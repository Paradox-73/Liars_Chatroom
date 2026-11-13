"""The model half of predict.py, run by it as a child process so that predict.py can stop it on time, restart it with
safer engine settings after a failure, and keep a valid predictions file whatever happens here.

It runs the thinking-mode judge of lc/llm (the prompt, sampling and answer parsing of lc/llm/judge_vllm.py and
kaggle/kjudge.py) with Qwen3-14B-AWQ on the cases in --cases, and appends one JSON line per finished answer to --out.
A free slot takes, in this order:
  1. a continuation: a first sample that stopped at 8,192 tokens while still thinking goes on to 16,384 tokens in
     total. It goes first only while the first samples are on schedule; otherwise it waits for them.
  2. the first sample of a case: seed 0, at most 8,192 output tokens.
  3. an extra sample (seed 1, 2, ...) for the least certain case: smallest vote margin over the speakers the rules
     did not clear, at most 5 samples a case, none once the margin reaches 1.5.
A sample with no "ANSWER:" line gets a forced answer: "ANSWER:" is appended and one more token is read.
No thinking work starts after --stop-new. Then every case without a finished sample gets a one-token answer without
thinking, and the process exits once nothing is left, or at --hard-stop.
Exit codes: 0 done, 1 error, 3 no GPU, 4 vLLM missing.
"""
import argparse
import json
import os
import sys
import time
from collections import deque

T0 = time.time()
try:
    import torch
    from lc.llm.judge_vllm import NUM_LOGPROBS, letter_probs, read_answer  # also turns off the FlashInfer sampler
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
except ImportError as e:
    print(f"judge: cannot import the model libraries: {e!r}", flush=True)
    sys.exit(4)
from lc.combine import tally
from lc.data import Case, Message
from lc.llm.prompt import build_prompt

MAIN_TOKENS = int(os.environ.get("LC_MAIN_TOKENS", 8192))  # first sample; lowered only to test the capped path
CONT_TOKENS = 16384  # a continued sample in total
MAX_SAMPLES, SETTLED = 5, 1.5
THINKING = ("main", "cont", "extra")
# per tensor-parallel size: max model length, thinking slots, one-token slots, tokens per engine step
ENGINE = {2: (CONT_TOKENS + 2048, 40, 24, 4096), 1: (8192 + 2048, 8, 8, 2048)}


def log(msg):
    print(f"[judge {(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, help="JSONL from predict.py: id, scenario, speakers, evidence, messages, cands")
    ap.add_argument("--out", required=True, help="JSONL of answers; appended to, and read back on a restart")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stop-new", type=float, required=True, help="unix time: no thinking work starts after it")
    ap.add_argument("--hard-stop", type=float, required=True, help="unix time: exit")
    ap.add_argument("--heartbeat", required=True, help="file rewritten every 10 s while the engine runs")
    ap.add_argument("--tp", type=int, default=0, help="tensor parallel size; default: 2 with two or more cards, else 1")
    ap.add_argument("--eager", action="store_true", help="no torch.compile and no CUDA graphs")
    a = ap.parse_args()

    cases, cands, order = {}, {}, []
    with open(a.cases, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cases[d["id"]] = Case(d["id"], d["scenario"], d["speakers"], d["evidence"], [Message(*m) for m in d["messages"]])
            cands[d["id"]] = sorted(d["cands"])
            order.append(d["id"])
    samples = {}  # id -> {k: record}; k is the sample number, -1 for the answer without thinking
    if os.path.exists(a.out):  # a restart: keep what the last attempt finished
        with open(a.out, encoding="utf-8") as f:
            for line in f:
                if line.endswith("\n"):
                    r = json.loads(line)
                    old = samples.setdefault(r["id"], {}).get(r["k"])
                    if old is None or r["stage"] >= old["stage"]:
                        samples[r["id"]][r["k"]] = r
    main_q = deque(i for i in order if 0 not in samples.get(i, {}))
    log(f"{len(order)} cases, {len(order) - len(main_q)} already have a first sample")

    gpus = torch.cuda.device_count()  # counted through NVML, so CUDA is not started before vLLM starts its workers
    if not gpus:
        log("no GPU found")
        sys.exit(3)
    tp = a.tp or min(gpus, 2)
    max_len, think_slots, quick_slots, step_tokens = ENGINE[tp]
    think_slots = int(os.environ.get("LC_SLOTS", think_slots))  # per model: fewer when the weights leave little room
    max_len = int(os.environ.get("LC_MAX_LEN", max_len))  # per model: a small one has room for continuations on one card
    presence = float(os.environ.get("LC_PRESENCE_PENALTY", 0))  # per model: the Qwen3.5 models use 1.5
    util = float(os.environ.get("LC_GPU_UTIL", "0.92"))  # lowered only to test on a shared card
    with open(os.path.join(a.model, "config.json")) as f:
        multimodal = "vision_config" in json.load(f)  # Qwen3.5 checkpoints carry a vision tower; it is not loaded
    log(f"{gpus} GPU(s): tensor parallel {tp}, max model length {max_len}, {think_slots}+{quick_slots} slots, "
        f"presence penalty {presence}, eager {a.eager}")
    llm = LLM(a.model, tensor_parallel_size=tp, dtype="float16", gpu_memory_utilization=util, max_model_len=max_len,
              max_num_seqs=think_slots + quick_slots, max_num_batched_tokens=step_tokens, seed=0,
              disable_custom_all_reduce=True, enforce_eager=a.eager, **({"language_model_only": True} if multimodal else {}))
    eng, tok = llm.llm_engine, llm.get_tokenizer()
    t_ready = time.time()
    log(f"engine ready after {(t_ready - T0) / 60:.1f} min")

    final = RequestOutputKind.FINAL_ONLY
    one_token = SamplingParams(max_tokens=1, temperature=0.0, logprobs=NUM_LOGPROBS, output_kind=final)
    prompts, plens = {}, {}

    def prompt(cid):
        if cid not in prompts:
            prompts[cid] = tok.apply_chat_template([{"role": "user", "content": build_prompt(cases[cid])}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True)
        return prompts[cid]

    def no_think(cid):
        # an empty thought, then the answer: Qwen3's own form without thinking, and it works for templates that
        # already open the thought themselves
        p = prompt(cid)
        return p + ("" if p.endswith("<think>\n") else "<think>\n") + "\n</think>\n\nANSWER:"

    def plen(cid):
        if cid not in plens:
            plens[cid] = len(tok.encode(prompt(cid)))
        return plens[cid]

    # Longest prompt first: on the 14B's test answers a long prompt means long reasoning (correlation 0.51; the top
    # quarter of prompts held 21 of the 29 capped samples), so the slow cases start early and the cases left for the
    # one-token answers at the stop time are the short ones.
    try:
        main_q = deque(sorted(main_q, key=lambda cid: -plen(cid)))
    except Exception as e:
        log(f"prompts not measured ({type(e).__name__}): input order")

    fout = open(a.out, "a", encoding="utf-8")
    flight = {}  # request id -> (kind, case id, k, text so far, tokens so far, kind of the sample it answers for)
    queued_cont, quick, dead = deque(), deque(), set()  # dead: cases whose request could not start
    counts = dict.fromkeys(("main", "cont", "extra", "force", "nothink", "main_capped", "cont_finished", "not_started"), 0)
    tokens_out = n_req = 0

    def record(cid, k, src, letter, p, forced, tokens):
        r = {"id": cid, "k": k, "stage": int(src == "cont"), "kind": src, "pred": letter, "p_mean": p, "forced": [forced],
             "out_tokens": [tokens], "min": round((time.time() - T0) / 60, 2)}
        fout.write(json.dumps(r) + "\n")
        fout.flush()
        old = samples.setdefault(cid, {}).get(k)
        if old is None or r["stage"] >= old["stage"]:  # a continuation's answer replaces its first part's
            samples[cid][k] = r

    def start(kind, cid, k, text="", tokens=0, src=None):
        nonlocal n_req
        rid = str(n_req)
        n_req += 1
        try:
            if kind in THINKING:
                room = max_len - plen(cid) - tokens - 64  # 64: re-tokenising the text so far can add a few tokens
                cap = min(MAIN_TOKENS if kind != "cont" else CONT_TOKENS - tokens, room)
                if cap < 256:
                    raise ValueError(f"only {room} tokens of room")
                eng.add_request(rid, prompt(cid) + text, SamplingParams(
                    temperature=0.6, top_p=0.95, top_k=20, presence_penalty=presence, seed=k, max_tokens=cap,
                    logprobs=NUM_LOGPROBS, output_kind=final))
            else:  # text is the whole prompt of a one-token answer
                eng.add_request(rid, text, one_token)
        except Exception as e:  # one bad case must not stop the others
            counts["not_started"] += 1
            dead.add(cid)
            log(f"case {cid}: {kind} request not started: {e!r}")
            return
        flight[rid] = (kind, cid, k, text, tokens, src or kind)

    def running(thinking):
        return sum((v[0] in THINKING) == thinking for v in flight.values())

    def margin(cid):
        recs = [r for k, r in sorted(samples.get(cid, {}).items()) if k >= 0]
        total, _ = tally(cid, cands[cid], [(1.0, {cid: r}) for r in recs])
        top = sorted(total.values(), reverse=True) + [0.0, 0.0]
        return top[0] - top[1]

    def pick_extra():
        best = None
        skip = dead | {v[1] for v in flight.values()} | {q[1] for q in quick} | {c[0] for c in queued_cont}  # one at a time
        for i, cid in enumerate(order):
            ks = [k for k in samples.get(cid, {}) if k >= 0]
            if not ks or cid in skip or len(ks) >= MAX_SAMPLES:
                continue
            m = margin(cid)
            if m < SETTLED:
                key = (m, -max(samples[cid][k]["out_tokens"][0] for k in ks), i)  # longer reasoning first: a harder case
                if best is None or key < best[0]:
                    best = (key, cid, max(ks) + 1)
        return best and ("extra", best[1], best[2])

    def on_schedule(now):
        done = sum(1 for i in order if 0 in samples.get(i, {}))
        if done < 20:
            return True
        left = len(main_q) + sum(v[0] == "main" for v in flight.values())
        return now + left * (now - t_ready) / done < a.stop_new - 120

    def next_work(now):
        if queued_cont and (not main_q or on_schedule(now)):
            return ("cont",) + queued_cont.popleft()
        if main_q:
            return ("main", main_q.popleft(), 0)
        return pick_extra()

    def beat():
        with open(a.heartbeat, "w") as f:
            f.write(str(time.time()))

    beat()
    last_beat = last_log = time.time()
    late = False
    while True:
        now = time.time()
        if now >= a.hard_stop:
            log(f"hard stop with {len(flight)} requests running")
            break
        if now >= a.stop_new and not late:
            late = True
            need = [i for i in order if not any(k >= 0 for k in samples.get(i, {})) and -1 not in samples.get(i, {})]
            log(f"stop time: {len(main_q)} first samples and {len(queued_cont)} continuations dropped; "
                f"{len(need)} cases get an answer without thinking")
            main_q.clear()
            queued_cont.clear()
            for cid in need:
                try:
                    quick.append(("nothink", cid, -1, no_think(cid)))
                except Exception as e:
                    log(f"case {cid}: no prompt: {e!r}")
        while not late and running(True) < think_slots:
            w = next_work(now)
            if not w:
                break
            start(*w)
        while quick and running(False) < quick_slots:
            start(*quick.popleft())
        if not flight:
            break
        for o in eng.step():
            if not o.finished:
                continue
            kind, cid, k, text, tokens, src = flight.pop(o.request_id)
            counts[kind] += 1
            out = o.outputs[0]
            if kind not in THINKING:
                p = letter_probs(out.logprobs[0])
                record(cid, k, src, max(p, key=p.get) if any(p.values()) else None, p, kind == "force", tokens)
                continue
            text, tokens = text + out.text, tokens + len(out.token_ids)
            tokens_out += len(out.token_ids)
            letter, p = read_answer(out)
            if letter:
                record(cid, k, kind, letter, p, False, tokens)
                counts["cont_finished"] += kind == "cont"
                continue
            suffix = "\n\nANSWER:" if "</think>" in text else "\n</think>\n\nANSWER:"
            quick.appendleft(("force", cid, k, prompt(cid) + text + suffix, tokens, kind))
            if kind == "main":
                counts["main_capped"] += 1
                if "</think>" not in text and max_len - plen(cid) - tokens - 64 >= 2048:
                    queued_cont.append((cid, k, text, tokens))
        if time.time() - last_beat > 10:
            beat()
            last_beat = time.time()
        if time.time() - last_log > 120:
            mins = (time.time() - t_ready) / 60
            log(f"{counts['main']}/{len(order)} first samples, {counts['main_capped']} capped, {counts['cont']} continued "
                f"({counts['cont_finished']} finished), {counts['extra']} extra; {tokens_out / mins / 60:.0f} output tokens/s; "
                f"{len(main_q)} waiting, {running(True)} thinking")
            last_log = time.time()
    fout.close()
    log(f"done: {counts}, {tokens_out} output tokens")
    os._exit(0)  # every line is flushed; predict.py stops the engine's processes


if __name__ == "__main__":  # vLLM starts its workers in spawned processes, which re-import this module
    try:
        main()
    except Exception as e:
        log(f"error: {e!r}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
