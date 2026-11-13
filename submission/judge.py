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
    from lc.llm.judge_vllm import ANSWER_RE, NUM_LOGPROBS, letter_probs, read_answer  # also turns off the FlashInfer sampler
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind
except ImportError as e:
    print(f"judge: cannot import the model libraries: {e!r}", flush=True)
    sys.exit(4)
from lc.combine import tally
from lc.data import Case, Message
from lc.llm.prompt import build_prompt

MAIN_TOKENS = int(os.environ.get("LC_MAIN_TOKENS", 8192))  # first sample
CONT_TOKENS = int(os.environ.get("LC_CONT_TOKENS", 16384))  # a continued sample in total
MAX_SAMPLES, SETTLED = 5, 1.5
# optional first pass without thinking: every case gets a short per-claim answer before any thinking starts
FLOOR, FLOOR_TOKENS = os.environ.get("LC_FLOOR", ""), int(os.environ.get("LC_FLOOR_TOKENS", 768))
THINKING = ("main", "cont", "extra", "floor")  # the kinds that generate many tokens and hold a thinking slot
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
    style = os.environ.get("LC_PROMPT", "full")  # the closing instruction of lc/llm/prompt.py
    # letter probabilities at the answer need log-probabilities on every sampled token; without them the letter alone votes
    token_logprobs = NUM_LOGPROBS if os.environ.get("LC_TOKEN_LOGPROBS", "1") == "1" else None
    util = float(os.environ.get("LC_GPU_UTIL", "0.92"))  # lowered only to test on a shared card
    with open(os.path.join(a.model, "config.json")) as f:
        multimodal = "vision_config" in json.load(f)  # Qwen3.5 checkpoints carry a vision tower; it is not loaded
    log(f"{gpus} GPU(s): tensor parallel {tp}, max model length {max_len}, {think_slots}+{quick_slots} slots, "
        f"presence penalty {presence}, eager {a.eager}")
    llm = LLM(a.model, tensor_parallel_size=tp, dtype="float16", gpu_memory_utilization=util, max_model_len=max_len,
              max_num_seqs=think_slots + quick_slots, max_num_batched_tokens=step_tokens, seed=0,
              disable_custom_all_reduce=True, enforce_eager=a.eager, disable_log_stats=os.environ.get("LC_ENGINE_STATS") != "1",
              **({"language_model_only": True} if multimodal else {}))
    eng, tok = llm.llm_engine, llm.get_tokenizer()
    t_ready = time.time()
    cache = eng.vllm_config.cache_config
    kv_tokens = (cache.num_gpu_blocks or 0) * cache.block_size
    if kv_tokens:  # more cases in flight than the cache holds makes vLLM evict and recompute them over and over
        think_slots = min(think_slots, max(4, kv_tokens // 2500))
    log(f"engine ready after {(t_ready - T0) / 60:.1f} min; cache {kv_tokens} tokens, {think_slots} thinking slots")

    final = RequestOutputKind.FINAL_ONLY
    one_token = SamplingParams(max_tokens=1, temperature=0.0, logprobs=NUM_LOGPROBS, output_kind=final)
    prompts, plens = {}, {}

    def prompt(cid):
        if cid not in prompts:
            prompts[cid] = tok.apply_chat_template([{"role": "user", "content": build_prompt(cases[cid], style)}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True)
        return prompts[cid]

    def floor_prompt(cid):
        return tok.apply_chat_template([{"role": "user", "content": build_prompt(cases[cid], FLOOR)}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

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
    counts = dict.fromkeys(("floor", "main", "cont", "extra", "force", "nothink", "main_capped", "cont_finished", "not_started"), 0)
    tokens_out = n_req = seq_tokens = seq_n = 0

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
            if kind == "floor":
                eng.add_request(rid, floor_prompt(cid), SamplingParams(temperature=0.7, top_p=0.8, top_k=20, seed=0,
                                                                       max_tokens=FLOOR_TOKENS, output_kind=final))
            elif kind in THINKING:
                room = max_len - plen(cid) - tokens - 64  # 64: re-tokenising the text so far can add a few tokens
                cap = min(MAIN_TOKENS if kind == "extra" else share() if kind == "main" else CONT_TOKENS - tokens, room)
                if cap < 256:
                    raise ValueError(f"only {room} tokens of room")
                eng.add_request(rid, prompt(cid) + text, SamplingParams(
                    temperature=0.6, top_p=0.95, top_k=20, presence_penalty=presence, seed=k, max_tokens=cap,
                    logprobs=token_logprobs, output_kind=final))
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

    def share():
        """The first-sample cap: MAIN_TOKENS, or less once the tokens left before the stop time do not cover every case
        still waiting. 1,024 tokens of thought and a forced answer beat no thought at all (67% against 40% for
        Qwen3-8B on our new-family cases)."""
        if seq_n < 20:
            return MAIN_TOKENS
        now = time.time()
        left = len(main_q) + 1 + sum(v[0] == "main" for v in flight.values())
        return max(768, min(MAIN_TOKENS, int(tokens_out / (now - t_ready) * (a.stop_new - now) / left)))

    def on_schedule(now):
        done = sum(1 for i in order if 0 in samples.get(i, {}))
        if done < 20:  # too early to tell: first samples go first
            return False
        left = len(main_q) + sum(v[0] == "main" for v in flight.values())
        return now + left * (now - t_ready) / done < a.stop_new - 120

    floor_q = deque(order if FLOOR else [])

    def next_work(now, slots):
        if floor_q:
            return ("floor", floor_q.popleft(), -1)
        # a continuation holds its slot for thousands of tokens, so at most half the slots while first samples wait
        if queued_cont and (not main_q or (on_schedule(now) and sum(v[0] == "cont" for v in flight.values()) < slots // 2)):
            return ("cont",) + queued_cont.popleft()
        if main_q:
            return ("main", main_q.popleft(), 0)
        return pick_extra()

    def beat():
        with open(a.heartbeat, "w") as f:
            f.write(str(time.time()))

    def abort(kinds):
        """Stops the running requests of these kinds, so that their cache goes to the one-token answers."""
        rids = [rid for rid, v in flight.items() if v[0] in kinds]
        if rids:
            eng.abort_request(rids)
            for rid in rids:
                flight.pop(rid)
        return len(rids)

    beat()
    last_beat = last_log = time.time()
    late = cut = False
    while True:
        now = time.time()
        if now >= a.hard_stop:
            log(f"hard stop with {len(flight)} requests running")
            break
        if now >= a.stop_new and not late:
            late = True
            need = [i for i in order if not any(k >= 0 for k in samples.get(i, {})) and -1 not in samples.get(i, {})]
            # continuations and extra samples only improve cases that already have an answer
            log(f"stop time: {len(main_q)} first samples and {len(queued_cont)} continuations dropped, "
                f"{abort(('cont', 'extra'))} running continuations and extra samples stopped; "
                f"{len(need)} cases get an answer without thinking")
            main_q.clear()
            queued_cont.clear()
            for cid in need:
                try:
                    quick.append(("nothink", cid, -1, no_think(cid)))
                except Exception as e:
                    log(f"case {cid}: no prompt: {e!r}")
        if late and not cut and now >= a.hard_stop - 90 and (quick or running(False)):
            cut = True  # the one-token answers still wait for cache: the unfinished first samples give way
            log(f"{abort(THINKING)} first samples stopped for {len(quick) + running(False)} one-token answers")
        # once 10 samples are done, keep the cases in flight within the cache at their mean length
        slots = think_slots if seq_n < 10 or not kv_tokens else min(think_slots, max(4, int(kv_tokens * seq_n / seq_tokens)))
        while not late and running(True) < slots:
            w = next_work(now, slots)
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
            if kind == "floor":
                found = ANSWER_RE.findall(out.text.rsplit("</think>", 1)[-1])
                if found:
                    record(cid, -1, "floor", found[-1], None, False, len(out.token_ids))
                else:
                    quick.appendleft(("force", cid, -1, floor_prompt(cid) + out.text + "\n\nANSWER:", len(out.token_ids), "floor"))
                continue
            text, tokens = text + out.text, tokens + len(out.token_ids)
            tokens_out += len(out.token_ids)
            seq_tokens, seq_n = seq_tokens + plen(cid) + tokens, seq_n + 1
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
                f"{len(main_q)} waiting, {running(True)} of {slots} slots thinking")
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
