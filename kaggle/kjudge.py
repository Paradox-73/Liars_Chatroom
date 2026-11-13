"""The thinking-mode judge of lc/llm/judge_vllm.py (same prompt, sampling and answer parsing) for a Kaggle session
with two T4 cards: tensor parallel over both cards, float16, and phases run one after another in one engine.

A phase is split:ids_file:out_jsonl[:min_accuracy]. A train phase also saves reasoning texts and, given
min_accuracy, stops the run when its accuracy is lower. Cases already in out_jsonl are skipped. No new case is
waited for after --deadline (unix time). Every case is one flushed JSON line with judge_vllm's fields.
Example: python -m kjudge --model /tmp/models/m --phase train:dev.txt:dev.jsonl:0.85 --phase test:t.txt:test.jsonl
"""
import argparse
import json
import os
import subprocess
import time
from collections import Counter

from lc.data import load
from lc.llm.judge_vllm import LETTERS, NUM_LOGPROBS, letter_probs, read_answer, vote  # also turns off the FlashInfer sampler
from lc.llm.prompt import build_prompt
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind

HEARTBEAT = "heartbeat"  # touched after start-up and after every case, for the launcher's hang check


def beat():
    with open(HEARTBEAT, "w") as f:
        f.write(str(time.time()))


def gpus():
    q = "index,name,memory.used,memory.total,temperature.gpu,clocks.sm,power.draw"
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True).stdout


def scores(out_path, cases):
    """(correct, labelled, forced answers, mean output tokens) over the cases' lines in out_path."""
    want = {c.id: c.label for c in cases}
    rows = [r for r in map(json.loads, open(out_path)) if r["id"] in want]
    correct = sum(r["pred"] == want[r["id"]] for r in rows)
    forced = sum(any(r["forced"]) for r in rows)
    return correct, len(rows), forced, sum(sum(r["out_tokens"]) for r in rows) / max(len(rows), 1)


def run_phase(eng, tok, name, cases, out_path, keep_texts, sample_params, force_params, deadline):
    """Runs the cases not yet in out_path until all finish or the deadline passes; True if the deadline passed."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [c for c in cases if c.id not in done]
    label = {c.id: c.label for c in todo}
    print(f"[{name}] {len(cases)} cases, {len(done & {c.id for c in cases})} already in {out_path}, {len(todo)} to run", flush=True)
    prompts = {}
    for c in todo:
        prompts[c.id] = tok.apply_chat_template([{"role": "user", "content": build_prompt(c)}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=True)
        eng.add_request(c.id, prompts[c.id], sample_params)

    pending, waiting, prompt_tokens = {}, Counter(), {}
    t0, n_written, tokens_written, correct = time.time(), 0, 0, 0
    texts_path = out_path[:-len(".jsonl")] + ".texts.jsonl"
    with open(out_path, "a") as fout, open(texts_path if keep_texts else os.devnull, "a") as ftxt:
        while eng.has_unfinished_requests():
            if time.time() > deadline:
                print(f"[{name}] deadline reached after {n_written} of {len(todo)} cases", flush=True)
                return True
            for o in eng.step():
                if not o.finished:
                    continue
                cid, _, k = o.request_id.partition("#force")
                if k:
                    p = letter_probs(o.outputs[0].logprobs[0])
                    pending[cid][int(k)].update(letter=max(p, key=p.get) if any(p.values()) else None, p=p, forced=True)
                    waiting[cid] -= 1
                else:
                    prompt_tokens[cid] = len(o.prompt_token_ids)
                    pending[cid] = []
                    for out in sorted(o.outputs, key=lambda x: x.index):
                        letter, p = read_answer(out)
                        pending[cid].append({"letter": letter, "p": p, "tokens": len(out.token_ids), "forced": False, "text": out.text})
                        if letter is None:
                            suffix = "\n\nANSWER:" if "</think>" in out.text else "\n</think>\n\nANSWER:"
                            eng.add_request(f"{cid}#force{out.index}", prompts[cid] + out.text + suffix, force_params)
                            waiting[cid] += 1
                if waiting[cid]:
                    continue
                s = pending.pop(cid)
                ps = [x["p"] for x in s if x["p"]]
                pred = vote(s)
                fout.write(json.dumps({
                    "id": cid, "pred": pred, "letters": [x["letter"] for x in s], "probs": [x["p"] for x in s],
                    "p_mean": {l: round(sum(p[l] for p in ps) / len(ps), 5) for l in LETTERS} if ps else None,
                    "out_tokens": [x["tokens"] for x in s], "forced": [x["forced"] for x in s], "prompt_tokens": prompt_tokens.pop(cid),
                }) + "\n")
                ftxt.write(json.dumps({"id": cid, "texts": [x["text"] for x in s]}) + "\n")
                fout.flush()
                ftxt.flush()
                beat()
                n_written += 1
                tokens_written += sum(x["tokens"] for x in s)
                correct += label[cid] is not None and pred == label[cid]
                if n_written % 25 == 0 or n_written == len(todo):
                    mins = (time.time() - t0) / 60
                    acc = f", {correct}/{n_written} correct" if label[cid] is not None else ""
                    print(f"[{name}] {n_written}/{len(todo)} cases, {mins:.1f} min, {n_written / mins:.2f} cases/min, "
                          f"{tokens_written / n_written:.0f} output tokens/case, {tokens_written / mins / 60:.0f} tokens/s{acc}", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local model folder")
    ap.add_argument("--phase", action="append", required=True, help="split:ids_file:out_jsonl[:min_accuracy], run in order")
    ap.add_argument("--deadline", type=float, default=float("inf"), help="unix time; no case is waited for after it")
    ap.add_argument("--tp", type=int, default=2, help="tensor parallel size (number of cards)")
    ap.add_argument("--eager", action="store_true", help="no torch.compile and no CUDA graphs")
    ap.add_argument("--n", type=int, default=1, help="samples per case, majority vote")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=6000, help="output tokens per sample, thinking included")
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.92, help="fraction of each card's memory for vLLM")
    args = ap.parse_args()

    phases = []
    splits = {}
    for spec in args.phase:
        split, ids_path, out_path, *gate = spec.split(":")
        if split not in splits:
            splits[split] = {c.id: c for c in load(split)}
        with open(ids_path) as f:
            ids = list(dict.fromkeys(f.read().split()))
        phases.append((f"{split}:{os.path.basename(ids_path)}", [splits[split][i] for i in ids], out_path, float(gate[0]) if gate else None))
    print(gpus(), flush=True)

    llm = LLM(args.model, tensor_parallel_size=args.tp, dtype="float16", gpu_memory_utilization=args.gpu_util,
              max_model_len=args.max_tokens + 2048, max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
              seed=args.seed, disable_custom_all_reduce=True, enforce_eager=args.eager)
    beat()
    print(gpus(), flush=True)
    tok = llm.get_tokenizer()
    final = RequestOutputKind.FINAL_ONLY
    sample_params = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                   presence_penalty=args.presence_penalty, seed=args.seed, max_tokens=args.max_tokens,
                                   logprobs=NUM_LOGPROBS, output_kind=final)
    force_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=NUM_LOGPROBS, output_kind=final)

    for name, cases, out_path, gate in phases:
        is_train = cases[0].label is not None
        late = run_phase(llm.llm_engine, tok, name, cases, out_path, is_train, sample_params, force_params, args.deadline)
        print(gpus(), flush=True)
        if is_train:
            correct, n, forced, mean_tokens = scores(out_path, cases)
            print(f"[{name}] accuracy {correct}/{n} = {correct / max(n, 1):.4f}, {forced} forced answers, "
                  f"{mean_tokens:.0f} output tokens/case", flush=True)
            if gate is not None and correct < gate * len(cases):
                print(f"[{name}] below the {gate} accuracy gate: stopping", flush=True)
                break
        if late:
            break
    os._exit(0)  # every line is flushed; do not wait for the engine's processes to wind down


if __name__ == "__main__":  # vLLM starts its workers in spawned processes, which re-import this module
    main()
