"""Thinking-mode LLM judge: batched offline vLLM inference over a split, one JSON line per case.

Resumable: ids already in --out are skipped. A sample that ends without an "ANSWER: <letter>" line
(usually because it hit --max-tokens) gets one extra greedy token after a forced "ANSWER:".
Example: LC_DATA_DIR=data python -m lc.llm.judge_vllm --model models/qwen3-8b-awq --ids configs/dev200.txt --out outputs/x.jsonl
"""
import argparse
import json
import math
import os
import re
import subprocess
import time
from collections import Counter

# FlashInfer compiles its sampling kernel at start-up and fails with this server's nvcc 11.5; use the PyTorch sampler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.sampling_params import RequestOutputKind  # noqa: E402

from lc.data import load  # noqa: E402
from lc.llm.prompt import build_prompt  # noqa: E402

LETTERS = "ABCD"
ANSWER_RE = re.compile(r"ANSWER\W{0,6}([ABCD])\b")
NUM_LOGPROBS = 20  # vLLM's default cap; enough to see all four letters at the answer position


def free_gpu_util():
    """Leaves 1.5 GiB of the card free for its other users. vLLM's sampler takes about 0.4 GiB beyond the
    gpu_memory_utilization budget (top-p sorting is not in its memory profile), so 1.9 GiB is held back."""
    free_mib = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).split()[0])
    return round((free_mib / 1024 - 1.9) / 48, 3)


def letter_probs(logprobs_at):
    """Probability of each letter in one position's top log-probabilities (" B", "B" and "**B" all count as B)."""
    p = dict.fromkeys(LETTERS, 0.0)
    for lp in logprobs_at.values():
        t = (lp.decoded_token or "").strip(" *")
        if t in p:
            p[t] += math.exp(lp.logprob)
    return {k: round(v, 5) for k, v in p.items()}


def read_answer(out):
    """(letter, letter probabilities at the answer token) from a finished thinking sample, or (None, None)."""
    if "</think>" not in out.text:
        return None, None
    found = ANSWER_RE.findall(out.text.rsplit("</think>", 1)[1])
    if not found:
        return None, None
    if not out.logprobs:  # sampled without log-probabilities: the letter alone
        return found[-1], None
    toks = [lp[t].decoded_token or "" for t, lp in zip(out.token_ids, out.logprobs)]
    for i in range(len(toks) - 1, -1, -1):
        if toks[i].strip(" *") == found[-1] and re.search(r"ANSWER\W{0,6}$", "".join(toks[max(0, i - 6):i])):
            return found[-1], letter_probs(out.logprobs[i])
    return found[-1], None


def complete_lines(path):
    """Lines of a JSONL file, after cutting off a half-written last line that a killed run can leave."""
    if not os.path.exists(path):
        return []
    with open(path, "rb+") as f:
        data = f.read()
        data = data[:data.rfind(b"\n") + 1]
        f.truncate(len(data))
    return [line for line in data.decode().splitlines() if line.strip()]


def vote(samples):
    counts = Counter(s["letter"] for s in samples if s["letter"])
    if not counts:
        return None
    top = max(counts.values())
    # ties go to the letter with the larger summed probability at the answer position
    return max((l for l in counts if counts[l] == top), key=lambda l: sum((s["p"] or {}).get(l, 0) for s in samples))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local model folder")
    ap.add_argument("--split", default="train", help="train or test (CSV in LC_DATA_DIR)")
    ap.add_argument("--ids", help="file with one case id per line, run in that order; default: every case in the split")
    ap.add_argument("--out", required=True, help="JSONL to append to; reasoning texts go to <out>.texts.jsonl")
    ap.add_argument("--n", type=int, default=1, help="samples per case, majority vote (self-consistency)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="sample k of every case uses seed + k")
    ap.add_argument("--max-tokens", type=int, default=8192, help="output tokens per sample, thinking included")
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096, help="lower it to free activation memory for the KV cache")
    ap.add_argument("--gpu-util", type=float, help="fraction of the 48 GB card; default (free GiB - 1.9) / 48 at launch")
    args = ap.parse_args()

    cases = load(args.split)
    if args.ids:
        with open(args.ids) as f:
            order = dict.fromkeys(f.read().split())
        by_id = {c.id: c for c in cases}
        cases = [by_id[i] for i in order]  # file order: vLLM serves requests first come, first served
    texts_path = args.out[:-len(".jsonl")] + ".texts.jsonl" if args.out.endswith(".jsonl") else args.out + ".texts"
    done = {json.loads(line)["id"] for line in complete_lines(args.out)}
    complete_lines(texts_path)
    todo = [c for c in cases if c.id not in done]
    print(f"{len(cases)} cases, {len(done)} already in {args.out}, {len(todo)} to run", flush=True)
    if not todo:
        return

    util = args.gpu_util or free_gpu_util()
    with open(os.path.join(args.model, "config.json")) as f:
        multimodal = "vision_config" in json.load(f)
    print(f"gpu_memory_utilization {util}", flush=True)
    llm = LLM(args.model, gpu_memory_utilization=util, max_model_len=args.max_tokens + 2048, max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_num_batched_tokens, seed=args.seed, **({"language_model_only": True} if multimodal else {}))
    print(subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout, flush=True)

    tok = llm.get_tokenizer()
    eng = llm.llm_engine
    final = RequestOutputKind.FINAL_ONLY
    sample_params = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                   presence_penalty=args.presence_penalty, seed=args.seed, max_tokens=args.max_tokens,
                                   logprobs=NUM_LOGPROBS, output_kind=final)
    force_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=NUM_LOGPROBS, output_kind=final)
    prompts = {}
    for c in todo:
        prompts[c.id] = tok.apply_chat_template([{"role": "user", "content": build_prompt(c)}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=True)
        eng.add_request(c.id, prompts[c.id], sample_params)

    pending, waiting, prompt_tokens = {}, Counter(), {}
    t0, n_written, tokens_written = time.time(), 0, 0
    with open(args.out, "a") as fout, open(texts_path, "a") as ftxt:
        while eng.has_unfinished_requests():
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
                fout.write(json.dumps({
                    "id": cid, "pred": vote(s), "letters": [x["letter"] for x in s], "probs": [x["p"] for x in s],
                    "p_mean": {l: round(sum(p[l] for p in ps) / len(ps), 5) for l in LETTERS} if ps else None,
                    "out_tokens": [x["tokens"] for x in s], "forced": [x["forced"] for x in s], "prompt_tokens": prompt_tokens.pop(cid),
                }) + "\n")
                ftxt.write(json.dumps({"id": cid, "texts": [x["text"] for x in s]}) + "\n")
                fout.flush()
                ftxt.flush()
                n_written += 1
                tokens_written += sum(x["tokens"] for x in s)
                if n_written % 25 == 0 or n_written == len(todo):
                    mins = (time.time() - t0) / 60
                    print(f"{n_written}/{len(todo)} cases, {mins:.1f} min, {n_written / mins:.2f} cases/min, "
                          f"{tokens_written / n_written:.0f} output tokens/case, {tokens_written / mins / 60:.0f} tokens/s", flush=True)


if __name__ == "__main__":  # vLLM's engine runs in a spawned process that re-imports this module
    main()
