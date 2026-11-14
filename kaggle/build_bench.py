"""Writes a Kaggle script kernel that measures decode throughput on two T4 cards for the settings that could limit it:
log-probabilities on every token, top-p/top-k sampling, the number of sequences in flight, the attention backend, and
one engine per card instead of tensor parallel. Prompts are the first new-family cases; every sequence generates
exactly --tokens tokens, so the runs are comparable. It prints one line per run: BENCH <label>: tokens/s.

python kaggle/build_bench.py kaggle/bench-t4 --zip ClaudeLaggye_code.zip --newfam <repo>/validation/newfam --owner <user>
"""
import argparse
import base64
import json
import os

BENCH = r'''
import csv, json, os, sys, time
sys.path.insert(0, "/tmp/sub")
from lc.data import case_from_row
from lc.llm.prompt import build_prompt
from vllm import LLM, SamplingParams

model, tp, runs = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
rows = list(csv.DictReader(open("/tmp/newfam.csv", encoding="utf-8")))
llm = LLM(model, tensor_parallel_size=tp, dtype="float16", gpu_memory_utilization=0.92, max_model_len=18432,
          max_num_seqs=96, max_num_batched_tokens=4096, seed=0, disable_custom_all_reduce=True)
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(case_from_row(r))}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True) for r in rows]
for label, n, kw in runs:
    t = time.time()
    outs = llm.generate(prompts[:n], SamplingParams(seed=0, ignore_eos=True, **kw), use_tqdm=False)
    dt = time.time() - t
    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"BENCH {os.environ.get('BENCH_TAG', '')} {label}: {n} sequences, {toks} tokens in {dt:.0f} s = {toks / dt:.0f} tokens/s",
          flush=True)
os._exit(0)
'''

DRIVER = r'''
import base64, io, json, os, subprocess, time, zipfile

T0 = time.time()


def log(msg):
    print(f"[bench {(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def sh(cmd, check=True):
    log(f"$ {cmd}")
    rc = subprocess.run(cmd, shell=True, check=check, executable="/bin/bash").returncode
    # bench.py leaves with os._exit, so its engine processes would keep the cards' memory for the next run
    subprocess.run("pkill -9 -f /tmp/sub/.venv/bin/python; sleep 5", shell=True)
    return rc


zipfile.ZipFile(io.BytesIO(base64.b64decode(ZIP))).extractall("/tmp/sub")
with open("/tmp/newfam.csv", "w", encoding="utf-8", newline="") as f:
    f.write(NEWFAM)
with open("/tmp/bench.py", "w") as f:
    f.write(BENCH)
sh("nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm,power.limit --format=csv; nproc")
sh(f"LC_MODEL_REPO={CFG['repo']} LC_MODEL_REV={CFG['rev']} LC_MODEL_DIR=/tmp/model bash /tmp/sub/setup.sh")
py = "VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 /tmp/sub/.venv/bin/python /tmp/bench.py /tmp/model"
T = CFG["tokens"]
base = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": T}
runs = [["48 in flight, top-p/top-k, logprobs 20 (as predict.py)", 48, dict(base, logprobs=20)],
        ["48 in flight, top-p/top-k, no logprobs", 48, base],
        ["48 in flight, plain temperature, no logprobs", 48, {"temperature": 0.6, "max_tokens": T}],
        ["16 in flight, as predict.py", 16, dict(base, logprobs=20)],
        ["96 in flight, as predict.py", 96, dict(base, logprobs=20)]]
sh(f"BENCH_TAG=tp2 {py} 2 '{json.dumps(runs)}'", check=False)
one = [runs[0]]
sh(f"BENCH_TAG=tp2-flex VLLM_ATTENTION_BACKEND=FLEX_ATTENTION {py} 2 '{json.dumps(one)}'", check=False)
half = [["24 in flight per card, as predict.py", 24, dict(base, logprobs=20)]]  # two engines at once, one per card
sh(f"(CUDA_VISIBLE_DEVICES=0 BENCH_TAG=card0 {py} 1 '{json.dumps(half)}' & CUDA_VISIBLE_DEVICES=1 BENCH_TAG=card1 {py} 1 "
   f"'{json.dumps(half)}' & wait)", check=False)
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--newfam", required=True, help="folder with newfam.csv")
    ap.add_argument("--repo", default="Qwen/Qwen3-8B-AWQ")
    ap.add_argument("--rev", default="4da05a8edb55c6046cce958586c33b61da07bb79")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--owner", default="YOUR_KAGGLE_USERNAME")
    a = ap.parse_args()
    with open(a.zip, "rb") as f:
        zip_b64 = base64.b64encode(f.read()).decode()
    newfam = open(os.path.join(a.newfam, "newfam.csv"), encoding="utf-8", newline="").read()
    cfg = {"repo": a.repo, "rev": a.rev, "tokens": a.tokens}
    run_py = f"ZIP = {zip_b64!r}\nNEWFAM = {newfam!r}\nBENCH = {BENCH!r}\nCFG = {cfg!r}\n{DRIVER}"
    compile(run_py, "run.py", "exec")
    compile(BENCH, "bench.py", "exec")
    slug = os.path.basename(os.path.normpath(a.kernel))
    meta = {"id": f"{a.owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    os.makedirs(a.kernel, exist_ok=True)
    with open(os.path.join(a.kernel, "run.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(run_py)
    with open(os.path.join(a.kernel, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{a.kernel}: {a.repo}@{a.rev[:8]}, {a.tokens} tokens per sequence")


if __name__ == "__main__":
    main()
