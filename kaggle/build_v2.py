"""Kaggle kernels for the checks of branch feat/predict-v2. Each kernel unpacks the code zip, runs setup.sh with internet
(vLLM and Qwen3-14B-AWQ, as the organisers will) and then one experiment on two T4 cards:

  pipe    predict.py on a labelled CSV with the time budget of the hidden run, as the bake-off driver of
          build_bakeoff.py does; one SUMMARY line.
  ending  the one-token forced answer of first samples that were still thinking at CUT tokens, read from the texts
          the explore-force14 kernel saved, under the pipeline's ending and a candidate ending; one ENDING line.

The zip must hold the committed code: the builder refuses uncommitted changes in submission/ and lc/ and any zip file
that differs from the file in the checkout, and the kernel logs the commit and the zip's md5.

python kaggle/build_v2.py pipe --zip Z --csv CASES --meta META --owner U --out kaggle/v2-set2
python kaggle/build_v2.py ending --zip Z --csv validation/newfam/newfam.csv --force force_out.jsonl --owner U --out kaggle/v2-ending
"""
import argparse
import base64
import csv
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_bakeoff import DRIVER as PIPE_DRIVER  # noqa: E402

MODEL = {"repo": "Qwen/Qwen3-14B-AWQ", "rev": "31c69efc29464b6bb0aee1398b5a7b50a99340c3"}
ENDINGS = {
    "current": "\n</think>\n\nANSWER:",
    "rule": "\n\nTime is up. Only one message cannot be true in any situation the evidence allows; a message that can be true "
            "under the exact words of the evidence is not the lie.\n</think>\n\nANSWER:",
}

ENDING = r'''
import csv, json, sys, time
sys.path.insert(0, "/tmp/sub")
from lc.data import case_from_row
from lc.llm.judge_vllm import NUM_LOGPROBS, letter_probs  # the pipeline's reading of a one-token answer
from lc.llm.prompt import build_prompt
from vllm import LLM, SamplingParams

c = json.load(open("/tmp/cfg.json"))
CUT = c["cut"]
rows = {r["id"]: r for r in csv.DictReader(open("/tmp/cases.csv", encoding="utf-8", newline=""))}
recs = [json.loads(x) for x in open("/tmp/force.jsonl", encoding="utf-8")]
llm = LLM("/tmp/model", tensor_parallel_size=2, dtype="float16", gpu_memory_utilization=0.92, max_model_len=CUT + 2048,
          max_num_seqs=40, max_num_batched_tokens=4096, seed=0, disable_custom_all_reduce=True)
tok = llm.get_tokenizer()
prompt = {i: tok.apply_chat_template([{"role": "user", "content": build_prompt(case_from_row(r))}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True) for i, r in rows.items()}
keep = []  # the samples a CUT-token cap would force: the thought has not ended within the first CUT tokens
for r in recs:
    ids = tok.encode(r["text"], add_special_tokens=False)
    text = tok.decode(ids[:CUT])
    if len(ids) >= CUT and "</think>" not in text:
        keep.append((r, text))
print(f"[ending] {len(recs)} samples, {len(keep)} still thinking at {CUT} tokens", flush=True)
one = SamplingParams(max_tokens=1, temperature=0.0, logprobs=NUM_LOGPROBS)
res = {(r["id"], r["k"]): {"id": r["id"], "k": r["k"], "label": r["label"]} for r, _ in keep}
for name, end in c["endings"].items():
    t = time.time()
    outs = llm.generate([prompt[r["id"]] + text + end for r, text in keep], one, use_tqdm=False)
    for (r, _), o in zip(keep, outs):
        p = letter_probs(o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {})
        res[(r["id"], r["k"])][name] = max(p, key=p.get) if any(p.values()) else None
    print(f"[ending] {name}: {len(keep)} answers in {time.time() - t:.0f} s", flush=True)
with open("/kaggle/working/ending_out.jsonl", "w", encoding="utf-8") as f:
    for key in sorted(res):
        f.write(json.dumps(res[key]) + "\n")
base, cand = list(c["endings"])
summary = {"samples": len(res), "cases": len({r["id"] for r in res.values()})}
for name in c["endings"]:
    summary[name] = sum(r[name] == r["label"] for r in res.values())
summary["wins"] = sum(r[base] != r["label"] and r[cand] == r["label"] for r in res.values())
summary["losses"] = sum(r[base] == r["label"] and r[cand] != r["label"] for r in res.values())
print("ENDING " + json.dumps(summary), flush=True)
import os
os._exit(0)
'''

ENDING_DRIVER = r'''
import base64, gzip, io, json, os, subprocess, time, zipfile

T0 = time.time()


def log(msg):
    print(f"[ending {(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def sh(cmd, check=False):
    log(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, executable="/bin/bash").returncode


zipfile.ZipFile(io.BytesIO(base64.b64decode(ZIP))).extractall("/tmp/sub")
with open("/tmp/cases.csv", "w", encoding="utf-8", newline="") as f:
    f.write(CASES)
with open("/tmp/force.jsonl", "wb") as f:
    f.write(gzip.decompress(base64.b64decode(FORCE)))
with open("/tmp/ending.py", "w") as f:
    f.write(PROGRAM)
with open("/tmp/cfg.json", "w") as f:
    json.dump(CFG, f)
sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")
sh(f"LC_MODEL_REPO={CFG['repo']} LC_MODEL_REV={CFG['rev']} LC_MODEL_DIR=/tmp/model bash /tmp/sub/setup.sh", check=True)
sh("VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 PYTHONPATH=/tmp/sub /tmp/sub/.venv/bin/python /tmp/ending.py")
log("done")
'''


def committed_zip(path):
    """(base64 of the zip, commit, md5), after checking that the zip holds the checkout's committed files."""
    dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--", "submission", "lc"], capture_output=True,
                           text=True).stdout
    assert not dirty, f"uncommitted changes in the code that goes into the zip:\n{dirty}"
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            src = os.path.join(ROOT, name if name.startswith("lc/") else os.path.join("submission", name))
            with open(src, "rb") as f:
                assert f.read() == z.read(name), f"{name} in the zip differs from {src}"
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    data = open(path, "rb").read()
    return base64.b64encode(data).decode(), commit, hashlib.md5(data).hexdigest()


def write_kernel(out, owner, run_py):
    compile(run_py, "run.py", "exec")
    slug = os.path.basename(os.path.normpath(out))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(run_py)
    meta = {"id": f"{owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "model_sources": []}
    with open(os.path.join(out, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{out}: {owner}/{slug}, {len(run_py) / 1024:.0f} KiB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["pipe", "ending"])
    ap.add_argument("--zip", required=True)
    ap.add_argument("--csv", required=True, help="labelled cases (train.csv columns)")
    ap.add_argument("--meta", help="pipe: id,family CSV for the per-family accuracy")
    ap.add_argument("--env", default="{}", help="pipe: JSON of environment settings for predict.py")
    ap.add_argument("--force", help="ending: force_out.jsonl of explore-force14 (id, k, label, text per first sample)")
    ap.add_argument("--cut", type=int, default=3072, help="ending: the first-sample cap to cut the texts at")
    ap.add_argument("--owner", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    zip_b64, commit, md5 = committed_zip(a.zip)
    head = f'"""Built by kaggle/build_v2.py from commit {commit}."""\nCOMMIT = {commit!r}\nZIP_MD5 = {md5!r}\nZIP = {zip_b64!r}\n'
    show = 'print(f"code: commit {COMMIT}, zip md5 {ZIP_MD5}", flush=True)\n'
    cases = open(a.csv, encoding="utf-8", newline="").read()
    if a.kind == "pipe":
        cfg = dict(MODEL, env=json.loads(a.env))
        meta = open(a.meta, encoding="utf-8", newline="").read()
        run_py = f"{head}NEWFAM = {cases!r}\nMETA = {meta!r}\nCFG = {cfg!r}\n{show}{PIPE_DRIVER}"
    else:
        recs = [json.loads(x) for x in open(a.force, encoding="utf-8")]
        ids = {r["id"] for r in recs}
        rows = [r for r in csv.DictReader(io.StringIO(cases)) if r["id"] in ids]  # only the cases the texts belong to
        assert len(rows) == len(ids), "every text needs its case in --csv"
        sub = io.StringIO()
        w = csv.DictWriter(sub, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
        force = "".join(json.dumps({k: r[k] for k in ("id", "k", "label", "text")}) + "\n" for r in recs)
        force_b64 = base64.b64encode(gzip.compress(force.encode("utf-8"))).decode()
        cfg = dict(MODEL, cut=a.cut, endings=ENDINGS)
        compile(ENDING, "ending.py", "exec")
        run_py = (f"{head}CASES = {sub.getvalue()!r}\nFORCE = {force_b64!r}\nPROGRAM = {ENDING!r}\nCFG = {cfg!r}\n"
                  f"{show}{ENDING_DRIVER}")
    write_kernel(a.out, a.owner, run_py)


if __name__ == "__main__":
    main()
