"""Writes one Kaggle script kernel per candidate judge model. Each kernel runs the code submission exactly as the
organisers will (setup.sh with internet, then predict.py) on validation/newfam/newfam.csv, our 300 labelled cases from
17 families that are in neither train nor test, with predict.py's own time budget for 300 cases (53 minutes, the
same time per case as 80 minutes for 450). At the end it prints one SUMMARY line: accuracy, first-sample accuracy,
capped answers, throughput, and accuracy per family.

python kaggle/build_bakeoff.py --zip ClaudeLaggye_code.zip --newfam <repo>/validation/newfam [--out kaggle]
"""
import argparse
import base64
import json
import os

# slug: (owner, Hugging Face repo, revision, judge.py environment)
KERNELS = {
    "bake-qwen3-8b-awq": ("paradox73", "Qwen/Qwen3-8B-AWQ", "4da05a8edb55c6046cce958586c33b61da07bb79", {"LC_SLOTS": "48"}),
    "bake-qwen35-4b-awq-dp": ("paradox73", "cyankiwi/Qwen3.5-4B-AWQ-4bit", "d3e78b50c727b1f34802f440ca84b25781625352",
                              {"LC_DATA_PARALLEL": "1", "LC_SLOTS": "48", "LC_MAX_LEN": "18432", "LC_PRESENCE_PENALTY": "1.5"}),
    "bake-qwen35-9b-awq": ("paradox73", "cyankiwi/Qwen3.5-9B-AWQ-4bit", "e4d2d39cede8c5d1716d824cd843e2889c648083",
                           {"LC_SLOTS": "40", "LC_PRESENCE_PENALTY": "1.5"}),
    "bake-r1-0528-qwen3-8b": ("nirbhaysharma504", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "6e8885a6ff5c1dc5201574c8fd700323f23c25fa",
                              {"LC_SLOTS": "16"}),
    "bake-qwen3-8b-fp16": ("nirbhaysharma504", "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", {"LC_SLOTS": "20"}),
    "bake-qwen35-4b-awq": ("bhavyajain1098", "cyankiwi/Qwen3.5-4B-AWQ-4bit", "d3e78b50c727b1f34802f440ca84b25781625352",
                           {"LC_SLOTS": "48", "LC_PRESENCE_PENALTY": "1.5"}),
    "bake-qwen3-14b-awq": ("bhavyajain1098", "Qwen/Qwen3-14B-AWQ", "31c69efc29464b6bb0aee1398b5a7b50a99340c3", {}),
    "bake-qwen3-14b-awq-v2": ("paradox73", "Qwen/Qwen3-14B-AWQ", "31c69efc29464b6bb0aee1398b5a7b50a99340c3",
                              {"LC_TOKEN_LOGPROBS": "0", "LC_MAIN_TOKENS": "4096", "LC_ENGINE_STATS": "1"}),
    "bake-final": ("nirbhaysharma504", "Qwen/Qwen3-14B-AWQ", "31c69efc29464b6bb0aee1398b5a7b50a99340c3", {"LC_ENGINE_STATS": "1"}),
    "bake-qwen3-8b-awq-v2": ("paradox73", "Qwen/Qwen3-8B-AWQ", "4da05a8edb55c6046cce958586c33b61da07bb79",
                             {"LC_SLOTS": "48", "LC_TOKEN_LOGPROBS": "0", "LC_MAIN_TOKENS": "2048", "LC_CONT_TOKENS": "8192",
                              "LC_ENGINE_STATS": "1"}),
    "bake-qwen35-9b-fp16": ("aaryanantala", "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                            {"LC_SLOTS": "24", "LC_PRESENCE_PENALTY": "1.5"}),
    "bake-qwen35-4b-fp16": ("aaryanantala", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                            {"LC_SLOTS": "40", "LC_PRESENCE_PENALTY": "1.5"}),
}

DRIVER = r'''
import base64, collections, csv, glob, io, json, os, re, subprocess, time, zipfile

T0, SUB, OUT = time.time(), "/tmp/sub", "/kaggle/working"


def log(msg):
    print(f"[bake {(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def sh(cmd, check=True):
    log(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, executable="/bin/bash").returncode


zipfile.ZipFile(io.BytesIO(base64.b64decode(ZIP))).extractall(SUB)
with open("/tmp/newfam.csv", "w", encoding="utf-8", newline="") as f:
    f.write(NEWFAM)
sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv; nproc; free -g | head -2; df -h /tmp | tail -1")
t = time.time()
sh(f"LC_MODEL_REPO={CFG['repo']} LC_MODEL_REV={CFG['rev']} LC_MODEL_DIR=/tmp/model bash {SUB}/setup.sh")
setup_min = (time.time() - t) / 60
env = " ".join(f"{k}={v}" for k, v in CFG["env"].items())
t = time.time()
rc = sh(f"cd {SUB} && {env} python predict.py --input /tmp/newfam.csv --output {OUT}/newfam_pred.csv --model /tmp/model "
        f"--workdir /tmp/w 2>&1 | tee /tmp/predict.log; exit ${{PIPESTATUS[0]}}", check=False)
wall = (time.time() - t) / 60
sh(f"cat /tmp/w/answers* > {OUT}/newfam_answers.jsonl", check=False)  # one answers file per model process

labels = {r["id"]: r["label"] for r in csv.DictReader(io.StringIO(NEWFAM))}
family = {r["id"]: r["family"] for r in csv.DictReader(io.StringIO(META))}
pred = {}
if os.path.exists(f"{OUT}/newfam_pred.csv"):
    pred = {r["id"]: r["label"] for r in csv.DictReader(open(f"{OUT}/newfam_pred.csv"))}
recs, lines = {}, []
for path in sorted(glob.glob("/tmp/w/answers*")):
    lines += [json.loads(x) for x in open(path)]
for r in lines:
    old = recs.get((r["id"], r["k"]))
    if old is None or r["stage"] >= old["stage"]:
        recs[(r["id"], r["k"])] = r
first = {c: r for (c, k), r in recs.items() if k == 0}
done = sorted(r["min"] for r in lines if r["kind"] == "main")
text = open("/tmp/predict.log", errors="replace").read() if os.path.exists("/tmp/predict.log") else ""
ready = re.findall(r"engine ready after ([\d.]+) min", text)
fam = collections.defaultdict(lambda: [0, 0])
for i, lab in labels.items():
    fam[family.get(i, "?").split("+")[0]][0] += pred.get(i) == lab
    fam[family.get(i, "?").split("+")[0]][1] += 1
summary = {
    "model": CFG["repo"], "env": CFG["env"], "exit": rc, "setup_min": round(setup_min, 1), "predict_min": round(wall, 1),
    "engine_ready_min": [float(x) for x in ready], "restarts": text.count("MODEL PROCESS FAILED"),
    "accuracy": round(sum(pred.get(i) == lab for i, lab in labels.items()) / len(labels), 4),
    "first_samples": len(first), "first_correct": sum(r["pred"] == labels[c] for c, r in first.items()),
    "first_forced": sum(r["forced"][0] for r in first.values()),
    "first_forced_correct": sum(r["forced"][0] and r["pred"] == labels[c] for c, r in first.items()),
    "continued": sum(r["kind"] == "cont" for r in first.values()),
    "continued_finished": sum(r["kind"] == "cont" and not r["forced"][0] for r in first.values()),
    "extra_samples": sum(1 for (c, k) in recs if k > 0), "no_think_answers": sum(1 for (c, k) in recs if k < 0),
    "first_tokens_mean": round(sum(r["out_tokens"][0] for r in first.values()) / max(len(first), 1)),
    "first_done_min": {q: done[min(len(done) - 1, int(len(done) * q))] for q in (0.5, 0.9, 1.0)} if done else {},
    "per_family": {k: f"{v[0]}/{v[1]}" for k, v in sorted(fam.items())},
}
print("SUMMARY " + json.dumps(summary), flush=True)
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=1)
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--newfam", required=True, help="folder with newfam.csv and newfam_meta.csv")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--only", nargs="*", help="slugs to write (default: all)")
    a = ap.parse_args()
    with open(a.zip, "rb") as f:
        zip_b64 = base64.b64encode(f.read()).decode()
    newfam = open(os.path.join(a.newfam, "newfam.csv"), encoding="utf-8", newline="").read()
    meta = open(os.path.join(a.newfam, "newfam_meta.csv"), encoding="utf-8", newline="").read()
    for slug, (owner, repo, rev, env) in KERNELS.items():
        if a.only and slug not in a.only:
            continue
        cfg = {"repo": repo, "rev": rev, "env": env}
        run_py = f"ZIP = {zip_b64!r}\nNEWFAM = {newfam!r}\nMETA = {meta!r}\nCFG = {cfg!r}\n{DRIVER}"
        compile(run_py, "run.py", "exec")
        folder = os.path.join(a.out, slug)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "run.py"), "w", encoding="utf-8", newline="\n") as f:
            f.write(run_py)
        meta_json = {"id": f"{owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
                     "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                     "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
                     "model_sources": []}
        with open(os.path.join(folder, "kernel-metadata.json"), "w") as f:
            json.dump(meta_json, f, indent=2)
        print(f"{folder}: {owner}, {repo}@{rev[:8]}, {env}")


if __name__ == "__main__":
    main()
