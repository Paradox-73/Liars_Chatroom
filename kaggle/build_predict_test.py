"""Writes a Kaggle script kernel that tests the code submission on two T4 cards exactly as the organisers will run it:
the zip from submission/make_zip.py is unpacked, setup.sh runs with internet, and then, with the network cut when the
session allows it, (1) the organisers' selftest.py on 200 train cases, (2) predict.py on the first 450 ids of
docs/ids/test_llm_order.txt, the test cases the rules leave open, and (3) predict.py on the first 40 of them with one
card. Test rows are only written to files in /tmp, never printed or saved as output.
With --full the kernel instead runs predict.py on all of test.csv and saves the predictions as submission.csv.

python kaggle/build_predict_test.py kaggle/predict-test --zip ClaudeLaggye_code.zip --selftest <kit>/selftest.py --owner <user>
python kaggle/build_predict_test.py kaggle/predict-full --full --zip ClaudeLaggye_code.zip --selftest <kit>/selftest.py --owner <user>
"""
import argparse
import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DRIVER = r'''
import base64, csv, glob, io, os, subprocess, time, zipfile

T0, SUB, OUT = time.time(), "/tmp/sub", "/kaggle/working"


def log(msg):
    print(f"[test {(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def sh(cmd, check=True):
    log(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check).returncode


zipfile.ZipFile(io.BytesIO(base64.b64decode(ZIP))).extractall(SUB)
with open(f"{SUB}/selftest.py", "w") as f:
    f.write(SELFTEST)
sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv; nproc; free -g | head -2; df -h /tmp | tail -1")
t = time.time()
sh(f"bash {SUB}/setup.sh")
log(f"setup.sh took {(time.time() - t) / 60:.1f} min")
data = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])

with open("/tmp/offline.sh", "w") as f:  # inside a new network namespace: bring the loopback up, as on any offline machine
    f.write("#!/bin/sh\nip link set lo up 2>/dev/null || python3 -c \"import fcntl, socket, struct; s = socket.socket(2, 2); "
            "f = struct.unpack('16sH', fcntl.ioctl(s, 0x8913, struct.pack('16sH', b'lo', 0)))[1]; "
            "fcntl.ioctl(s, 0x8914, struct.pack('16sH', b'lo', f | 1))\"\nexec \"$@\"\n")
os.chmod("/tmp/offline.sh", 0o755)
loopback = "python3 -c \"import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); socket.create_connection(s.getsockname(), 3)\""
internet = "python3 -c \"import socket; socket.create_connection(('pypi.org', 443), 5)\""
if (sh(f"unshare -rn /tmp/offline.sh {loopback}", check=False) == 0
        and sh(f"unshare -rn /tmp/offline.sh {internet}", check=False) != 0):
    offline = "unshare -rn /tmp/offline.sh "
    log("the runs below have no network at all (a new, empty network namespace)")
else:
    offline = "HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 "
    log("WARNING: the network cannot be cut here; the runs below only have HF_HUB_OFFLINE=1 and a dead proxy")

if FULL:
    t = time.time()
    rc = sh(f"cd {SUB} && {offline}python predict.py --input {data}/test.csv --output {OUT}/submission.csv --workdir /tmp/wfull",
            check=False)
    log(f"predict.py on all of test.csv: exit code {rc}, {(time.time() - t) / 60:.1f} min")
    sh(f"cp /tmp/wfull/answers.jsonl {OUT}/answers_full.jsonl; wc -l {OUT}/submission.csv", check=False)
    raise SystemExit(0)

sh(f"cp {data}/train.csv {SUB}/train.csv")
t = time.time()
sh(f"cd {SUB} && {offline}python selftest.py --train train.csv --n 200", check=False)
log(f"selftest on 200 train cases took {(time.time() - t) / 60:.1f} min")

sh("rm -rf ~/.cache/vllm ~/.triton; sync; echo 3 > /proc/sys/vm/drop_caches", check=False)  # a cold start, as for the organisers
cols = ["id", "scenario", "speakers", "evidence", "chat"]
with open(f"{data}/test.csv", encoding="utf-8", newline="") as f:
    rows = {r["id"]: r for r in csv.DictReader(f)}
for name, ids in (("hidden450", IDS), ("hidden40", IDS[:40])):
    with open(f"/tmp/{name}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows([rows[i][c] for c in cols] for i in ids)
log(f"/tmp/hidden450.csv: {len(IDS)} test cases")
t = time.time()
rc = sh(f"cd {SUB} && {offline}python predict.py --input /tmp/hidden450.csv --output {OUT}/hidden450_predictions.csv "
        f"--workdir /tmp/w450", check=False)
log(f"predict.py on 450 test cases: exit code {rc}, {(time.time() - t) / 60:.1f} min")
sh(f"cp /tmp/w450/answers.jsonl {OUT}/answers450.jsonl; wc -l {OUT}/hidden450_predictions.csv {OUT}/answers450.jsonl", check=False)

t = time.time()  # one card only: tensor parallel 1, smaller batch and context
rc = sh(f"cd {SUB} && CUDA_VISIBLE_DEVICES=0 {offline}python predict.py --input /tmp/hidden40.csv "
        f"--output {OUT}/one_t4_predictions.csv --workdir /tmp/w1 --minutes 15", check=False)
log(f"predict.py on 40 test cases with one T4: exit code {rc}, {(time.time() - t) / 60:.1f} min")
sh(f"cp /tmp/w1/answers.jsonl {OUT}/answers_one_t4.jsonl", check=False)
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--selftest", required=True)
    ap.add_argument("--ids", default=os.path.join(HERE, "..", "docs", "ids", "test_llm_order.txt"))
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--owner", default="YOUR_KAGGLE_USERNAME")
    ap.add_argument("--full", action="store_true", help="predict.py on all of test.csv, saved as submission.csv")
    a = ap.parse_args()
    with open(a.ids) as f:
        ids = f.read().split()[:a.n]
    with open(a.zip, "rb") as f:
        zip_b64 = base64.b64encode(f.read()).decode()
    with open(a.selftest, encoding="utf-8") as f:
        selftest = f.read()
    run_py = f"ZIP = {zip_b64!r}\nSELFTEST = {selftest!r}\nIDS = {ids!r}\nFULL = {a.full!r}\n{DRIVER}"
    compile(run_py, "run.py", "exec")
    slug = os.path.basename(os.path.normpath(a.kernel))
    meta = {"id": f"{a.owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "competition_sources": ["the-liars-chatroom"], "kernel_sources": [], "model_sources": []}
    os.makedirs(a.kernel, exist_ok=True)
    with open(os.path.join(a.kernel, "run.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(run_py)
    with open(os.path.join(a.kernel, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{a.kernel}/run.py: zip {len(zip_b64) * 3 // 4 // 1024} KiB, {len(ids)} test ids")


if __name__ == "__main__":
    main()
