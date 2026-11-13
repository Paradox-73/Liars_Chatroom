# Body of run.py (build.py puts CFG and FILES above it). Runs inside a Kaggle session: installs the GPU server's
# vLLM build in a fresh Python 3.12 environment, downloads the model at a pinned revision, writes the judge package
# to /tmp/lc_src and runs it over the embedded id lists, retrying with safer engine settings if an attempt fails.
import glob
import os
import signal
import subprocess
import time

T0 = time.time()
SRC = "/tmp/lc_src"
PY = "/tmp/venv/bin/python"
# Same vLLM build and pins as scripts/server/setup_env.sh on the GPU server.
INSTALL = ("https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl "
           "torchcodec==0.16.0+cu129 --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match")
ATTEMPTS = [  # (extra judge arguments, extra environment), tried in order until one exits cleanly
    ("", {}),
    ("--eager", {}),
    ("--eager", {"VLLM_USE_TRITON_AWQ": "1", "VLLM_ATTENTION_BACKEND": "TRITON_ATTN"}),
]
START_LIMIT, STALL_LIMIT = 25 * 60, 45 * 60  # seconds without a heartbeat before an attempt counts as hung


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def sh(cmd):
    log(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def run_judge(cmd, env):
    """Runs one judge attempt and returns its exit code; kills it when its heartbeat file goes stale."""
    hb = os.path.join(SRC, "heartbeat")
    if os.path.exists(hb):
        os.remove(hb)
    log(f"$ {cmd}")
    p = subprocess.Popen(cmd, shell=True, cwd=SRC, env=env, start_new_session=True)
    start = time.time()
    while p.poll() is None:
        time.sleep(20)
        stale = time.time() - (os.path.getmtime(hb) if os.path.exists(hb) else start)
        if stale > (STALL_LIMIT if os.path.exists(hb) else START_LIMIT):
            log(f"no heartbeat for {stale / 60:.0f} min: killing the attempt")
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    return p.returncode


def main():
    for rel, text in FILES.items():
        path = os.path.join(SRC, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    phases = []
    for i, (split, ids, out, gate) in enumerate(CFG["phases"]):
        with open(f"{SRC}/phase{i}.txt", "w") as f:
            f.write("\n".join(ids) + "\n")
        phases.append(f"--phase {split}:{SRC}/phase{i}.txt:/kaggle/working/{out}" + (f":{gate}" if gate else ""))
        log(f"phase {i}: {split}, {len(ids)} ids -> {out}" + (f", accuracy gate {gate}" if gate else ""))
    sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv; nproc; free -g | head -2; df -h /dev/shm /tmp")
    sh("uv venv /tmp/venv --python 3.12")
    sh(f"uv pip install --python {PY} {INSTALL}")
    sh(f"{PY} -c 'import torch, vllm, transformers; print(torch.__version__, torch.version.cuda, torch.cuda.device_count(), "
       f"vllm.__version__, transformers.__version__)'")
    m = CFG["model"]
    sh(f"HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1 {PY} -c 'from huggingface_hub import snapshot_download; "
       f"snapshot_download(\"{m['repo']}\", revision=\"{m['revision']}\", local_dir=\"/tmp/models/{m['name']}\")' "
       f"&& du -sh /tmp/models/{m['name']}")
    data_dir = os.path.dirname(glob.glob("/kaggle/input/**/train.csv", recursive=True)[0])
    deadline = T0 + CFG["budget_hours"] * 3600
    base = f"{PY} -m kjudge --model /tmp/models/{m['name']} {' '.join(phases)} --deadline {deadline:.0f} {CFG['judge_args']}"
    # Kaggle's PYTHONPATH points at its own Python 3.13 packages, which must not leak into the 3.12 environment.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for extra_args, extra_env in ATTEMPTS:
        rc = run_judge(f"{base} {extra_args}", dict(env, LC_DATA_DIR=data_dir, **extra_env))
        log(f"judge exited with code {rc}")
        if rc == 0 or time.time() > deadline:
            break
    sh("ls -la /kaggle/working; wc -l /kaggle/working/*.jsonl || true")


main()
