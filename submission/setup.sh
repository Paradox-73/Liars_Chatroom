#!/usr/bin/env bash
# Run once, with internet, from any folder: bash setup.sh
# 1. builds .venv beside this file: Python 3.12 and every package of requirements.txt at its pinned version
# 2. downloads the judge model at its pinned revision from Hugging Face to models/judge beside this file, or to
#    $LC_MODEL_DIR when it is set (then give predict.py the same folder with --model or LC_MODEL_DIR)
# predict.py starts its model process with .venv/bin/python, so `python predict.py ...` works with any Python 3.
set -eu  # no pipefail: sh (dash) has none, and this script has no pipes
cd "$(dirname "$0")"
REPO=${LC_MODEL_REPO:-Qwen/Qwen3-14B-AWQ}
REV=${LC_MODEL_REV:-31c69efc29464b6bb0aee1398b5a7b50a99340c3}
MODEL_DIR=${LC_MODEL_DIR:-models/judge}

retry() {  # a download that fails gets two more tries, 30 s apart
  "$@" || { sleep 30; "$@"; } || { sleep 30; "$@"; }
}

if command -v uv >/dev/null; then
  UV=uv
else  # a plain install first: pip refuses --user inside a virtualenv
  python3 -m pip install --quiet uv || python3 -m pip install --quiet --user uv ||
    python3 -m pip install --quiet --user --break-system-packages uv
  UV="python3 -m uv"
fi
# a Python that uv downloads goes beside this file, so .venv/bin/python does not point into this user's home
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
$UV venv .venv --python 3.12 --allow-existing
# torch and vLLM are CUDA 12.9 builds from the PyTorch index; unsafe-best-match lets every other pin come from PyPI
retry $UV pip install --python .venv/bin/python -r requirements.txt --index-strategy unsafe-best-match

export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
retry .venv/bin/hf download "$REPO" --revision "$REV" --local-dir "$MODEL_DIR"
du -sh "$MODEL_DIR"
# the check starts CUDA on a card when there is one, so a driver that cannot run CUDA 12.9 fails here, not in predict.py
.venv/bin/python -c "import torch, vllm, transformers; print('torch', torch.__version__, 'CUDA', torch.version.cuda, \
'GPUs', torch.cuda.device_count(), 'vLLM', vllm.__version__, 'transformers', transformers.__version__); \
torch.cuda.device_count() and print('CUDA works:', torch.ones(1, device='cuda').sum().item() == 1)"
