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

if command -v uv >/dev/null; then
  UV=uv
else  # a plain install first: pip refuses --user inside a virtualenv
  python3 -m pip install --quiet uv || python3 -m pip install --quiet --user uv ||
    python3 -m pip install --quiet --user --break-system-packages uv
  UV="python3 -m uv"
fi
$UV venv .venv --python 3.12 --allow-existing
# torch and vLLM are CUDA 12.9 builds from the PyTorch index; unsafe-best-match lets every other pin come from PyPI
$UV pip install --python .venv/bin/python -r requirements.txt --index-strategy unsafe-best-match

HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1 .venv/bin/hf download "$REPO" --revision "$REV" --local-dir "$MODEL_DIR"
du -sh "$MODEL_DIR"
.venv/bin/python -c "import torch, vllm, transformers; print('torch', torch.__version__, 'CUDA', torch.version.cuda, \
'GPUs', torch.cuda.device_count(), 'vLLM', vllm.__version__, 'transformers', transformers.__version__)"
