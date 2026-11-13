#!/usr/bin/env bash
# Builds ~/synergy_liars/.venv with vLLM on the GPU server.
# The PyPI vLLM wheels need CUDA 13 (driver 580 or newer); this server has driver 570 (CUDA 12.8),
# so the CUDA 12.9 build of vLLM 0.32.0 from the GitHub release is used with torch from the cu129 index.
# torchcodec is pinned to its cu129 build: the PyPI 0.17.0 wheel needs libnvrtc.so.13 and breaks `import vllm`.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || python3 -m pip install --user uv
cd ~/synergy_liars
uv venv .venv --python /usr/bin/python3.12
source .venv/bin/activate
uv pip install \
  "https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  "torchcodec==0.16.0+cu129" \
  --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
python -c "import torch, vllm; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), vllm.__version__)"
uv pip freeze > requirements.lock.txt
bash scripts/server/smoke_test.sh
