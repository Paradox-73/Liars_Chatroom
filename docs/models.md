# Models and libraries used by the LLM judge

All models are open-weight and run locally on the GPU server. Nothing in the pipeline calls a hosted model API.
Weights are downloaded at the pinned Hugging Face revisions in `scripts/server/get_models.sh` (the 0.8B check model in `scripts/server/smoke_test.sh`).

## Models

| Model | Revision | Licence | Parameters | Role |
|---|---|---|---|---|
| [`Qwen/Qwen3-8B-AWQ`](https://huggingface.co/Qwen/Qwen3-8B-AWQ) | `4da05a8edb55c6046cce958586c33b61da07bb79` | Apache-2.0 | 8.2B, 4-bit AWQ (official) | judge for the train and test predictions |
| [`cyankiwi/Qwen3.5-4B-AWQ-4bit`](https://huggingface.co/cyankiwi/Qwen3.5-4B-AWQ-4bit) | `d3e78b50c727b1f34802f440ca84b25781625352` | Apache-2.0 | 4B, 4-bit AWQ of `Qwen/Qwen3.5-4B` | judge candidate, compared on 200 train cases |
| [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | `2fc06364715b967f1860aea9cf38778875588b17` | Apache-2.0 | 0.8B | install check only (`scripts/server/smoke_test.sh`) |

The base models are by the Qwen team (Alibaba Cloud): Qwen3 technical report, arXiv:2505.09388, and the Qwen3.5 model cards on Hugging Face.

## Libraries

| Library | Version | Licence |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | 0.32.0, CUDA 12.9 build | Apache-2.0 |
| [PyTorch](https://pytorch.org) | 2.13.0+cu129 | BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers) (tokenizer and chat template) | 5.19.0 | Apache-2.0 |
| [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) (loads the 4-bit checkpoints) | 0.17.0 | Apache-2.0 |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) (`hf download`) | 2.4.0 | Apache-2.0 |

The full list of installed packages is in `requirements.lock.txt` on the server, written by `scripts/server/setup_env.sh`.
