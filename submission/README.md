# The Liar's Chatroom: code submission

**A rule solver that answers what it can prove, and Qwen3-14B-AWQ in thinking mode for the rest.**<br>
Team Claude Laggye. The same rules, prompt, sampling and vote as our Kaggle submissions (repository README), run in one
command under the organisers' limits: offline, two T4 cards, 90 minutes for about 450 cases.

## Run it

```
bash setup.sh                                                   # once, with internet
python predict.py --input hidden.csv --output predictions.csv   # offline
```

`setup.sh` builds `.venv` (Python 3.12 and the pinned packages of `requirements.txt`, with vLLM 0.32.0 and PyTorch
2.13.0 for CUDA 12.9) and downloads `Qwen/Qwen3-14B-AWQ` at revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3` to
`models/qwen3-14b-awq` (9.3 GB). `predict.py` needs only the standard library: it starts its model process with
`.venv/bin/python`, so any `python` 3.8 or newer runs it. Another weights folder: `--model <folder>` or
`LC_MODEL_DIR=<folder>`, for both scripts. The NVIDIA driver must support CUDA 12.9 (Kaggle's 580 does).

## How it works

1. **Rules.** `lc/rules` (27 families, standard library) parses the evidence and judges the claims it fully
   understands. A case is answered when exactly one speaker has a claim that cannot be true. A speaker whose claims
   were all judged consistent is cleared. On train the rules answer 2,257 of 2,500 cases with no error. On families
   they were not written for they are built to stay silent. A complete predictions file is written at once.
2. **Model.** `judge.py`, a child process, serves Qwen3-14B-AWQ with vLLM (tensor parallel over both cards, float16)
   and judges the cases the rules leave open: the prompt of `lc/llm/prompt.py`, thinking mode, temperature 0.6,
   top-p 0.95, top-k 20. It keeps 40 cases in flight, and a free slot takes, in order:
   - a sample that stopped at 8,192 tokens while still thinking, which continues to 16,384 tokens in total. It
     goes first only while the first samples are on schedule to finish in time;
   - the first sample of the next case, seed 0, at most 8,192 tokens. Cases go longest prompt first: on our
     earlier test answers a long prompt meant long reasoning (correlation 0.51), so slow cases start early;
   - an extra sample (seed 1, 2, ...) for the least certain case: smallest vote margin first, up to 5 samples a
     case, none once the margin reaches 1.5 (two agreeing answers).

   A sample without an `ANSWER:` line gets a forced answer: `ANSWER:` is appended and one token is read.
3. **Vote.** `lc/combine.py` decides each case: the rule answer, then elimination (one speaker left with claims nobody
   could judge), then the weighted vote of the samples over the speakers the rules did not clear, with forced
   answers at half weight. A case the model never reached gets its one-token answer without thinking, and failing
   that the letter C (or the first open speaker).

### Time

The clock starts with `predict.py`. For 450 to 600 cases the budget is 80 minutes; any other size gets the same time
per case (selftest's 200 cases get 35.6 minutes). No new model work starts in the last 5 minutes. Then every case
without a finished sample gets a one-token answer without thinking. At the end of the budget the model process is
stopped and the final file is written.

### Failure handling

- The predictions file is complete from the first second and is rewritten every 30 s as answers arrive, by writing
  a new file and renaming it over the old one.
- A model process that exits with an error, does not start within 20 minutes, or stops making progress for 5
  minutes is stopped and restarted with safer settings while time allows: no CUDA graphs, then Triton AWQ and
  attention kernels, then one card. Answers already written are kept.
- With one T4 the model runs on that card with 8 cases at a time and 8,192 tokens per sample. Without a GPU, vLLM or
  the weights, the rule answers and fallbacks stand, and the log says so.

## Files

| File | What it is |
|---|---|
| `predict.py` | entry point: rules, the model process, time limits, the vote, the predictions file |
| `judge.py` | the model process: vLLM engine, scheduling, forced answers, one JSON line per answer |
| `lc/` | our package: `rules/` (solver), `llm/prompt.py` (prompt), `llm/judge_vllm.py` (answer parsing), `combine.py` (vote) |
| `setup.sh`, `requirements.txt` | environment and weights |
| `team_info.json` | team, model, hardware, statement |

## Tested

| Check | Result |
|---|---|
| `selftest.py --n 200`, Kaggle, two T4 cards, network cut | to be filled |
| 450 test cases the rules leave open, Kaggle, two T4 cards | to be filled |
| `selftest.py --n 50` on a laptop without a GPU | OK, 0.980 (rules and fallbacks only) |

## Models and licences

| Component | Use | Licence |
|---|---|---|
| [Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ), Qwen team, 14.8B parameters, 4-bit AWQ | the judge | Apache-2.0 |
| [vLLM](https://github.com/vllm-project/vllm) 0.32.0 | serving | Apache-2.0 |
| [PyTorch](https://pytorch.org) 2.13.0 | tensors | BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers) 5.19.0 | tokenizer, chat template | Apache-2.0 |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) 2.4.0 | weight download in `setup.sh` | Apache-2.0 |

Qwen Team, Qwen3 Technical Report, 2025, https://arxiv.org/abs/2505.09388. Kwon et al., Efficient Memory Management
for Large Language Model Serving with PagedAttention, SOSP 2023. Lin et al., AWQ, MLSys 2024.
