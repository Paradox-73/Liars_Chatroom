# The Liar's Chatroom: code submission

**A rule solver that answers what it can prove, and Qwen3-14B-AWQ in thinking mode for the rest.**<br>
Team Claude Laggye. The rules, prompt and vote of our Kaggle entry, run in one command under the organisers' limits:
offline, two T4 cards, 90 minutes for about 450 cases.

## Run it

```
bash setup.sh                                                   # once, with internet
python predict.py --input hidden.csv --output predictions.csv   # offline
```

`setup.sh` builds `.venv` (Python 3.12 and the pinned packages of `requirements.txt`, with vLLM 0.32.0 and PyTorch
2.13.0 for CUDA 12.9) and downloads `Qwen/Qwen3-14B-AWQ` at revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3` to
`models/judge` (9.3 GB). `predict.py` needs only the standard library: it starts its model process with
`.venv/bin/python`, so any Python 3 runs it (tested with 3.10, 3.12 and Kaggle's default Python). Another weights folder:
`LC_MODEL_DIR=<folder>` for both scripts, or `--model <folder>` for `predict.py`. The NVIDIA driver must support
CUDA 12.9 (Kaggle's 580 does). In Docker, start the container with `--ipc=host` (or a large `--shm-size`): the engine
on two cards passes tensors between its processes through shared memory.

## How it works

1. **Rules.** `lc/rules` (27 families, standard library) parses the evidence and judges the claims it fully
   understands. A case is answered when exactly one speaker has a claim that cannot be true. A speaker whose claims
   were all judged consistent is cleared. On train the rules answer 2,257 of 2,500 cases with no error. Their answer
   and the cleared speakers count only when the rules read every evidence sentence (2,221 of those 2,257): a sentence
   no family reads can change what the rest means, so such a case goes to the model with no speaker cleared. On
   scenario families they were not written for they stay silent: on our 300 synthetic cases from 17 new families
   they answer none and misfire on none. A complete predictions file is written at once.
2. **Model.** `judge.py`, a child process, serves Qwen3-14B-AWQ with vLLM (tensor parallel over both cards, float16)
   on the cases the rules leave open: the prompt of `lc/llm/prompt.py`, thinking mode, temperature 0.6, top-p 0.95,
   top-k 20. Up to 40 cases are in flight, fewer when the KV cache would overflow. A free slot takes, in order:
   - the first sample of the next case, seed 0, shortest prompt first, at most 3,072 tokens (see Choosing the
     model). A long prompt means long reasoning, so the cases left without a first sample when time runs out are the
     slow ones, not the cheap ones that are almost always right once reached. Cutting the later cases shorter to fit
     the time costs accuracy: on our 300 new-family cases a cap that shared the remaining tokens out evenly scored
     0.930 against 0.957 without it (both with a 4,096-token first sample, longest prompt first);
   - an extra sample (seed 1, 2, ...) for the least certain case: smallest vote margin first, up to 5 samples a
     case, none once two clean samples agree.

   A sample without an `ANSWER:` line gets a forced answer: `ANSWER:` is appended and one token is read. A sample
   that reaches the cap is not continued: spare time goes to extra samples instead (see Choosing the model).
   `judge.py` can continue capped samples (`LC_CONT_TOKENS` above the cap), but `predict.py` sets it equal to the
   cap, which leaves the engine a context of 5,120 tokens on two cards. Our longest prompt (test and new-family
   files) is about 3,200 characters, so at most about 1,100 tokens.
3. **Vote.** `lc/combine.py` decides each case: the rule answer, then elimination (one speaker left with claims nobody
   could judge), then the vote of the samples over the speakers the rules did not clear, with forced answers at half
   weight. A case the model never reached gets a one-token answer without thinking, and failing that the letter C (or
   the first open speaker).

### Time

The clock starts with `predict.py`. A file of 450 to 1,500 cases gets 80 minutes; a smaller file the same time per case
(the kit's selftest with 200 cases gets 35.6 minutes); a larger one 80 minutes per 450 cases the model judges
(`test.csv`: about 190 minutes). No new model work starts in the last 5 minutes. Then every case without a finished
sample gets a one-token answer without thinking. While first samples are still waiting or running, extra samples stop
at once to give those answers their cache; when none are left, running extra samples go on until they end or time is
up. 90 seconds before the end, any sample still thinking stops if one-token answers are waiting. At the end the model
process is stopped and the final file is written. `--minutes` overrides it.

The 3,072-token cap on first samples is fixed: it does not shrink when time runs short. A case whose first sample has
not ended (finished, or stopped at the cap and forced) by the end gets the one-token answer, which is right far less
often (40% for Qwen3-8B on the new-family cases, against 95% with thinking). On two T4 cards that happened to none of
our 300 new-family cases and to 16 of the 168 cases of our second set (41 with a 4,096-token first sample).

### Failure handling

- The predictions file is complete from the first second and is rewritten every 30 s as answers arrive, by writing
  a new file and renaming it over the old one.
- A model process that exits with an error, does not start within 20 minutes, or stops making progress for 5
  minutes is stopped with all its engine processes and restarted with safer settings while time allows: no CUDA
  graphs, then Triton AWQ and attention kernels, then one card. Answers already written are kept.
- With one T4 the model runs on that card with 8 cases in flight. Without a GPU, vLLM or the weights, the rule
  answers and fallbacks stand, and the log says so.

## Files

| File | What it is |
|---|---|
| `predict.py` | entry point: rules, the model process, time limits, the vote, the predictions file; `MODEL_ENV` holds the judge's settings |
| `judge.py` | the model process: vLLM engine, scheduling, forced answers, one JSON line per answer |
| `lc/` | our package: `rules/` (solver), `llm/prompt.py` (prompt), `llm/judge_vllm.py` (answer parsing), `combine.py` (vote) |
| `setup.sh`, `requirements.txt` | environment and weights |
| `team_info.json` | team, model, hardware, statement |

## Choosing the model

The hidden cases come from scenario families in neither train nor test, so the rules will answer few of them and the
model does almost all the work. We wrote 300 labelled cases from 17 such families (`validation/newfam` in our
repository) and ran each candidate through this code on Kaggle's two T4 cards, with the time per case of the hidden
run (53 minutes for 300 cases). Accuracy counts every case, including those the model did not reach in time.

| Judge on two T4 cards | New-family accuracy | Notes |
|---|---|---|
| Qwen3-14B-AWQ, 4,096-token first pass (this submission's model) | **0.957** (287 of 300) | 297 cases reached; 150 to 175 tokens/s |
| Qwen3-8B-AWQ, 2,048-token first pass, continued to 8,192 | 0.893 | 299 cases reached, 199 of them capped |
| Qwen3-8B unquantised (fp16), 8,192-token cap | 0.643 | 115 cases reached; little room left for the KV cache |
| Qwen3-8B-AWQ, 8,192-token cap | 0.637 | 115 cases reached |
| DeepSeek-R1-0528-Qwen3-8B (fp16) | 0.313 | 13 cases reached: very long reasoning |
| Qwen3.5-4B-AWQ, one engine per card | stopped | 9 min to start, then 4 cases in 14 min |

T4 cards generate about 150 to 200 tokens per second for these models, whatever their size, so the number of tokens
a model writes per case decides how many cases it reaches. Qwen3-14B writes about half as many as Qwen3-8B.

**First-sample length.** With the model chosen, we compared a 3,072-token first sample with 4,096 on Kaggle's two T4
cards, same code, same time per case. We used a second labelled set of 168 cases: 153 from nine more families, written
apart from the first set and checked claim by claim against every world the evidence allows (`validation/set2` in our
repository), and 15 from a teammate's own set. There 3,072 scored 0.911 against 0.857: 10 cases better, 1 worse
(two-sided sign test, p = 0.012), mostly because more cases got a thinking answer in time (152 of 168, against 127).
On the first set both scored 0.953 (5 cases better, 5 worse); the 0.957 above is an earlier run at 4,096. This
submission uses the 3,072-token first sample.

**No continuations.** Continuing a capped sample rarely paid on T4 cards. On 49 hard cases whose first sample hit its
cap, continuing to 8,192 tokens gave a net 3 more right answers (p = 0.51) and took 2,595 s on two cards. A fresh
4,096-token sample on those cases finished 29 times out of 78 and was right 28 of those 29 times. So a capped sample
is forced at once and spare time goes to extra samples. With the 3,072-token first sample and no continuations, the
first set scored 0.957 (299 of 299 cases reached, 41 forced, 5 extra samples). On the second set no spare time is
left for continuations either way.

**Size limit.** Update 2 lowered the limit for the final code check to 14B. The organisers then clarified on the
Discussion tab: the limit is 15B total parameters, including embeddings, as listed on the model's official page, and a
quantised build of an allowed model, such as a 4-bit AWQ, counts by the original model's total. Qwen3-14B has 14.8B
total parameters on its official page.

## Tested

The rows below ran this code (commit 62523c8: 3,072-token first sample, no continuations) unless a row says otherwise.

| Check | Result |
|---|---|
| Kit `selftest.py --n 200`, Kaggle, two T4 cards | OK, accuracy 1.000, 5.4 min (the kit's estimate for 450: 12.0 min); 12 cases reached the model |
| 450 test cases the rules leave open (the closest stand-in for new families), Kaggle, two T4 cards, cold start | 75.0 min including model load (engine ready at 2.3 min), exit 0; the limit is 90. All 449 cases the model judged got a thinking answer: 35 first samples stopped at 3,072 tokens and were forced, and 92 extra samples went to 66 uncertain cases; none needed the answer without thinking. Mean 1,413 output tokens per first sample |
| 40 of those cases on one T4 (15-minute budget) | 11.9 min, exit 0. Cache room for 6 cases in flight, so one card writes about a third as many tokens a minute as two: 19 of 40 cases got a thinking answer (2 forced) and 21 the answer without thinking |
| The 300 new-family cases, Kaggle, two T4 cards | 0.957 (see No continuations): commit 5697d56 with these settings from the environment, which runs the same on two cards |
| Kit `selftest.py --n 50` on a laptop without a GPU | OK, 0.980 in 0.9 s (rules and fallbacks only) |
| GPU server (A6000, Qwen3-8B as a stand-in) | Earlier commits: restarts after a failed engine, stop-time answers and continuations all ran, and the final file was valid each time. This code on one card, 40 new-family cases: 12 capped samples forced, 13 extra samples, no continuation tried, exit 0 |

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
