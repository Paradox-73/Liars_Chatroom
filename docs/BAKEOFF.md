# Model bake-off (new-family cases)

Each run is the full `predict.py` with one candidate model on one Kaggle T4 notebook. Numbers come from the `SUMMARY` lines posted on issues #1 and #3 and the `summary.json` files in `bake_out/` / `explore_out/`. "Capped" is how many first answers hit the thinking cap (`main_capped`, or `first_forced` where only the SUMMARY line was posted).

## 300 new-family cases

| model | first think cap | account | accuracy | predict min | capped | cases/min | notes |
|---|---|---|---|---|---|---|---|
| DeepSeek-R1-0528-Qwen3-8B (fp16, 16 slots) | default | nirbhaysharma504 | 31.3% (94/300) | 53.4 | 11 of 16 | 5.6 | only 13 cases got a first answer, 291 preds were fallback |
| Qwen3-8B (fp16, 20 slots) | default | nirbhaysharma504 | 64.3% (193/300) | 53.4 | 5 of 115 | 5.6 | 112/115 right when it got to think, 184 answered without thinking |
| Qwen3-8B-AWQ (48 slots, 8k continuation) | 2,048 | paradox73 | 89.3% (268/300) | 50.6 | 162 forced | 5.9 | 44 answered without thinking |
| Qwen3-14B-AWQ | 4,096 | paradox73 | 95.7% (287/300) | 52.0 | - | 5.8 | picked as the winner, no engine stats in this run |
| Qwen3-14B-AWQ (bake-final v2) | 4,096 | nirbhaysharma504 | 93.0% (279/300) | 49.2 | 90 | 6.1 | confirmation of the final code, half the first samples only finished at 33 min |
| Qwen3-14B-AWQ | 3,072 | nirbhaysharma504 | 95.3% (286/300) | 50.1 | 42 | 6.0 | 5 continuations, all finished |
| Qwen3-14B-AWQ, no long continuations | 3,072 | nirbhaysharma504 | 95.7% (287/300) | 48.5 | 41 | 6.2 | leftover time goes to 5 extra tries, 1 answer without thinking |

The first bake-final run (0.927) is left out: it was built with a bug in the end-of-run time cap that cut thinking too early.

## Second new-family set (168 cases)

| model | first think cap | account | accuracy | predict min | capped | cases/min | notes |
|---|---|---|---|---|---|---|---|
| Qwen3-14B-AWQ | 4,096 | nirbhaysharma504 | 85.7% (144/168) | 28.9 | 17 | 5.8 | 41 answered without thinking (ran out of time) |
| Qwen3-14B-AWQ | 3,072 | nirbhaysharma504 | 91.1% (153/168) | 29.0 | 31 | 5.8 | without-thinking answers 41 -> 16 |

Per-family splits are in each run's `summary.json`.

**Takeaway:** the 8B models run out of think budget on a T4, so a lot of their answers are fallback or no-think, and Qwen3-14B in 4-bit is the clear winner at the same speed (~6 cases/min). When the 14B finishes thinking it's almost always right, so the misses are about time: lowering the first think cap from 4,096 to 3,072 gets +9 cases on the second set and loses nothing on the first one.
