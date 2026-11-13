# Model bake-off (new-family cases)

Each run is the full `predict.py` with one candidate model on the 300 new-family test cases, on one Kaggle T4 notebook. Numbers come from the `SUMMARY` lines posted on issue #1 and the `summary.json` files in `bake_out/`.

| model | account | new-family accuracy | predict min | capped first answers | cases/min | notes |
|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B (fp16, 16 slots) | nirbhaysharma504 | 31.3% (94/300) | 53.4 | 11 of 16 | 5.6 | only 13 cases got a first answer, 291 preds were fallback |
| Qwen/Qwen3-8B (fp16, 20 slots) | nirbhaysharma504 | 64.3% (193/300) | 53.4 | 5 of 115 | 5.6 | 112/115 right when it got to think, 184 answered without thinking |
| Qwen3-14B-AWQ (4k first think) | paradox73 | 95.7% | - | - | - | winner, SUMMARY not posted in the thread, only the score |

Setup took ~3.2-3.5 min on top of the predict time for both posted runs. Per-family splits are in each `bake_out/<run>/summary.json`.

**Takeaway:** the 8B models run out of think budget on T4 (R1 almost never finishes a first answer, Qwen3-8B only on ~40% of cases), so their scores are mostly fallback. Qwen3-14B in 4-bit fits the same time budget and gets 95.7%, so that's the one in `predict.py`.
