# Rebuild of s4: capped code, not final

s4 is the full `test.csv` run of the code submission at commit 8d46852 (zip md5 `aaab316e29b68dfdec2d5cd021a8064d`)
on Kaggle with two T4 cards. That code still had the tail cap, which commit 5697d56 removed, so s4 is kept for the
record only. The final file comes from the same run without the tail cap.

Model answers used (frozen here):
- `test_qwen3-14b-awq_predict_capped.jsonl`: Qwen3-14B-AWQ through `submission/judge.py` on the 1,064 test cases the
  rules leave open. Every case has a first sample: 65 stopped at 4,096 tokens and 2 were cut lower by the tail cap
  (1,819 and 1,884 tokens). 40 of those were continued to 16,384 tokens, and 27 stayed forced. The 2 answers without
  thinking are not used, because those cases also have a thinking answer.

Run: 3,500 cases; the rules answered 2,436 (8 of them by elimination). predict.py took 184.4 of its 189.2 minutes and
exited with code 0.

Command, from the repository root (`lc/` and `submission/predict.py` are the same at 8d46852 and 9122462):

```
python -c "import sys; sys.path.insert(0, 'submission'); import predict as p; \
ids, fixed, cases, cands = p.read_cases('<folder with the competition CSVs>/test.csv'); recs = {}; \
p.merge('results/frozen/s4/test_qwen3-14b-awq_predict_capped.jsonl', 0, recs); \
print(p.write_predictions('s4_rebuild.csv', ids, fixed, cands, recs))"
```
Expected md5 of the output: `442e35dbf56d8edb6a04481719247f43`, the md5 of the `submission.csv` the Kaggle run wrote
(checked: the rebuild matches). 23 labels differ from s3, all on cases the rules leave open.
