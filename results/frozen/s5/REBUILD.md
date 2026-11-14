# Rebuild of s5: 4,096 cap, fallback final

s5 is the full `test.csv` run of the code submission at commit 9122462 (zip md5 `ec99a0111a78a8d6ff25e8c296c40986`)
on Kaggle with two T4 cards: no tail cap, a 4,096-token first sample, continuations to 16,384 tokens. It is the
fallback final file. The final code (3,072-token first sample, no continuations) came later.

Model answers used (frozen here):
- `test_qwen3-14b-awq_predict_4096.jsonl`: Qwen3-14B-AWQ through `submission/judge.py` on the 1,064 test cases the
  rules leave open. Every case has a first sample. 63 of them stopped at 4,096 tokens; 46 of those were continued to
  16,384 tokens and finished, and 17 stayed forced. The 5 answers without thinking are not used, because those cases
  also have a thinking answer.

Run: 3,500 cases; the rules answered 2,436 (8 of them by elimination). predict.py took 185.3 of its 189.2 minutes and
exited with code 0.

Command, from the repository root at commit 9122462 (`lc/` and `submission/predict.py` there are the same as at
8d46852):

```
python -c "import sys; sys.path.insert(0, 'submission'); import predict as p; \
ids, fixed, cases, cands = p.read_cases('<folder with the competition CSVs>/test.csv'); recs = {}; \
p.merge('results/frozen/s5/test_qwen3-14b-awq_predict_4096.jsonl', 0, recs); \
print(p.write_predictions('s5_rebuild.csv', ids, fixed, cands, recs))"
```
Expected md5 of the output: `abc490334be67eec600c21a3157f6fc3`, the md5 of the `submission.csv` the Kaggle run wrote
(checked: the rebuild matches). 18 labels differ from s3 and 9 from s4, all on cases the rules leave open.
