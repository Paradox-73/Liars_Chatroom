# Rebuild of s6: the final code, 3,072-token first sample, no continuations

s6 is the full `test.csv` run of the final code submission at commit 62523c8 (zip md5 `2b05254abb9e645a0cad64433c4122c7`)
on Kaggle with two T4 cards (Kaggle notebook `predict-full-final`): a 3,072-token first sample, no continuations,
no tail cap. It is the final file.

Model answers used (frozen here):
- `test_qwen3-14b-awq_predict_3072.jsonl`: Qwen3-14B-AWQ through `submission/judge.py` on the 1,064 test cases the
  rules leave open. Every case has a first sample (1,064 of 1,064); 101 of them stopped at 3,072 tokens and were
  forced to an answer. 194 extra samples were added. No case is answered without thinking.

Run: 3,500 cases; the rules answered 2,436 (8 of them by elimination). predict.py took 184.2 minutes (budget 189.2)
and exited with code 0; the whole notebook step took 186.9 minutes.

Command, from the repository root at commit 62523c8 (`lc/` and `submission/predict.py` at later commits are the same;
only the README and `team_info.json` changed):

```
python -c "import sys; sys.path.insert(0, 'submission'); import predict as p; \
ids, fixed, cases, cands = p.read_cases('<folder with the competition CSVs>/test.csv'); recs = {}; \
p.merge('results/frozen/s6/test_qwen3-14b-awq_predict_3072.jsonl', 0, recs); \
print(p.write_predictions('s6_rebuild.csv', ids, fixed, cands, recs))"
```
Expected md5 of the output: `86ef374890c2aa1ab4be4340ab9bceb3`, the md5 of the `submission.csv` the Kaggle run wrote
(checked: the rebuild matches byte for byte). Labels: A 882, B 892, C 878, D 848. 14 labels differ from s3, 24 from s4
and 21 from s5.
