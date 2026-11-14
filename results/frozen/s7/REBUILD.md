# Rebuild of s7: version 2, the final code

s7 is the full `test.csv` run of version 2 of the code submission (zip md5 `67c4f145d20fc5341ca0cc60e0bd083f`;
`submission/` and `lc/` as at commit c5b55b1, the same on main) on Kaggle with two T4 cards (Kaggle notebook
`predict-full-v2-nirbhay`): a 3,072-token first sample, no continuations, shortest prompt first, extra samples kept
running at the stop time. It is the final file.

Model answers used (frozen here):
- `test_qwen3-14b-awq_v2.jsonl`: Qwen3-14B-AWQ through `submission/judge.py` on the 1,252 test cases the rules leave
  open. Every case has a first sample (1,252 of 1,252); 137 of them stopped at 3,072 tokens and were forced to an
  answer. 263 extra samples were added. No case is answered without thinking.

Run: 3,500 cases; the rules answered 2,248 (4 of them by elimination). predict.py took 222.3 minutes (budget 222.6)
and exited with code 0.

Command, from the repository root on main:

```
python -c "import sys; sys.path.insert(0, 'submission'); import predict as p; \
ids, fixed, cases, cands = p.read_cases('<folder with the competition CSVs>/test.csv'); recs = {}; \
p.merge('results/frozen/s7/test_qwen3-14b-awq_v2.jsonl', 0, recs); \
print(p.write_predictions('s7_rebuild.csv', ids, fixed, cands, recs))"
```
Expected md5 of the output: `2cb35f3b2e402387c44206144154f30e`, the md5 of the `submission.csv` the Kaggle run wrote
(checked: the rebuild matches byte for byte). Labels: A 883, B 893, C 882, D 842. 15 labels differ from s6, 28 from s5
and 23 from s3.
