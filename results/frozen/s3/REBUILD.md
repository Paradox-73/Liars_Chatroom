# Rebuild of upload s3

Model answers used (all frozen here):
- `test_qwen35-4b_n1.jsonl`: Qwen3.5-4B on the first 1,066 ids of `docs/ids/test_llm_order.txt`, max 8,192 tokens
- `test_qwen35-4b_forced16k.jsonl`: Qwen3.5-4B again on the 190 of those ids where it hit the 8,192-token cap, now with 16,384 tokens
- `test_qwen35-4b_best.jsonl`: the first file with those 190 ids replaced by the second (no other change)
- `test_qwen3-32b-awq_kaggle.jsonl`: Qwen3-32B-AWQ on the same 1,066 ids (Kaggle 2x T4)
- `test_qwen3-14b-awq_kaggle.jsonl`: Qwen3-14B-AWQ on the first 1,003 ids (Kaggle 2x T4)
- `test_qwen3-8b_n1.jsonl`: Qwen3-8B-AWQ on 469 of the ids (its first 339 in file order, then the 4B's capped ids)

Command, from the repository root:

```
LC_DATA_DIR=<folder with the competition CSVs> python -m lc.combine --split test \
  --llm results/frozen/s3/test_qwen35-4b_best.jsonl@1.2 \
  --llm results/frozen/s3/test_qwen3-32b-awq_kaggle.jsonl@1.1 \
  --llm results/frozen/s3/test_qwen3-14b-awq_kaggle.jsonl@1.0 \
  --llm results/frozen/s3/test_qwen3-8b_n1.jsonl@0.9 \
  --out s3_rebuild.csv
```
Expected md5 of the output: `18aa031a59bc8b33d24d4617c6d36c93` (checked: the rebuild matches).
