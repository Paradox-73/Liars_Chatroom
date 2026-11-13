# Rebuild of upload s2 (public 1.00000)

Inputs frozen here are the exact model answers the upload used:
- `test_qwen35-4b_n1.jsonl`: Qwen3.5-4B on the 1,066 test ids that start `docs/ids/test_llm_order.txt` (1,066 lines)
- `test_qwen3-8b_n1.first210.jsonl`: the first 210 lines of the Qwen3-8B output, which was all the 8B had written at upload time (the file is append-only)

Command, from the repository root with the rules at commit `f6b52de` or later (the rule code has not changed since):

```
LC_DATA_DIR=<folder with the competition CSVs> python -m lc.combine --split test \
  --llm results/frozen/s2/test_qwen35-4b_n1.jsonl@1.1 \
  --llm results/frozen/s2/test_qwen3-8b_n1.first210.jsonl@1 \
  --out s2_rebuild.csv
```

Expected md5 of the output: `32d33d2a8a1485ba12955a459e4e09b4`, the same as the uploaded file. Checked: identical.
