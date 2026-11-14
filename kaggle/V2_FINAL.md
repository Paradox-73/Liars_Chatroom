# v2 final runs: steps for Nirbhay

The code is branch `feat/predict-v2`. The kernels were built at commit 0dde465. `submission/` and `lc/` there are the same
as at c5b55b1, the commit that v2-set2 and v2-newfam ran. Zip md5: `67c4f145d20fc5341ca0cc60e0bd083f`. The first log line
of each kernel shows the commit and the md5. Check it.

## 1. Ship rule

- v2-set2 passed: 0.946 against 0.911, net +6, 1 case never reached.
- Push the kernels below only if v2-newfam scores **at least 286 of 300**. The baseline, cap3072x-newfam, scored 287.
- At 285 or lower, push nothing. s6 stays the final file.
- Push only one account's copies. If `paradox73/predict-full-v2` is already running, do not push yours.

## 2. Push

From the repository root, on `feat/predict-v2` after a pull:

```
kaggle kernels push -p kaggle/predict-full-v2-nirbhay
kaggle kernels push -p kaggle/predict-test-v2-nirbhay
```

- Together they take both of your GPU slots.
- `predict-full-v2-nirbhay` runs predict.py on all of test.csv and writes `submission.csv`.
  - Setup takes about 3 min.
  - predict.py then gets 80 min per 450 cases that the model judges, so about 195 min.
  - It must start by 15 Nov 03:00 IST.
- `predict-test-v2-nirbhay` takes about 1 h 45 min. It runs:
  - the kit selftest on 200 train cases;
  - 450 test cases that the rules leave open, with the 80-min budget;
  - 40 of those cases on one T4.
- If a push fails because the competition data is missing, accept the competition rules on Kaggle and push again.
- Status: `kaggle kernels status nirbhaysharma504/predict-full-v2-nirbhay`

## 3. Outputs

When a kernel is COMPLETE:

```
kaggle kernels output nirbhaysharma504/predict-full-v2-nirbhay -p final_out/predict-full-v2-nirbhay
kaggle kernels output nirbhaysharma504/predict-test-v2-nirbhay -p final_out/predict-test-v2-nirbhay
```

Commit both folders to `nirbhay-outputs`. The `.log` files need `git add -f`, because `*.log` is ignored.

- Full run: `submission.csv`, `answers_full.jsonl` and the log.
- Test run: `hidden450_predictions.csv`, `answers450.jsonl`, `one_t4_predictions.csv`, `answers_one_t4.jsonl` and
  the log.

On issue #6, paste these lines from each log, then the md5 of `submission.csv`:

- the `code:` line
- `[predict ...] N cases: {...}`
- `wrote ... predictions`
- `model: ...`
- the selftest result lines

## 4. Check submission.csv

```
kaggle competitions download the-liars-chatroom -f sample_submission.csv -p data
LC_DATA_DIR=data python -c "from lc.submit import check; p, c = check('final_out/predict-full-v2-nirbhay/submission.csv'); print(p or 'valid', dict(sorted(c.items())))"
```

If the download arrives zipped, unzip it into `data/` first. `valid` means:

- the header is `id,label`;
- every id of the sample submission appears once;
- every label is A, B, C or D.

For comparison, s6 had A 882, B 892, C 878, D 848.

Do not submit the file to the competition: Kanav does that.
