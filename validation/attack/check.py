"""Run predict.py's own rule step (read_cases, model off) on a labelled CSV and list every rule answer that is wrong
and every case where the rules cleared the true liar (so the model could never pick them).

python validation/attack/check.py validation/attack/cases.csv [--show]
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "submission"))
sys.path.insert(0, ROOT)
import predict as p  # noqa: E402
from lc.data import case_from_row  # noqa: E402
from lc.rules import solve  # noqa: E402
from lc.rules.run import show  # noqa: E402

path = sys.argv[1]
labels = {r["id"]: r["label"] for r in csv.DictReader(open(path, encoding="utf-8-sig"))}
rows = {r["id"]: r for r in csv.DictReader(open(path, encoding="utf-8-sig"))}
ids, fixed, cases, cands = p.read_cases(path)
bad = 0
for cid in ids:
    lab = labels[cid]
    if cid in fixed:
        letter, tier = fixed[cid]
        status = "OK" if letter == lab else "WRONG"
        print(f"{cid}: {tier} -> {letter} (label {lab}) {status}")
    else:
        status = "OK" if lab in cands[cid] else "LIAR-CLEARED"
        print(f"{cid}: model, open {cands[cid]} (label {lab}) {status}")
    if status != "OK":
        bad += 1
        if "--show" in sys.argv:
            c = case_from_row({k.lower(): v for k, v in rows[cid].items()})
            show(c, solve(c))
print(f"{len(ids)} cases, {len(fixed)} rule answers, {bad} problems")
