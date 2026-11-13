"""Check a submission file against the sample submission and log it.

python -m lc.submit submissions/sub.csv --note "rules + llm v1"
Checks: header is exactly id,label; the ids are exactly the sample submission's ids, once each; every label is
A, B, C or D. Prints the md5 and the label counts, and appends a row to submissions/LOG.md.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import os
import subprocess
import sys

from lc.data import DATA_DIR


def check(path):
    with open(os.path.join(DATA_DIR, "sample_submission.csv"), newline="", encoding="utf-8") as f:
        want = [r["id"] for r in csv.DictReader(f)]
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = list(rd)
    problems = []
    if header != ["id", "label"]:
        problems.append(f"header is {header}")
    ids = [r[0] for r in rows]
    dup = [k for k, v in collections.Counter(ids).items() if v > 1]
    if dup:
        problems.append(f"{len(dup)} duplicate ids")
    if set(ids) != set(want):
        problems.append(f"missing {len(set(want) - set(ids))}, extra {len(set(ids) - set(want))}")
    bad = [r for r in rows if len(r) != 2 or r[1] not in ("A", "B", "C", "D")]
    if bad:
        problems.append(f"{len(bad)} rows with a bad label")
    return problems, collections.Counter(r[1] for r in rows if len(r) == 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    problems, counts = check(a.path)
    md5 = hashlib.md5(open(a.path, "rb").read()).hexdigest()
    print("md5", md5, "labels", dict(sorted(counts.items())))
    if problems:
        print("NOT VALID:", "; ".join(problems))
        sys.exit(1)
    print("valid")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    log = os.path.join(os.path.dirname(a.path) or ".", "LOG.md")
    new = not os.path.exists(log)
    with open(log, "a", encoding="utf-8") as f:
        if new:
            f.write("| time | file | md5 | commit | note | public LB |\n|---|---|---|---|---|---|\n")
        f.write(f"| {dt.datetime.now():%d %b %H:%M} | {os.path.basename(a.path)} | {md5[:8]} | {commit} | {a.note} | |\n")


if __name__ == "__main__":
    main()
