"""Forced-answer accuracy by the cap the sample stopped at, from a pipeline kernel's newfam_answers.jsonl.

python explore/forced_stats.py <answers.jsonl> <labelled csv>
"""
import collections
import csv
import json
import sys

labels = {r["id"]: r["label"] for r in csv.DictReader(open(sys.argv[2], encoding="utf-8"))}
first = {}
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    if r["k"] == 0 and (r["id"] not in first or r["stage"] >= first[r["id"]]["stage"]):
        first[r["id"]] = r
bins = collections.defaultdict(lambda: [0, 0])
for cid, r in first.items():
    t = r["out_tokens"][0]
    key = ("finished" if not r["forced"][0] else
           "forced at 4096" if t >= 4096 else f"forced at {t // 512 * 512}-{t // 512 * 512 + 511}")
    bins[key][0] += r["pred"] == labels[cid]
    bins[key][1] += 1
for k in sorted(bins):
    print(f"{k:28s} {bins[k][0]:4d}/{bins[k][1]:<4d} {bins[k][0] / bins[k][1]:.3f}")
