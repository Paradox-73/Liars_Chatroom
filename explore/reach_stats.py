"""Accuracy split by how each case was answered (finished, forced, never reached) and by a meta column.

python explore/reach_stats.py <answers.jsonl> <pred.csv> <labelled csv> <meta csv> [column]
"""
import collections
import csv
import json
import sys

answers, pred_csv, cases_csv, meta_csv = sys.argv[1:5]
col = sys.argv[5] if len(sys.argv) > 5 else "family"
labels = {r["id"]: r["label"] for r in csv.DictReader(open(cases_csv, encoding="utf-8"))}
meta = {r["id"]: r for r in csv.DictReader(open(meta_csv, encoding="utf-8"))}
pred = {r["id"]: r["label"] for r in csv.DictReader(open(pred_csv, encoding="utf-8"))}
first = {}
for line in open(answers, encoding="utf-8"):
    r = json.loads(line)
    if r["k"] == 0 and (r["id"] not in first or r["stage"] >= first[r["id"]]["stage"]):
        first[r["id"]] = r
tab = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for cid, lab in labels.items():
    r = first.get(cid)
    how = "finished" if r and not r["forced"][0] else "forced" if r else "not reached"
    for key in (meta.get(cid, {}).get(col, "?"), "all"):
        tab[key][how][0] += pred.get(cid) == lab
        tab[key][how][1] += 1
for key in sorted(tab, key=lambda k: (k == "all", k)):
    print(f"{key:12s} " + "  ".join(f"{h} {v[0]}/{v[1]}" for h, v in sorted(tab[key].items())))
