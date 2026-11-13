"""Scores judge output on train cases: accuracy, confusion, accuracy by number of messages, output tokens.

Example: LC_DATA_DIR=data python -m lc.llm.eval outputs/x.jsonl --ids configs/dev200.txt
"""
import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict

from lc.data import load

# Rough evidence types from words in the evidence; a case can have several. Only for reading the scores.
EVIDENCE_TYPES = {
    "clock times": r"\d:\d\d [AP]M|\bnoon\b|\bmidnight\b",
    "weekdays": r"\b(Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\b|\bweek",
    "dates, ages": r"\bborn\b|\byears? old\b|\baged?\b|\b(19|20)\d\d\b",
    "speeds, distances": r"km/h|\bkm\b|\bmiles?\b|\bmph\b|\bspeed\b",
    "rates": r"\bper (second|minute|hour|day)\b|\bdrain|\bsteady\b",
    "counts, quantities": r"\bheld\b|\badded\b|\bremoved\b|\bcount",
    "lengths, weights": r"\b(centi|milli)?met(re|er)s?\b|\bcm\b|\bkg\b|\b(kilo)?grams?\b|\btonnes?\b|\bweigh",
    "ownership": r"\bsold\b|\bowned\b|\bbought\b",
    "rotas, bookings": r"\brota\b|\bshifts?\b|\bbook(ed|ing)\b",
    "logs, lists": r"\blogs?\b|\brecords\b|\blist\b|\bregister\b",
    "access rules": r"\bdoors?\b|\bcodes?\b|\bunlock|\bwristband|\bseal|\bgates?\b|\bkeys?\b",
    "places, travel": r"\bseparate locations?\b|\broutes?\b|\btravel",
    "relative times": r"\b(minutes|hours?) (after|before)\b",
}


def evidence_types(evidence):
    ev = re.sub(r"^All times in this case refer to [^.]*\.", "", evidence)
    return [t for t, rx in EVIDENCE_TYPES.items() if re.search(rx, ev, re.I)] or ["other"]


def p95(xs):
    return sorted(xs)[math.ceil(0.95 * len(xs)) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", help="JSONL written by lc.llm.judge_vllm")
    ap.add_argument("--ids", help="score only these ids (one per line)")
    args = ap.parse_args()

    train = load("train")
    label = {c.id: c.label for c in train}
    n_msgs = {c.id: len(c.messages) for c in train}
    ev_types = {c.id: evidence_types(c.evidence) for c in train}
    with open(args.preds) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if args.ids:
        with open(args.ids) as f:
            keep = set(f.read().split())
        recs = [r for r in recs if r["id"] in keep]
    recs = [r for r in recs if r["id"] in label]
    if not recs:
        raise SystemExit("no train ids in the file")
    n = len(recs)
    ok = [r["pred"] == label[r["id"]] for r in recs]
    print(f"cases {n}, accuracy {sum(ok) / n:.3f} ({sum(ok)}/{n}), no answer {sum(r['pred'] is None for r in recs)}, "
          f"samples with a forced answer {sum(sum(r['forced']) for r in recs)}")

    k = len(recs[0]["letters"])
    if k > 1:
        single = [sum(r["letters"][i] == label[r["id"]] for r in recs) / n for i in range(k)]
        print("accuracy of each sample alone:", ", ".join(f"{a:.3f}" for a in single), f"(mean {statistics.mean(single):.3f})")
        agree = sum(len(set(r["letters"])) == 1 for r in recs)
        print(f"all {k} samples agree on {agree} cases, accuracy there "
              f"{sum(o for r, o in zip(recs, ok) if len(set(r['letters'])) == 1) / max(agree, 1):.3f}")

    conf = Counter((label[r["id"]], r["pred"] or "-") for r in recs)
    print("confusion (rows: true sender, columns: predicted, -: no answer)")
    print("      " + "".join(f"{p:>5}" for p in "ABCD-"))
    for t in "ABCD":
        print(f"  {t}   " + "".join(f"{conf[t, p]:>5}" for p in "ABCD-"))

    by = defaultdict(list)
    for r, o in zip(recs, ok):
        by[n_msgs[r["id"]]].append(o)
    print("accuracy by number of messages:")
    for m in sorted(by):
        print(f"  {m:>2}: {sum(by[m]) / len(by[m]):.3f} of {len(by[m])}")

    by_type = defaultdict(list)
    for r, o in zip(recs, ok):
        for t in ev_types[r["id"]]:
            by_type[t].append(o)
    print("accuracy by evidence type (from words in the evidence; a case can count under several types):")
    for t in sorted(by_type, key=lambda t: sum(by_type[t]) / len(by_type[t])):
        print(f"  {t:28s} {sum(by_type[t]) / len(by_type[t]):.3f} of {len(by_type[t])}")

    per_case = [sum(r["out_tokens"]) for r in recs]
    per_sample = [t for r in recs for t in r["out_tokens"]]
    print(f"output tokens per case: mean {statistics.mean(per_case):.0f}, p95 {p95(per_case)}; "
          f"per sample: mean {statistics.mean(per_sample):.0f}, p95 {p95(per_sample)}, max {max(per_sample)}")


if __name__ == "__main__":
    main()
