"""Combine the rule solver with language-model predictions.

Order of trust, measured on train:
  rule      exactly one speaker has a message the rules judge impossible
  elim      no rule fired, but every claim of three speakers was judged consistent and the fourth has
            claims nobody could judge: exactly one liar exists, so it is the fourth
  llm       the language-model judge's vote
  fallback  no source available (should not happen in a real run)

python -m lc.combine --split train --llm outputs/llm_train.jsonl      # accuracy per tier on train
python -m lc.combine --split test --llm outputs/llm_test.jsonl --out submissions/sub.csv
"""
import argparse
import collections
import csv
import json
import os

from lc.data import load
from lc.rules import solve
from lc.rules.core import LIE, OK
from lc.rules.text import is_filler

FALLBACK = "C"  # most common train label; only used when no source has an answer


def load_llm(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            letter = r.get("pred") or r.get("vote") or r.get("letter")
            if letter in ("A", "B", "C", "D"):
                out[r["id"]] = r
    return out


def eliminate(case, res):
    """The only speaker left with unjudged claims when everyone else's claims were judged consistent."""
    if res.lie_labels:
        return None
    open_speakers = set()
    for m in case.messages:
        if is_filler(m.text):
            continue
        vs = res.verdicts.get(m.num, [])
        if not vs or not all(v.status == OK for v in vs):
            open_speakers.add(m.label)
    return next(iter(open_speakers)) if len(open_speakers) == 1 else None


def decide(case, llm):
    res = solve(case)
    if res.pred:
        return res.pred, "rule_confident" if res.confident else "rule"
    e = eliminate(case, res)
    if e:
        return e, "elim"
    r = llm.get(case.id)
    if r:
        return (r.get("pred") or r.get("vote") or r.get("letter")), "llm"
    return FALLBACK, "fallback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--llm", default=None, help="JSONL of language-model predictions for this split")
    ap.add_argument("--out", default=None, help="write id,label CSV (test)")
    a = ap.parse_args()
    cases = load(a.split)
    llm = load_llm(a.llm)
    tiers = collections.Counter()
    right = collections.Counter()
    llm_right_on_tier = collections.Counter()
    rows = []
    for c in cases:
        pred, tier = decide(c, llm)
        rows.append((c.id, pred))
        tiers[tier] += 1
        if a.split == "train":
            right[tier] += pred == c.label
            r = llm.get(c.id)
            if r:
                llm_right_on_tier[tier] += (r.get("pred") or r.get("vote") or r.get("letter")) == c.label
    print(f"{a.split}: {len(cases)} cases, llm predictions for {len(llm)}")
    for t in ("rule_confident", "rule", "elim", "llm", "fallback"):
        if tiers[t]:
            line = f"  {t:15s} {tiers[t]:5d}"
            if a.split == "train":
                line += f"  accuracy {right[t] / tiers[t]:.2%}"
                if llm:
                    line += f"  (llm alone on these: {llm_right_on_tier[t] / tiers[t]:.2%})"
            print(line)
    if a.split == "train":
        print(f"  overall accuracy {sum(right.values()) / len(cases):.2%}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "label"])
            w.writerows(rows)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
