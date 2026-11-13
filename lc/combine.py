"""Combine the rule solver with language-model predictions.

Order of trust, measured on train:
  rule      exactly one speaker has a message the rules judge impossible
  elim      no rule fired, but every claim of three speakers was judged consistent and the fourth has
            claims nobody could judge: exactly one liar exists, so it is the fourth
  llm       weighted vote of the language-model judges, restricted to the speakers the rules did not clear
  fallback  no source available (should not happen in a real run)

Each --llm file is a JSONL with one line per case: id, pred, and optionally p_mean (letter -> probability)
and forced (per-sample flags for answers forced after the length limit). Give a weight per file with
--llm path@weight (default weight 1).

python -m lc.combine --split train --llm outputs/dev200_8b.jsonl --llm outputs/train_4b.jsonl
python -m lc.combine --split test --llm outputs/test_8b.jsonl --llm outputs/test_14b.jsonl@1.5 --out sub.csv
"""
import argparse
import collections
import csv
import json
import os

from lc.data import load
from lc.rules import solve
from lc.rules.core import OK
from lc.rules.text import is_filler

FALLBACK = "C"  # most common train label; only used when no source has an answer
LETTERS = ("A", "B", "C", "D")


def load_llm(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            letter = r.get("pred") or r.get("vote") or r.get("letter")
            if letter in LETTERS:
                out[r["id"]] = r
    return out


def open_speakers(case, res):
    """Speakers with at least one claim the rules did not judge consistent; only they can be the liar."""
    out = set()
    for m in case.messages:
        if is_filler(m.text):
            continue
        vs = res.verdicts.get(m.num, [])
        if not vs or not all(v.status == OK for v in vs):
            out.add(m.label)
    return out


def eliminate(case, res):
    """The only speaker left with unjudged claims when everyone else's claims were judged consistent."""
    if res.lie_labels:
        return None
    s = open_speakers(case, res)
    return next(iter(s)) if len(s) == 1 else None


def model_scores(r):
    """Letter -> probability for one model's record; one-hot on pred when no probabilities were saved."""
    p = r.get("p_mean")
    if isinstance(p, dict) and sum(p.values()) > 0:
        return {k: float(p.get(k, 0.0)) for k in LETTERS}
    letter = r.get("pred") or r.get("vote") or r.get("letter")
    return {k: float(k == letter) for k in LETTERS}


def vote(case_id, candidates, models):
    """Weighted sum of model probabilities over the candidate speakers. Returns (letter, n_models_used)."""
    total = collections.Counter()
    used = 0
    for weight, preds in models:
        r = preds.get(case_id)
        if not r:
            continue
        sc = model_scores(r)
        forced = r.get("forced") or []
        w = weight * (0.5 if forced and all(forced) else 1.0)  # a forced answer is a weaker vote
        mass = sum(sc[c] for c in candidates)
        if mass <= 0:
            continue  # this model only named speakers the rules already cleared
        for c in candidates:
            total[c] += w * sc[c] / mass
        used += 1
    if not used:
        return None, 0
    best = max(total.values())
    winners = [c for c in LETTERS if c in total and total[c] == best]
    if len(winners) > 1:  # tie: the highest-weight model that voted decides
        for weight, preds in sorted(models, key=lambda m: -m[0]):
            r = preds.get(case_id)
            if r:
                sc = model_scores(r)
                return max(winners, key=lambda c: sc[c]), used
    return winners[0], used


def decide(case, models):
    res = solve(case)
    if res.pred:
        return res.pred, "rule_confident" if res.confident else "rule"
    cands = open_speakers(case, res)
    if len(cands) == 1:
        return next(iter(cands)), "elim"
    letter, used = vote(case.id, sorted(cands) or list(LETTERS), models)
    if letter:
        return letter, f"llm{used}"
    return FALLBACK, "fallback"


def parse_llm_arg(s):
    """'path' or 'path@weight' (@ because Windows paths contain ':')."""
    path, _, w = s.rpartition("@") if "@" in s else (s, "", "")
    return path, float(w) if w else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--llm", action="append", default=[], help="JSONL of model predictions, optionally path@weight")
    ap.add_argument("--out", default=None, help="write id,label CSV (test)")
    a = ap.parse_args()
    cases = load(a.split)
    models = []
    for s in a.llm:
        path, w = parse_llm_arg(s)
        preds = load_llm(path)
        print(f"model file {path}: {len(preds)} predictions, weight {w}")
        models.append((w, preds))
    tiers, right = collections.Counter(), collections.Counter()
    rows = []
    for c in cases:
        pred, tier = decide(c, models)
        rows.append((c.id, pred))
        tiers[tier] += 1
        if a.split == "train":
            right[tier] += pred == c.label
    print(f"{a.split}: {len(cases)} cases")
    for t in sorted(tiers, key=lambda t: ("rule_confident", "rule", "elim").index(t) if t in ("rule_confident", "rule", "elim") else 9):
        line = f"  {t:15s} {tiers[t]:5d}"
        if a.split == "train":
            line += f"  accuracy {right[t] / tiers[t]:.2%}"
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
