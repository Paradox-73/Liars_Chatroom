"""Score the rule solver on train.

python -m lc.rules.run                      # overall coverage and accuracy, per-family false-lie counts
python -m lc.rules.run --family ticket      # only cases where a family whose name contains 'ticket' is active
python -m lc.rules.run --errors 5           # print 5 wrong or false-lie cases in full
python -m lc.rules.run --missed 5 --family X  # print cases where family X is active but no liar was found
"""
import argparse
import collections

from lc.data import load
from lc.rules import REGISTRY, solve
from lc.rules.core import LIE


def show(case, res):
    print(f"===== {case.id} label={case.label} pred={res.pred} lies={res.lie_labels} families={res.families}")
    print("SP:", "; ".join(f"{k} = {v}" for k, v in case.speakers.items()))
    print("EV:", case.evidence)
    for m in case.messages:
        vs = res.verdicts.get(m.num, [])
        tag = " | ".join(f"{v.status}:{v.family}:{v.why}" for v in vs) or ("?" if m.num in res.unknown_claims else "")
        print(f"  {m.num}. {m.label}: {m.text}    <{tag}>")
    if res.unconsumed:
        print("  unparsed evidence:", res.unconsumed)
    if res.errors:
        print("  ERRORS:", res.errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--family", default=None)
    ap.add_argument("--errors", type=int, default=0)
    ap.add_argument("--missed", type=int, default=0)
    a = ap.parse_args()
    assert a.split == "train", "the rule solver is only ever scored on train"
    cases = load("train")
    fam_false = collections.Counter()
    fam_true = collections.Counter()
    fam_active = collections.Counter()
    n = answered = correct = confident = conf_correct = 0
    wrong, missed, errs = [], [], 0
    for c in cases:
        r = solve(c)
        if a.family and not any(a.family in f for f in r.families):
            continue
        n += 1
        errs += bool(r.errors)
        for f in r.families:
            fam_active[f] += 1
        for num, vs in r.verdicts.items():
            lab = next(m.label for m in c.messages if m.num == num)
            for v in vs:
                if v.status == LIE:
                    (fam_true if lab == c.label else fam_false)[v.family] += 1
        if r.pred:
            answered += 1
            correct += r.pred == c.label
            confident += r.confident
            conf_correct += r.confident and r.pred == c.label
            if r.pred != c.label:
                wrong.append((c, r))
        else:
            missed.append((c, r))
        if c.label not in r.lie_labels and r.lie_labels and (c, r) not in wrong:
            wrong.append((c, r))
    print(f"cases {n}  answered {answered} ({answered / max(n, 1):.1%})  correct {correct} "
          f"(precision {correct / max(answered, 1):.2%})  coverage-accuracy {correct / max(n, 1):.1%}")
    print(f"confident {confident}, of which correct {conf_correct}; cases with family exceptions {errs}")
    print("family: active cases / LIE verdicts on the liar / LIE verdicts on an innocent speaker (must be 0)")
    for f in sorted(fam_active, key=lambda k: -fam_active[k]):
        print(f"  {f:28s} {fam_active[f]:5d} {fam_true[f]:5d} {fam_false[f]:5d}")
    for c, r in wrong[: a.errors]:
        show(c, r)
    for c, r in missed[: a.missed]:
        show(c, r)


if __name__ == "__main__":
    main()
