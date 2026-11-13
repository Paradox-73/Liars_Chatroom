"""Build validation/newfam/newfam.csv: liar cases from scenario families that train.csv does not have.

python -m validation.newfam.gen.build             # writes newfam.csv and newfam_meta.csv
python -m validation.newfam.gen.build --show 3    # also prints the first three cases

Deterministic: the plan comes from one seed, each case from its own seeded generator. Before anything is written,
check_case() re-derives every case's liar from the fact models and asserts exactly one contradicting message.
"""
import argparse
import collections
import csv
import os
import random
import re

from . import base
from . import (fam_agelimit, fam_battery, fam_library, fam_lift, fam_medicine, fam_overnight, fam_parking, fam_periodic,
               fam_platform, fam_prices, fam_queue, fam_recipe, fam_scores, fam_seats, fam_temperature, fam_timezone,
               fam_whenever)

FAMILIES = [fam_agelimit, fam_battery, fam_library, fam_lift, fam_medicine, fam_parking, fam_periodic, fam_platform,
            fam_prices, fam_queue, fam_recipe, fam_scores, fam_seats, fam_temperature, fam_timezone, fam_whenever]
OVERNIGHT = [fam_overnight]  # families kept out of combined cases: their evidence spans two dates
PER_FAMILY = 13
PER_OVERNIGHT = 10
TOTAL = 300
LENGTHS = {6: 19, 7: 14, 8: 19, 9: 11, 10: 5, 11: 8, 12: 23}  # train's chat lengths, in percent
OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMBINED_SCENARIOS = ["{g} trade claims about {a} and {b}.", "{g} sort out what happened with {a} and {b}.",
                      "{g} compare notes on {a} and {b}."]


class Msg:
    def __init__(self, who, text, claim=None, bundle=None):
        self.who, self.text, self.claim, self.bundle = who, text, claim, bundle


def plan(rng):
    """A list of family tuples, one per case: (main,) or (main, second) for combined cases."""
    out = [(m,) for m in FAMILIES for _ in range(PER_FAMILY)]
    out += [(m,) for m in OVERNIGHT for _ in range(PER_OVERNIGHT)]
    while len(out) < TOTAL:
        out.append(tuple(rng.sample(FAMILIES, 2)))
    rng.shuffle(out)
    return out[:TOTAL]


def pick(pool, k, counts):
    """k claims from a shuffled pool: first one for each speaker with no claim yet, so most chats hear claims from
    three or four speakers as train chats do; then the rest, at most four claims per speaker while the pool allows."""
    out, rest = [], []
    for c, b in pool:
        if counts[c.who] == 0 and len(out) < k:
            out.append((c, b))
            counts[c.who] += 1
        else:
            rest.append((c, b))
    for cap in (4, len(pool) + 1):
        for c, b in rest:
            if len(out) < k and (c, b) not in out and counts[c.who] < cap:
                out.append((c, b))
                counts[c.who] += 1
    return out


def assemble(rng, bundles, day):
    main = bundles[0]
    n = rng.choices(list(LENGTHS), weights=list(LENGTHS.values()))[0]
    n_fill = rng.randint(2, min(4, n - 3))
    need = n - n_fill - 1
    pools = []
    for b in bundles:
        # another family's evidence may pin a named speaker somewhere else, which this family's claims cannot see
        pinned = {p for o in bundles if o is not b for p in main.people
                  if re.search(rf"\b{p}\b", " ".join(o.evidence))}
        seen, pool = {main.lie.text}, []
        for c in rng.sample(b.truths, len(b.truths)):
            if c.who == main.lie.who and c.key is not None and c.key == main.lie.key and b is main:
                continue  # the liar would contradict their own lie
            if c.text not in seen and c.who not in pinned:
                seen.add(c.text)
                pool.append((c, b))
        pools.append(pool)
    if need > sum(map(len, pools)):  # a short pool after the filters: drop to the longest chat it can fill
        need = sum(map(len, pools))
        n = need + n_fill + 1
        assert n >= 6, (main.family, n)
    counts = collections.Counter({main.lie.who: 1})
    if len(bundles) == 1:
        picked = pick(pools[0], need, counts)
    else:
        k2 = rng.randint(1, max(1, need // 2))
        k2 = max(min(k2, len(pools[1])), need - len(pools[0]))  # keep both parts within their pools
        picked = pick(pools[1], k2, counts)
        picked += pick(pools[0], need - k2, counts)
    assert len(picked) == need, (main.family, len(picked), need)
    msgs = [Msg(main.lie.who, base.lead(rng, main.lie.text), main.lie, main)]
    msgs += [Msg(c.who, base.lead(rng, c.text), c, b) for c, b in picked]
    fam_fill = [x for b in bundles for x in b.fillers]
    n_fam = min(len(fam_fill), sum(rng.random() < 0.35 for _ in range(n_fill)))
    texts = rng.sample(fam_fill, n_fam) + rng.sample(base.GENERIC_QUESTIONS + base.GENERIC_OPINIONS, n_fill - n_fam)
    for text in texts:  # train chats hear from all four speakers, rarely more than four messages each
        have = collections.Counter(m.who for m in msgs)
        low = min(have[p] for p in main.people)
        msgs.append(Msg(rng.choice([p for p in main.people if have[p] <= low + (low > 0)]), text))
    rng.shuffle(msgs)
    evidence = [f"All times in this case refer to {base.date_text(day)}."] if any(b.timed for b in bundles) else []
    blocks = [list(b.evidence) for b in bundles]
    decor = [d for b in bundles for d in b.decor]
    if decor and rng.random() < 0.45:
        rng.choice(blocks).append(rng.choice(decor))
    rng.shuffle(blocks)
    for blk in blocks:
        evidence += blk
    if len(bundles) == 1:
        scenario = main.scenario
    else:
        scenario = rng.choice(COMBINED_SCENARIOS).format(g=main.group, a=main.topic, b=bundles[1].topic)
    return scenario, evidence, msgs


def check_case(case):
    """Re-derive the liar from the fact models: run every claim's check against its own bundle's facts."""
    assert all(m.who in case["labels"] for m in case["msgs"]), case["id"]
    failing = [m for m in case["msgs"] if m.claim is not None and not m.claim.ok(m.bundle.facts)]
    assert len(failing) == 1, (case["id"], [m.text for m in failing])
    assert case["labels"][failing[0].who] == case["label"], case["id"]
    for m in case["msgs"]:
        if m.claim is None:  # fillers are questions or opinions with no digits
            assert m.text.endswith("?") or not any(ch.isdigit() for ch in m.text), m.text
    texts = [m.text for m in case["msgs"]]
    assert len(texts) == len(set(texts)), case["id"]
    assert 6 <= len(texts) <= 12


def build(seed=2026):
    rng = random.Random(seed)
    jobs = plan(rng)
    labels = [x for x in "ABCD" for _ in range(TOTAL // 4)]
    rng.shuffle(labels)
    cases, retries = [], collections.Counter()
    for i, fams in enumerate(jobs):
        for attempt in range(30):  # a combined case whose claim pools run short is drawn again
            crng = random.Random(f"newfam-{seed}-{i}-{attempt}")
            names = crng.sample(base.NAMES, 14)
            people, others = names[:4], names[4:]
            day = base.random_day(crng)
            bundles = []
            for j, fam in enumerate(fams):
                b = fam.make(crng, people, others[5 * j: 5 * j + 5], day)
                b.people = people
                bundles.append(b)
            try:
                scenario, evidence, msgs = assemble(crng, bundles, day)
                break
            except AssertionError:
                retries[fams[0].NAME + "+".join(f.NAME for f in fams[1:])] += 1
        else:
            raise RuntimeError(f"case {i} {[f.NAME for f in fams]} never assembled")
        liar = bundles[0].lie.who
        rest = [p for p in people if p != liar]
        crng.shuffle(rest)
        letters = [x for x in "ABCD" if x != labels[i]]
        lab = {liar: labels[i], **dict(zip(rest, letters))}
        case = dict(id=f"NF{i + 1:04d}", scenario=scenario, labels=lab, evidence=evidence, msgs=msgs,
                    label=labels[i], family="+".join(f.NAME for f in fams),
                    tags=bundles[0].lie.tags, lie_num=next(k + 1 for k, m in enumerate(msgs) if m.claim is bundles[0].lie))
        check_case(case)
        cases.append(case)
    print("redraws:", dict(retries))
    return cases


def row(case):
    by_label = sorted(case["labels"].items(), key=lambda kv: kv[1])
    speakers = "; ".join(f"{lab} = {name}" for name, lab in by_label)
    chat = "\n".join(f"{k + 1}. {case['labels'][m.who]}: {m.text}" for k, m in enumerate(case["msgs"]))
    return [case["id"], case["scenario"], speakers, " ".join(case["evidence"]), chat, case["label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    cases = build(a.seed)
    with open(os.path.join(OUT_DIR, "newfam.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "scenario", "speakers", "evidence", "chat", "label"])
        w.writerows(row(c) for c in cases)
    with open(os.path.join(OUT_DIR, "newfam_meta.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "family", "lie_message", "lie_tags", "messages"])
        w.writerows([c["id"], c["family"], c["lie_num"], " ".join(c["tags"]), len(c["msgs"])] for c in cases)
    print(f"wrote {len(cases)} cases to {OUT_DIR}; every case passed the self-check")
    for c in cases[: a.show]:
        r = row(c)
        print(f"===== {r[0]} label {r[5]} family {c['family']} tags {c['tags']}\nSC: {r[1]}\nSP: {r[2]}\nEV: {r[3]}\n{r[4]}")


if __name__ == "__main__":
    main()
