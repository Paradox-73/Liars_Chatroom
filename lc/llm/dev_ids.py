"""Prints a fixed, seeded sample of train ids, stratified by true sender and number of messages.

Example: LC_DATA_DIR=data python -m lc.llm.dev_ids --k 200 --seed 13 > configs/dev200.txt
"""
import argparse
import random
from collections import defaultdict

from lc.data import load


def stratified_ids(cases, k, seed):
    groups = defaultdict(list)
    for c in cases:
        groups[(c.label, len(c.messages))].append(c.id)
    keys = sorted(groups)
    quota = {g: k * len(groups[g]) / len(cases) for g in keys}
    take = {g: int(quota[g]) for g in keys}
    # largest remainders get the seats left over after rounding down
    for g in sorted(keys, key=lambda g: quota[g] - take[g], reverse=True)[: k - sum(take.values())]:
        take[g] += 1
    rng = random.Random(seed)
    return sorted(i for g in keys for i in rng.sample(sorted(groups[g]), take[g]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    ids = stratified_ids(load("train"), args.k, args.seed)
    assert len(ids) == len(set(ids)) == args.k
    print("\n".join(ids))
