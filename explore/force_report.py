"""Reads force_out.jsonl from the forcing kernel: accuracy of each way to answer a capped sample, paired flips against
the one-token forced answer with an exact two-sided sign test, and the extra tokens each way costs.

python explore/force_report.py <force_out.jsonl>
"""
import json
import math
import sys
from collections import Counter


def sign_test(a, b):
    """Two-sided exact binomial p-value for a wins against b wins (ties dropped)."""
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


rows = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8")]
fin = [r for r in rows if not r["capped"]]
cap = [r for r in rows if r["capped"]]
print(f"first samples {len(rows)}: finished {len(fin)} (right {sum(r['first'] == r['label'] for r in fin)}), "
      f"capped {len(cap)} on {len({r['id'] for r in cap})} cases")
for m in ("one_token", "budget", "continue"):
    print(f"  {m:10s} right {sum(r.get(m) == r['label'] for r in cap):3d}/{len(cap)}")
for m in ("budget", "continue"):
    win = sum(r.get(m) == r["label"] and r.get("one_token") != r["label"] for r in cap)
    loss = sum(r.get(m) != r["label"] and r.get("one_token") == r["label"] for r in cap)
    print(f"  {m} vs one_token: wrong->right {win}, right->wrong {loss}, sign test p = {sign_test(win, loss):.3f}")
bt = [r.get("budget_tokens", 0) for r in cap]
print(f"budget answer: mean {sum(bt) / max(len(bt), 1):.0f} tokens, max {max(bt, default=0)}; "
      f"no ANSWER line {sum(r.get('budget') is None for r in cap)}")
print(f"continuations that finished by 2x the cap: {sum(bool(r.get('continue_finished')) for r in cap)}/{len(cap)}")
# per case: how often a fresh seed finishes under the cap, and the letters
by = {}
for r in rows:
    by.setdefault(r["id"], []).append(r)
fresh = Counter()
for cid, rs in sorted(by.items()):
    fresh[sum(not r["capped"] for r in rs)] += 1
    print(f"  {cid} label {rs[0]['label']}: " + "  ".join(
        (f"k{r['k']} fin {r['first']}" if not r["capped"] else
         f"k{r['k']} cap 1tok {r.get('one_token')} bud {r.get('budget')} cont {r.get('continue')}") for r in sorted(rs, key=lambda r: r["k"])))
print("cases by number of seeds (of 3) that finished under the cap:", dict(sorted(fresh.items())))
