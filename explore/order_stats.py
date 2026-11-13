"""Does a long prompt predict long reasoning? First-sample tokens (capped at 4,096) by prompt-length quarter, and which
quarter the capped samples and the unreached cases fall in.

python explore/order_stats.py <answers.jsonl> <labelled csv>
"""
import csv
import json
import statistics
import sys

sys.path.insert(0, ".")
from lc.data import case_from_row  # noqa: E402
from lc.llm.prompt import build_prompt  # noqa: E402

rows = {r["id"]: r for r in csv.DictReader(open(sys.argv[2], encoding="utf-8"))}
plen = {i: len(build_prompt(case_from_row(r))) for i, r in rows.items()}  # characters, as a stand-in for tokens
first = {}
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    if r["k"] == 0 and (r["id"] not in first or r["stage"] >= first[r["id"]]["stage"]):
        first[r["id"]] = r
ids = sorted(rows, key=lambda i: plen[i])
q = len(ids) // 4
xs = [plen[i] for i in first]
ys = [first[i]["out_tokens"][0] for i in first]
mx, my = statistics.mean(xs), statistics.mean(ys)
corr = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) * statistics.pstdev(xs) * statistics.pstdev(ys))
print(f"correlation of prompt length and first-sample tokens: {corr:.2f} over {len(xs)} cases")
for n in range(4):
    part = ids[n * q:(n + 1) * q] if n < 3 else ids[3 * q:]
    reached = [first[i] for i in part if i in first]
    print(f"prompt quarter {n + 1} (shortest=1): {len(part)} cases, reached {len(reached)}, mean tokens "
          f"{statistics.mean(r['out_tokens'][0] for r in reached) if reached else 0:.0f}, capped "
          f"{sum(r['forced'][0] for r in reached)}")
