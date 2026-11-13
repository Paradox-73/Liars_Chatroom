"""Reads the outputs of the exploration kernels.

python explore/analyze.py pipe <kernel output folder> <labelled csv> <meta csv>   accuracy by family/style, misses
python explore/analyze.py probe <probe_out.jsonl>                                  per case and prompt arm
"""
import collections
import csv
import glob
import json
import os
import re
import sys


def log_text(folder):
    """The kernel log (a JSON list of stream records) as plain text."""
    logs = glob.glob(os.path.join(folder, "*.log"))
    if not logs:
        return ""
    with open(logs[0], encoding="utf-8", errors="replace") as f:
        return "".join(x["data"] for x in json.load(f))


def pipe(folder, cases_csv, meta_csv):
    labels = {r["id"]: r["label"] for r in csv.DictReader(open(cases_csv, encoding="utf-8"))}
    meta = {r["id"]: r for r in csv.DictReader(open(meta_csv, encoding="utf-8"))}
    pred = {r["id"]: r["label"] for r in csv.DictReader(open(os.path.join(folder, "newfam_pred.csv"), encoding="utf-8"))}
    recs = {}
    path = os.path.join(folder, "newfam_answers.jsonl")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            old = recs.get((r["id"], r["k"]))
            if old is None or r["stage"] >= old["stage"]:
                recs[(r["id"], r["k"])] = r
    first = {c: r for (c, k), r in recs.items() if k == 0}
    by = collections.defaultdict(lambda: [0, 0])
    for i, lab in labels.items():
        m = meta.get(i, {})
        for key in ("all", f"family={m.get('family', '?').split('+')[0]}", f"style={m.get('style', '?')}"):
            by[key][0] += pred.get(i) == lab
            by[key][1] += 1
    for key in sorted(by, key=lambda k: (k != "all", k)):
        print(f"{key:24s} {by[key][0]:4d}/{by[key][1]:<4d} {by[key][0] / by[key][1]:.3f}")
    if first:
        f_ok = [r for c, r in first.items() if not r["forced"][0]]
        f_forced = [r for c, r in first.items() if r["forced"][0]]
        print(f"first samples {len(first)}: unforced {sum(r['pred'] == labels[r['id']] for r in f_ok)}/{len(f_ok)}, "
              f"forced {sum(r['pred'] == labels[r['id']] for r in f_forced)}/{len(f_forced)}; "
              f"continued {sum(r['kind'] == 'cont' for r in first.values())}; "
              f"mean tokens {sum(r['out_tokens'][0] for r in first.values()) / len(first):.0f}")
        extra = collections.Counter(c for (c, k) in recs if k > 0)
        print(f"extra samples: {sum(extra.values())} on {len(extra)} cases; no-think answers {sum(1 for (c, k) in recs if k < 0)}")
    print("misses:")
    for i, lab in labels.items():
        if pred.get(i) != lab:
            r = first.get(i)
            how = (f"first {r['pred']} forced={r['forced'][0]} kind={r['kind']} tokens={r['out_tokens'][0]}" if r else
                   "no first sample")
            ks = sorted(k for (c, k) in recs if c == i)
            print(f"  {i} {meta.get(i, {}).get('family', '?'):12s} label {lab} pred {pred.get(i)}  {how}  samples {ks}")
    text = log_text(folder)
    for pat in (r"engine ready after [\d.]+ min.*", r"\[judge .*done: .*", r"\[predict .*wrote .*", r"SUMMARY .*"):
        for m in re.findall(pat, text)[-1:]:
            print(m[:400])
    rate = re.findall(r"(\d+) output tokens/s", text)
    if rate:
        print("last throughput line:", rate[-1], "tokens/s")


def probe(path):
    rows = [json.loads(x) for x in open(path, encoding="utf-8")]
    groups = {}
    try:
        for r in csv.DictReader(open(os.path.join(os.path.dirname(__file__), "probe_cases.csv"), encoding="utf-8")):
            groups[r["id"]] = r["group"]
    except OSError:
        pass
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])  # right, n, capped, tokens
    per = collections.defaultdict(list)
    for r in rows:
        g = groups.get(r["id"], "?")
        for key in ((g, r["arm"]), ("all", r["arm"])):
            a = agg[key]
            a[0] += r["pred"] == r["label"]
            a[1] += 1
            a[2] += r["capped"]
            a[3] += r["tokens"]
        per[(r["id"], r["arm"])].append(r)
    for key in sorted(agg):
        a = agg[key]
        print(f"{key[0]:12s} {key[1]:8s} right {a[0]:3d}/{a[1]:<3d} {a[0] / a[1]:.3f}  capped {a[2]:3d}  mean tokens {a[3] / a[1]:.0f}")
    print("per case (letters per sample; * = capped):")
    for cid in sorted({i for i, _ in per}):
        line = []
        for arm in ("base", "variant"):
            rs = sorted(per.get((cid, arm), []), key=lambda r: r["k"])
            if rs:
                line.append(f"{arm}: " + " ".join(f"{r['pred'] or '-'}{'*' if r['capped'] else ''}" for r in rs))
        print(f"  {cid} {groups.get(cid, '?'):12s} label {rs[0]['label'] if rs else '?'}  " + "   ".join(line))


if __name__ == "__main__":
    {"pipe": pipe, "probe": probe}[sys.argv[1]](*sys.argv[2:])
