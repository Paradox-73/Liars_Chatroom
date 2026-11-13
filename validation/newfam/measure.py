"""Score the rule solver on the new-family validation set, CPU only, no language model.

python -m validation.newfam.measure           # print the summary, every misfire and every wrongly cleared liar
python -m validation.newfam.measure --write   # also rewrite REPORT.md; the hand-written block between the
                                              # <!-- analysis --> markers is kept

Counts, over validation/newfam/newfam.csv:
  answered        the rules name a liar (exactly one speaker has a message judged a lie)
  misfire         a message of an innocent speaker is judged a lie (answered wrongly, or two speakers flagged)
  liar cleared    every claim of the true liar was judged consistent, so open_speakers() drops the liar and the
                  model vote can never pick the right letter
  elim            the elimination tier of lc.combine answers (no lie found, one speaker left with unjudged claims)
"""
import argparse
import collections
import csv
import os

from lc.combine import decide, open_speakers
from lc.data import load
from lc.rules import solve
from lc.rules.core import LIE, OK
from lc.rules.text import is_filler

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "REPORT.md")
START, END = "<!-- analysis -->", "<!-- /analysis -->"


def show(case, res, fam):
    """One case with every verdict, as plain text for a fenced block."""
    lines = [f"{case.id}  family {fam}  true liar {case.label}  rules said {res.pred or '-'}  "
             f"flagged {','.join(res.lie_labels) or '-'}  rule families active: {', '.join(res.families) or '-'}",
             f"scenario: {case.scenario}",
             "speakers: " + "; ".join(f"{k} = {v}" for k, v in case.speakers.items()),
             f"evidence: {case.evidence}"]
    for m in case.messages:
        vs = res.verdicts.get(m.num, [])
        tag = " | ".join(f"{v.status}:{v.family}:{v.why}" for v in vs)
        mark = "  <-- true liar" if m.label == case.label else ""
        lines.append(f"  {m.num}. {m.label}: {m.text}" + (f"    [{tag}]" if tag else "") + mark)
    if res.errors:
        lines.append(f"  errors: {res.errors}")
    return "\n".join(lines)


def run():
    with open(os.path.join(HERE, "newfam_meta.csv"), encoding="utf-8", newline="") as f:
        meta = {r["id"]: r for r in csv.DictReader(f)}
    rows = []
    for c in load("newfam", data_dir=HERE):
        res = solve(c)
        pred, tier = decide(c, [])
        sender = {m.num: m.label for m in c.messages}
        bad = [(n, v) for n, vs in res.verdicts.items() for v in vs if v.status == LIE and sender[n] != c.label]
        lie = int(meta[c.id]["lie_message"])
        lie_text = next(m.text for m in c.messages if m.num == lie)
        rows.append(dict(case=c, res=res, tier=tier, pred=pred, fam=meta[c.id]["family"], bad=bad,
                         lie_ok=[v for v in res.verdicts.get(lie, []) if v.status == OK], lie_filler=is_filler(lie_text),
                         cleared=c.label not in open_speakers(c, res),
                         overnight=("core", "overnight times: rules abstain") in res.errors,
                         crashed=[e for e in res.errors if e[0] != "core"]))
    return rows


def table(rows):
    def key(r):
        return r["fam"] if "+" not in r["fam"] else "combined (two families)"

    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = ["| family | cases | answered | right | misfires | liar cleared | elim right/wrong | overnight abstain "
           "| rule families active |", "|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(groups, key=lambda k: (k.startswith("combined"), k)) + ["all"]:
        rs = rows if name == "all" else groups[name]
        act = collections.Counter(f for r in rs for f in r["res"].families)
        elim = [r for r in rs if r["tier"] == "elim"]
        out.append(
            f"| {name} | {len(rs)} | {sum(bool(r['res'].pred) for r in rs)} "
            f"| {sum(r['res'].pred == r['case'].label for r in rs)} | {sum(bool(r['bad']) for r in rs)} "
            f"| {sum(r['cleared'] for r in rs)} | {sum(r['pred'] == r['case'].label for r in elim)}/"
            f"{sum(r['pred'] != r['case'].label for r in elim)} | {sum(r['overnight'] for r in rs)} "
            f"| {', '.join(f'{k} {v}' for k, v in act.most_common()) or '-'} |")
    return "\n".join(out)


def family_counts(rows):
    """Cases per family: alone, and in combined cases as the source of the lie or as the second family."""
    alone, lie_src, second = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        fams = r["fam"].split("+")
        (alone if len(fams) == 1 else lie_src)[fams[0]] += 1
        for f in fams[1:]:
            second[f] += 1
    out = ["| family | cases alone | combined, lie from it | combined, second family |", "|---|---|---|---|"]
    for f in sorted(set(alone) | set(lie_src) | set(second)):
        out.append(f"| {f} | {alone[f]} | {lie_src[f]} | {second[f]} |")
    out.append(f"| all | {sum(alone.values())} | {sum(lie_src.values())} | {sum(second.values())} |")
    return "\n".join(out)


def summary(rows):
    n = len(rows)
    answered = [r for r in rows if r["res"].pred]
    right = sum(r["res"].pred == r["case"].label for r in answered)
    mis = [r for r in rows if r["bad"]]
    cleared = [r for r in rows if r["cleared"]]
    elim = [r for r in rows if r["tier"] == "elim"]
    fired = collections.Counter(v.family for r in mis for _, v in r["bad"])
    lines = [
        f"- Cases: {n}",
        f"- Answered by the rules: {len(answered)} ({len(answered) / n:.1%}); right {right}, wrong {len(answered) - right}",
        f"- Misfires (an innocent speaker's message judged a lie): {len(mis)} cases"
        + (f"; rule families that fired: {', '.join(f'{k} {v}' for k, v in fired.most_common())}" if fired else ""),
        f"- Cases where open_speakers() wrongly drops the true liar: {len(cleared)}",
        f"- Cases where a rule judged the lie itself consistent: {sum(bool(r['lie_ok']) for r in rows)}",
        f"- Cases where is_filler() reads the lie as a question or opinion: {sum(r['lie_filler'] for r in rows)}",
        f"- Elimination tier answers: {len(elim)}, of which wrong {sum(r['pred'] != r['case'].label for r in elim)}",
        f"- Cases where the overnight guard made the rules abstain: {sum(r['overnight'] for r in rows)}",
        f"- Cases with an exception inside a rule family: {sum(bool(r['crashed']) for r in rows)}",
    ]
    return "\n".join(lines), mis, cleared, [r for r in elim if r["pred"] != r["case"].label]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="rewrite REPORT.md, keeping the analysis block")
    a = ap.parse_args()
    rows = run()
    text, mis, cleared, elim_wrong = summary(rows)
    parts = ["## Rule solver on newfam.csv", "", "Command: `python -m validation.newfam.measure --write`", "",
             text, "", "### Cases per family", "", family_counts(rows), "", "### Rule solver by family", "",
             table(rows), ""]
    for title, rs in (("Misfires", mis), ("True liar wrongly cleared", cleared), ("Wrong elimination answers",
                                                                                    elim_wrong)):
        parts += [f"### {title}: {len(rs)}", ""]
        for r in rs:
            why = "; ".join(f"message {n} ({v.family}: {v.why})" for n, v in r["bad"])
            parts += ([f"Fired on an innocent: {why}", ""] if why else []) + ["```", show(r["case"], r["res"], r["fam"]),
                                                                               "```", ""]
    body = "\n".join(parts)
    print(body)
    if a.write:
        old = open(REPORT, encoding="utf-8").read() if os.path.exists(REPORT) else ""
        keep = old[old.index(START): old.index(END) + len(END)] if START in old and END in old else \
            f"{START}\n(hand-written analysis goes here)\n{END}"
        with open(REPORT, "w", encoding="utf-8", newline="\n") as f:
            f.write("# New-family validation set: report\n\n" + keep + "\n\n" + body)
        print("wrote", REPORT)


if __name__ == "__main__":
    main()
