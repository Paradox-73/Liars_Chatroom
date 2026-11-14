"""python -m pytest tests"""
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "submission"))
import predict  # noqa: E402

SPEAKERS = "A = Kofi; B = Yusuf; C = Akira; D = Obi"
WINDOW = ("All times in this case refer to 30 June 2026. Nobody could enter the carpentry shed between 4:00 PM and "
          "5:00 PM.")
CHAT = "\n".join(["1. A: I entered the carpentry shed at 3:30 PM.", "2. B: I entered the carpentry shed at {b}.",
                  "3. C: I entered the carpentry shed at 5:30 PM.", "4. D: I had a long call with the supplier today."])


def test_rule_answers_only_when_the_rules_read_all_the_evidence(tmp_path):
    rows = [("ruled", WINDOW, CHAT.format(b="4:30 PM")),  # B entered inside the window: the rules can prove it
            ("hybrid", WINDOW + " Yusuf spent the whole afternoon at the dentist across town and only got back at "
             "7:30 PM.", CHAT.format(b="6:30 PM"))]  # the liar is B, through a sentence no rule family reads
    path = tmp_path / "cases.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "scenario", "speakers", "evidence", "chat"])
        w.writerows([cid, "Shed visits.", SPEAKERS, ev, chat] for cid, ev, chat in rows)
    ids, fixed, cases, cands = predict.read_cases(str(path))
    assert fixed == {"ruled": ("B", "rule")}  # before, the hybrid was fixed too: ("D", "elim"), a wrong answer
    assert list(cases) == ["hybrid"] and cands["hybrid"] == "ABCD"


def test_header_names_in_any_case_and_with_spaces(tmp_path):
    path = tmp_path / "cases.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", " Scenario", "SPEAKERS ", "Evidence", "chat"])  # before: no "id" column, no case read
        w.writerow(["ruled", "Shed visits.", SPEAKERS, WINDOW, CHAT.format(b="4:30 PM")])
    ids, fixed, cases, cands = predict.read_cases(str(path))
    assert ids == ["ruled"] and fixed == {"ruled": ("B", "rule")}


def test_a_bad_answer_line_is_skipped_not_the_whole_merge(tmp_path):
    path = tmp_path / "answers0"
    good = {"id": "x1", "k": 0, "stage": 0, "kind": "main", "pred": "B", "forced": [False]}
    later = dict(good, id="x2", pred="D")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(good) + "\n{not json\n" + json.dumps({"id": "x3"}) + "\n" + json.dumps(later) + "\n")
    recs = {}
    pos = predict.merge(str(path), 0, recs)  # before: JSONDecodeError, which stopped the model step
    assert pos == os.path.getsize(path) and sorted(recs) == [("x1", 0), ("x2", 0)]


def test_a_very_long_case_skips_the_rules(tmp_path):
    long_evidence = " ".join([WINDOW] * (20000 // len(WINDOW) + 1))  # about 20,000 characters
    path = tmp_path / "cases.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "scenario", "speakers", "evidence", "chat"])
        w.writerow(["long", "Shed visits.", SPEAKERS, long_evidence, CHAT.format(b="4:30 PM")])
        w.writerow(["ruled", "Shed visits.", SPEAKERS, WINDOW, CHAT.format(b="4:30 PM")])
    t = time.time()
    ids, fixed, cases, cands = predict.read_cases(str(path))
    assert time.time() - t < 5
    assert ids == ["long", "ruled"] and fixed == {"ruled": ("B", "rule")}
    assert list(cases) == ["long"] and cands["long"] == "ABCD"  # before: the rules ran on it and answered it
