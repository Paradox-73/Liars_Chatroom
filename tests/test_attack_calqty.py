"""Calendar and quantity rules must not judge a claim whose time of day they cannot read."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lc.data import case_from_row  # noqa: E402
from lc.rules import solve  # noqa: E402

POOL = ("1 April 2026 was a Wednesday. In April 2026 the pool was open only Wednesday to Sunday, from 9:00 AM to 5:00 PM, "
        "and nobody could use it while it was closed.")


def pool_case(liar_text):
    chat = "\n".join([
        "1. A: I swam at the pool at 10:00 AM on 11 April 2026.",
        f"2. C: {liar_text}",
        "3. D: I drove past the pool on 13 April 2026.",
    ])
    return case_from_row({"id": "t", "scenario": "s", "speakers": "A = Mateo; B = Maria; C = Zeynep; D = Carmen",
                          "evidence": POOL, "chat": chat})


def test_cal_hours_leaves_unread_times_to_the_model():
    # 11 April 2026 is an open day, but each of these times is after the 5:00 PM close: judging the day alone cleared
    # the liar, and elimination then named the innocent D
    for text in ("I swam at the pool at 17:45 on 11 April 2026.", "I swam at the pool at 7:30 p.m. on 11 April 2026.",
                 "I swam at the pool at night on 11 April 2026.", "I swam at the pool in the evening on 11 April 2026."):
        res = solve(pool_case(text))
        assert 2 not in res.verdicts, text


def test_cal_hours_leaves_claims_that_last_to_the_model():
    # each swim starts inside the hours but runs past the 5:00 PM close
    for text in ("I swam at the pool from 4:30 PM on 11 April 2026 for two hours.",
                 "I swam at the pool for three hours starting at 3:00 PM on 11 April 2026."):
        assert 2 not in solve(pool_case(text)).verdicts, text


def test_cal_session_leaves_claims_that_last_to_the_model():
    case = case_from_row({"id": "t", "scenario": "s", "speakers": "A = Mateo; B = Maria; C = Zeynep; D = Carmen",
                          "evidence": "All times in this case refer to 21 February 2026. That day's only tour began at "
                                      "12:45 PM and lasted exactly 1 hour.",
                          "chat": "1. A: I led the group during the tour at 1:00 PM.\n"
                                  "2. C: I led the group during the tour at 1:30 PM for an hour.\n"
                                  "3. D: I checked the museum at 3:00 PM."})
    res = solve(case)
    assert 1 in res.verdicts and 2 not in res.verdicts


def test_cal_hours_still_judges_readable_times():
    res = solve(pool_case("I swam at the pool at 5:30 PM on 11 April 2026."))
    assert res.pred == "C"
