"""Access rules: claims about a stretch of time, a later arrival, a code that did not work, someone else's wristband
and 'didn't go in until T' are left to the model instead of being judged as one moment."""
from lc.combine import open_speakers
from lc.data import case_from_row
from lc.rules import solve

SP = "A = Ravi; B = Mia; C = Leo; D = Zoe"
DAY = "All times in this case refer to 14 May 2026. "


def verdicts(evidence, msg):
    """Status of the one message 'X: text' (None when no family judged it)."""
    c = case_from_row({"id": "t", "scenario": "", "speakers": SP, "evidence": DAY + evidence, "chat": f"1. {msg}"})
    vs = solve(c).verdicts.get(1)
    return {v.status for v in vs} if vs else None


ARR = ("The door log accurately records when each of these four people first arrived at the office that day: "
       "Ravi at 8:30 AM, Mia at 8:45 AM, Leo at 9:00 AM and Zoe at 9:10 AM.")
WIN = "Nobody could enter the greenhouse between 9:00 AM and 11:00 AM."
CODE = ("Before 10:15 AM, only code 1764 could unlock the studio door. From 10:15 AM to the end of the day, only code "
        "6139 could unlock it.")
PRES = ("From 2:30 PM to 6:30 PM, only Ravi and Mia were at the security gate. Nobody else was at the security gate "
        "during that time.")
BADGE = ("Nobody could enter the pool area except by scanning their own active wristband. Leo's wristband was inactive "
         "before 10:05 AM, and active from 10:05 AM onward.")


def test_later_arrival_without_first_is_not_a_lie():
    assert verdicts(ARR, "A: I came to the office at 5:00 PM for the evening meeting.") is None
    assert verdicts(ARR, "A: I arrived at the office again at 1:30 PM.") is None
    assert verdicts(ARR, "A: I got to the office at 8:00 AM.") == {"lie"}  # earlier than the first arrival
    assert verdicts(ARR, "A: I first arrived at the office at 9:00 AM.") == {"lie"}


def test_denial_until_a_time_inside_the_window_is_not_cleared():
    assert verdicts(WIN, "A: I didn't go into the greenhouse until 10:00 AM.") is None
    assert verdicts(WIN, "A: I didn't enter the greenhouse before 10:30 AM, but I did go in then.") is None
    assert verdicts(WIN, "A: I never went into the greenhouse.") == {"ok"}


def test_a_stretch_of_time_is_not_one_moment():
    assert verdicts(PRES, "C: I was at the security gate at 2:15 PM for an hour.") is None
    assert verdicts(PRES, "C: I was at the security gate at 2:15 PM.") == {"ok"}
    assert verdicts(PRES, "C: I was at the security gate at 2:45 PM.") == {"lie"}


def test_code_that_did_not_open_the_door_is_not_a_lie():
    assert verdicts(CODE, "A: At 9:30 AM I typed code 6139 to unlock the studio door, but it stayed shut.") is None
    assert verdicts(CODE, "A: At 9:30 AM I used code 6139 to unlock the studio door.") == {"lie"}


def test_someone_elses_wristband_is_not_cleared():
    assert verdicts(BADGE, "A: I saw Leo go into the pool area at 10:30 AM using Zoe's wristband.") is None
    assert verdicts(BADGE, "A: I saw Leo go into the pool area at 9:30 AM.") == {"lie"}


def test_liar_stays_open_for_the_model():
    c = case_from_row({"id": "t", "scenario": "", "speakers": SP, "evidence": DAY + PRES,
                       "chat": "1. C: I was at the security gate at 2:15 PM for an hour.\n"
                               "2. A: I was at the security gate at 4:00 PM.\n"
                               "3. B: I was at the security gate at 5:00 PM.\n4. D: I bought a sandwich at 3:00 PM."})
    assert "C" in open_speakers(c, solve(c))
