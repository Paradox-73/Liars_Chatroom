"""Movement rules: 'the cabinet' is ambiguous when the evidence names two cabinets told apart only by 'by the ...'."""
from lc.data import case_from_row
from lc.rules import solve

SP = "A = Kofi; B = Ama; C = Priya; D = Tomas"
DAY = "All times in this case refer to 3 March 2026. "


def case(evidence, msgs):
    chat = "\n".join(f"{i}. {m}" for i, m in enumerate(msgs, 1))
    return case_from_row({"id": "t", "scenario": "", "speakers": SP, "evidence": DAY + evidence, "chat": chat})


def test_ambiguous_cabinet_is_not_a_lie():
    c = case("There was one red key. It stayed in the cabinet by the window from 1:00 PM to 3:00 PM. The cabinet by "
             "the window and the cabinet by the door are separate locations.",
             ["A: I saw the red key in the cabinet at 2:00 PM.",
              "C: At around 2:30 PM I took the red key out of the cabinet by the window."])
    assert solve(c).pred is None


def test_ambiguous_shop_is_not_a_lie():
    c = case("Kofi was at the warehouse at 2:00 PM. Every route from the warehouse to the shop by the river takes at "
             "least 40 minutes. Every route from the warehouse to the shop by the station takes at least 10 minutes. "
             "The shop by the river and the shop by the station are separate locations.",
             ["B: I saw Kofi at the shop at 2:20 PM.", "C: Kofi was at the shop by the river at about 2:15 PM."])
    assert solve(c).pred is None


def test_single_cabinet_short_name_still_judged():
    c = case("There was one red key. It stayed in the cabinet by the window from 1:00 PM to 3:00 PM. The cabinet by "
             "the window and the front desk are separate locations.",
             ["A: I saw the red key in the cabinet at 2:00 PM.", "C: I had the red key at the front desk at 2:00 PM."])
    assert solve(c).pred == "C"
