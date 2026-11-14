"""python -m pytest tests"""
from lc.combine import open_speakers
from lc.data import Case, Message
from lc.rules import solve


def test_an_opinion_like_claim_keeps_its_speaker_in_the_vote():
    case = Case("X", "Lab visits.", {"A": "Usha", "B": "Fatima", "C": "Rania", "D": "Vikram"},
                "The lab log shows Priya entering the lab once in the afternoon.",
                [Message(1, "A", "I want to be clear: Priya never entered the lab."), Message(2, "B", "Who saw her?"),
                 Message(3, "C", "Priya went in once."), Message(4, "D", "She was there in the afternoon.")])
    res = solve(case)
    assert not res.verdicts  # the rules judge nothing here, so only the questions are passed over
    assert open_speakers(case, res) == {"A", "C", "D"}
