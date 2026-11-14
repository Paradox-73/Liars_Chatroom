"""python -m pytest tests"""
import pytest

from lc.data import parse_chat, parse_speakers

SPEAKERS = {"A": "Usha", "B": "Fatima", "C": "Rania", "D": "Vikram"}
CHAT = [(1, "A", "I was in the lab at 9:40 AM."), (2, "B", "Priya never entered the lab."), (3, "C", "Same here.")]


def test_an_odd_line_continues_the_message_before_it():
    chat = "Chat log:\n1. A: I was in the lab at 9:40 AM.\n\n2. B: Priya never entered\nthe lab.\n3. C: Same here."
    assert [(m.num, m.label, m.text) for m in parse_chat(chat)] == CHAT


@pytest.mark.parametrize("chat", [
    "1. A: I was in the lab at 9:40 AM.\n2. B: Priya never entered the lab.\n3. C: Same here.",
    "1) A: I was in the lab at 9:40 AM.\n2) B: Priya never entered the lab.\n3) C: Same here.",
    "1. a: I was in the lab at 9:40 AM.\n2. b: Priya never entered the lab.\n3. c: Same here.",
    "1. A (Usha): I was in the lab at 9:40 AM.\n2. B (Fatima): Priya never entered the lab.\n3. C (Rania): Same here.",
    "A: I was in the lab at 9:40 AM.\nB: Priya never entered the lab.\nC: Same here.",
])
def test_chat_line_forms(chat):
    assert [(m.num, m.label, m.text) for m in parse_chat(chat)] == CHAT


def test_a_line_without_a_number_in_a_numbered_chat_is_a_continuation():
    msgs = parse_chat("1. A: I was in the lab at 9:40 AM.\nB: said so too.\n2. B: Priya never entered the lab.")
    assert [(m.num, m.label) for m in msgs] == [(1, "A"), (2, "B")] and msgs[0].text.endswith("B: said so too.")


def test_a_chat_without_any_message_line_is_unreadable():
    with pytest.raises(ValueError):
        parse_chat("Usha said she was in the lab.\nFatima said she was not.")


@pytest.mark.parametrize("speakers", [
    "A = Usha; B = Fatima; C = Rania; D = Vikram",
    "A = Usha; B = Fatima; C = Rania; D = Vikram;",
    "A: Usha; B: Fatima; C: Rania; D: Vikram",
    "A = Usha, B = Fatima, C = Rania, D = Vikram",
    "A (Usha); B (Fatima); C (Rania); D (Vikram)",
])
def test_speaker_forms(speakers):
    assert parse_speakers(speakers) == SPEAKERS


def test_a_comma_inside_a_name_stays_when_pairs_are_split_by_semicolons():
    assert parse_speakers("A = Rao, Usha; B = Fatima")["A"] == "Rao, Usha"
