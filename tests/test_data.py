"""python -m pytest tests"""
import pytest

from lc.data import parse_chat


def test_an_odd_line_continues_the_message_before_it():
    chat = "Chat log:\n1. A: I was in the lab at 9:40 AM.\n\n2. B: Priya never entered\nthe lab after noon.\n3. C: Same."
    assert [(m.num, m.label, m.text) for m in parse_chat(chat)] == [
        (1, "A", "I was in the lab at 9:40 AM."), (2, "B", "Priya never entered the lab after noon."), (3, "C", "Same.")]


def test_a_chat_without_any_message_line_is_unreadable():
    with pytest.raises(ValueError):
        parse_chat("A said she was in the lab.\nB said he was not.")
