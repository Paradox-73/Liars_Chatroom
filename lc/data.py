"""Load the competition CSVs into Case objects."""
import csv
import os
import re
from dataclasses import dataclass, field

DATA_DIR = os.environ.get("LC_DATA_DIR", "data")
# "12. B: text", also "12) B:", "b:" and "B (Name):"; the number may be missing when no line has one
MSG_RE = re.compile(r"^\s*(?:(\d+)\s*[.)]\s*)?([A-Da-d])(?:\s*\([^)]*\))?\s*:\s*(.*?)\s*$")
SPEAKER_RE = re.compile(r"\s*([A-Za-z])\s*([=:(])\s*(.*?)\s*$")  # "A = Name", "A: Name", "A (Name)"


@dataclass
class Message:
    num: int
    label: str
    text: str


@dataclass
class Case:
    id: str
    scenario: str
    speakers: dict  # label -> name
    evidence: str
    messages: list
    label: str = None  # train only
    extra: dict = field(default_factory=dict)

    @property
    def names(self):
        return {v: k for k, v in self.speakers.items()}  # name -> label


def parse_speakers(s):
    """label -> name from "A = Name; B = Name; ..."; commas between the pairs when there is no ';', empty parts skipped."""
    out = {}
    for part in s.split(";" if ";" in s else ","):
        m = SPEAKER_RE.match(part)
        if m:
            name = m.group(3)
            out[m.group(1).upper()] = name[:-1].strip() if m.group(2) == "(" and name.endswith(")") else name
    if not out:
        raise ValueError("no speaker in the form A = name")
    return out


def parse_chat(chat):
    """The "N. A: text" messages (see MSG_RE). A line in another form continues the message before it (one dropped before
    the first message), so one odd line no longer makes the whole case unreadable; blank lines are skipped. Without
    numbers the messages are numbered 1, 2, ...; in a numbered chat a line without a number is a continuation."""
    lines = chat.strip().splitlines()
    found = [MSG_RE.match(line) for line in lines]
    numbered = any(m and m.group(1) for m in found)
    msgs = []
    for line, m in zip(lines, found):
        if m and (m.group(1) or not numbered):
            msgs.append(Message(int(m.group(1)) if m.group(1) else len(msgs) + 1, m.group(2).upper(), m.group(3)))
        elif line.strip() and msgs:
            msgs[-1].text = f"{msgs[-1].text} {line.strip()}"
    if not msgs:
        raise ValueError("no chat line in the form N. A: text")
    return msgs


def case_from_row(r):
    return Case(r["id"], r["scenario"], parse_speakers(r["speakers"]), r["evidence"], parse_chat(r["chat"]), r.get("label"))


def load(split, data_dir=None):
    path = os.path.join(data_dir or DATA_DIR, f"{split}.csv")
    with open(path, encoding="utf-8", newline="") as f:
        return [case_from_row(r) for r in csv.DictReader(f)]
