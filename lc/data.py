"""Load the competition CSVs into Case objects."""
import csv
import os
import re
from dataclasses import dataclass, field

DATA_DIR = os.environ.get("LC_DATA_DIR", "data")
MSG_RE = re.compile(r"^\s*(\d+)\.\s*([ABCD]):\s*(.*?)\s*$")


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
    out = {}
    for part in s.split(";"):
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_chat(chat):
    msgs = []
    for line in chat.strip().splitlines():
        m = MSG_RE.match(line)
        if not m:
            raise ValueError(f"bad chat line: {line!r}")
        msgs.append(Message(int(m.group(1)), m.group(2), m.group(3)))
    return msgs


def case_from_row(r):
    return Case(r["id"], r["scenario"], parse_speakers(r["speakers"]), r["evidence"], parse_chat(r["chat"]), r.get("label"))


def load(split, data_dir=None):
    path = os.path.join(data_dir or DATA_DIR, f"{split}.csv")
    with open(path, encoding="utf-8", newline="") as f:
        return [case_from_row(r) for r in csv.DictReader(f)]
