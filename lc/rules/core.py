"""Rule-based solver core.

A family recognises one kind of evidence (for example "nobody could enter X between T1 and T2") and judges the chat
messages that make claims about it. Each family is precise rather than broad: it returns a verdict only for messages
it fully understands and stays silent otherwise. A case is answered when exactly one speaker has a message judged
to be a lie; everything else is left to the language-model judge.
"""
from dataclasses import dataclass, field

from .text import find_dates, is_filler, split_sentences

LIE = "lie"
OK = "ok"

REGISTRY = []


def register(cls):
    """Class decorator: add one instance of a Family subclass to the solver."""
    REGISTRY.append(cls())
    return cls


@dataclass
class Verdict:
    status: str  # LIE or OK
    family: str
    why: str = ""


class Family:
    """Base class. Subclasses set `name` and implement parse() and judge()."""

    name = "base"

    def parse(self, ctx):
        """Return parsed facts if this family's evidence is present, else None. Call ctx.consume(i) for every
        evidence sentence index you understood, so the solver knows which sentences nobody parsed."""
        return None

    def judge(self, ctx, facts, msg):
        """Return Verdict(LIE|OK, self.name, why) for a message this family fully understands, else None.
        LIE only when the claim cannot be true together with the evidence. msg has .num, .label, .text;
        ctx.speakers maps the sender's label to their name."""
        return None


class Ctx:
    """Everything a family may need about one case."""

    def __init__(self, case):
        self.case = case
        self.sentences = split_sentences(case.evidence)
        self.speakers = case.speakers  # label -> name
        self.label_of = {v: k for k, v in case.speakers.items()}  # name -> label
        self.consumed = set()
        self.day = None  # the date all times refer to, when the evidence says so
        for i, s in enumerate(self.sentences):
            if s.startswith("All times in this case refer to"):
                d = find_dates(s)
                if len(d) == 1:
                    self.day = d[0][0]
                    self.consume(i)

    def consume(self, i):
        self.consumed.add(i)

    def unconsumed(self):
        return [s for i, s in enumerate(self.sentences) if i not in self.consumed]


@dataclass
class Result:
    id: str
    pred: str = None  # the letter, when exactly one speaker has a LIE verdict
    lie_labels: list = field(default_factory=list)
    verdicts: dict = field(default_factory=dict)  # message num -> [Verdict]
    unknown_claims: list = field(default_factory=list)  # message nums neither filler nor judged
    unconsumed: list = field(default_factory=list)  # evidence sentences no family parsed
    families: list = field(default_factory=list)  # families whose evidence was found
    errors: list = field(default_factory=list)  # exceptions raised by families (family, message)

    @property
    def confident(self):
        """One liar found and every claim in the chat was judged."""
        return self.pred is not None and not self.unknown_claims


def solve(case, registry=None):
    ctx = Ctx(case)
    res = Result(case.id)
    active = []
    for fam in registry or REGISTRY:
        try:
            facts = fam.parse(ctx)
        except Exception as e:  # a family bug must never sink the whole case
            res.errors.append((fam.name, f"parse: {e!r}"))
            continue
        if facts is not None:
            active.append((fam, facts))
            res.families.append(fam.name)
    lie_labels = set()
    for msg in case.messages:
        vs = []
        for fam, facts in active:
            try:
                v = fam.judge(ctx, facts, msg)
            except Exception as e:
                res.errors.append((fam.name, f"judge {msg.num}: {e!r}"))
                continue
            if v is not None:
                vs.append(v)
        if vs:
            res.verdicts[msg.num] = vs
            if any(v.status == LIE for v in vs):
                lie_labels.add(msg.label)
        elif not is_filler(msg.text):
            res.unknown_claims.append(msg.num)
    res.lie_labels = sorted(lie_labels)
    if len(lie_labels) == 1:
        res.pred = res.lie_labels[0]
    res.unconsumed = ctx.unconsumed()
    return res
