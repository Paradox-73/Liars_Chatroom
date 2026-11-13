"""Shared pieces of the new-family generators: names, clock and calendar wording, the claim model.

A family module has NAME and make(rng, people, others, day) -> Bundle. `people` are the four speakers' names,
`others` are spare names for people outside the chat, and `day` is the date the "All times in this case refer to"
line names. The bundle holds the evidence sentences, the fact model, exactly one lie and a pool of claims that are
all true in one hidden world that satisfies the evidence.

Every claim carries ok(facts): a check that re-derives from the fact model whether the claim can be true together
with the evidence. The lie is built to fail it and every other claim to pass it; build.py runs the checks on every
message of every chat and asserts that exactly one fails.
"""
import datetime as dt
from dataclasses import dataclass, field

NAMES = [
    "Aarav", "Abena", "Adaeze", "Ade", "Aditi", "Ahmed", "Aiko", "Akira", "Alejandro", "Amara", "Amira", "Ananya",
    "Andrei", "Anika", "Aruna", "Asha", "Ayesha", "Bao", "Beatriz", "Bilal", "Bongani", "Bruno", "Camila", "Carlos",
    "Carmen", "Chen", "Chidi", "Chloe", "Daniel", "Dara", "Deepa", "Dev", "Diego", "Dmitri", "Elena", "Elif",
    "Emeka", "Esha", "Eun", "Farid", "Fatima", "Felix", "Femi", "Fiona", "Gabriel", "Gita", "Goran", "Grace",
    "Hana", "Hari", "Hassan", "Hira", "Iker", "Ines", "Ingrid", "Ishaan", "Isla", "Ivan", "Jana", "Jaya", "Jin",
    "Joel", "Jonas", "Jun", "Kabir", "Kai", "Karim", "Kavya", "Kemal", "Kenji", "Kiran", "Kofi", "Laila",
    "Lakshmi", "Lena", "Leo", "Liam", "Lin", "Lucia", "Lucy", "Malik", "Marco", "Maria", "Marta", "Mateo", "Maya",
    "Meera", "Mehmet", "Mei", "Mira", "Mohan", "Nadia", "Naledi", "Nell", "Nikhil", "Nina", "Noor", "Nora", "Obi",
    "Olga", "Omar", "Omari", "Oscar", "Pablo", "Pia", "Priya", "Quinn", "Rafael", "Rahul", "Rania", "Ravi", "Reza",
    "Rhea", "Rohan", "Rosa", "Rui", "Sami", "Sana", "Sara", "Selin", "Sofia", "Sunil", "Tara", "Tariq", "Taro",
    "Theo", "Tomas", "Ugo", "Uma", "Usha", "Vera", "Vida", "Vikram", "Wei", "Wren", "Ximena", "Yara", "Yuki",
    "Yusuf", "Zane", "Zara", "Zeynep", "Zoya",
    # names not used in train
    "Abel", "Adele", "Alma", "Anton", "Arash", "Aya", "Bella", "Cyrus", "Dario", "Ebru", "Ezra", "Freya", "Hugo",
    "Ilse", "Imani", "Kaito", "Keira", "Lars", "Leila", "Mina", "Nils", "Odette", "Pavel", "Rina", "Sergio",
    "Tomasz", "Vani", "Yosef", "Zainab",
]

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve"]
ORDINALS = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
DAY_END = 23 * 60 + 59  # last minute of a day; clock times are minutes since midnight


def clock(minutes):
    """Minutes since midnight as '9:05 AM' or '12:00 PM'."""
    h, m = divmod(int(minutes), 60)
    assert 0 <= h < 24, minutes
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def date_text(d):
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def weekday(d):
    return WEEKDAYS[d.weekday()]


def anchor(d):
    """A weekday anchor sentence for the first day of d's month, as in '1 April 2026 was a Wednesday.'"""
    first = d.replace(day=1)
    return f"{date_text(first)} was a {weekday(first)}."


def duration(minutes):
    """'2 hours', '1 hour 30 minutes', '45 minutes'."""
    h, m = divmod(int(minutes), 60)
    parts = []
    if h:
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m or not h:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    return " ".join(parts)


def money(cents, cur="£"):
    """Whole cents as '£4.50'."""
    return f"{cur}{cents // 100}.{cents % 100:02d}"


def an(noun):
    """'an iron tablet', 'a painkiller'."""
    return ("an " if noun[0].lower() in "aeiou" else "a ") + noun


def word(n):
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)


def random_day(rng, year=2026):
    return dt.date(year, 1, 1) + dt.timedelta(days=rng.randrange(365))


@dataclass
class Claim:
    who: str  # the sender's name
    text: str  # one sentence, without the number and label
    ok: object  # ok(facts) -> True when the claim can be true together with the evidence
    tags: tuple = ()  # boundary, two-fact, about-other, distractor, neutral, ...


@dataclass
class Bundle:
    family: str
    scenario: str
    group: str  # who is talking, for combined scenarios: "Carers"
    topic: str  # what about, for combined scenarios: "Kofi's tablets"
    evidence: list  # sentences, without the "All times in this case refer to" line
    facts: object  # the fact model every ok() reads
    lie: Claim
    truths: list  # claims all true in one hidden world that satisfies the evidence
    fillers: list = field(default_factory=list)  # family-specific questions and opinions
    decor: list = field(default_factory=list)  # harmless decoration sentences
    timed: bool = True  # the evidence needs the "All times in this case refer to <day>" line


def facts(**kw):
    """A plain attribute bag for fact models."""
    return type("Facts", (), kw)()


GENERIC_QUESTIONS = [
    "Who is taking the notes?", "Should we check with the manager?", "Can we keep this brief?",
    "Is there more to go on?", "Does anyone have anything to add?", "Can we go over it once more?",
    "Would the manager know more?", "Is someone writing this up?", "Shall we start again from the top?",
    "Has anyone got further details?", "Could we keep this thread short?", "Who's writing up the notes?",
    "Any more details, anyone?", "Should the manager be told?", "Can we get a plain answer?",
]
GENERIC_OPINIONS = [
    "This is getting muddled.", "I'd like a clear answer here.", "Let's keep this thread brief.",
    "Thanks, everyone, for checking.", "This is all rather confusing.", "Thanks for going through this, everyone.",
    "I'd like a straight answer on this.", "This is getting tangled.", "Let's keep it short.",
    "Some clarity would be good.", "Cheers for checking, all.", "Let's not drag this out.",
    "Thanks for sorting this out, everyone.", "This is a bit of a muddle.", "I want a clear answer on this.",
]

LEADS = ["Honestly, ", "For the record, ", "Okay so ", "To be clear, "]
# Sentence openers that may be lowercased after a lead-in; anything else (names, "I", numbers) keeps its case.
LOWER_OK = {"The", "At", "On", "In", "By", "Between", "During", "When", "After", "Before", "There", "My", "Our",
            "That", "This", "From", "Every", "Each", "Both", "Only", "Nobody", "It", "Just"}


def lead(rng, text):
    """Sometimes add a lead-in ('Honestly, ') or a tail (', for the record.') the way train chats do."""
    r = rng.random()
    first = text.split(" ", 1)[0]
    if r < 0.14 and (first in ("I", "I'd", "I'm") or first in LOWER_OK):
        body = text if first.startswith("I") else text[0].lower() + text[1:]
        return rng.choice(LEADS) + body
    if r < 0.19 and text.endswith("."):
        return text[:-1] + ", for the record."
    return text
