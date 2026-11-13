"""Parsing helpers shared by the rule families: clock times, dates, durations, numbers, sentences."""
import datetime as dt
import re

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUM_WORDS = {"zero": 0, "no": 0, "none": 0, "one": 1, "a single": 1, "single": 1, "two": 2, "both": 2, "three": 3,
             "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
             "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
             "eighteen": 18, "nineteen": 19, "twenty": 20, "half": None}

TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*(AM|PM)\b|\b(noon|midday|midnight)\b", re.I)
DATE_RE = re.compile(r"\b(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})\b")
WEEKDAY_RE = re.compile(r"\b(" + "|".join(WEEKDAYS) + r")s?\b")
DUR_RE = re.compile(
    r"\b(\d+|an?|one|two|three|four|five|six)\s*(hours?|hrs?)(?:\s*(?:and\s*)?(\d+)\s*(minutes?|mins?))?"
    r"|\b(\d+)\s*(minutes?|mins?)\b", re.I)
NUM_RE = re.compile(r"(?<![\d:])(\d+(?:\.\d+)?)(?![\d:])")


def time_to_min(h, m, ampm):
    h, m = int(h), int(m)
    if not (1 <= h <= 12 and 0 <= m < 60):
        raise ValueError("bad clock time")
    ampm = ampm.upper()
    if ampm == "AM":
        h = 0 if h == 12 else h
    else:
        h = 12 if h == 12 else h + 12
    return h * 60 + m


def find_times(s):
    """All clock times in s as (minutes since midnight, match). noon=720, midnight=0."""
    out = []
    for m in TIME_RE.finditer(s):
        if m.group(4):
            word = m.group(4).lower()
            out.append((0 if word == "midnight" else 720, m))
        else:
            out.append((time_to_min(m.group(1), m.group(2), m.group(3)), m))
    return out


def parse_time(s):
    """The single clock time in s, in minutes since midnight, or None when there is none or more than one."""
    t = find_times(s)
    return t[0][0] if len(t) == 1 else None


def fmt_time(minutes):
    h, m = divmod(int(minutes) % 1440, 60)
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def find_dates(s):
    """All dates like '14 April 2026' in s as (datetime.date, match)."""
    out = []
    for m in DATE_RE.finditer(s):
        out.append((dt.date(int(m.group(3)), MONTHS.index(m.group(2)) + 1, int(m.group(1))), m))
    return out


def parse_date(s):
    d = find_dates(s)
    return d[0][0] if len(d) == 1 else None


def find_weekdays(s):
    return [WEEKDAYS.index(m.group(1)) for m in WEEKDAY_RE.finditer(s)]


def word_num(w):
    w = w.lower().strip()
    if w in ("a", "an"):
        return 1
    if w.isdigit():
        return int(w)
    return NUM_WORDS.get(w)


def find_durations(s):
    """Durations in minutes found in s ('1 hour 20 minutes', '45 minutes', 'two hours')."""
    out = []
    for m in DUR_RE.finditer(s):
        if m.group(1):
            h = word_num(m.group(1))
            if h is None:
                continue
            out.append((h * 60 + (int(m.group(3)) if m.group(3) else 0), m))
        else:
            out.append((int(m.group(5)), m))
    return out


def find_numbers(s):
    """Plain numbers in s that are not part of a clock time or a date's day/year (floats kept)."""
    masked = TIME_RE.sub(lambda m: " " * len(m.group(0)), s)
    masked = DATE_RE.sub(lambda m: " " * len(m.group(0)), masked)
    return [float(m.group(1)) if "." in m.group(1) else int(m.group(1)) for m in NUM_RE.finditer(masked)]


def split_sentences(text):
    """Split evidence into sentences. Times like '9:40 AM.' end sentences; decimals do not."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def norm(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9:'% ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


OPINION_RE = re.compile(
    r"^(honestly,?\s*)?(this is|things are|it's|it is|that's|that is)\b.*\b(confus|muddl|tangl|mess|hard to follow|unclear|odd|strange|weird)"
    r"|^(honestly,?\s*)?(thanks|thank you|cheers|appreciate)\b"
    r"|^(i'd|i would|i want|we need|someone please|some clarity|can someone)\b.*\b(answer|clarity|clear)\b"
    r"|^(let's|let us|keep it|please keep)\b"
    r"|^(some clarity)\b"
    r"|^we need (a|an|more|better|faster)\b", re.I)


def is_filler(text):
    """Questions and opinions, which are never the answer."""
    t = text.strip()
    if t.endswith("?"):
        return True
    if re.search(r"\d", t):
        return False
    return bool(OPINION_RE.search(t))
