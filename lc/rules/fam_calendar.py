"""Rule families: calendar, schedules and ages. See lc/rules/core.py for the Family interface.

cal_hours    a place open only on some weekdays and hours of a month ("In April 2026 the pool was open only ...")
cal_closed   a place nobody could be inside on certain weekdays ("... on any Tuesday in April 2026")
cal_rota     exactly one person did a nightly duty over a date range, following a rota
cal_session  the only session of the day (tour, class, rehearsal), with a start time and an exact duration
cal_age      birth dates against claims like "I turned 28 in July 2026" and "On 12 March 2026 I was 27"
cal_owner    who owned an item on each day of a month, from a chain of sales

Every evidence pattern matches a whole sentence, so a sentence with a clause these families do not model (an
exception, a second schedule) is left unparsed. A family also abstains when another evidence sentence mentions the
same place, duty, session, person or item. Claims are judged only when plain: questions, hedged or negated
sentences and reported speech are left alone, and so are messages naming another person, except in sessions (nobody
could take part outside the session) and sales (the name is the other side of the sale).
"""
import datetime as dt
import re

from .core import LIE, OK, Family, Verdict, register
from . import text as T

MONTH_ALT = "|".join(T.MONTHS)
WD_ALT = "|".join(T.WEEKDAYS)
DATE_PAT = rf"\d{{1,2}} (?:{MONTH_ALT}) \d{{4}}"
MY_PAT = rf"(?:{MONTH_ALT}) \d{{4}}"
TIME_PAT = r"(?:\d{1,2}:\d{2} [AP]M|noon|midday|midnight)"
DUR_PAT = r"(?:(?:\d+|an?|one|two|three|four|five|six) hours?(?: (?:and )?\d+ minutes?)?|\d+ minutes?)"
NP = r"[a-z][a-z' -]*?"  # a short lower-case noun phrase
NAME = r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?"
NOBODY = r"(?:nobody|no one)"
DASH = "[-–—]"
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "last": -1}
# a month and year that is not the tail of a full date ("in April 2026", not "1 April 2026")
MONTH_YEAR_RE = re.compile(rf"(?<!\d\s)\b({MONTH_ALT})\s+(?:of\s+)?(\d{{4}})\b")
ANCHOR_RE = re.compile(rf"^({DATE_PAT}) was an? ({WD_ALT})\.$")

# conversational frames the generator wraps around claims
FRAME_HEAD_RE = re.compile(r"^(?:(?:honestly|for the record|okay so|ok so|so|well|frankly|to be honest|as it happens)"
                           r"\b\s*[,:]?\s*)+", re.I)
FRAME_TAIL_RE = re.compile(r"[,;]?\s*\b(?:honestly|for the record|as it happens|to be honest)\s*(?=[.!]?$)", re.I)
# any of these makes a message something other than a plain claim of fact
DOUBT_RE = re.compile(
    r"n't\b|\b(?:not|never|no|nobody|none|nothing|cannot|maybe|perhaps|probably|possibly|might|(?-i:may)|must|think|"
    r"thought|guess|believe|suppose|reckon|apparently|supposedly|unless|if|whether|almost|nearly|tried|try|trying|"
    r"planned|plan|meant|wanted|hoped|would|should|could|supposed|instead|except|only|until|before|after|between|"
    r"since|told|tells|said|says|claimed|claims|mentioned|according|heard|reported)\b",
    re.I)  # 'may' only in lower case: 'May' is a month
FIRST_PERSON_RE = re.compile(r"\b(?:I|[Mm]e|[Mm]y|[Mm]ine)\b")
CAPITAL_RE = re.compile(r"\b[A-Z][a-z][A-Za-z'-]*")
NOT_NAMES = set(T.MONTHS) | set(T.WEEKDAYS)
# claims of waiting near a place rather than using it; always possible
WAIT_RE = re.compile(r"\b(?:outside|waiting|waited|wait|stood|standing|queued|queuing|queueing)\b", re.I)
# words that may follow 'the pool' in a claim about the pool itself (not 'the pool cafe', not "the pool's")
AFTER_NOUN = (r"(?=\s*(?:[.,;:!]|$)|\s+(?:at|on|in|from|for|with|and|that|when|while|was|is|around|about|during|"
              r"by|to|then|so|but|as|a|I|today|yesterday|myself|again|alone|too)\b)")


def claim_text(text):
    """The message without 'Honestly,' / ', for the record' style frames."""
    t = FRAME_HEAD_RE.sub("", text.strip())
    return FRAME_TAIL_RE.sub("", t).strip()


def plain_claim(text, names_ok=False):
    """The framed-off claim, or None for a question, a hedged or negated sentence, reported speech, or (unless
    names_ok) a message that names someone: any capitalised word after the first that is not a month or weekday."""
    t = claim_text(text)
    if not t or t.endswith("?") or DOUBT_RE.search(t):
        return None
    if not names_ok and any(m.start() > 0 and m.group(0) not in NOT_NAMES for m in CAPITAL_RE.finditer(t)):
        return None
    return t


def the_noun(noun):
    """'the <noun>' as the place itself."""
    return r"\bthe " + re.escape(noun) + AFTER_NOUN


def the_date(t):
    """The one full date in a claim, or None when there is none, several, or another bare number besides clock
    times (so 'on 5 and 6 March 2026' and 'from 8 to 20 September 2026' are not read as one day)."""
    dates = T.find_dates(t)
    return dates[0][0] if len(dates) == 1 and not T.find_numbers(t) else None


def month_years(s):
    return [(int(m.group(2)), T.MONTHS.index(m.group(1)) + 1) for m in MONTH_YEAR_RE.finditer(s)]


def first_match(patterns, s):
    for rx in patterns:
        m = rx.match(s)
        if m:
            return m
    return None


def matches(ctx, patterns):
    """[(sentence index, match)] for every evidence sentence that one of the whole-sentence patterns matches."""
    return [(i, m) for i, s in enumerate(ctx.sentences) for m in [first_match(patterns, s)] if m]


def mentioned_elsewhere(ctx, used, words):
    """True when an evidence sentence this family did not parse (anchors aside) names one of the words: then the
    evidence may say more about the subject than the family understands."""
    rx = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.I)
    return any(i not in used and not ANCHOR_RE.match(s) and rx.search(s) for i, s in enumerate(ctx.sentences))


def find_anchor(ctx):
    """'1 April 2026 was a Wednesday.' -> (date, weekday index), consumed. With several anchors they must agree."""
    found = []
    for i, s in enumerate(ctx.sentences):
        m = ANCHOR_RE.match(s)
        if m:
            found.append((i, T.parse_date(m.group(1)), T.WEEKDAYS.index(m.group(2))))
    if not found:
        return None
    _, d0, wd0 = found[0]
    if any((wd0 + (d - d0).days) % 7 != wd for _, d, wd in found):
        return False  # contradictory anchors: no weekday arithmetic at all
    for i, _, _ in found:
        ctx.consume(i)
    return d0, wd0


def weekday(d, anchor):
    """Weekday index of d (Monday=0), counted from the evidence's anchor when it gives one."""
    if anchor:
        return (anchor[1] + (d - anchor[0]).days) % 7
    return d.weekday()


def weekday_span(a, b):
    """'Wednesday to Sunday' -> {2, 3, 4, 5, 6}; wraps past Sunday."""
    i, j = T.WEEKDAYS.index(a), T.WEEKDAYS.index(b)
    return {(i + k) % 7 for k in range((j - i) % 7 + 1)}


def waiting_only(t):
    """'I waited outside the pool on 14 April 2026': a single clause about waiting, with no clock time."""
    t = re.sub(rf"({DATE_PAT}),", r"\1", t)  # 'On 21 December 2026, I waited ...' is still one clause
    return WAIT_RE.search(t) and not T.find_times(t) and not re.search(r",|\b(?:and|then)\b", t)


# ----------------------------------------------------------------------------------------------- opening hours

MONTH_SCOPE = rf"(?:in|during|throughout) {MY_PAT}"
HOURS_PAT = rf"(?:from |between )?{TIME_PAT} (?:to|and) {TIME_PAT}"
CLOSE_IT = rf"{NOBODY} could use it while it was closed"
HOURS_RES = [
    re.compile(rf"^(?:(?i:{MONTH_SCOPE}),? )?(?i:the) (?P<noun>{NP}) (?:was open|opened) only (?:from )?"
               rf"(?P<d1>{WD_ALT}) to (?P<d2>{WD_ALT})(?:,? {MONTH_SCOPE})?, {HOURS_PAT}(?:,? {MONTH_SCOPE})?"
               rf"(?P<close>(?:,? and |; ){CLOSE_IT})?\.$"),
    re.compile(rf"^(?i:{NOBODY}) could use the (?P<noun>{NP}) while it was closed(?P<close>), and {MONTH_SCOPE} it was "
               rf"open only (?:from )?(?P<d1>{WD_ALT}) to (?P<d2>{WD_ALT}), {HOURS_PAT}\.$"),
]
CLOSE_RE = re.compile(rf"^(?i:{NOBODY}) could use (?:it|the (?P<noun>{NP})) while it was closed\.$")
# not a use of the place: being near it or on the way, contacting it, seeing it, working on it
NOT_USE_RE = re.compile(
    r"\b(?:near|nearby|past|passed|front|across|opposite|behind|next|beside|phoned|called|rang|emailed|messaged|"
    r"texted|wrote|looked|saw|seen|watched|noticed|website|online|closed|shut|locked|unlocked|booked|reserved|by|"
    r"arrived|drove|walked|cycled|headed|reached|travelled|traveled|left|got|came|turned|showed|parked|parking|"
    r"dropped|returned|posted|ticket|tickets|photo|photos|photographed|picture|pictures|cleaned|cleaning|repaired|"
    r"fixed|inspected|checked(?!(?: (?:a|an|some|the|my))? [a-z]+ out\b)|delivered|painted|guarded|patrolled|"
    r"maintained|worked(?! out)|working(?! out)|staff|security|volunteered)\b",
    re.I)  # 'worked out at the gym' and 'checked a book out of the library' are uses; 'worked at the gym' may be a job


def use_re(noun):
    """Claims of using the place, in the wordings train uses for that ('I swam at the pool', 'I borrowed a book from
    the library', 'I bought bread at the bakery', 'I visited the museum', 'My visit to the museum was ...') plus a few
    plain equivalents (used, trained, worked out, ate, played). Any other wording is left unjudged."""
    n = the_noun(noun)[2:]  # 'the pool' as the place itself
    obj = r"(?:(?:a|an|some|the|my) )?[a-z]+"  # 'a book', 'bread', 'some bread'
    return re.compile("|".join([
        rf"\bI (?:swam|went swimming|went for a swim|had a swim|trained|worked out|played|ate) (?:at|in) {n}",
        rf"\bI (?:visited|used|went to|paid a visit to) {n}",
        rf"\bI paid {n} a visit\b",
        rf"\bI (?:borrowed|bought|picked up) {obj} (?:at|from) {n}",
        rf"\bI (?:took|checked) {obj} out (?:of|from) {n}",
        rf"\bI borrowed (?:a|an) {re.escape(noun)} book\b",
        rf"^My (?:visit|swim|[a-z]+ purchase) (?:to|at) {n} was at\b",
        rf"^At {n}, .*\bI (?:bought|borrowed|picked up|swam|ate|played|trained|worked out)\b",
    ]), re.I)


@register
class OpeningHours(Family):
    """'In April 2026 the pool was open only Wednesday to Sunday, from 9:00 AM to 5:00 PM, and nobody could use it
    while it was closed.' A first-person claim of using the place on a closed weekday, or at a time outside the
    hours (opening and closing times count as open), is a lie. Waiting outside is always possible; other months are
    not covered."""

    name = "cal_hours"

    def parse(self, ctx):
        hits = matches(ctx, HOURS_RES)
        if len(hits) != 1:
            return None
        i, m = hits[0]
        s, noun, used = ctx.sentences[i], m.group("noun"), [i]
        if m.group("close") is None:  # 'Nobody could use it while it was closed.' as its own sentence
            closes = [(j, c) for j, x in enumerate(ctx.sentences) for c in [CLOSE_RE.match(x)] if c]
            if len(closes) != 1:
                return None
            j, c = closes[0]
            if c.group("noun") != noun and not (c.group("noun") is None and j == i + 1):
                return None
            used.append(j)
        times, months = [t for t, _ in T.find_times(s)], month_years(s)
        if len(times) != 2 or times[0] >= times[1] or len(months) > 1:
            return None
        if mentioned_elsewhere(ctx, used, [noun, noun.split()[-1]]):
            return None
        anchor = find_anchor(ctx)
        if anchor is False:
            return None
        for k in used:
            ctx.consume(k)
        return {"noun": noun, "days": weekday_span(m.group("d1"), m.group("d2")), "hours": times,
                "month": months[0] if months else None, "anchor": anchor, "use": use_re(noun)}

    def judge(self, ctx, f, msg):
        t = plain_claim(msg.text)
        if not t or not re.search(the_noun(f["noun"]), t, re.I) or not FIRST_PERSON_RE.search(t):
            return None
        d = the_date(t)
        if d is None:
            return None
        if waiting_only(t):
            return Verdict(OK, self.name, "waiting outside is always possible")
        times = T.find_times(t)
        if not f["use"].search(t) or WAIT_RE.search(t) or NOT_USE_RE.search(t) or len(times) > 1:
            return None  # only wordings known to mean using the place are judged
        if f["month"] and (d.year, d.month) != f["month"]:
            return Verdict(OK, self.name, "the opening days cover another month")
        wd = weekday(d, f["anchor"])
        if wd not in f["days"]:
            return Verdict(LIE, self.name, f"{d} was a {T.WEEKDAYS[wd]}, a closed day")
        if not times:
            return Verdict(OK, self.name, "an open day")
        if not f["hours"][0] <= times[0][0] <= f["hours"][1]:
            return Verdict(LIE, self.name, f"{T.fmt_time(times[0][0])} is outside the opening hours")
        return Verdict(OK, self.name, "open at that day and time")


# ----------------------------------------------------------------------------------------------- closed weekdays

ANY_DAY = rf"(?:any|every) (?P<wd>{WD_ALT}) (?:in|during) (?P<mon>{MONTH_ALT}) (?P<yr>\d{{4}})"
NTH_DAY = (rf"the (?P<ord>first|second|third|fourth|fifth|last) (?P<wd>{WD_ALT}) (?:of|in) (?P<mon>{MONTH_ALT}) "
           rf"(?P<yr>\d{{4}})")
INSIDE = rf"could be inside the (?P<noun>{NP})"
CLOSED_RES = [
    re.compile(rf"^(?i:{NOBODY}) {INSIDE} on {ANY_DAY}\.$"),
    re.compile(rf"^(?i:{NOBODY}) {INSIDE} on {NTH_DAY}\.$"),
    re.compile(rf"^On {ANY_DAY},? {NOBODY} {INSIDE}\.$"),
    re.compile(rf"^On {NTH_DAY},? {NOBODY} {INSIDE}\.$"),
    re.compile(rf"^(?:In|During) (?P<mon>{MONTH_ALT}) (?P<yr>\d{{4}}),? {NOBODY} {INSIDE} on (?:any|every) "
               rf"(?P<wd>{WD_ALT})\.$"),
]


def inside_re(noun):
    """Claims of being inside the place, in the wordings train uses ('I was inside the library', 'Inside the library
    is where I was', '14 April 2026 found me inside the library') plus visiting or entering it."""
    n = the_noun(noun)[2:]
    return re.compile(rf"\bI was (?:inside|in) {n}|\bI (?:visited|entered|went into|went inside) {n}"
                      rf"|\b{n} is where I was\b|\bfound me (?:inside|in) {n}", re.I)


def month_days(y, m):
    d = dt.date(y, m, 1)
    while d.month == m:
        yield d
        d += dt.timedelta(days=1)


@register
class ClosedWeekday(Family):
    """'Nobody could be inside the library on any Tuesday in April 2026' (or 'on the second Tuesday of ...').
    A first-person claim of being inside on one of those dates is a lie; other dates and waiting outside are fine."""

    name = "cal_closed"

    def parse(self, ctx):
        hits = matches(ctx, CLOSED_RES)
        if len(hits) != 1:
            return None
        i, m = hits[0]
        noun = m.group("noun")
        if mentioned_elsewhere(ctx, [i], [noun, noun.split()[-1]]):
            return None
        anchor = find_anchor(ctx)
        if anchor is False:
            return None
        y, mo, wd = int(m.group("yr")), T.MONTHS.index(m.group("mon")) + 1, T.WEEKDAYS.index(m.group("wd"))
        days = [d for d in month_days(y, mo) if weekday(d, anchor) == wd]
        if "ord" in m.groupdict():
            k = ORDINALS[m.group("ord")]
            if k > len(days):
                return None
            days = [days[k - 1] if k > 0 else days[-1]]
        ctx.consume(i)
        return {"noun": noun, "closed": set(days), "inside": inside_re(noun)}

    def judge(self, ctx, f, msg):
        t = plain_claim(msg.text)
        if not t or not re.search(the_noun(f["noun"]), t, re.I) or not FIRST_PERSON_RE.search(t):
            return None
        d = the_date(t)
        if d is None:
            return None
        if waiting_only(t):
            return Verdict(OK, self.name, "waiting outside is always possible")
        if WAIT_RE.search(t) or NOT_USE_RE.search(t) or not f["inside"].search(t):
            return None
        if d in f["closed"]:
            return Verdict(LIE, self.name, f"nobody could be inside on {d}")
        return Verdict(OK, self.name, "not a closed day")


# ----------------------------------------------------------------------------------------------- nightly rota

SPAN_PAT = rf"(?:from {DATE_PAT} to {DATE_PAT}|between {DATE_PAT} and {DATE_PAT})"
ROTA_TAIL = r",? (?:following this rota exactly|and the rota was followed exactly): (?P<list>.+)\.$"
ROTA_RES = [
    re.compile(rf"^(?i:{SPAN_PAT}),? exactly one person worked (?P<duty>the {NP}) (?:each|every) night{ROTA_TAIL}"),
    re.compile(rf"^(?i:each|every) night {SPAN_PAT},? exactly one person worked (?P<duty>the {NP}){ROTA_TAIL}"),
    re.compile(rf"^(?i:exactly) one person worked (?P<duty>the {NP}) (?:each|every) night {SPAN_PAT}{ROTA_TAIL}"),
    re.compile(rf"^(?i:{SPAN_PAT}),? (?P<duty>the {NP}) was covered (?:each|every) night by exactly one person"
               rf"{ROTA_TAIL}"),
]
PART_DATES_RE = re.compile(rf"^({NAME}) (?:on|took) (\d{{1,2}})(?: ({MONTH_ALT}))?(?:\s*{DASH}\s*(\d{{1,2}}))? "
                           rf"({MONTH_ALT})$")
PART_COUNT_RE = re.compile(rf"^(?:({NAME}) took the (first|next|last) (\w+)(?: nights?)?"
                           rf"|the (first|next|last) (\w+)(?: nights?)? (?:went )?to ({NAME}))$")
# words that name a different duty or shift when the evidence's duty lacks them
DUTY_QUALIFIERS = {"day", "night", "late", "early", "morning", "afternoon", "evening", "weekend", "weekday",
                   "call-out", "front", "desk", "back", "office", "kitchen", "door", "gate", "reception", "bar",
                   "double", "split", "overnight", "graveyard", "lunch", "breakfast", "dinner", "cleaning", "security"}
# the speaker as the one who worked it
WORK_RE = re.compile(r"\bI (?:worked|covered|did|took|was on|was working|was the one working)\b"
                     r"|\bwas mine\b|\bone I worked\b|\bby me\b|\bme working\b")
NOT_WORK_RE = re.compile(r"\b(?:off|swap|swapped|gave|handed|missed|skipped|leave|sick|cancelled|instead)\b", re.I)
DUTY_WORD_RE = re.compile(r"\b(?:shift|duty)\b", re.I)


def split_parts(s):
    return [p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", s.strip().rstrip(".")) if p.strip()]


def words(s):
    return re.findall(r"[a-z][a-z'-]*", s.lower())


def date_in(day, month_name, nights):
    """Day and month of a rota entry, in whichever year of the rota's range holds it."""
    for y in sorted({nights[0].year, nights[-1].year}):
        try:
            d = dt.date(y, T.MONTHS.index(month_name) + 1, int(day))
        except ValueError:
            return None
        if nights[0] <= d <= nights[-1]:
            return d
    return None


@register
class NightRota(Family):
    """'From 3 March 2026 to 8 March 2026, exactly one person worked the late shift each night, following this rota
    exactly: Usha took the first two, ...' (or 'Usha on 3-4 March, ...'). A first-person claim of working that duty
    on a night in the range that the rota gives to someone else is a lie. Other duties and other dates are fine."""

    name = "cal_rota"

    def parse(self, ctx):
        hits = matches(ctx, ROTA_RES)
        if len(hits) != 1:
            return None
        i, m = hits[0]
        dates = T.find_dates(ctx.sentences[i][:m.start("list")])
        if len(dates) != 2:
            return None
        start, end = dates[0][0], dates[1][0]
        if end < start or (end - start).days > 62:
            return None
        nights = [start + dt.timedelta(days=k) for k in range((end - start).days + 1)]
        who = self.assign(m.group("list"), nights)
        duty = [w for w in words(m.group("duty")) if w != "the"]
        if who is None or mentioned_elsewhere(ctx, [i], [" ".join(duty)] + sorted(set(who.values()))):
            return None
        ctx.consume(i)
        phrase = r"\b" + r" ".join(map(re.escape, duty)) + r"(?=\s*(?:[.,;:!]|$)|\s+(?:on|was|that|fell|is|I|at)\b)"
        return {"duty": duty, "phrase": re.compile(phrase, re.I), "who": who, "start": start, "end": end}

    @staticmethod
    def assign(rota, nights):
        """Rota text -> {date: name}, or None unless it covers every night exactly once."""
        who, pos = {}, 0
        for part in split_parts(rota):
            m = PART_DATES_RE.match(part)
            if m:
                name, d1, mon1, d2, mon2 = m.groups()
                a, b = date_in(d1, mon1 or mon2, nights), date_in(d2 or d1, mon2, nights)
                if a is None or b is None or b < a:
                    return None
                for k in range((b - a).days + 1):
                    d = a + dt.timedelta(days=k)
                    if d in who:
                        return None
                    who[d] = name
                continue
            m = PART_COUNT_RE.match(part)
            if not m:
                return None
            name, which, num = (m.group(1), m.group(2), m.group(3)) if m.group(1) else (m.group(6), m.group(4),
                                                                                         m.group(5))
            n = T.word_num(num)
            if not n or (which == "first" and pos != 0) or (which == "last" and pos + n != len(nights)):
                return None
            for d in nights[pos:pos + n]:
                if d in who:
                    return None
                who[d] = name
            pos += n
        if set(who) != set(nights) or (pos and pos != len(nights)):
            return None
        return who

    def judge(self, ctx, f, msg):
        t = plain_claim(msg.text)
        mine = t and (WORK_RE.search(t) or t.startswith("My " + " ".join(f["duty"])))  # 'My late shift was on ...'
        if not mine or NOT_WORK_RE.search(t):
            return None
        d = the_date(t)
        if d is None or T.find_times(t):
            return None
        said = set(words(t))
        extra = (said & DUTY_QUALIFIERS) - set(f["duty"])
        if not all(w in said for w in f["duty"]):
            return Verdict(OK, self.name, "a different duty") if extra and DUTY_WORD_RE.search(t) else None
        if extra or not f["phrase"].search(t):
            return None  # extra qualifiers, or the words not as the duty's name ('the late shift paperwork')
        if not f["start"] <= d <= f["end"]:
            return Verdict(OK, self.name, "outside the rota's dates")
        me = ctx.speakers.get(msg.label, "")
        if f["who"][d] in (me, me.split(" ")[0]):
            return Verdict(OK, self.name, "the rota gives that night to the speaker")
        return Verdict(LIE, self.name, f"the rota gives {d} to {f['who'][d]}")


# ----------------------------------------------------------------------------------------------- one session

SESSION_TIMING = [rf"(?:began|started) at (?P<t>{TIME_PAT}) and (?:lasted|ran for) exactly (?P<dur>{DUR_PAT})",
                  rf"(?:lasted|ran for) exactly (?P<dur>{DUR_PAT}),? and (?:it )?(?:began|started) at (?P<t>{TIME_PAT})"]
SESSION_SUBJECT = [rf"(?i:the) (?:one and )?only (?P<noun>{NP}) that day", rf"(?i:that) day's only (?P<noun>{NP})",
                   rf"(?i:the) (?P<noun>{NP}) that day was the only one, and it"]
# an add-on clause that rules nothing out: 'and it was open to everyone', 'and the field was cleaned before ...'
SESSION_EXTRA = rf"(?:, and (?:it|the (?P<open>{NP})) was open to everyone|, and the {NP} was cleaned before the (?P<cleaned>{NP}))?"
SESSION_RES = [re.compile(rf"^{s} {t}{SESSION_EXTRA}\.$") for s in SESSION_SUBJECT for t in SESSION_TIMING]
# 'The field was cleaned before the training session, which was the only one that day.' 'It began at ...'
SESSION_FIRST_RE = re.compile(rf"^(?i:the) {NP} was cleaned before the (?P<noun>{NP}), which was the only one that day\.$")
SESSION_IT_RES = [re.compile(rf"^It {t}\.$") for t in SESSION_TIMING]
# booking, asking or talking about a session can happen at any time ('I signed up for a place on the tour at 11 AM')
SESSION_NOT_RE = re.compile(r"\b(?:sign|signed|signing|registered|booked|book|bought|paid|reserved|ticket|tickets|"
                            r"place|spot|seat|queued|waited|waiting|asked|read|talked|emailed|phoned|called|posted|"
                            r"photo|photos|video|plan|planned)\b", re.I)


def session_support(noun):
    """Sentences that add nothing to check: 'The tour was open to everyone.', 'The stage was cleaned before the
    rehearsal.', 'Anyone could join the tour, as it was open to everyone.'"""
    n = re.escape(noun)
    return [re.compile(rf"^(?i:the) (?:{n}|class|session) was open to everyone\.$"),
            re.compile(r"^It was open to everyone\.$"),
            re.compile(rf"^(?i:anyone|everyone) could join the {n}(?:, as it was open to everyone)?\.$"),
            re.compile(rf"^(?i:anyone|everyone) was welcome (?:at|to join) the {n}\.$"),
            re.compile(rf"^(?i:the) {NP} (?:was|had been) cleaned before the {n}\.$"),
            re.compile(rf"^Before the {n}, the {NP} (?:was|had been) cleaned\.$")]


@register
class OnlySession(Family):
    """'The only tour that day began at 12:45 PM and lasted exactly 1 hour.' A claim of doing something during the
    tour (or taking part in it) at a time outside its start and end (endpoints included) is a lie."""

    name = "cal_session"

    def parse(self, ctx):
        hits = matches(ctx, SESSION_RES)
        firsts = matches(ctx, [SESSION_FIRST_RE])
        if len(hits) + len(firsts) != 1:
            return None
        if hits:
            (i, m), used = hits[0], [hits[0][0]]
            noun = m.group("noun")
            if m.group("open") not in (None, noun, "class", "session") or m.group("cleaned") not in (None, noun):
                return None
        else:
            i, noun = firsts[0][0], firsts[0][1].group("noun")
            m = first_match(SESSION_IT_RES, ctx.sentences[i + 1]) if i + 1 < len(ctx.sentences) else None
            if not m:
                return None
            used = [i, i + 1]
        durs = T.find_durations(m.group("dur"))
        if len(durs) != 1:
            return None
        start = T.parse_time(m.group("t"))
        end = start + durs[0][0]
        if end >= 24 * 60:
            return None
        support = session_support(noun)
        for k, s in enumerate(ctx.sentences):
            if k not in used and first_match(support, s) and (not s.startswith("It ") or k - 1 in used):
                used.append(k)
        if mentioned_elsewhere(ctx, used, [noun]):
            return None
        for k in used:
            ctx.consume(k)
        rel = re.compile(rf"\b(?:during|in|on) {the_noun(noun)[2:]}|\bwhile the {re.escape(noun)} was on\b", re.I)
        return {"noun": noun, "start": start, "end": end, "rel": rel}

    def judge(self, ctx, f, msg):
        t = plain_claim(msg.text, names_ok=True)  # nobody, not just the speaker, could take part at another time
        if not t or not f["rel"].search(t) or SESSION_NOT_RE.search(t):
            return None
        times = T.find_times(t)
        if len(times) != 1 or T.find_dates(t):
            return None
        tm = times[0][0]
        if f["start"] <= tm <= f["end"]:
            return Verdict(OK, self.name, "within the session")
        return Verdict(LIE, self.name, f"the {f['noun']} ran {T.fmt_time(f['start'])}-{T.fmt_time(f['end'])}")


# ----------------------------------------------------------------------------------------------- ages

BORN_RE = re.compile(rf"^({NAME}) was born on ({DATE_PAT})(?:,? and ({NAME}) was born on ({DATE_PAT}))?\.$")
AGE_WORDS = ["born", "birth", "birthday", "birthdays", "age", "aged", "ages", "years old", "older", "younger"]
# the number must be the age itself, not 'I was 2 hours late' or 'I turned 3 pages'
AGE_END = r"(?: years old)?(?= (?:in|on|during|was|is)\b|[.,!]|$)"
TURNED_RE = re.compile(rf"\b(?:I turned|turning) (\d{{1,3}}){AGE_END}")
WAS_AGE_RE = re.compile(rf"\bI was (\d{{1,3}}){AGE_END}")
SELF_BORN_RE = re.compile(r"\bI was born (?:on|in)\b")


def age_on(born, d):
    return d.year - born.year - ((d.month, d.day) < (born.month, born.day))


@register
class BirthAge(Family):
    """'Sana was born on 14 July 1993.' A speaker's own claims 'I turned N in MONTH YEAR', 'I turned N on DATE',
    'On DATE I was N' and 'I was born on/in ...' are checked against their birth date; a speaker whose birth date
    the evidence does not give is unconstrained."""

    name = "cal_age"

    def parse(self, ctx):
        born, used = {}, []
        for i, m in matches(ctx, [BORN_RE]):
            for name, d in ((m.group(1), m.group(2)), (m.group(3), m.group(4))):
                if name is None:
                    continue
                if name in born:
                    return None
                born[name] = T.parse_date(d)
            used.append(i)
        if not born or mentioned_elsewhere(ctx, used, AGE_WORDS + sorted(born)):
            return None
        for i in used:
            ctx.consume(i)
        return {"born": born}

    def judge(self, ctx, f, msg):
        t = plain_claim(msg.text)
        if not t or T.find_times(t) or re.search(r"\b(?:will|turn|turns|am|I'm|still|nearly|about|over|under)\b", t):
            return None
        kinds = [k for k in (TURNED_RE.search(t), WAS_AGE_RE.search(t), SELF_BORN_RE.search(t)) if k]
        if len(kinds) != 1:
            return None
        born = f["born"].get(ctx.speakers.get(msg.label))
        dates, months = T.find_dates(t), month_years(t)
        years = re.findall(r"\b(?:19|20)\d\d\b", t)
        if born is None:
            return Verdict(OK, self.name, "no known birth date")
        if born.month == 2 and born.day == 29:
            return None  # leap-day birthdays: the birthday in other years is a matter of convention
        k = kinds[0]
        if k.re is SELF_BORN_RE:
            if len(dates) == 1 and not months[1:]:
                ok = dates[0][0] == born
            elif not dates and len(months) == 1:
                ok = months[0] == (born.year, born.month)
            elif not dates and not months and len(years) == 1:
                ok = int(years[0]) == born.year
            else:
                return None
            return Verdict(OK if ok else LIE, self.name, f"born on {born}")
        n = int(k.group(1))
        if k.re is TURNED_RE:
            if len(dates) == 1 and not months[1:]:
                d = dates[0][0]
                ok = (d.month, d.day) == (born.month, born.day) and d.year - born.year == n
                return Verdict(OK if ok else LIE, self.name, f"born on {born}")
            if dates or len(months) != 1:
                return None
            y, mo = months[0]
            if born.month != mo:
                return Verdict(LIE, self.name, f"born in {T.MONTHS[born.month - 1]}, so no birthday in "
                                               f"{T.MONTHS[mo - 1]}")
            if y - born.year != n:
                return Verdict(LIE, self.name, f"turned {y - born.year} in {T.MONTHS[mo - 1]} {y}")
            return Verdict(OK, self.name, "matches the birth date")
        if len(dates) != 1 or months[1:]:
            return None
        d = dates[0][0]
        if age_on(born, d) != n:
            return Verdict(LIE, self.name, f"was {age_on(born, d)} on {d}")
        return Verdict(OK, self.name, "matches the birth date")


# ----------------------------------------------------------------------------------------------- owners over a month

OWNER_HEAD_RES = [
    re.compile(rf"^(?i:{NOBODY}) else owned the (?P<item>{NP}) in (?P<mon>{MONTH_ALT}) (?P<yr>\d{{4}})\.$"),
    re.compile(rf"^(?i:in) (?P<mon>{MONTH_ALT}) (?P<yr>\d{{4}}),? {NOBODY} else owned the (?P<item>{NP})\.$"),
]
PRONOUN = r"(?:he|she|they)"
SOLD_TO = rf"sold it (?:on )?to (?P<b>{NAME})"
WHEN = rf"(?:on (?P<d>{DATE_PAT})|exactly (?P<n>\w+) days? after buying it)"
START = rf"at the start of {MY_PAT}"
# one sale per clause; a sentence is split into clauses at ', and <Name>'. 'it' is the item of the head sentence;
# where a clause names the item, (?P<item>) must be that item, and a second seller name (?P<s2>) must repeat (?P<s>)
SALE_RES = [
    re.compile(rf"^(?P<s>{NAME}) owned the (?P<item>{NP}) until (?:selling|(?:{PRONOUN}|(?P<s2>{NAME})) sold) it to "
               rf"(?P<b>{NAME}) on (?P<d>{DATE_PAT})$"),
    re.compile(rf"^(?P<s>{NAME}) owned the (?P<item>{NP}) until (?P<d>{DATE_PAT}), when {PRONOUN} sold it to "
               rf"(?P<b>{NAME})$"),
    re.compile(rf"^(?P<s>{NAME}) was the (?P<item>{NP})'s owner until selling it to (?P<b>{NAME}) on (?P<d>{DATE_PAT})$"),
    re.compile(rf"^(?i:until) (?:selling|{PRONOUN} sold) it to (?P<b>{NAME}) on (?P<d>{DATE_PAT}), (?P<s>{NAME}) "
               rf"owned the (?P<item>{NP})$"),
    re.compile(rf"^(?P<s>{NAME}) owned the (?P<item>{NP}) {START} and {SOLD_TO} on (?P<d>{DATE_PAT})$"),
    re.compile(rf"^(?i:at) the start of {MY_PAT},? (?P<s>{NAME}) owned the (?P<item>{NP}), and "
               rf"(?:{PRONOUN}|(?P<s2>{NAME})) {SOLD_TO} on (?P<d>{DATE_PAT})$"),
    re.compile(rf"^(?i:at) the start of {MY_PAT},? the (?P<item>{NP}) belonged to (?P<s>{NAME}), who {SOLD_TO} on "
               rf"(?P<d>{DATE_PAT})$"),
    re.compile(rf"^(?P<s>{NAME}) (?:then )?{SOLD_TO} {WHEN}$"),
    re.compile(rf"^(?i:on) (?P<d>{DATE_PAT}),? (?P<s>{NAME}) (?:then )?{SOLD_TO}$"),
    re.compile(rf"^(?i:exactly) (?P<n>\w+) days? after buying it, (?P<s>{NAME}) (?:then )?{SOLD_TO}$"),
]
START_OWNER_RE = re.compile(rf"^(?:(?i:at) the start of {MY_PAT},? )?{NAME} owned the (?P<item>{NP})(?: {START})?$")


def sale(clause, item):
    """'Camila owned the kayak until selling it to Tomas on 8 September 2026' -> (seller, buyer, date), where the
    date is ('after', n) for '... exactly 14 days after buying it'; None when no sale of this item matches."""
    m = first_match(SALE_RES, clause)
    if not m:
        return None
    g = m.groupdict()
    if g.get("item") not in (None, item) or g.get("s2") not in (None, g["s"]):
        return None
    if g.get("d"):
        return g["s"], g["b"], T.parse_date(g["d"])
    n = T.word_num(g["n"])
    return (g["s"], g["b"], ("after", n)) if n else None


@register
class MonthOwners(Family):
    """'Camila owned the kayak until selling it to Tomas on 8 September 2026. Tomas sold it to Isla on 20 September
    2026. Nobody else owned the kayak in September 2026.' Within that month the owners follow one another: the first
    owner up to the first sale, each buyer from their purchase to their own sale (sale days count for both). A
    speaker's claim to own it, or to have sold or bought it, on a day of that month that does not fit is a lie."""

    name = "cal_owner"

    def parse(self, ctx):
        heads = matches(ctx, OWNER_HEAD_RES)
        if len(heads) != 1:
            return None
        hi, hm = heads[0]
        item, sales, used = hm.group("item"), [], [hi]
        for i, s in enumerate(ctx.sentences):
            if i == hi or not s.endswith("."):
                continue
            clauses = re.split(r",? and (?=[A-Z])", s[:-1])  # '..., and Ivan sold it to Jonas on 15 May 2026.'
            found = [sale(c, item) for c in clauses]
            starts = [START_OWNER_RE.match(c) for c in clauses]
            if all(x or (st and st.group("item") == item) for x, st in zip(found, starts)):
                sales += [x for x in found if x]
                used.append(i)
            # any other sentence naming the item or an owner makes the family abstain (checked below)
        bought = {b: d for _, b, d in sales if not isinstance(d, tuple)}
        resolved = []
        for s_, b, d in sales:
            if isinstance(d, tuple):
                if s_ not in bought:
                    return None
                d = bought[s_] + dt.timedelta(days=d[1])
                bought[b] = d
            resolved.append((s_, b, d))
        resolved.sort(key=lambda x: x[2])
        if not resolved or any(resolved[k][0] != resolved[k - 1][1] or resolved[k][2] <= resolved[k - 1][2]
                               for k in range(1, len(resolved))):
            return None
        names = [resolved[0][0]] + [b for _, b, _ in resolved]
        if len(set(names)) != len(names):
            return None  # someone owned it twice: keep out
        month = (int(hm.group("yr")), T.MONTHS.index(hm.group("mon")) + 1)
        if any((d.year, d.month) != month for _, _, d in resolved):
            return None
        if mentioned_elsewhere(ctx, used, [item, item.split()[-1]] + names):
            return None
        for i in used:
            ctx.consume(i)
        cuts = [None] + [d for _, _, d in resolved] + [None]
        spans = [(n, cuts[k], cuts[k + 1]) for k, n in enumerate(names)]
        return {"item": item, "month": month, "sales": set(resolved), "spans": spans}

    @staticmethod
    def owners_on(f, d):
        return {n for n, a, b in f["spans"] if (a is None or a <= d) and (b is None or d <= b)}

    def judge(self, ctx, f, msg):
        t = plain_claim(msg.text, names_ok=True)
        it = re.escape(f["item"])
        if not t or not re.search(rf"\b{it}\b", t):
            return None
        d = the_date(t)
        if d is None or (d.year, d.month) != f["month"]:
            return None
        me = ctx.speakers.get(msg.label)
        names = [w for w in CAPITAL_RE.findall(t)
                 if w not in NOT_NAMES and w not in ("On", "As", "For", "The", "At", "In", "During", "By")]
        if re.search(rf"\bI owned the {it}\b|\bI was the owner of the {it}\b|\bI was the {it}'s owner\b"
                     rf"|\b[Tt]he {it} (?:was mine|belonged to me)\b", t):
            if names:
                return None
            if me in self.owners_on(f, d):
                return Verdict(OK, self.name, "the speaker owned it that day")
            return Verdict(LIE, self.name, f"{'/'.join(sorted(self.owners_on(f, d)))} owned it on {d}")
        sold = re.search(rf"\bI sold the {it} (?:on )?to ({NAME})\b|\bI sold ({NAME}) the {it}\b"
                         rf"|\b[Tt]he {it} was sold by me to ({NAME})\b|\b({NAME}) bought the {it} (?:from|off) me\b",
                         t)
        bought = re.search(rf"\bI (?:bought|purchased) the {it} (?:from|off) ({NAME})\b|\b({NAME}) sold me the {it}\b", t)
        lent = re.search(rf"\bI borrowed the {it} (?:from|off) ({NAME})\b", t)
        hits = [(k, m) for k, m in (("sold", sold), ("bought", bought), ("lent", lent)) if m]
        if len(hits) != 1:
            return None
        kind, m = hits[0]
        other = next(g for g in m.groups() if g)
        if names != [other] or other == me:
            return None
        if kind == "lent":  # borrowing changes no ownership; only a lender who owned it that day is clearly fine
            return Verdict(OK, self.name, "the lender owned it that day") if other in self.owners_on(f, d) else None
        deal = (me, other, d) if kind == "sold" else (other, me, d)
        if deal in f["sales"]:
            return Verdict(OK, self.name, "matches a recorded sale")
        return Verdict(LIE, self.name, f"no such sale: the {f['item']} changed hands only as recorded that month")


if __name__ == "__main__":  # self-check of the calendar arithmetic and the whole-sentence patterns
    assert weekday_span("Wednesday", "Sunday") == {2, 3, 4, 5, 6} and weekday_span("Friday", "Monday") == {4, 5, 6, 0}
    assert weekday(dt.date(2026, 4, 14), (dt.date(2026, 4, 1), 2)) == 1
    assert age_on(dt.date(1993, 7, 14), dt.date(2026, 7, 13)) == 32
    assert age_on(dt.date(1993, 7, 14), dt.date(2026, 7, 14)) == 33
    nts = [dt.date(2026, 3, 3) + dt.timedelta(days=k) for k in range(6)]
    w = NightRota.assign("Usha took the first two, Jonas took the next two, and Akira took the last two", nts)
    assert w[dt.date(2026, 3, 5)] == "Jonas" and w[dt.date(2026, 3, 8)] == "Akira"
    w = NightRota.assign("Lin on 5–6 March, Fatima on 7–8 March, Farid on 3–4 March", nts)
    assert w[dt.date(2026, 3, 3)] == "Farid"
    nts = [dt.date(2026, 12, 30) + dt.timedelta(days=k) for k in range(4)]
    w = NightRota.assign("Lin on 30–31 December, Farid on 1–2 January", nts)
    assert w[dt.date(2027, 1, 2)] == "Farid"
    assert claim_text("Honestly, I swam at the pool at 1:25 PM, for the record.") == "I swam at the pool at 1:25 PM."
    assert plain_claim("Joel told me he swam at the pool.") is None and plain_claim("In May 2026 I turned 30.")
    assert not re.search(the_noun("tour"), "I was at the tour desk at 1 PM")
    s = ("In April 2026 the pool was open only Wednesday to Sunday, from 9:00 AM to 5:00 PM, and nobody could use it "
         "while it was closed.")
    assert first_match(HOURS_RES, s) and not first_match(HOURS_RES, s.replace(" and nobody", " except on 6 April "
                                                                              "2026, and nobody"))
    assert first_match(CLOSED_RES, "On the third Monday of April 2026, nobody could be inside the community hall.")
    assert not first_match(CLOSED_RES, "Nobody could be inside the library on any Tuesday in April 2026 after noon.")
    print("ok")
