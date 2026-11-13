"""Rule families: calendar, schedules and ages. See lc/rules/core.py for the Family interface.

cal_hours    a place open only on some weekdays and hours of a month ("In April 2026 the pool was open only ...")
cal_closed   a place nobody could be inside on certain weekdays ("... on any Tuesday in April 2026")
cal_rota     exactly one person did a nightly duty over a date range, following a rota
cal_session  the only session of the day (tour, class, rehearsal), with a start time and an exact duration
cal_age      birth dates against claims like "I turned 28 in July 2026" and "On 12 March 2026 I was 27"
"""
import datetime as dt
import re

from .core import LIE, OK, Family, Verdict, register
from . import text as T

MONTH_ALT = "|".join(T.MONTHS)
WD_ALT = "|".join(T.WEEKDAYS)
DASH = "[-–—]"
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "last": -1}
# a month and year that is not the tail of a full date ("in April 2026", not "1 April 2026")
MONTH_YEAR_RE = re.compile(rf"(?<!\d\s)\b({MONTH_ALT})\s+(?:of\s+)?(\d{{4}})\b")
ANCHOR_RE = re.compile(rf"^(\d{{1,2}} (?:{MONTH_ALT}) \d{{4}}) was an? ({WD_ALT})\.?$")

# conversational frames the generator wraps around claims
FRAME_HEAD_RE = re.compile(r"^(?:(?:honestly|for the record|okay so|ok so|so|well|frankly|to be honest|as it happens)"
                           r"\b\s*[,:]?\s*)+", re.I)
FRAME_TAIL_RE = re.compile(r"[,;]?\s*\b(?:honestly|for the record|as it happens|to be honest)\s*(?=[.!]?$)", re.I)
# any of these makes a message something other than a plain claim of fact: leave it alone
DOUBT_RE = re.compile(
    r"n't\b|\b(?:not|never|no|nobody|none|nothing|cannot|maybe|perhaps|probably|possibly|might|(?-i:may)|must|think|"
    r"thought|guess|believe|suppose|reckon|apparently|supposedly|unless|if|whether|almost|nearly|tried|try|trying|"
    r"planned|plan|meant|wanted|hoped|would|should|could|supposed|instead|except|only|until|before|after|between|"
    r"since)\b", re.I)  # 'may' only in lower case: 'May' is a month
FIRST_PERSON_RE = re.compile(r"\b(?:I|[Mm]e|[Mm]y|[Mm]ine)\b")
# claims of being near a place rather than using it; these are always possible
WAIT_RE = re.compile(r"\b(?:outside|waiting|waited|wait|stood|standing|queued|queuing|queueing)\b", re.I)


def claim_text(text):
    """The message without 'Honestly,' / ', for the record' style frames."""
    t = FRAME_HEAD_RE.sub("", text.strip())
    return FRAME_TAIL_RE.sub("", t).strip()


def month_years(s):
    return [(int(m.group(2)), T.MONTHS.index(m.group(1)) + 1) for m in MONTH_YEAR_RE.finditer(s)]


def find_anchor(ctx):
    """'1 April 2026 was a Wednesday.' -> (date, weekday index); consumed. None when absent or ambiguous."""
    found = []
    for i, s in enumerate(ctx.sentences):
        m = ANCHOR_RE.match(s)
        if m:
            found.append((i, T.parse_date(m.group(1)), T.WEEKDAYS.index(m.group(2))))
    if len(found) != 1:
        return None
    i, d, wd = found[0]
    ctx.consume(i)
    return d, wd


def weekday(d, anchor):
    """Weekday index of d (Monday=0), counted from the evidence's anchor when it gives one."""
    if anchor:
        return (anchor[1] + (d - anchor[0]).days) % 7
    return d.weekday()


def weekday_span(a, b):
    """'Wednesday to Sunday' -> {2, 3, 4, 5, 6}; wraps past Sunday."""
    i, j = T.WEEKDAYS.index(a), T.WEEKDAYS.index(b)
    return {(i + k) % 7 for k in range((j - i) % 7 + 1)}


def mentions(text, noun):
    return re.search(r"\b" + re.escape(noun) + r"\b", text, re.I) is not None


def is_question(text):
    return text.rstrip().endswith("?")


# ----------------------------------------------------------------------------------------------- opening hours

HOURS_RE = re.compile(rf"\bopen(?:ed)? only (?:from )?({WD_ALT}) to ({WD_ALT})\b")
HOURS_NOUN_RE = re.compile(r"\b(?:[Tt]he|use the) ([a-z][a-z' -]*?) (?:was open|opened|while it was closed)\b")
CLOSED_USE_RE = re.compile(r"\b(?:nobody|no one) could use (?:it|the ([a-z][a-z' -]*?)) while it was closed\b", re.I)
# not a use of the place: being near it, contacting it, seeing it
NOT_USE_RE = re.compile(
    r"\b(?:near|nearby|past|passed|front|across|opposite|behind|next|beside|phoned|called|rang|emailed|messaged|"
    r"texted|wrote|looked|saw|seen|watched|noticed|heard|website|online|closed|shut|locked|booked|by)\b", re.I)


@register
class OpeningHours(Family):
    """'In April 2026 the pool was open only Wednesday to Sunday, from 9:00 AM to 5:00 PM, and nobody could use it
    while it was closed.' A first-person claim of using the place on a closed weekday, or at a time outside the
    hours (endpoints are open), is a lie. Waiting outside is always possible; other months are not covered."""

    name = "cal_hours"

    def parse(self, ctx):
        hits = [(i, s) for i, s in enumerate(ctx.sentences) if HOURS_RE.search(s)]
        if len(hits) != 1:
            return None
        i, s = hits[0]
        m = HOURS_RE.search(s)
        nouns = {n for n in HOURS_NOUN_RE.findall(s)}
        closed = [(j, CLOSED_USE_RE.search(x)) for j, x in enumerate(ctx.sentences) if CLOSED_USE_RE.search(x)]
        if len(nouns) != 1 or len(closed) != 1:
            return None
        noun = nouns.pop()
        j, cm = closed[0]
        if cm.group(1) and cm.group(1) != noun or (j != i and cm.group(1) is None and j != i + 1):
            return None  # the closure rule must be about the same place
        times = [t for t, _ in T.find_times(s)]
        if len(times) not in (0, 2) or (times and times[0] >= times[1]):
            return None
        months = month_years(s)
        if len(months) > 1:
            return None
        ctx.consume(i)
        ctx.consume(j)
        return {"noun": noun, "days": weekday_span(m.group(1), m.group(2)), "hours": times or None,
                "month": months[0] if months else None, "anchor": find_anchor(ctx)}

    def judge(self, ctx, f, msg):
        t = claim_text(msg.text)
        if is_question(t) or not mentions(t, f["noun"]):
            return None
        dates = T.find_dates(t)
        if len(dates) != 1 or DOUBT_RE.search(t) or not FIRST_PERSON_RE.search(t):
            return None
        d = dates[0][0]
        if WAIT_RE.search(t):
            return Verdict(OK, self.name, "waiting outside is always possible")
        if NOT_USE_RE.search(t):
            return None
        times = T.find_times(t)
        if len(times) > 1:
            return None
        if f["month"] and (d.year, d.month) != f["month"]:
            return Verdict(OK, self.name, "the opening days cover another month")
        wd = weekday(d, f["anchor"])
        if wd not in f["days"]:
            return Verdict(LIE, self.name, f"{d} was a {T.WEEKDAYS[wd]}, a closed day")
        if times and f["hours"] and not f["hours"][0] <= times[0][0] <= f["hours"][1]:
            return Verdict(LIE, self.name, f"{T.fmt_time(times[0][0])} is outside the opening hours")
        if f["hours"] and not times:
            return None  # an open day, but the claim gives no time
        return Verdict(OK, self.name, "open at that day and time")


# ----------------------------------------------------------------------------------------------- closed weekdays

CLOSED_DAY_RE = re.compile(r"\b(?:nobody|no one) could be (?:inside|in) the ([a-z][a-z' -]*?)(?= on\b| during\b| in\b|[.,])",
                           re.I)
ANY_WD_RE = re.compile(rf"\b(?:any|every|each) ({WD_ALT})\b")
NTH_WD_RE = re.compile(rf"\bthe (first|second|third|fourth|fifth|last) ({WD_ALT}) (?:of|in) ({MONTH_ALT}) (\d{{4}})\b")
# claims of being inside a place
INSIDE_RE = re.compile(r"\b(?:inside|in|into|within)\b|\bwhere I was\b|\bfound me\b|\bvisited\b|\bentered\b", re.I)


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
        hits = [(i, s, CLOSED_DAY_RE.search(s)) for i, s in enumerate(ctx.sentences) if CLOSED_DAY_RE.search(s)]
        if len(hits) != 1:
            return None
        i, s, cm = hits[0]
        anchor = None
        nth, anyd = NTH_WD_RE.search(s), ANY_WD_RE.search(s)
        if bool(nth) == bool(anyd):
            return None
        if nth:
            y, mo, wd_name = int(nth.group(4)), T.MONTHS.index(nth.group(3)) + 1, nth.group(2)
        else:
            months = month_years(s)
            if len(months) != 1 or len(T.find_weekdays(s)) != 1:
                return None
            (y, mo), wd_name = months[0], anyd.group(1)
        ctx.consume(i)
        anchor = find_anchor(ctx)
        wd = T.WEEKDAYS.index(wd_name)
        days = [d for d in month_days(y, mo) if weekday(d, anchor) == wd]
        if nth:
            k = ORDINALS[nth.group(1)]
            if k > len(days):
                return None
            days = [days[k - 1] if k > 0 else days[-1]]
        return {"noun": cm.group(1), "closed": set(days)}

    def judge(self, ctx, f, msg):
        t = claim_text(msg.text)
        if is_question(t) or not mentions(t, f["noun"]):
            return None
        dates = T.find_dates(t)
        if len(dates) != 1 or DOUBT_RE.search(t) or not FIRST_PERSON_RE.search(t):
            return None
        d = dates[0][0]
        if WAIT_RE.search(t):
            return Verdict(OK, self.name, "waiting outside is always possible")
        if NOT_USE_RE.search(t) or not INSIDE_RE.search(t):
            return None
        if d in f["closed"]:
            return Verdict(LIE, self.name, f"nobody could be inside on {d}")
        return Verdict(OK, self.name, "not a closed day")


# ----------------------------------------------------------------------------------------------- nightly rota

ROTA_RE = re.compile(r"\bexactly one person worked (the [a-z][a-z' -]*?) (?:each|every) night\b"
                     r"|\bexactly one person worked (the [a-z][a-z' -]*?),"
                     r"|\b(the [a-z][a-z' -]*?) was covered (?:each|every) night by exactly one person\b", re.I)
ROTA_SPAN_RE = re.compile(rf"\b(?:[Ff]rom|[Bb]etween) (\d{{1,2}} (?:{MONTH_ALT}) \d{{4}}) (?:to|and) "
                          rf"(\d{{1,2}} (?:{MONTH_ALT}) \d{{4}})\b")
NAME = r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?"
PART_DATES_RE = re.compile(rf"^({NAME}) (?:on|took) (\d{{1,2}})(?: ({MONTH_ALT}))?(?:\s*{DASH}\s*(\d{{1,2}}))? "
                           rf"({MONTH_ALT})$")
PART_COUNT_RE = re.compile(r"^(?:(" + NAME + r") took the (first|next|last) (\w+)(?: nights?)?"
                           r"|the (first|next|last) (\w+)(?: nights?)? (?:went )?to (" + NAME + r"))$")
DUTY_STOP = {"the", "a", "an"}
# words that name a different duty or shift when the evidence's duty lacks them
DUTY_QUALIFIERS = {"day", "night", "late", "early", "morning", "afternoon", "evening", "weekend", "weekday",
                   "call-out", "front", "desk", "back", "office", "kitchen", "door", "gate", "reception", "bar",
                   "double", "split", "overnight", "graveyard", "lunch", "breakfast", "dinner", "cleaning", "security"}
WORK_RE = re.compile(r"\b(?:worked|work|working|covered|did|took|was on|mine|fell on|worked by me|me working)\b",
                     re.I)
DUTY_WORD_RE = re.compile(r"\b(?:shift|duty|rota)\b", re.I)


def split_parts(s):
    return [p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", s.strip().rstrip(".")) if p.strip()]


def words(s):
    return re.findall(r"[a-z][a-z'-]*", s.lower())


@register
class NightRota(Family):
    """'From 3 March 2026 to 8 March 2026, exactly one person worked the late shift each night, following this rota
    exactly: Usha took the first two, ...' (or 'Usha on 3-4 March, ...'). A first-person claim of working that duty
    on a night in the range that the rota gives to someone else is a lie. Other duties and other dates are fine."""

    name = "cal_rota"

    def parse(self, ctx):
        hits = [(i, s) for i, s in enumerate(ctx.sentences) if ROTA_RE.search(s) and "rota" in s and ":" in s]
        if len(hits) != 1:
            return None
        i, s = hits[0]
        rm, span = ROTA_RE.search(s), ROTA_SPAN_RE.search(s)
        if not span:
            return None
        duty = next(g for g in rm.groups() if g)
        start, end = T.parse_date(span.group(1)), T.parse_date(span.group(2))
        if end < start or (end - start).days > 62:
            return None
        nights = [start + dt.timedelta(days=k) for k in range((end - start).days + 1)]
        who = self.assign(s.split(":", 1)[1], nights, start.year)
        if who is None:
            return None
        ctx.consume(i)
        return {"duty": duty, "who": who, "start": start, "end": end}

    @staticmethod
    def assign(rota, nights, year):
        """Rota text -> {date: name}, or None unless it covers every night exactly once."""
        who, pos = {}, 0
        for part in split_parts(rota):
            m = PART_DATES_RE.match(part)
            if m:
                name, d1, mon1, d2, mon2 = m.groups()
                a = dt.date(year, T.MONTHS.index(mon1 or mon2) + 1, int(d1))
                b = dt.date(year, T.MONTHS.index(mon2) + 1, int(d2 or d1))
                if b < a:
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
            if not n:
                return None
            if which == "first" and pos != 0:
                return None
            if which == "last" and pos + n != len(nights):
                return None
            for d in nights[pos:pos + n]:
                who[d] = name
            pos += n
        if set(who) != set(nights) or (pos and pos != len(nights)):
            return None
        return who

    def judge(self, ctx, f, msg):
        t = claim_text(msg.text)
        if is_question(t) or not DUTY_WORD_RE.search(t):
            return None
        dates = T.find_dates(t)
        if len(dates) != 1 or DOUBT_RE.search(t) or not FIRST_PERSON_RE.search(t) or not WORK_RE.search(t):
            return None
        if T.find_times(t):
            return None
        d = dates[0][0]
        duty = [w for w in words(f["duty"]) if w not in DUTY_STOP]
        said = set(words(t))
        if not all(w in said for w in duty):
            extra = (said & DUTY_QUALIFIERS) - set(duty)
            return Verdict(OK, self.name, "a different duty") if extra else None
        if (said & DUTY_QUALIFIERS) - set(duty):
            return None  # names the duty with extra qualifiers: not sure it is the same one
        if not f["start"] <= d <= f["end"]:
            return Verdict(OK, self.name, "outside the rota's dates")
        me = ctx.speakers.get(msg.label)
        if f["who"][d] == me:
            return Verdict(OK, self.name, "the rota gives that night to the speaker")
        return Verdict(LIE, self.name, f"the rota gives {d} to {f['who'][d]}")


# ----------------------------------------------------------------------------------------------- one session

SESSION_RES = [
    re.compile(r"\b(?:[Tt]he|[Tt]hat day's) (?:one and )?only ([a-z][a-z' -]*?)(?: that day)?(?:,)? "
               r"(?:began|started|lasted|ran)\b"),
    re.compile(r"\b[Tt]he ([a-z][a-z' -]*?) that day was the only one\b"),
    re.compile(r"\bthe ([a-z][a-z' -]*?), which was the only one that day\b"),
]
START_RE = re.compile(r"\b(?:began|started|begun|starting) at\b", re.I)
LENGTH_RE = re.compile(r"\b(?:lasted|ran for|ran) exactly\b", re.I)
OPEN_TO_ALL_RE = re.compile(r"\bopen to everyone\b|\b(?:anyone|everyone) could join\b", re.I)


@register
class OnlySession(Family):
    """'The only tour that day began at 12:45 PM and lasted exactly 1 hour.' A claim of doing something during the
    tour (or taking part in it) at a time outside its start and end (endpoints included) is a lie."""

    name = "cal_session"

    def parse(self, ctx):
        found = []
        for i, s in enumerate(ctx.sentences):
            for rx in SESSION_RES:
                m = rx.search(s)
                if m:
                    found.append((i, m.group(1)))
                    break
        if len(found) != 1:
            return None
        i, noun = found[0]
        span = [i] if START_RE.search(ctx.sentences[i]) else [i, i + 1]
        if span[-1] >= len(ctx.sentences):
            return None
        text = " ".join(ctx.sentences[k] for k in span)
        if len(span) == 2 and not ctx.sentences[i + 1].startswith("It "):
            return None
        times, durs = T.find_times(text), T.find_durations(text)
        if len(times) != 1 or len(durs) != 1 or not START_RE.search(text) or not LENGTH_RE.search(text):
            return None
        start = times[0][0]
        end = start + durs[0][0]
        if end >= 24 * 60:
            return None
        for k in span:
            ctx.consume(k)
        for k, s in enumerate(ctx.sentences):  # 'The tour was open to everyone.' adds nothing to check
            if k not in span and OPEN_TO_ALL_RE.search(s) and (mentions(s, noun) or s.startswith("It ")):
                ctx.consume(k)
        return {"noun": noun, "start": start, "end": end}

    def judge(self, ctx, f, msg):
        t = claim_text(msg.text)
        if is_question(t):
            return None
        noun = re.escape(f["noun"])
        rel = re.search(rf"\b(?:during|in|on|at|joined|attended|while) the {noun}\b|\bthe {noun} was on\b", t, re.I)
        if not rel:
            return None
        times = T.find_times(t)
        if len(times) != 1 or DOUBT_RE.search(t) or T.find_dates(t):
            return None
        tm = times[0][0]
        if f["start"] <= tm <= f["end"]:
            return Verdict(OK, self.name, "within the session")
        return Verdict(LIE, self.name, f"the {f['noun']} ran {T.fmt_time(f['start'])}-{T.fmt_time(f['end'])}")


# ----------------------------------------------------------------------------------------------- ages

BORN_RE = re.compile(rf"\b({NAME}) was born on (\d{{1,2}} (?:{MONTH_ALT}) \d{{4}})")
TURNED_RE = re.compile(r"\b(?:I turned|turning) (\d{1,3})\b")
WAS_AGE_RE = re.compile(r"\bI was (\d{1,3})(?: years old)?\b")


def age_on(born, d):
    return d.year - born.year - ((d.month, d.day) < (born.month, born.day))


@register
class BirthAge(Family):
    """'Sana was born on 14 July 1993.' Claims 'I turned N in MONTH YEAR' and 'On DATE I was N' by a speaker whose
    birth date is known are checked by calendar arithmetic; speakers with no known birth date are unconstrained."""

    name = "cal_age"

    def parse(self, ctx):
        born = {}
        for i, s in enumerate(ctx.sentences):
            ms = BORN_RE.findall(s)
            if not ms:
                continue
            if len(ms) != s.count("born"):
                return None
            for name, d in ms:
                if name in born:
                    return None
                born[name] = T.parse_date(d)
            ctx.consume(i)
        return {"born": born} if born else None

    def judge(self, ctx, f, msg):
        t = claim_text(msg.text)
        if is_question(t) or re.search(r"\b(?:will|turn|turns|am|I'm)\b", t):
            return None
        tm, wm = TURNED_RE.search(t), WAS_AGE_RE.search(t)
        if bool(tm) == bool(wm) or DOUBT_RE.search(t) or T.find_times(t):
            return None
        me = ctx.speakers.get(msg.label)
        others = [n for n in f["born"] if n != me and re.search(rf"\b{re.escape(n)}\b", t)]
        if others:
            return None  # a claim that names someone else
        born = f["born"].get(me)
        dates, months = T.find_dates(t), month_years(t)
        if tm:
            if dates or len(months) != 1:
                return None
            n, (y, mo) = int(tm.group(1)), months[0]
            if born is None:
                return Verdict(OK, self.name, "no known birth date")
            if born.month == 2 and born.day == 29:
                return None
            if born.month != mo:
                return Verdict(LIE, self.name, f"born in {T.MONTHS[born.month - 1]}, so no birthday in {T.MONTHS[mo - 1]}")
            if y - born.year != n:
                return Verdict(LIE, self.name, f"turned {y - born.year} in {T.MONTHS[mo - 1]} {y}")
            return Verdict(OK, self.name, "matches the birth date")
        if len(dates) != 1 or months[1:]:
            return None
        n, d = int(wm.group(1)), dates[0][0]
        if born is None:
            return Verdict(OK, self.name, "no known birth date")
        if born.month == 2 and born.day == 29:
            return None
        if age_on(born, d) != n:
            return Verdict(LIE, self.name, f"was {age_on(born, d)} on {d}")
        return Verdict(OK, self.name, "matches the birth date")


if __name__ == "__main__":  # self-check of the calendar arithmetic
    assert weekday_span("Wednesday", "Sunday") == {2, 3, 4, 5, 6} and weekday_span("Friday", "Monday") == {4, 5, 6, 0}
    assert weekday(dt.date(2026, 4, 14), (dt.date(2026, 4, 1), 2)) == 1
    assert age_on(dt.date(1993, 7, 14), dt.date(2026, 7, 13)) == 32 and age_on(dt.date(1993, 7, 14), dt.date(2026, 7, 14)) == 33
    nights = [dt.date(2026, 3, 3) + dt.timedelta(days=k) for k in range(6)]
    w = NightRota.assign("Usha took the first two, Jonas took the next two, and Akira took the last two.", nights, 2026)
    assert w[dt.date(2026, 3, 5)] == "Jonas" and w[dt.date(2026, 3, 8)] == "Akira"
    w = NightRota.assign("Lin on 5–6 March, Fatima on 7–8 March, Farid on 3–4 March", nights, 2026)
    assert w[dt.date(2026, 3, 3)] == "Farid"
    assert claim_text("Honestly, I swam at the pool at 1:25 PM, for the record.") == "I swam at the pool at 1:25 PM."
    print("ok")
