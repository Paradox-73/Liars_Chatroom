"""Rule families: calendar, schedules and ages. See lc/rules/core.py for the Family interface.

cal_hours    a place open only on some weekdays and hours of a month ("In April 2026 the pool was open only ...")
cal_closed   a place nobody could be inside on certain weekdays ("... on any Tuesday in April 2026")
cal_rota     exactly one person did a nightly duty over a date range, following a rota
cal_session  the only session of the day (tour, class, rehearsal), with a start time and an exact duration
cal_age      birth dates against claims like "I turned 28 in July 2026" and "On 12 March 2026 I was 27"

Every family judges only plain first-person claims: questions, hedged or negated sentences, reported speech and
messages that name another person are left alone.
"""
import datetime as dt
import re

from .core import LIE, OK, Family, Verdict, register
from . import text as T

MONTH_ALT = "|".join(T.MONTHS)
WD_ALT = "|".join(T.WEEKDAYS)
DATE_PAT = rf"\d{{1,2}} (?:{MONTH_ALT}) \d{{4}}"
DASH = "[-–—]"
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "last": -1}
NAME = r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?"
# a month and year that is not the tail of a full date ("in April 2026", not "1 April 2026")
MONTH_YEAR_RE = re.compile(rf"(?<!\d\s)\b({MONTH_ALT})\s+(?:of\s+)?(\d{{4}})\b")
ANCHOR_RE = re.compile(rf"^({DATE_PAT}) was an? ({WD_ALT})\.?$")

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


# words that add an exception to a calendar rule, or limit it to part of a day; such rules are left alone
EXCEPTION_RE = re.compile(r"\b(?:except|excepting|also|apart|but|besides|plus|unless|holidays?|as well|alternate|"
                          r"other)\b", re.I)
PART_OF_DAY_RE = re.compile(r"\b(?:before|after|until|till|between|from|morning|afternoon|evening|night|noon|"
                            r"midday|midnight)\b", re.I)


def waiting_only(t):
    """'I waited outside the pool on 14 April 2026': a single clause about waiting, with no clock time."""
    t = re.sub(rf"({DATE_PAT}),", r"\1", t)  # 'On 21 December 2026, I waited ...' is still one clause
    return WAIT_RE.search(t) and not T.find_times(t) and not re.search(r",|\b(?:and|then)\b", t)


# ----------------------------------------------------------------------------------------------- opening hours

HOURS_RE = re.compile(rf"\bopen(?:ed)? only (?:from )?({WD_ALT}) to ({WD_ALT})\b")
HOURS_NOUN_RE = re.compile(r"\b(?:[Tt]he|use the) ([a-z][a-z' -]*?) (?:was open|opened|while it was closed)\b")
CLOSED_USE_RE = re.compile(r"\b(?:nobody|no one) could use (?:it|the ([a-z][a-z' -]*?)) while it was closed\b", re.I)
# not a use of the place: being near it, contacting it, seeing it
NOT_USE_RE = re.compile(
    r"\b(?:near|nearby|past|passed|front|across|opposite|behind|next|beside|phoned|called|rang|emailed|messaged|"
    r"texted|wrote|looked|saw|seen|watched|noticed|website|online|closed|shut|locked|booked|by)\b", re.I)


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
        nouns = set(HOURS_NOUN_RE.findall(s))
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
        if len(T.find_weekdays(s)) != 2 or T.find_dates(s) or EXCEPTION_RE.search(s):
            return None  # extra days, dates or exceptions would change the opening days
        months = month_years(s)
        anchor = find_anchor(ctx)
        if len(months) > 1 or anchor is False:
            return None
        ctx.consume(i)
        ctx.consume(j)
        return {"noun": noun, "days": weekday_span(m.group(1), m.group(2)), "hours": times or None,
                "month": months[0] if months else None, "anchor": anchor,
                # 'I was at the pool' may mean outside it, so it is not a claim of using it
                "at": re.compile(rf"\b(?:was|were) at the {re.escape(noun)}\b", re.I)}

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
        if WAIT_RE.search(t) or NOT_USE_RE.search(t) or len(times) > 1 or f["at"].search(t):
            return None
        if f["month"] and (d.year, d.month) != f["month"]:
            return Verdict(OK, self.name, "the opening days cover another month")
        wd = weekday(d, f["anchor"])
        if wd not in f["days"]:
            return Verdict(LIE, self.name, f"{d} was a {T.WEEKDAYS[wd]}, a closed day")
        if not times:
            return Verdict(OK, self.name, "an open day")
        if not f["hours"]:
            return None  # the hours were not given in the sentence with the days
        if not f["hours"][0] <= times[0][0] <= f["hours"][1]:
            return Verdict(LIE, self.name, f"{T.fmt_time(times[0][0])} is outside the opening hours")
        return Verdict(OK, self.name, "open at that day and time")


# ----------------------------------------------------------------------------------------------- closed weekdays

CLOSED_DAY_RE = re.compile(r"\b(?:nobody|no one) could be (?:inside|in) the ([a-z][a-z' -]*?)(?= on\b| during\b| in\b|[.,])",
                           re.I)
ANY_WD_RE = re.compile(rf"\b(?:any|every|each) ({WD_ALT})\b|\bon ({WD_ALT})s\b")
NTH_WD_RE = re.compile(rf"\bthe (first|second|third|fourth|fifth|last) ({WD_ALT}) (?:of|in) ({MONTH_ALT}) (\d{{4}})\b")


def inside_re(noun):
    """Claims of being inside the place."""
    n = re.escape(noun)
    return re.compile(rf"\b(?:inside|in|into|within|visited|entered) {the_noun(noun)[2:]}|\bthe {n} is where I was\b"
                      rf"|\bfound me (?:inside|in) the {n}\b", re.I)


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
        nth, anyd = NTH_WD_RE.search(s), ANY_WD_RE.search(s)
        if bool(nth) == bool(anyd) or len(T.find_weekdays(s)) != 1:
            return None
        if T.find_times(s) or T.find_dates(s) or EXCEPTION_RE.search(s) or PART_OF_DAY_RE.search(s):
            return None  # 'on any Tuesday after 5:00 PM' or '... except 14 April' close less than whole days
        if nth:
            y, mo, wd_name = int(nth.group(4)), T.MONTHS.index(nth.group(3)) + 1, nth.group(2)
        else:
            months = month_years(s)
            if len(months) != 1:
                return None
            (y, mo), wd_name = months[0], anyd.group(1) or anyd.group(2)
        anchor = find_anchor(ctx)
        if anchor is False:
            return None
        wd = T.WEEKDAYS.index(wd_name)
        days = [d for d in month_days(y, mo) if weekday(d, anchor) == wd]
        if nth:
            k = ORDINALS[nth.group(1)]
            if k > len(days):
                return None
            days = [days[k - 1] if k > 0 else days[-1]]
        ctx.consume(i)
        return {"noun": cm.group(1), "closed": set(days), "inside": inside_re(cm.group(1))}

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

ROTA_RE = re.compile(r"\bexactly one person worked (the [a-z][a-z' -]*?) (?:each|every) night\b"
                     r"|\bexactly one person worked (the [a-z][a-z' -]*?),"
                     r"|\b(the [a-z][a-z' -]*?) was covered (?:each|every) night by exactly one person\b", re.I)
ROTA_SPAN_RE = re.compile(rf"\b(?:[Ff]rom|[Bb]etween) ({DATE_PAT}) (?:to|and) ({DATE_PAT})\b")
PART_DATES_RE = re.compile(rf"^({NAME}) (?:on|took) (\d{{1,2}})(?: ({MONTH_ALT}))?(?:\s*{DASH}\s*(\d{{1,2}}))? "
                           rf"({MONTH_ALT})$")
PART_COUNT_RE = re.compile(rf"^(?:({NAME}) took the (first|next|last) (\w+)(?: nights?)?"
                           rf"|the (first|next|last) (\w+)(?: nights?)? (?:went )?to ({NAME}))$")
# words that name a different duty or shift when the evidence's duty lacks them
DUTY_QUALIFIERS = {"day", "night", "late", "early", "morning", "afternoon", "evening", "weekend", "weekday",
                   "call-out", "front", "desk", "back", "office", "kitchen", "door", "gate", "reception", "bar",
                   "double", "split", "overnight", "graveyard", "lunch", "breakfast", "dinner", "cleaning", "security"}
# the speaker as the one who worked it
WORK_RE = re.compile(r"\bI (?:worked|covered|did|took|was on|was working|was the one working)\b|\bI worked\b"
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
        who = self.assign(s.split(":", 1)[1], nights)
        if who is None:
            return None
        ctx.consume(i)
        return {"duty": [w for w in words(duty) if w != "the"], "who": who, "start": start, "end": end}

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
        if extra:
            return None  # the duty with extra qualifiers: maybe a different one
        if not f["start"] <= d <= f["end"]:
            return Verdict(OK, self.name, "outside the rota's dates")
        me = ctx.speakers.get(msg.label, "")
        if f["who"][d] in (me, me.split(" ")[0]):
            return Verdict(OK, self.name, "the rota gives that night to the speaker")
        return Verdict(LIE, self.name, f"the rota gives {d} to {f['who'][d]}")


# ----------------------------------------------------------------------------------------------- one session

SESSION_RES = [
    re.compile(r"\b(?:[Tt]he|[Tt]hat day's) (?:one and )?only ([a-z][a-z' -]*?)(?: that day)?(?:,)? "
               r"(?:began|started|lasted|ran)\b"),
    re.compile(r"\b[Tt]he ([a-z][a-z' -]*?) that day was the only one\b"),
    re.compile(r"\bthe ([a-z][a-z' -]*?), which was the only one that day\b"),
]
START_RE = re.compile(r"\b(?:began|started|starting) at\b", re.I)
LENGTH_RE = re.compile(r"\b(?:lasted|ran for|ran) exactly\b", re.I)
OPEN_TO_ALL_RE = re.compile(r"\bopen to everyone\b|\b(?:anyone|everyone) could join\b", re.I)
CLEANED_BEFORE_RE = re.compile(r"\bwas cleaned before the ([a-z][a-z' -]*?)[.,]")


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
        if span[-1] >= len(ctx.sentences) or (len(span) == 2 and not ctx.sentences[i + 1].startswith("It ")):
            return None
        text = " ".join(ctx.sentences[k] for k in span)
        times, durs = T.find_times(text), T.find_durations(text)
        if len(times) != 1 or len(durs) != 1 or not START_RE.search(text) or not LENGTH_RE.search(text):
            return None
        if re.search(r"\b(?:half|quarter)\b", text):
            return None  # 'half an hour' would read as an hour
        start = times[0][0]
        end = start + durs[0][0]
        if end >= 24 * 60:
            return None
        for k in span:
            ctx.consume(k)
        for k, s in enumerate(ctx.sentences):
            # 'The tour was open to everyone.' and 'The stage was cleaned before the rehearsal.' rule nothing out
            c = CLEANED_BEFORE_RE.search(s)
            if (OPEN_TO_ALL_RE.search(s) and (re.search(rf"\b{re.escape(noun)}\b", s) or s.startswith("It "))
                    or c and c.group(1) == noun and not T.find_times(s)):
                ctx.consume(k)
        rel = re.compile(rf"\b(?:during|in|on|at|joined|attended) {the_noun(noun)[2:]}|\bwhile the {re.escape(noun)} "
                         rf"was on\b", re.I)
        return {"noun": noun, "start": start, "end": end, "rel": rel}

    def judge(self, ctx, f, msg):
        t = plain_claim(msg.text, names_ok=True)  # nobody, not just the speaker, could take part at another time
        if not t or not f["rel"].search(t):
            return None
        times = T.find_times(t)
        if len(times) != 1 or T.find_dates(t):
            return None
        tm = times[0][0]
        if f["start"] <= tm <= f["end"]:
            return Verdict(OK, self.name, "within the session")
        return Verdict(LIE, self.name, f"the {f['noun']} ran {T.fmt_time(f['start'])}-{T.fmt_time(f['end'])}")


# ----------------------------------------------------------------------------------------------- ages

BORN_RE = re.compile(rf"\b({NAME}) was born on ({DATE_PAT})")
TURNED_RE = re.compile(r"\b(?:I turned|turning) (\d{1,3})\b")
WAS_AGE_RE = re.compile(r"\bI was (\d{1,3})(?: years old)?\b")
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
        born, other_ages = {}, False
        for i, s in enumerate(ctx.sentences):
            ms = BORN_RE.findall(s)
            if not ms:
                other_ages |= bool(re.search(r"\b(?:birthday|birthdays|years old|age|aged|older|younger)\b", s, re.I))
                continue
            if len(ms) != s.count("born"):
                return None
            for name, d in ms:
                if name in born:
                    return None
                born[name] = T.parse_date(d)
            ctx.consume(i)
        # with other age facts around, a speaker missing from the birth dates is not known to be unconstrained
        return {"born": born, "others_free": not other_ages} if born else None

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
            return Verdict(OK, self.name, "no known birth date") if f["others_free"] else None
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


# ----------------------------------------------------------------------------------------------- room bookings

TEAM_RE = re.compile(r"\b[Tt]he ([A-Za-z]+) team\b")
ROOM_RE = re.compile(r"\b(Room \d+|[Tt]he [a-z][a-z ]*?)(?= was booked\b| was reserved\b)"
                     r"|\bbooking for (Room \d+|the [a-z][a-z ]*?) ran\b")
ONLY_TEAM_RE = re.compile(r"\bonly (?:that|the booked) team was in the room\b", re.I)
GAPLESS_RE = re.compile(r"\bwithout gaps\b|\b[Ss]tarting at\b")


def booking_blocks(s):
    """'The studio was booked from 8:20 AM to 11:20 AM: the design team until 9:20 AM, the research team until
    10:35 AM, then the finance team.' (or '... booked without gaps from 8:20 AM: the design team for 1 hour, then
    ...') -> (room, [(start, end, team)]); None when the sentence does not fit either shape exactly."""
    rm = ROOM_RE.search(s)
    teams = list(TEAM_RE.finditer(s))
    if not rm or not teams:
        return None
    room = rm.group(1) or rm.group(2)
    head, segs = s[:teams[0].start()], []
    for k, m in enumerate(teams):
        seg = s[m.end():teams[k + 1].start() if k + 1 < len(teams) else len(s)]
        segs.append((m.group(1).lower(), [t for t, _ in T.find_times(seg)], [d for d, _ in T.find_durations(seg)]))
    times = [t for t, _ in T.find_times(head)]
    if re.search(r"\b(?:half|quarter)\b", s):
        return None
    blocks = []
    if GAPLESS_RE.search(head) and len(times) == 1 and all(len(d) == 1 and not t for _, t, d in segs):
        at = times[0]
        for team, _, (dur,) in segs:
            blocks.append((at, at + dur, team))
            at += dur
    elif len(times) == 2 and all(len(t) == 1 and not d for _, t, d in segs[:-1]) and segs[-1][1:] == ([], []):
        bounds = [times[0]] + [t[0] for _, t, _ in segs[:-1]] + [times[1]]
        blocks = [(bounds[k], bounds[k + 1], team) for k, (team, _, _) in enumerate(segs)]
    else:
        return None
    if any(a >= b for a, b, _ in blocks) or blocks[-1][1] >= 24 * 60 or len({b[2] for b in blocks}) != len(blocks):
        return None
    return room, blocks


@register
class RoomBooking(Family):
    """A room booked back to back by teams, with 'During each booking, only that team was in the room.' A claim that
    a team was in (or had) the room at a time inside another team's booking is a lie; times outside the bookings,
    other places, and a team inside its own booking are fine. Endpoints belong to both neighbouring bookings."""

    name = "cal_booking"

    def parse(self, ctx):
        only = [i for i, s in enumerate(ctx.sentences) if ONLY_TEAM_RE.search(s)]
        found = [(i, booking_blocks(s)) for i, s in enumerate(ctx.sentences)
                 if re.search(r"\b(?:booked|reserved|booking)\b", s) and i not in only]
        if len(only) != 1 or len(found) != 1 or found[0][1] is None:
            return None
        i, (room, blocks) = found[0]
        ctx.consume(i)
        ctx.consume(only[0])
        if room.lower().startswith("the "):  # 'the main hall'
            room_re, head = the_noun(room[4:].lower())[2:], room.split()[-1].lower()
        else:  # 'Room 4'
            room_re, head = re.escape(room) + r"\b(?!\s*\d)", "room"
        return {"room": room, "blocks": blocks, "head": head,
                "in_room": re.compile(rf"\b(?:in|inside|had) {room_re}", re.I),
                "past": re.compile(rf"^(?:At \S+ [AP]M,? )?I (?:walked|went|passed|cycled|ran) (?:past|by) {room_re}",
                                   re.I),
                "elsewhere": re.compile(rf"\b(?:in|inside) the (?!{re.escape(head)}\b)([a-z]+){AFTER_NOUN}", re.I)}

    def judge(self, ctx, f, msg):
        t = plain_claim(msg.text, names_ok=True)  # the subject is a team, not the speaker
        if not t:
            return None
        teams, times = TEAM_RE.findall(t), T.find_times(t)
        if not teams and f["past"].search(t):
            return Verdict(OK, self.name, "walking past the room")
        if len(teams) != 1 or len(times) != 1 or T.find_dates(t):
            return None
        team, tm = teams[0].lower(), times[0][0]
        if not f["in_room"].search(t):
            if f["elsewhere"].search(t) and not re.search(rf"\b{re.escape(f['head'])}\b", t, re.I):
                return Verdict(OK, self.name, "another place")
            return None
        holders = {b[2] for b in f["blocks"] if b[0] <= tm <= b[1]}
        if not holders:
            return Verdict(OK, self.name, "outside the booked hours")
        if team in holders:
            return Verdict(OK, self.name, "the team's own booking")
        return Verdict(LIE, self.name, f"the {'/'.join(sorted(holders))} team had {f['room']} at {T.fmt_time(tm)}")


# ----------------------------------------------------------------------------------------------- owners over a month

NOBODY_ELSE_RE = re.compile(
    rf"\b(?:[Nn]obody|[Nn]o one) else owned the ([a-z][a-z -]*?) (?:in|during) ({MONTH_ALT}) (\d{{4}})\."
    rf"|^(?:In|During) ({MONTH_ALT}) (\d{{4}}),? (?:nobody|no one) else owned the ([a-z][a-z -]*?)\.")
SALE_RE = re.compile(rf"\b(?:sold|selling) it (?:on )?to ({NAME})\b")
AFTER_BUYING_RE = re.compile(r"\bexactly (\w+) days? after buying it\b", re.I)
OTHER_HANDOVER_RE = re.compile(r"\b(?:gave|given|handed|lent|loaned|passed|transferred|inherited|swapped|traded|"
                               r"returned|donated|gifted|stole|stolen|lost|found|rented|shared|co-owned|jointly)\b",
                               re.I)
START_OWNER_RE = re.compile(rf"^(?:At the start of (?:{MONTH_ALT}) \d{{4}},? )?{NAME} owned the [a-z][a-z -]*?"
                            rf"(?: at the start of (?:{MONTH_ALT}) \d{{4}})?\.$")
NOT_SELLERS = NOT_NAMES | {"On", "Until", "At", "Exactly", "In", "During", "From", "Then", "After", "The"}


def sale(clause):
    """'Camila owned the kayak until selling it to Tomas on 8 September 2026.' -> (seller, buyer, date), where date
    is ('after', n) for 'Tariq sold it to Rahul exactly 14 days after buying it.'; None if it is not one sale."""
    m = SALE_RE.search(clause)
    if not m or len(SALE_RE.findall(clause)) != 1:
        return None
    buyer = m.group(1)
    sellers = {w for w in CAPITAL_RE.findall(clause) if w not in NOT_SELLERS and w != buyer}
    dates, rel = T.find_dates(clause), AFTER_BUYING_RE.search(clause)
    if len(sellers) != 1 or len(dates) + bool(rel) != 1:
        return None
    if rel:
        n = T.word_num(rel.group(1))
        return (sellers.pop(), buyer, ("after", n)) if n else None
    return sellers.pop(), buyer, dates[0][0]


@register
class MonthOwners(Family):
    """'Camila owned the kayak until selling it to Tomas on 8 September 2026. Tomas sold it to Isla on 20 September
    2026. Nobody else owned the kayak in September 2026.' Within that month the owners follow one another: the first
    owner up to the first sale, each buyer from their purchase to their own sale (sale days count for both). A
    speaker's claim to own it, or to have sold or bought it, on a day of that month that does not fit is a lie."""

    name = "cal_owner"

    def parse(self, ctx):
        heads = [(i, NOBODY_ELSE_RE.search(s)) for i, s in enumerate(ctx.sentences) if NOBODY_ELSE_RE.search(s)]
        if len(heads) != 1:
            return None
        hi, hm = heads[0]
        item, mon, yr = (hm.group(1), hm.group(2), hm.group(3)) if hm.group(1) else (hm.group(6), hm.group(4),
                                                                                    hm.group(5))
        sales, used = [], [hi]
        for i, s in enumerate(ctx.sentences):
            if i == hi:
                continue
            if OTHER_HANDOVER_RE.search(s):
                return None  # a gift, loan or swap would change who owned it when
            if not re.search(r"\b(?:sold|selling|bought|buying|owned|owner|belonged)\b", s):
                continue
            found = 0
            for clause in re.split(r",? and (?=[A-Z])", s):  # 'X sold it to Y on D, and Y sold it to Z on E.'
                if SALE_RE.search(clause):
                    one = sale(clause)
                    if one is None:
                        return None
                    sales.append(one)
                    found += 1
            if not found and not START_OWNER_RE.match(s):
                return None  # some other fact about owning or buying it
            used.append(i)
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
        month = (int(yr), T.MONTHS.index(mon) + 1)
        if any((d.year, d.month) != month for _, _, d in resolved):
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
        names = [w for w in CAPITAL_RE.findall(t) if w not in NOT_SELLERS and w not in ("I", "As", "For")]
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


if __name__ == "__main__":  # self-check of the calendar arithmetic
    assert weekday_span("Wednesday", "Sunday") == {2, 3, 4, 5, 6} and weekday_span("Friday", "Monday") == {4, 5, 6, 0}
    assert weekday(dt.date(2026, 4, 14), (dt.date(2026, 4, 1), 2)) == 1
    assert age_on(dt.date(1993, 7, 14), dt.date(2026, 7, 13)) == 32
    assert age_on(dt.date(1993, 7, 14), dt.date(2026, 7, 14)) == 33
    nts = [dt.date(2026, 3, 3) + dt.timedelta(days=k) for k in range(6)]
    w = NightRota.assign("Usha took the first two, Jonas took the next two, and Akira took the last two.", nts)
    assert w[dt.date(2026, 3, 5)] == "Jonas" and w[dt.date(2026, 3, 8)] == "Akira"
    w = NightRota.assign("Lin on 5–6 March, Fatima on 7–8 March, Farid on 3–4 March", nts)
    assert w[dt.date(2026, 3, 3)] == "Farid"
    nts = [dt.date(2026, 12, 30) + dt.timedelta(days=k) for k in range(4)]
    w = NightRota.assign("Lin on 30–31 December, Farid on 1–2 January", nts)
    assert w[dt.date(2027, 1, 2)] == "Farid"
    assert claim_text("Honestly, I swam at the pool at 1:25 PM, for the record.") == "I swam at the pool at 1:25 PM."
    assert plain_claim("Joel told me he swam at the pool.") is None and plain_claim("In May 2026 I turned 30.")
    assert not re.search(the_noun("tour"), "I was at the tour desk at 1 PM")
    print("ok")
