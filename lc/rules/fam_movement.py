"""Rule families: movement and custody. See lc/rules/core.py for the Family interface.

Each family reads one kind of evidence and judges only messages that name the same vehicle, place, person or object
the evidence names. The nouns come from the evidence, so the patterns carry no scenario nouns of their own:

- move_route: a vehicle leaves at a time and goes along a route with speed limits, so it cannot arrive before
  departure + distance / top speed.
- move_away: a vehicle leaves a stop and cannot be back before a time, so nobody boards it at that stop meanwhile.
- move_place: a person is at one place at a time, and every route to another place takes at least some minutes.
- move_room: a person stayed in one room for a window, so they were not in a separate room then.
- move_object: one object stayed in one place for a window, so nobody took it out or had it elsewhere then.
- move_seal: a container stayed sealed for a window, so nobody removed what it held then.
- move_chain: one object passed from hand to hand at known times, and only one person held it at any moment.
- move_count: a container held exactly N items for a window, and none were added or removed.

Claims at the exact edge of a window are left to the fallback judge, since "left at 2:15 PM" and "stayed until
1:20 PM" read both ways. Each judge returns LIE if any parsed fact rules the claim out, else OK if one allows it.
"""
import re

from .core import LIE, OK, Family, Verdict, register
from . import text as T

TM = r"(\d{1,2}:\d{2} (?:am|pm)|noon|midday|midnight)"  # evidence and claims are matched in lower case
NP = r"[a-z][a-z0-9' -]*?"  # a noun phrase after "the"; lazy, so the following words decide where it ends
END = r"(?=[,.;:]|$| and\b| which\b| with\b)"
NAME = (r"(?!(?:nobody|no one|noone|none|someone|somebody|anyone|anybody|everyone|everybody|nothing|not|never)\b)"
        r"[a-z]+")
PRONOUNS = {"i", "me", "you", "he", "him", "she", "her", "they", "them", "we", "us", "it", "this", "that", "there",
            "who", "one", "the", "nobody", "everyone", "someone"}
EPS = 1e-6


def low(s):
    return re.sub(r"\s+", " ", s.lower().replace("’", "'")).strip()


def esc(s):
    return re.escape(s)


def tm(s):
    """Minutes since midnight for one clock time such as '2:45 pm'."""
    return T.parse_time(s)


LOC_MOD = re.compile(r"^(.+?) (?:by|near|beside|next to|behind|under|opposite) the [a-z ]+$")


def np_re(np, sents):
    """Regex for a place or container as messages name it: 'the cabinet by the window' may be just 'the cabinet',
    unless the evidence also names another cabinet ('the grey cabinet')."""
    m = LOC_MOD.match(np)
    if not m:
        return esc(np)
    head = m.group(1)
    if any(w != "the" for s in sents for w in re.findall(rf"\b([a-z]+) {esc(head)}\b", s)):
        return esc(np)
    return rf"(?:{esc(np)}|{esc(head)})"


def is_person(name, original):
    """A lower-case word from an evidence sentence that is written as a name (capitalised) in the original."""
    return name not in PRONOUNS and re.search(rf"\b{esc(name.capitalize())}\b", original) is not None


# ---------------------------------------------------------------- claims

FILLER_PRE = re.compile(
    r"^(?:(?:honestly|for the record|okay so|ok so|okay|so|well|to be honest|to be clear|frankly|actually|"
    r"as it happens|by my count|i can tell you(?: that)?|i say(?: that)?|i can confirm(?: that)?)\s+)+")
FILLER_SUF = re.compile(r"(?:\s+(?:for the record|honestly|as it happens|to be honest|to be clear|by the way))+$")
OTHER_DAY = re.compile(r"\b(?:yesterday|tomorrow|day before|next day|previous day|following day|other day|"
                       r"last week|next week|last night)\b")


def claim(text):
    """(minutes, clause) for a message that states one thing at one clock time.

    The clause is lower case with the time phrase ('at 2:45 PM', 'Come 9:15 AM', 'It was 3:10 PM when', 'It was at
    3:10 PM that'), commas and discourse fillers ('Honestly,', ', for the record') removed. None for questions,
    messages with no time or several times, dates or other days, and times after any word but 'at' (before, after,
    around, by ...)."""
    s = text.strip().replace("’", "'")
    if "?" in s or T.find_dates(s) or T.find_weekdays(s):
        return None
    times = T.find_times(s)
    if len(times) != 1:
        return None
    minutes, m = times[0]
    before, after = s[:m.start()], s[m.end():]
    if re.fullmatch(r"\s*it was at\s*", before, re.I) and re.match(r"\s*,?\s*that\b", after, re.I):
        before, after = "", re.sub(r"^\s*,?\s*that\b", "", after, flags=re.I)
    elif re.fullmatch(r"\s*it was\s*", before, re.I) and re.match(r"\s*,?\s*when\b", after, re.I):
        before, after = "", re.sub(r"^\s*,?\s*when\b", "", after, flags=re.I)
    elif re.search(r"\b(?:at|come)\s*$", before, re.I):
        before = re.sub(r"\b(?:at|come)\s*$", "", before, flags=re.I)
    else:
        return None
    s = (before + " " + after).lower()
    if OTHER_DAY.search(s):
        return None
    s = re.sub(r"[,;:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .!")
    s = FILLER_PRE.sub("", s)
    s = FILLER_SUF.sub("", s)
    return minutes, s.strip()


def window_verdict(t, t1, t2, fam, inside_why):
    """LIE strictly inside (t1, t2), None on either edge, OK outside."""
    if t1 + EPS < t < t2 - EPS:
        return Verdict(LIE, fam, inside_why)
    if abs(t - t1) < EPS or abs(t - t2) < EPS:
        return None
    return Verdict(OK, fam, f"outside {T.fmt_time(t1)}-{T.fmt_time(t2)}")


def best(verdicts):
    """LIE if any fact rules the claim out, else the first OK, else None."""
    vs = [v for v in verdicts if v is not None]
    return next((v for v in vs if v.status == LIE), vs[0] if vs else None)


def any_full(res, s):
    return any(r.fullmatch(s) for r in res)


# shared ways of telling an event, a presence or a sighting
SAW = r"(?:saw|watched|noticed|spotted|observed|witnessed|caught)"
WITNESS_TAIL = (r"(?: and i (?:saw|watched|witnessed) (?:it|it happen|him do it|her do it|them do it)| i saw it"
                r"(?: happen)?| while i (?:watched|looked on|was watching))?")


def event_res(past, base, ing):
    """An event told by its actor or a witness: 'I took X', 'Kofi took X, and I saw it', 'I saw Kofi take X',
    'I was watching when Kofi took X', 'Kofi taking X was something I saw', 'It was me who took X'. past, base and
    ing are the event after its actor in the past tense, the base form and the -ing form."""
    return [re.compile(rf"(?:i|{NAME}) {past}{WITNESS_TAIL}"),
            re.compile(rf"i {SAW} {NAME} (?:{base}|{ing})"),
            re.compile(rf"i (?:was watching|was there|looked on) when {NAME} {past}"),
            re.compile(rf"{NAME} {ing} (?:is|was) something i (?:saw|watched|witnessed)"),
            re.compile(rf"it was (?:me|{NAME}) who {past}")]


TAKE_PAST = (r"(?:took|pulled|removed|lifted|got|grabbed|carried|moved|slid|fetched|brought|picked|took out|"
             r"pulled out|got out|lifted out|slid out|carried out)")
TAKE_BASE = (r"(?:take|pull|remove|lift|get|grab|carry|move|slide|fetch|bring|pick|take out|pull out|get out|"
             r"lift out|slide out|carry out)")
TAKE_ING = (r"(?:taking|pulling|removing|lifting|getting|grabbing|carrying|moving|sliding|fetching|bringing|picking|"
            r"taking out|pulling out|getting out)")
OUT_OF = r"(?:out of|from|out from|off)"


def removal_res(th, pl):
    """Patterns for 'somebody took THING out of PLACE', told by the actor or a witness. The helpers below take
    regex fragments for the nouns (from esc or np_re)."""
    rest = rf"the {th} {OUT_OF} the {pl}"
    return event_res(rf"{TAKE_PAST} {rest}", rf"{TAKE_BASE} {rest}", rf"{TAKE_ING} {rest}")


AT = (r"(?:at|in|inside|on|over at|over in|out at|out in|down at|down in|up at|up in|right at|already at|"
      r"already in|back at|back in)")
MEET = (r"(?:saw|spotted|noticed|caught sight of|met|ran into|bumped into|talked to|spoke to|spoke with|"
        r"chatted with|greeted)")
SEEN_TAIL = r"(?: when i (?:saw|spotted|noticed|met|ran into) (?:him|her|them)| i saw (?:him|her|them)(?: there)?)?"
NEAR = r"(?:next to|beside|near|by|right beside|right next to|close to)"


def self_presence(pl):
    """The sender at PLACE: 'I was at the shop', 'The shop is where I was', 'I got to the shop'."""
    return [re.compile(rf"i (?:was|stood|was standing|was sitting|was waiting|sat|waited|was working) {AT} the {pl}"
                       rf"(?: waiting)?"),
            re.compile(rf"(?:{AT} )?the {pl} (?:is|was) where i was"),
            re.compile(rf"i (?:got to|reached|arrived at|arrived in|made it to|walked into|went into|came into) "
                       rf"the {pl}")]


def other_presence(person, pl):
    """PERSON at PLACE as told by the sender: 'I saw Isla at the stadium', 'Isla was at the stadium when I saw her',
    'Over at the stadium, I saw Isla'."""
    p = esc(person)
    return [re.compile(rf"i {MEET} {p} {AT} the {pl}"),
            re.compile(rf"{p} was {AT} the {pl}{SEEN_TAIL}"),
            re.compile(rf"(?:over )?{AT} the {pl} i (?:saw|spotted|noticed|met) {p}")]


def near_res(pl):
    return [re.compile(rf"i (?:was|stood|was standing|was waiting|waited|sat|was sitting) (?:right )?{NEAR} the {pl}")]


# ---------------------------------------------------------------- move_route

DEPART = (r"(?:left|set off from|setting off from|set out from|pulled out of|pulled out from|pulled away from|"
          r"departed from|departed|leaving|departing from|departing|drove off from|drove out of|started from)")
GO = r"(?:went|drove|travelled|traveled|headed|ran|sailed|rode|moved|continued|going|driving|heading)"
STRAIGHT = r"(?:straight |directly |non-stop )?"
ROUTE_KM = r"(?:along the [\d.]+ km route |the [\d.]+ km route |along a [\d.]+ km route )?"
DEST_END = r"(?=[:,;.]|$| along\b| covering\b| first\b| never\b| and\b| at\b| without\b| on\b| via\b)"
ROUTE_A = re.compile(rf"^(?:at {TM},? )?(?:the|a|an) ({NP}) {DEPART} the ({NP})(?: at {TM})?,? (?:and |then |"
                     rf"and then )?{GO} {STRAIGHT}{ROUTE_KM}to the ({NP}){DEST_END}")
ROUTE_B = re.compile(rf"^(?:leaving|setting off from|departing from|departing|pulling out of) the ({NP}) at {TM},? "
                     rf"the ({NP}) {GO} {STRAIGHT}{ROUTE_KM}to the ({NP}){DEST_END}")
MAX_SPEED = re.compile(
    r"never (?:went|going|goes|got|getting|travelled|travelling|traveling) (?:above|over|faster than|beyond)"
    r"|never above|never exceed|without (?:ever )?(?:going|getting) (?:above|over|faster than)"
    r"|(?:did not|didn't) (?:go|travel|get) (?:above|over|faster than)|(?:did not|didn't) exceed"
    r"|no faster than|at no point did its speed (?:go|get) (?:above|over)|top speed|maximum speed|speed limit"
    r"|(?:never|no|not) more than|at most|(?:could not|couldn't|cannot|can't) (?:go|travel) (?:above|over|faster than)")
EXACT_SPEED = re.compile(r"\b(?:average|averaged|exactly|constant|steady|steadily|precisely)\b")
DELAY = re.compile(r"\b(?:stopp?ed|stops|waited|paused|break|delay|delayed|detour|broke down|refuel)")
ARRIVE = (r"(?:reached|got to|arrived at|arrived in|made it to|pulled up at|pulled up to|pulled into|pulled in at|"
          r"pulled in to|came into|rolled into|rolled up at|drew up at|docked at|turned up at|showed up at)")
ARRIVE_BASE = (r"(?:reach|get to|arrive at|arrive in|make it to|pull up at|pull into|pull in at|come into|roll into|"
               r"dock at|reaching|getting to|arriving at|arriving in|pulling up at|pulling into|pulling in at|"
               r"coming into|rolling into|docking at)")
ON_WAY = r"(?:on the way|on its way|en route|on the road|along the way|along the route|on the route|on its route)"


def route_legs(text):
    """Minimum travel minutes from 'N km' and 'N km/h' figures paired in order, or None when they do not pair."""
    kms = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:km|kilometres|kilometers)\b(?!\s*/\s*h|\s*per hour"
                                        r"|\s*an hour)", text)]
    sps = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:km/h|kph|km per hour|km an hour|kilometres per hour"
                                        r"|kilometers per hour)", text)]
    if not kms or not sps or min(sps) <= 0:
        return None
    if len(sps) == 1 and len(kms) == 1:
        return kms[0] / sps[0] * 60
    if len(kms) == len(sps) + 1 and abs(kms[0] - sum(kms[1:])) < EPS:  # a total, then the legs
        kms = kms[1:]
    if len(kms) == len(sps):
        return sum(k / v * 60 for k, v in zip(kms, sps))
    return None


def route_head(s):
    """(vehicle, origin, destination, departure minutes, sentence) for a departure sentence."""
    m = ROUTE_A.match(s)
    if m:
        veh, orig, dest, t0s = m.group(2), m.group(3), m.group(5), [x for x in (m.group(1), m.group(4)) if x]
    else:
        m = ROUTE_B.match(s)
        if not m:
            return None
        orig, veh, dest, t0s = m.group(1), m.group(3), m.group(4), [m.group(2)]
    if len(t0s) != 1 or len(T.find_times(s)) != 1 or orig == dest:
        return None
    return veh, orig, dest, tm(t0s[0]), s


@register
class MoveRoute(Family):
    name = "move_route"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        out = []
        for i, s in enumerate(sents):
            r = route_head(s)
            if not r:
                continue
            veh, _orig, dest, t0, tail = r
            v, d = esc(veh), esc(dest)
            used, parts = [i], [tail]
            head = (rf"(?:its|it|the {v}'s|the {v}|for the|on the|over the|the first|first|then|after that|"
                    rf"at no point)\b")
            for j in range(i + 1, len(sents)):
                if "km" in sents[j] and re.match(head, sents[j]) and not re.search(DEPART, sents[j]):
                    used.append(j)
                    parts.append(sents[j])
                else:
                    break
            text = " ".join(parts)
            speed_info = re.search(r"km/h|per hour|an hour|mph|mile|kph", text) is not None
            earliest = None
            if speed_info and MAX_SPEED.search(text) and not EXACT_SPEED.search(text) and \
                    not re.search(r"mph|mile", text):
                mins = route_legs(text)
                earliest = None if mins is None else t0 + mins
            if speed_info and earliest is None:
                continue  # speeds we could not read: claims stay unjudged
            for j in used:
                ctx.consume(j)
            noun = esc(veh.split()[-1])
            out.append({
                "veh": veh, "t0": t0, "earliest": earliest, "ok": not DELAY.search(text),
                "arrive": [re.compile(rf"(?:the )?{v} {ARRIVE} the {d}"),
                           re.compile(rf"i {SAW} the {v} {ARRIVE_BASE} the {d}"),
                           re.compile(rf"i (?:saw|spotted|noticed) the {v} (?:at|in) the {d}"),
                           re.compile(rf"the {v} was (?:at|in) the {d}")],
                "way": [re.compile(rf"i (?:saw|spotted|noticed|passed|caught sight of|glimpsed|watched) the {v} "
                                   rf"{ON_WAY}(?: to the {d})?"),
                        re.compile(rf"{ON_WAY} i (?:saw|spotted|noticed|passed) the {v}")],
                "other": [re.compile(rf"(?:another|some other|a different|some different|one other|a second) "
                                     rf"(?:vehicle|{noun}|[a-z]+ {noun}) {ARRIVE} the {d}")],
            })
        return out or None

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        return best(self.one(f, *c) for f in facts) if c else None

    def one(self, f, t, cl):
        if any_full(f["arrive"], cl):
            if t < f["t0"]:
                return None  # an arrival before it set off could be an earlier trip
            e = f["earliest"]
            if e is not None and t < e - EPS:
                return Verdict(LIE, self.name, f"the {f['veh']} left at {T.fmt_time(f['t0'])}; earliest arrival "
                                               f"{T.fmt_time(e)}")
            if f["ok"] and t > f["t0"]:
                return Verdict(OK, self.name, "arrival after the earliest possible time")
            return None
        if any_full(f["way"], cl):
            return Verdict(OK, self.name, "seen on the way after it left") if t >= f["t0"] else None
        if any_full(f["other"], cl):
            return Verdict(OK, self.name, "another vehicle")
        return None


# ---------------------------------------------------------------- move_away

AWAY_HEAD = re.compile(rf"^(?:at {TM},? )?(?:the|a|an) ({NP}) {DEPART} the ({NP})(?: at {TM})?(?=[,.;]|$| and\b)")
AWAY_RET = [re.compile(rf"(?:did not|didn't|was not|wasn't|could not be|couldn't be) (?:return|come back|get back|"
                       rf"arrive back|back)(?: (?:at|to) the ({NP}))? (?:before|until) {TM}"),
            re.compile(rf"(?:was|stayed) away(?: from the ({NP}))? (?:until|till) {TM}"),
            re.compile(rf"(?:returned|came back|got back)(?: to the ({NP}))? at {TM}")]  # (stop, time)
ROUND = re.compile(rf"round trip(?: of the ({NP}))? took at least ([^,;.]+)")
ROUND_BACK = re.compile(rf"could not be back(?: at the ({NP}))? before then")
BOARD = (r"(?:got on|got onto|got on board|got aboard|got into|boarded|climbed onto|climbed on|climbed aboard|"
         r"climbed into|stepped onto|stepped on|stepped aboard|stepped into|hopped on|hopped onto|hopped aboard|"
         r"hopped into|went aboard|went on board|jumped on|jumped onto|jumped aboard|caught|embarked on|got off|"
         r"stepped off|climbed off|hopped off|jumped off)")
BOARD_BASE = (r"(?:get on|get onto|get on board|get aboard|get into|board|climb onto|climb on|climb aboard|"
              r"climb into|step onto|step on|step aboard|step into|hop on|hop onto|hop aboard|go aboard|go on board|"
              r"jump on|jump onto|jump aboard|catch|embark on|get off|step off|climb off|hop off)")
BOARD_ING = (r"(?:getting on|getting onto|getting aboard|boarding|climbing onto|climbing on|climbing aboard|"
             r"stepping onto|stepping aboard|hopping on|hopping onto|going aboard|getting off|stepping off)")


@register
class MoveAway(Family):
    name = "move_away"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        out = []
        for i, s in enumerate(sents):
            m = AWAY_HEAD.match(s)
            if not m or route_head(s):
                continue
            veh, stop = m.group(2), m.group(3)
            t0s = [x for x in (m.group(1), m.group(4)) if x]
            if len(t0s) != 1:
                continue
            t0 = tm(t0s[0])
            used, t1 = [i], None
            for j in (i, i + 1):
                if j >= len(sents):
                    break
                sj = sents[j] if j > i else s[m.end():]
                if j > i and not re.match(rf"(?:its|it|the round trip of the {esc(veh)}|the {esc(veh)})\b", sj):
                    break
                for rx in AWAY_RET:
                    r = rx.search(sj)
                    if r and r.group(1) in (None, stop) and t1 is None:
                        t1 = tm(r.group(2))
                r = ROUND.search(sj)
                if r and r.group(1) in (None, veh):
                    back = ROUND_BACK.search(sj)
                    d = T.find_durations(r.group(2))
                    if len(d) == 1 and back and back.group(1) in (None, stop):
                        t1 = t0 + d[0][0]
                if t1 is not None:
                    used.append(j)
                    break
            if t1 is None or t1 <= t0:
                continue
            for j in set(used):
                ctx.consume(j)
            v, st = esc(veh), esc(stop)
            for j, sj in enumerate(sents):
                if re.search(rf"(?:^|, and )nobody could (?:get on|board) the {v} at the {st} unless (?:the {v}|it) "
                             rf"was at the {st}\.?$", sj):
                    ctx.consume(j)
            board = rf"the {v} at the {st}"
            out.append({
                "veh": veh, "stop": stop, "t0": t0, "t1": t1,
                "board": event_res(rf"{BOARD} {board}", rf"{BOARD_BASE} {board}", rf"{BOARD_ING} {board}") + [
                    re.compile(rf"at the {st} i {BOARD} the {v}"),
                    re.compile(rf"the {v} was at the {st}"),
                    re.compile(rf"i (?:saw|spotted|noticed) the {v} (?:at|waiting at|parked at) the {st}")],
                "wait": [re.compile(rf"i (?:was )?(?:standing )?(?:waiting )?(?:at|by|near) the {st}(?: waiting)?"),
                         re.compile(rf"i stood (?:waiting )?at the {st}")],
            })
        return out or None

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        return best(self.one(f, *c) for f in facts) if c else None

    def one(self, f, t, cl):
        if any_full(f["board"], cl):
            if f["t0"] + EPS < t < f["t1"] - EPS:
                return Verdict(LIE, self.name, f"the {f['veh']} was away from the {f['stop']} from "
                                               f"{T.fmt_time(f['t0'])} until at least {T.fmt_time(f['t1'])}")
            if abs(t - f["t0"]) < EPS:
                return None
            return Verdict(OK, self.name, "outside the time it was away")
        if any_full(f["wait"], cl):
            return Verdict(OK, self.name, "waiting at the stop")
        return None


# ---------------------------------------------------------------- move_place

PERSON_AT = re.compile(rf"^(?:at {TM},? )?([a-z]+) was at the ({NP})(?: at {TM})?(?=[,.;]|$| and\b)")
ROUTE_MIN = [re.compile(rf"every (?:route|road|path|way) from the (?P<a>{NP}) to the (?P<b>{NP}) (?:also )?takes at "
                        rf"least (?P<d>[^,;.]+)"),
             re.compile(rf"(?:getting|going|travelling|traveling|walking|driving|the journey|the trip) from the "
                        rf"(?P<a>{NP}) to the (?P<b>{NP}) (?:always )?takes at least (?P<d>[^,;.]+)"),
             re.compile(rf"it takes at least (?P<d>[^,;.]+?) to (?:get|go|travel|walk|drive) from the (?P<a>{NP}) to the "
                        rf"(?P<b>{NP})(?=[,.;]|$| and\b)")]
ROUTE_VIA = re.compile(rf"every (?:route|road|path|way) from the ({NP}) to the ({NP}) (?:passes|goes|runs) "
                       rf"through the ({NP})(?=[,.;]|$| and\b)")
SEPARATE = re.compile(rf"the ({NP})(?:, which [^,]*,)?(?:, the ({NP}))?,? and the ({NP})(?:, which [^,]*,)? are "
                      rf"(?:two )?separate (?:locations|places|sites|buildings|rooms|areas|spaces)")


def separate_groups(sents):
    return [{g for g in m.groups() if g} for s in sents for m in SEPARATE.finditer(s)]


def consume_separate(ctx, sents, names):
    for i, s in enumerate(sents):
        m = SEPARATE.search(s)
        if m and not re.search(r"\d", s) and names & {g for g in m.groups() if g}:
            ctx.consume(i)


@register
class MovePlace(Family):
    name = "move_place"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        edges, via, used = {}, {}, set()
        for i, s in enumerate(sents):
            for rx in ROUTE_MIN:
                for r in rx.finditer(s):
                    d = T.find_durations(r.group("d"))
                    if len(d) == 1 and d[0][0] > 0:
                        edges[(r.group("a"), r.group("b"))] = d[0][0]
                        used.add(i)
            for r in ROUTE_VIA.finditer(s):
                via[(r.group(1), r.group(2))] = r.group(3)
                used.add(i)
        if not edges:
            return None
        out = []
        for i, s in enumerate(sents):
            m = PERSON_AT.match(s)
            if not m or not is_person(m.group(2), ctx.sentences[i]):
                continue
            person, start = m.group(2), m.group(3)
            t0s = [x for x in (m.group(1), m.group(4)) if x]
            if len(t0s) != 1:
                continue
            bounds = {b: d for (a, b), d in edges.items() if a == start and b != start}
            for (a, b), mid in via.items():
                if a == start and (a, mid) in edges and (mid, b) in edges:
                    bounds[b] = max(bounds.get(b, 0), edges[(a, mid)] + edges[(mid, b)])
            if not bounds:
                continue
            ctx.consume(i)
            out.append({"person": person, "start": start, "t0": tm(t0s[0]), "bounds": bounds,
                        "places": {pl: (other_presence(person, np_re(pl, sents)), self_presence(np_re(pl, sents)))
                                   for pl in list(bounds) + [start]}})
        if not out:
            return None
        for i in used:
            ctx.consume(i)
        consume_separate(ctx, sents, {pl for f in out for pl in f["places"]})
        return out

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        if not c:
            return None
        if re.fullmatch(r"i was (?:out )?on the road", c[1]):
            return Verdict(OK, self.name, "on the road")
        sender = ctx.speakers.get(msg.label, "").lower()
        return best(self.one(f, sender, *c) for f in facts)

    def one(self, f, sender, t, cl):
        for pl, (others, selfs) in f["places"].items():
            about = f["person"] if any_full(others, cl) else sender if any_full(selfs, cl) else None
            if about is None:
                continue
            if about != f["person"] or pl == f["start"]:
                return Verdict(OK, self.name, "no travel limit applies")
            d, t0 = f["bounds"][pl], f["t0"]
            if t0 - EPS <= t < t0 + d - EPS:
                return Verdict(LIE, self.name, f"{f['person']} was at the {f['start']} at {T.fmt_time(t0)}; the "
                                               f"{pl} is at least {d} minutes away")
            if t >= t0 + d - EPS or t <= t0 - d:
                return Verdict(OK, self.name, "enough time to travel")
            return None  # shortly before: the evidence gives no limit for the way back
        return None


# ---------------------------------------------------------------- move_room

SPAN = r"from (?P<t1>\d{1,2}:\d{2} (?:am|pm)|noon) (?:to|until|till) (?P<t2>\d{1,2}:\d{2} (?:am|pm)|noon)"
BETWEEN = r"between (?P<t1>\d{1,2}:\d{2} (?:am|pm)|noon) and (?P<t2>\d{1,2}:\d{2} (?:am|pm)|noon)"
STAY_END = r"(?=[,.;]|$| and\b)"
NOT_LEAVE = r"(?:did not|didn't|never) (?:leave|left)"
WHOLE = r"(?:the whole time|the entire time|throughout)"
STAYS = [  # a person who stayed in one room for a window, as (p, r, t1, t2) named groups
    re.compile(rf"^(?P<p>[a-z]+) (?:stayed|remained) (?:in|inside) the (?P<r>{NP}) (?:{WHOLE} )?{SPAN}{STAY_END}"),
    re.compile(rf"^(?P<p>[a-z]+) was (?:in|inside) the (?P<r>{NP}) {WHOLE} {SPAN}{STAY_END}"),
    re.compile(rf"^{SPAN},? (?P<p>[a-z]+) (?:stayed|remained) (?:in|inside) the (?P<r>{NP}){STAY_END}"),
    re.compile(rf"^(?P<p>[a-z]+) {NOT_LEAVE} the (?P<r>{NP}) (?:at any point )?{BETWEEN}{STAY_END}"),
    re.compile(rf"^(?P<p>[a-z]+) {NOT_LEAVE} the (?P<r>{NP}) {SPAN}{STAY_END}"),
    re.compile(rf"^{BETWEEN},? (?P<p>[a-z]+) {NOT_LEAVE} the (?P<r>{NP}){STAY_END}"),
]
LEAVE = r"(?:left|walked out of|came out of|stepped out of|went out of|exited)"
LEAVE_BASE = r"(?:leave|walk out of|come out of|step out of|go out of|exit)"
LEAVE_ING = r"(?:leaving|walking out of|coming out of|stepping out of|going out of|exiting)"


@register
class MoveRoom(Family):
    name = "move_room"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        groups = separate_groups(sents)
        out = []
        for i, s in enumerate(sents):
            m = next((m for m in (rx.match(s) for rx in STAYS) if m), None)
            if not m:
                continue
            person, room, t1, t2 = m.group("p", "r", "t1", "t2")
            if not is_person(person, ctx.sentences[i]) or tm(t2) <= tm(t1):
                continue
            others = set().union(*[g - {room} for g in groups if room in g]) if groups else set()
            if not others:
                continue
            ctx.consume(i)
            p, r = esc(person), esc(room)
            out.append({"person": person, "room": room, "t1": tm(t1), "t2": tm(t2),
                        "rooms": {x: (other_presence(person, np_re(x, sents)), self_presence(np_re(x, sents)))
                                  for x in others | {room}},
                        "leave": [re.compile(rf"{p} {LEAVE} the {r}{WITNESS_TAIL}"),
                                  re.compile(rf"i {SAW} {p} (?:{LEAVE_BASE}|{LEAVE_ING}) the {r}")],
                        "self_leave": [re.compile(rf"i {LEAVE} the {r}")]})
        if not out:
            return None
        consume_separate(ctx, sents, {x for f in out for x in f["rooms"]})
        return out

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        if not c:
            return None
        sender = ctx.speakers.get(msg.label, "").lower()
        return best(self.one(f, sender, *c) for f in facts)

    def one(self, f, sender, t, cl):
        why = (f"{f['person']} stayed in the {f['room']} from {T.fmt_time(f['t1'])} to {T.fmt_time(f['t2'])}")
        if any_full(f["leave"], cl) or (sender == f["person"] and any_full(f["self_leave"], cl)):
            return window_verdict(t, f["t1"], f["t2"], self.name, why)
        for r, (others, selfs) in f["rooms"].items():
            about = f["person"] if any_full(others, cl) else sender if any_full(selfs, cl) else None
            if about is None:
                continue
            if about != f["person"] or r == f["room"]:
                return Verdict(OK, self.name, "not a different room for the person who stayed")
            return window_verdict(t, f["t1"], f["t2"], self.name, why)
        return None


# ---------------------------------------------------------------- move_object

OBJ_INTRO = re.compile(rf"^there was (?:one|a single|only one|exactly one) ({NP}){END}")
STAY_VERB = r"(?:stayed|remained|was kept|was locked|was left)"
OBJ_STAY = re.compile(rf"(?:^|, and |, which |; )(it|which|(?:the|that) {NP})? ?(?:{STAY_VERB}|was) in the ({NP})"
                      rf"(?:, which [^,]*,)? from {TM} (?:to|until|till) {TM}(?=[,.;]|$| and\b)")
OBJ_STAY_B = re.compile(rf"^from {TM} (?:to|until|till) {TM},? (it|(?:the|that) {NP}) {STAY_VERB} in the ({NP})"
                        rf"(?=[,.;]|$)")
STAYED_THERE = re.compile(r", and it (?:stayed|remained) there the whole time")
HAVE_AT = r"(?:at|in|on|inside)"


@register
class MoveObject(Family):
    name = "move_object"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        groups = separate_groups(sents)
        out, last = [], None
        for i, s in enumerate(sents):
            intro = OBJ_INTRO.match(s)
            if intro:
                last = intro.group(1)
            m = OBJ_STAY.search(s)
            if m:
                subj, place, t1, t2 = m.groups()
            else:
                m = OBJ_STAY_B.match(s)
                if not m:
                    continue
                t1, t2, subj, place = m.groups()
            if " was in the " in m.group(0) and not STAYED_THERE.search(s):
                continue  # 'was in the post room from ... to ...' alone does not say it never left
            if subj and subj.split(" ", 1)[0] in ("the", "that"):
                noun = subj.split(" ", 1)[1]
                if last and re.fullmatch(rf"(?:{esc(noun)}\b.*|.*\b{esc(noun)})", last):
                    obj = last  # 'The parcel stayed in ...' may shorten 'parcel marked X7'
                elif subj.startswith("the ") and not is_person(noun, ctx.sentences[i]):
                    obj = noun
                else:
                    continue
            elif last and (subj or intro):
                obj = last
            else:
                continue
            others = set().union(*[g - {place} for g in groups if place in g]) if groups else set()
            ctx.consume(i)
            o, pl = esc(obj), np_re(place, sents)
            bad = removal_res(o, pl)
            fine = [re.compile(rf"i (?:saw|spotted|noticed|checked) the {o} (?:sitting |lying |resting |still )?"
                               rf"(?:in|inside) the {pl}"),
                    re.compile(rf"the {o} was (?:still )?(?:in|inside) the {pl}(?: when i (?:saw|checked|spotted) it)?")
                    ] + near_res(pl)
            for other in others:
                ot = np_re(other, sents)
                bad += [
                    re.compile(rf"i (?:had|was holding|held|was carrying|carried|used|was using) the {o}(?: with me| "
                               rf"along with me| on me)? {HAVE_AT} the {ot}"),
                    re.compile(rf"{HAVE_AT} the {ot} i (?:had|was holding|was carrying) the {o}(?: with me)?"),
                    re.compile(rf"the {o} was (?:with me )?{HAVE_AT} the {ot}"),
                    re.compile(rf"i (?:saw|spotted|noticed) the {o} {HAVE_AT} the {ot}"),
                    re.compile(rf"i (?:saw|spotted|noticed) {NAME} (?:with|holding|carrying|using) the {o} {HAVE_AT} "
                               rf"the {ot}"),
                    re.compile(rf"(?:i|{NAME}) (?:moved|carried|took|brought) the {o} (?:over )?to the {ot}")]
                fine += self_presence(ot)
            out.append({"obj": obj, "place": place, "t1": tm(t1), "t2": tm(t2), "bad": bad, "fine": fine})
        if not out:
            return None
        objs = {f["obj"] for f in out}
        places = {f["place"] for f in out}
        for i, s in enumerate(sents):
            m = OBJ_INTRO.match(s)
            if m and m.group(1) in objs and re.fullmatch(rf"there was (?:one|a single|only one|exactly one) "
                                                         rf"{esc(m.group(1))}(?:, (?:with|which has|and it has) "
                                                         rf"[a-z ]+)?\.?", s):
                ctx.consume(i)
            if re.fullmatch(rf"the (?:{'|'.join(esc(x) for x in objs | places)}) (?:has|is|was) [a-z ]+\.?", s):
                ctx.consume(i)  # decoration such as 'The red key has a white label.'
        consume_separate(ctx, sents, places)
        return out

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        return best(self.one(f, *c) for f in facts) if c else None

    def one(self, f, t, cl):
        if any_full(f["bad"], cl):
            return window_verdict(t, f["t1"], f["t2"], self.name,
                                  f"the {f['obj']} stayed in the {f['place']} from {T.fmt_time(f['t1'])} to "
                                  f"{T.fmt_time(f['t2'])}")
        if any_full(f["fine"], cl):
            return Verdict(OK, self.name, "consistent with the object staying put")
        return None


# ---------------------------------------------------------------- move_seal

SEAL = re.compile(rf"(?:^|, and |; |, )(?:from {TM} (?:to|until|till) {TM},? )?(it|(?:the|that|this) {NP}) "
                  rf"(?:stayed|remained) sealed(?: from {TM} (?:to|until|till) {TM})?")
SEAL_ITEM = re.compile(rf"the ({NP}) could not be (?:removed|taken out|taken)(?: from it)? without breaking the seal")
HELD_IN = [re.compile(rf"(?:^|, )(?:one|a single|a|the) ({NP})(?:, which [^,]*,)? held the ({NP}){END}"),
           re.compile(rf"^the ({NP}) (?:was|were) (?:kept |held |stored |placed |sealed )?(?:in|inside) (?:one|a single|"
                      rf"a) ({NP})(?=[,.;]|$| and\b| which\b| that\b)")]
HELD_IN_ONLY = [re.compile(rf"(?:one|a single|a|the) {NP}(?:, which [^,]*,)? held the {NP}(?: and was kept on a shelf)?"
                           rf"\.?"),
                re.compile(rf"the {NP} (?:was|were) (?:kept |held |stored |placed )?(?:in|inside) (?:one|a single|a) "
                           rf"{NP}(?:,? (?:which|that) was (?:kept on a shelf|labelled in [a-z]+ ink))?\.?")]
OPEN = r"(?:opened|unsealed|broke open|tore open|cut open|ripped open|unwrapped)"


@register
class MoveSeal(Family):
    name = "move_seal"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        known = []  # (container, item) from 'One evidence bag held the USB drive.'
        out = []
        for i, s in enumerate(sents):
            for k, r in enumerate(HELD_IN):
                m = r.search(s)
                if m:
                    known.append((m.group(1), m.group(2)) if k == 0 else (m.group(2), m.group(1)))
                    if any_full(HELD_IN_ONLY, s):
                        ctx.consume(i)
            m = SEAL.search(s)
            it = SEAL_ITEM.search(s)
            if not m or not it:
                continue
            times = [x for x in m.groups()[:2] + m.groups()[3:] if x]
            if len(times) != 2:
                continue
            subj, item = m.group(3), it.group(1)
            if subj == "it":
                cont = known[-1][0] if known else None
            elif subj.startswith("the "):
                cont = subj[4:]
            else:  # 'that bag', 'that envelope': the container named earlier
                word = subj.split(" ", 1)[1]
                cand = [c for c, _ in known if c == word or c.endswith(" " + word)]
                cont = cand[-1] if cand else None
            if cont is None:
                continue
            ctx.consume(i)
            c, itm = np_re(cont, sents), esc(item)
            opened = rf"the {c}"
            out.append({
                "cont": cont, "item": item, "t1": tm(times[0]), "t2": tm(times[1]),
                "bad": removal_res(itm, c)
                + event_res(rf"{OPEN} {opened}", rf"(?:open|unseal|break open|tear open|cut open) {opened}",
                            rf"(?:opening|unsealing|breaking open|tearing open|cutting open) {opened}")
                + event_res(rf"broke the seal (?:on|of) {opened}", rf"break the seal (?:on|of) {opened}",
                            rf"breaking the seal (?:on|of) {opened}"),
                "fine": [re.compile(rf"i (?:was holding|held|had|was carrying|carried|picked up|had hold of) the {c}"),
                         re.compile(rf"when i (?:saw|looked at|checked|inspected|examined) (?:the {c} it|it the {c}) "
                                    rf"was still sealed"),
                         re.compile(rf"the {c} was still sealed(?: when i (?:saw|looked at|checked|inspected|examined) "
                                    rf"it)?"),
                         re.compile(rf"i (?:saw|checked|noticed) that the {c} was still sealed"),
                         re.compile(rf"i (?:saw|checked|looked at|inspected|examined) the {c} and it was still sealed"),
                         re.compile(rf"i saw the {itm} (?:in|inside) the {c}")],
            })
        if not out:
            return None
        names = {f["cont"] for f in out} | {f["cont"].split()[-1] for f in out}
        for i, s in enumerate(sents):
            if re.fullmatch(rf"the (?:{'|'.join(esc(x) for x in names)}) (?:was|is|has) [a-z ]+\.?", s):
                ctx.consume(i)  # decoration such as 'The sealed envelope was labelled in black ink.'
        return out

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        return best(self.one(f, *c) for f in facts) if c else None

    def one(self, f, t, cl):
        if any_full(f["bad"], cl):
            return window_verdict(t, f["t1"], f["t2"], self.name,
                                  f"the {f['cont']} stayed sealed from {T.fmt_time(f['t1'])} to "
                                  f"{T.fmt_time(f['t2'])}")
        if any_full(f["fine"], cl):
            return Verdict(OK, self.name, "consistent with the seal")
        return None


# ---------------------------------------------------------------- move_chain

CHAIN_SPLIT = re.compile(r",? and then |, then |; |,? before |, who |, and |\. ")
HOLD = r"(?:held|kept|had|holding|keeping)"


def parse_chain(text):
    """[(holder, start, end)] from 'Aiko held it from 10:50 AM to 11:50 AM, then handed it to Sofia. Sofia held it
    until 1:20 PM ...' and 'From 10:45 AM Fatima held it for exactly 1 hour, then handed it to Isla.'; None when a
    clause does not fit or the times do not join up."""
    segs, pending = [], None
    for cl in CHAIN_SPLIT.split(text.strip(" .")):
        cl = cl.strip(" ,.")
        if not cl:
            continue
        m = re.fullmatch(r"(?:then )?(?:hand(?:ed|ing)|pass(?:ed|ing)|g(?:ave|iving)) it (?:over )?to ([a-z]+)", cl)
        if m:
            if pending is not None or not segs:
                return None
            pending = m.group(1)
            continue
        m = re.fullmatch(rf"(?:from {TM}(?: (?:to|until) {TM})?,? )?(?:([a-z]+) )?{HOLD} it(?: from {TM})?"
                         rf"(?: (?:to|until|till) {TM})?(?:,? (?:and )?(?:kept|keeping|held|holding) it)?"
                         rf"(?: for exactly ([^,]+))?", cl)
        if not m:
            return None
        f1, t1, who, f2, t2, dur = m.groups()
        who = who or pending
        if who is None or who in PRONOUNS or (pending and who != pending):
            return None
        pending = None
        start = f1 or f2
        start = tm(start) if start else (segs[-1][2] if segs else None)
        if start is None or (segs and abs(start - segs[-1][2]) > EPS):
            return None
        end = t1 or t2
        if end:
            end = tm(end)
        elif dur:
            d = T.find_durations(dur)
            if len(d) != 1:
                return None
            end = start + d[0][0]
        else:
            return None
        if end <= start:
            return None
        segs.append((who, start, end))
    return segs if segs and pending is None else None


@register
class MoveChain(Family):
    name = "move_chain"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        out = []
        for i, s in enumerate(sents):
            m = re.fullmatch(rf"there was (?:one|a single|only one|exactly one) ({NP})(?:,? and (only one person held "
                             rf"it at any moment))?\.?", s)
            if not m:
                continue
            obj = m.group(1)
            o = esc(obj)
            chain_idx, rest_idx, window, exclusive = [], [], None, bool(m.group(2))
            for j in range(i + 1, len(sents)):
                sj = sents[j]
                w = re.search(rf"(?:nobody else held (?:the {o}|it) between {TM} and {TM}|between {TM} and {TM},? "
                              rf"nobody else held (?:the {o}|it))", sj)
                ex = re.search(rf"only one person held (?:the {o}|it) at any moment|at any moment,? only one person "
                               rf"held (?:the {o}|it)", sj)
                if w or ex:
                    if w:
                        g = [x for x in w.groups() if x]
                        window = (tm(g[0]), tm(g[1]))
                    exclusive = exclusive or bool(ex)
                    rest_idx.append(j)
                    continue
                if re.search(rf"\b{HOLD} it\b|hand(?:ed|ing) it", sj):
                    chain_idx.append(j)
                    continue
                break
            if not chain_idx:
                continue
            segs = parse_chain(" ".join(sents[j] for j in chain_idx))
            if not segs or not (exclusive or window):
                continue
            if window and (abs(window[0] - segs[0][1]) > EPS or abs(window[1] - segs[-1][2]) > EPS):
                continue
            for j in [i] + chain_idx + rest_idx:
                ctx.consume(j)
            out.append({
                "obj": obj, "segs": segs, "exclusive": exclusive, "nobody_else": window is not None,
                "names": {h for h, _, _ in segs},
                "self": re.compile(rf"i (?:had|held|was holding|was carrying|had hold of|was minding|kept|was keeping"
                                   rf"|was looking after) the {o}(?: with me| in my hands?| on me)?|the {o} was (?:with "
                                   rf"me|in my hands?|on me)"),
                "other": re.compile(rf"i {SAW} ({NAME}) (?:holding|hold|with|carrying|carry|keeping) the {o}|({NAME}) "
                                    rf"(?:had|held|was holding|was carrying) the {o}{SEEN_TAIL}"),
                "give": re.compile(rf"i (?:gave|handed|passed) (?:the {o} (?:over )?to ({NAME})|({NAME}) the {o})"),
                "take": re.compile(rf"({NAME}) (?:gave|handed|passed) (?:me the {o}|the {o} (?:over )?to me)|i "
                                   rf"(?:got|received|took) the {o} from ({NAME})")})
        return out or None

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        if not c:
            return None
        sender = ctx.speakers.get(msg.label, "").lower()
        return best(self.one(f, sender, *c) for f in facts)

    @staticmethod
    def holder(f, t):
        """The holder strictly inside a segment, 'edge' on a hand-over or window edge, None outside the chain."""
        for h, a, b in f["segs"]:
            if a + EPS < t < b - EPS:
                return h
            if abs(t - a) < EPS or abs(t - b) < EPS:
                return "edge"
        return None

    def one(self, f, sender, t, cl):
        start, end = f["segs"][0][1], f["segs"][-1][2]
        outside = t < start - EPS or t > end + EPS
        who = sender if f["self"].fullmatch(cl) else None
        m = None if who else f["other"].fullmatch(cl)
        if m:
            who = m.group(1) or m.group(2)
        if who:
            if who in PRONOUNS:
                return None  # 'I saw her holding it': we cannot tell who 'her' is
            if outside:
                return Verdict(OK, self.name, f"outside the hand-over chain of the {f['obj']}")
            h = self.holder(f, t)
            if h in (None, "edge"):
                return None
            if h == who:
                return Verdict(OK, self.name, f"{who} held the {f['obj']} then")
            if f["exclusive"] or (f["nobody_else"] and who not in f["names"]):
                return Verdict(LIE, self.name, f"{h} held the {f['obj']} at {T.fmt_time(t)}")
            return None
        giver = receiver = None
        m = f["give"].fullmatch(cl)
        if m:
            giver, receiver = sender, m.group(1) or m.group(2)
        m = f["take"].fullmatch(cl)
        if m:
            giver, receiver = m.group(1) or m.group(2), sender
        if giver is None or giver in PRONOUNS or receiver in PRONOUNS:
            return None
        if outside:
            return Verdict(OK, self.name, "hand-over outside the chain")
        for (h1, _, b1), (h2, _, _) in zip(f["segs"], f["segs"][1:]):
            if abs(t - b1) < EPS and h1 == giver and h2 == receiver:
                return Verdict(OK, self.name, "the recorded hand-over")
        h = self.holder(f, t)
        if h not in (None, "edge") and (f["exclusive"] or (f["nobody_else"] and receiver not in f["names"])):
            return Verdict(LIE, self.name, f"{h} held the {f['obj']} at {T.fmt_time(t)}")
        return None


# ---------------------------------------------------------------- move_count

T_A = r"(?P<a1>\d{1,2}:\d{2} (?:am|pm)|noon) (?:to|until|till) (?P<a2>\d{1,2}:\d{2} (?:am|pm)|noon)"
T_B = T_A.replace("<a1>", "<b1>").replace("<a2>", "<b2>")
COUNTS = [  # a container that held exactly N items for a window, as named groups
    re.compile(rf"^(?:from {T_A},? )?(?:the (?P<c>{NP})(?:, which [^,]*,)?|it) (?:held|contained) exactly "
               rf"(?P<n>[a-z0-9]+) (?P<items>{NP})(?: from {T_B})?(?=[,.;]|$| and\b)"),
    re.compile(rf"^(?:from {T_A},? )?there were exactly (?P<n>[a-z0-9]+) (?P<items>{NP}) in the (?P<c>{NP})"
               rf"(?: from {T_B})?(?=[,.;]|$| and\b)")]
NO_CHANGE = re.compile(rf"no ({NP}) (?:was|were) added (?:to it |to the {NP} )?or removed(?: from it)? during that time")
NUM_WORD = r"(\d+|no|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"


@register
class MoveCount(Family):
    name = "move_count"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        out, last_subject = [], None
        for i, s in enumerate(sents):
            m = next((m for m in (rx.match(s) for rx in COUNTS) if m), None)
            if not m:
                sm = re.match(rf"^the ({NP}) (?:stands|stood|sits|sat|is|was) ", s)
                last_subject = sm.group(1) if sm else last_subject
                continue
            cont = m.group("c") or last_subject
            times = [x for x in m.group("a1", "a2", "b1", "b2") if x]
            n = T.word_num(m.group("n"))
            items = m.group("items")
            if cont is None or len(times) != 2 or n is None or not items.endswith("s"):
                continue
            nc, nc_idx = NO_CHANGE.search(s), None
            if not nc and i + 1 < len(sents):
                nc = NO_CHANGE.search(sents[i + 1])
                nc_idx = i + 1 if nc else None
            item = nc.group(1) if nc else None
            if item is not None and not items.startswith(item[:-1] if item.endswith("y") else item):
                continue
            ctx.consume(i)
            if nc_idx is not None:
                ctx.consume(nc_idx)
            one = item or (items[:-3] + "y" if items.endswith("ies") else items[:-1])
            it, ones, c = esc(items), esc(one), np_re(cont, sents)
            things = rf"(?:{it}|{ones})"
            some = rf"(?:one of the {it}|{NUM_WORD} of the {it}|one {ones}|an? {ones}|a single {ones})"
            out.append({
                "cont": cont, "n": n, "t1": tm(times[0]), "t2": tm(times[1]), "no_change": item is not None,
                "count": [re.compile(rf"there (?:were|was) {NUM_WORD} {things} (?:in|inside) the {c}"),
                          re.compile(rf"(?:in|inside) the {c} there (?:were|was) {NUM_WORD} {things}"),
                          re.compile(rf"the {c} (?:held|had|contained|was holding|housed) {NUM_WORD} {things}"
                                     rf"(?: in it)?"),
                          re.compile(rf"{NUM_WORD} {things} (?:were|was) (?:in|inside) the {c}"),
                          # 'I saw five' may be some of them; 'I counted five' and 'all five' give the total
                          re.compile(rf"i (?:counted(?: all)?|(?:saw|photographed|checked) all) {NUM_WORD} {things} "
                                     rf"(?:in|inside) the {c}"),
                          re.compile(rf"(?:in|inside) the {c} i (?:counted(?: all)?|(?:saw|photographed) all) "
                                     rf"{NUM_WORD} {things}"),
                          re.compile(rf"i (?:took|got) (?:photos|pictures|a photo|a picture) of all {NUM_WORD} "
                                     rf"{things} (?:in|inside) the {c}")],
                "change": event_res(rf"{TAKE_PAST} (?:out )?{some} {OUT_OF} the {c}",
                                    rf"{TAKE_BASE} (?:out )?{some} {OUT_OF} the {c}",
                                    rf"{TAKE_ING} (?:out )?{some} {OUT_OF} the {c}") + [
                    re.compile(rf"(?:i|{NAME}) (?:added|put|placed|slipped|dropped) (?:another|an?|one|an extra) "
                               rf"{ones} (?:in|into|to) the {c}")],
                "near": near_res(c),
            })
        if not out:
            return None
        conts = {f["cont"] for f in out}
        for i, s in enumerate(sents):
            if re.fullmatch(rf"the (?:{'|'.join(esc(x) for x in conts)}) (?:is|was|stands|sits|has) [a-z ]+\.?", s):
                ctx.consume(i)
        return out

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        return best(self.one(f, *c) for f in facts) if c else None

    def one(self, f, t, cl):
        for r in f["count"]:
            m = r.fullmatch(cl)
            if m:
                k = T.word_num(m.group(1))
                if k is None:
                    return None
                inside = f["t1"] + EPS < t < f["t2"] - EPS
                edge = abs(t - f["t1"]) < EPS or abs(t - f["t2"]) < EPS
                if inside and k != f["n"]:
                    return Verdict(LIE, self.name, f"the {f['cont']} held exactly {f['n']} from "
                                                   f"{T.fmt_time(f['t1'])} to {T.fmt_time(f['t2'])}")
                if edge and k != f["n"]:
                    return None
                return Verdict(OK, self.name, "count consistent with the evidence")
        if any_full(f["change"], cl):
            if not f["no_change"]:
                return None
            return window_verdict(t, f["t1"], f["t2"], self.name,
                                  f"nothing was added to or removed from the {f['cont']} from "
                                  f"{T.fmt_time(f['t1'])} to {T.fmt_time(f['t2'])}")
        if any_full(f["near"], cl):
            return Verdict(OK, self.name, "beside the container")
        return None


if __name__ == "__main__":  # self-check: python -m lc.rules.fam_movement
    assert claim("Honestly, I took the red key out of the sealed cabinet at 3:30 PM.") == (
        930, "i took the red key out of the sealed cabinet")
    assert claim("It was 3:10 PM when the tractor got to the far field.") == (910, "the tractor got to the far field")
    assert claim("It was at 1:00 PM that I took the exam paper out of the sealed envelope.")[1] == \
        "i took the exam paper out of the sealed envelope"
    assert claim("I reached the shop before 9:00 AM.") is None and claim("Did the van reach the farm at 2:50 PM?") is None
    assert route_legs("first 12 km where it never went above 36 km/h, then 30 km where it never went above 60 km/h") == 50
    assert route_legs("along the 42 km route: first 12 km at 36 km/h, then 30 km at 60 km/h") == 50
    assert parse_chain("aiko held it from 10:50 am to 11:50 am, then handed it to sofia. sofia held it until 1:20 pm") == [
        ("aiko", 650, 710), ("sofia", 710, 800)]
    assert parse_chain("from 10:45 am fatima held it for exactly 1 hour, then handed it to isla. isla kept it for "
                       "exactly 1 hour, then handed it to sami, who kept it for exactly 2 hours") == [
        ("fatima", 645, 705), ("isla", 705, 765), ("sami", 765, 885)]
    assert window_verdict(600, 600, 660, "x", "") is None and window_verdict(630, 600, 660, "x", "").status == LIE
    print("ok")
