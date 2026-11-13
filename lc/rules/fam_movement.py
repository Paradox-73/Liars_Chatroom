"""Rule families: movement and custody. See lc/rules/core.py for the Family interface.

Each family reads one kind of evidence and judges only messages that name the same vehicle, place, person or object
the evidence names. The nouns come from the evidence, so the patterns carry no scenario nouns of their own:

- move_route: a vehicle leaves at a time and goes straight along a route with speed limits, so it cannot arrive
  before departure + distance / top speed.
- move_away: a vehicle leaves a stop and cannot be back before a time, so nobody boards it at that stop meanwhile.
- move_place: a person is at one place at a time, and every route to another place takes at least some minutes.
- move_room: a person stayed in one room for a window, so they were not in a separate room then.
- move_object: one object stayed in one place for a window, so nobody took it out or had it elsewhere then.
- move_seal: a container stayed sealed for a window, so nobody removed what it held then.
- move_chain: one object passed from hand to hand at known times, and only one person held it at any moment.
- move_count: a container held exactly N items for a window, and none were added or removed.

Verdicts at the exact edge of a window are left to the fallback judge: "left at 2:15 PM" or "stayed until 1:20 PM"
read both ways.
"""
import re

from .core import LIE, OK, Family, Verdict, register
from . import text as T

TM = r"(\d{1,2}:\d{2} (?:am|pm)|noon|midday|midnight)"  # evidence and claims are matched in lower case
NP = r"[a-z][a-z0-9' -]*?"  # a noun phrase after "the"; lazy, so the following words decide where it ends
END = r"(?=[,.;:]|$| and\b| which\b| with\b)"
NAME = r"(?!(?:nobody|no one|noone|none|someone|somebody|anyone|anybody|everyone|everybody|nothing)\b)[a-z]+"
NOT_PEOPLE = {"it", "he", "she", "they", "this", "that", "there", "nobody", "everyone", "someone", "the", "one"}
EPS = 1e-6


def low(s):
    return re.sub(r"\s+", " ", s.lower().replace("’", "'")).strip()


def esc(s):
    return re.escape(s)


def tm(s):
    """Minutes since midnight for one clock time such as '2:45 pm'."""
    return T.parse_time(s)


# ---------------------------------------------------------------- claims

FILLER_PRE = re.compile(
    r"^(?:(?:honestly|for the record|okay so|ok so|okay|so|well|to be honest|to be clear|frankly|actually|"
    r"as it happens|by my count|i can tell you(?: that)?|i say(?: that)?|i can confirm(?: that)?)\s+)+")
FILLER_SUF = re.compile(r"(?:\s+(?:for the record|honestly|as it happens|to be honest|to be clear))+$")
OTHER_DAY = re.compile(r"\b(?:yesterday|tomorrow|day before|next day|previous day|following day|other day|"
                       r"last week|next week|last night)\b")


def claim(text):
    """(minutes, clause) for a message that states one thing at one clock time.

    The clause is lower case with the time phrase ('at 2:45 PM', 'Come 9:15 AM', 'It was 3:10 PM when'), commas and
    discourse fillers ('Honestly,', ', for the record') removed. None for questions, messages with no time or several
    times, dates or other days, and times after any word but 'at' (before, after, around, by ...)."""
    s = text.strip().replace("’", "'")
    if "?" in s or T.find_dates(s) or T.find_weekdays(s):
        return None
    times = T.find_times(s)
    if len(times) != 1:
        return None
    minutes, m = times[0]
    before, after = s[:m.start()], s[m.end():]
    if re.search(r"\b(?:at|come)\s*$", before, re.I):
        before = re.sub(r"\b(?:at|come)\s*$", "", before, flags=re.I)
    elif re.fullmatch(r"\s*it was\s*", before, re.I) and re.match(r"\s*,?\s*when\b", after, re.I):
        before, after = "", re.sub(r"^\s*,?\s*when\b", "", after, flags=re.I)
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


def window_verdict(t, t1, t2, name, inside_why, fam):
    """LIE strictly inside (t1, t2), None on either edge, OK outside."""
    if t1 + EPS < t < t2 - EPS:
        return Verdict(LIE, fam, inside_why)
    if abs(t - t1) < EPS or abs(t - t2) < EPS:
        return None
    return Verdict(OK, fam, f"outside {name} {T.fmt_time(t1)}-{T.fmt_time(t2)}")


# verbs shared by several families
TAKE_PAST = (r"(?:took|pulled|removed|lifted|got|grabbed|carried|moved|slid|fetched|brought|picked|"
             r"took out|pulled out|got out|lifted out|slid out)")
TAKE_BASE = (r"(?:take|pull|remove|lift|get|grab|carry|move|slide|fetch|bring|pick|taking|pulling|removing|"
             r"lifting|getting|grabbing|carrying|moving|sliding|fetching|bringing|take out|pull out)")
TAKE_ING = r"(?:taking|pulling|removing|lifting|getting|grabbing|carrying|moving|sliding|fetching|bringing)"
OUT_OF = r"(?:out of|from|out from|off)"
SAW = r"(?:saw|watched|noticed|spotted|observed|witnessed|caught)"


def removal_res(thing, place):
    """Patterns for 'somebody took THING out of PLACE', said by the actor or by a witness."""
    th, pl = esc(thing), esc(place)
    return [
        re.compile(rf"(?:i|{NAME}) {TAKE_PAST} (?:out )?the {th} {OUT_OF} the {pl}(?: and i {SAW} (?:it|it happen|"
                   rf"him do it|her do it|them do it))?"),
        re.compile(rf"i {SAW} {NAME} {TAKE_BASE} (?:out )?the {th} {OUT_OF} the {pl}"),
        re.compile(rf"{NAME} {TAKE_ING} (?:out )?the {th} {OUT_OF} the {pl} is something i (?:saw|watched|witnessed)"),
        re.compile(rf"it was (?:me|{NAME}) who {TAKE_PAST} (?:out )?the {th} {OUT_OF} the {pl}"),
    ]


def any_full(res, s):
    return any(r.fullmatch(s) for r in res)


# ---------------------------------------------------------------- move_route

DEPART = (r"(?:left|set off from|setting off from|set out from|pulled out of|pulled out from|pulled away from|"
          r"departed from|departed|leaving|departing from|departing|drove off from|drove out of|started from)")
GO = r"(?:went|drove|travelled|traveled|headed|ran|sailed|rode|moved|continued|going|driving|heading)"
DEST_END = r"(?=[:,;.]|$| along\b| covering\b| first\b| never\b| and\b| at\b| without\b)"
ROUTE_A = re.compile(rf"^(?:at {TM},? )?the ({NP}) {DEPART} the ({NP})(?: at {TM})?,? (?:and |then |and then )?"
                     rf"{GO} straight(?: along the [\d.]+ km route)? to the ({NP}){DEST_END}")
ROUTE_B = re.compile(rf"^(?:leaving|setting off from|departing from|departing|pulling out of) the ({NP}) at {TM},? "
                     rf"the ({NP}) {GO} straight(?: along the [\d.]+ km route)? to the ({NP}){DEST_END}")
MAX_SPEED = re.compile(
    r"never (?:went|going|goes|got|getting|travelled|travelling|traveling) (?:above|over|faster than|beyond)"
    r"|never above|never exceed|without (?:ever )?(?:going|getting) (?:above|over|faster than)"
    r"|(?:did not|didn't) (?:go|travel|get) (?:above|over|faster than)|(?:did not|didn't) exceed"
    r"|no faster than|at no point did its speed (?:go|get) (?:above|over)|top speed|maximum speed|speed limit")
EXACT_SPEED = re.compile(r"\b(?:average|averaged|exactly|constant|steady|steadily|precisely)\b")
DELAY = re.compile(r"\b(?:stopp?ed|stops|waited|paused|break|delay|delayed|detour|broke down|refuel)")
ARRIVE = (r"(?:reached|got to|arrived at|arrived in|made it to|pulled up at|pulled up to|pulled into|pulled in at|"
          r"pulled in to|came into|rolled into|rolled up at|drew up at|docked at|turned up at|showed up at)")
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


@register
class MoveRoute(Family):
    name = "move_route"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        out = []
        for i, s in enumerate(sents):
            m = ROUTE_A.match(s)
            if m:
                veh, _orig, dest = m.group(2), m.group(3), m.group(5)
                t0s = [x for x in (m.group(1), m.group(4)) if x]
            else:
                m = ROUTE_B.match(s)
                if not m:
                    continue
                _orig, veh, dest = m.group(1), m.group(3), m.group(4)
                t0s = [m.group(2)]
            if len(t0s) != 1 or len(T.find_times(s)) != 1:
                continue
            t0 = tm(t0s[0])
            used, parts = [i], [s[s.index("straight"):]]
            head = rf"(?:its|it|the {esc(veh)}'s|the {esc(veh)}|for the|on the|over the|the first|first|then|after that|at no point)\b"
            for j in range(i + 1, len(sents)):
                if "km" in sents[j] and re.match(head, sents[j]) and not re.search(DEPART, sents[j]):
                    used.append(j)
                    parts.append(sents[j])
                else:
                    break
            text = " ".join(parts)
            speed_info = "km/h" in text or "per hour" in text or "mph" in text or "mile" in text
            earliest = None
            if speed_info and MAX_SPEED.search(text) and not EXACT_SPEED.search(text) and "mile" not in text \
                    and "mph" not in text:
                mins = route_legs(text)
                earliest = None if mins is None else t0 + mins
            if speed_info and earliest is None:
                continue  # speeds we could not read: claims stay unjudged
            for j in used:
                ctx.consume(j)
            v, d = esc(veh), esc(dest)
            head_noun = esc(veh.split()[-1])
            out.append({
                "veh": veh, "t0": t0, "earliest": earliest, "ok": not DELAY.search(text),
                "arrive": [re.compile(rf"(?:the )?{v} {ARRIVE} the {d}"),
                           re.compile(rf"i (?:saw|spotted|noticed) the {v} (?:at|in) the {d}"),
                           re.compile(rf"the {v} was (?:at|in) the {d}")],
                "way": [re.compile(rf"i (?:saw|spotted|noticed|passed|caught sight of|glimpsed|watched) the {v} "
                                   rf"{ON_WAY}(?: to the {d})?"),
                        re.compile(rf"{ON_WAY} i (?:saw|spotted|noticed|passed) the {v}")],
                "other": [re.compile(rf"(?:another|some other|a different|some different|one other|a second) "
                                     rf"(?:vehicle|{head_noun}|[a-z]+ {head_noun}) {ARRIVE} the {d}")],
            })
        return out or None

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        if not c:
            return None
        t, cl = c
        for f in facts:
            if any_full(f["arrive"], cl):
                if t < f["t0"]:
                    return None  # an arrival before it set off could be an earlier trip
                e = f["earliest"]
                if e is not None and t < e - EPS:
                    return Verdict(LIE, self.name, f"{f['veh']} left at {T.fmt_time(f['t0'])}; earliest arrival "
                                                   f"{T.fmt_time(e)}")
                if f["ok"] and t > f["t0"]:
                    return Verdict(OK, self.name, f"arrival after the earliest possible time")
                return None
            if any_full(f["way"], cl):
                return Verdict(OK, self.name, "seen on the way after it left") if t >= f["t0"] else None
            if any_full(f["other"], cl):
                return Verdict(OK, self.name, "another vehicle")
        return None


# ---------------------------------------------------------------- move_away

AWAY_HEAD = re.compile(rf"^(?:at {TM},? )?the ({NP}) {DEPART} the ({NP})(?: at {TM})?(?=[,.;]|$| and\b)")
AWAY_RET = re.compile(rf"(?:did not|didn't|was not|wasn't|could not be|couldn't be) (?:return|come back|get back|"
                      rf"arrive back|back)(?: (?:at|to) the ({NP}))? (?:before|until) {TM}")
ROUND = re.compile(rf"round trip(?: of the ({NP}))? took at least ([^,;.]+)")
ROUND_BACK = re.compile(rf"could not be back(?: at the ({NP}))? before then")
BOARD = (r"(?:got on|got onto|got on board|got aboard|got into|boarded|climbed onto|climbed on|climbed aboard|"
         r"climbed into|stepped onto|stepped on|stepped aboard|stepped into|hopped on|hopped onto|hopped aboard|"
         r"hopped into|went aboard|went on board|jumped on|jumped onto|jumped aboard|caught|embarked on)")
BOARD_BASE = (r"(?:get on|get onto|get on board|get aboard|get into|board|climb onto|climb on|climb aboard|"
              r"climb into|step onto|step on|step aboard|step into|hop on|hop onto|hop aboard|go aboard|go on board|"
              r"jump on|jump onto|jump aboard|catch|embark on|getting on|getting onto|boarding|climbing onto|"
              r"climbing on|climbing aboard|stepping onto|stepping aboard|hopping on|hopping onto|going aboard)")


@register
class MoveAway(Family):
    name = "move_away"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        out = []
        for i, s in enumerate(sents):
            if "straight" in s:
                continue
            m = AWAY_HEAD.match(s)
            if not m:
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
                r = AWAY_RET.search(sj)
                if r and (r.group(1) in (None, stop)):
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
                if re.match(rf"nobody could (?:get on|board) the {v} at the {st} unless (?:the {v}|it) was at the "
                            rf"{st}\.?$", sj) or re.search(rf", and nobody could (?:get on|board) the {v} at the {st} "
                                                           rf"unless (?:the {v}|it) was at the {st}\.?$", sj):
                    ctx.consume(j)
            out.append({
                "veh": veh, "stop": stop, "t0": t0, "t1": t1,
                "board": [re.compile(rf"i {BOARD} the {v} at the {st}"),
                          re.compile(rf"at the {st} i {BOARD} the {v}"),
                          re.compile(rf"i {SAW} {NAME} {BOARD_BASE} the {v} at the {st}"),
                          re.compile(rf"{NAME} {BOARD} the {v} at the {st}(?: and i {SAW} (?:it|him|her|them))?")],
                "wait": [re.compile(rf"i (?:was )?(?:standing )?(?:waiting )?(?:at|by|near) the {st}(?: waiting)?"),
                         re.compile(rf"i stood (?:waiting )?at the {st}")],
            })
        return out or None

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        if not c:
            return None
        t, cl = c
        for f in facts:
            if any_full(f["board"], cl):
                if f["t0"] + EPS < t < f["t1"] - EPS:
                    return Verdict(LIE, self.name, f"{f['veh']} away from the {f['stop']} from "
                                                   f"{T.fmt_time(f['t0'])} until at least {T.fmt_time(f['t1'])}")
                if abs(t - f["t0"]) < EPS:
                    return None
                return Verdict(OK, self.name, "boarding outside the time it was away")
            if any_full(f["wait"], cl):
                return Verdict(OK, self.name, "waiting at the stop")
        return None


# ---------------------------------------------------------------- move_place

PERSON_AT = re.compile(rf"^(?:at {TM},? )?([a-z]+) was at the ({NP})(?: at {TM})?(?=[,.;]|$| and\b)")
ROUTE_MIN = re.compile(rf"every route from the ({NP}) to the ({NP}) (?:also )?takes at least ([^,;.]+)")
ROUTE_VIA = re.compile(rf"every route from the ({NP}) to the ({NP}) (?:passes|goes|runs) through the ({NP})"
                       rf"(?=[,.;]|$| and\b)")
SEPARATE = re.compile(rf"the ({NP})(?:, which [^,]*,)?(?:, the ({NP}))?,? and the ({NP})(?:, which [^,]*,)? are "
                      rf"separate (?:locations|places|sites|buildings|rooms|areas|spaces)")
AT_PLACE = r"(?:at|over at|out at|down at|up at|in|inside|already at)"
MEET = (r"(?:saw|spotted|noticed|caught sight of|met|ran into|bumped into|talked to|spoke to|spoke with|"
        r"chatted with|greeted)")


def presence_res(person, place, prep=AT_PLACE):
    """Patterns placing PERSON at PLACE: 'I saw Isla at the stadium', 'Isla was at the stadium when I saw her',
    'Over at the stadium, I saw Isla'. Self claims ('I was at the stadium') are separate."""
    p, pl = esc(person), esc(place)
    return [
        re.compile(rf"i {MEET} {p} {prep} the {pl}"),
        re.compile(rf"{p} was {prep} the {pl}(?: when i (?:saw|spotted|noticed|met|ran into) (?:him|her|them))?"),
        re.compile(rf"(?:over )?(?:at|in) the {pl} i (?:saw|spotted|noticed|met) {p}"),
    ]


def self_res(place, prep=AT_PLACE):
    pl = esc(place)
    return [re.compile(rf"i was {prep} the {pl}"), re.compile(rf"the {pl} is where i was"),
            re.compile(rf"i (?:got to|reached|arrived at|made it to) the {pl}")]


@register
class MovePlace(Family):
    name = "move_place"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        edges, via, used = {}, {}, set()
        for i, s in enumerate(sents):
            for r in ROUTE_MIN.finditer(s):
                d = T.find_durations(r.group(3))
                if len(d) == 1:
                    edges[(r.group(1), r.group(2))] = d[0][0]
                    used.add(i)
            for r in ROUTE_VIA.finditer(s):
                via[(r.group(1), r.group(2))] = r.group(3)
                used.add(i)
        if not edges:
            return None
        out = []
        for i, s in enumerate(sents):
            m = PERSON_AT.match(s)
            if not m or m.group(2) in NOT_PEOPLE:
                continue
            person, start = m.group(2), m.group(3)
            if person in NOT_PEOPLE or not re.search(rf"\b{esc(person.capitalize())}\b", ctx.sentences[i]):
                continue
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
                        "places": {pl: (presence_res(person, pl), self_res(pl)) for pl in list(bounds) + [start]}})
        if not out:
            return None
        for i in used:
            ctx.consume(i)
        for i, s in enumerate(sents):
            if SEPARATE.search(s) and not re.search(r"\d", s):
                ctx.consume(i)
        return out

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        if not c:
            return None
        t, cl = c
        sender = ctx.speakers.get(msg.label, "").lower()
        if re.fullmatch(r"i was (?:out )?on the road", cl):
            return Verdict(OK, self.name, "on the road")
        for f in facts:
            for pl, (others, selfs) in f["places"].items():
                about = None
                if any_full(others, cl):
                    about = f["person"]
                elif any_full(selfs, cl):
                    about = sender
                if about is None:
                    continue
                if about != f["person"] or pl == f["start"]:
                    return Verdict(OK, self.name, "no travel limit applies")
                d, t0 = f["bounds"][pl], f["t0"]
                if t0 - EPS <= t < t0 + d - EPS:
                    return Verdict(LIE, self.name, f"{f['person']} was at the {f['start']} at {T.fmt_time(t0)}; "
                                                   f"the {pl} is at least {d} minutes away")
                if t >= t0 + d - EPS or t <= t0 - d:
                    return Verdict(OK, self.name, "enough time to travel")
                return None  # shortly before: the evidence gives no limit for the way back
        return None


# ---------------------------------------------------------------- move_room

STAY_A = re.compile(rf"^([a-z]+) stayed in the ({NP}) from {TM} (?:to|until|till) {TM}(?=[,.;]|$| and\b)")
STAY_B = re.compile(rf"^from {TM} (?:to|until|till) {TM},? ([a-z]+) stayed in the ({NP})(?=[,.;]|$| and\b)")
IN_ROOM = r"(?:in|inside|down in|over in|up in|at)"


def separate_groups(sents):
    groups = []
    for s in sents:
        for m in SEPARATE.finditer(s):
            groups.append({g for g in m.groups() if g})
    return groups


@register
class MoveRoom(Family):
    name = "move_room"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        groups = separate_groups(sents)
        out = []
        for i, s in enumerate(sents):
            m = STAY_A.match(s)
            if m:
                person, room, t1, t2 = m.groups()
            else:
                m = STAY_B.match(s)
                if not m:
                    continue
                t1, t2, person, room = m.groups()
            if person in NOT_PEOPLE or not re.search(rf"\b{esc(person.capitalize())}\b", ctx.sentences[i]):
                continue
            others = set().union(*[g - {room} for g in groups if room in g]) if groups else set()
            if not others:
                continue
            ctx.consume(i)
            out.append({"person": person, "room": room, "t1": tm(t1), "t2": tm(t2),
                        "rooms": {r: (presence_res(person, r, IN_ROOM), self_res(r, IN_ROOM))
                                  for r in others | {room}}})
        if not out:
            return None
        for i, s in enumerate(sents):
            if SEPARATE.search(s) and not re.search(r"\d", s):
                ctx.consume(i)
        return out

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        if not c:
            return None
        t, cl = c
        sender = ctx.speakers.get(msg.label, "").lower()
        for f in facts:
            for r, (others, selfs) in f["rooms"].items():
                about = f["person"] if any_full(others, cl) else sender if any_full(selfs, cl) else None
                if about is None:
                    continue
                if about != f["person"] or r == f["room"]:
                    return Verdict(OK, self.name, "not a different room for the person who stayed")
                return window_verdict(t, f["t1"], f["t2"], f"{f['person']} in the {f['room']}",
                                      f"{f['person']} stayed in the {f['room']} from {T.fmt_time(f['t1'])} to "
                                      f"{T.fmt_time(f['t2'])}", self.name)
        return None


# ---------------------------------------------------------------- move_object

OBJ_INTRO = re.compile(rf"^there was (?:one|a single|only one|exactly one) ({NP}){END}")
OBJ_STAY = re.compile(rf"(?:^|, and |, which |; )(it|which|the {NP})? ?stayed in the ({NP})(?:, which [^,]*,)? from "
                      rf"{TM} (?:to|until|till) {TM}(?=[,.;]|$| and\b)")
OBJ_STAY_B = re.compile(rf"^from {TM} (?:to|until|till) {TM},? (it|the {NP}) stayed in the ({NP})(?=[,.;]|$)")
HAVE_AT = r"(?:at|in|on|inside)"


@register
class MoveObject(Family):
    name = "move_object"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        groups = separate_groups(sents)
        out, last = [], None
        for i, s in enumerate(sents):
            m = OBJ_INTRO.match(s)
            if m:
                last = m.group(1)
            m = OBJ_STAY.search(s)
            subj = None
            if m:
                subj, place, t1, t2 = m.groups()
            else:
                m = OBJ_STAY_B.match(s)
                if m:
                    t1, t2, subj, place = m.groups()
            if not m or last is None:
                continue
            if subj and subj.startswith("the ") and subj[4:] != last:
                continue
            if not subj and not OBJ_INTRO.match(s):
                continue
            obj = last
            others = set().union(*[g - {place} for g in groups if place in g]) if groups else set()
            ctx.consume(i)
            o, pl = esc(obj), esc(place)
            elsewhere = []
            for other in others:
                ot = esc(other)
                elsewhere += [
                    re.compile(rf"i (?:had|was holding|held|was carrying|carried|used|was using) the {o}(?: with me| "
                               rf"along with me| on me)? {HAVE_AT} the {ot}"),
                    re.compile(rf"{HAVE_AT} the {ot} i (?:had|was holding|was carrying) the {o}(?: with me)?"),
                    re.compile(rf"the {o} was with me {HAVE_AT} the {ot}"),
                    re.compile(rf"i (?:saw|spotted|noticed) the {o} {HAVE_AT} the {ot}"),
                    re.compile(rf"i (?:saw|spotted|noticed) {NAME} (?:with|holding|carrying|using) the {o} "
                               rf"{HAVE_AT} the {ot}"),
                ]
            fine = [re.compile(rf"i (?:saw|spotted|noticed) the {o} (?:in|inside) the {pl}"),
                    re.compile(rf"the {o} was (?:in|inside) the {pl}(?: when i (?:saw|checked|spotted) it)?"),
                    re.compile(rf"i was (?:standing |right )?(?:next to|beside|near|by|right beside|right next to) "
                               rf"the {pl}")]
            for other in others:
                fine.append(re.compile(rf"i was (?:at|in|over at|on|inside|down at) the {esc(other)}"))
            out.append({"obj": obj, "place": place, "t1": tm(t1), "t2": tm(t2),
                        "bad": removal_res(obj, place) + elsewhere, "fine": fine})
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
            if SEPARATE.search(s) and not re.search(r"\d", s) and any(p in s for p in places):
                ctx.consume(i)
            if re.fullmatch(rf"the (?:{'|'.join(esc(x) for x in objs | places)}) (?:has|is|was) [a-z ]+\.?", s):
                ctx.consume(i)  # decoration such as 'The red key has a white label.'
        return out

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        if not c:
            return None
        t, cl = c
        for f in facts:
            if any_full(f["bad"], cl):
                return window_verdict(t, f["t1"], f["t2"], f"the {f['obj']} in the {f['place']}",
                                      f"the {f['obj']} stayed in the {f['place']} from {T.fmt_time(f['t1'])} to "
                                      f"{T.fmt_time(f['t2'])}", self.name)
            if any_full(f["fine"], cl):
                return Verdict(OK, self.name, "consistent with the object staying put")
        return None


# ---------------------------------------------------------------- move_seal

SEAL = re.compile(rf"(?:^|, and |; |, )(?:from {TM} (?:to|until|till) {TM},? )?(it|(?:the|that|this) {NP}) stayed "
                  rf"sealed(?: from {TM} (?:to|until|till) {TM})?")
SEAL_ITEM = re.compile(rf"the ({NP}) could not be (?:removed|taken out|taken)(?: from it)? without breaking the seal")
HELD_IN = [re.compile(rf"(?:^|, )(?:one|a single|a) ({NP}) held the ({NP}){END}"),
           re.compile(rf"^the ({NP}) was (?:kept |held |stored |placed |sealed )?(?:in|inside) (?:one|a single|a) "
                      rf"({NP})(?=[,.;]|$| and\b| which\b| that\b)")]
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
                    if re.fullmatch(rf"(?:one|a single|a) {NP} held the {NP}(?: and was kept on a shelf)?\.?", s) or \
                            re.fullmatch(rf"the {NP} was (?:kept |held |stored |placed )?(?:in|inside) (?:one|a single|"
                                         rf"a) {NP}(?:,? (?:which|that) was (?:kept on a shelf|labelled in [a-z]+ "
                                         rf"ink))?\.?", s):
                        ctx.consume(i)
            m = SEAL.search(s)
            it = SEAL_ITEM.search(s)
            if not m or not it:
                continue
            times = [x for x in m.groups()[:2] + m.groups()[3:] if x]
            if len(times) != 2:
                continue
            subj, item = m.group(3), it.group(1)
            cont = None
            if subj == "it":
                cont = known[-1][0] if known else None
            else:
                word = subj.split(" ", 1)[1]
                if subj.startswith("the "):
                    cont = word
                else:  # 'that bag', 'that envelope': the container named earlier
                    cand = [c for c, _ in known if c == word or c.endswith(" " + word)]
                    cont = cand[-1] if cand else None
            if cont is None:
                continue
            ctx.consume(i)
            c, itm = esc(cont), esc(item)
            out.append({
                "cont": cont, "item": item, "t1": tm(times[0]), "t2": tm(times[1]),
                "bad": removal_res(item, cont) + [
                    re.compile(rf"(?:i|{NAME}) {OPEN} the {c}"),
                    re.compile(rf"(?:i|{NAME}) broke the seal (?:on|of) the {c}"),
                    re.compile(rf"i {SAW} {NAME} (?:open|unseal|break open|tear open|cut open|opening) the {c}")],
                "fine": [re.compile(rf"i (?:was holding|held|had|was carrying|carried|picked up|had hold of) the {c}"),
                         re.compile(rf"when i (?:saw|looked at|checked|inspected|examined) the {c} it was still sealed"),
                         re.compile(rf"the {c} was still sealed when i (?:saw|looked at|checked|inspected|examined) it"),
                         re.compile(rf"i saw the {itm} in the {c}")],
            })
        if not out:
            return None
        conts = {f["cont"] for f in out}
        for i, s in enumerate(sents):
            if re.fullmatch(rf"the (?:{'|'.join(esc(x) for x in conts)}) (?:was|is|has) [a-z ]+\.?", s):
                ctx.consume(i)  # decoration such as 'The sealed envelope was labelled in black ink.'
        return out

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        if not c:
            return None
        t, cl = c
        for f in facts:
            if any_full(f["bad"], cl):
                return window_verdict(t, f["t1"], f["t2"], f"the {f['cont']} sealed",
                                      f"the {f['cont']} stayed sealed from {T.fmt_time(f['t1'])} to "
                                      f"{T.fmt_time(f['t2'])}", self.name)
            if any_full(f["fine"], cl):
                return Verdict(OK, self.name, "consistent with the seal")
        return None


# ---------------------------------------------------------------- move_chain

CHAIN_SPLIT = re.compile(r",? and then |, then |; |,? before |, who |, and |\. ")
HOLD = r"(?:held|kept|had|holding|keeping)"


def parse_chain(text):
    """[(holder, start, end)] from 'Aiko held it from 10:50 AM to 11:50 AM, then handed it to Sofia. Sofia held it
    until 1:20 PM ...' and 'From 10:45 AM Fatima held it for exactly 1 hour, then handed it to Isla.'; None when a
    clause does not fit."""
    segs, pending = [], None
    for cl in CHAIN_SPLIT.split(text.strip(" .")):
        cl = cl.strip(" ,.")
        if not cl:
            continue
        m = re.fullmatch(r"(?:then )?hand(?:ed|ing) it (?:over )?to ([a-z]+)", cl)
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
        if who is None or who in NOT_PEOPLE or (pending and who != pending):
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
            m = re.fullmatch(rf"there was (?:one|a single|only one|exactly one) ({NP})\.?", s)
            if not m:
                continue
            obj = m.group(1)
            o = esc(obj)
            chain_idx, rest_idx, window, exclusive = [], [], None, False
            for j in range(i + 1, len(sents)):
                sj = sents[j]
                w = re.search(rf"(?:nobody else held (?:the {o}|it) between {TM} and {TM}|between {TM} and {TM},? "
                              rf"nobody else held (?:the {o}|it))", sj)
                ex = re.search(rf"only one person held (?:the {o}|it) at any moment", sj)
                if w or ex:
                    if w:
                        g = [x for x in w.groups() if x]
                        window = (tm(g[0]), tm(g[1]))
                    exclusive = exclusive or bool(ex)
                    rest_idx.append(j)
                    continue
                if re.search(rf"\b{HOLD} it\b|hand(?:ed|ing) it", sj) and not rest_idx:
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
            out.append({"obj": obj, "segs": segs, "exclusive": exclusive, "nobody_else": window is not None,
                        "names": {h for h, _, _ in segs},
                        "self": re.compile(rf"(?:i (?:had|held|was holding|was carrying|had hold of|was minding|kept)"
                                           rf" the {o}(?: with me| in my hands?)?|the {o} was (?:with me|in my hands?|"
                                           rf"on me))"),
                        "other": re.compile(rf"(?:i {SAW} ({NAME}) (?:holding|with|carrying) the {o}|({NAME}) (?:had|"
                                            rf"held|was holding) the {o})"),
                        "give": re.compile(rf"i (?:gave|handed|passed) (?:the {o} to ({NAME})|({NAME}) the {o})")})
        return out or None

    def holder(self, f, t):
        """The holder strictly inside a segment, 'edge' on a hand-over or window edge, None outside the chain."""
        for h, a, b in f["segs"]:
            if a + EPS < t < b - EPS:
                return h
            if abs(t - a) < EPS or abs(t - b) < EPS:
                return "edge"
        return None

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        if not c:
            return None
        t, cl = c
        sender = ctx.speakers.get(msg.label, "").lower()
        for f in facts:
            who = None
            if f["self"].fullmatch(cl):
                who = sender
            else:
                m = f["other"].fullmatch(cl)
                if m:
                    who = m.group(1) or m.group(2)
            start, end = f["segs"][0][1], f["segs"][-1][2]
            if who:
                if who == "me":
                    who = sender
                if t < start - EPS or t > end + EPS:
                    return Verdict(OK, self.name, f"outside the hand-over chain of the {f['obj']}")
                h = self.holder(f, t)
                if h is None or h == "edge":
                    return None
                if h == who:
                    return Verdict(OK, self.name, f"{who} held the {f['obj']} then")
                if f["exclusive"] or (f["nobody_else"] and who not in f["names"]):
                    return Verdict(LIE, self.name, f"{h} held the {f['obj']} at {T.fmt_time(t)}")
                return None
            m = f["give"].fullmatch(cl)
            if m:
                to = m.group(1) or m.group(2)
                if t < start - EPS or t > end + EPS:
                    return Verdict(OK, self.name, "hand-over outside the chain")
                for (h1, _, b1), (h2, _, _) in zip(f["segs"], f["segs"][1:]):
                    if abs(t - b1) < EPS and h1 == sender and h2 == to:
                        return Verdict(OK, self.name, "the recorded hand-over")
                h = self.holder(f, t)
                if h not in (None, "edge") and (f["exclusive"] or (f["nobody_else"] and to not in f["names"])):
                    return Verdict(LIE, self.name, f"{h} held the {f['obj']} at {T.fmt_time(t)}")
                return None
        return None


# ---------------------------------------------------------------- move_count

COUNT = re.compile(rf"^(?:from {TM} (?:to|until|till) {TM},? )?(?:the ({NP})|it) held exactly ([a-z0-9]+) ({NP})"
                   rf"(?: from {TM} (?:to|until|till) {TM})?(?=[,.;]|$| and\b)")
NO_CHANGE = re.compile(rf"no ({NP}) (?:was|were) added (?:to it |to the {NP} )?or removed(?: from it)? during that time")
NUM_WORD = r"(\d+|no|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"


@register
class MoveCount(Family):
    name = "move_count"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        out, last_subject = [], None
        for i, s in enumerate(sents):
            m = COUNT.match(s)
            if not m:
                sm = re.match(rf"^the ({NP}) (?:stands|stood|sits|sat|is|was) ", s)
                last_subject = sm.group(1) if sm else last_subject
                continue
            g = m.groups()
            cont = g[2] or last_subject
            times = [x for x in g[:2] + g[5:] if x]
            n = T.word_num(g[3])
            items = g[4]
            if cont is None or len(times) != 2 or n is None or not items.endswith("s"):
                continue
            nc = NO_CHANGE.search(s)
            nc_idx = None
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
            it, ones, c = esc(items), esc(one), esc(cont)
            things = rf"(?:{it}|{ones})"
            out.append({
                "cont": cont, "n": n, "t1": tm(times[0]), "t2": tm(times[1]), "no_change": item is not None,
                "count": [re.compile(rf"there (?:were|was) {NUM_WORD} {things} in the {c}"),
                          re.compile(rf"the {c} (?:held|had|contained|was holding|housed) {NUM_WORD} {things}(?: in it)?"),
                          re.compile(rf"{NUM_WORD} {things} (?:were|was) in the {c}"),
                          re.compile(rf"i (?:counted|saw|photographed|checked) (?:all )?{NUM_WORD} {things} in the {c}")],
                "change": [re.compile(rf"(?:i|{NAME}) {TAKE_PAST} (?:out )?(?:one of the {it}|one {ones}|an? {ones}|"
                                      rf"a single {ones}) {OUT_OF} the {c}"),
                           re.compile(rf"i {SAW} {NAME} {TAKE_BASE} (?:out )?(?:one of the {it}|one {ones}|an? {ones}) "
                                      rf"{OUT_OF} the {c}"),
                           re.compile(rf"(?:i|{NAME}) (?:added|put|placed|slipped|dropped) (?:another|an?|one|an extra) "
                                      rf"{ones} (?:in|into|to) the {c}")],
                "near": [re.compile(rf"i was (?:standing )?(?:beside|next to|near|by) the {c}")],
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
        if not c:
            return None
        t, cl = c
        for f in facts:
            for r in f["count"]:
                m = r.fullmatch(cl)
                if m:
                    k = T.word_num(m.group(1))
                    if k is None:
                        return None
                    if f["t1"] + EPS < t < f["t2"] - EPS and k != f["n"]:
                        return Verdict(LIE, self.name, f"the {f['cont']} held exactly {f['n']} from "
                                                       f"{T.fmt_time(f['t1'])} to {T.fmt_time(f['t2'])}")
                    if (f["t1"] - EPS <= t <= f["t2"] + EPS) and k != f["n"]:
                        return None
                    return Verdict(OK, self.name, "count consistent with the evidence")
            if any_full(f["change"], cl):
                if not f["no_change"]:
                    return None
                return window_verdict(t, f["t1"], f["t2"], f"the {f['cont']} unchanged",
                                      f"nothing was added to or removed from the {f['cont']} from "
                                      f"{T.fmt_time(f['t1'])} to {T.fmt_time(f['t2'])}", self.name)
            if any_full(f["near"], cl):
                return Verdict(OK, self.name, "beside the container")
        return None
