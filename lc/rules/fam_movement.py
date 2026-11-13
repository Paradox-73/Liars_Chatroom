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

Evidence is read only through whole-sentence grammars: a sentence with any clause the grammar does not know (an
exception, a second window, a condition) is not read. A fact is dropped when another sentence the grammars cannot
read names one of its nouns or people, or follows it starting with a pronoun, since that sentence may change it.

Claims at the exact edge of a window are left to the fallback judge, since "left at 2:15 PM" and "stayed until
1:20 PM" read both ways. Each judge returns LIE if any parsed fact rules the claim out, else OK if one allows it.
"""
import re

from .core import LIE, OK, Family, Verdict, register
from . import text as T

# a noun phrase after "the": lazy, so the following words decide where it ends, and never a clause word, so
# 'the white packet held the signed form and was kept on a shelf' cannot be read as one long noun
NP_WORD = (r"(?!(?:and|or|but|was|were|is|are|be|been|held|stayed|remained|left|had|has|have|could|would|did|does|"
           r"not|never|from|to|until|till|between|at|on|in|during|for|with|without|after|before|which|that|who|whom|"
           r"then|so|its|it|when|while|unless|except|if|only|every|each|daily|weekdays|weekends)\b)[a-z0-9][a-z0-9'-]*")
NP = rf"(?:{NP_WORD}(?: {NP_WORD})*?)"
NAME = (r"(?!(?:nobody|no one|noone|none|someone|somebody|anyone|anybody|everyone|everybody|nothing|not|never|who|"
        r"whom|whoever|what|which)\b)[a-z]+")
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


def dur(s):
    """Minutes in one duration such as '1 hour 20 minutes', or None."""
    d = T.find_durations(s)
    return d[0][0] if len(d) == 1 and d[0][0] > 0 else None


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
    r"^(?:(?:honestly|for the record|okay so|ok so|okay|so|well|to be honest|to be clear|to be fair|frankly|actually|"
    r"as it happens|by my count|i can tell you(?: that)?|i say(?: that)?|i can confirm(?: that)?|look|listen|anyway|"
    r"just so you know|for what it's worth)\s+)+")
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


# ---------------------------------------------------------------- whole-sentence evidence grammar
# Sentences are matched in lower case. Every grammar below must cover a sentence from its first word to its last.

TM_ = r"(?:\d{1,2}:\d{2} (?:am|pm)|noon|midday|midnight)"
DUR_ = r"(?:(?:\d+|an?|one|two|three|four|five|six) hours?(?: (?:and )?\d+ minutes?)?|\d+ minutes?)"
NUM_ = r"\d+(?:\.\d+)?"
COLOUR = r"(?:grey|gray|blue|red|green|white|black|yellow|brown|orange|silver|wooden|metal)"
DECOR_PRED = (rf"(?:has an? {COLOUR} (?:door|label|lid)|is (?:painted )?{COLOUR}|(?:is|sits|stands) on the [a-z]+ "
              rf"floor|seats [a-z]+ people|has seating for [a-z]+ people|(?:stands|sits) (?:by|beside|next to|near) "
              rf"the window|was labelled in {COLOUR} ink|was kept on a shelf)")
DECOR_SENTS = [re.compile(p) for p in (  # decoration that changes no time or place: 'The red key has a white label.'
    rf"^the (?P<d>{NP}) {DECOR_PRED}\.?$",
    rf"^an? {COLOUR} label is (?:stuck|fixed|attached) (?:on|to) the (?P<d>{NP})\.?$",
    rf"^{COLOUR} ink was used to label the (?P<d>{NP})\.?$",
    rf"^[a-z]+ people can sit in the (?P<d>{NP})\.?$")]
JOIN = re.compile(r", and |, while |; ")
BOUND = r"(?=$|, and |, while |; )"


def decor(s):
    """The noun a decoration sentence describes, or None."""
    m = next((m for m in (r.match(s) for r in DECOR_SENTS) if m), None)
    return m.group("d") if m else None


def clause(p):
    return re.compile(p + BOUND)


def split_clauses(s, kinds):
    """[(kind, match)] when the sentence is a chain of known clauses joined by ', and', ', while' or ';', else None."""
    s = s.rstrip(".")

    def go(pos):
        for kind, rx in kinds:
            m = rx.match(s, pos)
            if not m:
                continue
            if m.end() == len(s):
                return [(kind, m)]
            j = JOIN.match(s, m.end())
            rest = go(j.end()) if j else None
            if rest is not None:
                return [(kind, m)] + rest
        return None
    return go(0)


SEP_P = (rf"the (?P<s1>{NP})(?:, which {DECOR_PRED},)?(?:, the (?P<s2>{NP}))?,? and the (?P<s3>{NP})"
         rf"(?:, which {DECOR_PRED},)? are (?:two )?separate (?:locations|places|rooms|sites|buildings|areas)")
SEP_C = clause(SEP_P)
SEP_SENT = re.compile(rf"^{SEP_P}\.?$")
DECOR_C = clause(rf"the (?P<d>{NP}) {DECOR_PRED}")


def separate_groups(sents):
    """Sets of places named in 'The A and the B are separate locations' clauses of fully read sentences."""
    groups = []
    for s in sents:
        for kind, m in split_clauses(s, PLACE_KINDS) or split_clauses(s, ROOM_KINDS) or []:
            if kind == "sep":
                groups.append({g for g in m.group("s1", "s2", "s3") if g})
    return groups


def previous_fact(sents, i):
    """Index of the sentence before i, skipping decoration sentences ('The red key has a white label.')."""
    j = i - 1
    while j >= 0 and decor(sents[j]):
        j -= 1
    return j


def understood(ctx):
    """Indices of evidence sentences some grammar of this module reads in full (cached on ctx)."""
    got = getattr(ctx, "_move_read", None)
    if got is None:
        sents = [low(s) for s in ctx.sentences]
        got = {i for i, s in enumerate(sents) if s.startswith("all times in this case refer to") or reads(s)}
        ctx._move_read = got
    return got


def tainted(ctx, used, nouns, names=()):
    """True when an evidence sentence this module cannot read names one of the fact's nouns or people, or follows
    one of the fact's sentences and starts with a pronoun."""
    read = understood(ctx)
    for j, orig in enumerate(ctx.sentences):
        if j in read:
            continue
        s = low(orig)
        if any(re.search(rf"\b{esc(n)}\b", s) for n in nouns if n):
            return True
        if any(re.search(rf"\b{esc(n.capitalize())}\b", orig) for n in names if n):
            return True
        if j - 1 in used and re.match(r"(?:it|its|this|that|they|their|he|she|his|her|there)\b", s):
            return True
    return False


# ---------------------------------------------------------------- move_route

DEPART = (r"(?:left|set off from|set out from|pulled out of|pulled out from|pulled away from|departed from|"
          r"drove off from|drove out of)")
GO = r"(?:went|drove|travelled|traveled|headed|sailed)"
SPD = rf"{NUM_} km/h"
MAXP = (r"(?:where (?:it|its speed) never went (?:above|over|faster than)|(?:it|its speed) never went (?:above|over|"
        r"faster than)|never going (?:above|over|faster than)|without (?:ever )?going (?:above|over|faster than)|"
        r"at never above|never exceeding)")
LEAD = (r"(?:(?:and )?then |and )?(?:for |on |over |covering |first covering |it covered |it first covered |"
        r"it then covered )?(?:the )?(?:first |next |following |remaining |last )?")
LEG = rf"{LEAD}{NUM_} km(?: first)?,? {MAXP} {SPD}"
LEGS = rf"{LEG}(?:,? {LEG})+"
ONE_SPEED = rf"(?:its speed|its top speed|it) never went (?:above|over|faster than) {SPD}"
ROUTE_TO = (rf"{GO} (?:straight|directly) (?:along the {NUM_} km route )?to the (?P<dest>{NP})"
            rf"(?: along the {NUM_} km route)?")
ROUTE_TAIL = rf"(?:(?:,? and|;|,) {ONE_SPEED}|,? never going (?:above|over|faster than) {SPD}|(?::|,|, and) {LEGS})"
ROUTE_SENT = [
    re.compile(rf"^(?:at (?P<ta>{TM_}),? )?(?:the|a|an) (?P<veh>{NP}) {DEPART} the (?P<orig>{NP})(?: at (?P<tb>{TM_}))?"
               rf",? (?:and |and then |then )?{ROUTE_TO}(?:{ROUTE_TAIL})?\.?$"),
    re.compile(rf"^(?:leaving|setting off from|departing from|pulling out of) the (?P<orig>{NP}) at (?P<ta>{TM_}),? "
               rf"the (?P<veh>{NP}) {ROUTE_TO}(?:{ROUTE_TAIL})?\.?$")]
SPEED_SENT = re.compile(rf"^(?:{ONE_SPEED}|the (?P<veh>{NP})'s speed never went (?:above|over|faster than) {SPD}|"
                        rf"at no point did its speed go (?:above|over) {SPD}|it never exceeded {SPD}|{LEGS})\.?$")
ARRIVE = (r"(?:reached|got to|arrived at|arrived in|made it to|pulled up at|pulled up to|pulled into|pulled in at|"
          r"pulled in to|came into|rolled into|rolled up at|drew up at|docked at|turned up at|showed up at)")
ARRIVE_BASE = (r"(?:reach|get to|arrive at|arrive in|make it to|pull up at|pull into|pull in at|come into|roll into|"
               r"dock at|reaching|getting to|arriving at|arriving in|pulling up at|pulling into|pulling in at|"
               r"coming into|rolling into|docking at)")
ON_WAY = r"(?:on the way|on its way|en route|on the road|along the way|along the route|on the route|on its route)"


def route_legs(text):
    """Minimum travel minutes from 'N km' and 'N km/h' figures paired in order, or None when they do not pair."""
    kms = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*km\b(?!\s*/\s*h)", text)]
    sps = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*km/h", text)]
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
            m = next((m for m in (r.match(s) for r in ROUTE_SENT) if m), None)
            if not m:
                continue
            d = m.groupdict()
            veh, orig, dest = d["veh"], d["orig"], d["dest"]
            t0s = [x for x in (d.get("ta"), d.get("tb")) if x]
            if len(t0s) != 1 or orig == dest:
                continue
            t0, used, text = tm(t0s[0]), [i], s
            if "km/h" not in s and i + 1 < len(sents):
                sm = SPEED_SENT.match(sents[i + 1])
                if sm and sm.group("veh") in (None, veh):
                    used.append(i + 1)
                    text += " " + sents[i + 1]
            earliest = None
            if "km/h" in text:
                mins = route_legs(text)
                if mins is None:
                    continue  # speeds we could not pair: claims stay unjudged
                earliest = t0 + mins
            if tainted(ctx, used, {veh, orig, dest}):
                continue
            for j in used:
                ctx.consume(j)
            v, dd = esc(veh), esc(dest)
            noun = esc(veh.split()[-1])
            out.append({
                "veh": veh, "t0": t0, "earliest": earliest,
                "arrive": [re.compile(rf"(?:the )?{v} {ARRIVE} the {dd}"),
                           re.compile(rf"i {SAW} the {v} {ARRIVE_BASE} the {dd}"),
                           re.compile(rf"i (?:saw|spotted|noticed) the {v} (?:at|in) the {dd}"),
                           re.compile(rf"the {v} was (?:at|in) the {dd}")],
                "way": [re.compile(rf"i (?:saw|spotted|noticed|passed|caught sight of|glimpsed|watched) the {v} "
                                   rf"{ON_WAY}(?: to the {dd})?"),
                        re.compile(rf"{ON_WAY} i (?:saw|spotted|noticed|passed) the {v}")],
                "other": [re.compile(rf"(?:another|some other|a different|some different|one other|a second) "
                                     rf"(?:vehicle|{noun}|[a-z]+ {noun}) {ARRIVE} the {dd}")],
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
            if t > f["t0"]:
                return Verdict(OK, self.name, "arrival after the earliest possible time")
            return None
        if any_full(f["way"], cl):
            return Verdict(OK, self.name, "seen on the way after it left") if t >= f["t0"] else None
        if any_full(f["other"], cl):
            return Verdict(OK, self.name, "another vehicle")
        return None


# ---------------------------------------------------------------- move_away

AWAY_HEAD = rf"(?:at (?P<ta>{TM_}),? )?(?:the|a|an) (?P<veh>{NP}) {DEPART} the (?P<stop>{NP})(?: at (?P<tb>{TM_}))?"
AWAY_RET = (rf"(?:,? and (?:it )?|, it )(?:(?:did not|didn't) (?:return|come back|get back)"
            rf"(?: to the (?P<rstop>{NP}))? (?:before|until) (?P<t1>{TM_})|(?:was|stayed) away until (?P<t1b>{TM_}))")
AWAY_ROUND = (rf",? and its round trip took at least (?P<dur>{DUR_}),? so it could not be back at the (?P<stop2>{NP}) "
              rf"before then")
AWAY_NOBODY = (rf",? and nobody could (?:get on|board) the (?P<nveh>{NP}) at the (?P<nstop>{NP}) unless (?:the "
               rf"(?P<nveh2>{NP})|it) was at the (?P<nstop2>{NP})")
AWAY_SENT = re.compile(rf"^{AWAY_HEAD}(?:{AWAY_RET}|{AWAY_ROUND})?(?:{AWAY_NOBODY})?\.?$")
ROUND_SENT = re.compile(rf"^(?:its round trip|the round trip of the (?P<veh>{NP})) took at least (?P<dur>{DUR_}),? so "
                        rf"it could not be back at the (?P<stop>{NP}) before then\.?$")
RET_SENT = re.compile(rf"^it (?:did not|didn't) (?:return|come back|get back)(?: to the (?P<stop>{NP}))? "
                      rf"(?:before|until) (?P<t1>{TM_})\.?$")
NOBODY_SENT = re.compile(rf"^nobody could (?:get on|board) the (?P<veh>{NP}) at the (?P<stop>{NP}) unless (?:the "
                         rf"(?P<veh2>{NP})|it) was at the (?P<stop2>{NP})\.?$")
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
            m = AWAY_SENT.match(s)
            if not m:
                continue
            d = m.groupdict()
            veh, stop = d["veh"], d["stop"]
            t0s = [x for x in (d["ta"], d["tb"]) if x]
            if len(t0s) != 1:
                continue
            if d["nveh"] and not (d["nveh"] == veh and d["nstop"] == d["nstop2"] == stop and d["nveh2"] in (None, veh)):
                continue
            t0, t1, used = tm(t0s[0]), None, [i]
            if d["t1"] or d["t1b"]:
                t1 = tm(d["t1"] or d["t1b"]) if d["rstop"] in (None, stop) else None
            elif d["dur"]:
                t1 = t0 + dur(d["dur"]) if d["stop2"] == stop and dur(d["dur"]) else None
            elif i + 1 < len(sents):
                r = ROUND_SENT.match(sents[i + 1])
                if r and r.group("veh") in (None, veh) and r.group("stop") == stop and dur(r.group("dur")):
                    t1 = t0 + dur(r.group("dur"))
                r = RET_SENT.match(sents[i + 1])
                if r and r.group("stop") in (None, stop):
                    t1 = tm(r.group("t1"))
                if t1 is not None:
                    used.append(i + 1)
            if t1 is None or t1 <= t0:
                continue
            for j, sj in enumerate(sents):
                r = NOBODY_SENT.match(sj)
                if r and r.group("veh") == veh and r.group("stop") == r.group("stop2") == stop and \
                        r.group("veh2") in (None, veh):
                    used.append(j)
            if tainted(ctx, used, {veh, stop}):
                continue
            for j in used:
                ctx.consume(j)
            v, st = esc(veh), esc(stop)
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

PLACE_KINDS = [
    ("at", clause(rf"(?:at (?P<ta>{TM_}),? )?(?P<p>[a-z]+) was at the (?P<l>{NP})(?: at (?P<tb>{TM_}))?")),
    ("min", clause(rf"every route from the (?P<a>{NP}) to the (?P<b>{NP}) (?:also )?takes at least (?P<d>{DUR_})")),
    ("min", clause(rf"it takes at least (?P<d>{DUR_}) to (?:get|go|travel) from the (?P<a>{NP}) to the (?P<b>{NP})")),
    ("via", clause(rf"every route from the (?P<a>{NP}) to the (?P<b>{NP}) passes through the (?P<m>{NP})")),
    ("sep", SEP_C),
    ("decor", DECOR_C),
]


@register
class MovePlace(Family):
    name = "move_place"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        edges, via, starts, used = {}, {}, [], set()
        for i, s in enumerate(sents):
            parts = split_clauses(s, PLACE_KINDS)
            for kind, m in parts or []:
                if kind == "min" and dur(m.group("d")):
                    edges[(m.group("a"), m.group("b"))] = dur(m.group("d"))
                    used.add(i)
                elif kind == "via":
                    via[(m.group("a"), m.group("b"))] = m.group("m")
                    used.add(i)
                elif kind == "at":
                    t0s = [x for x in m.group("ta", "tb") if x]
                    if len(t0s) == 1 and is_person(m.group("p"), ctx.sentences[i]):
                        starts.append((i, m.group("p"), m.group("l"), tm(t0s[0])))
        out = []
        for i, person, start, t0 in starts:
            bounds = {b: d for (a, b), d in edges.items() if a == start and b != start}
            for (a, b), mid in via.items():
                if a == start and (a, mid) in edges and (mid, b) in edges:
                    bounds[b] = max(bounds.get(b, 0), edges[(a, mid)] + edges[(mid, b)])
            places = list(bounds) + [start] + list(via.values())
            if not bounds or tainted(ctx, used | {i}, set(places), {person}):
                continue
            ctx.consume(i)
            out.append({"person": person, "start": start, "t0": t0, "bounds": bounds,
                        "places": {pl: (other_presence(person, np_re(pl, sents)), self_presence(np_re(pl, sents)))
                                   for pl in list(bounds) + [start]}})
        if not out:
            return None
        names = {pl for f in out for pl in f["places"]}
        for i, s in enumerate(sents):
            parts = split_clauses(s, PLACE_KINDS)
            if i in used or (parts and any(k == "sep" and names & set(m.group("s1", "s2", "s3")) for k, m in parts)):
                ctx.consume(i)
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

SPAN = rf"from (?P<t1>{TM_}) (?:to|until|till) (?P<t2>{TM_})"
BETWEEN = rf"between (?P<t1>{TM_}) and (?P<t2>{TM_})"
NOT_LEAVE = r"(?:did not|didn't|never) (?:leave|left)"
WHOLE = r"(?:the whole time|the entire time|throughout)"
ROOM_KINDS = [  # a person who stayed in one room for a window
    ("stay", clause(rf"(?P<p>[a-z]+) (?:stayed|remained) (?:in|inside) the (?P<r>{NP}) (?:{WHOLE} )?{SPAN}")),
    ("stay", clause(rf"(?P<p>[a-z]+) was (?:in|inside) the (?P<r>{NP}) {WHOLE} {SPAN}")),
    ("stay", clause(rf"{SPAN},? (?P<p>[a-z]+) (?:stayed|remained) (?:in|inside) the (?P<r>{NP})")),
    ("stay", clause(rf"(?P<p>[a-z]+) {NOT_LEAVE} the (?P<r>{NP}) (?:at any point )?{BETWEEN}")),
    ("stay", clause(rf"(?P<p>[a-z]+) {NOT_LEAVE} the (?P<r>{NP}) {SPAN}")),
    ("stay", clause(rf"{BETWEEN},? (?P<p>[a-z]+) {NOT_LEAVE} the (?P<r>{NP})")),
    ("sep", SEP_C),
    ("decor", DECOR_C),
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
            for kind, m in split_clauses(s, ROOM_KINDS) or []:
                if kind != "stay":
                    continue
                person, room, t1, t2 = m.group("p", "r", "t1", "t2")
                if not is_person(person, ctx.sentences[i]) or tm(t2) <= tm(t1):
                    continue
                others = set().union(*[g - {room} for g in groups if room in g]) if groups else set()
                if not others or tainted(ctx, [i], others | {room}, {person}):
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
        names = {x for f in out for x in f["rooms"]}
        for i, s in enumerate(sents):
            parts = split_clauses(s, ROOM_KINDS)
            if parts and all(k == "decor" or (k == "sep" and names & set(m.group("s1", "s2", "s3")))
                             for k, m in parts if k != "stay"):
                if any(k == "sep" for k, _ in parts):
                    ctx.consume(i)
        return out

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        if not c:
            return None
        sender = ctx.speakers.get(msg.label, "").lower()
        return best(self.one(f, sender, *c) for f in facts)

    def one(self, f, sender, t, cl):
        why = f"{f['person']} stayed in the {f['room']} from {T.fmt_time(f['t1'])} to {T.fmt_time(f['t2'])}"
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

OBJ_INTRO = rf"there was (?:one|a single) (?P<o>{NP})"
INTRO_DECOR = rf"(?:, with an? {COLOUR} label|, (?:which|and it) has an? {COLOUR} label)"
STAY_V = r"(?:stayed|remained|was kept|was locked|was left)"
OBJ_SPAN = rf"from (?P<t1>{TM_}) (?:to|until|till) (?P<t2>{TM_})"
PLACE_DECOR = rf"(?:, which has an? {COLOUR} door,)"
OBJ_SENTS = [re.compile(p) for p in (
    rf"^{OBJ_INTRO}{INTRO_DECOR}?\.?$",
    rf"^{OBJ_INTRO}(?:, and it|, which) {STAY_V} in the (?P<pl>{NP}){PLACE_DECOR}? {OBJ_SPAN}\.?$",
    rf"^{OBJ_INTRO}, and it was in the (?P<pl>{NP}) {OBJ_SPAN}, and it stayed there the whole time\.?$",
    rf"^(?:it|the (?P<o2>{NP})) {STAY_V} in the (?P<pl>{NP}){PLACE_DECOR}? {OBJ_SPAN}\.?$",
    rf"^{OBJ_SPAN},? (?:it|the (?P<o2>{NP})) {STAY_V} in the (?P<pl>{NP})\.?$")]
HAVE_AT = r"(?:at|in|on|inside)"


@register
class MoveObject(Family):
    name = "move_object"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        groups = separate_groups(sents)
        out, last, last_i = [], None, None
        for i, s in enumerate(sents):
            m = next((m for m in (r.match(s) for r in OBJ_SENTS) if m), None)
            if not m:
                continue
            d = m.groupdict()
            if d.get("o"):
                last, last_i = d["o"], i
            if not d.get("pl"):
                continue
            o2 = d.get("o2")
            if o2:
                if last and re.fullmatch(rf"(?:{esc(o2)}\b.*|.*\b{esc(o2)})", last):
                    obj = last  # 'The parcel stayed in ...' may shorten 'parcel marked X7'
                elif not is_person(o2, ctx.sentences[i]):
                    obj = o2
                else:
                    continue
            elif d.get("o"):
                obj = d["o"]
            elif last and last_i == previous_fact(sents, i):
                obj = last  # 'It stayed in ...' right after 'There was one blue case.'
            else:
                continue
            place = d["pl"]
            others = set().union(*[g - {place} for g in groups if place in g]) if groups else set()
            used = [i] + ([last_i] if obj == last and last_i is not None else [])
            if tainted(ctx, used, {obj, place} | others | ({o2} if o2 else set())):
                continue
            for j in used:
                ctx.consume(j)
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
            out.append({"obj": obj, "place": place, "t1": tm(d["t1"]), "t2": tm(d["t2"]), "bad": bad, "fine": fine})
        if not out:
            return None
        nouns = {f["obj"] for f in out} | {f["place"] for f in out}
        for i, s in enumerate(sents):
            m = SEP_SENT.match(s)
            if decor(s) in nouns or (m and nouns & set(m.group("s1", "s2", "s3"))):
                ctx.consume(i)
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

CONT_DECOR = rf"(?:,? (?:which|that) was (?:kept on a shelf|labelled in {COLOUR} ink)| and was kept on a shelf)"
HELD_SENTS = [re.compile(p) for p in (
    rf"^(?:one|a single|the) (?P<c>{NP})(?:, which was kept on a shelf,)? held the (?P<i>{NP})(?: and was kept on a "
    rf"shelf|, and it was (?:kept on a shelf|labelled in {COLOUR} ink))?\.?$",
    rf"^the (?P<i>{NP}) (?:was|were) (?:kept |held |stored |placed )?(?:in|inside) (?:one|a single|a) (?P<c>{NP})"
    rf"{CONT_DECOR}?\.?$")]
SEAL_SENT = re.compile(
    rf"^(?:the (?P<c0>{NP}) was kept on a shelf, and )?(?:from (?P<a1>{TM_}) (?:to|until|till) (?P<a2>{TM_}),? )?"
    rf"(?P<subj>it|(?:the|that|this) {NP}) (?:stayed|remained) sealed(?: from (?P<b1>{TM_}) (?:to|until|till) "
    rf"(?P<b2>{TM_}))?(?:, and |; )the (?P<i>{NP}) could not be removed without breaking the seal\.?$")
OPEN = r"(?:opened|unsealed|broke open|tore open|cut open|ripped open|unwrapped)"


@register
class MoveSeal(Family):
    name = "move_seal"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        known = []  # (container, item, sentence) from 'One evidence bag held the USB drive.'
        out = []
        for i, s in enumerate(sents):
            for r in HELD_SENTS:
                m = r.match(s)
                if m:
                    known.append((m.group("c"), m.group("i"), i))
            m = SEAL_SENT.match(s)
            if not m:
                continue
            d = m.groupdict()
            times = [x for x in (d["a1"], d["a2"], d["b1"], d["b2"]) if x]
            if len(times) != 2:
                continue
            subj, item, used = d["subj"], d["i"], [i]
            if d["c0"]:
                cont = d["c0"] if subj == "it" else None
            elif subj == "it":
                cont = known[-1][0] if known and known[-1][2] == previous_fact(sents, i) else None
            elif subj.startswith("the "):
                cont = subj[4:]
            else:  # 'that bag', 'that envelope': the container named earlier
                word = subj.split(" ", 1)[1]
                cand = [c for c, _, _ in known if c == word or c.endswith(" " + word)]
                cont = cand[-1] if cand else None
            if cont is None:
                continue
            held = [(c, it, j) for c, it, j in known if c == cont]
            if held and not any(it == item for _, it, _ in held):
                continue  # 'One bag held the drive' but 'the cash could not be removed': not the same thing
            used += [j for _, _, j in held]
            if tainted(ctx, used, {cont, item}):
                continue
            for j in used:
                ctx.consume(j)
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
            if decor(s) in names:
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
HAND_CLAUSE = re.compile(r"(?:then )?(?:hand(?:ed|ing)|pass(?:ed|ing)|g(?:ave|iving)) it (?:over )?to ([a-z]+)")
HOLD_CLAUSE = re.compile(rf"(?:from ({TM_})(?: (?:to|until) ({TM_}))?,? )?(?:([a-z]+) )?{HOLD} it(?: from ({TM_}))?"
                         rf"(?: (?:to|until|till) ({TM_}))?(?:,? (?:and )?(?:kept|keeping|held|holding) it)?"
                         rf"(?: for exactly ({DUR_}))?")
CHAIN_INTRO = re.compile(rf"^there was (?:one|a single) (?P<o>{NP})(?:,? and (?P<ex>only one person held it at any "
                         rf"moment))?\.?$")


def chain_clauses(s):
    """The clauses of a hand-over sentence when every clause is a holding or a hand-over, else None."""
    cls = [c.strip(" ,.") for c in CHAIN_SPLIT.split(s.strip(" ."))]
    cls = [c for c in cls if c]
    ok = cls and all(HAND_CLAUSE.fullmatch(c) or HOLD_CLAUSE.fullmatch(c) for c in cls)
    return cls if ok and any(HOLD_CLAUSE.fullmatch(c) for c in cls) else None


def parse_chain(text):
    """[(holder, start, end)] from 'Aiko held it from 10:50 AM to 11:50 AM, then handed it to Sofia. Sofia held it
    until 1:20 PM ...' and 'From 10:45 AM Fatima held it for exactly 1 hour, then handed it to Isla.'; None when a
    clause does not fit or the times do not join up."""
    segs, pending = [], None
    for cl in chain_clauses(text) or [None]:
        if cl is None:
            return None
        m = HAND_CLAUSE.fullmatch(cl)
        if m:
            if pending is not None or not segs:
                return None
            pending = m.group(1)
            continue
        f1, t1, who, f2, t2, dd = HOLD_CLAUSE.fullmatch(cl).groups()
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
        elif dd and dur(dd):
            end = start + dur(dd)
        else:
            return None
        if end <= start:
            return None
        segs.append((who, start, end))
    return segs if segs and pending is None else None


def chain_rest_kinds(o):
    """'Nobody else held the X between T and T', 'only one person held the X at any moment' and their variants."""
    it = rf"(?:the {o}|it)"
    return [("window", clause(rf"nobody else held {it} between (?P<w1>{TM_}) and (?P<w2>{TM_})")),
            ("window", clause(rf"between (?P<w1>{TM_}) and (?P<w2>{TM_}),? nobody else held {it}")),
            ("only", clause(rf"only one person held {it} at any moment")),
            ("only", clause(rf"at any moment,? only one person held {it}"))]


@register
class MoveChain(Family):
    name = "move_chain"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        out = []
        for i, s in enumerate(sents):
            m = CHAIN_INTRO.match(s)
            if not m:
                continue
            obj = m.group("o")
            o = esc(obj)
            kinds = chain_rest_kinds(o)
            chain_idx, rest_idx, window, exclusive = [], [], None, bool(m.group("ex"))
            for j in range(i + 1, len(sents)):
                parts = split_clauses(sents[j], kinds)
                if parts:
                    for kind, r in parts:
                        if kind == "window":
                            window = (tm(r.group("w1")), tm(r.group("w2")))
                        exclusive = exclusive or kind == "only"
                    rest_idx.append(j)
                    continue
                if chain_clauses(sents[j]):
                    chain_idx.append(j)
                    continue
                break
            if not chain_idx:
                continue
            segs = parse_chain(" ".join(sents[j].rstrip(".") + "." for j in chain_idx))
            if not segs or not (exclusive or window):
                continue
            if window and (abs(window[0] - segs[0][1]) > EPS or abs(window[1] - segs[-1][2]) > EPS):
                continue
            used = [i] + chain_idx + rest_idx
            if tainted(ctx, used, {obj}, {h for h, _, _ in segs}):
                continue
            for j in used:
                ctx.consume(j)
            out.append({
                "obj": obj, "segs": segs, "exclusive": exclusive, "nobody_else": window is not None,
                "names": {h for h, _, _ in segs},
                "self": re.compile(rf"i (?:had|held|was holding|was carrying|had hold of|was minding|kept|was "
                                   rf"keeping|was looking after) the {o}(?: with me| in my hands?| on me)?|the {o} "
                                   rf"was (?:with me|in my hands?|on me)"),
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

T_A = rf"(?P<a1>{TM_}) (?:to|until|till) (?P<a2>{TM_})"
T_B = rf"(?P<b1>{TM_}) (?:to|until|till) (?P<b2>{TM_})"
NO_CHANGE_C = rf"no (?P<item>{NP}) was added or removed during that time"
COUNT_SENTS = [re.compile(p) for p in (  # a container that held exactly N items for a window
    rf"^(?:from {T_A},? )?(?:the (?P<c>{NP})(?:, which is {COLOUR},)?|it) (?:held|contained) exactly (?P<n>[a-z0-9]+) "
    rf"(?P<items>{NP})(?: from {T_B})?(?:, and {NO_CHANGE_C})?\.?$",
    rf"^(?:from {T_A},? )?there were exactly (?P<n>[a-z0-9]+) (?P<items>{NP}) in the (?P<c>{NP})(?: from {T_B})?"
    rf"(?:, and {NO_CHANGE_C})?\.?$")]
NO_CHANGE_SENT = re.compile(rf"^{NO_CHANGE_C}\.?$")
NUM_WORD = r"(\d+|no|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"


@register
class MoveCount(Family):
    name = "move_count"

    def parse(self, ctx):
        sents = [low(s) for s in ctx.sentences]
        out = []
        for i, s in enumerate(sents):
            m = next((m for m in (rx.match(s) for rx in COUNT_SENTS) if m), None)
            if not m:
                continue
            used = [i]
            cont = m.group("c")
            if cont is None:  # 'It held exactly ...' after 'The display case stands by the window.'
                cont = decor(sents[i - 1]) if i else None
                if cont is None:
                    continue
                used.append(i - 1)
            times = [x for x in m.group("a1", "a2", "b1", "b2") if x]
            n = T.word_num(m.group("n"))
            items = m.group("items")
            if len(times) != 2 or n is None or not items.endswith("s"):
                continue
            item = m.group("item")
            if item is None and i + 1 < len(sents) and NO_CHANGE_SENT.match(sents[i + 1]):
                item = NO_CHANGE_SENT.match(sents[i + 1]).group("item")
                used.append(i + 1)
            if item is not None and not items.startswith(item[:-1] if item.endswith("y") else item):
                continue
            one = item or (items[:-3] + "y" if items.endswith("ies") else items[:-1])
            if tainted(ctx, used, {cont, items, one}):
                continue
            for j in used:
                ctx.consume(j)
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
                "saw": [re.compile(rf"i (?:saw|spotted|noticed) {NUM_WORD} {things} (?:in|inside) the {c}"),
                        re.compile(rf"(?:in|inside) the {c} i (?:saw|spotted|noticed) {NUM_WORD} {things}")],
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
            if decor(s) in conts:
                ctx.consume(i)
        return out

    def judge(self, ctx, facts, msg):
        c = claim(msg.text)
        return best(self.one(f, *c) for f in facts) if c else None

    def one(self, f, t, cl):
        for total, res in ((True, f["count"]), (False, f["saw"])):
            m = next((m for m in (r.fullmatch(cl) for r in res) if m), None)
            if not m:
                continue
            k = T.word_num(m.group(1))
            if k is None:
                return None
            inside = f["t1"] + EPS < t < f["t2"] - EPS
            edge = abs(t - f["t1"]) < EPS or abs(t - f["t2"]) < EPS
            wrong = k != f["n"] if total else k > f["n"]  # seeing five of nine is possible, seeing nine of five not
            if inside and wrong:
                return Verdict(LIE, self.name, f"the {f['cont']} held exactly {f['n']} from "
                                               f"{T.fmt_time(f['t1'])} to {T.fmt_time(f['t2'])}")
            if (inside or edge) and k != f["n"]:
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


def reads(s):
    """True when one of this module's grammars reads the whole lower-case sentence."""
    return bool(
        any(r.match(s) for r in ROUTE_SENT) or SPEED_SENT.match(s)
        or AWAY_SENT.match(s) or ROUND_SENT.match(s) or RET_SENT.match(s) or NOBODY_SENT.match(s)
        or split_clauses(s, PLACE_KINDS) is not None or split_clauses(s, ROOM_KINDS) is not None
        or any(r.match(s) for r in OBJ_SENTS) or decor(s) or SEP_SENT.match(s)
        or any(r.match(s) for r in HELD_SENTS) or SEAL_SENT.match(s)
        or CHAIN_INTRO.match(s) or chain_clauses(s) is not None
        or split_clauses(s, chain_rest_kinds(NP)) is not None
        or any(r.match(s) for r in COUNT_SENTS) or NO_CHANGE_SENT.match(s))


if __name__ == "__main__":  # self-check: python -m lc.rules.fam_movement
    assert claim("Honestly, I took the red key out of the sealed cabinet at 3:30 PM.") == (
        930, "i took the red key out of the sealed cabinet")
    assert claim("It was 3:10 PM when the tractor got to the far field.") == (910, "the tractor got to the far field")
    assert claim("It was at 1:00 PM that I took the exam paper out of the sealed envelope.")[1] == \
        "i took the exam paper out of the sealed envelope"
    assert claim("I reached the shop before 9:00 AM.") is None
    assert claim("Did the van reach the farm at 2:50 PM?") is None
    legs = "first 12 km where it never went above 36 km/h, then 30 km where it never went above 60 km/h"
    assert route_legs(legs) == 50 and reads("it first covered " + legs[6:] + ".")
    assert route_legs("along the 42 km route: first 12 km at 36 km/h, then 30 km at 60 km/h") == 50
    assert parse_chain("aiko held it from 10:50 am to 11:50 am, then handed it to sofia. sofia held it until "
                       "1:20 pm") == [("aiko", 650, 710), ("sofia", 710, 800)]
    assert parse_chain("from 10:45 am fatima held it for exactly 1 hour, then handed it to isla. isla kept it for "
                       "exactly 1 hour, then handed it to sami, who kept it for exactly 2 hours") == [
        ("fatima", 645, 705), ("isla", 705, 765), ("sami", 765, 885)]
    assert window_verdict(600, 600, 660, "x", "") is None and window_verdict(630, 600, 660, "x", "").status == LIE
    # whole-sentence reading: an extra clause leaves the sentence unread
    assert reads("the shuttle bus left the hotel stop at 2:15 pm and did not return before 3:45 pm.")
    assert not reads("the shuttle bus left the hotel stop at 2:15 pm and did not return before 3:45 pm, except on "
                     "weekdays.")
    assert reads("the warehouse and the shop are separate locations, and every route from the warehouse to the shop "
                 "takes at least 30 minutes.")
    assert not reads("every route from the warehouse to the shop takes at least 30 minutes unless the bridge is open.")
    assert reads("the delivery van left the warehouse at 2:45 pm and went straight to the farm: first 12 km where it "
                 "never went above 36 km/h, then 30 km where it never went above 60 km/h.")
    print("ok")
