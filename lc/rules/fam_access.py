"""Rule families: access. See lc/rules/core.py for the Family interface.

Each family reads one kind of evidence about who could get into, use or be at something, and when:
  access_window    nobody could enter / be in / use a place between two times
  access_supply    a machine works only while a supply is on, and the supply was off for a while
  access_code      which door code unlocked a door at which time of day
  access_list      a door opened only for one group before a time, and it was the only way in
  access_badge     entry only by scanning one's own active card, with each person's active times
  access_log       a complete log of every entry to (or exit from) a place
  access_arrival   first-arrival times, as a log or as minutes before/after someone else
  access_presence  only these people were at a place during a time window
  access_booking   a room booked by teams in turn, with only the booked team inside

Chat messages are first rewritten by `claim()` so that the person the message is about becomes "@": "I saw Joel go
into the vault at 9:40 AM" -> actor Joel, text "@ go into the vault at 9:40 am". Families then match short patterns
against that text. A family says LIE only when the message clearly refers to its evidence and cannot be true with it.
"""
import re
from dataclasses import dataclass

from .core import LIE, OK, Family, Verdict, register
from . import text as T

TIME = r"(?:\d{1,2}:\d{2}\s*[AP]M|noon|midday|midnight)"
NAME = r"[A-Z][a-z]+(?:-[A-Z][a-z]+)?"
NAMES = rf"{NAME}(?:, {NAME})*(?:,? and {NAME})?"
NP = r"[a-z][a-z0-9' -]*?"  # a short lowercase noun phrase, matched lazily
NOBODY = r"(?:nobody|no one|no-one|no person)"
DAY_END = 24 * 60

# Words that start with a capital in chat text but are not people.
NOT_NAMES = {
    "I", "At", "The", "A", "An", "On", "In", "By", "From", "Before", "After", "Between", "During", "Room", "Code",
    "Honestly", "Okay", "Ok", "So", "For", "Through", "Using", "With", "Come", "Over", "Down", "Remotely", "Locally",
    "Exactly", "Nobody", "Thanks", "Yes", "No", "Well", "Then", "Around", "About", "Until", "Since", "When", "While",
    "My", "Me", "We", "Our", "It", "This", "That", "There", "Here", "Someone", "Somebody", "Everyone", "Anyone",
    "Marketing", "Sales", "Finance", "Research", "Legal", "Design", "HR", "Back", "Up", "Out", "Off", "Just", "Only",
    "Right", "Actually", "Look", "Sure", "Of", "To", "And", "But", "Or", "If", "As", "Not", "Never", "Today", "Earlier",
    "Later", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
    "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "VPN", "AM",
    "PM", "Cheers", "Shall", "Should", "Could", "Would", "Can", "Does", "Do", "Is", "Are", "Was", "Who", "What",
    "Why", "How", "Let", "Some", "Clarity", "Grateful", "Appreciate", "Good", "Keep", "Please", "Everything", "None",
    "Security", "Reception", "Staff", "Management", "Admin", "Facilities", "Maintenance", "Cleaning", "Catering",
    "Support", "Mum", "Mom", "Dad", "Grandma", "Grandpa", "Doctor", "Nurse", "Officer", "Professor", "Mr", "Mrs",
    "Ms", "Dr", "Sir", "Madam", "Team", "Everybody", "Folks", "Guys", "All", "Both", "Each", "Either", "Neither",
}

LEAD_RE = re.compile(
    r"^\s*(?:(?:honestly|for the record|okay so|ok so|okay|ok|so|well|to be honest|frankly|actually|look|right|yes|"
    r"yeah|anyway|i think|i believe|i'm sure|i am sure|as far as i know|just so you know|i can tell you(?: that)?|"
    r"i can confirm(?: that)?|i can say(?: that)?|i know(?: that)?)\s*[,:]?\s+)+", re.I)
TAIL_RE = re.compile(
    r"(?:\s*,\s*(?:honestly|for the record|okay|ok|to be honest|i think|i believe|i'm sure|for sure|i promise|"
    r"definitely|i swear))+\s*$", re.I)
SAW_IT_RE = re.compile(r",?\s*(?:and )?I (?:saw|watched|noticed|spotted|witnessed) (?:it|this|that|them|him|her)"
                       r"(?: happen)?(?: myself)?\s*$")
WITNESS_RE = re.compile(r"\bI (?:saw|watched|noticed|spotted|witnessed|observed)(?: that)? (you|" + NAME + r")\b")
FIRST_PERSON = re.compile(r"\b(?:I|[Mm]e|[Mm]y|[Mm]yself|[Mm]ine)\b")
# Messages these match are not plain claims about that day: another day, someone's report or belief, a hedge.
OTHER_DAY_RE = re.compile(r"\b(?:yesterday|tomorrow|last (?:week|month|year|night)|next (?:week|month|year|day)|the day "
                          r"before|the previous day|the following day|the next day|Monday|Tuesday|Wednesday|Thursday|"
                          r"Friday|Saturday|Sunday|January|February|March|April|May|June|July|August|September|October|"
                          r"November|December)\b", re.I)
REPORT_RE = re.compile(r"\b(?:thinks?|thought|said|says|claims?|claimed|believes?|believed|reckons?|suspects?|"
                       r"suspected|assumed?|assumes|told|tells|heard|hears|guess(?:ed|es)?|wondered|asked|asks|accused|"
                       r"accuses|if|unless|whether|would|might|maybe|perhaps|probably|possibly|supposedly|apparently|"
                       r"allegedly|should|must have|could have)\b", re.I)
# Two joined actions ("@ left the office at 9:40 am and went into the vault"): the time may belong to either one.
JOINED_RE = re.compile(r"\b(?:and|then|but|before|after|while) (?:then )?(?:@ )?([a-z]+ed|went|got|came|made|took|left|"
                       r"ran|saw|rode|sat|met|had|was|were|stayed|did|go|get|come)\b")
NEG_RE = re.compile(r"\b(?:not|never|n't|didnt|no point|nobody|no one|nowhere|neither|nor|cannot|without|almost|"
                    r"nearly|tried to|try to|wanted to|meant to|planned to|hoped to|was going to|were going to)\b|n't\b")
TIMES_WORD_RE = re.compile(r"(\w+)\W*$")


@dataclass
class Claim:
    actor: str  # the person the message is about (the sender for "I ...")
    text: str  # lowercase, fillers removed, actor replaced by "@", commas removed
    times: list  # [(minutes, preceding word)] for each clock time in the message
    names: dict  # lowercase name -> name, for names known in this case


def case_names(ctx):
    """Names of people known in this case: the speakers, capitalised words of the evidence, and people the chat says
    it saw. Used to tell a person's name from an ordinary capitalised word."""
    if getattr(ctx, "_access_names", None) is None:
        names = set(ctx.speakers.values())
        for s in ctx.sentences:
            for w in re.findall(NAME, s)[1:] if s else []:
                names.add(w)
            m = re.match(NAME, s)
            if m and re.match(rf"{NAME}(?:'s| and | first| was| is| are| held| left| went| stayed)", s):
                names.add(m.group(0))
        for msg in ctx.case.messages:
            for m in WITNESS_RE.finditer(msg.text):
                names.add(m.group(1))
        ctx._access_names = {n for n in names if n not in NOT_NAMES}
    return ctx._access_names


def find_clock(s):
    """Clock times in s as [(minutes, preceding word)]. 'noon'/'midnight' count only in 'at noon'."""
    out = []
    for v, m in T.find_times(s):
        before = s[:m.start()]
        w = TIMES_WORD_RE.search(before)
        prev = w.group(1).lower() if w else ""
        if m.group(4) and prev != "at":
            continue
        out.append((v, prev))
    return out


def claim(ctx, msg):
    """Rewrite one chat message as a Claim, or None when it is a question or the actor is unclear."""
    t = msg.text.strip().replace("’", "'")
    if t.endswith("?") or OTHER_DAY_RE.search(t) or REPORT_RE.search(t) or T.find_dates(t):
        return None
    t = t.rstrip(" .!")
    t = LEAD_RE.sub("", t)
    t = TAIL_RE.sub("", t).rstrip(" .!")
    names = case_names(ctx)
    addressee = None
    m = re.match(rf"^({NAME}),\s+(.*)$", t)
    if m and m.group(1) not in NOT_NAMES:
        addressee, t = m.group(1), m.group(2)
    m = re.match(rf"^(.*?),\s+({NAME})$", t)
    if m and m.group(2) not in NOT_NAMES:
        addressee, t = m.group(2), m.group(1)
    m = re.search(rf",\s+({NAME}),\s+", t)  # "At 6:50 PM, Sana, I saw you ..."
    if addressee is None and m and m.group(1) not in NOT_NAMES and re.search(r"\byou\b", t):
        addressee, t = m.group(1), t[:m.start()] + ", " + t[m.end():]
    t = LEAD_RE.sub("", t)
    saw_it = SAW_IT_RE.search(t)
    if saw_it:
        t = t[:saw_it.start()]
    m = WITNESS_RE.search(t)
    if m:
        who = m.group(1)
        if who == "you":
            who = addressee
        if who is None or who in NOT_NAMES:
            return None
        actor = who
        t = t[:m.start()] + "@" + t[m.end():]
        if FIRST_PERSON.search(t) or re.search(r"\byou\b", t, re.I):
            return None
    else:
        m = re.match(rf"^({NAME}) (?=[a-z])", t)
        if m and m.group(1) in names and not FIRST_PERSON.search(t):
            actor = m.group(1)
            t = "@" + t[m.end() - 1:]
        elif FIRST_PERSON.search(t):
            if re.search(r"\byou\b", t, re.I):
                return None
            actor = ctx.speakers.get(msg.label)
            t = re.sub(r"\b[Mm]yself\b", "@", t)
            t = re.sub(r"\b(?:[Mm]y|[Mm]ine)\b", "@'s", t)
            t = re.sub(r"\b(?:I|[Mm]e)\b", "@", t)
            t = t.replace("@'m", "@ am")
        else:
            actor = None  # a statement about things, not people ("The stage lamps were lit at 2:00 PM")
    times = find_clock(t)
    low = re.sub(r"\s+", " ", t.replace(",", " ").lower()).strip()
    return Claim(actor, low, times, {n.lower(): n for n in names})


def event_time(c, same_event=()):
    """The single clock time of the claim when it is given as 'at T' and the claim names one action, else None.
    same_event lists verbs that may be joined on without changing the event ('punched in code 1311 and unlocked')."""
    if len(c.times) != 1:
        return None
    m = JOINED_RE.search(c.text)
    if m and m.group(1) not in same_event:
        return None
    v, prev = c.times[0]
    return v if prev == "at" else None


def span(c):
    """(T1, T2) for a claim of the form '... between T1 and T2 ...' (both ends included), else None."""
    m = re.search(rf"\b(?:between|from) ({TIME}) (?:and|to|until) ({TIME})", c.text, re.I)
    if not m:
        return None
    return T.parse_time(m.group(1)), T.parse_time(m.group(2))


def negated(c):
    return bool(NEG_RE.search(c.text))


def tm(s):
    v = T.parse_time(s)
    if v is None:
        raise ValueError(f"no single time in {s!r}")
    return v


def clean_np(s):
    s = s.strip().lower()
    s = re.sub(r"^(?:the|a|an)\s+", "", s)
    return s


# What may follow a place or machine name in a claim: the end, or a word that cannot extend the noun phrase. This keeps
# "the vault" from matching inside "the vault corridor" or "the lift key".
NP_END = (r"(?=$| @| (?:at|to|up|down|in|into|on|onto|with|without|through|via|by|from|for|and|before|after|around|"
          r"when|while|until|so|because|but|then|again|first|alone|together|today|that|this|just|once|too|as|using|"
          r"is|was|were|had|has|came|went|got|sat|unlocked|opened)\b)")


def place_re(p):
    """Regex for a place noun phrase as it appears in a claim, with an optional article and nothing more after it."""
    return rf"(?:the )?{re.escape(p)}{NP_END}"


def match_pieces(sentence, pats):
    """Match anchored patterns against a whole sentence or its clauses. A sentence is cut at '; ', ', while ' and
    ', and ' into clauses; every run of adjacent clauses is tried, longest first, and runs overlapping an earlier
    match are skipped. Returns [(pattern index, match)] in sentence order, or [] unless the matches cover every
    clause: a sentence with any part we do not model is not read at all."""
    parts = re.split(r"(;\s+|,\s+while\s+|,\s+and\s+)", sentence.strip().rstrip("."))
    cls, seps = parts[0::2], parts[1::2]
    runs = [(i, j) for i in range(len(cls)) for j in range(i, len(cls))]
    runs.sort(key=lambda r: r[0] - r[1])  # longest first; stable, so equal lengths stay in sentence order
    taken, found = set(), []
    for i, j in runs:
        if taken & set(range(i, j + 1)):
            continue
        text = "".join(cls[k] + (seps[k] if k < j else "") for k in range(i, j + 1))
        for n, p in enumerate(pats):
            m = p.match(text)
            if m:
                taken.update(range(i, j + 1))
                found.append((i, n, m))
                break
    if len(taken) != len(cls):
        return []
    return [(n, m) for _, n, m in sorted(found, key=lambda f: f[0])]


DATE_LINE = re.compile(r"^All times in this case refer to \d{1,2} [A-Z][a-z]+ \d{4}\.?$")


class Unread:
    """The evidence sentences a family did not read. A family stays silent about any place, object or person that one
    of them mentions, since that sentence might change what the family's own evidence means."""

    def __init__(self, ctx, seen):
        self.sents = [s for i, s in enumerate(ctx.sentences) if i not in seen and not DATE_LINE.match(s)]
        self.low = [s.lower() for s in self.sents]

    def noun(self, *phrases):
        """A lowercase phrase (or its plural) appears in an unread sentence."""
        pats = [rf"\b{re.escape(p)}(?:s|es)?\b" for p in phrases if p]
        return any(re.search(p, s) for p in pats for s in self.low)

    def person(self, *names):
        pats = [rf"\b{re.escape(n)}\b" for n in names if n]
        return any(re.search(p, s) for p in pats for s in self.sents)


def verdict(fam, ok, why):
    return Verdict(OK if ok else LIE, fam.name, why)


# Verbs for "went into a place". Base forms come from "I saw X go into ...", past forms from "I went into ...".
ENTER = (r"(?:entered|enters|enter|entering|went into|go into|goes into|going into|got into|get into|getting into|"
         r"walked into|walk into|walking into|stepped into|step into|stepping into|slipped into|slip into|"
         r"headed into|head into|heading into|came into|come into|coming into|let @ into|let (?:him|her|them)self "
         r"into|made it into|make it into|went inside|go inside|got inside|get inside|snuck into|sneaked into|"
         r"ducked into|popped into|hurried into|hurry into|rushed into|rush into|strolled into|stroll into|"
         r"wandered into|wander into|went in to|moved into|went onto|go onto|got onto|get onto|walked onto|"
         r"walk onto|stepped onto|step onto|came onto|come onto|let @ in to|was let into|"
         r"(?:made|make|making|found|find) (?:@'s|his|her|their) way into|"
         r"(?:came|come|walked|walk|went|go|got|get|stepped|step|headed|head|slipped|slip) in to)")


# Helper words allowed between the actor and the verb ("@ didn't go into", "@ then walked into").
AUX = (r"(?:(?:did|didn't|did not|do not|don't|never|also|then|just|finally|quickly|eventually|actually|definitely|"
       r"really|only|briefly|later|once|certainly) ){0,2}")


def entry_re(p):
    return rf"@ {AUX}{ENTER} {place_re(p)}"


def entry_any_door_re(p):
    """Entry into p, possibly naming the door used ('came in through the side gate into the yard')."""
    return (rf"@ (?:{ENTER} {place_re(p)}|(?:came|come|walked|walk|went|go|got|get|stepped|step|slipped|slip|headed|"
            rf"head|made it|make it)(?: in)? (?:through|via|by way of) (?:the )?[a-z -]+? (?:into|in to|onto) "
            rf"{place_re(p)})")


# ---------------------------------------------------------------------------------------------------------------
@register
class AccessWindow(Family):
    """'Nobody could enter the vault between 9:00 AM and 11:00 AM.' An entry claim at a time inside the window is a
    lie; 'be in/inside' also rules out being there, 'use' rules out using it. Both ends of 'between' are included."""

    name = "access_window"
    VERB = r"(?P<verb>enter|get into|go into|go inside|access|get inside|be in|be inside|use|set foot in)"
    PATS = [
        re.compile(rf"^{NOBODY} (?:could|was able to|was allowed to|could ever) {VERB} (?:the )?(?P<p>{NP}) "
                   rf"(?:between (?P<a>{TIME}) and (?P<b>{TIME})|from (?P<c>{TIME}) (?:to|until) (?P<d>{TIME}))$",
                   re.I),
        re.compile(rf"^(?:between (?P<a>{TIME}) and (?P<b>{TIME})|from (?P<c>{TIME}) (?:to|until) (?P<d>{TIME})),? "
                   rf"{NOBODY} (?:could|was able to|was allowed to) {VERB} (?:the )?(?P<p>{NP})$", re.I),
        re.compile(rf"^(?:the )?(?P<p>{NP}) could not be (?P<verb>entered|used) (?:by anyone )?"
                   rf"(?:between (?P<a>{TIME}) and (?P<b>{TIME})|from (?P<c>{TIME}) (?:to|until) (?P<d>{TIME}))$",
                   re.I),
    ]
    FLAVOUR = re.compile(r"^(?:the )?(?P<p>[a-z ]+?)(?:'s)? (?:door )?(?:has a \w+ door|door is (?:painted )?\w+|"
                         r"is painted \w+|holds \w+ shelves|has \w+ shelves|was repainted in \d{4}|(?:was given|got) a "
                         r"fresh coat of paint in \d{4})$|^in \d{4} the (?P<q>[a-z ]+?) (?:was repainted|got a fresh "
                         r"coat of paint)$|^\w+ shelves stand inside (?:the )?(?P<r>[a-z ]+?)$", re.I)

    def parse(self, ctx):
        wins, seen = [], set()
        for i, s in enumerate(ctx.sentences):
            body = s.rstrip(".")
            for pat in self.PATS:
                m = pat.match(body)
                if not m:
                    continue
                p = clean_np(m.group("p"))
                if len(p.split()) > 4:
                    continue
                verb = m.group("verb").lower()
                kind = "use" if verb in ("use", "used") else "in" if verb.startswith("be ") else "enter"
                between = m.group("a") is not None
                t1 = tm(m.group("a") or m.group("c"))
                t2 = tm(m.group("b") or m.group("d"))
                wins.append((p, kind, t1, t2, between))
                seen.add(i)
                break
        if not wins:
            return None
        places = {w[0] for w in wins}
        for i, s in enumerate(ctx.sentences):
            m = self.FLAVOUR.match(s.rstrip("."))
            if m and clean_np(m.group("p") or m.group("q") or m.group("r")) in places:
                seen.add(i)
        for i in seen:
            ctx.consume(i)
        return {"wins": wins, "unread": Unread(ctx, seen)}

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None or c.actor is None:
            return None
        for p, kind, t1, t2, between in facts["wins"]:
            pr = place_re(p)
            if not re.search(pr, c.text):
                continue
            if facts["unread"].noun(p):
                return None
            hit = re.search(entry_re(p), c.text) and kind in ("enter", "in")
            if kind == "in" and re.search(rf"@ (?:was|were|stayed|remained|sat|worked|waited) (?:still )?"
                                          rf"(?:in|inside) {pr}", c.text):
                hit = True
            if kind == "use" and re.search(rf"@ (?:used|use|using) {pr}", c.text):
                hit = True
            if hit:
                if negated(c):
                    plain = (re.search(rf"@ (?:didn't|did not|never) (?:{ENTER}|set foot in) {pr}", c.text)
                             and not re.search(r"\bwithout\b", c.text))
                    return verdict(self, True, f"denies entering {p}") if plain else None
                t = event_time(c)
                if t is None:
                    return None
                inside = t1 <= t <= t2 if between else t1 < t < t2
                edge = not between and t in (t1, t2)
                if edge:
                    return None
                return verdict(self, not inside, f"{p} closed {T.fmt_time(t1)}-{T.fmt_time(t2)}, claim {T.fmt_time(t)}")
            if re.search(rf"@ (?:was |were )?(?:\w+ )?(?:waiting|standing|sitting|loitering)? ?(?:outside|by|near|"
                         rf"beside|next to|in front of) {re.escape('the ' + p) if not p.startswith('room') else pr}"
                         rf"(?: door| gate)?(?: at|$)", c.text):
                return verdict(self, True, f"near {p}, not inside")
        return None


# ---------------------------------------------------------------------------------------------------------------
IRREGULAR = {"make": "made", "run": "ran", "ride": "rode", "take": "took", "light": "lit", "go": "went", "get": "got",
             "grind": "ground", "freeze": "froze", "spin": "spun", "send": "sent", "do": "did", "write": "wrote",
             "draw": "drew", "cut": "cut", "shine": "shone", "sing": "sang", "scan": "scanned", "stop": "stopped",
             "ship": "shipped"}
CONVEYANCE = {"lift", "elevator", "escalator", "funicular", "tram", "train", "monorail", "chairlift", "gondola",
              "cable car", "ski lift", "hoist", "dumbwaiter"}
LIT_WORDS = r"(?:lit|lit up|on|switched on|turned on|glowing|shining|illuminated|burning|bright)"
RUN_WORDS = r"(?:running|operating|in operation|moving)"
OFF_WORDS = (r"(?:off|switched off|turned off|shut off|cut off|stopped|offline|down|out|out of service|not running|"
             r"unavailable|disconnected|shut down|out of action|not working|dead|disabled|interrupted)")


def verb_forms(v):
    """Inflected forms of a verb lemma: make -> make, made, makes, making."""
    v = v.lower()
    past = IRREGULAR.get(v)
    if past is None:
        past = v + "d" if v.endswith("e") else v[:-1] + "ied" if re.search(r"[^aeiou]y$", v) else v + "ed"
    ing = v[:-1] + "ing" if v.endswith("e") and not v.endswith("ee") else v + "ing"
    third = v + "es" if re.search(r"(?:s|sh|ch|x|z|o)$", v) else v + "s"
    return {v, past, ing, third}


@register
class AccessSupply(Family):
    """'The lift could run only while the main power was on. The main power was off from 5:15 PM to 5:45 PM.'
    Chains are followed (lamps need the generator, the generator needs fuel from the red tank). Also 'Every remote
    sign-in to the archive server required the VPN' with the VPN offline for a while; a local sign-in is fine.
    A claim that the machine did its job (or was lit/running) inside an outage is a lie. 'From T1 to T2' ends are
    left alone; both ends of 'between' count."""

    name = "access_supply"
    ON = r"(?:on|running|working|operating|available|online|up|flowing|switched on|turned on|connected|in service)"
    DEP = [
        re.compile(rf"^(?:the )?(?P<a>{NP}) could (?P<v>{NP}) only (?:while|when|if|as long as) (?:the )?(?P<b>{NP}) "
                   rf"(?:was|were) {ON}$", re.I),
        re.compile(rf"^(?:the )?(?P<a>{NP}) could (?P<v>{NP}) only (?:while|when|if|as long as) (?:the )?(?P<b>{NP}) "
                   rf"(?:supplied|fed|powered|gave) (?:it|them)(?: with \w+)?$", re.I),
        re.compile(rf"^only (?:while|when|if) (?:the )?(?P<b>{NP}) (?:was|were) {ON},? could (?:the )?(?P<a>{NP}) "
                   rf"(?P<v>{NP})$", re.I),
    ]
    SIGN = r"(?:sign-in|sign in|log-in|login|log in|connection)"
    REMOTE = [
        re.compile(rf"^every remote {SIGN} to (?:the )?(?P<s>{NP}) (?:required|needed|went through|used) "
                   rf"(?:the )?(?P<b>{NP})$", re.I),
        re.compile(rf"^(?:the )?(?P<b>{NP}) was (?:required|needed) for every remote {SIGN} to (?:the )?(?P<s>{NP})$",
                   re.I),
        re.compile(rf"^there was no other remote route: every remote {SIGN} to (?:the )?(?P<s>{NP}) "
                   rf"(?:required|needed) (?:the )?(?P<b>{NP})$", re.I),
    ]
    WIN = rf"(?:from (?P<t1>{TIME}) (?:to|until) (?P<t2>{TIME})|between (?P<t3>{TIME}) and (?P<t4>{TIME}))"
    OFF = [
        re.compile(rf"^(?:the )?(?P<b>{NP}) (?:was|were) {OFF_WORDS} {WIN}$", re.I),
        re.compile(rf"^{WIN},? (?:the )?(?P<b>{NP}) (?:was|were) {OFF_WORDS}$", re.I),
        re.compile(rf"^(?:the )?(?P<b>{NP}) (?:supplied|provided|delivered|gave) no \w+(?: to (?:it|them|the {NP}))? "
                   rf"{WIN}$", re.I),
        re.compile(rf"^{WIN},? (?:the )?(?P<b>{NP}) (?:supplied|provided|delivered|gave) no \w+$", re.I),
    ]
    NO_ROUTE = re.compile(rf"^there was no other remote route(?: to (?:the )?(?P<s>{NP}))?$", re.I)

    def parse(self, ctx):
        deps, outs, remote, seen = [], [], [], set()
        pats = [self.NO_ROUTE] + self.DEP + self.REMOTE + self.OFF
        kinds = ["route"] + ["dep"] * len(self.DEP) + ["remote"] * len(self.REMOTE) + ["off"] * len(self.OFF)
        server = None  # "There was no other remote route to the archive server, and every remote sign-in to it ..."
        for i, s in enumerate(ctx.sentences):
            found = match_pieces(s, pats)
            d, o, r = [], [], []
            for n, m in found:
                if kinds[n] == "route" and m.group("s"):
                    server = clean_np(m.group("s"))
                elif kinds[n] == "dep":
                    d.append((clean_np(m.group("a")), m.group("v").lower(), clean_np(m.group("b"))))
                elif kinds[n] == "remote":
                    r.append((server if clean_np(m.group("s")) == "it" else clean_np(m.group("s")),
                              clean_np(m.group("b"))))
                elif kinds[n] == "off":
                    a, b = m.group("t1") or m.group("t3"), m.group("t2") or m.group("t4")
                    o.append((clean_np(m.group("b")), tm(a), tm(b), m.group("t3") is not None))
            if found and all(x[0] for x in r):  # "it" with no server named before it: not read
                deps += d
                outs += o
                remote += r
                seen.add(i)
        if not outs or not (deps or remote):
            return None

        def same(x, y):  # "the VPN" and "the VPN service" name the same thing; "the lift lobby lights" is not the lift
            tail = r" (?:service|supply|system|network|connection|line|feed|circuit)$"
            return re.sub(tail, "", x) == re.sub(tail, "", y)

        def windows(thing, path=()):
            w = [(t1, t2, btw) for b, t1, t2, btw in outs if same(b, thing)]
            for a, _, b in deps:
                if a == thing and b not in path:
                    w += windows(b, path + (thing,))
            return w

        def chain(thing, path=()):  # the thing, what it needs, and the names its outages use
            names = {thing} | {b for b, _, _, _ in outs if same(b, thing)}
            for a, _, b in deps:
                if a == thing and b not in path:
                    names |= chain(b, path + (thing,))
            return names

        machines = [(a, v, windows(a), chain(a)) for a, v, _ in deps if windows(a)]
        servers = [(s, windows(b), chain(b) | {s}) for s, b in remote if windows(b)]
        if not machines and not servers:
            return None
        for i in seen:
            ctx.consume(i)
        return {"machines": machines, "servers": servers, "heads": [m[0].split()[-1] for m in machines],
                "unread": Unread(ctx, seen)}

    def uses(self, c, a, v, heads):
        """True when the claim says machine a did its job v (or was in its working state)."""
        head = a.split()[-1]
        # the full name, or "the" + its last word when no other machine shares it ("the machine", "the lamps"),
        # never inside a longer name ("the reading lamps", "the lift key")
        obj = (rf"(?:\bthe (?:{re.escape(a)}|{re.escape(head)})|\b{re.escape(a)})"
               if heads.count(head) == 1 and head != a else rf"\b(?:the )?{re.escape(a)}")
        if not re.search(rf"{obj}\b", c.text):
            return False
        words = v.split()
        end = r"(?= at | by | around | when |$)"  # "were on at 2:35 pm", not "were on the floor"
        if words[0] == "be" and len(words) >= 2:  # "could be lit": a state claim about the machine
            states = LIT_WORDS if words[1] in ("lit", "on", "illuminated") else re.escape(" ".join(words[1:]))
            return bool(re.search(rf"{obj} (?:was|were) (?:still |already )?{states}{end}", c.text)
                        or re.search(rf"@ saw {obj} {states}{end}", c.text))
        obj_end = obj + NP_END
        if words in (["run"], ["operate"], ["work"]):
            if head in CONVEYANCE or a in CONVEYANCE:
                if re.search(rf"@ {AUX}(?:rode|ride|riding|took|take|taking|used|use|using|caught|catch) {obj_end}",
                             c.text):
                    return True
                if re.search(rf"@ {AUX}(?:went|came|got|travelled|traveled|rode) (?:up|down) (?:\S+ ){{0,4}}?in "
                             rf"{obj_end}", c.text):
                    return True
            return bool(re.search(rf"{obj} (?:was|were) (?:still |already )?{RUN_WORDS}{end}", c.text))
        # an action, maybe with an object: "make coffee" -> "made a coffee with the machine", "with the machine I
        # made a coffee", "used the machine to make a coffee"; the machine must be what the action was done with
        forms = "|".join(sorted(verb_forms(words[0]), key=len, reverse=True))
        rest = "".join(rf" (?:\S+ ){{0,3}}?{re.escape(w)}" for w in words[1:])
        act = rf"(?:{forms}){rest}\b"
        tool = rf"(?:with|using|on|from|in) {obj_end}"
        return bool(re.search(rf"@ {AUX}{act} (?:\S+ ){{0,3}}?{tool}", c.text)
                    or re.search(rf"\b{tool} @ {AUX}{act}", c.text)
                    or re.search(rf"@ {AUX}(?:used|use|using) {obj} to {act}", c.text))

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None:
            return None
        for a, v, wins, names in facts["machines"]:
            if self.uses(c, a, v, facts["heads"]):
                t = event_time(c)
                if negated(c) or t is None or facts["unread"].noun(*names):
                    return None
                return self.check(wins, t, f"{a} could not {v}")
        for s, wins, names in facts["servers"]:
            if not re.search(rf"(?:the )?{re.escape(s)}{NP_END}", c.text):
                continue
            if facts["unread"].noun(*names):
                return None
            if not re.search(r"\b(?:sign(?:ed|s|ing)? (?:in|on|into)|sign-in|log(?:ged|s|ging)? (?:in|on|into)|"
                             r"log-in|login)\b", c.text):
                continue
            remote = re.search(r"\bremote(?:ly)?\b", c.text)
            local = re.search(r"\b(?:local(?:ly)?|on site|on-site|in person)\b", c.text)
            t = event_time(c)
            if negated(c) or t is None or c.actor is None or bool(remote) == bool(local):
                return None
            if local:
                return verdict(self, True, "a local sign-in needs no remote route")
            return self.check(wins, t, f"no remote sign-in to {s}")
        return None

    def check(self, wins, t, why):
        for t1, t2, between in wins:
            if t1 < t < t2 or (between and t in (t1, t2)):
                return verdict(self, False, f"{why} {T.fmt_time(t1)}-{T.fmt_time(t2)}, claim {T.fmt_time(t)}")
            if t in (t1, t2):
                return None
        return verdict(self, True, f"outside every outage, claim {T.fmt_time(t)}")


# ---------------------------------------------------------------------------------------------------------------
@register
class AccessCode(Family):
    """'Before 10:40 AM, only code 9230 could unlock the storeroom door. From 10:40 AM to the end of the day, only
    code 1311 could unlock it.' Unlocking that door with any other code at a time a rule covers is a lie, whoever
    did it. 'Until T' is read as 'before T' with T itself left alone. 'Codes are changed every month.' and keypad
    descriptions carry no constraint."""

    name = "access_code"
    DOOR = rf"(?:the )?(?P<d>{NP})"
    BEFORE = [
        re.compile(rf"^(?P<r>before|until) (?P<t>{TIME}),? only code (?P<c>\d+) could unlock {DOOR}$", re.I),
        re.compile(rf"^only code (?P<c>\d+) could unlock {DOOR} (?P<r>before|until) (?P<t>{TIME})$", re.I),
        re.compile(rf"^(?P<r>before|until) (?P<t>{TIME}),? {DOOR} could (?:only )?be unlocked (?:only )?(?:by|with) "
                   rf"code (?P<c>\d+)$", re.I),
        re.compile(rf"^{DOOR} could (?:only )?be unlocked (?:only )?(?:by|with) code (?P<c>\d+) (?P<r>before|until) "
                   rf"(?P<t>{TIME})$", re.I),
        re.compile(rf"^code (?P<c>\d+) was the only (?:code|one) that could unlock {DOOR} (?P<r>before|until) "
                   rf"(?P<t>{TIME})$", re.I),
    ]
    TO_END = r"(?:to|until|till|through|through to) the end of the day|onwards?"
    AFTER = [
        re.compile(rf"^from (?P<t>{TIME}) (?:{TO_END}),? only code (?P<c>\d+) could unlock (?:it|{DOOR})$", re.I),
        re.compile(rf"^only code (?P<c>\d+) could unlock (?:it|{DOOR}) from (?P<t>{TIME}) (?:{TO_END})$", re.I),
        re.compile(rf"^from (?P<t>{TIME}) (?:{TO_END}),? code (?P<c>\d+) was the only (?:code|one) that could unlock "
                   rf"(?:it|{DOOR})$", re.I),
        re.compile(rf"^from (?P<t>{TIME}) (?:{TO_END}),? (?:it|{DOOR}) could (?:only )?be unlocked (?:only )?"
                   rf"(?:by|with) code (?P<c>\d+)$", re.I),
    ]
    FLAVOUR = re.compile(r"^(?:the )?codes? (?:are|is|were|was) changed every \w+$|^(?:the )?[a-z ]+ (?:has|had) a "
                         r"keypad(?: with a \w+ light)?$|^the keypad on (?:the )?[a-z ]+ (?:has|had) a \w+ light$|"
                         r"^(?:the )?[a-z ]+'s keypad (?:has|had) a \w+ light$", re.I)

    def parse(self, ctx):
        rules = {}  # door -> [(start, end, code, end_is_unclear)]
        last_door = None
        flavour, seen = [], set()
        pats = self.BEFORE + self.AFTER + [self.FLAVOUR]
        for i, s in enumerate(ctx.sentences):
            used = False
            found, new = match_pieces(s, pats), []
            for n, m in found:
                if n >= len(self.BEFORE) + len(self.AFTER):
                    continue
                d = clean_np(m.group("d")) if m.group("d") else "it"
                d = last_door if d == "it" else d  # "... only code 8932 could unlock it."
                if d is None:
                    break
                last_door = d
                if n < len(self.BEFORE):
                    new.append((d, (0, tm(m.group("t")), m.group("c"), m.group("r").lower() == "until")))
                else:
                    new.append((d, (tm(m.group("t")), DAY_END, m.group("c"), False)))
            else:
                if not found:
                    continue
                if not new:
                    flavour.append(i)
                    continue
                for d, rule in new:
                    rules.setdefault(d, []).append(rule)
                used = True
            if used:
                seen.add(i)
        if not rules:
            return None
        seen.update(flavour)
        for i in seen:
            ctx.consume(i)
        return {"rules": rules, "unread": Unread(ctx, seen)}

    UNLOCKED = r"(?:unlock|unlocked|unlocks|unlocking|open|opened|opens)"
    USE = [  # the code must be the one the unlocking was done with
        re.compile(rf"\b{UNLOCKED}\b[^@]*? (?:with|using|by entering|by typing|by keying in) code (\d+)\b"),
        re.compile(rf"\b(?:used|use|using|entered|typed|punched in|keyed in|tapped in) code (\d+) (?:to|and) "
                   rf"{UNLOCKED}\b"),
        re.compile(rf"\bcode (\d+) {UNLOCKED}\b"),
        re.compile(rf"\b(?:with|using) code (\d+) @ {UNLOCKED}\b"),
        re.compile(rf"\bcode (\d+) (?:was|is) what @ used to {UNLOCKED}\b"),
    ]
    ATTEMPT = re.compile(r"\b(?:tried|try|tries|trying|attempt\w*|failed|wouldn't|couldn't|didn't|refused|wrong)\b")

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None:
            return None
        codes = re.findall(r"\bcode (\d+)\b", c.text)
        used = {m.group(1) for p in self.USE for m in p.finditer(c.text)}
        if len(set(codes)) != 1 or used != set(codes):
            return None
        if negated(c) or self.ATTEMPT.search(c.text):
            return None
        code = codes[0]
        for d, rules in facts["rules"].items():
            names = [re.escape(d)] + ([re.escape(d[:-5])] if d.endswith(" door") else [])  # "the storeroom"
            if not re.search(rf"(?:the )?(?:{'|'.join(names)}){NP_END}", c.text):
                continue
            t = event_time(c, same_event=("unlocked", "opened"))
            if t is None or facts["unread"].noun(d, *(f"code {r[2]}" for r in rules), f"code {code}"):
                return None
            for start, end, good, until in rules:
                if until and t == end:
                    return None
                if start <= t < end:
                    return verdict(self, code == good,
                                   f"at {T.fmt_time(t)} only code {good} could unlock the {d}, claim code {code}")
            return None
        return None


# ---------------------------------------------------------------------------------------------------------------
def group_key(g):
    """'people on the safety-trained list' / 'one of the cast members' / 'a lifeguard' / 'staff badges' -> a key."""
    g = g.lower().strip()
    g = re.sub(r"^(?:people|persons|those|anyone|everyone|members|holders)(?: who are)? (?:on|in|of) ", "", g)
    g = re.sub(r"^(?:both |all )?(?:on|in) ", "", g)
    g = re.sub(r"^(?:one of the|a member of the|members of the|the|a|an) ", "", g)
    g = re.sub(r" (?:badges?|cards?|passes?|holders?)$", "", g)
    g = re.sub(r"(?<=[a-z])(?<!s)s$", "", g)  # plural -> singular, "lifeguards" -> "lifeguard"
    return g.strip()


@register
class AccessList(Family):
    """'Before 9:45 AM the lab door opened only for people on the safety-trained list, and it was the only way into
    the lab.' plus who is (not) on the list. Entering the lab before 9:45 AM is a lie only for someone the evidence
    puts off the list ('Bruno is on the volunteer rota. Nobody on the volunteer rota is on the safety-trained
    list.', 'Carmen is a visitor, not staff'); for anyone else it cannot be checked."""

    name = "access_list"
    ONLY = r"(?:it|that door|that gate|this door|the [a-z ]+?)"
    DOORS = [
        re.compile(rf"^(?P<r>before|until) (?P<t>{TIME}),? (?:the )?(?P<door>{NP}) opened only for (?P<g>{NP}),? and "
                   rf"{ONLY} was the only way into (?:the )?(?P<p>{NP})$", re.I),
        re.compile(rf"^(?:the )?(?P<door>{NP}) opened only for (?P<g>{NP}) (?P<r>before|until) (?P<t>{TIME}),? and "
                   rf"{ONLY} was the only way into (?:the )?(?P<p>{NP})$", re.I),
        re.compile(rf"^(?:the )?(?P<door>{NP}) was the only way into (?:the )?(?P<p>{NP}),? and (?P<r>before|until) "
                   rf"(?P<t>{TIME}),? it opened only for (?P<g>{NP})$", re.I),
    ]
    MEMBERS = re.compile(rf"^(?P<n>{NAMES}) (?:is|are) (?:both |all )?(?P<g>{NP})$")
    NOT_MEMBER = re.compile(rf"^(?P<n>{NAME}) is (?:a|an) [a-z -]+?,? (?:and )?not (?P<g>{NP})$")
    NOT_MEMBER2 = re.compile(rf"^(?P<n>{NAME}) is not (?P<g>{NP})$")
    EXCLUDES = re.compile(rf"^(?:nobody|no one) (?:on|in) (?P<r>{NP}) is (?P<g>{NP})$", re.I)

    def parse(self, ctx):
        doors, read = [], {}  # read: sentence index -> [(kind, data)]
        pats = self.DOORS + [self.NOT_MEMBER, self.NOT_MEMBER2, self.EXCLUDES, self.MEMBERS]
        nd = len(self.DOORS)
        for i, s in enumerate(ctx.sentences):
            for n, m in match_pieces(s, pats):
                if n < nd:
                    doors.append((clean_np(m.group("door")), clean_np(m.group("p")), group_key(m.group("g")),
                                  tm(m.group("t")), m.group("r").lower() == "until"))
                    read.setdefault(i, []).append(("door", None))
                elif n in (nd, nd + 1):
                    read.setdefault(i, []).append(("neg", (m.group("n"), group_key(m.group("g")))))
                elif n == nd + 2:
                    read.setdefault(i, []).append(("excl", (group_key(m.group("r")), group_key(m.group("g")))))
                else:
                    for name in re.findall(NAME, m.group("n")):
                        read.setdefault(i, []).append(("pos", (name, group_key(m.group("g")))))
        if not doors:
            return None
        # A membership sentence counts as read only if it is about a door's group, or about a group that an
        # exclusion ties to one ("Nobody on the volunteer rota is on the safety-trained list").
        groups = {d[2] for d in doors}
        excl = [x for parts in read.values() for k, x in parts if k == "excl" and x[1] in groups]
        groups |= {r for r, _ in excl}
        seen = {i for i, parts in read.items()
                if all(k == "door" or (k == "excl" and x in excl) or (k in ("pos", "neg") and x[1] in groups)
                       for k, x in parts)}
        member = {}  # (name, group) -> True/False
        for k, (n, g) in ((k, x) for i in seen for k, x in read[i] if k == "pos"):
            member[(n, g)] = True
        for k, (n, g) in ((k, x) for i in seen for k, x in read[i] if k == "neg"):
            if member.get((n, g)) is True:
                return None  # the evidence contradicts itself as read; stay out
            member[(n, g)] = False
        for r, g in excl:
            for (n, h), v in list(member.items()):
                if h == r and v:
                    if member.get((n, g)) is True:
                        return None
                    member[(n, g)] = False
        for i in seen:
            ctx.consume(i)
        return {"doors": doors, "member": member, "unread": Unread(ctx, seen)}

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None or c.actor is None:
            return None
        for door, p, g, t0, until in facts["doors"]:
            if re.search(entry_any_door_re(p), c.text):
                t = event_time(c)
                if negated(c) or t is None:
                    return None
                if facts["unread"].noun(door, p, g) or facts["unread"].person(c.actor):
                    return None
                if t > t0 or (t == t0 and not until):
                    return verdict(self, True, f"after {T.fmt_time(t0)} the {door} rule no longer applies")
                if t == t0:
                    return None
                m = facts["member"].get((c.actor, g))
                if m is None:
                    return None
                return verdict(self, m, f"before {T.fmt_time(t0)} only {g} could get into the {p}; "
                                        f"{c.actor} {'is' if m else 'is not'} {g}")
            if re.search(rf"@ (?:was )?(?:\w+ )?(?:by|beside|near|next to|outside) (?:the )?{re.escape(door)}\b",
                         c.text):
                return verdict(self, True, f"by the {door}, not inside")
        return None


# ---------------------------------------------------------------------------------------------------------------
@register
class AccessBadge(Family):
    """'Nobody could enter the staff floor except by scanning their own active badge.' plus 'Femi's badge was
    inactive before 12:40 PM and active from 12:40 PM onward.' Femi entering before 12:40 PM is a lie; people whose
    card times are not given cannot be checked. Scanning an inactive card is possible, so that is never a lie."""

    name = "access_badge"
    RULES = [
        re.compile(rf"^{NOBODY} could (?:enter|get into|go into) (?:the )?(?P<p>{NP}) (?:except|other than|unless) "
                   rf"(?:by )?(?:scanning|they scanned|swiping|using) (?:their|his or her|his|her) own active "
                   rf"(?P<card>{NP})$", re.I),
        re.compile(rf"^the only way (?:anybody|anyone|a person|someone|people) could (?:enter|get into) (?:the )?"
                   rf"(?P<p>{NP}) was by (?:scanning|swiping|using) (?:their|his or her|his|her) own active "
                   rf"(?P<card>{NP})$", re.I),
        re.compile(rf"^(?:the )?(?P<p>{NP}) opened at {TIME},? and {NOBODY} could (?:enter|get into) it (?:except|other "
                   rf"than) by (?:scanning|swiping|using) (?:their|his or her|his|her) own active (?P<card>{NP})$",
                   re.I),
    ]
    ST = r"(?P<s{k}>inactive|active|deactivated|disabled|valid|invalid|switched off|switched on)"
    REL = r"(?P<r{k}>before|until|from|after|since)"
    TWO = [  # "X's badge was inactive before T and active from T onward", "Before T X's badge was inactive; ..."
        re.compile(rf"^(?P<n>{NAME})'s (?P<card>{NP}) was {ST.format(k=1)} {REL.format(k=1)} (?P<t1>{TIME}),? "
                   rf"(?:and |then |, then |and then )?(?:it )?(?:was |became |turned )?{ST.format(k=2)} "
                   rf"{REL.format(k=2)} (?P<t2>{TIME})(?: onwards?)?$"),
        re.compile(rf"^{REL.format(k=1)} (?P<t1>{TIME}),? (?P<n>{NAME})'s (?P<card>{NP}) was {ST.format(k=1)}[;,]? "
                   rf"(?:and |then )?{REL.format(k=2)} (?P<t2>{TIME})(?: onwards?)?,? it (?:was|became) "
                   rf"{ST.format(k=2)}$", re.I),
        re.compile(rf"^(?P<n>{NAME})'s (?P<card>{NP}) was {ST.format(k=1)} {REL.format(k=1)} (?P<t1>{TIME})[;,]? "
                   rf"(?:and |then )?{REL.format(k=2)} (?P<t2>{TIME})(?: onwards?)?,? it (?:was|became) "
                   rf"{ST.format(k=2)}$"),
    ]
    ACTIVE = {"active": True, "valid": True, "switched on": True, "inactive": False, "deactivated": False,
              "disabled": False, "invalid": False, "switched off": False}
    NUMBERED = re.compile(rf"^(?:each|every) {NP} (?:is printed with a number|has a number printed on it|carries a "
                          rf"printed number)$|^a number is printed on (?:each|every) {NP}$", re.I)
    OPENED = re.compile(rf"^(?:the )?(?P<p>{NP}) opened at (?P<t>{TIME})$|^at (?P<t2>{TIME}),? (?:the )?(?P<p2>{NP}) "
                        rf"opened$", re.I)

    def parse(self, ctx):
        places, cards, opened, seen = [], {}, {}, set()
        nr, nt = len(self.RULES), len(self.TWO)
        for i, s in enumerate(ctx.sentences):
            found, ok, new = match_pieces(s, self.RULES + self.TWO + [self.NUMBERED, self.OPENED]), True, []
            for n, m in found:
                if n < nr:
                    new.append(("place", (clean_np(m.group("p")), clean_np(m.group("card")))))
                elif n == nr + nt:
                    pass  # "Each wristband is printed with a number."
                elif n > nr + nt:
                    new.append(("open", (clean_np(m.group("p") or m.group("p2")), tm(m.group("t") or m.group("t2")))))
                else:
                    s1, s2 = self.ACTIVE.get(m.group("s1").lower()), self.ACTIVE.get(m.group("s2").lower())
                    r1, r2 = m.group("r1").lower(), m.group("r2").lower()
                    t1, t2 = tm(m.group("t1")), tm(m.group("t2"))
                    if (s1 is None or s2 is None or s1 == s2 or r1 not in ("before", "until") or t1 != t2
                            or r2 not in ("from", "since", "after") or m.group("n") in cards):
                        ok = False
                        break
                    # (start, end, active, end_is_unclear): before T -> [0, T); from T -> [T, end of day)
                    new.append(("card", (m.group("n"), [(0, t1, s1, r1 == "until"), (t2, DAY_END, s2, r2 == "after")],
                                         clean_np(m.group("card")), i)))
            if not found or not ok:
                continue
            for k, x in new:
                if k == "place":
                    places.append(x)
                elif k == "open":
                    opened[x[0]] = x[1]
                else:
                    cards[x[0]] = x[1:]
            seen.add(i)
        if not places:
            return None
        rule_cards = {card for _, card in places}
        for n, (spans, card, i) in list(cards.items()):
            if card not in rule_cards:  # a different kind of card than the one the door needs: not read
                del cards[n]
                seen.discard(i)
        cards = {n: v[0] for n, v in cards.items()}
        places = [p for p, _ in places]
        for i in seen:
            ctx.consume(i)
        return {"places": places, "cards": cards, "opened": opened, "unread": Unread(ctx, seen)}

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None or c.actor is None:
            return None
        for p in facts["places"]:
            if not re.search(entry_re(p), c.text):
                continue
            t = event_time(c)
            if negated(c) or t is None or facts["unread"].noun(p) or facts["unread"].person(c.actor):
                return None
            spans = facts["cards"].get(c.actor)
            if spans is None:
                return None
            for start, end, active, unclear in spans:
                if unclear and t in (start, end):
                    return None
                if start <= t < end:
                    if active and t < facts["opened"].get(p, 0):
                        return None  # an active card, but before the place opened
                    return verdict(self, active, f"{c.actor}'s card was {'active' if active else 'inactive'} at "
                                                 f"{T.fmt_time(t)}")
            return None
        return None


# ---------------------------------------------------------------------------------------------------------------
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10}


def name_times(items):
    """'Ivan at 8:35 AM, Mohan at 9:25 AM and Gabriel at 1:45 PM' -> [('Ivan', 515), ...], or None if any item
    is not 'Name at time'."""
    parts = re.split(r",\s+(?:and\s+)?|\s+and\s+", items.strip().rstrip("."))
    out = []
    for p in parts:
        m = re.fullmatch(rf"({NAME})(?: was logged| was recorded| arrived| first arrived| came| got there)? at ({TIME})",
                         p.strip())
        if not m:
            return None
        out.append((m.group(1), tm(m.group(2))))
    return out


@register
class AccessLog(Family):
    """'The archive badge log accurately records every entry to the archive that day. It shows exactly two
    entries: Joel at 9:30 AM and Kiran at 11:45 AM.' The log is complete, so an entry it does not list is a lie,
    and so is denying a listed entry. Exit logs ('... records every time someone left the clinic that day') work
    the same way for leaving. Being inside after one's own logged entry is fine; being inside with no entry might
    still be true (from the day before), so it is left alone."""

    name = "access_log"
    LIST = r"exactly (?P<k>\w+) (?P<kind>entries|entry|exits|exit)"
    PATS = [
        re.compile(rf"^(?:that day,? )?(?:the )?{NP} accurately records every (?:(?P<e>entry to|entry into|time someone "
                   rf"entered|time someone went into)|(?P<x>exit from|time someone left|time someone exited|departure "
                   rf"from)) (?:the )?(?P<p>{NP})(?: that day)?(?: and (?:is checked every evening|(?:shows|lists) "
                   rf"{LIST}: (?P<items>.+)))?$", re.I),
        re.compile(rf"^(?:it |the (?:{NP} )?log )?(?:shows|lists|contains) {LIST}: (?P<items>.+)$", re.I),
        re.compile(rf"^{LIST} appears? (?:in|on) it: (?P<items>.+)$", re.I),
        re.compile(rf"^(?:the )?{NP} has (?:a single|one|only one) door$|^(?:the )?{NP} is checked every evening$|"
                   rf"^every evening (?:the )?{NP} is checked$|^someone checks (?:the )?{NP} every evening$", re.I),
    ]

    def parse(self, ctx):
        logs, seen = [], set()  # logs: [kind, place, entries or None]
        for i, s in enumerate(ctx.sentences):
            found, ok = match_pieces(s, self.PATS), True
            for n, m in found:
                if n == 0:
                    logs.append(["exit" if m.group("x") else "entry", clean_np(m.group("p")), None])
                if n in (0, 1, 2) and m.group("items"):
                    kind = "exit" if m.group("kind").lower().startswith("exit") else "entry"
                    items = name_times(m.group("items"))
                    k = NUMBER_WORDS.get(m.group("k").lower()) or (int(m.group("k")) if m.group("k").isdigit()
                                                                   else None)
                    if not logs or logs[-1][2] is not None or kind != logs[-1][0] or items is None or k != len(items):
                        ok = False
                        break
                    logs[-1][2] = items
            if found and ok:
                seen.add(i)
        logs = [tuple(lg) for lg in logs if lg[2] is not None]
        if not logs:
            return None
        for i in seen:
            ctx.consume(i)
        return {"logs": logs, "unread": Unread(ctx, seen)}

    NEVER = r"(?: at all| that day| all day| today)?$"  # the whole denial, nothing after the place
    EXCEPT = re.compile(r"\b(?:except|apart from|other than|besides|but|only|briefly|unless|save for|aside from)\b")

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None or c.actor is None:
            return None
        for kind, p, items in facts["logs"]:
            mine = [t for n, t in items if n == c.actor]
            pr = place_re(p)
            if not re.search(pr, c.text):
                continue
            if facts["unread"].noun(p) or facts["unread"].person(c.actor):
                return None
            if kind == "entry":
                return self.judge_entry(c, p, pr, mine)
            return self.judge_exit(c, p, pr, mine)
        return None

    def judge_entry(self, c, p, pr, mine):
        went = re.search(entry_re(p), c.text) or re.search(rf"@ {AUX}set foot in {pr}", c.text)
        if went and negated(c):
            if self.EXCEPT.search(c.text):
                return None
            sp = span(c)
            if sp:
                inside = [t for t in mine if sp[0] <= t <= sp[1]]
                return verdict(self, not inside, f"log entries for {c.actor}: {[T.fmt_time(t) for t in mine]}")
            if not c.times and re.search(rf"@ {AUX}(?:{ENTER}|set foot in) {pr}{self.NEVER}", c.text):
                return verdict(self, not mine, f"log entries for {c.actor}: {[T.fmt_time(t) for t in mine]}")
            return None
        if went:
            if re.search(r"\bbefore noon\b", c.text) and not c.times:
                return verdict(self, any(t < 720 for t in mine), f"{c.actor} entries {self.fmt(mine)}")
            if re.search(r"\b(?:after noon|in the afternoon)\b", c.text) and not c.times:
                return verdict(self, any(t > 720 for t in mine), f"{c.actor} entries {self.fmt(mine)}")
            t = event_time(c)
            if t is None:
                return None
            return verdict(self, t in mine, f"log shows {c.actor} entering the {p} at {self.fmt(mine)}")
        if re.search(rf"@ (?:was|were) (?:still )?(?:in|inside) {pr}", c.text) and not negated(c):
            t = event_time(c)
            if t is not None and any(e <= t for e in mine):
                return verdict(self, True, f"{c.actor} entered at {self.fmt(mine)}")
        return None

    LEAVE = (r"(?:leave|left|leaving|exit|exited|exiting|step out of|stepped out of|go out of|went out of|walk out of|"
             r"walked out of)")

    def judge_exit(self, c, p, pr, mine):
        left = re.search(rf"@ {AUX}{self.LEAVE} {pr}", c.text) or \
            re.search(rf"\bdid @ (?:leave|exit|step out of|go out of|walk out of) {pr}", c.text)
        if not left:
            return None
        if negated(c):
            if self.EXCEPT.search(c.text):
                return None
            sp = span(c)
            if sp:
                inside = [t for t in mine if sp[0] <= t <= sp[1]]
                return verdict(self, not inside, f"exit log for {c.actor}: {self.fmt(mine)}")
            if not c.times and re.search(rf"@ {AUX}{self.LEAVE} {pr}{self.NEVER}", c.text):
                return verdict(self, not mine, f"exit log for {c.actor}: {self.fmt(mine)}")
            return None
        t = event_time(c)
        if t is None:
            return None
        return verdict(self, t in mine, f"exit log for {c.actor}: {self.fmt(mine)}")

    @staticmethod
    def fmt(ts):
        return ", ".join(T.fmt_time(t) for t in ts) or "none"


# ---------------------------------------------------------------------------------------------------------------
@register
class AccessArrival(Family):
    """First arrivals, either logged ('The door log accurately records when each of these four people first arrived
    at the depot that day: Emeka at 9:50 AM, ...') or chained ('Wei first arrived at the workshop at 8:15 AM.
    Exactly 15 minutes after Wei, Sara first arrived.'). Checks 'my first arrival was at T', 'I first got there
    before/after X', 'I got there at T' (a lie only if T is earlier than the first arrival) and 'I was the first
    person in'. Only the listed people are known, so being the earliest of them never proves anything."""

    name = "access_arrival"
    LOG = re.compile(rf"^(?:that day,? )?(?:the )?{NP} accurately records (?:when|the time) each of these "
                     rf"(?:\w+ )?people first arrived at (?:the )?(?P<p>{NP})(?: that day)?(?:: (?P<items>.+))?$", re.I)
    LOG2 = re.compile(rf"^for each of these (?:\w+ )?people,? (?:the )?{NP} accurately records when they first arrived "
                      rf"at (?:the )?(?P<p>{NP})(?: that day)?: (?P<items>.+)$", re.I)
    LOG3 = re.compile(rf"^(?P<items>.+?) are the times when each of these (?:\w+ )?people first arrived at (?:the )?"
                      rf"(?P<p>{NP})(?: that day)?,? and the {NP} accurately records them$", re.I)
    ITEMS = re.compile(rf"^(?P<items>{NAME} (?:was logged|was recorded|came|arrived|first arrived|got there) at .+)$")
    ABS = [  # found anywhere in a sentence, between commas: "Elena first arrived at the workshop at 8:35 AM, ..."
        re.compile(rf"(?:^|, (?:and )?)(?P<n>{NAME}) first arrived at (?:the )?(?P<p>{NP}) at (?P<t>{TIME})(?=,|$)"),
        re.compile(rf"(?:^|, (?:and )?)[Aa]t (?P<t>{TIME}),? (?P<n>{NAME}) first arrived at (?:the )?(?P<p>{NP})"
                   rf"(?=,|$)"),
        re.compile(rf"(?:^|, (?:and )?)(?P<n>{NAME})'s first arrival at (?:the )?(?P<p>{NP}) was at (?P<t>{TIME})"
                   rf"(?=,|$)"),
    ]
    REL = [
        re.compile(rf"(?:^|, (?:and )?)(?P<n>{NAME}) first arrived exactly (?P<k>\d+) minutes (?P<d>before|after) "
                   rf"(?P<m>{NAME})(?: did)?(?=,|$)"),
        re.compile(rf"(?:^|, (?:and )?)[Ee]xactly (?P<k>\d+) minutes (?P<d>before|after) (?P<m>{NAME}),? "
                   rf"(?P<n>{NAME}) first arrived(?=,|$)"),
    ]

    def parse(self, ctx):
        place, known, rels, logged, seen = None, {}, [], False, set()
        for i, s in enumerate(ctx.sentences):
            body = s.rstrip(".")
            m = self.LOG.match(body) or self.LOG2.match(body) or self.LOG3.match(body)
            if m:
                place = clean_np(m.group("p"))
                logged = True
                if m.group("items"):
                    items = name_times(m.group("items"))
                    if items is None:
                        return None
                    known.update(items)
                seen.add(i)
                continue
            m = self.ITEMS.match(body)
            if m and logged and not known:
                items = name_times(m.group("items"))
                if items:
                    known.update(items)
                    seen.add(i)
                continue
            # chain clauses ("X first arrived exactly 15 minutes after Y, and ...") must tile the whole sentence
            hits = sorted(((mm.start(), mm.end(), kind, mm) for kind, pats in (("abs", self.ABS), ("rel", self.REL))
                           for pat in pats for mm in pat.finditer(body)), key=lambda h: h[0])
            pos = 0
            for start, end, _, _ in hits:
                if start != pos:
                    break
                pos = end
            if not hits or pos != len(body):
                continue
            for _, _, kind, mm in hits:
                if kind == "abs":
                    p = clean_np(mm.group("p"))
                    if place not in (None, p):
                        return None
                    place = p
                    known[mm.group("n")] = tm(mm.group("t"))
                else:
                    k = int(mm.group("k")) * (1 if mm.group("d") == "after" else -1)
                    rels.append((mm.group("n"), mm.group("m"), k))
            seen.add(i)
        if place is None or not known:
            return None
        for i in seen:
            ctx.consume(i)
        changed = True
        while changed:  # n first arrived k minutes after m
            changed = False
            for n, m, k in rels:
                if m in known and n not in known:
                    known[n] = known[m] + k
                    changed = True
                elif n in known and m not in known:
                    known[m] = known[n] - k
                    changed = True
                elif n in known and m in known and known[n] != known[m] + k:
                    return None  # misread evidence; stay out
        return {"place": place, "first": known, "unread": Unread(ctx, seen)}

    ARRIVE = (r"(?:arrived at|arrive at|got to|get to|reached|reach|showed up at|show up at|made it to|make it to|"
              r"came to|come to|turned up at|turn up at)")
    ENTERED = r"(?:got in to|got into|walked into|came into|entered)"  # an arrival only with "first"
    BEFORE = r"(?:before|ahead of|earlier than|sooner than)"
    AFTER = r"(?:after|later than|behind)"
    TAIL = r"(?:'s(?: first arrival)?)?(?: did| had| arrived| got there| first arrived| did so)?$"

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None or c.actor is None:
            return None
        p, first = facts["place"], facts["first"]
        pr = place_re(p)
        if not re.search(pr, c.text) or negated(c):
            return None
        me = first.get(c.actor)
        if me is None:
            return None
        named = [n for w, n in c.names.items() if re.search(rf"\b{re.escape(w)}\b", c.text)]
        if facts["unread"].noun(p) or facts["unread"].person(c.actor, *named):
            return None
        if re.search(rf"@ was the (?:very )?first (?:person|one|to arrive|to get there)?\s*(?:in|into|at|to arrive at|"
                     rf"to get to|to reach)? {pr}(?: that day| today| this morning)?$", c.text) \
                or re.search(rf"@ (?:got to|reached|arrived at) {pr} first$", c.text):
            others = [t for n, t in first.items() if n != c.actor]
            if any(t < me for t in others):
                return verdict(self, False, f"{c.actor} first arrived at {T.fmt_time(me)}, someone listed was earlier")
            if all(t > me for t in others):
                return verdict(self, True, f"{c.actor} is the earliest of the listed people")
            return None
        arrival = (re.search(rf"@ {AUX}(?:first )?{self.ARRIVE} {pr}", c.text)
                   or re.search(rf"@ {AUX}first {self.ENTERED} {pr}", c.text)
                   or re.search(rf"@'s first arrival at {pr}", c.text)
                   or re.search(rf"\bfirst time @ {AUX}{self.ARRIVE} {pr}", c.text))
        if not arrival:
            return None
        is_first = bool(re.search(r"\bfirst\b", c.text))
        m = (re.search(rf"\b(?P<rel>{self.BEFORE}|{self.AFTER}) (?P<who>[a-z]+){self.TAIL}", c.text)
             or re.search(rf"^(?P<rel>{self.BEFORE}|{self.AFTER}) (?P<who>[a-z]+)(?: is when)? @ ", c.text))
        if not m and re.search(rf"\b(?:{self.BEFORE}|{self.AFTER})\b", c.text):
            return None  # an order against something we do not model ("after Mira left")
        if m and m.group("who") in c.names:
            other = first.get(c.names[m.group("who")])
            if other is None or c.times:
                return None
            before = re.fullmatch(self.BEFORE, m.group("rel")) is not None
            ok = me < other if before else me > other
            return verdict(self, ok, f"first arrivals: {c.actor} {T.fmt_time(me)}, {c.names[m.group('who')]} "
                                     f"{T.fmt_time(other)}")
        if m:
            return None
        if len(c.times) != 1 or c.times[0][1] not in ("at", "was") or JOINED_RE.search(c.text):
            return None
        t = c.times[0][0]
        # Without "first" ("I got to the office at 9:55 AM") train always means the first arrival: innocents give
        # exactly the logged time (27 of 27), liars an earlier or later one.
        return verdict(self, t == me, f"{c.actor} first arrived at {T.fmt_time(me)}"
                                      f"{'' if is_first else ' (arrival read as the first one)'}")


# ---------------------------------------------------------------------------------------------------------------
@register
class AccessPresence(Family):
    """'From 12:55 PM to 4:55 PM, only Ivan and Carmen were at the ticket booth. Nobody else was at the ticket booth
    during that time.' Anyone else said to be at the ticket booth inside the window is a lie. Both ends of 'between'
    count; the ends of 'from T1 to T2' are left alone. 'Yuki was the only cashier on duty that day' lets 'I saw the
    cashier at ...' name Yuki."""

    name = "access_presence"
    WIN = (rf"(?:from (?P<a>{TIME}) (?:to|until) (?P<b>{TIME})|between (?P<c>{TIME}) and (?P<d>{TIME}))")
    PATS = [
        re.compile(rf"^{WIN},? only (?P<n>{NAMES}) (?:was|were) at (?:the )?(?P<p>{NP})$", re.I),
        re.compile(rf"^only (?P<n>{NAMES}) (?:was|were) at (?:the )?(?P<p>{NP}) {WIN}$", re.I),
    ]
    ELSE = re.compile(rf"^(?:and )?{NOBODY} else was (?:at (?:the )?(?P<p>{NP})|there) (?:during|in|at|throughout) "
                      rf"(?:that|this|the same) (?:time|period)$|^(?:and )?{NOBODY} else was (?:at (?:the )?{NP}|there) "
                      rf"then$", re.I)
    ROLE = re.compile(rf"^(?P<n>{NAME}) was the only (?P<r>[a-z -]+?) on duty(?: that day)?$")

    def parse(self, ctx):
        wins, roles, seen = [], {}, set()
        for i, s in enumerate(ctx.sentences):
            found = match_pieces(s, self.PATS + [self.ELSE, self.ROLE])
            for n, m in found:
                if n < len(self.PATS):
                    names = set(re.findall(NAME, m.group("n")))
                    t1, t2 = tm(m.group("a") or m.group("c")), tm(m.group("b") or m.group("d"))
                    wins.append((clean_np(m.group("p")), names, t1, t2, m.group("c") is not None))
                elif n > len(self.PATS):
                    roles[m.group("r").lower()] = m.group("n")
            if found:
                seen.add(i)
        if not wins:
            return None
        for i in seen:
            ctx.consume(i)
        return {"wins": wins, "roles": roles, "unread": Unread(ctx, seen)}

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None:
            return None
        for p, names, t1, t2, between in facts["wins"]:
            pr = place_re(p)
            if not re.search(pr, c.text) or negated(c):
                continue
            actor = c.actor
            m = re.search(rf"@ (?:saw|spotted|noticed) the (?P<r>[a-z -]+?) (?:at|by) {pr}", c.text)
            if m:
                actor = facts["roles"].get(m.group("r"))
                text = "@ at " + c.text[m.end("r") + 4:]
            else:
                text = c.text
            if actor is None:
                return None
            if not (re.search(rf"@ (?:was |were )?(?:standing |sitting |working |waiting |stationed |posted |"
                              rf"over |down |right |still )?at {pr}", text)
                    or re.search(rf"\b{pr} is where @ was\b", text)):
                continue
            t = event_time(c)
            if t is None or (actor not in names and not between and t in (t1, t2)):
                return None
            if facts["unread"].noun(p) or facts["unread"].person(actor):
                return None
            inside = t1 <= t <= t2
            ok = actor in names or not inside
            return verdict(self, ok, f"only {', '.join(sorted(names))} at the {p} {T.fmt_time(t1)}-{T.fmt_time(t2)}; "
                                     f"{actor} at {T.fmt_time(t)}")
        return None


# ---------------------------------------------------------------------------------------------------------------
@register
class AccessBooking(Family):
    """'The main hall was booked from 11:35 AM to 4:05 PM: the sales team had it until 1:05 PM, the finance team
    until 2:35 PM, and then the research team. During each booking, only that team was in the room.' (or 'booked
    without gaps from 10:10 AM: the finance team for 1 hour 15 minutes, then ...'). Another team in the room during
    a booking is a lie; outside the bookings anything goes; times on a changeover are left alone."""

    name = "access_booking"
    ROOM = r"(?P<room>room \d+|[a-z][a-z ]*?)"
    SPAN = [
        re.compile(rf"^(?:the )?{ROOM} was (?:booked|reserved) from (?P<s>{TIME}) to (?P<e>{TIME})(?::|,) (?P<l>.+)$",
                   re.I),
        re.compile(rf"^from (?P<s>{TIME}) to (?P<e>{TIME}),? (?:the )?{ROOM} was (?:booked|reserved)(?::|,) (?P<l>.+)$",
                   re.I),
        re.compile(rf"^the booking for (?:the )?{ROOM} ran from (?P<s>{TIME}) to (?P<e>{TIME})(?::|,) (?P<l>.+)$", re.I),
    ]
    GAPLESS = [
        re.compile(rf"^(?:the )?{ROOM} was (?:booked|reserved) without gaps from (?P<s>{TIME})(?::|,) (?P<l>.+)$", re.I),
        re.compile(rf"^(?:from|starting at|starting from) (?P<s>{TIME}),? (?:the )?{ROOM} was (?:booked|reserved) "
                   rf"without gaps(?::|,) (?P<l>.+)$", re.I),
    ]
    ONLY = re.compile(r"^(?:(?:during|throughout) each booking|while each booking lasted|whenever a team had the "
                      r"booking),? only (?:that|the booked) team was in the room$|^only (?:that|the booked) team was in "
                      r"the room (?:during|throughout) each booking$", re.I)
    STEP = r"(?:first |then |next |and then |and )?(?:by |with )?(?:the )?(?P<team>[a-z]+) team"
    UNTIL = re.compile(rf"{STEP}(?: had it| holding it| had the room)? until (?P<t>{TIME})", re.I)
    LAST = re.compile(r"(?:and )?(?:then|next|finally)(?: by)? (?:the )?(?P<team>[a-z]+) team", re.I)
    FOR = re.compile(rf"{STEP} for (?P<d>.+)", re.I)

    def slots_until(self, start, end, items):
        """'the sales team until 1:05 PM, the finance team until 2:35 PM, and then the research team' -> slots."""
        parts = re.split(r",\s+", items)
        slots, t0 = [], start
        for k, part in enumerate(parts):
            m = self.UNTIL.fullmatch(part)
            if m and k < len(parts) - 1:
                t = tm(m.group("t"))
                slots.append((t0, t, m.group("team").lower()))
                t0 = t
                continue
            m = self.LAST.fullmatch(part)
            if not m or k != len(parts) - 1:
                return None
            slots.append((t0, end, m.group("team").lower()))
        return slots if len(slots) >= 2 and all(a < b for a, b, _ in slots) else None

    def slots_for(self, start, items):
        """'the finance team for 1 hour 30 minutes, then the sales team for 45 minutes, ...' -> slots."""
        slots, t0 = [], start
        for part in re.split(r",\s+", items):
            m = self.FOR.fullmatch(part)
            d = T.find_durations(m.group("d")) if m else []
            if len(d) != 1 or d[0][1].group(0) != m.group("d"):
                return None
            slots.append((t0, t0 + d[0][0], m.group("team").lower()))
            t0 += d[0][0]
        return slots if len(slots) >= 2 else None

    def parse(self, ctx):
        rooms, exclusive, seen = [], False, set()
        for i, s in enumerate(ctx.sentences):
            body = s.rstrip(".")
            if self.ONLY.match(body):
                exclusive = True
                seen.add(i)
                continue
            m = next((m for m in (p.match(body) for p in self.SPAN) if m), None)
            slots = self.slots_until(tm(m.group("s")), tm(m.group("e")), m.group("l")) if m else None
            if not m:
                m = next((m for m in (p.match(body) for p in self.GAPLESS) if m), None)
                slots = self.slots_for(tm(m.group("s")), m.group("l")) if m else None
            if slots:
                rooms.append((m.group("room").lower(), slots))
                seen.add(i)
        if not rooms or not exclusive:
            return None
        for i in seen:
            ctx.consume(i)
        return {"rooms": rooms, "unread": Unread(ctx, seen)}

    PV = (r"(?:was|were|met|met up|had|held|sat|gathered|convened|got together|was meeting|were meeting|was sitting|"
          r"were sitting|was present|were present|was having|was holding|was working|were working)")
    THERE = r"(?:met|met up|was present|were present|was meeting|were meeting|gathered|sat|convened|was there|were there)"
    AWAY = re.compile(r"\b(?:past|outside|left|leave|leaving|booked|book|reserved|cancel\w*|clean\w*|near|beside|"
                      r"waiting|walked|walk|passed|toward|towards|empty|vacated|locked|instead|moved|arrived|came|"
                      r"went|headed|after|before|until|since|from|between)\b")

    def team_in(self, text, rr):
        """The team the claim puts in the room, or None."""
        if len(re.findall(r"\bteams?\b", text)) != 1 or self.AWAY.search(text):
            return None
        pats = [
            rf"\b(?P<team>[a-z]+) team {self.PV}(?: (?:its|their|a|the) meeting| a meeting| meeting| together)? "
            rf"(?:in|inside) {rr}",
            rf"\b{rr} had (?:the )?(?P<team>[a-z]+) team in it\b",
            rf"\b(?P<team>[a-z]+) team (?:had|occupied|was using|were using|used) {rr}",
            rf"^(?:at \S+ [ap]m )?(?:in|inside) {rr} (?:\S+ ){{0,4}}?(?:the )?(?P<team>[a-z]+) team (?:{self.THERE}|"
            rf"held its meeting|had its meeting)\b",
            rf"\bit was the (?P<team>[a-z]+) team (?:in|inside) {rr}",
            rf"\b(?P<team>[a-z]+) team's meeting (?:in|inside) {rr} was\b",
            rf"\b{rr} (?:hosted (?:the )?(?P<team>[a-z]+) team's meeting|saw (?:the )?(?P<team2>[a-z]+) team meet)\b",
            rf"^(?:at \S+ [ap]m )?(?:in|inside) {rr} (?:\S+ ){{0,4}}?was the (?P<team>[a-z]+) team$",
        ]
        for p in pats:
            m = re.search(p, text)
            if m:
                return m.groupdict().get("team") or m.groupdict().get("team2")
        return None

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None or negated(c):
            return None
        for room, slots in facts["rooms"]:
            rr = rf"(?:the )?{re.escape(room)}(?:{NP_END}|(?= (?:hosted|saw)\b))"
            team = self.team_in(c.text, rr)
            if team is None:
                continue
            t = event_time(c)
            if t is None or facts["unread"].noun(room, f"{team} team", *(f"{o} team" for _, _, o in slots)):
                return None
            first, last = slots[0][0], slots[-1][1]
            if t < first or t > last:
                return verdict(self, True, f"no booking of the {room} at {T.fmt_time(t)}")
            if t in {b for _, b, _ in slots}:
                return None  # a changeover, or the very end
            owner = next(o for a, b, o in slots if a <= t < b)
            return verdict(self, team == owner, f"the {room} was booked by the {owner} team at {T.fmt_time(t)}")
        return None


# ---------------------------------------------------------------------------------------------------------------
# Self-check on made-up wordings and nouns that train does not use:
#   python -c "import lc.rules.fam_access as f; raise SystemExit(f._selfcheck())"
_CHECKS = [  # (evidence, sender, message, expected status or None)
    ("Nobody could get into the boiler room between 1:00 PM and 2:00 PM.", "A",
     "For the record, at 1:30 PM I went into the boiler room.", LIE),
    ("Nobody could get into the boiler room between 1:00 PM and 2:00 PM.", "A", "I went into the boiler room at 2:00 PM.",
     LIE),
    ("Nobody could get into the boiler room between 1:00 PM and 2:00 PM.", "A", "I went into the boiler room at 2:30 PM.",
     OK),
    ("Nobody could get into the boiler room between 1:00 PM and 2:00 PM.", "A",
     "I was waiting outside the boiler room at 1:30 PM.", OK),
    ("Nobody could get into the boiler room between 1:00 PM and 2:00 PM.", "A",
     "I almost went into the boiler room at 1:30 PM.", None),
    ("Nobody could get into the boiler room between 1:00 PM and 2:00 PM.", "A",
     "Dan, I saw you slip into the boiler room at 1:45 PM.", LIE),
    ("The printer could print only while the network was up. The network was down from 10:00 AM to 11:00 AM.", "B",
     "I printed my boarding pass on the printer at 10:30 AM.", LIE),
    ("The printer could print only while the network was up. The network was down from 10:00 AM to 11:00 AM.", "B",
     "I fixed the printer at 10:30 AM.", None),
    ("The printer could print only while the network was up. The network was down from 10:00 AM to 11:00 AM.", "B",
     "I printed my boarding pass on the printer at 11:30 AM.", OK),
    ("The stage lamps could be lit only while the generator was running. The generator was stopped from 2:00 PM to "
     "3:00 PM.", "C", "The stage lamps were on the floor at 2:30 PM.", None),
    ("The stage lamps could be lit only while the generator was running. The generator was stopped from 2:00 PM to "
     "3:00 PM.", "C", "Honestly, the stage lamps were glowing at 2:30 PM.", LIE),
    ("The coffee machine could make coffee only while the water supply was on. The water supply was shut off from "
     "11:15 AM to 11:45 AM.", "C", "I made a coffee at 11:30 AM before cleaning the machine.", None),
    ("Every remote sign-in to the archive server required the VPN. The VPN service was offline from 9:00 AM to "
     "9:30 AM.", "D", "I signed in locally to the archive server at 9:10 AM.", OK),
    ("Until 10:00 AM, only code 1111 could unlock the shed gate. From 10:00 AM onward, only code 2222 could unlock it.",
     "A", "I opened the shed gate with code 1111 at 10:30 AM.", LIE),
    ("Until 10:00 AM, only code 1111 could unlock the shed gate. From 10:00 AM onward, only code 2222 could unlock it.",
     "A", "I tried code 1111 on the shed gate at 10:30 AM.", None),
    ("Until 10:00 AM, only code 1111 could unlock the shed gate. From 10:00 AM onward, only code 2222 could unlock it.",
     "A", "I unlocked the shed gate with code 2222 at 10:15 AM.", OK),
    ("Until 10:00 AM, only code 1111 could unlock the shed gate. From 10:00 AM onward, only code 2222 could unlock it.",
     "A", "I unlocked the shed gate with code 1111 at 10:00 AM.", None),
    ("Before 8:00 AM the boathouse door opened only for registered rowers, and it was the only way into the boathouse. "
     "Ana and Ben are registered rowers. Cleo is a guest, not a registered rower.", "C",
     "At 7:30 AM I came into the boathouse.", LIE),
    ("Before 8:00 AM the boathouse door opened only for registered rowers, and it was the only way into the boathouse. "
     "Ana and Ben are registered rowers. Cleo is a guest, not a registered rower.", "D",
     "At 7:30 AM I came into the boathouse.", None),
    ("Before 8:00 AM the boathouse door opened only for registered rowers, and it was the only way into the boathouse. "
     "Ana and Ben are registered rowers. Cleo is a guest, not a registered rower.", "C",
     "At 8:30 AM I came into the boathouse.", OK),
    ("Nobody could enter the darkroom except by scanning their own active pass. Ana's pass was inactive before 9:00 AM "
     "and active from 9:00 AM onward.", "B", "I saw Ana walk into the darkroom at 8:30 AM.", LIE),
    ("Nobody could enter the darkroom except by scanning their own active pass. Ana's pass was inactive before 9:00 AM "
     "and active from 9:00 AM onward.", "A", "I scanned my pass at 8:30 AM.", None),
    ("The darkroom entry log accurately records every entry to the darkroom that day. It shows exactly one entry: Ana "
     "at 9:00 AM.", "B", "I went into the darkroom at 9:00 AM.", LIE),
    ("The darkroom entry log accurately records every entry to the darkroom that day. It shows exactly one entry: Ana "
     "at 9:00 AM.", "B", "I never went into the darkroom at all.", OK),
    ("The darkroom entry log accurately records every entry to the darkroom that day. It shows exactly one entry: Ana "
     "at 9:00 AM.", "B", "I was in the darkroom at 10:00 AM.", None),
    ("The gate log accurately records when each of these four people first arrived at the depot that day: Ana at "
     "8:00 AM, Ben at 8:10 AM, Cleo at 8:20 AM and Dan at 8:30 AM.", "B", "I was the first person at the depot.", LIE),
    ("The gate log accurately records when each of these four people first arrived at the depot that day: Ana at "
     "8:00 AM, Ben at 8:10 AM, Cleo at 8:20 AM and Dan at 8:30 AM.", "A", "I was the first person at the depot.", OK),
    ("The gate log accurately records when each of these four people first arrived at the depot that day: Ana at "
     "8:00 AM, Ben at 8:10 AM, Cleo at 8:20 AM and Dan at 8:30 AM.", "C", "I first arrived at the depot before Ben.",
     LIE),
    ("Only Ana and Ben were at the kiosk from 1:00 PM to 3:00 PM.", "C", "I was at the kiosk at 2:00 PM.", LIE),
    ("Only Ana and Ben were at the kiosk from 1:00 PM to 3:00 PM.", "A", "I was at the kiosk at 2:00 PM.", OK),
    ("Only Ana and Ben were at the kiosk from 1:00 PM to 3:00 PM.", "C", "I was near the kiosk at 2:00 PM.", None),
    ("Only Ana and Ben were at the kiosk from 1:00 PM to 3:00 PM.", "C", "I was at the kiosk at 3:00 PM.", None),
    ("The lab was booked from 9:00 AM to 12:00 PM: the biology team until 10:00 AM, the chemistry team until 11:00 AM, "
     "then the physics team. During each booking, only that team was in the room.", "D",
     "The physics team was in the lab at 9:30 AM.", LIE),
    ("The lab was booked from 9:00 AM to 12:00 PM: the biology team until 10:00 AM, the chemistry team until 11:00 AM, "
     "then the physics team. During each booking, only that team was in the room.", "D",
     "The physics team walked past the lab at 9:30 AM.", None),
    ("The lab was booked from 9:00 AM to 12:00 PM: the biology team until 10:00 AM, the chemistry team until 11:00 AM, "
     "then the physics team. During each booking, only that team was in the room.", "D",
     "The physics team met in the lab at 1:00 PM.", OK),
    ("The lab was booked from 9:00 AM to 12:00 PM: the biology team until 10:00 AM, the chemistry team until 11:00 AM, "
     "then the physics team. During each booking, only that team was in the room.", "D",
     "The physics team was in the lab corridor at 9:30 AM.", None),
    # Sentences with a part we do not model are not read, and anything an unread sentence mentions is left alone.
    ("Nobody could enter the vault between 9:00 AM and 11:00 AM except staff with a pass.", "A",
     "I entered the vault at 10:00 AM.", None),
    ("Nobody could enter the vault between 9:00 AM and 11:00 AM and between 2:00 PM and 3:00 PM.", "A",
     "I entered the vault at 10:00 AM.", None),
    ("Nobody could enter the vault between 9:00 AM and 11:00 AM. The manager could open the vault at any time.", "A",
     "I entered the vault at 10:00 AM.", None),
    ("The lift could run only while the main power was on, and the main power was off from 1:00 PM to 2:00 PM on "
     "weekdays.", "B", "I rode the lift at 1:30 PM.", None),
    ("Only code 1111 could unlock the shed gate before 10:00 AM, unless the alarm was off.", "A",
     "I unlocked the shed gate with code 2222 at 9:00 AM.", None),
    ("Only Ana and Ben were at the kiosk from 1:00 PM to 3:00 PM. Cleo covered the kiosk during lunch.", "C",
     "I was at the kiosk at 2:00 PM.", None),
    ("Nobody could enter the darkroom except by scanning their own active pass. Ana's pass was inactive before 9:00 AM "
     "and active from 9:00 AM onward. Ana borrowed a spare pass at 8:00 AM.", "B",
     "I saw Ana walk into the darkroom at 8:30 AM.", None),
    # Wordings that look like a match but are about something else, another day, or a report.
    ("Nobody could enter the vault between 9:00 AM and 11:00 AM.", "A", "I went into the vault corridor at 10:00 AM.",
     None),
    ("Nobody could enter the vault between 9:00 AM and 11:00 AM.", "A", "Yesterday I went into the vault at 10:00 AM.",
     None),
    ("Nobody could enter the vault between 9:00 AM and 11:00 AM.", "A", "Ben thinks I went into the vault at 10:00 AM.",
     None),
    ("Nobody could enter the vault between 9:00 AM and 11:00 AM.", "A",
     "I left the office at 9:40 AM and went into the vault.", None),
    ("The lift could run only while the main power was on. The main power was off from 1:00 PM to 2:00 PM.", "B",
     "I took the lift key to the office at 1:30 PM.", None),
    ("The lift could run only while the main power was on. The lift lobby lights were off from 1:00 PM to 2:00 PM.",
     "B", "I rode the lift at 1:30 PM.", None),
    ("The stage lamps could be lit only while the generator was running. The generator was stopped from 2:00 PM to "
     "3:00 PM.", "C", "The reading lamps were on at 2:30 PM.", None),
    ("The coffee machine could make coffee only while the water supply was on. The water supply was shut off from "
     "11:15 AM to 11:45 AM.", "C", "I made a coffee in the machine room at 11:30 AM.", None),
    ("Before 10:40 AM, only code 9230 could unlock the back gate.", "A",
     "I went back and unlocked the storeroom door with code 1311 at 10:00 AM.", None),
    ("Before 10:40 AM, only code 9230 could unlock the storeroom door.", "A",
     "I unlocked the storeroom door at 10:00 AM after Ben used code 1311.", None),
    ("The vault access log accurately records every entry to the vault that day. It shows exactly one entry: Ana at "
     "2:00 PM.", "A", "I never went into the vault before noon.", None),
    ("The clinic exit log accurately records every time someone left the clinic that day. It shows exactly one exit: "
     "Ana at 12:30 PM.", "A", "I didn't leave the clinic between 10:00 AM and 2:00 PM except for lunch.", None),
    ("The gate log accurately records when each of these four people first arrived at the depot that day: Ana at "
     "8:00 AM, Ben at 8:10 AM, Cleo at 8:20 AM and Dan at 8:30 AM.", "B",
     "I was the first person in the depot to make coffee.", None),
    ("The gate log accurately records when each of these four people first arrived at the depot that day: Ana at "
     "8:00 AM, Ben at 8:10 AM, Cleo at 8:20 AM and Dan at 8:30 AM.", "B", "I walked into the depot at 2:00 PM.", None),
    ("The gate log accurately records when each of these four people first arrived at the depot that day: Ana at "
     "8:00 AM, Ben at 8:10 AM, Cleo at 8:20 AM and Dan at 8:30 AM.", "A",
     "I first arrived at the depot after Ben left.", None),
]


def _selfcheck():
    from lc.data import Case, Message
    from .core import solve
    fams = [f for f in __import__("lc.rules", fromlist=["REGISTRY"]).REGISTRY if f.name.startswith("access_")]
    speakers = {"A": "Ana", "B": "Ben", "C": "Cleo", "D": "Dan"}
    bad = 0
    for ev, label, text, want in _CHECKS:
        res = solve(Case("x", "", speakers, ev, [Message(1, label, text)]), fams)
        got = [v.status for v in res.verdicts.get(1, [])]
        status = LIE if LIE in got else OK if got else None
        if status != want:
            bad += 1
            print(f"FAIL want {want} got {status}: {text!r} | {ev}")
    print(f"{len(_CHECKS) - bad}/{len(_CHECKS)} checks pass")
    return bad

