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
}

LEAD_RE = re.compile(
    r"^\s*(?:(?:honestly|for the record|okay so|ok so|okay|ok|so|well|to be honest|frankly|actually|look|right|yes|"
    r"yeah|anyway|i think|i believe|i'm sure|i am sure|as far as i know|just so you know)\s*[,:]?\s+)+", re.I)
TAIL_RE = re.compile(
    r"(?:\s*,\s*(?:honestly|for the record|okay|ok|to be honest|i think|i believe|i'm sure|for sure|i promise|"
    r"definitely|i swear))+\s*$", re.I)
SAW_IT_RE = re.compile(r",?\s*(?:and )?I (?:saw|watched|noticed|spotted|witnessed) (?:it|this|that|them|him|her)"
                       r"(?: happen)?(?: myself)?\s*$")
WITNESS_RE = re.compile(r"\bI (?:saw|watched|noticed|spotted|witnessed|observed)(?: that)? (you|" + NAME + r")\b")
FIRST_PERSON = re.compile(r"\b(?:I|[Mm]e|[Mm]y|[Mm]yself|[Mm]ine)\b")
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
    if t.endswith("?"):
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
    low = low.replace("@'s", "@'s")
    return Claim(actor, low, times, {n.lower(): n for n in names})


def event_time(c):
    """The single clock time of the claim when it is given as 'at T', else None."""
    if len(c.times) != 1:
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


def place_re(p):
    """Regex for a place noun phrase as it appears in a claim, with an optional article, not followed by door/gate."""
    return rf"(?:the )?{re.escape(p)}(?![a-z])(?! door| gate| keypad)"


def match_pieces(sentence, pats):
    """Match patterns against a sentence and its clauses. A sentence is cut at '; ', ', while ' and ', and ' into
    clauses; every run of adjacent clauses is tried, longest first, and runs overlapping an earlier match are
    skipped. Returns [(pattern index, match)] in sentence order."""
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
    return [(n, m) for _, n, m in sorted(found, key=lambda f: f[0])]


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
                         r"is painted \w+|holds \w+ shelves|was repainted in \d{4})$|^in \d{4} the (?P<q>[a-z ]+?) "
                         r"(?:was repainted|got a fresh coat of paint)$", re.I)

    def parse(self, ctx):
        wins = []
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
                ctx.consume(i)
                break
        if not wins:
            return None
        places = {w[0] for w in wins}
        for i, s in enumerate(ctx.sentences):
            m = self.FLAVOUR.match(s.rstrip("."))
            if m and clean_np(m.group("p") or m.group("q")) in places:
                ctx.consume(i)
        return wins

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None or c.actor is None:
            return None
        for p, kind, t1, t2, between in facts:
            pr = place_re(p)
            if not re.search(pr, c.text):
                continue
            hit = re.search(entry_re(p), c.text) and kind in ("enter", "in")
            if kind == "in" and re.search(rf"@ (?:was|were|stayed|remained|sat|worked|waited) (?:still )?"
                                          rf"(?:in|inside) {pr}", c.text):
                hit = True
            if kind == "use" and re.search(rf"@ (?:used|use|using) {pr}", c.text):
                hit = True
            if hit:
                if negated(c):
                    return verdict(self, True, f"denies entering {p}")
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
    NO_ROUTE = re.compile(r"^there was no other remote route$", re.I)

    def parse(self, ctx):
        deps, outs, remote = [], [], []
        pats = [self.NO_ROUTE] + self.DEP + self.REMOTE + self.OFF
        kinds = ["route"] + ["dep"] * len(self.DEP) + ["remote"] * len(self.REMOTE) + ["off"] * len(self.OFF)
        for i, s in enumerate(ctx.sentences):
            found = match_pieces(s, pats)
            for n, m in found:
                if kinds[n] == "dep":
                    deps.append((clean_np(m.group("a")), m.group("v").lower(), clean_np(m.group("b"))))
                elif kinds[n] == "remote":
                    remote.append((clean_np(m.group("s")), clean_np(m.group("b"))))
                elif kinds[n] == "off":
                    a, b = m.group("t1") or m.group("t3"), m.group("t2") or m.group("t4")
                    outs.append((clean_np(m.group("b")), tm(a), tm(b), m.group("t3") is not None))
            if found:
                ctx.consume(i)
        if not outs or not (deps or remote):
            return None

        def same(x, y):  # "the VPN" and "the VPN service" name the same thing
            return x == y or bool(re.search(rf"\b{re.escape(x)}\b", y) or re.search(rf"\b{re.escape(y)}\b", x))

        def windows(thing, seen=()):
            w = [(t1, t2, btw) for b, t1, t2, btw in outs if same(b, thing)]
            for a, _, b in deps:
                if a == thing and b not in seen:
                    w += windows(b, seen + (thing,))
            return w

        machines = [(a, v, windows(a)) for a, v, _ in deps if windows(a)]
        servers = [(s, windows(b)) for s, b in remote if windows(b)]
        if not machines and not servers:
            return None
        return {"machines": machines, "servers": servers, "heads": [a.split()[-1] for a, _, _ in machines]}

    def uses(self, c, a, v, heads):
        """True when the claim says machine a did its job v (or was in its working state)."""
        head = a.split()[-1]
        objs = [re.escape(a)] + ([re.escape(head)] if heads.count(head) == 1 and head != a else [])
        obj = rf"(?:the )?(?:{'|'.join(objs)})\b"
        if not re.search(obj, c.text):
            return False
        words = v.split()
        if words[0] == "be" and len(words) >= 2:  # "could be lit": a state claim about the machine
            states = LIT_WORDS if words[1] in ("lit", "on", "illuminated") else re.escape(" ".join(words[1:]))
            return bool(re.search(rf"{obj} (?:was|were) (?:still |already )?{states}\b", c.text)
                        or re.search(rf"@ saw {obj} {states}\b", c.text))
        if words in (["run"], ["operate"], ["work"]):
            if head in CONVEYANCE or a in CONVEYANCE:
                if re.search(rf"@ (?:rode|ride|riding|took|take|taking|used|use|using|caught|catch) {obj}", c.text):
                    return True
                if re.search(rf"@ (?:went|came|got|travelled|traveled|rode) (?:up|down)\b.*\bin {obj}", c.text):
                    return True
            return bool(re.search(rf"{obj} (?:was|were) (?:still |already )?{RUN_WORDS}\b", c.text))
        # an action with an object: "make coffee" -> "made a coffee with the machine"
        forms = "|".join(sorted(verb_forms(words[0]), key=len, reverse=True))
        rest = r" (?:\S+ ){0,3}?".join(re.escape(w) for w in words[1:])
        if not rest:
            return bool(re.search(rf"@ (?:{forms}) (?:with |using |on )?{obj}", c.text))
        return bool(re.search(rf"@ (?:\S+ ){{0,4}}?(?:{forms}) (?:\S+ ){{0,3}}?{rest}\b", c.text))

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None:
            return None
        for a, v, wins in facts["machines"]:
            if self.uses(c, a, v, facts["heads"]):
                t = event_time(c)
                if negated(c) or t is None:
                    return None
                return self.check(wins, t, f"{a} could not {v}")
        for s, wins in facts["servers"]:
            if not re.search(rf"(?:the )?{re.escape(s)}\b", c.text):
                continue
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
                         r"keypad(?: with a \w+ light)?$", re.I)

    def parse(self, ctx):
        rules = {}  # door -> [(start, end, code, end_is_unclear)]
        last_door = None
        flavour = []
        pats = self.BEFORE + self.AFTER + [self.FLAVOUR]
        for i, s in enumerate(ctx.sentences):
            used = False
            for n, m in match_pieces(s, pats):
                if n < len(self.BEFORE):
                    last_door = clean_np(m.group("d"))
                    t = tm(m.group("t"))
                    rules.setdefault(last_door, []).append((0, t, m.group("c"), m.group("r").lower() == "until"))
                    used = True
                elif n < len(self.BEFORE) + len(self.AFTER):
                    d = clean_np(m.group("d")) if m.group("d") else last_door
                    if d is None:
                        continue
                    rules.setdefault(d, []).append((tm(m.group("t")), DAY_END, m.group("c"), False))
                    used = True
                else:
                    flavour.append(i)
            if used:
                ctx.consume(i)
        if not rules:
            return None
        for i in flavour:
            ctx.consume(i)
        return rules

    UNLOCK = re.compile(r"\b(?:unlock|unlocked|unlocks|unlocking|opened|open|opens)\b")
    ATTEMPT = re.compile(r"\b(?:tried|try|tries|trying|attempt\w*|failed|wouldn't|couldn't|didn't|refused|wrong)\b")

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None:
            return None
        codes = re.findall(r"\bcode (\d+)\b", c.text)
        if len(set(codes)) != 1 or not self.UNLOCK.search(c.text):
            return None
        if negated(c) or self.ATTEMPT.search(c.text):
            return None
        code = codes[0]
        for d, rules in facts.items():
            short = re.sub(r" (?:door|gate)$", "", d)
            if not re.search(rf"(?:the )?(?:{re.escape(d)}|{re.escape(short)})\b", c.text):
                continue
            t = event_time(c)
            if t is None:
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
        doors, pos, neg, excl = [], [], [], []
        used = set()
        pats = self.DOORS + [self.NOT_MEMBER, self.NOT_MEMBER2, self.EXCLUDES, self.MEMBERS]
        nd = len(self.DOORS)
        for i, s in enumerate(ctx.sentences):
            for n, m in match_pieces(s, pats):
                used.add(i)
                if n < nd:
                    doors.append((clean_np(m.group("door")), clean_np(m.group("p")), group_key(m.group("g")),
                                  tm(m.group("t")), m.group("r").lower() == "until"))
                elif n in (nd, nd + 1):
                    neg.append((m.group("n"), group_key(m.group("g"))))
                elif n == nd + 2:
                    excl.append((group_key(m.group("r")), group_key(m.group("g"))))
                else:
                    for name in re.findall(NAME, m.group("n")):
                        pos.append((name, group_key(m.group("g"))))
        if not doors:
            return None
        member = {}  # (name, group) -> True/False
        for n, g in pos:
            member[(n, g)] = True
        for n, g in neg:
            if member.get((n, g)) is True:
                return None  # the evidence contradicts itself as read; stay out
            member[(n, g)] = False
        for r, g in excl:
            for (n, h), v in list(member.items()):
                if h == r and v:
                    if member.get((n, g)) is True:
                        return None
                    member[(n, g)] = False
        for i in used:
            ctx.consume(i)
        return {"doors": doors, "member": member}

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None or c.actor is None:
            return None
        for door, p, g, t0, until in facts["doors"]:
            if re.search(entry_any_door_re(p), c.text):
                t = event_time(c)
                if negated(c) or t is None:
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
    FLAVOUR = re.compile(rf"^(?:each|every) {NP} (?:is printed with a number|has a number printed on it|carries a "
                         rf"printed number)$|^a number is printed on (?:each|every) {NP}$|^(?:the )?{NP} opened at "
                         rf"{TIME}$|^at {TIME},? (?:the )?{NP} opened$", re.I)

    def parse(self, ctx):
        places, cards, flavour = [], {}, []
        nr, nt = len(self.RULES), len(self.TWO)
        for i, s in enumerate(ctx.sentences):
            for n, m in match_pieces(s, self.RULES + self.TWO + [self.FLAVOUR]):
                if n < nr:
                    places.append(clean_np(m.group("p")))
                    ctx.consume(i)
                    continue
                if n >= nr + nt:
                    flavour.append(i)
                    continue
                s1, s2 = self.ACTIVE.get(m.group("s1").lower()), self.ACTIVE.get(m.group("s2").lower())
                r1, r2 = m.group("r1").lower(), m.group("r2").lower()
                t1, t2 = tm(m.group("t1")), tm(m.group("t2"))
                if s1 is None or s2 is None or s1 == s2 or r1 not in ("before", "until") or t1 != t2:
                    continue
                if r2 not in ("from", "since", "after"):
                    continue
                # (start, end, active, end_is_unclear): before T -> [0, T); from T -> [T, end of day)
                cards[m.group("n")] = [(0, t1, s1, r1 == "until"), (t2, DAY_END, s2, r2 == "after")]
                ctx.consume(i)
        if not places:
            return None
        for i in flavour:
            ctx.consume(i)
        return {"places": places, "cards": cards}

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None or c.actor is None:
            return None
        for p in facts["places"]:
            if not re.search(entry_re(p), c.text):
                continue
            t = event_time(c)
            if negated(c) or t is None:
                return None
            spans = facts["cards"].get(c.actor)
            if spans is None:
                return None
            for start, end, active, unclear in spans:
                if unclear and t in (start, end):
                    return None
                if start <= t < end:
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
    EVERY = re.compile(rf"\baccurately records every (?:(?P<e>entry to|entry into|time someone entered|time someone "
                       rf"went into)|(?P<x>exit from|time someone left|time someone exited|departure from)) "
                       rf"(?:the )?(?P<p>{NP})(?= that day|,|$| and)", re.I)
    ITEMS = re.compile(rf"\bexactly (?P<k>\w+) (?P<kind>entries|entry|exits|exit)\b[^:]*: (?P<items>.+?)\.?$", re.I)
    FLAVOUR = re.compile(rf"^(?:the )?(?P<p>{NP}) has (?:a single|one|only one) door$|^(?:someone checks )?(?:the )?"
                         rf"(?:{NP} )?log is checked every evening$|^every evening (?:the )?{NP} log is checked$|"
                         rf"^someone checks the {NP} log every evening$", re.I)

    def parse(self, ctx):
        logs = []  # [kind, place, entries or None]
        flavour = []
        for i, s in enumerate(ctx.sentences):
            body = s.rstrip(".")
            m = self.EVERY.search(body)
            if m:
                logs.append(["exit" if m.group("x") else "entry", clean_np(m.group("p")), None])
                ctx.consume(i)
            m = self.ITEMS.search(body)
            if m and logs and logs[-1][2] is None:
                kind = "exit" if m.group("kind").lower().startswith("exit") else "entry"
                items = name_times(m.group("items"))
                k = NUMBER_WORDS.get(m.group("k").lower()) or (int(m.group("k")) if m.group("k").isdigit() else None)
                if kind == logs[-1][0] and items is not None and k == len(items):
                    logs[-1][2] = items
                    ctx.consume(i)
            for cl in [c for c in re.split(r",\s+and\s+|;\s+", body)]:
                if self.FLAVOUR.match(cl):
                    flavour.append(i)
        logs = [tuple(lg) for lg in logs if lg[2] is not None]
        if not logs:
            return None
        for i in flavour:
            ctx.consume(i)
        return logs

    NEVER = r"(?:at all|that day|all day)"

    def judge(self, ctx, facts, msg):
        c = claim(ctx, msg)
        if c is None or c.actor is None:
            return None
        for kind, p, items in facts:
            mine = [t for n, t in items if n == c.actor]
            pr = place_re(p)
            if not re.search(pr, c.text):
                continue
            if kind == "entry":
                return self.judge_entry(c, p, pr, mine)
            return self.judge_exit(c, p, pr, mine)
        return None

    def judge_entry(self, c, p, pr, mine):
        went = re.search(entry_re(p), c.text) or re.search(rf"@ {AUX}set foot in {pr}", c.text)
        if went and negated(c):
            sp = span(c)
            if sp:
                inside = [t for t in mine if sp[0] <= t <= sp[1]]
                return verdict(self, not inside, f"log entries for {c.actor}: {[T.fmt_time(t) for t in mine]}")
            if re.search(rf"\b{self.NEVER}$|@ never ", c.text) and not c.times:
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

    def judge_exit(self, c, p, pr, mine):
        left = re.search(rf"@ {AUX}(?:leave|left|leaving|exit|exited|exiting|step out of|stepped out of|go out of|"
                         rf"went out of|walk out of|walked out of) {pr}", c.text) or \
            re.search(rf"\bdid @ (?:leave|exit|step out of|go out of|walk out of) {pr}", c.text)
        if not left:
            return None
        if negated(c):
            sp = span(c)
            if sp:
                inside = [t for t in mine if sp[0] <= t <= sp[1]]
                return verdict(self, not inside, f"exit log for {c.actor}: {self.fmt(mine)}")
            if re.search(rf"\b{self.NEVER}$", c.text) and not c.times:
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
    LOG = re.compile(rf"\brecords (?:when|the time) each of these (?:\w+ )?people first arrived at (?:the )?(?P<p>{NP})"
                     rf"(?: that day)?(?:: (?P<items>.+))?$", re.I)
    LOG2 = re.compile(rf"\brecords when they first arrived at (?:the )?(?P<p>{NP})(?: that day)?: (?P<items>.+)$", re.I)
    LOG3 = re.compile(rf"^(?P<items>.+?) are the times when each of these (?:\w+ )?people first arrived at (?:the )?"
                      rf"(?P<p>{NP})(?: that day)?,? and the door log accurately records them$", re.I)
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
        place, known, rels, logged = None, {}, [], False
        for i, s in enumerate(ctx.sentences):
            body = s.rstrip(".")
            m = self.LOG.search(body) or self.LOG2.search(body) or self.LOG3.match(body)
            if m:
                place = clean_np(m.group("p"))
                logged = True
                ctx.consume(i)
                if m.group("items"):
                    items = name_times(m.group("items"))
                    if items is None:
                        return None
                    known.update(items)
                continue
            m = self.ITEMS.match(body)
            if m and logged and not known:
                items = name_times(m.group("items"))
                if items:
                    known.update(items)
                    ctx.consume(i)
                continue
            used = False
            for mm in (mm for pat in self.ABS for mm in pat.finditer(body)):
                p = clean_np(mm.group("p"))
                if place not in (None, p):
                    return None
                place = p
                known[mm.group("n")] = tm(mm.group("t"))
                used = True
            for pat in self.REL:
                for mm in pat.finditer(body):
                    k = int(mm.group("k")) * (1 if mm.group("d") == "after" else -1)
                    rels.append((mm.group("n"), mm.group("m"), k))
                    used = True
            if used:
                ctx.consume(i)
        if place is None or not known:
            return None
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
        return {"place": place, "first": known}

    ARRIVE = (r"(?:arrived at|arrive at|got to|get to|reached|reach|showed up at|show up at|made it to|make it to|"
              r"came to|come to|turned up at|turn up at|got in to|got into|walked into|came into|entered)")
    BEFORE = r"(?:before|ahead of|earlier than|sooner than)"
    AFTER = r"(?:after|later than|behind)"

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
        if re.search(rf"@ was the (?:very )?first (?:person|one|to arrive|to get there)?\s*(?:in|into|at|to arrive at|"
                     rf"to get to|to reach)? {pr}", c.text) or re.search(rf"@ (?:got to|reached|arrived at) {pr} first\b",
                                                                        c.text):
            others = [t for n, t in first.items() if n != c.actor]
            if any(t < me for t in others):
                return verdict(self, False, f"{c.actor} first arrived at {T.fmt_time(me)}, someone listed was earlier")
            if all(t > me for t in others):
                return verdict(self, True, f"{c.actor} is the earliest of the listed people")
            return None
        arrival = (re.search(rf"@ {AUX}(?:first )?{self.ARRIVE} {pr}", c.text)
                   or re.search(rf"@'s first arrival at {pr}", c.text)
                   or re.search(rf"\bfirst time @ {AUX}{self.ARRIVE} {pr}", c.text))
        if not arrival:
            return None
        is_first = bool(re.search(r"\bfirst\b", c.text))
        m = re.search(rf"\b(?P<rel>{self.BEFORE}|{self.AFTER}) (?P<who>[a-z]+)(?:'s)?(?: did| had| arrived)?\b", c.text)
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
        if len(c.times) != 1 or c.times[0][1] not in ("at", "was"):
            return None
        t = c.times[0][0]
        # Without "first" ("I got to the office at 9:55 AM") train always means the first arrival: innocents give
        # exactly the logged time (27 of 27), liars an earlier or later one.
        return verdict(self, t == me, f"{c.actor} first arrived at {T.fmt_time(me)}"
                                      f"{'' if is_first else ' (arrival read as the first one)'}")
