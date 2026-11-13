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
NEG_RE = re.compile(r"\b(?:not|never|n't|didnt|no point|nobody|no one|nowhere|neither|nor|cannot|without)\b|n't\b")
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
        if re.search(r"\b(?:I|me|my|myself|you)\b", t):
            return None
    else:
        m = re.match(rf"^({NAME}) (?=[a-z])", t)
        if m and m.group(1) in names and not re.search(r"\b(?:I|me|my|myself)\b", t):
            actor = m.group(1)
            t = "@" + t[m.end() - 1:]
        elif re.search(r"\b(?:I|me|my|myself)\b", t):
            if re.search(r"\byou\b", t, re.I):
                return None
            actor = ctx.speakers.get(msg.label)
            t = re.sub(r"\bmyself\b", "@", t)
            t = re.sub(r"\b(?:my|mine)\b", "@'s", t)
            t = re.sub(r"\b(?:I|me)\b", "@", t)
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


def entry_re(p):
    return rf"@ (?:\w+ )?{ENTER} {place_re(p)}"


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
