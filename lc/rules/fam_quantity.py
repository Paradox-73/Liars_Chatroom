"""Rule families: quantity. See lc/rules/core.py for the Family interface.

Each family parses one kind of evidence that pins a number exactly and judges chat claims about that number with
exact arithmetic (fractions, never floats). A family answers only for wordings it fully parses; anything else it
leaves to the language-model judge.

qty_level       a container's amount over time: a reading, exact additions and removals, steady fill or drain rates
                and "nothing else was added or removed" windows; also fixed item counts ("the drawer held exactly six
                folders from T1 to T2")
qty_tickets     tickets sold for an event, the share sold before noon, the share of the morning ones sold online
qty_pieces      an object of exact length or weight cut into equal pieces, or into one long piece and equal short ones
qty_attendance  an attendance list that the evidence says is complete
"""
import re
from fractions import Fraction

from . import text as T
from .core import LIE, OK, Family, Verdict, register

# ------------------------------------------------------------------------------------------------ shared helpers

WORDNUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
NUM = r"(?:\d+(?:\.\d+)?|" + "|".join(sorted(WORDNUM, key=len, reverse=True)) + r")\b"

LEAD = re.compile(r"^(?:honestly|frankly|for the record|okay so|ok so|okay|so|well|actually|to be honest|in fact)\b,?\s*",
                  re.I)
TAIL = re.compile(r",?\s+(?:for the record|honestly|to be honest|yes|no doubt|frankly)$", re.I)


def body(text):
    """The claim without the chat's hedges ('Honestly, ...', '..., for the record') and the final full stop."""
    s, prev = text.strip().rstrip(".!").strip(), None
    while s != prev:
        prev = s
        s = TAIL.sub("", LEAD.sub("", s)).strip()
    return s


def mask_times(s):
    """Replace each clock time in s with '@T'; return (masked text, minutes since midnight in order). An impossible
    time such as '13:30 PM' stays as written, so no pattern reads it."""
    out = []

    def rep(m):
        if m.group(4):
            out.append(0 if m.group(4).lower() == "midnight" else 720)
            return "@T"
        try:
            out.append(T.time_to_min(m.group(1), m.group(2), m.group(3)))
        except ValueError:
            return m.group(0)
        return "@T"

    return T.TIME_RE.sub(rep, s), out


def num(s):
    s = s.lower()
    return Fraction(s) if s[0].isdigit() else Fraction(WORDNUM[s])


def precision(s):
    """The step a written number claims: 1 for '60' or 'six', 1/10 for '60.5'."""
    return Fraction(1, 10 ** len(s.split(".")[1])) if "." in s else Fraction(1)


def settle(claimed, prec, possible):
    """OK when the claim equals a possible value; LIE when it is at least one step of its own precision away from
    every possible value; None when only rounding separates them."""
    if claimed in possible:
        return OK
    if all(abs(claimed - p) >= prec for p in possible):
        return LIE
    return None


def sing(phrase):
    """Singular of the last word: 'litres' -> 'litre', 'trophies' -> 'trophy', 'sealed envelopes' -> 'sealed envelope'."""
    words = phrase.lower().split()
    w = words[-1]
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("ches", "shes", "sses", "xes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        w = w[:-1]
    return " ".join(words[:-1] + [w])


UNITS = {"litre": ("vol", 1), "liter": ("vol", 1), "l": ("vol", 1), "millilitre": ("vol", Fraction(1, 1000)),
         "milliliter": ("vol", Fraction(1, 1000)), "ml": ("vol", Fraction(1, 1000)),
         "kilogram": ("mass", 1), "kilo": ("mass", 1), "kg": ("mass", 1), "gram": ("mass", Fraction(1, 1000)),
         "g": ("mass", Fraction(1, 1000)), "tonne": ("mass", 1000),
         "metre": ("len", 1), "meter": ("len", 1), "m": ("len", 1), "centimetre": ("len", Fraction(1, 100)),
         "centimeter": ("len", Fraction(1, 100)), "cm": ("len", Fraction(1, 100)),
         "millimetre": ("len", Fraction(1, 1000)), "millimeter": ("len", Fraction(1, 1000)),
         "mm": ("len", Fraction(1, 1000)), "kilometre": ("len", 1000), "kilometer": ("len", 1000), "km": ("len", 1000)}


def unit_of(u):
    """(dimension, size) of a unit phrase; any other noun is a count of that thing: ('count:folder', 1)."""
    w = sing(u)
    if w in UNITS:
        d, f = UNITS[w]
        return d, Fraction(f)
    return "count:" + w, Fraction(1)


def same_dim(a, b):
    """Same measure, or exactly the same counted noun: 'five empty bags' says nothing about how many bags there are."""
    return a == b


def head(c):
    """Container name without its article or a trailing place phrase: 'the cabinet by the window' -> 'cabinet'."""
    c = re.sub(r"^(?:the|that|this|a|an) ", "", c.strip().lower())
    return re.sub(r" (?:by|near|beside|next to|in|on|at|under|behind|outside|inside) .*$", "", c)


def same_name(a, b):
    a, b = head(a), head(b)
    return a == b or a.endswith(" " + b) or b.endswith(" " + a)


def unparsed_mention(ctx, used, key):
    """True when an evidence sentence the family did not parse names its subject (a word for which key() is true):
    the parsed facts may then be incomplete, so the family abstains."""
    return any(i not in used and not sent.startswith("All times in this case refer to")
               and any(key(w) for w in re.findall(r"[a-z]+", sent.lower()))
               for i, sent in enumerate(ctx.sentences))


# -------------------------------------------------------------------------------------- qty_level: containers

C = r"(?:the|that|this) (?P<c>[a-z][a-z'\- ]*?)"
# a unit after a number; "twenty five bags" or "5 more bags" is not "N bags"
UNIT = (r"(?P<u>(?!(?:" + "|".join(WORDNUM) + r"|hundred|thousand|million|dozen|more|fewer|less|extra|other|"
        r"additional|further)\b)[a-z]+(?: [a-z]+)?)")
QTY = r"(?:exactly )?(?P<n>" + NUM + r") " + UNIT + r"(?: of [a-z]+)?"  # "150 bags (of sand)"
SPAN = r"(?:from @T (?:to|until)|between @T and) @T"
STATES = [re.compile(r"^(?:(?P<pre>at @T|" + SPAN + r"),? )?" + p + r"(?: (?P<post>at @T|" + SPAN + r"))?$") for p in (
    r"(?:" + C + r"|(?P<it>it)) (?:held|contained|had|was holding) " + QTY + r"(?: in it)?",
    r"there (?:were|was) " + QTY + r" in (?:" + C + r"|(?P<it>it))")]
RATE = re.compile(r"^(?:(?P<pre>" + SPAN + r"|from then (?:to|until) @T|until @T),? )?(?:(?:" + C + r"|(?P<it>it)) )?"
                  r"(?:then )?(?:was )?(?P<v>drained|filled|refilled|emptied|leaked|pumped out) "
                  r"(?:at a steady|at a constant|steadily at)(?: rate of)? "
                  r"(?P<r>\d+(?:\.\d+)?) (?P<u>[a-z]+) (?:per|an|a|each|every) (?P<per>hour|minute)"
                  r"(?: (?P<post>" + SPAN + r"|from then (?:to|until) @T|until @T))?$")
ADDED = ("added", "poured in", "put in", "tipped in", "pumped in", "delivered")
EVENT = re.compile(r"^(?:then )?(?:at @T,? )?(?:exactly )?(?P<n>" + NUM + r") " + UNIT + r" (?:was|were) "
                   r"(?P<v>" + "|".join(ADDED) + r"|removed|taken out|taken|taken away|drawn off|used|poured out|"
                   r"tipped out|scooped out|pumped out|drained off)"
                   r"(?: (?:to|into|from|out of|off) (?:it|" + C + r"))?(?: at @T)?$")
CLOSE = re.compile(r"^(?:and )?(?:with )?(?:nothing (?:else )?|no other [a-z]+ )(?:was )?"
                   r"(?P<what>added or removed|removed or added|put in or taken out|added)"
                   r"(?: (?:to|from|in|out of) it)?(?: (?:before|until) @T)?$"
                   r"|^(?:and )?there (?:were|was) no other (?:additions or removals|changes)(?: (?:before|until) @T)?$")
NOCHANGE = re.compile(r"^(?:and )?no (?P<u>[a-z]+(?: [a-z]+)?) (?:was|were) (?:added or removed|removed or added)"
                      r"(?: during that time| in that time| during that period)?$")
COLOUR = r"(?:grey|gray|white|black|blue|red|green|brown|yellow|orange|purple|pink|silver)"
# a plain description of a container, which only tells what "it" in the next sentence refers to
SUBJECT = re.compile(r"^" + C + r" (?:(?:is|was) (?:painted )?" + COLOUR
                     + r"|(?:stands|stood|sits|sat|is|was) (?:by|beside|next to|near|against|under) the [a-z]+)$")
LEADING_TIME = re.compile(r"^(?:at @T|" + SPAN + r"|until @T|from then (?:to|until) @T)$")
SPLIT = re.compile(r"\s*;\s*|,\s*(?:and |then |while )?|\s+and\s+(?=(?:then |it |drained|filled|emptied|leaked|exactly"
                   r"|nothing|no |from |until |at @T))")
ASIDE = re.compile(r", (?:which|that) (?:is|was) " + COLOUR + ",")  # "the cabinet, which is grey, held ..."
CHANGE_WORD = re.compile(r"\b(?:added|removed|poured|drained|filled|refilled|emptied|leaked|topped up|taken out|"
                         r"tipped|scooped|pumped|delivered|used up)\b")
NUMBER_WORD = re.compile(r"\d|\b(?:" + "|".join(WORDNUM) + r")\b")


def clauses(masked, times):
    """Split a masked sentence into clauses, each with its own times; a bare leading time joins the next clause."""
    out, k, carry = [], 0, None
    for p in SPLIT.split(masked):
        p = p.strip()
        if not p:
            continue
        n = p.count("@T")
        ts = times[k:k + n]
        k += n
        if carry:
            p, ts, carry = carry[0] + " " + p, carry[1] + ts, None
        if LEADING_TIME.match(p):
            carry = (p, ts)
            continue
        out.append((p, ts))
    if carry:
        out.append(carry)
    return out


def span_times(phrase, ts):
    """(start, end) for a time phrase whose @T values are ts; start None means 'from the reading'."""
    if len(ts) == 2:
        return ts[0], ts[1]
    return None, ts[0]  # 'until @T' or 'from then until @T'


class Store:
    """One container: an anchor (a reading at an instant, or a fixed amount over a span) and every change."""

    def __init__(self, name, unit):
        self.name, self.unit = name, unit  # unit: (dimension, size); amounts are kept in this unit
        self.reading = None  # (minute, amount)
        self.span = None  # (t1, t2, amount): amount fixed over the whole closed range
        self.span_closed = False  # "no folder was added or removed during that time"
        self.events = []  # (minute, signed amount)
        self.rates = []  # [start or None, end, signed amount per minute]
        self.close = []  # ("all" | "add", end minute or None)
        self.kin = {}  # last words of the name -> container phrases in the evidence that end with them
        self.bad = False

    def finish(self):
        if (self.reading is None) == (self.span is None):
            self.bad = True
            return
        if self.span:
            self.bad |= bool(self.events or self.rates or self.close)
            return
        t0 = self.reading[0]
        for r in self.rates:
            r[0] = t0 if r[0] is None else r[0]
            self.bad |= r[0] < t0 or r[1] <= r[0]
        self.bad |= any(te <= t0 for te, _ in self.events)

    def pinned_until(self):
        """(end, end included, kind) of the stretch after the reading where every change is listed, or None."""
        kinds = [k for k, _ in self.close]
        if not kinds:
            return None
        kind = "all" if "all" in kinds else "add"
        ends = [e for _, e in self.close if e is not None]
        if ends:
            return min(ends), False, kind
        if self.rates:
            return max(r[1] for r in self.rates), True, kind
        return None

    def amount_at(self, t):
        """(set of amounts possible at minute t, whether a mismatch is a sure lie), or None when nothing is fixed."""
        if self.span:
            t1, t2, a = self.span
            return ({a}, True) if t1 <= t <= t2 else None
        t0, a = self.reading
        if t < t0:
            # "nothing else was added or removed (before T)" also covers the time before the reading, so the reading
            # holds then too; only agreement is judged, as train never shows a different number there
            closed = [e for k, e in self.close if k == "all"]
            return ({a}, False) if closed and all(e is None or t < e for e in closed) else None
        if t > t0:
            w = self.pinned_until()
            if w is None or t > w[0] or (t == w[0] and not w[1]):
                return None
            if w[2] == "add" and (self.events or not self.rates):
                return None  # only additions are ruled out, and the rate alone does not pin the amount
        at_t = []
        for te, d in self.events:
            if te < t:
                a += d
            elif te == t:
                at_t.append(d)
        for ra, rb, r in self.rates:
            lo, hi = max(ra, t0), min(rb, t)
            if hi > lo:
                a += r * (hi - lo)
        possible = {a}
        for d in at_t:  # changes at that very minute may or may not have happened yet, in any order
            possible |= {p + d for p in possible}
        return (possible, True) if min(possible) >= 0 else None  # below empty: the reading of the evidence is off

    def removal(self, t):
        """LIE/OK for 'someone took some out at minute t', or None."""
        if self.span:
            t1, t2, _ = self.span
            if not self.span_closed:
                return None
            return LIE if t1 < t < t2 else OK if t < t1 or t > t2 else None
        w = self.pinned_until()
        if w is None or w[2] != "all":
            return None
        t0 = self.reading[0]
        end, incl, _ = w
        if t < t0 or t == t0:
            return None
        if t > end or (t == end and not incl):
            return OK if not incl else None  # after a "before T" window anything may happen
        if any(ra <= t <= rb for ra, rb, _ in self.rates):
            return None
        at = [d for te, d in self.events if te == t]
        if at:
            return OK if any(d < 0 for d in at) else None
        return LIE


def pick(stores, name, claim=False):
    """The one store with this name. A claim may shorten the name ('the tank' for 'the water tank') but not add to
    it ('the paddling pool' may be a different pool from 'the pool'), and may shorten it only when the evidence
    names no other container with that word ('the red tank' elsewhere makes 'the tank' ambiguous)."""
    if claim:
        c = head(name)
        found = [s for s in stores if head(s.name) == c or (head(s.name).endswith(" " + c) and not any(
            p != head(s.name) and p != c for p in s.kin.get(c, ())))]
    else:
        found = [s for s in stores if same_name(name, s.name)]
    return found[0] if len(found) == 1 else None


CQ = r"(?:exactly |only |just )?(?P<n>" + NUM + r") " + UNIT
AMOUNT = [re.compile(r"^(?:at @T,? )?(?:" + f.format(Q=CQ, C=C) + r")(?: at @T)?$") for f in (
    r"there (?:were|was) {Q} in {C}",
    r"{C} (?:held|had|contained|still held|still had|was holding) {Q}(?: in it)?",
    r"{Q} (?:were|was) in {C}",
    r"i (?:counted|make it|measured|recorded|noted|photographed all|counted all|saw all"
    r"|(?:got|took) (?:photos|pictures|a photo|a picture) of all) {Q} in {C}",
    r"in {C}(?: at @T)? there (?:were|was) {Q}")]
TOOK = re.compile(r"^(?:at @T,? )?i (?:took|drew|scooped|removed|pulled|lifted|grabbed|got|fetched|helped myself to) (?:some|a little|a bit|a few|one|a|an)"
                  r"(?: of (?:the|them)(?: (?P<u>[a-z]+(?: [a-z]+)?))?| (?P<u2>[a-z]+))? (?:out of|from|out from) " + C
                  + r"(?: at @T)?$")
NEUTRAL = re.compile(r"^(?:at @T,? )?i (?:checked|checked on|inspected|looked at|looked over|looked in|had a look at|"
                     r"took a look at|had a check of|went and checked|went to check|walked past|"
                     r"was standing (?:beside|next to|by|near)|stood (?:beside|next to|by|near)|"
                     r"was (?:beside|next to|by|near)) " + C + r"(?: at @T)?$")
OPINION = re.compile(r"\b(?:too (?:small|big|large|little)|bigger|smaller|larger|badly sized|poor design|"
                     r"suits? our needs|big enough|large enough|would help|would be useful|wish|keep a log|log book)\b")


@register
class Level(Family):
    name = "qty_level"

    def parse(self, ctx):
        stores, cur, subjects, used = [], None, [], []

        def store_for(name, unit):
            st = next((s for s in stores if same_name(name, s.name)), None)
            if st is None:
                st = Store(name, unit)
                stores.append(st)
            elif not same_dim(st.unit[0], unit[0]):
                st.bad = True
            return st

        unparsed, mine = [], set()  # mine: sentences read here, so "it" in the next one is a store
        for i, sent in enumerate(ctx.sentences):
            low, times = mask_times(ASIDE.sub("", sent.lower().rstrip(". ")))
            if i - 1 not in mine:
                cur = None
            m = SUBJECT.match(low)
            if m:
                subjects.append((i, m["c"]))
                mine.add(i)
                if cur is not None and not same_name(m["c"], cur.name):
                    cur = None  # "it" now means the container just described
                continue
            fresh = [name for j, name in subjects if j == i - 1]
            parts = clauses(low, times)
            done, touched = [], []
            for p, ts in parts:
                ok = self.apply(p, ts, stores, store_for, cur, fresh, touched)
                if not ok:
                    break
                done.append(p)
                cur = touched[-1] if touched else cur
            if len(done) == len(parts) and done:
                used.append(i)
                mine.add(i)
                continue
            for st in touched:  # half-understood sentence about a store: its model may be incomplete
                st.bad = True
            if not low.startswith("all times in this case refer to"):
                unparsed.append(low)
        changed = any(CHANGE_WORD.search(low) and NUMBER_WORD.search(low) for low in unparsed)
        for st in stores:
            # any other sentence about the container, its items, amounts in its unit, or a numbered addition or
            # removal of anything may change what it held
            words = {head(st.name).split()[-1]} | ({st.unit[0][6:].split()[-1]} if st.unit[0].startswith("count:")
                                                   else set())
            for low in unparsed:
                if any(sing(w) in words or w in words for w in re.findall(r"[a-z]+", low)) or any(
                        sing(w) in UNITS and UNITS[sing(w)][0] == st.unit[0]
                        for w in re.findall(r"\d+(?:\.\d+)? ([a-z]+)", low)):
                    st.bad = True
            st.bad |= changed
            st.finish()
        good = [s for s in stores if not s.bad]
        if not good:
            return None
        evidence = ctx.case.evidence.lower()
        for st in good:
            words = head(st.name).split()
            for k in range(1, len(words)):
                tail = " ".join(words[k:])
                st.kin[tail] = set(re.findall(r"\b(?:the|a|an|that|this) ((?:[a-z\-]+ )?" + re.escape(tail) + r")\b",
                                              evidence))
        for i in used:
            ctx.consume(i)
        for i, name in subjects:
            if any(same_name(name, s.name) for s in good):
                ctx.consume(i)
        return good

    def apply(self, p, ts, stores, store_for, cur, fresh, touched):
        """Add one clause to the stores; False when the clause is not understood. fresh: containers described in
        the sentence just before, which "it" may name."""
        m = next((m for m in (rx.match(p) for rx in STATES) if m), None)
        if m and bool(m["pre"]) != bool(m["post"]):
            phrase = m["pre"] or m["post"]
            st = self.subject(m, stores, store_for, cur, fresh, m["u"])
            if st is None:
                return False
            touched.append(st)
            u = unit_of(m["u"])
            if not same_dim(u[0], st.unit[0]):
                st.bad = True
                return True
            amount = num(m["n"]) * u[1] / st.unit[1]
            if phrase == "at @T" and len(ts) == 1:
                st.bad |= st.reading is not None or st.span is not None
                st.reading = (ts[0], amount)
            elif len(ts) == 2 and m.group(0).count("exactly"):
                st.bad |= st.reading is not None or st.span is not None or ts[1] < ts[0]
                st.span = (ts[0], ts[1], amount)
            else:
                return False
            return True
        m = RATE.match(p)
        if m and bool(m["pre"]) != bool(m["post"]):
            st = self.subject(m, stores, store_for, cur, fresh, m["u"], need_store=True)
            if st is None:
                return False
            touched.append(st)
            u = unit_of(m["u"])
            if not same_dim(u[0], st.unit[0]):
                st.bad = True
                return True
            start, end = span_times(m["pre"] or m["post"], ts)
            per = Fraction(60) if m["per"] == "hour" else Fraction(1)
            sign = 1 if m["v"] in ("filled", "refilled") else -1
            st.rates.append([start, end, sign * num(m["r"]) * u[1] / st.unit[1] / per])
            return True
        m = EVENT.match(p)
        if m and len(ts) == 1:
            u = unit_of(m["u"])
            if m["c"]:
                st = pick(stores, m["c"])
            else:
                fits = [s for s in stores if same_dim(u[0], s.unit[0])]
                st = cur if cur in fits else fits[0] if len(fits) == 1 else None
            if st is None:
                return False
            touched.append(st)
            sign = 1 if m["v"] in ADDED else -1
            st.events.append((ts[0], sign * num(m["n"]) * u[1] / st.unit[1]))
            return True
        m = CLOSE.match(p)
        if m and cur is not None and len(ts) <= 1:
            touched.append(cur)
            cur.close.append(("add" if m["what"] == "added" else "all", ts[0] if ts else None))  # changes ruled out
            return True
        m = NOCHANGE.match(p)
        if m and cur is not None and cur.span and same_dim(unit_of(m["u"])[0], cur.unit[0]):
            touched.append(cur)
            cur.span_closed = True
            return True
        return False

    @staticmethod
    def subject(m, stores, store_for, cur, fresh, u, need_store=False):
        if m["c"]:
            if need_store:
                return pick(stores, m["c"])
            return store_for(m["c"], unit_of(u))
        if m["it"] or need_store:
            if cur is not None:
                return cur
            if m["it"] and len(fresh) == 1 and not need_store:
                return store_for(fresh[0], unit_of(u))
        return None

    def judge(self, ctx, stores, msg):
        if msg.text.strip().endswith("?"):
            return None
        try:
            s, times = mask_times(body(msg.text).lower())
        except ValueError:
            return None
        for rx in AMOUNT:
            m = rx.match(s)
            if m is None:
                continue
            st = pick(stores, m["c"], claim=True)
            if st is None or len(times) != 1:
                return None
            u = unit_of(m["u"])
            if not same_dim(u[0], st.unit[0]):
                return None
            known = st.amount_at(times[0])
            if known is None:
                return None
            possible, sure = known
            k = u[1] / st.unit[1]
            res = settle(num(m["n"]) * k, precision(m["n"]) * k, possible)
            if res is None or (res == LIE and not sure):
                return None
            want = ", ".join(str(float(p)) for p in sorted(possible))
            return Verdict(res, self.name, f"{st.name} at {T.fmt_time(times[0])}: evidence {want}")
        m = TOOK.match(s)
        if m:
            st = pick(stores, m["c"], claim=True)
            u = m["u"] or m["u2"]
            if st is None or len(times) != 1 or (u and not same_dim(unit_of(u)[0], st.unit[0])):
                return None
            res = st.removal(times[0])
            if res == LIE and st.unit[0].startswith("count:") and not u:
                return None  # "some" from a drawer of folders may be something other than a folder
            return Verdict(res, self.name, f"removal from {st.name} at {T.fmt_time(times[0])}") if res else None
        m = NEUTRAL.match(s)
        if m and len(times) <= 1 and pick(stores, m["c"], claim=True):
            return Verdict(OK, self.name, "no claim about the amount")
        if not re.search(r"\d", s) and OPINION.search(s) and any(
                re.search(r"\b" + re.escape(head(st.name).split()[-1]) + r"\b", s) for st in stores):
            return Verdict(OK, self.name, "opinion")
        return None


# ---------------------------------------------------------------------------------------- qty_tickets: sales

FRACW = {"half": Fraction(1, 2), "a half": Fraction(1, 2), "one half": Fraction(1, 2), "a third": Fraction(1, 3),
         "one third": Fraction(1, 3), "two thirds": Fraction(2, 3), "a quarter": Fraction(1, 4),
         "one quarter": Fraction(1, 4), "three quarters": Fraction(3, 4), "a fifth": Fraction(1, 5),
         "one fifth": Fraction(1, 5), "two fifths": Fraction(2, 5), "three fifths": Fraction(3, 5),
         "four fifths": Fraction(4, 5), "a tenth": Fraction(1, 10), "one tenth": Fraction(1, 10)}
SHARE = r"(?:(?P<p>\d+(?:\.\d+)?)%|(?P<f>" + "|".join(sorted(FRACW, key=len, reverse=True)) + r"))"
EVT = r"(?P<e>[a-z][a-z'\- ]*?)"
SOLD_TOTAL = [re.compile(r"^" + p + r"$") for p in (
    r"(?:a total of )?exactly (?P<n>\d+) (?P<u>[a-z]+) (?:were|was) sold(?: for the " + EVT + r")?(?: in total)?",
    r"for the " + EVT + r",? exactly (?P<n>\d+) (?P<u>[a-z]+) (?:were|was) sold(?: in total)?",
    r"the " + EVT + r" (?:sold exactly (?P<n>\d+) (?P<u>[a-z]+)|had exactly (?P<n2>\d+) (?P<u2>[a-z]+) sold)(?: in total)?")]
SOLD_SHARE = re.compile(r"^(?:of (?:them|these|those),? )?exactly (?:" + SHARE + r"|(?P<k>\d+))"
                        r"(?: of (?:them|these|those|(?:those|these|the) (?P<u>[a-z]+)))? "
                        r"(?:were sold|were bought|went|sold) before (?:noon|midday)$")
SOLD_ONLINE = re.compile(r"^exactly (?:" + SHARE + r"|(?P<k>\d+)) of the morning (?P<u>[a-z]+) were (?:sold|bought) online$")
PERIODS = [(re.compile(r"\b(?:before (?:noon|midday)|in the morning|morning)\b"), " _m_ "),
           (re.compile(r"\b(?:after (?:noon|midday)|in the afternoon|afternoon)\b"), " _a_ "),
           (re.compile(r"\b(?:at the box office|box office)\b"), " _b_ "),
           (re.compile(r"\bonline\b"), " _o_ "),
           (re.compile(r"^i make it\b"), " ")]
SALE_WORDS = set("there were was sold got went bought purchased in the of at number count came to sales total "
                 "for only just exactly a all overall altogether".split())
BOUGHT = [re.compile(p) for p in (
    r"^(?:at @T,? )?i (?:bought|purchased|got) (?:mine|my (?:own )?(?P<u>[a-z]+)|one|a (?P<u2>[a-z]+))"
    r"(?: (?P<ch>online|at the box office))?(?: at @T)?$",
    r"^my (?:own )?(?P<u>[a-z]+) was (?:bought|purchased)(?: (?P<ch>online|at the box office))? at @T$")]


@register
class Tickets(Family):
    name = "qty_tickets"

    def parse(self, ctx):
        f, used = {}, []
        for i, sent in enumerate(ctx.sentences):
            parts = [p for p in re.split(r",\s*and\s+|;\s*", sent.lower().rstrip(". ")) if p]
            g = dict(f)  # kept only when every clause of the sentence is understood
            for p in parts:
                m = next((m for m in (rx.match(p) for rx in SOLD_TOTAL) if m), None)
                if m and "total" not in g:
                    gd = m.groupdict()
                    g.update(total=int(gd.get("n2") or m["n"]), unit=sing(gd.get("u2") or m["u"]), event=m["e"])
                    continue
                m = SOLD_SHARE.match(p)
                if m and "total" in g and "before" not in g and (not m["u"] or sing(m["u"]) == g["unit"]):
                    g["before"] = Fraction(int(m["k"])) if m["k"] else g["total"] * (
                        Fraction(m["p"]) / 100 if m["p"] else FRACW[m["f"]])
                    continue
                m = SOLD_ONLINE.match(p)
                if m and "before" in g and "online" not in g and sing(m["u"]) == g["unit"]:
                    g["online"] = (Fraction(m["p"]) / 100 if m["p"] else FRACW[m["f"]]) if not m["k"] else int(m["k"])
                    continue
                break
            else:
                if parts:
                    f = g
                    used.append(i)
        if "before" not in f or unparsed_mention(ctx, used, lambda w: sing(w) == f["unit"]):
            return None
        before = f["before"]
        if before.denominator != 1 or not 0 <= before <= f["total"]:
            return None
        counts = {("m", None): before, ("a", None): f["total"] - before, (None, None): Fraction(f["total"])}
        if "online" in f:
            on = f["online"] if isinstance(f["online"], int) else before * f["online"]
            if Fraction(on).denominator != 1 or not 0 <= on <= before:
                return None
            counts[("m", "o")] = Fraction(on)
            counts[("m", "b")] = before - on  # the generator's two channels: online or the box office
        for i in used:
            ctx.consume(i)
        words = SALE_WORDS | set((f["event"] or "").split())
        return {"counts": counts, "unit": f["unit"], "words": words, "event": f["event"]}

    def judge(self, ctx, f, msg):
        if msg.text.strip().endswith("?"):
            return None
        low = body(msg.text).lower()
        try:
            masked, times = mask_times(low)
        except ValueError:
            return None
        for rx in BOUGHT:
            m = rx.match(masked)
            if m is None:
                continue
            u = m["u"] or (m.groupdict().get("u2"))
            if (u and sing(u) != f["unit"]) or len(times) != 1 or times[0] == 720:
                return None
            ch = {"online": "o", "at the box office": "b"}.get(m["ch"])
            n = f["counts"].get(("m" if times[0] < 720 else "a", ch))
            if n is None:
                return None
            return Verdict(OK if n > 0 else LIE, self.name, f"{n} sold in that part of the day")
        if re.search(r"\d:\d\d", low):
            return None
        s = re.sub(r"\b" + re.escape(f["event"]) + r"\b", " _e_ ", low) if f["event"] else low  # "the morning walk"
        for rx, tag in PERIODS:
            s = rx.sub(tag, s)
        toks = re.findall(r"_[mabeo]_|\d+(?:\.\d+)?|[a-z']+", s)
        nums = [t for t in toks if t[0].isdigit()]
        if len(nums) != 1 or "." in nums[0] or any(
                t not in f["words"] and sing(t) != f["unit"] for t in toks if t[0].isalpha()):
            return None
        period = {t for t in toks if t in ("_m_", "_a_")}
        chan = {t for t in toks if t in ("_o_", "_b_")}
        if len(period) > 1 or len(chan) > 1:
            return None
        key = (period.pop()[1] if period else None, chan.pop()[1] if chan else None)
        want = f["counts"].get(key)
        if want is None:
            return None
        return Verdict(OK if int(nums[0]) == want else LIE, self.name, f"evidence {want} for {key}")


# ------------------------------------------------------------------------------------ qty_pieces: cut objects

OBJ = r"(?P<o>[a-z][a-z\-]*(?: [a-z\-]+)??)"
MEAS = r"exactly (?P<n>\d+(?:\.\d+)?) (?P<u>[a-z]+)"
CUT = r"cut into (?P<k>" + NUM + r") (?P<eq>equal )?(?P<p>[a-z]+),? (?:with nothing left over|and nothing was left over)"
UNEQUAL = r"(?:: (?P<d>.+))?"
# every accepted sentence, whole: the object and its size, optionally with the cut in the same sentence
WHOLE = [re.compile(r"^" + p + r"$") for p in (
    r"(?:the|a) " + OBJ + r" (?:was|is) " + MEAS + r" long(?:(?:,? and (?:it )?was|; it was) (?:then )?" + CUT
    + UNEQUAL + r")?",
    r"(?:the|a) " + OBJ + r" (?:measured|weighed|measures|weighs) " + MEAS + r"(?:(?:,? and (?:it )?was|; it was) "
    r"(?:then )?" + CUT + UNEQUAL + r")?",
    r"(?:the|a) " + OBJ + r",? (?:which (?:was|weighed|measured) )?" + MEAS + r"(?: long)?, was (?:then )?" + CUT + UNEQUAL,
    r"a " + OBJ + r" (?:measuring )?" + MEAS + r"(?: long)? was (?:then )?" + CUT + UNEQUAL,
    MEAS + r" long, the " + OBJ + r" was (?:then )?" + CUT + UNEQUAL,
    MEAS + r" was the (?:weight|length) of the " + OBJ + r", which was (?:then )?" + CUT + UNEQUAL)]
CUT_ONLY = re.compile(r"^(?:it|the (?P<o>[a-z][a-z\- ]*?)) was (?:then )?" + CUT + UNEQUAL + r"$")
RATIO = r"(?P<m>twice|three times|four times|\d+ times) as (?:long|heavy) as each of"
SHORTS = r"(?P<s>" + NUM + r") short (?P<p2>[a-z]+)"
LONG = r"one long (?P<p1>[a-z]+)"
SHAPES = [re.compile(r"^" + p + r"$") for p in (  # one long piece and equal short ones, after the colon
    LONG + r",? (?:that was |which was )?" + RATIO + r" the " + SHORTS + r",? which were equal",
    LONG + r",? (?:that was |which was )?" + RATIO + r" the " + SHORTS + r",? and the (?:" + NUM + r" )?short "
    r"[a-z]+ were equal",
    r"the " + SHORTS + r" were equal,? and the " + LONG + r" was " + RATIO + r" them",
    LONG + r",? and " + SHORTS + r" that were equal")]
RATIO_ALONE = re.compile(r"^the (?:one )?long (?P<p1>[a-z]+) was " + RATIO + r" the (?:" + NUM + r" )?short "
                         r"(?:ones|(?P<p2>[a-z]+))$")
PIECE_WORDS = {"piece", "slice", "length", "portion", "share", "part", "bit", "section", "segment", "strip", "chunk",
               "wedge"}


def ratio(m):
    w = m["m"]
    return 2 if w == "twice" else WORDNUM.get(w.split()[0]) or int(w.split()[0])


@register
class Pieces(Family):
    name = "qty_pieces"

    def parse(self, ctx):
        obj = cut = shape = rat = None
        idx = []
        for i, sent in enumerate(ctx.sentences):
            low = sent.lower().rstrip(". ")
            if obj is None:
                obj = next((m for m in (rx.match(low) for rx in WHOLE) if m), None)
                if obj is not None:
                    idx.append(i)
                    cut = obj if obj["k"] else None
                continue
            if cut is None:  # the cut must come in the very next sentence
                cut = CUT_ONLY.match(low)
                if cut is None or (cut["o"] and not same_name(cut["o"], obj["o"])):
                    return None
                idx.append(i)
                continue
            m = RATIO_ALONE.match(low)
            if m and rat is None and not cut["eq"]:
                rat = m
                idx.append(i)
            break
        if obj is None or cut is None:
            return None
        d, size = unit_of(obj["u"])
        k, noun = num(cut["k"]), sing(cut["p"])
        if d not in ("len", "mass") or k < 2:
            return None
        total = num(obj["n"]) * size
        if cut["eq"]:
            if cut["d"] or rat:
                return None
            sizes = {"mine": {total / k}, "each": {total / k}}
        else:
            shape = next((m for m in (rx.match(cut["d"] or "") for rx in SHAPES) if m), None)
            gd = shape.groupdict() if shape else {}
            r = shape if gd.get("m") else rat
            nouns = {sing(x) for x in (gd.get("p1"), gd.get("p2"), rat and rat["p1"], rat and rat["p2"]) if x}
            if shape is None or r is None or (gd.get("m") and rat) or nouns != {noun} or num(gd["s"]) != k - 1:
                return None
            short = total / (ratio(r) + k - 1)
            sizes = {"mine": {short, ratio(r) * short}, "long": {ratio(r) * short}, "short": {short},
                     "two short": {short * (k - 1)} if k == 3 else None}
        sizes["total"] = {total}
        words = set(obj["o"].split())
        if unparsed_mention(ctx, idx, lambda w: w in words or sing(w) == noun):
            return None
        for i in idx:
            ctx.consume(i)
        return {"name": obj["o"], "dim": d, "sizes": sizes, "frames": piece_frames(obj["o"], PIECE_WORDS | {noun})}

    def judge(self, ctx, f, msg):
        if msg.text.strip().endswith("?"):
            return None
        s = body(msg.text).lower()
        if not re.search(r"\d", s):
            if re.search(r"\bcut(?:ting)?\b", s) and set(f["name"].split()) <= set(re.findall(r"[a-z]+", s)) \
                    and not re.search(r"\b(?:not|never|nobody|no one)\b|n't\b", s):
                return Verdict(OK, self.name, "says who cut it")
            return None
        for ref, rx in f["frames"]:
            m = rx.match(s)
            if m is None:
                continue
            d, size = unit_of(m["u"])
            possible = f["sizes"].get(ref)
            if d != f["dim"] or not possible:
                return None
            res = settle(num(m["n"]) * size, precision(m["n"]) * size, possible)
            if res is None:
                return None
            return Verdict(res, self.name, f"{ref}: evidence {', '.join(str(float(p)) for p in sorted(possible))}")
        return None


def piece_frames(name, nouns):
    """Whole-message patterns for claims about one piece, the long or short ones, all pieces, or the whole object."""
    P, O = "(?:" + "|".join(sorted(nouns)) + ")", re.escape(name)
    Q = r"(?P<n>\d+(?:\.\d+)?) (?P<u>[a-z]+)"
    was = r"(?:was|is|measured|came to|came out at|weighed|weighed in at|ran to)"
    end = r"(?: long| in length)?"
    got = r"(?:got|had|ended up with|received|was given|took|kept|was handed|came away with)"
    made = r"(?:made|came to|totalled|totaled|added up to)"
    together = r"(?:together|combined|put together|added together)"
    return [(ref, re.compile(p)) for ref, p in (
        ("mine", r"^my (?:own )?" + P + r"(?: of (?:the )?" + O + r")? " + was + " " + Q + end + "$"),
        ("mine", r"^the " + P + r" (?:that )?i " + got + " " + was + " " + Q + end + "$"),
        ("mine", r"^i " + got + r" (?:a|one) " + P + r" (?:that|which) " + was + " " + Q + end + "$"),
        ("mine", r"^the (?:length|weight) of my " + P + r" (?:was|came to) " + Q + "$"),
        ("long", r"^the (?:one )?long " + P + " " + was + " " + Q + end + "$"),
        ("short", r"^(?:the|each|one|either) short " + P + " " + was + " " + Q + end + "$"),
        ("short", r"^each of the (?:two |three |four )?short " + P + "s " + was + " " + Q + end + "$"),
        ("two short", r"^" + together + r",? the two short " + P + r"s (?:" + made + r"|measured|weighed) " + Q + "$"),
        ("two short", r"^the two short " + P + r"s " + together + r" (?:" + made + r"|measured|weighed) " + Q + "$"),
        ("two short", r"^the two short " + P + r"s " + made + " " + Q
         + r"(?: together| combined| between them| in total)?$"),
        ("each", r"^(?:each|every) " + P + " " + was + " " + Q + end + "$"),
        ("each", r"^(?:all )?the " + P + r"s (?:were|measured|weighed) " + Q + end + " each$"),
        ("total", r"^the (?:whole |entire |full |original )?" + O + " " + was + " " + Q + end + r"(?: in total)?$"))]


# ------------------------------------------------------------------------------- qty_attendance: complete lists

LISTNOUN = r"[\w' -]*?(?:sheet|list|register|roster|record)"
PEOPLE = r"(?:people|names|attendees|persons|members)"
LIST_RES = [re.compile(p, re.I) for p in (
    r"^(?:the|its) (?:" + LISTNOUN + r")(?: for (?:the )?(?P<ev>(?:\d{1,2}:\d{2}|[^:,])+?))? (?:lists|names|shows|records) exactly "
    r"(?P<k>" + NUM + r") " + PEOPLE + r"[:,] (?P<names>.+)$",
    r"^exactly (?P<k>" + NUM + r") " + PEOPLE + r" (?:are listed on|appear on|are on|are named on) the (?:" + LISTNOUN
    + r") for (?:the )?(?P<ev>(?:\d{1,2}:\d{2}|[^:,])+?): (?P<names>.+)$")]
PLACE = r"(?:in|at) the [a-z]+(?: [a-z]+)?"
HELD = re.compile(r"^the (?P<ev>.+?) (?:was held|took place) " + PLACE + r"$")
DESCRIBED = [r"^the {ev} (?:was held|took place) " + PLACE + r"$", r"^the [a-z]+(?: [a-z]+)? (?:was|is) where the {ev} was held$",
             r"^the {ev} was the first of the year$", r"^it was the first {ev} of the year$"]
ALL_CAME = re.compile(r"^(?:(?:everyone|everybody) (?:on (?:it|the (?:[a-z\-]+ )?(?:list|sheet|register|roster))|listed)"
                      r"|all of them|they all|each of them|all " + NUM + r"(?: of them)?) attended$")
ONLY_THEM = re.compile(r"^(?:(?:everyone|everybody) who attended (?:is|was) (?:on it|listed|on the (?:[a-z\-]+ )?"
                       r"(?:list|sheet|register|roster))|(?:nobody|no one) else (?:attended|did|came|was there))$")
CAME = (r"(?:attended|went to|was at|was there at|sat through|sat in on|made it to|turned up (?:to|at|for)|"
        r"showed up (?:to|at|for)|took part in|was present at)")
STAYED_AWAY = (r"(?:missed|skipped|wasn't at|was not at|didn't attend|did not attend|didn't go to|did not go to|"
               r"never went to|stayed away from|didn't make it to|did not make it to|was absent from|sat out|"
               r"didn't turn up (?:to|at|for)|did not turn up (?:to|at|for)|wasn't present at|was not present at)")
NOT_A_NAME = set("i me everyone everybody nobody someone somebody anyone anybody he she they we it one all none no "
                 "the this that".split())
# a seat in the room where the event ran; "near the window in the office during the drill" is somewhere else
SEAT = (r"(?:sat|took a seat|was sitting|was seated|stood|was standing) (?:near|by|at|next to|beside) the "
        r"(?:door|doors|window|front|back|exit|entrance|stage|wall|aisle)")


@register
class Attendance(Family):
    name = "qty_attendance"

    def parse(self, ctx):
        f, used, held = {}, [], None
        for i, sent in enumerate(ctx.sentences):
            s = sent.rstrip(". ")
            low, times = mask_times(s.lower())
            m = HELD.match(low)
            if m and len(times) <= 1 and "names" not in f:
                held = (m["ev"], times, i)  # "The 1:30 PM safety briefing was held ... Its sign-in register lists ..."
                continue
            m = next((m for m in (rx.match(s) for rx in LIST_RES) if m), None)
            if m and "names" not in f:
                names = [x for x in re.split(r",\s*(?:and\s+)?|\s+and\s+", m["names"]) if x]
                if len(names) != num(m["k"]) or not all(re.fullmatch(r"[A-Z][a-zA-Z'\-]+", x) for x in names):
                    return None
                if m["ev"]:
                    ev, ts = mask_times(m["ev"].lower())
                else:  # "Its ... register": the event named in the sentence just before
                    ev, ts = held[:2] if held and held[2] == i - 1 else (None, None)
                if ev is None or len(ts) != 1:
                    return None
                noun = re.sub(r"^@T |^the @T | at @T$", "", ev)
                if "@T" in noun:
                    return None
                f.update(names={x.lower() for x in names}, noun=noun, time=ts[0])
                used.append(i)
                continue
            if "names" in f:
                parts = [p for p in re.split(r",\s*and\s+|;\s*|,\s*|\s+and\s+", low) if p]
                kinds = ["a" if ALL_CAME.match(p) else "b" if ONLY_THEM.match(p) else None for p in parts]
                if parts and all(kinds):
                    f["all_came"] = f.get("all_came") or "a" in kinds
                    f["only_them"] = f.get("only_them") or "b" in kinds
                    used.append(i)
        if not f.get("names") or not (f.get("all_came") or f.get("only_them")):
            return None
        word = f["noun"].split()[-1]
        ev = r"(?:@T )?(?:" + re.escape(f["noun"]) + "|" + re.escape(word) + ")"
        described = [re.compile(p.format(ev=ev)) for p in DESCRIBED]
        for i, sent in enumerate(ctx.sentences):  # where it was held, "the first of the year": nothing to judge
            low, times = mask_times(sent.lower().rstrip(". "))
            if i not in used and any(rx.match(low) for rx in described) and all(t == f["time"] for t in times):
                used.append(i)
        if unparsed_mention(ctx, used, lambda w: w == word or w in (
                "list", "sheet", "register", "roster", "attended", "attend", "attendance", "present", "absent")):
            return None  # another session of the event, or more about the list than this family reads
        for i in used:
            ctx.consume(i)
        n = re.escape(f["noun"])
        ev = r"(?:the )?(?:@T )?" + n + r"(?: at @T)?"
        f["frames"] = [(re.compile(p), sign) for p, sign in (
            (r"^(?P<who>[a-z][a-z'\-]*) " + CAME + " " + ev + r"$", True),
            (r"^(?P<who>[a-z][a-z'\-]*) " + STAYED_AWAY + " " + ev + r"$", False),
            (r"^(?P<who>[a-z][a-z'\-]*) gave " + ev + r" a miss$", False),
            (r"^" + ev + r" (?:was|is) one (?P<who>[a-z][a-z'\-]*) (?:attended|went to)$", True),
            (r"^" + ev + r" (?:was|is) one (?P<who>[a-z][a-z'\-]*) (?:skipped|missed)$", False),
            (r"^" + ev + r" (?:passed|went ahead|happened|took place) without (?P<who>[a-z][a-z'\-]*)$", False),
            (r"^during " + ev + r",? (?P<who>i) " + SEAT + r"$", True),
            (r"^(?P<who>i) " + SEAT + r" during " + ev + r"$", True))]
        f["headcount"] = [re.compile(p) for p in (
            r"^(?:exactly |only )?(?P<k>" + NUM + r") (?:people|attendees|persons) (?:attended|went to|were at|"
            r"turned up (?:to|at|for)|showed up (?:to|at|for)|took part in) " + ev + r"$",
            r"^" + ev + r" had (?:exactly |only )?(?P<k>" + NUM + r") (?:people|attendees)(?: there| in attendance)?$")]
        f["where"] = re.compile(r"^(?:at @T,? )?i was (?:in|at|out in) the (?P<loc>[a-z][a-z ]*?)(?: at @T)?$")
        f["evidence"] = ctx.case.evidence.lower()
        return f

    def judge(self, ctx, f, msg):
        if msg.text.strip().endswith("?"):
            return None
        text = body(msg.text)
        try:
            s, times = mask_times(text.lower())
        except ValueError:
            return None
        me = ctx.speakers.get(msg.label, "").lower()
        for rx, came in f["frames"]:
            m = rx.match(s)
            if m is None:
                continue
            if any(t != f["time"] for t in times) or len(times) > 1:
                return None  # a different session of the same kind of event
            who = m["who"]
            if who in ("i", "me"):
                who = me
            elif who in NOT_A_NAME or who in WORDNUM or not re.search(r"\b" + re.escape(who.capitalize()) + r"\b", text):
                return None
            listed = who in f["names"]
            if came and not listed and f.get("only_them"):
                if who not in {n.lower() for n in ctx.speakers.values()}:
                    return None  # a capitalised first word need not be a person; speakers surely are
                return Verdict(LIE, self.name, f"{who} is not on the complete list")
            if not came and listed and f.get("all_came"):
                return Verdict(LIE, self.name, f"{who} is on the list and everyone on it attended")
            return Verdict(OK, self.name, "agrees with the list")
        m = next((m for m in (rx.match(s) for rx in f["headcount"]) if m), None)
        if m and all(t == f["time"] for t in times) and f.get("all_came") and f.get("only_them"):
            ok = num(m["k"]) == len(f["names"])
            return Verdict(OK if ok else LIE, self.name, f"{len(f['names'])} attended")
        m = f["where"].match(s)
        if m and len(times) == 1:
            if m["loc"] in f["evidence"] or (me in f["names"] and times[0] == f["time"]):
                return None
            return Verdict(OK, self.name, "somewhere the evidence never mentions")
        return None



def _selfcheck():
    """python -m lc.rules.fam_quantity: made-up cases (not train rows) for each family, including abstentions."""
    from lc.data import Case, Message
    from .core import Ctx

    speakers = {"A": "Nina", "B": "Ade", "C": "Taro", "D": "Lena"}
    checks = [
        (Level, "The tank held 400 litres at 9:00 AM. Exactly 30 litres were removed at 10:00 AM. Nothing else was "
                "added or removed before 2:00 PM.",
         [("At 11:00 AM the tank held 370 litres.", OK), ("At 11:00 AM the tank held 400 litres.", LIE),
          ("At 10:00 AM the tank held 400 litres.", OK), ("I took some from the tank at 11:00 AM.", LIE),
          ("At 3:00 PM the tank held 10 litres.", None)]),
        (Level, "The pool held 150 litres at 9:00 AM and drained at a steady 30 litres per hour until 12:00 PM, with "
                "nothing added.", [("At 10:30 AM there were 105 litres in the pool.", OK),
                                   ("The pool had 120 litres in it at 10:30 AM.", LIE)]),
        (Level, "The tank held 400 litres at 9:00 AM, except on weekdays. Nothing else was added or removed before "
                "2:00 PM.", [("At 11:00 AM the tank held 300 litres.", None)]),
        (Tickets, "Exactly 50 passes were sold for the museum tour, and exactly 20% of them were sold before noon. "
                  "Exactly half of the morning passes were sold online.",
         [("Before noon, 10 passes were sold.", OK), ("After noon, 30 passes were sold.", LIE),
          ("5 of the morning passes were sold at the box office.", OK), ("At least 10 passes were sold.", None)]),
        (Pieces, "The rope was exactly 3 metres long. It was cut into three pieces with nothing left over: one long "
                 "piece twice as long as each of the two short pieces, which were equal.",
         [("The long piece measured 150 centimetres.", OK), ("My piece measured 100 centimetres.", LIE),
          ("Together, the two short pieces made 150 centimetres.", OK), ("The rope I got was 75 centimetres.", None)]),
        (Attendance, "The sign-in register for the 9:00 AM safety briefing lists exactly two people: Nina and Ade. "
                     "Everyone on the list attended, and everyone who attended is on it.",
         [("I skipped the 9:00 AM safety briefing.", LIE), ("Ade attended the safety briefing.", OK),
          ("I skipped the 4:00 PM safety briefing.", None)]),
    ]
    for fam_cls, evidence, msgs in checks:
        case = Case("check", "", speakers, evidence, [Message(i + 1, "A", t) for i, (t, _) in enumerate(msgs)])
        fam, ctx = fam_cls(), Ctx(case)
        facts = fam.parse(ctx)
        for m, (_, want) in zip(case.messages, msgs):
            v = fam.judge(ctx, facts, m) if facts is not None else None
            assert (v.status if v else None) == want, (fam.name, m.text, want, v)
    print("quantity self-check passed")


if __name__ == "__main__":
    _selfcheck()
