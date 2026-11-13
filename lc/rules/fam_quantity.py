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
    """Replace each clock time in s with '@T'; return (masked text, minutes since midnight in order)."""
    out = []

    def rep(m):
        if m.group(4):
            out.append(0 if m.group(4).lower() == "midnight" else 720)
        else:
            out.append(T.time_to_min(m.group(1), m.group(2), m.group(3)))
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
    """Same measure, or the same counted noun allowing a shorter name ('envelope' for 'sealed envelope')."""
    if a == b:
        return True
    return a.startswith("count:") and b.startswith("count:") and (
        a.endswith(" " + b[6:]) or b.endswith(" " + a[6:]))


def head(c):
    """Container name without its article or a trailing place phrase: 'the cabinet by the window' -> 'cabinet'."""
    c = re.sub(r"^(?:the|that|this|a|an) ", "", c.strip().lower())
    return re.sub(r" (?:by|near|beside|next to|in|on|at|under|behind|outside|inside) .*$", "", c)


def same_name(a, b):
    a, b = head(a), head(b)
    return a == b or a.endswith(" " + b) or b.endswith(" " + a)


NUMBER_IN = re.compile(r"\d|%|\b(?:" + "|".join(w for w in WORDNUM if w != "one") + r")\b")


def unparsed_mention(ctx, used, key, time_ok=lambda t: True):
    """True when an evidence sentence the family did not parse names its subject (a word for which key() is true)
    together with a number, so the parsed facts may be incomplete. Clock times count only when time_ok rejects them."""
    for i, sent in enumerate(ctx.sentences):
        if i in used:
            continue
        low, times = mask_times(T.DATE_RE.sub("", sent).lower())
        if any(key(w) for w in re.findall(r"[a-z]+", low)) and (
                NUMBER_IN.search(low) or not all(time_ok(t) for t in times)):
            return True
    return False


# -------------------------------------------------------------------------------------- qty_level: containers

C = r"(?:the|that|this) (?P<c>[a-z][a-z'\- ]*?)"
QTY = r"(?:exactly )?(?P<n>" + NUM + r") (?P<u>[a-z]+(?: [a-z]+)?)(?: of [a-z]+)?"  # "150 bags (of sand)"
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
EVENT = re.compile(r"^(?:then )?(?:at @T,? )?(?:exactly )?(?P<n>" + NUM + r") (?P<u>[a-z]+(?: [a-z]+)?) (?:was|were) "
                   r"(?P<v>" + "|".join(ADDED) + r"|removed|taken out|taken|taken away|drawn off|used|poured out|"
                   r"tipped out|scooped out|pumped out|drained off)"
                   r"(?: (?:to|into|from|out of|off) (?:it|" + C + r"))?(?: at @T)?$")
CLOSE = re.compile(r"^(?:and )?(?:with )?(?:nothing (?:else )?|no other [a-z]+ )(?:was )?"
                   r"(?P<what>added or removed|removed or added|put in or taken out|added)"
                   r"(?: (?:to|from|in|out of) it)?(?: (?:before|until) @T)?$"
                   r"|^(?:and )?there (?:were|was) no other (?:additions or removals|changes)(?: (?:before|until) @T)?$")
NOCHANGE = re.compile(r"^(?:and )?no (?P<u>[a-z]+(?: [a-z]+)?) (?:was|were) (?:added or removed|removed or added)"
                      r"(?: during that time| in that time| during that period)?$")
SUBJECT = re.compile(r"^" + C + r" (?:stands|stood|sits|sat|is|was) (?!.*@T)[a-z ,']+$")
LEADING_TIME = re.compile(r"^(?:at @T|" + SPAN + r"|until @T|from then (?:to|until) @T)$")
SPLIT = re.compile(r"\s*;\s*|,\s*(?:and |then |while )?|\s+and\s+(?=(?:then |it |drained|filled|emptied|leaked|exactly"
                   r"|nothing|no |from |until |at @T))")
ASIDE = re.compile(r", (?:which|that) (?:is|was) [a-z ]+?,")  # "the cabinet, which is grey, held ..."


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
        at_t = 0
        for te, d in self.events:
            if te < t:
                a += d
            elif te == t:
                at_t += d
        for ra, rb, r in self.rates:
            lo, hi = max(ra, t0), min(rb, t)
            if hi > lo:
                a += r * (hi - lo)
        possible = {a, a + at_t} if at_t else {a}
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


def pick(stores, name):
    found = [s for s in stores if same_name(name, s.name)]
    return found[0] if len(found) == 1 else None


CQ = r"(?:exactly |only |just )?(?P<n>" + NUM + r") (?P<u>[a-z]+(?: [a-z]+)?)(?: of [a-z]+)?"
AMOUNT = [re.compile(r"^(?:at @T,? )?(?:" + f.format(Q=CQ, C=C) + r")(?: at @T)?$") for f in (
    r"there (?:were|was) {Q} in {C}",
    r"{C} (?:held|had|contained|still held|still had|was holding) {Q}(?: in it)?",
    r"{Q} (?:were|was) in {C}",
    r"i (?:counted|make it|measured|recorded|noted|found|photographed all|counted all|saw all"
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

        for i, sent in enumerate(ctx.sentences):
            low, times = mask_times(ASIDE.sub("", sent.lower().rstrip(". ")))
            m = SUBJECT.match(low)
            if m:
                subjects.append((i, m["c"]))
                continue
            parts = clauses(low, times)
            done, touched = [], []
            for p, ts in parts:
                ok = self.apply(p, ts, stores, store_for, cur, subjects, touched)
                if not ok:
                    break
                done.append(p)
                cur = touched[-1] if touched else cur
            if len(done) == len(parts) and done:
                used.append(i)
                continue
            for st in touched:  # half-understood sentence about a store: its model may be incomplete
                st.bad = True
            for st in stores:  # an unparsed sentence that mentions a store's container or unit
                if re.search(r"\b" + re.escape(head(st.name)) + r"\b", low) or (
                        not st.unit[0].startswith("count:") and re.search(r"\d+(?:\.\d+)? " + r"[a-z]+", low)
                        and any(sing(w) in UNITS and UNITS[sing(w)][0] == st.unit[0]
                                for w in re.findall(r"\d+(?:\.\d+)? ([a-z]+)", low))):
                    st.bad = True
        for st in stores:
            st.finish()
        good = [s for s in stores if not s.bad]
        if not good:
            return None
        for i in used:
            ctx.consume(i)
        for i, name in subjects:
            if any(same_name(name, s.name) for s in good):
                ctx.consume(i)
        return good

    def apply(self, p, ts, stores, store_for, cur, subjects, touched):
        """Add one clause to the stores; False when the clause is not understood."""
        m = next((m for m in (rx.match(p) for rx in STATES) if m), None)
        if m and bool(m["pre"]) != bool(m["post"]):
            phrase = m["pre"] or m["post"]
            st = self.subject(m, stores, store_for, cur, subjects, m["u"])
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
            st = self.subject(m, stores, store_for, cur, subjects, m["u"], need_store=True)
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
    def subject(m, stores, store_for, cur, subjects, u, need_store=False):
        if m["c"]:
            if need_store:
                return pick(stores, m["c"])
            return store_for(m["c"], unit_of(u))
        if m["it"] or need_store:
            if cur is not None:
                return cur
            if m["it"] and subjects and not need_store:
                return store_for(subjects[-1][1], unit_of(u))
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
            st = pick(stores, m["c"])
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
            st = pick(stores, m["c"])
            u = m["u"] or m["u2"]
            if st is None or len(times) != 1 or (u and not same_dim(unit_of(u)[0], st.unit[0])):
                return None
            res = st.removal(times[0])
            return Verdict(res, self.name, f"removal from {st.name} at {T.fmt_time(times[0])}") if res else None
        m = NEUTRAL.match(s)
        if m and len(times) <= 1 and pick(stores, m["c"]):
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
    r"the " + EVT + r" (?:sold|had) exactly (?P<n>\d+) (?P<u>[a-z]+)(?: sold)?(?: in total)?")]
SOLD_SHARE = re.compile(r"^(?:of (?:them|these|those),? )?exactly (?:" + SHARE + r"|(?P<k>\d+))"
                        r"(?: of (?:them|these|those|(?:those|these|the) (?P<u>[a-z]+)))? "
                        r"(?:were sold|were bought|went|sold) before (?:noon|midday)$")
SOLD_ONLINE = re.compile(r"^exactly (?:" + SHARE + r"|(?P<k>\d+)) of the morning (?P<u>[a-z]+) were (?:sold|bought) online$")
PERIODS = [(re.compile(r"\b(?:before (?:noon|midday)|in the morning|morning)\b"), " _m_ "),
           (re.compile(r"\b(?:after (?:noon|midday)|in the afternoon|afternoon)\b"), " _a_ "),
           (re.compile(r"\b(?:at the box office|box office)\b"), " _b_ "),
           (re.compile(r"\bonline\b"), " _o_ "),
           (re.compile(r"^i make it\b"), " ")]
SALE_WORDS = set("there were was sold got went bought purchased booked in the of at number count came to sales total "
                 "for only just exactly a all overall altogether".split())
BOUGHT = [re.compile(p) for p in (
    r"^(?:at @T,? )?i (?:bought|purchased|got|booked) (?:mine|my (?:own )?(?P<u>[a-z]+)|one|a (?P<u2>[a-z]+))"
    r"(?: (?P<ch>online|at the box office))?(?: at @T)?$",
    r"^my (?:own )?(?P<u>[a-z]+) was (?:bought|purchased|booked)(?: (?P<ch>online|at the box office))? at @T$")]


@register
class Tickets(Family):
    name = "qty_tickets"

    def parse(self, ctx):
        f, used = {}, []
        for i, sent in enumerate(ctx.sentences):
            parts = [p for p in re.split(r",\s*and\s+|;\s*", sent.lower().rstrip(". ")) if p]
            got = []
            for p in parts:
                m = next((m for m in (rx.match(p) for rx in SOLD_TOTAL) if m), None)
                if m and "total" not in f:
                    f.update(total=int(m["n"]), unit=sing(m["u"]), event=m["e"])
                    got.append(p)
                    continue
                m = SOLD_SHARE.match(p)
                if m and "total" in f and "before" not in f and (not m["u"] or sing(m["u"]) == f["unit"]):
                    f["before"] = Fraction(int(m["k"])) if m["k"] else f["total"] * (
                        Fraction(m["p"]) / 100 if m["p"] else FRACW[m["f"]])
                    got.append(p)
                    continue
                m = SOLD_ONLINE.match(p)
                if m and "before" in f and sing(m["u"]) == f["unit"]:
                    f["online"] = (Fraction(m["p"]) / 100 if m["p"] else FRACW[m["f"]]) if not m["k"] else int(m["k"])
                    got.append(p)
                    continue
                break
            if got and len(got) == len(parts):
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
        return {"counts": counts, "unit": f["unit"], "words": words}

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
        s = low
        for rx, tag in PERIODS:
            s = rx.sub(tag, s)
        toks = re.findall(r"_[mabo]_|\d+(?:\.\d+)?|[a-z']+", s)
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

MEASURE = re.compile(r"\bexactly (?P<n>\d+(?:\.\d+)?) (?P<u>[a-z]+)\b")
OBJECT = re.compile(r"\b(?:the|a) (?P<o>[a-z][a-z\-]*?(?: [a-z\-]+?)??)(?=,| was\b| weighed\b| measured\b| measuring\b"
                    r"| exactly\b| which\b|$)")
CUT = re.compile(r"\bcut into (?P<k>" + NUM + r") (?P<eq>equal )?(?P<p>[a-z]+)")
LEFT_OVER = re.compile(r"\bnothing (?:was )?left over\b")
RATIO = re.compile(r"\b(?P<m>twice|three times|four times|\d+ times) as (?:long|heavy|big|large) as each of "
                   r"(?:the (?:[a-z]+ )?short (?:[a-z]+|ones)|them)\b")
PIECE_WORDS = {"piece", "slice", "length", "portion", "share", "part", "bit", "section", "segment", "strip", "chunk",
               "wedge"}
PIECE_TALK = set("my own the a one of i got had ended up with received was given took kept handed came away that which "
                 "measured measures measure to out at weighed weighs weigh in long length is were made make together "
                 "put added combined between them two both short each every all ran totalled totaled total ones it "
                 "me from came".split())


def ratio(m):
    w = m["m"]
    return 2 if w == "twice" else WORDNUM.get(w.split()[0]) or int(w.split()[0])


@register
class Pieces(Family):
    name = "qty_pieces"

    def parse(self, ctx):
        obj = None
        for i, sent in enumerate(ctx.sentences):
            low = sent.lower().rstrip(". ")
            if obj is None:
                m, o = MEASURE.search(low), OBJECT.search(low)
                if not (m and o and re.search(r"\b(?:long|measur\w*|weigh\w*|weight)\b", low)):
                    continue
                d, size = unit_of(m["u"])
                if d not in ("len", "mass") or len(re.findall(r"\d+(?:\.\d+)?", low)) != 1:
                    continue
                obj = {"name": o["o"], "dim": d, "total": num(m["n"]) * size, "idx": [i], "text": low}
                cut = CUT.search(low)
                if cut is None:
                    continue
            else:
                if not re.match(r"^(?:it|the " + re.escape(obj["name"]) + r") was (?:then )?cut into ", low) \
                        and "cut" in obj and not re.search(r"\b(?:long|short) (?:[a-z]+|ones)\b", low):
                    break
                obj["idx"].append(i)
                obj["text"] += " | " + low
                cut = CUT.search(low)
                if cut is None and "cut" not in obj:
                    return None
            if cut is not None:
                if "cut" in obj:
                    return None
                obj["cut"] = cut
            if "cut" in obj and (obj["cut"]["eq"] or RATIO.search(obj["text"])):
                break
        if obj is None or "cut" not in obj:
            return None
        cut, text = obj["cut"], obj["text"]
        k = num(cut["k"])
        if not LEFT_OVER.search(text) or k < 2 or re.search(r"\b(?:except|apart from|but|about|roughly|around)\b", text):
            return None
        noun = sing(cut["p"])
        if cut["eq"]:
            if re.search(r"\b(?:long|short) " + noun, text):
                return None
            sizes = {"mine": {obj["total"] / k}, "each": {obj["total"] / k}}
        else:
            r = RATIO.search(text)
            shorts = re.search(r"\b(?P<s>" + NUM + r") short " + noun + r"s\b", text)
            if not (r and shorts and re.search(r"\b(?:one|the) long " + noun + r"\b", text) and "equal" in text
                    and num(shorts["s"]) == k - 1):
                return None
            short = obj["total"] / (ratio(r) + k - 1)
            sizes = {"mine": {short, ratio(r) * short}, "long": {ratio(r) * short}, "short": {short},
                     "two short": {short * (k - 1)} if k == 3 else None}
        sizes["total"] = {obj["total"]}
        words = set(obj["name"].split())
        if unparsed_mention(ctx, obj["idx"], lambda w: w in words or sing(w) == noun):
            return None
        for i in obj["idx"]:
            ctx.consume(i)
        return {"name": obj["name"], "dim": obj["dim"], "sizes": sizes, "nouns": PIECE_WORDS | {noun}}

    def judge(self, ctx, f, msg):
        if msg.text.strip().endswith("?"):
            return None
        s = body(msg.text).lower()
        if re.search(r"\d:\d\d", s):
            return None
        toks = re.findall(r"\d+(?:\.\d+)?|[a-z']+", s)
        obj_words = set(f["name"].split())
        if not re.search(r"\d", s):
            if re.search(r"\bcut(?:ting)?\b", s) and obj_words <= set(toks) and not re.search(
                    r"\b(?:not|never|nobody|no one|didn't|wasn't|isn't)\b|n't\b", s):
                return Verdict(OK, self.name, "says who cut it")
            return None
        qty = re.findall(r"(\d+(?:\.\d+)?) ([a-z]+)", s)
        if len(qty) != 1 or len(re.findall(r"\d+(?:\.\d+)?", s)) != 1:
            return None
        n, u = qty[0]
        d, size = unit_of(u)
        if d != f["dim"]:
            return None
        nouns = {w for w in toks if sing(w) in f["nouns"]}
        if any(w not in PIECE_TALK and w not in obj_words and w not in nouns and w != u and not w[0].isdigit()
               for w in toks):
            return None
        P = r"(?:" + "|".join(sorted(f["nouns"])) + r")"
        if re.search(r"\b(?:two|both) short " + P + r"s\b", s):
            ref = "two short" if re.search(r"\b(?:together|combined|between them|in total|altogether)\b", s) else None
        elif re.search(r"\b(?:the|each|one|a) short " + P + r"\b", s):
            ref = "short"
        elif re.search(r"\bthe (?:one )?long " + P + r"\b", s):
            ref = "long"
        elif re.search(r"\b(?:each|every) " + P + r"\b|\b(?:the|all the|all) " + P + r"s\b.*\beach\b", s):
            ref = "each"
        elif re.search(r"\bmy (?:own )?" + P + r"\b|\b" + P + r" (?:that )?i (?:got|had|ended up with|received|was given|"
                       r"took|kept|was handed|came away with)\b|\bi (?:got|had|received|was given|ended up with|took|"
                       r"kept|came away with) (?:a|one|the) " + P + r"\b", s):
            ref = "mine"
        elif not nouns and re.search(r"\bthe (?:whole |entire |full )?" + re.escape(f["name"]) + r"\b", s):
            ref = "total"
        else:
            ref = None
        possible = f["sizes"].get(ref) if ref else None
        if not possible:
            return None
        res = settle(num(n) * size, precision(n) * size, possible)
        if res is None:
            return None
        return Verdict(res, self.name, f"{ref}: evidence {', '.join(str(float(p)) for p in sorted(possible))}")


# ------------------------------------------------------------------------------- qty_attendance: complete lists

LISTNOUN = r"[\w' -]*?(?:sheet|list|register|roster|record)"
PEOPLE = r"(?:people|names|attendees|persons|members)"
LIST_RES = [re.compile(p, re.I) for p in (
    r"^(?:the|its) (?:" + LISTNOUN + r")(?: for (?:the )?(?P<ev>(?:\d{1,2}:\d{2}|[^:,])+?))? (?:lists|names|shows|records) exactly "
    r"(?P<k>" + NUM + r") " + PEOPLE + r"[:,] (?P<names>.+)$",
    r"^exactly (?P<k>" + NUM + r") " + PEOPLE + r" (?:are listed on|appear on|are on|are named on) the (?:" + LISTNOUN
    + r") for (?:the )?(?P<ev>(?:\d{1,2}:\d{2}|[^:,])+?): (?P<names>.+)$")]
HELD = re.compile(r"^the (?P<ev>.+?) (?:was held|took place)\b")
DESCRIBED = re.compile(r"\b(?:was held|took place|where the|of the year)\b")
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
        word = re.escape(f["noun"].split()[-1])
        for i, sent in enumerate(ctx.sentences):  # where it was held, "the first of the year": nothing to judge
            low, times = mask_times(sent.lower())
            if i not in used and re.search(r"\b" + word + r"\b", low) and DESCRIBED.search(low) and not re.search(
                    r"\d", low) and all(t == f["time"] for t in times):
                used.append(i)
        if unparsed_mention(ctx, used, lambda w: w == f["noun"].split()[-1] or w in ("list", "sheet", "register"),
                            lambda t: t == f["time"]):
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
            (r"^during " + ev + r",? (?P<who>i) (?:sat|took a seat|was sitting|was seated|stood|was standing) "
             r"(?:near|by|at|next to|beside) the [a-z ]+$", True),
            (r"^(?P<who>i) (?:sat|took a seat|was sitting|was seated|stood|was standing) (?:near|by|at|next to|beside) "
             r"the [a-z ]+? during " + ev + r"$", True))]
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

