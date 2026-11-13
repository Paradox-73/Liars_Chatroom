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


# -------------------------------------------------------------------------------------- qty_level: containers

C = r"(?:the|that|this) (?P<c>[a-z][a-z'\- ]*?)"
QTY = r"(?:exactly )?(?P<n>" + NUM + r") (?P<u>[a-z]+(?: [a-z]+)?)"
SPAN = r"from @T (?:to|until) @T"
STATE = re.compile(r"^(?:(?P<pre>at @T|" + SPAN + r"),? )?(?:" + C + r"|(?P<it>it)) (?:held|contained|had) " + QTY
                   + r"(?: in it)?(?: (?P<post>at @T|" + SPAN + r"))?$")
RATE = re.compile(r"^(?:(?P<pre>" + SPAN + r"|from then (?:to|until) @T|until @T),? )?(?:(?:" + C + r"|(?P<it>it)) )?"
                  r"(?:then )?(?P<v>drained|filled|emptied|leaked) (?:at a steady|at a constant|steadily at)(?: rate of)? "
                  r"(?P<r>\d+(?:\.\d+)?) (?P<u>[a-z]+) (?:per|an|a|each|every) (?P<per>hour|minute)"
                  r"(?: (?P<post>" + SPAN + r"|from then (?:to|until) @T|until @T))?$")
EVENT = re.compile(r"^(?:then )?(?:at @T,? )?exactly (?P<n>" + NUM + r") (?P<u>[a-z]+(?: [a-z]+)?) (?:was|were) "
                   r"(?P<v>added|removed|taken out|poured in|put in|tipped in|drawn off|taken away)"
                   r"(?: (?:to|into|from|out of) (?:it|" + C + r"))?(?: at @T)?$")
CLOSE = re.compile(r"^(?:and )?(?:with )?nothing (?:else )?(?:was )?(?P<what>added or removed|removed or added|added)"
                   r"(?: (?:to|from) it)?(?: (?:before|until) @T)?$")
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
    if phrase.startswith("from @T"):
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
        return ({a, a + at_t} if at_t else {a}), True

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


CQ = r"(?:exactly |only |just )?(?P<n>" + NUM + r") (?P<u>[a-z]+(?: [a-z]+)?)"
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
        m = STATE.match(p)
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
            sign = 1 if m["v"] == "filled" else -1
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
            sign = 1 if m["v"] in ("added", "poured in", "put in", "tipped in") else -1
            st.events.append((ts[0], sign * num(m["n"]) * u[1] / st.unit[1]))
            return True
        m = CLOSE.match(p)
        if m and cur is not None and len(ts) <= 1:
            touched.append(cur)
            cur.close.append(("add" if m["what"] == "added" else "all", ts[0] if ts else None))
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
