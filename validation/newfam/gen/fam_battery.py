"""A device battery that falls at a steady number of percentage points per hour from a known level, and sometimes then
charges at a steady rate up to 100% and stays there. Outside the span the evidence covers, the level is unknown."""
import math
from fractions import Fraction as Fr

from .base import Bundle, Claim, clock, facts

NAME = "battery"
DEVICES = ["phone", "tablet", "e-reader", "laptop"]

# (first-person templates when the sender owns the device, third-person templates otherwise)
T = {
    "level": (["My {d} was at {v}% at {t}.", "At {t} my {d} battery showed {v}%.", "I had {v}% left on my {d} at {t}."],
              ["{p}'s {d} was at {v}% at {t}.", "At {t} {p}'s {d} battery showed {v}%.",
               "{p}'s {d} had {v}% left at {t}."]),
    "flat": (["My {d} was completely flat at {t}.", "At {t} my {d} battery was at 0%."],
             ["{p}'s {d} was completely flat at {t}.", "At {t} {p}'s {d} battery was at 0%."]),
    "full": (["My {d} was fully charged at {t}.", "At {t} my {d} showed 100%."],
             ["{p}'s {d} was fully charged at {t}.", "At {t} {p}'s {d} showed 100%."]),
    "lost": (["My {d} lost {x} percentage points between {a} and {b}."],
             ["{p}'s {d} lost {x} percentage points between {a} and {b}.",
              "Between {a} and {b} {p}'s {d} dropped by {x} percentage points."]),
    "least": (["My {d} still had at least {v}% at {t}."], ["{p}'s {d} still had at least {v}% at {t}."]),
    "between": (["At {t} my {d} was somewhere between {lo}% and {hi}%."],
                ["At {t} {p}'s {d} was somewhere between {lo}% and {hi}%."]),
    "back_up": (["At {t} my {d} was back up to {v}%."], ["At {t} {p}'s {d} was back up to {v}%."]),
    "down_to": (["At {t} my {d} was down to {v}%."], ["At {t} {p}'s {d} was down to {v}%."]),
}


def level(f, t):
    """The battery level at minute t as a Fraction, or None outside the span the evidence covers."""
    if not f.t0 <= t <= f.end:
        return None
    if t <= f.t1:
        return f.b0 - Fr(f.r * (t - f.t0), 60)
    return min(Fr(100), f.b1 + Fr(f.c * (t - f.t1), 60))


def can_level(f, t, v):
    lv = level(f, t)
    return 0 <= v <= 100 if lv is None else lv == v


def can_least(f, t, v):
    lv = level(f, t)
    return v <= 100 if lv is None else lv >= v


def can_between(f, t, lo, hi):
    lv = level(f, t)
    return lo <= 100 and hi >= 0 if lv is None else lo <= lv <= hi


def can_lost(f, ta, tb, x):
    la, lb = level(f, ta), level(f, tb)
    return la is None or lb is None or la - lb == x


def make(rng, people, others, day):
    d = rng.choice(DEVICES)
    p = rng.choice(people) if rng.random() < 0.7 else others[0]
    r = rng.choice([4, 5, 6, 8, 10, 12, 15, 20])
    b0 = rng.randrange(60, 101)
    t0 = rng.randrange(6 * 60 + 30, 11 * 60 + 1, 5)
    spans = [x for x in range(120, 481, 5) if r * x % 60 == 0 and b0 - r * x // 60 >= 5 and t0 + x <= 18 * 60]
    t1 = t0 + rng.choice(spans)
    b1 = b0 - r * (t1 - t0) // 60
    charge = rng.random() < 0.5
    c, end = 0, t1
    if charge:
        rates = [x for x in (20, 25, 30, 40, 50, 60) if t1 + (100 - b1) * 60 / x <= 22 * 60]
        c = rng.choice(rates)
        full_at = 5 * math.ceil((t1 + Fr((100 - b1) * 60, c)) / 5)  # first 5-minute mark at 100%
        end = full_at + rng.choice([x for x in (20, 40, 60, 90) if full_at + x + 60 <= 23 * 60 + 30])
    f = facts(t0=t0, b0=b0, r=r, t1=t1, b1=b1, c=c, end=end)
    grid = [t for t in range(t0, end + 1, 5) if level(f, t).denominator == 1]
    drain = [t for t in grid if t0 < t < t1]
    charging = [t for t in grid if t1 < t and level(f, t) < 100]
    full = [t for t in grid if t > t1 and level(f, t) == 100]
    sp = list(people)

    def say(kind, s, **kw):
        return rng.choice(T[kind][0] if s == p else T[kind][1]).format(p=p, d=d, **kw)

    truths = []

    def add(kind, ok, tags=(), **kw):
        s = rng.choice(sp)
        truths.append(Claim(s, say(kind, s, **kw), ok, tags))

    def lv(t):
        return int(level(f, t))

    add("level", lambda f: can_level(f, t0, b0), v=b0, t=clock(t0))
    add("level", lambda f: can_level(f, t1, b1), v=b1, t=clock(t1))
    for t in rng.sample(drain, min(3, len(drain))):
        add("level", lambda f, t=t, v=lv(t): can_level(f, t, v), v=lv(t), t=clock(t))
    ta, tb = sorted(rng.sample([t0, t1] + drain, 2))
    add("lost", lambda f, ta=ta, tb=tb, x=lv(ta) - lv(tb): can_lost(f, ta, tb, x), x=lv(ta) - lv(tb), a=clock(ta),
        b=clock(tb))
    t = rng.choice(drain)
    add("least", lambda f, t=t, v=lv(t): can_least(f, t, v), ("boundary",), v=lv(t), t=clock(t))
    t = rng.choice(drain)
    lo, hi = (lv(t), lv(t) + 10) if rng.random() < 0.5 else (max(lv(t) - 10, 0), lv(t))
    add("between", lambda f, t=t, lo=lo, hi=hi: can_between(f, t, lo, hi), ("boundary",), lo=lo, hi=hi, t=clock(t))
    if charge:
        for t in rng.sample(charging, min(2, len(charging))):
            add("level", lambda f, t=t, v=lv(t): can_level(f, t, v), v=lv(t), t=clock(t))
        t = rng.choice(full)
        add("full", lambda f, t=t: can_level(f, t, 100), ("boundary",) if level(f, t - 5) < 100 else (), t=clock(t))
        ta = end + 60  # after the span the level is unknown: lower again is possible
        v = rng.randrange(80, 100)
        add("down_to", lambda f, ta=ta, v=v: can_level(f, ta, v), ("distractor",), v=v, t=clock(ta))
    else:
        ta = t1 + rng.choice([60, 90, 120])
        v = min(100, b1 + rng.choice([20, 30, 40, 50]))
        if ta <= 23 * 60:  # charged again after the span: sounds wrong for a draining battery, but possible
            add("back_up", lambda f, ta=ta, v=v: can_level(f, ta, v), ("distractor",), v=v, t=clock(ta))
    tb = t0 - rng.choice([30, 60, 90])
    if tb >= 5 * 60 and (100 - b0) * 60 <= r * (t0 - tb):  # full earlier, at the same drain rate: unknown, possible
        add("full", lambda f, tb=tb: can_level(f, tb, 100), ("distractor",), t=clock(tb))
    for _ in range(3):
        t = clock(rng.randrange(7 * 60, 22 * 60, 5))
        truths.append(Claim(rng.choice(sp), rng.choice([
            f"I bought a sandwich at {t}.", f"At {t} I looked up the bus times.",
            f"I sent {others[-1]} an email at {t}."]), lambda f: True, ("neutral",)))

    s = rng.choice(sp)
    about = () if s == p else ("about-other",)
    kinds = ["level_wrong", "flat", "full_early", "lost_wrong", "least_wrong"] + (["charge_wrong"] if charge else [])
    lie = None
    while lie is None:
        kind = rng.choice(kinds)
        if kind == "level_wrong":
            t = rng.choice(drain + charging)
            cand = [lv(t) + k for k in (-r, r, -2 * r, 2 * r) if 0 <= lv(t) + k <= 100]
            v = rng.choice(cand)
            lie = Claim(s, say("level", s, v=v, t=clock(t)), lambda f, t=t, v=v: can_level(f, t, v),
                        (("two-fact",) if t > t1 else ()) + about)  # a charging level needs both rates
        elif kind == "flat":
            cand = [t for t in drain if lv(t) >= 5]
            if cand:
                t = rng.choice(cand)
                lie = Claim(s, say("flat", s, t=clock(t)), lambda f, t=t: can_level(f, t, 0), about)
        elif kind == "full_early":
            t = rng.choice(drain + charging)
            lie = Claim(s, say("full", s, t=clock(t)), lambda f, t=t: can_level(f, t, 100),
                        (("two-fact",) if t > t1 else ()) + about)
        elif kind == "lost_wrong":
            ta, tb = sorted(rng.sample([t0, t1] + drain, 2))
            x = lv(ta) - lv(tb)
            x2 = rng.choice([y for y in (x + r, x - r, 2 * x, x // 2) if y > 0 and y != x])
            lie = Claim(s, say("lost", s, x=x2, a=clock(ta), b=clock(tb)),
                        lambda f, ta=ta, tb=tb, x2=x2: can_lost(f, ta, tb, x2), about)
        elif kind == "least_wrong":
            t = rng.choice(drain)
            v = lv(t) + rng.choice([5, 10])
            if v <= 100:
                lie = Claim(s, say("least", s, v=v, t=clock(t)), lambda f, t=t, v=v: can_least(f, t, v), about)
        else:  # while charging, the level the drain alone would give: needs both rates
            cand = [t for t in charging if 0 <= b0 - Fr(r * (t - t0), 60) <= 100
                    and (b0 - Fr(r * (t - t0), 60)).denominator == 1]
            if cand:
                t = rng.choice(cand)
                v = int(b0 - Fr(r * (t - t0), 60))
                lie = Claim(s, say("level", s, v=v, t=clock(t)), lambda f, t=t, v=v: can_level(f, t, v),
                            ("two-fact",) + about)

    ev = [f"At {clock(t0)} {p}'s {d} battery was at {b0}%.",
          rng.choice([f"From {clock(t0)} to {clock(t1)} the {d} was never charged, and its battery level fell at a "
                      f"steady rate of {r} percentage points per hour.",
                      f"It was not charged at any time from {clock(t0)} to {clock(t1)}, and over that time its "
                      f"battery level dropped steadily by {r} percentage points every hour."])]
    if charge:
        ev.append(f"At {clock(t1)} {p} put it on charge, and it stayed on charge until {clock(end)}.")
        ev.append(f"While it was charging, its level rose at a steady {c} percentage points per hour until it "
                  f"reached 100%, and then it stayed at 100%.")
    group = rng.choice(["Friends", "Colleagues", "Housemates", "Travellers"])
    return Bundle(
        NAME,
        rng.choice([f"{group} argue over how fast {p}'s {d} battery ran down.",
                    f"{group} compare notes on {p}'s {d} battery during a long day.",
                    f"{group} try to work out how much charge {p}'s {d} had."]),
        group, f"{p}'s {d} battery", ev, f, lie, truths,
        fillers=["Should we buy a power bank?", "Does anyone have a spare charger?", "Is the battery getting old?",
                 "Batteries never last a whole day.", "Who keeps borrowing the charger?"],
        decor=[f"{p}'s {d} has a cracked case.", "The charger was a spare one from the office.",
               f"The {d} is a year old."])
