"""Two sites whose clocks differ by a whole number of hours. A phone call starts at a stated time on one site's clock
and lasts exactly m minutes; sometimes a video meeting starts at a stated time on the other site's clock. Claims give
times on either clock, always naming which one."""
from .base import Bundle, Claim, clock, duration, facts

NAME = "timezone"
PAIRS = [("head office", "branch office"), ("main lab", "field station"), ("city office", "island office"),
         ("north site", "south site")]
LO, HI = 5 * 60 + 30, 22 * 60 + 30  # every clock time the family writes stays inside this range on both clocks


def at(f, t_a, z):
    """A time on site A's clock, read on site z's clock."""
    return t_a if z == "A" else t_a + f.d


def on_call(f, x, z):
    t_a = x if z == "A" else x - f.d
    return f.t <= t_a <= f.t + f.m


def meet_at(f, z):
    """The meeting's start on site z's clock (the evidence gives it on site B's clock)."""
    return f.t2 if z == "B" else f.t2 - f.d


def make(rng, people, others, day):
    a, b = rng.sample(rng.choice(PAIRS), 2)
    Z = {"A": a, "B": b}
    sp = list(people)
    while True:
        h = rng.randint(1, 7)
        d = h * 60 * rng.choice([-1, 1])
        t, m = rng.randrange(8 * 60, 17 * 60 + 1, 5), rng.choice([15, 20, 25, 30, 40, 45, 50])
        meet, t2, m2 = rng.random() < 0.6, rng.randrange(8 * 60, 18 * 60 + 1, 15), rng.choice([30, 45, 60, 90])
        used = [t - 30, t + m + 70, t - d] + [x + d for x in (t - 30, t + m + 70)]
        used += [t2, t2 - d, t2 + m2, t2 + m2 - d] if meet else []
        if all(LO <= x <= HI for x in used):
            break
    p = rng.choice(sp) if rng.random() < 0.7 else others[0]
    q = rng.choice([x for x in sp if x != p]) if rng.random() < 0.6 else others[1]
    f = facts(d=d, t=t, m=m, t2=t2 if meet else None, m2=m2)
    ev = [f"The {b}'s clocks are exactly {h} hour{'s' if h > 1 else ''} {'behind' if d < 0 else 'ahead of'} the "
          f"{a}'s clocks.",
          f"At {clock(t)} {a} time, {p} at the {a} phoned {q} at the {b}, and they talked for exactly {m} minutes "
          f"without a break."]
    if meet:
        ev.append(f"The video meeting between the two sites started at {clock(t2)} {b} time and lasted exactly "
                  f"{duration(m2)}.")

    def zt(t_a, z):
        return f"{clock(at(f, t_a, z))} {Z[z]} time"

    def pair(s):
        if s in (p, q):
            return f"{q if s == p else p} and I"
        return f"{p} and {q}"

    def start(s, x, z):  # a sentence saying the call started at clock time x on site z's clock
        if s == p and z == "A":
            return f"I rang {q} at {clock(x)} {Z[z]} time."
        if s == q and z == "B":
            return f"I picked up {p}'s call at {clock(x)} {Z[z]} time."
        return rng.choice([f"The call between {p} and {q} started at {clock(x)} {Z[z]} time.",
                           f"{pair(s)} started talking at {clock(x)} {Z[z]} time."])

    # the lie is picked first, so the liar never also states the truth the lie denies
    kind, liar = rng.choice(["sign", "mix", "still", "early", "dur"] + (["meeting"] if meet else [])), rng.choice(sp)
    key = {"sign": "start", "early": "start", "mix": "end", "still": "end", "dur": "dur", "meeting": "meet"}[kind]

    def snd(k):
        return rng.choice([y for y in sp if y != liar] if k == key else sp)

    truths = []
    for z in "AB":
        s = snd("start")
        truths.append(Claim(s, start(s, at(f, t, z), z), lambda f, x=at(f, t, z), z=z: at(f, f.t, z) == x))
    z, s = rng.choice("AB"), snd("end")
    truths.append(Claim(s, rng.choice([f"The call ended at {zt(t + m, z)}.", f"{pair(s)} hung up at {zt(t + m, z)}."]),
                        lambda f, x=at(f, t + m, z), z=z: at(f, f.t + f.m, z) == x))
    z, s, tm = rng.choice("AB"), rng.choice(sp), t + rng.randrange(5, m - 4, 5)
    truths.append(Claim(s, f"At {zt(tm, z)}, {pair(s)} were still on the phone.",
                        lambda f, x=at(f, tm, z), z=z: on_call(f, x, z)))
    z, s = rng.choice("AB"), snd("start")
    x = at(f, t, z)
    truths.append(Claim(s, f"The call started between {clock(x)} and {clock(x + 30)} {Z[z]} time.",
                        lambda f, x=x, z=z: x <= at(f, f.t, z) <= x + 30, ("boundary",)))
    s = snd("dur")
    truths.append(Claim(s, f"{pair(s)} talked for {m} minutes.", lambda f: f.m == m))
    s = snd("start")
    pn, qn = ("I" if s == p else p), ("I" if s == q else q)
    if d < 0:  # site B's clock is behind: answered early on B's clock although rung late on A's clock
        xb, ya = at(f, t, "B") + 15, t - 15
        text = f"{qn} answered before {clock(xb)} {b} time, even though {pn} rang after {clock(ya)} {a} time."
        ok = lambda f, xb=xb, ya=ya: at(f, f.t, "B") < xb and f.t > ya  # noqa: E731
    else:
        xb, ya = at(f, t, "B") - 15, t + 15
        text = f"{qn} answered after {clock(xb)} {b} time, even though {pn} rang before {clock(ya)} {a} time."
        ok = lambda f, xb=xb, ya=ya: at(f, f.t, "B") > xb and f.t < ya  # noqa: E731
    truths.append(Claim(s, text, ok, ("distractor",)))
    if meet:
        for z in "AB":
            s = snd("meet")
            truths.append(Claim(s, f"The video meeting started at {clock(meet_at(f, z))} {Z[z]} time.",
                                lambda f, x=meet_at(f, z), z=z: meet_at(f, z) == x))
        z, s = rng.choice("AB"), snd("meet")
        truths.append(Claim(s, f"The video meeting ended at {clock(meet_at(f, z) + m2)} {Z[z]} time.",
                            lambda f, x=meet_at(f, z) + m2, z=z: meet_at(f, z) + f.m2 == x))
    for tpl in [f"I printed {{k}} copies of the agenda at the {a}.", f"I sent {{k}} emails to the {b} that day.",
                f"I booked a room for {{k}} people at the {a}."]:
        truths.append(Claim(rng.choice(sp), tpl.format(k=rng.randint(2, 9)), lambda f: True, ("neutral",)))

    s = liar
    about = () if s in (p, q) else ("about-other",)
    if kind == "sign":  # the offset applied the wrong way round
        x = t - d
        lie = Claim(s, start(s, x, "B"), lambda f, x=x: at(f, f.t, "B") == x, ("two-fact",) + about)
    elif kind == "mix":  # an end time on one clock passed off as the other clock's
        z = rng.choice("AB")
        x = at(f, t + m, "B" if z == "A" else "A")
        lie = Claim(s, rng.choice([f"The call ended at {clock(x)} {Z[z]} time.", f"{pair(s)} hung up at {clock(x)} "
                                                                                f"{Z[z]} time."]),
                    lambda f, x=x, z=z: at(f, f.t + f.m, z) == x, ("two-fact",) + about)
    elif kind in ("still", "early"):
        z = rng.choice("AB")
        x = at(f, t + m + rng.randrange(10, 71, 5) if kind == "still" else t - rng.randrange(10, 31, 5), z)
        adv = "still" if kind == "still" else "already"
        lie = Claim(s, f"At {clock(x)} {Z[z]} time, {pair(s)} were {adv} on the phone.",
                    lambda f, x=x, z=z: on_call(f, x, z), ("two-fact",) + about)
    elif kind == "dur":
        m3 = rng.choice([x for x in (m - 10, m - 5, m + 5, m + 10, m + 15) if x > 0])
        lie = Claim(s, f"{pair(s)} talked for {m3} minutes.", lambda f, m3=m3: f.m == m3, about)
    else:  # the meeting's start on site B's clock passed off as site A's
        lie = Claim(s, f"The video meeting started at {clock(t2)} {a} time.", lambda f: meet_at(f, "A") == t2,
                    ("two-fact",))
    group = rng.choice(["Office staff", "Project team members", "Colleagues"])
    assert len(truths) >= 10, len(truths)
    return Bundle(
        NAME,
        rng.choice([f"{group} at the {a} and the {b} argue over what time the call happened.",
                    f"{group} compare call times across two clocks.",
                    f"{group} try to line up the {a} and {b} clocks for one day's call."]),
        group, "a call between two offices", ev, f, lie, truths,
        fillers=["Can we put both clocks on the calendar invite?", "Who set up the video link?",
                 "Should we pick one clock for all meetings?", "Honestly, the time difference trips everyone up."],
        decor=[f"The {a} has a large clock above its front desk.", f"The {b} opened two years ago."])
