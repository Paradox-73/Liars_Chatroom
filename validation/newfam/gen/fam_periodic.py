"""Repeating schedules. Day level: recycling collected every 14 days on one weekday, or greenhouse plants watered every
n days from a start date within one month. Minute level: a ferry or a cable car leaving every n minutes from a first
to a last departure, and at no other time."""
import calendar
import datetime as dt

from .base import MONTHS, WEEKDAYS, Bundle, Claim, anchor, clock, date_text, facts, weekday

NAME = "periodic"
DAY = dt.timedelta(days=1)
STREETS = ["Elm Road", "Mill Lane", "Station Street", "Orchard Close", "Harbour Row", "Chapel Walk"]
RIDES = [
    dict(veh="ferry", vehs="ferries", stop="harbour", dests=["Gull Island", "the north shore", "Kell Island"],
         first="That day the ferry to {dest} left the harbour every {n} minutes, with the first departure at {t0} and "
               "the last at {t1}.", none="No ferry left the harbour at any other time that day.",
         me=["I caught the {t} ferry to {dest}.", "I took the {t} ferry."],
         other=["{x} took the {t} ferry to {dest}.", "I saw {x} board the {t} ferry."],
         watch="At {t} I watched a ferry pull out of the harbour.",
         group="Day trippers", topic="the ferry times",
         neutral=["I bought {k} coffees at the harbour kiosk.", "I paid for {k} hours of parking by the harbour.",
                  "I took {k} photos of the lighthouse."],
         fill=["Is the ferry crossing ever rough?", "Should we go earlier next time?", "Who has the timetable?"],
         decor=["The ferry has a red funnel.", "The harbour has a small fish market."]),
    dict(veh="cable car", vehs="cable cars", stop="base station", dests=["the summit", "the ridge café", "the top station"],
         first="That day the cable car to {dest} left the base station every {n} minutes, with the first car at {t0} "
               "and the last at {t1}.", none="No cable car left the base station at any other time that day.",
         me=["I rode the {t} cable car up to {dest}.", "I took the {t} cable car."],
         other=["{x} rode the {t} cable car up to {dest}.", "I saw {x} get into the {t} cable car."],
         watch="At {t} I watched a cable car leave the base station.",
         group="Hikers", topic="the cable car times",
         neutral=["I bought {k} postcards at the base station.", "I walked {k} kilometres along the ridge.",
                  "I paid for {k} hours of parking at the base station."],
         fill=["Is the view worth the queue?", "Does the cable car run in strong wind?", "Who has the timetable?"],
         decor=["Each cable car holds eight people.", "The base station has a gift shop."]),
]


def is_day(f, d):
    """A watering or collection day under the evidence; plants: only dates in the stated month are known."""
    if f.month and (d.year, d.month) != f.month:
        return False
    return d >= (f.d0 if f.month else d) and (d - f.d0).days % f.n == 0


def in_scope(f, d):
    return not f.month or (d.year, d.month) == f.month


def dep(f, t):
    return f.t0 <= t <= f.t1 and (t - f.t0) % f.n == 0


def bins(rng, people, others, day):
    D, sp = date_text, list(people)
    street = rng.choice(STREETS)
    wd = rng.randrange(7)
    d0 = dt.date(2026, 2, 1) + rng.randrange(0, 260) * DAY
    d0 += ((wd - d0.weekday()) % 7) * DAY
    f = facts(d0=d0, n=14, month=None)
    ev = [f"The recycling on {street} is collected every second {WEEKDAYS[wd]}, that is once every 14 days, and on no "
          f"other day.", f"One collection was on {D(d0)}.", anchor(d0)]
    cs = [d0 + 14 * j * DAY for j in range(-2, 6)]
    rng.shuffle(cs)
    # the lie is picked first, so the liar never also states the truth the lie denies
    kind, liar = rng.choice(["offweek", "wrongday", "weekday", "count", "saw"]), rng.choice(sp)
    rest = [y for y in sp if y != liar]
    truths = []
    for c in cs[:3]:
        s = rng.choice(sp)
        truths.append(Claim(s, rng.choice([f"The recycling on {street} was collected on {D(c)}.",
                                           f"The bin lorry took the recycling from {street} on {D(c)}.",
                                           f"I watched the bin lorry collect the recycling on {street} on {D(c)}."]),
                            lambda f, c=c: is_day(f, c)))
    c = cs[3]
    truths.append(Claim(rng.choice(sp), f"There was no recycling collection on {street} on {D(c + 7 * DAY)}.",
                        lambda f, d=c + 7 * DAY: not is_day(f, d)))
    c = cs[4]
    truths.append(Claim(rng.choice(sp), f"I put my recycling out on {D(c + 7 * DAY)}, but nobody collected it that day.",
                        lambda f, d=c + 7 * DAY: not is_day(f, d), ("distractor",)))
    c = cs[5]
    truths.append(Claim(rng.choice(rest if kind == "weekday" else sp), f"The collection on {D(c)} fell on a {WEEKDAYS[wd]}.",
                        lambda f, c=c, w=WEEKDAYS[wd]: is_day(f, c) and weekday(c) == w))
    m = (d0.year, d0.month)
    k = sum(1 for i in range(1, calendar.monthrange(*m)[1] + 1) if is_day(f, dt.date(*m, i)))
    truths.append(Claim(rng.choice(rest if kind == "count" else sp), f"{street} had {k} recycling collections in {MONTHS[m[1] - 1]} {m[0]}.",
                        lambda f, k=k: sum(is_day(f, dt.date(*m, i)) for i in range(1, calendar.monthrange(*m)[1] + 1)) == k))
    for tpl in ["I bought a roll of {k} bin bags that month.", "I rinsed out {k} glass jars for the recycling.",
                "I flattened {k} cardboard boxes that week."]:
        truths.append(Claim(rng.choice(sp), tpl.format(k=rng.randint(3, 12)), lambda f: True, ("neutral",)))

    s, c = liar, cs[6]
    if kind == "offweek":
        d = c + 7 * DAY
        lie = Claim(s, f"The recycling on {street} was collected on {D(d)}.", lambda f, d=d: is_day(f, d), ("two-fact",))
    elif kind == "wrongday":
        d = c + rng.choice([1, 2, 3, -1, -2]) * DAY
        lie = Claim(s, f"The bin lorry took the recycling from {street} on {D(d)}.", lambda f, d=d: is_day(f, d),
                    ("two-fact",))
    elif kind == "weekday":
        w = rng.choice([x for x in WEEKDAYS if x != WEEKDAYS[wd]])
        lie = Claim(s, f"The collection on {D(c)} fell on a {w}.", lambda f, c=c, w=w: is_day(f, c) and weekday(c) == w,
                    ("two-fact",))
    elif kind == "count":
        k2 = k + rng.choice([1, -1])
        lie = Claim(s, f"{street} had {k2} recycling collection{'s' if k2 != 1 else ''} in {MONTHS[m[1] - 1]} {m[0]}.",
                    lambda f, k=k2: sum(is_day(f, dt.date(*m, i)) for i in range(1, calendar.monthrange(*m)[1] + 1)) == k,
                    ("two-fact",))
    else:
        x, d = rng.choice([y for y in sp if y != s] + others[:1]), c + 7 * DAY
        lie = Claim(s, f"{x} watched the bin lorry collect the recycling on {street} on {D(d)}.",
                    lambda f, d=d: is_day(f, d), ("two-fact", "about-other"))
    g = rng.choice(["Neighbours", "Residents", "Flatmates"])
    return Bundle(NAME, rng.choice([f"{g} on {street} argue over which days the recycling was collected.",
                                    f"{g} compare dates of recycling collections on {street}.",
                                    f"{g} try to pin down the bin days on {street}."]),
                  g, f"the recycling days on {street}", ev, f, lie, truths,
                  fillers=["Is the council changing the bin days?", "Who left the lid open last time?",
                           "Can glass go in the same bin?"],
                  decor=[f"{street} has twelve houses.", "The recycling bins have green lids."], timed=False)


def plants(rng, people, others, day):
    D, sp = date_text, list(people)
    m, n = rng.randrange(1, 13), rng.choice([2, 3, 4, 5])
    d0 = dt.date(2026, m, rng.randint(2, 5))
    month = f"{MONTHS[m - 1]} 2026"
    cyc = [d0 + n * j * DAY for j in range(20) if (d0 + n * j * DAY).month == m]
    p = rng.choice(sp) if rng.random() < 0.6 else others[0]
    f = facts(d0=d0, n=n, month=(2026, m), who=p)
    ev = [f"In {month} the greenhouse plants were watered every {n} days, starting on {D(d0)}, and on no other days "
          f"that month.", f"{p} did all of that watering."]
    off = [d for d in (d0 + i * DAY for i in range(31)) if d.month == m and not is_day(f, d)]

    def who(s):
        return "I" if s == p else p

    # the lie is picked first, so the liar never also states the truth the lie denies
    kind = rng.choice(["offcycle", "before", "who", "row", "count"])
    liar = rng.choice([y for y in sp if y != p]) if kind == "who" else rng.choice(sp)
    rest = [y for y in sp if y != liar]
    d_l = rng.choice([x for x in off if x > d0])  # the off-cycle date an 'offcycle' lie names
    truths = []
    for c in rng.sample(cyc, 3):
        s = rng.choice(sp)
        truths.append(Claim(s, rng.choice([f"{who(s)} watered the greenhouse plants on {D(c)}.",
                                           f"The greenhouse plants were watered on {D(c)}."]),
                            lambda f, c=c: is_day(f, c)))
    s = rng.choice(rest if kind == "before" else sp)
    truths.append(Claim(s, f"{who(s)} watered the greenhouse plants on {D(d0)}, the first watering of {month}.",
                        lambda f: is_day(f, d0) and not any(is_day(f, d0 - i * DAY) for i in range(1, 6))))
    d = rng.choice([x for x in off if x != d_l])
    truths.append(Claim(rng.choice(sp), f"Nobody watered the greenhouse plants on {D(d)}.", lambda f, d=d: not is_day(f, d)))
    truths.append(Claim(rng.choice(rest if kind == "count" else sp), f"The greenhouse plants were watered {len(cyc)} times in {month}.",
                        lambda f, k=len(cyc): sum(is_day(f, d0 + i * DAY) for i in range(31)) == k))
    nxt = dt.date(2026 + (m == 12), m % 12 + 1, rng.randint(1, 20))
    s = rng.choice([y for y in sp if y != p])
    truths.append(Claim(s, f"I watered the greenhouse plants on {D(nxt)}.", lambda f, d=nxt: not in_scope(f, d),
                        ("distractor",)))
    s = rng.choice(rest if kind == "row" else sp)
    truths.append(Claim(s, f"{who(s)} went exactly {n} days between waterings in {month}.",
                        lambda f: f.n == n, ("distractor",)))
    for tpl in ["I bought {k} bags of compost that month.", "I repotted {k} pepper plants in the greenhouse.",
                "I replaced {k} cracked panes in the greenhouse roof."]:
        truths.append(Claim(rng.choice(sp), tpl.format(k=rng.randint(2, 9)), lambda f: True, ("neutral",)))

    s = liar
    if kind == "offcycle":
        d = d_l
        if s != p and rng.random() < 0.5:
            lie = Claim(s, f"I saw {p} water the greenhouse plants on {D(d)}.", lambda f, d=d: is_day(f, d),
                        ("two-fact", "about-other"))
        else:
            lie = Claim(s, f"The greenhouse plants were watered on {D(d)}.", lambda f, d=d: is_day(f, d), ("two-fact",))
    elif kind == "before":
        d = d0 - rng.randint(1, d0.day - 1) * DAY
        lie = Claim(s, f"{who(s)} watered the greenhouse plants on {D(d)}.", lambda f, d=d: is_day(f, d),
                    () if s == p else ("about-other",))
    elif kind == "who":  # a cycle day, but somebody other than p did the watering
        c = rng.choice(cyc)
        lie = Claim(s, f"I watered the greenhouse plants on {D(c)}.", lambda f, c=c, s=s: is_day(f, c) and f.who == s,
                    ("two-fact",))
    elif kind == "row":
        lie = Claim(s, f"{who(s)} watered the greenhouse plants on two days in a row in {month}.",
                    lambda f: f.n == 1, ("two-fact",) + (() if s == p else ("about-other",)))
    else:
        k2 = len(cyc) + rng.choice([1, -1])
        lie = Claim(s, f"The greenhouse plants were watered {k2} times in {month}.",
                    lambda f, k=k2: sum(is_day(f, d0 + i * DAY) for i in range(31)) == k, ("two-fact",))
    g = rng.choice(["Allotment volunteers", "Housemates", "Garden club members"])
    return Bundle(NAME, rng.choice([f"{g} compare dates when the greenhouse plants were watered.",
                                    f"{g} argue over who watered the greenhouse plants in {month}.",
                                    f"{g} piece together the greenhouse watering days."]),
                  g, "the greenhouse watering days", ev, f, lie, truths,
                  fillers=["Do the tomatoes need more water in summer?", "Who has the greenhouse key?",
                           "Should we buy a timer for the hose?"],
                  decor=["The greenhouse has a sliding door.", "The watering can is kept by the shed."], timed=False)


def rides(rng, people, others, day):
    sp, r = list(people), rng.choice(RIDES)
    dest, n = rng.choice(r["dests"]), rng.choice([15, 20, 25, 30, 40, 45])
    t0 = rng.randrange(6 * 60, 8 * 60 + 31, 5)
    k = rng.randint(6, (20 * 60 + 30 - t0) // n)
    t1 = t0 + k * n
    f = facts(t0=t0, n=n, t1=t1)
    ev = [r["first"].format(dest=dest, n=n, t0=clock(t0), t1=clock(t1)), r["none"]]
    grid = [t0 + i * n for i in range(1, k)]  # departures strictly inside the span
    picks = rng.sample(grid, 4)
    # the lie is picked first, so the liar never also states the truth the lie denies
    kind, liar = rng.choice(["offgrid", "offgrid", "after", "before", "interval", "count"]), rng.choice(sp)
    rest = [y for y in sp if y != liar]
    u = rng.choice(grid)  # an 'offgrid' lie names a time between departure u and the next one
    truths = []
    for t in picks[:2]:
        s = rng.choice(sp)
        truths.append(Claim(s, rng.choice(r["me"]).format(t=clock(t), dest=dest), lambda f, t=t: dep(f, t)))
    x, t = rng.choice(sp + others[:1]), picks[2]
    s = rng.choice([y for y in sp if y != x])
    truths.append(Claim(s, rng.choice(r["other"]).format(x=x, t=clock(t), dest=dest), lambda f, t=t: dep(f, t)))
    truths.append(Claim(rng.choice(sp), r["watch"].format(t=clock(picks[3])), lambda f, t=picks[3]: dep(f, t)))
    truths.append(Claim(rng.choice(rest if kind == "before" else sp), f"The first {r['veh']} of the day left at {clock(t0)}.",
                        lambda f: dep(f, t0) and not dep(f, t0 - f.n)))
    s = rng.choice(rest if kind == "after" else sp)
    truths.append(Claim(s, rng.choice([f"The last {r['veh']} of the day left at {clock(t1)}.",
                                       f"I caught the last {r['veh']} of the day, at {clock(t1)}."]),
                        lambda f: dep(f, t1) and not dep(f, t1 + f.n), ("boundary",)))
    t = rng.choice([x for x in grid[:-1] if x != u])
    truths.append(Claim(rng.choice(sp), f"No {r['veh']} left between {clock(t + 5)} and {clock(t + n - 5)}.",
                        lambda f, t=t: not any(dep(f, u) for u in range(t + 5, t + n - 4)), ("distractor",)))
    t, w = rng.choice(grid), n + rng.choice([5, 10, 15])
    truths.append(Claim(rng.choice(sp), f"I got to the {r['stop']} at {clock(t - w)} and caught the {clock(t)} {r['veh']}.",
                        lambda f, t=t: dep(f, t), ("distractor",)))
    truths.append(Claim(rng.choice(rest if kind == "count" else sp), f"There were {k + 1} {r['veh']} departures that day.",
                        lambda f, c=k + 1: sum(dep(f, u) for u in range(f.t0, f.t1 + 1)) == c))
    for tpl in r["neutral"]:
        truths.append(Claim(rng.choice(sp), tpl.format(k=rng.randint(2, 9)), lambda f: True, ("neutral",)))

    s = liar
    if kind in ("offgrid", "after", "before"):
        t = (u + rng.randrange(5, n - 4, 5) if kind == "offgrid" else
             t1 + rng.choice([n, 10, 20]) if kind == "after" else t0 - n)
        if rng.random() < 0.5:
            x = rng.choice([y for y in sp if y != s] + others[:1])
            lie = Claim(s, rng.choice(r["other"]).format(x=x, t=clock(t), dest=dest), lambda f, t=t: dep(f, t),
                        ("two-fact", "about-other"))
        else:
            lie = Claim(s, rng.choice(r["me"]).format(t=clock(t), dest=dest), lambda f, t=t: dep(f, t), ("two-fact",))
    elif kind == "interval":
        n2 = rng.choice([x for x in (n - 10, n - 5, n + 5, n + 10, 2 * n) if x > 0 and x != n])
        lie = Claim(s, f"The {r['vehs']} left every {n2} minutes that day.", lambda f, n2=n2: f.n == n2)
    else:
        c = k + 1 + rng.choice([1, -1, 2])
        lie = Claim(s, f"There were {c} {r['veh']} departures that day.",
                    lambda f, c=c: sum(dep(f, u) for u in range(f.t0, f.t1 + 1)) == c, ("two-fact",))
    return Bundle(NAME, rng.choice([f"{r['group']} compare which {r['veh']} each of them caught.",
                                    f"{r['group']} argue over the {r['veh']} times to {dest}.",
                                    f"{r['group']} piece together a day of {r['veh']} trips to {dest}."]),
                  r["group"], r["topic"], ev, f, lie, truths, fillers=r["fill"], decor=r["decor"], timed=True)


def make(rng, people, others, day):
    return rng.choice([bins, plants, rides, rides])(rng, people, others, day)
