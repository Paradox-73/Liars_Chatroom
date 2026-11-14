"""Second new-family validation set (set2), written independently of validation/newfam.

Nine scenario families that train does not have and the first new-family set does not have either: league tables,
currency exchange, race finishing order, round-table seating, fuel use on a trip, meetings on the n-th weekday of a
month, locker-code rules, a bank balance from a complete list of payments, and auction bidding. Half the cases use the
competition's formal wording and 12-hour clock; the other half a casual chat style with a 24-hour clock.

Every claim is checked against the evidence model by its family's check, which enumerates every world the evidence
allows where it leaves freedom (finishing orders, seatings, win/draw/loss splits, bid sequences, codes). build()
asserts that exactly one message per case fails its check and that its sender is the label.

python validation/set2/gen_set2.py [--n 17] [--seed 2026] [--out validation/set2]
"""
import argparse
import csv
import datetime as dt
import itertools
import os
import random
from fractions import Fraction

NAMES = ["Alvaro", "Amaya", "Anouk", "Arjun", "Astrid", "Babak", "Beatrix", "Benedikt", "Callum", "Cecilia", "Dalia",
         "Dmitra", "Edgar", "Elodie", "Emil", "Esther", "Fabian", "Farah", "Gideon", "Greta", "Hamid", "Helga", "Ilan",
         "Irene", "Jasper", "Jovana", "Kasimir", "Klara", "Lorenzo", "Luana", "Magnus", "Malia", "Matteo", "Mirela",
         "Nadir", "Nkechi", "Orla", "Otto", "Paloma", "Petra", "Rasmus", "Renata", "Sahil", "Saskia", "Silas",
         "Svetla", "Teodor", "Thandi", "Ulla", "Valentin", "Wanjiru", "Xavier", "Yasmin", "Yohannes", "Zofia", "Zoltan"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th", 8: "8th", 9: "9th"}
NTH = {1: "first", 2: "second", 3: "third", 4: "fourth", -1: "last"}
WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

FILLERS = {
    "formal": ["Can we settle this quickly?", "Does anyone have the receipt?", "Who is writing this up?",
               "This is getting confusing.", "Let's go through it once more.", "Should we ask the organiser?",
               "Was anyone else there?", "Is that everything we know?", "I'd rather we got this right.",
               "Thanks for checking, everyone."],
    "casual": ["wait what", "can someone just check the notes", "ugh this is confusing", "ok who has the details?",
               "honestly no idea", "anyone else remember?", "can we wrap this up pls", "lol this chat",
               "hang on, let me think", "does it even matter?"],
}
CASUAL_LEAD = ["", "", "tbh ", "fwiw ", "ok so ", "pretty sure ", "honestly ", "for the record "]


class Claim:
    """One checkable message. ok: can it be true together with the evidence (computed by the family's check).
    who: the person it is about, or None; first: the wording when that person says it, third: anyone else's."""

    def __init__(self, ok, third, first=None, who=None, world=None):
        self.ok, self.third, self.first, self.who = bool(ok), third, first, who
        self.world = self.ok if world is None else bool(world)  # true in the one hidden world behind the evidence
        assert self.ok or not self.world


def clock(minutes, st):
    h, m = divmod(int(minutes), 60)
    return f"{h:02d}:{m:02d}" if st == "casual" else f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def gbp(cents):
    return f"£{cents // 100}.{cents % 100:02d}" if cents % 100 else f"£{cents // 100}"


def eur(cents):
    return f"€{cents // 100}.{cents % 100:02d}"


def pick_claims(rng, cands, n_true):
    """cands: Claims; returns (one lie, up to n_true distinct true claims) or None when no lie is available."""
    lies = [c for c in cands if not c.ok]
    trues = [c for c in cands if c.world]  # all true in one world, so the true claims never clash with each other
    if not lies or len(trues) < 3:
        return None
    rng.shuffle(trues)
    seen, out = set(), []
    for c in trues:
        if c.third not in seen:
            seen.add(c.third)
            out.append(c)
    return rng.choice(lies), out[:n_true]


# ---------------------------------------------------------------- families
def fam_league(rng, people, others, st):
    teams = rng.sample(["the Kestrels", "the Otters", "the Comets", "the Badgers", "the Herons", "the Lynxes",
                        "the Pilots", "the Ravens"], 3)
    G, win = rng.randint(4, 8), rng.choice([3, 3, 2])
    rec = {}
    for t in teams:
        w = rng.randint(0, G)
        d = rng.randint(0, G - w)
        rec[t] = (w, d, G - w - d)
    pts = {t: win * w + d for t, (w, d, l) in rec.items()}

    def splits(t):  # every win/draw/loss split the evidence allows
        return [(w, d, G - w - d) for w in range(G + 1) for d in range(G + 1 - w) if win * w + d == pts[t]]

    cands = []
    for t in teams:
        T = t[0].upper() + t[1:]
        sp = splits(t)
        preds = [(f"{T} won exactly {k} of their games.", lambda w, d, l, k=k: w == k) for k in range(G + 1)]
        preds += [(f"{T} drew exactly {k} game{'s' if k != 1 else ''}.", lambda w, d, l, k=k: d == k) for k in range(1, G + 1)]
        preds += [(f"{T} lost exactly {k} game{'s' if k != 1 else ''}.", lambda w, d, l, k=k: l == k) for k in range(1, G + 1)]
        preds += [(f"{T} went the whole season unbeaten.", lambda w, d, l: l == 0),
                  (f"{T} did not draw a single game.", lambda w, d, l: d == 0),
                  (f"{T} won more games than they lost.", lambda w, d, l: w > l),
                  (f"{T} lost more games than they won.", lambda w, d, l: l > w),
                  (f"{T} won at least half of their games.", lambda w, d, l: 2 * w >= G)]
        for text, p in preds:
            cands.append(Claim(any(p(*x) for x in sp), text, world=p(*rec[t])))
    scen = rng.choice(["Players in a five-a-side league argue about how their teams did this season.",
                       "Members of a hockey club compare notes on the league table."])
    ev = [f"In the league every team played exactly {G} games. A win earned {win} points, a draw 1 point and a loss "
          f"no points; no other points were given or taken away.",
          f"At the end of the season {teams[0]} had {pts[teams[0]]} points, {teams[1]} had {pts[teams[1]]} points "
          f"and {teams[2]} had {pts[teams[2]]} points."]
    return scen, ev, cands


def fam_exchange(rng, people, others, st):
    rate = rng.choice([Fraction(23, 20), Fraction(6, 5), Fraction(5, 4), Fraction(11, 10), Fraction(13, 10)])
    fee = rng.choice([2, 3, 4, 5])
    open_, close = rng.choice([(9 * 60, 16 * 60 + 30), (10 * 60, 17 * 60), (8 * 60 + 30, 15 * 60)])
    cands = []
    for p in people + others[:1]:
        x = fee + 20 * rng.randint(2, 15)
        y = (x - fee) * rate * 100
        assert y.denominator == 1
        y = int(y)
        for yy in (y, int(x * rate * 100), int((x + fee) * rate * 100), y + 500):  # right, fee ignored, fee added, off
            cands.append(Claim(yy == y, f"{p} handed over £{x} at the booth and got {eur(yy)} back.",
                               f"I handed over £{x} at the booth and got {eur(yy)} back.", p))
        t_ok = rng.choice([open_, close, rng.randrange(open_ + 15, close - 15, 5)])
        cands.append(Claim(True, f"{p} changed money at the booth at {clock(t_ok, st)}.",
                           f"I changed money at the booth at {clock(t_ok, st)}.", p))
        t_bad = rng.choice([open_ - rng.choice([10, 20, 30]), close + rng.choice([5, 10, 25])])
        cands.append(Claim(open_ <= t_bad <= close, f"{p} changed money at the booth at {clock(t_bad, st)}.",
                           f"I changed money at the booth at {clock(t_bad, st)}.", p))
        cands.append(Claim(True, f"{p} paid the booth's £{fee} fee.", f"I paid the £{fee} fee at the booth.", p))
        f2 = fee + rng.choice([1, 2])
        cands.append(Claim(f2 == fee, f"{p} paid a £{f2} fee at the booth.", f"The booth charged me a £{f2} fee.", p))
    r = f"{float(rate):.2f}".rstrip("0").rstrip(".")
    scen = rng.choice(["Travellers compare what they got at the airport exchange booth.",
                       "Friends on a trip argue about the money they changed before leaving."])
    ev = [f"The exchange booth took a flat fee of £{fee} from every amount handed over and changed the rest at exactly "
          f"{r} euros per pound, with no rounding. It changed no money before {clock(open_, st)} or after "
          f"{clock(close, st)} that day.",
          "Everyone in the chat changed money there exactly once that day."]
    return scen, ev, cands


def fam_race(rng, people, others, st):
    N = rng.randint(6, 9)
    named = people + others[:1]
    while True:
        pos = dict(zip(named, rng.sample(range(1, N + 1), len(named))))
        x, y, z, u, v = rng.sample(named, 5)
        facts = [(f"{x} finished {ORD[pos[x]]}.", lambda P, x=x, k=pos[x]: P[x] == k)]
        if pos[y] == pos[x] + 1:
            facts.append((f"{y} finished directly behind {x}.", lambda P, x=x, y=y: P[y] == P[x] + 1))
        else:
            m = abs(pos[y] - pos[x]) - 1
            text = (f"Nobody finished between {x} and {y}." if m == 0 else
                    f"Exactly {WORDS[m]} runner{'s' if m != 1 else ''} finished between {x} and {y}.")
            facts.append((text, lambda P, x=x, y=y, m=m: abs(P[y] - P[x]) - 1 == m))
        a, b = (z, u) if pos[z] < pos[u] else (u, z)
        facts.append((f"{a} finished ahead of {b}.", lambda P, a=a, b=b: P[a] < P[b]))
        if pos[v] > 3:
            facts.append((f"{v} was not in the top three.", lambda P, v=v: P[v] > 3))
        sols = [dict(zip(named, s)) for s in itertools.permutations(range(1, N + 1), len(named))
                if all(f(dict(zip(named, s))) for _, f in facts)]
        if 2 <= len(sols) <= 400:
            break
    assert pos in sols
    cands = []

    def add(pred, third, first=None, who=None):
        cands.append(Claim(any(pred(P) for P in sols), third, first, who, world=pred(pos)))

    for p in named:
        for k in range(1, N + 1):
            add(lambda P, p=p, k=k: P[p] == k, f"{p} finished {ORD[k]}.", f"I finished {ORD[k]}.", p)
        add(lambda P, p=p: P[p] <= 3, f"{p} made the top three.", "I made the top three.", p)
        add(lambda P, p=p: P[p] == N, f"{p} finished last.", "I finished last.", p)
        for q in named:
            if q != p:
                add(lambda P, p=p, q=q: P[p] < P[q], f"{p} finished ahead of {q}.", f"I finished ahead of {q}.", p)
                add(lambda P, p=p, q=q: abs(P[p] - P[q]) == 1, f"Nobody finished between {p} and {q}.",
                    f"Nobody finished between me and {q}.", p)
    scen = rng.choice(["Club runners argue about the finishing order of the 800 m final.",
                       "Members of a running club piece together the results of the 5 km race."])
    ev = [f"{N} runners finished the race and every finishing place from 1st to {ORD[N]} was taken by exactly one "
          f"runner; there were no ties."] + [t for t, _ in facts]
    return scen, ev, cands


def fam_table(rng, people, others, st):
    six = people + others[:2]

    def dist(P, a, b):
        d = abs(P[a] - P[b])
        return min(d, 6 - d)

    while True:
        seat = dict(zip(six, rng.sample(range(6), 6)))
        facts = []
        pairs = rng.sample([(a, b) for a, b in itertools.combinations(six, 2)], 3)  # three different pairs
        for a, b in pairs:
            if rng.random() < 0.5:
                a, b = b, a
            d = dist(seat, a, b)
            text = {1: f"{a} sat next to {b}.", 2: f"{a} sat two seats away from {b}.",
                    3: f"{a} sat directly opposite {b}."}[d]
            facts.append((text, lambda P, a=a, b=b, d=d: dist(P, a, b) == d))
        sols = [dict(zip(six, s)) for s in itertools.permutations(range(6)) if all(f(dict(zip(six, s))) for _, f in facts)]
        if 12 <= len(sols) <= 240 and len({t for t, _ in facts}) == 3:
            break
    assert seat in sols
    cands = []
    for p in six:
        for q in six:
            if q == p:
                continue
            for d, third, first in ((1, f"{p} sat next to {q}.", f"I sat next to {q}."),
                                    (3, f"{q} sat directly opposite {p}.", f"{q} sat directly opposite me."),
                                    (2, f"{p} sat two seats away from {q}.", f"I sat two seats away from {q}.")):
                cands.append(Claim(any(dist(P, p, q) == d for P in sols), third, first, p, world=dist(seat, p, q) == d))
            cands.append(Claim(any(dist(P, p, q) != 1 for P in sols), f"{p} did not sit next to {q}.",
                               f"I did not sit next to {q}.", p, world=dist(seat, p, q) != 1))
    scen = rng.choice(["Six friends try to remember the seating at a birthday dinner.",
                       "Colleagues argue about who sat where at the round table during the team lunch."])
    ev = [f"Six people sat around a round table with six evenly spaced seats: {', '.join(six[:-1])} and {six[-1]}. "
          f"Each person had exactly two neighbours. Two seats away means exactly one seat between them on the shorter "
          f"side, and directly opposite means three seats away either way."]
    ev += [t for t, _ in facts]
    return scen, ev, cands


def fam_fuel(rng, people, others, st):
    T = rng.choice([40, 45, 50, 55, 60])
    c = rng.choice([Fraction(6), Fraction(15, 2), Fraction(8), Fraction(10)])  # litres per 100 km
    step = {Fraction(6): 50, Fraction(15, 2): 40, Fraction(8): 25, Fraction(10): 10}[c]  # km that use whole litres
    vmax, floor = rng.choice([80, 90, 100]), rng.choice([4, 5, 6])
    t0 = rng.choice([7 * 60, 7 * 60 + 30, 8 * 60])
    t1 = t0 + rng.choice([8, 9, 10]) * 60
    driver = people[0]
    reach = (T - floor) / c * 100  # km the fuel allows

    def ok_dist(d, t):
        return d * c / 100 <= T - floor and d <= Fraction(vmax) * (t - t0) / 60

    vh = min(Fraction(vmax) * rng.choice([5, 6, 7, 8]) / 10, reach * Fraction(9, 10) * 60 / (t1 - t0))

    def hidden(t):  # trip meter of the one hidden day: a steady average speed, rounded down to whole litres
        return int(vh * (t - t0) / 60) // step * step

    cands = []
    for t in sorted(rng.sample(range(t0 + 60, t1, 15), 8)):
        d = hidden(t)
        L = T - d * c / 100
        assert L.denominator == 1 and ok_dist(d, t)
        L = int(L)
        cands.append(Claim(True, f"By {clock(t, st)} the van had done {d} km."))
        cands.append(Claim(True, f"At {clock(t, st)} the trip meter showed {d} km and the gauge showed {L} litres."))
        cands.append(Claim(True, f"{driver} drove {d} km before {clock(t, st)}.", f"I drove {d} km before {clock(t, st)}.",
                           driver))
        fast = (int(Fraction(vmax) * (t - t0) / 60) // step + 1) * step  # one step beyond what the speed limit allows
        cands.append(Claim(ok_dist(fast, t), f"By {clock(t, st)} the van had done {fast} km.", world=False))
        cands.append(Claim(False, f"At {clock(t, st)} the trip meter showed {d} km and the gauge showed {L + 4} litres."))
        cands.append(Claim(False, f"At {clock(t, st)} the gauge showed {floor - 1} litres.", world=False))
    D = int(reach) // 10 * 10 + 60
    cands.append(Claim(ok_dist(D, t1), f"The van covered {D} km in total that day.", world=False))
    cands.append(Claim(False, f"{driver} kept the van at a steady {vmax + 20} km/h for an hour.",
                       f"I kept the van at a steady {vmax + 20} km/h for an hour.", driver))
    cv = str(float(c)).rstrip("0").rstrip(".")
    scen = rng.choice(["Workers argue about how far the delivery van went on a single tank.",
                       "A camping group reconstructs how far their van went before it was refuelled."])
    ev = [f"The van left at {clock(t0, st)} with exactly {T} litres of fuel and was not refuelled before "
          f"{clock(t1, st)}. It used fuel at a steady {cv} litres per 100 km, the fuel gauge was always accurate and "
          f"never read below {floor} litres, and the van never went faster than {vmax} km/h.",
          f"{driver} drove the van all day; nobody else drove it, and the trip meter started at 0 km."]
    return scen, ev, cands


def fam_meeting(rng, people, others, st):
    year = rng.choice([2027, 2028])
    wd = rng.randrange(7)
    nth = rng.choice([1, 2, 3, 4, -1])
    months = sorted(rng.sample(range(1, 13), 2))

    def meet(m):
        days = [dt.date(year, m, d) for d in range(1, 32) if _valid(year, m, d) and dt.date(year, m, d).weekday() == wd]
        return days[nth - 1] if nth > 0 else days[-1]

    cands = []
    for m in months:
        md = meet(m)
        for p in people:
            for delta in (0, 7, -7, 1, -1):
                d = md + dt.timedelta(days=delta)
                if d.month == m:
                    cands.append(Claim(delta == 0, f"{p} went to the {MONTHS[m - 1]} meeting on {d.day} {MONTHS[m - 1]}.",
                                       f"I went to the {MONTHS[m - 1]} meeting on {d.day} {MONTHS[m - 1]}.", p))
            other = md + dt.timedelta(days=rng.choice([2, 3, 4, -2, -3]))
            if other.month == m:
                cands.append(Claim(True, f"{p} was at the gym on {other.day} {MONTHS[m - 1]}.",
                                   f"I was at the gym on {other.day} {MONTHS[m - 1]}.", p))
        wrong_wd = (wd + rng.choice([1, 2, 5])) % 7
        cands.append(Claim(False, f"The {MONTHS[m - 1]} meeting was held on a {WEEKDAYS[wrong_wd]}."))
        cands.append(Claim(True, f"The {MONTHS[m - 1]} meeting was held on a {WEEKDAYS[wd]}."))
    scen = rng.choice(["Residents argue about which committee meetings they went to.",
                       "Members of a chess club try to sort out the dates of their monthly meetings."])
    ev = [f"In {year} the committee met once a month, always on the {NTH[nth]} {WEEKDAYS[wd]} of the month, and never on "
          f"any other day."]
    ev += [f"1 {MONTHS[m - 1]} {year} was a {WEEKDAYS[dt.date(year, m, 1).weekday()]}." for m in months]
    return scen, ev, cands


def _valid(y, m, d):
    try:
        dt.date(y, m, d)
        return True
    except ValueError:
        return False


def fam_locker(rng, people, others, st):
    while True:
        distinct = rng.random() < 0.7
        first = rng.choice(["odd", "even", "nonzero"])
        total = rng.randint(14, 24)
        no_zero = rng.random() < 0.4

        def rule_ok(code):
            ds = [int(x) for x in code]
            if first == "odd":
                f_ok = ds[0] % 2 == 1
            elif first == "even":
                f_ok = ds[0] % 2 == 0 and ds[0] != 0
            else:
                f_ok = ds[0] != 0
            return (len(ds) == 4 and (not distinct or len(set(ds)) == 4) and sum(ds) == total and f_ok
                    and (not no_zero or 0 not in ds))

        valid = ["".join(x) for x in itertools.product("0123456789", repeat=4) if rule_ok("".join(x))]
        if len(valid) >= 20:
            break
    cands = []
    for p in people:
        good = rng.choice(valid)
        cands.append(Claim(True, f"{p}'s locker code was {good}.", f"My locker code was {good}.", p))
        cands.append(Claim(True, f"{p}'s code ended in {good[-1]}.", f"My code ended in {good[-1]}.", p))
        cands.append(Claim(True, f"{p}'s code started with {good[0]}.", f"My code started with {good[0]}.", p))
        bad = [int(x) for x in good]
        i = rng.randrange(1, 4)
        bad[i] = (bad[i] + rng.choice([1, 2, 9])) % 10
        bad = "".join(map(str, bad))
        cands.append(Claim(rule_ok(bad), f"{p}'s locker code was {bad}.", f"My locker code was {bad}.", p,
                           world=bad == good))
        s = rng.randrange(10)
        cands.append(Claim(any(v[0] == str(s) for v in valid), f"{p}'s code started with {s}.",
                           f"My code started with {s}.", p, world=good[0] == str(s)))
        r = rng.randrange(10)
        cands.append(Claim(any(v.count(str(r)) >= 2 for v in valid), f"{p}'s code had the digit {r} in it twice.",
                           f"My code had the digit {r} in it twice.", p, world=good.count(str(r)) >= 2))
    scen = rng.choice(["Gym members argue about the codes they picked for their lockers.",
                       "Students compare the four-digit codes they set on their lockers."])
    rules = ["every digit is different" if distinct else "digits may repeat",
             {"odd": "the first digit is odd", "even": "the first digit is even and not 0",
              "nonzero": "the first digit is not 0"}[first],
             f"the four digits add up to exactly {total}"] + (["no digit is 0"] if no_zero else [])
    ev = ["The lockers only accept four-digit codes that follow every one of these rules: " + "; ".join(rules) + ".",
          "Every person in the chat set exactly one code, and the lockers accepted it."]
    return scen, ev, cands


def fam_bank(rng, people, others, st):
    holder = people[0]
    t0 = 8 * 60
    while True:
        b0 = rng.randrange(150, 900) * 100 + rng.choice([0, 0, 50, 25])
        times = sorted(rng.sample(range(9 * 60, 19 * 60, 10), 4))
        tx = []
        for t in times:
            amt = rng.randrange(10, 260) * 100 + rng.choice([0, 0, 50, 99])
            sign = rng.choice([-1, -1, 1])
            kind = (rng.choice(["card payment", "direct debit", "cash withdrawal"]) if sign < 0
                    else rng.choice(["transfer in", "refund", "salary payment"]))
            tx.append((t, sign * amt, kind))

        def bal(t):
            return b0 + sum(a for tt, a, _ in tx if tt <= t)

        low = min(bal(t) for t in [t0] + times)
        if low >= 1000:
            break
    cands = []
    for t in rng.sample(range(t0 + 10, 21 * 60, 5), 12):
        if t in times:
            continue
        b = bal(t)
        cands.append(Claim(True, f"At {clock(t, st)} {holder}'s balance was {gbp(b)}.",
                           f"At {clock(t, st)} my balance was {gbp(b)}.", holder))
        done = [a for tt, a, _ in tx if tt <= t]
        wrong = b - rng.choice(done) if done else b + tx[0][1]  # one payment left out, or the first one too early
        cands.append(Claim(wrong == b, f"At {clock(t, st)} {holder}'s balance was {gbp(wrong)}.",
                           f"At {clock(t, st)} my balance was {gbp(wrong)}.", holder))
    for th in (low // 1000 * 1000, (low // 1000 + 2) * 1000):
        cands.append(Claim(low >= th, f"{holder}'s balance never went below {gbp(th)} that day.",
                           f"My balance never went below {gbp(th)} that day.", holder))
    lines = ", ".join(f"a {k} of {gbp(abs(a))} at {clock(t, st)}" for t, a, k in tx)
    scen = rng.choice([f"Flatmates help {holder} work out what happened to the money in a shared account.",
                       f"Family members go through {holder}'s bank app after a confusing day of spending."])
    ev = [f"At {clock(t0, st)} {holder}'s account held exactly {gbp(b0)}.",
          f"The only payments into or out of the account that day were, in order: {lines}. Each one changed the balance "
          f"at the moment it was made."]
    return scen, ev, cands


def auction_ok(f, z, inc):
    """Checks for claims about a bid sequence that opens at f, ends with the winning bid z and rises by at least inc
    each time (whole pounds). Worked by hand; self_test compares them with enumeration on small cases."""
    return {
        "count": lambda k: (k == 1 and z == f) or (k >= 2 and z - f >= (k - 1) * inc),
        "middle": lambda x: f + inc <= x <= z - inc,  # some bid other than the first and the last
        "second": lambda x: f + inc <= x <= z - inc,  # the second of at least three bids
        "margin": lambda d: d >= inc and (z - d == f or z - d - f >= inc),
    }


def fam_auction(rng, people, others, st):
    inc = rng.choice([5, 10, 20])
    f = rng.choice([20, 40, 50, 60, 100])
    s = [f]
    for _ in range(rng.randint(3, 7)):
        s.append(s[-1] + inc + rng.choice([0, 0, 0, inc // 2, inc]))
    z = s[-1]
    chk = auction_ok(f, z, inc)
    item = rng.choice(["the painting", "the antique clock", "the signed guitar", "the vintage bike"])
    cands = []
    for k in range(2, (z - f) // inc + 3):
        cands.append(Claim(chk["count"](k), f"There were exactly {k} bids on {item} in total.", world=k == len(s)))
    middles = s[1:-1]
    bidders = rng.sample(people, min(len(people), len(middles)))
    for p, x in zip(bidders, rng.sample(middles, len(bidders))):
        cands.append(Claim(True, f"{p} bid £{x} on {item} at one point.", f"I bid £{x} on {item} at one point.", p))
    for p in people:
        for x in (f + inc // 2, z - inc // 2, z + inc):
            if x not in (f, z):
                cands.append(Claim(chk["middle"](x), f"{p} bid £{x} on {item} at one point.",
                                   f"I bid £{x} on {item} at one point.", p, world=False))
    for x in (f + inc // 2, s[1]):
        cands.append(Claim(chk["second"](x), f"The second bid on {item} was £{x}.", world=x == s[1]))
    for d in (inc // 2, z - s[-2]):
        if d:
            cands.append(Claim(chk["margin"](d), f"The winning bid on {item} beat the bid before it by exactly £{d}.",
                               world=d == z - s[-2]))
    scen = rng.choice([f"Bidders at a charity auction argue about how the bidding for {item} went.",
                       f"Friends who went to a village auction compare notes on the sale of {item}."])
    ev = [f"The bidding for {item} opened with a first bid of £{f}. Every later bid was a whole number of pounds and at "
          f"least £{inc} higher than the bid before it. The last bid, £{z}, won, and no bid was withdrawn."]
    return scen, ev, cands


FAMILIES = {"league": fam_league, "exchange": fam_exchange, "race": fam_race, "table": fam_table, "fuel": fam_fuel,
            "meeting": fam_meeting, "locker": fam_locker, "bank": fam_bank, "auction": fam_auction}


# ---------------------------------------------------------------- chat assembly
def casual(text, rng):
    """A claim in chat style: a lead-in, and a lower-case first word unless it is a name or 'I'."""
    first = text.split()[0]
    if first in ("The", "There", "Nobody", "At", "By", "My", "Exactly"):
        text = text[0].lower() + text[1:]
    return rng.choice(CASUAL_LEAD) + text


def build_case(rng, fam, cid, liar_letter, st):
    names = rng.sample(NAMES, 7)
    people, others = names[:4], names[4:]
    for _ in range(50):
        scen, ev, cands = FAMILIES[fam](rng, people, others, st)
        picked = pick_claims(rng, cands, rng.randint(5, 8))
        if picked:
            break
    else:
        raise RuntimeError(f"{fam}: no lie available")
    lie, trues = picked
    # the liar: the lie's subject when it is a speaker and the coin says first person, else another speaker
    if lie.who in people and lie.first and rng.random() < 0.55:
        liar, lie_text = lie.who, lie.first
    else:
        liar = rng.choice([p for p in people if p != lie.who])
        lie_text = lie.third
    letters = list("ABCD")
    order = [liar] + [p for p in people if p != liar]
    rng.shuffle(order)
    order.remove(liar)
    order.insert(letters.index(liar_letter), liar)
    lab = dict(zip(order, letters))
    msgs = [(liar, lie_text, False)]
    for c in trues:
        if c.who in people and c.first and rng.random() < 0.6:
            msgs.append((c.who, c.first, True))
        else:
            msgs.append((rng.choice([p for p in people if p != c.who]), c.third, True))
    for f in rng.sample(FILLERS[st], rng.randint(2, 4)):
        msgs.append((rng.choice(people), f, None))
    for p in people:  # everyone speaks at least once
        if not any(m[0] == p for m in msgs):
            msgs.append((p, rng.choice(FILLERS[st]), None))
    rng.shuffle(msgs)
    lines, lie_num = [], None
    for i, (p, text, ok) in enumerate(msgs, 1):
        if st == "casual" and ok is not None:  # fillers are already in chat style
            text = casual(text, rng)
        lines.append(f"{i}. {lab[p]}: {text}")
        if ok is False:
            lie_num = i
    # checks: one lie, said by the liar; every other checkable message passed its family check
    assert sum(ok is False for _, _, ok in msgs) == 1 and lab[liar] == liar_letter
    assert all(ok is not False for p, _, ok in msgs if p != liar)
    evidence = " ".join(ev)
    if st == "formal" and any(x in evidence for x in (" AM", " PM")):
        evidence = f"All times in this case refer to {rng.randint(1, 28)} {rng.choice(MONTHS)} 2026. " + evidence
    elif st == "casual":
        evidence = "Facts checked by the organisers: " + evidence
    row = {"id": cid, "scenario": scen, "speakers": "; ".join(f"{l} = {p}" for p, l in sorted(lab.items(), key=lambda x: x[1])),
           "evidence": evidence, "chat": "\n".join(lines), "label": liar_letter}
    meta = {"id": cid, "family": fam, "style": st, "lie_message": lie_num, "lie_about": "self" if lie_text == lie.first else "other",
            "messages": len(lines)}
    return row, meta


def self_test():
    """The families' checks against hand-worked cases."""
    assert clock(14 * 60 + 5, "formal") == "2:05 PM" and clock(14 * 60 + 5, "casual") == "14:05"
    assert dt.date(2027, 1, 1).weekday() == 4  # a Friday
    for f, z, inc in ((20, 45, 5), (20, 40, 10), (50, 50, 5), (30, 75, 10), (40, 95, 10)):
        seqs = []

        def go(q):
            if q[-1] == z:
                seqs.append(q)
            for nxt in range(q[-1] + inc, z + 1):
                go(q + [nxt])

        go([f])
        chk = auction_ok(f, z, inc)
        for k in range(1, 12):
            assert chk["count"](k) == any(len(q) == k for q in seqs), (f, z, inc, k)
        for x in range(f - 5, z + 6):
            assert chk["middle"](x) == any(x in q[1:-1] for q in seqs), (f, z, inc, x)
            assert chk["second"](x) == any(len(q) > 2 and q[1] == x for q in seqs), (f, z, inc, x)
        for d in range(0, z - f + 2):
            assert chk["margin"](d) == any(len(q) > 1 and q[-1] - q[-2] == d for q in seqs), (f, z, inc, d)
    rng = random.Random(0)
    for fam in FAMILIES:
        for _ in range(20):
            scen, ev, cands = FAMILIES[fam](rng, ["P1", "P2", "P3", "P4"], ["O1", "O2", "O3"], "formal")
            assert any(c.ok for c in cands), fam


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=17, help="cases per family")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    self_test()
    rng = random.Random(a.seed)
    rows, metas = [], []
    k = 0
    for fam in FAMILIES:
        for j in range(a.n):
            k += 1
            row, meta = build_case(rng, fam, f"S2{k:04d}", "ABCD"[k % 4], "formal" if j % 2 == 0 else "casual")
            rows.append(row)
            metas.append(meta)
    order = list(range(len(rows)))
    rng.shuffle(order)
    rows, metas = [rows[i] for i in order], [metas[i] for i in order]
    with open(os.path.join(a.out, "set2.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["id", "scenario", "speakers", "evidence", "chat", "label"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "set2_meta.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["id", "family", "style", "lie_message", "lie_about", "messages"])
        w.writeheader()
        w.writerows(metas)
    print(f"{len(rows)} cases, {len(FAMILIES)} families, labels {sorted(set(r['label'] for r in rows))}")


if __name__ == "__main__":
    main()
