"""Parking tickets. A ticket costs a fixed price per hour and is valid from the moment it is bought for exactly the time
paid for. Variant 'length': the evidence gives each ticket's length, so its price is derived. Variant 'paid': it gives
the amount paid, so the length and the expiry are derived. No vehicle was fined while it had a valid ticket, and
sometimes the warden checked only at two named times and gave fines only during those checks."""
from .base import Bundle, Claim, clock, duration, facts, money

NAME = "parking"
LOTS = ["the station car park", "the Market Street car park", "the hospital car park", "the beach car park",
        "the library car park", "the leisure centre car park"]
VEHICLES = ["red hatchback", "blue van", "white estate car", "green pickup", "silver saloon", "black minibus"]
RATES = [100, 120, 150, 180, 200, 240, 300]  # cents per hour
LENGTHS = [30, 45, 60, 90, 120, 150, 180, 240]  # minutes

# (first-person templates when the sender owns the ticket, third-person templates otherwise)
T = {
    "bought": (["I bought my parking ticket at {t}.", "At {t} I bought the ticket for my {v}."],
               ["{o} bought a parking ticket at {t}.", "At {t} {o} bought a ticket for the {v}."]),
    "valid": (["My ticket was still valid at {t}.", "At {t} my parking ticket was still valid."],
              ["{o}'s ticket was still valid at {t}.", "At {t} the ticket for {o}'s {v} was still valid."]),
    "expired": (["My ticket had run out by {t}.", "By {t} my parking ticket had run out."],
                ["{o}'s ticket had run out by {t}.", "By {t} the ticket for {o}'s {v} had run out."]),
    "ends": (["My ticket ran out at {t}.", "My parking ticket expired at {t}."],
             ["{o}'s ticket ran out at {t}.", "The ticket for {o}'s {v} expired at {t}."]),
    "ends_between": (["My ticket ran out between {x} and {y}."], ["{o}'s ticket ran out between {x} and {y}."]),
    "paid": (["I paid {amt} for my ticket.", "My parking ticket cost me {amt}."],
             ["{o} paid {amt} for the ticket.", "{o}'s parking ticket cost {amt}."]),
    "length": (["I paid for {dur} of parking.", "My ticket covered {dur}."],
               ["{o}'s ticket covered {dur}.", "{o} paid for {dur} of parking."]),
    "fined": (["My {v} got a parking fine at {t}.", "At {t} the warden fined my {v}."],
              ["The warden fined {o}'s {v} at {t}.", "I saw the warden fine {o}'s {v} at {t}."]),
}


def valid(f, o, t):
    b, m = f.tickets[o]
    return b <= t < b + m


def can_fine(f, o, t):
    return not valid(f, o, t) and (f.rounds is None or t in f.rounds)


def make(rng, people, others, day):
    lot, cur = rng.choice(LOTS), rng.choice(["£", "£", "$"])
    paid = rng.random() < 0.5  # True: the evidence gives the amount paid, False: the ticket's length
    rate = rng.choice(RATES)
    lengths = [m for m in LENGTHS if rate * m % 60 == 0]
    sp = list(people)
    owners = rng.sample(sp, 1 if rng.random() < 0.45 else 2)
    if rng.random() < 0.25:
        owners[-1] = others[0]
    vehicles = dict(zip(owners, rng.sample(VEHICLES, len(owners))))
    tickets, stay = {}, {}
    for o in owners:
        b, m = rng.randrange(7 * 60, 15 * 60 + 1, 5), rng.choice(lengths)
        tickets[o] = (b, m)
        # hidden world: the vehicle arrived shortly before the ticket was bought and left before or after expiry
        leave = b + m + rng.choice([20, 40, 60, 90]) if rng.random() < 0.5 else b + m - rng.choice([5, 10, 20])
        stay[o] = (b - rng.choice([5, 10, 15, 25]), leave)
    rounds = None
    if rng.random() < 0.5:
        w1 = rng.randrange(8 * 60, 14 * 60, 5)
        rounds = (w1, w1 + rng.randrange(90, 6 * 60, 5))
    f = facts(tickets=tickets, rounds=rounds, rate=rate)

    def say(kind, s, o, **kw):
        return rng.choice(T[kind][0] if s == o else T[kind][1]).format(o=o, v=vehicles[o], **kw)

    truths = []

    def add(kind, o, ok, tags=(), **kw):
        s = rng.choice(sp)
        truths.append(Claim(s, say(kind, s, o, **kw), ok, tags))

    for o in owners:
        b, m = tickets[o]
        e = b + m
        add("bought", o, lambda f, o=o, t=b: f.tickets[o][0] == t, t=clock(b))
        inner = list(range(b + 5, e - 9, 5))
        for t in rng.sample(inner, min(2, len(inner))):
            add("valid", o, lambda f, o=o, t=t: valid(f, o, t), t=clock(t))
        add("valid", o, lambda f, o=o, t=e - 5: valid(f, o, t), ("distractor",), t=clock(e - 5))  # 5 minutes to go
        for t in rng.sample(range(e + 5, e + 91, 5), 2):
            add("expired", o, lambda f, o=o, t=t: t >= sum(f.tickets[o]), t=clock(t))
        add("ends", o, lambda f, o=o, t=e: sum(f.tickets[o]) == t, t=clock(e))
        d = rng.choice([15, 30, 60])
        x, y = (e - d, e) if rng.random() < 0.5 else (e, e + d)
        add("ends_between", o, lambda f, o=o, x=x, y=y: x <= sum(f.tickets[o]) <= y, ("boundary",),
            x=clock(x), y=clock(y))
        if paid:
            add("length", o, lambda f, o=o, m=m: f.tickets[o][1] == m, ("two-fact",), dur=duration(m))
        else:
            amt = rate * m // 60
            add("paid", o, lambda f, o=o, a=amt: f.rate * f.tickets[o][1] == a * 60, ("two-fact",),
                amt=money(amt, cur))
        arrive, leave = stay[o]
        if rounds:
            fines = [w for w in rounds if arrive <= w <= leave and w != e and not valid(f, o, w)]
        elif leave > e + 10 and rng.random() < 0.7:
            fines = [rng.randrange(e + 5, leave, 5)]
        elif b - arrive >= 15 and rng.random() < 0.5:
            fines = [arrive + 5]  # fined before the ticket was bought: sounds wrong, but nothing forbids it
        else:
            fines = []
        for w in fines:
            add("fined", o, lambda f, o=o, t=w: can_fine(f, o, t), ("distractor",), t=clock(w))
    for w in rounds or ():
        truths.append(Claim(rng.choice(sp), rng.choice([f"The warden checked {lot} at {clock(w)}.",
                                                        f"I saw the warden going round {lot} at {clock(w)}."]),
                            lambda f, t=w: t in f.rounds))
    for _ in range(3):
        t = clock(rng.randrange(7 * 60, 21 * 60, 5))
        truths.append(Claim(rng.choice(sp), rng.choice([
            f"I bought a coffee at {t}.", f"At {t} I rang {others[-1]}.", f"I checked the bus times at {t}.",
            f"At {t} I was looking for my keys."]), lambda f: True, ("neutral",)))

    o = rng.choice(owners)
    b, m = tickets[o]
    e = b + m
    s = rng.choice(sp)
    about = () if s == o else ("about-other",)
    two = ("two-fact",) if paid else ()
    far = [t for t in range(8 * 60, 21 * 60, 5) if rounds and min(abs(t - w) for w in rounds) >= 30]
    kinds = ["valid_late", "expired_early", "ends", "bought", "derived", "fined_valid"]
    if rounds:
        kinds += ["fined_offround", "round"]
    lie = None
    while lie is None:
        kind = rng.choice(kinds)
        if kind == "valid_late":
            t = rng.randrange(e + 10, e + 121, 5)
            lie = Claim(s, say("valid", s, o, t=clock(t)), lambda f, t=t, o=o: valid(f, o, t), two + about)
        elif kind == "expired_early":
            t = rng.randrange(b + 10, e - 9, 5)
            lie = Claim(s, say("expired", s, o, t=clock(t)), lambda f, t=t, o=o: t >= sum(f.tickets[o]), two + about)
        elif kind == "ends":
            t = e + rng.choice([d for d in (-60, -45, -30, -15, 15, 30, 45, 60) if e + d > b])
            lie = Claim(s, say("ends", s, o, t=clock(t)), lambda f, t=t, o=o: sum(f.tickets[o]) == t, two + about)
        elif kind == "bought":
            t = b + rng.choice([-30, -20, -15, 15, 20, 30])
            lie = Claim(s, say("bought", s, o, t=clock(t)), lambda f, t=t, o=o: f.tickets[o][0] == t, about)
        elif kind == "derived":
            m2 = rng.choice([x for x in lengths if x != m])
            if paid:
                lie = Claim(s, say("length", s, o, dur=duration(m2)), lambda f, m2=m2, o=o: f.tickets[o][1] == m2,
                            ("two-fact",) + about)
            else:
                a2 = rate * m2 // 60
                lie = Claim(s, say("paid", s, o, amt=money(a2, cur)),
                            lambda f, a2=a2, o=o: f.rate * f.tickets[o][1] == a2 * 60, ("two-fact",) + about)
        elif kind == "fined_valid":
            cand = [w for w in rounds if b + 10 <= w <= e - 10] if rounds else list(range(b + 10, e - 9, 5))
            if cand:
                t = rng.choice(cand)
                lie = Claim(s, say("fined", s, o, t=clock(t)), lambda f, t=t, o=o: can_fine(f, o, t),
                            ("two-fact",) + about)
        elif kind == "fined_offround":
            cand = [t for t in far if t >= e + 10]
            if cand:
                t = rng.choice(cand)
                lie = Claim(s, say("fined", s, o, t=clock(t)), lambda f, t=t, o=o: can_fine(f, o, t),
                            ("two-fact",) + about)
        else:  # a warden check at a time the evidence rules out
            t = rng.choice(far)
            lie = Claim(s, rng.choice([f"The warden checked {lot} at {clock(t)}.",
                                       f"I saw the warden checking {lot} at {clock(t)}."]),
                        lambda f, t=t: t in f.rounds)

    ev = rng.choice([
        [f"In {lot}, parking cost {money(rate, cur)} per hour, and every ticket was valid from the moment it was "
         f"bought for exactly the length of time paid for."],
        [f"Parking in {lot} cost {money(rate, cur)} an hour.",
         "Each ticket was valid from the moment it was bought for exactly as long as had been paid for."],
    ])
    for o in owners:
        b, m = tickets[o]
        v = vehicles[o]
        if paid:
            ev.append(f"At {clock(b)} {o} bought a ticket for {o}'s {v} and paid {money(rate * m // 60, cur)} for it.")
        else:
            ev.append(f"At {clock(b)} {o} bought a ticket for {o}'s {v} covering exactly {duration(m)} of parking.")
    ev.append(f"{owners[0]} bought no other parking ticket that day." if len(owners) == 1 else
              f"Neither {owners[0]} nor {owners[1]} bought any other parking ticket that day.")
    ev.append(f"No vehicle in {lot} was fined while it had a valid ticket.")
    if rounds:
        ev.append(f"That day the warden checked {lot} only twice, at {clock(rounds[0])} and at {clock(rounds[1])}, "
                  f"and gave fines only during those two checks.")
    group = rng.choice(["Drivers", "Shoppers", "Neighbours", "Colleagues"])
    lot_cap = lot[0].upper() + lot[1:]
    return Bundle(
        NAME,
        rng.choice([f"{group} argue over parking tickets at {lot}.",
                    f"{group} compare notes on when their parking tickets ran out.",
                    f"{group} try to work out who got a parking fine at {lot}."]),
        group, f"the parking tickets at {lot}", ev, f, lie, truths,
        fillers=["Is the car park always this full?", "Should we pay by phone next time?",
                 "Does the ticket machine take cards?", "Parking here is far too expensive.",
                 "Who parked nearest the exit?"],
        decor=[f"{lot_cap} has two levels.", "The ticket machine is painted blue.",
               f"{lot_cap} has a height barrier at the entrance."])
