"""Medicine doses. Variant 'gap': a first dose at t0, every later dose at least g hours after the one before, at
most k doses that day. Variant 'fixed': exactly k doses, one every g hours exactly, starting at t0."""
from .base import DAY_END, ORDINALS, Bundle, Claim, an, clock, duration, facts, word

NAME = "medicine"
DRUGS = [("painkiller", "painkillers"), ("antibiotic capsule", "antibiotic capsules"),
         ("allergy tablet", "allergy tablets"), ("iron tablet", "iron tablets"),
         ("dose of cough syrup", "doses of cough syrup"), ("anti-sickness tablet", "anti-sickness tablets")]


def can_take(f, t):
    if f.fixed:
        return t in f.sched
    return t == f.t0 or (f.k >= 2 and f.t0 + f.gap <= t <= DAY_END)


def can_nth(f, n, t):
    if f.fixed:
        return 1 <= n <= f.k and f.sched[n - 1] == t
    if n == 1:
        return t == f.t0
    return 2 <= n <= f.k and f.t0 + (n - 1) * f.gap <= t <= DAY_END


def can_count(f, n):
    if f.fixed:
        return n == f.k
    return 1 <= n <= f.k and f.t0 + (n - 1) * f.gap <= DAY_END


def can_pair(f, d):
    """Two of the doses exactly d minutes apart."""
    if f.fixed:
        return any(b - a == d for a in f.sched for b in f.sched)
    return f.k >= 2 and d >= f.gap and f.t0 + d <= DAY_END


def make(rng, people, others, day):
    one, many = rng.choice(DRUGS)
    p = rng.choice(people) if rng.random() < 0.5 else others[0]
    fixed = rng.random() < 0.35
    g = rng.choice([3, 4, 5, 6] if fixed else [3, 4, 5, 6, 8])
    gap = g * 60
    t0 = rng.randrange(6 * 60, 10 * 60 + 1, 15)
    kmax = min(4, (22 * 60 - t0) // gap + 1)
    k = rng.randint(2, kmax) if not fixed else rng.randint(3, kmax) if kmax >= 3 else 2
    if fixed:
        sched = [t0 + i * gap for i in range(k)]
        ev = [f"{p} took {word(k)} {many} that day, one every {g} hours exactly, starting at {clock(t0)}."]
    else:
        # hidden world: the doses actually taken, each at least the gap after the one before
        sched, t = [t0], t0
        for _ in range(rng.randint(2, k) - 1):
            t += gap + rng.choice([0, 0, 15, 30, 45, 60, 90])
            if t > 23 * 60:
                break
            sched.append(t)
        ev = rng.choice([
            [f"{p} took the first {one} of the day at {clock(t0)}.",
             f"Every later {one} that day came at least {g} hours after the one before it, and {p} took no more "
             f"than {word(k)} {many} that day."],
            [f"{p} took no more than {word(k)} {many} that day, and each one after the first came at least {g} "
             f"hours after the previous one.", f"The first one was at {clock(t0)}."],
        ])
    f = facts(fixed=fixed, t0=t0, gap=gap, k=k, sched=sched)

    def subj(s):
        return ("I", "my") if s == p else (p, f"{p}'s")

    def took(s, t):
        if s == p:
            return rng.choice([f"I took {an(one)} at {clock(t)}.", f"At {clock(t)} I took {an(one)}."])
        return rng.choice([f"I gave {p} {an(one)} at {clock(t)}.", f"{p} took {an(one)} at {clock(t)}.",
                           f"At {clock(t)} I saw {p} take {an(one)}."])

    def nth(s, n, t):
        who, whose = subj(s)
        return rng.choice([f"{whose.capitalize()} {ORDINALS[n]} {one} that day was at {clock(t)}.",
                           f"{who} took the {ORDINALS[n]} {one} of the day at {clock(t)}."])

    def count(s, n):
        who, _ = subj(s)
        return f"{who} took {word(n)} {many} that day."

    sp = list(people)
    truths = []
    for i, t in enumerate(sched):
        s = rng.choice(sp)
        truths.append(Claim(s, took(s, t), lambda f, t=t: can_take(f, t)))
        s = rng.choice(sp)
        truths.append(Claim(s, nth(s, i + 1, t), lambda f, n=i + 1, t=t: can_nth(f, n, t)))
    s = rng.choice(sp)
    truths.append(Claim(s, count(s, len(sched)), lambda f, n=len(sched): can_count(f, n)))
    gaps = [b - a for a, b in zip(sched, sched[1:])]
    if gap in gaps:  # the hidden world has a dose exactly at the minimum gap: true, and it sounds risky
        s = rng.choice(sp)
        who, _ = subj(s)
        truths.append(Claim(s, f"{who} took two {many} exactly {g} hours apart.", lambda f: can_pair(f, gap),
                            ("boundary", "distractor")))
    if len(sched) < k and not fixed:
        s = rng.choice(sp)
        who, _ = subj(s)
        truths.append(Claim(s, f"{who} took fewer than {word(k)} {many} that day.",
                            lambda f: can_count(f, len(sched)), ("distractor",)))
    for _ in range(3):
        s = rng.choice([x for x in sp if x != p] or sp)
        t = rng.randrange(7 * 60, 21 * 60, 5)
        truths.append(Claim(s, rng.choice([
            f"I picked up {p}'s prescription at {clock(t)}.", f"At {clock(t)} I reminded {p} about the {many}.",
            f"I put the {many} back in the cupboard at {clock(t)}.", f"I rang the pharmacy at {clock(t)}."]),
            lambda f: True, ("neutral",)))

    s = rng.choice(sp)
    about = () if s == p else ("about-other",)
    kinds = ["early", "count", "pair", "nth", "between" if fixed else "gap"]
    kind = rng.choice(kinds)
    if kind == "early":
        t = rng.randrange(max(5 * 60 + 30, t0 - 150), t0 - 10, 5)
        lie = Claim(s, took(s, t), lambda f, t=t: can_take(f, t), about)
    elif kind == "gap":  # needs the first dose and the minimum gap
        t = rng.randrange(t0 + 20, t0 + gap - 10, 5)
        lie = Claim(s, took(s, t), lambda f, t=t: can_take(f, t), ("two-fact",) + about)
    elif kind == "between":  # fixed schedule only: a time strictly between two of the doses
        a = rng.randrange(len(sched) - 1)
        t = rng.randrange(sched[a] + 20, sched[a + 1] - 10, 5)
        lie = Claim(s, took(s, t), lambda f, t=t: can_take(f, t), ("two-fact",) + about)
    elif kind == "count":
        n = k + rng.choice([1, 1, 2])
        lie = Claim(s, count(s, n), lambda f, n=n: can_count(f, n), about)
    elif kind == "pair":
        d = rng.randrange(60, gap - 29, 30)
        who, _ = subj(s)
        lie = Claim(s, f"{who} took two {many} only {duration(d)} apart.", lambda f, d=d: can_pair(f, d),
                    ("two-fact",) + about)
    else:  # nth dose earlier than the chain of gaps allows
        n = rng.randint(2, k)
        t = rng.randrange(t0 + (n - 2) * gap + 20, t0 + (n - 1) * gap - 10, 5)
        lie = Claim(s, nth(s, n, t), lambda f, n=n, t=t: can_nth(f, n, t), ("two-fact",) + about)

    group = rng.choice(["Carers", "Family members", "Ward staff", "Housemates"])
    return Bundle(
        NAME,
        rng.choice([f"{group} compare notes on when {p} took {many} during the day.",
                    f"{group} argue over how many {many} {p} took in one day.",
                    f"{group} piece together {p}'s {one} times."]),
        group, f"{p}'s {many}", ev, f, lie, truths,
        fillers=[f"Is {p} feeling any better?", "Should we write the times on the box?",
                 "Who has the medicine chart?", "Does the pharmacy open on Sundays?",
                 f"Should {p} see a doctor about this?"],
        decor=[f"The {many} are kept in a blue box.", "The pharmacy is next to the bakery.",
               f"{p}'s prescription was renewed last month."])
