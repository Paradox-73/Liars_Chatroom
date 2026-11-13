"""Recipe scaling. A recipe for s0 people lists exact amounts; it was made once for s1 people with every amount scaled
exactly in proportion and nothing else cooked. Baked dishes may add trays baked one after another from a start time,
each for exactly m minutes, one tray in the oven at a time."""
import itertools
from fractions import Fraction as Fr

from .base import ORDINALS, Bundle, Claim, clock, duration, facts, word

NAME = "recipe"
# (ingredient, unit, low, high): unit "g" or "ml", or the plural for counted items; low and high keep amounts sensible
DISHES = {
    "pancakes": (False, [("flour", "g", 100, 300), ("egg", "eggs", 1, 3), ("milk", "ml", 200, 600)]),
    "flapjacks": (True, [("oats", "g", 200, 450), ("butter", "g", 100, 250), ("honey", "ml", 50, 200)]),
    "dal": (False, [("red lentils", "g", 150, 350), ("water", "ml", 400, 900), ("onion", "onions", 1, 3)]),
    "vegetable soup": (False, [("carrot", "carrots", 2, 4), ("potato", "potatoes", 2, 4),
                               ("stock", "ml", 500, 1200)]),
    "scones": (True, [("flour", "g", 200, 500), ("butter", "g", 50, 150), ("milk", "ml", 100, 300)]),
    "fried rice": (False, [("rice", "g", 200, 450), ("egg", "eggs", 1, 3), ("peas", "g", 100, 250)]),
    "muffins": (True, [("flour", "g", 200, 400), ("sugar", "g", 100, 250), ("egg", "eggs", 1, 3)]),
    "banana bread": (True, [("flour", "g", 200, 350), ("banana", "bananas", 2, 4), ("butter", "g", 75, 150)]),
}
RATIO_WORDS = {Fr(1, 2): "half", Fr(2, 3): "two thirds of", Fr(3, 2): "one and a half times", Fr(2): "double",
               Fr(5, 2): "two and a half times", Fr(3): "three times", Fr(4): "four times", Fr(5): "five times"}
OCCASIONS = ["the street party", "a family lunch", "the club picnic", "the school fair", "a birthday tea",
             "the volunteers' supper"]


def qty(ing, unit, n):
    if unit == "g":
        return f"{n} grams of {ing}"
    if unit == "ml":
        return f"{n} millilitres of {ing}"
    return f"{n} {ing if n == 1 else unit}"


def amount_ok(f, i, n):
    return f.base[i] * f.ratio == n


def in_oven_ok(f, k, t):
    """Tray k (1-based) was in the oven at t: strictly inside its slot, so the edges never decide."""
    return f.trays and f.t0 + (k - 1) * f.m < t < f.t0 + k * f.m


def sl(claim, *key):
    """Set the single-valued fact a claim is about; build never gives the liar a true claim with the lie's key."""
    claim.key = key
    return claim


def make(rng, people, others, day):
    dish = rng.choice(sorted(DISHES))
    baked, spec = DISHES[dish]
    ings = [(ing, unit) for ing, unit, _, _ in spec]
    while True:
        s0 = rng.choice([2, 3, 4, 6])
        s1 = rng.randint(2, 20)
        r = Fr(s1, s0)
        if s1 != s0 and r.denominator <= 3 and r <= 5:
            break
    den = r.denominator
    while True:  # whole scaled amounts by construction; keep counted items to a sane number
        base = []
        for ing, unit, lo, hi in spec:
            step = (25 * den if den > 1 else 50) if unit == "g" else 50 * den if unit == "ml" else den
            k_lo = max(1, -(-lo // step))
            base.append(step * rng.randint(k_lo, max(k_lo, hi // step)))
        scaled = [b * r for b in base]
        if all(x <= 24 for x, (_, u) in zip(scaled, ings) if u not in ("g", "ml")):
            break
    assert all(x.denominator == 1 for x in scaled)
    scaled = [int(x) for x in scaled]
    p = rng.choice(people) if rng.random() < 0.6 else others[0]
    occasion = rng.choice(OCCASIONS)
    trays = baked and rng.random() < 0.6
    n, m, t0 = rng.randint(2, 4), rng.choice([15, 20, 25, 30, 35, 40]), rng.randrange(9 * 60, 17 * 60, 5)
    f = facts(base=base, ratio=r, s1=s1, once=True, trays=trays, n=n, m=m, t0=t0)
    listed = [qty(ing, unit, b) for (ing, unit), b in zip(ings, base)]
    ev = [f"{p}'s recipe for {dish} serves {s0} and uses exactly {listed[0]}, {listed[1]} and {listed[2]}.",
          rng.choice([f"For {occasion}, {p} made the recipe once, for {s1} people, scaling every ingredient exactly "
                      f"in proportion, and cooked nothing else that day.",
                      f"{p} made it just once, for {s1} people at {occasion}, with every amount scaled exactly in "
                      f"proportion, and cooked nothing else that day."])]
    if trays:
        ev.append(f"{p} baked it in {word(n)} trays, one after another with no gap, starting at {clock(t0)}. Each tray "
                  f"took exactly {m} minutes, and only one tray fitted in the oven at a time.")
    sp = list(people)

    def who(s):
        return "I" if s == p else p

    def used(s, i, x):
        ing, unit = ings[i]
        q = qty(ing, unit, x)
        if s != p and rng.random() < 0.4:
            return f"I watched {p} {'measure out' if unit in ('g', 'ml') else 'use'} {q} in total for the {dish}."
        return rng.choice([f"{who(s)} used {q} in the {dish}.", f"The {dish} took {q}."])

    truths = []
    turn = itertools.cycle(rng.sample(sp, len(sp)))  # spread the senders over all four
    for i in range(3):
        s = next(turn)
        truths.append(sl(Claim(s, used(s, i, scaled[i]), lambda f, i=i, x=scaled[i]: amount_ok(f, i, x)), "amount", i))
    s = next(turn)
    truths.append(sl(Claim(s, f"{who(s)} made the {dish} for {s1} people.", lambda f: f.s1 == s1), "people"))
    if r in RATIO_WORDS:
        s = next(turn)
        truths.append(sl(Claim(s, f"{who(s)} made exactly {RATIO_WORDS[r]} the recipe.", lambda f: f.ratio == r,
                               () if r.denominator == 1 else ("distractor",)), "ratio"))
    i = rng.randrange(3)
    s = next(turn)
    more = r > 1
    ing, unit = ings[i]
    what = f"{'more' if more else 'less'} {ing}" if unit in ("g", "ml") else f"{'more' if more else 'fewer'} {unit}"
    truths.append(sl(Claim(s, f"{who(s)} used {what} than the recipe lists.",
                           lambda f, more=more: (f.ratio > 1) == more), "amount", i))
    i = rng.randrange(3)
    s = next(turn)
    ing, unit = ings[i]
    truths.append(sl(Claim(s, f"{who(s)} used {qty(ing, unit, scaled[i])}, not the {base[i]} on the recipe card.",
                           lambda f, i=i, b=base[i], x=scaled[i]: f.base[i] == b and amount_ok(f, i, x),
                           ("distractor",)), "amount", i))
    if trays:
        k = rng.randint(1, n)
        truths.append(Claim(next(turn), f"The {ORDINALS[k]} tray went into the oven at {clock(t0 + (k - 1) * m)}.",
                            lambda f, k=k, t=t0 + (k - 1) * m: f.trays and f.t0 + (k - 1) * f.m == t,
                            ("boundary",)))
        k = rng.randint(1, n)
        t = rng.randrange(t0 + (k - 1) * m + 5, t0 + k * m - 4, 5)
        truths.append(Claim(next(turn), f"At {clock(t)} the {ORDINALS[k]} tray was in the oven.",
                            lambda f, k=k, t=t: in_oven_ok(f, k, t)))
        truths.append(sl(Claim(next(turn), f"The last tray came out of the oven at {clock(t0 + n * m)}.",
                               lambda f, t=t0 + n * m: f.trays and f.t0 + f.n * f.m == t), "last"))
        truths.append(Claim(next(turn), f"Baking {'both' if n == 2 else 'all ' + word(n)} trays took "
                                            f"{duration(n * m)} without a break.",
                            lambda f: f.trays and f.n * f.m == n * m))
        neutral = [f"I washed up at {clock(t0 + n * m + 30)}.", f"I set the table at {clock(t0 - 30)}.",
                   f"I found the recipe on page {rng.randint(12, 240)} of the cookbook.",
                   f"I brought {rng.randint(2, 6)} chairs over from next door."]
    else:
        neutral = [f"I found the recipe on page {rng.randint(12, 240)} of the cookbook.",
                   f"I bought the {ings[0][0] if ings[0][1] in ('g', 'ml') else ings[0][1]} from stall "
                   f"{rng.randint(2, 30)} at the market.", f"I brought {rng.randint(2, 6)} chairs over from next door.",
                   f"I lent {p} my {rng.choice([24, 26, 28, 30])} centimetre mixing bowl.",
                   f"I put the leftovers in the fridge on shelf {rng.randint(2, 4)}."]
    for text in neutral:
        truths.append(Claim(next(x for x in turn if x != p), text, lambda f: True, ("neutral",)))
    s = next(x for x in turn if x != p)
    truths.append(Claim(s, f"I made the same recipe for {rng.choice([x for x in range(2, 13) if x != s1])} people "
                           f"last month.", lambda f: True, ("neutral", "distractor")))

    s = rng.choice(sp)
    about = () if s == p else ("about-other",)
    kinds = ["amount", "amount", "people", "twice"] + (["ratio"] if r in RATIO_WORDS else []) + (
        ["tray", "tray"] if trays else [])
    kind = rng.choice(kinds)
    if kind == "amount":
        i = rng.randrange(3)
        unit = ings[i][1]
        step = 25 if unit == "g" else 50 if unit == "ml" else 1
        wrong = [x for x in {base[i], scaled[i] + step, scaled[i] - step, scaled[i] + 2 * step,
                             base[i] * (s1 + 1) // s0} if x > 0 and x != scaled[i]]
        x = rng.choice(sorted(wrong))
        lie = sl(Claim(s, used(s, i, x), lambda f, i=i, x=x: amount_ok(f, i, x), ("two-fact",) + about), "amount", i)
    elif kind == "people":
        x = rng.choice([y for y in range(2, 21) if y != s1])
        lie = sl(Claim(s, f"{who(s)} made the {dish} for {x} people.", lambda f, x=x: f.s1 == x, about), "people")
    elif kind == "twice":
        lie = Claim(s, f"{who(s)} made the {dish} twice that day, in {word(2)} separate batches.",
                    lambda f: not f.once, about)
    elif kind == "ratio":
        x = rng.choice([y for y in RATIO_WORDS if y != r])
        lie = sl(Claim(s, f"{who(s)} made exactly {RATIO_WORDS[x]} the recipe.", lambda f, x=x: f.ratio == x,
                       ("two-fact",) + about), "ratio")
    else:
        sub = rng.choice(["last", "other_slot", "empty"])
        if sub == "last":
            t = t0 + (n - 1) * m
            lie = sl(Claim(s, f"The last tray came out of the oven at {clock(t)}.",
                           lambda f, t=t: f.trays and f.t0 + f.n * f.m == t, ("two-fact",)), "last")
        elif sub == "other_slot":
            k = rng.randint(1, n)
            j = rng.choice([y for y in range(1, n + 1) if y != k])
            t = rng.randrange(t0 + (j - 1) * m + 5, t0 + j * m - 4, 5)
            lie = Claim(s, f"At {clock(t)} the {ORDINALS[k]} tray was in the oven.",
                        lambda f, k=k, t=t: in_oven_ok(f, k, t), ("two-fact",))
        else:  # never exactly at a tray change, where 'empty' could be argued for a moment
            t = rng.choice([x for x in range(t0 + 5, t0 + n * m - 4, 5) if (x - t0) % m])
            lie = Claim(s, f"At {clock(t)} the oven was empty.",
                        lambda f, t=t: not (f.trays and f.t0 <= t <= f.t0 + f.n * f.m), ("two-fact",))
    group = rng.choice(["Friends", "Housemates", "Community kitchen volunteers", "Neighbours"])
    return Bundle(
        NAME,
        rng.choice([f"{group} argue over how much went into {p}'s {dish}.",
                    f"{group} compare notes on the {dish} {p} made for {occasion}.",
                    f"{group} piece together how {p} scaled the {dish} recipe."]),
        group, f"{p}'s {dish}", ev, f, lie, truths,
        fillers=["Was it any good?", "Can I have the recipe?", "Who brought the dessert?", "That smelled lovely.",
                 "Should we make more next time?"],
        decor=["The kitchen has two sinks.", "The bowls were washed afterwards.",
               "The recipe card has a coffee stain on it."],
        timed=trays)
