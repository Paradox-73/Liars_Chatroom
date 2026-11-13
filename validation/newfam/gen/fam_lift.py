"""Lift capacity. Variant 'weight': the lift carries at most cap kilograms and never carried more; some people's
weights and the loads of two trips are stated. Variant 'people': at most N people at once; on one trip exactly a got
in at the ground floor and nobody got out before floor F."""
from .base import ORDINALS, Bundle, Claim, clock, facts, word

NAME = "lift"
LOADS = ["pallet of tiles", "crate of paint", "trolley of books", "cage of laundry", "pallet of bricks",
         "rack of costumes", "trolley of printer paper", "crate of tinned food"]
BUILDINGS = ["the library", "the hotel", "the office tower", "the hospital", "the multi-storey car park",
             "the student halls", "the shopping centre"]


def fits(f, trip, riders):
    """Riders could have gone with that trip's load without passing the limit (an unknown weight counts as 1 kg)."""
    return f.load[trip] + sum(f.w.get(p) or 1 for p in riders) <= f.cap


def carried_ok(f, trip, kg):
    """The lift's whole load on that trip: at least the stated load, at most the limit."""
    return f.load[trip] <= kg <= f.cap


def count_ok(f, k, c):
    """People aboard between floor k and k + 1 (k + 1 <= F): nobody got out, so at least a; never more than N."""
    return f.a <= c <= f.N


def sl(claim, *key):
    """Set the single-valued fact a claim is about; build never gives the liar a true claim with the lie's key."""
    claim.key = key
    return claim


def floor(k):
    return "ground" if k == 0 else ORDINALS[k]


def make_weight(rng, people, others, day):
    cap = rng.choice([350, 400, 450, 500, 600, 630])
    while True:
        known = list(people) if rng.random() < 0.5 else rng.sample(people, 3)
        w = {p: rng.randint(52, 108) for p in known}
        if len(set(w.values())) == len(w) and max(w.values()) - min(w.values()) >= 8:
            break
    order = sorted(known, key=w.get)
    light, heavy = order[0], order[-1]
    a, b = rng.sample(known, 2)  # the pair that rode the first load, landing exactly on the limit
    c = rng.choice([p for p in known if p not in (a, b)])
    load1, load2, load3 = rng.sample(LOADS, 3)
    l1 = cap - w[a] - w[b]
    d = (cap - w[light]) % 5
    l2 = cap - w[light] - (d if d < w[heavy] - w[light] else 0)  # the lightest fits, the heaviest does not
    l3 = rng.randrange(100, cap - 150, 10)  # a third trip the evidence never mentions
    t1 = rng.randrange(8 * 60, 11 * 60, 5)
    t2 = t1 + rng.randrange(45, 180, 5)
    t3 = t2 + rng.randrange(40, 150, 5)
    f = facts(cap=cap, w={p: w.get(p) for p in people}, load={1: l1, 2: l2, 3: l3})
    kg = lambda p: f"{p} weighs {w[p]} kilograms"  # noqa: E731
    listed = rng.sample(known, len(known))
    ev = [rng.choice([f"The goods lift can carry at most {cap} kilograms at a time, and it never carried more than "
                      f"that.", f"The goods lift never carried more than its limit of {cap} kilograms."]),
          ", ".join(kg(p) for p in listed[:-1]) + f" and {kg(listed[-1])}.",
          rng.choice([f"At {clock(t1)} the lift took {an_kg(l1, load1)} up to the {ORDINALS[rng.randint(2, 6)]} "
                      f"floor.", f"At {clock(t1)} the {load1}, which weighed {l1} kilograms, went up in the lift."]),
          rng.choice([f"At {clock(t2)} it took {an_kg(l2, load2)} up to the {ORDINALS[rng.randint(2, 6)]} floor.",
                      f"At {clock(t2)} the lift carried {an_kg(l2, load2)} upstairs."])]
    rode = lambda load, t: rng.choice([f"I rode up with the {load} at {clock(t)}.",  # noqa: E731
                                       f"At {clock(t)} I went up in the lift with the {load}."])
    saw = lambda x, load, t: rng.choice([f"I saw {x} ride up with the {load} at {clock(t)}.",  # noqa: E731
                                         f"{x} went up in the lift with the {load} at {clock(t)}."])
    sp = list(people)
    truths = [
        sl(Claim(b, f"{a} and I rode up together with the {load1} at {clock(t1)}.",
                 lambda f: fits(f, 1, (a, b)), ("boundary",)), "ride1"),
        sl(Claim(a, rode(load1, t1), lambda f: fits(f, 1, (a,))), "ride1"),
        sl(Claim(rng.choice([p for p in sp if p not in (a, b)]),
                 f"I saw {a} and {b} go up in the lift with the {load1} at {clock(t1)}.",
                 lambda f: fits(f, 1, (a, b))), "ride1"),
        sl(Claim(rng.choice(sp), f"On the {clock(t1)} trip the lift was loaded right up to its {cap} kilogram limit.",
                 lambda f: carried_ok(f, 1, cap), ("boundary", "distractor")), "total", 1),
        sl(Claim(light, rode(load2, t2), lambda f: fits(f, 2, (light,)),
                 ("boundary",) if l2 + w[light] == cap else ()), "ride2", light),
        sl(Claim(rng.choice([p for p in sp if p != light]), saw(light, load2, t2), lambda f: fits(f, 2, (light,))),
           "ride2", light),
        sl(Claim(rng.choice(sp), f"On the {clock(t2)} trip the lift carried {l2 + w[light]} kilograms in all.",
                 lambda f, x=l2 + w[light]: carried_ok(f, 2, x)), "total", 2),
        sl(Claim(heavy, f"I was too heavy to ride up with the {load2}, so I took the stairs at {clock(t2)}.",
                 lambda f: not fits(f, 2, (heavy,)), ("distractor",)), "ride2", heavy),
    ]
    # a trip the evidence does not list: lists of trips are not complete, so it cannot be contradicted
    s = rng.choice(sp)
    truths.append(Claim(s, f"At {clock(t3)} I took {an_kg(l3, load3)} up in the lift on my own.",
                        lambda f, s=s: fits(f, 3, (s,)), ("distractor",)))
    for t in rng.sample([t1 - 40, t2 - 30, t3 + 35, t3 + 70], 3):
        truths.append(Claim(rng.choice(sp), rng.choice([
            f"I signed for a delivery at {clock(t)}.", f"I swept the loading bay at {clock(t)}.",
            f"I was checking the stock list at {clock(t)}."]), lambda f: True, ("neutral",)))

    kind = rng.choice(["heavy", "saw_heavy", "third", "over", "under"])
    if kind == "heavy":
        lie = sl(Claim(heavy, rode(load2, t2), lambda f: fits(f, 2, (heavy,)), ("two-fact",)), "ride2", heavy)
    elif kind == "saw_heavy":
        lie = sl(Claim(rng.choice([p for p in sp if p != heavy]), saw(heavy, load2, t2),
                       lambda f: fits(f, 2, (heavy,)), ("two-fact", "about-other")), "ride2", heavy)
    elif kind == "third":
        if rng.random() < 0.5:
            lie = sl(Claim(c, f"{a}, {b} and I all rode up together with the {load1} at {clock(t1)}.",
                           lambda f: fits(f, 1, (a, b, c)), ("two-fact",)), "ride1")
        else:
            s = rng.choice([p for p in sp if p not in (a, b, c)] or [c])
            lie = sl(Claim(s, f"I saw {a}, {b} and {c} all go up together with the {load1} at {clock(t1)}.",
                           lambda f: fits(f, 1, (a, b, c)), ("two-fact", "about-other")), "ride1")
    elif kind == "over":
        x = cap + rng.randrange(15, 150, 5)
        lie = sl(Claim(rng.choice(sp), f"On the {clock(t2)} trip the lift carried {x} kilograms in all.",
                       lambda f, x=x: carried_ok(f, 2, x)), "total", 2)
    else:
        x = l1 - rng.randrange(20, 90, 5)
        lie = sl(Claim(rng.choice(sp), f"On the {clock(t1)} trip the lift carried only {x} kilograms in all.",
                       lambda f, x=x: carried_ok(f, 1, x)), "total", 1)
    return ev, f, lie, truths, "the goods lift"


def make_people(rng, people, others, day):
    n_max = rng.choice([6, 8, 10, 12])
    a = rng.randint(max(2, n_max - 4), n_max)
    top = rng.randint(4, 7)  # nobody got out before this floor
    t = rng.randrange(7 * 60, 19 * 60, 5)
    where = rng.choice(BUILDINGS)
    f = facts(N=n_max, a=a, F=top)
    cap_text = rng.choice([str(n_max), word(n_max)])
    ev = [rng.choice([f"The lift in {where} holds at most {cap_text} people at a time, and it never held more than "
                      f"that.", f"The lift in {where} never had more than {cap_text} people in it at once."]),
          f"On its {clock(t)} trip up from the ground floor, exactly {word(a)} people got in at the ground floor, and "
          f"nobody got out before the {ORDINALS[top]} floor."]
    # hidden world: extra people get in below the top floor without passing the limit
    ins = {k: 0 for k in range(1, top)}
    for _ in range(rng.randint(min(1, n_max - a), n_max - a)):
        ins[rng.randint(1, top - 1)] += 1
    aboard = {0: a}
    for k in range(1, top):
        aboard[k] = aboard[k - 1] + ins[k]
    sp = list(people)
    rng.shuffle(sp)
    g1, g2, late, off = sp  # two rode from the ground floor, one got in higher up (if anyone did), one took the stairs
    trip = f"the {clock(t)} trip"
    truths = [
        sl(Claim(g1, rng.choice([f"I got in at the ground floor for {trip}.",
                                 f"On {trip} I got into the lift at the ground floor."]), lambda f: f.a >= 1),
           "in", g1),
        sl(Claim(g1, f"On {trip} I got out at the {ORDINALS[top]} floor.", lambda f, j=top: j >= f.F,
                 ("boundary", "distractor")), "out", g1),
        sl(Claim(g2, f"I rode {trip} from the ground floor up to the {ORDINALS[top + 1]} floor.",
                 lambda f, j=top + 1: j >= f.F), "out", g2),
        Claim(off, f"I took the stairs instead of {trip}.", lambda f: True, ("neutral",)),
        sl(Claim(rng.choice(sp), f"There were exactly {word(a)} people in the lift between the ground and first "
                                 f"floors on {trip}.", lambda f, c=a: count_ok(f, 0, c)), "count", 0),
    ]
    for k in rng.sample(range(1, top), min(3, top - 1)):
        c = aboard[k]
        truths.append(sl(Claim(rng.choice(sp), rng.choice([
            f"There were {c} people in the lift between the {ORDINALS[k]} and {ORDINALS[k + 1]} floors on {trip}.",
            f"Between the {ORDINALS[k]} and {ORDINALS[k + 1]} floors, {trip} had {c} people aboard."]),
            lambda f, k=k, c=c: count_ok(f, k, c), ("boundary",) if c == n_max else ()), "count", k))
    joined = [k for k in range(1, top) if ins[k]]
    nobody = [k for k in range(1, top) if not ins[k]]
    for k in joined[:2] + rng.sample(nobody, min(1, len(nobody))):
        b = ins[k]
        truths.append(sl(Claim(rng.choice(sp), f"{more(b)} got in at the {ORDINALS[k]} floor on {trip}." if b
                               else f"Nobody got in at the {ORDINALS[k]} floor on {trip}.",
                               lambda f, b=b: f.a + b <= f.N), "ins", k))
    if joined:
        k = rng.choice(joined)
        truths.append(sl(Claim(late, f"I got in at the {ORDINALS[k]} floor on {trip}.",
                               lambda f, k=k: k >= f.F or f.a < f.N), "in", late))
    else:  # nobody joined below the top floor; getting in above it is possible even if the lift had been full
        truths.append(sl(Claim(late, f"I got in at the {ORDINALS[top + 1]} floor on {trip}.",
                               lambda f, k=top + 1: k >= f.F or f.a < f.N, ("distractor",)), "in", late))
    if aboard[top - 1] == n_max:
        truths.append(Claim(rng.choice(sp), f"The lift was full when it reached the {ORDINALS[top]} floor on {trip}.",
                            lambda f: f.a <= f.N, ("boundary",)))
    for m in rng.sample([t - 50, t - 25, t + 30, t + 55], 2):
        truths.append(Claim(rng.choice(sp), rng.choice([
            f"I pressed the call button on the second floor at {clock(m)}.", f"I collected my post at {clock(m)}.",
            f"At {clock(m)} I was at the front desk."]), lambda f: True, ("neutral",)))

    kinds = ["over", "under", "early_out", "saw_out", "more_in"] + (["full_in", "full_in"] if a == n_max else [])
    kind = rng.choice(kinds)
    k = rng.randint(1, top - 1)
    s = rng.choice(sp)
    if kind == "over":
        c = n_max + rng.randint(1, 3)
        lie = sl(Claim(s, f"There were {c} people in the lift between the {floor(k - 1)} and {ORDINALS[k]} floors "
                          f"on {trip}.", lambda f, k=k - 1, c=c: count_ok(f, k, c)), "count", k - 1)
    elif kind == "under":
        c = rng.randint(max(1, a - 3), a - 1)
        who = "was only one person" if c == 1 else f"were only {word(c)} people"
        lie = sl(Claim(s, f"There {who} in the lift between the {ORDINALS[k]} and {ORDINALS[k + 1]} floors on "
                          f"{trip}.", lambda f, k=k, c=c: count_ok(f, k, c), ("two-fact",)), "count", k)
    elif kind == "early_out":
        lie = sl(Claim(s, f"On {trip} I got out at the {ORDINALS[k]} floor.", lambda f, k=k: k >= f.F), "out", s)
    elif kind == "saw_out":
        x = rng.choice([p for p in sp if p != s] + others[:2])
        lie = sl(Claim(s, f"On {trip} I saw {x} get out at the {ORDINALS[k]} floor.", lambda f, k=k: k >= f.F,
                       ("about-other",)), "out", x)
    elif kind == "more_in":
        b = n_max - a + rng.randint(1, 3)
        lie = sl(Claim(s, f"{more(b)} got in at the {ORDINALS[k]} floor on {trip}.",
                       lambda f, b=b: f.a + b <= f.N, ("two-fact",)), "ins", k)
    else:  # the lift left the ground floor full and nobody got out before the top floor
        lie = sl(Claim(s, f"I got in at the {ORDINALS[k]} floor on {trip}.", lambda f, k=k: k >= f.F or f.a < f.N,
                       ("two-fact",)), "in", s)
    return ev, f, lie, truths, f"the lift in {where}"


def more(b):
    return "One more person" if b == 1 else f"{word(b).capitalize()} more people"


def an_kg(kg, load):
    return f"a {kg} kilogram {load}"


def make(rng, people, others, day):
    ev, f, lie, truths, topic = (make_weight if rng.random() < 0.5 else make_people)(rng, people, others, day)
    group = rng.choice(["Warehouse staff", "Building staff", "Office workers", "Caretakers"])
    return Bundle(
        NAME,
        rng.choice([f"{group} argue over who rode {topic} and with what.",
                    f"{group} piece together the trips of {topic}.",
                    f"{group} compare notes on how full {topic} was."]),
        group, topic, ev, f, lie, truths,
        fillers=["Should the lift get a bigger motor?", "Is the lift ever serviced?", "Who keeps the lift key?",
                 "That lift is far too slow.", "Why does the lift always smell of paint?"],
        decor=["The lift has a mirror on the back wall.", "The lift doors are painted grey.",
               "The lift was installed when the building opened."])
