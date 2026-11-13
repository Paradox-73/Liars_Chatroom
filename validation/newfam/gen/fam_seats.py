"""Seats, desks or lockers with 'exactly' and 'only'. N places in one row, numbered 1 to N in order, go to exactly N
named people, one place each, and only that person used the place (sat there the whole time, or held its key).
Clues fix part of the assignment ('Ana had locker 3', 'Ben's locker number was one higher than Di's', 'Cy did not
sit in seat 2', 'Ed sat at one end of the row', 'Fay sat directly next to Gil'). Some puzzles have one solution,
others leave places open. ok() is brute force over every assignment that satisfies the clues."""
from itertools import permutations

from .base import Bundle, Claim, facts, word

NAME = "seats"

SURFACES = [
    dict(setup="Lockers 1 to {N} stand side by side in one row, in number order.",
         share="Each of {names} was given exactly one of these lockers, no two of them shared one, and nobody else was "
               "given one. Only the person given a locker had a key to it.",
         fixed=["{x} had locker {k}.", "Locker {k} was given to {x}."],
         no=["{x} did not have locker {k}.", "Locker {k} was not {x}'s."],
         next=["{x}'s locker was right next to {y}'s.", "{x} and {y} had lockers side by side."],
         offset=["{x}'s locker number was {d} higher than {y}'s."],
         end=["{x}'s locker was at one end of the row."],
         notnext=["{x}'s locker was not next to {y}'s."],
         lower=["{x}'s locker number was lower than {y}'s."],
         p_me=["My locker was number {k}.", "I opened locker {k} with my own key.", "I had the key to locker {k}."],
         p=["{x} had locker {k}.", "{x}'s key opened locker {k}.", "Locker {k} was {x}'s."],
         n_me=["My locker was right next to {y}'s.", "{y}'s locker was next to mine."],
         n=["{x}'s locker was right next to {y}'s."],
         e_me=["My locker was at one end of the row."], e=["{x}'s locker was at one end of the row."],
         e_too=["My locker was at one end of the row too.", "{x}'s locker was at the other end of the row."],
         b_me=["My locker was directly between {x}'s and {y}'s."], b=["{m}'s locker was directly between {x}'s and {y}'s."],
         h_me=["My locker number was higher than {y}'s."], h=["{x}'s locker number was higher than {y}'s."],
         gap=["There was exactly one locker between {x}'s and {y}'s."],
         out=["I didn't have a locker in that row.", "My own locker is in another corridor."],
         neutral=["I keep {n} books in my locker.", "I go to the gym {n} times a week."],
         group="Gym members", topic="the row of lockers",
         scen=["{g} argue over who had which locker.", "{g} try to work out whose locker was whose.",
               "{g} compare notes on the lockers in the changing room."],
         fill=["Who keeps the spare keys?", "Can lockers be swapped?", "Should the lockers have name tags?"],
         decor=["The lockers are painted blue.", "The changing room has a long bench."]),
    dict(setup="Seats 1 to {N} in row {R} of the {hall} are side by side, in number order.",
         share="For the whole {event}, those seats were taken by exactly {names}, one person per seat; nobody else sat in "
               "them and nobody changed seats.",
         fixed=["{x} sat in seat {k}.", "Seat {k} was {x}'s."],
         no=["{x} did not sit in seat {k}."],
         next=["{x} sat directly next to {y}.", "{x} and {y} sat side by side."],
         offset=["{x}'s seat number was {d} higher than {y}'s."],
         end=["{x} sat at one end of the row.", "{x} had an end seat."],
         notnext=["{x} and {y} did not sit next to each other."],
         lower=["{x}'s seat number was lower than {y}'s."],
         p_me=["I sat in seat {k}.", "My seat was number {k}."], p=["{x} sat in seat {k}.", "{x} was in seat {k}."],
         n_me=["I sat right next to {y}.", "{y} was sitting right next to me."], n=["{x} sat right next to {y}."],
         e_me=["I had an end seat.", "I sat at one end of the row."], e=["{x} had an end seat."],
         e_too=["I had an end seat as well.", "{x} sat at the other end of the row."],
         b_me=["I sat directly between {x} and {y}."], b=["{m} sat directly between {x} and {y}."],
         h_me=["My seat number was higher than {y}'s."], h=["{x}'s seat number was higher than {y}'s."],
         gap=["There was exactly one seat between {x} and {y}."],
         out=["I sat further back, in row {R2}.", "I wasn't in row {R} at all."],
         neutral=["I bought my ticket {n} weeks before the {event}.", "I took the number {b} bus home afterwards."],
         group="Audience members", topic="who sat where in row {R}",
         scen=["{g} piece together who sat where in row {R} at the {event}.", "{g} argue over their seats at the {event}.",
               "{g} compare notes on row {R} of the {hall}."],
         fill=["Was the sound any good from that row?", "Should we book the balcony next time?", "Who chose these seats?"],
         decor=["Row {R} is close to the stage.", "The seats in the {hall} are red velvet."]),
    dict(setup="Desks 1 to {N} stand in a single row at the front of the exam hall, in number order.",
         share="During the exam each of these desks was used by exactly one of {names}, one person per desk; nobody else "
               "used them and nobody moved desks.",
         fixed=["{x} sat at desk {k}.", "Desk {k} was {x}'s."],
         no=["{x} was not at desk {k}."],
         next=["{x}'s desk was next to {y}'s.", "{x} and {y} sat at neighbouring desks."],
         offset=["{x}'s desk number was {d} higher than {y}'s."],
         end=["{x} sat at one end of the row."],
         notnext=["{x} and {y} did not sit at neighbouring desks."],
         lower=["{x}'s desk number was lower than {y}'s."],
         p_me=["I sat at desk {k}.", "My desk was number {k}."], p=["{x} sat at desk {k}.", "{x} was at desk {k}."],
         n_me=["My desk was next to {y}'s.", "{y} sat at the desk next to mine."], n=["{x}'s desk was next to {y}'s."],
         e_me=["I sat at one end of the row."], e=["{x} sat at one end of the row."],
         e_too=["I sat at one end of the row too.", "{x} sat at the other end of the row."],
         b_me=["My desk was directly between {x}'s and {y}'s."], b=["{m}'s desk was directly between {x}'s and {y}'s."],
         h_me=["My desk number was higher than {y}'s."], h=["{x}'s desk number was higher than {y}'s."],
         gap=["There was exactly one desk between {x}'s and {y}'s."],
         out=["I sat my exam in the back row.", "I wasn't in the front row."],
         neutral=["I brought {n} pencils to the exam.", "I answered {n} of the essay questions."],
         group="Students", topic="the front row of desks in the exam hall",
         scen=["{g} try to work out who sat at which desk in the exam.", "{g} argue over the seating in the exam hall.",
               "{g} compare notes on the front row of the exam hall."],
         fill=["Was the exam hall too cold?", "Who set the seating plan?", "Do we get the same desks next time?"],
         decor=["The exam hall has tall windows.", "A clock hangs above the exam hall door."]),
]
HALLS = [("concert hall", "concert"), ("theatre", "play"), ("cinema", "film"), ("town hall", "talk")]


def at(a):
    return {p: i + 1 for i, p in enumerate(a)}


def holds(c, a):
    p = at(a)
    k = c[0]
    if k == "fixed":
        return p[c[1]] == c[2]
    if k == "no":
        return p[c[1]] != c[2]
    if k == "next":
        return abs(p[c[1]] - p[c[2]]) == 1
    if k == "offset":
        return p[c[1]] == p[c[2]] + c[3]
    if k == "end":
        return p[c[1]] in (1, len(a))
    if k == "notnext":
        return abs(p[c[1]] - p[c[2]]) != 1
    return p[c[1]] < p[c[2]]  # lower


def solutions(members, clues):
    return [a for a in permutations(members) if all(holds(c, a) for c in clues)]


def possible(f, test, names):
    """Some assignment allowed by the clues makes the claim true; a claim placing someone outside the listed people
    in the row is impossible, because the row went to exactly those people."""
    if any(x not in f.members for x in names):
        return False
    return any(test(at(a), len(a)) for a in solutions(f.members, f.clues))


def make(rng, people, others, day):
    sf = rng.choice(SURFACES)
    hall, event = rng.choice(HALLS)
    R = rng.choice("BCDEFGHJK")
    R2 = chr(ord(R) + 1)
    fmt = lambda t, **kw: t.format(R=R, R2=R2, hall=hall, event=event, **kw)  # noqa: E731
    sp = list(people)
    N = rng.choice([4, 5, 5])
    n_sp = rng.randint(2, min(4, N))
    members = rng.sample(sp, n_sp) + others[: N - n_sp]
    a = tuple(rng.sample(members, N))  # hidden: a[i] holds place i + 1
    pos = at(a)
    unique = rng.random() < 0.45
    pool = [("fixed", x, pos[x]) for x in members]
    pool += rng.sample([("no", x, k) for x in members for k in range(1, N + 1) if k != pos[x]], 3)
    pool += [("next", a[i], a[i + 1]) for i in range(N - 1)]
    pool += [("offset", x, y, pos[x] - pos[y]) for x in members for y in members if pos[x] - pos[y] in (1, 2)]
    pool += [("end", a[0]), ("end", a[-1])]
    pool += [("notnext", x, y) for x in members for y in members if pos[x] < pos[y] - 1]
    pool += rng.sample([("lower", x, y) for x in members for y in members if pos[x] < pos[y]], 3)
    rng.shuffle(pool)
    pool.sort(key=lambda c: c[0] == "fixed" and rng.random() < 0.7)  # fixed places mostly come last
    clues, count = [], len(solutions(members, []))
    for c in pool:
        new = len(solutions(members, clues + [c]))
        if new == count or (not unique and new < 2):
            continue
        clues.append(c)
        count = new
        if count == 1 or (not unique and count <= 4 and len(clues) >= 3):
            break
    f = facts(members=members, clues=clues)

    def ctext(c):
        k = c[0]
        if k in ("fixed", "no"):
            return fmt(rng.choice(sf[k]), x=c[1], k=c[2])
        if k == "offset":
            return fmt(rng.choice(sf[k]), x=c[1], y=c[2], d=word(c[3]))
        if k == "end":
            return fmt(rng.choice(sf[k]), x=c[1])
        return fmt(rng.choice(sf[k]), x=c[1], y=c[2])

    names = ", ".join(members[:-1]) + " and " + members[-1]
    ev = [fmt(sf["setup"], N=N), fmt(sf["share"], names=names)] + [ctext(c) for c in clues]

    # claim makers: (sender, text, test(positions, N), names the claim puts in the row)
    def place(s, x, k):
        text = fmt(rng.choice(sf["p_me"]), k=k) if s == x else fmt(rng.choice(sf["p"]), x=x, k=k)
        return s, text, lambda p, n: p[x] == k, (x,)

    def nxt(s, x, y):
        if s == y:
            x, y = y, x
        text = fmt(rng.choice(sf["n_me"]), y=y) if s == x else fmt(rng.choice(sf["n"]), x=x, y=y)
        return s, text, lambda p, n: abs(p[x] - p[y]) == 1, (x, y)

    def end(s, x):
        text = fmt(rng.choice(sf["e_me"])) if s == x else fmt(rng.choice(sf["e"]), x=x)
        return s, text, lambda p, n: p[x] in (1, n), (x,)

    def between(s, m, x, y):
        text = fmt(rng.choice(sf["b_me"]), x=x, y=y) if s == m else fmt(rng.choice(sf["b"]), m=m, x=x, y=y)
        return s, text, lambda p, n: abs(p[m] - p[x]) == 1 and abs(p[m] - p[y]) == 1 and p[x] != p[y], (m, x, y)

    def higher(s, x, y):
        text = fmt(rng.choice(sf["h_me"]), y=y) if s == x else fmt(rng.choice(sf["h"]), x=x, y=y)
        return s, text, lambda p, n: p[x] > p[y], (x, y)

    def key_of(spec):
        """The fact a claim is about: the sender's own place, a pair's relation, or one person's place."""
        s, _, _, nm = spec
        if s in nm:
            return ("self", s)
        if len(nm) == 2:
            return ("pair",) + tuple(sorted(nm))
        return ("place", nm[0])

    def claim(spec, tags=()):
        s, text, test, nm = spec
        return Claim(s, text, lambda f, test=test, nm=nm: possible(f, test, nm), tags, key_of(spec))

    def pick_sender(*subjects):
        mine = [x for x in subjects if x in sp]
        return rng.choice(mine) if mine and rng.random() < 0.6 else rng.choice(sp)

    truths = []

    def add(spec, tags=()):
        assert all(x in pos for x in spec[3]) and spec[2](pos, N), spec[1]
        truths.append(claim(spec, tags))

    for x in rng.sample(members, 3):
        add(place(pick_sender(x), x, pos[x]))
    for i in rng.sample(range(N - 1), 2):
        add(nxt(pick_sender(a[i], a[i + 1]), a[i], a[i + 1]))
    x = rng.choice([a[0], a[-1]])
    add(end(pick_sender(x), x))
    i = rng.randrange(1, N - 1)
    add(between(pick_sender(a[i]), a[i], a[i - 1], a[i + 1]))
    for _ in range(2):
        i, j = sorted(rng.sample(range(N), 2))
        add(higher(pick_sender(a[j]), a[j], a[i]))
    ends = [c[1] for c in clues if c[0] == "end"]
    if len(ends) == 1:  # one end is named in the evidence; the other end exists too
        y = a[-1] if ends[0] == a[0] else a[0]
        s = pick_sender(y)
        t = fmt(sf["e_too"][0]) if s == y else fmt(sf["e_too"][1], x=y)
        add((s, t, lambda p, n, y=y: p[y] in (1, n), (y,)), ("distractor",))
    gaps = [(x, y) for x in members for y in members if pos[y] - pos[x] == 2]
    if gaps and any(c[0] in ("notnext", "next") for c in clues):  # sounds like it clashes with a neighbour clue
        x, y = rng.choice(gaps)
        add((rng.choice(sp), fmt(sf["gap"][0], x=x, y=y), lambda p, n, x=x, y=y: abs(p[x] - p[y]) == 2, (x, y)),
            ("distractor",))
    outs = [x for x in sp if x not in members]
    for s in outs:
        truths.append(Claim(s, fmt(rng.choice(sf["out"])), lambda f, s=s: s not in f.members, (), ("self", s)))
    for i, s in enumerate([x for x in sp if not any(c.who == x for c in truths)] + [rng.choice(sp), rng.choice(sp)]):
        truths.append(Claim(s, fmt(sf["neutral"][i % 2], n=rng.randint(2, 9), b=rng.randint(3, 48)), lambda f: True,
                            ("neutral",)))  # every speaker gets a claim; two neutral claims at least

    def single_fact(spec):
        _, _, test, nm = spec
        if not possible(facts(members=members, clues=[]), test, nm):
            return True
        return any(not possible(facts(members=members, clues=[c]), test, nm) for c in clues)

    kinds = ["place", "place", "next", "end", "between", "higher"] + (["outsider"] if outs else [])
    rng.shuffle(kinds)
    lie = None
    for kind in kinds:
        if kind == "place":
            cands = [place(s, x, k) for x in members for k in range(1, N + 1) for s in sp]
        elif kind == "next":
            cands = [nxt(s, x, y) for x in members for y in members if x < y for s in sp]
        elif kind == "end":
            cands = [end(s, x) for x in members for s in sp]
        elif kind == "between":
            cands = [between(s, m, x, y) for m in members for x in members for y in members
                     if len({m, x, y}) == 3 and x < y for s in sp]
        elif kind == "higher":
            cands = [higher(s, x, y) for x in members for y in members if x != y for s in sp]
        else:  # someone who was not given a place in the row claims one
            cands = [place(o, o, k) for o in outs for k in range(1, N + 1)]
            cands += [place(s, o, k) for o in outs for s in sp if s != o for k in range(1, N + 1)]
        cands = [c for c in cands if not possible(f, c[2], c[3])]
        if cands:
            mine = [c for c in cands if c[0] in c[3]]
            if mine and rng.random() < 0.5:
                cands = mine
            if kind != "outsider":
                two = [c for c in cands if not single_fact(c)]
                if two and rng.random() < 0.5:
                    cands = two
            spec = rng.choice(cands)
            tags = () if spec[0] in spec[3] else ("about-other",)
            if kind != "outsider" and not single_fact(spec):
                tags = ("two-fact",) + tags
            lie = claim(spec, tags)
            break
    assert lie is not None

    g = sf["group"]
    return Bundle(NAME, fmt(rng.choice(sf["scen"]), g=g), g, fmt(sf["topic"]), ev, f, lie, truths,
                  fillers=[fmt(x) for x in sf["fill"]], decor=[fmt(x) for x in sf["decor"]], timed=False)
