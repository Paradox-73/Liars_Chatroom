"""Queue order at a counter. One person served only people from the queue, one at a time and strictly in the order
they joined it, each person once, and nobody left the queue before being served, so the order of service is the
order of joining. The evidence gives some of the joins ('Mia joined the queue before Rahul', 'Tom joined straight
after Ava', 'Ava was the first to join') and leaves other pairs open.

Variant 'complete': the evidence names everyone who queued, so places in the queue are fixed and nobody else was
served. Variant 'open': the names are not a complete list, so someone unnamed may also have queued.
Timed (complete only): serving each person took at least M minutes, and one or two start times are given.

ok() is brute force: a claim can be true iff some order of the queue (with any unnamed person the claim mentions
added, when the list is open) and start times on that order satisfy every clue and the claim."""
from itertools import permutations

from .base import ORDINALS, Bundle, Claim, clock, facts, word

NAME = "queue"

PLACES = [
    dict(place="the post office counter", clerk="the clerk", group="Customers",
         fill=["Is the post office always this busy?", "Should there be a second counter?"],
         decor=["The post office has a red door.", "There is a bench by the post office window."]),
    dict(place="the bakery counter", clerk="the baker", group="Neighbours",
         fill=["Are the rye loaves worth the wait?", "Why was it so slow this morning?"],
         decor=["The bakery sells rye bread on Fridays.", "The bakery counter has a brass bell."]),
    dict(place="the pharmacy counter", clerk="the pharmacist", group="Patients",
         fill=["Does the pharmacy take phone orders?", "Is the queue always this long?"],
         decor=["The pharmacy has a green cross sign.", "The pharmacy counter is next to the door."]),
    dict(place="the passport desk", clerk="the officer", group="Applicants",
         fill=["Do we need an appointment next time?", "Was the officer new?"],
         decor=["The passport office has a waiting area with plants.", "The passport desk has a glass screen."]),
    dict(place="the ice-cream van", clerk="the vendor", group="Beachgoers",
         fill=["Was the mint flavour any good?", "Should the van come earlier?"],
         decor=["The ice-cream van is pink.", "The van parks by the pier."]),
    dict(place="the library help desk", clerk="the librarian", group="Library users",
         fill=["Does the help desk open on Sundays?", "Was the librarian working alone?"],
         decor=["The help desk is by the main stairs.", "The library has a reading garden."]),
]
NEUTRAL = ["I had {n} errands to run that morning.", "I came on the number {b} bus.", "I live {n} streets away.",
           "I had {n} bags of shopping with me."]


def holds(c, o):
    if c[0] == "before":
        return o.index(c[1]) < o.index(c[2])
    if c[0] == "next":
        return o.index(c[2]) == o.index(c[1]) + 1
    if c[0] == "first":
        return o[0] == c[1]
    return o[-1] == c[1]  # last


def clue_text(rng, c):
    k, x = c[0], c[1]
    y = c[2] if len(c) > 2 else None
    return rng.choice({
        "before": [f"{x} joined the queue before {y}.", f"{y} joined the queue after {x}.",
                   f"{x} was already in the queue when {y} joined it."],
        "next": [f"{y} joined the queue straight after {x}, with nobody in between.",
                 f"{y} joined the queue immediately after {x}."],
        "first": [f"{x} was the first to join the queue.", f"Nobody joined the queue before {x}."],
        "last": [f"{x} was the last to join the queue.", f"Nobody joined the queue after {x}."],
    }[k])


def fits(f, o, times):
    """Start times: the ones the evidence gives plus the claimed ones; turns at least M minutes apart."""
    known = dict(f.starts)
    for x, t in times:
        if known.setdefault(x, t) != t:
            return False
    pts = sorted((o.index(x), t) for x, t in known.items())
    return all(t2 - t1 >= f.M * (p2 - p1) for (p1, t1), (p2, t2) in zip(pts, pts[1:]))


def possible(f, test, names=(), times=()):
    members = list(f.members)
    for x in names:
        if x not in members:
            if f.complete:
                return False  # the evidence lists everyone who queued
            members.append(x)
    return any(test(o) and fits(f, o, times) for o in permutations(members) if all(holds(c, o) for c in f.clues))


def absent_ok(f, s):
    return s not in f.members


def n_orders(members, clues):
    return sum(all(holds(c, o) for c in clues) for o in permutations(members))


def pick_clues(rng, order, unique):
    n = len(order)
    pool = [("before", order[i], order[j]) for i in range(n) for j in range(i + 1, n)]
    pool += [("next", order[i], order[i + 1]) for i in range(n - 1)] + [("first", order[0]), ("last", order[-1])]
    rng.shuffle(pool)
    clues, count = [], n_orders(order, [])
    for c in pool:
        new = n_orders(order, clues + [c])
        if new == count or (not unique and new < 2):
            continue
        clues.append(c)
        count = new
        if count == 1 or (not unique and count <= 4 and len(clues) >= 2):
            break
    return clues


def make(rng, people, others, day):
    pl = rng.choice(PLACES)
    place, clerk = pl["place"], pl["clerk"]
    sp = list(people)
    complete = rng.random() < 0.6
    n = rng.choice([4, 5]) if complete else rng.choice([3, 4])
    n_sp = rng.randint(2, min(4 if complete else 3, n))
    named = rng.sample(sp, n_sp) + others[: n - n_sp]
    order = rng.sample(named, n)
    clues = pick_clues(rng, order, unique=rng.random() < 0.4)
    timed = complete and rng.random() < 0.5
    M = rng.choice([5, 10, 15]) if timed else 0
    hs = {}  # hidden start of each turn
    if timed:
        t = rng.randrange(8 * 60 + 30, 11 * 60 + 31, 5)
        for x in order:
            hs[x] = t
            t += M + 5 * rng.choice([0, 0, 1, 2, 3])
    starts = {x: hs[x] for x in rng.sample(order, rng.choice([1, 2]))} if timed else {}
    full = list(order)  # the hidden queue; with an open list one unnamed speaker may have queued too
    outside = [x for x in sp if x not in named]
    if not complete and outside and rng.random() < 0.7:
        u = rng.choice(outside)
        spots = [i for i in range(n + 1) if all(holds(c, full[:i] + [u] + full[i:]) for c in clues)]
        if spots:
            i = rng.choice(spots)
            full = full[:i] + [u] + full[i:]
    f = facts(members=named, complete=complete, clues=clues, M=M, starts=starts)

    listed = ", ".join(named[:-1]) + " and " + named[-1]
    if complete:
        ev = [rng.choice([f"That morning exactly {word(n)} people queued at {place}: {listed}.",
                          f"That morning {listed} were the only people who queued at {place}."])]
    else:
        ev = [rng.choice([f"That morning {listed} all queued at {place}.",
                          f"{listed} were among the people who queued at {place} that morning."])]
    ev.append(f"{clerk.capitalize()} served only people from that queue, one at a time and strictly in the order they "
              f"joined it.")
    ev.append("Each person in the queue was served exactly once, and nobody left the queue before being served.")
    ev += [clue_text(rng, c) for c in clues]
    if timed:
        ev.append(f"Serving each person took {clerk} at least {M} minutes.")
        ev += [f"{x}'s turn at {place} began at {clock(t)}." for x, t in starts.items()]

    def ordw(k, size):
        return "last" if k == size and rng.random() < 0.5 else ORDINALS[k]

    # claim makers: (sender, text, test on an order, names the claim puts in the queue, claimed start times)
    def before(s, x, y):
        if s == x:
            text = rng.choice([f"I was served before {y}.", f"I got served ahead of {y}.", f"{y} was served after me."])
        elif s == y:
            text = rng.choice([f"I was served after {x}.", f"{x} got served before me.", f"{x} was served ahead of me."])
        else:
            text = rng.choice([f"{x} was served before {y}.", f"{y} got served after {x}.",
                               f"I watched {x} get served before {y}."])
        return s, text, lambda o: o.index(x) < o.index(y), (x, y), ()

    def nxt(s, x, y):
        if s == y:
            text = rng.choice([f"I was served straight after {x}.", f"I was served right after {x}, with nobody in between."])
        elif s == x:
            text = rng.choice([f"{y} was served straight after me.", f"Right after me, {y} was served."])
        else:
            text = rng.choice([f"{y} was served straight after {x}.", f"{y} got served right after {x}."])
        return s, text, lambda o: o.index(y) == o.index(x) + 1, (x, y), ()

    def pos(s, x, k):
        if complete:
            w = ordw(k, n)
            test = lambda o: o.index(x) == k - 1  # noqa: E731
        else:  # open list: only first and last make sense
            w = "first" if k == 1 else "last"
            test = (lambda o: o[0] == x) if k == 1 else (lambda o: o[-1] == x)
        text = (rng.choice([f"I was the {w} person served.", f"I was served {w}."]) if s == x
                else rng.choice([f"{x} was the {w} person served.", f"{x} was served {w}."]))
        return s, text, test, (x,), ()

    def start(s, x, t):
        text = (f"My turn at {place} began at {clock(t)}." if s == x else
                rng.choice([f"{x}'s turn at {place} began at {clock(t)}.", f"{x} was called up to {place} at {clock(t)}."]))
        return s, text, lambda o: True, (x,), ((x, t),)

    def present(s):
        return s, rng.choice(["I was in the queue too that morning.", f"I queued at {place} that morning as well."]), \
            lambda o: True, (s,), ()

    def key_of(spec):
        """The fact a claim is about: the sender's own place, a pair's order, or one person's place or start."""
        s, _, _, names, times = spec
        if s in names:
            return ("self", s)
        if len(names) == 2:
            return ("pair",) + tuple(sorted(names))
        return ("start" if times else "pos", names[0])

    def claim(spec, tags=()):
        s, text, test, names, times = spec
        return Claim(s, text, lambda f, test=test, names=names, times=times: possible(f, test, names, times), tags,
                     key_of(spec))

    def hidden_true(spec):
        _, _, test, names, times = spec
        return all(x in full for x in names) and test(tuple(full)) and all(hs.get(x) == t for x, t in times)

    def pick_sender(*subjects):
        mine = [x for x in subjects if x in sp]
        return rng.choice(mine) if mine and rng.random() < 0.6 else rng.choice(sp)

    truths = []

    def add(spec, tags=()):
        assert hidden_true(spec), spec[1]
        truths.append(claim(spec, tags))

    m = len(full)
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    for i, j in rng.sample(pairs, min(6, len(pairs))):
        add(before(pick_sender(full[i], full[j]), full[i], full[j]))
    for i in rng.sample(range(m - 1), min(2, m - 1)):
        add(nxt(pick_sender(full[i], full[i + 1]), full[i], full[i + 1]))
    if complete:
        for k in rng.sample(range(1, m + 1), 2):
            add(pos(pick_sender(full[k - 1]), full[k - 1], k))
        for s in outside:
            truths.append(Claim(s, rng.choice(["I wasn't in the queue at all that morning.",
                                               f"I didn't queue at {place} that morning."]),
                                lambda f, s=s: absent_ok(f, s), (), ("self", s)))
    else:
        add(pos(pick_sender(full[0]), full[0], 1))
        add(pos(pick_sender(full[-1]), full[-1], m))
        for u in [x for x in full if x not in named]:  # an unnamed speaker who queued: the list was not complete
            add(present(u), ("distractor",))
            later = [y for y in full[full.index(u) + 1:]]
            if later:
                add(before(u, u, rng.choice(later)), ("distractor",))
        for s in [x for x in outside if x not in full]:
            truths.append(Claim(s, rng.choice(["I wasn't in the queue myself.", f"I didn't queue at {place} that morning."]),
                                lambda f, s=s: absent_ok(f, s), (), ("self", s)))
    if timed:
        for x in rng.sample([x for x in order if x not in starts], min(2, n - len(starts))):
            add(start(pick_sender(x), x, hs[x]))
        for x in starts:
            i = order.index(x)
            if i + 1 < n:
                y = order[i + 1]
                gap = hs[y] - hs[x]
                s = pick_sender(y)
                if gap == M:  # exactly the minimum: allowed, "at least" includes it
                    text = (f"My turn began exactly {M} minutes after {x}'s." if s == y
                            else f"{y}'s turn began exactly {M} minutes after {x}'s.")
                    add((s, text, lambda o: True, (y,), ((y, hs[y]),)), ("boundary",))
            for y in order[i + 1:]:
                gap = hs[y] - hs[x]
                if gap >= 2 * M + 10:  # long wait: no upper limit on service, so possible
                    s = pick_sender(y)
                    text = (f"My turn began a full {gap} minutes after {x}'s." if s == y
                            else f"{y}'s turn began a full {gap} minutes after {x}'s.")
                    add((s, text, lambda o: True, (y,), ((y, hs[y]),)), ("distractor",))
                    break
    neutral = rng.sample(NEUTRAL, len(NEUTRAL))
    extra = [rng.choice(sp) for _ in range(max(2, 12 - len(truths)))]  # small queues give few order claims
    for i, s in enumerate([x for x in sp if not any(c.who == x for c in truths)] + extra):
        truths.append(Claim(s, neutral[i % len(neutral)].format(n=rng.randint(2, 9), b=rng.randint(3, 48)),
                            lambda f: True, ("neutral",)))  # every speaker gets a claim

    # lies: every candidate of a kind that no order allows; tagged two-fact when no single clue rules it out alone
    def single_fact(spec):
        _, _, test, names, times = spec
        bare = facts(members=named, complete=complete, clues=[], M=M, starts=starts)
        if not possible(bare, test, names, times):
            return True
        return any(not possible(facts(members=named, complete=complete, clues=[c], M=M, starts=starts),
                                test, names, times) for c in clues)

    everyone = named + [x for x in sp if x not in named]
    kinds = ["order", "order", "next"] + (["position", "outsider"] if complete else ["firstlast"])
    if timed:
        kinds += ["time", "time", "time-direct"]
    rng.shuffle(kinds)
    lie = None
    for kind in kinds:
        cands = []
        if kind == "order":
            cands = [before(s, x, y) for x in everyone for y in everyone if x != y for s in sp
                     if (complete and x in named and y in named) or (not complete and (x in named or x == s)
                                                                    and (y in named or y == s))]
        elif kind == "next":
            cands = [nxt(s, x, y) for x in named for y in named if x != y for s in sp]
        elif kind == "position":
            cands = [pos(s, x, k) for x in named for k in range(1, n + 1) for s in sp]
        elif kind == "firstlast":
            cands = [pos(s, x, k) for x in named for k in (1, n) for s in sp]
        elif kind == "outsider":
            outs = [x for x in sp if x not in named]
            cands = [before(o, o, y) for o in outs for y in named] + [nxt(o, y, o) for o in outs for y in named]
            cands += [present(o) for o in outs]
        elif kind == "time":
            cands = [start(s, x, t) for x in order if x not in starts for s in sp
                     for t in range(max(5 * 60, hs[x] - 3 * M), hs[x] + 3 * M + 1, 5)]
        elif kind == "time-direct":
            cands = [start(s, x, starts[x] + d) for x in starts for s in sp for d in (-15, -10, -5, 5, 10, 15)]
        cands = [c for c in cands if not possible(f, c[2], c[3], c[4])]
        if cands:
            mine = [c for c in cands if c[0] in c[3]]
            if mine and rng.random() < 0.5:
                cands = mine
            if kind in ("order", "next", "position", "firstlast"):
                two = [c for c in cands if not single_fact(c)]
                if two and rng.random() < 0.5:
                    cands = two
            spec = rng.choice(cands)
            s, text, _, names, times = spec
            tags = () if s in names else ("about-other",)
            if kind == "time" or (kind not in ("outsider", "time-direct") and not single_fact(spec)):
                tags = ("two-fact",) + tags
            lie = claim(spec, tags)
            break
    assert lie is not None

    group = pl["group"]
    return Bundle(
        NAME,
        rng.choice([f"{group} argue over the order they were served at {place}.",
                    f"{group} try to work out who was served first at {place}.",
                    f"{group} compare notes on the queue at {place}."]),
        group, f"the queue at {place}", ev, f, lie, truths, fillers=pl["fill"] + ["Is the queue always this slow?"],
        decor=pl["decor"], timed=timed)
