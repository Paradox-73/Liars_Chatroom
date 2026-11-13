"""Age limits on one event day. Only people aged at least N on the day (or only people under N) took part, and every
date of birth was checked, so nobody outside the limit did. Birth dates are given for some people; anyone else could
be any age."""
import datetime as dt

from .base import Bundle, Claim, date_text, facts

NAME = "agelimit"
EVENTS = [
    dict(kind="min", n=18, short="screening",
         rule="Only people aged at least {n} on the day could watch the late-night horror screening at the {place} on "
              "{d}, and staff checked every viewer's date of birth at the door, so nobody younger got in.",
         places=["town cinema", "arts centre", "film club hall"],
         me=["I watched the late-night horror screening.", "I was in the audience at the late-night horror screening."],
         other=["{p} watched the late-night horror screening.", "I saw {p} walk into the late-night horror screening."],
         out="get into the late-night horror screening", group="Film club members",
         topic="the late-night horror screening",
         neutral=["I queued for {k} minutes at the box office.", "I bought {k} tubs of popcorn that evening.",
                  "I parked {k} streets away from the {place}."],
         fill=["Was the film any good?", "Who picked the film?", "Should the club show more horror films?"],
         decor=["The screening was in the smaller of the two rooms.", "The {place} has red velvet seats."]),
    dict(kind="min", n=16, short="race",
         rule="Only drivers aged at least {n} on the day could race in the go-karting race at the {place} on {d}, and "
              "the marshals checked every driver's date of birth, so nobody younger raced.",
         places=["kart track", "leisure park", "old airfield track"],
         me=["I raced in the go-karting race.", "I drove a kart in the go-karting race."],
         other=["{p} raced in the go-karting race.", "I watched {p} drive a kart in the go-karting race."],
         out="race in the go-karting race", group="Friends", topic="the go-karting race",
         neutral=["I paid for {k} practice laps the week before.", "I parked in bay {k} at the {place}.",
                  "I brought {k} bottles of water for everyone."],
         fill=["Who won the race in the end?", "Are the karts safe?", "Should we book again next year?"],
         decor=["The karts were painted orange.", "The {place} has a small café."]),
    dict(kind="min", n=21, short="course",
         rule="Only people aged at least {n} on the day could drive on the minibus driving course at the {place} on "
              "{d}, and the instructors checked every driver's date of birth, so nobody younger drove.",
         places=["college car park", "test centre", "sports ground"],
         me=["I drove on the minibus driving course.", "I took the wheel on the minibus driving course."],
         other=["{p} drove on the minibus driving course.", "I watched {p} drive on the minibus driving course."],
         out="drive on the minibus driving course", group="Club volunteers", topic="the minibus driving course",
         neutral=["I set out {k} cones in the {place}.", "I made {k} cups of tea for the instructors.",
                  "I parked in bay {k}."],
         fill=["Do we need more drivers for the club trips?", "Was the course worth the money?",
               "Who books the minibus?"],
         decor=["The minibus has sixteen seats.", "The {place} was dry all day."]),
    dict(kind="min", n=18, short="tasting",
         rule="Only people aged at least {n} on the day could join the cider tasting at the {place} on {d}, and staff "
              "checked every taster's date of birth, so nobody younger joined in.",
         places=["orchard", "farm shop", "village barn"],
         me=["I joined the cider tasting at the {place}.", "I tried the ciders at the {place} tasting."],
         other=["{p} joined the cider tasting at the {place}.", "I saw {p} taste the ciders at the {place} tasting."],
         out="join the cider tasting", group="Neighbours", topic="the cider tasting",
         neutral=["I bought {k} bottles of apple juice in the shop.", "I walked {k} kilometres round the {place}.",
                  "I parked in bay {k}."],
         fill=["Which cider was the best?", "Is the shop open all year?", "Should we go again in autumn?"],
         decor=["The {place} grows six kinds of apple.", "The tasting table stood under an old oak."]),
    dict(kind="max", n=16, short="tournament",
         rule="Only players under {n} on the day could play in the junior chess tournament at the {place} on {d}, "
              "and the organisers checked every player's date of birth, so nobody aged {n} or over played.",
         places=["library hall", "school gym", "town hall"],
         me=["I played in the junior chess tournament.", "I had a board in the junior chess tournament."],
         other=["{p} played in the junior chess tournament.", "I watched {p} play a game in the junior chess "
                                                              "tournament."],
         out="play in the junior chess tournament", group="Chess club members", topic="the junior chess tournament",
         neutral=["I brought {k} chess clocks from the club.", "I set up {k} boards before the first round.",
                  "I bought {k} sandwiches at the café."],
         fill=["Who won the junior section?", "Is there a senior tournament too?", "Should the club enter a team?"],
         decor=["The {place} was very cold that morning.", "Every board had a wooden set."]),
    dict(kind="max", n=14, short="gala",
         rule="Only swimmers under {n} on the day could swim in the junior swimming gala at the {place} on {d}, and "
              "the officials checked every swimmer's date of birth, so nobody aged {n} or over swam.",
         places=["leisure centre", "town baths", "school pool"],
         me=["I swam in the junior swimming gala.", "I raced in the junior swimming gala."],
         other=["{p} swam in the junior swimming gala.", "I watched {p} swim in the junior swimming gala."],
         out="swim in the junior swimming gala", group="Swimming club families", topic="the junior swimming gala",
         neutral=["I brought {k} towels.", "I sat in row {k} of the spectator seats.",
                  "I bought {k} cartons of juice at the café."],
         fill=["Who timed the races?", "Was the water warm enough?", "Should the gala start later next year?"],
         decor=["The {place} has six lanes.", "The gala ended with a relay."]),
]


def age_on(b, d):
    return d.year - b.year - ((d.month, d.day) < (b.month, b.day))


def birth(day, age, back):
    """A birth date that makes someone exactly `age` on `day`, with the last birthday `back` days before it."""
    b = dt.date(day.year - age, day.month, day.day) - dt.timedelta(days=back)
    if (b.month, b.day) == (2, 29):
        b -= dt.timedelta(days=1)
    assert age_on(b, day) == age
    return b


def allowed(f, x):
    b = f.born.get(x)
    if b is None:  # no birth date in the evidence: any age is possible
        return True
    return age_on(b, f.day) >= f.n if f.kind == "min" else age_on(b, f.day) < f.n


def ok_out(f, x):
    return x not in f.born or not allowed(f, x)


def ok_age(f, x, a):
    return x not in f.born or age_on(f.born[x], f.day) == a


def birthday(f, x, k):
    b = f.born[x]
    return b.replace(year=b.year + k)


def make(rng, people, others, day):
    e = rng.choice(EVENTS)
    n, kind, E, D = e["n"], e["kind"], day, date_text
    place = rng.choice(e["places"])
    fmt = lambda s, **kw: s.format(place=place, **kw)  # noqa: E731
    young = "young" if kind == "min" else "old"
    sp = list(people)
    lie_kind = rng.choice(["underage_self", "underage_other", "wrong_age", "excluded", "turned_side"])
    pool = rng.sample(sp, 4)
    inel = pool.pop() if lie_kind == "underage_self" or rng.random() < 0.5 else others[0]
    elig = [pool.pop()] + ([others[1]] if rng.random() < 0.5 else [])
    bd = None
    if rng.random() < 0.45:
        bd = pool.pop() if rng.random() < 0.6 else others[2]
    unknown = pool  # speakers with no birth date in the evidence
    lo, hi = (n, n + 20) if kind == "min" else (n - 5, n - 1)
    born = {x: birth(E, rng.randint(lo, hi), rng.randint(1, 360)) for x in elig}
    # outside the limit, but close to it: turns n a few weeks after the day, or turned n a few weeks before it
    born[inel] = birth(E, n - 1, rng.randint(305, 355)) if kind == "min" else birth(E, n, rng.randint(1, 60))
    if bd:  # on the limit: turns n on the day (min), or turns n the day after (max)
        b = dt.date(E.year - n, E.month, E.day) + (dt.timedelta(days=0) if kind == "min" else dt.timedelta(days=1))
        if (b.month, b.day) == (2, 29):
            bd = None
        else:
            born[bd] = b
    f = facts(kind=kind, n=n, day=E, born=born)
    ev = [fmt(e["rule"], n=n, d=D(E))]
    births = [f"{x} was born on {D(b)}." for x, b in born.items()]
    rng.shuffle(births)
    ev += births

    def part(s, x):
        return fmt(rng.choice(e["me"])) if s == x else fmt(rng.choice(e["other"]), p=x)

    def name(s, x):
        return "I" if s == x else x

    # the lie is picked first, so the liar never also states a truth about the lie's subject
    subj = (inel if lie_kind in ("underage_self", "underage_other", "turned_side") else
            rng.choice(list(born)) if lie_kind == "wrong_age" else rng.choice(elig + ([bd] if bd else [])))
    liar = (inel if lie_kind == "underage_self" else
            rng.choice([y for y in sp if y != inel]) if lie_kind == "underage_other" else rng.choice(sp))

    def snd(x, self_ok=False):
        pool_ = [y for y in sp if y != liar] if x == subj else sp
        return x if self_ok and x in pool_ and rng.random() < 0.5 else rng.choice(pool_)

    truths = []
    attendees = elig + ([bd] if bd else []) + rng.sample(unknown, min(len(unknown), 2))
    for x in attendees:
        s = snd(x, self_ok=True)
        tags = ("distractor",) if x in unknown else ("boundary",) if x == bd else ()
        truths.append(Claim(s, part(s, x), lambda f, x=x: allowed(f, x), tags))
    for x, b in born.items():
        s, a = snd(x), age_on(b, E)
        text = (f"I was {a} on the day of the {e['short']}." if s == x else
                rng.choice([f"{x} was {a} on the day of the {e['short']}.", f"On {D(E)} {x} was {a}."]))
        truths.append(Claim(s, text, lambda f, x=x, a=a: ok_age(f, x, a)))
    s = snd(inel)
    truths.append(Claim(s, f"{name(s, inel)} was too {young} to {e['out']}.", lambda f: ok_out(f, inel)))
    x = rng.choice(elig)
    s, k = snd(x), age_on(born[x], E)
    truths.append(Claim(s, f"{name(s, x)} turned {k} on {D(born[x].replace(year=born[x].year + k))}.",
                        lambda f, x=x, k=k, d=born[x].replace(year=born[x].year + k): birthday(f, x, k) == d))
    s = snd(x)
    truths.append(Claim(s, f"{name(s, x)} was old enough to {e['out']}." if kind == "min" else
                        f"{name(s, x)} was young enough to {e['out']}.", lambda f, x=x: allowed(f, x)))
    if bd:
        s = snd(bd)
        if kind == "min":
            truths.append(Claim(s, f"{name(s, bd)} turned {n} on the day of the {e['short']}.",
                                lambda f: birthday(f, bd, n) == f.day, ("boundary",)))
        else:
            truths.append(Claim(s, f"{name(s, bd)} turned {n} the day after the {e['short']}.",
                                lambda f: birthday(f, bd, n) == f.day + dt.timedelta(days=1), ("boundary", "distractor")))
    for tpl in e["neutral"]:
        truths.append(Claim(rng.choice(sp), fmt(tpl, k=rng.randint(2, 9)), lambda f: True, ("neutral",)))

    if lie_kind == "underage_self":  # inel is a speaker and claims to have taken part
        lie = Claim(inel, fmt(rng.choice(e["me"])), lambda f: allowed(f, inel), ("two-fact",))
    elif lie_kind == "underage_other":
        s = liar
        lie = Claim(s, fmt(rng.choice(e["other"]), p=inel), lambda f: allowed(f, inel), ("two-fact", "about-other"))
    elif lie_kind == "wrong_age":
        x, s = subj, liar
        a = age_on(born[x], E) + rng.choice([1, -1])
        text = f"I was {a} on the day of the {e['short']}." if s == x else f"{x} was {a} on the day of the {e['short']}."
        lie = Claim(s, text, lambda f, x=x, a=a: ok_age(f, x, a), ("two-fact",) + (() if s == x else ("about-other",)))
    elif lie_kind == "excluded":  # someone inside the limit said to be outside it
        x, s = subj, liar
        lie = Claim(s, f"{name(s, x)} was too {young} to {e['out']}.", lambda f, x=x: ok_out(f, x),
                    ("two-fact",) + (() if s == x else ("about-other",)))
    else:  # which side of the birthday the event fell on
        s = liar
        about = ("two-fact",) + (() if s == inel else ("about-other",))
        if kind == "min":
            lie = Claim(s, f"{name(s, inel)} had already turned {n} by the day of the {e['short']}.",
                        lambda f: birthday(f, inel, n) <= f.day, about)
        else:
            lie = Claim(s, f"{name(s, inel)} turned {n} only after the {e['short']}.",
                        lambda f: birthday(f, inel, n) > f.day, about)
    assert len(truths) >= 10, len(truths)
    return Bundle(
        NAME,
        rng.choice([f"{e['group']} argue over who was old enough for {e['topic']}.",
                    f"{e['group']} check who could take part in {e['topic']}.",
                    f"{e['group']} compare birth dates against the age limit for {e['topic']}."])
        if kind == "min" else
        rng.choice([f"{e['group']} argue over who was young enough for {e['topic']}.",
                    f"{e['group']} check who could take part in {e['topic']}.",
                    f"{e['group']} compare birth dates against the age limit for {e['topic']}."]),
        e["group"], e["topic"], ev, f, lie, truths, fillers=e["fill"], decor=[fmt(x) for x in e["decor"]], timed=False)
