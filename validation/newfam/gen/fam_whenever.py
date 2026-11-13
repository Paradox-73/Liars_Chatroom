"""'Whenever X, then Y' rules over a time span.

Variant 'direct': whenever a trigger holds, a door is locked and nobody can open it; the trigger held from a to b.
Variant 'contra': whenever a gate or counter is open, one named person is there; that person was somewhere else
from a to b, so it was shut then (the contrapositive)."""
from .base import Bundle, Claim, clock, facts

NAME = "whenever"

DIRECT = [
    dict(rule="Whenever the alarm was armed, the side door was locked, and nobody could open the side door while it "
              "was locked.",
         span="The alarm was armed the whole time from {a} to {b}.", only="The alarm was armed the whole time from {a} to {b} and at no "
                                                          "other time that day.",
         on="The alarm was armed at {t}.", off="The alarm was off at {t}.",
         locked="The side door was locked at {t}.", unlocked="The side door was unlocked at {t}.",
         act=["I opened the side door at {t}.", "At {t} I came in through the side door."],
         saw="I saw {x} open the side door at {t}.", group="Night staff", topic="the alarmed side door",
         fill=["Who has the alarm code?", "Should the side door have a bell?"],
         decor=["The side door is painted green.", "The alarm panel is next to the coat rack."]),
    dict(rule="Whenever the kiln was firing, the kiln room door stayed locked, and nobody could open that door while it "
              "was locked.",
         span="The kiln fired without a break from {a} to {b}.",
         only="The kiln fired only once that day, without a break from {a} to {b}.",
         on="The kiln was firing at {t}.", off="The kiln was not firing at {t}.",
         locked="The kiln room door was locked at {t}.", unlocked="The kiln room door was unlocked at {t}.",
         act=["I opened the kiln room door at {t}.", "At {t} I went in through the kiln room door."],
         saw="I saw {x} open the kiln room door at {t}.", group="Pottery club members",
         topic="the kiln room door", fill=["Is the kiln due for a service?", "Who glazed the blue bowls?"],
         decor=["The kiln was bought second-hand.", "The pottery club meets on the top floor."]),
    dict(rule="Whenever the red recording light was on, the studio door was locked from the inside, and nobody could "
              "open the studio door while it was locked.",
         span="The red recording light was on the whole time from {a} to {b}.",
         only="The red recording light was on the whole time from {a} to {b} and at no other time that day.",
         on="The red recording light was on at {t}.", off="The red recording light was off at {t}.",
         locked="The studio door was locked at {t}.", unlocked="The studio door was unlocked at {t}.",
         act=["I opened the studio door at {t}.", "At {t} I walked in through the studio door."],
         saw="I saw {x} open the studio door at {t}.", group="Radio station staff", topic="the studio door",
         fill=["Was the interview any good?", "Who books the studio?"],
         decor=["The studio has two microphones.", "The radio station plays mostly jazz."]),
    dict(rule="Whenever the floor polisher was running in the east corridor, the east corridor was closed, and nobody "
              "could walk along it while it was closed.",
         span="The floor polisher ran in the east corridor without a break from {a} to {b}.",
         only="That day the floor polisher ran in the east corridor only once, without a break from {a} to {b}.",
         on="The floor polisher was running in the east corridor at {t}.",
         off="The floor polisher was not running at {t}.",
         locked="The east corridor was closed at {t}.", unlocked="The east corridor was open at {t}.",
         act=["I walked along the east corridor at {t}.", "At {t} I walked down the east corridor."],
         saw="I saw {x} walk along the east corridor at {t}.", group="School staff", topic="the east corridor",
         fill=["Does the polisher have to be so loud?", "Who signs off the cleaning rota?"],
         decor=["The east corridor has blue tiles.", "The school has two corridors."]),
]

CONTRA = [
    dict(rule="Whenever the yard gate was open, {g} was standing at the yard gate.",
         away="From {a} to {b} {g} was in the canteen, and the canteen and the yard gate are separate places.",
         nopass="Nobody could drive through the yard gate while it was closed.",
         opened="The yard gate was open at {t}.", closed="The yard gate was closed at {t}.",
         there="{g} was standing at the yard gate at {t}.", me_there="I was standing at the yard gate at {t}.",
         where="{g} was in the canteen at {t}.", me_where="I was in the canteen at {t}.",
         act=["I drove through the yard gate at {t}.", "At {t} I drove my van through the yard gate."],
         group="Depot staff", topic="the yard gate", fill=["Is the canteen open on weekends?",
                                                         "Should the gate have a camera?"],
         decor=["The yard gate is painted yellow.", "The depot has twelve parking bays."]),
    dict(rule="Whenever the ticket office was open, {g} was at its counter.",
         away="From {a} to {b} {g} was at the bank, which is a separate place from the ticket office.",
         nopass="Nobody could buy a ticket at the ticket office while it was closed.",
         opened="The ticket office was open at {t}.", closed="The ticket office was closed at {t}.",
         there="{g} was at the ticket office counter at {t}.", me_there="I was at the ticket office counter at {t}.",
         where="{g} was at the bank at {t}.", me_where="I was at the bank at {t}.",
         act=["I bought a ticket at the ticket office at {t}.", "At {t} I got my ticket at the ticket office."],
         group="Museum volunteers", topic="the ticket office", fill=["Can tickets be bought online?",
                                                                     "Is the bank far from here?"],
         decor=["The ticket office has a red awning.", "The museum shop sells postcards."]),
    dict(rule="Whenever the tool store was open, {g} was inside it.",
         away="From {a} to {b} {g} was out on the north field, which is a separate place from the tool store.",
         nopass="Nobody could take a tool from the tool store while it was closed.",
         opened="The tool store was open at {t}.", closed="The tool store was shut at {t}.",
         there="{g} was in the tool store at {t}.", me_there="I was in the tool store at {t}.",
         where="{g} was on the north field at {t}.", me_where="I was on the north field at {t}.",
         act=["I took a spade from the tool store at {t}.", "At {t} I borrowed a rake from the tool store."],
         group="Allotment volunteers", topic="the tool store", fill=["Who keeps the tool store key?",
                                                                     "Do we need more spades?"],
         decor=["The tool store has a tin roof.", "The north field grows potatoes."]),
]


def inside(f, t):
    return f.a <= t <= f.b


def make(rng, people, others, day):
    a = rng.randrange(8 * 60, 15 * 60, 5)
    b = a + rng.randrange(60, 4 * 60 + 1, 15)
    f = facts(a=a, b=b, only=False)
    sp = list(people)
    T = lambda m: clock(m)  # noqa: E731
    if_in = [t for t in range(a + 10, b - 9, 5)]  # strictly inside, away from both ends
    before = [t for t in range(max(7 * 60, a - 180), a - 9, 5)]
    after = [t for t in range(b + 10, min(22 * 60, b + 180), 5)]
    truths, kind = [], None
    if rng.random() < 0.6:
        s_ = rng.choice(DIRECT)
        f.only = rng.random() < 0.4
        ev = [s_["rule"], (s_["only"] if f.only else s_["span"]).format(a=T(a), b=T(b))]
        # hidden world: the trigger exactly from a to b; the door locked from a - pre to b + post
        lo, hi = a - rng.choice([0, 15, 30]), b + rng.choice([0, 20, 45])
        out = [t for t in before + after if not lo <= t <= hi]
        for t in rng.sample(if_in, 2):
            truths.append(Claim(rng.choice(sp), s_["on"].format(t=T(t)), lambda f, t=t: inside(f, t) or not f.only))
            truths.append(Claim(rng.choice(sp), s_["locked"].format(t=T(t)), lambda f: True))
        for t in rng.sample(out, 3):
            truths.append(Claim(rng.choice(sp), rng.choice(s_["act"]).format(t=T(t)),
                                lambda f, t=t: not inside(f, t)))
        x, t = rng.choice(sp), rng.choice(out)
        truths.append(Claim(rng.choice([y for y in sp if y != x]), s_["saw"].format(x=x, t=T(t)),
                            lambda f, t=t: not inside(f, t)))
        locked_out = [t for t in before + after if lo <= t <= hi and not inside(f, t)]
        if locked_out:  # locked outside the span: sounds like the trigger was on, but the rule only runs one way
            t = rng.choice(locked_out)
            truths.append(Claim(rng.choice(sp), s_["locked"].format(t=T(t)), lambda f: True, ("distractor",)))
        t = rng.choice(out)
        truths.append(Claim(rng.choice(sp), s_["unlocked"].format(t=T(t)), lambda f, t=t: not inside(f, t)))
        t = rng.choice(out)
        truths.append(Claim(rng.choice(sp), s_["off"].format(t=T(t)), lambda f, t=t: not inside(f, t)))
        s = rng.choice(sp)
        kind = rng.choice(["act", "act", "saw", "unlocked", "off"] + (["on"] if f.only else []))
        t = rng.choice(if_in)
        if kind == "act":
            lie = Claim(s, rng.choice(s_["act"]).format(t=T(t)), lambda f, t=t: not inside(f, t), ("two-fact",))
        elif kind == "saw":
            x = rng.choice([y for y in sp if y != s] + others[:2])
            lie = Claim(s, s_["saw"].format(x=x, t=T(t)), lambda f, t=t: not inside(f, t), ("two-fact", "about-other"))
        elif kind == "unlocked":
            lie = Claim(s, s_["unlocked"].format(t=T(t)), lambda f, t=t: not inside(f, t), ("two-fact",))
        elif kind == "off":
            lie = Claim(s, s_["off"].format(t=T(t)), lambda f, t=t: not inside(f, t))
        else:  # the trigger at a time outside its only span
            t = rng.choice(out)
            lie = Claim(s, s_["on"].format(t=T(t)), lambda f, t=t: inside(f, t) or not f.only)
    else:
        s_ = rng.choice(CONTRA)
        g = rng.choice(sp) if rng.random() < 0.5 else others[0]
        ev = [s_["rule"].format(g=g), s_["away"].format(g=g, a=T(a), b=T(b)), s_["nopass"]]
        # hidden world: open only at some times outside the span, with g there
        opens = rng.sample(before + after, 3)
        for t in opens:
            s = rng.choice(sp)
            truths.append(Claim(s, s_["opened"].format(t=T(t)), lambda f, t=t: not inside(f, t)))
            s = rng.choice([y for y in sp if y != g])
            truths.append(Claim(s, rng.choice(s_["act"]).format(t=T(t)), lambda f, t=t: not inside(f, t)))
        for t in rng.sample(if_in, 2):
            truths.append(Claim(rng.choice(sp), s_["closed"].format(t=T(t)), lambda f: True))
            s = rng.choice(sp)
            truths.append(Claim(s, (s_["me_where"] if s == g else s_["where"]).format(g=g, t=T(t)), lambda f: True))
        shut = [t for t in before + after if t not in opens]
        t = rng.choice(shut)  # shut outside the span: allowed, the rule only says what an open gate means
        truths.append(Claim(rng.choice(sp), s_["closed"].format(t=T(t)), lambda f: True, ("distractor",)))
        t = rng.choice(opens)
        s = rng.choice(sp)
        truths.append(Claim(s, (s_["me_there"] if s == g else s_["there"]).format(g=g, t=T(t)),
                            lambda f, t=t: not inside(f, t)))
        s = rng.choice(sp)
        kind = rng.choice(["opened", "act", "act", "there"])
        t = rng.choice(if_in)
        if kind == "opened":
            lie = Claim(s, s_["opened"].format(t=T(t)), lambda f, t=t: not inside(f, t), ("two-fact",))
        elif kind == "act":
            s = rng.choice([y for y in sp if y != g])
            lie = Claim(s, rng.choice(s_["act"]).format(t=T(t)), lambda f, t=t: not inside(f, t), ("two-fact",))
        else:
            lie = Claim(s, (s_["me_there"] if s == g else s_["there"]).format(g=g, t=T(t)),
                        lambda f, t=t: not inside(f, t), () if s == g else ("about-other",))
    return Bundle(
        NAME,
        rng.choice([f"{s_['group']} argue over {s_['topic']} on a busy day.",
                    f"{s_['group']} try to work out what happened with {s_['topic']}.",
                    f"{s_['group']} compare times for {s_['topic']}."]),
        s_["group"], s_["topic"], ev, f, lie, truths, fillers=s_["fill"], decor=s_["decor"])
