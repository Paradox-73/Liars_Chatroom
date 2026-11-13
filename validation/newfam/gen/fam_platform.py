"""A train with a platform change and a delay. The evidence gives the timetabled platform and time, the platform the
train actually stood at (it never stopped at the timetabled one), when it pulled in, exactly how late it left, and a
non-stop journey of fixed length, so the departure and the arrival are known. Sometimes a second train left on time
from the timetabled platform, which makes 'I caught a train at that platform' true for the other train."""
from .base import Bundle, Claim, clock, duration, facts

NAME = "platform"
STATIONS = ["Millbrook", "Eastfield", "Kingsway", "Northgate", "Riverside", "Elm Park", "Harbour Street", "Ashby Road",
            "Westcombe", "Fernhill", "Stonebridge", "Oakvale"]


def can_stand(f, p, t):
    return p in f.stood and f.t_in <= t <= f.dep


def make(rng, people, others, day):
    origin, dest, dest2 = rng.sample(STATIONS, 3)
    p1, p2 = rng.sample(range(1, 13), 2)
    sched = rng.randrange(6 * 60 + 30, 19 * 60, 5)
    delay = rng.choice([5, 10, 12, 15, 18, 20, 25, 30, 40])
    dep = sched + delay
    t_in = dep - rng.choice([3, 5, 8, 10, 12])
    jt = rng.choice([25, 35, 40, 48, 55, 65, 80, 95])
    arr = dep + jt
    t2 = None
    if rng.random() < 0.5:
        t2 = sched + rng.choice([d for d in (-40, -30, -20, -10, 10, 20, 30, 40) if d != delay])
    f = facts(sched=sched, p1=p1, p2=p2, stood={p2}, t_in=t_in, dep=dep, arr=arr, delay=delay, jt=jt, t2=t2)
    sp = list(people)
    rider, waiter, rider2, _ = rng.sample(sp, 4)

    def name():
        return rng.choice([f"the {clock(sched)} to {dest}", f"the {clock(sched)} train to {dest}"])

    def from_():  # arrival claims name the train by its origin, so the destination is not said twice
        return rng.choice([f"the {clock(sched)} from {origin}", f"the {clock(sched)} train from {origin}"])

    def cap(s):
        return s[0].upper() + s[1:]

    truths = []

    def add(s, text, ok, tags=()):
        truths.append(Claim(s, text, ok, tags))

    add(rng.choice(sp), cap(f"{name()} was timetabled to leave from platform {p1}."), lambda f: f.p1 == p1)
    add(rng.choice(sp), cap(f"{name()} left from platform {p2}."), lambda f: f.p2 == p2)
    add(rng.choice(sp), rng.choice([cap(f"{name()} left at {clock(dep)}."),
                                    cap(f"{name()} pulled out of platform {p2} at {clock(dep)}.")]),
        lambda f: f.dep == dep and f.p2 == p2)
    add(rng.choice(sp), rng.choice([cap(f"{from_()} got into {dest} at {clock(arr)}."),
                                    cap(f"{from_()} reached {dest} at {clock(arr)}.")]), lambda f: f.arr == arr)
    tb = rng.randrange(t_in + 1, dep)
    add(rider, f"I got on {name()} at platform {p2} at {clock(tb)}.", lambda f, t=tb: can_stand(f, p2, t))
    add(rider, f"I got off {from_()} in {dest} at {clock(arr)}.", lambda f: arr >= f.arr)
    add(waiter, f"I waited on platform {p1} at {clock(sched)}, but {name()} never came to that platform.",
        lambda f: p1 not in f.stood, ("distractor",))
    tw = rng.randrange(max(t_in, sched) + 1, dep)
    add(waiter, f"In the end I boarded {name()} at platform {p2} at {clock(tw)}.", lambda f, t=tw: can_stand(f, p2, t))
    ts = rng.randrange(t_in + 1, dep)
    add(rng.choice(sp), f"At {clock(ts)} {name()} was standing at platform {p2}.", lambda f, t=ts: can_stand(f, p2, t))
    add(rng.choice(sp), cap(f"{name()} never stopped at platform {p1}."), lambda f: p1 not in f.stood, ("distractor",))
    add(rng.choice(sp), cap(f"{name()} left between {clock(sched)} and {clock(dep)}."),
        lambda f: sched <= f.dep <= dep, ("boundary",))
    add(rng.choice(sp), cap(f"{name()} left at least {delay} minutes after its timetabled time."),
        lambda f: f.dep - f.sched >= delay, ("boundary",))
    add(rng.choice(sp), cap(f"{name()} took {duration(jt)} to get from {origin} to {dest}."), lambda f: f.jt == jt)
    if t2 is not None:
        add(rider2, f"I caught the {clock(t2)} to {dest2} from platform {p1}.",
            lambda f: f.t2 == t2 and f.p1 == p1, ("distractor",))
    for _ in range(3):
        t = clock(rng.randrange(6 * 60, 22 * 60, 5))
        # no places here: a rider is on the train for the whole journey
        add(rng.choice(sp), rng.choice([f"At {t} I topped up my travel card online.", f"I texted {others[0]} at {t}.",
                                        f"I checked the weather forecast at {t}."]), lambda f: True, ("neutral",))

    s = rng.choice(sp)
    x = rng.choice([y for y in sp if y != s] + others[:2])
    kind = rng.choice(["board_old", "on_time", "arr_nodelay", "still_there", "left_old", "got_off_early",
                       "wrong_delay"])
    if kind == "board_old":  # needs 'never stopped at p1' and 'board only where it stands'
        if rng.random() < 0.5:
            lie = Claim(s, f"I got on {name()} at platform {p1}.", lambda f: p1 in f.stood, ("two-fact",))
        else:
            lie = Claim(s, f"I saw {x} board {name()} at platform {p1}.", lambda f: p1 in f.stood,
                        ("two-fact", "about-other"))
    elif kind == "on_time":
        lie = Claim(s, rng.choice([cap(f"{name()} left on time at {clock(sched)}."),
                                   cap(f"{name()} left at {clock(sched)}.")]), lambda f: f.dep == sched)
    elif kind == "arr_nodelay":  # the timetabled time plus the journey, forgetting the delay
        t = sched + jt
        lie = Claim(s, cap(f"{from_()} got into {dest} at {clock(t)}."), lambda f, t=t: f.arr == t, ("two-fact",))
    elif kind == "still_there":
        t = rng.randrange(dep + 10, dep + 41, 5)
        lie = Claim(s, f"At {clock(t)} {name()} was still standing at platform {p2}.",
                    lambda f, t=t: can_stand(f, p2, t), ("two-fact",))
    elif kind == "left_old":
        lie = Claim(s, cap(f"{name()} left from platform {p1}."), lambda f: f.p2 == p1)
    elif kind == "got_off_early":
        t = rng.randrange(dep + 5, arr - 9, 5)
        if rng.random() < 0.5:
            lie = Claim(s, f"I got off {from_()} in {dest} at {clock(t)}.", lambda f, t=t: t >= f.arr, ("two-fact",))
        else:
            lie = Claim(s, f"I saw {x} get off {from_()} in {dest} at {clock(t)}.", lambda f, t=t: t >= f.arr,
                        ("two-fact", "about-other"))
    else:
        d = rng.choice([v for v in (delay - 5, delay + 5, delay + 10, delay // 2) if v > 0 and v != delay])
        lie = Claim(s, cap(f"{name()} left exactly {d} minutes late."), lambda f, d=d: f.delay == d)

    ev = [f"The {clock(sched)} train from {origin} to {dest} was timetabled to leave from platform {p1}."]
    ev += rng.choice([
        [f"That day it never stopped at platform {p1}: it pulled in to platform {p2} at {clock(t_in)} and left from "
         f"there exactly {delay} minutes late."],
        [f"On the day it pulled in to platform {p2} at {clock(t_in)} and left from platform {p2} exactly {delay} "
         f"minutes late.", f"It never stopped at platform {p1} that day."],
    ])
    ev.append(f"It ran non-stop to {dest}, and the journey took exactly {duration(jt)}.")
    ev.append("Passengers could board a train only at the platform where it was standing.")
    if t2 is not None:
        ev.append(f"The {clock(t2)} train from {origin} to {dest2} left on time from platform {p1}.")
    group = rng.choice(["Commuters", "Passengers", "Travellers"])
    return Bundle(
        NAME,
        rng.choice([f"{group} argue over the delayed train to {dest}.",
                    f"{group} compare notes on the {clock(sched)} train from {origin}.",
                    f"{group} try to work out where and when the train to {dest} left."]),
        group, f"the delayed train to {dest}", ev, f, lie, truths,
        fillers=["Why do they keep changing platforms?", "Is there a waiting room at the station?",
                 "Should we complain about the delay?", "Did anyone get a refund?",
                 "The announcements here are hard to hear."],
        decor=[f"{origin} station has a small café.", "The station clock hangs above the ticket hall.",
               f"The line to {dest} runs along the river."])
