"""Temperature thresholds with an automatic action. Every thermometer reading that day was accurate.

Variant 'alarm': a cold store alarm sounds exactly when the store is above T, and its complete log shows one unbroken
span, so the store was above T exactly then. Variant 'heater': a greenhouse heater is on exactly when the greenhouse is
below T, and it was on only during one span. Variant 'fan': a fan always runs when a room is warmer than T, and it did
not run during one span, so the room was at most T then; outside the span the fan and the temperature are free."""
from .base import Bundle, Claim, clock, facts

NAME = "temperature"
SETUPS = {
    "alarm": dict(
        places=["cold store", "walk-in fridge", "vaccine fridge", "dairy cold room"], T=[4, 5, 6, 8],
        inside=(1, 4), outside=(-4, 0), on="The alarm was sounding at {t}.", off="The alarm was silent at {t}.",
        groups=["Kitchen staff", "Shop staff", "Clinic staff"],
        fill=["Should the alarm be louder?", "Is the fridge due for a service?", "Who checks the thermostat?",
              "That alarm is far too sensitive."],
        decor=["The {place} has a steel door.", "The kitchen has two sinks."]),
    "heater": dict(
        places=["greenhouse", "orchid house", "seedling tunnel", "palm house"], T=[8, 10, 12, 15],
        inside=(-5, -1), outside=(0, 8), on="The heater was on at {t}.", off="The heater was off at {t}.",
        groups=["Gardeners", "Volunteers", "Nursery staff"],
        fill=["Should we insulate the glass?", "Who waters the seedlings?", "Is the heater expensive to run?",
              "The tomatoes look healthy this year."],
        decor=["The {place} grows tomatoes and peppers.", "The heater runs on gas."]),
    "fan": dict(
        places=["server room", "print room", "comms room", "plant room"], T=[24, 26, 27, 28, 30],
        inside=(-5, 0), outside=(-3, 4), on="The fan was running at {t}.", off="The fan was not running at {t}.",
        groups=["IT staff", "Office staff", "Technicians"],
        fill=["Should we get air conditioning?", "Who last cleaned the fan?", "Is that room always this noisy?",
              "That fan is far too loud."],
        decor=["The {place} has a raised floor.", "The fan was fitted last spring."]),
}


def can_temp(f, t, v):
    inside = f.a <= t <= f.b
    if f.kind == "alarm":
        return (v > f.T) == inside
    if f.kind == "heater":
        return (v < f.T) == inside
    return not (inside and v > f.T)


def can_device(f, t, on):
    inside = f.a <= t <= f.b
    if f.kind == "fan":
        return not (inside and on)
    return on == inside


def make(rng, people, others, day):
    kind = rng.choice(sorted(SETUPS))
    st = SETUPS[kind]
    place, T = rng.choice(st["places"]), rng.choice(st["T"])
    u = (lambda v: f"{v} degrees") if rng.random() < 0.5 else (lambda v: f"{v} °C")
    a = rng.randrange(9 * 60, 15 * 60 + 1, 5)
    b = a + rng.randrange(40, 181, 5)
    f = facts(kind=kind, T=T, a=a, b=b)
    sp = list(people)
    ins = list(range(a + 10, b - 9, 5))
    outs = list(range(7 * 60, a - 14, 5)) + list(range(b + 15, 21 * 60 + 1, 5))
    t_in, t_out = rng.sample(ins, 3), rng.sample(outs, 6)
    # hidden world: one temperature per claim time, on the right side of T for that time
    world = {t: T + rng.randint(*st["inside"]) for t in t_in}
    world.update({t: T + rng.randint(*st["outside"]) for t in t_out})

    def reading(t, v, x=None):
        if x is not None:
            return rng.choice([f"{x} read {u(v)} on the {place} thermometer at {clock(t)}.",
                               f"{x}'s reading in the {place} at {clock(t)} was {u(v)}."])
        if rng.random() < 0.5:
            return rng.choice([f"I read {u(v)} on the {place} thermometer at {clock(t)}.",
                               f"When I checked at {clock(t)}, the {place} was at {u(v)}."])
        return rng.choice([f"The {place} was at {u(v)} at {clock(t)}.",
                           f"At {clock(t)} the thermometer in the {place} read {u(v)}."])

    truths = []
    for t in t_in + t_out[:4]:
        s = rng.choice(sp)
        x = rng.choice([y for y in sp if y != s] + others[:2]) if rng.random() < 0.25 else None
        truths.append(Claim(s, reading(t, world[t], x), lambda f, t=t, v=world[t]: can_temp(f, t, v)))
    t = rng.choice(t_in)
    truths.append(Claim(rng.choice(sp), st["off" if kind == "fan" else "on"].format(t=clock(t)),
                        lambda f, t=t, on=kind != "fan": can_device(f, t, on)))
    t = rng.choice(t_out[:4])
    on_out = kind == "fan" and world[t] > T
    truths.append(Claim(rng.choice(sp), st["on" if on_out else "off"].format(t=clock(t)),
                        lambda f, t=t, on=on_out: can_device(f, t, on)))
    tb = t_out[4]
    if kind == "alarm":  # rising up to exactly T, outside the alarm span: sounds close to an alarm, but allowed
        side = [x for x in outs if (x < a) == (tb < a) and x < tb - 20 and x not in world]
        if not side or rng.random() < 0.4:  # exactly T is not above T, so the alarm stays silent
            truths.append(Claim(rng.choice(sp), f"At {clock(tb)} the alarm was silent even though the {place} was "
                                                f"at {u(T)}.",
                                lambda f: can_device(f, tb, False) and can_temp(f, tb, T), ("boundary", "distractor")))
        else:
            t0 = rng.choice(side)
            v0 = T - rng.randint(2, 4)
            truths.append(Claim(rng.choice(sp), f"The {place} went from {u(v0)} at {clock(t0)} up to {u(T)} at "
                                                f"{clock(tb)}.",
                                lambda f, t0=t0, v0=v0: can_temp(f, t0, v0) and can_temp(f, tb, T),
                                ("boundary", "distractor")))
    elif kind == "heater":  # exactly T is not below T, so the heater stays off
        truths.append(Claim(rng.choice(sp), f"At {clock(tb)} the heater was off even though the {place} was only "
                                            f"at {u(T)}.",
                            lambda f: can_device(f, tb, False) and can_temp(f, tb, T), ("boundary", "distractor")))
    else:  # the rule only says what heat forces; the fan may run when it is cool
        v = T - rng.randint(1, 3)
        truths.append(Claim(rng.choice(sp), f"At {clock(tb)} the fan was running even though the {place} was only "
                                            f"at {u(v)}.",
                            lambda f, v=v: can_device(f, tb, True) and can_temp(f, tb, v), ("distractor",)))
        t = rng.choice([x for x in ins if x not in t_in])
        truths.append(Claim(rng.choice(sp), reading(t, T), lambda f, t=t: can_temp(f, t, T), ("boundary",)))
    if kind != "fan":  # exactly T on the allowed side of the threshold
        t = t_out[5]
        truths.append(Claim(rng.choice(sp), reading(t, T), lambda f, t=t: can_temp(f, t, T), ("boundary",)))
    for _ in range(3):
        t = clock(rng.randrange(7 * 60, 21 * 60, 5))
        truths.append(Claim(rng.choice(sp), rng.choice([
            f"I restocked the shelves at {t}.", f"At {t} I signed for a delivery.", f"I swept the floor at {t}.",
            f"I made a cup of tea at {t}."]), lambda f: True, ("neutral",)))

    s = rng.choice(sp)
    x = rng.choice([y for y in sp if y != s] + others[:2])
    if kind == "alarm":
        kinds = {"hot_outside": (outs, (2, 5)), "cold_inside": (ins, (-3, -1))}
        devs = {"alarm_outside": (outs, True), "silent_inside": (ins, False)}
    elif kind == "heater":
        kinds = {"cold_outside": (outs, (-4, -1)), "warm_inside": (ins, (1, 5))}
        devs = {"heater_on_outside": (outs, True), "heater_off_inside": (ins, False)}
    else:
        kinds = {"hot_inside": (ins, (1, 4))}
        devs = {"fan_inside": (ins, True)}
    pick = rng.choice(sorted(kinds) + sorted(devs) + ["reading_other"])
    if pick in devs:  # contradicts the log or the stated span directly
        times, on = devs[pick]
        t = rng.choice(times)
        lie = Claim(s, st["on" if on else "off"].format(t=clock(t)), lambda f, t=t, on=on: can_device(f, t, on))
    else:  # a temperature on the wrong side of T: needs the rule and the span
        times, (lo, hi) = kinds[rng.choice(sorted(kinds))] if pick == "reading_other" else kinds[pick]
        t, v = rng.choice(times), T + rng.randint(lo, hi)
        other = x if pick == "reading_other" else None
        lie = Claim(s, reading(t, v, other), lambda f, t=t, v=v: can_temp(f, t, v),
                    ("two-fact", "about-other") if other else ("two-fact",))

    if kind == "alarm":
        ev = [f"The alarm on the {place} sounds whenever the temperature inside it is above {u(T)}, and only then.",
              f"The alarm log accurately records every time the alarm sounded that day: it sounded only once, without "
              f"a break from {clock(a)} to {clock(b)}."]
    elif kind == "heater":
        ev = [f"The heater in the {place} is on whenever the temperature inside is below {u(T)}, and off whenever "
              f"it is {u(T)} or warmer.", f"That day the heater was on only once, the whole time from {clock(a)} to "
                                         f"{clock(b)}."]
    else:
        ev = [f"The fan in the {place} always runs whenever the room is warmer than {u(T)}.",
              f"The fan did not run at any time from {clock(a)} to {clock(b)}."]
    ev.append("Every thermometer reading taken that day was accurate.")
    group = rng.choice(st["groups"])
    return Bundle(
        NAME,
        rng.choice([f"{group} argue over how warm the {place} got during the day.",
                    f"{group} compare thermometer readings from the {place}.",
                    f"{group} try to work out what the {place} temperature was and when."]),
        group, f"the {place} temperature", ev, f, lie, truths, fillers=st["fill"],
        decor=[d.format(place=place) for d in st["decor"]])
