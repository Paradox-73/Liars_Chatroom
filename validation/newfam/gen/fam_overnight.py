"""Windows that cross midnight. The night starts on d0 (the case's day) and ends on d1 = d0 + 1; times are minutes
from the midnight that starts d0, so 1:15 AM on d1 is 1440 + 75.

Variants: 'deliveries' (a depot took deliveries only between w0 and w1), 'shift' (someone worked without a break
from w0 to w1 and never left), 'outage' (a website was offline from w0 to w1 and nobody could use it), 'ferry'
(an overnight ferry left at w0, reached the other port at w1, no stops, nobody got on or off at sea).

About half the cases use no clock time before 5:00 AM anywhere (the window ends at 5:30 AM or later, no claim
time falls in the small hours and the word 'midnight' never appears), because the rule solver steps aside only
when it sees such a time. Never combined with another family: the evidence names its own two dates."""
import datetime as dt
import re

from .base import Bundle, Claim, clock, date_text, duration, facts

NAME = "overnight"
SMALL_HOURS = re.compile(r"\b(12|[1-4]):\d\d AM\b|midnight", re.I)


def inside(f, t):
    return f.w0 <= t <= f.w1


def make(rng, people, others, day):
    d0 = day
    d1 = day + dt.timedelta(days=1)
    early = rng.random() < 0.5
    w0 = rng.randrange(21 * 60, 23 * 60 + 31, 15)
    w1 = 1440 + (rng.randrange(60, 4 * 60 + 31, 15) if early else rng.randrange(5 * 60 + 30, 7 * 60 + 31, 15))
    f = facts(w0=w0, w1=w1, P=None)
    sp = list(people)

    def stamp(t):
        """'11:15 PM on 9 October 2026', or 'that night' / 'the next morning' when that is unambiguous."""
        d, m = divmod(t, 1440)
        if rng.random() < 0.55 or (d == 1 and m < 5 * 60):
            return f"{clock(m)} on {date_text(d1 if d else d0)}"
        if d == 0:
            return f"{clock(m)} that evening" if m < 21 * 60 else f"{clock(m)} that night"
        return f"{clock(m)} the next morning"

    a_txt, b_txt = f"{clock(w0)} on {date_text(d0)}", f"{clock(w1 - 1440)} on {date_text(d1)}"
    shown = lambda t: early or t < 1440 or t % 1440 >= 5 * 60  # noqa: E731  (a time this case may print)
    ins = [t for t in range(w0 + 15, w1 - 14, 5) if shown(t)]
    before = list(range(max(18 * 60, w0 - 180), w0 - 19, 5))
    after = list(range(w1 + 20, min(1440 + 11 * 60 + 55, w1 + 180), 5))
    length = w1 - w0
    wrong_len = rng.choice([length + 60, length - 60, length + 120, length - 30, 1440 - length])
    hours_in = [k for k in range(1, 10) if w0 + 60 * k <= w1 - 15 and shown(w0 + 60 * k)]
    hours_out = [k for k in range(1, 12) if w0 + 60 * k >= w1 + 15]
    truths = []
    variant = rng.choice(["deliveries", "shift", "outage", "ferry"])

    if variant == "deliveries":
        depot, goods = rng.choice([("depot", "my load"), ("warehouse", "the pallets"), ("market hall", "the crates"),
                                   ("sorting office", "the mail sacks")])
        ev = [f"On {date_text(d0)} and {date_text(d1)}, the {depot} accepted deliveries at all times between {a_txt} "
              f"and {b_txt}, and at no other time on those two days."]

        def deliver(s, t, x=None):
            if x is None or x == s:
                return rng.choice([f"I delivered {goods} to the {depot} at {stamp(t)}.",
                                   f"At {stamp(t)} I unloaded my van at the {depot}."])
            return rng.choice([f"I saw {x} unloading at the {depot} at {stamp(t)}.",
                               f"{x} made a delivery to the {depot} at {stamp(t)}."])
        # hidden world: each speaker made at most one delivery; the fourth one made it exactly at a window edge
        *drivers, edge = rng.sample(sp, 4)
        times = {d: rng.choice(ins) for d in drivers}
        times[edge] = rng.choice([w0, w1])  # "between" includes both ends
        if hours_in:
            k = rng.choice(hours_in)
            times[drivers[0]] = w0 + 60 * k
            truths.append(Claim(drivers[0], f"I made my delivery exactly {k} hours after the {depot} started taking "
                                            f"deliveries.", lambda f, k=k: inside(f, f.w0 + 60 * k)))
        for d, t in times.items():
            truths.append(Claim(d, deliver(d, t), lambda f, t=t: inside(f, t), ("boundary",) if d == edge else ()))
        x = rng.choice(drivers)
        s = rng.choice([y for y in sp if y != x])
        truths.append(Claim(s, deliver(s, times[x], x), lambda f, t=times[x]: inside(f, t)))
        t = rng.choice(before)
        s = rng.choice(drivers)
        truths.append(Claim(s, f"I got to the {depot} at {stamp(t)} and waited until it started taking deliveries.",
                            lambda f, t=t: t < f.w0, ("distractor",)))
        truths.append(Claim(rng.choice(sp), f"The {depot} took deliveries for {duration(length)} that night.",
                            lambda f: f.w1 - f.w0 == length))
        t = rng.choice(ins)
        truths.append(Claim(rng.choice(sp), f"The {depot} was taking deliveries at {stamp(t)}.",
                            lambda f, t=t: inside(f, t)))
        t = rng.choice(before + after)
        truths.append(Claim(rng.choice(sp), f"The {depot} was closed to deliveries at {stamp(t)}.",
                            lambda f, t=t: not inside(f, t)))
        neutral = ["My lorry carries {n} pallets.", "I drove {m} miles that night."]
        kind = rng.choice(["before", "after", "after", "other", "hours", "length"])
        s = rng.choice(sp)
        if kind in ("before", "after"):
            t = rng.choice(before if kind == "before" else after)
            lie = Claim(s, deliver(s, t), lambda f, t=t: inside(f, t))
        elif kind == "other":
            x = rng.choice([y for y in sp if y != s])
            t = rng.choice(before + after)
            lie = Claim(s, deliver(s, t, x), lambda f, t=t: inside(f, t), ("about-other",))
        elif kind == "hours" and hours_out:
            k = rng.choice(hours_out[:3])
            lie = Claim(s, f"I made my delivery exactly {k} hours after the {depot} started taking deliveries.",
                        lambda f, k=k: inside(f, f.w0 + 60 * k), ("two-fact",))
        else:
            lie = Claim(s, f"The {depot} took deliveries for {duration(wrong_len)} that night.",
                        lambda f: f.w1 - f.w0 == wrong_len, ("two-fact",))
        group, topic = "Night drivers", f"the overnight deliveries to the {depot}"
        scen = [f"{group} argue over when they delivered to the {depot} overnight.",
                f"Drivers compare delivery times at the {depot} across one night.",
                f"Night drivers try to work out who delivered to the {depot} when."]
        fill = [f"Should the {depot} open earlier?", "Is the night rate worth it?", "Who signs the delivery notes?"]
        decor = [f"The {depot} has four loading bays.", f"The {depot} yard is lit all night."]

    elif variant == "shift":
        place, other = rng.choice([("bakery", "all-night café"), ("hospital ward", "car park"),
                                   ("hotel front desk", "train station"), ("signal box", "village pub"),
                                   ("radio station", "petrol station")])
        p = rng.choice(sp) if rng.random() < 0.5 else others[0]
        f.P = p
        ev = [rng.choice([f"{p} worked a night shift at the {place} from {a_txt} until {b_txt}, without a break, and "
                          f"did not leave the {place} at any point during the shift.",
                          f"From {a_txt} until {b_txt}, {p} worked a night shift at the {place} without a break and "
                          f"never left the {place} during it."]),
              f"The {place} and the {other} are separate places."]

        def seen(s, t, where):
            if s == p:
                return rng.choice([f"I was at the {where} at {stamp(t)}.", f"At {stamp(t)} I was in the {where}."])
            return rng.choice([f"I saw {p} at the {where} at {stamp(t)}.", f"{p} was at the {where} at {stamp(t)}."])
        for t in rng.sample(ins, 3):
            s = rng.choice(sp)
            text = (f"At {stamp(t)} I was on shift at the {place}." if s == p else
                    rng.choice([f"{p} was on shift at the {place} at {stamp(t)}.",
                                f"At {stamp(t)} {p} was still on shift at the {place}."]))
            truths.append(Claim(s, text, lambda f, t=t: inside(f, t)))
        truths.append(Claim(rng.choice(sp), f"{p}'s shift began at {stamp(w0)}.", lambda f: f.w0 == w0))
        truths.append(Claim(rng.choice(sp), f"{p} finished the shift at {stamp(w1)}.", lambda f: f.w1 == w1,
                            ("boundary",)))
        truths.append(Claim(rng.choice(sp), f"{p}'s shift lasted {duration(length)}.", lambda f: f.w1 - f.w0 == length))
        t = rng.choice(after)  # after the shift: free to be elsewhere
        s = rng.choice(sp)
        truths.append(Claim(s, seen(s, t, other), lambda f, t=t: not inside(f, t), ("distractor",)))
        t = rng.choice(before)
        s = rng.choice(sp)
        truths.append(Claim(s, seen(s, t, other), lambda f, t=t: not inside(f, t)))
        for s in rng.sample([y for y in sp if y != p], 2):
            t = rng.choice(ins)
            truths.append(Claim(s, f"I was at the {other} at {stamp(t)}.", lambda f, s=s: s != f.P))
        neutral = ["I made {n} cups of tea that night.", "I live {n} streets from the " + place + "."]
        kind = rng.choice(["other", "other", "left", "length", "end"])
        s = rng.choice(sp)
        t = rng.choice(ins)
        if kind == "other":
            lie = Claim(s, seen(s, t, other), lambda f, t=t: not inside(f, t),
                        ("two-fact",) + (() if s == p else ("about-other",)))
        elif kind == "left":
            text = (f"I popped out of the {place} at {stamp(t)}." if s == p else
                    rng.choice([f"{p} left the {place} at {stamp(t)}.", f"I saw {p} walk out of the {place} at {stamp(t)}."]))
            lie = Claim(s, text, lambda f, t=t: not inside(f, t), () if s == p else ("about-other",))
        elif kind == "length":
            lie = Claim(s, f"{p}'s shift lasted {duration(wrong_len)}.", lambda f: f.w1 - f.w0 == wrong_len,
                        ("two-fact",))
        else:
            t = rng.choice([x for x in (w1 - 60, w1 - 90, w1 - 120, w1 + 60) if shown(x)])
            lie = Claim(s, f"{p} finished the shift at {stamp(t)}.", lambda f, t=t: f.w1 == t)
        group, topic = "Night staff", f"{p}'s night shift"
        scen = [f"{group} try to pin down where {p} was during the night shift.",
                f"Colleagues argue over {p}'s night shift at the {place}.",
                f"Friends compare sightings of {p} across one night."]
        fill = ["Are night shifts paid extra?", f"Is the {other} open all night?", "Who covers the next night?"]
        decor = [f"The {place} has a back door onto a lane.", f"The {other} has a blue sign."]

    elif variant == "outage":
        system, act, did, did_x = rng.choice([
            ("the ticket website", "book tickets on the ticket website", "booked my tickets on the ticket website",
             "booked tickets on the ticket website"),
            ("the library's booking system", "reserve a book through the library's booking system",
             "reserved a book through the library's booking system",
             "reserved a book through the library's booking system"),
            ("the bank's app", "send money with the bank's app", "sent money with the bank's app",
             "sent money with the bank's app"),
            ("the parking website", "pay for parking on the parking website", "paid for parking on the parking website",
             "paid for parking on the parking website")])
        ev = [rng.choice([f"{system[0].upper() + system[1:]} went offline at {a_txt} and came back online at {b_txt}, "
                          f"and it stayed offline without a break in between.",
                          f"At {a_txt}, {system} went offline; it stayed offline without a break until it came back "
                          f"online at {b_txt}."]),
              f"Nobody could {act} while it was offline."]

        def used(s, t, x=None):
            if x is None or x == s:
                return rng.choice([f"I {did} at {stamp(t)}.", f"At {stamp(t)} I {did}."])
            return f"{x} {did_x} at {stamp(t)}."
        for s in rng.sample(sp, 3):
            t = rng.choice(before + after)
            truths.append(Claim(s, used(s, t), lambda f, t=t: not inside(f, t)))
        for _ in range(2):
            s, x = rng.sample(sp, 2)
            t = rng.choice(before + after)
            truths.append(Claim(s, used(s, t, x), lambda f, t=t: not inside(f, t)))
        for t in rng.sample(before + after, 2):
            truths.append(Claim(rng.choice(sp), f"{system[0].upper() + system[1:]} was working normally at {stamp(t)}.",
                                lambda f, t=t: not inside(f, t)))
        t = w1 + 5
        truths.append(Claim(rng.choice(sp), f"I {did} at {stamp(t)}, just after it came back.",
                            lambda f, t=t: t > f.w1, ("distractor", "boundary")))
        k = rng.choice(hours_out[:3])
        truths.append(Claim(rng.choice(sp), f"I {did} {k} hours after it went offline.",
                            lambda f, k=k: not inside(f, f.w0 + 60 * k)))
        truths.append(Claim(rng.choice(sp), f"The outage lasted {duration(length)}.", lambda f: f.w1 - f.w0 == length))
        neutral = ["I had {n} tabs open that night.", "My phone was on {p} percent battery that night."]
        kind = rng.choice(["use", "use", "other", "hours", "length"])
        s = rng.choice(sp)
        t = rng.choice(ins)
        if kind == "use":
            lie = Claim(s, used(s, t), lambda f, t=t: not inside(f, t), ("two-fact",))
        elif kind == "other":
            x = rng.choice([y for y in sp if y != s] + others[:1])
            lie = Claim(s, used(s, t, x), lambda f, t=t: not inside(f, t), ("two-fact", "about-other"))
        elif kind == "hours" and hours_in:
            k = rng.choice(hours_in)
            lie = Claim(s, f"I {did} {k} hours after it went offline.", lambda f, k=k: not inside(f, f.w0 + 60 * k),
                        ("two-fact",))
        else:
            lie = Claim(s, f"The outage lasted {duration(wrong_len)}.", lambda f: f.w1 - f.w0 == wrong_len,
                        ("two-fact",))
        group, topic = "Neighbours", f"the night {system} was down"
        scen = [f"{group} compare when they managed to use {system} around an overnight outage.",
                f"Friends argue over who got through on {system} during the night.",
                f"Users try to work out when {system} was really down."]
        fill = ["Does this outage happen every month?", "Should there be a warning next time?",
                "Who do we complain to?"]
        decor = [f"{system[0].upper() + system[1:]} has a blue logo.", "The outage was announced by email."]

    else:  # ferry
        A, B = rng.choice([("Port Ellis", "Kestrel Bay"), ("Northhaven", "Saltmarsh"), ("Larkport", "Greyholm"),
                           ("Westcliff", "Amber Quay")])
        ev = rng.choice([[f"The overnight ferry left {A} at {a_txt}, made no stops, and reached {B} at {b_txt}.",
                          f"Nobody got on or off the ferry between leaving {A} and reaching {B}."],
                         [f"The overnight ferry from {A} to {B} set off at {a_txt} and docked at {B} at {b_txt}, with "
                          f"no stops on the way.", "Nobody boarded or left the ferry while it was at sea."]])
        passengers = rng.sample(sp, 2)
        for s in passengers:
            t = rng.choice([x for x in before if x >= w0 - 90])
            truths.append(Claim(s, rng.choice([f"I boarded the ferry at {A} at {stamp(t)}.",
                                               f"At {stamp(t)} I was already aboard the ferry at {A}."]),
                                lambda f, t=t: t < f.w0))
            t = rng.choice(ins)
            truths.append(Claim(s, rng.choice([f"I was on the ferry, out at sea, at {stamp(t)}.",
                                               f"At {stamp(t)} I was out at sea on the ferry."]),
                                lambda f, t=t: f.w0 < t < f.w1))
            off = rng.choice(after[2:12])
            truths.append(Claim(s, f"I got off the ferry at {B} at {stamp(off)}.", lambda f, t=off: t >= f.w1))
        t = rng.choice(range(w1 + 5, off - 4, 5))  # the last passenger lingered aboard after docking
        truths.append(Claim(passengers[-1], f"I was still on board at {stamp(t)}, after the ferry had docked at {B}.",
                            lambda f, t=t: t > f.w1, ("distractor",)))
        waver, greeter = [y for y in sp if y not in passengers]
        truths.append(Claim(waver, f"I waved the ferry off from the quay at {A} at {stamp(w0)}.", lambda f: f.w0 == w0))
        truths.append(Claim(greeter, f"I met the ferry when it docked at {B} at {stamp(w1)}.", lambda f: f.w1 == w1,
                            ("boundary",)))
        truths.append(Claim(rng.choice(sp), f"The crossing took {duration(length)}.", lambda f: f.w1 - f.w0 == length))
        neutral = ["I had {n} bags with me.", "The ferry ticket cost me {m} pounds."]
        kind = rng.choice(["on", "off", "other", "length", "arrive"])
        s = rng.choice(sp)
        t = rng.choice(ins)
        if kind == "on":
            lie = Claim(s, f"I got on the ferry at {A} at {stamp(t)}.", lambda f, t=t: t < f.w0, ("two-fact",))
        elif kind == "off":
            lie = Claim(s, f"I got off the ferry at {B} at {stamp(t)}.", lambda f, t=t: t >= f.w1, ("two-fact",))
        elif kind == "other":
            x = rng.choice([y for y in sp if y != s])
            lie = Claim(s, f"I saw {x} get off the ferry at {B} at {stamp(t)}.", lambda f, t=t: t >= f.w1,
                        ("two-fact", "about-other"))
        elif kind == "length":
            lie = Claim(s, f"The crossing took {duration(wrong_len)}.", lambda f: f.w1 - f.w0 == wrong_len,
                        ("two-fact",))
        else:
            t = rng.choice([x for x in ins if abs(x - w1) >= 30])
            lie = Claim(s, f"The ferry reached {B} at {stamp(t)}.", lambda f, t=t: f.w1 == t)
        group, topic = "Passengers", "the overnight ferry crossing"
        scen = [f"{group} piece together the overnight ferry crossing from {A} to {B}.",
                "Travellers argue over when they got on and off the night ferry.",
                f"Friends compare notes on the overnight ferry to {B}."]
        fill = ["Was the sea rough that night?", "Is there a cabin next time?", "Who booked the ferry?"]
        decor = ["The ferry has a red funnel.", f"{B} has a small lighthouse."]

    for i, s in enumerate([x for x in sp if not any(c.who == x for c in truths)] + [rng.choice(sp), rng.choice(sp)]):
        truths.append(Claim(s, neutral[i % 2].format(n=rng.randint(2, 9), m=rng.randint(12, 90),
                                                     p=rng.randrange(20, 95, 5)), lambda f: True, ("neutral",)))
        # every speaker gets a claim; two neutral claims at least
    if not early:  # this half must not trip the rule solver's overnight guard
        for text in ev + [c.text for c in truths] + [lie.text]:
            assert not SMALL_HOURS.search(text), text
    return Bundle(NAME, rng.choice(scen), group, topic, ev, f, lie, truths, fillers=fill, decor=decor, timed=False)
