"""Library loans. Every book is due back exactly L days after the day it was borrowed, and a late return costs a fee
for each day after the due date, up to and including the day it comes back. Variant 'paid': the evidence gives the
late fee paid instead of the return date, so the return date has to be worked out from the fee."""
import datetime as dt

from .base import Bundle, Claim, an, anchor, date_text, facts, money, weekday

NAME = "library"
ITEMS = ["travel guide", "cookery book", "book on bees", "atlas", "bird guide", "poetry collection",
         "chess puzzle book", "garden handbook", "history of the railways", "biography of a pianist", "crime novel",
         "book of sea charts"]
LIBS = ["town", "village", "college", "community", "harbour", "district"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY = dt.timedelta(days=1)


def due_of(f, it):
    return f.loans[it].d0 + f.L * DAY


def late_of(f, it):
    ln = f.loans[it]
    if ln.d1 is None:  # variant 'paid': the fee paid fixes the number of late days
        return ln.paid // f.fee
    return max(0, (ln.d1 - due_of(f, it)).days)


def back_of(f, it):
    ln = f.loans[it]
    return ln.d1 if ln.d1 is not None else due_of(f, it) + late_of(f, it) * DAY


def make(rng, people, others, day):
    D = date_text
    L = rng.choice([7, 14, 14, 21, 28])
    fee = rng.choice([10, 15, 20, 25, 30, 50])
    cur = rng.choice(["£", "£", "$"])
    lib = rng.choice(LIBS)
    items = rng.sample(ITEMS, 3)
    sp = list(people)
    borrowers = rng.sample(sp, 1) + rng.sample(sp + others[:2], 1)
    if borrowers[1] == borrowers[0] or rng.random() < 0.55:
        borrowers = borrowers[:1]
    if rng.random() < 0.3:  # the borrower is someone outside the chat
        borrowers[0] = others[2]
    paid = rng.random() < 0.3
    ev = [f"Every book from the {lib} library is due back exactly {L} days after the day it is borrowed.",
          f"A late return costs {money(fee, cur)} for each day after the due date, up to and including the day the "
          f"book comes back, and the library charges nothing else."]
    loans = {}
    start = dt.date(2026, 1, 5) + rng.randrange(0, 290) * DAY
    for i, p in enumerate(borrowers):
        it, d0 = items[i], start + rng.randrange(0, 20) * DAY
        if i == 0 and paid:
            n = rng.randint(1, 9)
            loans[it] = facts(who=p, d0=d0, d1=None, paid=n * fee)
            ev.append(f"{p} borrowed {an(it)} from the library on {D(d0)}, returned it late and paid exactly "
                      f"{money(n * fee, cur)} in late fees for it.")
            continue
        r = rng.random()
        d1 = d0 + (L if r < 0.2 else rng.randint(2, L - 1) if r < 0.35 else L + rng.randint(1, 12)) * DAY
        loans[it] = facts(who=p, d0=d0, d1=d1, paid=None)
        ev.append(f"{p} borrowed {an(it)} from the library on {D(d0)} and returned it on {D(d1)}.")
    has_anchor = rng.random() < 0.5
    if has_anchor:
        ev.append(anchor(min(ln.d0 for ln in loans.values())))
    f = facts(L=L, fee=fee, loans=loans)
    # the lie is picked first, so the liar never also states a truth about the same loan
    it_l = rng.choice(list(loans))
    p_l = loans[it_l].who
    liar = p_l if p_l in sp and rng.random() < 0.4 else rng.choice(sp)
    kind = rng.choice(["due", "late", "fee", "status"] + (["weekday"] if has_anchor else []) +
                      (["back"] if loans[it_l].d1 is None else []))

    def snd(it):
        return rng.choice([y for y in sp if y != liar] if it == it_l else sp)

    def X(s, p):
        return "I" if s == p else p

    truths = []
    for it, ln in loans.items():
        p, due, n, back = ln.who, due_of(f, it), late_of(f, it), back_of(f, it)
        s = snd(it)
        truths.append(Claim(s, rng.choice([f"The {it} was due back on {D(due)}.",
                                           f"{'My' if s == p else p + chr(39) + 's'} {it} was due back on {D(due)}."]),
                            lambda f, it=it, d=due: due_of(f, it) == d))
        s = snd(it)
        truths.append(Claim(s, f"{X(s, p)} borrowed the {it} on {D(ln.d0)}.", lambda f, it=it, d=ln.d0: f.loans[it].d0 == d))
        s = snd(it)
        truths.append(Claim(s, rng.choice([f"{X(s, p)} returned the {it} on {D(back)}.",
                                           f"{X(s, p)} brought the {it} back on {D(back)}."]),
                            lambda f, it=it, d=back: back_of(f, it) == d))
        s = snd(it)
        if n:
            truths.append(Claim(s, rng.choice([f"{X(s, p)} returned the {it} {n} day{'s' if n > 1 else ''} late.",
                                               f"{X(s, p)} brought the {it} back {n} day{'s' if n > 1 else ''} after "
                                               f"it was due."]), lambda f, it=it, n=n: late_of(f, it) == n))
            s = snd(it)
            truths.append(Claim(s, rng.choice([f"{X(s, p)} paid {money(n * fee, cur)} in late fees for the {it}.",
                                               f"The late fee for the {it} came to {money(n * fee, cur)}."]),
                                lambda f, it=it, c=n * fee: late_of(f, it) * f.fee == c))
            s = snd(it)
            truths.append(Claim(s, f"{X(s, p)} kept the {it} for longer than {L} days.",
                                lambda f, it=it: late_of(f, it) >= 1, ("distractor",)))
        elif back == due:  # returned on the due date itself: not late, no fee
            truths.append(Claim(s, f"{X(s, p)} returned the {it} on the day it was due.",
                                lambda f, it=it: back_of(f, it) == due_of(f, it), ("boundary",)))
            s = snd(it)
            truths.append(Claim(s, f"{X(s, p)} paid no late fee for the {it}.", lambda f, it=it: late_of(f, it) == 0,
                                ("boundary", "distractor")))
            s = snd(it)
            truths.append(Claim(s, f"{X(s, p)} returned the {it} exactly {L} days after borrowing it.",
                                lambda f, it=it: (back_of(f, it) - f.loans[it].d0).days == f.L, ("boundary",)))
        else:
            truths.append(Claim(s, f"{X(s, p)} returned the {it} before it was due.",
                                lambda f, it=it: back_of(f, it) < due_of(f, it)))
            s = snd(it)
            truths.append(Claim(s, f"{X(s, p)} paid no late fee for the {it}.", lambda f, it=it: late_of(f, it) == 0))
            s, k = snd(it), (due - back).days
            truths.append(Claim(s, f"{X(s, p)} returned the {it} {k} days before it was due.",
                                lambda f, it=it, k=k: (due_of(f, it) - back_of(f, it)).days == k))
        if has_anchor:
            s = snd(it)
            truths.append(Claim(s, f"{X(s, p)} returned the {it} on a {weekday(back)}.",
                                lambda f, it=it, wd=weekday(back): weekday(back_of(f, it)) == wd))
            s = snd(it)
            truths.append(Claim(s, f"The {it} was due back on a {weekday(due)}.",
                                lambda f, it=it, wd=weekday(due): weekday(due_of(f, it)) == wd))
    extra = items[2]
    s = rng.choice(sp)
    truths.append(Claim(s, f"I borrowed {an(extra)} from the {lib} library on {D(start + rng.randrange(0, 25) * DAY)} "
                           f"as well.", lambda f, it=extra: it not in f.loans, ("distractor",)))
    for tpl in rng.sample(["I printed {k} pages at the {lib} library on {d}.",
                           "I spent {k} hours in the reading room on {d}.",
                           "I made {k} photocopies at the library on {d}.",
                           "I renewed my library card on {d}, which took {k} minutes."], 3):
        s = rng.choice(sp)
        truths.append(Claim(s, tpl.format(k=rng.randint(2, 9), lib=lib, d=D(start + rng.randrange(0, 40) * DAY)),
                            lambda f: True, ("neutral",)))

    it, s = it_l, liar
    ln = loans[it]
    p, due, n, back = ln.who, due_of(f, it), late_of(f, it), back_of(f, it)
    about = () if s == p else ("about-other",)
    if kind == "due":
        d = due + rng.choice([-7, -2, -1, 1, 2, 7]) * DAY
        lie = Claim(s, rng.choice([f"The {it} was due back on {D(d)}.",
                                   f"{'My' if s == p else p + chr(39) + 's'} {it} was due back on {D(d)}."]),
                    lambda f, it=it, d=d: due_of(f, it) == d, ("two-fact",) + about)
    elif kind == "late":
        m = rng.choice([x for x in (n - 1, n + 1, n + 2) if x >= 1])
        lie = Claim(s, f"{X(s, p)} returned the {it} {m} day{'s' if m > 1 else ''} late.",
                    lambda f, it=it, m=m: late_of(f, it) == m, ("two-fact",) + about)
    elif kind == "fee":
        m = rng.choice([x for x in (n - 1, n + 1, n + 2) if x >= 1])
        lie = Claim(s, rng.choice([f"{X(s, p)} paid {money(m * fee, cur)} in late fees for the {it}.",
                                   f"The late fee for the {it} came to {money(m * fee, cur)}."]),
                    lambda f, it=it, c=m * fee: late_of(f, it) * f.fee == c,
                    (() if ln.d1 is None else ("two-fact",)) + about)  # 'paid' states the fee outright
    elif kind == "status":
        if n:  # in the 'paid' variant the evidence says outright that the return was late and a fee was paid
            lie = Claim(s, rng.choice([f"{X(s, p)} paid no late fee for the {it}.",
                                       f"{X(s, p)} returned the {it} on time."]),
                        lambda f, it=it: late_of(f, it) == 0, (() if ln.d1 is None else ("two-fact",)) + about)
        else:
            lie = Claim(s, f"{X(s, p)} had to pay a late fee for the {it}.", lambda f, it=it: late_of(f, it) >= 1,
                        ("two-fact",) + about)
    elif kind == "weekday":
        target = rng.choice(["back", "due"])
        real = weekday(back if target == "back" else due)
        wd = rng.choice([w for w in WEEKDAYS if w != real])
        if target == "back":
            lie = Claim(s, f"{X(s, p)} returned the {it} on a {wd}.",
                        lambda f, it=it, wd=wd: weekday(back_of(f, it)) == wd, ("two-fact",) + about)
        else:
            lie = Claim(s, f"The {it} was due back on a {wd}.", lambda f, it=it, wd=wd: weekday(due_of(f, it)) == wd,
                        ("two-fact",) + about)
    else:  # 'paid' variant: the return date implied by the fee, shifted by a day or two
        d = back + rng.choice([-2, -1, 1, 2]) * DAY
        lie = Claim(s, f"{X(s, p)} brought the {it} back on {D(d)}.", lambda f, it=it, d=d: back_of(f, it) == d,
                    ("two-fact",) + about)

    group = rng.choice(["Book club members", "Neighbours", "Classmates", "Flatmates"])
    main = loans[next(iter(loans))].who
    assert len(truths) >= 10, len(truths)
    return Bundle(
        NAME,
        rng.choice([f"{group} sort out what {main} owed the {lib} library.",
                    f"{group} compare library due dates and late fees.",
                    f"{group} argue over when a library book was due back.",
                    f"{group} try to work out the late fees on some library loans."]),
        group, "some library loans", ev, f, lie, truths,
        fillers=["Does the library open on Sundays?", "Can loans be renewed online?",
                 "Are the late fees always this steep?", "Is the reading room quiet in the mornings?",
                 "Honestly, the late fees are far too high."],
        decor=[f"The {lib} library has a blue front door.", "The library café sells soup.",
               "The returns box is next to the car park."],
        timed=False)
