"""Match scores. Variant 'football': the half-time and full-time scores; 'cricket': runs in two innings per team;
'basketball': points in four quarters. Some speakers' teams are stated. A player's own score in a part of the match
fits only within their team's score for that part; a speaker with no stated team could be on either side."""
from .base import ORDINALS, Bundle, Claim, date_text, facts, word

NAME = "scores"
TEAMS = ["Harbour Hawks", "Mill Lane Rovers", "Riverside Otters", "Northgate Foxes", "Elm Park Comets",
         "Quarry Rangers", "Valley Kestrels", "Brookside Badgers", "Hilltop Herons", "Canal Street Wolves"]


def seg_ok(f, team, i, n):
    return f.sc[team][i] == n


def total_ok(f, team, n):
    return sum(f.sc[team]) == n


def upto(f, team, k, n):
    """Points of `team` after the first k parts."""
    return sum(f.sc[team][:k]) == n


def lead_ok(f, team, k, state):
    """After the first k parts, `team` was ahead, behind or level."""
    a, b = sum(f.sc[team][:k]), sum(f.sc[f.other[team]][:k])
    return {"ahead": a > b, "behind": a < b, "level": a == b}[state]


def player_ok(f, who, t, i, k):
    """`who` scored k for team t in part i: t is their stated team (or theirs is not stated) and t scored k or more."""
    return f.team.get(who, t) == t and k <= f.sc[t][i]


def sl(claim, *key):
    """Set the single-valued fact a claim is about; build never gives the liar a true claim with the lie's key."""
    claim.key = key
    return claim


def split(rng, n, k):
    """n units shared among k scorers and the rest of the team."""
    cuts = sorted(rng.randint(0, n) for _ in range(k))
    return [b - a for a, b in zip([0] + cuts, cuts)]


def make(rng, people, others, day):
    t1, t2 = rng.sample(TEAMS, 2)
    sport = rng.choice(["football", "football", "cricket", "basketball"])
    dash = rng.choice(["–", "-"])
    sp = list(people)
    rng.shuffle(sp)
    if sport == "football":
        while True:
            a, b = rng.randint(0, 3), rng.randint(0, 3)
            x, y = a + rng.randint(0, 3), b + rng.randint(0, 3)
            if x + y >= 2 and (x, y) != (a, b):
                break
        sc = {t1: [a, x - a], t2: [b, y - b]}
        parts, unit, verb = ["first half", "second half"], "goals", "play for"
        ev = [f"On {date_text(day)} the {t1} played the {t2}.",
              f"At half-time the score was {t1} {a}, {t2} {b}, and the final score was {t1} {x}, {t2} {y}."]
    elif sport == "cricket":
        sc = {t: [rng.randint(60, 320), rng.randint(40, 300)] for t in (t1, t2)}
        parts, unit, verb = ["first innings", "second innings"], "runs", "bat for"
        ev = [f"In their two-innings match on {date_text(day)}, the {t1} made {sc[t1][0]} runs in their first innings "
              f"and {sc[t1][1]} in their second, and the {t2} made {sc[t2][0]} and {sc[t2][1]}."]
    else:
        while True:
            sc = {t: [rng.randint(10, 34) for _ in range(4)] for t in (t1, t2)}
            if sum(sc[t1]) != sum(sc[t2]):
                break
        parts, unit, verb = ["first quarter", "second quarter", "third quarter", "fourth quarter"], "points", "play for"
        q = lambda t: ", ".join(str(v) for v in sc[t][:3]) + f" and {sc[t][3]}"  # noqa: E731
        ev = [f"The {t1} played the {t2} on {date_text(day)}.",
              f"Quarter by quarter, the {t1} scored {q(t1)} points, and the {t2} scored {q(t2)}. The game had no "
              f"overtime."]
    team = {sp[0]: t1, sp[1]: t1, sp[2]: t2}  # sp[3] has no stated team
    ev.append(f"{sp[0]} and {sp[1]} {verb} the {t1}, and {sp[2]} {verb.replace(' ', 's ', 1)} the {t2}.")
    f = facts(sc=sc, team=team, other={t1: t2, t2: t1})
    # hidden world: sp[3] plays for one side; each part's score is shared among that side's speakers and others
    hidden = dict(team, **{sp[3]: rng.choice([t1, t2])})
    mine = {p: [0] * len(parts) for p in sp}
    for t in (t1, t2):
        squad = [p for p in sp if hidden[p] == t]
        for i in range(len(parts)):
            for p, v in zip(squad, split(rng, sc[t][i], len(squad))):
                mine[p][i] = v
    n_t = lambda n: f"{word(n) if sport == 'football' else n} {unit[:-1] if n == 1 else unit}"  # noqa: E731
    made = "made" if sport == "cricket" else "scored"

    def seg_text(t, i, n):
        if sport == "cricket":
            return f"The {t} made {n} runs in their {parts[i]}."
        if n == 0:
            return f"The {t} didn't score in the {parts[i]}."
        return rng.choice([f"The {t} scored {n_t(n)} in the {parts[i]}.",
                           f"In the {parts[i]} the {t} scored {n_t(n)}."])

    def mine_text(p, t, i, k, speaker):  # names the team, so an own goal can never count
        who = "I" if p == speaker else p
        if sport == "cricket":
            return f"{who} made {k} runs in the {t}' {parts[i]}."
        return f"{who} scored {n_t(k)} for the {t} in the {parts[i]}."

    truths = []
    for t, i in rng.sample([(t, i) for t in (t1, t2) for i in range(len(parts))], 3):
        truths.append(sl(Claim(rng.choice(sp), seg_text(t, i, sc[t][i]),
                               lambda f, t=t, i=i, n=sc[t][i]: seg_ok(f, t, i, n)), "seg", t, i))
    for t in (t1, t2):
        n = sum(sc[t])
        truths.append(sl(Claim(rng.choice(sp), rng.choice([f"The {t} finished with {n_t(n)}.",
                                                           f"Over the whole match the {t} {made} {n_t(n)}."]),
                               lambda f, t=t, n=n: total_ok(f, t, n)), "total", t))
    for p in sp:
        zero = [i for i in range(len(parts)) if not mine[p][i]]
        if zero and sport != "cricket":
            i = rng.choice(zero)
            truths.append(sl(Claim(p, f"I didn't score in the {parts[i]}.",
                                   lambda f, p=p, t=hidden[p], i=i: player_ok(f, p, t, i, 0)),
                             "player", p, i))
        hits = [i for i in range(len(parts)) if mine[p][i]]
        if hits:
            i = rng.choice(hits)
            tags = ("distractor",) if p == sp[3] and mine[p][i] > min(sc[t1][i], sc[t2][i]) else ()
            truths.append(sl(Claim(p, mine_text(p, hidden[p], i, mine[p][i], p),
                                   lambda f, p=p, t=hidden[p], i=i, k=mine[p][i]: player_ok(f, p, t, i, k), tags),
                             "player", p, i))
            s = rng.choice([y for y in sp if y != p])
            truths.append(sl(Claim(s, mine_text(p, hidden[p], i, mine[p][i], s),
                                   lambda f, p=p, t=hidden[p], i=i, k=mine[p][i]: player_ok(f, p, t, i, k)),
                             "player", p, i))
    half = len(parts) // 2 if sport != "cricket" else None
    if half:
        h1, h2 = sum(sc[t1][:half]), sum(sc[t2][:half])
        lead = t1 if h1 > h2 else t2
        hi, lo = max(h1, h2), min(h1, h2)
        truths.append(sl(Claim(rng.choice(sp), f"It was level at {hi}{dash}{lo} at half-time." if h1 == h2 else
                               f"At half-time it was {hi}{dash}{lo} to the {lead}.",
                               lambda f, h1=h1, h2=h2: upto(f, t1, half, h1) and upto(f, t2, half, h2)), "ht"))
        w1, w2 = sum(sc[t1]), sum(sc[t2])
        for t in (t1, t2):  # won the second half of the match but lost overall, or the reverse: true, sounds wrong
            o = f.other[t]
            late = sum(sc[t][half:]) - sum(sc[o][half:])
            final = sum(sc[t]) - sum(sc[o])
            if late * final < 0:
                txt = (f"The {t} lost the second half but still won the match." if final > 0 else
                       f"The {t} won the second half and still lost the match.")
                truths.append(sl(Claim(rng.choice(sp), txt, lambda f, t=t, won=final > 0: lead_ok(
                    f, t, len(parts), "ahead" if won else "behind") and (sum(f.sc[t][half:]) > sum(
                    f.sc[f.other[t]][half:])) != won, ("distractor",)), "result"))
        t = rng.choice([t1, t2])
        state = next(st for st in ("ahead", "behind", "level") if lead_ok(f, t, half, st))
        truths.append(sl(Claim(rng.choice(sp), {"ahead": f"The {t} were ahead at half-time.",
                                                "behind": f"The {t} were behind at half-time.",
                                                "level": "The scores were level at half-time."}[state],
                               lambda f, t=t, st=state: lead_ok(f, t, half, st)), "ht"))
        if w1 == w2:
            truths.append(sl(Claim(rng.choice(sp), "The match ended in a draw.",
                                   lambda f: lead_ok(f, t1, len(parts), "level")), "result"))
        else:
            win, wa, wb = (t1, w1, w2) if w1 > w2 else (t2, w2, w1)
            truths.append(sl(Claim(rng.choice(sp), f"The {win} won {wa}{dash}{wb}.", lambda f, win=win, wa=wa, wb=wb:
                                   total_ok(f, win, wa) and total_ok(f, f.other[win], wb)), "result"))
    else:
        for t in (t1, t2):
            d = (sc[t][1] > sc[t][0]) - (sc[t][1] < sc[t][0])
            how = {1: "more runs in their second innings than", -1: "fewer runs in their second innings than",
                   0: "exactly as many runs in their second innings as"}[d]
            truths.append(Claim(rng.choice(sp), f"The {t} made {how} in their first.",
                                lambda f, t=t, d=d: (f.sc[t][1] > f.sc[t][0]) - (f.sc[t][1] < f.sc[t][0]) == d))
    for text in rng.sample([f"I parked in bay {rng.randint(2, 60)}.",
                            f"I took the number {rng.randint(3, 99)} bus to the game.",
                            f"I sat in row {rng.choice('ABCDEFGH')} of the stand, seat {rng.randint(1, 30)}."], 2):
        truths.append(Claim(rng.choice(sp), text, lambda f: True, ("neutral",)))

    kinds = ["seg", "total", "player", "player_other"] + (["ht", "result"] if half else ["compare"])
    kind = rng.choice(kinds)
    s = rng.choice(sp)
    if kind == "seg":
        t, i = rng.choice(sorted(sc)), rng.randrange(len(parts))
        n = rng.choice([v for v in range(max(0, sc[t][i] - 3), sc[t][i] + 4) if v != sc[t][i]])
        lie = sl(Claim(s, seg_text(t, i, n), lambda f, t=t, i=i, n=n: seg_ok(f, t, i, n),
                       ("two-fact",) if sport == "football" and i == 1 else ()), "seg", t, i)
    elif kind == "total":
        t = rng.choice(sorted(sc))
        n = sum(sc[t]) + rng.choice([-20, -10, -2, -1, 1, 2, 10] if sport != "football" else [-1, 1, 2])
        n = n if n >= 0 else sum(sc[t]) + 1
        lie = sl(Claim(s, f"Over the whole match the {t} {made} {n_t(n)}.", lambda f, t=t, n=n: total_ok(f, t, n),
                       ("two-fact",) if sport != "football" else ()), "total", t)
    elif kind in ("player", "player_other"):
        p = rng.choice(sp[:3])  # a speaker whose team is stated
        i = rng.randrange(len(parts))
        k = sc[team[p]][i] + rng.randint(1, 2 if sport == "football" else 9)
        if kind == "player":
            lie = Claim(p, mine_text(p, team[p], i, k, p), lambda f, p=p, t=team[p], i=i, k=k: player_ok(f, p, t, i, k),
                        ("two-fact",))
        else:
            s = rng.choice([y for y in sp if y != p])
            lie = Claim(s, mine_text(p, team[p], i, k, s), lambda f, p=p, t=team[p], i=i, k=k: player_ok(f, p, t, i, k),
                        ("two-fact", "about-other"))
        sl(lie, "player", p, i)
    elif kind == "ht":
        t = rng.choice([t1, t2])
        state = rng.choice([st for st in ("ahead", "behind", "level") if not lead_ok(f, t, half, st)])
        txt = {"ahead": f"The {t} were ahead at half-time.", "behind": f"The {t} were behind at half-time.",
               "level": "The scores were level at half-time."}[state]
        lie = sl(Claim(s, txt, lambda f, t=t, st=state: lead_ok(f, t, half, st),
                       ("two-fact",) if sport == "basketball" else ()), "ht")
    elif kind == "result":
        t = rng.choice([t1, t2])
        state = rng.choice([st for st in ("ahead", "behind", "level") if not lead_ok(f, t, len(parts), st)])
        txt = {"ahead": f"The {t} won the match.", "behind": f"The {t} lost the match.",
               "level": "The match ended in a draw."}[state]
        lie = sl(Claim(s, txt, lambda f, t=t, st=state: lead_ok(f, t, len(parts), st),
                       ("two-fact",) if sport == "basketball" else ()), "result")
    else:  # cricket: compare the two teams over both innings
        t = rng.choice([t1, t2])
        o = f.other[t]
        d = sum(sc[t]) - sum(sc[o])
        n = rng.choice([v for v in (abs(d) - 10, abs(d) - 5, abs(d) + 5, abs(d) + 10, abs(d) + 20)
                        if v >= 1 and v != d])
        txt = f"Over both innings the {t} made {n} more runs than the {o}."
        lie = sl(Claim(s, txt, lambda f, t=t, o=o, n=n: sum(f.sc[t]) - sum(f.sc[o]) == n, ("two-fact",)), "compare")
    topic = f"the {t1}' {sport} match against the {t2}"
    group = rng.choice(["Teammates", "Club members", "Supporters", "Parents at the club"])
    return Bundle(
        NAME,
        rng.choice([f"{group} argue over the scores in {topic}.", f"{group} go back over {topic}.",
                    f"{group} compare what they remember of {topic}."]),
        group, topic, ev, f, lie, truths,
        fillers=["Who was in charge of the match?", "When is the next fixture?", "Were many people watching?",
                 "That was a tense game.", "Should we have played the reserves?"],
        decor=["The game was played in the sports hall." if sport == "basketball" else
               "The match was played on the school field.", "Tea was served after the match.",
               "The club has a new kit this season."],
        timed=False)
