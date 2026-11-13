"""Prices, discounts and change, in whole cents. Variant 'discount': listed prices, a percentage off the whole bill for
a stated group and nobody else, cash paid with the correct change. Variant 'deal': one item has a three-for deal the
till always applied to as many threes as possible; another has no deal."""
from .base import Bundle, Claim, an, date_text, facts, money, word

NAME = "prices"
# (singular, plural, lowest and highest price in cents); prices are drawn in steps of 20 cents
SHOPS = {
    "café": [("sandwich", "sandwiches", 320, 560), ("coffee", "coffees", 160, 300), ("muffin", "muffins", 160, 300),
             ("bowl of soup", "bowls of soup", 300, 500)],
    "bakery": [("loaf", "loaves", 180, 360), ("croissant", "croissants", 100, 220), ("pie", "pies", 240, 420),
               ("slice of cake", "slices of cake", 200, 380)],
    "museum shop": [("postcard", "postcards", 60, 160), ("poster", "posters", 400, 900),
                    ("guidebook", "guidebooks", 600, 1400), ("pencil", "pencils", 60, 160)],
    "canteen": [("curry", "curries", 400, 700), ("salad", "salads", 300, 560), ("pot of tea", "pots of tea", 120, 240),
                ("fruit pot", "fruit pots", 120, 260)],
}
GROUPS = [
    dict(rule="Students got {pct}% off their whole bill, and nobody else got any discount.",
         yes2="{x} and {y} are students", yes1="{x} is a student", no1="{z} is not a student",
         no2="{z} and {w} are not students"),
    dict(rule="Customers with a loyalty card got {pct}% off their whole bill, and nobody else got any discount.",
         yes2="{x} and {y} have loyalty cards", yes1="{x} has a loyalty card", no1="{z} does not have one",
         no2="neither {z} nor {w} has one"),
    dict(rule="Members of the walking club got {pct}% off their whole bill, and nobody else got any discount.",
         yes2="{x} and {y} are members of the walking club", yes1="{x} is a member of the walking club",
         no1="{z} is not a member", no2="{z} and {w} are not members"),
]
NOTES = [500, 1000, 2000, 5000, 10000]


def total(f, basket):
    """Before any discount: deal items in as many threes as possible, then singles."""
    out = 0
    for item, n in basket:
        if item == f.deal_item:
            out += (n // 3) * f.deal + (n % 3) * f.price[item]
        else:
            out += n * f.price[item]
    return out


def payable(f, who, basket):
    """Every amount this person could have paid for the basket: one if their status is known, two if not."""
    full = total(f, basket)
    disc = full * (100 - f.pct) // 100
    if who in f.members:
        return {disc}
    if who in f.others or not f.pct:
        return {full}
    return {full, disc}


def paid_ok(f, who, basket, x):
    return x in payable(f, who, basket)


def change_ok(f, who, basket, note, c):
    return any(note - x == c for x in payable(f, who, basket) if x <= note)


def sl(claim, *key):
    """Set the single-valued fact a claim is about; build never gives the liar a true claim with the lie's key."""
    claim.key = key
    return claim


def basket_text(basket, f_names):
    parts = [an(f_names[i][0]) if n == 1 else f"{word(n)} {f_names[i][1]}" for i, n in basket]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def make(rng, people, others, day):
    shop = rng.choice(sorted(SHOPS))
    names = rng.sample(SHOPS[shop], 3)
    cur = rng.choice(["£", "$"])
    note_word = "note" if cur == "£" else "bill"
    cash = lambda c: money(c, cur)  # noqa: E731
    sp = list(people)
    rng.shuffle(sp)
    deal_variant = rng.random() < 0.4
    price = {i: 20 * rng.randint(names[i][2] // 20, names[i][3] // 20) for i in range(3)}  # keeps discounts whole
    if deal_variant:
        pct, members, nonmem = 0, set(), set(sp)
        deal_item = 0
        p = price[0]
        deal = 20 * rng.randint(max(3, (3 * p) // 40), (3 * p) // 20 - 1)  # cheaper than three singles
        ev = [f"At the {shop} on {date_text(day)}, {names[0][1]} cost {cash(p)} each, or {cash(deal)} for any three, "
              f"and the till always charged as many threes as possible at the deal price.",
              f"{names[1][1].capitalize()} cost {cash(price[1])} each, with no deal.",
              "Everyone paid in cash and was given the correct change."]
        items = [0, 1]
    else:
        pct = rng.choice([10, 20, 25, 50])
        deal_item, deal = None, None
        g = rng.choice(GROUPS)
        k_mem = rng.choice([1, 2])
        unknown = rng.random() < 0.6
        members = set(sp[:k_mem])
        nonmem = set(sp[k_mem: 4 - unknown])
        m, nm = sorted(members), sorted(nonmem)
        yes = g["yes2"].format(x=m[0], y=m[1]) if len(m) == 2 else g["yes1"].format(x=m[0])
        no = g["no2"].format(z=nm[0], w=nm[1]) if len(nm) == 2 else g["no1"].format(z=nm[0])
        listed = [f"{an(names[i][0])} cost {cash(price[i])}" for i in range(3)]
        ev = [f"At the {shop} on {date_text(day)}, {listed[0]}, {listed[1]} and {listed[2]}.",
              g["rule"].format(pct=pct), f"{yes}, and {no}.",
              "Everyone paid in cash and was given the correct change."]
        items = [0, 1, 2]
    f = facts(price=price, pct=pct, members=members, others=nonmem, deal_item=deal_item, deal=deal)
    # hidden world: each speaker's basket and status (an unknown status gets one), paid with the smallest note above it
    status = {x: (x in members) or (x not in nonmem and rng.random() < 0.5) for x in sp}
    basket = {}
    for x in sp:
        if deal_variant:
            b = [(0, rng.randint(1, 7))] + ([(1, rng.randint(1, 2))] if rng.random() < 0.4 else [])
        else:
            b = sorted((i, rng.choice([1, 1, 2])) for i in rng.sample(items, rng.randint(1, 3)))
        basket[x] = tuple(b)
    if not deal_variant:  # one non-member bought exactly what one member bought
        mm, nn = rng.choice(sorted(members)), rng.choice(sorted(nonmem))
        basket[nn] = basket[mm]
    elif deal < 2 * price[0]:  # this deal makes three cheaper than two; two speakers bought exactly those
        three_by, two_by = rng.sample(sp, 2)
        basket[three_by], basket[two_by] = ((0, 3),), ((0, 2),)

    def pay(x):
        full = total(f, basket[x])
        return full * (100 - pct) // 100 if status[x] else full

    def note_for(x):
        return next(n for n in NOTES if n > pay(x))

    bt = lambda x: basket_text(basket[x], names)  # noqa: E731
    paid_t = lambda x, amount: rng.choice([f"I paid {cash(amount)} for {bt(x)}.",  # noqa: E731
                                           f"{bt(x)[0].upper()}{bt(x)[1:]} cost me {cash(amount)} altogether."])
    change_t = lambda x, note, c: rng.choice([  # noqa: E731
        f"I paid for {bt(x)} with a {cash(note)[:-3]} {note_word} and got {cash(c)} back.",
        f"I handed over a {cash(note)[:-3]} {note_word} for {bt(x)} and got {cash(c)} in change."])
    other_t = lambda x, amount: rng.choice([f"{x} paid {cash(amount)} for {bt(x)}.",  # noqa: E731
                                            f"I saw {x} pay {cash(amount)} for {bt(x)}."])
    truths = []
    for x in sp:
        tags = ("distractor",) if x not in members and x not in nonmem and status[x] else ()
        truths.append(sl(Claim(x, paid_t(x, pay(x)), lambda f, x=x, b=basket[x], v=pay(x): paid_ok(f, x, b, v), tags),
                         "pay", x))
    for x in rng.sample(sp, 2):
        n = note_for(x)
        truths.append(sl(Claim(x, change_t(x, n, n - pay(x)),
                               lambda f, x=x, b=basket[x], n=n, c=n - pay(x): change_ok(f, x, b, n, c)), "pay", x))
    for x in rng.sample(sp, 3):
        s = rng.choice([y for y in sp if y != x])
        truths.append(sl(Claim(s, other_t(x, pay(x)), lambda f, x=x, b=basket[x], v=pay(x): paid_ok(f, x, b, v)),
                         "pay", x))
    i = rng.choice(items)
    truths.append(Claim(rng.choice(sp), rng.choice([f"{an(names[i][0]).capitalize()} was {cash(price[i])} before any "
                                                    f"discount.", f"A single {names[i][0]} was {cash(price[i])}."])
                        if not deal_variant else f"A single {names[i][0]} was {cash(price[i])}.",
                        lambda f, i=i, v=price[i]: f.price[i] == v))
    truths[-1].key = ("list", i)
    if not deal_variant:
        truths.append(sl(Claim(nn, f"{mm} bought exactly what I bought but paid less.",
                               lambda f, a=mm, b=nn: a not in f.others and b not in f.members and f.pct > 0,
                               ("distractor",)), "pay", nn))
    else:
        if deal < 2 * price[0]:  # true, and sounds wrong
            truths.append(sl(Claim(three_by, f"Three {names[0][1]} cost me less than two cost {two_by}.",
                                   lambda f: total(f, ((0, 3),)) < total(f, ((0, 2),)), ("distractor",)),
                             "pay", three_by))
        truths.append(Claim(rng.choice(sp), f"Three {names[0][1]} together came to {cash(deal)}, not "
                                            f"{cash(3 * price[0])}.",
                            lambda f, v=deal: total(f, ((0, 3),)) == v, ("distractor",)))
    for text in rng.sample([f"I sat at table {rng.randint(2, 19)}.", f"I came on the number {rng.randint(3, 99)} bus.",
                            f"My receipt number was {rng.randint(100, 9999)}.",
                            f"The queue had about {rng.randint(3, 9)} people in it when I arrived.",
                            f"I waited about {rng.randint(3, 12)} minutes to be served."], 4):
        truths.append(Claim(rng.choice(sp), text, lambda f: True, ("neutral",)))

    kinds = ["change", "about_other", "list_price"] + (["nodeal", "overdeal"] if deal_variant else
                                                       ["member_full", "nonmember_disc"])
    kind = rng.choice(kinds)
    known = sorted(members | nonmem)
    if kind == "member_full":
        x = rng.choice(sorted(members))
        v = total(f, basket[x])
        lie = Claim(x, paid_t(x, v), lambda f, x=x, b=basket[x], v=v: paid_ok(f, x, b, v), ("two-fact",))
    elif kind == "nonmember_disc":
        x = rng.choice(sorted(nonmem))
        v = total(f, basket[x]) * (100 - pct) // 100
        lie = Claim(x, paid_t(x, v), lambda f, x=x, b=basket[x], v=v: paid_ok(f, x, b, v), ("two-fact",))
    elif kind == "change":
        x = rng.choice(sp)
        n = note_for(x)
        ok_set = {n - v for v in payable(f, x, basket[x]) if v <= n}
        c = rng.choice([y for y in (n - pay(x) + d for d in (-100, -50, -40, -20, -10, 10, 20, 40, 50, 100))
                        if y >= 0 and y not in ok_set])
        lie = Claim(x, change_t(x, n, c), lambda f, x=x, b=basket[x], n=n, c=c: change_ok(f, x, b, n, c),
                    ("two-fact",))
    elif kind == "about_other":
        x = rng.choice(known)
        right = payable(f, x, basket[x])
        full = total(f, basket[x])
        v = rng.choice([y for y in {full, full * (100 - pct) // 100, pay(x) + 20, pay(x) - 20,
                                    full + (f.price[0] if deal_variant else 0)} if y > 0 and y not in right])
        s = rng.choice([y for y in sp if y != x])
        lie = Claim(s, other_t(x, v), lambda f, x=x, b=basket[x], v=v: paid_ok(f, x, b, v),
                    ("two-fact", "about-other"))
    elif kind == "list_price":  # never the discounted single price, which a member could have meant
        i = rng.choice(items)
        v = rng.choice([price[i] + d for d in (-40, -20, 20, 40, 60) if price[i] + d != price[i] * (100 - pct) // 100])
        lie = Claim(rng.choice(sp), f"Before any discount, a single {names[i][0]} was {cash(v)}." if pct else
                    f"A single {names[i][0]} was {cash(v)}.", lambda f, i=i, v=v: f.price[i] == v)
    else:
        x = rng.choice(sp)
        n = rng.choice(([4, 7] + ([5] if deal != 2 * price[0] else [])) if kind == "overdeal" else [3, 4, 5, 6])
        v = -(-n // 3) * deal if kind == "overdeal" else n * price[0]
        lie = Claim(x, f"I bought {word(n)} {names[0][1]} for {cash(v)}.",
                    lambda f, n=n, v=v: total(f, ((0, n),)) == v, ("two-fact",))
    if kind in ("member_full", "nonmember_disc", "change", "about_other", "nodeal", "overdeal"):
        lie.key = ("pay", x)
    else:
        lie.key = ("list", i)
    group = rng.choice(["Friends", "Colleagues", "Flatmates", "Classmates"])
    return Bundle(
        NAME,
        rng.choice([f"{group} argue over what they paid at the {shop}.",
                    f"{group} check their change from the {shop}.",
                    f"{group} compare what the {shop} charged them."]),
        group, f"what the {shop} charged", ev, f, lie, truths,
        fillers=["Do they take cards?", "Who keeps the receipts?", "Prices there keep going up.",
                 f"Is the {shop} open on Sundays?", "That place is always busy."],
        decor=[f"The {shop} has a green awning.", "The till is next to the door.",
               "The price board is written in chalk."],
        timed=False)
