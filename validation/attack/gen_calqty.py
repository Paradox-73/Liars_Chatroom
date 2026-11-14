"""Writes cases_calqty.csv: hand-written cases for the calendar and quantity rule families in logic that train lacks."""
import csv
import os

POOL = ("1 April 2026 was a Wednesday. In April 2026 the pool was open only {days}, from {o} to {c}, "
        "and nobody could use it while it was closed.")
LIB = "1 April 2026 was a Wednesday. Nobody could be inside the library on any Tuesday in April 2026{q}."
AGE = "Sana was born on 14 July 1993, and Ishaan was born on 27 March 1998."
KAYAK = ("Camila owned the kayak until selling it to Tomas on 8 September 2026. Tomas sold it to Isla on "
         "20 September 2026. Nobody else owned the kayak in September 2026.")
SEATS = ("All times in this case refer to 4 March 2026. Exactly 60 seats were sold for the workshop, and exactly 75% of "
         "them were sold before noon.")
BUTT = ("All times in this case refer to 4 March 2026. The water butt held 60 litres at 10:00 AM and drained at a steady "
        "6 litres per hour until 2:00 PM, with nothing added.")
ROPE5 = "The rope was exactly 3 metres long and was cut into five equal pieces, with nothing left over."
ROPE3 = ("The rope was exactly 3 metres long. It was cut into three pieces with nothing left over: one long piece twice "
         "as long as each of the two short pieces, which were equal.")
DRILL = ("All times in this case refer to 15 May 2026. The roll call sheet for the 10:45 AM fire drill lists exactly four "
         "people: Jun, Rahul, Gita and Yusuf. {both} The fire drill was held in the main hall.")
BOTH = "Everyone on the list attended, and everyone who attended is on it."
TOUR = "All times in this case refer to 21 February 2026. That day's only tour began at 12:45 PM and lasted exactly {d}."
N4 = "A = Mateo; B = Maria; C = Zeynep; D = Carmen"
DRILL_SP = "A = Jun; B = Rahul; C = Anika; D = Mira"
POOL_STD = POOL.format(days="Wednesday to Sunday", o="9:00 AM", c="5:00 PM")

CASES = [  # (scenario, speakers, evidence, [(label, text)], liar); the comment says why the label holds
    # 01: 5:00 PM is inside "from 9:00 AM to 5:00 PM"; the liar swam on a Tuesday (closed every Tuesday)
    ("Pool hours end point.", N4, POOL_STD,
     [("A", "I swam at the pool at 5:00 PM on 11 April 2026."), ("B", "On 16 April 2026 at 10:00 AM I swam at the pool."),
      ("C", "I swam at the pool on a Tuesday afternoon in April 2026."), ("D", "I waited outside the pool on 13 April 2026.")], "C"),
    # 02: Friday to Monday wraps over the weekend: 6 April (Monday) and 3 April (Friday) are open days
    ("Pool open Friday to Monday.", N4, POOL.format(days="Friday to Monday", o="9:00 AM", c="5:00 PM"),
     [("A", "I swam at the pool at 10:00 AM on 6 April 2026."), ("B", "I swam at the pool at 3:00 PM on 3 April 2026."),
      ("C", "I swam at the pool on a Wednesday in April 2026."), ("D", "On 8 April 2026 I waited outside the pool.")], "C"),
    # 03: 29 February 2028 is a Tuesday (open); 26 February 2028 is a Saturday (closed)
    ("Gym in a leap year.", N4, "1 February 2028 was a Tuesday. In February 2028 the gym was open only Monday to Friday, "
     "from 9:00 AM to 5:00 PM, and nobody could use it while it was closed.",
     [("A", "I trained at the gym at 10:00 AM on 29 February 2028."), ("B", "On 1 February 2028 I trained at the gym at 4:00 PM."),
      ("C", "I trained at the gym at 11:00 AM on 26 February 2028."), ("D", "I waited outside the gym on 27 February 2028.")], "C"),
    # 04: 24-hour clock: 17:45 is after the 5:00 PM close; 16:30 is inside
    ("Pool with a 24-hour clock.", N4, POOL_STD,
     [("A", "I swam at the pool at 16:30 on 11 April 2026."), ("B", "On 16 April 2026 at 10:00 AM I swam at the pool."),
      ("C", "I swam at the pool at 17:45 on 11 April 2026."), ("D", "I waited outside the pool on 13 April 2026.")], "C"),
    # 05: opening at 12:00 PM (noon): 12:15 PM is open, 11:45 AM is not
    ("Pool opens at noon.", N4, POOL.format(days="Wednesday to Sunday", o="12:00 PM", c="8:00 PM"),
     [("A", "I swam at the pool at 12:15 PM on 11 April 2026."), ("B", "On 16 April 2026 at 7:30 PM I swam at the pool."),
      ("C", "I swam at the pool at 11:45 AM on 11 April 2026."), ("D", "I waited outside the pool on 13 April 2026.")], "C"),
    # 06: the closure is only before 1:00 PM: 3:00 PM on Tuesday 14 April is fine; the liar was in on a Tuesday morning
    ("Library closed on Tuesday mornings.", N4, LIB.format(q=" before 1:00 PM"),
     [("A", "I was inside the library at 3:00 PM on 14 April 2026."), ("B", "I was inside the library on 16 April 2026."),
      ("C", "I was inside the library on the morning of 21 April 2026."), ("D", "I waited outside the library on 7 April 2026.")], "C"),
    # 07: 31 March 2026 is a Tuesday but March is not covered; the last Tuesday of April (28th) is
    ("Library across a month end.", N4, LIB.format(q=""),
     [("A", "I was inside the library on 31 March 2026."), ("B", "I was inside the library on 16 April 2026."),
      ("C", "I was inside the library on the last Tuesday of April 2026."), ("D", "I waited outside the library on 7 April 2026.")], "C"),
    # 08: Sana turns 33 on 14 July 2026, so she is 33 that day; Ishaan is 27 until 27 March 2026; C lies about Sana
    ("Ages on a birthday.", "A = Sana; B = Ishaan; C = Lucy; D = Omar", AGE,
     [("A", "On 14 July 2026 I was 33."), ("B", "On 26 March 2026 I was 27."), ("C", "Sana turned 32 in July 2026."),
      ("D", "Who is organising the party?")], "C"),
    # 09: born 29 February 2000: 26 in March 2026; Bruno born 19 October 2002 turns 24 in October 2026, not 23
    ("Leap-day birthday.", "A = Pia; B = Bruno; C = Lucy; D = Omar",
     "Pia was born on 29 February 2000. Bruno was born on 19 October 2002.",
     [("A", "In March 2026 I was 26."), ("B", "I turned 23 in October 2026."), ("C", "On 1 March 2026 Pia was 26."),
      ("D", "Who is organising the party?")], "B"),
    # 10: Tomas owned it from 8 to 20 September, so Isla did not own it on 12 September; October and August not covered
    ("Kayak owners, claim about someone else.", "A = Tomas; B = Isla; C = Hari; D = Camila", KAYAK,
     [("A", "I owned the kayak on 15 September 2026."), ("B", "I owned the kayak on 1 October 2026."),
      ("C", "Isla owned the kayak on 12 September 2026."), ("D", "I owned the kayak on 30 August 2026.")], "C"),
    # 11: Camila sold on 8 September, so she did not own it on 10 September
    ("Kayak owners, seller.", "A = Tomas; B = Isla; C = Hari; D = Camila", KAYAK,
     [("A", "I sold the kayak to Isla on 20 September 2026."), ("B", "I bought the kayak from Tomas on 20 September 2026."),
      ("C", "I never owned the kayak in September 2026."), ("D", "I still owned the kayak on 10 September 2026.")], "D"),
    # 12: 45 sold before noon, 15 after: "at least 40" holds; "most after noon" does not
    ("Seats with at least.", N4, SEATS,
     [("A", "At least 40 seats were sold before noon."), ("B", "After noon, 15 seats were sold."),
      ("C", "Most of the seats were sold after noon."), ("D", "I bought mine at 9:50 AM.")], "C"),
    # 13: "more than 10 after noon" holds (15); a third before noon is 20, not 45
    ("Seats with more than.", N4, SEATS,
     [("A", "More than 10 seats were sold after noon."), ("B", "Before noon, 45 seats were sold."),
      ("C", "Only a third of the seats were sold before noon."), ("D", "I bought mine at 9:50 AM.")], "C"),
    # 14: a quarter after noon is 15 (true); two thirds before noon is 40, not 45
    ("Seats as fractions.", N4, SEATS,
     [("A", "Exactly a quarter of the seats were sold after noon."), ("B", "After noon, 15 seats were sold."),
      ("C", "Two thirds of the seats were sold before noon."), ("D", "I bought mine at 9:50 AM.")], "C"),
    # 15: 15 sold after noon: "fewer than 20" holds; "no more than 10" does not
    ("Seats with fewer than.", N4, SEATS,
     [("A", "Fewer than 20 seats were sold after noon."), ("B", "Before noon, 45 seats were sold."),
      ("C", "No more than 10 seats were sold after noon."), ("D", "I bought mine at 9:50 AM.")], "C"),
    # 16: noon: 48 litres = 48,000 ml; 1:00 PM: 42; the butt fell to 36 litres by 2:00 PM, so a floor of 50 is false
    ("Water butt in millilitres.", N4, BUTT,
     [("A", "At 12:00 PM the water butt held 48,000 millilitres."), ("B", "At 1:00 PM the water butt held 42 litres."),
      ("C", "The water butt never dropped below 50 litres before 2:00 PM."), ("D", "I checked the water butt at 11:00 AM.")], "C"),
    # 17: 11:00 AM: 54 litres, so "at least 50" holds; by 1:00 PM it lost 18 litres, not 30
    ("Water butt with at least.", N4, BUTT,
     [("A", "At 11:00 AM the water butt held at least 50 litres."), ("B", "At 1:00 PM the water butt held 42 litres."),
      ("C", "By 1:00 PM the water butt had lost 30 litres."), ("D", "I checked the water butt at 11:00 AM.")], "C"),
    # 18: noon: 48 litres is between 45 and 50; 13:00 is 1:00 PM (42 litres); at 2:00 PM it held 36, not 40
    ("Water butt with ranges.", N4, BUTT,
     [("A", "At noon the water butt held between 45 and 50 litres."), ("B", "At 13:00 the water butt held 42 litres."),
      ("C", "At 2:00 PM the water butt was down to 40 litres."), ("D", "I checked the water butt at 11:00 AM.")], "C"),
    # 19: 12 kg less 500 g = 11.5 kg = 11,500 g from 11:00 AM; 11 kg at 2:00 PM is false
    ("Rice bin in grams.", N4, "All times in this case refer to 4 March 2026. The rice bin held exactly 12 kilograms at "
     "9:00 AM. At 11:00 AM exactly 500 grams were removed. Nothing else was added or removed before 3:00 PM.",
     [("A", "At noon the rice bin held 11.5 kilograms."), ("B", "At 10:00 AM the rice bin held 12000 grams."),
      ("C", "At 2:00 PM the rice bin held 11 kilograms."), ("D", "At 1:00 PM the rice bin held 11500 grams.")], "C"),
    # 20: each piece is 60 cm: "longer than 50 cm" holds; "shorter than half a metre" does not
    ("Rope pieces with comparisons.", N4, ROPE5,
     [("A", "My piece measured 0.6 metres."), ("B", "My piece measured 600 millimetres."),
      ("C", "My piece was shorter than half a metre."), ("D", "My piece was longer than 50 centimetres.")], "C"),
    # 21: long 1.5 m, short 0.75 m each: the long one is twice, not three times, a short one
    ("Rope long and short.", N4, ROPE3,
     [("A", "The long piece was 1.5 metres."), ("B", "Each short piece was 75 cm."),
      ("C", "The long piece was three times as long as a short piece."), ("D", "I was the one who cut the rope.")], "C"),
    # 22: without "everyone who attended is on it" an unlisted speaker may have attended; Rahul (listed) did attend
    ("Fire drill, one-way list.", DRILL_SP, DRILL.format(both="Everyone on the list attended."),
     [("A", "I attended the 10:45 AM fire drill."), ("B", "I skipped the 10:45 AM fire drill."),
      ("C", "I attended the 10:45 AM fire drill."), ("D", "Gita attended the 10:45 AM fire drill.")], "B"),
    # 23: leaving early or arriving late is still attending; Anika is not listed so she did not take part
    ("Fire drill, late and early.", DRILL_SP, DRILL.format(both=BOTH),
     [("A", "I left the 10:45 AM fire drill early."), ("B", "I arrived late to the fire drill."),
      ("C", "I took part in the fire drill in the main hall."), ("D", "I was in the car park at 10:45 AM.")], "C"),
    # 24: double negation about someone else is true; Yusuf attended, so "never came in" is false
    ("Fire drill, double negation.", DRILL_SP, DRILL.format(both=BOTH),
     [("A", "I attended the 10:45 AM fire drill."), ("B", "I was there for the whole fire drill."),
      ("C", "Yusuf was off sick that day and never came in."), ("D", "Jun did not miss the 10:45 AM fire drill.")], "C"),
    # 25: 90 minutes from 12:45 PM ends at 2:15 PM, so 2:10 PM is inside; three hours after the start is not
    ("Tour of 90 minutes.", N4, TOUR.format(d="90 minutes"),
     [("A", "At 2:10 PM I led the group during the tour."), ("B", "I led the group during the tour at 12:50 PM."),
      ("C", "I led the group during the tour three hours after it began."), ("D", "I checked the museum at 3:00 PM.")], "C"),
    # 26: tour 12:45-1:45 PM: not on the tour at 2:00 PM is true; the liar claims two full hours
    ("Tour with a negation.", N4, TOUR.format(d="1 hour"),
     [("A", "I was not on the tour at 2:00 PM."), ("B", "I led the group during the tour at 1:00 PM."),
      ("C", "I led the group for the whole two hours of the tour."), ("D", "I checked the museum at 3:00 PM.")], "C"),
    # 27: the tour starts at 12:45 PM, so it was running at 12:45 PM; 2:30 PM is after its end
    ("Tour start point.", N4, TOUR.format(d="1 hour"),
     [("A", "I led the group during the tour at 12:45 PM."), ("B", "I led the group during the tour at 1:30 PM."),
      ("C", "I led the group during the tour at 2:30 PM."), ("D", "I checked the museum at 2:30 PM.")], "C"),
    # 28: rota across a month end: Usha 30 Apr-1 May, Jonas 2-3 May, Akira 4-5 May
    ("Rota across a month end.", "A = Jonas; B = Usha; C = Akira; D = Lena",
     "From 30 April 2026 to 5 May 2026, exactly one person worked the weekend call-out duty each night, following this "
     "rota exactly: Usha took the first two, Jonas took the next two, and Akira took the last two.",
     [("A", "I worked the weekend call-out duty on 2 May 2026."), ("B", "I worked the weekend call-out duty on 1 May 2026."),
      ("C", "I worked the weekend call-out duty on 3 May 2026."), ("D", "I worked the day shift on 30 April 2026.")], "C"),
    # 29: the liar talks about someone else: Mei had 18 September, not Obi
    ("Rota, claim about someone else.", "A = Eun; B = Obi; C = Lena; D = Kofi",
     "From 13 September 2026 to 19 September 2026, exactly one person worked the front desk night duty each night, and "
     "the rota was followed exactly: Eun on 13-15 September, Obi on 16-17 September, Mei on 18-19 September.",
     [("A", "I worked the front desk night duty on 14 September 2026."), ("B", "On 17 September 2026 I worked the front desk night duty."),
      ("C", "Obi worked the front desk night duty on 18 September 2026."), ("D", "I worked the day shift on 18 September 2026.")], "C"),
    # 30: "not before 9:00 AM" is true (it opened at 9); the liar swam at 5:30 PM, after closing
    ("Pool with not before.", N4, POOL_STD,
     [("A", "I did not swim at the pool before 9:00 AM on 11 April 2026."), ("B", "I swam at the pool at 10:00 AM on 16 April 2026."),
      ("C", "I swam at the pool at 5:30 PM on 11 April 2026."), ("D", "I waited outside the pool on 13 April 2026.")], "C"),
    # 31-33: the liar uses a clock form the parser cannot read (after the 5:00 PM close on an open day); D is innocent
    # and unjudged, so a day-only OK verdict on the liar would hand D the answer by elimination
    ("Pool, 24-hour liar.", N4, POOL_STD,
     [("A", "I swam at the pool at 10:00 AM on 11 April 2026."), ("B", "On 16 April 2026 at 10:00 AM I swam at the pool."),
      ("C", "I swam at the pool at 17:45 on 11 April 2026."), ("D", "I drove past the pool on 13 April 2026.")], "C"),
    ("Pool, p.m. liar.", N4, POOL_STD,
     [("A", "I swam at the pool at 10:00 AM on 11 April 2026."), ("B", "On 16 April 2026 at 10:00 AM I swam at the pool."),
      ("C", "I swam at the pool at 7:30 p.m. on 11 April 2026."), ("D", "I drove past the pool on 13 April 2026.")], "C"),
    # 'at night' on an open day: the pool closes at 5:00 PM
    ("Pool, at night.", N4, POOL_STD,
     [("A", "I swam at the pool at 10:00 AM on 11 April 2026."), ("B", "On 16 April 2026 at 10:00 AM I swam at the pool."),
      ("C", "I swam at the pool at night on 11 April 2026."), ("D", "I drove past the pool on 13 April 2026.")], "C"),
    # 34-35: a swim that starts in opening hours but runs past the 5:00 PM close is a use while closed
    ("Pool, swim runs past closing.", N4, POOL_STD,
     [("A", "I swam at the pool at 10:00 AM on 11 April 2026."), ("B", "On 16 April 2026 at 10:00 AM I swam at the pool."),
      ("C", "I swam at the pool from 4:30 PM on 11 April 2026 for two hours."), ("D", "I drove past the pool on 13 April 2026.")], "C"),
    ("Pool, swim starting at 3:00 PM.", N4, POOL_STD,
     [("A", "I swam at the pool at 10:00 AM on 11 April 2026."), ("B", "On 16 April 2026 at 10:00 AM I swam at the pool."),
      ("C", "I swam at the pool for three hours starting at 3:00 PM on 11 April 2026."),
      ("D", "I drove past the pool on 13 April 2026.")], "C"),
    # 36: the tour ended at 1:45 PM, so leading it for an hour from 1:30 PM is impossible; D's claim is unjudged
    ("Tour, claim that outlasts it.", N4, TOUR.format(d="1 hour"),
     [("A", "I led the group during the tour at 1:00 PM."), ("B", "I led the group during the tour at 1:15 PM."),
      ("C", "I led the group during the tour at 1:30 PM for an hour."), ("D", "I checked the museum at 3:00 PM.")], "C"),
]

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases_calqty.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "scenario", "speakers", "evidence", "chat", "label"])
    for i, (sc, sp, ev, msgs, liar) in enumerate(CASES, 1):
        w.writerow([f"ATK_CQ_{i:02d}", sc, sp, ev, "\n".join(f"{n}. {lab}: {t}" for n, (lab, t) in enumerate(msgs, 1)), liar])
print(len(CASES), "cases ->", out)
