"""Writes cases_access.csv: new access-style cases whose logic the access rules were not built for.
Each case: (scenario, speakers, evidence, [(sender, text)], label). The comment above each case says why the label
is right under the task rules."""
import csv
import os

D = "All times in this case refer to 14 May 2026. "
SP4 = "A = Ravi; B = Mia; C = Leo; D = Zoe"
ARR = D + ("The door log accurately records when each of these four people first arrived at the office that day: "
           "Ravi at 8:30 AM, Mia at 8:45 AM, Leo at 9:00 AM and Zoe at 9:10 AM.")
WIN = D + "Nobody could enter the greenhouse between 9:00 AM and 11:00 AM."
INW = D + "Nobody could be in the greenhouse between 9:00 AM and 11:00 AM."
CODE = D + ("Before 10:15 AM, only code 1764 could unlock the studio door. From 10:15 AM to the end of the day, "
            "only code 6139 could unlock it.")
PRES = D + ("From 2:30 PM to 6:30 PM, only Ravi and Mia were at the security gate. Nobody else was at the security "
            "gate during that time.")
LIFT = D + "The lift could run only while the main power was on. The main power was off from 5:15 PM to 5:45 PM."
LOG = D + ("The archive badge log accurately records every entry to the archive that day. It shows exactly two "
           "entries: Leo at 9:30 AM and Zoe at 11:45 AM.")
BADGE = D + ("Nobody could enter the pool area except by scanning their own active wristband. Leo's wristband was "
             "inactive before 10:05 AM, and active from 10:05 AM onward.")
BOOK = D + ("The main hall was booked from 11:35 AM to 4:05 PM: the sales team had it until 1:05 PM, the finance team "
            "until 2:35 PM, and then the research team. During each booking, only that team was in the room.")
VPN = D + ("Every remote sign-in to the archive server required the VPN, and there was no other remote route. "
           "The VPN service was offline from 1:00 PM to 2:00 PM.")
LIST = D + ("Mia and Zoe are on the safety-trained list, and Leo is on the volunteer rota. Nobody on the volunteer "
            "rota is on the safety-trained list. Before 9:45 AM the lab door opened only for people on the "
            "safety-trained list, and it was the only way into the lab.")
SHED = D + "Nobody could enter the carpentry shed between 4:00 PM and 5:00 PM."
WIN_FT = D + "Nobody could enter the greenhouse from 9:00 AM to 11:00 AM."

CASES = [
    # 01 "didn't go in until 10:00 AM, when I went in" = went in at 10:00 AM, inside the closed window
    ("Gardeners argue over the greenhouse.", SP4, WIN,
     [("A", "I didn't go into the greenhouse until 10:00 AM, when I went in to water the tomatoes."),
      ("B", "I went into the greenhouse at 11:30 AM."), ("C", "I was waiting outside the greenhouse at 9:30 AM."),
      ("D", "I watered the lawn at 12:15 PM."), ("B", "Who has the hose?")], "A"),
    # 02 coming back later is not the first arrival (true); Leo was third of the four, not second
    ("Office staff argue over who came in when.", SP4, ARR,
     [("A", "I arrived at the office again at 1:30 PM."), ("B", "My first arrival at the office was at 8:45 AM."),
      ("C", "I was the second of us four to arrive at the office."), ("D", "I first arrived at the office at 9:10 AM.")],
     "C"),
    # 03 Leo came 15 minutes after Mia, not twenty
    ("Office staff argue over who came in when.", SP4, ARR,
     [("C", "I got to the office twenty minutes after Mia."), ("A", "My first arrival at the office was at 8:30 AM."),
      ("B", "I first arrived at the office at 8:45 AM."), ("D", "At 12:00 PM I ate lunch at my desk.")], "C"),
    # 04 coming to the office for an evening meeting is a later arrival (true); Ravi (8:30) beat Mia (8:45)
    ("Office staff argue over who came in when.", SP4, ARR,
     [("A", "I came to the office at 5:00 PM for the evening meeting."),
      ("B", "I was the first of us four to arrive at the office."), ("C", "I first arrived at the office at 9:00 AM."),
      ("D", "My first arrival at the office was at 9:10 AM.")], "B"),
    # 05 typing a code that does not work is possible when the door stays shut; the door opening to 1764 after
    # 10:15 AM is not
    ("Studio members argue over the door codes.", SP4, CODE,
     [("A", "At 9:30 AM I typed code 6139 to unlock the studio door, but it stayed shut."),
      ("B", "At 11:00 AM the studio door opened for me when I keyed in 1764."),
      ("C", "I unlocked the studio door with code 6139 at 12:00 PM."),
      ("D", "At 9:40 AM I used code 1764 to unlock the studio door.")], "B"),
    # 06 at the gate from 2:15 PM for an hour = there at 2:30-3:15 PM; Leo is not Ravi or Mia
    ("Guards argue over who stood at the gate.", SP4, PRES,
     [("C", "I was at the security gate at 2:15 PM for an hour."), ("A", "I was at the security gate at 4:00 PM."),
      ("B", "I was at the security gate at 5:00 PM."), ("D", "I bought a sandwich at 3:00 PM.")], "C"),
    # 07 in the greenhouse from 8:30 AM for two hours = in it at 9:00-10:30 AM
    ("Gardeners argue over the greenhouse.", SP4, INW,
     [("A", "I was in the greenhouse at 8:30 AM for two hours."), ("B", "I was in the greenhouse at 11:30 AM."),
      ("C", "I was in the greenhouse at 8:00 AM."), ("D", "I repotted the ferns in the shed at 10:00 AM.")], "A"),
    # 08 the lift running from 5:00 PM for an hour covers the 5:15-5:45 PM power cut
    ("Residents argue over the lift.", SP4, LIFT,
     [("B", "The lift was running at 5:00 PM for an hour."), ("A", "I took the lift at 6:00 PM."),
      ("C", "I took the lift at 5:00 PM."), ("D", "At 5:30 PM I carried the boxes up the stairs.")], "B"),
    # 09 the complete log has one entry for Leo, so "twice" is false
    ("Archivists argue over who went into the archive.", SP4, LOG,
     [("C", "I went into the archive twice, the first time at 9:30 AM."),
      ("D", "I went into the archive at 11:45 AM."), ("A", "I didn't go into the archive at all."),
      ("B", "At 10:00 AM I sorted the post at reception.")], "C"),
    # 10 entering with someone else's wristband is not entering by scanning one's own active wristband
    ("Swimmers argue over the pool area.", SP4, BADGE,
     [("A", "I saw Leo go into the pool area at 10:30 AM using Zoe's wristband."),
      ("B", "I saw Leo go into the pool area at 11:00 AM."), ("D", "I saw Leo go into the pool area at 12:00 PM."),
      ("C", "I swam lengths at 1:00 PM.")], "A"),
    # 11 the sales team in the hall from 12:40 PM for three hours would be there in the finance booking
    ("Staff argue over the main hall bookings.", SP4, BOOK,
     [("D", "The sales team was in the main hall at 12:40 PM for three hours."),
      ("A", "The finance team was in the main hall at 1:35 PM."),
      ("B", "At 3:20 PM the research team was in the main hall."), ("C", "I booked a taxi at 4:30 PM.")], "D"),
    # 12 control: entering at 3:30 PM and staying is allowed (only entering is barred); a 4:59 PM entry is not
    ("Carpenters argue over the shed.", SP4, SHED,
     [("B", "I went into the carpentry shed at 3:30 PM for an hour."),
      ("A", "I saw Leo go into the carpentry shed at 4:59 PM."), ("C", "I went into the carpentry shed at 5:10 PM."),
      ("D", "I was waiting outside the carpentry shed at 4:30 PM.")], "A"),
    # 13 Leo first arrived 15 minutes after Mia, not 5
    ("Office staff argue over who came in when.", SP4, ARR,
     [("C", "I first arrived at the office 5 minutes after Mia."), ("A", "I first arrived at the office at 8:30 AM."),
      ("B", "My first arrival at the office was at 8:45 AM."), ("D", "I made tea at 9:30 AM.")], "C"),
    # 14 "did go in then" = entered at 10:30 AM, inside the window
    ("Gardeners argue over the greenhouse.", SP4, WIN,
     [("B", "I didn't enter the greenhouse before 10:30 AM, but I did go in then."),
      ("A", "I went into the greenhouse at 11:15 AM."), ("C", "I went into the greenhouse at 8:45 AM."),
      ("D", "I mowed the lawn at 10:00 AM.")], "B"),
    # 15 control: a liar about someone else; only 6139 worked after 10:15 AM
    ("Studio members argue over the door codes.", SP4, CODE,
     [("A", "I saw Zoe unlock the studio door with code 1764 at 11:00 AM."),
      ("B", "I unlocked the studio door with code 6139 at 10:20 AM."),
      ("C", "At 9:00 AM I used code 1764 to unlock the studio door."), ("D", "Who changed the code?")], "A"),
    # 16 control: signing in before the outage and working on is not a remote sign-in during it
    ("Archivists argue over server sign-ins.", SP4, VPN,
     [("A", "I signed in to the archive server remotely at 12:30 PM and worked for two hours."),
      ("D", "I signed in to the archive server remotely at 1:30 PM."),
      ("B", "I signed in to the archive server locally at 1:30 PM."),
      ("C", "I signed in to the archive server remotely at 2:30 PM.")], "D"),
    # 17 coming back after a dentist visit is a later arrival (true); Zoe (9:10) came after Leo (9:00)
    ("Office staff argue over who came in when.", SP4, ARR,
     [("B", "I got to the office at 2:00 PM after my dentist appointment."),
      ("D", "I was in the office before Leo arrived."), ("A", "I first arrived at the office at 8:30 AM."),
      ("C", "My first arrival at the office was at 9:00 AM.")], "D"),
    # 18 control: Leo is off the list, so he could not come in at 9:00 AM
    ("Lab staff argue over the lab door.", SP4, LIST,
     [("A", "At 9:00 AM I saw Leo come into the lab through the lab door."),
      ("B", "I came into the lab through the lab door at 9:10 AM."),
      ("D", "I came into the lab through the lab door at 9:20 AM."),
      ("C", "I came into the lab through the lab door at 10:30 AM.")], "A"),
    # 19 control: Leo coming in at 9:50 AM is allowed; Leo coming in at 9:10 AM is not
    ("Lab staff argue over the lab door.", SP4, LIST,
     [("C", "I came into the lab through the lab door at 9:50 AM and stayed for an hour."),
      ("A", "I saw Leo come into the lab at 9:10 AM."), ("B", "I came into the lab at 8:00 AM."),
      ("D", "I came into the lab at 8:30 AM.")], "A"),
    # 20 control with "from ... to": 10:00 AM is inside, 8:50 AM and 11:30 AM are outside
    ("Gardeners argue over the greenhouse.", SP4, WIN_FT,
     [("A", "I went into the greenhouse at 8:50 AM."), ("B", "I went into the greenhouse at 10:00 AM."),
      ("C", "I went into the greenhouse at 11:30 AM."), ("D", "I was standing outside the greenhouse at 10:00 AM.")],
     "B"),
    # 21 a two-minute ride from 5:10 PM ends before the cut; a ride at 5:40 PM is inside it
    ("Residents argue over the lift.", SP4, LIFT,
     [("A", "I took the lift at 5:10 PM for two minutes."), ("B", "I rode the lift at 5:40 PM."),
      ("C", "I took the lift at 6:00 PM."), ("D", "I took the lift at 5:00 PM.")], "B"),
    # 22 Leo inside after his 9:30 AM entry is fine; an entry by Zoe at 10:15 AM is not in the complete log
    ("Archivists argue over who went into the archive.", SP4, LOG,
     [("C", "I was in the archive at 10:00 AM."), ("D", "I went into the archive at 10:15 AM."),
      ("A", "I didn't go into the archive at all."), ("B", "I didn't go into the archive that day.")], "D"),
    # 23 Mia at the gate is allowed; Zoe at the gate at 2:45 PM is not
    ("Guards argue over who stood at the gate.", SP4, PRES,
     [("A", "I saw Mia at the security gate at 3:00 PM."), ("D", "Honestly, I was at the security gate at 2:45 PM."),
      ("C", "I was at the security gate at 7:00 PM."), ("B", "I was at the security gate at 6:00 PM.")], "D"),
    # 24 the research booking starts at 2:35 PM; marketing in the hall at 3:00 PM is not allowed
    ("Staff argue over the main hall bookings.", SP4, BOOK,
     [("A", "The research team was in the main hall at 2:35 PM."),
      ("C", "The marketing team met in the main hall at 3:00 PM."),
      ("B", "The sales team was in the main hall at 12:00 PM."),
      ("D", "The marketing team met in the canteen at 3:00 PM.")], "C"),
]

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases_access.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "scenario", "speakers", "evidence", "chat", "label"])
    for n, (sc, sp, ev, msgs, lab) in enumerate(CASES, 1):
        chat = "\n".join(f"{i}. {s}: {t}" for i, (s, t) in enumerate(msgs, 1))
        w.writerow([f"ATK_ACC_{n:02d}", sc, sp, ev, chat, lab])
print(len(CASES), "cases ->", out)
