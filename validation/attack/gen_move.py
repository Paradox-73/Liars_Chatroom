"""Writes cases_move.csv: new movement-style cases whose evidence reads like train's templates but whose logic differs."""
import csv
import os

D = "All times in this case refer to 3 March 2026. "
SP = "A = Kofi; B = Ama; C = Priya; D = Tomas"
SP2 = "A = Aiko; B = Sofia; C = Ravi; D = Lena"
CASES = [
    # an ambiguous 'the cabinet' when two cabinets differ only by 'by the ...'
    ("Two cabinets, one key.", SP, D + "There was one red key. It stayed in the cabinet by the window from 1:00 PM to "
     "3:00 PM. The cabinet by the window and the cabinet by the door are separate locations.",
     ["A: I saw the red key in the cabinet at 2:00 PM.", "B: Who has the spare?",
      "C: At around 2:30 PM I took the red key out of the cabinet by the window.",
      "D: I was at the cabinet by the door at 2:15 PM."], "C"),
    ("Two store rooms.", SP, D + "Priya stayed in the store room by the gate from 1:00 PM to 3:00 PM. The store room by "
     "the gate and the store room by the dock are separate locations.",
     ["A: I saw Priya in the store room at 2:00 PM.", "B: Can we check the camera?",
      "D: Around 2:00 PM Priya was in the store room by the dock.", "C: I was busy all afternoon."], "D"),
    ("Two shops.", SP, D + "Kofi was at the warehouse at 2:00 PM. Every route from the warehouse to the shop by the river "
     "takes at least 40 minutes. Every route from the warehouse to the shop by the station takes at least 10 minutes. "
     "The shop by the river and the shop by the station are separate locations.",
     ["B: I saw Kofi at the shop at 2:20 PM.", "C: Kofi was at the shop by the river at about 2:15 PM.",
      "D: Did anyone else see him?", "A: I was at the warehouse at 2:00 PM."], "C"),
    ("Ferry away.", SP, D + "The ferry left the pier at 2:15 PM and did not return before 3:45 PM.",
     ["A: I got on the ferry at the pier at 4:30 PM.", "B: I saw the ferry at the pier at 3:00 PM.",
      "C: I was waiting at the pier at 3:00 PM.", "D: Weird that it took so long?"], "B"),
    ("Shuttle, liar about someone else.", "A = Kofi; B = Mira; C = Priya; D = Tomas",
     D + "The shuttle bus left the hotel stop at 1:00 PM and did not return before 2:30 PM. Nobody could get on the "
     "shuttle bus at the hotel stop unless the shuttle bus was at the hotel stop.",
     ["A: I saw Mira get on the shuttle bus at the hotel stop at 1:45 PM.",
      "B: I got on the shuttle bus at the hotel stop at 12:50 PM.", "C: I was at the hotel stop at 2:00 PM.",
      "D: Who booked it?"], "A"),
    ("Van to farm.", SP, D + "The delivery van left the depot at 2:00 PM and went straight to the farm along the 30 km "
     "route, never going above 60 km/h.",
     ["A: The delivery van reached the farm at 2:20 PM.", "B: I saw the delivery van on the way at 2:10 PM.",
      "C: The delivery van reached the farm at 2:45 PM.", "D: Was the gate open?"], "A"),
    ("Van at the exact earliest time.", SP, D + "The delivery van left the depot at 2:00 PM and went straight to the "
     "farm along the 30 km route, never going above 60 km/h.",
     ["A: The delivery van reached the farm at 2:30 PM.", "B: I saw the delivery van at the farm at 2:25 PM.",
      "C: I saw the delivery van on the way at 2:05 PM.", "D: Fast driver, that one."], "B"),
    ("Van across noon.", SP, D + "The delivery van left the depot at 11:40 AM and went straight to the farm along the "
     "40 km route, never going above 60 km/h.",
     ["A: The delivery van reached the farm at 12:25 PM.", "B: The delivery van reached the farm at 12:10 PM.",
      "C: I saw the delivery van on the way at 11:50 AM.", "D: Lunch was late."], "B"),
    ("Another van.", SP, D + "The delivery van left the depot at 11:40 AM and went straight to the farm along the 40 km "
     "route, never going above 60 km/h.",
     ["A: Another van arrived at the farm at 11:50 AM.", "B: The delivery van arrived at the farm at 12:00 PM.",
      "C: The delivery van reached the farm at 1:00 PM.", "D: Which van was it?"], "B"),
    ("Ravi before and after.", SP2, D + "Ravi was at the warehouse at 10:00 AM. Every route from the warehouse to the "
     "shop takes at least 30 minutes.",
     ["C: I was at the shop at 9:20 AM.", "A: I saw Ravi at the shop at 10:20 AM.",
      "B: I saw Ravi at the warehouse at 10:15 AM.", "D: The shop was busy."], "A"),
    ("Ravi far.", SP2, D + "Ravi was at the warehouse at 10:00 AM. Every route from the warehouse to the shop takes at "
     "least 30 minutes.",
     ["A: I was at the shop at 10:10 AM.", "D: Ravi was at the shop at 10:10 AM.",
      "B: I saw Ravi at the shop at 11:00 AM.", "C: I left the warehouse late."], "D"),
    ("Lena did not leave.", SP2, D + "Lena did not leave the lab between 1:00 PM and 3:00 PM. The lab and the office are "
     "separate locations.",
     ["B: Lena was in the office at 2:10 PM.", "C: I was in the office at 2:10 PM.",
      "A: I saw Lena in the lab at 1:30 PM.", "D: The lab was cold."], "B"),
    ("Blue case.", SP, D + "There was one blue case. It stayed in the locker from 9:00 AM to 11:00 AM. The locker and the "
     "front desk are separate locations.",
     ["A: I had the blue case at the front desk at 10:00 AM.", "B: I saw the blue case in the locker at 10:30 AM.",
      "C: I was at the front desk at 10:00 AM.", "D: I took the blue case out of the locker at 11:30 AM."], "A"),
    ("Blue case moved.", SP, D + "There was one blue case. It stayed in the locker from 9:00 AM to 11:00 AM. The locker "
     "and the front desk are separate locations.",
     ["A: I took the blue case out of the locker at 8:30 AM.", "B: I was at the front desk at 9:30 AM.",
      "C: I moved the blue case to the front desk at 10:15 AM.", "D: Is the locker locked?"], "C"),
    ("Sealed bag.", SP, D + "One evidence bag held the USB drive. It stayed sealed from 2:00 PM to 4:00 PM, and the USB "
     "drive could not be removed without breaking the seal.",
     ["A: I opened the evidence bag at 3:00 PM.", "B: I was holding the evidence bag at 3:00 PM.",
      "C: The evidence bag was still sealed when I checked it at 3:30 PM.",
      "D: I took the USB drive out of the evidence bag at 4:30 PM."], "A"),
    ("Sealed bag, someone else.", "A = Kofi; B = Omar; C = Priya; D = Tomas", D + "One evidence bag held the USB drive. "
     "It stayed sealed from 2:00 PM to 4:00 PM, and the USB drive could not be removed without breaking the seal.",
     ["A: I was holding the evidence bag at 2:30 PM.", "C: I saw Omar take the USB drive out of the evidence bag at "
      "2:30 PM.", "B: I took the USB drive out of the evidence bag at 1:30 PM.", "D: Who sealed it?"], "C"),
    ("Baton.", SP2, D + "There was one relay baton, and only one person held it at any moment. Aiko held it from 10:50 AM "
     "to 11:50 AM, then handed it to Sofia. Sofia held it until 1:20 PM.",
     ["A: I had the relay baton at 11:30 AM.", "B: I was holding the relay baton at 11:20 AM.",
      "C: I saw Sofia holding the relay baton at 12:30 PM.", "D: Who has it now?"], "B"),
    ("Baton handed early.", SP2, D + "There was one relay baton, and only one person held it at any moment. Aiko held it "
     "from 10:50 AM to 11:50 AM, then handed it to Sofia. Sofia held it until 1:20 PM.",
     ["C: Sofia handed the relay baton to me at 1:20 PM.", "A: I gave the relay baton to Sofia at 11:00 AM.",
      "B: I had the relay baton at 12:00 PM.", "D: Nice race."], "A"),
    ("Medals.", SP, D + "The display case held exactly 6 medals from 1:00 PM to 3:00 PM, and no medal was added or removed "
     "during that time.",
     ["A: I counted 5 medals in the display case at 2:00 PM.", "B: I saw 4 medals in the display case at 2:00 PM.",
      "C: There were 7 medals in the display case at 4:00 PM.",
      "D: I took one medal out of the display case at 12:30 PM."], "A"),
    ("Tank litres.", SP, D + "The water tank held exactly 40 litres from 8:00 AM to 10:00 AM, and no litre was added or "
     "removed during that time.",
     ["A: I was next to the water tank at 9:00 AM.", "B: There were 35 litres in the water tank at 9:00 AM.",
      "C: There were 30 litres in the water tank at 11:00 AM.", "D: Who checked the gauge?"], "B"),
    ("Two dozen eggs.", SP, D + "The crate held exactly 2 dozen eggs from 9:00 AM to 11:00 AM, and no dozen egg was "
     "added or removed during that time.",
     ["A: There were 24 eggs in the crate at 10:00 AM.", "B: I counted 2 dozen eggs in the crate at 10:00 AM.",
      "C: There were 3 dozen eggs in the crate at 10:30 AM.", "D: Breakfast?"], "C"),
    ("Ferry past midnight.", SP, D + "The ferry left the pier at 11:30 PM and did not return before 1:00 AM.",
     ["A: I got on the ferry at the pier at 12:15 AM.", "B: I was waiting at the pier at 12:30 AM.",
      "C: I got on the ferry at the pier at 11:00 PM.", "D: Late crossing?"], "A"),
    ("Coach round trip.", SP, D + "The tour coach left the museum gate at 10:00 AM, and its round trip took at least 1 hour "
     "30 minutes, so it could not be back at the museum gate before then.",
     ["A: I boarded the tour coach at the museum gate at 11:20 AM.",
      "B: I boarded the tour coach at the museum gate at 11:40 AM.", "C: I was at the museum gate at 11:00 AM.",
      "D: Was it full?"], "A"),
    ("Via the bridge.", "A = Jonas; B = Ama; C = Priya; D = Tomas", D + "Jonas was at the depot at 3:00 PM. Every route "
     "from the depot to the bridge takes at least 15 minutes. Every route from the bridge to the market takes at least "
     "20 minutes. Every route from the depot to the market passes through the bridge.",
     ["B: I saw Jonas at the market at 3:30 PM.", "C: Jonas was at the bridge at 3:20 PM.",
      "D: I saw Jonas at the market at 3:40 PM.", "A: Traffic was bad."], "B"),
    ("Two display cases.", SP, D + "The display case by the door held exactly 6 medals from 1:00 PM to 3:00 PM. The "
     "display case by the stairs held exactly 2 medals from 1:00 PM to 3:00 PM.",
     ["A: There were 2 medals in the display case at 2:00 PM.",
      "B: Around 2:00 PM there were 5 medals in the display case by the stairs.",
      "C: I was beside the display case by the door at 2:10 PM.", "D: Shiny."], "B"),
    ("Van 24-hour clock.", SP, D + "The delivery van left the depot at 2:00 PM and went straight to the farm along the 30 "
     "km route, never going above 60 km/h.",
     ["A: The delivery van reached the farm at 14:45.", "B: The delivery van reached the farm at 2:15 PM.",
      "C: I saw the delivery van on the way at 2:10 PM.", "D: Was it raining?"], "B"),
    ("Two lockers, one case.", SP, D + "There was one blue case. It stayed in the locker by the gym from 9:00 AM to 11:00 "
     "AM. The locker by the gym and the locker by the pool are separate locations.",
     ["A: The blue case was in the locker at 10:00 AM.", "B: I was at the locker by the pool at 10:00 AM.",
      "C: Around 10:00 AM I saw the blue case in the locker by the pool.", "D: Whose case is it?"], "C"),
]


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases_move.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "scenario", "speakers", "evidence", "chat", "label"])
        for k, (sc, sp, ev, msgs, lab) in enumerate(CASES, 1):
            chat = "\n".join(f"{i}. {m}" for i, m in enumerate(msgs, 1))
            w.writerow([f"ATK_MOV_{k:02d}", sc, sp, ev, chat, lab])
    print(len(CASES), "cases ->", out)


if __name__ == "__main__":
    main()
