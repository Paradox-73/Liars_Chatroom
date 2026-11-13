# Validation sets

## open_model_cases.csv (Nirbhay)

15 new-family cases (NF001-NF015) generated with Qwen3-8B, then checked and cleaned into the train.csv format (`id,scenario,speakers,evidence,chat,label`). They are not hand-written, so treat them as an extra check rather than a fully independent one.

- 11 scenario types not in train: battery drain, parking meter pricing, medication intervals, lift capacity, library loans, train platforms/delays, recipe scaling, bundle pricing, match points, class caps, plant care.
- Each case has 8 messages and exactly one impossible claim; the label is the liar's speaker letter.
- Cases that need arithmetic: NF001, NF002, NF003, NF004, NF006, NF007, NF008, NF009, NF013, NF015.
- Cases that mix two kinds of evidence: NF012, NF013, NF014, NF015.
- IDs use three digits (NF001), unlike the four-digit ids of the 300-case set in `newfam/` (NF0001).
