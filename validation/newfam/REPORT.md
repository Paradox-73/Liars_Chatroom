# New-family validation set: report

<!-- analysis -->
(hand-written analysis goes here)
<!-- /analysis -->

## Rule solver on newfam.csv

Command: `python -m validation.newfam.measure --write`

- Cases: 300
- Answered by the rules: 0 (0.0%); right 0, wrong 0
- Misfires (an innocent speaker's message judged a lie): 0 cases
- Cases where open_speakers() wrongly drops the true liar: 0
- Cases where a rule judged the lie itself consistent: 0
- Cases where is_filler() reads the lie as a question or opinion: 0
- Elimination tier answers: 1, of which wrong 0
- Cases where the overnight guard made the rules abstain: 5
- Cases with an exception inside a rule family: 0

### Cases per family

| family | cases alone | combined, lie from it | combined, second family |
|---|---|---|---|
| agelimit | 13 | 6 | 2 |
| battery | 13 | 4 | 5 |
| library | 13 | 4 | 4 |
| lift | 13 | 9 | 3 |
| medicine | 13 | 4 | 4 |
| overnight | 10 | 0 | 0 |
| parking | 13 | 1 | 6 |
| periodic | 13 | 3 | 6 |
| platform | 13 | 4 | 7 |
| prices | 13 | 6 | 4 |
| queue | 13 | 6 | 6 |
| recipe | 13 | 2 | 5 |
| scores | 13 | 4 | 6 |
| seats | 13 | 4 | 8 |
| temperature | 13 | 5 | 7 |
| timezone | 13 | 7 | 4 |
| whenever | 13 | 13 | 5 |
| all | 218 | 82 | 82 |

### Rule solver by family

| family | cases | answered | right | misfires | liar cleared | elim right/wrong | overnight abstain | rule families active |
|---|---|---|---|---|---|---|---|---|
| agelimit | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| battery | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| library | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| lift | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| medicine | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| overnight | 10 | 0 | 0 | 0 | 0 | 0/0 | 5 | - |
| parking | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| periodic | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| platform | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| prices | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| queue | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| recipe | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| scores | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| seats | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| temperature | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| timezone | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| whenever | 13 | 0 | 0 | 0 | 0 | 0/0 | 0 | - |
| combined (two families) | 82 | 0 | 0 | 0 | 0 | 1/0 | 0 | - |
| all | 300 | 0 | 0 | 0 | 0 | 1/0 | 5 | - |

### Misfires: 0

### True liar wrongly cleared: 0

### Wrong elimination answers: 0

### Reading the numbers

- The same solver answers 273 of the first 300 train cases (0 wrong), so the zero here is real: none of its rule families recognises these scenario shapes, and it abstains instead of guessing.
- No misfires and no wrongly excluded liars, so the open-speaker restriction is safe on this set. It does not help either: it has no rule verdicts to narrow on.
- Every case passed the generator's own check: exactly one message fails its fact model, every other claim passes it.
