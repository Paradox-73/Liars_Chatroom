"""Builds notebooks/methodology.ipynb from the repository: embeds the code zip of the final Kaggle kernel, our labelled
validation files and, when results/frozen/s8/ exists, the frozen model answers of the final run.

python notebooks/build_notebook.py             # from the repository root; writes notebooks/methodology.ipynb
python notebooks/build_notebook.py --execute   # also runs the CPU cells and saves their outputs (needs nbformat,
                                               # nbclient and ipykernel; LC_NB_DATA = a folder with train.csv only)
When s8 lands: set FINAL_MD5 below to its md5, add results/frozen/s8/<answers>.jsonl, and run with --execute.
"""
import ast, base64, glob, gzip, hashlib, json, os, sys

FINAL_MD5 = "PENDING_S8"  # md5 of the final submission file s8
RUN = "kaggle/predict-full-v3/run.py"
src = open(RUN, encoding="utf-8").read()
consts = {n.targets[0].id: n.value.value for n in ast.parse(src).body
          if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("ZIP", "ZIP_MD5", "COMMIT")}
assert hashlib.md5(base64.b64decode(consts["ZIP"])).hexdigest() == consts["ZIP_MD5"] == "2ec97551db7044ea0b9da95460a410fd"

def blob(path):
    raw = open(path, "rb").read()
    return hashlib.md5(raw).hexdigest(), base64.b64encode(gzip.compress(raw, mtime=0)).decode()

DATA = {name: blob(path) for name, path in (
    ("newfam.csv", "validation/newfam/newfam.csv"),
    ("set2.csv", "validation/set2/set2.csv"),
    ("open_model_cases.csv", "validation/open_model_cases.csv"),
    ("attack_access.csv", "validation/attack/cases_access.csv"),
    ("attack_move.csv", "validation/attack/cases_move.csv"),
    ("attack_calqty.csv", "validation/attack/cases_calqty.csv"),
    *[("final_answers.jsonl", p) for p in sorted(glob.glob("results/frozen/s8/*.jsonl"))[:1]])}

cells = []
def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})
def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})

md("""
# The Liar's Chatroom: methodology

**Team Claude Laggye** (Kanav Bhardwaj, Nirbhay Sharma, Bhavya Jain, Aaryan Antala)

Each case is a group chat with four speakers, some verified evidence, and exactly one message that cannot be true
together with the evidence; the task is to name its sender. Our entry runs in two stages. A rule solver written in
plain Python (27 evidence families, standard library only) answers the cases it can prove and clears the speakers whose
claims it judged consistent; it stays silent on anything it does not fully read. Every other case goes to one open
model, Qwen3-14B-AWQ, run offline with vLLM on two T4 cards in thinking mode, with extra samples for the least certain
cases and a vote over the speakers the rules did not clear. On train the rules answer 2,218 of 2,500 cases with no
error. On 300 labelled cases from 17 scenario families that appear in neither train nor test, the rules answer none and
the full pipeline scores 0.957 (287 of 300); on a second, separately written set of 168 cases it scores 0.946. On
Kaggle's two T4 cards, 450 cases the rules leave open took 78.5 of the 90 allowed minutes, including model load. The
final submission file (s8) is the output of exactly this code on `test.csv`; section 8 rebuilds it from the frozen
model answers of that run and checks its md5.
""")

md("""
## 1. Task, constraints and how we comply

| Constraint | What our code does |
|---|---|
| Offline at prediction time | `setup.sh` (run once, with internet) installs pinned packages and downloads the weights. `predict.py` then needs no network: it sets `HF_HUB_OFFLINE=1`, and our Kaggle kernels ran it with the network cut (an empty network namespace through `unshare -rn` where the session allows it, else a dead proxy), as in section 8. |
| 2x T4 (16 GB each) | One vLLM engine, tensor parallel over both cards, float16 activations, 4-bit AWQ weights. With one card it still runs, with 8 cases in flight instead of 40. |
| 90 minutes for about 450 cases | `predict.py` gives itself 80 minutes for 450 to 1,500 cases (the same time per case for smaller files), stops new model work 5 minutes before its end, and always leaves a complete predictions file. Measured: 78.5 minutes for 450 hard cases on Kaggle. |
| Model size at most 15B total parameters | Qwen3-14B has 14.8B total parameters on its official page; the organisers clarified that a quantised build counts by the original model's total. |
| No hosted or closed model API | No network call to any model service exists in the code. The model runs locally from downloaded weights. |
| No hand labels on test | No test label was typed, read or chosen by a person. Test labels come only from the rule solver and the model. |
| No test-specific design | Rule families were written from train cases and scored on train only (`lc/rules/run.py` refuses any other split). No test row was read or printed while building the method, and no logic is keyed to a test id or a family seen in test. This notebook never prints or inspects a test row: `test.csv` is only passed to `predict.py`. |

### How to run this notebook

- **CPU cells** (sections 2 to 6) run anywhere in a few seconds: they read `train.csv` and our own validation files only.
- **GPU cells** (section 8) need a Kaggle notebook with *GPU T4 x2* and *Internet on* (internet only for `setup.sh`).
  Each is guarded: without a GPU it prints why it was skipped. The flags in the next cell choose what runs.
- The code is embedded in this notebook as the exact zip our final Kaggle kernel ran (md5 `2ec97551db7044ea0b9da95460a410fd`,
  repository commit `2c00db8`); the next cells check the md5 and unpack it. Nothing is downloaded except by `setup.sh`.
- The saved outputs in this notebook come from a CPU-only run, so the GPU cells there show "skipped". The GPU
  results are in section 5, with the Kaggle kernel that produced each.
""")

code("""
# Flags. Defaults make "Run all" on Kaggle (GPU T4 x2, internet on) take about 20 minutes.
RUN_SETUP = True       # GPU cells: run setup.sh (about 3 min, needs internet): packages and the 9.3 GB weights
RUN_DEMO = True        # GPU cell: predict.py with the model on 40 of our new-family cases (about 15 min)
RUN_FULL_TEST = False  # HEAVY GPU cell: predict.py on all of test.csv, as the final kernel did (about 225 min)

import base64, collections, csv, glob, gzip, hashlib, io, os, shutil, statistics, subprocess, sys, tempfile, time, zipfile

IS_KAGGLE = os.path.isdir("/kaggle/input")
SUB = "/tmp/sub" if IS_KAGGLE else os.path.join(tempfile.gettempdir(), "lc_sub")  # the unpacked code, as in the final kernel
OUT = "/kaggle/working" if IS_KAGGLE else os.path.abspath("nb_out")
os.makedirs(OUT, exist_ok=True)

def find_data(name):
    \"\"\"Path of a competition file: /kaggle/input/the-liars-chatroom/ on Kaggle, else $LC_NB_DATA, else ./data.\"\"\"
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) if IS_KAGGLE else []
    if not hits:
        hits = [p for p in (os.path.join(os.environ.get("LC_NB_DATA", "data"), name),) if os.path.exists(p)]
    return hits[0] if hits else None

TRAIN = find_data("train.csv")
try:
    N_GPU = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout.splitlines()
    N_GPU = sum(l.startswith("GPU ") for l in N_GPU)
except (OSError, subprocess.SubprocessError):
    N_GPU = 0
GPU_OK = N_GPU >= 1 and sys.platform.startswith("linux")
print("train.csv found:", TRAIN is not None, "| GPUs:", N_GPU, "| GPU cells will run:", GPU_OK)
""")

code(f"""
# The code submission, exactly as the final Kaggle kernel (paradox73/predict-full-v3) embedded it:
# kaggle/predict-full-v3/run.py, built from commit {consts["COMMIT"]}.
ZIP_MD5 = "{consts["ZIP_MD5"]}"
ZIP = "{consts["ZIP"]}"
raw = base64.b64decode(ZIP)
assert hashlib.md5(raw).hexdigest() == ZIP_MD5, "embedded zip changed"
shutil.rmtree(SUB, ignore_errors=True)
zipfile.ZipFile(io.BytesIO(raw)).extractall(SUB)
sys.path.insert(0, SUB)
print("zip md5", ZIP_MD5, "unpacked")
print(sorted(os.path.relpath(p, SUB).replace(os.sep, "/") for p in glob.glob(f"{{SUB}}/**/*.*", recursive=True)))
""")

code("""
# Our own labelled validation files and the frozen model answers of the final run, gzip + base64.
# newfam.csv: 300 cases, 17 families; set2.csv: 153 cases, 9 families (both made by our generator programs in
# validation/ of the repository); open_model_cases.csv: 15 cases a teammate generated with Qwen3-8B and checked.
# attack_*.csv: 87 cases in train's evidence wording with different logic (validation/attack, section 3b).
# final_answers.jsonl (when present): one line per model sample of the final test.csv run (case id, sample number,
# letter, token count).
FILES = {
""" + "".join(f'    "{k}": ("{m}", "{b}"),\n' for k, (m, b) in DATA.items()) + """}
DATA = os.path.join(SUB, "nbdata")
os.makedirs(DATA, exist_ok=True)
for name, (md5, b64) in FILES.items():
    raw = gzip.decompress(base64.b64decode(b64))
    assert hashlib.md5(raw).hexdigest() == md5, name
    with open(os.path.join(DATA, name), "wb") as f:
        f.write(raw)
    print(f"{name:22s} {len(raw):7d} bytes  md5 {md5}")
""")

md("""
## 2. The data (train.csv only)

`train.csv` has 2,500 labelled cases with the columns `id, scenario, speakers, evidence, chat, label`. Train has no
family column; the family tags below are the evidence parsers of our rule solver that recognise at least one evidence
sentence of the case. The cell reads train only.
""")

code("""
from lc.data import case_from_row
from lc.rules import solve
from lc.combine import open_speakers
from lc.rules.text import split_sentences

with open(TRAIN, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
cases = [case_from_row(r) for r in rows]
print(f"{len(cases)} cases; {len(set(r['scenario'] for r in rows))} distinct scenario sentences")
print("label balance:", dict(sorted(collections.Counter(c.label for c in cases).items())))

def describe(name, xs):
    q = statistics.quantiles(xs, n=20)
    print(f"{name:32s} min {min(xs):5d}  median {int(statistics.median(xs)):5d}  95th pct {int(q[18]):5d}  max {max(xs):5d}")

describe("evidence length (characters)", [len(c.evidence) for c in cases])
describe("evidence sentences", [len(split_sentences(c.evidence)) for c in cases])
describe("chat length (characters)", [sum(len(m.text) for m in c.messages) for c in cases])
describe("messages per chat", [len(c.messages) for c in cases])
describe("questions per chat", [sum(m.text.strip().endswith("?") for m in c.messages) for c in cases])
print("speakers per case:", dict(collections.Counter(len(c.speakers) for c in cases)))
print("the liar sends how many messages:", dict(sorted(collections.Counter(
    sum(m.label == c.label for m in c.messages) for c in cases).items())))

results = {c.id: solve(c) for c in cases}
fam = collections.Counter(f for r in results.values() for f in set(r.families))
print(f"\\nrule families recognised in train: {len(fam)}; cases with none: {sum(not r.families for r in results.values())}")
for name, n in fam.most_common():
    print(f"  {name:28s} {n:5d}")
""")

md("""
## 3. Stage 1: the rule solver (`lc/rules`)

**What it does.** Each family (27 of them, in `fam_access.py`, `fam_calendar.py`, `fam_movement.py`,
`fam_quantity.py`) recognises one kind of evidence sentence, for example "nobody could enter X between T1 and T2", a
complete log, a schedule, a capacity or a price, and judges the chat messages that make a claim about it. A family
returns a verdict only for a message it fully understands: **lie** when the claim cannot be true together with the
evidence, **ok** when it must be consistent, nothing otherwise. Precision comes before coverage:

- an evidence pattern must match a whole sentence, every clause; a clause the family does not model leaves the sentence unread;
- when an unread sentence names the same person, place or object as a parsed fact, the family drops that fact;
- questions, hedged or negated sentences, reported speech and claims at the exact edge of a window are not judged;
- quantities use exact fractions, times are whole minutes in one day, and any case that mentions midnight or a time
  before 5:00 AM is left alone (the overnight guard);
- an exception inside a family is recorded and that family is skipped; it never stops a case.

**Tiers** (`lc/combine.py`, `decide`):

| Tier | When |
|---|---|
| `rule_confident` / `rule` | exactly one speaker has a message judged a lie (`rule_confident` when, in addition, every claim in the chat was judged) |
| `elim` | no lie found, but every claim of three speakers was judged consistent and the fourth has claims nobody could judge |
| left to the model | anything else; the model's vote is limited to the speakers the rules did not clear |

**When it abstains.** Besides the cases above, `predict.py` keeps a rule or `elim` answer, and the cleared speakers,
only when the rules read *every* evidence sentence (the date line "All times in this case refer to ..." aside). A
sentence no family reads can change what the rest means, so such a case goes to the model with all speakers open.

The next cell runs `predict.py`'s own reading step (`read_cases`) on train with the model off, so the numbers are those
of the submitted pipeline.
""")

code("""
import predict
t = time.time()
ids, fixed, left, cands = predict.read_cases(TRAIN)
label = {c.id: c.label for c in cases}
print(f"read_cases on train: {time.time() - t:.1f} s")
raw_answered = sum(bool(r.pred) or len(open_speakers(c, r)) == 1 for c, r in zip(cases, results.values()))
print(f"rules answer (before the every-sentence-read check): {raw_answered}")
print(f"rules answer (kept by predict.py): {len(fixed)} of {len(ids)} = {len(fixed) / len(ids):.1%}")
by_tier = collections.defaultdict(lambda: [0, 0])
for cid, (letter, tier) in fixed.items():
    by_tier[tier][0] += 1
    by_tier[tier][1] += letter == label[cid]
for tier, (n, right) in sorted(by_tier.items()):
    print(f"  {tier:15s} {n:5d} cases, {right} right ({right / n:.2%})")
print(f"left to the model: {len(left)}; the true liar is among the open speakers in "
      f"{sum(label[c] in cands[c] for c in left)} of them")
print("open speakers per case left to the model:", dict(sorted(collections.Counter(len(v) for v in cands.values()).items())))
""")

md("""
Per-family check on train with the repository's scorer (`python -m lc.rules.run`): for each family, the cases where it
is active, its lie verdicts on the true liar, and its lie verdicts on an innocent speaker (must be 0).
""")

code("""
out = subprocess.run([sys.executable, "-m", "lc.rules.run"], cwd=SUB, capture_output=True, text=True,
                     env=dict(os.environ, LC_DATA_DIR=os.path.dirname(TRAIN)))
print(out.stdout[-6000:], out.stderr[-2000:])
""")

md("""
**On new families the rules stay silent.** The hidden cases come from families in neither train nor test, so what
matters is that the rules do not misfire there. The same reading step on our three labelled new-family files:
""")

code("""
for name in ("newfam.csv", "set2.csv", "open_model_cases.csv"):
    path = os.path.join(DATA, name)
    with open(path, encoding="utf-8", newline="") as f:
        gold = {r["id"]: r["label"] for r in csv.DictReader(f)}
    ids_v, fixed_v, left_v, cands_v = predict.read_cases(path)
    wrong = sum(letter != gold[cid] for cid, (letter, _) in fixed_v.items())
    dropped = sum(gold[cid] not in cands_v[cid] for cid in left_v)
    print(f"{name:22s} {len(ids_v):4d} cases: rules answer {len(fixed_v)} ({wrong} wrong); "
          f"{len(left_v)} to the model; true liar wrongly cleared in {dropped}")
""")

md("""
### 3b. Rule fixes after the new-family attack

The rules were written from train, so we tried to make them misfire: 87 labelled cases (24 access, 27 movement,
36 calendar and quantity; `validation/attack/gen_*.py` writes them, each with a comment on why its label is right)
reuse train's evidence wording with logic train does not have. On these, the rules gave 21 wrong answers or cleared
the true liar. Each fix makes a rule step aside, so the model decides:

- **Lasting claims and unusual access wording**: a claim that lasts ("for an hour"), "didn't go in until T", a code
  that did not work, someone else's wristband, and a later arrival without "first" are no longer judged.
- **Times the calendar rules do not read**: opening hours and sessions skip claims with 24-hour or "p.m." times,
  times of day in words, or a length of time.
- **Shared place names**: a short place name stays unread when the evidence has two places with that name ("the
  cabinet by the door", "the cabinet by the window").

The cost on train is 3 rule answers (2,218 instead of 2,221, all still right); those cases now go to the model. The
rules still answer none of the newfam and set2 cases, so the validation runs of section 5 are unaffected. The check
of `validation/attack/check.py`, on the embedded copies (expected: no wrong rule answer and no cleared liar):
""")

code("""
for name in ("attack_access.csv", "attack_move.csv", "attack_calqty.csv"):
    path = os.path.join(DATA, name)
    with open(path, encoding="utf-8-sig", newline="") as f:
        gold = {r["id"]: r["label"] for r in csv.DictReader(f)}
    ids_v, fixed_v, left_v, cands_v = predict.read_cases(path)
    wrong = sum(letter != gold[cid] for cid, (letter, _) in fixed_v.items())
    cleared = sum(gold[cid] not in cands_v[cid] for cid in left_v)
    print(f"{name:18s} {len(ids_v):3d} cases: rules answer {len(fixed_v)} ({wrong} wrong); {len(left_v)} to the model; "
          f"true liar cleared in {cleared}")
""")

md("""
## 4. Stage 2: the model judge (`judge.py`)

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-14B-AWQ`, revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3`, the Qwen team's 4-bit AWQ build of Qwen3-14B (14.8B total parameters), Apache-2.0 |
| Serving | vLLM 0.32.0, one engine, tensor parallel 2 over the two T4 cards, float16, 92% of card memory, max model length 5,120 tokens |
| Mode | thinking mode (`enable_thinking=True` in the Qwen3 chat template), temperature 0.6, top-p 0.95, top-k 20 |
| In flight | up to 40 thinking requests (fewer when the KV cache would overflow: at most cache tokens / 2,500) plus 24 one-token requests |
| First sample | seed 0, shortest prompt first, at most 3,072 tokens |
| Forced answer | a sample with no `ANSWER:` line (it hit the cap) gets `ANSWER:` appended and one more token read; it counts at half weight in the vote |
| Extra samples | seeds 1, 2, ... for the least certain case (smallest vote margin first, longer reasoning first on ties), up to 5 samples a case, none once the margin reaches 1.5 (two clean samples that agree) |
| No continuations | a capped sample is not continued; spare time goes to extra samples (section 6) |
| Vote | weighted vote of the samples over the speakers the rules did not clear (`lc/combine.py`, `vote`) |

**Why shortest prompt first.** A long prompt means long reasoning (correlation 0.51 between prompt length and output
tokens on the model's test answers, 0.34 on our second set). When time runs out, the cases still waiting are then the
slow, hard ones, not the cheap ones that are almost always right once reached.

**Time budget** (`predict.py`). The clock starts with `predict.py`. A file of 450 to 1,500 cases gets 80 minutes; a
smaller file gets the same time per case (40 cases: 7.1 minutes); a larger file gets 80 minutes per 450 cases the
model judges (all of `test.csv`: 222.6 minutes for 1,252 model cases). No new model work starts in the last 5 minutes:
every case without a finished sample then gets a one-token answer without thinking. Extra samples already running
stop at once while first samples are still waiting; when every case has its first sample they may run to the end.
90 seconds before the end, any sample still thinking stops if one-token answers are waiting. At the end the model
process is stopped and the final file is written.

**Fallbacks.** The predictions file is complete from the first second (rule answers, letter C elsewhere) and is
rewritten every 30 seconds through a temporary file and a rename. A case the model never reached gets the one-token
answer without thinking, and failing that the letter C (the most common train label), or the first open speaker when
the rules cleared C. A model process that exits with an error, does not start within 10 minutes, or stops making
progress for 5 minutes is stopped and restarted with safer settings: no CUDA graphs, then Triton AWQ and attention
kernels with more memory left free, then one card. Without a GPU, vLLM or the weights, the rule answers and fallbacks
stand and the log says so.

**Robustness fixes in version 2 (kept in the final code).** Chat and speaker lines in other forms (`1)` or no numbers,
lower-case labels, `A (Name):`, `A: Name`, commas) are read instead of getting the letter C; a rule answer counts only
when every evidence sentence was read; only questions leave a speaker out of the vote (an opinion pattern can match a
real claim); `setup.sh` runs under `sh` as well as `bash` and retries downloads; a hung engine start goes to one card
at once; the 30-second rewrite keeps model answers already read even when the polling loop fails.

The prompt (`lc/llm/prompt.py`) and the engine settings, printed from the unpacked code, with a train case as the example:
""")

code("""
from lc.llm.prompt import build_prompt
print(build_prompt(cases[0]))
print("\\n--- predict.py MODEL_ENV:", predict.MODEL_ENV)
print("--- restart attempts (judge.py arguments, environment):")
for a in predict.ATTEMPTS:
    print("   ", a)
judge_src = open(os.path.join(SUB, "judge.py"), encoding="utf-8").read().splitlines()
start = next(i for i, l in enumerate(judge_src) if l.strip().startswith("def prompt(cid)"))
print("\\n--- judge.py: chat template, answer without thinking")
print("\\n".join(judge_src[start:start + 11]))
print("\\n--- judge.py: sampling of a thinking request")
i = next(i for i, l in enumerate(judge_src) if "temperature=0.6" in l)
print("\\n".join(judge_src[i - 1:i + 2]))
""")

md("""
## 5. Validation

The hidden cases come from scenario families in neither train nor test, so the rules will answer few of them and the
model does almost all the work. We therefore validated on labelled cases from families we wrote ourselves, never on
test labels (there are none):

- **newfam**: 300 cases from 17 families (age limits, battery drain, library loans, lifts, medicine intervals,
  parking, periodic events, platforms, prices, queues, recipes, scores, seats, temperature, time zones, overnight
  spans, ...), produced by generator programs that check each claim against the evidence.
- **set2**: 168 cases: 153 from nine more families (league tables, currency exchange, finishing order, seating,
  fuel use, n-th weekday meetings, locker codes, bank balances, auctions), written apart from newfam and checked claim
  by claim against every world the evidence allows, plus the 15 cases of `open_model_cases.csv`.

Each run below used the time per case of the hidden run (80 minutes per 450 cases), on Kaggle's two T4 cards unless
stated. Accuracy counts every case, including any the model did not reach in time.

The Kaggle rows ran version 2 of the code (zip md5 `67c4f145`, commit `c5b55b1`). The final code (version 3, zip
`2ec97551`) differs from it only in the rule fixes of section 3b, which make the rules answer fewer cases. On newfam
and set2 the rules answer none in either version, so those inputs reach the model the same way. On test the rules
leave a few more cases to the model, so the timing row is a close stand-in rather than a measurement of version 3.

| Check | Result | Kaggle kernel |
|---|---|---|
| newfam, 300 cases | **0.957** (287 of 300), 52.3 of 53.3 min, exit 0; all 300 got a thinking answer (44 forced at 3,072 tokens); 23 extra samples | `nirbhaysharma504/v2-newfam` |
| set2, 168 cases | **0.946** (159 of 168) against 0.911 for version 1 (11 cases better, 5 worse), 28.8 of 29.9 min, exit 0; 167 got a thinking answer (27 forced), 1 the answer without thinking | `nirbhaysharma504/v2-set2` |
| Timing: 450 test cases the rules leave open (closest stand-in for new families), cold start, offline | **78.5 min** including model load (engine ready at 2.4 min), exit 0; limit 90. All 449 cases the model judged got a thinking answer (39 forced), 133 extra samples; none needed the answer without thinking | `nirbhaysharma504/predict-test-v2-nirbhay` |
| One T4: 40 of those cases, 15-minute budget | 12.6 min, exit 0: 34 of 40 got a thinking answer (1 forced), 6 the answer without thinking | `nirbhaysharma504/predict-test-v2-nirbhay` |
| Participant kit `selftest.py --n 200` | OK, accuracy 1.000, 6.5 min (kit's estimate for 450: 14.7 min); 15 cases reached the model | `nirbhaysharma504/predict-test-v2-nirbhay` |
| Participant kit `selftest.py --n 50`, laptop without GPU | OK, 0.980 in 1.0 s (rules and fallbacks only) | local |
| Unit tests, `python -m pytest tests`, version 3 | 36 passed (judge.py against a fake engine, parsing, the vote, rule answers on unread evidence, the attack cases) | local |
| Version 2's final file s7 (history): all of `test.csv` | 3,500 cases; rules 2,248 (4 by elimination); 1,252 to the model, every one with a first sample (137 forced), 263 extra samples; 222.3 of 222.6 min, exit 0 | `nirbhaysharma504/predict-full-v2-nirbhay` |
| Final file s8, version 3: all of `test.csv` | pending: running at the time of writing | `paradox73/predict-full-v3` |

For the timing rows, the 450 test ids were chosen by the rules (cases they leave open) and the rows were passed to
`predict.py` unchanged; nothing about their content was looked at, and no accuracy is measured on them.

**Model choice** (newfam, two T4 cards, same code and time per case):

| Judge | newfam accuracy | Notes |
|---|---|---|
| Qwen3-14B-AWQ, 4,096-token first sample | 0.957 | 297 of 300 reached; 150 to 175 tokens/s |
| Qwen3-8B-AWQ, 2,048-token first sample, continued to 8,192 | 0.893 | 199 of 299 capped |
| Qwen3-8B fp16, 8,192 cap | 0.643 | 115 reached; little room for the KV cache |
| Qwen3-8B-AWQ, 8,192 cap | 0.637 | 115 reached |
| DeepSeek-R1-0528-Qwen3-8B fp16 | 0.313 | 13 reached: very long reasoning |
| Qwen3.5-4B-AWQ, one engine per card | stopped | 9 min to start, then 4 cases in 14 min |

T4 cards generate about 150 to 200 tokens per second for these models whatever their size, so the number of tokens a
model writes per case decides how many cases it reaches. Qwen3-14B writes about half as many as Qwen3-8B.

Summary of the frozen model answers of the final run s8 (aggregates only), when they are embedded:
""")

code("""
import json
FINAL_ANSWERS = os.path.join(DATA, "final_answers.jsonl")
recs = [json.loads(l) for l in open(FINAL_ANSWERS, encoding="utf-8")] if os.path.exists(FINAL_ANSWERS) else []
first = [r for r in recs if r["k"] == 0]
if not recs:
    print("skipped: the frozen answers of s8 are not embedded yet (results/frozen/s8/ in the repository)")
else:
  print(f"{len(recs)} samples for {len({r['id'] for r in recs})} cases: {len(first)} first samples "
      f"({sum(r['forced'][0] for r in first)} forced at the cap), {sum(r['k'] > 0 for r in recs)} extra samples, "
      f"{sum(r['k'] < 0 for r in recs)} answers without thinking")
  toks = [r["out_tokens"][0] for r in first]
  print(f"output tokens per first sample: mean {statistics.mean(toks):.0f}, median {statistics.median(toks):.0f}, max {max(toks)}")
  print("samples per case:", dict(sorted(collections.Counter(collections.Counter(r["id"] for r in recs).values()).items())))
""")

md("""
## 6. What we tried and dropped

- **A vote of several models** (Qwen3.5-4B, Qwen3-8B, Qwen3-32B, QwQ-32B). Our first plan. The 32B models are over
  the 15B limit, and on T4 cards speed in tokens per second hardly depends on size, so the model that writes fewer
  tokens per case wins: one Qwen3-14B.
- **A 4,096-token first sample.** With 3,072, set2 rose from 0.857 to 0.911 (10 cases better, 1 worse, sign test
  p = 0.012), mostly because more cases got a thinking answer in time (152 of 168 against 127); newfam was 0.953 both ways.
- **Continuing capped samples.** On 49 hard cases whose first sample hit its cap, continuing to 8,192 tokens gave a
  net 3 more right answers (p = 0.51) and took 2,595 s on two cards; a fresh sample on those cases finished 29 times
  out of 78 and was right 28 of those 29 times. So capped samples are forced at once and spare time goes to extra samples.
- **A tail cap** that shortened later samples to fit the time. It measured throughput from finished samples only and
  cut thinking to 768 tokens early; even fixed, sharing the remaining tokens out evenly scored 0.930 on newfam against
  0.957 without it. Dropped: the 3,072 cap is fixed.
- **Speed options on T4** (Qwen3-14B-AWQ, steady state): current engine 205.5 tokens/s; FlashInfer 206.1; split
  kernels 203.9; n-gram speculative decoding 156.1; one engine per card 125.7; fp8 KV cache not supported on T4.
  Nothing beat the current engine by the 15% we required for a change.
- **A forced answer that restates the one-lie rule** before `ANSWER:` changed none of 62 forced answers.

## 7. Limitations

- Our accuracy estimate for new families (0.95 to 0.96) comes from cases we generated ourselves; the hidden cases may differ.
- The rules score 100% on train because they were built against it; their precision on test is seen only through
  the public leaderboard.
- vLLM batching can change near-ties between two runs, so a fresh run of the judge is not bit-exact; the vote is
  deterministic given the model's answers, which is how section 8 rebuilds the final file.
- On one T4 the model reaches about a third as many cases per minute, so more cases get the weaker answer without thinking.
""")

md("""
## 8. Reproduction

These cells mirror our final Kaggle kernel `predict-full-v3` (`kaggle/predict-full-v3/run.py` in the
repository): unpack the embedded zip (done above), run `setup.sh` with internet, then run `predict.py` with the network
cut. Turn on *GPU T4 x2* and *Internet* in the notebook settings. The heavy cell is off by default (`RUN_FULL_TEST`).

All output of the child processes is streamed into the cell.
""")

code("""
def sh(cmd):
    \"\"\"Runs a shell command and streams its output into the cell; returns the exit code.\"\"\"
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()

SETUP_DONE = False
if not GPU_OK:
    print("skipped: no GPU (or not Linux); the rule answers above need none")
elif not RUN_SETUP:
    print("skipped: RUN_SETUP is False")
else:
    sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv; nproc; free -g | head -2; df -h /tmp | tail -1")
    t = time.time()
    SETUP_DONE = sh(f"bash {SUB}/setup.sh") == 0  # about 3 min: .venv with pinned vLLM/PyTorch, weights to models/judge
    print(f"setup.sh: {'ok' if SETUP_DONE else 'FAILED'}, {(time.time() - t) / 60:.1f} min")
""")

code("""
# Cut the network for predict.py, as the final kernel did: a new, empty network namespace with only the loopback up.
OFFLINE = ""
if SETUP_DONE:
    with open("/tmp/offline.sh", "w") as f:
        f.write("#!/bin/sh\\nip link set lo up 2>/dev/null || python3 -c \\"import fcntl, socket, struct; s = socket.socket(2, 2); "
                "f = struct.unpack('16sH', fcntl.ioctl(s, 0x8913, struct.pack('16sH', b'lo', 0)))[1]; "
                "fcntl.ioctl(s, 0x8914, struct.pack('16sH', b'lo', f | 1))\\"\\nexec \\"$@\\"\\n")
    os.chmod("/tmp/offline.sh", 0o755)
    loopback = "python3 -c \\"import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); socket.create_connection(s.getsockname(), 3)\\""
    internet = "python3 -c \\"import socket; socket.create_connection(('pypi.org', 443), 5)\\""
    if sh(f"unshare -rn /tmp/offline.sh {loopback}") == 0 and sh(f"unshare -rn /tmp/offline.sh {internet}") != 0:
        OFFLINE = "unshare -rn /tmp/offline.sh "
        print("predict.py runs with no network at all")
    else:
        OFFLINE = "HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 "
        print("WARNING: the network cannot be cut here; predict.py has HF_HUB_OFFLINE=1 and a dead proxy only")
else:
    print("skipped: setup.sh did not run")
""")

md("""
**Demo (GPU, about 15 minutes).** `predict.py` with the model on the first 40 newfam cases, labels removed from its
input. The rules answer none of them, so every case goes to the model. The budget is set to 15 minutes with
`--minutes 15`: the default for 40 cases (80 x 40 / 450 = 7.1 minutes) leaves too little time after the engine load
(about 2.5 minutes) and the 5-minute stop margin, and `predict.py` would then keep the fallbacks without starting the
model. This demo shows the model path working; the accuracy figures to rely on are the full runs in section 5.
""")

code("""
if not (SETUP_DONE and RUN_DEMO):
    print("skipped:", "setup.sh did not run" if not SETUP_DONE else "RUN_DEMO is False")
else:
    with open(os.path.join(DATA, "newfam.csv"), encoding="utf-8", newline="") as f:
        demo = list(csv.DictReader(f))[:40]
    with open("/tmp/demo40.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "scenario", "speakers", "evidence", "chat"])
        w.writerows([r[c] for c in ("id", "scenario", "speakers", "evidence", "chat")] for r in demo)
    t = time.time()
    rc = sh(f"cd {SUB} && {OFFLINE}python predict.py --input /tmp/demo40.csv --output {OUT}/demo40_predictions.csv --workdir /tmp/wdemo --minutes 15")
    with open(f"{OUT}/demo40_predictions.csv", encoding="utf-8", newline="") as f:
        pred = {r["id"]: r["label"] for r in csv.DictReader(f)}
    right = sum(pred.get(r["id"]) == r["label"] for r in demo)
    print(f"exit {rc}, {(time.time() - t) / 60:.1f} min; accuracy {right}/{len(demo)} = {right / len(demo):.3f}")
""")

md("""
**Full run on `test.csv` (HEAVY: about 225 minutes on two T4 cards).** The exact command of the final kernel. Set
`RUN_FULL_TEST = True` in the first code cell to run it. Because vLLM batching is not bit-exact, a fresh run can differ
from s8 on a few near-tie cases; the cell after it checks the final file itself.
""")

code("""
TEST = find_data("test.csv")
if not (SETUP_DONE and RUN_FULL_TEST and TEST):
    print("skipped:", "RUN_FULL_TEST is False" if not RUN_FULL_TEST else "setup.sh did not run" if not SETUP_DONE else "no test.csv")
else:
    t = time.time()
    rc = sh(f"cd {SUB} && {OFFLINE}python predict.py --input {TEST} --output {OUT}/submission.csv --workdir /tmp/wfull")
    print(f"predict.py on all of test.csv: exit code {rc}, {(time.time() - t) / 60:.1f} min")
    print("md5 of submission.csv:", hashlib.md5(open(f"{OUT}/submission.csv", "rb").read()).hexdigest())
""")

md("""
**Rebuild of the final file (CPU, seconds).** `predict.py`'s own functions read `test.csv` (rule answers), merge
the frozen model answers of the final run (embedded above, from `results/frozen/s8/`), and write the predictions file.
Its md5 must equal `FINAL_MD5`, the md5 of the `submission.csv` the Kaggle run wrote. No test row is printed; only the
md5 and the label counts.

History: the previous final file s7 came from version 2 of the code (md5 `2cb35f3b2e402387c44206144154f30e`, rebuilt
byte for byte from its frozen answers with version 2). With version 3's rules, the same answers give a file that
differs from s7 in 1 label.
""")

code("""
FINAL_MD5 = "@FINAL_MD5@"  # md5 of the final submission file s8
S7_MD5 = "2cb35f3b2e402387c44206144154f30e"  # history: version 2's final file, not rebuilt here
TEST = find_data("test.csv")
if not os.path.exists(FINAL_ANSWERS):
    print("skipped: the frozen answers of s8 are not embedded yet (results/frozen/s8/ in the repository)")
elif not TEST:
    print("skipped: test.csv not found (on Kaggle, add the competition data the-liars-chatroom)")
else:
    ids, fixed, left, cands = predict.read_cases(TEST)
    recs = {}
    predict.merge(FINAL_ANSWERS, 0, recs)
    tiers = predict.write_predictions(f"{OUT}/final_rebuild.csv", ids, fixed, cands, recs)
    got = hashlib.md5(open(f"{OUT}/final_rebuild.csv", "rb").read()).hexdigest()
    with open(f"{OUT}/final_rebuild.csv", encoding="utf-8", newline="") as f:
        labels = collections.Counter(r["label"] for r in csv.DictReader(f))
    print("tiers:", dict(tiers), "| labels:", dict(sorted(labels.items())))
    print("md5", got, "MATCHES the final file" if got == FINAL_MD5 else f"DIFFERS from FINAL_MD5 ({FINAL_MD5})")
""")

md("""
## 9. Compliance statement

We confirm that:

1. no closed or hosted model API is called by the solution or was used to label data or produce training data;
2. no test labels were created by hand or by a person;
3. `test.csv` was not read to design family-specific rules;
4. `predict.py` reproduces the Final Submission when run on `test.csv`.

Model: `Qwen/Qwen3-14B-AWQ` (revision `31c69efc29464b6bb0aee1398b5a7b50a99340c3`), Apache-2.0, 14.8B total parameters.
Libraries: vLLM 0.32.0 (Apache-2.0), PyTorch 2.13.0 (BSD-3-Clause), Transformers 5.19.0 (Apache-2.0),
huggingface_hub 2.4.0 (Apache-2.0). Nothing is trained or fine-tuned; no external data is used.
References: Qwen Team, Qwen3 Technical Report, 2025 (arXiv:2505.09388); Kwon et al., Efficient Memory Management for
Large Language Model Serving with PagedAttention, SOSP 2023; Lin et al., AWQ, MLSys 2024.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
for c in cells:
    c["source"] = [l.replace("@FINAL_MD5@", FINAL_MD5) for l in c["source"]]
out = "notebooks/methodology.ipynb"
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
print("wrote", out, len(cells), "cells; final answers embedded:", "final_answers.jsonl" in DATA)
if "--execute" in sys.argv:  # CPU cells only: without a GPU the GPU cells print "skipped"
    import nbformat, tempfile
    from nbclient import NotebookClient
    nbx = nbformat.read(out, as_version=4)
    NotebookClient(nbx, timeout=900, kernel_name="python3", resources={"metadata": {"path": tempfile.mkdtemp()}}).execute()
    for c in nbx.cells:
        c.metadata.pop("execution", None)
    nbformat.validate(nbx)
    nbformat.write(nbx, out)
    print("executed;", sum(o.output_type == "error" for c in nbx.cells if c.cell_type == "code" for o in c.outputs), "errors")
