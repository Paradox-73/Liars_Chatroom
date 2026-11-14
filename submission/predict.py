#!/usr/bin/env python3
"""The Liar's Chatroom:  python predict.py --input hidden.csv --output predictions.csv

1. The rule solver (lc/rules, standard library only) answers the cases it can prove and clears the speakers whose
   claims it judged consistent. A complete predictions file is written straight away.
2. judge.py, a child process, runs the judge model with vLLM on the other cases. Its answers are merged as they arrive
   and the predictions file is rewritten every 30 s, so a crash or a hang still leaves a valid file.
3. Each case's answer follows lc/combine.py: the rule answer, then elimination, then the vote of the model's samples
   over the speakers the rules did not clear (a forced answer at half weight), then the model's one-token answer
   without thinking, then the letter C (or the first open speaker if the rules cleared C).

Time is measured from the start of this process: 80 min for an input of 450 to 1,500 cases (the limit is 90 min for
about 450), the same time per case for a smaller input, and the same time per model-judged case for a larger one.
Model work stops 5 min before the end and the child is stopped at the end. A model process that fails or hangs is
restarted with safer engine settings while time allows.
"""
import time

T0 = time.time()

import argparse  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.isdir(os.path.join(HERE, "lc")) else os.path.dirname(HERE)  # lc/ sits beside this file in the zip
sys.path.insert(0, ROOT)
from lc.combine import FALLBACK, LETTERS, decide, open_speakers, vote  # noqa: E402
from lc.data import case_from_row  # noqa: E402
from lc.rules import solve  # noqa: E402
from lc.rules.core import Result  # noqa: E402

ATTEMPTS = [  # (judge.py arguments, environment), tried in order until one exits cleanly; as in kaggle/launcher.py
    ([], {}),
    (["--eager"], {}),
    (["--eager"], {"VLLM_USE_TRITON_AWQ": "1", "VLLM_ATTENTION_BACKEND": "TRITON_ATTN", "LC_GPU_UTIL": "0.85"}),
    (["--eager", "--tp", "1"], {"LC_GPU_UTIL": "0.85", "LC_MAX_LEN": "5120"}),  # 5,120 as on two cards: prompt + 3,072
]
# How the judge model that setup.sh downloads runs; the environment can override each value. First samples stop at
# 3,072 tokens: on two T4 cards more cases then get a thinking answer in time than with 4,096 (README, Choosing the model).
MODEL_ENV = {"LC_MAIN_TOKENS": "3072",
             "LC_CONT_TOKENS": "3072",  # = the cap, so no continuations: spare time goes to extra samples (README)
             "LC_TOKEN_LOGPROBS": "0"}  # no log-probabilities on every token: 15% faster on T4, the letter votes alone
START_LIMIT, STALL_LIMIT = 10 * 60, 5 * 60  # seconds without a heartbeat from judge.py before it counts as hung;
# Kaggle's two T4 cards start the engine in under 3 min
NO_RETRY = (3, 4)  # judge.py exit codes: no GPU, no vLLM


def log(msg):
    print(f"[predict {(time.time() - T0) / 60:6.1f} min] {msg}", file=sys.stderr, flush=True)


def read_cases(path):
    """(ids in file order, fixed answers id -> (letter, tier), cases left to the model, their open speakers)."""
    csv.field_size_limit(2**31 - 1)  # the default 128 KiB per field would stop the whole run on one long chat
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:  # a stray byte changes one case, not all
        rows = list(csv.DictReader(f))
    ids, fixed, cases, cands = [], {}, {}, {}
    for r in rows:
        cid = r.get("id")
        if cid is None or cid in fixed or cid in cases:
            continue
        ids.append(cid)
        try:
            case = case_from_row(r)
        except Exception as e:  # only the type is logged: the message can quote the case's text
            log(f"case {cid} cannot be read ({type(e).__name__}): fallback letter")
            fixed[cid] = (FALLBACK, "unreadable")
            continue
        try:
            res = solve(case)
            if any(not s.startswith("All times in this case refer to") for s in res.unconsumed):
                # evidence no rule family read can change what the rest means: no rule answer, nobody cleared
                letter, tier, res = None, "fallback", Result(cid)
            else:
                letter, tier = decide(case, [])  # no model files yet: a case the rules leave open comes back as "fallback"
            open_ = "".join(sorted(open_speakers(case, res))) if tier == "fallback" else ""
        except Exception as e:
            log(f"case {cid}: the rules failed ({type(e).__name__}); the model judges every speaker")
            letter, tier, open_ = None, "fallback", ""
        if tier == "fallback":
            cases[cid], cands[cid] = case, open_ or "".join(LETTERS)
        else:
            fixed[cid] = (letter, tier)
    return ids, fixed, cases, cands


def merge(path, pos, recs):
    """Reads the complete lines judge.py appended after byte pos into recs ((id, k) -> latest record); new pos."""
    if not os.path.exists(path):
        return pos
    with open(path, "rb") as f:
        f.seek(pos)
        data = f.read()
    end = data.rfind(b"\n") + 1
    for line in data[:end].decode("utf-8").splitlines():
        r = json.loads(line)
        old = recs.get((r["id"], r["k"]))
        if old is None or r["stage"] >= old["stage"]:
            recs[(r["id"], r["k"])] = r
    return pos + end


def write_predictions(path, ids, fixed, cands, recs):
    by_k = collections.defaultdict(dict)  # sample number -> {id: record}, one vote each; -1 = without thinking
    for (cid, k), r in recs.items():
        if r["pred"] in LETTERS:
            by_k[k][cid] = r
    models = [(1.0, by_k[k]) for k in sorted(by_k) if k >= 0]
    judged = {cid for _, m in models for cid in m}
    models.append((1.0, {cid: r for cid, r in by_k[-1].items() if cid not in judged}))
    rows, tiers = [], collections.Counter()
    for cid in ids:
        if cid in cands:
            letter, _ = vote(cid, sorted(cands[cid]), models)
            tier = "model" if cid in judged else "model without thinking" if letter else "fallback"
            letter = letter or (FALLBACK if FALLBACK in cands[cid] else cands[cid][0])
        else:
            letter, tier = fixed[cid]
        rows.append((cid, letter))
        tiers[tier] += 1
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        w.writerows(rows)
    os.replace(path + ".tmp", path)  # a reader never sees a half-written file
    return tiers


def stop(child):
    if os.name == "posix":
        try:
            os.killpg(child.pid, signal.SIGKILL)  # the child's session holds the engine's processes too
        except OSError:
            pass
    elif child.poll() is None:
        child.kill()
    try:
        child.wait(timeout=60)
    except subprocess.TimeoutExpired:  # stuck in the driver: the final file matters more than reaping it
        log(f"model process {child.pid} did not exit within 60 s of SIGKILL; going on without it")


def gpu_ids():
    """The cards judge.py may use: CUDA_VISIBLE_DEVICES when it is set, else every card nvidia-smi lists."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return [d.strip() for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if d.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [str(i) for i in range(out.count("GPU "))]


def run_model(a, cases, cands, recs, write, stop_new, hard):
    """Runs judge.py until it finishes or time is up, restarting it after a failure, and merges its answers into recs
    as they arrive, so an error here keeps them. With LC_DATA_PARALLEL=1 and two or more cards, one judge.py per card
    takes every n-th case on its own."""
    work = a.workdir or tempfile.mkdtemp(prefix="liars_")
    os.makedirs(work, exist_ok=True)
    settings = dict(MODEL_ENV, **{k: v for k, v in os.environ.items() if k in MODEL_ENV})
    gpus = gpu_ids() if os.environ.get("LC_DATA_PARALLEL", settings.get("LC_DATA_PARALLEL")) == "1" else []
    parts = len(gpus) if len(gpus) >= 2 else 1
    py = next((p for p in (os.environ.get("LC_PYTHON"), os.path.join(HERE, ".venv", "bin", "python")) if p and os.path.exists(p)),
              sys.executable)
    if os.path.isdir(os.path.join(HERE, ".venv")) and not os.path.exists(os.path.join(HERE, ".venv", "bin", "python")):
        log(f"ERROR: .venv/bin/python does not resolve (a moved folder, or the Python it links to is gone): the model "
            f"process runs with {py}; run setup.sh again")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # a PYTHONPATH made for another Python must not leak in
    env.update(settings, PYTHONPATH=ROOT, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
    workers = []
    for i in range(parts):
        w = {x: os.path.join(work, f"{x}{i}") for x in ("cases", "answers", "heartbeat")}
        with open(w["cases"], "w", encoding="utf-8") as f:
            for cid in list(cases)[i::parts]:
                c = cases[cid]
                f.write(json.dumps({"id": cid, "scenario": c.scenario, "speakers": c.speakers, "evidence": c.evidence,
                                    "messages": [[m.num, m.label, m.text] for m in c.messages], "cands": cands[cid]}) + "\n")
        w.update(n=i, attempt=0, child=None, pos=0, env=dict(env, CUDA_VISIBLE_DEVICES=gpus[i]) if parts > 1 else env)
        workers.append(w)
    log(f"model answers go to {work}; {parts} model process(es)")

    def launch(w):
        extra, attempt_env = ATTEMPTS[w["attempt"]]
        if os.path.exists(w["heartbeat"]):
            os.remove(w["heartbeat"])
        if os.path.exists(w["answers"]):  # cut a line a killed process left half-written, before the next one appends
            with open(w["answers"], "rb+") as f:
                f.truncate(f.read().rfind(b"\n") + 1)
        cmd = [py, os.path.join(HERE, "judge.py"), "--cases", w["cases"], "--out", w["answers"], "--model", a.model,
               "--stop-new", f"{stop_new:.0f}", "--hard-stop", f"{hard:.0f}", "--heartbeat", w["heartbeat"], *extra,
               *(["--tp", "1"] if parts > 1 else [])]
        log(f"$ {' '.join(cmd)} {attempt_env or ''}")
        w.update(child=subprocess.Popen(cmd, env=dict(w["env"], **attempt_env), start_new_session=os.name == "posix"),
                 started=time.time())

    last_write, unwritten = time.time(), False
    try:
        for w in workers:
            launch(w)
        while any(w["child"] for w in workers):
            time.sleep(5)
            for w in workers:
                if not w["child"]:
                    continue
                pos = merge(w["answers"], w["pos"], recs)
                unwritten, w["pos"] = unwritten or pos != w["pos"], pos
                now = time.time()
                beat = os.path.getmtime(w["heartbeat"]) if os.path.exists(w["heartbeat"]) else None
                why = ("time is up" if now > hard + 60 else
                       "the engine did not start" if beat is None and now - w["started"] > START_LIMIT else
                       "no progress" if beat is not None and now - beat > STALL_LIMIT else None)
                if w["child"].poll() is None and not why:
                    continue
                if why:
                    log(f"stopping model process {w['n']}: {why}")
                stop(w["child"])
                pos = merge(w["answers"], w["pos"], recs)
                unwritten, w["pos"] = unwritten or pos != w["pos"], pos
                rc, w["child"] = w["child"].returncode, None
                log(f"model process {w['n']} exited with code {rc}; {len(recs)} answers so far")
                if rc == 0 or rc is None or rc in NO_RETRY or time.time() > stop_new - 10 * 60:  # None: still holds the cards
                    continue
                if w["attempt"] + 1 < len(ATTEMPTS):
                    log(f"MODEL PROCESS {w['n']} FAILED; restarting it with safer settings")
                    # after a start that hung, one card at once: the attempts in between start on two cards again
                    w["attempt"] = len(ATTEMPTS) - 1 if why == "the engine did not start" else w["attempt"] + 1
                    launch(w)
                else:
                    log(f"MODEL PROCESS {w['n']} FAILED on every attempt")
            if unwritten and time.time() - last_write > 30:
                write(recs)
                last_write, unwritten = time.time(), False
    finally:
        for w in workers:
            if w["child"] is not None and w["child"].poll() is None:
                stop(w["child"])
        if not a.workdir:
            shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=os.environ.get("LC_MODEL_DIR", os.path.join(HERE, "models", "judge")),
                    help="local model folder (default: models/judge, where setup.sh puts it, or $LC_MODEL_DIR)")
    ap.add_argument("--minutes", type=float, help="time budget; default 80 for 450 to 1,500 cases, 80 x cases / 450 below, 80 x model cases / 450 above")
    ap.add_argument("--no-llm", action="store_true", help="rules and fallbacks only")
    ap.add_argument("--workdir", help="folder for the model's answers (default: a new temporary folder)")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # runs the finally blocks, which stop the model process

    ids, fixed, cases, cands = read_cases(a.input)
    # The limit is 90 min for about 450 cases. 450 to 1,500 cases get 80 min and fewer get the same time per case. A bigger
    # file (test.csv, when predict.py is checked against our submission) gets that time per case the model judges.
    n, n_model = len(ids), len(cases)
    minutes = a.minutes or 80 * (n / 450 if n < 450 else 1 if n <= 1500 else max(1, n_model / 450))
    hard = T0 + 60 * minutes
    stop_new = hard - 5 * 60
    tiers = write_predictions(a.output, ids, fixed, cands, {})
    log(f"{len(ids)} cases: {dict(tiers)}; {len(cases)} left to the model; time budget {minutes:.1f} min")

    recs = {}
    if not cases:
        pass
    elif a.no_llm:
        log("--no-llm: the rule answers and fallbacks stand")
    elif not os.path.isdir(a.model):
        log(f"NO MODEL FOLDER AT {a.model}: run setup.sh or pass --model; the rule answers and fallbacks stand")
    elif stop_new - time.time() < 3 * 60:
        log("too little time to start the model: the rule answers and fallbacks stand")
    else:
        try:
            run_model(a, cases, cands, recs, lambda r: write_predictions(a.output, ids, fixed, cands, r), stop_new, hard)
        except Exception as e:
            log(f"MODEL STEP FAILED: {e!r}; the answers so far stand")
    tiers = write_predictions(a.output, ids, fixed, cands, recs)
    first = [r for (cid, k), r in recs.items() if k == 0]
    log(f"wrote {len(ids)} predictions to {a.output}: {dict(tiers)}")
    log(f"model: {len(first)} of {len(cases)} cases have a first sample, {sum(r['forced'][0] for r in first)} of them forced; "
        f"{sum(r['kind'] == 'cont' for r in first)} continued to 16,384 tokens; "
        f"{sum(1 for (cid, k) in recs if k > 0)} extra samples; {sum(1 for (cid, k) in recs if k < 0)} answers without thinking")


if __name__ == "__main__":
    main()
