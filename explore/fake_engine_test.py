"""Runs submission/judge.py against a fake vLLM engine (no GPU): every case must end with one answer for its first sample,
capped samples must take the forced path that LC_FORCE selects, and the counts must add up.

python explore/fake_engine_test.py
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAKE_VLLM = r'''
import random
random.seed(7)


class SamplingParams:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _LP:
    def __init__(self, tok, lp):
        self.decoded_token, self.logprob = tok, lp


class _Out:
    def __init__(self, text, ids, logprobs=None):
        self.text, self.token_ids, self.logprobs = text, ids, logprobs


class _RO:
    def __init__(self, rid, out):
        self.request_id, self.outputs, self.finished = rid, [out], True


class _Cache:
    num_gpu_blocks, block_size = 6000, 16


class _Cfg:
    cache_config = _Cache()


class _Eng:
    vllm_config = _Cfg()

    def __init__(self):
        self.q = []

    def add_request(self, rid, prompt, sp):
        self.q.append((rid, prompt, sp))

    def abort_request(self, rids):
        self.q = [x for x in self.q if x[0] not in set(rids)]

    def step(self):
        outs, self.q = self.q[:8], self.q[8:]
        res = []
        for rid, prompt, sp in outs:
            if sp.max_tokens == 1:  # one-token answer
                res.append(_RO(rid, _Out(" B", [1], [{1: _LP(" B", -0.1), 2: _LP(" C", -2.5)}])))
            elif sp.temperature == 0.0:  # the short answer after the budget sentence
                text = "So the answer is C.\n\nANSWER: C" if random.random() < 0.7 else "I am not sure."
                res.append(_RO(rid, _Out(text, [1] * 12)))
            elif random.random() < 0.4:  # capped while still thinking
                res.append(_RO(rid, _Out("hmm " * 10, [1] * sp.max_tokens)))
            else:
                res.append(_RO(rid, _Out("checked.</think>\n\nANSWER: " + random.choice("ABCD"), [1] * 50)))
        return res


class _Tok:
    def apply_chat_template(self, m, **kw):
        return "<user>" + m[0]["content"] + "<assistant><think>\n"

    def encode(self, s):
        return s.split()


class LLM:
    def __init__(self, model, **kw):
        self.llm_engine = _Eng()

    def get_tokenizer(self):
        return _Tok()
'''

FAKE_SP = "class RequestOutputKind:\n    FINAL_ONLY = 2\n    DELTA = 1\n"
FAKE_TORCH = "class cuda:\n    @staticmethod\n    def device_count():\n        return 2\n"


def run(env_extra):
    d = tempfile.mkdtemp(prefix="fake_judge_")
    os.makedirs(os.path.join(d, "vllm"))
    with open(os.path.join(d, "vllm", "__init__.py"), "w") as f:
        f.write(FAKE_VLLM)
    with open(os.path.join(d, "vllm", "sampling_params.py"), "w") as f:
        f.write(FAKE_SP)
    with open(os.path.join(d, "torch.py"), "w") as f:
        f.write(FAKE_TORCH)
    os.makedirs(os.path.join(d, "model"))
    with open(os.path.join(d, "model", "config.json"), "w") as f:
        json.dump({}, f)
    sys.path.insert(0, ROOT)
    from lc.data import case_from_row
    rows = list(csv.DictReader(open(os.path.join(ROOT, "explore", "newfam.csv"), encoding="utf-8")))[:30]
    with open(os.path.join(d, "cases.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            c = case_from_row(r)
            f.write(json.dumps({"id": c.id, "scenario": c.scenario, "speakers": c.speakers, "evidence": c.evidence,
                                "messages": [[m.num, m.label, m.text] for m in c.messages], "cands": "ABCD"}) + "\n")
    now = time.time()
    env = dict(os.environ, PYTHONPATH=d + os.pathsep + ROOT, **env_extra)
    cmd = [sys.executable, os.path.join(ROOT, "submission", "judge.py"), "--cases", os.path.join(d, "cases.jsonl"),
           "--out", os.path.join(d, "answers.jsonl"), "--model", os.path.join(d, "model"), "--stop-new", str(now + 600),
           "--hard-stop", str(now + 900), "--heartbeat", os.path.join(d, "hb")]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    recs = [json.loads(x) for x in open(os.path.join(d, "answers.jsonl"), encoding="utf-8")]
    return p, recs, len(rows)


for mode in ({}, {"LC_FORCE": "budget"}):
    p, recs, n = run(mode)
    first = {}
    for r in recs:
        if r["k"] == 0 and (r["id"] not in first or r["stage"] >= first[r["id"]]["stage"]):
            first[r["id"]] = r
    forced = [r for r in first.values() if r["forced"][0]]
    done = [x for x in p.stdout.splitlines() if "done:" in x]
    print(mode or "default", f"exit {p.returncode}; {len(first)}/{n} cases have a first answer; forced {len(forced)} "
          f"(letters {sorted({r['pred'] for r in forced})}); {done[-1][-160:] if done else p.stdout[-600:] + p.stderr[-600:]}")
    assert p.returncode == 0 and len(first) == n, "every case must end with a first answer"
    if mode:
        assert any(r["pred"] == "C" for r in forced) and any(r["pred"] == "B" for r in forced), \
            "budget answers (C) and their one-token fallback (B) must both appear"
    else:
        assert all(r["pred"] == "B" for r in forced), "default: one-token answers only"
print("ok")
