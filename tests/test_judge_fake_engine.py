"""submission/judge.py against a fake vLLM engine (no GPU): the order first samples start in, and what happens at the
stop time. Each scenario runs judge.py for a few seconds of wall time.  python -m pytest tests"""
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAKE_VLLM = r'''
import json, os, time


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
    """Thinking requests: a first sample (seed 0) ends at once unless its prompt holds SLOWCASE; an extra sample ends
    only after FAKE_EXTRA_END (unix time). One-token requests end at once."""
    vllm_config = _Cfg()

    def __init__(self):
        self.q = []

    def add_request(self, rid, prompt, sp):
        if sp.max_tokens > 1 and sp.seed == 0:
            with open(os.environ["FAKE_LOG"], "a") as f:
                f.write(json.dumps(len(prompt.split())) + "\n")
        self.q.append((rid, prompt, sp))

    def abort_request(self, rids):
        self.q = [x for x in self.q if x[0] not in set(rids)]

    def step(self):
        time.sleep(0.01)
        done, keep = [], []
        for rid, prompt, sp in self.q:
            if sp.max_tokens == 1:
                done.append(_RO(rid, _Out(" B", [1], [{1: _LP(" B", -0.1), 2: _LP(" C", -2.5)}])))
            elif (sp.seed == 0 and "SLOWCASE" not in prompt) or (sp.seed > 0 and time.time() >= float(os.environ["FAKE_EXTRA_END"])):
                done.append(_RO(rid, _Out("checked.</think>\n\nANSWER: B", [1] * 50)))
            else:
                keep.append((rid, prompt, sp))
        self.q = keep
        return done


class _Tok:
    def apply_chat_template(self, m, **kw):
        return "<user> " + m[0]["content"] + " <assistant><think>\n"

    def encode(self, s):
        return s.split()


class LLM:
    def __init__(self, model, **kw):
        self.llm_engine = _Eng()

    def get_tokenizer(self):
        return _Tok()
'''


def run_judge(tmp_path, slow_case, stop_in, hard_after_stop, extra_after_stop):
    """judge.py on 30 new-family cases (plus a slow one); returns (exit code, seconds, answer records, prompt lengths
    of the first samples in start order, log)."""
    fake = tmp_path / "fake"
    (fake / "vllm").mkdir(parents=True)
    (fake / "vllm" / "__init__.py").write_text(FAKE_VLLM)
    (fake / "vllm" / "sampling_params.py").write_text("class RequestOutputKind:\n    FINAL_ONLY = 2\n")
    (fake / "torch.py").write_text("class cuda:\n    @staticmethod\n    def device_count():\n        return 2\n")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    sys.path.insert(0, ROOT)
    from lc.data import case_from_row
    rows = list(csv.DictReader(open(os.path.join(ROOT, "validation", "newfam", "newfam.csv"), encoding="utf-8")))[:30]
    if slow_case:
        rows.append(dict(rows[0], id="SLOW", evidence=rows[0]["evidence"] + " SLOWCASE" * 3000))
    with open(tmp_path / "cases.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            c = case_from_row(r)
            f.write(json.dumps({"id": c.id, "scenario": c.scenario, "speakers": c.speakers, "evidence": c.evidence,
                                "messages": [[m.num, m.label, m.text] for m in c.messages], "cands": "ABCD"}) + "\n")
    t0 = time.time()
    stop = t0 + stop_in
    env = dict(os.environ, PYTHONPATH=str(fake) + os.pathsep + ROOT, FAKE_LOG=str(tmp_path / "starts.jsonl"),
               FAKE_EXTRA_END=str(stop + extra_after_stop), LC_SLOTS="4" if slow_case else "40", LC_TOKEN_LOGPROBS="0")
    cmd = [sys.executable, os.path.join(ROOT, "submission", "judge.py"), "--cases", str(tmp_path / "cases.jsonl"),
           "--out", str(tmp_path / "answers.jsonl"), "--model", str(tmp_path / "model"), "--stop-new", str(stop),
           "--hard-stop", str(stop + hard_after_stop), "--heartbeat", str(tmp_path / "hb")]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    recs = [json.loads(x) for x in open(tmp_path / "answers.jsonl", encoding="utf-8")]
    starts = [json.loads(x) for x in open(tmp_path / "starts.jsonl")]
    return p.returncode, time.time() - t0, recs, starts, p.stdout + p.stderr


def test_first_samples_start_shortest_prompt_first(tmp_path):
    rc, _, recs, starts, log = run_judge(tmp_path, False, 5, 3, 1)
    assert rc == 0, log
    assert len(starts) == 30 and starts == sorted(starts), starts


def test_extra_samples_run_on_at_the_stop_time_when_no_first_sample_is_left(tmp_path):
    rc, secs, recs, _, log = run_judge(tmp_path, False, 5, 4, 1.5)  # extras end 1.5 s after the stop time
    assert rc == 0, log
    assert "0 running continuations stopped" in log, log
    assert len([r for r in recs if r["k"] > 0]) == 30 and secs < 5 + 4, log  # every extra kept, and done before the hard stop


def test_extra_samples_stop_while_a_first_sample_is_still_running(tmp_path):
    rc, _, recs, _, log = run_judge(tmp_path, True, 5, 3, 1)
    assert rc == 0, log
    assert "3 running continuations and extra samples stopped" in log, log
    assert not [r for r in recs if r["k"] > 0]
    assert [r["kind"] for r in recs if r["id"] == "SLOW"] == ["nothink"]  # its first sample gave way to the one-token answer


def test_the_hard_stop_ends_extra_samples_that_run_on(tmp_path):
    rc, secs, recs, _, log = run_judge(tmp_path, False, 5, 2, 1000)  # extras would never end
    assert rc == 0, log
    assert not [r for r in recs if r["k"] > 0] and secs < 5 + 2 + 3, (secs, log)
