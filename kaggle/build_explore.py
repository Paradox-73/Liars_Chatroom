"""Kaggle kernels for the accuracy exploration on branch explore/accuracy. Each kernel unpacks the code zip, runs
setup.sh (vLLM and Qwen3-14B-AWQ, as the organisers will) and then one experiment on two T4 cards:

  bench  steady-state decode throughput of the judge's engine for several engine settings: tensor parallel over both
         cards (as judge.py), the same with n-gram speculative decoding, and one engine per card. Every setting gets
         the same new-family prompts (longest first, as judge.py), the judge's sampling and its 4,096-token cap; tokens
         are counted per engine step in a fixed window after a warm-up. One BENCH line per setting.
  probe  several samples per case with the reasoning texts saved, for the failure analysis: the base prompt and,
         optionally, a variant with extra rule lines. Writes probe_out.jsonl.
  pipe   predict.py on a labelled CSV with the time budget of the hidden run (the bake-off driver), with optional
         environment overrides; prints the SUMMARY line.

python kaggle/build_explore.py bench --zip Z --csv validation/newfam/newfam.csv --owner U --out kaggle/explore-bench14-U
python kaggle/build_explore.py probe --zip Z --csv probe.csv --owner U --out kaggle/explore-probe14-U [--variant rules.txt]
python kaggle/build_explore.py pipe --zip Z --csv set2.csv --meta set2_meta.csv --owner U --out kaggle/explore-set2-U
"""
import argparse
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_bakeoff import DRIVER as PIPE_DRIVER  # noqa: E402

MODEL = {"repo": "Qwen/Qwen3-14B-AWQ", "rev": "31c69efc29464b6bb0aee1398b5a7b50a99340c3"}

BENCH = r'''
import csv, json, os, sys, time
sys.path.insert(0, "/tmp/sub")
from lc.data import case_from_row
from lc.llm.prompt import build_prompt
from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind

c = json.loads(sys.argv[1])
rows = list(csv.DictReader(open("/tmp/cases.csv", encoding="utf-8")))
kw = dict(tensor_parallel_size=c["tp"], dtype="float16", gpu_memory_utilization=0.92, max_model_len=c["max_len"],
          max_num_seqs=c["seqs"], max_num_batched_tokens=4096, seed=0, disable_custom_all_reduce=True)
kw.update(c.get("engine", {}))
t_load = time.time()
llm = LLM("/tmp/model", **kw)
eng, tok = llm.llm_engine, llm.get_tokenizer()
print(f"BENCHLOG {c['tag']}: engine ready in {time.time() - t_load:.0f} s", flush=True)
prompts = {r["id"]: tok.apply_chat_template([{"role": "user", "content": build_prompt(case_from_row(r))}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=True) for r in rows}
labels = {r["id"]: r["label"] for r in rows}
order = sorted(prompts, key=lambda i: -len(tok.encode(prompts[i])))[c.get("part", 0)::c.get("parts", 1)]
for i in order:
    eng.add_request(i, prompts[i], SamplingParams(temperature=0.6, top_p=0.95, top_k=20, seed=0, max_tokens=4096,
                                                  output_kind=RequestOutputKind.DELTA))
warm, window = c["warm"], c["window"]
t0 = time.time()
t_start = t_end = None
tokens = steps = finished = right = 0
texts = {}
while eng.has_unfinished_requests():
    t = time.time()
    if t - t0 > warm + window:
        break
    outs = eng.step()
    counting = t - t0 >= warm
    if counting and t_start is None:
        t_start = t
    for o in outs:
        new = len(o.outputs[0].token_ids)
        texts[o.request_id] = texts.get(o.request_id, "") + o.outputs[0].text
        if counting:
            tokens += new
        if o.finished:
            finished += 1
            tail = texts[o.request_id].rsplit("</think>", 1)[-1]
            right += f"ANSWER: {labels[o.request_id]}" in tail
    if counting:
        steps += 1
        t_end = time.time()
dt = (t_end or time.time()) - (t_start or time.time())
print(f"BENCH {c['tag']}: {tokens} tokens in {dt:.0f} s = {tokens / max(dt, 1e-9):.1f} tokens/s over {steps} steps; "
      f"{finished} finished in {time.time() - t0:.0f} s, {right} of them with the right letter", flush=True)
import psutil  # the engine's processes rename themselves (VLLM::EngineCore, VLLM::Worker_TP0) and outlive os._exit
for child in psutil.Process().children(recursive=True):
    try:
        child.kill()
    except psutil.Error:
        pass
os._exit(0)
'''

BENCH_DRIVER = r'''
import base64, io, json, os, subprocess, time, zipfile

T0 = time.time()


def log(msg):
    print(f"[bench {(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def used_mib():
    out = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", shell=True,
                         capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def sh(cmd, check=False):
    log(f"$ {cmd}")
    rc = subprocess.run(cmd, shell=True, check=check, executable="/bin/bash").returncode
    # engines left behind hold the cards; vLLM renames its processes, so match those names too, then wait for the cards
    for _ in range(36):
        # the brackets keep each pattern from matching this shell's own command line
        subprocess.run("pkill -9 -f '[/]tmp/sub/.venv/bin/python'; pkill -9 -f '[V]LLM::'", shell=True)
        time.sleep(5)
        if all(u < 1500 for u in used_mib()):
            break
    log(f"cards after cleanup: {used_mib()} MiB used")
    return rc


zipfile.ZipFile(io.BytesIO(base64.b64decode(ZIP))).extractall("/tmp/sub")
with open("/tmp/cases.csv", "w", encoding="utf-8", newline="") as f:
    f.write(CASES)
with open("/tmp/bench.py", "w") as f:
    f.write(BENCH)
sh("nvidia-smi --query-gpu=index,name,memory.total,clocks.max.sm,power.limit --format=csv; nproc; free -g | head -2")
sh(f"LC_MODEL_REPO={CFG['repo']} LC_MODEL_REV={CFG['rev']} LC_MODEL_DIR=/tmp/model bash /tmp/sub/setup.sh", check=True)
py = "VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 PYTHONPATH=/tmp/sub /tmp/sub/.venv/bin/python /tmp/bench.py"
W = {"warm": 90, "window": 330}
for run in CFG["runs"]:
    env = " ".join(f"{k}={v}" for k, v in run.get("env", {}).items())  # per-setting environment, e.g. a backend
    if run.get("dp"):  # one engine per card, each with every second prompt; the two BENCH lines add up
        cmds = [f"CUDA_VISIBLE_DEVICES={i} {env} {py} '{json.dumps({**W, **run, 'tag': run['tag'] + f'-card{i}', 'part': i, 'parts': 2})}'"
                for i in (0, 1)]
        sh(f"({cmds[0]} & {cmds[1]} & wait)")
    else:
        sh(f"{env} {py} '{json.dumps({**W, **run})}'")
log("done")
'''

PROBE = r'''
import csv, json, os, re, sys, time
sys.path.insert(0, "/tmp/sub")
from lc.data import case_from_row
from lc.llm.prompt import build_prompt
from vllm import LLM, SamplingParams

c = json.loads(open("/tmp/probe_cfg.json").read())
rows = list(csv.DictReader(open("/tmp/cases.csv", encoding="utf-8")))
llm = LLM("/tmp/model", tensor_parallel_size=2, dtype="float16", gpu_memory_utilization=0.92,
          max_model_len=c["max_tokens"] + 2560, max_num_seqs=40, max_num_batched_tokens=4096, seed=0,
          disable_custom_all_reduce=True)
tok = llm.get_tokenizer()
ANCHOR = "\n- Questions and opinions are never the answer."
reqs, prompts, params = [], [], []
for r in rows:
    base = build_prompt(case_from_row(r))
    arms = {"base": base}
    if c.get("variant"):
        assert ANCHOR in base
        arms["variant"] = base.replace(ANCHOR, ANCHOR + "\n" + c["variant"].strip("\n"))
    for arm, text in arms.items():
        for k in range(int(r["samples"])):
            reqs.append((r["id"], arm, k))
            prompts.append(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True))
            params.append(SamplingParams(temperature=0.6, top_p=0.95, top_k=20, seed=k, max_tokens=c["max_tokens"]))
print(f"[probe] {len(rows)} cases, {len(reqs)} samples, cap {c['max_tokens']}", flush=True)
t = time.time()
outs = llm.generate(prompts, params, use_tqdm=False)
labels = {r["id"]: r["label"] for r in rows}
right = {}
with open("/kaggle/working/probe_out.jsonl", "w", encoding="utf-8") as f:
    for (cid, arm, k), o in zip(reqs, outs):
        text = o.outputs[0].text
        tail = text.rsplit("</think>", 1)[1] if "</think>" in text else ""
        found = re.findall(r"ANSWER\W{0,6}([ABCD])\b", tail)
        letter = found[-1] if found else None
        f.write(json.dumps({"id": cid, "arm": arm, "k": k, "pred": letter, "label": labels[cid],
                            "tokens": len(o.outputs[0].token_ids), "capped": "</think>" not in text, "text": text}) + "\n")
        right.setdefault(arm, [0, 0])
        right[arm][0] += letter == labels[cid]
        right[arm][1] += 1
tok_total = sum(len(o.outputs[0].token_ids) for o in outs)
print(f"PROBE {json.dumps({a: f'{v[0]}/{v[1]}' for a, v in right.items()})}; {tok_total} tokens in {time.time() - t:.0f} s",
      flush=True)
os._exit(0)
'''

PROBE_DRIVER = r'''
import base64, io, json, os, subprocess, time, zipfile

T0 = time.time()


def log(msg):
    print(f"[probe {(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def sh(cmd, check=False):
    log(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, executable="/bin/bash").returncode


zipfile.ZipFile(io.BytesIO(base64.b64decode(ZIP))).extractall("/tmp/sub")
with open("/tmp/cases.csv", "w", encoding="utf-8", newline="") as f:
    f.write(CASES)
with open("/tmp/probe.py", "w") as f:
    f.write(PROBE)
with open("/tmp/probe_cfg.json", "w") as f:
    json.dump(CFG, f)
sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")
sh(f"LC_MODEL_REPO={CFG['repo']} LC_MODEL_REV={CFG['rev']} LC_MODEL_DIR=/tmp/model bash /tmp/sub/setup.sh", check=True)
sh("VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 PYTHONPATH=/tmp/sub /tmp/sub/.venv/bin/python /tmp/probe.py")
log("done")
'''

BENCH_RUNS = [
    {"tag": "tp2-40", "tp": 2, "seqs": 40, "max_len": 18432},  # judge.py's engine and its 40 thinking slots
    {"tag": "tp2-40-ngram4", "tp": 2, "seqs": 40, "max_len": 18432,
     "engine": {"speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4,
                                       "prompt_lookup_min": 2}}},
    {"tag": "dp-10", "tp": 1, "seqs": 10, "max_len": 8192, "dp": True},
    {"tag": "dp-16-fp8kv", "tp": 1, "seqs": 16, "max_len": 8192, "dp": True, "engine": {"kv_cache_dtype": "fp8_e5m2"}},
]
# second set: the settings with a smaller expected gain, each against the baseline on the same machine: a larger
# cache block, 56 slots with an 8-bit cache (twice the room), a shorter speculation, FlexAttention
BENCH_RUNS_B = [
    {"tag": "tp2-40", "tp": 2, "seqs": 40, "max_len": 18432},
    {"tag": "tp2-40-block32", "tp": 2, "seqs": 40, "max_len": 18432, "engine": {"block_size": 32}},
    {"tag": "tp2-56-fp8e4m3", "tp": 2, "seqs": 56, "max_len": 18432, "engine": {"kv_cache_dtype": "fp8_e4m3"}},
    {"tag": "tp2-40-ngram2", "tp": 2, "seqs": 40, "max_len": 18432,
     "engine": {"speculative_config": {"method": "ngram", "num_speculative_tokens": 2, "prompt_lookup_max": 3,
                                       "prompt_lookup_min": 2}}},
    {"tag": "tp2-40-flex", "tp": 2, "seqs": 40, "max_len": 18432, "env": {"VLLM_ATTENTION_BACKEND": "FLEX_ATTENTION"}},
]
# third set, most likely gains first: fitting the engine log, each step costs about 30 ms plus 2.6 ms per 1,000 tokens
# of cache in use, about a tenth of the T4's memory bandwidth, so the attention kernel limits decode: FlashInfer, the
# separate prefill and decode kernels, an 8-bit cache, n-gram speculation, and one engine per card
BENCH_RUNS_C = [
    {"tag": "tp2-40", "tp": 2, "seqs": 40, "max_len": 18432},
    {"tag": "tp2-40-flashinfer", "tp": 2, "seqs": 40, "max_len": 18432, "env": {"VLLM_ATTENTION_BACKEND": "FLASHINFER"}},
    {"tag": "tp2-40-splitkernels", "tp": 2, "seqs": 40, "max_len": 18432,
     "env": {"VLLM_V1_USE_PREFILL_DECODE_ATTENTION": "1"}},
    {"tag": "tp2-40-fp8e4m3", "tp": 2, "seqs": 40, "max_len": 18432, "engine": {"kv_cache_dtype": "fp8_e4m3"}},
    {"tag": "tp2-40-ngram4", "tp": 2, "seqs": 40, "max_len": 18432,
     "engine": {"speculative_config": {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4,
                                       "prompt_lookup_min": 2}}},
    {"tag": "dp-10", "tp": 1, "seqs": 10, "max_len": 8192, "dp": True},
]


def variant_zip(path, lines):
    """The code zip with lines appended to the rule list of lc/llm/prompt.py (the prompt arm the probe calls variant)."""
    import io
    import zipfile
    end = '- Questions and opinions are never the answer."""'
    buf = io.BytesIO()
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info)
            if info.filename == "lc/llm/prompt.py":
                text = data.decode("utf-8")
                assert text.count(end) == 1
                data = text.replace(end, end[:-3] + "\n" + lines.strip("\n") + '"""').encode("utf-8")
            zout.writestr(info, data)
    return buf.getvalue()


def write_kernel(out, owner, run_py):
    compile(run_py, "run.py", "exec")
    slug = os.path.basename(os.path.normpath(out))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(run_py)
    meta = {"id": f"{owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "model_sources": []}
    with open(os.path.join(out, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{out}: {owner}/{slug}, {len(run_py) / 1024:.0f} KiB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["bench", "probe", "pipe"])
    ap.add_argument("--zip", required=True)
    ap.add_argument("--csv", required=True, help="labelled cases (train.csv columns); probe: plus a samples column")
    ap.add_argument("--meta", help="pipe: id,family CSV for the per-family accuracy")
    ap.add_argument("--owner", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", help="probe: extra rule lines for a second prompt arm; pipe: run with them")
    ap.add_argument("--max-tokens", type=int, default=12288, help="probe: cap per sample")
    ap.add_argument("--env", default="{}", help="pipe: JSON of environment overrides for predict.py")
    ap.add_argument("--runs", default="a", choices=["a", "b", "c"], help="bench: which set of engine settings")
    a = ap.parse_args()
    with open(a.zip, "rb") as f:
        zip_b64 = base64.b64encode(f.read()).decode()
    cases = open(a.csv, encoding="utf-8", newline="").read()
    if a.kind == "bench":
        cfg = dict(MODEL, runs={"a": BENCH_RUNS, "b": BENCH_RUNS_B, "c": BENCH_RUNS_C}[a.runs])
        run_py = f"ZIP = {zip_b64!r}\nCASES = {cases!r}\nBENCH = {BENCH!r}\nCFG = {cfg!r}\n{BENCH_DRIVER}"
        compile(BENCH, "bench.py", "exec")
    elif a.kind == "probe":
        cfg = dict(MODEL, max_tokens=a.max_tokens, variant=open(a.variant, encoding="utf-8").read() if a.variant else "")
        run_py = f"ZIP = {zip_b64!r}\nCASES = {cases!r}\nPROBE = {PROBE!r}\nCFG = {cfg!r}\n{PROBE_DRIVER}"
        compile(PROBE, "probe.py", "exec")
    else:
        meta = open(a.meta, encoding="utf-8", newline="").read()
        cfg = dict(MODEL, env=json.loads(a.env))
        if a.variant:  # the same code with the variant's rule lines at the end of the prompt's rule list
            zip_b64 = base64.b64encode(variant_zip(a.zip, open(a.variant, encoding="utf-8").read())).decode()
        run_py = f"ZIP = {zip_b64!r}\nNEWFAM = {cases!r}\nMETA = {meta!r}\nCFG = {cfg!r}\n{PIPE_DRIVER}"
    write_kernel(a.out, a.owner, run_py)


if __name__ == "__main__":
    main()
