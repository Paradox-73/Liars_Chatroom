"""One-token generation with log-probabilities from a small Qwen3.5 model, to check vLLM on this card and driver."""
import os

# FlashInfer compiles its sampling kernel at start-up and fails with this server's nvcc 11.5; use the PyTorch sampler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams  # noqa: E402

if __name__ == "__main__":  # vLLM starts its engine in a spawned process, which re-imports this file
    llm = LLM("models/qwen35-0.8b", gpu_memory_utilization=0.12, max_model_len=4096,
              limit_mm_per_prompt={"image": 0, "video": 0})
    p = ("Evidence: The cafe closes at 6 pm.\nChat:\nA: We stayed at the cafe until 8 pm.\n"
         "B: The coffee was good.\nWhich speaker contradicts the evidence? Answer with one letter.\nAnswer:")
    o = llm.generate([p], SamplingParams(max_tokens=1, temperature=0, logprobs=5))[0].outputs[0]
    print("SMOKE", repr(o.text), o.logprobs)
