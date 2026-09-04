"""
Config-sweep tuner for OneAMR.

Finds the best-performing llama.cpp runtime configuration (n_threads,
n_batch) for THIS model + hardware target, measured via the same
llama-cpp-python runtime app.py actually uses at inference time -- not a
separate llama-bench binary, so the numbers this produces are directly
trustworthy for what gets shipped.

IMPORTANT CONSTRAINT: the ADTC judges' machine has exactly 4 CPU cores.
This sweep never tests n_threads above 4, even if this dev machine has
more -- a setting that only wins here would be meaningless (or actively
misleading) for the actual evaluation hardware.

This is a separate, narrower optimization from cache_warmup.py's prefix
caching: this picks the best GLOBAL runtime config (threads/batch size),
while prefix caching exploits a property specific to this workload's
fixed, repeated system prompts. Both apply -- run this first to pick the
config, then cache_warmup.py to warm the persona prefixes under that
config.

Usage:
    python tune.py
"""

import itertools
import json
import time
from pathlib import Path

from llama_cpp import Llama

MODEL_PATH = Path(__file__).parent / "model" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
RESULTS_PATH = Path(__file__).parent / "tune_results.json"

# ADTC target hardware has 4 CPU cores -- never sweep above that, even if
# this dev machine has more. Also try below 4 in case hyperthreading /
# contention makes fewer threads faster in practice, which does happen.
N_THREADS_GRID = [2, 3, 4]
N_BATCH_GRID = [128, 256, 512]

N_CTX = 4096
GENERATE_TOKENS = 128  # fixed-length generation for a fair timing comparison

TEST_PROMPT = (
    "Adult, watery diarrhoea for 2 days, no blood, no fever. "
    "Which antibiotic should I give?"
)


def time_one_config(n_threads: int, n_batch: int) -> dict:
    print(f"Testing n_threads={n_threads}, n_batch={n_batch} ...")

    load_start = time.perf_counter()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_threads=n_threads,
        n_batch=n_batch,
        verbose=False,
    )
    load_time = time.perf_counter() - load_start

    # Warm-up call (not timed) -- first call after load can include extra
    # one-time setup cost that would distort the measurement.
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=8,
    )

    gen_start = time.perf_counter()
    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": TEST_PROMPT}],
        max_tokens=GENERATE_TOKENS,
        temperature=0.2,
    )
    gen_time = time.perf_counter() - gen_start

    completion_tokens = result["usage"]["completion_tokens"]
    tokens_per_sec = completion_tokens / gen_time if gen_time > 0 else 0.0

    del llm  # free memory before next config

    return {
        "n_threads": n_threads,
        "n_batch": n_batch,
        "load_time_sec": round(load_time, 2),
        "completion_tokens": completion_tokens,
        "gen_time_sec": round(gen_time, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
    }


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Run `bash download_model.sh` first.")
        return

    results = []
    for n_threads, n_batch in itertools.product(N_THREADS_GRID, N_BATCH_GRID):
        try:
            r = time_one_config(n_threads, n_batch)
            results.append(r)
            print(f"  -> {r['tokens_per_sec']} tok/s "
                  f"(load {r['load_time_sec']}s, gen {r['gen_time_sec']}s)")
        except Exception as e:
            print(f"  -> FAILED: {e}")

    if not results:
        print("No successful runs -- nothing to report.")
        return

    results.sort(key=lambda r: r["tokens_per_sec"], reverse=True)

    print("\n=== Results, best to worst ===")
    for r in results:
        print(f"  threads={r['n_threads']}, batch={r['n_batch']}: "
              f"{r['tokens_per_sec']} tok/s")

    best = results[0]
    print(f"\nBest config: n_threads={best['n_threads']}, "
          f"n_batch={best['n_batch']} -> {best['tokens_per_sec']} tok/s")
    print("\nUpdate app.py's --n_threads default and add/update n_batch in "
          "the Llama(...) constructor to match this, then rerun "
          "cache_warmup.py so the warmed prefixes match the new config.")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
