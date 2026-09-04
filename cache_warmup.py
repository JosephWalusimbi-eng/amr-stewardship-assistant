"""
Prefix-cache warmup for the AMR Stewardship Assistant.

This is a second, narrower optimization layer on top of whatever config the
tuner picks: it exploits a property specific to THIS workload -- three fixed,
repeated system prompts (health_worker / patient / farmer) -- that a generic
tuning pass has no visibility into.

Without this, every single query reprocesses the persona's system prompt
tokens from scratch through the model, even though that prompt never
changes for a given persona. This script runs each persona's system prompt
through the model once, captures the resulting KV-cache state via
llama-cpp-python's save_state(), and pickles it to cache/<persona>.pkl.
app.py then loads the matching pickle at startup (see load_state() call)
and only has to evaluate the new user-turn tokens on each query, instead of
the full system+user prompt every time.

IMPORTANT: this only pays off if the system message fed to the model is
byte-identical between warmup and inference. That's why app.py's
build_prompt() keeps RAG context out of the system message entirely --
see the comment there. If you edit SYSTEM_PROMPTS in app.py, you MUST
rerun this script, or the cached prefix will silently mismatch and
load_state() will fall back to full reprocessing (safe, but you lose
the speedup).

Usage:
    python cache_warmup.py
"""

import pickle
import sys
from pathlib import Path

from llama_cpp import Llama

from app import SYSTEM_PROMPTS, MODEL_PATH, CACHE_DIR


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Run `bash download_model.sh` first.", file=sys.stderr)
        sys.exit(1)

    CACHE_DIR.mkdir(exist_ok=True)

    for persona, system_prompt in SYSTEM_PROMPTS.items():
        print(f"Warming cache for persona: {persona}")

        # Fresh model instance per persona so each pickle captures a clean,
        # independent KV-cache state -- no cross-persona contamination.
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_threads=4,
            n_batch=512,
            verbose=False,
        )

        # Process just the system prompt (max_tokens=1 forces the forward
        # pass over the prompt without wasting time generating output we
        # don't need -- the state we want is the KV cache *after* the
        # prompt, not any particular completion of it).
        llm.create_chat_completion(
            messages=[{"role": "system", "content": system_prompt}],
            max_tokens=1,
        )

        state = llm.save_state()
        out_path = CACHE_DIR / f"{persona}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(state, f)

        print(f"  -> saved {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

        del llm  # free before next persona to keep memory bounded

    print("\nDone. app.py will automatically pick up these cached prefixes.")


if __name__ == "__main__":
    main()
