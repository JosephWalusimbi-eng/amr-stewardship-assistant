"""
AMR Stewardship Assistant -- ADTC 2026 Gate 1 submission.

An offline antimicrobial-stewardship assistant for Uganda, grounded in the
WHO AWaRe antibiotic book, Uganda Clinical Guidelines 2023, and Uganda's
NAP-AMR II (2024/25-2028/29). Runs entirely on-device via llama.cpp.

Three registers (system prompts):
    health_worker -- prescribing/dosing/AWaRe-classification support
    patient       -- plain-language adherence and education
    farmer        -- livestock antibiotic use, withdrawal periods (One Health)

Usage:
    python app.py --persona patient
    python app.py --persona health_worker --prompt "Is ceftriaxone Access, Watch or Reserve?"
"""

import argparse
import pickle
import sys
from pathlib import Path

from llama_cpp import Llama

from rag.retrieve import retrieve, format_context
from system_prompts import SYSTEM_PROMPTS

MODEL_PATH = Path(__file__).parent / "model" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
CACHE_DIR = Path(__file__).parent / "cache"


def build_prompt(persona: str, user_query: str, history: list[dict] | None = None) -> list[dict]:
    """
    IMPORTANT for prefix caching: the system message must be byte-identical
    across every call for a given persona -- it is the reusable KV-cache
    prefix. Retrieved RAG context therefore goes in the user turn, NOT the
    system message, even though it changes per query. Putting it in system
    content would silently defeat both llama.cpp's automatic in-process
    prefix reuse and the on-disk cache warmed by cache_warmup.py.

    `history` is the accumulated prior turns in this conversation (list of
    {"role": ..., "content": ...} dicts), needed for follow-ups that
    reference earlier answers -- e.g. a patient pushing back after an
    initial refusal. Pass None or [] for a fresh, single-turn exchange.
    """
    context_chunks = retrieve(user_query, k=4)
    context_block = format_context(context_chunks)

    user_content = user_query
    if context_block:
        user_content = (
            f"Relevant reference material (cite source + page when you use it):\n"
            f"{context_block}\n\n"
            f"Question: {user_query}"
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPTS[persona]}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persona",
        choices=list(SYSTEM_PROMPTS.keys()),
        default="patient",
        help="Which register to respond in.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to answer, then exit. Omit for interactive chat.",
    )
    parser.add_argument("--n_ctx", type=int, default=4096)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--n_batch", type=int, default=512)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Run `bash download_model.sh` first.", file=sys.stderr)
        sys.exit(1)

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_batch=args.n_batch,
        verbose=False,
    )

    cache_path = CACHE_DIR / f"{args.persona}.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                state = pickle.load(f)
            llm.load_state(state)
            print(f"[cache] loaded warmed prefix for persona '{args.persona}' "
                  f"({cache_path.stat().st_size / 1024:.0f} KB)", file=sys.stderr)
        except Exception as e:
            # Never let a stale/incompatible cache file break inference --
            # worst case we just fall back to full reprocessing.
            print(f"[cache] could not load {cache_path.name}, continuing without "
                  f"it: {e}", file=sys.stderr)
    else:
        print(f"[cache] no warmed cache for '{args.persona}' -- run "
              f"cache_warmup.py to speed up cold starts. Continuing without it.",
              file=sys.stderr)

    conversation_history: list[dict] = []

    def respond(user_query: str):
        messages = build_prompt(args.persona, user_query, history=conversation_history)
        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.2,
            stream=True,
        )
        answer = ""
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            token = delta.get("content")
            if token:
                print(token, end="", flush=True)
                answer += token
        print()
        # Record this turn (using the plain user_query, not the RAG-augmented
        # version, so history doesn't compound retrieved context across turns)
        conversation_history.append({"role": "user", "content": user_query})
        conversation_history.append({"role": "assistant", "content": answer})

    if args.prompt:
        respond(args.prompt)
        return

    print(f"AMR Stewardship Assistant -- persona: {args.persona}. Ctrl+C to exit.")
    while True:
        try:
            user_query = input("\nYou: ").strip()
            if not user_query:
                continue
            print()
            respond(user_query)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
