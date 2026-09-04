"""
Loads the prebuilt BM25 index (rag/index.pkl) and retrieves the top-k most
relevant passages for a given query. This is a cheap, dependency-light
retrieval step that runs outside the llama.cpp process.
"""

import pickle
import re
from pathlib import Path

INDEX_PATH = Path(__file__).parent / "index.pkl"

_index_cache = None

# Cues that identify the patient's age group from the query text, so
# retrieval can avoid handing the model a paediatric-dosing passage (e.g.
# from the WHO IMCI or Pocket Book) for a query that's clearly about an
# adult, and vice versa. Deliberately conservative: ambiguous or unmarked
# queries (both/neither pattern set matches) are left unfiltered rather
# than guessing.
ADULT_PATTERNS = [
    r"\badults?\b",
    r"\bwom[ae]n\b",
    r"\bmen\b",
    r"\bmale\b",
    r"\bfemale\b",
    r"\belderly\b",
    r"\b(1[89]|[2-9]\d)\s*[- ]?\s*(years?|yrs?|y\.?o\.?)\b",  # 18-99 years
]

CHILD_PATTERNS = [
    r"\bchild(ren)?\b",
    r"\binfant\b",
    r"\bneonat(e|al)\b",
    r"\bnewborn\b",
    r"\bbab(y|ies)\b",
    r"\btoddler\b",
    r"\bp[ae]ediatric\b",
    r"\b([0-9]|1[0-7])\s*[- ]?\s*(years?|yrs?|y\.?o\.?)\b",  # 0-17 years
    r"\b\d+\s*[- ]?\s*(months?|mo)\b",  # X months (old)
]

# Which "audience" tag (see build_index.py's SOURCE_AUDIENCE) should be
# deprioritized for a detected patient audience. Only adult->pediatric is
# mapped: "general" sources (AWaRe, UCG) are written to cover all ages, so a
# child query has no mismatched source to avoid.
AUDIENCE_MISMATCH = {
    "adult": "pediatric",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def detect_patient_audience(query: str) -> str | None:
    """Returns 'adult', 'child', or None (no clear/unambiguous cue)."""
    q = query.lower()
    is_adult = any(re.search(p, q) for p in ADULT_PATTERNS)
    is_child = any(re.search(p, q) for p in CHILD_PATTERNS)
    if is_adult and not is_child:
        return "adult"
    if is_child and not is_adult:
        return "child"
    return None


def _load_index():
    global _index_cache
    if _index_cache is None:
        with open(INDEX_PATH, "rb") as f:
            _index_cache = pickle.load(f)
    return _index_cache


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Returns the top-k chunks (dicts with source, page, text) for a query.

    Ranks by BM25 relevance, then -- if the query gives an unambiguous
    adult/child cue -- deprioritizes chunks whose source audience is known
    to mismatch (e.g. paediatric dosing guidance for an adult query) by
    excluding them from the top-k pool. If excluding them would leave the
    pool empty (the only relevant hits happen to be from a mismatched
    source), falls back to the unfiltered ranking rather than returning
    nothing.
    """
    index = _load_index()
    bm25 = index["bm25"]
    chunks = index["chunks"]

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    ranked = [i for i in ranked if scores[i] > 0]  # drop zero-relevance matches

    audience = detect_patient_audience(query)
    mismatch_tag = AUDIENCE_MISMATCH.get(audience)
    if mismatch_tag:
        filtered = [i for i in ranked if chunks[i].get("audience") != mismatch_tag]
        if filtered:
            pool = filtered
        else:
            # Safe fallback: still exclude the mismatched audience entirely --
            # a mismatched-audience chunk must never reach the result set,
            # even when filtering would otherwise empty the pool.
            pool = [i for i in ranked if chunks[i].get("audience") == "general"]
    else:
        pool = ranked

    return [chunks[i] for i in pool[:k]]


def format_context(chunks: list[dict]) -> str:
    """Formats retrieved chunks into a context block for the prompt."""
    if not chunks:
        return ""
    lines = []
    for c in chunks:
        lines.append(f"[{c['source']}, p.{c['page']}]: {c['text']}")
    return "\n\n".join(lines)
