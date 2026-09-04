"""
Builds a lightweight, offline BM25 retrieval index over the AMR stewardship
corpus: the WHO AWaRe antibiotic book, the WHO IMCI chart booklet, the WHO
Pocket Book of Hospital Care for Children, and the antimicrobial-relevant
chapters of the Uganda Clinical Guidelines 2023.

This runs entirely outside the llama.cpp inference process, so it does not
count against the ADTC 7GB inference memory cap -- it's a one-time offline
preprocessing step whose output (index.pkl) is loaded cheaply at runtime.

Each chunk is tagged with an "audience" ("general" or "pediatric", see
SOURCE_AUDIENCE below) so retrieve.py can deprioritize e.g. paediatric
dosing guidance for a query that's clearly about an adult patient.

Usage:
    python rag/build_index.py

Expects PDFs in corpus/:
    corpus/aware_book.pdf
    corpus/WHO-IMCI.pdf
    corpus/WHO-pocketbook.pdf
    corpus/ucg_2023.pdf        (only antimicrobial-relevant chapters needed --
                                 see NOTE below on trimming this file first)

Produces:
    rag/index.pkl
"""

import pickle
import re
from pathlib import Path

from pypdf import PdfReader
from rank_bm25 import BM25Okapi

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
INDEX_PATH = Path(__file__).parent / "index.pkl"

# Chunking parameters -- tuned for short, retrievable passages rather than
# whole pages, since prescribing/dosing facts are usually a paragraph long.
CHUNK_SIZE_CHARS = 900
CHUNK_OVERLAP_CHARS = 150

# Which patient population each source is written for. retrieve.py uses this
# to deprioritize e.g. paediatric dosing guidance when the query is clearly
# about an adult (and vice versa) -- without it, BM25's pure keyword overlap
# can rank a paediatric passage above a better-matching adult one just
# because it shares more words with the query.
SOURCE_AUDIENCE = {
    "AWaRe Book (WHO)": "general",
    "WHO IMCI Chart Booklet": "pediatric",
    "WHO Pocket Book of Hospital Care for Children": "pediatric",
    "Uganda Clinical Guidelines 2023": "general",
}

# Uganda Clinical Guidelines 2023 is tagged "general" at the source level,
# but it embeds a dedicated pediatric section (Chapter 17: Childhood
# Illness) inside an otherwise adult/general-audience document -- e.g. its
# child-specific zinc dosing was leaking into adult diarrhoea queries
# because the whole document's chunks were tagged "general" and so passed
# straight through retrieve.py's adult/pediatric audience filter.
# Chapter 17's running header repeats "CHAPTER 17: Childhood Illness" on
# every one of its pages, so detect it per-page and override the audience
# tag for just those chunks, rather than trusting the whole-document tag.
PEDIATRIC_CHAPTER_MARKERS = {
    "Uganda Clinical Guidelines 2023": "CHAPTER 17: Childhood Illness",
}


def extract_pdf_text(path: Path) -> list[tuple[int, str]]:
    """Returns a list of (page_number, page_text) tuples."""
    if not path.exists():
        print(f"  [skip] {path.name} not found in corpus/")
        return []
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i + 1, text))
    return pages


def chunk_text(source: str, page_num: int, text: str, audience: str) -> list[dict]:
    """Splits page text into overlapping chunks, tagged with source + page + audience."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE_CHARS
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({
                "source": source,
                "page": page_num,
                "text": chunk.strip(),
                "audience": audience,
            })
        start += CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def main():
    sources = {
        "AWaRe Book (WHO)": CORPUS_DIR / "aware_book.pdf",
        "WHO IMCI Chart Booklet": CORPUS_DIR / "WHO-IMCI.pdf",
        "WHO Pocket Book of Hospital Care for Children": CORPUS_DIR / "WHO-pocketbook.pdf",
        "Uganda Clinical Guidelines 2023": CORPUS_DIR / "ucg_2023.pdf",
    }

    all_chunks = []
    for source_name, path in sources.items():
        print(f"Extracting: {source_name}")
        audience = SOURCE_AUDIENCE[source_name]
        marker = PEDIATRIC_CHAPTER_MARKERS.get(source_name)
        pages = extract_pdf_text(path)
        override_count = 0
        for page_num, text in pages:
            page_audience = audience
            if marker and marker in text:
                page_audience = "pediatric"
                override_count += 1
            all_chunks.extend(chunk_text(source_name, page_num, text, page_audience))
        print(f"  -> {len(pages)} pages read (audience: {audience})"
              + (f", {override_count} pages overridden to 'pediatric' ({marker!r})"
                 if marker else ""))

    if not all_chunks:
        print("No chunks produced -- check that corpus/*.pdf files exist.")
        return

    print(f"Total chunks: {len(all_chunks)}")
    tokenized_corpus = [tokenize(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)

    print(f"Index written to {INDEX_PATH}")


if __name__ == "__main__":
    main()
