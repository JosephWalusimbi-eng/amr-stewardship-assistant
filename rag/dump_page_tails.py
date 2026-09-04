"""
Dumps the last ~60 characters of extracted text from a range of physical
PDF pages, so we can see the REAL page-number/footer pattern by eye instead
of guessing its exact format (which failed twice already -- the actual
extracted text layout doesn't match assumptions about spacing/wording).

Usage:
    python rag/dump_page_tails.py corpus/ucg_2023_full.pdf 0 130
    (dumps physical pages 0 through 130, adjust the range as needed)
"""

import sys
from pypdf import PdfReader


def main():
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <input_pdf> <start_page> <end_page>")
        sys.exit(1)

    path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    reader = PdfReader(path)
    total = len(reader.pages)
    end = min(end, total - 1)

    for i in range(start, end + 1):
        text = (reader.pages[i].extract_text() or "").strip()
        tail = text[-70:].replace("\n", " | ")
        head = text[:40].replace("\n", " | ")
        print(f"[{i:4d}] HEAD: {head!r}")
        print(f"       TAIL: {tail!r}")


if __name__ == "__main__":
    main()
