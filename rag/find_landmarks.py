"""
Improved diagnostic: finds the PHYSICAL PDF page index for a set of known
PRINTED page numbers, by searching for each page's own footer stamp
("<printed_number>Uganda Clinical Guidelines 2023" -- squished together by
text extraction, but a reliable per-page fingerprint) instead of chapter
titles, which also appear in the Table of Contents and produce false matches.

Usage:
    python rag/find_landmarks.py corpus/ucg_2023_full.pdf
"""

import sys
from pypdf import PdfReader

# Printed page numbers we actually need offsets for for trim_ucg.py's
# KEEP_RANGES_PRINTED -- the start/end of each chapter range we're keeping.
TARGET_PRINTED_PAGES = [1, 95, 185, 328, 391, 392, 407, 446, 465,
                         552, 559, 793, 871, 944, 1002, 1043]


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input_pdf>")
        sys.exit(1)

    reader = PdfReader(sys.argv[1])
    total = len(reader.pages)
    print(f"Scanning {total} pages...\n")

    found = {n: None for n in TARGET_PRINTED_PAGES}

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").replace(" ", "")
        for n in TARGET_PRINTED_PAGES:
            if found[n] is not None:
                continue
            stamp = f"{n}UgandaClinicalGuidelines2023"
            if stamp in text:
                found[n] = i
                offset = i - n
                print(f"  printed p.{n:4d}  ->  physical page {i:4d} (0-indexed)  "
                      f"(offset = {offset:+d})")

    print("\nPrinted page numbers not found (footer stamp may differ, or "
          "this page number doesn't exist / is duplicated in the source):")
    for n, idx in found.items():
        if idx is None:
            print(f"  - printed p.{n}")

    resolved = [(n, idx) for n, idx in found.items() if idx is not None]
    if len(resolved) >= 2:
        offsets = [idx - n for n, idx in resolved]
        if len(set(offsets)) == 1:
            print(f"\nConsistent offset across all matched landmarks: {offsets[0]:+d}")
            print("Use this in trim_ucg.py's offset logic (or just hardcode it "
                  "and skip auto-detection).")
        else:
            print(f"\nWARNING: offsets are NOT consistent across landmarks: {offsets}")
            print("This means pagination resets or shifts somewhere in the "
                  "document (common in books with roman-numeral front matter "
                  "vs. numbered body, or inserted unnumbered pages). You may "
                  "need a different offset for different chapter ranges -- "
                  "share this output and we'll work out per-range offsets.")


if __name__ == "__main__":
    main()
