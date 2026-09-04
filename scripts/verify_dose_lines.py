"""
STEP 3b -- pick a sample of extracted dose lines for cross-checking against the
PDF pages they claim to come from.

This exists because "the extractor ran without erroring" is not evidence. The
same corpus encoded a 250 mg paediatric dose as three private-use codepoints
that looked like nothing at all, and the first text-quality audit passed it.
A dose line is not trusted until a human has read the page image and confirmed
the number, the unit, the frequency and the column it belongs to.

Usage:
    python3 scripts/verify_dose_lines.py            # print a sample to check
    python3 scripts/verify_dose_lines.py --seed 5   # a different sample

Prints, for each sampled line, the page of the source PDF to open. Record the
outcome in source_material/SOURCES.md and set "verified": true in
dose_lines.json only once that is done.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SM = ROOT / "source_material"

PDF_FOR_DOC = {
    "uganda_clinical_guidelines_2023": "corpus/ucg_2023_full.pdf",
    "who_aware_book_2022": "corpus/aware_book.pdf",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min-doses", type=int, default=1)
    args = ap.parse_args()

    data = json.loads((SM / "dose_lines.json").read_text(encoding="utf-8"))
    drugs = data["drugs"]

    # Sample the lines that matter most: unflagged, carrying a real regimen, and
    # spread across both documents. Weight-band and mg/kg lines are drawn first
    # because those are the ones a wrong column silently corrupts.
    pool = []
    for key, recs in drugs.items():
        for r in recs:
            if r["flags"]:
                continue
            if len(r["doses"]) < args.min_doses:
                continue
            score = (r["has_weight_band"] * 2 + any("/kg" in d for d in r["doses"]) * 2
                     + r["has_frequency"] + r["has_duration"])
            pool.append((score, key, r))

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    pool.sort(key=lambda t: -t[0])

    chosen, by_doc = [], {}
    for score, key, r in pool:
        if len(chosen) >= args.n:
            break
        if by_doc.get(r["doc"], 0) >= args.n // 2 + 1:
            continue
        chosen.append((key, r))
        by_doc[r["doc"]] = by_doc.get(r["doc"], 0) + 1

    print("CROSS-CHECK SAMPLE -- open each PDF page and confirm the extracted")
    print("numbers, units and frequency match, and that no value has been pulled")
    print("across a column boundary.")
    print()
    for i, (key, r) in enumerate(chosen, 1):
        print("[%d] drug: %s" % (i, key))
        print("    doc      : %s" % r["doc_label"])
        print("    open     : %s  page %d" % (PDF_FOR_DOC[r["doc"]], r["page"]))
        print("    doses    : %s" % ", ".join(r["doses"]))
        print("    freq=%s duration=%s route=%s weight_band=%s"
              % (r["has_frequency"], r["has_duration"], r["has_route"], r["has_weight_band"]))
        print("    extracted: %s" % r["text"][:300])
        print()
    print("Pages to open, by PDF:")
    for doc in sorted(by_doc):
        pages = sorted({r["page"] for k, r in chosen if r["doc"] == doc})
        print("   %-44s %s" % (PDF_FOR_DOC[doc], ", ".join(str(p) for p in pages)))


if __name__ == "__main__":
    sys.exit(main())
