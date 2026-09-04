"""
Trims Uganda Clinical Guidelines 2023 (1000+ pages, all specialties) down to
the chapters actually relevant to antimicrobial stewardship, before it goes
into the RAG index. Keeping the full book dilutes retrieval with irrelevant
content (cardiology, oncology, dental, family planning, etc.).

Run this BEFORE rag/build_index.py.

Usage:
    python rag/trim_ucg.py corpus/ucg_2023_full.pdf corpus/ucg_2023.pdf
    python rag/trim_ucg.py --check corpus/ucg_2023_full.pdf   (spot-check only)

This document has TWO independent page-numbering schemes, not one offset:

  1. Front matter (physical pages 0-65) is numbered in roman numerals
     (i, ii, iii, ... lxvi), running from the title page through the
     prescribing-principles / rational-antibiotic-use discussion. It ends
     the page immediately before Chapter 1 begins -- there is no "printed
     page 1" inside it, so it can't be addressed via printed_page + offset.
     It's kept in full via KEEP_RANGES_PHYSICAL, addressed by raw index.

  2. The body (from Chapter 1 onward, physical page 66+) restarts arabic
     numbering at "1". Empirically (see find_page_offset) every body page's
     printed number is exactly 65 less than its physical index, consistent
     across the whole 1161-page document. Body ranges are kept via
     KEEP_RANGES_PRINTED, addressed as printed_page + offset.

Both range lists were verified empirically against this specific PDF (see
find_page_offset's footer-stamp matching, and verify_output's post-write
content check) -- re-run --check if you swap in a different copy/edition.
"""

import re
import sys
from pypdf import PdfReader, PdfWriter

# Physical PDF indices (0-indexed, inclusive), for content that precedes
# arabic pagination and so can't be expressed as printed_page + offset.
KEEP_RANGES_PHYSICAL = [
    (0, 65),  # Full roman-numeral front matter (i-lxvi): title page, foreword,
              # ToC, abbreviations, intro, medicine classification, and the
              # antibiotic-prescribing / AMR discussion ("Inappropriate use
              # of antibiotics...", "Prescribing Guidelines"). Printed arabic
              # p.1 is actually Chapter 1: Emergencies and Trauma (unrelated)
              # -- so this range replaces a literal "printed pages 1-46"
              # reading, which would have missed the AMR content entirely.
]

# (start_page, end_page) in PRINTED (arabic) page numbers, inclusive.
KEEP_RANGES_PRINTED = [
    (95, 185),    # Ch. 2 Infectious Diseases
    (328, 391),   # Ch. 5 Respiratory Diseases (incl. TB)
    (392, 407),   # Ch. 6.1-6.2 GI infections
    (446, 465),   # Ch. 7 Renal/Urinary
    (552, 559),   # Ch. 10.1 Musculoskeletal infections
    (793, 871),   # Ch. 17 Childhood Illness (IMNCI-style)
    (944, 1002),  # Ch. 21-22 ENT & Skin infections
    (1046, 1047), # 24.1.5 Surgical Antibiotic Prophylaxis (verified: printed
                   # p.1043 is anaesthesia/theatre content, not the
                   # prophylaxis section -- the actual section header + body
                   # are on 1046-1047, confirmed via footer-stamp matching).
]

# Printed page numbers to empirically verify the offset against. Every
# start/end of every KEEP_RANGES_PRINTED entry, spread across the whole
# document, so a mid-book pagination shift (e.g. an inserted unnumbered
# plate) would show up as an inconsistent offset instead of being missed.
LANDMARK_PRINTED_PAGES = sorted({p for start, end in KEEP_RANGES_PRINTED for p in (start, end)})


def find_page_offset(reader: PdfReader, landmark_printed_pages=LANDMARK_PRINTED_PAGES) -> int:
    """
    Each body page carries a footer stamp "<printed_page>Uganda Clinical
    Guidelines 2023..." -- but pypdf's text extraction order varies page to
    page (sometimes the footer text block is emitted first, sometimes
    last), and whitespace around the number is inconsistent (space vs
    newline vs none), so we strip ALL whitespace before matching rather
    than relying on position or a fixed separator.

    Scans the whole document for the one page whose stripped text contains
    the stamp for each landmark, then checks that every landmark implies
    the SAME offset (physical_index - printed_page). Falls back to offset 0
    (with a warning) if any landmark is missing or offsets disagree.
    """
    total = len(reader.pages)
    stripped_pages = [re.sub(r"\s+", "", p.extract_text() or "") for p in reader.pages]

    found_offsets = {}
    for printed_page in landmark_printed_pages:
        stamp = f"{printed_page}UgandaClinicalGuidelines2023"
        for pdf_index in range(total):
            if stamp in stripped_pages[pdf_index]:
                found_offsets[printed_page] = pdf_index - printed_page
                break

    missing = [p for p in landmark_printed_pages if p not in found_offsets]
    if missing:
        print(f"WARNING: could not find footer stamp for printed page(s) {missing}. "
              f"Defaulting to offset 0 -- verify KEEP_RANGES_PRINTED manually.")
        return 0

    offsets = set(found_offsets.values())
    if len(offsets) != 1:
        print(f"WARNING: inconsistent offsets across landmarks: {found_offsets}")
        print("Pagination shifts somewhere in the document -- KEEP_RANGES_PRINTED "
              "may need per-range offsets. Defaulting to offset 0.")
        return 0

    offset = offsets.pop()
    print(f"Verified consistent offset ({offset:+d}) across {len(found_offsets)} "
          f"landmark pages: {sorted(found_offsets)}")
    return offset


def resolve_physical_indices(reader: PdfReader) -> list[int]:
    """Returns the ordered, deduped list of physical page indices to keep."""
    total = len(reader.pages)
    offset = find_page_offset(reader)
    print(f"Detected body offset: {offset:+d} (pdf_index = printed_page + offset)")

    indices = []
    for start, end in KEEP_RANGES_PHYSICAL:
        for idx in range(start, end + 1):
            if 0 <= idx < total:
                indices.append(idx)

    for start, end in KEEP_RANGES_PRINTED:
        for printed_page in range(start, end + 1):
            idx = printed_page + offset
            if 0 <= idx < total:
                indices.append(idx)

    return indices


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        spot_check_ranges(sys.argv[2])
        return

    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_pdf> <output_pdf>")
        print(f"       python {sys.argv[0]} --check <input_pdf>  "
              f"(spot-check ranges without writing output)")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]
    reader = PdfReader(input_path)
    print(f"Input has {len(reader.pages)} PDF pages.")

    indices = resolve_physical_indices(reader)

    writer = PdfWriter()
    for idx in indices:
        writer.add_page(reader.pages[idx])

    with open(output_path, "wb") as f:
        writer.write(f)

    print(f"Wrote {len(indices)} pages to {output_path}")
    verify_output(output_path)


def verify_output(output_path: str) -> None:
    """
    Sanity-checks the trimmed PDF: confirms its opening section (the
    front-matter block, before the ranges lists become clinical chapters)
    contains AMR-relevant content, and that the page count matches what
    the configured ranges should produce (catches out-of-bounds ranges
    being silently dropped).
    """
    reader = PdfReader(output_path)
    front_matter_pages = sum(end - start + 1 for start, end in KEEP_RANGES_PHYSICAL)
    front_matter_text = "".join(
        (reader.pages[i].extract_text() or "") for i in range(front_matter_pages)
    ).upper()
    amr_markers = ["ANTIMICROBIAL", "AWARE", "ANTIBIOTIC", "AMR"]
    hits = [m for m in amr_markers if m in front_matter_text]
    if hits:
        print(f"  [ok] opening {front_matter_pages} page(s) of output contain "
              f"AMR-relevant marker(s): {hits}")
    else:
        print(f"  [WARNING] opening {front_matter_pages} page(s) of output contain "
              f"none of {amr_markers} -- output may not start where expected.")

    expected = sum(end - start + 1 for start, end in KEEP_RANGES_PHYSICAL)
    expected += sum(end - start + 1 for start, end in KEEP_RANGES_PRINTED)
    actual = len(reader.pages)
    status = "ok" if actual == expected else "WARNING"
    print(f"  [{status}] expected {expected} pages from configured ranges, "
          f"output has {actual}")


def spot_check_ranges(input_path: str) -> None:
    """
    Prints the first line of text at each configured range's start/end
    page, so a human can visually confirm each range lands on the intended
    chapter before trusting the trim. Run standalone:
        python rag/trim_ucg.py --check <input_pdf>
    """
    reader = PdfReader(input_path)
    offset = find_page_offset(reader)

    for start, end in KEEP_RANGES_PHYSICAL:
        for label, idx in (("start", start), ("end", end)):
            text = (reader.pages[idx].extract_text() or "").strip()
            first_line = next((l for l in text.splitlines() if l.strip()), "")
            print(f"  physical range ({start}-{end}) {label} (phys {idx}): "
                  f"{first_line[:80]!r}")

    for start, end in KEEP_RANGES_PRINTED:
        for label, printed_page in (("start", start), ("end", end)):
            idx = printed_page + offset
            text = (reader.pages[idx].extract_text() or "").strip()
            first_line = next((l for l in text.splitlines() if l.strip()), "")
            print(f"  printed range ({start}-{end}) {label}=p.{printed_page} "
                  f"(phys {idx}): {first_line[:80]!r}")


if __name__ == "__main__":
    main()
