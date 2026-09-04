"""
Extracts each corpus PDF to a page-marked text file under source_material/,
normalising the artefacts that PDF text layers introduce (broken ligatures,
line-break hyphenation, symbol-font glyphs left as private-use codepoints)
WITHOUT touching digits -- a corrupted dose number is worse than no data, so
nothing in here rewrites a numeric character except to restore one that font
subsetting had hidden (see GLYPH_FIXES). 

Output: source_material/<stem>.txt with a "[[page N]]" marker before each page.
"""
import hashlib, json, re, sys
from pathlib import Path
from pypdf import PdfReader

ROOT = Path.home() / "amr-stewardship-assistant"
CORPUS = ROOT / "corpus"
OUT = ROOT / "source_material"
OUT.mkdir(exist_ok=True)

DOCS = {
    "aware_book.pdf": "who_aware_book_2022",
    "ucg_2023_full.pdf": "uganda_clinical_guidelines_2023",
    "emhslu_2023.pdf": "emhslu_2023",
    "WHO-IMCI.pdf": "who_imci_chart_booklet",
    "WHO-pocketbook.pdf": "who_pocketbook_hospital_care_children",
}

# ---------------------------------------------------------------------------
# Symbol- and dingbat-font glyphs that the PDF text layer leaves behind as
# private-use (U+E000-F8FF) or C1 (U+0080-009F) codepoints.
#
# Keys are written as chr(0x....) rather than as literal glyphs so the table
# stays readable and diffable in plain ASCII -- these characters have no visible
# form in most editors.
#
# They are NOT interchangeable between documents: the same codepoint is a
# different glyph in a different font. U+F067 is a decorative separator in the
# AWaRe book running header, while U+F077 is a bullet in UCG Note/Caution
# blocks -- neither is the ASCII letter its low byte would suggest. So the
# table is keyed by document, and every entry was read off its own context in
# the extracted text.
#
# The IMCI booklet is the dangerous one. It uses a subsetted font that maps
# plain ASCII into U+F0xx: U+F032 U+F035 U+F030 followed by " mg" is literally
# "250 mg" -- the paediatric mebendazole dose -- and U+F042 + "reastfeed" is
# "Breastfeed". Stripping unrecognised private-use characters -- the obvious
# cleanup -- would have silently deleted the digits of a dose. So nothing here
# is stripped blind: an unmapped codepoint is a hard error, see
# assert_no_stray_glyphs.
GLYPH_FIXES = {
    "who_imci_chart_booklet": {
        # subsetted font: U+F0xx carries ASCII 0xxx
        chr(0xF030): "0", chr(0xF032): "2", chr(0xF035): "5",
        chr(0xF042): "B", chr(0xF02D): "-",
        # Symbol and Wingdings list markers
        chr(0xF0B7): "- ", chr(0xF0A8): "- ", chr(0xF0FC): "- ",
        chr(0xF0D8): "- ", chr(0xF0BA): "- ",
        # the arrow carrying the "then" of an IMCI decision rule:
        # "GENERAL DANGER SIGN -> Give one dose of pre-referral antibiotic"
        chr(0xF0AE): " -> ",
    },
    "who_aware_book_2022": {
        # running-header separator: "PRIMARY HEALTH CARE * 12. Pneumonia"
        chr(0xF067): " ",
    },
    "uganda_clinical_guidelines_2023": {
        chr(0x89): "- ",     # bullet on TREATMENT lines -- the drug and dose lines
        chr(0x81): "- ",     # bullet on Investigations lines
        chr(0x83): "- ",     # bullet on Notes lines
        chr(0x8D): "- ",     # bullet on Clinical features lines
        chr(0xF077): "- ",   # bullet on Note/Caution lines
    },
    "who_pocketbook_hospital_care_children": {
        chr(0x81): "- ",
    },
}

# Deliberately NOT mapped: U+F0E2, and the mis-decoded Latin-1 bytes beside it,
# in the UCG shock-severity tables ("Pulse pressure Normal a aa <U+F0E2>").
# Those are arrows whose direction the text layer no longer records, and
# guessing "decreased" against "increased" in a triage table is not a call this
# script gets to make. Recorded as unusable for grounding in SOURCES.md.
ALLOWED_STRAY = {chr(0xF0E2)}


def stray_glyphs(text):
    """Private-use and C1 codepoints still present, with counts."""
    out = {}
    for ch in text:
        o = ord(ch)
        if (0xE000 <= o <= 0xF8FF or 0x80 <= o <= 0x9F) and ch not in ALLOWED_STRAY:
            out[ch] = out.get(ch, 0) + 1
    return out


def assert_no_stray_glyphs(stem, text):
    stray = stray_glyphs(text)
    if not stray:
        return
    detail = ", ".join("U+%04X x%d" % (ord(c), n) for c, n in sorted(stray.items()))
    raise SystemExit(
        "%s: %d unmapped symbol-font codepoints (%s).%s"
        "Read each one in context before adding it to GLYPH_FIXES -- in this "
        "corpus some of them are digits inside doses."
        % (stem, sum(stray.values()), detail, chr(10)))

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
             "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}

def clean(text: str, stem: str) -> str:
    for glyph, repl in GLYPH_FIXES.get(stem, {}).items():
        text = text.replace(glyph, repl)
    for lig, repl in LIGATURES.items():
        # "classiﬁ cation" -> "classification": the ligature glyph is often
        # emitted with a spurious space after it mid-word.
        text = re.sub(r"(?<=[A-Za-z])" + lig + r"\s+(?=[a-z])", repl, text)
        text = text.replace(lig, repl)
    # De-hyphenate only across a real line break ("work-\ners" -> "workers").
    # UCG's typesetter emits the wrap hyphen with a leading space
    # ("health fa -\ncilities"), so handle that form too. Both rules require a
    # lowercase letter on each side, which keeps ranges ("20 mg -\n30 mg") and
    # bullet dashes intact; nothing here can touch a digit.
    text = re.sub(r"(?<=[a-z]) ?-\n(?=[a-z])", "", text)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def garble_report(name: str, text: str) -> dict:
    total = len(text)
    return {
        "doc": name,
        "chars": total,
        "replacement_chars": text.count("�"),
        "residual_ligatures": sum(text.count(l) for l in LIGATURES),
        "cid_artifacts": len(re.findall(r"\(cid:\d+\)", text)),
        "stray_symbol_glyphs": sum(stray_glyphs(text).values()),
        "allowed_stray_glyphs": sum(text.count(c) for c in ALLOWED_STRAY),
        "pct_non_ascii": round(100 * sum(1 for c in text if ord(c) > 127) / max(total, 1), 3),
        "dose_patterns_mg": len(re.findall(r"\b\d+(?:\.\d+)?\s?mg\b", text, re.I)),
        "dose_patterns_mg_per_kg": len(re.findall(r"\bmg/kg\b", text, re.I)),
        "spaced_digits": len(re.findall(r"\b\d\s\d\s\d\b", text)),  # "5 0 0" = broken numerals
    }

reports = []
for pdf_name, stem in DOCS.items():
    src = CORPUS / pdf_name
    if not src.exists():
        print(f"[skip] {pdf_name} missing", file=sys.stderr); continue
    reader = PdfReader(str(src))
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        t = clean(page.extract_text() or "", stem)
        if t:
            parts.append(f"[[page {i}]]\n{t}")
    text = "\n\n".join(parts)
    assert_no_stray_glyphs(stem, text)
    dest = OUT / f"{stem}.txt"
    dest.write_text(text, encoding="utf-8")
    rep = garble_report(stem, text)
    rep.update(pdf=pdf_name, pdf_pages=len(reader.pages), pages_with_text=len(parts),
               sha256=hashlib.sha256(src.read_bytes()).hexdigest())
    reports.append(rep)
    print(f"{stem}: {len(parts)}/{len(reader.pages)} pages, {len(text):,} chars -> {dest.name}")

(OUT / "extraction_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
print("\n=== GARBLE CHECK ===")
for r in reports:
    print(f"{r['doc']:<42} nonascii={r['pct_non_ascii']:>6}%  U+FFFD={r['replacement_chars']:<4} "
          f"lig={r['residual_ligatures']:<4} cid={r['cid_artifacts']:<4} "
          f"stray={r['stray_symbol_glyphs']:<4} kept-stray={r['allowed_stray_glyphs']:<3} "
          f"broken-numerals={r['spaced_digits']:<4} 'N mg'={r['dose_patterns_mg']:<6} 'mg/kg'={r['dose_patterns_mg_per_kg']}")
