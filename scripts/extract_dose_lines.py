"""
STEP 3a -- per-drug dose lines, the grounding unit for Category 4.

Category 4 is the category where a wrong number is the worst thing this project
can produce, so it does not get a whole chunk attached and a hope that the model
finds the right row. Each generation call is handed the specific extracted lines
for one drug, and the generated dose is checked back against those exact lines
before the pair is accepted.

Two sources, both prose-style dosing:
  * Uganda Clinical Guidelines 2023 -- national first-line practice
  * WHO AWaRe antibiotic book 2022 -- weight-band paediatric dosing

Everything here is CANDIDATE material. Extraction parsing without an error
proves nothing: the same corpus quietly encoded "250 mg" as three private-use
codepoints until it was audited. So every candidate carries the page it came
from, and `flags` records why a line might be untrustworthy. Nothing is trusted
as a generation source until it has been cross-checked against the PDF page --
see scripts/verify_dose_lines.py.

Output: source_material/dose_lines.json
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SM = ROOT / "source_material"
sys.path.insert(0, str(ROOT))

from prepare_dataset import Validator, DRUG_ALIASES, drug_pattern, norm_drug  # noqa: E402

DOCS = {
    "uganda_clinical_guidelines_2023": "Uganda Clinical Guidelines 2023 (4th ed.)",
    "who_aware_book_2022": "WHO AWaRe Antibiotic Book (2022)",
}

# A dose: a number with a unit that can carry a dose. Deliberately narrow --
# "for 7 days" is a duration, not a dose, and is captured separately.
DOSE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:-\s*\d+(?:[.,]\d+)?\s*)?"
                  r"(?:mg/kg(?:/day|/dose)?|mg|g|mcg|micrograms?|IU|MU|ml)\b", re.I)
FREQ = re.compile(r"\b(?:once|twice|three times|four times|every\s+\d+\s*hours?|"
                  r"[qb]\.?i\.?d\.?|t\.?i\.?d\.?|o\.?d\.?|daily|hourly|"
                  r"\d+\s*(?:times|hourly))\b", re.I)
DURATION = re.compile(r"\b(?:for\s+)?\d+(?:\s*-\s*\d+)?\s*(?:days?|weeks?|months?)\b", re.I)
ROUTE = re.compile(r"\b(?:oral(?:ly)?|IV|intravenous(?:ly)?|IM|intramuscular(?:ly)?|"
                   r"by mouth|PO|rectal(?:ly)?|topical(?:ly)?)\b", re.I)
WEIGHT_BAND = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:-|–|to)\s*\d+(?:[.,]\d+)?\s*kg\b", re.I)

# Table-row shapes. UCG weight/age dosing tables flatten to rows whose headers no
# longer bind to their values ("10-20 2-7 years 1 1 1"), which SOURCES.md records
# as a known limitation. Anything matching these is kept but flagged, never
# silently used.
BARE_NUMBER_RUN = re.compile(r"(?:(?<=\s)|^)\d+(?:\.\d+)?(?:\s+\d+(?:\.\d+)?){2,}(?:\s|$)")
AGE_BAND_ROW = re.compile(r"\d+\s*(?:-|–)\s*\d+\s*(?:years?|months?|kg)\b.*\b\d+\s+\d+", re.I)
PAGE_MARK = re.compile(r"^\[\[page (\d+)\]\]$")


def page_map(text):
    """Line index -> source PDF page number."""
    pages, current = [], 0
    for line in text.split("\n"):
        m = PAGE_MARK.match(line.strip())
        if m:
            current = int(m.group(1))
        pages.append(current)
    return pages


RULE_RUN = re.compile(r"[-_.]{6,}")
SPACED_DIGITS = re.compile(r"\b\d(?:\s\d){2,}\b")


def table_flags(window, rec_doses):
    flags = []
    if BARE_NUMBER_RUN.search(window):
        flags.append("bare_number_run")
    if AGE_BAND_ROW.search(window):
        flags.append("age_band_row")
    if window.count("|") >= 2:
        flags.append("pipe_delimited")
    if RULE_RUN.search(window):
        # dotted leaders and form rules: a table or a fill-in line, not prose
        flags.append("rule_or_leader_run")
    if SPACED_DIGITS.search(window):
        flags.append("spaced_digits")
    if len(rec_doses) > 3:
        # several doses crammed into one window is a table row, not a sentence
        flags.append("many_doses_one_line")
    return flags


def attributed_doses(window, key, patterns):
    """Doses that belong to `key`, not to a neighbouring drug in the same window.

    Found by cross-checking a sample against the PDF pages. UCG p.183 yields
    "Or cloxacillin 50 mg/kg IV every 4-6 hours - Or benzylpenicillin 50,000
    IU/kg IV", and the naive reading files 50 mg/kg under benzylpenicillin when
    it is cloxacillin's dose. Attributing a neighbour's number to a drug is the
    exact failure this category exists to avoid.

    So a dose counts for `key` only if it falls between that drug's mention and
    the next drug mention. A dose before any drug name is orphaned -- AWaRe
    p.256 has "(oral): 500 mg every 6 hours" with the drug named on the previous
    line -- and belongs to nobody.
    """
    spans = []
    for k, pat in patterns:
        for m in pat.finditer(window):
            spans.append((m.start(), m.end(), k))
    spans.sort()
    mine, start, end = [], None, len(window)
    for i, (s, e, k) in enumerate(spans):
        if k == key:
            start = e
            end = spans[i + 1][0] if i + 1 < len(spans) else len(window)
            segment = window[start:end]
            mine.extend(m.group(0).strip() for m in DOSE.finditer(segment))
            break
    return sorted(set(mine)), (start is not None)


def main():
    v = Validator()
    # Longest first so "amoxicillin/clavulanic acid" wins over "amoxicillin".
    names = sorted(v.known_drugs, key=len, reverse=True)
    patterns = [(norm_drug(re.sub(r"_(IV|oral)$", "", n)), re.compile(drug_pattern(n), re.I))
                for n in names]

    out = defaultdict(list)
    stats = defaultdict(int)

    for stem, label in DOCS.items():
        path = SM / ("%s.txt" % stem)
        if not path.exists():
            print("[skip] %s missing" % path, file=sys.stderr)
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        pages = page_map(text)

        for i, line in enumerate(lines):
            if PAGE_MARK.match(line.strip()) or not line.strip():
                continue
            # Dose statements wrap; look at the line plus its two successors.
            window = " ".join(l.strip() for l in lines[i:i + 3]
                              if not PAGE_MARK.match(l.strip()))
            if not DOSE.search(window):
                continue
            hit_keys = set()
            masked = window
            for key, pat in patterns:
                m = pat.search(masked)
                if m:
                    hit_keys.add(DRUG_ALIASES.get(key, key))
                    masked = masked[:m.start()] + "#" * (m.end() - m.start()) + masked[m.end():]
            if not hit_keys:
                continue

            doses_found = sorted(set(m.group(0).strip() for m in DOSE.finditer(window)))
            flags = table_flags(window, doses_found)
            # A dose with no frequency, duration or route is a formulation
            # mention ("Amoxicillin 250 mg dispersible tablets"), not a regimen.
            # Category 4 answers must state route and total duration, so a line
            # that cannot support that is not a usable grounding line.
            if not (FREQ.search(window) or DURATION.search(window) or ROUTE.search(window)):
                flags.append("no_regimen_context")
            rec = {
                "doc": stem,
                "doc_label": label,
                "page": pages[i],
                "line_no": i + 1,
                "text": re.sub(r"\s+", " ", window).strip()[:400],
                "doses": doses_found,
                "has_frequency": bool(FREQ.search(window)),
                "has_duration": bool(DURATION.search(window)),
                "has_route": bool(ROUTE.search(window)),
                "has_weight_band": bool(WEIGHT_BAND.search(window)),
                "flags": flags,
            }
            for key in hit_keys:
                mine, named = attributed_doses(window, key, patterns)
                r = dict(rec)
                r["attributed_doses"] = mine
                r["flags"] = list(rec["flags"])
                if not mine:
                    # every dose in this window belongs to another drug, or to
                    # no named drug at all
                    r["flags"].append("no_dose_attributable_to_this_drug")
                elif len(doses_found) > len(mine):
                    r["flags"].append("other_drugs_doses_in_window")
                out[key].append(r)
                stats["candidates"] += 1
                if r["flags"]:
                    stats["flagged"] += 1
                if "no_dose_attributable_to_this_drug" in r["flags"]:
                    stats["unattributable"] += 1
                if "other_drugs_doses_in_window" in r["flags"]:
                    stats["shared_window"] += 1

    # Deduplicate: the 3-line window means consecutive lines repeat content.
    cleaned = {}
    for key, recs in out.items():
        seen, keep = set(), []
        for r in sorted(recs, key=lambda r: (r["doc"], r["line_no"])):
            sig = (r["doc"], r["text"][:120])
            if sig in seen:
                continue
            seen.add(sig)
            keep.append(r)
        cleaned[key] = keep

    usable = {k: [r for r in rs if not r["flags"]] for k, rs in cleaned.items()}
    usable = {k: rs for k, rs in usable.items() if rs}

    payload = {
        "note": "CANDIDATE dose lines. Not verified against the PDF pages -- run "
                "scripts/verify_dose_lines.py and record the outcome in SOURCES.md "
                "before using these to generate anything.",
        "verified": False,
        "drugs": cleaned,
    }
    (SM / "dose_lines.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print("drugs with any candidate line : %d" % len(cleaned))
    print("drugs with an unflagged line  : %d" % len(usable))
    print("candidate lines               : %d" % stats["candidates"])
    print("  unattributable to their drug: %d" % stats["unattributable"])
    print("  sharing a window with another drug: %d" % stats["shared_window"])
    print("  of which table-shaped       : %d (kept, flagged, excluded from use)"
          % stats["flagged"])
    print()
    print("top drugs by unflagged line count:")
    for k, rs in sorted(usable.items(), key=lambda kv: -len(kv[1]))[:12]:
        rich = sum(1 for r in rs if r["has_frequency"] and r["has_duration"])
        print("   %-32s %3d lines  (%d with frequency+duration)" % (k, len(rs), rich))


if __name__ == "__main__":
    main()
