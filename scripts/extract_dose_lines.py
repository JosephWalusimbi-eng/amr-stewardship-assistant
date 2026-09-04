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
from collections import Counter, defaultdict
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


# --- printed page numbers -------------------------------------------------
# The [[page N]] markers are pypdf indices, not the numbers printed in the book:
# UCG PDF page 183 prints as 117. A citation a reviewer cannot open to is worse
# than no citation, and this project's whole claim is hand-checkability. The
# printed folio survives extraction as a bare number line near each page break,
# so it can be recovered and cross-validated against a modal offset.
FOLIO = re.compile(r"^\d{1,4}$")


def folio_map(text):
    """pypdf page index -> page number printed in the book."""
    lines = text.split("\n")
    seen = defaultdict(list)
    current = None
    for line in lines:
        m = PAGE_MARK.match(line.strip())
        if m:
            current = int(m.group(1))
            continue
        s = line.strip()
        if current and FOLIO.match(s):
            n = int(s)
            if 0 < current - n < 400:
                seen[current].append(n)
    offsets = [p - n for p, ns in seen.items() for n in ns]
    if not offsets:
        return {}, None
    modal = Counter(offsets).most_common(1)[0][0]
    out = {}
    for page, cands in seen.items():
        exact = [n for n in cands if page - n == modal]
        out[page] = exact[0] if exact else page - modal
    return out, modal


# --- indications ----------------------------------------------------------
# A dose is only safe with the clinical question it answers. Benzylpenicillin is
# 3-4 MU for pneumococcal meningitis and 5-6 MU for meningococcal, on facing
# pages, and only 9 of its 55 extracted lines carry an indication word at all --
# the indication sits in a heading above the line.
SECTION_HEAD = re.compile(r"^\d+(?:\.\d+)+\.?\s+\S")          # UCG "2.1.5 Meningitis"
CHAPTER_HEAD = re.compile(r"^\d+\.\s+[A-Z]")                  # AWaRe "12. Pneumonia"
COURSE_HINT = re.compile(r"\((?:[^)]*\b(?:course|days?|weeks?)\b[^)]*)\)", re.I)
BINOMIAL = re.compile(r"^[A-Z][a-z]{3,}\s+[a-z]{3,}")          # "Streptococcus pneumoniae"
SKIP_LINE = re.compile(
    r"^(?:TREATMENT|LOC|TREATMENT\s+LOC|Adult|Child|Adults|Children|Management|"
    r"Prevention|Investigations|Notes?|Cautions?|Clinical features)\b|"
    r"^Uganda Clinical Guidelines|CHAPTER \d|^The WHO AWaRe|^\d{1,4}$|^[-•\s]*$", re.I)
CONTINUES = re.compile(r"^[a-z(]")

# Things that sit where a heading sits but say nothing clinical. Spot-checking
# benzylpenicillin found three of its lines -- including the 5-6 MU
# meningococcal dose, the exact case this work exists for -- picking up "H",
# which is the facility-level (LOC) column, and one picking up the table
# sub-header "Causative organisms identified". A junk indication is worse than
# none: it would pass a gate that only checks a field is populated.
NON_INDICATION = re.compile(
    r"^(?:H|HC\d|RR|NR|V|E|N|C|DS|STR|L)$|"                   # LOC / VEN column codes
    r"^(?:Causative organisms?(?: identified)?|Organisms?|Treatment|Diagnosis|"
    r"Differential diagnosis|Complications?|Referral|Follow[- ]up|Dosage|Dose|"
    r"Regimen|First line|Second line|Alternative|If |Once |Continue|Change to|"
    r"Supportive care|General measures|Non-pharmacological)\b", re.I)


# Structural furniture that sits in heading position. Sampling the 674 lines
# that resolved to a "nearby heading" showed most were not indications at all:
# "First Choice" (26), "Second Choice" (36), table headers like "DRUG DOSE
# INDICATION LOC" (12), and -- worst -- bare drug names like
# "Amoxicillin+clavulanic acid" (29). Naming a drug as its own indication would
# let a gate that merely checks the field is populated pass anything.
STRUCTURAL_LABEL = re.compile(
    r"^(?:First|Second|Third)\s+(?:Choice|line)\b|"
    r"^(?:Alternatives?|Medicines?|Administration|Adults?|Notes?|Table\s|"
    r"CONSIDER|DRUG\b|Antibiotic\s*\(|Duration|Comments?|Remarks?|Use\s|Give\s)|"
    r"\b(?:Dose|DOSE)\b.*\b(?:Indication|INDICATION|LOC)\b|"
    r"^[A-Z][a-z]+\s*(?:\(oral\)|\(IV\)|\(IM\))\s*:?$", re.I)


# A nearby heading only counts as an indication if it actually reads like a
# clinical condition. Without this test the tier was 34% structural furniture:
# "First Choice", "Administration", bare drug names. Anything that fails falls
# through to the numbered section heading, which is structural and always a real
# condition -- coarser than the organism, never invented.
CLINICAL_HINT = re.compile(
    r"\b\w+(?:itis|osis|aemia|emia|pathy|algia|coccal|cocci)\b|"
    r"\b(?:infection|infections|disease|syndrome|fever|abscess|sepsis|"
    r"septicaemia|septicemia|pneumonia|meningitis|malaria|tuberculosis|"
    r"diarrhoea|diarrhea|dysentery|cholera|typhoid|ulcer|wound|burns?|bite|"
    r"cellulitis|impetigo|trachoma|conjunctivitis|otitis|sinusitis|"
    r"pharyngitis|tonsillitis|bronchitis|cystitis|pyelonephritis|"
    r"urethritis|vaginosis|gonorrhoea|gonorrhea|syphilis|chlamydia|"
    r"chancroid|plague|anthrax|tetanus|measles|pertussis|diphtheria|"
    r"osteomyelitis|arthritis|endocarditis|peritonitis|appendicitis|"
    r"cholecystitis|prophylaxis|neutropenia|empiric(?:al)?)\b", re.I)


# A wrapped heading can pick up the regimen label that follows it, giving
# "Meningitis First choice". The condition is the part worth keeping.
TRAILING_LABEL = re.compile(
    r"\s+(?:First|Second|Third)\s+(?:choice|line)\b.*$|"
    r"\s+(?:Adults?|Children?|Dose|Dosage|Regimen|Alternatives?)\s*:?\s*$", re.I)
PROSE_START = re.compile(r"^(?:Although|However|For use in|Refer to|See |Note that|These )", re.I)


def looks_clinical(s):
    return bool(CLINICAL_HINT.search(s or ""))


def tidy_indication(s):
    """Trim a regimen label the line wrap glued onto a condition."""
    return TRAILING_LABEL.sub("", (s or "").strip()).strip(" .,;:-")


def usable_indication(s, drug_patterns=()):
    """True if a candidate heading actually names a clinical condition."""
    if not s:
        return False
    s = s.strip()
    if NON_INDICATION.match(s) or STRUCTURAL_LABEL.match(s):
        return False
    if len(s) <= 3:
        return False
    # A line that is essentially just a drug name is a regimen label, not an
    # indication. Falling through to the section heading gives a real one.
    stripped = re.sub(r"[^A-Za-z]+", "", s)
    for _key, pat in drug_patterns:
        m = pat.search(s)
        if m and len(re.sub(r"[^A-Za-z]+", "", m.group(0))) >= 0.6 * max(len(stripped), 1):
            return False
    return True


def indication_for(lines, i, max_up=14, drug_patterns=()):
    """Nearest indication heading above a dose line, and the section it sits in.

    Returns (indication, kind, section). `kind` records how it was found so the
    gate can treat a weak recovery differently from a strong one.
    """
    section = None
    for j in range(i - 1, max(-1, i - 400), -1):
        s = lines[j].strip()
        if SECTION_HEAD.match(s) or CHAPTER_HEAD.match(s):
            section = s
            break

    for j in range(i - 1, max(-1, i - max_up), -1):
        s = lines[j].strip()
        if not s or PAGE_MARK.match(s) or SKIP_LINE.match(s):
            continue
        if DOSE.search(s) or s.startswith("-") or s.startswith("Or ") or s.startswith("Plus"):
            continue
        if SECTION_HEAD.match(s) or CHAPTER_HEAD.match(s):
            return s, "section_heading", section
        # a heading may wrap: "Streptococcus pneumoniae (10-14 day course; up to"
        # / "21 days in severe case)"
        merged = s
        if j + 1 < len(lines) and CONTINUES.match(lines[j + 1].strip()) and j + 1 != i:
            nxt = lines[j + 1].strip()
            if not DOSE.search(nxt) and not nxt.startswith("-"):
                merged = (s + " " + nxt).strip()
        if not usable_indication(merged, drug_patterns):
            continue
        if PROSE_START.match(merged):
            continue
        merged = tidy_indication(merged)
        if not merged or not usable_indication(merged, drug_patterns):
            continue
        if BINOMIAL.match(merged) or COURSE_HINT.search(merged):
            return merged[:200], "organism_heading", section
        if len(merged) < 120 and merged[0].isupper() and looks_clinical(merged):
            return merged[:200], "nearby_heading", section
        # not clinical -- keep walking, then fall back to the section heading
        continue
    # The numbered section heading is the reliable fallback: "2.1.5 Meningitis",
    # "2.1.5.1 Neonatal Meningitis", "1.2.3 Burns". Disease-level rather than
    # organism-level, but it is a real clinical indication and it is structural
    # rather than guessed.
    if section:
        return section, "section_only", section
    return None, "none", None


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
        folios, modal_offset = folio_map(text)
        print("%s: printed-page offset %s (recovered for %d pages)"
              % (stem, modal_offset, len(folios)), file=sys.stderr)

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
                "printed_page": folios.get(pages[i]),
                # filled per drug below, anchored on that drug's own line
                "indication": None, "indication_kind": "none", "section_heading": None,
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
                # Anchor the indication search on the line the DRUG is on, not
                # the window start. The window spans three lines, so a heading
                # sitting between the window start and the drug was being walked
                # straight past: that is how "Neisseria meningitidis (up to 14
                # day course)" was lost from the 5-6 MU benzylpenicillin line,
                # leaving it indistinguishable from the 3-4 MU pneumococcal one.
                anchor = i
                for off in range(3):
                    if i + off < len(lines) and any(
                            p.search(lines[i + off]) for k, p in patterns if k == key):
                        anchor = i + off
                        break
                ind, ind_kind, section = indication_for(lines, anchor, drug_patterns=patterns)
                r["indication"] = ind
                r["indication_kind"] = ind_kind
                r["section_heading"] = section
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
