"""The Category 1 / Category 2 scenario pool, and the checks that keep it honest.

Categories 1 (`antibiotic_indicated`) and 2 (`no_antibiotic_indicated`) are not
two independent categories. They are the two sides of one boundary: for each
clinical presentation the guidelines state a default (usually "no antibiotic")
and a named, enumerable set of criteria that move a patient across it. Generating
them independently produces pairs whose only difference is the generator's mood.

So the pool is built first, reviewed by a clinician BEFORE generation scales, and
then every generated vignette is checked against it. Each statement records:

  * the verbatim source string it rests on, and the chunk that contains it --
    verified mechanically by `verify()`, so a criterion cannot enter the pool
    because it is clinically true, only because the corpus says it;
  * whether it is the default, an exception criterion, an explicit NON-criterion
    (things guidelines name specifically so they are not mistaken for
    indications), the symptomatic substitute, an escalation trigger, or the
    first-choice agent for the far side of the boundary;
  * for exception criteria, surface cues and any numeric bound, so a vignette can
    be tested for whether it actually satisfies the criterion.

The numeric bounds are the load-bearing part. An answer that loosens "≥10 days"
to "3 days", or "bilateral in children under 2" to "under 5", withholds the
antibiotic correctly for the patient in front of it while teaching a lower
threshold for every future patient. That answer passes a drug-name gate, passes a
"did it refuse" gate, and is the failure mode this pool exists to make checkable.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNK_DIR = ROOT / "source_material" / "chunks"

# Symbol-font residue left by the AWaRe book's subsetted fonts. These are the
# ligature glyphs, not the private-use codepoints repaired upstream: the text
# layer spells "significant" as "signi/g246cant". Mapped here rather than in the
# extractor because the extractor's job is fidelity, not readability -- a quote
# check has to see through the encoding without the corpus being rewritten.
GLYPH = {"/g246": "fi", "/g248": "fl", "/g255": "ff", "/c162": "", "/g255i": "ffi"}


def normalise(s):
    """Collapse a string to what a quote check can compare.

    The AWaRe infographics are letter-spaced ("fe ve r ≥39.0 °C"), so whitespace
    cannot be trusted and is removed entirely -- which incidentally repairs the
    letter-spacing exactly. Digits and the decimal point are kept because they
    are the part of a criterion that matters; every other symbol goes, so "≥ 39.0
    °C" and "≥39.0°C" compare equal.
    """
    s = (s or "").lower()
    for k, v in GLYPH.items():
        s = s.replace(k, v)
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[^a-z0-9.]", "", s)


_chunk_cache = {}


def chunk_text(topic):
    if topic not in _chunk_cache:
        p = CHUNK_DIR / ("%s.txt" % topic)
        _chunk_cache[topic] = normalise(p.read_text(encoding="utf-8"))
    return _chunk_cache[topic]


def chunk_index():
    """chunk_id -> normalised text, from the manifest."""
    out = {}
    with open(CHUNK_DIR / "_manifest.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[r["chunk_id"]] = normalise(r["text"])
    return out


# --- negation -----------------------------------------------------------------
# A Category 2 vignette states the exception criteria in the NEGATIVE far more
# often than it omits them: "no blood in the stool", "afebrile", "sputum
# unchanged in colour and volume". A substring test reads those as the criterion
# being present and flips the vignette to the wrong side of the boundary. This is
# the same shape as the meningitidis/meningitis prefix collision in the dose
# gate: the cue is there, its polarity is not.
NEGATORS = (
    r"\b(?:no|not|non|without|denies|denied|absent|neither|nor|never|"
    r"unchanged|normal|negative|free\s+of|rather\s+than|instead\s+of|"
    r"lacks?|lacking|apart\s+from|other\s+than|except)\b"
)
# Single words that are themselves the negation of a criterion cue.
LEXICAL_NEGATIONS = {
    "fever": [r"\bafebrile\b", r"\bno\s+fever\b", r"\bnormothermic\b"],
    "cough": [r"\bno\s+cough\b"],
}


# A negator distributes over a coordinated LIST: "no malnutrition, measles,
# cardiac disease or anaemia" rules out all four, and that form is unavoidable
# here because BRO-05 names six comorbidities in one criterion, so a Category 2
# bronchitis vignette must rule out all six at once. Ending the negator's scope
# at the first comma -- which is what stopped "no cough -- a Centor score of 4"
# from over-reaching -- makes every item after the first read as present, and the
# gate then rejects a correct answer. Scope therefore crosses a comma only while
# the crossed text is still bare list items: letters and spaces, no article, no
# number, no verb of assertion. "a Centor score of 4" fails all three tests.
_LIST_ITEM = re.compile(r"^\s*(?:(?:or|and|nor)\s+)?[a-z][a-z\s'-]*$", re.I)
_ASSERTS = re.compile(
    r"\b(?:is|are|was|were|has|have|had|shows?|showing|present|noted|seen|"
    r"reports?|reported|develops?|developed|complains?|with|plus)\b", re.I)


def _negated_across_list(before):
    """True if `before` ends inside a list still governed by an earlier negator."""
    last = None
    for m in re.finditer(NEGATORS, before, re.I):
        last = m
    if last is None:
        return False
    tail = before[last.end():]
    if "," not in tail or len(tail) > 90:
        return False
    if re.search(r"[.;()\[\]–—-]", tail):
        return False
    pieces = [p for p in tail.split(",") if p.strip()]
    return bool(pieces) and all(_LIST_ITEM.match(p) and not _ASSERTS.search(p)
                                for p in pieces)


def _negated(text, start, end):
    """True if this cue occurrence sits inside a negation."""
    before = text[max(0, start - 90):start]
    after = text[end:end + 30]
    if re.search(NEGATORS + r"[^.;,()\[\]–—-]{0,18}$", before, re.I):
        return True
    if _negated_across_list(before):
        return True
    if re.match(r"[^.;]{0,20}\b(?:is|are|was|were)\s+(?:absent|normal|not\b)", after, re.I):
        return True
    return False


def cue_present(text, pattern):
    """True if `pattern` occurs in `text` at least once un-negated."""
    for m in re.finditer(pattern, text, re.I):
        if not _negated(text, m.start(), m.end()):
            return True
    return False


# --- numeric bounds -----------------------------------------------------------
UNIT_PATTERNS = {
    "temp_c": r"(\d{2}(?:\.\d)?)\s*(?:°\s*c|degrees?\b|c\b)",
    "days": r"(\d{1,3})\s*(?:-\s*\d{1,3}\s*)?days?\b",
    # Hyphens matter: "4-year-old" and "18-month-old" are how every paediatric
    # vignette states an age. The earlier patterns required whitespace, or a
    # second digit after the hyphen (reading it as a range), so neither form
    # matched -- and the AOM "bilateral in children under 2 years" bound, the
    # most-loosened criterion in this pool, could not fire on the one age band
    # it was written for.
    "years": r"(\d{1,2})\s*(?:-\s*\d{1,2}(?=\s*(?:years?|yrs?)))?[\s-]*(?:years?|yrs?|y/?o)\b",
    "months": r"(\d{1,3})\s*(?:-\s*\d{1,3}(?=\s*months?))?[\s-]*months?\b",
    "bpm": r"(\d{2,3})\s*(?:beats?|bpm)\b",
    "rr": r"(\d{2,3})\s*(?:breaths?|/min)\b",
}
OPS = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
}


def numbers_of_kind(text, kind):
    pat = UNIT_PATTERNS[kind]
    out = []
    for m in re.finditer(pat, text, re.I):
        try:
            out.append((float(m.group(1)), m.start(), m.end()))
        except ValueError:
            pass
    if kind == "years":
        # Paediatric vignettes give age in months, and the guideline bound that
        # matters most in this pool -- bilateral otitis in children under 2 --
        # is written in years. An 18-month-old is under two; without this
        # conversion the criterion silently fails to fire for exactly the age
        # band it was written for, and every such pair lands on the wrong side
        # of the boundary. Age- and weight-framed paediatric questions are the
        # commonest shape in this dataset, so this is not an edge case.
        for m in re.finditer(UNIT_PATTERNS["months"], text, re.I):
            try:
                out.append((float(m.group(1)) / 12.0, m.start(), m.end()))
            except ValueError:
                pass
    return out


def bound_satisfied(text, bound):
    """True if the vignette states a value on the trigger side of the bound."""
    kind, op, value = bound["kind"], bound["op"], bound["value"]
    for val, s, e in numbers_of_kind(text, kind):
        if _negated(text, s, e):
            continue
        if OPS[op](val, value):
            return True
    return False


from scenario_pool_data import POOL  # noqa: E402


def presentations():
    out = []
    for e in POOL:
        if e["presentation"] not in out:
            out.append(e["presentation"])
    return out


def entries(presentation, kind=None):
    return [e for e in POOL
            if e["presentation"] == presentation and (kind is None or e["kind"] == kind)]


# --- boundary variants --------------------------------------------------------
# Two presentations have more than one boundary, because two corpus sources draw
# it in different places. The ruling on both was to keep BOTH rather than pick a
# winner, generate from each, and tag every pair with the source whose criteria
# it followed -- so the disagreement is recorded in the dataset itself and can be
# reported, rather than resolved by fiat and lost.
AWARE = "who_aware_book_2022"
UCG = "uganda_clinical_guidelines_2023"


def entry_source(e):
    doc = e.get("doc") or ""
    if "AWaRe" in doc and "Uganda" in doc:
        return "both"
    if "AWaRe" in doc:
        return AWARE
    if "Uganda" in doc:
        return UCG
    return "unknown"


def variants(presentation):
    """The distinct boundaries drawn for this presentation, one per source.

    A variant is what a single generated pair is written against: one default,
    one set of exception criteria, and the substitute and escalation material
    that goes with them. `cross_source` marks a variant whose two halves come
    from different documents -- acute bronchitis, where AWaRe states the default
    and names no exception at all, and UCG 5.2.2 names the only exception in the
    corpus. Such a pair is still generated; it is tagged so the paper can say how
    many pairs rest on a criterion the other source does not recognise.
    """
    exc = entries(presentation, "exception")
    by_src = {}
    for e in exc:
        by_src.setdefault(entry_source(e), []).append(e)
    defaults = entries(presentation, "default")
    out = []
    for src, es in sorted(by_src.items()):
        # The default is preferably from the same document as the criteria; if
        # that document states none, the other source's default is used and the
        # variant is flagged cross-source.
        own = [d for d in defaults if entry_source(d) == src]
        cross = not own
        out.append(dict(
            presentation=presentation,
            criteria_source=src,
            cross_source=cross,
            default_ids=[d["id"] for d in (own or defaults)],
            default_source=entry_source((own or defaults)[0]) if defaults else None,
            exception_ids=[e["id"] for e in es],
            substitute_ids=[e["id"] for e in entries(presentation, "substitute")],
            escalation_ids=[e["id"] for e in entries(presentation, "escalation")],
            non_criterion_ids=[e["id"] for e in entries(presentation, "non_criterion")],
            first_choice_ids=[e["id"] for e in entries(presentation, "first_choice")],
            chunk_ids=sorted({e["chunk_id"] for e in es}
                             | {d["chunk_id"] for d in (own or defaults)}),
        ))
    return out


def all_variants():
    return [v for p in presentations() for v in variants(p)]


def verify():
    """Every quote must occur in the chunk it names. Returns a list of failures.

    This is the pool's own acceptance gate, and it runs before anything is
    generated. A statement that is clinically correct but not in the chunk it
    cites is exactly the confabulation this project exists to prevent -- there is
    no reason to hold the pool to a looser standard than the model.
    """
    idx = chunk_index()
    bad = []
    for e in POOL:
        needle = normalise(e.get("quote_check") or e["quote"])
        cid = e["chunk_id"]
        if cid not in idx:
            bad.append((e["id"], "chunk_id not in manifest: %s" % cid))
            continue
        if needle in idx[cid]:
            continue
        # Fall back to the whole topic file, so a quote that is real but landed
        # in the neighbouring 6,000-character window is reported as a
        # misattribution rather than as a fabrication -- different defects.
        if needle in chunk_text(e["presentation"]):
            bad.append((e["id"], "quote is in the topic but NOT in %s" % cid))
        else:
            bad.append((e["id"], "quote not found in corpus: %r" % e["quote"][:70]))
    return bad


def satisfied_criteria(presentation, vignette, source=None):
    """Which exception criteria this vignette actually satisfies.

    A criterion is a set of LIMBS -- each cue alternation is one limb, and the
    numeric bound, if any, is another -- of which `min_limbs` must fire.

    Counting one limb as satisfaction does not work, and the reason is the same
    one the dose gate ran into with "meningitidis" against "meningitis": the cue
    that fires is the one every case of the presentation shares. A Category 2
    COPD vignette necessarily mentions breathlessness, because breathlessness is
    what a COPD exacerbation IS; a bare `breathless` cue therefore marks every
    correct Category 2 vignette as crossing the boundary. What discriminates is
    never the presenting symptom -- it is the qualifier the guideline attached to
    it (purulence WITH increased volume, admission, a temperature over a stated
    threshold). Non-discriminating cues are removed from the pool for the same
    reason GENERIC_DISEASE_WORDS are excluded from dose discriminators.

    Returns [(id, [what fired])].
    """
    hits = []
    for e in entries(presentation, "exception"):
        # When a pair was generated against one source's boundary, it is judged
        # against that source's criteria only. A COPD vignette at 38.5 C crosses
        # UCG's threshold (temp >38) and does not cross AWaRe's (severe /
        # hospitalised); which of those is the right answer depends entirely on
        # which boundary the pair was written for, so the tag has to be honoured
        # or the discrimination check is measuring the wrong thing.
        if source and entry_source(e) not in (source, "both"):
            continue
        fired = []
        for pat in e.get("cues", ()):
            if cue_present(vignette, pat):
                fired.append("cue:%s" % pat[:38])
        b = e.get("bound")
        if b and bound_satisfied(vignette, b):
            fired.append("bound:%s%s%s" % (b["kind"], b["op"], b["value"]))
        if len(fired) >= e.get("min_limbs", 1):
            hits.append((e["id"], fired))
    return hits


def stated_bounds(text):
    """(kind, value) pairs the text asserts, for threshold-loosening checks."""
    out = set()
    for kind in UNIT_PATTERNS:
        for val, _s, _e in numbers_of_kind(text, kind):
            out.add((kind, val))
    return out


def pool_bounds(presentation, source=None):
    """(kind, value) pairs the corpus states as CRITERIA for this presentation."""
    out = set()
    for e in entries(presentation, "exception"):
        if source and entry_source(e) not in (source, "both"):
            continue
        b = e.get("bound")
        if b:
            out.add((b["kind"], float(b["value"])))
    return out


def main():
    bad = verify()
    idx = chunk_index()
    print("scenario pool: %d statements across %d presentations"
          % (len(POOL), len(presentations())))
    print()
    for p in presentations():
        es = entries(p)
        kinds = {}
        for e in es:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        print("  %-22s %2d  %s" % (p, len(es),
              " ".join("%s=%d" % kv for kv in sorted(kinds.items()))))
    print()
    print("quote verification against source chunks:")
    if not bad:
        print("  all %d quotes found verbatim in the chunk they cite" % len(POOL))
    else:
        for pid, why in bad:
            print("  FAIL %-8s %s" % (pid, why))
    print()
    print("chunks carrying BOTH sides of the boundary (for interleaved C1/C2 pairs):")
    for p in presentations():
        both = []
        for cid, txt in idx.items():
            if not cid.startswith(p + "::"):
                continue
            has_default = any(normalise(e.get("quote_check") or e["quote"]) in txt
                              for e in entries(p) if e["kind"] == "default")
            has_exc = any(normalise(e.get("quote_check") or e["quote"]) in txt
                          for e in entries(p) if e["kind"] == "exception")
            if has_default and has_exc:
                both.append(cid)
        print("  %-22s %s" % (p, ", ".join(both) if both else "NONE -- pairs must span chunks"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
