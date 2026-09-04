"""Acceptance gates applied to each generated pair BEFORE it is written to raw/.

These exist because of a 20-pair pilot on `aware_classification` (2026-09-04) in
which the model got 20/20 AWaRe tiers right -- the tier table was attached to
the prompt, and it copied it faithfully -- while confabulating everything the
prompt did not pin down:

  * a pair labelled `grounded` whose attached passage was EMHSLU nutrition and
    neurosurgery text, answered confidently anyway;
  * four national-formulary availability claims invented outright ("fosfomycin
    is available intravenously according to EMHSLU 2023" -- fosfomycin is not on
    EMHSLU at all);
  * free-text rationales that contradicted each other across pairs, including
    amoxicillin described as "narrow-spectrum".

That last one is subtle and worth stating: the AWaRe book says *Access
antibiotics* have "a narrow spectrum of activity". That is a property of the
GROUP. Amoxicillin is a broad-spectrum aminopenicillin that happens to be in the
Access group. The model collapsed the group property onto the member drug. So
the guard is not "never say narrow-spectrum" -- it is "never assert a spectrum
property OF A NAMED DRUG unless the passage says it of that drug".

Everything here rejects rather than repairs. A rejected pair is dropped and
regenerated; nothing is silently edited into looking correct.

Drug-name matching is reused from prepare_dataset.Validator rather than
reimplemented, so the two stages cannot drift apart -- in particular the
longest-match-first span masking that stops "ceftazidime" matching inside
"ceftazidime+avibactam".
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from prepare_dataset import Validator, norm_drug, DRUG_ALIASES  # noqa: E402


# An assertion that a drug is or is not on the national formulary / obtainable
# in Uganda. Deliberately catches the negative form too: "X is not on EMHSLU" is
# just as unsupported as the positive claim when no formulary passage is
# attached, and the pilot produced both.
AVAILABILITY = re.compile(
    r"(?:"
    r"\bavailab\w*\b|\bunavailab\w*\b|\bnot\s+available\b|"
    r"\bstocked\b|\bobtainable\b|\bin\s+stock\b|"
    r"\b(?:on|in|from)\s+the\s+EMHSLU\b|\bEMHSLU\b[^.]{0,40}\blist\b|"
    r"\b(?:listed|included|present)\b[^.]{0,30}\bEMHSLU\b|"
    r"\bessential\s+medicines\s+list\b"
    r")", re.I)

# The model saying, in some form, that the material it was given does not answer
# the question. This is the REQUIRED output when a grounded passage does not
# cover the drug asked about, so we must be able to recognise it.
NONCOVERAGE = re.compile(
    r"(?:"
    r"\bdoes\s+not\s+(?:cover|contain|include|mention|address|state|specify|list)\b|"
    r"\bdo\s+not\s+(?:cover|contain|include|mention|address|state|specify|list)\b|"
    r"\bnot\s+(?:covered|contained|included|mentioned|addressed|stated|specified|listed)\b|"
    r"\bis\s+not\s+in\s+the\s+(?:passage|material|extract|text|reference)\b|"
    r"\bno\s+information\b|\bcannot\s+(?:be\s+)?(?:confirm\w*|determine\w*|verif\w*)\b|"
    r"\bnot\s+confirmed\b|\bnot\s+possible\s+to\s+confirm\b|"
    r"\b(?:passage|material|extract|reference|source)[^.]{0,40}\bnot\b|"
    r"\bbeyond\s+(?:the\s+)?(?:scope|material)\b"
    r")", re.I)

# A spectrum / resistance-potential property asserted of something.
SPECTRUM_CLAIM = re.compile(
    r"\b(?:narrow|broad)[\s-]*spectrum\b|"
    r"\b(?:low|high|lower|higher|no)\s+(?:risk|potential)\s+(?:of|for)\s+resistance\b|"
    r"\bresistance\s+potential\b", re.I)

# Tier words, used to tell a group-level statement from a drug-level one.
TIER_WORD = re.compile(r"\b(Access|Watch|Reserve)\b", re.I)

# A dose or a duration: every number a Category 4 answer is allowed to state.
DOSE_TOKEN = re.compile(
    r"\b\d[\d,.]*\s*(?:-\s*\d[\d,.]*\s*)?"
    r"(?:mg/kg(?:/day|/dose)?|mg|g|mcg|micrograms?|IU/kg|IU|MU|ml|"
    r"days?|weeks?|hours?|months?)\b", re.I)


class Guards:
    def __init__(self):
        self.v = Validator()

    # -- helpers ------------------------------------------------------------
    def drug_names(self, text):
        """Set of normalised drug keys mentioned in text."""
        return set(self.v.drugs_in(text or "").keys())

    @staticmethod
    def question_of(record):
        """The question as the trainee sees it, minus any prepended passage."""
        first = next((m["content"] for m in record["messages"] if m["role"] == "user"), "")
        marker = "\n\nQuestion: "
        return first.split(marker, 1)[1] if marker in first else first

    @staticmethod
    def answer_of(record):
        return "\n".join(m["content"] for m in record["messages"] if m["role"] == "assistant")

    @staticmethod
    def passage_text(chunks):
        return "\n".join((c or {}).get("text", "") for c in chunks if c)

    @staticmethod
    def formulary_text(chunks):
        """Only the attached EMHSLU passages -- the sole basis for availability."""
        return "\n".join(c.get("text", "") for c in chunks
                         if c and "EMHSLU" in (c.get("doc_label") or ""))

    def drug_in_text(self, drug_key, text):
        return drug_key in self.drug_names(text)

    def unsupported_doses(self, answer, passage):
        """Dose tokens in the answer that do not occur in the attached lines.

        Compared with whitespace and thousands separators removed, so "50000
        IU/kg" matches "50,000 IU/kg" and "50mg/kg" matches "50 mg/kg", but no
        digit is ever normalised away. A dose the source does not contain is the
        worst output this project can produce, so this is a rejection, not a
        flag, and it runs before the pair is written.
        """
        hay = re.sub(r"[\s,]", "", passage).lower()
        bad = []
        for m in DOSE_TOKEN.finditer(answer):
            tok = re.sub(r"[\s,]", "", m.group(0)).lower()
            if tok not in hay:
                bad.append(m.group(0).strip())
        return sorted(set(bad))

    def true_tier(self, drug_key):
        """AWaRe tier(s) for a drug: a string, a {route: tier} dict, or None."""
        key = DRUG_ALIASES.get(drug_key, drug_key)
        rd = self.v.lookup["route_dependent"].get(key)
        return rd if rd else self.v.lookup["flat"].get(key)

    # -- the gate -----------------------------------------------------------
    def screen(self, record, chunks):
        """Return a list of reasons this pair must be rejected. Empty == accept."""
        reasons = []
        question = self.question_of(record)
        answer = self.answer_of(record)

        # prepare_dataset blocks answers under 80 chars; catch them here too so
        # the pair is regenerated rather than written and rejected later. One
        # batch produced the bare "The passage provided does not cover
        # spiramycin." -- a refusal with no reasoning is not a training example,
        # and in that case the drug was in fact classified.
        if len(answer.strip()) < 80:
            reasons.append("answer_too_short:%d_chars" % len(answer.strip()))
        mode = record.get("context_mode")
        passage = self.passage_text(chunks)
        formulary = self.formulary_text(chunks)
        says_noncoverage = bool(NONCOVERAGE.search(answer))

        q_drugs = self.drug_names(question)
        a_drugs = self.drug_names(answer)

        # (1) GROUNDING INTEGRITY.
        # If we told the model "here is the relevant reference material" and the
        # drug being asked about is not in it, the only honest answer is to say
        # so. Answering confidently is the pilot's worst failure: it reads as
        # correct and is sourced to a passage that never mentioned the drug.
        if mode == "grounded":
            if not passage.strip():
                reasons.append("grounded_but_no_passage")
            else:
                absent = sorted(d for d in q_drugs if not self.drug_in_text(d, passage))
                if absent and not says_noncoverage:
                    reasons.append("grounded_passage_lacks_question_drug:%s" % ",".join(absent))
                # A grounded answer that names drugs the passage never mentions,
                # without flagging that, is drawing on parametric memory.
                extra = sorted(d for d in a_drugs
                               if not self.drug_in_text(d, passage) and d not in q_drugs)
                if extra and not says_noncoverage:
                    reasons.append("grounded_answer_adds_unsourced_drug:%s" % ",".join(extra))

        # (1b) DOSE GROUNDING, Category 4 only.
        # Every number must be in the lines the call was given. The passage here
        # is not a whole chunk but the specific extracted dose lines for this
        # drug, with each line's doses already attributed to the drug they
        # actually belong to -- so a number that is not present is either
        # invented or borrowed from a neighbouring drug.
        if record.get("category") == "dosing_duration":
            if not passage.strip():
                reasons.append("dosing_without_source_lines")
            else:
                bad = self.unsupported_doses(answer, passage)
                if bad:
                    reasons.append("dose_not_in_source_lines:%s" % ",".join(bad[:4]))

        # (2) AVAILABILITY CLAIMS.
        # Only an attached EMHSLU passage can support one, and only for a drug
        # actually named in it. Absence of a drug from an attached formulary
        # extract is NOT evidence of absence from the formulary either, because
        # the extract is a fragment -- so "not available" needs the same support
        # as "available", and without it the model must decline to say.
        # Only counts as a formulary claim if the sentence actually ties supply
        # to a drug or to Uganda/EMHSLU. A generic line like "Reserve agents
        # should be available only in referral hospitals" is not a claim about
        # what Uganda stocks, and rejecting it would starve the generator.
        for m in AVAILABILITY.finditer(answer):
            window = answer[max(0, m.start() - 130):m.end() + 130]
            scoped = self.drug_names(window)
            if not scoped and not re.search(r"\b(Uganda|EMHSLU)\b", window, re.I):
                continue
            if not formulary.strip():
                reasons.append("availability_claim_without_formulary_passage")
                break
            unsupported = sorted(d for d in (scoped or q_drugs)
                                 if not self.drug_in_text(d, formulary))
            if unsupported:
                reasons.append("availability_claim_for_drug_absent_from_formulary:%s"
                               % ",".join(unsupported))
                break

        # (3) TIER CORRECTNESS.
        # The classification table is attached to every call in this category,
        # so a wrong tier is the model overriding the source it was handed. It
        # happens: one batch answered "meropenem is Reserve" with the row
        # "Meropenem | Carbapenems | J01DH02 | Watch | Yes" in its own prompt.
        # A wrong tier is the single most damaging thing this category can emit
        # -- it is a confident, quotable, checkable claim -- so it is rejected
        # here rather than merely flagged downstream.
        for key in sorted(a_drugs | q_drugs):
            truth = self.true_tier(key)
            if truth is None:
                continue
            m = TIER_WORD.search(answer)
            if not m:
                continue
            stated = {t.capitalize() for t in TIER_WORD.findall(answer)}
            wanted = set(truth.values()) if isinstance(truth, dict) else {truth}
            # The rationale block quotes all three group names, so require only
            # that the correct tier is present and no other tier is asserted OF
            # this drug within a short window around its mention.
            near = set()
            for tm in TIER_WORD.finditer(answer):
                window = answer[max(0, tm.start() - 90):tm.end() + 40]
                if key in self.drug_names(window):
                    near.add(tm.group(0).capitalize())
            claimed = near or stated
            if claimed and not (wanted & claimed):
                reasons.append("tier_contradicts_classification:%s_stated_%s_expected_%s"
                               % (key, "/".join(sorted(claimed)), "/".join(sorted(wanted))))

        # (4) DRUG-LEVEL PROPERTY CLAIMS.
        # A spectrum or resistance-potential statement is allowed as a property
        # of an AWaRe GROUP (the book defines the groups that way) but not as a
        # property of a named drug unless the passage says it of that drug.
        for m in SPECTRUM_CLAIM.finditer(answer):
            window = answer[max(0, m.start() - 120):m.end() + 120]
            named = self.drug_names(window)
            if not named:
                continue                     # group-level statement, fine
            if TIER_WORD.search(window) and not named:
                continue
            unsupported = sorted(d for d in named if not self.drug_in_text(d, passage))
            if unsupported:
                reasons.append("drug_level_property_claim_unsourced:%s" % ",".join(unsupported))
                break

        return reasons


def primary_drug(guards, record):
    """The drug a pair is 'about', for coverage-spreading. None if unclear."""
    q = guards.drug_names(guards.question_of(record))
    if len(q) == 1:
        return next(iter(q))
    if q:
        return sorted(q)[0]
    a = guards.drug_names(guards.answer_of(record))
    return sorted(a)[0] if a else None
