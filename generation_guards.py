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

# Organism names as guidelines write them, against the adjectival forms
# clinicians actually use. Without this bridge an answer saying "meningococcal
# meningitis" would be judged not to match a source line headed "Neisseria
# meningitidis", and correct pairs would be thrown away.
ORGANISM_ADJECTIVES = {
    "streptococcus pneumoniae": "pneumococc",
    "neisseria meningitidis": "meningococc",
    "neisseria gonorrhoeae": "gonococc",
    "haemophilus influenzae": "haemophilus",
    "staphylococcus aureus": "staphylococc",
    "streptococcus pyogenes": "streptococc",
    "listeria monocytogenes": "listeri",
    "escherichia coli": "coliform",
    "clostridioides difficile": "difficile",
    "salmonella typhi": "typhoid",
}
GENERIC_DISEASE_WORDS = {
    # shared by every entry in a section, so they discriminate nothing
    "meningitis", "meningitidis", "pneumonia", "infection", "infections",
    "disease", "syndrome", "sepsis", "septicaemia", "fever",
}
INDICATION_STOPWORDS = {
    "course", "cases", "adult", "adults", "child", "children", "severe", "mild",
    "acute", "chronic", "first", "second", "third", "choice", "empiric",
    "empirical", "regimen", "treatment", "single", "daily", "hours", "weeks",
    "identified", "causative", "organisms", "patients", "including", "before",
}

# A claim that AWaRe does not classify a drug AT ALL, as opposed to the honest
# local statement that the material provided does not cover it. The qualifiers
# ("provided", "above", "in this extract") mark the local form and are excluded,
# because saying the extract does not contain a drug is true and wanted.
#
# A 550-pair run produced 13 answers asserting that rifampicin, sisomicin,
# daptomycin and tinidazole are unclassified. All four have tiers. The model was
# generalising from the absence of a row in its own excerpt, which is exactly
# what a focused-retrieval prompt invites unless it is told not to.
NONCLASSIFICATION_GLOBAL = re.compile(
    r"AWaRe\s+(?:classification\s+)?does\s+not\s+(?:classify|specify|include|list|cover|assign)|"
    r"(?:is\s+)?not\s+classified\s+(?:by|under|in)\s+AWaRe|"
r"AWaRe[^.]{0,40}does\s+not\s+specify\s+a\s+(?:category|tier|group)|"
    r"not\s+(?:in|included\s+in|listed\s+in|part\s+of)\s+the\s+AWaRe\s+classification"
    r"(?!\s+(?:provided|above|shown|given|extract|rows|here|in\s+this))",
    re.I)

# An ATC code as printed in the classification rows, e.g. J01CA11. Category 3
# answers must cite the one belonging to the row they used, which is a claim that
# can be checked -- unlike "I read the table", which cannot.
ATC_CODE = re.compile(r"(?<![A-Za-z0-9])([A-Z][0-9]{2}[A-Z]{2}[0-9]{2})(?![A-Za-z0-9])")

# A dose or a duration: every number a Category 4 answer is allowed to state.
DOSE_TOKEN = re.compile(
    r"\b\d[\d,.]*\s*(?:-\s*\d[\d,.]*\s*)?"
    r"(?:mg/kg(?:/day|/dose)?|mg|g|mcg|micrograms?|IU/kg|IU|MU|ml|"
    r"days?|weeks?|hours?|months?)\b", re.I)


class Guards:
    def __init__(self):
        self.v = Validator()
        self._build_atc_index()
        self._build_component_index()
        from c2_exception_guard import BoundaryGuards
        self.boundary = BoundaryGuards(self.v)

    def _build_atc_index(self):
        """ATC code -> the drugs and tiers that code actually belongs to.

        Requiring an answer to quote a code only proves it copied something
        code-shaped from the excerpt. Resolving the code back to its row is what
        makes the citation checkable: it catches an answer citing a real code
        that belongs to a different drug, or to a row whose tier contradicts the
        one asserted. A field that is present but unverified is the same bug
        shape as the facility-level column standing in for an indication.
        """
        self.atc = {}
        for v in self.v.aware["antibiotics"].values():
            code = (v.get("atc") or "").strip().upper()
            if not ATC_CODE.fullmatch(code):
                continue          # "to be assigned" and similar placeholders
            entry = self.atc.setdefault(code, {"drugs": set(), "tiers": set()})
            entry["drugs"].add(norm_drug(re.sub(r"_(IV|oral)$", "", v["antibiotic"])))
            if v.get("category"):
                entry["tiers"].add(v["category"])

    def _build_component_index(self):
        """Combination name -> its component drugs, when each is itself known.

        "rifampicin-clofazimine-dapsone" matches the EMHSLU combination, which
        carries no AWaRe tier, and the bare "rifampicin" inside it is masked out
        by longest-match-first. That let an answer claim AWaRe does not classify
        it, while rifampicin on its own is Watch.
        """
        known = {norm_drug(re.sub(r"_(IV|oral)$", "", n)) for n in self.v.known_drugs}
        self.components = {}
        for n in self.v.known_drugs:
            base = re.sub(r"_(IV|oral)$", "", n)
            parts = [norm_drug(x) for x in re.split(r"[+/\-]| and ", base) if x.strip()]
            parts = [x for x in parts if x and x in known and x != norm_drug(base)]
            if len(parts) > 1:
                self.components[norm_drug(base)] = parts

    def expand_components(self, keys):
        """Drug keys plus, for combinations, the components they are made of."""
        out = set(keys)
        for k in list(keys):
            out.update(self.components.get(k, ()))
        return out

    def atc_citation_problems(self, answer):
        """Cited codes that do not support what the answer actually says."""
        named = self.drug_names(answer)
        stated = {t.capitalize() for t in TIER_WORD.findall(answer)}
        problems = []
        for m in ATC_CODE.finditer(answer):
            code = m.group(1).upper()
            entry = self.atc.get(code)
            if not entry:
                problems.append("%s is not an ATC code in the classification" % code)
                continue
            if named and not (entry["drugs"] & named):
                problems.append("%s belongs to %s, not to the drug named"
                                % (code, "/".join(sorted(entry["drugs"]))))
                continue
            if stated and entry["tiers"] and not (entry["tiers"] & stated):
                problems.append("%s is %s but the answer says %s"
                                % (code, "/".join(sorted(entry["tiers"])),
                                   "/".join(sorted(stated))))
        return problems

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

    @staticmethod
    def is_age_reference(answer, match):
        """True if this number is somebody's age, not a dose or a duration.

        "infants up to 2 months old" is a patient description restated from the
        question, not a treatment duration, and demanding it appear in a dose
        line throws away correct paediatric pairs -- the commonest question
        shape in this category.
        """
        before = answer[max(0, match.start() - 40):match.start()].lower()
        after = answer[match.end():match.end() + 18].lower()
        return bool(re.search(r"\b(?:aged?|age of|older than|younger than|under|over|"
                              r"up to|infants?|neonates?|children|child|babies|baby)\s*$", before)
                    or re.match(r"\s*(?:old\b|of age\b|-old\b)", after))

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
            if self.is_age_reference(answer, m):
                continue
            tok = re.sub(r"[\s,]", "", m.group(0)).lower()
            if tok not in hay:
                bad.append(m.group(0).strip())
        return sorted(set(bad))

    @staticmethod
    def source_lines(passage):
        """[(indication, line_text)] parsed back out of a dose-line passage."""
        out = []
        for line in passage.split("\n"):
            m = re.match(r"\[[^\]]*\]\s*\(indication:\s*(.*?)\)\s*(.*)$", line.strip())
            if m:
                out.append((m.group(1).strip(), m.group(2).strip()))
        return out

    @staticmethod
    def discriminators(indication):
        """Terms that distinguish THIS indication from its neighbours.

        Species epithets are useless here: "meningitidis" shares nine leading
        characters with "meningitis", so prefix-matching let an answer about
        pneumococcal meningitis satisfy a Neisseria meningitidis source line.
        The genus and the clinical adjective are distinctive; the disease word
        shared by every entry in the section is not.
        """
        s = (indication or "").lower()
        out = set()
        for binom, adj in ORGANISM_ADJECTIVES.items():
            if binom in s:
                out.add(adj)
                out.add(binom.split()[0])          # genus, e.g. "neisseria"
        if out:
            return out
        for w in re.findall(r"[a-z]{5,}", s):
            if w not in INDICATION_STOPWORDS and w not in GENERIC_DISEASE_WORDS:
                out.add(w)
        return out

    @staticmethod
    def indication_terms(indication):
        """Words an answer could legitimately use to name this indication.

        Guidelines name the organism ("Neisseria meningitidis"); clinicians name
        the disease ("meningococcal meningitis"). Matching only the literal
        string would reject correct answers, so the adjectival forms are mapped
        explicitly rather than guessed.
        """
        s = (indication or "").lower()
        terms = set()
        for binom, adj in ORGANISM_ADJECTIVES.items():
            if binom in s:
                terms.add(adj)
                terms.update(binom.split())
        for w in re.findall(r"[a-z]{5,}", s):
            if w not in INDICATION_STOPWORDS:
                terms.add(w)
        return {t[:7] for t in terms if len(t) >= 5}

    def indication_mismatch(self, answer, passage):
        """Doses stated for a condition other than the one their source line is for.

        For each number in the answer, find which source lines carry it; if none
        of those lines' indications is named anywhere in the answer, the answer
        has taken a real dose and attached it to the wrong clinical question.
        That is the failure the dose gate cannot see, because the number is
        genuinely in the source.
        """
        lines = self.source_lines(passage)
        if not lines:
            return []
        low = answer.lower()
        distinct = {ind for ind, _t in lines if ind and ind.lower() != "not recorded"}
        if len(distinct) < 2:
            # Only one indication in play: naming it adds nothing to check
            # against, and the dose gate already covers the numbers.
            return []

        claimed = {ind for ind in distinct
                   if any(t in low for t in self.discriminators(ind))}
        problems = []
        for m in DOSE_TOKEN.finditer(answer):
            if self.is_age_reference(answer, m):
                continue
            tok = re.sub(r"[\s,]", "", m.group(0)).lower()
            owners = {ind for ind, text in lines
                      if tok in re.sub(r"[\s,]", "", text).lower()}
            owners = {o for o in owners if o and o.lower() != "not recorded"}
            if not owners:
                continue
            if not claimed:
                problems.append("%s stated with no indication named (source: %s)"
                                % (m.group(0).strip(), sorted(owners)[0][:55]))
            elif not (owners & claimed):
                problems.append("%s belongs to '%s' but the answer names '%s'"
                                % (m.group(0).strip(), sorted(owners)[0][:45],
                                   sorted(claimed)[0][:45]))
        return problems

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
        stripped = answer.strip()
        if len(stripped) < 80 and not (self.drug_names(stripped) and TIER_WORD.search(stripped)):
            reasons.append("answer_lacks_substance:%d_chars" % len(stripped))
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
                else:
                    # Only meaningful once the numbers themselves check out.
                    mism = self.indication_mismatch(answer, passage)
                    if mism:
                        reasons.append("dose_stated_for_wrong_indication:%s" % "; ".join(mism[:2]))

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
            # Every route, not any of them. Fosfomycin is Reserve IV and Watch
            # oral; an answer naming only "Reserve" is clinically incomplete and
            # a set-intersection test lets it through.
            if claimed and not wanted.issubset(claimed):
                reasons.append("tier_contradicts_classification:%s_stated_%s_expected_%s"
                               % (key, "/".join(sorted(claimed)), "/".join(sorted(wanted))))

        # (3a) CITED ATC CODE MUST SUPPORT THE ANSWER.
        # A tier assertion has to quote the code from the row it used, and that
        # code is then resolved back to its drug and tier. Copying any code from
        # the excerpt is not enough.
        if record.get("category") == "aware_classification":
            stated_tiers = {t.capitalize() for t in TIER_WORD.findall(answer)}
            drug_named = bool(a_drugs | q_drugs)
            cited = ATC_CODE.findall(answer)
            if stated_tiers and drug_named and not says_noncoverage and not cited:
                reasons.append("tier_asserted_without_atc_citation")
            bad_cite = self.atc_citation_problems(answer)
            if bad_cite:
                reasons.append("atc_citation_does_not_support_answer:%s" % "; ".join(bad_cite[:2]))

        # (3b) FALSE NON-CLASSIFICATION.
        # Saying "the extract does not cover X" is honest and wanted. Saying
        # "AWaRe does not classify X" about a drug that has a tier is a factual
        # error, and a damaging one to train on: it teaches the model to deny
        # classification for real antibiotics.
        if NONCLASSIFICATION_GLOBAL.search(answer):
            candidates = self.expand_components(q_drugs | a_drugs)
            wrongly = sorted(d for d in candidates if self.true_tier(d) is not None)
            if wrongly:
                reasons.append("false_nonclassification_claim:%s" % ",".join(wrongly[:3]))

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

        # (5) THE CATEGORY 1/2 BOUNDARY.
        # Categories 1 and 2 are two sides of one exception boundary, so they are
        # gated together against the reviewed scenario pool: no conditional
        # hand-off of a prescribing rule, no criterion stated with a threshold the
        # source does not contain, and the vignette must actually fall on the side
        # its category claims. See c2_exception_guard for why a drug-name gate
        # cannot see any of that.
        reasons.extend(self.boundary.screen(record, question, answer))

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
