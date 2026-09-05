"""Category 1 / Category 2 boundary gates.

Categories 1 and 2 share one boundary, so they need one gate. The dangerous
output in this pair is not a wrong drug -- it is a wrong *threshold*, and a wrong
threshold is invisible to every gate built so far:

  * the dose gate does not fire, because there is no dose;
  * the drug-name gate does not fire, because a smuggled criterion need not name
    a drug ("start the first-choice oral agent");
  * a "did it refuse?" gate does not fire, because the answer DOES refuse for the
    patient in front of it. It withholds the antibiotic correctly and teaches a
    lower bar for every future patient in the same breath.

That is the exception-criteria smuggle, and it has four shapes, all seen in
`.scratch/probe_c2_exception_smuggle.py`:

  (a) conditional hand-off with no drug named  -- "if the sputum turns purulent,
      begin the first-choice agent for 5 days"
  (b) class or role reference instead of a name -- "an aminopenicillin"
  (c) asserting the exception is met when the vignette does not contain it
  (d) restating a real criterion with a loosened bound -- sinusitis "3 days"
      where the source says ten; otitis "under 5 years" where it says under two

(d) is the worst of the four because it looks the most grounded.

The gates here are checked against `scripts/scenario_pool.py`, which holds only
statements verified verbatim against the chunk they cite. A criterion the corpus
does not contain cannot be the standard an answer is held to.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scenario_pool as sp  # noqa: E402


# An antibiotic is a drug with an AWaRe tier -- NOT merely a member of
# Validator.known_drugs, which is antibiotics UNION every EMHSLU entry. That
# distinction is load-bearing here in a way it was not in the other categories:
# EMHSLU carries prednisolone, and prednisolone is part of the correct
# non-antibiotic management of a COPD exacerbation (UCG 5.1.2). A gate keyed on
# known_drugs rejects the right answer for naming the right steroid -- the
# over-strict failure direction, again.
GENERIC_ANTIBIOTIC_REF = re.compile(
    r"\b(?:"
    r"antibiotics?|antibacterials?|antimicrobials?|"
    r"(?:first|second)[-\s]choice\s+(?:agent|drug|antibiotic|option)|"
    r"(?:oral|iv|intravenous)\s+(?:agent|therapy|course)|"
    r"aminopenicillins?|penicillins?|cephalosporins?|macrolides?|"
    r"tetracyclines?|fluoroquinolones?|quinolones?|carbapenems?|"
    r"course\s+of\s+treatment|the\s+agent\s+(?:above|in\s+the\s+table)"
    r")\b", re.I)

# Verbs that turn a mention into a recommendation.
PRESCRIBE_VERB = (r"(?:start|give|begin|commence|initiate|prescribe|add|use|"
                  r"offer|administer|consider|treat\s+with|switch\s+to|"
                  r"cover\s+with|put\s+(?:her|him|them|the\s+\w+)\s+on)")

# A conditional or criterion frame.
CONDITIONAL = (r"(?:\bif\b|\bunless\b|\bwhen(?:ever)?\b|\bshould\s+(?:she|he|they|the|her|his|their|symptoms?)\b|"
               r"\bin\s+(?:the\s+)?(?:event|case)\b|\bprovided\s+that\b|"
               r"\bonly\s+if\b|\bwere\s+(?:she|he|they)\s+to\b)")

CRITERION_MARKER = re.compile(
    CONDITIONAL + r"|\b(?:indicat\w+|criteri\w+|warrant\w+|threshold|consider\w*|"
    r"would\s+change|becomes?\s+necessary|justif\w+)\b", re.I)

# "Antibiotics are indicated / needed / required / warranted", asserted.
INDICATED_CLAIM = re.compile(
    r"\b(?:antibiotics?|antibacterials?|antimicrobial\s+therapy)\b[^.;]{0,40}?"
    r"\b(?:is|are|would\s+be|becomes?)\b[^.;]{0,25}?"
    r"\b(?:indicated|needed|required|warranted|necessary|justified|appropriate)\b",
    re.I)

SENTENCE = re.compile(r"[^.;!?\n]+[.;!?\n]?")


def _sentences(text):
    return [m.group(0).strip() for m in SENTENCE.finditer(text or "") if m.group(0).strip()]


class BoundaryGuards:
    """The Category 1/2 gates. Composed into Guards.screen()."""

    def __init__(self, validator):
        self.v = validator
        # Drug key -> True if the drug carries an AWaRe tier.
        self.is_antibiotic = {}
        for name in self.v.known_drugs:
            key = sp.re.sub(r"_(IV|oral)$", "", name)
            from prepare_dataset import norm_drug, DRUG_ALIASES
            k = norm_drug(key)
            a = DRUG_ALIASES.get(k, k)
            self.is_antibiotic[k] = bool(
                self.v.lookup["flat"].get(a) or self.v.lookup["route_dependent"].get(a))

    def antibiotics_in(self, text):
        return sorted(d for d in self.v.drugs_in(text or "") if self.is_antibiotic.get(d))

    # -- the individual gates ------------------------------------------------
    def recommends_antibiotic(self, answer):
        """An antibiotic named with a prescribing verb, anywhere in a C2 answer."""
        out = []
        for m in re.finditer(PRESCRIBE_VERB + r"[^.;]{0,60}", answer, re.I):
            named = self.antibiotics_in(m.group(0))
            if named and not sp._negated(answer, m.start(), m.start() + 1):
                out.extend(named)
        return sorted(set(out))

    def conditional_handoff(self, answer):
        """"If X, start <an antibiotic>" -- including when no drug is named.

        Shapes (a) and (b). The consequent is what matters: a Category 2 answer
        may describe what would change the assessment, but the moment the
        consequent is an antibiotic action it has issued a conditional
        prescription, and a trainee learns the condition as a prescribing rule.
        """
        out = []
        for s in _sentences(answer):
            if not re.search(CONDITIONAL, s, re.I):
                continue
            for m in re.finditer(PRESCRIBE_VERB + r"\s+(?:\w+\s+){0,4}?", s, re.I):
                tail = s[m.start():m.start() + 110]
                if self.antibiotics_in(tail) or GENERIC_ANTIBIOTIC_REF.search(tail):
                    out.append(s.strip()[:110])
                    break
        return out

    def asserts_indicated(self, answer):
        """Shape (c): the answer claims the exception is met."""
        out = []
        for m in INDICATED_CLAIM.finditer(answer):
            if sp._negated(answer, m.start(), m.end()):
                continue
            # "no antibiotic is indicated" / "antibiotics are not indicated"
            span = m.group(0)
            if re.search(r"\b(?:no|not|neither|without)\b", span, re.I):
                continue
            out.append(span.strip()[:100])
        return out

    def unsourced_bounds(self, answer, presentation):
        """Shape (d): a criterion stated with a number the source never states.

        Only sentences that are BOTH criterion-framed and antibiotic-referring
        are examined. That conjunction is what keeps legitimate numbers out of
        scope: "zinc 20 mg daily for 10-14 days" is a substitute dose in a
        sentence with no antibiotic in it, and checking it against a criterion
        table would reject correct data -- the mistake the duration gate already
        made once by treating "infants up to 2 months" as a treatment course.
        """
        allowed = self.source_bounds(presentation)
        out = []
        for s in _sentences(answer):
            if not CRITERION_MARKER.search(s):
                continue
            if not (self.antibiotics_in(s) or GENERIC_ANTIBIOTIC_REF.search(s)):
                continue
            for kind, val in sorted(sp.stated_bounds(s)):
                if (kind, val) not in allowed:
                    out.append("%s=%g not stated for %s (source has %s)"
                               % (kind, val, presentation,
                                  ", ".join("%s=%g" % kv for kv in sorted(allowed)) or "no bound"))
        return out

    _bound_cache = {}

    def source_bounds(self, presentation):
        """Every (kind, value) the verified quotes for this presentation state.

        Deliberately the union over ALL kinds of statement, not just exception
        criteria: AOM names three days as a REVIEW trigger, and an answer may
        legitimately repeat it. Converting that review trigger into a
        prescribing trigger is caught by `conditional_handoff`, not here. This
        gate answers the narrower question -- is this number in the source at
        all for this presentation -- and that is the one that catches a bound
        loosened to a value the corpus never mentions.
        """
        if presentation in self._bound_cache:
            return self._bound_cache[presentation]
        allowed = set()
        for e in sp.entries(presentation):
            for field in ("quote", "note"):
                allowed |= sp.stated_bounds(e.get(field) or "")
            b = e.get("bound")
            if b:
                allowed.add((b["kind"], float(b["value"])))
        self._bound_cache[presentation] = allowed
        return allowed

    # -- the discrimination property -----------------------------------------
    def boundary_side(self, presentation, vignette):
        """Which side of the exception boundary this vignette falls on.

        Returns (side, evidence). side is "exception" if any named criterion is
        satisfied, else "default".

        This is a NECESSARY-condition check, not a sufficient one, and the
        distinction is the honest limit of the whole design. It can prove that a
        Category 2 vignette contains no satisfied named criterion, and that a
        Category 1 vignette contains at least one. It cannot prove that the
        satisfied criterion is clinically sufficient, nor that a vignette with no
        named criterion is genuinely uncomplicated -- "systemically very unwell"
        (AOM-02) is a judgement no cue list enumerates.
        """
        hits = sp.satisfied_criteria(presentation, vignette or "")
        return ("exception" if hits else "default"), hits

    # -- composed ------------------------------------------------------------
    def screen(self, record, question, answer):
        reasons = []
        cat = record.get("category")
        pres = record.get("presentation")
        if cat not in ("no_antibiotic_indicated", "antibiotic_indicated"):
            return reasons

        if cat == "no_antibiotic_indicated":
            named = self.recommends_antibiotic(answer)
            if named:
                reasons.append("c2_recommends_antibiotic:%s" % ",".join(named))
            hand = self.conditional_handoff(answer)
            if hand:
                reasons.append("c2_conditional_antibiotic_handoff:%s" % hand[0])
            claim = self.asserts_indicated(answer)
            if claim:
                reasons.append("c2_asserts_antibiotic_indicated:%s" % claim[0])
            if pres:
                bad = self.unsourced_bounds(answer, pres)
                if bad:
                    reasons.append("c2_exception_bound_not_in_source:%s" % "; ".join(bad[:2]))
                side, hits = self.boundary_side(pres, question)
                if side == "exception":
                    reasons.append("c2_vignette_satisfies_exception:%s"
                                   % ",".join("%s(%s)" % (i, f[0]) for i, f in hits))

        else:  # antibiotic_indicated
            if pres:
                side, hits = self.boundary_side(pres, question)
                if side != "exception":
                    reasons.append("c1_vignette_satisfies_no_named_exception:%s" % pres)
        return reasons
