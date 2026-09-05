"""Deliberately try to smuggle a prescribing rule through a Category 2 answer.

Same method as the dose-indication probe: build the gate we would plausibly have
shipped, run the adversarial cases against it, and only then run the fix. A gate
that has only ever seen a clean batch has not been shown to work.

The naive gate is not a straw man. It is exactly what the generation spec calls
for -- amr_dataset_generation_prompts.md, Category 2: "The validator blocks any
Category 2 answer that names an antibiotic in a recommending frame." Implemented
literally, against Validator.known_drugs, which is the drug list every other gate
in this pipeline already uses.

Every vignette below is on the Category 2 side of its boundary: the named
exception criteria are NOT satisfied. Any answer that hands over a prescribing
rule is therefore teaching a threshold the source did not set.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_dataset import Validator                     # noqa: E402
from c2_exception_guard import BoundaryGuards, PRESCRIBE_VERB  # noqa: E402

V = Validator()
FIXED = BoundaryGuards(V)


def naive_gate(answer):
    """What the spec literally asks for: an antibiotic name in a recommending frame.

    Uses Validator.known_drugs, like every other gate in the pipeline.
    """
    hits = []
    for m in re.finditer(PRESCRIBE_VERB + r"[^.;]{0,60}", answer, re.I):
        hits.extend(V.drugs_in(m.group(0)).keys())
    return sorted(set(hits))


# ---------------------------------------------------------------- vignettes
COPD_C2 = (
    "A 62-year-old man with known COPD presents to a health centre III with three days of "
    "worsening breathlessness and cough. Sputum volume is unchanged and it remains clear. "
    "He is afebrile at 37.1 C, pulse 88, respiratory rate 22. He is managed at home.")

SIN_C2 = (
    "A 34-year-old woman has had nasal congestion, facial pressure and yellow nasal "
    "discharge for six days. She reports she is slowly improving. She is afebrile.")

AOM_C2 = (
    "A 4-year-old girl has pain in her right ear and a temperature of 38.2 C. She is "
    "alert, feeding normally, and the pain settles with paracetamol. The left ear is normal.")

CASES = [
    # name, vignette, presentation, answer, should_fire
    ("CORRECT C2 -- COPD, full symptomatic management named",
     COPD_C2, "copd_exacerbation",
     "No antibiotic is indicated. Most COPD exacerbations are not caused by bacterial "
     "infection, and this man has no increase in sputum volume or purulence and is not "
     "severe enough to require hospital admission. Manage with inhaled salbutamol 2 puffs "
     "2-4 times a day and oral prednisolone 40 mg once daily in the morning for 5 days. "
     "Refer urgently if his pulse rises above 100 beats per minute, his breathing above 30 "
     "breaths per minute, his lips or tongue turn blue, he becomes confused, or he fails to improve.",
     False),

    ("CORRECT C2 -- sinusitis, quotes the real 10-day bound",
     SIN_C2, "acute_sinusitis",
     "No antibiotic is indicated. Most sinusitis follows a viral upper respiratory infection "
     "and is self-limiting, and yellow or green nasal discharge alone is not a sign of "
     "bacterial infection. She is under ten days from onset and improving, so watchful waiting "
     "with symptom relief is the recommended approach: analgesia and antipyretics, saline "
     "nasal irrigation, and topical intranasal decongestants. Bacterial sinusitis would only "
     "be suspected if signs and symptoms persisted without improvement for more than 10 days, "
     "or if she became significantly worse after this initial mild phase.",
     False),

    ("SMUGGLE (a) -- conditional hand-off, no drug named",
     COPD_C2, "copd_exacerbation",
     "No antibiotic is indicated at present. His sputum is clear and unchanged in volume and "
     "he does not need admission. Give supplementary oxygen if needed and a short-acting "
     "inhaled beta-2-agonist. If the sputum becomes purulent and increases in volume, begin "
     "the first-choice oral agent for 5 days.",
     True),

    ("SMUGGLE (b) -- drug class instead of a drug name",
     SIN_C2, "acute_sinusitis",
     "No antibiotic is needed today; she is improving and yellow discharge alone does not "
     "indicate bacterial infection. Use analgesia and saline nasal irrigation. Should her "
     "symptoms fail to settle, prescribe an aminopenicillin.",
     True),

    ("SMUGGLE (c) -- asserts the exception is met when the vignette lacks it",
     AOM_C2, "acute_otitis_media",
     "This child has acute otitis media. Given the severity of her symptoms an antibiotic is "
     "indicated here, though most cases resolve without one. Continue paracetamol for pain "
     "and review her in three days.",
     True),

    ("SMUGGLE (d) -- real criterion, loosened bound: sinusitis 3 days for 10",
     SIN_C2, "acute_sinusitis",
     "No antibiotic is indicated. Sinusitis is usually viral and self-limiting, and the colour "
     "of the discharge does not indicate bacterial infection. Manage with analgesia and saline "
     "nasal irrigation. Consider an antibiotic if her symptoms persist without improvement for "
     "more than 3 days.",
     True),

    ("SMUGGLE (d) -- real criterion, loosened bound: otitis under 5 for under 2",
     AOM_C2, "acute_otitis_media",
     "No antibiotic is indicated. Most non-severe acute otitis media resolves on its own and "
     "watchful waiting with analgesia is appropriate. Antibiotics are considered for bilateral "
     "otitis media in children under 5 years, for immunocompromised children, and for severe "
     "symptoms. Reassess if she is no better in three days.",
     True),

    ("OVERT -- names the antibiotic outright (control: the naive gate must catch this)",
     COPD_C2, "copd_exacerbation",
     "No antibiotic is strictly needed, but given his age it is reasonable to start amoxicillin "
     "500 mg every 8 hours for 5 days alongside his inhaler.",
     True),
]


def run():
    print("=" * 94)
    print("NAIVE GATE -- 'a Category 2 answer must not name an antibiotic in a recommending frame'")
    print("=" * 94)
    naive_missed, naive_false = [], []
    for name, vig, pres, ans, should in CASES:
        hit = naive_gate(ans)
        fired = bool(hit)
        ok = fired == should
        if should and not fired:
            naive_missed.append(name)
        if not should and fired:
            naive_false.append((name, hit))
        print("%-4s %-62s expected=%-5s fired=%-5s %s"
              % ("PASS" if ok else "FAIL", name[:62], should, fired,
                 ",".join(hit) if hit else ""))

    print()
    print("=" * 94)
    print("FIXED GATE -- scenario-pool-checked boundary gates")
    print("=" * 94)
    fixed_bad = 0
    for name, vig, pres, ans, should in CASES:
        rec = {"category": "no_antibiotic_indicated", "presentation": pres}
        reasons = FIXED.screen(rec, vig, ans)
        fired = bool(reasons)
        ok = fired == should
        fixed_bad += not ok
        print("%-4s %-62s expected=%-5s fired=%-5s"
              % ("PASS" if ok else "FAIL", name[:62], should, fired))
        for r in reasons:
            print("        -> %s" % r[:150])

    print()
    print("-" * 94)
    print("naive gate: missed %d of %d smuggles" % (len(naive_missed), sum(1 for c in CASES if c[4])))
    for n in naive_missed:
        print("   MISSED  %s" % n)
    print("naive gate: %d false rejection(s) of correct Category 2 answers" % len(naive_false))
    for n, h in naive_false:
        print("   WRONGLY REJECTED  %s  (matched %s)" % (n, ",".join(h)))
    print("fixed gate: %d of %d cases behaved as expected" % (len(CASES) - fixed_bad, len(CASES)))
    return 1 if fixed_bad else 0


if __name__ == "__main__":
    sys.exit(run())
