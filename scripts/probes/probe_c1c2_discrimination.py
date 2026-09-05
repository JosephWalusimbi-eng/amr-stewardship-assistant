"""Can the gate actually tell the two sides of the boundary apart?

The interleaved design pairs a Category 1 and a Category 2 vignette on the SAME
chunk, so the only difference between them is which side of the exception
criteria the patient falls on. That is only worth doing if the gate can assert
the discrimination -- otherwise the ratio is balanced by assertion, not by check.

So this probe tests the property directly, and it tests it adversarially, because
the interesting failure is not "the gate misses a criterion". It is the opposite:
a Category 2 vignette DESCRIBES the criteria in order to rule them out ("no blood
in the stool", "afebrile", "the left ear is normal"), and a cue matcher without
polarity reads every one of those as the criterion being present. A Category 2
vignette is the densest possible source of criterion vocabulary. That is the same
shape as the meningitidis/meningitis collision: the cue is there, its polarity is
not.

The naive matcher below is the obvious implementation -- substring cues, any one
limb satisfies, no negation handling.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scenario_pool as sp  # noqa: E402


def naive_side(presentation, vignette):
    """Any cue substring, any single limb, no negation. Returns fired criterion ids."""
    hits = []
    for e in sp.entries(presentation, "exception"):
        for pat in e.get("cues", ()):
            if re.search(pat, vignette, re.I):
                hits.append(e["id"])
                break
    return hits


def fixed_side(presentation, vignette):
    return [i for i, _f in sp.satisfied_criteria(presentation, vignette)]


# Interleaved pairs. Each pair shares one chunk; the C2 member deliberately
# states the criteria in the negative, which is how a real clinician writes one.
PAIRS = [
    dict(presentation="copd_exacerbation",
         chunk="copd_exacerbation::who_aware_book_2022::p178-184::2",
         c1="A 66-year-old woman with COPD is admitted to hospital with a severe "
            "exacerbation. She has an increased volume of purulent sputum and is "
            "breathless at rest. Temperature 38.4 C.",
         c2="A 62-year-old man with COPD attends a health centre III with three days of "
            "worsening breathlessness and cough. Sputum volume is unchanged and it is not "
            "purulent. He is afebrile at 37.1 C and is managed at home."),
    dict(presentation="acute_otitis_media",
         chunk="acute_otitis_media::who_aware_book_2022::p52-61::3",
         c1="An 18-month-old boy has bilateral acute otitis media. His temperature is "
            "39.4 C and he is systemically very unwell.",
         c2="A 4-year-old girl has right ear pain and a temperature of 38.2 C. She is alert "
            "and feeding normally, the pain settles with paracetamol, and the infection is "
            "not bilateral -- the left ear is normal."),
    dict(presentation="acute_sinusitis",
         chunk="acute_sinusitis::who_aware_book_2022::p77-87::1",
         c1="A 41-year-old man has had facial pain, nasal congestion and purulent nasal "
            "discharge persisting without improvement for 12 days.",
         c2="A 34-year-old woman has had nasal congestion, facial pressure and yellow nasal "
            "discharge for six days and is slowly improving. She is afebrile and has had no "
            "worsening after an initial mild phase."),
    dict(presentation="pharyngitis",
         chunk="pharyngitis::who_aware_book_2022::p62-76::1",
         c1="A 9-year-old girl has a sore throat with tonsillar exudates, tender anterior "
            "cervical lymphadenitis, a temperature of 38.6 C and no cough -- a Centor score "
            "of 4 -- in a district with a high risk of rheumatic fever.",
         c2="A 26-year-old man has a sore throat, hoarseness, a runny nose and a cough. "
            "There are no tonsillar exudates and no tender cervical lymphadenitis. He is "
            "afebrile."),
    dict(presentation="acute_diarrhoea",
         chunk="acute_diarrhoea::who_aware_book_2022::p185-200::1",
         c1="A 30-year-old man has severe bloody diarrhoea with visible blood and mucus in "
            "the stool for two days, with fever and abdominal cramps.",
         c2="A 3-year-old boy has passed six loose watery stools today. There is no blood in "
            "the stool and no dysentery. He is drinking eagerly, is not immunocompromised, "
            "and there is no cholera outbreak in the district."),
    # Recorded even though it cannot be a same-chunk pair -- see the report below.
    dict(presentation="acute_bronchitis",
         chunk="SPANS TWO DOCUMENTS",
         c1="A 58-year-old smoker with acute bronchitis has a temperature of 39.0 C, "
            "dyspnoea and purulent expectorations, on a background of severe anaemia.",
         c2="A 29-year-old woman has had a cough for nine days with scanty mucoid sputum "
            "and mild chest tightness. She is afebrile, has no dyspnoea and no purulent "
            "expectorations, and her respiratory rate is normal."),
    # BRO-05 names six comorbidities in ONE criterion, so a Category 2 bronchitis
    # vignette has to rule out all six, and it does that the way a clinician
    # writes it -- as one negated list. This case exists because the first
    # stubbed generation run produced exactly this sentence and the gate rejected
    # a correct answer on the second item in the list. The C1 member is the
    # control: the same comorbidity, asserted rather than ruled out, must still
    # fire.
    dict(presentation="acute_bronchitis",
         chunk="SPANS TWO DOCUMENTS",
         c1="A 70-year-old man with cardiac disease has had a cough for eight days, and "
            "now has dyspnoea and purulent expectorations at 38.9 C.",
         c2="A 34-year-old market trader has had a dry cough for six days after a cold. He "
            "is afebrile at 37.2 C, has no dyspnoea, his sputum is unchanged and not "
            "purulent, and he has no malnutrition, measles, rickets, severe anaemia or "
            "cardiac disease."),
]


def run():
    print("=" * 96)
    print("DISCRIMINATION PROPERTY: C1 must satisfy >=1 named criterion; C2 must satisfy none")
    print("=" * 96)
    naive_bad = fixed_bad = 0
    for p in PAIRS:
        pres = p["presentation"]
        for side, vig, want_hits in (("C1", p["c1"], True), ("C2", p["c2"], False)):
            n = naive_side(pres, vig)
            f = fixed_side(pres, vig)
            n_ok = bool(n) == want_hits
            f_ok = bool(f) == want_hits
            naive_bad += not n_ok
            fixed_bad += not f_ok
            print("%-20s %s  naive:%-4s %-26s   fixed:%-4s %s"
                  % (pres, side,
                     "OK" if n_ok else "WRONG", ",".join(n) or "-",
                     "OK" if f_ok else "WRONG", ",".join(f) or "-"))
        print()

    total = len(PAIRS) * 2
    print("-" * 96)
    print("naive cue matcher: %d of %d vignettes placed on the correct side" % (total - naive_bad, total))
    print("fixed matcher    : %d of %d vignettes placed on the correct side" % (total - fixed_bad, total))
    print()
    print("Same-chunk feasibility (from scripts/scenario_pool.py):")
    print("  5 of 6 presentations have one chunk carrying both the default and an exception.")
    print("  acute_bronchitis does NOT: AWaRe states the default and names no exception at all,")
    print("  UCG 5.2.2 states the only exception. An interleaved bronchitis pair must therefore")
    print("  span two documents, and its C1 member rests on a criterion the WHO source does not")
    print("  recognise. That is BRO-90, and it is a finding, not a defect to engineer around.")
    return 1 if fixed_bad else 0


if __name__ == "__main__":
    sys.exit(run())
