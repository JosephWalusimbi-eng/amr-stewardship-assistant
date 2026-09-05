"""Run the real boundary-pair loop against a stubbed model.

The GPU box is offline, so the 20-pair batch cannot be generated yet. This
rehearsal exercises everything between the model and the dataset: prompt build,
pair parsing, whole-pair acceptance, the source tag reaching the guard, and the
file write. The canned responses are written so that each required variant
returns one correct pair and one that commits the failure mode the ruling asked
to be checked -- a Category 2 answer that withholds the antibiotic correctly and
teaches a loosened or foreign threshold in the same breath.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import generate_boundary_pairs as g  # noqa: E402

BRO_C2_OK = dict(
    pair=1, side="not_indicated",
    question=("A 34-year-old market trader at HC3 has had a dry cough for 6 days after a "
              "cold. He is afebrile at 37.2 C, has no dyspnoea, his sputum is unchanged and "
              "not purulent, and he has no malnutrition, measles, cardiac disease or anaemia. "
              "He asks for an antibiotic."),
    answer=("No antibiotic is indicated. Antibiotic treatment is not recommended for acute "
            "bronchitis and should be avoided. His sputum is not purulent, he has no dyspnoea "
            "and he is afebrile, so none of the features of secondary bacterial infection are "
            "present. Yellow/green colour of the sputum does not indicate bacterial infection "
            "and the need for antibiotics. Give Paracetamol 1 g every 4-6 hours (max: 4 g "
            "daily) and fluids. Cough usually persists for 10-20 days and it can last longer, "
            "so tell him that is expected. Ask him to come back for reassessment if he becomes "
            "clinically unwell with systemic signs of infection: fever, increased heart rate, "
            "increased respiratory rate or focal chest signs."))
BRO_C1_OK = dict(
    pair=1, side="indicated",
    question=("A 71-year-old woman at HC4 has had a cough for 8 days. Today her temperature is "
              "38.9 C, she has dyspnoea at rest and purulent expectorations."),
    answer=("An antibiotic is indicated. She meets the criterion for secondary bacterial "
            "infection: fever > 38.5degC, dyspnoea, purulent expectorations, and she is "
            "elderly. Give Amoxicillin 500 mg every 8 hours. Amoxicillin is in the Access "
            "group. Review in 48-72 hours; if she deteriorates or does not improve, refer."))
BRO_C2_BAD = dict(
    pair=2, side="not_indicated",
    question=("A 29-year-old teacher at HC2 has had a cough for 5 days. She is afebrile at "
              "37.4 C, has no dyspnoea and her sputum is unchanged and not purulent."),
    answer=("No antibiotic is indicated today. Antibiotic treatment is not recommended for "
            "acute bronchitis and should be avoided. Give Paracetamol 1 g every 4-6 hours "
            "(max: 4 g daily). Consider an antibiotic if her temperature rises above 38 C or "
            "the cough persists for more than 3 days."))
BRO_C1_BAD = dict(
    pair=2, side="indicated",
    question=("A 64-year-old man with severe anaemia has a cough, dyspnoea and purulent "
              "expectorations at 38.7 C."),
    answer=("An antibiotic is indicated: fever > 38.5degC, dyspnoea and purulent "
            "expectorations, in a patient with severe anaemia. Give Amoxicillin 500 mg every "
            "8 hours. Amoxicillin is in the Access group."))

CPD_AW_C2_OK = dict(
    pair=1, side="not_indicated",
    question=("A 58-year-old man with known COPD comes to HC3 with 2 days of increased "
              "breathlessness. He is not hospitalised and does not require hospital admission, "
              "this is not a severe exacerbation, and his sputum volume and colour are "
              "unchanged and not purulent."),
    answer=("No antibiotic is indicated. Antibiotic treatment is not required in the great "
            "majority of cases of acute exacerbations of chronic obstructive pulmonary "
            "disease, and he is not hospitalised, does not require admission, and has no "
            "increased volume or purulence of sputum. The core treatment of an exacerbation "
            "episode consists of supplementary oxygen and short-acting inhaled beta-2-agonists "
            "(with or without anticholinergics): give Inhaled salbutamol 2 puffs 2-4 times a "
            "day. Send him back at once if his pulse goes above 100 beats per minute or his "
            "breathing above 30 breaths per minute."))
CPD_AW_C1_OK = dict(
    pair=1, side="indicated",
    question=("A 66-year-old woman with COPD is admitted to hospital with a severe "
              "exacerbation. Her sputum has increased in volume and is purulent."),
    answer=("An antibiotic is indicated. She is hospitalized because of an acute exacerbation "
            "of COPD with an increased volume and purulence of sputum, which is the criterion "
            "this source names. Give Amoxicillin+clavulanic acid (oral): 500 mg + 125 mg given "
            "every 8 hours. It is in the Access group. Reassess at 48-72 hours."))
CPD_AW_C2_BAD = dict(
    pair=2, side="not_indicated",
    question=("A 61-year-old man with COPD at HC3 is breathless. He is not hospitalised, does "
              "not require admission, and his sputum is unchanged and not purulent."),
    answer=("No antibiotic is indicated at present. Antibiotic treatment is not required in "
            "the great majority of cases of acute exacerbations of chronic obstructive "
            "pulmonary disease. Give Inhaled salbutamol 2 puffs 2-4 times a day. Start an "
            "antibiotic if his temperature goes above 38 C or his sputum turns green."))
CPD_AW_C1_BAD = dict(
    pair=2, side="indicated",
    question=("A 59-year-old man with COPD has more sputum, changed to green, rapid breathing "
              "and a temperature of 38.6 C. He is managed at HC3 and is not admitted."),
    answer=("An antibiotic is indicated because his sputum has changed and he is febrile. Give "
            "Amoxicillin+clavulanic acid (oral): 500 mg + 125 mg given every 8 hours."))

CPD_UG_C2_OK = dict(
    pair=1, side="not_indicated",
    question=("A 54-year-old woman with COPD at HC3 has 3 days of worse breathlessness. Her "
              "sputum is unchanged in amount and colour and is not purulent, she is afebrile "
              "at 37.4 C, and her breathing is not rapid."),
    answer=("No antibiotic is indicated. Her sputum is not increased or changed in colour, she "
            "is afebrile and her breathing is not rapid, so the criteria for antibiotics in an "
            "exacerbation are not met. Give Inhaled salbutamol 2 puffs 2-4 times a day. Come "
            "back immediately if the pulse goes above 100 beats per minute or breathing above "
            "30 breaths per minute."))
CPD_UG_C1_OK = dict(
    pair=1, side="indicated",
    question=("A 63-year-old man with COPD at HC4 has more sputum, changed to green, rapid "
              "breathing at 32 breaths per minute and a temperature of 38.6 C."),
    answer=("An antibiotic is indicated. He has more sputum, changed to more yellow/green "
            "coloured, with rapid breathing and temp >38degC. Give Amoxicillin+clavulanic acid "
            "(oral): 500 mg + 125 mg given every 8 hours. It is in the Access group. Reassess "
            "in 48-72 hours and refer if he worsens."))

CANNED = {
    ("acute bronchitis", "uganda"): [BRO_C2_OK, BRO_C1_OK, BRO_C2_BAD, BRO_C1_BAD],
    ("copd exacerbation", "who"): [CPD_AW_C2_OK, CPD_AW_C1_OK, CPD_AW_C2_BAD, CPD_AW_C1_BAD],
    ("copd exacerbation", "uganda"): [CPD_UG_C2_OK, CPD_UG_C1_OK],
}


def stub_call(session, base_url, model, prompt, temperature, max_tokens, timeout):
    pres = re.search(r"Presentation: (.+)", prompt).group(1).strip()
    src = re.search(r"Criteria source: (.+)", prompt).group(1).strip()
    key = (pres, "uganda" if src.startswith("uganda") else "who")
    items = CANNED.get(key)
    if items is None:
        raise g.CallFailed("no canned response for %s" % (key,))
    return json.dumps(items)


g.call_model = stub_call
sys.argv = ["rehearse", "--model", "stub", "--target", "3", "--pairs-per-call", "2",
            "--grounded-fraction", "0", "--max-stalled-calls", "3",
            "--out", ".scratch/rehearsal_pairs.jsonl",
            "--discarded", ".scratch/rehearsal_discarded.jsonl",
            "--require-variant", "acute_bronchitis:uganda_clinical_guidelines_2023",
            "--require-variant", "copd_exacerbation:who_aware_book_2022",
            "--require-variant", "copd_exacerbation:uganda_clinical_guidelines_2023"]
rc = g.main()

print("\n" + "=" * 92)
for line in open(".scratch/rehearsal_discarded.jsonl", encoding="utf-8"):
    d = json.loads(line)
    print("REJECTED  %s" % d["variant"])
    for r in d["reasons"]:
        print("     %s" % r)
print("=" * 92)
for line in open(".scratch/rehearsal_pairs.jsonl", encoding="utf-8"):
    r = json.loads(line)
    print("KEPT  %-14s %-32s cross=%-5s pair=%s"
          % (r["pair_side"], r["criteria_source"], r["cross_source"], r["pair_id"]))
raise SystemExit(rc)
