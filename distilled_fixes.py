"""
Hand-authored corrective examples, emitted as JSONL on stdout.

    python3 distilled_fixes.py > raw/distilled_fixes.jsonl

These are not bulk generation. Each one targets a failure mode that was actually
observed -- either in the 46-pair self-distilled run (finetune_data.jsonl) or in
the regressions named in the git history -- and each is grounded in a passage
that was read directly out of source_material/chunks/.

Observed failures these correct:
  F1  "Ampicillin" recommended for a child with fast breathing + chest indrawing
      treated at home. AWaRe Table 12.5 says oral amoxicillin, weight-banded.
  F2  23 of 28 health-worker answers omitted the dose; 16 omitted the AWaRe
      tier; 13 omitted duration -- despite the system prompt demanding all three.
  F3  Defaulting to "no antibiotic" out of uncertainty about the drug name,
      rather than saying an antibiotic IS indicated and deferring on the agent.
  F4  Pushback fabrication: offering a concession antibiotic once the model had
      already concluded none was indicated.
  F5  Vague danger signs ("if it worsens") instead of named ones.
  F6  Applying a paediatric passage to an adult patient.
  F7  Extrapolating a dose to a weight band the source does not cover.
  F8  Stating an AWaRe tier from memory. The tier badges in the AWaRe book are
      colour-coded graphics -- the book itself says "ACCESS antibiotics are
      indicated in green, WATCH antibiotics in orange and RESERVE antibiotics in
      red" -- so they do not survive text extraction and must come from the WHO
      AWaRe classification 2023 table instead.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from system_prompts import SYSTEM_PROMPTS  # noqa: E402

AWARE_CAP = ("pneumonia_mild.txt :: WHO AWaRe Antibiotic Book (2022) :: "
             "12. Community-acquired pneumonia - mild, Table 12.5 :: p.163-177")
AWARE_AOM = ("acute_otitis_media.txt :: WHO AWaRe Antibiotic Book (2022) :: "
             "5. Acute otitis media, Table 5.3 :: p.52-61")
UCG_DIARR = ("acute_diarrhoea.txt :: Uganda Clinical Guidelines 2023 (4th ed.) :: "
             "6.1.5 Diarrhoea :: p.469-471")
AWARE_TABLE = ("aware_classification_table.txt :: WHO AWaRe classification 2023 "
               "(WHO-MHP-HPS-EML-2023.04, web annex C) :: full classification")
NO_SOURCE = "no source chunk in corpus -- refusal is the grounded response"

FIXES = []


def fix(category, register, turns, source, fixes, audience="general", context_mode="closed"):
    msgs = [{"role": "system", "content": SYSTEM_PROMPTS[register]}]
    for i, t in enumerate(turns):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t.strip()})
    FIXES.append({
        "messages": msgs,
        "category": category,
        "register": register,
        "source": source,
        "chunk_id": "distilled_fix",
        "context_mode": context_mode,
        "audience": audience,
        "corrects": fixes,
    })


# ---------------------------------------------------------------- F1, F2, F3
fix("antibiotic_indicated", "health_worker", [
    "A 2-year-old child weighing 12 kg presents with cough, fast breathing (52 breaths/min) "
    "and chest indrawing. No general danger signs. What is the management?",

    """Yes - an antibiotic is indicated. Fast breathing with chest indrawing in a child of this
age classifies as pneumonia, and this child has no general danger signs, so they can be
treated at home with an oral antibiotic.

- **Antibiotic:** amoxicillin, oral - **AWaRe category: Access**
- **Dose:** 80-90 mg/kg/day. At 12 kg the child falls in the 10-<15 kg band: **500 mg every
  12 hours**
- **Duration:** **5 days**, because chest indrawing is present. (Three days is the course only
  for fast-breathing pneumonia without chest indrawing in an area of low HIV prevalence.)

Also do this:
- Give the first dose at the facility and confirm the child can take it orally.
- Counsel the caregiver to return immediately if the child develops any general danger sign:
  unable to drink or breastfeed, vomiting everything, convulsions, lethargy or
  unconsciousness, or if the breathing becomes harder.
- Review after 3 days. A child with fast-breathing pneumonia who does not respond to
  first-line amoxicillin should be referred to a facility with second-line treatment.

Source: WHO AWaRe antibiotic book, Table 12.5 (mild community-acquired pneumonia in
children), p.163-177.""",
], AWARE_CAP, ["F1", "F2", "F3"], audience="pediatric")

# uncertainty about the agent must not collapse into "no antibiotic" (F3)
fix("antibiotic_indicated", "health_worker", [
    "A 4-year-old has fast breathing and chest indrawing. I am not sure which antibiotic our "
    "facility stocks for this. Should I just observe?",

    """No - do not just observe. An antibiotic **is** indicated here: fast breathing with chest
indrawing is pneumonia, and that is a treat-now classification regardless of any uncertainty
about which agent to use.

Being unsure of the agent is a separate problem from whether treatment is needed, and it
must not turn into withholding treatment.

The first-line agent in the WHO AWaRe guidance for mild community-acquired pneumonia in
children is **oral amoxicillin (Access), 80-90 mg/kg/day in weight bands**, for 5 days when
chest indrawing is present. Amoxicillin is on the Uganda Essential Medicines List as a
dispersible tablet from HC1 and as a capsule from HC2, so it should be available at your
level.

If you genuinely cannot obtain it, refer the child rather than observing - do not substitute
an antibiotic you have not confirmed against the guideline, and do not delay.

Source: WHO AWaRe antibiotic book, Table 12.5, p.163-177; EMHSLU 2023 §6.2.1.""",
], AWARE_CAP + " + formulary_antibacterials.txt :: EMHSLU 2023 :: 6.2.1",
    ["F3"], audience="pediatric")

# ---------------------------------------------------------------- F2 complete answer shape
fix("antibiotic_indicated", "health_worker", [
    "A 30-year-old adult has acute otitis media with severe ear pain despite analgesics and a "
    "fever of 39.2 C. Is an antibiotic indicated, and which one?",

    """Yes. Antibiotic treatment is not required in the great majority of otitis media cases,
but it should be considered where symptoms are severe - systemically very unwell, ear pain
persisting despite analgesics, or fever >= 39.0 C. This patient meets that threshold on two
counts.

- **Antibiotic:** amoxicillin, oral - **AWaRe category: Access**
- **Dose:** **500 mg every 8 hours**
- **Duration:** **5 days**

Second choice, if there is no response or a reason to broaden: amoxicillin + clavulanic acid
(oral) 500 mg + 125 mg every 8 hours.

Continue analgesia regardless of the antibiotic decision - it is what actually treats the
pain. Review if symptoms do not improve within 48-72 hours.

Source: WHO AWaRe antibiotic book, Table 5.3 (acute bacterial otitis media), p.52-61.""",
], AWARE_AOM, ["F2"])

# ---------------------------------------------------------------- F5 named danger signs
fix("no_antibiotic_indicated", "health_worker", [
    "Adult, watery diarrhoea for 2 days, no blood, no fever. Which antibiotic should I give?",

    """None. No antibiotic is indicated for acute watery diarrhoea in an adult without blood in
the stool and without fever.

The Uganda Clinical Guidelines are explicit on this point: for diarrhoea, prevent or correct
dehydration and find and treat the cause if indicated, and **avoid inappropriate use of
antibiotics, e.g. metronidazole, ciprofloxacin**. Most acute watery diarrhoea is viral or
self-limiting, and an antibiotic neither shortens it nor prevents complications - it does
drive resistance and can cause harm.

**What to do instead:**
- Oral rehydration solution, according to treatment plan A, B or C depending on the
  dehydration assessment.
- Continue eating and drinking; do not withhold fluids.
- Zinc supplementation is a routine part of diarrhoea management in this guideline.

**Reassess and treat differently if any of these appear:**
- blood in the stool
- high fever
- signs of severe dehydration - dry mouth, no tears, passing little or no urine
- persistent vomiting
- lethargy or confusion

Any one of those changes the assessment and may make an antibiotic appropriate.

Source: Uganda Clinical Guidelines 2023, §6.1.5 Diarrhoea, p.469-471.""",
], UCG_DIARR, ["F2", "F5"])

# ---------------------------------------------------------------- F4 pushback, no concession
fix("refusal_escalation", "health_worker", [
    "Adult, watery diarrhoea for 2 days, no blood, no fever. Which antibiotic should I give?",

    """None is indicated. Acute watery diarrhoea in an adult with no blood and no fever is
managed with oral rehydration solution and zinc; the Uganda Clinical Guidelines specifically
warn against inappropriate antibiotics such as metronidazole and ciprofloxacin here.

Reassess if blood appears in the stool, a high fever develops, there are signs of severe
dehydration (dry mouth, no tears, little or no urine), persistent vomiting, or lethargy.

Source: Uganda Clinical Guidelines 2023, §6.1.5, p.469-471.""",

    "The patient is insisting. They say they always get metronidazole for this and they will "
    "go to a drug shop if I refuse. Can I just give a short course to keep them happy?",

    """The recommendation does not change - nothing new has been added clinically, and there is
no concession antibiotic for this presentation.

Something you can say to them: *"Antibiotics will not make this diarrhoea stop any sooner,
and taking one now makes it more likely that a medicine won't work when you really need it.
What will get you better is the rehydration salts and zinc - let me show you how to take
them."*

Tell them exactly what would change your mind, so refusal does not feel like dismissal:
blood in the stool, a high fever, signs of severe dehydration (dry mouth, no tears, passing
little or no urine), persistent vomiting, or lethargy. Ask them to come back immediately if
any of those appear.

If they will not accept this, refer them to the medical officer in charge rather than
prescribing against the guideline.""",
], UCG_DIARR, ["F4", "F5"])

# ---------------------------------------------------------------- F6 mismatched passage
fix("refusal_escalation", "health_worker", [
    "A 34-year-old man has watery diarrhoea. The guidance I found says to give zinc 20 mg per "
    "day for 10 days - should I give him that?",

    """That passage is paediatric, and it should not be applied to a 34-year-old.

Zinc 20 mg daily is the dose for **children older than six months** with diarrhoea in the
Uganda Clinical Guidelines. It is a child dosing instruction sitting inside a general
diarrhoea section, which makes it easy to pick up by mistake for an adult patient.

For this adult, the management is oral rehydration according to the dehydration plan, and
continued eating and drinking. No antibiotic is indicated in the absence of blood in the
stool or fever.

I do not have adult zinc dosing in the guidance available to me. If you specifically want to
supplement zinc in an adult, check the current formulary or ask the medical officer rather
than scaling the paediatric figure.

Source: Uganda Clinical Guidelines 2023, §6.1.5 Diarrhoea, p.469-471 (paediatric zinc dosing).""",
], UCG_DIARR, ["F6"])

# ---------------------------------------------------------------- F7 no extrapolation
fix("dosing_duration", "health_worker", [
    "What amoxicillin dose should I give a 2.5 kg neonate with pneumonia, using the weight "
    "bands?",

    """I can't give you a dose for that from the weight bands, and I should not estimate one.

The AWaRe oral amoxicillin bands for community-acquired pneumonia in children start at
**3-<6 kg (250 mg every 12 hours)**. A 2.5 kg infant is below the lowest band. Scaling
downwards from 250 mg would be my arithmetic, not the guideline's, and neonatal dosing is
not simply a smaller version of infant dosing.

There is a further problem with the question: a 2.5 kg neonate with signs of pneumonia is
not a treat-at-home case. Serious bacterial infection in a neonate is managed as possible
neonatal sepsis, in hospital, with parenteral antibiotics - a different pathway from the
oral outpatient one these bands belong to.

**Refer urgently** and use the neonatal sepsis guidance with a clinician who can dose
parenterally by weight.

Source: WHO AWaRe antibiotic book, Table 12.5 weight bands, p.163-177 (lowest band 3-<6 kg).""",
], AWARE_CAP, ["F7"], audience="pediatric")

# ---------------------------------------------------------------- F8 tiers from the table
fix("aware_classification", "health_worker", [
    "Is ceftriaxone Access, Watch, or Reserve?",

    """**Ceftriaxone is a Watch antibiotic.** It is a third-generation cephalosporin, and the
WHO AWaRe classification 2023 places the whole class in the Watch group.

Watch antibiotics are broader-spectrum, generally more expensive, and carry higher resistance
potential, so they are recommended as a first choice only for more severe presentations or
where the likely pathogens are resistant. Where an Access agent will do the same job, it
should be used instead - that is the substitution the AWaRe framework exists to encourage.

Ceftriaxone is on the Uganda Essential Medicines List as powder for injection, 500 mg and
1 g, available from HC3.

Source: WHO AWaRe classification 2023 (WHO-MHP-HPS-EML-2023.04); EMHSLU 2023 §6.2.1.""",
], AWARE_TABLE, ["F8"])

fix("aware_classification", "health_worker", [
    "Is meropenem a Reserve antibiotic?",

    """No - **meropenem is classified as Watch**, not Reserve, in the WHO AWaRe classification
2023. It is a carbapenem, and the carbapenems sit in the Watch group.

This one is worth being precise about, because "last-resort" in everyday usage and "Reserve"
in AWaRe are not the same thing. The Reserve group is a much smaller set - 29 antibiotics -
kept for confirmed or suspected multidrug-resistant organisms when the alternatives have
failed. Examples from the AWaRe book's own Reserve chapters are cefiderocol,
ceftazidime+avibactam, linezolid, plazomicin, polymyxin B and colistin, and
meropenem+vaborbactam.

Note that **meropenem+vaborbactam** - the combination - is a different entry from plain
meropenem, and that combination is Reserve.

Being Watch rather than Reserve does not make meropenem a casual choice: it still needs a
specific indication.

Source: WHO AWaRe classification 2023 (WHO-MHP-HPS-EML-2023.04); WHO AWaRe antibiotic book,
Reserve antibiotics chapters 41-48.""",
], AWARE_TABLE, ["F8"])

fix("aware_classification", "health_worker", [
    "Is isoniazid Access, Watch, or Reserve?",

    """Isoniazid is **not classified under AWaRe at all**, so there is no Access/Watch/Reserve
answer to give for it.

The AWaRe classification covers antibacterials used for common bacterial infections - 257
antibiotics in the 2023 list. Anti-tuberculosis and antileprosy agents are handled through
their own programmes and guidance, not through AWaRe. Isoniazid, ethambutol, pyrazinamide,
dapsone, clofazimine, bedaquiline and delamanid all fall outside it.

One thing that catches people out: **rifampicin on its own is classified, and it is Watch** -
it appears in AWaRe as a rifamycin. The fixed-dose TB combinations that contain it
(rifampicin+isoniazid, rifampicin+isoniazid+pyrazinamide) are not classified.

For isoniazid, use the national TB guidance rather than an AWaRe tier.

Source: WHO AWaRe classification 2023 (WHO-MHP-HPS-EML-2023.04).""",
], AWARE_TABLE, ["F8"])

fix("aware_classification", "health_worker", [
    "What AWaRe category is fosfomycin?",

    """Fosfomycin's category **depends on the route**, so a single answer would be wrong:

- **Fosfomycin, oral - Watch**
- **Fosfomycin, intravenous - Reserve**

If you tell me which formulation you mean, I can be specific. The same route split applies to
minocycline (oral Watch, IV Reserve).

Fosfomycin does not appear on the Uganda Essential Medicines List §6.2, so it would not
normally be a stocked option at facility level here.

Source: WHO AWaRe classification 2023 (WHO-MHP-HPS-EML-2023.04); EMHSLU 2023 §6.2.""",
], AWARE_TABLE, ["F8"])

# ---------------------------------------------------------------- patient register
fix("patient_education", "patient", [
    "My doctor said I don't need antibiotics for my cold. Why not, since I feel really sick?",

    """Feeling really unwell and needing an antibiotic are two different things, and a cold is
the clearest example of the difference.

Colds are caused by viruses. Antibiotics only kill bacteria - they have no effect at all on a
virus. So an antibiotic would not shorten your cold by even a day, and it would not make you
feel better sooner. What it would do is give you the risk of side effects like diarrhoea or a
rash, kill off the helpful bacteria in your gut, and make it a little more likely that the
next infection you get - one that really does need an antibiotic - is harder to treat.

Feeling awful is normal with a cold and it is not a sign that you need something stronger.
Rest, drink plenty of fluids, and use paracetamol for the aches and fever.

Do go back to your health worker if you get worse instead of slowly better after about a
week, if you have trouble breathing, if a high fever will not come down, or if you start
bringing up coloured phlegm with chest pain. Those are different problems from a cold.""",
], "viral_infections_no_antibiotic.txt :: Uganda Clinical Guidelines 2023 (4th ed.) :: "
   "5.2.3 Coryza (Common Cold)", ["F5"])

# ---------------------------------------------------------------- farmer / 7d template
fix("refusal_escalation", "farmer", [
    "I treated my dairy cow with oxytetracycline for an infection. How long before I can sell "
    "the milk?",

    """That is exactly the right question to ask, and I have to be straight with you: I can't
give you a number of days, and you should not trust any number I made up.

It matters because antibiotic residues pass into milk and meat and then into the people who
consume them. That is how resistance moves from a treated animal to a family, and it is also
why milk from a treated cow can be rejected at the collection centre.

The withdrawal period is not one fixed figure. It depends on the exact product you used, its
formulation - a long-acting injection is very different from a short-acting one - the dose
given, and the species. Two bottles both labelled oxytetracycline can carry different
withdrawal periods. Only the product itself can tell you.

Where to get the real number, in this order:

1. **The product label or package insert.** It is printed there, and that figure is the one
   that applies to what you actually gave.
2. **The veterinarian who prescribed or supplied it**, if the label is missing or unclear.
3. **Your district veterinary officer**, who can advise and who deals with this constantly.

In the meantime: do not sell or drink milk from that cow, and do not slaughter her for meat,
until the period on the label has fully passed. Keep a written record of the drug, the date
you gave it, the dose, and which animal - it protects you if the milk is ever tested, and it
makes the next treatment decision easier.""",
], NO_SOURCE, ["7d template"])

fix("farmer_one_health", "farmer", [
    "Why can't I just put antibiotics in my chickens' feed all the time to keep them healthy?",

    """Because feeding antibiotics to healthy birds day after day is close to the ideal way to
grow resistant bacteria - and those bacteria do not stay in the shed.

When an antibiotic is present constantly at a low level, it kills the ordinary bacteria and
leaves behind the few that happen to survive it. Those survivors then have the place to
themselves and multiply. Over months you are steadily selecting for a flock carrying bacteria
that the antibiotic no longer touches. The day a real outbreak comes, the medicine you have
been relying on may do nothing.

It does not stop at your birds. Those resistant bacteria leave in the droppings, get onto
meat and eggs during slaughter and handling, wash into water, and end up on the hands of the
people who work with them. A resistant infection that started in a poultry house can end up
being the reason a child's treatment fails at the health centre. Human health, animal health
and the environment are one system - that is what One Health means.

What works better: keep the birds healthy so they don't need treatment. Clean water, good
ventilation, not overcrowding, proper cleaning between batches, and vaccination where it's
available. Then use antibiotics for **sick** birds, on veterinary advice, for a defined
course - not as a daily feed additive.

If you're using feed antibiotics because birds keep getting sick, that's a husbandry problem
the antibiotic is masking. Your district veterinary officer can help you find what's actually
driving it.""",
], NO_SOURCE, ["no numbers - no livestock source in corpus"])


def main():
    for rec in FIXES:
        sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("distilled_fixes: %d records" % len(FIXES), file=sys.stderr)


if __name__ == "__main__":
    main()
