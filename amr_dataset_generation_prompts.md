# OneAMR instruction-tuning dataset — generation spec

**Status: DRAFT written by Claude on 2026-09-03**, because the original file was not
present in the repo or in any commit. Everything here is a proposal. Review it —
especially the Category 7 withdrawal-period template and the per-category
targets — before or after the first generation run; the validator
(`prepare_dataset.py`) enforces what this file says, so changing a rule here
means re-running validation, not regenerating.

Target: **~5,000 pairs** across seven categories, three registers.

---

## 0. The one rule that overrides everything

**No clinical content from model memory.** Every generation call carries at least
one source chunk. A drug name, dose, frequency, route, or duration may appear in
an assistant answer only if it is:

- present verbatim (or as an unambiguous restatement) in an attached chunk, **or**
- present in `source_material/aware_tiers.json` (AWaRe tier only), **or**
- present in `source_material/emhslu_drugs.json` (availability/formulation only).

If the chunk does not support the answer, the correct output is a Category 7
refusal, not a plausible guess. The validator treats an unsupported number as a
**blocking** issue.

---

## 1. Registers

Three registers, matching `SYSTEM_PROMPTS` in `app.py` exactly. The `system`
message of every record is copied byte-for-byte from `app.py` — do not paraphrase
it, or the fine-tuned model will diverge from the deployed prompt.

| Register | Audience | Categories |
|---|---|---|
| `health_worker` | Clinicians prescribing in Uganda | 1, 2, 3, 4, 7 |
| `patient` | Patients and caregivers, plain language | 5, 7 |
| `farmer` | Livestock keepers, One Health | 6, 7 |

---

## 2. Output schema

One JSON object per line in `raw/<category>.jsonl`:

```json
{
  "messages": [
    {"role": "system",    "content": "<verbatim SYSTEM_PROMPTS[register] from app.py>"},
    {"role": "user",      "content": "<the question>"},
    {"role": "assistant", "content": "<the grounded answer>"}
  ],
  "category": "no_antibiotic_indicated",
  "register": "health_worker",
  "source": "acute_diarrhoea.txt :: Uganda Clinical Guidelines 2023 (4th ed.) :: 6.1.5 Diarrhoea :: p.469-471",
  "chunk_id": "acute_diarrhoea::uganda_clinical_guidelines_2023::p469-471::1",
  "context_mode": "closed",
  "audience": "general"
}
```

- `source` — human-readable: `<chunk file> :: <document> :: <section> :: <pages>`.
  Multi-chunk calls join with `+`.
- `chunk_id` — the exact `chunk_id` from `source_material/chunks/_manifest.jsonl`,
  so any pair can be traced back to the bytes it came from.
- `context_mode` — see §3.
- `audience` — `general` or `pediatric`, inherited from the chunk.

### 2.1 `context_mode` — grounded vs closed

`app.py:build_prompt` injects retrieved passages into the **user** turn at
inference. If every training pair were a bare question, the tuned model would
learn to answer from its weights and then meet a context block it was never
trained to read. So the dataset carries both modes:

- **`closed` (70%)** — user turn is the question alone. Teaches the stewardship
  behaviour, register, and refusal discipline.
- **`grounded` (30%)** — user turn is formatted exactly as `build_prompt` does:

  ```
  Relevant reference material (cite source + page when you use it):
  [<source>, p.<page>]: <chunk text>

  Question: <the question>
  ```

  Teaches the model to read, cite, and — critically — to **reject a passage that
  does not match the patient** (wrong age group, wrong presentation). At least
  15% of `grounded` pairs must attach a **deliberately mismatched** passage that
  the answer must explicitly set aside.

---

## 3. The generation call

For each call, send the generator model:

1. A generation system prompt (below).
2. The attached chunk(s), verbatim, delimited.
3. The category instruction block for the target category.
4. A request for `n` pairs as JSON.

### 3.1 Generator system prompt (verbatim)

```
You are building a supervised fine-tuning dataset for an antimicrobial
stewardship assistant used in Uganda. You will be given verbatim source
passages from the WHO AWaRe antibiotic book, the Uganda Clinical Guidelines
2023, the Uganda Essential Medicines List (EMHSLU 2023), or WHO paediatric
guidance, and a category of training example to produce.

Hard rules:
1. Every drug name, dose, frequency, route and duration in an answer MUST be
   traceable to the passages provided in this call. Do not add a drug, a dose,
   or a duration from your own knowledge, not even a correct one.
2. If the passages do not support a confident answer, produce a refusal example
   instead: state what is known, state what is not, and escalate.
3. Never extrapolate a dose to a weight, age, or route the passage does not
   cover. "The passage gives 3-6 kg and 6-10 kg bands" does not license you to
   invent a 2 kg dose.
4. Uganda context: prefer what EMHSLU 2023 actually stocks and the facility
   level that stocks it. Do not recommend a drug that is not on that list.
5. Output valid JSON only, matching the requested schema. No commentary.
```

### 3.2 Batch size and self-check

Generate in batches of **50**. After each batch, hand-check **5** at random
against the attached chunk. If any of the 5 contains a number or drug name not
in the chunk, **stop that category and report** — do not continue.

---

## 4. Categories

### Category 1 — `antibiotic_indicated` (target 1,050, register `health_worker`)

A clinical vignette where an antibiotic **is** indicated. The answer must state,
in this order: (a) that an antibiotic is indicated and why; (b) the first-choice
agent **with its AWaRe tier**; (c) dose; (d) duration; (e) what to do if it
fails or the patient deteriorates.

- Vignettes must vary age, sex, weight, severity, comorbidity, and setting
  (HC2 / HC4 / hospital).
- The AWaRe tier comes from `aware_tiers.json`, not from the chunk's badges —
  the AWaRe book's tier badges are **graphics and are absent from the extracted
  text**, so a tier claimed "from the passage" is a hallucination.
- If the chunk gives a weight-band table, the vignette's weight must fall inside
  a stated band.

### Category 2 — `no_antibiotic_indicated` (target 900, register `health_worker`) ⚠ high risk

A vignette where an antibiotic is **not** indicated. The answer must:

1. Say plainly that no antibiotic is indicated.
2. Give the reason **as the source states it** — viral aetiology, self-limiting
   course, or an explicit UCG instruction such as §6.1.5's *"Avoid inappropriate
   use of antibiotics e.g. metronidazole, ciprofloxacin"*. Do not substitute
   generic medical intuition for what the guideline actually says.
3. Give symptomatic management **matched to that presentation** — ORS and zinc
   for watery diarrhoea; rest, fluids and antipyretics for viral URTI. Never
   transplant one presentation's management onto another.
4. Name **specific** danger signs that would change the assessment. For
   diarrhoea these must include: blood in the stool, high fever, signs of severe
   dehydration (dry mouth, no tears, little or no urination), persistent
   vomiting, lethargy. "If it gets worse" is a validator failure.

**No concession antibiotic, ever** — not hedged, not "if you insist", not "not
first-line but". The validator blocks any Category 2 answer that names an
antibiotic in a recommending frame.

Ratio: Categories 1 and 2 together should land near **55:45 indicated
: not-indicated**. Over-weighting refusal produces a model that withholds
treatment from a child with pneumonia; the earlier 46-pair run failed in exactly
that direction.

### Category 3 — `aware_classification` (target 550, register `health_worker`)

"Is X Access, Watch, or Reserve?" and the reasoning behind the tiers.

- Ground on `source_material/chunks/aware_classification_table.txt` (all 257
  antibiotics: 87 Access, 141 Watch, 29 Reserve) plus the AWaRe book's Chapter 2
  narrative for the *why*.
- **Route-dependent drugs**: `fosfomycin` and `minocycline` have different tiers
  IV vs oral. An answer that gives a single tier for these without naming the
  route is a validator failure. `aware_lookup.json` lists all 12 route-split
  entries.
- Include ~80 pairs on drugs that are **not** in the AWaRe classification at all
  (isoniazid, rifampicin, ethambutol, dapsone, clofazimine…). The correct answer
  is that AWaRe does not classify anti-TB and antileprosy agents — not a guessed
  tier.
- Include ~60 pairs grounded on EMHSLU: whether the drug is available in Uganda
  and at which facility level.

### Category 4 — `dosing_duration` (target 900, register `health_worker`) ⚠ high risk

Dose and duration questions. **Every number must trace to the attached chunk.**

- Prefer the AWaRe book's weight-band tables, which extract cleanly
  (`3–6 kg: 250 mg given every 12 hours`) and the adult/paediatric dosing
  guidance chapters (`dosing_guidance_adults`, `dosing_guidance_children`).
- **Do not extrapolate.** If the bands are 3–6 / 6–<10 / 10–<15 kg, there is no
  pair for a 2 kg neonate. If the chunk gives an adult dose only, there is no
  paediatric pair from that chunk.
- **UCG tables are unreliable in text form** — column alignment is lost
  (`10–20 2–7 years 1 1 1`). Do **not** generate dosing pairs from a UCG table.
  UCG prose dosing (`zinc 20 mg per day`) is fine. AWaRe and EMHSLU tables are fine.
- **Never** use WHO Pocket Book pp. 47–62; those tables are letter-spaced and
  corrupt (`1 0 % g l u c o s e`).
- Every answer states the route and the total duration, and flags renal dose
  adjustment only when the chunk mentions it.

### Category 5 — `patient_education` (target 700, register `patient`)

Adherence, why antibiotics don't work on viruses, resistance, not sharing or
saving leftovers, not buying part-courses. Plain language, no jargon, no drug
names unless the chunk supplies them, roughly 120–250 words.

Must not contain a dose. Patients do not dose themselves from this model.

### Category 6 — `farmer_one_health` (target 450, register `farmer`)

⚠ **There is no source document in this corpus covering livestock antibiotic
use.** AWaRe, UCG, EMHSLU, IMCI and the Pocket Book are all human-health texts.
Uganda's NAP-AMR II is referenced in `metadata.json` but is **not** in `corpus/`.

Therefore Category 6 pairs are grounded only in transferable One Health
reasoning — why resistant organisms move between animals, people and the
environment; why routine prophylactic dosing of healthy flocks drives
resistance; what a withdrawal period is *conceptually*; who to ask.

**Hard constraint: no numbers.** No withdrawal period in days, no MRL value, no
veterinary dose, no drug-specific claim. Any such number would be fabricated.
Every question that reaches for one gets the Category 7 withdrawal template.

If NAP-AMR II or a Ugandan veterinary formulary is added to `corpus/` later,
this category can be regenerated with real figures.

### Category 7 — `refusal_escalation` (target 450, all registers) ⚠ high risk

Four sub-types, roughly evenly split:

**7a — out of grounding.** Question the corpus doesn't cover. Say what is known,
say plainly that the specific answer isn't in the guidance available, escalate.

**7b — pushback.** Multi-turn: the model correctly says no antibiotic is needed,
the user pushes back **without new clinical information**, and the model holds.
Format: `[system, user, assistant, user, assistant]`. The second answer must
restate the reasoning in **one** sentence (not re-deliver the first answer), give
one concrete line the clinician can say to the patient, name specific danger
signs, and offer escalation. It must **not** name any antibiotic.

**7c — mismatched passage.** A passage is attached that does not fit the patient
(paediatric dosing for an adult, bloody-diarrhoea guidance for watery). The
answer must say so explicitly and refuse to apply it.

**7d — withdrawal period.** Farmer asks how long before selling milk/meat.

#### 7d template — this wording is mandatory

The answer must **never state a number of days**. It must contain, in order:

1. An acknowledgement that the withdrawal period is the right thing to ask about,
   and one sentence on why it matters (residues reach people through milk and meat).
2. An explicit statement that the withdrawal period **depends on the specific
   product, its formulation, the dose given, and the species**, and that it is
   **not something this assistant can give a number for**.
3. Where the number actually comes from, in this order: the **product label or
   package insert**, the prescribing **veterinarian**, and the **district
   veterinary officer**.
4. What to do meanwhile: do not sell or consume milk or meat from the treated
   animal until the labelled period has passed; keep a written record of the
   drug, date, dose and animal.
5. No number. No range. No "usually about a week". No "typically 3–7 days".

The validator blocks any Category 7d answer containing a day-count pattern
(`\d+\s*(day|days|hours|weeks)`) or the words "usually", "typically", or
"generally" adjacent to a duration.

---

## 5. Per-category targets

| # | Category | Register | Target | Risk |
|---|---|---|---|---|
| 1 | `antibiotic_indicated` | health_worker | 1,050 | |
| 2 | `no_antibiotic_indicated` | health_worker | 900 | ⚠ |
| 3 | `aware_classification` | health_worker | 550 | |
| 4 | `dosing_duration` | health_worker | 900 | ⚠ |
| 5 | `patient_education` | patient | 700 | |
| 6 | `farmer_one_health` | farmer | 450 | ⚠ no source |
| 7 | `refusal_escalation` | all | 450 | ⚠ |
| | **Total** | | **5,000** | |

Distribution constraints the validator checks:

- indicated : not-indicated within 50:50 – 60:40.
- No single `chunk_id` accounts for more than 2.5% of a category —
  **except `aware_classification`, see the exception below.**
- Register split within ±3 points of: health_worker 68%, patient 15%, farmer 10%,
  mixed-register Category 7 making up the rest.
- `context_mode`: grounded 30% ±5.
- Paediatric `audience` ≥ 20% of Categories 1, 2 and 4 combined.

### Exception: `aware_classification` uses drug diversity, not chunk diversity

**Decided 2026-09-04, deliberately and for this category only.**

The 2.5% chunk cap exists to stop a category resting on a handful of source
passages. For `aware_classification` it measures the wrong thing. The
authoritative source for an AWaRe tier legitimately *is* one document — the WHO
AWaRe classification 2023 — so chunk-level concentration is expected rather than
a defect, and chunk identity carries no information about whether the category
covers the antibiotic space well.

It is also arithmetically unsatisfiable here. Only **32 chunks** are eligible for
this category. At a 550-pair target the cap is 13 pairs per chunk, giving a hard
ceiling of **32 × 13 = 416** — below the target — and in practice the candidate
pool collapses as chunks saturate, stalling three separate runs at 323, 320 and
326 pairs. The target and the constraint cannot both be satisfied.

So for `aware_classification`:

- the **chunk cap does not apply**;
- diversity is enforced on **drug identity** instead, via the per-drug cap
  (`--drug-cap-fraction`, default 0.08; runs at scale use 0.02, i.e. no single
  drug exceeds 2% of the category). 264 drugs are reachable from the pool, so
  this is a real constraint, not a formality;
- every other category keeps the chunk cap unchanged.

**Rejected alternative:** splitting the AWaRe table into more, smaller chunks
would have satisfied the 2.5% rule arithmetically while leaving the actual source
concentration identical. That is compliance with the letter of the rule against
its purpose, and it was rejected on those grounds.

**Consequence to accept honestly:** if the realised count still falls short of
550 once the arithmetic wall is removed, the limit is how many antibiotics the
model can classify correctly from a real excerpt. Take the smaller number and
record it. Do not relax a correctness guard to reach a target.

---

## 6. Pipeline

```
scripts/extract_sources.py        PDFs        -> source_material/*.txt
scripts/chunk_by_topic.py         text        -> source_material/chunks/*.txt + _manifest.jsonl
scripts/build_reference_lists.py  xlsx + text -> aware_tiers.json, aware_lookup.json, emhslu_drugs.json
scripts/generate_pairs.py         chunks      -> raw/<category>.jsonl        [runs on the GPU box]
distilled_fixes.py                -           -> raw/distilled_fixes.jsonl
prepare_dataset.py                raw/*.jsonl -> dataset/{train,val}.jsonl + validation report
```
