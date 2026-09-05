# OneAMR — running notes for the IEEE journal paper

**Status: living document, not a submission draft.** Started 2026-09-04, during
dataset construction. Write to it *as work happens* — several findings below
could not have been reconstructed after the fact.

Target format: IEEE journal style (IEEEtran, two-column, numbered `[n]`
citations). Likely venues in order of fit: *IEEE Journal of Biomedical and
Health Informatics* (JBHI), *IEEE Access*, *IEEE Transactions on Technology and
Society*. JBHI is the strongest fit if we have a real evaluation; IEEE Access is
the fallback if the contribution stays mostly systems/engineering.

---

## 1. Working title candidates

1. "OneAMR: An Offline, Guideline-Grounded Language Model for Antimicrobial
   Stewardship in Low-Resource Settings"
2. "Teaching a Small Language Model to Withhold: Guideline-Grounded
   Antimicrobial Stewardship on Consumer Hardware"
3. "Silent Corruption in Guideline-Derived Training Corpora, and What It Costs
   an Antimicrobial Stewardship Model"

(2) frames the actual novelty — *declining* to recommend. (3) is the honest
methods paper if the evaluation underdelivers. Decide after results.

---

## 2. The claim we are actually making

Not "an LLM that prescribes antibiotics." The contribution is a model that
**correctly declines** when no antibiotic is indicated, runs **fully offline in
8 GB RAM**, and grounds every clinical specific in a **named, versioned,
checksummed source**. Three registers — health worker, patient, farmer.

Framing to keep: over-prescription, not under-prescription, is the dominant
failure mode in the target setting, so a stewardship assistant that answers
every question with an antibiotic is worse than useless. Refusal is the feature.

---

## 3. Section plan, and what we can already support

### I. Introduction
- AMR burden framing: 2019 attributable and associated deaths [Murray2022],
  updated and projected in [GBD2024]. Sub-Saharan Africa carries the highest
  age-standardised mortality — this is the motivating statistic.
- Antibiotic consumption growth [Klein2018]; East Africa and Uganda specifics
  [Ampaire2016], [Kiggundu2022].
- Offline constraint is not a nicety: intermittent power and connectivity in
  Ugandan health facilities. **TODO: cite a real infrastructure source, do not
  assert this from memory.**

### II. Background and Related Work
- WHO AWaRe framework [WHO_AWaRe_Book2022], [WHO_AWaRe_Class2023]; national
  formulary layer [UCG2023], [EMHSLU2023]; paediatric [WHO_IMCI], [WHO_Pocket].
- Medical LLMs [Singhal2023], [Singhal2025], benchmarks [Jin2021].
- RAG [Lewis2020] vs. instruction tuning — we use both, at different stages.
- Hallucination [Ji2023] — frame the refusal work against this.
- Synthetic instruction data [Wang2023SelfInstruct], [Taori2023], [Xu2024].
- Efficient adaptation [Hu2022LoRA], [Dettmers2023QLoRA]; quantization
  [Frantar2023GPTQ], [Lin2024AWQ]; on-device SLMs [Abdin2024Phi3].
- Gap we claim: guideline-grounded synthetic data with *provenance to the
  passage*, plus explicit non-prescription supervision, on consumer hardware.

### III. Corpus construction and provenance — STRONGEST SECTION, ALREADY DONE

See `source_material/SOURCES.md` for the authoritative record. Paper-worthy:

**(a) Colour-encoded classification does not survive text extraction.**
The AWaRe book encodes each antibiotic tier as a *coloured badge*, not text —
the book says so itself ("ACCESS antibiotics are indicated in green, WATCH in
orange and RESERVE in red"). Colour is lost in extraction: an audit found tier
words attached to roughly 6 drugs in 1.46 M characters. Ceftriaxone,
metronidazole, gentamicin, doxycycline and cloxacillin had none. Without the
separate 2023 classification spreadsheet, every tier answer would have come from
model memory. **This is a generalisable warning about PDF-derived clinical
corpora and belongs in the paper.**
Recovered: 257 antibiotics, 87 Access / 141 Watch / 29 Reserve — matching the
counts WHO publishes, which is our correctness check.

**(b) Invisible symbol-font corruption, including a hidden dose.** (commit `fa738a5`)
The first audit checked U+FFFD, `(cid:N)` and residual ligatures, found zero of
each — and missed **6,467** invisible characters:

| Class | Before | After |
|---|---|---|
| C1 controls (U+0080 to U+009F) | 5,506 | 0 |
| Private-use (U+E000 to U+F8FF) | 961 | 4 (whitelisted) |

Most were list bullets — U+0089 x3,111 is the bullet on *every UCG TREATMENT
line*, i.e. the drug-and-dose lines. But the IMCI booklet uses a subsetted font
that maps plain ASCII into U+F0xx, so `U+F032 U+F035 U+F030` followed by `" mg"`
is literally **`250 mg`**, the paediatric mebendazole dose, and `U+F042` plus
`"reastfeed"` is `"Breastfeed"`.

The obvious remedy — strip unrecognised private-use characters — **would have
silently deleted the digits of a dose.** The mapping must be keyed by document,
because the same codepoint is a different glyph in a different font (U+F067 is a
header separator in the AWaRe book; U+F077 is a bullet in UCG; neither is the
ASCII letter its low byte suggests). Unmapped codepoints are now a hard error.
Relevant prior work on extraction fidelity: [Bast2017], [Ramakrishnan2012].
**This is a genuine, citable methods contribution.** Argue: corpus audits for
clinical NLP must test for symbol-font and C1 residue, not only U+FFFD.

**(c) Deliberate non-repair.** Four U+F0E2 remain in UCG shock-severity tables
alongside Latin-1 mojibake; they are directional arrows whose direction the text
layer no longer records. We refused to guess "decreased" against "increased" in
a triage table and flagged those tables unusable for grounding. Same treatment
for UCG dose tables that lose column alignment and for letter-spaced Pocket Book
tables. Good material for limitations — principled abstention in *data cleaning*,
mirroring the model behaviour we are training.

### IV. Dataset generation
- Generator: Qwen2.5-14B-Instruct Q5_K_M via llama.cpp `llama-server`, used
  purely as a data engine — never fine-tuned, never shipped. Deployment target
  is a separate 3B. **Make this distinction explicit; reviewers will ask.**
- Every call carries source passages; no clinical content from parametric memory.
- Seven categories, ~5,000 pairs, three registers. Targets in
  `amr_dataset_generation_prompts.md`.
- **Negative supervision is the point**: Category 2 (no antibiotic indicated)
  and Category 7 (refusal and escalation), including a withdrawal-period
  template that must never state a number.
- **The grounding unit is chosen per category, and that choice is a result.**
  Attaching a topic chunk and hoping the model finds the right row does not
  work: the model answered "meropenem is Reserve" with the Watch row in its own
  prompt, and 36 of 47 rejections in an early run were a passage that did not
  name the drug asked about. Generation is now planned drug-first — pick the
  target, find material that provably contains it, and hand over only that.
  Category 3 gets the classification rows for its target drugs; Category 4 gets
  per-drug dose lines with the indication attached. Yield rose from 30% to 83%
  with no guard loosened, which is the evidence that the earlier losses were a
  retrieval failure and not a model failure.
- **Acceptance gates reject, never repair.** A rejected pair is dropped and
  regenerated; nothing is edited into looking correct. This matters for the
  dataset-integrity claim — no pair in the final set was touched by hand.

### V. Validation
`prepare_dataset.py` — blocking issues against flags; drug names checked against
50 EMHSLU section 6.2 entries and 257 AWaRe drugs loaded from generated JSON,
never a hand-typed list; AWaRe tier consistency with route-dependence handling
(longest-match-first with span masking, so `ceftazidime+avibactam` does not also
match `ceftazidime`); indicated to not-indicated ratio; register and chunk-id
distribution caps.

### VI. Fine-tuning and deployment
Unsloth plus LoRA on Qwen2.5-3B-Instruct, export to GGUF Q4_K_M, target under
8 GB RAM. **TODO: record every hyperparameter at the time of the run.**

### VII. Evaluation
**Not yet designed. This is the paper's main risk — see section 5.**

---

## 4. Findings log (append-dated, do not rewrite history)

### 2026-09-04 — Generator failure modes, 20-pair Category 3 pilot

Tiers were **20/20 correct** — tier-table grounding works. Everything around the
tier did not:

- Four fabricated EMHSLU availability claims (e.g. "fosfomycin is available
  intravenously according to EMHSLU 2023" — fosfomycin is not on EMHSLU at all;
  same error for clarithromycin; "minocycline is not classified for IV use" — it
  is, Reserve).
- Fabricated and self-contradictory rationales: amoxicillin called
  "narrow-spectrum" in one pair and "broad-spectrum" in another; Access glossed
  as "without the risk of promoting antibiotic resistance"; tigecycline's
  Reserve status attributed to "high cost".
- A `grounded` pair whose attached passage was EMHSLU *nutrition and
  neurosurgery* text — answered confidently anyway. Root cause: chunk
  `formulary_antibacterials::emhslu_2023::p31-51::7` is labelled "6.2
  Antibacterials" but contains **zero antibiotics** (the topic slice spans
  pp.31 to 51 and the last 6,000-character window is off-topic residue).

**Paper value:** a clean, quantified demonstration that *retrieval grounding
constrains the fields it covers and nothing else*. A structured lookup table
(AWaRe tiers) was reproduced perfectly; free-text justification, and a second
knowledge base (national formulary availability) that was not in the prompt,
were confabulated. Report it as a finding, not an embarrassment — with the
caveat that n=20 and it is a pilot. **Re-measure at scale with the fixed
pipeline and report both numbers.**

### 2026-09-04 — Category 4: three failure modes a dose check cannot see

Dosing is where a wrong number does the most harm, so it got the most scrutiny,
and the scrutiny kept finding things the previous layer had missed. Each of
these is a distinct failure class and each is worth a paragraph in the paper,
because each was invisible to the check that preceded it.

**(a) Dose–drug misattribution.** Extracting dose lines with a sliding window
files a neighbouring drug's number under the wrong drug. UCG p.183 reads "Or
cloxacillin 50 mg/kg IV every 4-6 hours - Or benzylpenicillin 50,000 IU/kg IV",
and the naive reading gives benzylpenicillin 50 mg/kg. Fixed by attributing a
dose only to the drug whose mention precedes it, with the next drug mention
closing the span. **1,509 of 4,239 candidate lines have no dose attributable to
their drug at all; 890 more share a window with another drug's dose.** Confirmed
against the rendered PDF page: they are separate bullets.

**(b) Right number, wrong clinical question.** This one survives every check
that asks "is this number in the source", because the number *is* in the source.
Benzylpenicillin in UCG carries at least five doses by indication -- 3-4 MU for
*Streptococcus pneumoniae*, 5-6 MU for *Neisseria meningitidis*, 3 MU for
*Listeria monocytogenes*, 2.5 MU q6h for tetanus, 100,000 IU/kg q12h for
neonatal tetanus. **25 of 36 drugs with clean lines carry more than three
distinct doses**, and only 9 of benzylpenicillin's 55 lines name an indication
on the line itself; the indication lives in a heading above. Recovering it
required anchoring the upward walk on the drug's own line, not the window start.

**(c) The guard's own matching can be wrong in a way that looks right.**
Adversarial probing -- not a clean run -- found that species epithets are
useless discriminators: "meningitidis" shares nine leading characters with
"meningitis", so an answer about *pneumococcal* meningitis satisfied a
*Neisseria meningitidis* source line. The gate would have passed the exact swap
it was built to catch. Discrimination now uses genus plus clinical adjective.

**Method claim worth making explicitly:** a gate that never fires has not been
shown to work. Every gate in this pipeline was probed with deliberately
malformed input before being trusted, and in two of three cases the probe found
a hole a clean batch would not have. The indication gate then fired 8 times on a
real 20-pair batch, which is the evidence that it catches live errors rather
than only synthetic ones.

**Precision costs too.** Treating "months" as a duration unit meant "infants up
to 2 months old" -- an age restated from the question -- was checked as if it
were a treatment duration and rejected. Paediatric age- and weight-framed
questions are the commonest shape in this category, so an over-strict gate
discards exactly the data the category most needs. Worth stating that guard
design has two failure directions, not one.

**Provenance.** The `[[page N]]` markers are pypdf indices, not printed page
numbers: UCG PDF page 183 prints as 117, a constant offset of 66 recovered for
1,094 of 1,097 pages by reading the folio out of the page text. A citation a
reviewer cannot open to undermines the hand-checkability the whole design rests
on. The AWaRe book recovers only 148 of 697 and still needs a different method
-- **do not put an unqualified page-citation claim in the paper until that is
fixed.**

### 2026-09-04 — Infrastructure note

A 4,096-token context was insufficient: `aware_classification` prompts run about
4,576 tokens before any output, because the prompt carries a full chunk plus a
12,000-character tier table. Raised to 16,384 with q8_0 KV cache: 13,188 of
16,303 MiB VRAM on an RTX 5070 Ti. Minor, but it belongs in a reproducibility
appendix.

---

### 2026-09-04 — Scale-up to 550: three failures that a 20-pair batch cannot show

The strongest single methodological result so far, and it should be stated
plainly in the paper: **the same pipeline that produced 0 tier errors in 20
pairs produced 15 in 323.** Small clean batches are not evidence of a clean
pipeline. Every gate here had passed a 20-pair batch and an adversarial probe
before this run.

Run: `aware_classification` targeted at 550, aborted by its own stall detector at
**323 accepted / 106 rejected (75% yield)**, 107 distinct drugs, grounded 30%.
Yield degraded across the run — 92% at 70 pairs, 86% at 275, collapsing at the
end — which is itself the signal that something was drifting.

**(a) "Absent from my context" became "absent from the world".** This is the
interesting one, because it was *created by the fix for an earlier problem*.
Giving the model only the classification rows for its target drugs eliminated
tier errors (the 366-row table was unusable — it answered "meropenem is Reserve"
with the Watch row in its own prompt). But when the model drifted to a drug
outside its excerpt, it converted "not in the rows I was given" into "AWaRe does
not classify this agent". Rifampicin is Watch; sisomicin is Watch; daptomycin is
Reserve; tinidazole is Access. **13 of the 15 errors are this shape.**

The first clause of those answers is honest — the material provided genuinely
does not cover the drug. The second is a false universal negative. For a
stewardship model this is a bad thing to learn: it teaches denial of
classification for real antibiotics.

Worth framing as a general trade-off in the paper: **narrowing context to what
is relevant improves faithfulness to the context and simultaneously invites
over-generalisation from it.** A retrieval system that hands a model a focused
extract must also tell it what the extract is *not* evidence about. We had it
backwards — the prompt said "not present in the AWaRe classification at all …
the correct answer is that AWaRe does not classify the agent", which is only
valid for genuinely unclassified agents (anti-TB, antileprosy) and which the
model applied to anything missing from its excerpt.

**(b) Partial answers to multi-valued facts pass a set-intersection check.**
Fosfomycin is Reserve intravenously and Watch orally; an answer saying only
"Reserve" is incomplete in a way that matters clinically. The guard tested
`expected ∩ stated ≠ ∅`, which a partial answer satisfies. It needs
`expected ⊆ stated`. Small bug, general lesson: when a fact is multi-valued,
membership is the wrong test.

**(c) An over-strict guard is a data-quality problem too.** The minimum accepted
answer length was exactly the 80-character floor, i.e. the floor was binding.
"Mecillinam belongs to the Access group according to the WHO AWaRe
classification" (81 chars) passed; "Mecillinam is in the Access group" (33) —
correct and complete — did not. As the run reached less familiar drugs the model
answered more tersely, rejections rose, and the run stalled. The floor was a
proxy for "has substance"; for a classification category, substance is *drug +
tier*, not length.

Guard design therefore has **two** failure directions, and papers on
guardrailed generation usually only discuss one. A guard that is too loose
admits wrong data; a guard that is too tight discards right data and, at scale,
can halt the pipeline. Both were observed here within a single run.

**Engineering note worth an appendix line:** the stall detector — abort after N
consecutive calls yield nothing acceptable — is what turned a silent degradation
into a diagnosable stop. Without it the run would have burned GPU time producing
nothing, which is exactly what an earlier version did for fifteen minutes on
HTTP 400s.

### 2026-09-05 -- Categories 1 and 2: the exception boundary, and a gate that can see it

Categories 1 and 2 were specified as separate categories with a target ratio.
They are not separate. For each presentation the guideline states a default and a
named set of criteria that move a patient across it, so the two categories are
the two sides of one boundary, and generating them independently leaves the
boundary itself unsupervised. The scenario pool (`scripts/scenario_pool_data.py`)
is built first, reviewed by a clinician before generation scales, and then used
as the standard every generated vignette is checked against.

**50 statements across 6 presentations** (acute bronchitis, COPD exacerbation,
acute otitis media, acute sinusitis, pharyngitis, acute diarrhoea), each carrying
the verbatim source string it rests on. **All 50 quotes verify against the chunk
they cite** -- the pool is held to the same standard as the model, which is the
only defensible position: a criterion may not enter because it is clinically
true, only because the corpus states it.

**(a) The COPD symptomatic-management gap was real, and closed by a second source
already in the corpus.** The AWaRe COPD chapter names the non-antibiotic
management -- oxygen, short-acting inhaled beta-2-agonists, systemic steroids --
and then says "A detailed discussion of non-antibiotic management of COPD is
beyond the scope of this chapter". Named but undosed, which is not enough for a
Category 2 answer that has to offer a real substitute. UCG 2023 section 5.1.2
(pp.406-410, printed 340-343) supplies the specifics and is already chunked as
`copd_exacerbation::uganda_clinical_guidelines_2023::p406-410::1`. So the pool
entry is grounded, not invented, and COPD stays a full Category 2 presentation
rather than being reclassified as a refusal. Worth stating in the paper that the
gap was found by asking what the *substitute* would be, not by auditing the
antibiotic advice: a chapter can be complete about when to withhold and silent
about what to do instead.

**(b) The exception-criteria smuggle: four shapes, and the naive gate caught one.**
The generation spec asks the validator to block "any Category 2 answer that names
an antibiotic in a recommending frame". Implemented literally against
`Validator.known_drugs` -- the drug list every other gate uses -- that gate
**missed 5 of 6 adversarial cases** (`scripts/probes/probe_c2_exception_smuggle.py`).
It caught only the case that names the drug outright. The four shapes it cannot
see:

  (a) conditional hand-off with no drug named: *"If the sputum becomes purulent
      and increases in volume, begin the first-choice oral agent for 5 days."*
  (b) class instead of name: *"Should her symptoms fail to settle, prescribe an
      aminopenicillin."*
  (c) asserting the exception is met when the vignette does not contain it:
      *"Given the severity of her symptoms an antibiotic is indicated here"*, on a
      vignette describing a comfortable, alert 4-year-old at 38.2 C.
  (d) a real criterion restated with a loosened bound: *"Consider an antibiotic if
      her symptoms persist without improvement for more than 3 days"* where the
      source says ten; *"bilateral otitis media in children under 5 years"* where
      the source says under two.

(d) is the worst, and is the reason this category needed a pool at all. That
answer is **correct about the patient in front of it** -- it withholds the
antibiotic -- and teaches a lower threshold for every future patient in the same
breath. The dose gate cannot see it because there is no dose; the drug gate
cannot see it because there may be no drug; a "did it refuse?" check cannot see
it because it did refuse. It is visible only against an enumerated,
source-verified statement of what the threshold actually is. **The fixed gate is
8 of 8 on the same cases.**

This is the third gate in this pipeline whose adversarial probe found a hole a
clean batch would not have, and the pattern is now worth stating as a claim: the
hole is never in the check the gate advertises. The dose gate checked numbers
correctly and mis-attributed indications; the tier gate checked tiers correctly
and confabulated availability; this gate checks drug names correctly and lets the
prescribing *rule* through untouched.

**(c) A guard keyed on the wrong drug list rejects the right answer.**
`Validator.known_drugs` is antibiotics UNION every EMHSLU entry, and EMHSLU
carries prednisolone -- which is part of the correct non-antibiotic management of
a COPD exacerbation. A name-based Category 2 gate therefore rejects the best
available Category 2 answer for naming the right steroid. "Antibiotic" here has to
mean "carries an AWaRe tier", not "is in the drug list". The over-strict failure
direction again, and this time created by reusing a list built to answer a
different question.

**(d) The discrimination property holds, but only after the naive version failed
completely.** The interleaved design pairs a Category 1 and a Category 2 vignette
on the same chunk, which is only worth doing if the gate can assert which side
each falls on. The obvious matcher -- substring cues, any single limb, no
negation -- placed **6 of 12 vignettes** correctly, and the failures were not
scattered: it got **every Category 1 vignette right and every Category 2 vignette
wrong** (`scripts/probes/probe_c1c2_discrimination.py`). The reason is structural and
belongs in the paper. **A Category 2 vignette is the densest possible source of
exception-criterion vocabulary**, because ruling those criteria out is what makes
it a Category 2 vignette: "no blood in the stool", "afebrile", "the left ear is
normal", "sputum unchanged and not purulent". A matcher without polarity reads
every one of those as the criterion being present. Same shape as the
meningitidis/meningitis collision in the dose gate -- the cue is there, its
polarity is not. With negation scoping and multi-limb criteria the fixed matcher
is **12 of 12**.

Two further bugs surfaced only because the probe was adversarial rather than
clean, and both would have failed silently:

  * **Hyphenated ages did not parse.** "4-year-old" and "18-month-old" are how
    every paediatric vignette states an age; the year pattern expected a digit
    after the hyphen (reading it as a range) and the month pattern expected
    whitespace. So AOM-04 -- *bilateral acute otitis media in children under 2
    years*, the most-loosened criterion in the pool -- **could not fire on the one
    age band it was written for.** Months are now converted to years, because the
    guideline bound is written in years and the vignette is written in months.
  * **The negation window reached across clause boundaries.** In "...and no cough
    -- a Centor score of 4...", the "no" attached to "cough" was read as negating
    "Centor score of 4" thirteen characters later, suppressing a criterion that
    genuinely fired. Commas, dashes and brackets now end a negator's scope.

**(e) Same-chunk interleaving works for 5 of 6 presentations, and the sixth is a
finding.** One chunk carries both the default and an exception for COPD, otitis
media, sinusitis, pharyngitis and diarrhoea. **Acute bronchitis has none**: the
AWaRe chapter states the default and names *no exception at all*, while UCG 5.2.2
names the only exception in the corpus (fever >38.5 C, dyspnoea, purulent
expectorations). A bronchitis C1/C2 pair must therefore span two documents, and
its Category 1 member rests on a criterion the WHO source does not recognise.

That is not an isolated case. COPD has the same disagreement in a sharper form:
AWaRe restricts antibiotics to severe exacerbations requiring hospital admission,
5 days; UCG treats at health-centre level on sputum change plus temp >38 C, 7-10
days. **Same presentation, different boundary, different duration, both sources
in the corpus, both authoritative for this setting.** Recorded as pool entries
BRO-90 and CPD-90 and left unresolved, because resolving it would mean choosing a
national standard on the model's behalf. This is the concrete instance of the
source-consistency limitation, and should be cited as such rather than described
in the abstract.

**Pool size note for T3:** 50 statements, not the ~37 estimated. The increase is
the COPD supplementary-source entries and the two conflict entries; the estimate
predated finding either.

---

### 2026-09-06 -- Keeping the disagreement: source-tagged boundary variants, and a negation bug only a generation run could show

The ruling on the two conflicts found yesterday was to keep both sides rather
than pick a winner. That decision changes the unit of generation, and the change
is worth stating in the paper because it is what makes the disagreement
*measurable* rather than merely disclosed.

**The unit of generation is a boundary VARIANT, not a presentation.** A variant
is one presentation plus one source's exception criteria, with the default,
substitute, escalation and first-choice material that goes with them
(`scenario_pool.variants()`). Six presentations yield **7 variants**, because
COPD has two: AWaRe restricts antibiotics to severe exacerbations requiring
hospital admission; UCG treats at health-centre level on sputum change plus
temp >38 C. Pairs are generated from each, and every record carries
`criteria_source`, `criteria_ids`, `default_source` and `cross_source`. The
dataset therefore records which national-vs-WHO boundary each example was written
against, and the paper can report the count per side instead of asserting that
one was chosen.

**Two of the seven variants are cross-source, and both are generated, not worked
around.** Acute bronchitis: AWaRe states the default and names no exception at
all, UCG 5.2.2 names the only one in the corpus. COPD under UCG's threshold: UCG
states the criteria but no "not required in the great majority of cases"
sentence, so the default is AWaRe's. `cross_source: true` on every such record
gives the exact figure for how many pairs rest on a criterion the other source
does not recognise.

**(a) "Wrong source's threshold" is a different defect from "invented number",
and the gate now says so.** The bound check originally reported anything not in
the pool as unsourced. Under two retained boundaries that is wrong in a way that
matters for the paper: an answer citing temp >38 C on an AWaRe-tagged COPD pair
is quoting a real, corpus-present, *authoritative* threshold attached to the
wrong variant. Logging it as "unsourced" would make a genuine guideline
disagreement appear in the rejection table as a hallucination. The guard now
emits `is the OTHER source's threshold for X, not this pair's`, and the two
counts stay separate in T6.

**(b) The failure mode from yesterday's probe does occur in the generation path,
and is caught there.** Verified by running the real loop with the model call
stubbed (`scripts/probes/rehearse_boundary_run.py`), so prompt build, pair parsing,
whole-pair acceptance, the source tag reaching the guard and the file write are
all exercised. Both "technically withholds, teaches the wrong threshold" answers
were rejected as records, not just as probe strings: the loosened bronchitis
bound (`days=3` and `temp_c=38` where the source says 10 days and 38.5 C) and the
foreign COPD bound (`temp_c=38` on an AWaRe-tagged pair). A Category 1 half whose
vignette satisfied only the *other* source's criteria was also rejected --
evidence the tag is doing work, since that same vignette is correct under the
UCG variant.

**(c) The finding: negation scope is a two-sided error surface, and fixing one
side opened the other.** Yesterday's fix ended a negator's scope at commas and
dashes, because "...and no cough -- a Centor score of 4..." was suppressing a
criterion that genuinely fired. The first stubbed run produced this sentence, and
it is exactly what the spec asks a Category 2 vignette to look like:

> "...his sputum is unchanged and not purulent, and he has **no malnutrition,
> measles, rickets, severe anaemia or cardiac disease**."

BRO-05 names six comorbidities in **one** criterion, so ruling it out means one
negated list. With scope ending at the first comma, every item after the first
read as present, and **the gate rejected a correct Category 2 answer** -- the
over-strict direction, for the third time in this pipeline, and this time created
by the fix for the under-strict direction of the same rule. Scope now crosses a
comma while the crossed text is still bare list items (letters and spaces, no
article, no number, no verb of assertion), which readmits the list and still
excludes "a Centor score of 4". Both directions now have a probe case: the
discrimination probe is **14 of 14** (naive 7 of 14), the smuggle gate stays
**8 of 8**, and all 50 pool quotes still verify.

The general claim for the methods section: a polarity rule cannot be validated in
one direction. Every tightening that fixes a false accept is a candidate false
reject, so the probe has to carry a matched pair for each rule -- one vignette
that must fire and one that must not -- or the next fix silently trades one error
for the other. Note also *how* this was found: the adversarial probe missed it,
because a probe written by hand rules criteria out the way the author already
thinks about them. It surfaced only when the generation loop ran end to end. A
stubbed-model rehearsal is cheap and belongs in the method.

**Status:** the 20-pair test batch is **not yet generated** -- the GPU node has
been offline since 2026-09-05 (power loss), so no llama-server. Everything up to
the model call is verified and the run is a single command when the node returns:

```
python scripts/generate_boundary_pairs.py --model qwen2.5-14b-instruct-q5_k_m \
  --target 20 --discarded .scratch/boundary_discarded.jsonl \
  --require-variant acute_bronchitis:uganda_clinical_guidelines_2023 \
  --require-variant copd_exacerbation:who_aware_book_2022 \
  --require-variant copd_exacerbation:uganda_clinical_guidelines_2023
```

The run's own rejection table answers the question the ruling asked -- whether
the loosened-bound and foreign-bound answers recur in real generation and at what
rate -- and those counts go straight into T6.

## 5. Things we must not forget to capture

Ordered by how irrecoverable they are.

1. **Evaluation design — decide before training, not after.** Needs a held-out
   set the model never saw, and ideally a small clinician-reviewed subset.
   The primary metric should be *appropriate non-prescription*, not accuracy.
   Candidates: sensitivity and specificity of "antibiotic indicated"; tier
   accuracy; dose exact-match against source; refusal rate on out-of-corpus
   questions; and a pushback test (does it hold the line under patient
   insistence?). **A baseline is mandatory** — untuned Qwen2.5-3B-Instruct, and
   ideally the existing RAG-only Gate 1 build, on the identical test set.
2. **Hyperparameters and seeds at run time** — LoRA rank, alpha, dropout, target
   modules, LR schedule, epochs, batch and accumulation, max sequence length,
   quantization config, `--seed`. Capture from the actual command, not from
   memory afterwards.
3. **On-device measurements on the 8 GB target**, not on the RTX box: peak RSS,
   model file size, tokens/s prefill and decode, cold-start latency, and the
   exact hardware. `adtc_run*.log` and `tune_results.json` already hold Gate 1
   profiler numbers — reuse that methodology for continuity.
4. **Final dataset statistics** — per-category counts, register split,
   context_mode split, paediatric share, indicated to not-indicated ratio,
   unique chunk coverage, and the full `prepare_dataset.py` report.
5. **Yield and error rate at every batch size run**, not just the final one. The
   n=20 against n=323 divergence is a headline methods result and only exists
   because the intermediate numbers were recorded. Keep logging per-run:
   accepted, rejected, rejection reason counts, distinct drugs, and independent
   error count.
6. **The hand-review outcome** — how many flagged drug names were real errors.
   That number is a headline result: it quantifies what human review caught that
   automated validation did not.
7. **Inter-rater material** if any clinician reviews outputs — who, how many
   items, agreement. Without this the clinical claims stay weak.
8. **Ablations, if time allows**: with and without source grounding in the
   prompt; with and without Category 2 and 7 negative supervision (this directly
   tests the central claim); Q4_K_M against Q5_K_M against fp16 degradation.
9. **Ethics and responsible use**: not a medical device, decision *support*
   only, licence terms of each source (the AWaRe book is CC BY-NC-SA 3.0 IGO —
   check what that permits for derived training data **before** publishing the
   dataset), and a clear statement of failure modes.
10. **Reproducibility appendix**: exact model revisions and quant files,
   llama.cpp build `b10791`, Unsloth 2026.9.2, Torch 2.11.0+cu128, commit hashes.
11. **Negative results are worth keeping.** The 400-error retry loop, the context
    sizing, the glyph bug — a short "engineering lessons" subsection is genuinely
    useful to this literature and costs half a column.

---

## 6. Figures and tables to plan for

- **T1** Source corpus: publisher, edition, checksum, retrieval date.
  (Buildable now from SOURCES.md.)
- **T2** Corpus-integrity audit before and after — the C1 and private-use table.
  (Buildable now.)
- **T3** Seven categories with target and realised counts. (Pending.)
- **T4** Main results against baselines. (Pending.)
- **T5** Acceptance yield and error rate against batch size — buildable now, and
  it carries the paper's strongest methods claim. Category 3: 20 pairs → 0 tier
  errors, 83% yield; 323 pairs → 15 tier errors, 75% yield degrading across the
  run. Category 4: 20 pairs → 0 unsupported numbers, 54% yield.
- **T6** Gate-by-gate rejection counts per run, showing which failure mode each
  gate actually caught in practice rather than in principle.
- **F1** Pipeline diagram: PDFs to text to chunks to generation to validation to
  LoRA to GGUF, with the human checkpoint drawn in — the checkpoint is part of
  the method, not an omission.
- **F2** Confusion matrix for indicated against not-indicated.
- **F3** Example transcript showing a correct refusal holding under pushback.

---

## 7. Known limitations to state plainly

- Single-country guideline scope; generalisation untested.
- UCG 2023 verified by *content*, not checksum — `library.health.go.ug` refused
  connections on 2026-09-03. Disclose this honestly.
- Synthetic training data from a 14B generator; grounded, but not
  clinician-authored.
- Category 6 (farmer, One Health) has **no Ugandan veterinary source** in the
  corpus; pairs rest on transferable principles and the withdrawal-period
  template refuses specific numbers. Say so explicitly — a reviewer will find it.
- The pilot confabulation rate above is n=20 and pre-fix.
- **Error rates measured on small batches do not hold at scale.** 20 pairs gave
  0 tier errors; 323 gave 15. Any per-category quality figure in the paper must
  state the n it was measured at, and the final figures must come from the full
  generated set, not from a pilot.
- Guards were developed against observed failures, so they are fitted to the
  failure modes we happened to see. The adversarial probes reduce this but do
  not remove it; an unseen failure class would pass every gate silently. Say so.
- No prospective clinical evaluation. Do not imply one.
