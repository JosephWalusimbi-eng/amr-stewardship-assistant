# OneAMR -- ADTC 2026 Gate 1 Submission
### An offline antimicrobial-stewardship assistant for Uganda

## Team

B.Eng Electronics and Computer Engineering students, Soroti University, Uganda.

- Joseph Walusimbi -- Team Lead
- Joshua Benjamin Ssentongo
- Jacinta Achieng Oketch

## Problem

Antimicrobial resistance (AMR) is one of the largest health threats facing
Uganda, driven less by a diagnostic gap than by a behavioural one: what
prescribers prescribe, what patients do with what they're given, and how
antibiotics are used on farms. Uganda's second National Action Plan on AMR
(NAP-AMR II, 2024/25-2028/29) is explicitly anchored under a One Health
approach for this reason -- resistant organisms move between human, animal,
and environmental reservoirs.

This submission targets three users who never get a purpose-built tool:

- **Health workers** in primary care and drug shops, who need fast,
  checkable answers on whether an antibiotic is indicated, which one, at
  what dose and AWaRe tier, and for how long -- often without reliable
  internet.
- **Patients**, whose adherence decisions (stopping early, sharing leftover
  medicine, buying partial courses) are where AMR is arguably lost in
  practice, and who are rarely addressed directly by clinical tools.
- **Farmers**, whose livestock antibiotic use (prophylactic feed-dosing,
  skipped withdrawal periods) is a real, causal driver of resistance that
  reaches humans through the food chain -- and the domain most other
  entrants in Healthcare & Medical will not touch.

A 3B-class model cannot be a competent general practitioner. It can be a
reliable, narrow-scope stewardship assistant, because the decision space
-- indicated or not, which drug, what dose, how long, when to defer -- is
small and rule-shaped. Narrow scope is the deliberate design choice that
makes small-model accuracy achievable.

## Cross-Disciplinary Integration: Healthcare x Agriculture (One Health)

Primary domain: **Healthcare & Medical**. Cross-disciplinary pairing:
**Agriculture**, specifically livestock antibiotic use.

This pairing is load-bearing, not decorative: antibiotics used
prophylactically in livestock feed, or sold without observing a
withdrawal period, produce resistant organisms that reach humans through
milk, meat, and the environment -- the same resistance problem, one causal
pathway. Uganda's own NAP-AMR II is anchored under One Health for exactly
this reason.

Because veterinary withdrawal periods vary by product, formulation, and
species, and there is no single Uganda-specific number that applies
across products, the assistant is deliberately built to **explain the
concept and defer to the product label / district veterinary officer**
for the specific figure rather than invent one. It grounds that answer in
a stable, citable reference point instead: the ~100 ng/mL maximum residue
limit for oxytetracycline in milk set by Codex Alimentarius and the EU.
A model that reasons correctly and defers on the unavailable specific
number is safer, and more honest, than one that fabricates a figure --
and fabricated numbers on food safety would be an easy and damaging
failure for a judge to catch.

## Design Decisions

- **Base model:** Qwen2.5-3B-Instruct, quantized to GGUF Q4_K_M
  (~1.9GB on disk). Chosen for strong instruction-following at small scale
  and broad llama.cpp compatibility, leaving comfortable headroom under
  the 7GB inference cap for context and KV cache.
- **No fine-tuning for Gate 1.** Given the build timeline, domain
  grounding is delivered through **retrieval, not weight updates**: a
  lightweight offline BM25 index (`rank_bm25`) over four source
  documents -- the WHO AWaRe antibiotic book, the WHO IMCI chart booklet,
  the WHO Pocket Book of Hospital Care for Children, and the
  antimicrobial-relevant sections of the Uganda Clinical Guidelines 2023
  -- retrieved at query time and injected into the prompt alongside
  source + page attribution. This keeps the safety-critical facts
  (dosing, AWaRe tier, duration) traceable to a real document rather than
  model-generated, and avoids the verification burden and hallucination
  risk of a rushed fine-tune. Fine-tuning on a larger, verified
  instruction dataset is the planned Gate 2/3 upgrade if this advances.
- **Retrieval runs outside the capped inference process.** The BM25
  index and PDF preprocessing are a one-time offline step and a
  lightweight Python retrieval call at inference time -- not part of the
  llama.cpp process the ADTC profiler measures -- so they do not count
  against the 7GB memory budget.
- **Three system-prompt personas** (`health_worker`, `patient`, `farmer`)
  rather than three separate models, so the same base model and index
  serve all three registers demonstrated in the two required test
  prompts.
- **Prefix caching for fixed, repeated system prompts.** Each persona's
  system prompt never changes between queries, so we precompute and
  persist its KV-cache state to disk once (`cache_warmup.py`) and load it
  at process startup, rather than reprocessing that fixed prefix from
  scratch on every query. Measured on the same query
  ("Adult, watery diarrhoea for 2 days, no blood, no fever...",
  health_worker persona, model-load time excluded from both), this cut
  time-to-first-token from ~68.2s cold to ~47.6s warmed -- roughly a 30%
  reduction. Both absolute numbers reflect a contended shared development
  CPU (Intel i5-8350U under WSL2), not the ADTC evaluation hardware, so
  they should be read directionally; the relative improvement from
  caching is the meaningful result. RAG-retrieved context is still
  appended to the user turn and reprocessed every query by design (see
  `app.py`'s `build_prompt()`), so even the warmed case still does real
  prefill work -- caching applies specifically to the fixed persona
  prefix, not the full prompt.
- **Runtime configuration chosen by measurement, not default.** We swept
  `n_threads` and `n_batch` combinations directly through the same
  `llama-cpp-python` runtime the app ships with (`tune.py`), constrained
  to the ADTC target hardware's 4 CPU cores, and selected the
  best-measured configuration rather than using framework defaults.
- **llama.cpp + GGUF only**, per the competition's fixed evaluation
  pipeline; no other inference runtime is used.
- **Guarding against pressure-induced sycophancy.** We tested whether the
  health-worker persona holds a correct no-antibiotic recommendation
  under direct patient pressure, using a real two-turn conversation
  through the live interactive session (not a reimplementation): an
  initial adult-diarrhoea query, followed by "The patient is insisting.
  Just give me something to be safe." The first tested version caved --
  not by naming an antibiotic outright, but by fabricating an ungrounded
  loperamide recommendation under pressure, a softer instance of the same
  fabrication failure documented above. We hardened the health-worker
  system prompt with an explicit anti-fabrication instruction (never
  introduce a drug not present in retrieved material or the prompt's own
  baseline instructions) and a requirement to name five specific danger
  signs -- blood in stool, high fever, signs of severe dehydration,
  persistent vomiting, lethargy -- rather than a vague "if it worsens."
  Retested, the model now holds its position under pushback, restates its
  reasoning briefly rather than repeating the full original answer,
  offers concrete language for the clinical conversation, and names all
  five required danger signs, without introducing any unsourced drug.
- **A regression this hardening caused, caught and fixed immediately.**
  Stacking successive "no antibiotic" and anti-fabrication instructions
  into the same system prompt over-generalized: tested against the
  classic IMCI pneumonia case (child, fast breathing, chest indrawing,
  fever) -- a presentation that clearly requires an antibiotic -- the
  model produced a self-contradictory answer, stating "no antibiotic is
  indicated" while simultaneously listing "give recommended antibiotic"
  as a bullet. The prompt had conflated two separate judgments: whether
  an antibiotic is clinically indicated at all, versus whether the model
  can confidently name a specific drug from retrieved material. We added
  an explicit instruction separating these two judgments, so uncertainty
  about the exact drug name no longer collapses into "no antibiotic
  needed." Retested: the pneumonia case now clearly states an antibiotic
  is indicated and names amoxicillin with dose and duration from the
  retrieved Uganda Clinical Guidelines passage, with no self-contradiction,
  and the original diarrhoea no-antibiotic case was re-confirmed
  unaffected.

### Alternatives considered

- **Fine-tuning a 5,000-pair synthetic dataset** (prescriber stewardship,
  AWaRe classification, patient education, livestock/agriculture) was the
  original plan and remains the intended Gate 2/3 direction. It was
  deprioritized for Gate 1 given the submission timeline and the risk of
  shipping unverified antibiotic facts baked into model weights, which is
  harder to correct after the fact than a retrieval passage.
- **A general One Health platform** covering human medicine, veterinary
  medicine, and agricultural advisory equally was rejected: splitting a
  3B model's effective capacity three ways produces three mediocre
  capabilities rather than one strong, checkable one. AMR stewardship
  stayed the spine; livestock antibiotic use is the one load-bearing
  cross-disciplinary limb, not a third parallel domain.

## A Retrieval Failure We Found and Fixed

During end-to-end testing of the exact "does this adult need an
antibiotic for watery diarrhoea" case that anchors this submission's
central claim, we found a real failure: retrieval pulled paediatric
dysentery/Shigella guidance from the WHO Pocket Book of Hospital Care for
Children -- a children's clinical manual -- instead of the correct adult
guidance (no antibiotic, ORS and zinc only). The model followed the
mismatched retrieved evidence and recommended ceftriaxone, which is
wrong for this presentation.

The root cause: plain BM25 keyword retrieval has no notion of which
patient population a passage was written for. Two of our four sources
(WHO IMCI, WHO Pocket Book) are paediatric-only; two (AWaRe, UCG) are
general. For a query like "adult, watery diarrhoea," raw keyword overlap
with the paediatric dysentery passage scored higher than the correct
general-audience passage, and nothing in the pipeline caught the mismatch
before it reached the model.

We consider this a meaningful finding, not just a bug fix: it is exactly
the kind of silent, plausible-sounding failure mode that makes clinical
AI dangerous, and it surfaced specifically because we tested the
system's actual behaviour on the headline case rather than trusting the
architecture to be correct by construction. The fix has two parts:

1. **Audience-tagged retrieval.** Each corpus source is tagged
   `pediatric` or `general` at index-build time. At query time, simple
   cues in the question (adult/child/infant, age-in-months, etc.) are
   used to filter or deprioritize audience-mismatched sources before the
   top-k passages are selected, with a safe fallback restricted to
   general-tagged sources only, never falling back to fully unfiltered
   retrieval (hardened after a second occurrence of this bug, see below).
2. **System-prompt hardening.** The health-worker persona now explicitly
   instructs the model to check whether a retrieved passage's age group
   and clinical picture actually match the patient described, and to
   disregard it if not -- retrieved context is treated as supporting
   evidence to be checked, not a mandate to be followed. Rather than a
   single hardcoded fact (which we found causes small models to
   over-generalize it as a template to unrelated presentations -- see
   below), the prompt states a general principle: match symptomatic
   management to the actual presentation, with scoped examples for both
   viral URTI and uncomplicated adult watery diarrhoea.

We verified the fix resolves the original failing case, then caught and
fixed a second-order issue from our own fix: an early version stated the
diarrhoea-specific fact directly, and the model began applying "no
antibiotic -> ORS and zinc" as a generic template to unrelated
presentations (a common-cold query received diarrhoea management). The
principle-plus-scoped-examples version above resolved this without
regressing the original fix -- both cases were re-tested together to
confirm neither broke the other.

The same bug class recurred once more during later testing: a
pediatric-specific dosing detail leaked into an adult diarrhoea answer
despite the fix above. Diagnostic logging traced this to the primary
audience filter, not the fallback path -- the Uganda Clinical Guidelines
2023 is tagged `general` as a whole document, but its Chapter 17
("Childhood Illness") is pediatric-specific content that inherited the
document-level tag and so passed the filter unflagged. We fixed this at
the source in `build_index.py` by detecting Chapter 17's running header
and overriding those chunks' audience to `pediatric`, and separately
hardened `retrieve.py`'s fallback path as defense-in-depth (described
above).

## Constraints

- **Hardware:** ADTC Standard Laptop -- 7GB inference memory ceiling, 4
  CPU cores, no discrete GPU (Intel UHD/Iris Xe integrated graphics only).
  Exceeding the memory ceiling is an automatic disqualification, which
  directly motivated choosing a 3B rather than 7B+ base model.
- **Connectivity:** the target users -- rural clinics, drug shops, and
  farms -- often lack reliable connectivity or power, which is the reason
  this is an offline, on-device tool rather than a cloud API wrapper in
  the first place.
- **Data reliability:** veterinary withdrawal periods are not
  standardized for Uganda/East Africa and vary by product; the assistant
  is designed to recognize this and defer rather than fabricate, per the
  design decision above.

## Benchmarks

Measured on development hardware (Intel i5-8350U, WSL2) inside a Linux
cgroup hard-capped at 7GB to verify behaviour under the actual ADTC
memory ceiling, not just this machine's full available RAM. Zero
out-of-memory events across all runs (5 independent profiler runs).

| Metric | Value |
|---|---|
| Tokens/sec (generation, official profiler) | 7.05 |
| First-token latency (official profiler) | 28,136 ms |
| Peak RAM (RSS) | 3,285.39 MB (~3.21 GB) |
| Peak core temperature | Not exposed by this WSL2 dev environment |
| Thermal throttling detected | False |
| Self-reported Sperf | 47.0 (100 x (7.05 / 15)) |
| Self-reported Seff | 53.1 (100 x ((7 - 3.21) / 7)) |

**A note on the runtime tuning and this benchmark:** `tune.py`'s sweep
(see Design Decisions) selected `n_threads=4, n_batch=512` as the
best-measured configuration for `app.py`'s actual chat runtime, and both
`app.py` and `cache_warmup.py` use it. However, the ADTC profiler's
throughput measurement calls `llama-bench` directly with its own
threading defaults rather than routing through `app.py`, so this
particular number is effectively unaffected by the tuning (7.05 tok/s
here vs. 7.02-7.08 across pre- and post-tuning runs -- within normal
run-to-run variance on this shared dev machine). The tuning's measured
benefit is specific to the actual interactive chat path: combined with
prefix caching, it contributes to the ~30% first-token latency
improvement documented in Design Decisions, which the profiler's
raw-model benchmark does not capture. We're reporting this discrepancy
directly rather than presenting the profiler number as if it reflects
the tuned application.

## Test Prompts

See `metadata.json` -> `test_prompts`. The two prompts are chosen to
demonstrate, in the first exchange, both halves of the submission's
central claim, and both were the specific cases used during our own
bug-finding and fixing (see "A Retrieval Failure We Found and Fixed"
above) -- these are not untested prompts picked for narrative fit, they
are the ones we actually broke, fixed, and re-verified:

1. `tp_001` -- an adult presenting with uncomplicated watery diarrhoea,
   where the correct answer is **no antibiotic**, ORS and zinc instead --
   the single most consequential stewardship judgment in Ugandan
   outpatient care, and the exact case where we found and fixed a
   real retrieval failure during development.
2. `tp_002` -- a livestock withdrawal-period question, demonstrating the
   One Health cross-disciplinary integration and the deliberate
   defer-rather-than-fabricate behaviour on unsourced specifics.
