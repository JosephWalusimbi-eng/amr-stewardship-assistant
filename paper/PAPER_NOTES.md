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

### 2026-09-04 — Infrastructure note

A 4,096-token context was insufficient: `aware_classification` prompts run about
4,576 tokens before any output, because the prompt carries a full chunk plus a
12,000-character tier table. Raised to 16,384 with q8_0 KV cache: 13,188 of
16,303 MiB VRAM on an RTX 5070 Ti. Minor, but it belongs in a reproducibility
appendix.

---

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
5. **The hand-review outcome** — how many flagged drug names were real errors.
   That number is a headline result: it quantifies what human review caught that
   automated validation did not.
6. **Inter-rater material** if any clinician reviews outputs — who, how many
   items, agreement. Without this the clinical claims stay weak.
7. **Ablations, if time allows**: with and without source grounding in the
   prompt; with and without Category 2 and 7 negative supervision (this directly
   tests the central claim); Q4_K_M against Q5_K_M against fp16 degradation.
8. **Ethics and responsible use**: not a medical device, decision *support*
   only, licence terms of each source (the AWaRe book is CC BY-NC-SA 3.0 IGO —
   check what that permits for derived training data **before** publishing the
   dataset), and a clear statement of failure modes.
9. **Reproducibility appendix**: exact model revisions and quant files,
   llama.cpp build `b10791`, Unsloth 2026.9.2, Torch 2.11.0+cu128, commit hashes.
10. **Negative results are worth keeping.** The 400-error retry loop, the context
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
- No prospective clinical evaluation. Do not imply one.
