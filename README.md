# OneAMR — Offline Antimicrobial Stewardship Assistant for Uganda

An on-device assistant that helps **health workers, patients, and farmers** use antibiotics responsibly. It runs a 3B-parameter language model entirely offline through [llama.cpp](https://github.com/ggerganov/llama.cpp) and grounds every answer in WHO and Ugandan clinical guidance retrieved from a local index.

Built for the **Africa Deep Tech Challenge (ADTC) 2026**, Laptop LLM track. Domain: Healthcare & Medical. Cross-disciplinary pairing: Agriculture.

## Why this exists

Antimicrobial resistance (AMR) is driven as much by behaviour as by diagnostics: what is prescribed, what patients do with it, and how antibiotics are used on farms. Uganda's second National Action Plan on AMR (NAP-AMR II, 2024/25–2028/29) is anchored in a **One Health** approach because resistant organisms move between people, animals, and the environment.

OneAMR targets three users who rarely get a purpose-built tool, often without reliable internet:

| User | What the assistant does |
|---|---|
| **Health worker** | Says whether an antibiotic is indicated at all, and if so which one, its AWaRe tier (Access / Watch / Reserve), dose, and duration. Says "no antibiotic" plainly when that is correct. |
| **Patient** | Explains adherence, why antibiotics may be withheld, and the risks of sharing leftovers, in plain language. Never names a drug or dose as a recommendation. |
| **Farmer** | Explains withdrawal periods and why prophylactic feed-dosing drives resistance. Defers to the product label or district veterinary officer instead of inventing a figure. |

A 3B model cannot be a general practitioner. It can be a reliable, **narrow-scope** stewardship assistant, because the decision space is small and rule-shaped. The feature that matters most is **correctly declining** to recommend an antibiotic when none is indicated.

## How it works

```
 user question
      │
      ▼
 BM25 retrieval (rag/retrieve.py) ──► top passages with source + page
      │
      ▼
 persona system prompt + retrieved context + question
      │
      ▼
 Qwen2.5-3B-Instruct (GGUF Q4_K_M) via llama.cpp, fully offline
      │
      ▼
 answer that cites its sources
```

- **Retrieval-grounded.** Dosing and tier facts come from real documents with page attribution, not from model weights, so they are traceable and correctable.
- **Three personas, one model.** `health_worker`, `patient`, and `farmer` are system prompts (`system_prompts.py`) over the same weights.
- **Prefix caching.** Each persona's system prompt is byte-identical across calls, and `cache_warmup.py` pre-computes its KV-cache so cold starts skip reprocessing it. Retrieved context goes in the user turn to keep that prefix reusable.
- **Tuned runtime.** `tune.py` sweeps thread and batch settings (never above the 4 cores of the target laptop) and records the results in `tune_results.json`.
- **Safe fallbacks.** If retrieved context does not cover a case, the assistant says so and recommends escalation.

### Knowledge sources

- WHO AWaRe antibiotic book
- WHO IMCI chart booklet
- WHO Pocket Book of Hospital Care for Children
- Uganda Clinical Guidelines 2023 (antimicrobial-relevant chapters only)

Extracted text and lookup tables are in `source_material/`, with provenance in `source_material/SOURCES.md`.

## Quick start

**Requirements:** Python 3.10+, `curl`, about 2 GB free disk for the model, and 4 CPU cores. No GPU needed.

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download the model (~1.9 GB, resumes if interrupted)
bash download_model.sh

# 3. (Optional) Warm the per-persona prefix cache for faster first responses
python cache_warmup.py

# 4. Run
python app.py --persona patient
```

The retrieval index (`rag/index.pkl`) is committed, so you can skip rebuilding it. To rebuild it from the source PDFs, place them in `corpus/` and run `python rag/build_index.py`. The PDFs are git-ignored.

### Usage

```bash
# Interactive chat
python app.py --persona health_worker

# Single question, then exit
python app.py --persona farmer --prompt "I treated my cow with oxytetracycline. How long before I can sell the milk?"

# Browser UI
streamlit run streamlit_app.py
```

| Flag | Default | Meaning |
|---|---|---|
| `--persona` | `patient` | `health_worker`, `patient`, or `farmer` |
| `--prompt` | none | Answer one prompt and exit |
| `--n_ctx` | `4096` | Context window size |
| `--n_threads` | `4` | CPU threads |

### Example prompts

- *Health worker:* "Adult, watery diarrhoea for 2 days, no blood, no fever. Which antibiotic should I give?" The expected answer is **no antibiotic**, with oral rehydration and zinc instead.
- *Farmer:* "I treated my dairy cow with oxytetracycline for an infection. How long before I can sell the milk?" The expected answer explains the concept, cites the ~100 ng/mL Codex/EU residue limit, and defers to the label or district veterinary officer for the exact period.

## Benchmarks

Measured with the official ADTC profiler on development hardware (Intel i5-8350U, WSL2) inside a 7 GB memory cap. Five runs, no out-of-memory events.

| Metric | Value |
|---|---|
| Generation throughput | 7.05 tokens/s |
| Peak RAM | about 3.2 GB |
| Thermal throttling | none detected |

`REPORT.md` explains how these numbers relate to the tuned chat path and what the profiler does not capture.

## Fine-tuning dataset pipeline

The repo also holds the pipeline for building a verified instruction dataset to fine-tune the model in a later stage. Antibiotic facts in training data must be checkable, so every generated record is tied to a source passage and passes automated guards before it is kept.

- `scripts/` extracts and chunks the source documents, builds drug and dose-line reference lists, and generates grounded question and answer pairs against a locally served open model.
- `generation_guards.py` and `c2_exception_guard.py` reject pairs that state unsourced numbers or smuggle in exceptions the source does not support.
- `scripts/probes/` tests those guards adversarially.
- `paper/PAPER_NOTES.md` keeps running notes for a planned journal paper.

## Repository layout

```
├── app.py                  Chat CLI: retrieval, prompt building, llama.cpp inference
├── streamlit_app.py        Browser UI wrapping app.py
├── system_prompts.py       The three persona prompts
├── cache_warmup.py         Pre-computes per-persona KV-cache prefixes
├── tune.py                 Runtime config sweep (threads, batch size)
├── download_model.sh       Downloads the GGUF model into model/
├── metadata.json           ADTC submission metadata and test prompts
├── REPORT.md               Technical write-up: design, constraints, benchmarks
├── rag/                    PDF extraction, BM25 index build and retrieval
├── scripts/                Dataset pipeline and adversarial probes
├── source_material/        Extracted source text, chunks, lookup tables
├── paper/                  Working notes for the paper
└── model/                  Downloaded .gguf weights (git-ignored)
```

## Design constraints

- **Hardware target:** ADTC standard laptop, 7 GB inference memory ceiling, 4 CPU cores, integrated graphics only. This is why the base model is 3B rather than 7B or larger.
- **Fully offline:** zero network calls during inference. The only network step is the one-time model download.
- **llama.cpp and GGUF only**, per the competition's evaluation pipeline.

See [`REPORT.md`](REPORT.md) for the full rationale and alternatives considered.

## Disclaimer

OneAMR is a decision-support and education tool, **not** a substitute for clinical judgement, a prescriber's examination, or a veterinarian's advice. Verify doses and durations against current national guidelines before acting.

## Team

B.Eng Electronics and Computer Engineering students, Soroti University, Uganda.

- Joseph Walusimbi (team lead)
- Joshua Benjamin Ssentongo
- Jacinta Achieng Oketch

## License

Licensed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE).
