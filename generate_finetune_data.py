"""
Generates instruction-tuning data for the AMR Stewardship Assistant by
running a fixed set of prompts through the existing app.py logic (retrieval
+ prompt construction via build_prompt, generation via the same Llama
config/params app.py uses) and saving each system/user/assistant turn as a
JSONL line.

Reuses app.py directly -- no duplication of retrieval or prompt-building
logic. app.py itself is not modified.

Usage:
    python generate_finetune_data.py
"""

import json
import pickle
import sys
import time
from pathlib import Path

from llama_cpp import Llama

from app import SYSTEM_PROMPTS, build_prompt, MODEL_PATH, CACHE_DIR

OUTPUT_PATH = Path(__file__).parent / "finetune_data.jsonl"

# Prompts grouped by persona so each group can run consecutively and reuse
# the warmed system-prompt prefix (see build_prompt's docstring on prefix
# caching). Within health_worker, prompts are further tagged by category
# purely for readability here -- all use the same persona/system prompt.

ANTIBIOTIC_INDICATED_PROMPTS = [
    "A 2-year-old child presents with cough, fast breathing (52 breaths/min), and chest indrawing. No fever. What is the management?",
    "A 34-year-old woman has dysuria, urinary frequency, and suprapubic pain for 2 days, no fever, no flank pain. Is an antibiotic indicated?",
    "A 6-month-old infant has a fever of 39.5C, a bulging fontanelle, and neck stiffness. What is the appropriate management?",
    "A 45-year-old man presents with a red, warm, swollen, tender area on his lower leg with fever, consistent with cellulitis. What antibiotic should be used?",
    "A 28-year-old pregnant woman at 32 weeks has fever, foul-smelling vaginal discharge, and uterine tenderness. What is the management?",
    "A 5-year-old child has ear pain, fever, and a bulging red tympanic membrane on otoscopy, consistent with acute otitis media. Is an antibiotic needed?",
    "A 60-year-old man with diabetes has a foot ulcer with surrounding cellulitis, purulent discharge, and fever. What is the appropriate antibiotic approach?",
    "A 3-year-old child has bloody diarrhoea with mucus, fever, and abdominal cramping for 2 days, consistent with dysentery. What is the management?",
    "A 25-year-old man has a deep laceration from a dog bite on his hand, with surrounding redness and swelling 24 hours later. Is an antibiotic indicated?",
    "A newborn, 5 days old, has fever, poor feeding, and lethargy. What is the appropriate management for suspected neonatal sepsis?",
]

ANTIBIOTIC_NOT_INDICATED_PROMPTS = [
    "A 30-year-old man has a runny nose, mild sore throat, and cough for 2 days, no fever, consistent with a viral URTI. Should he receive an antibiotic?",
    "A 4-year-old child has watery diarrhoea without blood or fever for 1 day, is alert and drinking well. Is an antibiotic indicated?",
    "A 22-year-old woman has a mild sore throat with no fever, no tonsillar exudate, and no cervical lymphadenopathy. Should she be given an antibiotic?",
    "A 35-year-old man has nasal congestion and facial pressure for 3 days, no fever, consistent with viral sinusitis. Is an antibiotic needed?",
    "A 6-year-old child has a mild cough and clear nasal discharge for 2 days, normal breathing rate, no chest indrawing. What is the management?",
    "A 40-year-old woman has mild watery diarrhoea for half a day after eating street food, no blood, no fever, mild dehydration. Should she be given an antibiotic?",
    "A 50-year-old man has a common cold with sneezing and mild cough, no fever, for 1 day. Should he receive an antibiotic?",
    "A 2-year-old child has loose, watery stools 3 times today, no blood, no fever, still playful and drinking. What is the management?",
    "A 27-year-old woman has a scratchy throat and mild cough for 1 day with no fever, no difficulty swallowing. Is an antibiotic indicated?",
]

AWARE_CLASSIFICATION_PROMPTS = [
    "Is ceftriaxone Access, Watch, or Reserve?",
    "Is amoxicillin Access, Watch, or Reserve?",
    "Is ciprofloxacin Access, Watch, or Reserve?",
    "Is metronidazole Access, Watch, or Reserve?",
    "Is azithromycin Access, Watch, or Reserve?",
    "Is meropenem Access, Watch, or Reserve?",
    "Is doxycycline Access, Watch, or Reserve?",
    "Is gentamicin Access, Watch, or Reserve?",
    "Why is it important to prefer Access-category antibiotics over Watch-category antibiotics?",
]

PATIENT_PROMPTS = [
    "Why do I need to finish all my antibiotic pills even though I feel better after 2 days?",
    "Can I share my leftover antibiotics with my neighbor who has similar symptoms?",
    "My doctor said I don't need antibiotics for my cold. Why not, since I feel really sick?",
    "Is it okay to buy just a few antibiotic tablets from the pharmacy instead of a full course?",
    "What happens if I stop taking my antibiotics as soon as my fever goes away?",
    "Why did the health worker refuse to give my child antibiotics for diarrhoea?",
    "Can I take antibiotics left over from my last illness for a new cough?",
    "What is antibiotic resistance and why should I care about it?",
    "My child's fever is not fully gone after 2 days of antibiotics. Should I get a stronger antibiotic myself?",
]

FARMER_PROMPTS = [
    "How long after giving my cow antibiotics can I sell its milk?",
    "Why can't I use antibiotics in chicken feed all the time to keep my flock healthy?",
    "What is a withdrawal period and why does it matter for my livestock?",
    "Can resistant bacteria from my animals affect my family's health?",
    "My veterinary drug label doesn't state a withdrawal period. What should I do?",
    "Why do I need to consult the district veterinary officer before treating sick cattle with antibiotics?",
    "Is it safe to eat meat from an animal I treated with antibiotics yesterday?",
    "What is the maximum residue limit for antibiotics in milk, and why does it matter?",
    "Why does routine antibiotic use in healthy livestock contribute to resistance?",
]

# (persona, prompts) groups, in the order they'll be generated. Grouping by
# persona keeps the system-prompt prefix warm across consecutive calls.
PERSONA_GROUPS = [
    ("health_worker", ANTIBIOTIC_INDICATED_PROMPTS + ANTIBIOTIC_NOT_INDICATED_PROMPTS + AWARE_CLASSIFICATION_PROMPTS),
    ("patient", PATIENT_PROMPTS),
    ("farmer", FARMER_PROMPTS),
]


def load_persona_cache(llm: Llama, persona: str) -> None:
    cache_path = CACHE_DIR / f"{persona}.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                state = pickle.load(f)
            llm.load_state(state)
            print(f"[cache] loaded warmed prefix for persona '{persona}'", file=sys.stderr)
        except Exception as e:
            print(f"[cache] could not load {cache_path.name}, continuing without it: {e}", file=sys.stderr)
    else:
        print(f"[cache] no warmed cache for '{persona}', continuing without it.", file=sys.stderr)


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Run `bash download_model.sh` first.", file=sys.stderr)
        sys.exit(1)

    total_prompts = sum(len(prompts) for _, prompts in PERSONA_GROUPS)
    print(f"Loading model ({total_prompts} prompts queued)...", file=sys.stderr)

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=4096,
        n_threads=8,
        n_batch=512,
        verbose=False,
    )

    done = 0
    t_start = time.time()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for persona, prompts in PERSONA_GROUPS:
            load_persona_cache(llm, persona)

            for prompt in prompts:
                done += 1
                t0 = time.time()
                messages = build_prompt(persona, prompt, history=None)

                completion = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=512,
                    temperature=0.2,
                    stream=False,
                )
                answer = completion["choices"][0]["message"]["content"]

                record = {
                    "messages": messages + [{"role": "assistant", "content": answer}]
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                elapsed = time.time() - t0
                print(
                    f"[{done}/{total_prompts}] persona={persona} ({elapsed:.1f}s) "
                    f"prompt={prompt[:60]!r}",
                    file=sys.stderr,
                )

    total_elapsed = time.time() - t_start
    print(
        f"Done. Wrote {done} examples to {OUTPUT_PATH} in {total_elapsed / 60:.1f} min.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
