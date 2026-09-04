"""
STEP 3 -- grounded pair generation. Runs on the GPU box.

Talks to a locally served open model over an OpenAI-compatible HTTP endpoint,
which is what `vllm serve` exposes:

    # on the GPU box
    pip install vllm
    vllm serve Qwen/Qwen2.5-72B-Instruct \
        --tensor-parallel-size 2 --max-model-len 16384 --port 8000

    python3 scripts/generate_pairs.py \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-72B-Instruct \
        --category all

Every call carries source chunks. Nothing is generated from model memory -- see
amr_dataset_generation_prompts.md section 0.

Batches of 50, with a spot-check dump after each batch. The run stops on a
category if a batch's spot-check sample contains a number that is not in the
attached chunk, rather than continuing to burn GPU time on bad data.

Resumable: existing raw/<category>.jsonl is read on start and its chunk_ids are
skipped.
"""
import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SM = ROOT / "source_material"
CHUNKS = SM / "chunks"
RAW = ROOT / "raw"

sys.path.insert(0, str(ROOT))
from system_prompts import SYSTEM_PROMPTS  # noqa: E402
from generation_guards import Guards, primary_drug  # noqa: E402
from prepare_dataset import DRUG_ALIASES, norm_drug  # noqa: E402

# Verbatim from the AWaRe book, chunks/aware_classification.txt. Supplied to the
# model as the ONLY permitted basis for a "why is it in this group" answer.
#
# Note what these say: they are properties of the GROUP. The 2026-09-04 pilot
# produced "amoxicillin is a narrow-spectrum antibiotic" by collapsing the
# Access definition onto a member drug -- amoxicillin is a broad-spectrum
# aminopenicillin that sits in the Access group. Hence rule (c) below.
TIER_RATIONALE = """AWaRe GROUP DEFINITIONS (verbatim, WHO AWaRe antibiotic book, ch.2).
These are the ONLY reasons you may give for why a group exists:

Access: "Access antibiotics have a narrow spectrum of activity, lower cost, a
good safety profile and generally low resistance potential. They are often
recommended as empiric first- or second-choice treatment options for common
infections."

Watch: "Watch antibiotics are broader-spectrum antibiotics, generally with
higher costs and are recommended only as first-choice options for patients with
more severe clinical presentations or for infections where the causative
pathogens are more likely to be resistant to Access antibiotics."

Reserve: "Reserve antibiotics are last-choice antibiotics used to treat
multidrug-resistant infections."
"""

GEN_SYSTEM = """You are building a supervised fine-tuning dataset for an antimicrobial
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
5. Output valid JSON only, matching the requested schema. No commentary."""

# category -> (register, topic selector, per-call pair count, instruction)
EXCLUDE_CLINICAL = {"aware_classification", "aware_classification_table",
                    "formulary_antibacterials", "formulary_other_anti_infectives",
                    "reserve_antibiotics", "dosing_guidance_adults",
                    "dosing_guidance_children", "antibiotic_allergy"}
NO_ANTIBIOTIC_TOPICS = {"viral_infections_no_antibiotic", "acute_diarrhoea",
                        "acute_bronchitis", "pharyngitis", "acute_sinusitis",
                        "acute_otitis_media", "copd_exacerbation"}
AWARE_TOPICS = {"aware_classification", "aware_classification_table",
                "reserve_antibiotics", "formulary_antibacterials"}
DOSING_TOPICS_DOC = {"who_aware_book_2022", "emhslu_2023"}

INSTRUCTIONS = {
    "antibiotic_indicated": """Write {n} training examples where an antibiotic IS indicated.
Each: a specific clinical vignette (vary age, weight, sex, severity, comorbidity, and
facility level HC2/HC4/hospital), then an answer that states in order:
(a) that an antibiotic is indicated and why, (b) the first-choice agent AND its AWaRe
category, (c) the dose, (d) the duration, (e) what to do on failure or deterioration.
If the passage gives weight bands, the vignette's weight must fall inside a stated band
and the dose must be that band's dose, copied exactly.
The AWaRe category must come from the AWaRe CLASSIFICATION TABLE below if one is
attached; the treatment tables encode tiers as colour, which is not in the text.""",

    "no_antibiotic_indicated": """Write {n} training examples where an antibiotic is NOT indicated.
Each answer must: (1) say plainly that no antibiotic is indicated; (2) give the reason AS THE
SOURCE STATES IT, quoting or closely paraphrasing the guideline rather than substituting
general medical intuition; (3) give symptomatic management matched to THAT presentation;
(4) name SPECIFIC danger signs that would change the assessment. For any diarrhoea case the
danger signs must include blood in the stool, high fever, signs of severe dehydration (dry
mouth, no tears, little or no urination), persistent vomiting, and lethargy.
Never name an antibiotic in a recommending frame. There is no concession antibiotic.
"If it gets worse" is not an acceptable danger sign.""",

    "aware_classification": """Write {n} training examples about AWaRe categories.
Use ONLY the classification table below for tier assignments.

(a) TIER STATEMENTS are the core of this category. Ask about one drug at a time and
state its group. Vary the drug: do NOT keep returning to amoxicillin and the other
few obvious ones. Work across the classification table, including less familiar
agents.
If a drug's tier differs by route (fosfomycin, minocycline), the answer MUST give both
routes rather than one tier.
If a drug is not in the table at all (isoniazid, ethambutol, pyrazinamide, dapsone,
clofazimine, bedaquiline, delamanid), the correct answer is that AWaRe does not classify
anti-TB and antileprosy agents -- never guess a tier.

(b) AVAILABILITY IN UGANDA. Say something about whether a drug is on the national
formulary ONLY if an EMHSLU passage is attached to this call AND that drug is named
in it. Otherwise state that the material provided does not confirm its availability
in Uganda, and stop there. Do NOT assert that a drug IS available and do NOT assert
that it is NOT available -- an EMHSLU extract is a fragment, so a drug missing from
it proves nothing either way. Inventing formulary status is the single most common
error in this category.

(c) WHY A GROUP EXISTS may be answered only by quoting or closely paraphrasing the
AWaRe GROUP DEFINITIONS block. Those describe the GROUP. Never restate a group
property as a property of an individual drug: do not write "amoxicillin is a
narrow-spectrum antibiotic" because the Access definition mentions narrow spectrum.
Write "amoxicillin is in the Access group; Access antibiotics are described as ...".
Never invent a pharmacological reason (spectrum, cost, resistance risk, safety) for a
NAMED drug unless the attached passage states it of that drug.

(d) IF A REFERENCE PASSAGE IS ATTACHED and it does not mention the drug being asked
about, say so explicitly -- "the passage provided does not cover X" -- and answer only
the part you can support from the classification table. Never answer from memory as
though the passage supported you.""",

    "dosing_duration": """Write {n} dose-and-duration training examples for the target drug.

The passages above are extracted DOSE LINES for that one drug, each carrying its
source document and page. They are the entire permitted basis for every number.

EVERY number you write -- dose, frequency, duration -- must appear verbatim in
those lines. Copy it, do not restate it in different units: if the line says
"50,000 IU/kg" do not write "50000 units per kg", and if it says "every 4-6
hours" do not write "four times a day". A number that is not in the lines will
be rejected, and rightly: these lines were checked against the source PDF pages
and the model's memory was not.

Do NOT convert, scale, round, or combine numbers. Do not compute a total daily
dose from a per-dose figure. Do not extrapolate to a weight, age or route the
lines do not cover.

If the question would need something the lines do not give -- a paediatric dose
when only adult lines are present, a route that is not stated, a duration that
is absent -- write the example as a refusal: state precisely what the source
does give, state what it does not, and decline to extrapolate. That is a
valuable training example, not a failure, and this category needs them.

State route and total duration whenever the lines support it. Vary the question
shape: a direct dose question, a weight-band question, a "how long" question, a
renal or paediatric question the lines cannot answer.""",

    "patient_education": """Write {n} patient-facing training examples in plain language,
120-250 words, no jargon, warm and direct. Topics: finishing a course, why antibiotics do
nothing for viruses, not sharing or saving leftovers, not buying part-courses, what
resistance means for their family.
Never state a dose -- patients do not dose themselves from this model.
Explain rather than lecture. Give concrete "come back if" advice with named signs.""",

    "farmer_one_health": """Write {n} farmer-facing One Health training examples.
CRITICAL: there is NO livestock source document in this corpus, so the answers must contain
NO NUMBERS AT ALL -- no withdrawal period in days, no maximum residue limit, no veterinary
dose, no drug-specific figure. Any number would be fabricated.
Ground the answers in transferable reasoning: how resistant bacteria move between animals,
people and the environment; why constant low-dose antibiotics in healthy animals select for
resistance; why good husbandry beats prophylactic dosing; who to ask (product label,
veterinarian, district veterinary officer).
Warm, practical, respectful of the farmer's judgement.""",

    "refusal_escalation": """Write {n} refusal and escalation training examples, mixing these
sub-types roughly evenly:
7a OUT OF GROUNDING -- the question is not covered; say what is known, say plainly the
   specific answer is not in the available guidance, escalate.
7b PUSHBACK -- 5 turns [user, assistant, user, assistant]: the assistant correctly says no
   antibiotic is needed; the user pushes back WITHOUT new clinical information; the second
   answer restates the reasoning in ONE sentence, gives one concrete line the clinician can
   say to the patient, names specific danger signs, and offers escalation. It must not name
   any antibiotic.
7c MISMATCHED PASSAGE -- a passage that does not fit the patient (paediatric dosing for an
   adult, bloody-diarrhoea guidance for watery); the answer says so explicitly and refuses
   to apply it.
7d WITHDRAWAL PERIOD -- a farmer asks how long before selling milk or meat. The answer must
   NEVER state a number of days, a range, or a hedged figure ("usually about a week"). It
   must: acknowledge the question and why residues matter; state that the period depends on
   the specific product, formulation, dose and species and that you cannot give a number;
   point to the product label or package insert, then the veterinarian, then the district
   veterinary officer; say not to sell or consume milk or meat until the labelled period has
   passed and to keep a written record of drug, date, dose and animal.""",
}

SCHEMA = """Return a JSON array of exactly {n} objects, nothing else:
[{{"question": "...", "answer": "..."}}]
For multi-turn pushback examples use:
[{{"turns": ["user text", "assistant text", "user text", "assistant text"]}}]"""


def load_manifest():
    rows = [json.loads(l) for l in open(CHUNKS / "_manifest.jsonl", encoding="utf-8")]
    table = CHUNKS / "aware_classification_table.txt"
    if table.exists():
        text = table.read_text(encoding="utf-8")
        for i in range(0, len(text), 6000):
            piece = text[i:i + 6000]
            if len(piece) > 400:
                rows.append({"chunk_id": "aware_classification_table::%d" % (i // 6000),
                             "topic": "aware_classification_table",
                             "doc": "who_aware_classification_2023",
                             "doc_label": "WHO AWaRe classification 2023 "
                                          "(WHO-MHP-HPS-EML-2023.04, web annex C)",
                             "section": "full classification", "page_start": 0, "page_end": 0,
                             "audience": "general", "chars": len(piece), "text": piece})
    return rows


def pick_chunks(manifest, category):
    if category == "farmer_one_health":
        return [None]                                  # no source document exists
    if category == "aware_classification":
        return [c for c in manifest if c["topic"] in AWARE_TOPICS]
    if category == "no_antibiotic_indicated":
        return [c for c in manifest if c["topic"] in NO_ANTIBIOTIC_TOPICS]
    if category == "dosing_duration":
        return [c for c in manifest
                if c["doc"] in DOSING_TOPICS_DOC and c["topic"] not in {"aware_classification"}]
    if category == "patient_education":
        return [c for c in manifest
                if c["topic"] in NO_ANTIBIOTIC_TOPICS | {"aware_classification"}]
    if category == "refusal_escalation":
        return [c for c in manifest if c["topic"] not in EXCLUDE_CLINICAL] + [None] * 30
    return [c for c in manifest if c["topic"] not in EXCLUDE_CLINICAL]


# Categories whose examples are about a specific named drug, and which are
# therefore worth planning around a target drug rather than around a passage.
DRUG_TARGETED = {"aware_classification", "antibiotic_indicated", "dosing_duration"}


def load_tier_rows(guards):
    """Normalised drug key -> its row(s) from the AWaRe classification table.

    The table has 366 rows. Handing the model all of them and asking it to look
    one up does not work reliably: one batch answered "meropenem is Reserve"
    with `Meropenem | Carbapenems | J01DH02 | Watch | Yes` in its own prompt,
    and another declared talampicillin and penamecillin unclassified when both
    are Access and both rows were present. Truncating the table to 12,000
    characters made it worse, silently hiding everything alphabetically later.

    So the prompt now carries only the rows for the drugs a call is about. The
    lookup stops being a needle-in-a-haystack task the model can fail.
    """
    path = CHUNKS / "aware_classification_table.txt"
    rows, header = {}, ""
    if not path.exists():
        return rows, header
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Antibiotic |"):
            header = line
            continue
        if "|" not in line:
            continue
        name = line.split("|", 1)[0].strip()
        if not name:
            continue
        key = norm_drug(re.sub(r"_(IV|oral)$", "", name))
        rows.setdefault(key, []).append(line.strip())
    return rows, header


def tier_excerpt(targets, tier_rows, header):
    """The classification rows for this call's drugs, and nothing else."""
    if not targets:
        return ""
    out = [header] if header else []
    missing = []
    for d in targets:
        key = DRUG_ALIASES.get(d, d)
        hits = tier_rows.get(key) or tier_rows.get(d)
        if hits:
            out.extend(hits)
        else:
            missing.append(d)
    body = "\n".join(out)
    if missing:
        body += ("\n\nNOT PRESENT in the AWaRe classification at all: %s. For these the "
                 "correct answer is that AWaRe does not classify the agent -- do not "
                 "guess a tier." % ", ".join(missing))
    return ("\nAWaRe CLASSIFICATION -- the authoritative rows for this call "
            "(tier is the 4th column):\n" + body + "\n")


def load_dose_lines():
    """Verified per-drug dose lines, keyed by drug. Category 4's whole source.

    Only lines with no flags are offered: flags cover table-shaped rows, lines
    with no regimen context, and -- most importantly -- lines where the dose
    belongs to a neighbouring drug rather than this one.
    """
    path = SM / "dose_lines.json"
    if not path.exists():
        return {}, False
    data = json.loads(path.read_text(encoding="utf-8"))
    usable = {}
    for key, recs in data.get("drugs", {}).items():
        clean = [r for r in recs if not r["flags"] and r["attributed_doses"]]
        if clean:
            usable[key] = clean
    return usable, bool(data.get("verified"))


def dose_chunk(drug, recs, rng, k=6):
    """A synthetic passage of just this drug's dose lines, with page citations.

    Shaped like a manifest chunk so it flows through to_record, the grounding
    guard and the dose gate unchanged.
    """
    picked = rng.sample(recs, min(k, len(recs)))
    picked.sort(key=lambda r: (r["doc"], r["page"]))
    body = "\n".join("[%s p.%d] %s" % (r["doc_label"], r["page"], r["text"]) for r in picked)
    pages = [r["page"] for r in picked]
    docs = sorted({r["doc_label"] for r in picked})
    return {
        "chunk_id": "dose_lines::%s::%s" % (drug, "-".join(str(p) for p in pages[:4])),
        "topic": "dose_lines",
        "doc": "dose_lines",
        "doc_label": " + ".join(docs),
        "section": "extracted dose lines for %s" % drug,
        "page_start": min(pages), "page_end": max(pages),
        "audience": "general",
        "chars": len(body),
        "text": body,
    }


def build_drug_index(manifest, guards):
    """drug key -> chunk_ids that actually contain that drug, and the reverse.

    Selection used to run passage-first: pick a chunk, let the model choose
    whatever drugs it liked, then check afterwards whether the passage happened
    to support them. It usually did not -- 36 of 47 rejections in the first
    gated run were a passage that did not name the drug asked about, or an
    availability claim with no formulary passage attached. Both are decidable
    before spending a generation call, so decide them before.
    """
    drug_to_chunks, chunk_to_drugs = defaultdict(list), {}
    for c in manifest:
        if c is None:
            continue
        drugs = guards.drug_names(c["text"])
        chunk_to_drugs[c["chunk_id"]] = drugs
        for d in drugs:
            drug_to_chunks[d].append(c["chunk_id"])
    return drug_to_chunks, chunk_to_drugs


def plan_call(category, rng, pool, drug_to_chunks, chunk_to_drugs, by_id,
              formulary_ids, seen_drugs, drug_cap, n, dose_lines=None,
              classified=frozenset(), unclassified_fraction=0.15):
    """Choose target drugs and a passage guaranteed to contain them.

    Returns (chunks, targets). Falls back to the old passage-first behaviour for
    categories that are not about one named drug.
    """
    if category == "dosing_duration" and dose_lines:
        eligible = sorted(d for d in dose_lines if seen_drugs[d] < drug_cap)
        if eligible:
            rng.shuffle(eligible)
            eligible.sort(key=lambda d: seen_drugs[d])
            lead = eligible[0]
            return [dose_chunk(lead, dose_lines[lead], rng)], [lead]

    if category not in DRUG_TARGETED:
        return [rng.choice(pool)], []

    pool_ids = {c["chunk_id"] for c in pool if c}
    eligible = sorted(d for d, ids in drug_to_chunks.items()
                      if seen_drugs[d] < drug_cap and any(i in pool_ids for i in ids))
    if not eligible:
        return [rng.choice(pool)], []

    # Most examples should be about drugs AWaRe actually classifies. A minority
    # deliberately are not -- "AWaRe does not classify anti-TB agents" is a
    # wanted answer, and the spec asks for roughly 80 of 550 such pairs -- but
    # left unweighted the pool drifts onto whatever the formulary happens to
    # list, including agents that are not antibacterials at all.
    in_aware = [d for d in eligible if d in classified] if classified else eligible
    out_aware = [d for d in eligible if d not in classified] if classified else []
    bucket = out_aware if (out_aware and rng.random() < unclassified_fraction) else (in_aware or eligible)

    # Prefer drugs we have used least, so coverage spreads across the table
    # instead of pooling on the handful the model finds most salient.
    rng.shuffle(bucket)
    bucket.sort(key=lambda d: seen_drugs[d])
    lead = bucket[0]

    # A passage that names the lead drug, preferring one that also covers other
    # under-used drugs so a single call can carry several examples.
    cand_ids = [i for i in drug_to_chunks[lead] if i in pool_ids]

    # For tier questions, prefer a passage that carries tiers. The formulary
    # chunks are long and mention many drugs, so ranking by coverage alone kept
    # picking them -- and a pair whose passage is the formulary answers the tier
    # from memory. One batch put "Meropenem | ... | Watch" in the prompt via the
    # tier table and still said Reserve; when the passage itself carries the
    # tier, the model has to disagree with the text in front of it.
    def rank(i):
        c = by_id[i]
        tier_bearing = 0 if "EMHSLU" in (c.get("doc_label") or "") else 1
        covered = len([d for d in chunk_to_drugs.get(i, []) if seen_drugs[d] < drug_cap])
        return (-tier_bearing, -covered) if category == "aware_classification" else (0, -covered)

    cand_ids.sort(key=rank)
    chunk = by_id[cand_ids[0]]

    same_bucket = set(bucket)
    targets = [d for d in chunk_to_drugs.get(chunk["chunk_id"], [])
               if seen_drugs[d] < drug_cap and (d in same_bucket or d == lead)]
    targets.sort(key=lambda d: seen_drugs[d])
    targets = targets[:n] or [lead]

    chunks = [chunk]
    # Attach the formulary extract naming these drugs, so availability is either
    # answerable from source or visibly absent -- rather than invented.
    for fid in formulary_ids:
        if fid == chunk["chunk_id"]:
            break                       # the passage already IS the formulary
        if any(d in chunk_to_drugs.get(fid, []) for d in targets):
            chunks.append(by_id[fid])
            break
    return chunks, targets


def source_label(chunk):
    if chunk is None:
        return "no source chunk in corpus -- refusal is the grounded response"
    return "%s.txt :: %s :: %s :: p.%s-%s" % (chunk["topic"], chunk["doc_label"],
                                              chunk["section"], chunk["page_start"],
                                              chunk["page_end"])


def build_user_prompt(category, chunks, n, tier_table, targets=()):
    parts = []
    real = [c for c in chunks if c]
    if real:
        parts.append("SOURCE PASSAGES (the only material you may draw clinical content from):\n")
        for c in real:
            parts.append("--- [%s | %s | p.%s-%s] ---\n%s\n"
                         % (c["doc_label"], c["section"], c["page_start"], c["page_end"], c["text"]))
    else:
        parts.append("SOURCE PASSAGES: none. This corpus contains no document covering this "
                     "question. The answer must therefore contain no clinical numbers.\n")
    if tier_table and category in ("antibiotic_indicated", "aware_classification"):
        # `tier_table` is the focused excerpt when the call has target drugs,
        # and the (truncated) full table only as a fallback.
        parts.append(tier_table if tier_table.startswith("\nAWaRe CLASSIFICATION --")
                     else "\nAWaRe CLASSIFICATION TABLE (authoritative for tiers):\n"
                          + tier_table + "\n")
    if category == "aware_classification":
        parts.append("\n" + TIER_RATIONALE)
    if real:
        parts.append(
            "\nGROUNDING RULE FOR THIS CALL: the source passages above are the only "
            "material you may treat as evidence. If a passage does not mention the drug "
            "or condition an example asks about, the answer must say so in plain words "
            "rather than answering as if it did. An answer that sounds confident while "
            "resting on a passage that never mentioned its subject is the worst outcome "
            "here, because it reads as correct.\n")
    if targets:
        has_formulary = any("EMHSLU" in (c.get("doc_label") or "") for c in real)
        parts.append(
            "\nTARGET DRUGS FOR THIS CALL: %s.\n"
            "Write your %d example(s) about these drugs, one drug per example, in this "
            "order. Every one of them is named in the source passages above, so you can "
            "answer from the passages rather than from memory. Do not substitute a "
            "different drug.\n" % (", ".join(targets), n))
        parts.append(
            ("An EMHSLU formulary extract IS attached. You may state a drug's Uganda "
             "formulary status only where that extract names the drug; if it does not, "
             "say the material provided does not confirm it.\n")
            if has_formulary else
            ("NO EMHSLU formulary extract is attached to this call, so you cannot know "
             "any drug's Uganda availability. Do not claim a drug is available and do not "
             "claim it is unavailable. If availability comes up, say the material "
             "provided does not confirm it.\n"))
    parts.append("\nTASK\n" + INSTRUCTIONS[category].format(n=n))
    parts.append("\n\n" + SCHEMA.format(n=n))
    return "\n".join(parts)


class CallFailed(RuntimeError):
    """HTTP failure that carries the server's response body.

    requests' raise_for_status() reports only the status line, which is how a
    run once spent fifteen minutes emitting "400 Client Error" with no hint that
    the real cause was a prompt longer than the server's context window.
    """


def call_model(session, base_url, model, user_prompt, temperature, max_tokens, timeout):
    r = session.post(base_url.rstrip("/") + "/chat/completions", timeout=timeout, json={
        "model": model,
        "messages": [{"role": "system", "content": GEN_SYSTEM},
                     {"role": "user", "content": user_prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    })
    if r.status_code >= 400:
        raise CallFailed("HTTP %d from %s\n   prompt was %d chars (~%d tokens)\n   body: %s"
                         % (r.status_code, r.url, len(user_prompt), len(user_prompt) // 4,
                            r.text[:800]))
    return r.json()["choices"][0]["message"]["content"]


def parse_pairs(text):
    """Pull the JSON array out of a response that may be fenced or prefaced."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


def to_record(item, category, register, chunks, context_mode):
    turns = item.get("turns")
    if not turns:
        q, a = item.get("question"), item.get("answer")
        if not q or not a:
            return None
        turns = [q, a]
    if len(turns) % 2:
        return None

    real = [c for c in chunks if c]
    user0 = turns[0]
    if context_mode == "grounded" and real:
        ctx = "\n\n".join("[%s, p.%s]: %s" % (c["doc_label"], c["page_start"], c["text"])
                          for c in real)
        user0 = ("Relevant reference material (cite source + page when you use it):\n"
                 "%s\n\nQuestion: %s" % (ctx, turns[0]))

    msgs = [{"role": "system", "content": SYSTEM_PROMPTS[register]}]
    for i, t in enumerate(turns):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant",
                     "content": (user0 if i == 0 else t)})
    return {
        "messages": msgs,
        "category": category,
        "register": register,
        "source": " + ".join(source_label(c) for c in chunks) if chunks else source_label(None),
        "chunk_id": "+".join(c["chunk_id"] for c in real) if real else "no_chunk",
        "context_mode": context_mode,
        "audience": (real[0]["audience"] if real else "general"),
    }


NUM = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|IU|MU|ml|kg|days?|hours?|weeks?)\b", re.I)


def spot_check(records, chunks_by_id, sample=5, rng=None):
    """Return numbers in an answer that do not appear in its own source chunks."""
    rng = rng or random.Random(0)
    problems = []
    for rec in rng.sample(records, min(sample, len(records))):
        haystack = ""
        for cid in (rec.get("chunk_id") or "").split("+"):
            c = chunks_by_id.get(cid)
            if c:
                haystack += c["text"]
        if not haystack:
            continue
        answer = "\n".join(m["content"] for m in rec["messages"] if m["role"] == "assistant")
        unsupported = []
        for m in NUM.finditer(answer):
            tok = re.sub(r"\s+", "", m.group(0)).lower()
            hay = re.sub(r"\s+", "", haystack).lower()
            if tok not in hay:
                unsupported.append(m.group(0))
        if unsupported:
            problems.append((rec.get("chunk_id"), sorted(set(unsupported))))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--category", default="all")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--pairs-per-call", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--grounded-fraction", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--targets", default="", help="override, e.g. dosing_duration=100")
    ap.add_argument("--dry-run", action="store_true", help="print one prompt and exit")
    ap.add_argument("--max-failures", type=int, default=3,
                    help="consecutive call failures before aborting the category")
    ap.add_argument("--drug-cap-fraction", type=float, default=0.08,
                    help="max share of a category that may be about one drug")
    ap.add_argument("--max-stalled-calls", type=int, default=8,
                    help="calls yielding no accepted pair before aborting the category")
    args = ap.parse_args()

    spec = json.loads((ROOT / "scripts" / "targets.json").read_text()) \
        if (ROOT / "scripts" / "targets.json").exists() else {
        "antibiotic_indicated": 1050, "no_antibiotic_indicated": 900,
        "aware_classification": 550, "dosing_duration": 900, "patient_education": 700,
        "farmer_one_health": 450, "refusal_escalation": 450}
    for override in filter(None, args.targets.split(",")):
        k, v = override.split("=")
        spec[k.strip()] = int(v)

    registers = {"antibiotic_indicated": "health_worker", "no_antibiotic_indicated": "health_worker",
                 "aware_classification": "health_worker", "dosing_duration": "health_worker",
                 "patient_education": "patient", "farmer_one_health": "farmer",
                 "refusal_escalation": "health_worker"}

    manifest = load_manifest()
    chunks_by_id = {c["chunk_id"]: c for c in manifest}
    tier_path = CHUNKS / "aware_classification_table.txt"
    tier_table = tier_path.read_text(encoding="utf-8")[:12000] if tier_path.exists() else ""

    cats = list(spec) if args.category == "all" else [args.category]
    RAW.mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    session = requests.Session()
    guards = Guards()
    drug_to_chunks, chunk_to_drugs = build_drug_index(manifest, guards)
    tier_rows, tier_header = load_tier_rows(guards)
    dose_lines, doses_verified = load_dose_lines()
    if dose_lines:
        print("dose lines: %d drugs with attributed, unflagged lines (verified=%s)"
              % (len(dose_lines), doses_verified))
    formulary_ids = [c["chunk_id"] for c in manifest
                     if c and "EMHSLU" in (c.get("doc_label") or "")]
    classified_drugs = (frozenset(guards.v.lookup["flat"])
                        | frozenset(guards.v.lookup["route_dependent"])
                        | frozenset(k for k, v in DRUG_ALIASES.items()
                                    if v in guards.v.lookup["flat"]
                                    or v in guards.v.lookup["route_dependent"]))
    print("drug index: %d drugs across %d chunks; %d formulary chunk(s); "
          "%d drugs carry an AWaRe tier"
          % (len(drug_to_chunks), len(chunk_to_drugs), len(formulary_ids),
             len(classified_drugs)))

    for category in cats:
        target = spec[category]
        out_path = RAW / ("%s.jsonl" % category)
        done, seen_chunks = [], defaultdict(int)
        if out_path.exists():
            for line in open(out_path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.append(r)
                seen_chunks[r.get("chunk_id")] += 1
        print("\n=== %s : have %d, target %d ===" % (category, len(done), target), flush=True)
        if len(done) >= target:
            continue

        pool = pick_chunks(manifest, category)
        if not pool:
            print("   no chunks match this category -- skipping", file=sys.stderr)
            continue

        # A passage with no antibiotic in it cannot ground an antibiotic answer.
        # The pilot attached an EMHSLU chunk of nutrition and neurosurgery text
        # to a fosfomycin question; chunk 7 of formulary_antibacterials is
        # labelled "6.2 Antibacterials" and contains no drug at all.
        if category in ("aware_classification", "antibiotic_indicated", "dosing_duration"):
            kept = [c for c in pool if c is None or guards.drug_names(c["text"])]
            dropped = len([c for c in pool if c is not None]) - len([c for c in kept if c is not None])
            if dropped:
                print("   dropped %d chunk(s) containing no known drug name" % dropped)
            pool = kept or pool

        # No chunk over 2.5% of the category -- but one call emits
        # pairs_per_call records from a single chunk, so a cap below that can
        # never be honoured and silently did nothing.
        cap = max(args.pairs_per_call, int(target * 0.025))
        # No single drug over this share of the category, so coverage spreads
        # across the classification table instead of pooling on amoxicillin.
        drug_cap = max(1, int(target * args.drug_cap_fraction))
        grounded_count = sum(1 for r in done if r.get("context_mode") == "grounded")
        seen_drugs = defaultdict(int)
        for r in done:
            d = primary_drug(guards, r)
            if d is not None:
                seen_drugs[d] += 1
        rejected = defaultdict(int)
        rejected_total = 0
        consecutive_failures = 0

        fh = open(out_path, "a", encoding="utf-8")
        batch, t0 = [], time.time()
        stalled_calls = 0
        while len(done) < target:
            before = len(done)
            candidates = [c for c in pool
                          if c is None or seen_chunks[c["chunk_id"]] < cap]
            if not candidates:
                print("   every chunk hit the 2.5%% cap at %d records -- stopping this "
                      "category" % len(done), file=sys.stderr)
                break
            n = min(args.pairs_per_call, target - len(done))
            chunks, targets = plan_call(category, rng, candidates, drug_to_chunks,
                                        chunk_to_drugs, chunks_by_id, formulary_ids,
                                        seen_drugs, drug_cap, n, dose_lines, classified_drugs)
            chunk = chunks[0]
            excerpt = tier_excerpt(targets, tier_rows, tier_header) or tier_table
            prompt = build_user_prompt(category, chunks, n, excerpt, targets)

            if args.dry_run:
                print("PROMPT: %d chars, ~%d tokens (rough chars/4), "
                      "%d source chunks"
                      % (len(prompt), len(prompt) // 4, len([c for c in chunks if c])))
                print(prompt[:4000])
                return 0

            try:
                raw = call_model(session, args.base_url, args.model, prompt,
                                 args.temperature, args.max_tokens, args.timeout)
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                print("   call failed (attempt %d/%d):\n   %s"
                      % (consecutive_failures, args.max_failures, e), file=sys.stderr)
                if consecutive_failures >= args.max_failures:
                    print("\n   ABORTING %s after %d consecutive failures. The error above "
                          "is the server's own response -- read it before rerunning.\n"
                          "   %d records were written."
                          % (category, consecutive_failures, len(done)), file=sys.stderr)
                    fh.close()
                    return 3
                time.sleep(5)
                continue

            for item in parse_pairs(raw):
                # Deterministic grounded/closed split. Randomising per item made
                # a 20-pair run land at 15% against a 30% target; we now steer
                # toward the target using what has actually been written.
                want_grounded = (len(done) + 1) * args.grounded_fraction > grounded_count
                mode = "grounded" if (want_grounded and chunk is not None) else "closed"
                rec = to_record(item, category, registers[category], chunks, mode)
                if not rec:
                    continue

                reasons = guards.screen(rec, chunks)
                if reasons:
                    rejected[reasons[0].split(":")[0]] += 1
                    rejected_total += 1
                    continue

                # Spread coverage: stop the model circling the same few drugs.
                drug = primary_drug(guards, rec)
                if drug is not None and seen_drugs[drug] >= drug_cap:
                    rejected["drug_over_represented"] += 1
                    rejected_total += 1
                    continue

                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done.append(rec)
                batch.append(rec)
                if drug is not None:
                    seen_drugs[drug] += 1
                if mode == "grounded":
                    grounded_count += 1
                if chunk is not None:
                    seen_chunks[chunk["chunk_id"]] += 1

            fh.flush()

            # A call that yields nothing acceptable is as much a dead end as a
            # call that errors -- it just fails quietly. Treat it the same way.
            if len(done) == before:
                stalled_calls += 1
                if stalled_calls >= args.max_stalled_calls:
                    print("\n   ABORTING %s: %d consecutive calls produced no acceptable "
                          "pair. Rejections so far:" % (category, stalled_calls),
                          file=sys.stderr)
                    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
                        print("      %-52s %d" % (reason, count), file=sys.stderr)
                    print("   Either the prompt is not steering the model, or a guard is "
                          "too strict. Do not just raise the limit.", file=sys.stderr)
                    fh.close()
                    return 4
            else:
                stalled_calls = 0

            if len(batch) >= args.batch_size:
                probs = spot_check(batch, chunks_by_id, sample=5, rng=rng)
                rate = len(done) / max(time.time() - t0, 1e-6) * 60
                print("   %d/%d  (%.1f pairs/min)  grounded %d%%  rejected %d  spot-check: %s"
                      % (len(done), target, rate,
                         round(100 * grounded_count / max(len(done), 1)), rejected_total,
                         "clean" if not probs else "%d of 5 have unsupported numbers" % len(probs)),
                      flush=True)
                if probs:
                    print("\n   STOPPING %s -- the model is inventing numbers that are not in "
                          "the attached chunk:" % category, file=sys.stderr)
                    for cid, nums in probs:
                        print("      %s -> %s" % (cid, ", ".join(nums)), file=sys.stderr)
                    print("   %d records were written before the stop; review them before "
                          "using." % len(done), file=sys.stderr)
                    fh.close()
                    return 2
                batch = []
        fh.close()
        print("   %s finished: %d records -> %s" % (category, len(done), out_path))
        print("      grounded %d/%d (%d%%, target %d%%)  distinct drugs %d"
              % (grounded_count, len(done),
                 round(100 * grounded_count / max(len(done), 1)),
                 # count drugs actually used, not every key plan_call read:
                 # seen_drugs is a defaultdict and reading a key creates it.
                 round(100 * args.grounded_fraction),
                 sum(1 for v in seen_drugs.values() if v)))
        if rejected:
            print("      rejected %d pair(s) at the gate:" % rejected_total)
            for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
                print("         %-50s %d" % (reason, count))

    print("\nAll requested categories done. Next:")
    print("   python3 distilled_fixes.py > raw/distilled_fixes.jsonl")
    print("   python3 prepare_dataset.py --input \"raw/*.jsonl\" --out dataset/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
