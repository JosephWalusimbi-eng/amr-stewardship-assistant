"""Interleaved Category 1 / Category 2 generation, one boundary at a time.

Separate from generate_pairs.py on purpose. That script produced the 550-pair
aware_classification run and the Category 4 batches; this mode changes the unit
of generation (a matched PAIR, not a single example) and the planning axis (a
boundary variant, not a target drug), and folding both into the existing loop
would put the working generator at risk for no benefit. The HTTP, parsing and
gating machinery is imported rather than copied, so there is one implementation
of each.

Each call is written against ONE boundary variant -- one presentation, one
source's criteria -- and asks for matched pairs: same presentation, one vignette
that satisfies no named criterion (Category 2) and one that satisfies at least
one (Category 1). Both halves carry the variant tag, so the dataset records which
source's threshold each pair was written against.

Two variants are cross-source and are generated deliberately, not worked around:

  * acute bronchitis -- AWaRe states the default and names no exception at all;
    UCG 5.2.2 names the only exception in the corpus.
  * COPD under UCG's threshold -- UCG states the criteria but no "antibiotics are
    not needed by default" sentence, so the default is AWaRe's.

Both are tagged `cross_source: true` so the paper can report how many pairs rest
on a criterion the other source does not recognise.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import scenario_pool as sp                                          # noqa: E402
from scenario_pool_data import POOL                                 # noqa: E402
from generation_guards import Guards                                # noqa: E402
from generate_pairs import (CallFailed, call_model, parse_pairs,    # noqa: E402
                            load_manifest, source_label)
from system_prompts import SYSTEM_PROMPTS                           # noqa: E402

RAW = ROOT / "raw"
BY_ID = {e["id"]: e for e in POOL}

SIDE_TO_CATEGORY = {
    "not_indicated": "no_antibiotic_indicated",
    "indicated": "antibiotic_indicated",
}

SCHEMA = """Return a JSON array of exactly {m} objects, nothing else:
[{{"pair": 1, "side": "not_indicated", "question": "...", "answer": "..."}},
 {{"pair": 1, "side": "indicated", "question": "...", "answer": "..."}}]
Every pair number from 1 to {n} appears exactly twice: once with side
"not_indicated" and once with side "indicated". The two halves of a pair are the
same presentation and differ ONLY in whether the exception criteria are met."""


def criterion_block(e):
    """One exception criterion, as the generator must treat it."""
    lines = ["  [%s] %s" % (e["id"], e["quote"])]
    b = e.get("bound")
    if b:
        sym = {">=": "at least", ">": "above", "<=": "at most", "<": "under"}[b["op"]]
        unit = {"temp_c": "degrees C", "days": "days", "years": "years",
                "months": "months", "bpm": "bpm", "rr": "breaths/min"}[b["kind"]]
        v = b["value"]
        v = int(v) if float(v).is_integer() else v
        lines.append("      numeric threshold: %s %s %s -- this exact number, "
                     "never a rounder or lower one" % (sym, v, unit))
    if e.get("min_limbs", 1) > 1:
        lines.append("      CONJUNCTIVE: at least %d of its parts must be present "
                     "together; one alone does not satisfy it" % e["min_limbs"])
    return "\n".join(lines)


def build_prompt(variant, chunks, n):
    v = variant
    P = []
    P.append("SOURCE PASSAGES (the only material you may draw clinical content from):\n")
    for c in chunks:
        P.append("--- [%s | %s | p.%s-%s] ---\n%s\n"
                 % (c["doc_label"], c["section"], c["page_start"], c["page_end"], c["text"]))

    P.append("\nTHE BOUNDARY FOR THIS CALL")
    P.append("Presentation: %s" % v["presentation"].replace("_", " "))
    P.append("Criteria source: %s" % v["criteria_source"].replace("_", " "))
    if v["cross_source"]:
        P.append("NOTE: the default and the exception criteria come from DIFFERENT documents. "
                 "Use the default as stated and the criteria as stated; do not blend them, and "
                 "do not claim either document says what the other says.")

    P.append("\nDEFAULT (what to do when no criterion below is met):")
    for i in v["default_ids"]:
        P.append("  [%s] %s" % (i, BY_ID[i]["quote"]))

    P.append("\nEXCEPTION CRITERIA -- the COMPLETE list. There are no others:")
    for i in v["exception_ids"]:
        P.append(criterion_block(BY_ID[i]))

    if v["non_criterion_ids"]:
        P.append("\nEXPLICITLY NOT INDICATIONS -- the guideline names these so they are not "
                 "mistaken for reasons to treat:")
        for i in v["non_criterion_ids"]:
            P.append("  [%s] %s" % (i, BY_ID[i]["quote"]))

    P.append("\nSYMPTOMATIC MANAGEMENT to offer in the Category 2 answer:")
    for i in v["substitute_ids"]:
        e = BY_ID[i]
        P.append("  [%s] %s" % (i, e["quote"]))
        if e.get("note"):
            P.append("      also: %s" % e["note"])

    if v["escalation_ids"]:
        P.append("\nESCALATION / REVIEW TRIGGERS (these send the patient back or onward -- "
                 "they are NOT antibiotic triggers):")
        for i in v["escalation_ids"]:
            e = BY_ID[i]
            P.append("  [%s] %s" % (i, e["quote"]))
            if e.get("note"):
                P.append("      also: %s" % e["note"])

    P.append("\nFIRST CHOICE for the Category 1 half:")
    for i in v["first_choice_ids"]:
        e = BY_ID[i]
        P.append("  [%s] %s" % (i, e["quote"]))
        if e.get("note"):
            P.append("      also: %s" % e["note"])

    P.append("""
TASK
Write {n} matched PAIRS of clinical training examples for this boundary.

For each pair, invent one presentation and two patients who differ in whether the
exception criteria are met. Vary age, sex, weight, comorbidity and facility level
(HC2 / HC3 / HC4 / hospital) across pairs.

THE "not_indicated" HALF (Category 2):
  - The vignette must satisfy NONE of the exception criteria above, and must say
    so positively: state the relevant findings as ABSENT ("sputum unchanged and
    not purulent", "afebrile at 37.1 C", "the left ear is normal"). Do not leave
    them unmentioned -- a clinician reading it must be able to see the criteria
    were checked.
  - The answer must: say plainly that no antibiotic is indicated; give the reason
    AS THE SOURCE STATES IT; offer the symptomatic management above, with the
    doses as written; and name specific danger signs from the escalation triggers.
  - The answer must NOT name an antibiotic, an antibiotic class, or "the
    first-choice agent" in any recommending frame -- including a conditional one.
    "If X happens, start an antibiotic" is forbidden. Say what would change the
    assessment and that the patient should be reassessed; stop there.
  - If the answer states a threshold at all, it must be one of the exact numbers
    listed under EXCEPTION CRITERIA. Never round it, never lower it, never
    substitute a number from a different presentation.

THE "indicated" HALF (Category 1):
  - The vignette must satisfy at least one criterion above, EXPLICITLY and in the
    terms the criterion uses. For a conjunctive criterion every part must be
    present in the vignette.
  - The answer states: that an antibiotic is indicated and WHICH criterion is met,
    naming it; the first-choice agent and its AWaRe group; the dose; the duration;
    and what to do on failure or deterioration.

Both halves stay inside the passages above. Do not import a criterion, a drug, a
dose or a threshold from another guideline or from memory.""".format(n=n))
    P.append("\n\n" + SCHEMA.format(n=n, m=2 * n))
    return "\n".join(P)


def to_record(item, variant, chunks, context_mode, pair_uid):
    q, a = item.get("question"), item.get("answer")
    side = (item.get("side") or "").strip()
    if not q or not a or side not in SIDE_TO_CATEGORY:
        return None
    user0 = q
    if context_mode == "grounded":
        ctx = "\n\n".join("[%s, p.%s]: %s" % (c["doc_label"], c["page_start"], c["text"])
                          for c in chunks)
        user0 = ("Relevant reference material (cite source + page when you use it):\n"
                 "%s\n\nQuestion: %s" % (ctx, q))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS["health_worker"]},
            {"role": "user", "content": user0},
            {"role": "assistant", "content": a},
        ],
        "category": SIDE_TO_CATEGORY[side],
        "register": "health_worker",
        "source": " + ".join(source_label(c) for c in chunks),
        "chunk_id": "+".join(c["chunk_id"] for c in chunks),
        "context_mode": context_mode,
        "audience": chunks[0]["audience"],
        # --- boundary metadata, the point of this mode -----------------------
        "presentation": variant["presentation"],
        "criteria_source": variant["criteria_source"],
        "criteria_ids": variant["exception_ids"],
        "default_source": variant["default_source"],
        "cross_source": variant["cross_source"],
        "pair_id": pair_uid,
        "pair_side": side,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", type=int, default=20, help="pairs (each = 2 records)")
    ap.add_argument("--pairs-per-call", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--grounded-fraction", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", default=str(RAW / "boundary_pairs.jsonl"))
    ap.add_argument("--discarded", default="")
    ap.add_argument("--max-stalled-calls", type=int, default=8)
    ap.add_argument("--require-variant", action="append", default=[],
                    help="presentation:source that MUST appear; repeatable")
    ap.add_argument("--dry-run", action="store_true",
                    help="print one prompt per variant and exit -- no server needed")
    args = ap.parse_args()

    manifest = {c["chunk_id"]: c for c in load_manifest()}
    variants = sp.all_variants()

    # Order the variants so the required ones are generated FIRST. A 20-pair test
    # batch that reaches the hard cases last learns nothing if it stalls early,
    # and the two hardest cases here -- the cross-source bronchitis pair and the
    # two COPD thresholds -- are precisely the ones the batch exists to test.
    def vkey(v):
        return "%s:%s" % (v["presentation"], v["criteria_source"])
    required = set(args.require_variant)
    unknown = required - {vkey(v) for v in variants}
    if unknown:
        print("unknown variant(s): %s" % ", ".join(sorted(unknown)), file=sys.stderr)
        print("available: %s" % ", ".join(sorted(vkey(v) for v in variants)), file=sys.stderr)
        return 2
    variants.sort(key=lambda v: (vkey(v) not in required, vkey(v)))

    if args.dry_run:
        for v in variants:
            chunks = [manifest[i] for i in v["chunk_ids"] if i in manifest]
            print("=" * 100)
            print("VARIANT %s  cross_source=%s  chunks=%d"
                  % (vkey(v), v["cross_source"], len(chunks)))
            print("=" * 100)
            p = build_prompt(v, chunks, args.pairs_per_call)
            print(p[:1400])
            print("   ... [%d chars total, ~%d tokens]" % (len(p), len(p) // 4))
            print()
        return 0

    guards = Guards()
    rng = random.Random(args.seed)
    session = requests.Session()
    RAW.mkdir(exist_ok=True)

    out_path = Path(args.out)
    kept = []
    if out_path.exists():
        kept = [json.loads(l) for l in open(out_path, encoding="utf-8") if l.strip()]
    have_pairs = len({r.get("pair_id") for r in kept})
    print("boundary pairs: have %d, target %d" % (have_pairs, args.target), flush=True)

    rejected = defaultdict(int)
    discarded = []
    per_variant = defaultdict(int)
    stalled = 0
    vi = 0

    while have_pairs < args.target and stalled < args.max_stalled_calls:
        v = variants[vi % len(variants)]
        vi += 1
        chunks = [manifest[i] for i in v["chunk_ids"] if i in manifest]
        if not chunks:
            print("   %s: no chunks resolvable -- skipping" % vkey(v), file=sys.stderr)
            continue
        n = min(args.pairs_per_call, args.target - have_pairs)
        prompt = build_prompt(v, chunks, n)
        mode = "grounded" if rng.random() < args.grounded_fraction else "closed"

        try:
            text = call_model(session, args.base_url, args.model, prompt,
                              args.temperature, args.max_tokens, args.timeout)
        except (CallFailed, requests.RequestException) as exc:
            print("   call failed: %s" % str(exc)[:300], file=sys.stderr)
            stalled += 1
            continue

        items = parse_pairs(text)
        by_pair = defaultdict(dict)
        for it in items:
            side = (it.get("side") or "").strip()
            if side in SIDE_TO_CATEGORY:
                by_pair[it.get("pair")][side] = it

        accepted_this_call = 0
        for pnum, halves in sorted(by_pair.items(), key=lambda kv: str(kv[0])):
            if set(halves) != set(SIDE_TO_CATEGORY):
                rejected["incomplete_pair"] += 1
                continue
            pair_uid = "%s::%s::%d" % (v["presentation"], v["criteria_source"],
                                       have_pairs + accepted_this_call + 1)
            recs, reasons = [], []
            for side, item in halves.items():
                rec = to_record(item, v, chunks, mode, pair_uid)
                if rec is None:
                    reasons.append("%s:malformed" % side)
                    continue
                # The gate sees the same record the dataset would.
                why = guards.screen(rec, chunks)
                if why:
                    reasons.extend("%s:%s" % (side, w) for w in why)
                else:
                    recs.append(rec)
            # A pair is accepted only whole. Keeping one half would silently
            # unbalance the ratio the interleaved design exists to control.
            if len(recs) == 2:
                kept.extend(recs)
                accepted_this_call += 1
                per_variant[vkey(v)] += 1
            else:
                for r in reasons:
                    rejected[r.split(":", 1)[1].split(":")[0]] += 1
                discarded.append({"variant": vkey(v), "reasons": reasons, "halves": halves})

        have_pairs += accepted_this_call
        stalled = 0 if accepted_this_call else stalled + 1
        print("   %-52s +%d pair(s)  total=%d" % (vkey(v), accepted_this_call, have_pairs),
              flush=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.discarded and discarded:
        with open(args.discarded, "w", encoding="utf-8") as fh:
            for d in discarded:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("\nwrote %d records (%d pairs) to %s" % (len(kept), have_pairs, out_path))
    print("per-variant pairs:")
    for k in sorted(per_variant):
        print("   %-52s %d" % (k, per_variant[k]))
    missing = required - set(per_variant)
    if missing:
        print("REQUIRED VARIANTS WITH NO ACCEPTED PAIR: %s" % ", ".join(sorted(missing)),
              file=sys.stderr)
    if rejected:
        print("rejections by reason:")
        for k, n_ in sorted(rejected.items(), key=lambda kv: -kv[1]):
            print("   %-52s %d" % (k, n_))
    if stalled >= args.max_stalled_calls:
        print("STALLED: %d consecutive calls produced nothing acceptable"
              % stalled, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
