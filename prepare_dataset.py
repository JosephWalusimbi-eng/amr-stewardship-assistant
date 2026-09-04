"""
Validates the generated pairs and assembles the train/val split.

    python3 prepare_dataset.py --input "raw/*.jsonl" --out dataset/

Two severities:

  BLOCKING  -- the record is wrong in a way that would teach the model something
               unsafe or malformed. Blocked records never reach dataset/.
  FLAG      -- the record needs a human to look at it. Flagged records DO reach
               dataset/ (they are not necessarily wrong), and every one is listed
               in the report so it can be hand-checked. `unlisted_drug_name` is
               the main one: a drug that is not on Uganda's essential medicines
               list may be a hallucination, or may be a legitimate mention in a
               referral or "not available here" answer.

Nothing here is a substitute for reading the dosing pairs by hand.
"""
import argparse
import glob
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SM = ROOT / "source_material"

sys.path.insert(0, str(ROOT))
# Hard import, deliberately. This used to fall back to {} when app.py could not
# be imported, which silently disabled the system_prompt_mismatch check instead
# of failing. system_prompts.py has no third-party imports, so there is no
# longer any environment where the fallback is the right answer.
from system_prompts import SYSTEM_PROMPTS

VALID_CATEGORIES = {
    "antibiotic_indicated", "no_antibiotic_indicated", "aware_classification",
    "dosing_duration", "patient_education", "farmer_one_health", "refusal_escalation",
}
VALID_REGISTERS = {"health_worker", "patient", "farmer"}
TARGETS = {
    "antibiotic_indicated": 1050, "no_antibiotic_indicated": 900,
    "aware_classification": 550, "dosing_duration": 900, "patient_education": 700,
    "farmer_one_health": 450, "refusal_escalation": 450,
}

# Diarrhoea danger signs the spec requires by name (Category 2).
DANGER_SIGNS = [
    (r"blood", "blood in stool"),
    (r"fever", "fever"),
    (r"dehydrat|dry mouth|no tears|urin", "dehydration signs"),
    (r"vomit", "persistent vomiting"),
    (r"letharg|unconscious|drowsy", "lethargy"),
]
VAGUE_ONLY = re.compile(r"\bif (it|they|things|symptoms) (get|gets|become|becomes) worse\b", re.I)

# A recommending frame around an antibiotic name -- what Category 2 must never do.
RECOMMEND = re.compile(
    r"\b(give|prescrib\w*|start|commence|administer|use|take|consider|offer|"
    r"recommend\w*|could try|short course of|course of)\b", re.I)

DURATION_NUM = re.compile(r"\b\d+(?:\.\d+)?\s*(?:-\s*\d+\s*)?(day|days|hour|hours|week|weeks)\b", re.I)
HEDGED_DURATION = re.compile(
    r"\b(usually|typically|generally|normally|about|around|roughly|approximately)\b[^.]{0,40}?"
    r"\b\d+[^.]{0,20}?\b(day|days|hour|hours|week|weeks)\b", re.I)

DOSE_PAT = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|µg|IU|MU|ml)\b(?:\s*/\s*\w+)?", re.I)


def load_json(name):
    p = SM / name
    if not p.exists():
        print("WARNING: %s missing -- related checks disabled. Run "
              "scripts/build_reference_lists.py" % p, file=sys.stderr)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# Nomenclature bridges between the national formulary and the WHO classification.
# EMHSLU and UCG prose use the Ugandan common names; AWaRe uses INN combination
# names. Same drug, different register. Without these the tier check silently
# skipped two of the most-used antibacterials in the corpus -- "cotrimoxazole"
# alone appears 23 times in UCG prose, and it is Access, not unclassified.
#
# Keep this list short and evidence-based: each entry must be the SAME drug
# under two names, verified against both reference files. It is not a place to
# map a drug onto a therapeutic neighbour.
DRUG_ALIASES = {
    "cotrimoxazole": "sulfamethoxazole/trimethoprim",   # EMHSLU "Cotrimoxazole"
    "benzathinepenicillin": "benzathine-benzylpenicillin",  # EMHSLU "Benzathine penicillin"
}
DRUG_ALIASES = {k: "".join(ch for ch in v.lower() if ch.isalnum())
                for k, v in DRUG_ALIASES.items()}


def drug_pattern(name):
    """Regex for one reference drug name, tolerant of separator style.

    The reference lists and the prose disagree about separators: AWaRe stores
    "Polymyxin-B_IV", a guideline writes "polymyxin B", a model writes
    "Polymyxin B". Matching the stored spelling literally made every
    space-separated mention invisible to validation -- and a drug that is not
    recognised is a drug whose tier is never checked and whose absence from the
    formulary is never flagged, so the failure was silent and fail-open.

    Combination products matter most. AWaRe stores "Ceftazidime/avibactam"
    (Reserve) while prose writes "ceftazidime+avibactam" or "ceftazidime-
    avibactam". Matching only the stored separator left the combination
    unrecognised and matched the bare "Ceftazidime" (Watch) instead, checking a
    Reserve product against the wrong tier.

    Route suffixes are stripped: tier-by-route is resolved separately via
    aware_lookup.json's route_dependent map.
    """
    base = re.sub(r"_(IV|oral)$", "", name)
    parts = [re.escape(p) for p in re.split(r"[\s\-+/]+", base) if p]
    return r"\b" + r"[\s\-+/]*".join(parts) + r"\b"


def norm_drug(s):
    s = s.lower()
    s = re.sub(r"\b(sodium|potassium|sulphate|sulfate|hydrochloride|citrate|forte|base)\b", " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


class Validator:
    def __init__(self):
        self.aware = load_json("aware_tiers.json") or {"antibiotics": {}}
        self.lookup = load_json("aware_lookup.json") or {"flat": {}, "route_dependent": {}}
        self.emhslu = load_json("emhslu_drugs.json") or {"drugs": {}}
        self.emhslu_norm = {norm_drug(d["drug"]) for d in self.emhslu["drugs"].values()}
        # every antibiotic name we know about, longest first so multiword names win
        self.known_drugs = sorted(
            {v["antibiotic"] for v in self.aware["antibiotics"].values()}
            | {d["drug"] for d in self.emhslu["drugs"].values()},
            key=len, reverse=True)
        self.drug_res = [(d, re.compile(drug_pattern(d), re.I)) for d in self.known_drugs]

    def drugs_in(self, text):
        """Longest-match-first, with matched spans masked out.

        Without masking, "ceftazidime+avibactam" (Reserve) also matches the bare
        name "ceftazidime" (Watch), and the tier check then reports a
        contradiction that isn't there. Masking each hit stops a shorter name
        matching inside a longer one.
        """
        found, mask = {}, list(text)
        for name, rx in self.drug_res:                 # self.drug_res is length-sorted
            hay = "".join(mask)
            m = rx.search(hay)
            if not m:
                continue
            base = re.sub(r"_(IV|oral)$", "", name)
            found.setdefault(norm_drug(base), (base, m.start(), m.end()))
            for i in range(m.start(), m.end()):
                mask[i] = "\x00"
        return found

    @staticmethod
    def in_combination(text, start, end):
        """True if the matched name is one half of a combination product.

        'ceftazidime+avibactam' and 'amoxicillin/clavulanic acid' carry their own
        AWaRe tiers, different from the single agent, so the single-agent tier
        must not be asserted over them.
        """
        before = text[max(0, start - 2):start]
        after = text[end:end + 2]
        return bool(re.search(r"[+/]\s*$", before) or re.match(r"\s*[+/]\s*[A-Za-z]", after))

    def check(self, rec, idx, path):
        blocking, flags = [], []

        def B(code, detail=""):
            blocking.append((code, detail))

        def F(code, detail=""):
            flags.append((code, detail))

        # ---- structure
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 3:
            B("malformed_messages", "expected >=3 messages, got %r" % (len(msgs) if isinstance(msgs, list) else None))
            return blocking, flags
        if msgs[0].get("role") != "system":
            B("missing_system_turn")
        roles = [m.get("role") for m in msgs]
        if roles[-1] != "assistant":
            B("last_turn_not_assistant", "|".join(roles))
        if len(msgs) % 2 == 0:
            B("even_turn_count", "|".join(roles))

        cat = rec.get("category")
        reg = rec.get("register")
        if cat not in VALID_CATEGORIES:
            B("bad_category", str(cat))
        if reg not in VALID_REGISTERS:
            B("bad_register", str(reg))
        if not rec.get("source"):
            B("missing_source")
        if rec.get("context_mode") not in ("closed", "grounded"):
            B("bad_context_mode", str(rec.get("context_mode")))

        # system prompt must match app.py byte-for-byte
        if reg in SYSTEM_PROMPTS and msgs[0].get("content") != SYSTEM_PROMPTS.get(reg):
            B("system_prompt_mismatch", "register=%s" % reg)

        answer = "\n".join(m.get("content", "") for m in msgs if m.get("role") == "assistant")
        question = "\n".join(m.get("content", "") for m in msgs if m.get("role") == "user")
        if not answer.strip():
            B("empty_answer")
            return blocking, flags
        if len(answer) < 80:
            B("answer_too_short", "%d chars" % len(answer))
        if not re.search(r"[.!?\"'*)\]]\s*$", answer.strip()):
            B("truncated_answer", repr(answer.strip()[-60:]))

        # ---- drug-name grounding
        found = self.drugs_in(answer)
        for key, (base, _s, _e) in found.items():
            if key not in self.emhslu_norm:
                F("unlisted_drug_name", "%s (not on EMHSLU 2023 s6.2) [%s]" % (base, cat))

        # ---- AWaRe tier correctness
        for raw_key, (base, s, e) in found.items():
            key = DRUG_ALIASES.get(raw_key, raw_key)
            if self.in_combination(answer, s, e):
                continue                      # combination product, different tier
            window = answer[max(0, s - 160):e + 160]
            stated = set(t.capitalize() for t in re.findall(r"\b(Access|Watch|Reserve)\b", window, re.I))
            if not stated:
                continue
            if key in self.lookup["route_dependent"]:
                routes = self.lookup["route_dependent"][key]
                if len(set(routes.values())) > 1 and len(stated) == 1:
                    if not re.search(r"\b(oral|intravenous|\bIV\b|by mouth|injection)\b", window, re.I):
                        B("route_dependent_tier_without_route",
                          "%s is %s" % (base, routes))
            elif key in self.lookup["flat"]:
                true_tier = self.lookup["flat"][key]
                if stated and true_tier not in stated:
                    B("wrong_aware_tier",
                      "%s: answer says %s, WHO 2023 says %s" % (base, "/".join(sorted(stated)), true_tier))

        # ---- category-specific
        if cat == "no_antibiotic_indicated":
            if not re.search(r"\bno antibiotic|not indicated|antibiotic is not|none is indicated"
                             r"|do(es)? not need an antibiotic", answer, re.I):
                B("category2_missing_refusal")
            for key, (base, s, e) in found.items():
                before = answer[max(0, s - 120):s]
                # a warning mention ("avoid ciprofloxacin") is fine; a recommendation is not
                if RECOMMEND.search(before) and not re.search(
                        r"\b(avoid|not|never|without|instead of|rather than|do not|don't|unnecessar)\b",
                        before, re.I):
                    B("concession_antibiotic", "%s in a recommending frame" % base)
            if re.search(r"diarrh|dysenter|loose stool", question + answer, re.I):
                missing = [label for pat, label in DANGER_SIGNS
                           if not re.search(pat, answer, re.I)]
                if missing:
                    B("missing_named_danger_signs", ", ".join(missing))
            if VAGUE_ONLY.search(answer) and not any(
                    re.search(p, answer, re.I) for p, _ in DANGER_SIGNS[:1]):
                F("vague_worsening_language")

        if cat == "antibiotic_indicated":
            if not DOSE_PAT.search(answer):
                B("missing_dose")
            if not DURATION_NUM.search(answer):
                B("missing_duration")
            if not re.search(r"\b(Access|Watch|Reserve)\b", answer):
                B("missing_aware_tier")

        if cat == "dosing_duration":
            if not DOSE_PAT.search(answer) and not re.search(
                    r"cannot|can't|not (able|possible)|no (dose|band)|outside", answer, re.I):
                B("dosing_pair_without_a_dose")
            if re.search(r"uganda clinical guidelines", rec.get("source", ""), re.I) and \
                    re.search(r"\btable\b", rec.get("source", ""), re.I):
                F("dose_from_ucg_table", "column alignment is lost in extraction - verify against the page image")

        if cat == "patient_education":
            if DOSE_PAT.search(answer):
                B("dose_in_patient_register", DOSE_PAT.search(answer).group(0))

        if cat == "farmer_one_health":
            if DURATION_NUM.search(answer) or DOSE_PAT.search(answer):
                B("number_in_farmer_answer",
                  "no livestock source in corpus; any figure is fabricated")

        if cat == "refusal_escalation" and re.search(
                r"withdrawal period|sell (the )?milk|slaughter|meat from", question, re.I):
            if DURATION_NUM.search(answer):
                B("withdrawal_period_states_a_number", DURATION_NUM.search(answer).group(0))
            if HEDGED_DURATION.search(answer):
                B("withdrawal_period_hedged_number", HEDGED_DURATION.search(answer).group(0))
            for need, label in [(r"label|package insert|insert", "label/insert"),
                                (r"veterinar", "veterinarian"),
                                (r"district veterinary officer", "district veterinary officer")]:
                if not re.search(need, answer, re.I):
                    B("withdrawal_template_missing_referral", label)

        return blocking, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="raw/*.jsonl")
    ap.add_argument("--out", default="dataset/")
    ap.add_argument("--val-fraction", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input))
    if not paths:
        print("no input files matched %r" % args.input, file=sys.stderr)
        print("\nNothing to validate. Generate raw/<category>.jsonl first "
              "(scripts/generate_pairs.py) and emit raw/distilled_fixes.jsonl "
              "(python3 distilled_fixes.py).", file=sys.stderr)
        return 1

    v = Validator()
    records, blocked, flagged = [], [], []
    parse_errors = 0
    per_file = Counter()

    for path in paths:
        for i, line in enumerate(open(path, encoding="utf-8"), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                parse_errors += 1
                blocked.append((path, i, [("json_decode_error", str(e))], None))
                continue
            per_file[path] += 1
            b, f = v.check(rec, i, path)
            if f:
                flagged.append((path, i, f, rec))
            if b:
                blocked.append((path, i, b, rec))
            else:
                records.append(rec)

    # ---- duplicate detection
    seen, dupes = {}, []
    for r in records:
        key = "\n".join(m.get("content", "") for m in r["messages"] if m["role"] != "system")
        if key in seen:
            dupes.append(r)
        else:
            seen[key] = r
    if dupes:
        dup_keys = {id(d) for d in dupes}
        records = [r for r in records if id(r) not in dup_keys]

    # ---- report
    W = 78
    print("=" * W)
    print("prepare_dataset.py -- validation report")
    print("=" * W)
    print("input pattern : %s" % args.input)
    for p in paths:
        print("   %-46s %5d records" % (p, per_file[p]))
    print("\nparsed        : %d" % (len(records) + len(blocked) + len(dupes)))
    print("blocked       : %d" % len(blocked))
    print("duplicates    : %d (dropped)" % len(dupes))
    print("flagged       : %d (kept, need hand-check)" % len(flagged))
    print("passing       : %d" % len(records))

    print("\n" + "-" * W)
    print("BLOCKING ISSUES")
    print("-" * W)
    if not blocked:
        print("   none")
    else:
        by_code = Counter(c for _, _, issues, _ in blocked for c, _ in issues)
        for code, n in by_code.most_common():
            print("   %-42s %5d" % (code, n))
        print("\n   every blocking record:")
        for path, ln, issues, rec in blocked:
            cat = (rec or {}).get("category", "?")
            print("   %s:%d  [%s]" % (path, ln, cat))
            for code, detail in issues:
                print("        %-40s %s" % (code, detail))

    print("\n" + "-" * W)
    print("UNLISTED DRUG NAME FLAGS  (hand-check these)")
    print("-" * W)
    unlisted = [(p, ln, d, (rec or {}).get("category"))
                for p, ln, issues, rec in flagged
                for c, d in issues if c == "unlisted_drug_name"]
    if not unlisted:
        print("   none")
    else:
        counts = Counter(d.split(" (")[0] for _, _, d, _ in unlisted)
        print("   distinct drugs: %d, mentions: %d" % (len(counts), len(unlisted)))
        for name, n in counts.most_common():
            print("      %-34s %4d mentions" % (name, n))
        print("\n   every occurrence:")
        for p, ln, d, cat in unlisted:
            print("      %s:%d  [%s]  %s" % (p, ln, cat, d))

    other_flags = [(p, ln, c, d, (rec or {}).get("category"))
                   for p, ln, issues, rec in flagged
                   for c, d in issues if c != "unlisted_drug_name"]
    print("\n" + "-" * W)
    print("OTHER FLAGS")
    print("-" * W)
    if not other_flags:
        print("   none")
    for p, ln, c, d, cat in other_flags:
        print("   %s:%d  [%s]  %-26s %s" % (p, ln, cat, c, d))

    # ---- distribution
    print("\n" + "-" * W)
    print("CATEGORY COUNTS")
    print("-" * W)
    cats = Counter(r.get("category") for r in records)
    print("   %-30s %7s %8s %8s" % ("category", "passing", "target", "delta"))
    for c in sorted(VALID_CATEGORIES):
        t = TARGETS.get(c, 0)
        print("   %-30s %7d %8d %+8d" % (c, cats.get(c, 0), t, cats.get(c, 0) - t))
    extra = set(cats) - VALID_CATEGORIES
    for c in sorted(extra):
        print("   %-30s %7d %8s %8s" % (str(c), cats[c], "-", "UNKNOWN"))
    print("   %-30s %7d %8d" % ("TOTAL", len(records), sum(TARGETS.values())))

    print("\n" + "-" * W)
    print("REGISTER COUNTS")
    print("-" * W)
    regs = Counter(r.get("register") for r in records)
    for r_, n in regs.most_common():
        print("   %-30s %7d  %5.1f%%" % (r_, n, 100.0 * n / max(len(records), 1)))

    print("\n" + "-" * W)
    print("INDICATED : NOT-INDICATED RATIO")
    print("-" * W)
    ind = cats.get("antibiotic_indicated", 0)
    notind = cats.get("no_antibiotic_indicated", 0)
    tot = ind + notind
    if tot:
        pi, pn = 100.0 * ind / tot, 100.0 * notind / tot
        print("   indicated       %5d   %5.1f%%" % (ind, pi))
        print("   not indicated   %5d   %5.1f%%" % (notind, pn))
        print("   ratio           %.2f : 1" % (ind / notind if notind else float("inf")))
        verdict = "OK" if 50.0 <= pi <= 60.0 else "OUT OF SPEC (want 50-60% indicated)"
        print("   %s" % verdict)
    else:
        print("   no records in either category")

    print("\n" + "-" * W)
    print("CONTEXT MODE / AUDIENCE")
    print("-" * W)
    cm = Counter(r.get("context_mode") for r in records)
    for k, n in cm.most_common():
        print("   context_mode %-16s %6d  %5.1f%%" % (k, n, 100.0 * n / max(len(records), 1)))
    aud = Counter(r.get("audience") for r in records)
    for k, n in aud.most_common():
        print("   audience     %-16s %6d  %5.1f%%" % (k, n, 100.0 * n / max(len(records), 1)))

    print("\n" + "-" * W)
    print("SOURCE CONCENTRATION (top chunk_ids)")
    print("-" * W)
    ch = Counter(r.get("chunk_id") for r in records)
    for k, n in ch.most_common(10):
        pct = 100.0 * n / max(len(records), 1)
        mark = "  <-- over 2.5% of dataset" if pct > 2.5 else ""
        print("   %-52s %5d  %4.1f%%%s" % (str(k)[:52], n, pct, mark))

    # ---- write split
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r.get("category")].append(r)
    rng = random.Random(args.seed)
    train, val = [], []
    for c, rs in by_cat.items():                     # stratified by category
        rs = sorted(rs, key=lambda x: json.dumps(x, sort_keys=True))
        rng.shuffle(rs)
        k = int(round(len(rs) * args.val_fraction))
        val.extend(rs[:k])
        train.extend(rs[k:])
    rng.shuffle(train)
    rng.shuffle(val)

    for name, rows in (("train", train), ("val", val)):
        with open(out / ("%s.jsonl" % name), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "-" * W)
    print("OUTPUT")
    print("-" * W)
    print("   %-40s %6d" % (str(out / "train.jsonl"), len(train)))
    print("   %-40s %6d" % (str(out / "val.jsonl"), len(val)))
    print("=" * W)
    if blocked:
        print("%d blocking issue(s) -- those records were EXCLUDED from dataset/." % len(blocked))
    print("Hand-check required before training: the %d unlisted-drug flags and every "
          "dosing_duration pair." % len(unlisted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
