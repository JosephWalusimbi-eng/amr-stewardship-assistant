"""
STEP 2 -- topic-aligned chunking.

Splits each source document along its own structural boundaries (PDF outline
for the AWaRe book and EMHSLU; numbered section headings for UCG 2023;
heading keywords for the two paediatric WHO booklets) and files every span
under a canonical *condition* topic, so a chunk is grounding context for one
generation call rather than an arbitrary page break.

Outputs:
  source_material/chunks/<topic>.txt        human-readable, all spans for a topic
  source_material/chunks/_manifest.jsonl    generation-ready sub-chunks (<= MAX_CHARS)
  source_material/chunks/_index.json        topic -> sources/sections/pages summary
"""
import json
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

ROOT = Path.home() / "amr-stewardship-assistant"
SM = ROOT / "source_material"
OUT = SM / "chunks"
OUT.mkdir(parents=True, exist_ok=True)
MAX_CHARS = 6000          # ceiling for one generation call's grounding context
MIN_CHARS = 400

DOC_LABEL = {
    "who_aware_book_2022": "WHO AWaRe Antibiotic Book (2022)",
    "uganda_clinical_guidelines_2023": "Uganda Clinical Guidelines 2023 (4th ed.)",
    "emhslu_2023": "EMHSLU 2023",
    "who_imci_chart_booklet": "WHO IMCI Chart Booklet",
    "who_pocketbook_hospital_care_children": "WHO Pocket Book of Hospital Care for Children (2nd ed.)",
}


def load_pages(stem):
    txt = (SM / (stem + ".txt")).read_text(encoding="utf-8")
    parts = re.split(r"\[\[page (\d+)\]\]\n", txt)
    it = iter(parts[1:])
    return [(int(n), b.strip()) for n, b in zip(it, it)]


def norm(t):
    """NFKC + fold every dash variant to ASCII '-'.

    The AWaRe outline titles use en-dashes ("27. Community-acquired pneumonia
    - severe"), so topic keys written with a plain hyphen silently miss unless
    the dashes are folded first.
    """
    t = unicodedata.normalize("NFKC", t)
    for dash in "‐‑‒–—―−�":
        t = t.replace(dash, "-")
    return re.sub(r"\s+", " ", t)


# ---------------------------------------------------------------- taxonomy
# AWaRe outline title (lowercased prefix) -> canonical topic slug
AWARE_TOPIC = {
    "3. allergy": "antibiotic_allergy",
    "4. bronchitis": "acute_bronchitis",
    "5. acute otitis media": "acute_otitis_media",
    "6. pharyngitis": "pharyngitis",
    "7. acute sinusitis": "acute_sinusitis",
    "8. oral and dental": "oral_dental_infections",
    "9. localized acute bacterial": "lymphadenitis",
    "10. bacterial eye": "eye_infections",
    "11. trachoma": "trachoma",
    "12. community-acquired pneumonia": "pneumonia_mild",
    "13. exacerbation": "copd_exacerbation",
    "14. acute infectious diarrhoea": "acute_diarrhoea",
    "15. enteric fever": "enteric_fever",
    "16. skin and soft tissue": "skin_soft_tissue_mild",
    "17. burn wound": "burn_wound_infections",
    "18. wound and bite": "wound_bite_infections",
    "19. sexually transmitted infections - chlamydial": "sti_chlamydia",
    "20. sexually transmitted infections - gonococcal": "sti_gonorrhoea",
    "21. sexually transmitted infections - syphilis": "sti_syphilis",
    "22. sexually transmitted infections - trichomoniasis": "sti_trichomoniasis",
    "23. lower urinary tract": "lower_uti",
    "24. sepsis in adults": "sepsis_adults",
    "25. sepsis in neonates": "sepsis_neonatal_paediatric",
    "26. bacterial meningitis": "bacterial_meningitis",
    "27. community-acquired pneumonia - severe": "pneumonia_severe",
    "28. hospital-acquired pneumonia": "pneumonia_hospital_acquired",
    "29. intra-abdominal infections - acute cholecystitis": "intraabdominal_cholecystitis",
    "30. intra-abdominal infections - pyogenic liver": "intraabdominal_liver_abscess",
    "31. intra-abdominal infections - acute appendicitis": "intraabdominal_appendicitis",
    "32. intra-abdominal infections - acute diverticulitis": "intraabdominal_diverticulitis",
    "33. intra-abdominal infections - clostridioides": "c_difficile",
    "34. upper urinary tract": "upper_uti",
    "35. acute bacterial osteomyelitis": "osteomyelitis",
    "36. septic arthritis": "septic_arthritis",
    "37. skin and soft tissue infections - necrotizing": "necrotizing_fasciitis",
    "38. skin and soft tissue infections - pyomyositis": "pyomyositis",
    "39. febrile neutropenia": "febrile_neutropenia",
    "40. surgical prophylaxis": "surgical_prophylaxis",
    "2. improving the use": "aware_classification",
    "41. overview": "reserve_antibiotics",
    "49. dosing guidance - adults": "dosing_guidance_adults",
    "50. dosing guidance - children": "dosing_guidance_children",
}
for _t in ["42. cefiderocol", "43. ceftazidime", "44. fosfomycin", "45. linezolid",
           "46. meropenem+vaborbactam", "47. plazomicin", "48. polymyxin"]:
    AWARE_TOPIC[_t] = "reserve_antibiotics"

# UCG section-number prefix -> topic. Longest prefix wins.
UCG_TOPIC = {
    "1.2.1.3": "wound_bite_infections", "1.2.1.4": "wound_bite_infections",
    "1.2.3": "burn_wound_infections", "1.2.4": "wound_bite_infections",
    "1.1.3": "acute_diarrhoea",
    "2.1.1": "other_bacterial_infections", "2.1.2": "other_bacterial_infections",
    "2.1.3": "other_bacterial_infections", "2.1.4": "other_bacterial_infections",
    "2.1.5": "bacterial_meningitis",
    "2.1.6": "other_bacterial_infections",
    "2.1.7": "sepsis_adults", "2.1.7.1": "sepsis_neonatal_paediatric",
    "2.1.8": "other_bacterial_infections",
    "2.1.9": "enteric_fever", "2.1.10": "other_bacterial_infections",
    "2.3": "viral_infections_no_antibiotic",
    "3.2.1": "sti_gonorrhoea", "3.2.2": "sti_trichomoniasis",
    "3.2.3": "pelvic_inflammatory_disease",
    "3.2.4": "sti_syphilis", "3.2.5": "sti_chlamydia", "3.2.7": "sti_syphilis",
    "3.2.9.1": "eye_infections", "3.2.9.2": "sti_syphilis",
    "4.1.2": "infective_endocarditis", "4.1.9": "pharyngitis",
    "5.2.1": "acute_bronchitis", "5.2.2": "acute_bronchitis",
    "5.2.3": "viral_infections_no_antibiotic", "5.2.4": "other_respiratory_infections",
    "5.2.5": "viral_infections_no_antibiotic", "5.2.6": "other_respiratory_infections",
    "5.2.7": "other_respiratory_infections", "5.2.8": "other_respiratory_infections",
    "5.2.9": "pneumonia_mild", "5.2.10": "pneumonia_mild",
    "5.1.2": "copd_exacerbation",
    "6.1.4": "intraabdominal_appendicitis", "6.1.5": "acute_diarrhoea",
    "6.2.1": "acute_diarrhoea", "6.2.2": "acute_diarrhoea",
    "6.2.3": "acute_diarrhoea", "6.2.4": "acute_diarrhoea",
    "6.5.9": "intraabdominal_cholecystitis",
    "7.2.1": "lower_uti", "7.2.2": "upper_uti", "7.2.3": "lower_uti",
    "10.1.1": "septic_arthritis", "10.1.2": "osteomyelitis", "10.1.3": "pyomyositis",
    "14.1.2": "pelvic_inflammatory_disease",
}

UCG_TOPIC_KEYWORD = [   # applied to heading text (catches chapter 17 childhood illness)
    (r"pneumonia", "pneumonia_mild"),
    (r"diarrhoea|dysentery|cholera|gastroenteritis", "acute_diarrhoea"),
    (r"otitis|ear infection", "acute_otitis_media"),
    (r"tonsill|pharyng|sore throat", "pharyngitis"),
    (r"sinusitis", "acute_sinusitis"),
    (r"meningitis", "bacterial_meningitis"),
    (r"septicaemia|sepsis", "sepsis_neonatal_paediatric"),
    (r"cellulitis|impetigo|abscess", "skin_soft_tissue_mild"),
    (r"urinary|cystitis|pyelonephritis", "lower_uti"),
    (r"conjunctivitis", "eye_infections"),
]


def ucg_topic(secno, title):
    for n in range(4, 0, -1):
        pref = ".".join(secno.split(".")[:n])
        if pref in UCG_TOPIC:
            return UCG_TOPIC[pref]
    low = title.lower()
    for pat, topic in UCG_TOPIC_KEYWORD:
        if re.search(pat, low):
            return topic
    return None


# ---------------------------------------------------------------- span builders
spans = []


def add_span(topic, doc, section, p0, p1, text, audience="general"):
    text = text.strip()
    if len(text) < MIN_CHARS:
        return
    spans.append(dict(topic=topic, doc=doc, section=section, page_start=p0,
                      page_end=p1, audience=audience, text=text))


def outline_spans(pdf_name, stem, mapper, audience="general"):
    reader = PdfReader(str(ROOT / "corpus" / pdf_name))
    entries = []

    def walk(node):
        for it in node:
            if isinstance(it, list):
                walk(it)
            else:
                try:
                    pg = reader.get_destination_page_number(it) + 1
                except Exception:
                    continue
                entries.append((norm(str(it.title)).strip(), pg))

    walk(reader.outline)
    entries.sort(key=lambda e: e[1])
    pages = dict(load_pages(stem))
    maxp = max(pages) if pages else 0
    for i, (title, p0) in enumerate(entries):
        p1 = (entries[i + 1][1] - 1) if i + 1 < len(entries) else maxp
        topic = mapper(title)
        if not topic:
            continue
        body = "\n".join(pages.get(p, "") for p in range(p0, p1 + 1) if pages.get(p))
        add_span(topic, stem, title, p0, p1, body, audience)


def aware_lookup(title):
    t = norm(title).lower()
    for k in sorted(AWARE_TOPIC, key=len, reverse=True):
        if t.startswith(k):
            return AWARE_TOPIC[k]
    return None


outline_spans("aware_book.pdf", "who_aware_book_2022", aware_lookup)


def emhslu_lookup(title):
    t = title.strip().lower()
    if t.startswith("6.2") or t.startswith("6. anti infective") or "antibacterial" in t:
        return "formulary_antibacterials"
    if t.startswith("6.") or "anti infective" in t:
        return "formulary_other_anti_infectives"
    if "aware classification" in t:
        return "aware_classification"
    return None


outline_spans("emhslu_2023.pdf", "emhslu_2023", emhslu_lookup)

# --- UCG 2023: numbered headings in the body ---
ucg_pages = load_pages("uganda_clinical_guidelines_2023")
HEAD = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\s+([A-Za-z][^\n]{2,90})$")
body_started = False
cur = None
ucg_sections = []
for pno, body in ucg_pages:
    for line in body.split("\n"):
        line = line.strip()
        if not body_started:
            if line.startswith("1.1 COMMON EMERGENCIES"):
                body_started = True
            else:
                continue
        m = HEAD.match(line)
        if m and "...." not in line and not re.search(r"\.\d{2,}$", line):
            secno, title = m.group(1), m.group(2).strip()
            title = re.sub(r"\s*ICD\s?1[01]\s?CODES?:?.*$", "", title, flags=re.I).strip()
            if len(title) < 3:
                continue
            cur = dict(secno=secno, title=title, p0=pno, p1=pno, lines=[])
            ucg_sections.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
            cur["p1"] = pno

for s in ucg_sections:
    topic = ucg_topic(s["secno"], s["title"])
    if not topic:
        continue
    aud = "pediatric" if re.search(r"child|neonat|infant|newborn|under 5", s["title"], re.I) else "general"
    add_span(topic, "uganda_clinical_guidelines_2023",
             s["secno"] + " " + s["title"], s["p0"], s["p1"], "\n".join(s["lines"]), aud)

# --- WHO paediatric booklets: keyword-tagged page groups ---
PED_KEY = [
    (r"\bpneumonia\b|fast breathing|chest indrawing", "pneumonia_mild"),
    (r"diarrhoea|dysentery|cholera|dehydration", "acute_diarrhoea"),
    (r"meningitis", "bacterial_meningitis"),
    (r"\bsepsis\b|septicaemia|serious bacterial infection", "sepsis_neonatal_paediatric"),
    (r"ear problem|ear infection|otitis", "acute_otitis_media"),
    (r"sore throat|pharyng|tonsill", "pharyngitis"),
    (r"conjunctivitis|eye infection", "eye_infections"),
    (r"cellulitis|impetigo|skin infection", "skin_soft_tissue_mild"),
]
for stem in ("who_imci_chart_booklet", "who_pocketbook_hospital_care_children"):
    for pno, body in load_pages(stem):
        low = body.lower()
        for pat, topic in PED_KEY:
            # >=3 hits: a page that merely mentions "dehydration" once or twice
            # in passing is not a diarrhoea page.
            if len(re.findall(pat, low)) >= 3:
                add_span(topic, stem, "p." + str(pno), pno, pno, body, "pediatric")
                break

# ---------------------------------------------------------------- write out
topics = {}
for s in spans:
    topics.setdefault(s["topic"], []).append(s)

manifest = []
index = {}
for topic, ss in sorted(topics.items()):
    ss.sort(key=lambda s: (s["doc"] != "who_aware_book_2022", s["page_start"]))
    lines = ["# TOPIC: " + topic, ""]
    for s in ss:
        header = ("--- [" + DOC_LABEL[s["doc"]] + " | " + s["section"] +
                  " | p." + str(s["page_start"]) + "-" + str(s["page_end"]) +
                  " | audience=" + s["audience"] + "] ---")
        lines += [header, s["text"], ""]
        t, i, part = s["text"], 0, 0
        while i < len(t):
            piece = t[i:i + MAX_CHARS]
            if len(piece) >= MIN_CHARS:
                part += 1
                manifest.append(dict(
                    chunk_id=(topic + "::" + s["doc"] + "::p" + str(s["page_start"]) +
                              "-" + str(s["page_end"]) + "::" + str(part)),
                    topic=topic, doc=s["doc"], doc_label=DOC_LABEL[s["doc"]],
                    section=s["section"], page_start=s["page_start"],
                    page_end=s["page_end"], audience=s["audience"],
                    chars=len(piece), text=piece))
            i += MAX_CHARS
    (OUT / (topic + ".txt")).write_text("\n".join(lines), encoding="utf-8")
    index[topic] = [dict(doc=DOC_LABEL[s["doc"]], section=s["section"],
                         pages=str(s["page_start"]) + "-" + str(s["page_end"]),
                         audience=s["audience"], chars=len(s["text"])) for s in ss]

with open(OUT / "_manifest.jsonl", "w", encoding="utf-8") as f:
    for m in manifest:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
(OUT / "_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

print("topics: %d   spans: %d   generation sub-chunks: %d" % (len(topics), len(spans), len(manifest)))
print("%-36s%6s%11s%10s  %s" % ("TOPIC", "SPANS", "SUBCHUNKS", "CHARS", "SOURCES"))
SHORT = {"who_aware_book_2022": "AWaRe", "uganda_clinical_guidelines_2023": "UCG",
         "emhslu_2023": "EMHSLU", "who_imci_chart_booklet": "IMCI",
         "who_pocketbook_hospital_care_children": "PocketBk"}
for topic, ss in sorted(topics.items()):
    n_sub = sum(1 for m in manifest if m["topic"] == topic)
    srcs = sorted({SHORT[s["doc"]] for s in ss})
    print("%-36s%6d%11d%10d  %s" % (topic, len(ss), n_sub,
                                    sum(len(s["text"]) for s in ss), ",".join(srcs)))
