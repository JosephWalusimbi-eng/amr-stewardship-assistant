"""
Builds the two reference lists the generator grounds on and the validator checks
against.

1. source_material/aware_tiers.json          antibiotic -> AWaRe category
   From the WHO AWaRe classification 2023 workbook (web annex C of "The selection
   and use of essential medicines 2023"). This is needed because the AWaRe
   *book*'s per-drug tier badges are graphics -- they do not survive PDF text
   extraction, so the book alone cannot ground a "which tier is X?" answer.

2. source_material/emhslu_drugs.json         antibacterial -> formulations
   From EMHSLU 2023 section 6.2, so the validator can flag any drug a generated
   pair names that is not actually on Uganda's essential medicines list.

Also writes source_material/chunks/aware_classification_table.txt so the tier
table is available as ordinary grounding context like every other chunk.

Usage:  python scripts/build_reference_lists.py
"""
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
ROOT = Path(__file__).resolve().parent.parent
SM = ROOT / "source_material"
XLSX = ROOT / "corpus" / "aware_classification_2023.xlsx"

VALID_TIERS = {"Access", "Watch", "Reserve"}


# ----------------------------------------------------------------- xlsx (stdlib)
def read_workbook(path):
    """Yields every populated row of every sheet as a {column_letter: value} dict."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
    for name in sorted(n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)):
        for row in ET.fromstring(z.read(name)).iter(NS + "row"):
            cells = {}
            for c in row.findall(NS + "c"):
                v, isel = c.find(NS + "v"), c.find(NS + "is")
                if c.get("t") == "s" and v is not None:
                    val = shared[int(v.text)]
                elif isel is not None:
                    val = "".join(t.text or "" for t in isel.iter(NS + "t"))
                elif v is not None:
                    val = v.text
                else:
                    continue
                cells[re.match(r"[A-Z]+", c.get("r")).group(0)] = (val or "").strip()
            if cells:
                yield cells


def build_aware_tiers():
    tiers, conflicts, not_recommended = {}, [], []
    seen_not_rec_header = False
    for row in read_workbook(XLSX):
        name, tier = row.get("A", "").strip(), row.get("D", "").strip()
        if not name or name.lower() == "antibiotic":
            if name.lower() == "antibiotic" and "B" not in row:
                seen_not_rec_header = True     # the "Not recommended" sheet is name-only
            continue
        if tier in VALID_TIERS:
            key = name.lower()
            prev = tiers.get(key)
            if prev and prev["category"] != tier:
                conflicts.append((name, prev["category"], tier))
                continue
            tiers[key] = {"antibiotic": name, "category": tier,
                          "class": row.get("B", ""), "atc": row.get("C", ""),
                          "on_eml": row.get("E", "")}
        elif seen_not_rec_header and "B" not in row and "/" in name and len(name) < 120:
            not_recommended.append(name)
    return tiers, conflicts, sorted(set(not_recommended))


# ----------------------------------------------------------------- EMHSLU 6.2
DOSAGE_FORM = (r"(?:Dispersible tablet|Powder for ?injection|Oral suspension|Oral liquid|"
               r"Eye ointment|Eye drops|Ear drops|Tablets?|Capsules?|Injection|Suspension|"
               r"Syrup|Cream|Ointment|Solution|Infusion|Granules|Sachet|Powder)")
LEVEL = r"(?:HC[1-4]|RR|NRH|RRH|H|GH)"


def build_emhslu_drugs():
    txt = (SM / "emhslu_2023.txt").read_text(encoding="utf-8")
    # Anchor on the BODY heading, not the identically-worded table-of-contents
    # entry: only the body is immediately followed by subsection 6.2.1.
    m = re.search(r"6\.2 Antibacterials\s*\n6\.2\.1(.*?)(?:\n6\.[3-9] |\n7\. )", txt, re.S)
    body = m.group(1) if m else ""

    # Rejoin drug names broken across a line ("Rifampicin + Clofazimine +\nDapsone
    # Tablet ..."), otherwise the row parses under the wrong drug name.
    raw = body.split("\n")
    lines, i = [], 0
    while i < len(raw):
        cur = raw[i].strip()
        if cur.endswith("+") and i + 1 < len(raw):
            cur = cur + " " + raw[i + 1].strip()
            i += 1
        lines.append(cur)
        i += 1

    drugs, subsection = {}, "6.2.1"
    for line in lines:
        line = line.strip()
        if re.match(r"^6\.2\.\d+ ", line):
            subsection = line
            continue
        mm = re.match(r"^(.+?)\s+(" + DOSAGE_FORM + r")\s+(.*?)\s+(" + LEVEL + r")\s+([VEN])$", line)
        if not mm:
            continue
        name = re.sub(r"\s+", " ", mm.group(1)).strip(" .")
        if len(name) < 3 or name.lower().startswith(("specialist", "section")):
            continue
        entry = drugs.setdefault(name.lower(), {"drug": name, "subsection": subsection,
                                                "formulations": []})
        entry["formulations"].append({"form": mm.group(2), "strength": mm.group(3),
                                      "level": mm.group(4), "ven": mm.group(5)})
    return drugs


def main():
    if not XLSX.exists():
        raise SystemExit("missing %s -- download the AWaRe classification workbook first" % XLSX)

    tiers, conflicts, not_recommended = build_aware_tiers()
    counts = {}
    for v in tiers.values():
        counts[v["category"]] = counts.get(v["category"], 0) + 1
    (SM / "aware_tiers.json").write_text(json.dumps(
        {"source": "WHO AWaRe classification 2023 (WHO-MHP-HPS-EML-2023.04, web annex C)",
         "counts": counts, "not_recommended": not_recommended,
         "antibiotics": tiers}, indent=2), encoding="utf-8")
    print("aware_tiers.json: %d antibiotics  %s  not-recommended=%d  conflicts=%d"
          % (len(tiers), counts, len(not_recommended), len(conflicts)))
    for c in conflicts[:5]:
        print("   conflict:", c)

    drugs = build_emhslu_drugs()
    (SM / "emhslu_drugs.json").write_text(json.dumps(
        {"source": "EMHSLU 2023 section 6.2 Antibacterials", "drugs": drugs},
        indent=2), encoding="utf-8")
    print("emhslu_drugs.json: %d antibacterials, %d formulations"
          % (len(drugs), sum(len(d["formulations"]) for d in drugs.values())))

    # cross-check: which EMHSLU antibacterials carry a WHO AWaRe tier?
    #
    # Two wrinkles in WHO's naming. (a) Route-dependent tiers are encoded as a
    # suffix -- Metronidazole_oral is Access but Metronidazole_IV is Watch, same
    # for Vancomycin and Colistin -- so "what tier is metronidazole?" has no
    # single correct answer and the validator must know that. (b) EMHSLU uses
    # Uganda's drug names, which differ from WHO's INN spellings.
    ALIASES = {
        "cotrimoxazole": "sulfamethoxazole/trimethoprim",
        "benzathine penicillin": "benzathine-benzylpenicillin",
        "procaine benzylpenicillinforte": "procaine-benzylpenicillin",
        "procaine benzylpenicillin forte": "procaine-benzylpenicillin",
    }
    SALTS = r"\b(sodium|potassium|sulphate|sulfate|hydrochloride|citrate|forte|base)\b"

    def norm(s):
        s = s.lower()
        s = ALIASES.get(s.strip(), s)
        s = re.sub(SALTS, " ", s)
        return re.sub(r"[^a-z0-9]", "", s)

    route_map = {}     # base drug -> {route: category}
    flat_map = {}      # base drug -> category (only when unambiguous)
    for v in tiers.values():
        name = v["antibiotic"]
        m = re.match(r"^(.*)_(IV|oral)$", name)
        if m:
            route_map.setdefault(norm(m.group(1)), {})[m.group(2).lower()] = v["category"]
        else:
            flat_map[norm(name)] = v["category"]

    matched, unmatched, route_dep = [], [], []
    for d in drugs.values():
        key = norm(d["drug"])
        if key in route_map:
            route_dep.append((d["drug"], route_map[key]))
            matched.append(d["drug"])
        elif key in flat_map:
            matched.append(d["drug"])
        else:
            unmatched.append(d["drug"])
    print("EMHSLU antibacterials with a WHO AWaRe tier: %d/%d" % (len(matched), len(drugs)))
    if route_dep:
        print("   route-dependent tier (no single answer without a route):")
        for name, routes in sorted(route_dep):
            print("      %-16s %s" % (name, routes))
    if unmatched:
        print("   no tier match (expected for anti-TB / antileprosy / non-antibacterials):")
        print("      " + ", ".join(sorted(unmatched)))

    # persist the lookup the validator uses
    (SM / "aware_lookup.json").write_text(json.dumps(
        {"note": "normalised name -> AWaRe tier. 'route_dependent' entries have no "
                 "single correct tier; an answer must state the route.",
         "flat": flat_map, "route_dependent": route_map,
         "aliases": ALIASES}, indent=2), encoding="utf-8")
    print("wrote aware_lookup.json (%d flat, %d route-dependent)"
          % (len(flat_map), len(route_map)))

    # emit the tier table as an ordinary grounding chunk
    lines = ["# TOPIC: aware_classification_table", "",
             "--- [WHO AWaRe classification 2023 (WHO-MHP-HPS-EML-2023.04, web annex C) "
             "| full classification | audience=general] ---",
             "Antibiotic | Class | ATC code | AWaRe category | On EML/EMLc 2023"]
    for v in sorted(tiers.values(), key=lambda x: (x["category"], x["antibiotic"])):
        lines.append("%s | %s | %s | %s | %s" % (v["antibiotic"], v["class"], v["atc"],
                                                 v["category"], v["on_eml"]))
    if not_recommended:
        lines += ["", "Not recommended (fixed-dose combinations WHO advises against):"]
        lines += not_recommended
    (SM / "chunks" / "aware_classification_table.txt").write_text("\n".join(lines), encoding="utf-8")
    print("wrote chunks/aware_classification_table.txt (%d lines)" % len(lines))


if __name__ == "__main__":
    main()
