# Source corpus — provenance

Retrieval date for every network fetch below: **2026-09-03**.
Checksums are of the PDF as it sits in `corpus/`. Extracted text lives in
`source_material/<stem>.txt` with `[[page N]]` markers; per-document extraction
statistics are in `source_material/extraction_report.json`.

---

## 1. WHO AWaRe (Access, Watch, Reserve) antibiotic book — ✅ obtained, verified against publisher

| | |
|---|---|
| Publisher | World Health Organization, 2022 |
| ISBN | 978-92-4-006238-2 |
| Landing page | https://www.who.int/publications/i/item/9789240062382 |
| IRIS record | https://iris.who.int/handle/10665/365237 (item uuid `e3a52660-f2d4-4e3c-a311-e48418ad21e0`) |
| Canonical PDF | https://iris.who.int/server/api/core/bitstreams/61517f8b-3a34-413c-ae36-67a059ebe485/content |
| Local file | `corpus/aware_book.pdf` — 697 pages, 10,323,301 bytes |
| md5 | `4dfb4232bf2d192657831d98146a41c8` |
| sha256 | `4960ecc4fa5bab8281feda656a0da3176dafa1fd21971d87e3b2092d7dac562f` |
| Licence | CC BY-NC-SA 3.0 IGO |

**Verification:** the repo already held this file. Its md5 was compared against the
checksum WHO's IRIS DSpace API reports for the master bitstream and is **byte-identical**.

Note: the legacy `iris.who.int/bitstream/handle/...` URLs no longer serve the PDF —
IRIS now runs a DSpace 7 single-page app and those paths return the HTML shell.
Use the `/server/api/core/bitstreams/<uuid>/content` URL above.

---

## 2. Uganda Clinical Guidelines 2023, 4th edition — ✅ obtained, publisher site unreachable

| | |
|---|---|
| Publisher | Republic of Uganda, Ministry of Health |
| Canonical URL | https://library.health.go.ug/sites/default/files/resources/Uganda%20Clinical%20Guidelines%202023.pdf |
| Local file | `corpus/ucg_2023_full.pdf` — 1,161 pages, 13,527,788 bytes |
| md5 | `c2be0c0be8bc37cf55f58327fc4d1059` |
| sha256 | `d7814fcc7f0b0575a54bcdaeae9e1c88d04122c2732d6a0a9ff1d181ae456f02` |
| PDF creation date | 2023-12-19 (Adobe PDF Library 17.0) |

**Verification:** could not be re-downloaded — `library.health.go.ug` refused all
connections on 2026-09-03 from two independent networks (`ECONNREFUSED 154.72.196.19:443`;
`curl` exit with no response). The MoH root `www.health.go.ug` was likewise unreachable.
The repo copy was therefore verified **by content**, not by checksum: MoH/Republic of
Uganda title page, Foreword and Acknowledgements naming the MoH Update Task Force and
URMCHIP, 1,161 pages, and the full chapter structure (1 Emergencies … 17 Childhood
Illness) matching the published 4th edition.

`corpus/ucg_2023.pdf` (405 pages) is a **derived** file — a locally produced pypdf trim of
the full document used by the existing RAG index. All new chunking works from
`ucg_2023_full.pdf`, never the trim.

⚠️ Open item: re-verify this checksum against the MoH site when it comes back up.

---

## 3. Essential Medicines and Health Supplies List for Uganda (EMHSLU) 2023 — ✅ obtained

| | |
|---|---|
| Publisher | Republic of Uganda, Ministry of Health. First ed. 1997; revised Nov 2012, Dec 2016, **Sept 2023** |
| WHO landing page | https://www.who.int/publications/m/item/uganda--essential-medicines-and-health-supplies-list-for-uganda-(emhslu)-2023-(english) |
| Canonical PDF (used) | https://cdn.who.int/media/docs/default-source/essential-medicines/national-essential-medicines-lists-(neml)/afro_neml/uganda-2023.pdf?sfvrsn=5205bf98_3&download=true |
| MoH portal (unreachable) | https://library.health.go.ug/medical-products-technologies/pharmaceuticals-and-drugs/essential-medicines-and-health-supplies |
| Local file | `corpus/emhslu_2023.pdf` — 118 pages, 2,947,434 bytes |
| md5 | `45e5170bb767b62df47ebf2230b8ddb0` |

**Verification:** first downloaded from a third-party mirror (guluhospital.net), then
re-downloaded from WHO's own CDN. The two PDFs differ by one byte of container metadata
but their **extracted text is identical** (118 pages, 236,029 chars, same sha256). The
WHO-hosted copy is the one kept in `corpus/`.

Antibiotic section confirmed present and clean: §6 ANTI INFECTIVE MEDICINES, §6.2
Antibacterials (6.2.1 Beta-lactams onward), with strengths, dosage forms, facility level
codes (HC1–H, RR) and VEN codes intact.

---

## 4. WHO AWaRe classification of antibiotics 2023 — ✅ added, not originally requested

| | |
|---|---|
| Publisher | WHO, 14 July 2023 — web annex C of *The selection and use of essential medicines 2023* |
| Document no. | WHO-MHP-HPS-EML-2023.04 |
| Landing page | https://www.who.int/publications/i/item/WHO-MHP-HPS-EML-2023.04 |
| IRIS record | item uuid `e1cd5318-656d-4d76-8799-fa6be5ee846e` |
| Canonical file | https://iris.who.int/server/api/core/bitstreams/abba5c2a-8457-431c-9695-16cb2317dd0e/content |
| Local file | `corpus/aware_classification_2023.xlsx` — 118,671 bytes |
| md5 | `f06ab89d84fe72e39189eff623965d4d` |

**Why this was added.** The AWaRe *book* encodes each antibiotic's tier as a **coloured
badge**, not as text — the book says so itself: *"Legend: ACCESS antibiotics are indicated in
green, WATCH antibiotics in orange and RESERVE antibiotics in red"* (Table 12.5 footer).
Colour does not survive PDF text extraction. An audit of the extracted text found tier words
attached to only ~6 drugs anywhere in 1.46M characters, all incidental prose; ceftriaxone,
metronidazole, gentamicin, doxycycline, cloxacillin and most others had **none**.

Without this file, every "state the AWaRe tier" answer would have come from model memory —
exactly what the brief forbids. This is not a substitution for a requested document; all
three requested documents were found. It is an additional grounding source that the task
turned out to require.

Yields 257 antibiotics — **87 Access, 141 Watch, 29 Reserve**, matching WHO's published
counts — plus 103 not-recommended fixed-dose combinations. Parsed by
`scripts/build_reference_lists.py` into `source_material/aware_tiers.json`,
`aware_lookup.json`, and `chunks/aware_classification_table.txt`.

A newer *2025* classification exists (IRIS uuid `4fa2de82-388c-46d9-a6cb-41ffbd10677d`). The
2023 edition was chosen for consistency with the rest of the corpus, which is all 2022–2023
vintage. Worth revisiting if the corpus is refreshed.

---

## Supplementary paediatric sources (already in the repo, not part of the three requested)

| File | Document | Pages | md5 |
|---|---|---|---|
| `corpus/WHO-IMCI.pdf` | WHO IMCI Chart Booklet | 35 | `800d6e2281dba1433fd3f6754f8aa798` |
| `corpus/WHO-pocketbook.pdf` | WHO Pocket Book of Hospital Care for Children, 2nd ed. (2013) | 438 | `37833d107e2f90a0d164ba49201c11b3` |

These pre-date this task and carry no recorded retrieval URL. They are used only as
paediatric supplements, tagged `audience=pediatric`.

---

## Text-quality / garbling audit

Re-run 2026-09-04 after the first audit was found to be incomplete. All five
PDFs carry real text layers (none is a scan), so no OCR was involved.

### What the first pass checked, and what it missed

The original audit looked for U+FFFD replacement characters, `(cid:N)` artifacts
and residual `ﬁ`/`ﬂ` ligature glyphs, and found **zero of each** — which is still
true. But it did not look for **symbol-font glyphs**, and those were everywhere:

| Class | Before | After |
|---|---|---|
| C1 control codepoints (U+0080–U+009F) | **5,506** | 0 |
| Private-use codepoints (U+E000–U+F8FF) | **961** | 4 (kept deliberately, below) |
| U+FFFD replacement | 0 | 0 |
| `(cid:N)` artifacts | 0 | 0 |
| Residual ligature glyphs | 0 | 0 |

These arise when a PDF draws a bullet, arrow or subsetted character from a
Symbol/Wingdings font: the glyph has no Unicode identity, so extraction emits a
private-use or C1 codepoint instead. They are invisible in most editors, which
is why the first pass missed them.

**The same codepoint means different things in different documents**, so the fix
table in `scripts/extract_sources.py` is keyed by document. U+F067 is a
decorative separator in the AWaRe book's running header; U+F077 is a bullet in
UCG's Note/Caution blocks. Neither is the ASCII letter its low byte suggests.

What they were:

- **U+0089 ×3,111 — UCG `TREATMENT` bullets.** The bullet on every treatment
  line, i.e. the drug-and-dose lines that Categories 1 and 4 are grounded in.
- **U+0081 ×1,041, U+0083 ×819, U+008D ×535** — UCG Investigations, Notes and
  Clinical-features bullets; U+0081 also appears 306× in the WHO Pocket Book.
- **U+F067 ×669** — AWaRe book running-header separator.
- **U+F0B7 / U+F0A8 / U+F0FC / U+F0D8 / U+F0BA ×~250** — IMCI list bullets.
- **U+F0AE ×3** — the arrow carrying the *then* of an IMCI decision rule:
  `GENERAL DANGER SIGN → Give one dose of pre-referral antibiotic`.

### The one that mattered: a dose hidden inside private-use codepoints

The IMCI chart booklet uses a subsetted font that maps **plain ASCII** into the
U+F0xx block. In the deworming block, the sequence U+F032 U+F035 U+F030 followed
by ` mg` is literally **`250 mg`** — the paediatric mebendazole dose — and
`U+F042 + "reastfeed"` is `Breastfeed`.

The obvious cleanup for unrecognised private-use characters is to strip them.
That would have **silently deleted the digits of a dose**. Nothing is stripped
blind now: `assert_no_stray_glyphs()` makes an unmapped codepoint a hard error
that stops extraction, so a new document cannot quietly lose characters. Both
strings are verified present in the current extract:

```
- 250 mg if the child is less than 2 years
- 500 mg if the child is 2years of age or older
```

### Deliberately left alone — do not use for grounding

**UCG shock-severity tables.** Four U+F0E2 remain, beside Latin-1 mojibake
(`â`, `á`, `ââ`, `áá`) in the same rows:

```
Pulse pressure   Normal  â  ââ  <U+F0E2>/A
Systolic BP      Normal  N  â   ââ
Capillary refill Normal  á  áá  Absent
```

These are directional arrows whose direction the text layer no longer records.
Physiology makes the intent guessable, but guessing *decreased* against
*increased* in a triage table is not a call an extraction script should make, so
they are kept as-is and flagged. `ALLOWED_STRAY` in `extract_sources.py` is the
single place that whitelists them.

### Other normalisation

Ligature expansion including the `classiﬁ cation` → `classification`
mid-word-space form; de-hyphenation across line breaks for both `work-\ners` and
UCG's `health fa -\ncilities` form, restricted to a lowercase letter on each
side so **no rule can touch a digit**.

### Remaining limitations

1. **Tabular dosing loses column alignment.** UCG weight/age dose tables flatten
   to rows like `10–20 2–7 years 1 1 1`, where the headers no longer bind to the
   values. Any dosing pair drawn from a UCG *table* must be checked against the
   page image, not the text alone. The generation spec forbids Category 4 pairs
   from UCG tables for this reason.
2. **Some WHO Pocket Book front-matter tables are letter-spaced and unusable** —
   e.g. the IV fluid composition table renders as
   `1 0 % g l u c o s e ––––– 1 0 0 4 0 0`. These are on pp. 47–62 of the
   extracted text; not antibiotic dosing tables, but not usable as grounding.
3. **The UCG shock tables above.**

Verified un-garbled by spot check: AWaRe weight-band dosing (`3–6 kg: 250 mg
given every 12 hours`, `80–90 mg/kg/day`, `5 days`), UCG §6.1.5 Diarrhoea
treatment block including `Avoid inappropriate use of antibiotics e.g.
metronidazole, ciprofloxacin` and zinc `20 milligrams per day`, and EMHSLU §6.2.1
beta-lactam strengths and facility levels.

`cotrimoxazole` (23×, unhyphenated) was confirmed to be UCG's **own** spelling by
checking the raw PDF text — it is not a de-hyphenation artifact.

Dose-pattern counts after cleaning, as a regression guard (`extraction_report.json`):

| Document | `N mg` | `mg/kg` | broken numerals |
|---|---|---|---|
| WHO AWaRe book | 2,057 | 688 | 4 |
| UCG 2023 | 1,532 | 417 | 62 |
| EMHSLU 2023 | 711 | 0 | 4 |
| WHO IMCI | 59 | 0 | 5 |
| WHO Pocket Book | 665 | 367 | 45 |
