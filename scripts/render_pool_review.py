"""Render the scenario pool as the clinical review page.

Generated from scenario_pool_data.POOL rather than written by hand, so the page
and the gate cannot disagree about what the pool says. Every quote on the page
has passed scenario_pool.verify() against the chunk it cites.
"""
import html
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario_pool as sp
from scenario_pool_data import POOL

TITLES = {
    "acute_bronchitis": "Acute bronchitis",
    "copd_exacerbation": "COPD exacerbation",
    "acute_otitis_media": "Acute otitis media",
    "acute_sinusitis": "Acute sinusitis",
    "pharyngitis": "Pharyngitis",
    "acute_diarrhoea": "Acute infectious diarrhoea",
}
KIND_LABEL = {
    "default": "Default",
    "exception": "Exception criterion",
    "non_criterion": "Explicitly not a criterion",
    "substitute": "Substitute management",
    "escalation": "Escalation trigger",
    "first_choice": "First choice",
    "conflict": "Source conflict",
}
WITHHOLD = ["default", "non_criterion", "substitute", "escalation"]
TREAT = ["exception", "first_choice"]

E = html.escape


def op_words(b):
    sym = {">=": "≥", ">": ">", "<=": "≤", "<": "<"}[b["op"]]
    unit = {"temp_c": "°C", "days": " days", "years": " years",
            "months": " months", "bpm": " bpm", "rr": "/min"}[b["kind"]]
    label = {"temp_c": "Temperature", "days": "Duration", "years": "Age",
             "months": "Age", "bpm": "Pulse", "rr": "Resp. rate"}[b["kind"]]
    v = b["value"]
    v = int(v) if float(v).is_integer() else v
    return "%s %s%s%s" % (label, sym, v, unit)


def entry_html(e):
    parts = ['<article class="stmt %s">' % e["kind"]]
    parts.append('<header class="stmt-head">')
    parts.append('<span class="sid">%s</span>' % E(e["id"]))
    parts.append('<span class="kind k-%s">%s</span>' % (e["kind"], E(KIND_LABEL[e["kind"]])))
    b = e.get("bound")
    if b:
        parts.append('<span class="bound">%s</span>' % E(op_words(b)))
    if e.get("min_limbs", 1) > 1:
        n = len(e.get("cues", ())) + (1 if b else 0)
        parts.append('<span class="limbs">needs %d of %d</span>' % (e["min_limbs"], n))
    parts.append("</header>")
    parts.append('<blockquote class="quote">%s</blockquote>' % E(e["quote"]))
    parts.append('<p class="src">%s &middot; %s &middot; printed p.%s</p>'
                 % (E(e["doc"]), E(e["section"]), E(e["printed_pages"])))
    if e.get("note"):
        parts.append('<p class="note">%s</p>' % E(e["note"]))
    if e.get("non_discriminating"):
        for nd in e["non_discriminating"]:
            parts.append('<p class="flag nd"><span>Cue removed</span>%s</p>' % E(nd))
    if e.get("review_flag"):
        parts.append('<p class="flag rf"><span>Needs a ruling</span>%s</p>' % E(e["review_flag"]))
    parts.append("</article>")
    return "\n".join(parts)


def build():
    pres = sp.presentations()
    idx = sp.chunk_index()
    n_exc = sum(1 for e in POOL if e["kind"] == "exception")

    # same-chunk feasibility, recomputed rather than asserted
    shared = {}
    for p in pres:
        found = []
        for cid, txt in idx.items():
            if not cid.startswith(p + "::"):
                continue
            d = any(sp.normalise(x.get("quote_check") or x["quote"]) in txt
                    for x in sp.entries(p, "default"))
            x_ = any(sp.normalise(x.get("quote_check") or x["quote"]) in txt
                     for x in sp.entries(p, "exception"))
            if d and x_:
                found.append(cid)
        shared[p] = found

    out = []
    A = out.append
    A('<title>AMR Scenario Pool Review</title>')
    A('<link rel="preconnect" href="https://fonts.googleapis.com">')
    A('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=Newsreader:opsz,wght@6..72,400;6..72,600&'
      'family=Public+Sans:wght@400;500;600&'
      'family=IBM+Plex+Mono:wght@400;500&display=swap">')
    A("<style>%s</style>" % CSS)

    A('<div class="wrap">')

    # ---- header
    A('<header class="page-head">')
    A('<p class="eyebrow">OneAMR &middot; dataset construction &middot; review checkpoint</p>')
    A('<h1>The Category 1 / Category 2 exception boundary</h1>')
    A('<p class="lede">Categories 1 and 2 are not two categories. For each presentation the '
      'guidelines state a default &mdash; almost always <em>no antibiotic</em> &mdash; and a named set '
      'of criteria that move a patient across it. This is every statement the corpus makes '
      'about those six boundaries. Generation is held until it is reviewed.</p>')
    A('<dl class="stats">')
    for k, v in (("Statements", len(POOL)), ("Presentations", len(pres)),
                 ("Exception criteria", n_exc), ("Quotes verified", "%d / %d" % (len(POOL), len(POOL)))):
        A('<div><dt>%s</dt><dd>%s</dd></div>' % (k, v))
    A('</dl>')
    A('<p class="verify">Every quote below was matched mechanically against the source chunk it '
      'cites before this page was built. Nothing entered the pool because it is clinically '
      'true &mdash; only because the corpus states it.</p>')
    A('</header>')

    # ---- what needs a ruling
    A('<section class="asks">')
    A('<h2>Four things that need your ruling</h2>')
    A('<p class="asks-lede">Everything else is transcription. These four are judgement calls I '
      'made to keep the pool machine-checkable, and each one could reasonably go the other way.</p>')
    A('<ol class="asklist">')
    A('<li><h3>How many limbs does the UCG COPD criterion require?</h3>'
      '<p>UCG 5.1.2 reads <q>If more sputum, changed to more yellow/green coloured, and/or '
      'breathlessness, temp &gt;38&deg;C and or rapid breathing</q>. The <em>and/or</em> does not '
      'fix how many of those must be present. I encoded it as <strong>two of three</strong> '
      '(sputum increase, colour change, rapid breathing), with the temperature counting as a '
      'limb. Confirm or correct.</p></li>')
    A('<li><h3>Dropping breathlessness as a COPD cue.</h3>'
      '<p>Every COPD exacerbation is breathless, on both sides of the boundary, so the cue '
      'marked every correct Category&nbsp;2 vignette as crossing. I removed it. The same '
      'reasoning removed <em>sore throat</em> from pharyngitis. The risk is that a real '
      'criterion is now invisible to the check.</p></li>')
    A('<li><h3>Acute bronchitis cannot have a same-chunk pair.</h3>'
      '<p>The AWaRe chapter states the default and names <strong>no exception at all</strong>. '
      'UCG 5.2.2 names the only exception in the corpus. So a bronchitis Category&nbsp;1 vignette '
      'rests on a criterion the WHO source does not recognise. Options: generate the pair across '
      'the two documents anyway, or drop bronchitis from Category&nbsp;1 and keep it as a pure '
      'Category&nbsp;2 presentation.</p></li>')
    A('<li><h3>Two source conflicts, recorded and left unresolved.</h3>'
      '<p>BRO-90 and CPD-90 below. For COPD the two sources disagree on the threshold, the '
      'facility level and the duration. I did not pick a winner, because picking one means '
      'choosing a national standard on the model&rsquo;s behalf. Tell me if you want a house '
      'rule instead.</p></li>')
    A('</ol>')
    A('</section>')

    # ---- legend
    A('<section class="legend">')
    A('<h2>How to read a statement</h2>')
    A('<div class="legend-grid">')
    for k in ("default", "exception", "non_criterion", "substitute", "escalation",
              "first_choice", "conflict"):
        desc = {
            "default": "What the guideline says to do when no criterion is met.",
            "exception": "A criterion that moves the patient to antibiotic indicated. Only these are machine-checked against a vignette.",
            "non_criterion": "Named by the guideline specifically so it is <em>not</em> mistaken for an indication, or declared unsettled.",
            "substitute": "The management that replaces the antibiotic. A Category 2 answer without one is incomplete.",
            "escalation": "What should trigger review or referral. Not an antibiotic trigger.",
            "first_choice": "The agent for the far side of the boundary, for Category 1.",
            "conflict": "Two corpus sources that do not agree. Recorded, not resolved.",
        }[k]
        A('<div class="leg"><span class="kind k-%s">%s</span><p>%s</p></div>'
          % (k, KIND_LABEL[k], desc))
    A('</div>')
    A('<p class="legend-foot"><span class="bound">Temperature &ge;39&deg;C</span> is a numeric bound the '
      'source states. <span class="limbs">needs 2 of 2</span> marks a conjunctive criterion &mdash; '
      'every limb must be present, not just one.</p>')
    A('</section>')

    # ---- presentations
    for p in pres:
        es = sp.entries(p)
        A('<section class="pres" id="%s">' % E(p))
        A('<header class="pres-head">')
        A("<h2>%s</h2>" % E(TITLES[p]))
        sc = shared[p]
        if sc:
            A('<p class="chunkline ok">Same-chunk C1/C2 pairs possible &middot; '
              '<code>%s</code></p>' % E(sc[0]))
        else:
            A('<p class="chunkline no">No single chunk carries both sides &mdash; '
              'a pair here must span two documents</p>')
        A('</header>')

        A('<div class="cols">')
        A('<div class="col withhold"><h3 class="colhead">Withhold</h3>')
        for e in es:
            if e["kind"] in WITHHOLD:
                A(entry_html(e))
        A("</div>")
        A('<div class="col treat"><h3 class="colhead">Cross the boundary</h3>')
        for e in es:
            if e["kind"] in TREAT:
                A(entry_html(e))
        A("</div>")
        A("</div>")

        for e in es:
            if e["kind"] == "conflict":
                A(entry_html(e))
        A("</section>")

    # ---- limitations
    A('<section class="limits">')
    A("<h2>Two limitations that stay</h2>")
    A('<p class="asks-lede">Both are honest boundaries of the method, not gaps to patch. They '
      'are stated here in the form they will take in the paper.</p>')
    A('<div class="limit"><h3>Residual judgement beyond the named criteria</h3>'
      '<p>The pool enumerates what the guidelines write down. It cannot enumerate '
      '<q>systemically very unwell</q> (AOM-02), or the <q>severe</q> qualifying '
      '<q>severe bloody diarrhoea</q> (DIA-02), or what makes an exacerbation severe enough '
      'to need admission. So the check is a <strong>necessary condition, not a sufficient '
      'one</strong>: it can prove a Category&nbsp;2 vignette satisfies no named criterion, and '
      'that a Category&nbsp;1 vignette satisfies at least one. It cannot prove the satisfied '
      'criterion is clinically sufficient, nor that a vignette with no named criterion is '
      'genuinely uncomplicated.</p></div>')
    A('<div class="limit"><h3>Source consistency versus explanation quality</h3>'
      '<p>Grounding every statement in a named passage guarantees the statement is in the '
      'corpus. It guarantees nothing about whether the surrounding explanation is good, and '
      'where two corpus sources disagree it cannot say which is right &mdash; it can only record '
      'that they differ. BRO-90 and CPD-90 are the concrete instances: same presentation, '
      'different boundary, different duration, both sources authoritative for this setting.</p>'
      '</div>')
    A("</section>")

    A('<footer class="foot"><p>Generated from '
      '<code>scripts/scenario_pool_data.py</code> &middot; quotes verified by '
      '<code>scripts/scenario_pool.py</code> &middot; gates in '
      '<code>c2_exception_guard.py</code>, probed by '
      '<code>.scratch/probe_c2_exception_smuggle.py</code> and '
      '<code>.scratch/probe_c1c2_discrimination.py</code></p></footer>')
    A("</div>")
    return "\n".join(out)


CSS = """
:root{
  --paper:#f7f8f7; --card:#ffffff; --ink:#16232b; --ink-2:#3d4f57; --muted:#6b7a80;
  --rule:#dfe4e3; --rule-2:#eceff0;
  --withhold:#2f5d6b; --withhold-bg:#eaf1f2;
  --treat:#9a5b2c; --treat-bg:#f7efe6;
  --conflict:#8c3b3b; --conflict-bg:#f8ecec;
  --accent:#2f5d6b;
  --shadow:0 1px 2px rgba(22,35,43,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#101a1f; --card:#16232b; --ink:#e6ecec; --ink-2:#b3c2c6; --muted:#8b9a9f;
    --rule:#26363d; --rule-2:#1d2c33;
    --withhold:#8fc0cc; --withhold-bg:#162b32;
    --treat:#d9a273; --treat-bg:#2c211a;
    --conflict:#d78e8e; --conflict-bg:#2c1c1c;
    --accent:#8fc0cc;
    --shadow:0 1px 2px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"]{
  --paper:#101a1f; --card:#16232b; --ink:#e6ecec; --ink-2:#b3c2c6; --muted:#8b9a9f;
  --rule:#26363d; --rule-2:#1d2c33;
  --withhold:#8fc0cc; --withhold-bg:#162b32;
  --treat:#d9a273; --treat-bg:#2c211a;
  --conflict:#d78e8e; --conflict-bg:#2c1c1c;
  --accent:#8fc0cc;
  --shadow:0 1px 2px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{
  background:var(--paper); color:var(--ink);
  font-family:"Public Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.6; margin:0;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px; margin:0 auto; padding:56px 28px 80px;
  display:flex; flex-direction:column; gap:56px}
h1,h2,h3{font-family:Newsreader,Georgia,serif; font-weight:600;
  text-wrap:balance; margin:0; line-height:1.2}
h1{font-size:2.45rem; letter-spacing:-.015em}
h2{font-size:1.6rem}
h3{font-size:1.05rem}
p{margin:0}
q{font-style:italic}
code{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.82em}

.eyebrow{font-size:.72rem; text-transform:uppercase; letter-spacing:.13em;
  color:var(--muted); font-weight:600; margin-bottom:14px}
.page-head{display:flex; flex-direction:column; gap:18px;
  border-bottom:2px solid var(--ink); padding-bottom:30px}
.lede{font-size:1.12rem; color:var(--ink-2); max-width:64ch}
.verify{font-size:.9rem; color:var(--muted); max-width:70ch;
  border-left:2px solid var(--accent); padding-left:14px}

.stats{display:flex; flex-wrap:wrap; gap:38px; margin:6px 0 0; padding:0}
.stats div{display:flex; flex-direction:column; gap:2px}
.stats dt{font-size:.7rem; text-transform:uppercase; letter-spacing:.1em;
  color:var(--muted); font-weight:600}
.stats dd{margin:0; font-family:Newsreader,Georgia,serif; font-size:1.9rem;
  font-variant-numeric:tabular-nums; line-height:1}

.asks{display:flex; flex-direction:column; gap:18px}
.asks-lede{color:var(--ink-2); max-width:66ch}
.asklist{margin:0; padding:0; list-style:none; counter-reset:ask;
  display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(310px,1fr))}
.asklist li{counter-increment:ask; background:var(--card); border:1px solid var(--rule);
  border-top:3px solid var(--accent); padding:18px 20px 20px;
  display:flex; flex-direction:column; gap:8px; box-shadow:var(--shadow)}
.asklist h3::before{content:counter(ask); font-family:"IBM Plex Mono",monospace;
  font-size:.75rem; color:var(--accent); margin-right:9px; vertical-align:2px}
.asklist p{font-size:.93rem; color:var(--ink-2)}

.legend{display:flex; flex-direction:column; gap:16px}
.legend-grid{display:grid; gap:12px 26px;
  grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
.leg{display:flex; flex-direction:column; gap:6px; align-items:flex-start}
.leg p{font-size:.88rem; color:var(--ink-2)}
.legend-foot{font-size:.88rem; color:var(--ink-2); padding-top:12px;
  border-top:1px solid var(--rule)}

.kind{font-size:.68rem; text-transform:uppercase; letter-spacing:.09em;
  font-weight:600; padding:3px 8px; border-radius:2px; white-space:nowrap}
.k-default{background:var(--withhold-bg); color:var(--withhold)}
.k-exception{background:var(--treat-bg); color:var(--treat)}
.k-non_criterion{background:var(--rule-2); color:var(--ink-2)}
.k-substitute{background:var(--rule-2); color:var(--ink-2)}
.k-escalation{background:var(--rule-2); color:var(--ink-2)}
.k-first_choice{background:var(--treat-bg); color:var(--treat)}
.k-conflict{background:var(--conflict-bg); color:var(--conflict)}

.pres{display:flex; flex-direction:column; gap:20px;
  border-top:1px solid var(--rule); padding-top:34px}
.pres-head{display:flex; flex-direction:column; gap:6px}
.chunkline{font-size:.82rem; font-family:"IBM Plex Mono",monospace}
.chunkline.ok{color:var(--muted)}
.chunkline.no{color:var(--conflict)}
.chunkline code{font-size:.95em}

.cols{display:grid; gap:26px; grid-template-columns:1fr 1fr; align-items:start}
@media (max-width:820px){.cols{grid-template-columns:1fr}}
.col{display:flex; flex-direction:column; gap:12px; min-width:0}
.colhead{font-size:.72rem; text-transform:uppercase; letter-spacing:.12em;
  font-family:"Public Sans",sans-serif; font-weight:600; padding-bottom:7px;
  border-bottom:2px solid currentColor}
.withhold .colhead{color:var(--withhold)}
.treat .colhead{color:var(--treat)}

.stmt{background:var(--card); border:1px solid var(--rule);
  padding:15px 17px; display:flex; flex-direction:column; gap:9px; min-width:0}
.stmt.exception{border-left:3px solid var(--treat)}
.stmt.first_choice{border-left:3px solid var(--treat)}
.stmt.default{border-left:3px solid var(--withhold)}
.stmt.conflict{border-left:3px solid var(--conflict); background:var(--conflict-bg)}
.stmt-head{display:flex; flex-wrap:wrap; gap:8px; align-items:center}
.sid{font-family:"IBM Plex Mono",monospace; font-size:.76rem; font-weight:500;
  color:var(--muted); letter-spacing:.02em}
.bound{font-family:"IBM Plex Mono",monospace; font-size:.72rem; font-weight:500;
  color:var(--treat); border:1px solid currentColor; padding:2px 7px; border-radius:2px;
  font-variant-numeric:tabular-nums}
.limbs{font-size:.68rem; text-transform:uppercase; letter-spacing:.08em;
  font-weight:600; color:var(--conflict)}
.quote{margin:0; font-family:"IBM Plex Mono",monospace; font-size:.83rem;
  line-height:1.55; color:var(--ink); padding-left:12px;
  border-left:2px solid var(--rule); overflow-wrap:anywhere}
.src{font-size:.75rem; color:var(--muted)}
.note{font-size:.88rem; color:var(--ink-2)}
.flag{font-size:.83rem; display:flex; flex-direction:column; gap:3px;
  padding:9px 11px; border-radius:2px}
.flag span{font-size:.66rem; text-transform:uppercase; letter-spacing:.09em; font-weight:600}
.flag.nd{background:var(--rule-2); color:var(--ink-2)}
.flag.nd span{color:var(--muted)}
.flag.rf{background:var(--conflict-bg); color:var(--ink-2)}
.flag.rf span{color:var(--conflict)}

.limits{display:flex; flex-direction:column; gap:16px;
  border-top:2px solid var(--ink); padding-top:34px}
.limit{display:flex; flex-direction:column; gap:8px; max-width:72ch}
.limit p{color:var(--ink-2); font-size:.95rem}

.foot{border-top:1px solid var(--rule); padding-top:22px;
  font-size:.78rem; color:var(--muted); line-height:1.9}
"""


if __name__ == "__main__":
    bad = sp.verify()
    if bad:
        print("REFUSING to render: %d quote(s) do not verify" % len(bad))
        for pid, why in bad:
            print("  %s  %s" % (pid, why))
        sys.exit(1)
    dest = sys.argv[1] if len(sys.argv) > 1 else "pool_review.html"
    io.open(dest, "w", encoding="utf-8").write(build())
    print("wrote %s (%d statements)" % (dest, len(POOL)))
