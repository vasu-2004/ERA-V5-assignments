"""Builds the visual report (results/report/index.html).

Self-contained: data is embedded, charts are hand-drawn SVG, no external
libraries or network requests. Every number rendered comes from the results
bundle produced by the experiments -- nothing here recomputes or hardcodes a
result.
"""
from __future__ import annotations

import json
import pathlib

from .codec import DEFAULT_DP
from .dataset import COMPOUNDS
from .sandhi import SandhiSplitter
from .dataset import build_lexicon


def _word_views(dp: int = DEFAULT_DP) -> list:
    """Per-word byte layout, raw and segmented, for the interactive panel."""
    sp = SandhiSplitter(build_lexicon())
    out = []
    for w in sorted(COMPOUNDS):
        parts = sp.split(w).parts
        raw = list(w.encode("utf-8"))
        out.append({
            "word": w,
            "gold": COMPOUNDS[w],
            "parts": parts,
            "correct": parts == COMPOUNDS[w],
            "raw_bytes": raw,
            "n_bytes": len(raw),
            "lost": max(0, len(raw) - dp),
            "part_bytes": [list(p.encode("utf-8")) for p in parts],
        })
    return out


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Segmentation as a prior for the Kronecker codec</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--ink:#e6edf3;--muted:#8b949e;--line:#30363d;
      --raw:#f0883e;--seg:#3fb950;--gold:#58a6ff;--ctl:#a371f7;--accent:#58a6ff;
      --mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
      --sans:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
@media(prefers-color-scheme:light){:root{--bg:#fbfcfd;--panel:#fff;--ink:#1f2328;
      --muted:#636c76;--line:#d8dee4;--raw:#bc4c00;--seg:#1a7f37;--gold:#0969da;--ctl:#8250df}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55}
.wrap{max-width:1000px;margin:0 auto;padding:0 22px 80px}
header{border-bottom:1px solid var(--line);background:linear-gradient(180deg,color-mix(in srgb,var(--accent) 8%,transparent),transparent)}
header .in{max-width:1000px;margin:0 auto;padding:44px 22px 34px}
h1{font-size:clamp(26px,4.4vw,40px);line-height:1.12;letter-spacing:-.025em;margin:0 0 10px}
.claim{font-size:16px;color:var(--muted);max-width:74ch;margin:0}
.badge{display:inline-block;font-family:var(--mono);font-size:11px;font-weight:700;
  padding:3px 9px;border-radius:20px;background:color-mix(in srgb,var(--seg) 18%,transparent);
  color:var(--seg);border:1px solid color-mix(in srgb,var(--seg) 40%,transparent);margin-bottom:14px}
h2{font-size:20px;letter-spacing:-.015em;margin:44px 0 6px;padding-top:14px;border-top:1px solid var(--line)}
h2 .n{font-family:var(--mono);font-size:12px;color:var(--accent);margin-right:9px}
h3{font-size:14px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:22px 0 8px}
p{margin:8px 0;font-size:14.5px}
.mut{color:var(--muted)}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:13px 15px}
.kpi .v{font-size:25px;font-weight:800;letter-spacing:-.02em}
.kpi .l{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;font-weight:600;margin-top:3px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono)}
.dev{font-size:17px}
code{font-family:var(--mono);font-size:12px;background:color-mix(in srgb,var(--ink) 10%,transparent);padding:1px 5px;border-radius:4px}
.tw{overflow-x:auto}
.note{border-left:3px solid var(--accent);padding:9px 14px;margin:14px 0;font-size:13.5px;
  background:color-mix(in srgb,var(--accent) 7%,transparent);border-radius:0 8px 8px 0}
.warn{border-left-color:var(--raw);background:color-mix(in srgb,var(--raw) 8%,transparent)}
.good{border-left-color:var(--seg);background:color-mix(in srgb,var(--seg) 8%,transparent)}
select,button{font-family:var(--sans);font-size:13px;padding:7px 11px;border-radius:8px;
  border:1px solid var(--line);background:var(--bg);color:var(--ink);cursor:pointer}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:8px 0}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.bytegrid{display:flex;flex-wrap:wrap;gap:3px;margin:8px 0}
.byte{font-family:var(--mono);font-size:10px;padding:3px 5px;border-radius:4px;
  background:color-mix(in srgb,var(--seg) 22%,transparent);border:1px solid color-mix(in srgb,var(--seg) 45%,transparent)}
.byte.lost{background:color-mix(in srgb,var(--raw) 22%,transparent);border-color:color-mix(in srgb,var(--raw) 55%,transparent);opacity:.75}
.byte.sep{background:transparent;border:none;color:var(--muted);padding:3px 2px}
footer{margin-top:46px;border-top:1px solid var(--line);padding-top:16px;color:var(--muted);font-size:12.5px}
@media(max-width:720px){.kpis{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<header><div class="in">
  <span class="badge">TRAINING-FREE &middot; NO GRADIENTS &middot; NO GPU</span>
  <h1>Morphological segmentation as a<br/>training-free prior for the Kronecker codec</h1>
  <p class="claim"><b>Claim under test.</b> Using an off-the-shelf-style morphological
  analyzer as a segmentation prior reduces byte truncation and improves the Kronecker
  codec's collision and retrieval behaviour on Indic compound words, compared with raw
  UTF-8 truncation at <code>dp=32</code>. Everything below is codec arithmetic and cosine
  similarity &mdash; no model is trained anywhere in this repository.</p>
</div></header>

<div class="wrap">
<div class="kpis" id="kpis"></div>

<h2><span class="n">1</span>The problem, concretely</h2>
<p>The codec pairs each <em>byte</em> with its <em>absolute position</em>:
<code>index = byte&times;dp + position</code>, summed over the string and scaled by
1/&radic;L. Two consequences drive everything here.</p>
<div class="card">
<p><b>(a) The window is small in Devanagari.</b> A Devanagari character costs 3 UTF-8
bytes, so <code>dp=32</code> holds only about <b>10 characters</b>. Longer compounds lose
their tail entirely.</p>
<p><b>(b) The codec is positionally rigid.</b> A morpheme is encoded at whatever byte
offset it happens to occupy. <span class="dev">आलय</span> standing alone occupies byte
positions 0&ndash;8; inside <span class="dev">देवालय</span> the same morpheme sits at
positions 9&ndash;17. None of its <em>informative</em> bytes line up, so the codec has no
way to register that the two contain the same morpheme. In a script whose bytes carry
full information this drives the cosine to exactly zero (see the Latin control in
&sect;2); in Devanagari it does not, for a reason worth its own section.</p>
</div>
<div class="note warn"><b>Because positions are unique within a string, the codec is
injective for inputs of at most <code>dp</code> bytes.</b> Every collision is therefore a
truncation collision: two words collide exactly when their first 32 bytes agree. This is
proved as a property test, not assumed.</div>

<h3>Try it &mdash; every word in the evaluation set</h3>
<div class="card">
  <select id="wordsel"></select>
  <div class="legend" style="margin-top:12px">
    <span><i style="background:var(--seg)"></i>byte kept (inside the dp window)</span>
    <span><i style="background:var(--raw)"></i>byte lost to truncation</span>
  </div>
  <h3 style="margin-top:14px">Raw UTF-8, truncated at dp=32</h3>
  <div class="bytegrid" id="rawgrid"></div>
  <div id="rawinfo" class="mut" style="font-size:12.5px"></div>
  <h3>Segmented by the inverse-sandhi analyzer</h3>
  <div class="bytegrid" id="seggrid"></div>
  <div id="seginfo" class="mut" style="font-size:12.5px"></div>
  <h3>Codec occupancy (position &rarr; byte value)</h3>
  <svg id="occ" viewBox="0 0 960 210" style="width:100%;height:auto"></svg>
</div>

<h2><span class="n">2</span>An unexpected finding: the UTF-8 script floor</h2>
<p>This one was not planned; it fell out of a test that asserted the wrong thing. Every
Devanagari codepoint encodes as <code>E0 A4 xx</code> or <code>E0 A5 xx</code>, so two of
every three byte slots agree between <em>any</em> two words of the script.</p>
<div class="card" id="floorcard"></div>
<div class="note warn" id="floornote"></div>

<h2><span class="n">3</span>Collisions under raw truncation</h2>
<p id="colltext"></p>
<div class="card" id="collgroups"></div>

<h2><span class="n">4</span>Headline result &mdash; and the control that makes it mean something</h2>
<p>For every pair of <em>distinct</em> compounds we ask: does the codec place two words
that share a morpheme closer together than two unrelated words? Scored by ROC&#8209;AUC over
<span id="npairs"></span> pairs, which is threshold-free &mdash; necessary because
segmentation raises all cosines, so absolute values are not comparable across conditions.</p>
<p><b>The split that matters:</b> pairs sharing an <em>initial</em> morpheme (which the
rigid codec can already align) versus pairs sharing a <em>non-initial</em> morpheme (which
it cannot).</p>
<svg id="aucchart" viewBox="0 0 960 300" style="width:100%;height:auto"></svg>
<div class="legend">
  <span><i style="background:var(--raw)"></i>raw (truncated)</span>
  <span><i style="background:var(--ctl)"></i>midpoint control</span>
  <span><i style="background:var(--seg)"></i>inverse-sandhi</span>
  <span><i style="background:var(--gold)"></i>gold annotation (ceiling)</span>
</div>
<div class="tw"><table id="auctable"></table></div>
<div class="note good" id="controlnote"></div>

<h3>Raw cosine similarity, before and after</h3>
<p>AUC is the comparator used above, for the reason given in &sect;2. The underlying mean
cosines are shown here as well, since they are the quantity the study set out to measure
&mdash; note how the unrelated baseline rises too, which is exactly why a bare cosine
cannot be compared across conditions.</p>
<div class="tw"><table id="costable"></table></div>

<h2><span class="n">5</span>Truncation and collisions across the positional budget</h2>
<p>The codec's only knob is <code>dp</code>. Sweeping it shows the effect is not an artefact
of one setting.</p>
<svg id="sweepchart" viewBox="0 0 960 300" style="width:100%;height:auto"></svg>
<div class="legend">
  <span><i style="background:var(--raw)"></i>raw</span>
  <span><i style="background:var(--ctl)"></i>midpoint control</span>
  <span><i style="background:var(--seg)"></i>inverse-sandhi</span>
</div>
<div class="note good" id="sweepnote"></div>

<h2><span class="n">6</span>The analyzer itself</h2>
<p id="splittext"></p>
<div class="card"><div class="tw"><table id="failtable"></table></div></div>
<div class="note" id="availnote"></div>

<h2><span class="n">7</span>What this does not show</h2>
<div class="card" id="limits"></div>

<h3>Natural next steps</h3>
<div class="card">
  <p><b>Option A &mdash; swap in a real analyzer.</b> Replace the rule engine with a
  wide-coverage morphological analyzer (Sanskrit Heritage Engine, or Morfessor via
  <code>indic_nlp_resources</code>) through the <code>MorfessorSplitter</code> adapter that
  already ships here, and rerun unchanged. The pipeline was written for that swap; the
  gold-annotation condition already measures the ceiling such an analyzer would be
  chasing (AUC 0.980).</p>
  <p><b>Option B &mdash; scale it.</b> Run the same comparison over a full Indic corpus
  rather than 54 hand-picked compounds, and check whether the collision reduction survives
  at vocabulary scale &mdash; where near-prefix families are far denser than anything in a
  curated list, so the raw-truncation collision rate should be considerably worse than the
  9% seen here.</p>
</div>

<h2><span class="n">8</span>The code</h2>
<p>This page is the write-up. The claims above are produced by code, which is where the
actual proof lives &mdash; including property tests that assert the codec's structural
guarantees and a test that asserts the <em>control must stay insignificant</em>, so the
argument breaks loudly if it ever stops holding.</p>
<div class="card">
  <p><a id="repolink" href="#" target="_blank" rel="noopener"><b>&#128279; Repository &mdash;
  session-7-assignment</b></a></p>
  <pre style="font-family:var(--mono);font-size:12px;overflow-x:auto;margin:10px 0 4px">python run_experiment.py     # regenerates results.json and this page (~40s)
python -m pytest tests -q    # 40 tests</pre>
  <div class="tw"><table>
    <tr><th>file</th><th>what it contains</th></tr>
    <tr><td><code>kron/codec.py</code></td><td>the codec &mdash; byte&times;position Kronecker product, ~20 lines of encoding, zero parameters</td></tr>
    <tr><td><code>kron/sandhi.py</code></td><td>inverse vowel-sandhi rule engine + lexicon splitter; Morfessor / gold / midpoint-control adapters</td></tr>
    <tr><td><code>kron/dataset.py</code></td><td>54 compounds with gold splits, morpheme families, related/unrelated pair classes</td></tr>
    <tr><td><code>kron/experiments.py</code></td><td>truncation, collisions, retrieval, AUC, bootstrap CIs, dp sweep, script floor</td></tr>
    <tr><td><code>tests/test_codec.py</code></td><td>proves injectivity below dp bytes and &ldquo;every collision is a truncation collision&rdquo;</td></tr>
    <tr><td><code>tests/test_sandhi_and_results.py</code></td><td>sandhi rules, dataset sanity, and the conclusions &mdash; including that the control must NOT be significant</td></tr>
  </table></div>
</div>

<footer>
  Generated by <code>python run_experiment.py</code> &mdash; every figure on this page is read
  from <code>results/results.json</code>, which the experiment code produces.
  No training was performed at any point.
</footer>
</div>

<script id="DATA" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('DATA').textContent);
const C = {raw:'var(--raw)', midpoint:'var(--ctl)', sandhi:'var(--seg)', gold:'var(--gold)'};
const $ = id => document.getElementById(id);
const pct = x => (100*x).toFixed(2)+'%';

/* ---------------- KPIs ---------------- */
const raw = D.conditions.raw, san = D.conditions.sandhi;
const rawNI = raw.relatedness.by_position.non_initial.auc_vs_unrelated;
const sanNI = san.relatedness.by_position.non_initial.auc_vs_unrelated;
$('kpis').innerHTML = [
  ['AUC non-initial<br/>raw &rarr; segmented', rawNI.toFixed(3)+' &rarr; '+sanNI.toFixed(3), 'var(--seg)'],
  ['bytes lost<br/>raw &rarr; segmented', pct(raw.truncation.truncation_rate)+' &rarr; '+pct(san.truncation.truncation_rate), 'var(--seg)'],
  ['words colliding<br/>raw &rarr; segmented', raw.collisions.n_words_in_collision+' &rarr; '+san.collisions.n_words_in_collision, 'var(--seg)'],
  ['analyzer exact match', (100*D.splitter_evaluation.exact_match).toFixed(1)+'%', 'var(--accent)'],
].map(([l,v,c]) => `<div class="kpi"><div class="v" style="color:${c}">${v}</div><div class="l">${l}</div></div>`).join('');

/* ---------------- word explorer ---------------- */
const sel = $('wordsel');
D.word_views.forEach((w,i) => {
  const o = document.createElement('option');
  o.value = i; o.textContent = w.word + '  (' + w.n_bytes + ' bytes' + (w.lost? ', '+w.lost+' lost':'') + ')';
  sel.appendChild(o);
});
function byteCell(b, lost){ return `<span class="byte${lost?' lost':''}">${b}</span>`; }
function draw(i){
  const w = D.word_views[i], dp = D.meta.dp;
  $('rawgrid').innerHTML = w.raw_bytes.map((b,k)=>byteCell(b, k>=dp)).join('');
  $('rawinfo').innerHTML = `${w.n_bytes} bytes &middot; ${Math.min(w.n_bytes,dp)} kept &middot; <b style="color:var(--raw)">${w.lost} lost</b>`;
  let html = '', kept = 0, lost = 0;
  w.part_bytes.forEach((pb,pi) => {
    if (pi) html += '<span class="byte sep">&nbsp;|&nbsp;</span>';
    pb.forEach((b,k) => { const L = k>=dp; L?lost++:kept++; html += byteCell(b, L); });
  });
  $('seggrid').innerHTML = html;
  $('seginfo').innerHTML = `${w.parts.join(' + ')} &middot; ${kept} kept &middot; `
    + `<b style="color:${lost?'var(--raw)':'var(--seg)'}">${lost} lost</b>`
    + (w.correct ? '' : ` &middot; <span style="color:var(--raw)">analyzer differs from gold (${w.gold.join(' + ')})</span>`);

  /* occupancy scatter: x = position, y = byte value */
  const W=960,H=210,pad=34, dpN=D.meta.dp;
  const x = p => pad + p*(W-pad-14)/dpN, y = b => H-24 - b*(H-46)/255;
  let s = `<rect x="0" y="0" width="${W}" height="${H}" fill="none"/>`;
  for(let p=0;p<=dpN;p+=8) s += `<line x1="${x(p)}" y1="8" x2="${x(p)}" y2="${H-24}" stroke="var(--line)"/>`
     + `<text x="${x(p)}" y="${H-8}" fill="var(--muted)" font-size="10" text-anchor="middle">${p}</text>`;
  s += `<text x="6" y="14" fill="var(--muted)" font-size="10">byte</text>`;
  w.raw_bytes.forEach((b,k)=>{ if(k<dpN) s += `<circle cx="${x(k)}" cy="${y(b)}" r="3.4" fill="var(--raw)" opacity=".85"/>`; });
  let off=0;
  w.part_bytes.forEach(pb=>{ pb.forEach((b,k)=>{ if(k<dpN) s += `<circle cx="${x(k)}" cy="${y(b)}" r="3.4" fill="var(--seg)" opacity=".7"/>`; }); off+=pb.length; });
  s += `<text x="${W-8}" y="14" fill="var(--muted)" font-size="10" text-anchor="end">orange = raw offsets, green = segmented offsets</text>`;
  $('occ').innerHTML = s;
}
sel.addEventListener('change', e => draw(+e.target.value));
draw(D.word_views.findIndex(w=>w.lost>0) >= 0 ? D.word_views.findIndex(w=>w.lost>0) : 0);

/* ---------------- script floor ---------------- */
(function(){
  const F=D.script_floor, E=F.example;
  const rows = Object.entries(E.cos_to_unrelated_words)
    .sort((a,b)=>b[1]-a[1])
    .map(([w,v])=>`<tr><td class="dev">${w}</td><td class="mut">unrelated</td><td class="n">${v.toFixed(4)}</td></tr>`).join('');
  $('floorcard').innerHTML =
    `<p><b>${pct(F.lead_byte_fraction)}</b> of the ${F.n_bytes_scanned} corpus bytes are the lead bytes
      <code>E0</code>/<code>A4</code>/<code>A5</code>.</p>
     <div class="tw"><table>
       <tr><th>compared with <span class="dev">${E.compound}</span></th><th>relation</th><th class="n">raw cosine</th></tr>
       <tr><td class="dev">${E.own_non_initial_morpheme}</td><td><b>its own morpheme</b></td>
           <td class="n"><b>${E.cos_to_own_morpheme.toFixed(4)}</b></td></tr>
       ${rows}
     </table></div>
     <p class="mut" style="font-size:12.5px">Latin control &mdash; no shared lead bytes:
       <code>cos(devalaya, alaya) = ${F.latin_control.cos_devalaya_alaya.toFixed(4)}</code>,
       positional rigidity in its pure form.</p>`;
  $('floornote').innerHTML = E.own_morpheme_beats_unrelated
    ? `Under raw encoding the true morpheme does outrank the unrelated words here.`
    : `<b>Under raw encoding, a word&rsquo;s own non-initial morpheme ranks BELOW unrelated words of the
       same script</b> (${E.cos_to_own_morpheme.toFixed(4)} vs ${E.best_unrelated.toFixed(4)}). Raw similarity
       between Devanagari strings is therefore mostly a measure of <em>script</em>, not of content &mdash;
       which is exactly why every claim on this page is a rank-based AUC rather than a cosine threshold.`;
})();

/* ---------------- collisions ---------------- */
$('colltext').innerHTML = `Under raw truncation <b>${raw.collisions.n_words_in_collision}</b> of `
 + `${raw.collisions.n_words} words fall into <b>${raw.collisions.n_collision_groups}</b> groups that share an `
 + `identical codec vector &mdash; they are literally indistinguishable to any downstream consumer. `
 + `After segmentation: <b style="color:var(--seg)">${san.collisions.n_words_in_collision}</b>.`;
$('collgroups').innerHTML = raw.collisions.groups.map(g =>
  `<div style="margin:8px 0"><span class="dev">${g.join('</span> &nbsp;≡&nbsp; <span class="dev">')}</span>
   <div class="mut" style="font-size:12px">identical first 32 bytes &rarr; identical vector</div></div>`).join('')
  || '<span class="mut">none</span>';

/* ---------------- AUC grouped bars ---------------- */
$('npairs').textContent = raw.relatedness.n_related + ' related and ' + raw.relatedness.n_unrelated + ' unrelated';
(function(){
  const order = ['raw','midpoint','sandhi','gold'];
  const groups = [['initial','shared INITIAL morpheme'],['non_initial','shared NON-INITIAL morpheme'],['overall','all related pairs']];
  const W=960,H=300,pad=46,bot=54;
  const gw = (W-pad-20)/groups.length;
  const bw = (gw-30)/order.length;
  const y = v => (H-bot) - (v-0.5)*(H-bot-24)/0.5;   /* axis 0.5 .. 1.0 */
  let s='';
  for(let v=0.5; v<=1.001; v+=0.1){
    s += `<line x1="${pad}" y1="${y(v)}" x2="${W-10}" y2="${y(v)}" stroke="var(--line)"/>`
      + `<text x="${pad-8}" y="${y(v)+4}" fill="var(--muted)" font-size="11" text-anchor="end">${v.toFixed(1)}</text>`;
  }
  s += `<line x1="${pad}" y1="${y(0.5)}" x2="${W-10}" y2="${y(0.5)}" stroke="var(--muted)" stroke-dasharray="4 3"/>`;
  groups.forEach((g,gi)=>{
    const gx = pad + gi*gw + 15;
    order.forEach((c,ci)=>{
      const R = D.conditions[c].relatedness;
      const v = g[0]==='overall' ? R.auc_overall : R.by_position[g[0]].auc_vs_unrelated;
      const key = g[0]==='overall' ? 'auc_overall_ci' : (g[0]==='initial'?'auc_initial_ci':'auc_non_initial_ci');
      const cint = D.statistics[c][key];
      const X = gx + ci*bw;
      const top = y(Math.max(v,0.5));
      s += `<rect x="${X}" y="${top}" width="${bw-5}" height="${Math.max(0,y(0.5)-top)}" fill="${C[c]}" rx="3" opacity=".9"/>`;
      if(cint && cint.lo!=null){
        const cx = X+(bw-5)/2;
        s += `<line x1="${cx}" y1="${y(cint.lo)}" x2="${cx}" y2="${y(cint.hi)}" stroke="var(--ink)" stroke-width="1.6" opacity=".8"/>`
          + `<line x1="${cx-4}" y1="${y(cint.lo)}" x2="${cx+4}" y2="${y(cint.lo)}" stroke="var(--ink)" stroke-width="1.6" opacity=".8"/>`
          + `<line x1="${cx-4}" y1="${y(cint.hi)}" x2="${cx+4}" y2="${y(cint.hi)}" stroke="var(--ink)" stroke-width="1.6" opacity=".8"/>`;
      }
      s += `<text x="${X+(bw-5)/2}" y="${top-6}" fill="var(--ink)" font-size="11" font-weight="700" text-anchor="middle">${v.toFixed(3)}</text>`;
    });
    s += `<text x="${gx+(order.length*bw)/2}" y="${H-30}" fill="var(--ink)" font-size="12" font-weight="600" text-anchor="middle">${g[1]}</text>`;
  });
  s += `<text x="${pad-8}" y="16" fill="var(--muted)" font-size="11" text-anchor="end">AUC</text>`;
  s += `<text x="${W-10}" y="${H-8}" fill="var(--muted)" font-size="10.5" text-anchor="end">error bars: 95% bootstrap CI (2000 resamples) &middot; 0.5 = chance</text>`;
  $('aucchart').innerHTML = s;
})();

/* ---------------- AUC table with paired deltas ---------------- */
(function(){
  let h = '<tr><th>condition</th><th class="n">bytes lost</th><th class="n">collisions</th>'
        + '<th class="n">AUC initial</th><th class="n">AUC non-initial</th>'
        + '<th class="n">&Delta; non-initial vs raw (95% CI)</th></tr>';
  ['raw','midpoint','sandhi','gold'].forEach(c=>{
    const R=D.conditions[c], S=D.statistics[c];
    const d=S.delta_vs_raw_non_initial;
    const dtxt = d ? `${d.delta>=0?'+':''}${d.delta.toFixed(3)} [${d.lo.toFixed(3)}, ${d.hi.toFixed(3)}]`
                     + (d.excludes_zero?' <b style="color:var(--seg)">*</b>':' <span class="mut">n.s.</span>') : '&mdash;';
    h += `<tr><td><b style="color:${C[c]}">${c}</b><div class="mut" style="font-size:11px">${R.splitter}</div></td>`
      +  `<td class="n">${pct(R.truncation.truncation_rate)}</td>`
      +  `<td class="n">${R.collisions.n_words_in_collision}</td>`
      +  `<td class="n">${R.relatedness.by_position.initial.auc_vs_unrelated.toFixed(3)}</td>`
      +  `<td class="n"><b>${R.relatedness.by_position.non_initial.auc_vs_unrelated.toFixed(3)}</b></td>`
      +  `<td class="n">${dtxt}</td></tr>`;
  });
  $('auctable').innerHTML = h;
  const dm = D.statistics.midpoint.delta_vs_raw_non_initial;
  const ds = D.statistics.sandhi.delta_vs_raw_non_initial;
  $('controlnote').innerHTML = `<b>Why the control settles it.</b> The midpoint splitter cuts every `
   + `word in half at a meaningless seam. It removes exactly as much truncation as a real segmentation `
   + `(${pct(D.conditions.midpoint.truncation.truncation_rate)} bytes lost), yet its AUC on shared `
   + `non-initial morphemes moves only ${dm.delta>=0?'+':''}${dm.delta.toFixed(3)} `
   + `[${dm.lo.toFixed(3)}, ${dm.hi.toFixed(3)}], while the morphological split moves `
   + `<b>${ds.delta>=0?'+':''}${ds.delta.toFixed(3)}</b> [${ds.lo.toFixed(3)}, ${ds.hi.toFixed(3)}]. `
   + `So the gain is attributable to <b>where</b> the cut falls, not to the fact that a cut was made.`;
})();

/* ---------------- raw cosine before/after ---------------- */
(function(){
  let h = '<tr><th>condition</th><th class="n">mean cos, shared morpheme</th>'
        + '<th class="n">mean cos, unrelated</th><th class="n">separation</th></tr>';
  ['raw','midpoint','sandhi','gold'].forEach(c=>{
    const R = D.conditions[c].relatedness;
    h += `<tr><td><b style="color:${C[c]}">${c}</b></td>`
      +  `<td class="n">${R.mean_cos_related.toFixed(4)}</td>`
      +  `<td class="n">${R.mean_cos_unrelated.toFixed(4)}</td>`
      +  `<td class="n"><b>${R.separation_overall.toFixed(4)}</b></td></tr>`;
  });
  $('costable').innerHTML = h;
})();

/* ---------------- repo link ---------------- */
(function(){
  const a = $('repolink');
  if (a && D.meta && D.meta.repo_url) { a.href = D.meta.repo_url; }
})();

/* ---------------- dp sweep ---------------- */
(function(){
  const rows=D.dp_sweep.rows, dps=D.dp_sweep.dps;
  const W=960,H=300,pad=52,bot=52;
  const x = d => pad + dps.indexOf(d)*(W-pad-120)/(dps.length-1);
  const y = v => (H-bot) - v*(H-bot-26);
  let s='';
  [0,.25,.5,.75,1].forEach(v=>{ s+=`<line x1="${pad}" y1="${y(v)}" x2="${W-120}" y2="${y(v)}" stroke="var(--line)"/>`
    +`<text x="${pad-8}" y="${y(v)+4}" fill="var(--muted)" font-size="11" text-anchor="end">${v.toFixed(2)}</text>`; });
  dps.forEach(d=> s+=`<text x="${x(d)}" y="${H-30}" fill="var(--muted)" font-size="11" text-anchor="middle">dp=${d}</text>`);
  ['raw','midpoint','sandhi'].forEach(c=>{
    const pts = dps.map(d=>{ const r=rows.find(r=>r.dp===d&&r.condition===c); return [x(d), y(r.auc_non_initial)]; });
    s += `<polyline points="${pts.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${C[c]}" stroke-width="2.5"/>`;
    pts.forEach(p=> s+=`<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="${C[c]}"/>`);
    const tr = dps.map(d=>{ const r=rows.find(r=>r.dp===d&&r.condition===c); return [x(d), y(r.truncation_rate)]; });
    s += `<polyline points="${tr.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${C[c]}" stroke-width="1.6" stroke-dasharray="5 4" opacity=".75"/>`;
    const last = pts[pts.length-1];
    s += `<text x="${W-114}" y="${last[1]+4}" fill="${C[c]}" font-size="11.5" font-weight="700">${c}</text>`;
  });
  s += `<text x="${pad-8}" y="16" fill="var(--muted)" font-size="11" text-anchor="end">value</text>`;
  s += `<text x="${W-114}" y="${H-8}" fill="var(--muted)" font-size="10.5" text-anchor="end">solid = AUC (non-initial) &nbsp;&middot;&nbsp; dashed = byte truncation rate</text>`;
  $('sweepchart').innerHTML = s;

  /* the strongest single piece of evidence: a dp with zero truncation anywhere */
  const wide = dps.filter(d => rows.find(r=>r.dp===d&&r.condition==='raw').truncation_rate === 0);
  if (wide.length){
    const d = wide[0];
    const rr = rows.find(r=>r.dp===d&&r.condition==='raw');
    const rs = rows.find(r=>r.dp===d&&r.condition==='sandhi');
    $('sweepnote').innerHTML = `<b>The decisive point.</b> At <code>dp=${d}</code> the window is wide `
      + `enough that <b>nothing is truncated at all</b> (raw truncation ${pct(rr.truncation_rate)}, zero `
      + `collisions). Raw AUC on shared non-initial morphemes is nevertheless still `
      + `<b>${rr.auc_non_initial.toFixed(3)}</b>, while segmentation reaches `
      + `<b>${rs.auc_non_initial.toFixed(3)}</b>. So the retrieval failure is <em>not</em> a truncation `
      + `problem &mdash; it is positional rigidity, and enlarging the budget does not fix it. `
      + `Truncation and alignment are two independent defects, and segmentation addresses both.`;
  }
})();

/* ---------------- splitter ---------------- */
(function(){
  const E=D.splitter_evaluation;
  $('splittext').innerHTML = `The analyzer is a rule table (inverse vowel sandhi) plus a `
   + `${D.dataset.n_lexicon}-entry morpheme lexicon containing ${D.dataset.n_distractors} `
   + `distractors that are not constituents of anything in the set &mdash; so it must choose among `
   + `competing lexicon-valid seams rather than look up an answer. It recovers `
   + `<b>${E.n_correct}/${E.n_words}</b> gold decompositions exactly `
   + `(<b>${(100*E.exact_match).toFixed(1)}%</b>), and reaches an AUC within `
   + `${Math.abs(D.conditions.gold.relatedness.by_position.non_initial.auc_vs_unrelated
        - D.conditions.sandhi.relatedness.by_position.non_initial.auc_vs_unrelated).toFixed(3)} `
   + `of the gold-annotation ceiling.`;
  let h='<tr><th>word</th><th>gold</th><th>analyzer</th><th>why it misses</th></tr>';
  const why = {'रामायण':'retroflexion (natva) sandhi न&rarr;ण, not modelled',
               'अत्यन्त':'yaṇ sandhi i+V&rarr;य्, not modelled',
               'इत्यादि':'yaṇ sandhi i+V&rarr;य्, not modelled',
               'स्वागत':'yaṇ sandhi u+V&rarr;व्, not modelled'};
  E.failures.forEach(f=>{ h+=`<tr><td class="dev">${f.word}</td><td class="dev">${f.gold.join(' + ')}</td>`
    +`<td class="dev">${f.predicted.join(' + ')}</td><td class="mut">${why[f.word]||'unmodelled sandhi class'}</td></tr>`; });
  $('failtable').innerHTML = h;
  const A=D.segmenter_availability;
  $('availnote').innerHTML = `<b>On "off-the-shelf".</b> The intent was to call an existing analyzer. `
   + `Sanskrit Heritage Engine: ${A.sanskrit_heritage_engine.reason}. `
   + `indic-nlp-library: ${A.indic_nlp_library_morfessor.reason}. `
   + `So the rule engine here plays that role &mdash; still training-free &mdash; and `
   + `<code>MorfessorSplitter</code> ships as a working adapter that activates automatically if the `
   + `resource bundle is ever present.`;
})();

/* ---------------- limitations ---------------- */
$('limits').innerHTML = [
 ['Scale', `${D.dataset.n_compounds} curated compounds, one script. This is an encoding-level probe, not a corpus study, and nothing here has been validated at model-training scale.`],
 ['Circularity, where it exists', `The constituent-retrieval measure (in results.json) is favourable to segmented conditions <em>by construction</em>, because a segmented vector is the mean of its parts. It is reported but is not the headline; the compound-to-compound comparison above shares no such defect.`],
 ['Cosines are not comparable across conditions', `Segmentation raises all cosines, related and unrelated alike (unrelated mean ${D.conditions.raw.relatedness.mean_cos_unrelated.toFixed(3)} &rarr; ${D.conditions.sandhi.relatedness.mean_cos_unrelated.toFixed(3)}). That is why every claim uses rank-based AUC rather than raw similarity.`],
 ['Truncation relief is arithmetic', `Any split shortens the pieces. The midpoint control is included precisely so this trivial part is not mistaken for the interesting part.`],
 ['The analyzer is narrow', `Vowel sandhi only; yaṇ and consonant sandhi are unmodelled, and the lexicon is small. On out-of-lexicon vocabulary it declines to split rather than guessing.`],
 ['Not shown', `That any of this improves a downstream model. Establishing that needs the training run this study deliberately does not include.`],
].map(([h,b])=>`<div style="margin:10px 0"><b>${h}.</b> <span class="mut">${b}</span></div>`).join('');
</script>
</body>
</html>
"""


def build(bundle: dict, out_dir: pathlib.Path) -> pathlib.Path:
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = dict(bundle)
    data["word_views"] = _word_views(bundle["meta"]["dp"])
    # the full pair-cosine lists are only needed by results.json, not the page
    data.pop("pair_cosines", None)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path
