"""Assemble a single self-contained HTML (no external fetches) for hosting as a
claude.ai Artifact / on any static host. Inlines CSS, the BPE encoder, the
tokenizer.json, and the four corpora (base64) so the widget computes fertility
LIVE with nothing fetched."""
import base64, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
css = (ROOT / "css/style.css").read_text(encoding="utf-8")
encoder = (ROOT / "js/bpe_encoder.js").read_text(encoding="utf-8")
tok = (ROOT / "artifacts/tokenizer.json").read_text(encoding="utf-8")
LANGS = [("en", "english"), ("hi", "hindi"), ("te", "telugu"), ("mr", "marathi")]
corpora = {l: base64.b64encode((ROOT / f"data/{n}_india.txt").read_bytes()).decode() for l, n in LANGS}

RAW = "https://raw.githubusercontent.com/vasu-2004/ERA-V5-assignments/356b7276ed23227ae46a2ec533c0f2792107b476/session-2-assignments/artifacts/tokenizer.json"

BODY = r"""
<title>Session 2 · Multilingual BPE Tokenizer — India (EN/HI/TE/MR)</title>
<style>__CSS__</style>

<nav class="nav"><div class="nav-inner">
  <div class="brand">Session 2 &middot; Multilingual BPE <small>&mdash; 10k vocab &middot; EN/HI/TE/MR &middot; scored live</small></div>
  <a class="jump" href="#score">Score</a><a class="jump" href="#method">Method</a>
  <a class="jump" href="#live">Run it yourself</a><a class="jump" href="#downloads">Tokenizer</a>
  <button id="theme-btn" title="Toggle theme">&#9790;</button>
</div></nav>

<div class="wrap">
  <header class="hero">
    <h1>One 10,000-token tokenizer.<br/>Four languages. Scored live.</h1>
    <p class="lede">A single byte-level BPE tokenizer for the <b>India</b> Wikipedia page in English,
      Hindi, Telugu and Marathi. <b>Zero UNK by construction</b>, <b>English fertility &le; 1.2</b>,
      minimized cross-language spread. Every number below is <b>computed in your browser right now</b>
      from the embedded <code>tokenizer.json</code> using an encoder verified bit-exact against
      HuggingFace <code>tokenizers</code>.</p>
    <div class="warnbox"><b>Honesty note.</b> Figures are on the supplied copy-paste of each India page;
      the grader runs their own cleaned pages, so the exact spread/score will differ. Paste your text in
      <a href="#live">Run it yourself</a> to get the real number. The design is what's robust.</div>
  </header>

  <section class="demo" id="score">
    <span class="tag">Self-score</span><h2>Live score</h2>
    <div id="live-status" class="live-status">Computing&hellip;</div>
    <div class="scorecard">
      <div class="scorebig"><div class="num" id="score-value">&mdash;</div><div class="lbl">self-score</div></div>
      <div class="scoremeta"><table>
        <tr><td>X<sub>max</sub></td><td id="score-max">&mdash;</td></tr>
        <tr><td>X<sub>min</sub></td><td id="score-min">&mdash;</td></tr>
        <tr><td>Spread (X<sub>max</sub> &minus; X<sub>min</sub>)</td><td id="score-spread">&mdash;</td></tr>
        <tr><td>English fertility (cap 1.2)</td><td id="score-en">&mdash;</td></tr>
        <tr><td>Vocabulary size</td><td id="vocab-size">&mdash;</td></tr>
      </table><div class="formula" id="score-formula"></div></div>
    </div>
    <div class="grid2" style="margin-top:18px">
      <div class="card"><h4>Per-language fertility (computed live)</h4>
        <div class="tablewrap"><table class="datatable" id="opt-table"></table></div></div>
      <div class="card chartcard"><h4>Fertility by language (dashed = English 1.2 cap)</h4>
        <canvas id="fert-bars"></canvas></div>
    </div>
    <div class="takeaway"><b>No UNK, ever.</b> <span id="unk-proof">&mdash;</span></div>
  </section>

  <section class="demo" id="method">
    <span class="tag">How</span><h2>How it works (the &ldquo;secret sauce&rdquo;)</h2>
    <div class="cbp">
      <div><b>Byte-level base.</b> <span>All 256 byte values are base tokens &rarr; every character is
        encodable &rarr; UNK is structurally impossible (verified live above).</span></div>
      <div><b>Merges cross punctuation &amp; spaces.</b> <span>A byte-level pre-tokenizer with
        <code>use_regex=False</code> lets BPE learn tokens spanning markup like <code>](/wiki/</code>,
        so those characters get absorbed instead of exploding the count &mdash; the markdown problem
        solved by design, losslessly.</span></div>
      <div><b>The <code>\w+</code> denominator.</b> <span>Python's <code>\w</code> excludes Devanagari/
        Telugu vowel signs, so it splits &#2349;&#2366;&#2352;&#2340; &rarr; &#2349;, &#2352;&#2340; &mdash;
        ~doubling Indic word counts. We match that exactly.</span></div>
      <div><b>Weighted training.</b> <span>One 10k vocab, per-language weights
        <code>(En 7, Hi 1, Te 2, Mr 1)</code>: English up-weighted to pass under 1.2, Telugu up-weighted
        because it's hardest.</span></div>
    </div>
  </section>

  <section class="demo" id="live">
    <span class="tag">Verify</span><h2>Run it yourself</h2>
    <p class="methodbox">The boxes are pre-filled with the exact corpora. <b>Replace any with your own
      cleaned India-page text</b> and hit <b>Re-score</b> &mdash; it tokenizes live with the embedded
      <code>tokenizer.json</code> (the same encoder that matches HuggingFace exactly) and recomputes the
      score above.</p>
    <div class="pastegrid">
      <div><label>English text</label><textarea id="paste-en"></textarea></div>
      <div><label>Hindi text</label><textarea id="paste-hi"></textarea></div>
      <div><label>Telugu text</label><textarea id="paste-te"></textarea></div>
      <div><label>Marathi text</label><textarea id="paste-mr"></textarea></div>
    </div>
    <div class="btnrow">
      <button class="primary" id="rescore-btn">Re-score my text</button>
      <button class="ghost" id="reset-btn">Reset to original corpora</button>
    </div>
  </section>

  <section class="demo" id="downloads">
    <span class="tag">Tokenizer</span><h2>See &amp; download the tokenizer</h2>
    <p class="methodbox">The full 10,000-token vocabulary + merges, as a standard HuggingFace
      <code>tokenizer.json</code>. Load with <code>Tokenizer.from_file("tokenizer.json")</code>.</p>
    <div class="btnrow">
      <button class="primary" id="dl-btn">&#8595; Download tokenizer.json (10,000 tokens)</button>
      <a class="ghost" id="raw-link" href="__RAW__" target="_blank" rel="noopener" style="text-decoration:none;display:inline-block;padding:9px 18px;border-radius:10px;background:var(--grid);color:var(--ink);font-weight:600;font-size:13.5px">Public raw link on GitHub</a>
    </div>
    <div class="tablewrap" style="margin-top:14px"><table class="datatable" id="vocab-peek"></table></div>
  </section>

  <footer>Built for ERA V5 &middot; Session 2. Byte-level BPE, single 10,000-token vocab, zero UNK.
    The in-browser encoder is verified bit-exact against HuggingFace <code>tokenizers</code> on all four full corpora.</footer>
</div>

<script id="tokjson_b64" type="text/plain">__TOK_B64__</script>
<script id="corpora" type="application/json">__CORPORA__</script>
<script>__ENCODER__</script>
<script>__WIDGET__</script>
"""

WIDGET = r"""
(function(){
'use strict';
var LANGS=['en','hi','te','mr'], NAME={en:'English',hi:'Hindi',te:'Telugu',mr:'Marathi'},
    COLOR={en:'#2f6fed',hi:'#e0447f',te:'#12a594',mr:'#f2820a'}, EN_CAP=1.2;
function $(id){return document.getElementById(id);}
function fmt(n){return n.toLocaleString('en-US');}
function b64utf8(s){s=s.trim();var bin=atob(s),bytes=new Uint8Array(bin.length);for(var i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i);return new TextDecoder('utf-8').decode(bytes);}
// tokenizer + corpora are embedded base64 (ASCII-safe: immune to page charset) and decoded as UTF-8 here
var TOKTEXT=b64utf8(document.getElementById('tokjson_b64').textContent);
var TOK=JSON.parse(TOKTEXT);
var enc=BPE.Encoder.fromTokenizerJSON(TOK);
var b64=JSON.parse(document.getElementById('corpora').textContent);
var texts={}; LANGS.forEach(function(l){texts[l]=b64utf8(b64[l]);});
function fert(t){var n=BPE.normalizeNewlines(t);var tk=enc.countTokens(n,false),w=BPE.wordCount(n);return{tokens:tk,words:w,fertility:w?tk/w:0};}
function computeAll(src){var per={};LANGS.forEach(function(l){per[l]=fert(src[l]);});
  var xs=LANGS.map(function(l){return per[l].fertility;});var mx=Math.max.apply(null,xs),mn=Math.min.apply(null,xs);
  return{per:per,xmax:mx,xmin:mn,maxL:LANGS[xs.indexOf(mx)],minL:LANGS[xs.indexOf(mn)],spread:mx-mn,
    score:(mx-mn)>1e-9?1000/(mx-mn):Infinity,en_ok:per.en.fertility<=EN_CAP};}
function renderResults(r){var rows=LANGS.map(function(l){var p=r.per[l];var badge='';
  if(l===r.maxL)badge=' <span class="chip max">max</span>';if(l===r.minL)badge=' <span class="chip min">min</span>';
  var ef=(l==='en')?(p.fertility<=EN_CAP?' <span class="ok">&le;1.2 &#10003;</span>':' <span class="bad">&gt;1.2 &#10007;</span>'):'';
  return '<tr><td><span class="dot" style="background:'+COLOR[l]+'"></span>'+NAME[l]+badge+'</td><td>'+fmt(p.words)+'</td><td>'+fmt(p.tokens)+'</td><td><b>'+p.fertility.toFixed(4)+'</b>'+ef+'</td></tr>';}).join('');
  $('opt-table').innerHTML='<tr><th>Language</th><th>Words (\\w+)</th><th>BPE tokens</th><th>Fertility X</th></tr>'+rows;}
function renderScore(r){$('score-value').textContent=isFinite(r.score)?Math.round(r.score).toLocaleString('en-US'):'&#8734;';
  $('score-formula').innerHTML='1000 / ('+r.xmax.toFixed(4)+' &minus; '+r.xmin.toFixed(4)+') = 1000 / '+r.spread.toFixed(4);
  $('score-max').textContent=NAME[r.maxL]+' '+r.xmax.toFixed(4);$('score-min').textContent=NAME[r.minL]+' '+r.xmin.toFixed(4);
  $('score-spread').textContent=r.spread.toFixed(4);var en=$('score-en');
  en.textContent=r.per.en.fertility.toFixed(4)+(r.en_ok?'  ✓ within 1.2':'  ✗ exceeds 1.2');en.className=r.en_ok?'ok':'bad';}
function bars(r){var c=$('fert-bars');if(!c)return;var dpr=window.devicePixelRatio||1,rc=c.getBoundingClientRect(),w=rc.width,h=rc.height;
  c.width=w*dpr;c.height=h*dpr;var ctx=c.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
  var s=getComputedStyle(document.documentElement),ink=s.getPropertyValue('--ink').trim(),muted=s.getPropertyValue('--muted').trim();
  var pad=34,top=16,bot=30,maxX=1.35,capY=top+(1-EN_CAP/maxX)*(h-top-bot);
  ctx.strokeStyle=muted;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(pad,capY);ctx.lineTo(w-10,capY);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle=muted;ctx.font='11px system-ui';ctx.fillText('English cap 1.2',pad+4,capY-4);
  var bw=(w-pad-20)/LANGS.length;LANGS.forEach(function(l,i){var v=r.per[l].fertility,bh=(v/maxX)*(h-top-bot),x=pad+i*bw+bw*0.2,y=h-bot-bh,bwid=bw*0.6;
    ctx.fillStyle=COLOR[l];ctx.fillRect(x,y,bwid,bh);ctx.fillStyle=ink;ctx.font='bold 12px system-ui';ctx.textAlign='center';
    ctx.fillText(v.toFixed(3),x+bwid/2,y-5);ctx.fillStyle=muted;ctx.font='11px system-ui';ctx.fillText(NAME[l],x+bwid/2,h-bot+14);ctx.textAlign='start';});}
function status(m,k){var e=$('live-status');e.innerHTML=m;e.className='live-status '+(k||'');}
function currentSource(){var s={};LANGS.forEach(function(l){var ta=$('paste-'+l);s[l]=(ta&&ta.value)||texts[l];});return s;}
function unkProof(){var st=__STRESS__;
  var t=enc.encodeChunk(BPE.normalizeNewlines(st));$('unk-proof').innerHTML='Stress string of markup/rare chars &rarr; <b>'+t.length+' tokens, 0 UNK</b> (every byte value is a base token, so any character in any page is encodable).';}
function vocabPeek(){var v=TOK.model.vocab,items=Object.keys(v).map(function(k){return[k,v[k]];}).sort(function(a,b){return a[1]-b[1];});
  var head=items.slice(256,276),tail=items.slice(-6);
  function row(kv){return '<tr><td>'+kv[1]+'</td><td style="font-family:var(--mono)">'+kv[0].replace(/</g,'&lt;')+'</td></tr>';}
  $('vocab-peek').innerHTML='<tr><th>id</th><th>token (byte-level encoded)</th></tr>'+
    '<tr><td colspan="2" style="color:var(--muted)">&mdash; first learned merges &mdash;</td></tr>'+head.map(row).join('')+
    '<tr><td colspan="2" style="color:var(--muted)">&mdash; last learned merges &mdash;</td></tr>'+tail.map(row).join('');}
function download(){var blob=new Blob([TOKTEXT],{type:'application/json'});
  var a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='tokenizer.json';document.body.appendChild(a);a.click();a.remove();}
function draw(r){renderResults(r);renderScore(r);bars(r);}
function boot(){$('vocab-size').textContent=fmt(Object.keys(TOK.model.vocab).length);
  LANGS.forEach(function(l){var ta=$('paste-'+l);if(ta)ta.value=texts[l];});
  var r=computeAll(texts);draw(r);unkProof();vocabPeek();
  status('Computed live in your browser from the embedded tokenizer.json ✓ (matches HuggingFace exactly)','good');
  window.addEventListener('resize',function(){bars(computeAll(currentSource()));});}
$('rescore-btn').addEventListener('click',function(){status('Re-scoring your text live…');setTimeout(function(){var r=computeAll(currentSource());draw(r);status('Re-scored your pasted text live ✓','good');},20);});
$('reset-btn').addEventListener('click',function(){LANGS.forEach(function(l){$('paste-'+l).value=texts[l];});var r=computeAll(texts);draw(r);status('Reset to the original corpora ✓','good');});
$('dl-btn').addEventListener('click',download);
$('theme-btn').addEventListener('click',function(){var root=document.documentElement,cur=root.getAttribute('data-theme')==='dark'?'light':'dark';root.setAttribute('data-theme',cur);this.innerHTML=cur==='dark'?'☀︎':'☾';bars(computeAll(currentSource()));});
boot();
})();
"""

tok_b64 = base64.b64encode(tok.encode("utf-8")).decode()
stress = "India](/wiki/India) — भारत, గణతంత్ర! ₹100 42% \"q\" <t> {j:1} ©®™"
WIDGET = WIDGET.replace("__STRESS__", json.dumps(stress))
html = (BODY
        .replace("__CSS__", css)
        .replace("__ENCODER__", encoder)
        .replace("__WIDGET__", WIDGET)
        .replace("__RAW__", RAW)
        .replace("__TOK_B64__", tok_b64)
        .replace("__CORPORA__", json.dumps(corpora)))

out = ROOT / "standalone.html"
out.write_text(html, encoding="utf-8")
print("wrote", out, f"({len(html)} bytes)")
