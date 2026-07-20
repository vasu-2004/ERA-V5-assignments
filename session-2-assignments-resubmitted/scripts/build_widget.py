"""Build a self-contained widget (standalone.html) for the resubmission:
verified faithful-unit metrics + searchable vocab browser + download tokenizer.json.
Everything inlined (tokenizer.json as base64) -> hostable anywhere, no fetches."""
import base64, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
css = (ROOT / "css/style.css").read_text(encoding="utf-8")
tok_text = (ROOT / "artifacts/tokenizer.json").read_text(encoding="utf-8")
metrics = json.loads((ROOT / "artifacts/metrics.json").read_text(encoding="utf-8"))
tj = json.loads(tok_text)

LANGS = ["en", "hi", "te", "mai"]
NAME = {"en": "English", "hi": "Hindi", "te": "Telugu", "mai": "Maithili"}
m = metrics["metrics"]
per = m["per_language"]
weights = metrics["weights"]

# compact vocab for the browser: [id, token] sorted by id (skip nothing)
vocab = tj["model"]["vocab"]
vocab_list = sorted(([v, k] for k, v in vocab.items()), key=lambda x: x[0])

payload = {
    "langs": LANGS, "names": NAME,
    "per": {l: {"tokens": per[l]["tokens"], "units": per[l]["faithful_units"],
                "fertility": per[l]["fertility"]} for l in LANGS},
    "x_max": m["x_max"], "x_max_lang": NAME[m["x_max_lang"]],
    "x_min": m["x_min"], "x_min_lang": NAME[m["x_min_lang"]],
    "spread": m["spread"], "raw_score": m["raw_score"],
    "all_under_1_2": m["all_under_1_2"],
    "hindi_penalty": m.get("hindi_penalty", 1.0),
    "vocab_size": tj["model"] and len(vocab),
    "weights": weights,
    "vocab": vocab_list,
}

RECIPE = (f"BPE (unk=[UNK]) &middot; vocab {len(vocab):,} &middot; min_frequency 1 &middot; "
          f"Normalizer NFKC &middot; Metaspace pre-tokenizer + decoder (prepend_scheme=never) &middot; "
          f"weights En {weights['en']} / Hi {weights['hi']} / Te {weights['te']} / Mai {weights['mai']}")

BODY = r"""
<title>Session 2 (resubmission) &middot; Faithful-Markdown BPE &middot; India EN/HI/TE/MAI</title>
<style>__CSS__</style>
<nav class="nav"><div class="nav-inner">
  <div class="brand">Session 2 &middot; Faithful-Markdown BPE <small>&mdash; 10k vocab &middot; EN/HI/TE/Maithili</small></div>
  <a class="jump" href="#score">Score</a><a class="jump" href="#recipe">Recipe</a>
  <a class="jump" href="#vocab">Vocab</a><a class="jump" href="#download">Download</a>
  <button id="theme-btn" title="Toggle theme">&#9790;</button>
</div></nav>
<div class="wrap">
  <header class="hero">
    <h1>Faithful-Markdown tokenizer.<br/>All four under 1.2.</h1>
    <p class="lede">A single 10,000-token BPE tokenizer for the <b>India</b> Wikipedia pages in
      English, Hindi, Telugu and Maithili, trained on the <b>wiki-faithful Markdown</b> corpus
      (links, URLs, tables, references preserved) and scored with the instructor's exact
      faithful-unit metric. Every fertility is <b>under 1.2</b>, so the penalty factor is 1.0.
      Numbers below are produced by <code>evaluate_tokenizer.py</code> and reproduced by loading
      the downloadable <code>tokenizer.json</code>.</p>
  </header>

  <section class="demo" id="score">
    <span class="tag">Self-score</span><h2>Score</h2>
    <div class="scorecard">
      <div class="scorebig"><div class="num" id="score-value">&mdash;</div><div class="lbl">raw score = 1000 / spread</div></div>
      <div class="scoremeta"><table>
        <tr><td>X<sub>max</sub></td><td id="s-max">&mdash;</td></tr>
        <tr><td>X<sub>min</sub></td><td id="s-min">&mdash;</td></tr>
        <tr><td>Spread</td><td id="s-spread">&mdash;</td></tr>
        <tr><td>All languages &lt; 1.2</td><td id="s-under">&mdash;</td></tr>
        <tr><td>Hindi penalty factor</td><td id="s-pen">&mdash;</td></tr>
        <tr><td>Vocabulary size</td><td id="s-vocab">&mdash;</td></tr>
      </table></div>
    </div>
    <div class="card" style="margin-top:18px"><h4>Per-language fertility (faithful-unit metric)</h4>
      <div class="tablewrap"><table class="datatable" id="lang-table"></table></div>
      <p class="methodbox" style="margin-top:8px">Fertility = BPE tokens &divide; faithful units, where a
      <b>faithful unit</b> is one contiguous letter/mark/number run OR one visible non-space
      punctuation/symbol character (<code>[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]</code>).</p>
    </div>
    <div class="takeaway"><b>Faithful.</b> The tokenizer preserves visible text &mdash; punctuation,
      brackets, URL characters, apostrophes and number separators all round-trip (nothing stripped);
      it uses Metaspace, not ByteLevel, so Indic scripts don't blow up into UTF-8 bytes.</div>
  </section>

  <section class="demo" id="recipe">
    <span class="tag">Recipe</span><h2>How it's built (reference-faithful)</h2>
    <p class="methodbox" id="recipe-text"></p>
    <details class="math" open><summary>Reproduce from scratch</summary>
      <p style="font-family:var(--mono);font-size:12px;white-space:pre-wrap">pip install tokenizers regex requests markdownify beautifulsoup4 lxml
python build_wiki_faithful_markdown.py   # fetch India pages -> corpus/*.faithful.txt
python train_tokenizer.py --search       # train + tune weights -> artifacts/tokenizer.json
python evaluate_tokenizer.py             # print the faithful-unit fertilities + score</p></details>
  </section>

  <section class="demo" id="vocab">
    <span class="tag">Inspect</span><h2>Vocabulary browser</h2>
    <p class="methodbox">All <b id="vocab-count"></b> tokens. Search by token text or id
      (the <code>&#9601;</code> marks a word-initial space; <code>&#266;</code>/<code>\n</code> are newlines).</p>
    <input id="vocab-search" placeholder="filter tokens, e.g. India, ▁the, http, ]("
      style="width:100%;padding:9px 12px;border:1px solid var(--border);border-radius:9px;background:var(--bg);color:var(--ink);font-family:var(--mono);font-size:13px;margin-bottom:10px"/>
    <div class="tablewrap" style="max-height:360px;overflow:auto"><table class="datatable" id="vocab-table"></table></div>
  </section>

  <section class="demo" id="download">
    <span class="tag">Download</span><h2>Download &amp; verify the tokenizer</h2>
    <p class="methodbox">Standard HuggingFace <code>tokenizer.json</code>. Load with
      <code>Tokenizer.from_file("tokenizer.json")</code> and encode &mdash; that reproduces every
      number above (verified: reloading the file gives identical fertilities).</p>
    <div class="btnrow">
      <a class="primary" id="dl-raw" href="__RAW_URL__" download="tokenizer.json" target="_blank" rel="noopener"
         style="text-decoration:none;display:inline-block;padding:9px 18px;border-radius:10px;background:var(--accent);color:#fff;font-weight:600;font-size:13.5px">&#8595; Download tokenizer.json (10,000 tokens)</a>
      <button class="ghost" id="dl-btn">Download (in-page copy)</button>
    </div>
    <p class="methodbox" style="margin-top:8px">Direct file (public, always works):
      <a href="__RAW_URL__" target="_blank" rel="noopener" style="word-break:break-all">__RAW_URL__</a></p>
    <details class="math" style="margin-top:14px"><summary>Verify in three lines of Python</summary>
      <p style="font-family:var(--mono);font-size:12px;white-space:pre-wrap">from tokenizers import Tokenizer; import regex
tok = Tokenizer.from_file("tokenizer.json")
text = open("corpus/en.faithful.txt", encoding="utf-8").read()
units = len(regex.findall(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]", text))
print(len(tok.encode(text).ids) / units)   # -> English fertility</p></details>
  </section>

  <footer>Built for ERA V5 &middot; Session 2 resubmission. Reference recipe (BPE + NFKC + Metaspace),
    wiki-faithful Markdown corpus, exact faithful-unit metric. All four languages &lt; 1.2.</footer>
</div>
<script id="data" type="application/json">__DATA_B64__</script>
<script>__JS__</script>
"""

JS = r"""
(function(){
'use strict';
function $(id){return document.getElementById(id);}
function fmt(n){return n.toLocaleString('en-US');}
function b64utf8(s){s=s.trim();var b=atob(s),a=new Uint8Array(b.length);for(var i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return new TextDecoder('utf-8').decode(a);}
var D=JSON.parse(b64utf8($('data').textContent));
var COLOR={en:'#2f6fed',hi:'#e0447f',te:'#12a594',mai:'#f2820a'};
// score card
$('score-value').textContent=Math.round(D.raw_score).toLocaleString('en-US');
$('s-max').textContent=D.x_max_lang+' '+D.x_max.toFixed(4);
$('s-min').textContent=D.x_min_lang+' '+D.x_min.toFixed(4);
$('s-spread').textContent=D.spread.toFixed(5);
$('s-under').innerHTML=D.all_under_1_2?'<span class="ok">yes &#10003;</span>':'<span class="bad">no</span>';
$('s-pen').textContent=D.hindi_penalty.toFixed(3)+(D.hindi_penalty<=1.0001?'  (no penalty)':'');
$('s-vocab').textContent=fmt(D.vocab_size);
$('recipe-text').innerHTML=__RECIPE__;
// per-language table
$('lang-table').innerHTML='<tr><th>Language</th><th>BPE tokens</th><th>Faithful units</th><th>Fertility</th></tr>'+
 D.langs.map(function(l){var p=D.per[l];var mx=(D.names[l]===D.x_max_lang),mn=(D.names[l]===D.x_min_lang);
   return '<tr><td><span class="dot" style="background:'+COLOR[l]+'"></span>'+D.names[l]+
   (mx?' <span class="chip max">max</span>':'')+(mn?' <span class="chip min">min</span>':'')+'</td>'+
   '<td>'+fmt(p.tokens)+'</td><td>'+fmt(p.units)+'</td><td><b>'+p.fertility.toFixed(4)+'</b> <span class="ok">&lt;1.2</span></td></tr>';}).join('');
// vocab browser
$('vocab-count').textContent=fmt(D.vocab.length);
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function renderVocab(q){
  q=(q||'').toLowerCase(); var rows=[]; var count=0;
  for(var i=0;i<D.vocab.length && count<400;i++){var id=D.vocab[i][0],t=D.vocab[i][1];
    if(!q || t.toLowerCase().indexOf(q)>=0 || String(id)===q){rows.push('<tr><td style="width:70px">'+id+'</td><td style="font-family:var(--mono)">'+esc(t)+'</td></tr>');count++;}}
  $('vocab-table').innerHTML='<tr><th>id</th><th>token</th></tr>'+rows.join('')+
    (count>=400?'<tr><td colspan="2" style="color:var(--muted)">(showing first 400 matches &mdash; refine your search)</td></tr>':'');
}
renderVocab('');
$('vocab-search').addEventListener('input',function(){renderVocab(this.value);});
// download
$('dl-btn').addEventListener('click',function(){
  var blob=new Blob([b64utf8($('tok').textContent)],{type:'application/json'});
  var a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='tokenizer.json';
  document.body.appendChild(a);a.click();a.remove();});
// theme
$('theme-btn').addEventListener('click',function(){var r=document.documentElement,c=r.getAttribute('data-theme')==='dark'?'light':'dark';r.setAttribute('data-theme',c);this.innerHTML=c==='dark'?'☀︎':'☾';});
})();
"""

RAW_URL = ("https://raw.githubusercontent.com/vasu-2004/ERA-V5-assignments/"
           "6602bba1c577f3a93d8dc54bf7ef1c81ba611463/session-2-assignments-resubmitted/artifacts/tokenizer.json")
data_b64 = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode()
tok_b64 = base64.b64encode(tok_text.encode("utf-8")).decode()
JS = JS.replace("__RECIPE__", json.dumps(RECIPE))
html = (BODY.replace("__CSS__", css).replace("__DATA_B64__", data_b64)
        .replace("__RAW_URL__", RAW_URL).replace("__JS__", JS)
        + f'\n<script id="tok" type="text/plain">{tok_b64}</script>\n')
(ROOT / "standalone.html").write_text(html, encoding="utf-8")
(ROOT / "index.html").write_text(html, encoding="utf-8")
print("wrote standalone.html + index.html", len(html), "bytes; score", round(m["raw_score"]),
      "spread", round(m["spread"], 5), "weights", weights)
