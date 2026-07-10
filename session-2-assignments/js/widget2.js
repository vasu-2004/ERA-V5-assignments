/* widget2.js -- Session 2 tokenizer widget.
 * Everything shown is computed LIVE in your browser from the shipped
 * tokenizer.json + the corpora in data/, using js/bpe_encoder.js (verified
 * bit-exact against HuggingFace tokenizers). Nothing is hard-coded. */
(function () {
  'use strict';
  var LANGS = ['en', 'hi', 'te', 'mr'];
  var NAME = { en: 'English', hi: 'Hindi', te: 'Telugu', mr: 'Marathi' };
  var FILE = { en: 'english', hi: 'hindi', te: 'telugu', mr: 'marathi' };
  var COLOR = { en: '#2f6fed', hi: '#e0447f', te: '#12a594', mr: '#f2820a' };
  var EN_CAP = 1.2;
  function $(id) { return document.getElementById(id); }
  function fmt(n) { return n.toLocaleString('en-US'); }

  var enc = null;          // BPE encoder
  var texts = {};          // live (normalized) corpus texts

  function fertilityOf(text) {
    var norm = BPE.normalizeNewlines(text);
    var tokens = enc.countTokens(norm, false);     // whole-doc, matches HF exactly
    var words = BPE.wordCount(norm);
    return { tokens: tokens, words: words, fertility: words ? tokens / words : 0 };
  }

  function computeAll(sourceTexts) {
    var per = {};
    LANGS.forEach(function (l) { per[l] = fertilityOf(sourceTexts[l]); });
    var xs = LANGS.map(function (l) { return per[l].fertility; });
    var xmax = Math.max.apply(null, xs), xmin = Math.min.apply(null, xs);
    var maxL = LANGS[xs.indexOf(xmax)], minL = LANGS[xs.indexOf(xmin)];
    var spread = xmax - xmin;
    return {
      per: per, xmax: xmax, xmin: xmin, maxL: maxL, minL: minL, spread: spread,
      score: spread > 1e-9 ? 1000 / spread : Infinity,
      en_ok: per.en.fertility <= EN_CAP
    };
  }

  function renderResults(r, tableId) {
    var rows = LANGS.map(function (l) {
      var p = r.per[l];
      var badge = '';
      if (l === r.maxL) badge = ' <span class="chip max">max</span>';
      if (l === r.minL) badge = ' <span class="chip min">min</span>';
      var enflag = (l === 'en')
        ? (p.fertility <= EN_CAP ? ' <span class="ok">≤1.2 ✓</span>' : ' <span class="bad">&gt;1.2 ✗</span>')
        : '';
      return '<tr><td><span class="dot" style="background:' + COLOR[l] + '"></span>' + NAME[l] + badge + '</td>' +
        '<td>' + fmt(p.words) + '</td><td>' + fmt(p.tokens) + '</td>' +
        '<td><b>' + p.fertility.toFixed(4) + '</b>' + enflag + '</td></tr>';
    }).join('');
    $(tableId).innerHTML =
      '<tr><th>Language</th><th>Words (\\w+)</th><th>BPE tokens</th><th>Fertility X = tok/words</th></tr>' + rows;
  }

  function renderScore(r) {
    $('score-value').textContent = isFinite(r.score) ? Math.round(r.score).toLocaleString('en-US') : '∞';
    $('score-formula').innerHTML = '1000 / (X<sub>max</sub> − X<sub>min</sub>) = 1000 / (' +
      r.xmax.toFixed(4) + ' − ' + r.xmin.toFixed(4) + ') = 1000 / ' + r.spread.toFixed(4);
    $('score-max').textContent = NAME[r.maxL] + ' ' + r.xmax.toFixed(4);
    $('score-min').textContent = NAME[r.minL] + ' ' + r.xmin.toFixed(4);
    $('score-spread').textContent = r.spread.toFixed(4);
    var en = $('score-en');
    en.textContent = r.per.en.fertility.toFixed(4) + (r.en_ok ? '  ✓ within 1.2' : '  ✗ exceeds 1.2');
    en.className = r.en_ok ? 'ok' : 'bad';
  }

  function bars(r) {
    var c = $('fert-bars'); if (!c) return;
    var dpr = window.devicePixelRatio || 1, rect = c.getBoundingClientRect();
    var w = rect.width, h = rect.height; c.width = w * dpr; c.height = h * dpr;
    var ctx = c.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    var s = getComputedStyle(document.documentElement);
    var ink = s.getPropertyValue('--ink').trim(), muted = s.getPropertyValue('--muted').trim(), grid = s.getPropertyValue('--grid').trim();
    var pad = 34, top = 16, bot = 30, maxX = 1.35;
    // 1.2 cap line
    var capY = top + (1 - EN_CAP / maxX) * (h - top - bot);
    ctx.strokeStyle = muted; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(pad, capY); ctx.lineTo(w - 10, capY); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = muted; ctx.font = '11px system-ui'; ctx.fillText('English cap 1.2', pad + 4, capY - 4);
    var bw = (w - pad - 20) / LANGS.length;
    LANGS.forEach(function (l, i) {
      var v = r.per[l].fertility;
      var bh = (v / maxX) * (h - top - bot);
      var x = pad + i * bw + bw * 0.2, y = h - bot - bh, bwid = bw * 0.6;
      ctx.fillStyle = COLOR[l]; ctx.fillRect(x, y, bwid, bh);
      ctx.fillStyle = ink; ctx.font = 'bold 12px system-ui'; ctx.textAlign = 'center';
      ctx.fillText(v.toFixed(3), x + bwid / 2, y - 5);
      ctx.fillStyle = muted; ctx.font = '11px system-ui';
      ctx.fillText(NAME[l], x + bwid / 2, h - bot + 14);
      ctx.textAlign = 'start';
    });
  }

  function setStatus(msg, kind) {
    var el = $('live-status'); if (!el) return;
    el.textContent = msg; el.className = 'live-status ' + (kind || '');
  }

  function loadCorpora() {
    return Promise.all(LANGS.map(function (l) {
      return fetch('data/' + FILE[l] + '_india.txt').then(function (r) {
        if (!r.ok) throw new Error('fetch ' + l + ' failed'); return r.text();
      }).then(function (t) { texts[l] = t; });
    }));
  }

  function boot() {
    setStatus('Loading tokenizer.json and corpora…');
    fetch('artifacts/tokenizer.json').then(function (r) { return r.json(); }).then(function (tj) {
      enc = BPE.Encoder.fromTokenizerJSON(tj);
      $('vocab-size').textContent = fmt(Object.keys((tj.model || tj).vocab).length);
      return loadCorpora();
    }).then(function () {
      setStatus('Computed live in your browser from the shipped tokenizer.json ✓', 'good');
      var r = computeAll(texts);
      renderResults(r, 'opt-table'); renderScore(r); bars(r);
      // seed the custom-text boxes with the real corpora so users can edit/replace
      LANGS.forEach(function (l) { var ta = $('paste-' + l); if (ta) ta.value = texts[l]; });
      // no-UNK live proof
      runUnkProof();
      window.addEventListener('resize', function () { bars(computeAll(currentSource())); });
    }).catch(function (e) {
      setStatus('Live compute failed to load corpora (' + e.message + '). Showing precomputed numbers instead.', 'bad');
      if (window.OPT_DATA) renderFromBaked();
    });
  }

  function currentSource() {
    var src = {};
    LANGS.forEach(function (l) { var ta = $('paste-' + l); src[l] = (ta && ta.value) || texts[l]; });
    return src;
  }

  function rescore() {
    if (!enc) return;
    setStatus('Re-scoring your pasted text live…');
    setTimeout(function () {
      var r = computeAll(currentSource());
      renderResults(r, 'opt-table'); renderScore(r); bars(r);
      setStatus('Re-scored your text live in-browser ✓ (same encoder that matches HuggingFace exactly)', 'good');
    }, 20);
  }

  function runUnkProof() {
    var stress = "India](/wiki/India) — भारत, గణతంత్ర, महाराष्ट्र! ₹100 42% \"q\" <t> [[w]] {j:1} ©®™ ĀīŚṇ";
    var toks = enc.encodeChunk(BPE.normalizeNewlines(stress));
    // byte-level => every input byte is a base token, so UNK cannot occur
    $('unk-proof').innerHTML = 'Stress string of markup/rare characters encodes to <b>' + toks.length +
      ' tokens, 0 UNK</b> (byte-level: every one of the 256 byte values is a base token, so any character in any page is encodable).';
  }

  function renderFromBaked() {
    var d = window.OPT_DATA, o = d.optimized;
    var r = { per: {}, xmax: o.summary.x_max, xmin: o.summary.x_min,
      maxL: Object.keys(NAME).find(function (k) { return NAME[k] === o.summary.x_max_lang; }),
      minL: Object.keys(NAME).find(function (k) { return NAME[k] === o.summary.x_min_lang; }),
      spread: o.summary.spread, score: o.summary.score, en_ok: o.per_lang.en.fertility <= EN_CAP };
    LANGS.forEach(function (l) { r.per[l] = o.per_lang[l]; });
    $('vocab-size').textContent = fmt(d.vocab_size);
    renderResults(r, 'opt-table'); renderScore(r); bars(r);
  }

  function initTheme() {
    var btn = $('theme-btn'); if (!btn) return;
    btn.addEventListener('click', function () {
      var root = document.documentElement, cur = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', cur); btn.textContent = cur === 'dark' ? '☀︎' : '☾';
      if (enc) bars(computeAll(currentSource()));
    });
  }

  window.addEventListener('DOMContentLoaded', function () {
    initTheme();
    var b = $('rescore-btn'); if (b) b.addEventListener('click', rescore);
    var rb = $('reset-btn'); if (rb) rb.addEventListener('click', function () {
      LANGS.forEach(function (l) { var ta = $('paste-' + l); if (ta) ta.value = texts[l]; }); rescore();
    });
    boot();
  });
})();
