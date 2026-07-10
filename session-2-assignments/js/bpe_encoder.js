/* ============================================================================
 * bpe_encoder.js -- a faithful, efficient byte-level BPE encoder that loads an
 * HuggingFace tokenizer.json (ByteLevel, use_regex=False) and reproduces its
 * token counts exactly. Works in the browser and in Node.
 *
 * Algorithm: canonical BPE (always apply the globally lowest-rank adjacent
 * merge) implemented with a doubly-linked list + binary min-heap of candidate
 * merges -> O(n log n), fast enough to re-tokenize a whole Wikipedia page live.
 * ==========================================================================*/
(function (global) {
  'use strict';

  // GPT-2 byte<->unicode table (same one HF ByteLevel uses)
  function byteToUnicode() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);       // ! .. ~
    for (let i = 161; i <= 172; i++) bs.push(i);      // ¡ .. ¬
    for (let i = 174; i <= 255; i++) bs.push(i);      // ® .. ÿ
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    }
    const map = new Array(256);
    for (let i = 0; i < bs.length; i++) map[bs[i]] = String.fromCharCode(cs[i]);
    return map;
  }
  const B2U = byteToUnicode();

  function Encoder(vocab, merges) {
    this.vocab = vocab;                 // {tokenString: id}
    this.rank = new Map();              // "a b" -> rank
    for (let i = 0; i < merges.length; i++) {
      const m = merges[i];
      const key = Array.isArray(m) ? (m[0] + ' ' + m[1]) : m;
      this.rank.set(key, i);
    }
    this.enc = new TextEncoder();
  }

  // encode one chunk (string) -> array of token strings
  Encoder.prototype.encodeChunk = function (text) {
    if (!text) return [];
    const bytes = this.enc.encode(text);
    const N = bytes.length;
    if (N === 0) return [];
    // symbols as linked list
    const sym = new Array(N);
    const prev = new Int32Array(N);
    const next = new Int32Array(N);
    const alive = new Uint8Array(N);
    for (let i = 0; i < N; i++) { sym[i] = B2U[bytes[i]]; prev[i] = i - 1; next[i] = i + 1; alive[i] = 1; }
    next[N - 1] = -1;
    // min-heap of [rank, i] ordered by (rank, i): lower rank first, and for
    // EQUAL rank the leftmost pair first -- this reproduces canonical BPE's
    // left-to-right merging of overlapping equal-rank pairs (e.g. "yyyy" -> yy yy).
    const heap = [];
    const rankOf = (a, b) => { const r = this.rank.get(a + ' ' + b); return r === undefined ? -1 : r; };
    const lt = (x, y) => (x[0] !== y[0] ? x[0] < y[0] : x[1] < y[1]);
    const push = (i) => {
      const j = next[i];
      if (j === -1) return;
      const r = rankOf(sym[i], sym[j]);
      if (r < 0) return;
      const e = [r, i];
      heap.push(e);
      let c = heap.length - 1;
      while (c > 0) { const p = (c - 1) >> 1; if (!lt(heap[c], heap[p])) break; const t = heap[p]; heap[p] = heap[c]; heap[c] = t; c = p; }
    };
    const pop = () => {
      const top = heap[0], last = heap.pop();
      if (heap.length) { heap[0] = last; let c = 0; const n = heap.length;
        while (true) { let l = 2 * c + 1, r = 2 * c + 2, s = c;
          if (l < n && lt(heap[l], heap[s])) s = l;
          if (r < n && lt(heap[r], heap[s])) s = r;
          if (s === c) break; const t = heap[s]; heap[s] = heap[c]; heap[c] = t; c = s; } }
      return top;
    };
    for (let i = 0; i < N; i++) push(i);
    while (heap.length) {
      const [r, i] = pop();
      if (!alive[i]) continue;
      const j = next[i];
      if (j === -1 || !alive[j]) continue;
      if (rankOf(sym[i], sym[j]) !== r) continue;   // stale entry
      // merge j into i
      sym[i] = sym[i] + sym[j];
      alive[j] = 0;
      const k = next[j];
      next[i] = k;
      if (k !== -1) prev[k] = i;
      // new candidate pairs (prev[i], i) and (i, k)
      if (prev[i] !== -1 && alive[prev[i]]) push(prev[i]);
      push(i);
    }
    const out = [];
    for (let i = 0; i !== -1; i = next[i]) if (alive[i]) out.push(sym[i]);
    return out;
  };

  // encode a full document. `chunkOnNewlines`=false => faithful whole-doc BPE
  // (matches HF encode(full_text)). true => faster per-line approximation.
  Encoder.prototype.countTokens = function (text, chunkOnNewlines) {
    if (chunkOnNewlines) {
      let total = 0;
      const lines = text.split('\n');
      for (let i = 0; i < lines.length; i++) total += this.encodeChunk(lines[i]).length;
      return total;
    }
    return this.encodeChunk(text).length;
  };

  Encoder.fromTokenizerJSON = function (tj) {
    const model = tj.model || tj;
    return new Encoder(model.vocab, model.merges);
  };

  // Match Python's universal-newline read (\r\n, \r -> \n) so token counts agree
  // with the HuggingFace pipeline the tokenizer was trained/evaluated with.
  function normalizeNewlines(text) {
    return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
  }
  // Reproduces Python's len(re.findall(r"\w+", text)) EXACTLY (verified on all
  // four corpora): \p{L}\p{N}_ , which -- like Python's \w -- excludes the
  // Devanagari/Telugu combining vowel signs (category \p{M}).
  const WORD_RE = /[\p{L}\p{N}_]+/gu;
  function wordCount(text) {
    const m = text.match(WORD_RE);
    return m ? m.length : 0;
  }

  const api = { Encoder: Encoder, byteToUnicode: byteToUnicode,
                normalizeNewlines: normalizeNewlines, wordCount: wordCount };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.BPE = api;
})(typeof window !== 'undefined' ? window : globalThis);
