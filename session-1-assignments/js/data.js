/* ============================================================================
 * data.js — dataset generators for the four demos
 *   Data.rings()        concentric rings (S1-1, S1-2)
 *   Data.grammar        toy language + next-token pairs (S1-3)
 *   Data.blobs()        learnable noisy 2-D classification with a test split (S1-4)
 * All generators take a seed so every plot is reproducible.
 * ========================================================================== */
(function (global) {
  'use strict';

  /* ---- concentric rings ------------------------------------------------- *
   * inner ring = class 0, outer ring = class 1. Not linearly separable.
   * Returns { X:[n][2], y:[n] } roughly centered on the origin.
   * -------------------------------------------------------------------------*/
  function rings(opts) {
    opts = opts || {};
    const n = opts.n || 300;
    const noise = opts.noise == null ? 0.12 : opts.noise;
    const rInner = opts.rInner == null ? 1.0 : opts.rInner;
    const rOuter = opts.rOuter == null ? 2.4 : opts.rOuter;
    const rand = NN.rng(opts.seed || 1);
    const X = [], y = [];
    for (let i = 0; i < n; i++) {
      const cls = i % 2;                       // balanced classes
      const r = cls === 0 ? rInner : rOuter;
      const theta = rand() * 2 * Math.PI;
      const rr = r + rand.normal(0, noise);
      X.push([rr * Math.cos(theta), rr * Math.sin(theta)]);
      y.push(cls);
    }
    return { X: X, y: y };
  }

  /* ---- two interleaving moons style noisy blobs (S1-4) ------------------ *
   * Learnable structure (two gaussian clusters per class arranged so the
   * true boundary is curved) plus label noise, split into train / test.
   * -------------------------------------------------------------------------*/
  function blobs(opts) {
    opts = opts || {};
    const nTrain = opts.nTrain || 200;
    const nTest = opts.nTest || 500;
    const noise = opts.noise == null ? 0.22 : opts.noise;
    const labelNoise = opts.labelNoise == null ? 0.05 : opts.labelNoise;
    const rand = NN.rng(opts.seed || 7);
    // two moons
    function sample(n) {
      const X = [], y = [];
      for (let i = 0; i < n; i++) {
        const cls = i % 2;
        const t = rand() * Math.PI;
        let px, py;
        if (cls === 0) { px = Math.cos(t); py = Math.sin(t); }
        else { px = 1 - Math.cos(t); py = 0.5 - Math.sin(t); }
        px += rand.normal(0, noise); py += rand.normal(0, noise);
        let label = cls;
        if (rand() < labelNoise) label = 1 - label;   // flip a few labels
        X.push([px, py]); y.push(label);
      }
      return { X: X, y: y };
    }
    return { train: sample(nTrain), test: sample(nTest) };
  }

  /* ---- toy grammar (S1-3) ----------------------------------------------- *
   * Categories deliberately share slots in the templates, so same-category
   * tokens end up with identical next-token distributions. The model never
   * sees the categories — only (prev -> next) token pairs.
   * -------------------------------------------------------------------------*/
  const categories = {
    animal: ['cat', 'dog', 'cow'],
    fruit: ['apple', 'mango', 'grape'],
    verb: ['eat', 'chase', 'see'],
    det: ['the', 'a']
  };
  // '.' is an explicit end-of-sentence marker. It matters: without it, whichever
  // category sits last would never be a *previous* token, so its embeddings would
  // get no gradient and never cluster. With '.', every category token has a
  // consistent next-token signature and clusters cleanly:
  //   det -> animal|fruit   animal -> verb   verb -> det   fruit -> '.'
  const STOP = '.';
  const catOf = {};
  const tokens = [];
  Object.keys(categories).forEach(function (c) {
    categories[c].forEach(function (t) { tokens.push(t); catOf[t] = c; });
  });
  tokens.push(STOP); catOf[STOP] = 'stop';
  const tokenId = {};
  tokens.forEach(function (t, i) { tokenId[t] = i; });

  // Category-slot templates; every sentence is terminated with STOP.
  const templates = [
    ['det', 'animal', 'verb', 'det', 'fruit'],
    ['det', 'animal', 'verb', 'det', 'fruit']   // duplicated for a clean, consistent signature
  ];

  function sampleSentence(rand) {
    const tmpl = templates[Math.floor(rand() * templates.length)];
    const words = tmpl.map(function (slot) {
      const choices = categories[slot];
      return choices[Math.floor(rand() * choices.length)];
    });
    words.push(STOP);
    return words;
  }

  function grammarPairs(opts) {
    opts = opts || {};
    const nSentences = opts.n || 800;
    const rand = NN.rng(opts.seed || 3);
    const pairs = [];
    const sample = [];
    for (let s = 0; s < nSentences; s++) {
      const sent = sampleSentence(rand);
      if (s < 8) sample.push(sent.join(' '));
      for (let i = 0; i < sent.length - 1; i++) pairs.push([tokenId[sent[i]], tokenId[sent[i + 1]]]);
    }
    return { pairs: pairs, sampleSentences: sample };
  }

  const grammar = {
    categories: categories,
    tokens: tokens,
    catOf: catOf,
    tokenId: tokenId,
    templates: templates,
    pairs: grammarPairs
  };

  global.Data = { rings: rings, blobs: blobs, grammar: grammar };
})(window);
