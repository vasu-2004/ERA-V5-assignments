/* ============================================================================
 * demos.js — wires each section's controls -> training loop -> plots.
 * Each demo owns a requestAnimationFrame loop; re-running cancels the old loop.
 * ========================================================================== */
(function () {
  'use strict';

  /* ---- shared helpers --------------------------------------------------- */
  function $(id) { return document.getElementById(id); }

  // pack {X:[n][2], y:[n]} into NN matrices (Y is a single column, 0/1)
  function toMat(data) {
    const n = data.X.length;
    const X = NN.mat(n, 2), Y = NN.mat(n, 1);
    for (let i = 0; i < n; i++) { X.data[i * 2] = data.X[i][0]; X.data[i * 2 + 1] = data.X[i][1]; Y.data[i] = data.y[i]; }
    return { X: X, Y: Y };
  }
  // class-1 probability for a list of points, in one batched forward
  function batchProbs(mlp, points) {
    const n = points.length, X = NN.mat(n, 2);
    for (let i = 0; i < n; i++) { X.data[i * 2] = points[i][0]; X.data[i * 2 + 1] = points[i][1]; }
    const pred = mlp.forward(X);
    return pred.data;
  }
  // a cancellable animation driver: stepFn() does work, returns true when done
  function runLoop(state, stepFn) {
    if (state.raf) cancelAnimationFrame(state.raf);
    const tick = function () { if (stepFn()) { state.raf = null; return; } state.raf = requestAnimationFrame(tick); };
    state.raf = requestAnimationFrame(tick);
  }

  const WIN = { xmin: -3.4, xmax: 3.4, ymin: -3.4, ymax: 3.4 };
  const RES = 46;
  const grid = Plot.gridPoints(RES, WIN);
  const DRAW_EVERY = 3;   // redraw expensive boundary grids every N frames

  /* ======================================================================
   * S1-1 — Activations exist for a reason
   * ==================================================================== */
  function initS1() {
    const st = { raf: null };
    function build() {
      if (st.raf) cancelAnimationFrame(st.raf);
      const noise = parseFloat($('s1-noise').value);
      const hidden = parseInt($('s1-hidden').value, 10);
      const useRelu = $('s1-relu-toggle').checked;
      const seed = (Math.random() * 1e6) | 0;
      $('s1-noise-val').textContent = noise.toFixed(2);
      $('s1-hidden-val').textContent = hidden;

      const raw = Data.rings({ n: 300, noise: noise, seed: seed });
      const d = toMat(raw);
      // model A: linear + sigmoid (no hidden layer)
      const A = new NN.MLP(2, [{ units: 1, activation: 'sigmoid' }], 'bce', seed + 1);
      // model B: one hidden layer, ReLU (or linear if toggle off)
      const B = new NN.MLP(2, [{ units: hidden, activation: useRelu ? 'relu' : 'linear' }, { units: 1, activation: 'sigmoid' }], 'bce', seed + 2);

      const lossA = [], lossB = [];
      let epoch = 0, tick = 0; const maxEpoch = 600;
      const lr = 0.05;
      runLoop(st, function () {
        for (let k = 0; k < 8 && epoch < maxEpoch; k++) {
          lossA.push(A.trainStep(d.X, d.Y, lr));
          lossB.push(B.trainStep(d.X, d.Y, lr));
          epoch++;
        }
        const done = epoch >= maxEpoch;
        if (tick++ % DRAW_EVERY === 0 || done) {
          Plot.decision($('s1-lin'), batchProbs(A, grid), RES, raw, WIN);
          Plot.decision($('s1-relu'), batchProbs(B, grid), RES, raw, WIN);
          Plot.curves($('s1-loss'), [
            { data: lossA, color: '#94a3b8', label: 'linear+sigmoid' },
            { data: lossB, color: Plot.COLORS.cat.verb, label: useRelu ? 'ReLU hidden' : 'linear hidden' }
          ], { yMax: 0.8 });
          $('s1-lin-acc').textContent = (A.accuracy(d.X, d.Y) * 100).toFixed(1) + '%';
          $('s1-relu-acc').textContent = (B.accuracy(d.X, d.Y) * 100).toFixed(1) + '%';
        }
        $('s1-epoch').textContent = epoch;
        $('s1-relu-title').textContent = useRelu ? 'One ReLU hidden layer' : 'Hidden layer, NO activation';
        return done;
      });
    }
    ['s1-noise', 's1-hidden', 's1-relu-toggle'].forEach(function (id) { $(id).addEventListener('input', build); });
    $('s1-run').addEventListener('click', build);
    build();
    return build;
  }

  /* ======================================================================
   * S1-2 — Depth without nonlinearity is a lie
   * ==================================================================== */
  function initS2() {
    const st = { raf: null };
    function build() {
      if (st.raf) cancelAnimationFrame(st.raf);
      const seed = (Math.random() * 1e6) | 0;
      const raw = Data.rings({ n: 300, noise: 0.12, seed: seed });
      const d = toMat(raw);
      const lr = 0.04;
      // 1 linear layer
      const M1 = new NN.MLP(2, [{ units: 1, activation: 'sigmoid' }], 'bce', seed + 1);
      // 5 LINEAR layers (widths 4,4,4,4,1) — no activations
      const spec5 = [{ units: 4, activation: 'linear' }, { units: 4, activation: 'linear' }, { units: 4, activation: 'linear' }, { units: 4, activation: 'linear' }, { units: 1, activation: 'sigmoid' }];
      const M5lin = new NN.MLP(2, spec5, 'bce', seed + 2);
      // 5 layers with ReLU between them
      const spec5r = [{ units: 8, activation: 'relu' }, { units: 8, activation: 'relu' }, { units: 8, activation: 'relu' }, { units: 8, activation: 'relu' }, { units: 1, activation: 'sigmoid' }];
      const M5relu = new NN.MLP(2, spec5r, 'bce', seed + 3);

      let epoch = 0, tick = 0; const maxEpoch = 1000;
      runLoop(st, function () {
        for (let k = 0; k < 10 && epoch < maxEpoch; k++) {
          M1.trainStep(d.X, d.Y, lr); M5lin.trainStep(d.X, d.Y, lr); M5relu.trainStep(d.X, d.Y, lr); epoch++;
        }
        const done = epoch >= maxEpoch;
        if (tick++ % DRAW_EVERY === 0 || done) {
          Plot.decision($('s2-lin1'), batchProbs(M1, grid), RES, raw, WIN);
          Plot.decision($('s2-lin5'), batchProbs(M5lin, grid), RES, raw, WIN);
          Plot.decision($('s2-relu5'), batchProbs(M5relu, grid), RES, raw, WIN);
          $('s2-lin1-acc').textContent = (M1.accuracy(d.X, d.Y) * 100).toFixed(1) + '%';
          $('s2-lin5-acc').textContent = (M5lin.accuracy(d.X, d.Y) * 100).toFixed(1) + '%';
          $('s2-relu5-acc').textContent = (M5relu.accuracy(d.X, d.Y) * 100).toFixed(1) + '%';
        }
        $('s2-epoch').textContent = epoch;

        if (done || epoch % 200 === 0) {
          // collapse proof on the 5-linear net (drop its sigmoid output squash:
          // fold only the linear layers). We fold layers[0..3] (all linear) and
          // show the effective pre-activation map, which is a single 2xN matrix.
          const linLayers = M5lin.layers.slice(0, M5lin.layers.length); // last has sigmoid but its weights are still a linear map into the logit
          renderCollapse(linLayers);
        }
        return done;
      });
    }
    function renderCollapse(layers) {
      // render each weight matrix, then the folded product W_eff = W0·W1·…·W4
      for (let i = 0; i < 5; i++) Plot.matrixGrid($('s2-w' + i), layers[i].W);
      let W = layers[0].W;
      for (let i = 1; i < 5; i++) W = NN.matmul(W, layers[i].W);
      Plot.matrixGrid($('s2-weff'), W);
      $('s2-eff-note').textContent =
        'Five weight matrices (' + layers.map(function (l) { return l.W.rows + '×' + l.W.cols; }).join(', ') +
        ') multiply into ONE ' + W.rows + '×' + W.cols + ' matrix. All those parameters describe a single linear map.';
    }
    $('s2-run').addEventListener('click', build);
    build();
    return build;
  }

  /* ======================================================================
   * S1-3 — Embeddings learn similarity from next-token alone
   * ==================================================================== */
  function initS3() {
    const st = { raf: null };
    const g = Data.grammar;
    function build() {
      if (st.raf) cancelAnimationFrame(st.raf);
      const dim = parseInt($('s3-dim').value, 10);
      $('s3-dim-val').textContent = dim;
      const seed = (Math.random() * 1e6) | 0;
      const gp = g.pairs({ n: 900, seed: seed });
      $('s3-sentences').innerHTML = gp.sampleSentences.map(function (s) { return '<code>' + s + '</code>'; }).join('');
      const model = new NN.EmbeddingModel(g.tokens.length, dim, seed + 5);
      const loss = [];
      let epoch = 0, tick = 0; const maxEpoch = 350; const lr = 0.4;
      runLoop(st, function () {
        for (let k = 0; k < 4 && epoch < maxEpoch; k++) { loss.push(model.trainStep(gp.pairs, lr)); epoch++; }
        const done = epoch >= maxEpoch;
        if (tick++ % DRAW_EVERY === 0 || done) {
          const rows0 = model.embeddingRows();
          Plot.embedding($('s3-embed'), NN.pca2d(rows0), g.tokens, g.tokens.map(function (t) { return g.catOf[t]; }));
          Plot.curves($('s3-loss'), [{ data: loss, color: Plot.COLORS.cat.verb, label: 'next-token loss' }], {});
        }
        $('s3-epoch').textContent = epoch;
        if (done || epoch % 40 === 0) {
          // cosine similarity heatmap + nearest neighbours
          const rows = model.embeddingRows();
          const n = rows.length, sim = [];
          for (let i = 0; i < n; i++) { sim.push([]); for (let j = 0; j < n; j++) sim[i].push(NN.cosine(rows[i], rows[j])); }
          Plot.heatmap($('s3-heat'), sim, g.tokens);
          renderNN(rows, sim, g);
        }
        return done;
      });
    }
    function renderNN(rows, sim, g) {
      // the '.' end-marker is a lone structural token with no category peers, so
      // it is not part of the clustering claim — skip it as query and candidate.
      const real = g.tokens.map(function (t, i) { return i; }).filter(function (i) { return g.catOf[g.tokens[i]] !== 'stop'; });
      let html = '<tr><th>token</th><th>nearest neighbour</th><th>same category?</th></tr>';
      real.forEach(function (i) {
        let best = -1, bestv = -Infinity;
        real.forEach(function (j) { if (j !== i && sim[i][j] > bestv) { bestv = sim[i][j]; best = j; } });
        const same = g.catOf[g.tokens[i]] === g.catOf[g.tokens[best]];
        html += '<tr><td>' + g.tokens[i] + '</td><td>' + g.tokens[best] + '</td><td>' +
          (same ? '<span class="ok">yes</span>' : '<span class="bad">no</span>') + '</td></tr>';
      });
      $('s3-nn').innerHTML = html;
    }
    $('s3-dim').addEventListener('input', build);
    $('s3-run').addEventListener('click', build);
    build();
    return build;
  }

  /* ======================================================================
   * S1-4 — Memorization vs generalization, and data closes the gap
   * ==================================================================== */
  function initS4() {
    const st = { raf: null };
    const sizes = [20, 200, 2000];
    const canv = { 20: 's4-b20', 200: 's4-b200', 2000: 's4-b2000' };
    function build() {
      if (st.raf) cancelAnimationFrame(st.raf);
      const hidden = parseInt($('s4-hidden').value, 10);
      $('s4-hidden-val').textContent = hidden;
      const seed = (Math.random() * 1e6) | 0;
      const spec = [{ units: hidden, activation: 'relu' }, { units: hidden, activation: 'relu' }, { units: 1, activation: 'sigmoid' }];
      const runs = sizes.map(function (nTrain, idx) {
        const ds = Data.blobs({ nTrain: nTrain, nTest: 500, noise: 0.22, labelNoise: 0.05, seed: seed + idx });
        return {
          n: nTrain,
          train: toMat(ds.train), test: toMat(ds.test), rawTrain: ds.train,
          mlp: new NN.MLP(2, spec, 'bce', seed + 100 + idx),
          gap: []
        };
      });
      const win = { xmin: -1.6, xmax: 2.6, ymin: -1.4, ymax: 1.9 };
      const grid4 = Plot.gridPoints(RES, win);
      let epoch = 0, tick = 0; const maxEpoch = 700; const lr = 0.03;
      runLoop(st, function () {
        for (let k = 0; k < 8 && epoch < maxEpoch; k++) {
          runs.forEach(function (r) { r.mlp.trainStep(r.train.X, r.train.Y, lr); });
          epoch++;
        }
        const done = epoch >= maxEpoch;
        runs.forEach(function (r) {
          const trAcc = r.mlp.accuracy(r.train.X, r.train.Y);
          const teAcc = r.mlp.accuracy(r.test.X, r.test.Y);
          r.gap.push(trAcc - teAcc);
          if (tick % DRAW_EVERY === 0 || done) {
            Plot.decision($(canv[r.n]), batchProbs(r.mlp, grid4), RES, r.rawTrain, win);
            $('s4-' + r.n + '-tr').textContent = (trAcc * 100).toFixed(0) + '%';
            $('s4-' + r.n + '-te').textContent = (teAcc * 100).toFixed(0) + '%';
          }
        });
        if (tick++ % DRAW_EVERY === 0 || done) {
          Plot.curves($('s4-gap'), runs.map(function (r, i) {
            return { data: r.gap, color: [Plot.COLORS.class1, Plot.COLORS.cat.verb, Plot.COLORS.class0][i], label: 'n=' + r.n };
          }), { yMax: 0.5 });
        }
        $('s4-epoch').textContent = epoch;
        return done;
      });
    }
    $('s4-hidden').addEventListener('input', build);
    $('s4-run').addEventListener('click', build);
    build();
    return build;
  }

  /* ---- theme toggle + boot ---------------------------------------------- */
  function initTheme(rebuilds) {
    const btn = $('theme-btn'); if (!btn) return;
    btn.addEventListener('click', function () {
      const root = document.documentElement;
      const cur = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', cur);
      btn.textContent = cur === 'dark' ? '☀︎' : '☾';
      rebuilds.forEach(function (b) { b(); });   // re-train + redraw in new theme
    });
  }

  window.addEventListener('DOMContentLoaded', function () {
    const builds = [initS1(), initS2(), initS3(), initS4()];
    initTheme(builds);
  });
})();
