/* ============================================================================
 * nn.js — a tiny, dependency-free neural-network engine
 * ----------------------------------------------------------------------------
 * Everything here runs in the browser on plain Float64Arrays. The nets we train
 * are small (2-D inputs, <=32 hidden units, tiny embeddings), so naive matmul is
 * plenty fast and the whole thing stays readable.
 *
 * Exposes a global `NN` object with:
 *   NN.rng(seed)            -> seeded PRNG (mulberry32)
 *   NN.MLP                  -> a simple sequential multi-layer perceptron
 *   NN.EmbeddingModel       -> embedding -> softmax next-token model
 *   NN.pca2d(rows)          -> project N-d vectors down to 2-D
 *   NN.cosine(a, b)         -> cosine similarity
 *   matrix helpers (mat, matmul, transpose, ...)
 * ========================================================================== */
(function (global) {
  'use strict';

  /* ---- seeded RNG so every demo is reproducible ------------------------- */
  function rng(seed) {
    let a = (seed >>> 0) || 1;
    const next = function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    // standard-normal via Box-Muller
    next.normal = function (mean, std) {
      const u1 = Math.max(next(), 1e-12), u2 = next();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      return (mean || 0) + (std == null ? 1 : std) * z;
    };
    return next;
  }

  /* ---- flat row-major matrices ------------------------------------------ */
  function mat(rows, cols, data) {
    return { rows: rows, cols: cols, data: data || new Float64Array(rows * cols) };
  }
  function matmul(A, B) {                 // (rows x k) * (k x cols)
    const rows = A.rows, k = A.cols, cols = B.cols, a = A.data, b = B.data;
    const out = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      const arow = i * k, orow = i * cols;
      for (let p = 0; p < k; p++) {
        const av = a[arow + p];
        if (av === 0) continue;
        const brow = p * cols;
        for (let j = 0; j < cols; j++) out[orow + j] += av * b[brow + j];
      }
    }
    return mat(rows, cols, out);
  }
  function transpose(A) {
    const out = new Float64Array(A.rows * A.cols);
    for (let i = 0; i < A.rows; i++)
      for (let j = 0; j < A.cols; j++) out[j * A.rows + i] = A.data[i * A.cols + j];
    return mat(A.cols, A.rows, out);
  }

  /* ---- activations ------------------------------------------------------ */
  const ACT = {
    relu: { f: function (x) { return x > 0 ? x : 0; }, dfz: function (z) { return z > 0 ? 1 : 0; } },
    tanh: { f: function (x) { return Math.tanh(x); }, dfz: function (z) { const t = Math.tanh(z); return 1 - t * t; } },
    sigmoid: { f: function (x) { return 1 / (1 + Math.exp(-x)); }, dfz: null },
    linear: { f: function (x) { return x; }, dfz: function () { return 1; } }
  };

  /* ---- a single dense layer: y = act(x W + b) --------------------------- */
  function Dense(inDim, outDim, activation, rand) {
    this.inDim = inDim; this.outDim = outDim; this.activation = activation;
    this.W = mat(inDim, outDim);
    this.b = new Float64Array(outDim);
    // He init for relu, Xavier otherwise
    const scale = activation === 'relu' ? Math.sqrt(2 / inDim) : Math.sqrt(1 / inDim);
    for (let i = 0; i < this.W.data.length; i++) this.W.data[i] = rand.normal(0, scale);
    // adam state
    this.mW = new Float64Array(inDim * outDim); this.vW = new Float64Array(inDim * outDim);
    this.mb = new Float64Array(outDim); this.vb = new Float64Array(outDim);
    this.t = 0;
  }
  Dense.prototype.forward = function (X) {
    this.X = X;
    const Z = matmul(X, this.W);
    for (let i = 0; i < Z.rows; i++)
      for (let j = 0; j < Z.cols; j++) Z.data[i * Z.cols + j] += this.b[j];
    this.Z = Z;
    const act = ACT[this.activation];
    const A = mat(Z.rows, Z.cols, Float64Array.from(Z.data, act.f));
    this.A = A;
    return A;
  };
  // dOut is dL/dA (or dL/dZ directly when isOutput=true).
  Dense.prototype.backward = function (dOut, isOutput) {
    const n = this.X.rows;
    let dZ;
    if (isOutput) {
      dZ = dOut;                                  // loss already gives dL/dZ
    } else {
      const dfz = ACT[this.activation].dfz;
      dZ = mat(dOut.rows, dOut.cols, new Float64Array(dOut.data.length));
      for (let i = 0; i < dOut.data.length; i++) dZ.data[i] = dOut.data[i] * dfz(this.Z.data[i]);
    }
    this.dW = matmul(transpose(this.X), dZ);      // (in x out)
    this.db = new Float64Array(this.outDim);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < this.outDim; j++) this.db[j] += dZ.data[i * this.outDim + j];
    for (let k = 0; k < this.dW.data.length; k++) this.dW.data[k] /= n;
    for (let j = 0; j < this.outDim; j++) this.db[j] /= n;
    return matmul(dZ, transpose(this.W));         // dL/dX for the previous layer
  };
  Dense.prototype.step = function (lr) {
    // Adam
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const bc1 = 1 - Math.pow(b1, this.t), bc2 = 1 - Math.pow(b2, this.t);
    const W = this.W.data, dW = this.dW.data;
    for (let i = 0; i < W.length; i++) {
      this.mW[i] = b1 * this.mW[i] + (1 - b1) * dW[i];
      this.vW[i] = b2 * this.vW[i] + (1 - b2) * dW[i] * dW[i];
      W[i] -= lr * (this.mW[i] / bc1) / (Math.sqrt(this.vW[i] / bc2) + eps);
    }
    for (let j = 0; j < this.b.length; j++) {
      this.mb[j] = b1 * this.mb[j] + (1 - b1) * this.db[j];
      this.vb[j] = b2 * this.vb[j] + (1 - b2) * this.db[j] * this.db[j];
      this.b[j] -= lr * (this.mb[j] / bc1) / (Math.sqrt(this.vb[j] / bc2) + eps);
    }
  };

  /* ---- MLP: a stack of dense layers ------------------------------------- *
   * spec = [{units, activation}, ...]; the last layer is the output.
   * loss = 'bce' (sigmoid output) | 'ce' (softmax output) | 'mse'
   * -------------------------------------------------------------------------*/
  function MLP(inDim, spec, loss, seed) {
    this.loss = loss;
    this.rand = rng(seed);
    this.layers = [];
    let d = inDim;
    for (let i = 0; i < spec.length; i++) {
      this.layers.push(new Dense(d, spec[i].units, spec[i].activation, this.rand));
      d = spec[i].units;
    }
  }
  function softmaxRows(Z) {
    const out = mat(Z.rows, Z.cols, new Float64Array(Z.data.length));
    for (let i = 0; i < Z.rows; i++) {
      let mx = -Infinity;
      for (let j = 0; j < Z.cols; j++) mx = Math.max(mx, Z.data[i * Z.cols + j]);
      let s = 0;
      for (let j = 0; j < Z.cols; j++) { const e = Math.exp(Z.data[i * Z.cols + j] - mx); out.data[i * Z.cols + j] = e; s += e; }
      for (let j = 0; j < Z.cols; j++) out.data[i * Z.cols + j] /= s;
    }
    return out;
  }
  MLP.prototype.forward = function (X) {
    let A = X;
    for (let i = 0; i < this.layers.length; i++) A = this.layers[i].forward(A);
    if (this.loss === 'ce') { this.pred = softmaxRows(this.layers[this.layers.length - 1].Z); return this.pred; }
    this.pred = A; return A;
  };
  // Y: target matrix (same shape as pred). Returns scalar loss and runs backward.
  MLP.prototype.trainStep = function (X, Y, lr) {
    const pred = this.forward(X);
    const n = pred.rows, m = pred.cols;
    let lossVal = 0;
    const dZ = mat(n, m, new Float64Array(n * m));
    if (this.loss === 'bce') {
      for (let i = 0; i < n * m; i++) {
        const p = Math.min(Math.max(pred.data[i], 1e-9), 1 - 1e-9), y = Y.data[i];
        lossVal += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
        dZ.data[i] = p - y;                          // sigmoid+BCE => dL/dZ = p - y
      }
    } else if (this.loss === 'ce') {
      for (let i = 0; i < n; i++)
        for (let j = 0; j < m; j++) {
          const p = Math.max(pred.data[i * m + j], 1e-9), y = Y.data[i * m + j];
          if (y > 0) lossVal += -y * Math.log(p);
          dZ.data[i * m + j] = pred.data[i * m + j] - y;  // softmax+CE => dL/dZ = p - y
        }
    } else { // mse (linear output)
      for (let i = 0; i < n * m; i++) {
        const e = pred.data[i] - Y.data[i];
        lossVal += 0.5 * e * e; dZ.data[i] = e;
      }
    }
    lossVal /= n;
    // backprop
    let grad = this.layers[this.layers.length - 1].backward(dZ, true);
    for (let i = this.layers.length - 2; i >= 0; i--) grad = this.layers[i].backward(grad, false);
    for (let i = 0; i < this.layers.length; i++) this.layers[i].step(lr);
    return lossVal;
  };
  // evaluation loss (no gradient)
  MLP.prototype.evalLoss = function (X, Y) {
    const pred = this.forward(X);
    const n = pred.rows, m = pred.cols;
    let lossVal = 0;
    if (this.loss === 'bce') {
      for (let i = 0; i < n * m; i++) { const p = Math.min(Math.max(pred.data[i], 1e-9), 1 - 1e-9); lossVal += -(Y.data[i] * Math.log(p) + (1 - Y.data[i]) * Math.log(1 - p)); }
    } else if (this.loss === 'ce') {
      for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { if (Y.data[i * m + j] > 0) lossVal += -Y.data[i * m + j] * Math.log(Math.max(pred.data[i * m + j], 1e-9)); }
    } else {
      for (let i = 0; i < n * m; i++) { const e = pred.data[i] - Y.data[i]; lossVal += 0.5 * e * e; }
    }
    return lossVal / n;
  };
  // accuracy for binary (bce, single output) problems
  MLP.prototype.accuracy = function (X, Y) {
    const pred = this.forward(X);
    let correct = 0;
    for (let i = 0; i < pred.rows; i++) {
      const guess = pred.data[i] >= 0.5 ? 1 : 0;
      if (guess === Y.data[i]) correct++;
    }
    return correct / pred.rows;
  };
  // effective linear map: multiply all weight matrices (only meaningful if every
  // layer is linear). Returns {W: (in x out), b: (out)}.
  MLP.prototype.effectiveLinear = function () {
    let W = this.layers[0].W, b = this.layers[0].b.slice();
    for (let i = 1; i < this.layers.length; i++) {
      const Wi = this.layers[i].W, bi = this.layers[i].b;
      // new W = W * Wi ; new b = b * Wi + bi
      W = matmul(W, Wi);
      const nb = new Float64Array(Wi.cols);
      for (let j = 0; j < Wi.cols; j++) {
        let s = bi[j];
        for (let p = 0; p < Wi.rows; p++) s += b[p] * Wi.data[p * Wi.cols + j];
        nb[j] = s;
      }
      b = nb;
    }
    return { W: W, b: b };
  };

  /* ---- Embedding -> softmax next-token model ---------------------------- *
   * Input is a single previous-token id; predicts the next-token distribution.
   * Trains an embedding table E (vocab x dim) plus output weights W (dim x vocab).
   * Same-category tokens share next-token statistics, so their E-rows converge.
   * -------------------------------------------------------------------------*/
  function EmbeddingModel(vocab, dim, seed) {
    this.vocab = vocab; this.dim = dim; this.rand = rng(seed);
    this.E = mat(vocab, dim);
    this.W = mat(dim, vocab);
    this.b = new Float64Array(vocab);
    for (let i = 0; i < this.E.data.length; i++) this.E.data[i] = this.rand.normal(0, 0.4);
    for (let i = 0; i < this.W.data.length; i++) this.W.data[i] = this.rand.normal(0, 1 / Math.sqrt(dim));
  }
  EmbeddingModel.prototype.trainStep = function (pairs, lr) {
    // pairs: array of [prevId, nextId]; one full sweep (batch grad averaged)
    const V = this.vocab, D = this.dim, n = pairs.length;
    const dE = new Float64Array(this.E.data.length);
    const dW = new Float64Array(this.W.data.length);
    const db = new Float64Array(V);
    let loss = 0;
    for (let s = 0; s < n; s++) {
      const prev = pairs[s][0], next = pairs[s][1];
      const e = this.E.data.subarray(prev * D, prev * D + D);
      // logits = e·W + b
      const logits = new Float64Array(V);
      let mx = -Infinity;
      for (let j = 0; j < V; j++) { let z = this.b[j]; for (let k = 0; k < D; k++) z += e[k] * this.W.data[k * V + j]; logits[j] = z; if (z > mx) mx = z; }
      let sum = 0;
      for (let j = 0; j < V; j++) { logits[j] = Math.exp(logits[j] - mx); sum += logits[j]; }
      for (let j = 0; j < V; j++) logits[j] /= sum;         // now probabilities
      loss += -Math.log(Math.max(logits[next], 1e-9));
      // gradients: dlogits = p - onehot(next)
      for (let j = 0; j < V; j++) {
        const dz = logits[j] - (j === next ? 1 : 0);
        db[j] += dz;
        for (let k = 0; k < D; k++) {
          dW[k * V + j] += e[k] * dz;
          dE[prev * D + k] += this.W.data[k * V + j] * dz;
        }
      }
    }
    const inv = lr / n;
    for (let i = 0; i < dW.length; i++) this.W.data[i] -= inv * dW[i];
    for (let j = 0; j < V; j++) this.b[j] -= inv * db[j];
    for (let i = 0; i < dE.length; i++) this.E.data[i] -= inv * dE[i];
    return loss / n;
  };
  EmbeddingModel.prototype.embeddingRows = function () {
    const rows = [];
    for (let i = 0; i < this.vocab; i++) rows.push(Array.from(this.E.data.subarray(i * this.dim, i * this.dim + this.dim)));
    return rows;
  };

  /* ---- helpers: cosine + PCA to 2-D ------------------------------------- */
  function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
  }
  function pca2d(rows) {
    const n = rows.length, d = rows[0].length;
    const mean = new Float64Array(d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += rows[i][j] / n;
    const X = rows.map(function (r) { return r.map(function (v, j) { return v - mean[j]; }); });
    // covariance
    const C = [];
    for (let i = 0; i < d; i++) C.push(new Float64Array(d));
    for (let s = 0; s < n; s++) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += X[s][i] * X[s][j] / n;
    function powerIter(C) {
      let v = new Float64Array(d).map(function () { return Math.random() + 0.1; });
      for (let it = 0; it < 200; it++) {
        const nv = new Float64Array(d);
        for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += C[i][j] * v[j]; nv[i] = s; }
        let norm = 0; for (let i = 0; i < d; i++) norm += nv[i] * nv[i]; norm = Math.sqrt(norm) + 1e-12;
        for (let i = 0; i < d; i++) nv[i] /= norm;
        v = nv;
      }
      return v;
    }
    const v1 = powerIter(C);
    // deflate: C' = C - (v1^T C v1) v1 v1^T
    let lam = 0; for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += C[i][j] * v1[j]; lam += v1[i] * s; }
    const C2 = [];
    for (let i = 0; i < d; i++) { C2.push(new Float64Array(d)); for (let j = 0; j < d; j++) C2[i][j] = C[i][j] - lam * v1[i] * v1[j]; }
    const v2 = powerIter(C2);
    return X.map(function (r) {
      let a = 0, b = 0;
      for (let j = 0; j < d; j++) { a += r[j] * v1[j]; b += r[j] * v2[j]; }
      return [a, b];
    });
  }

  global.NN = {
    rng: rng, mat: mat, matmul: matmul, transpose: transpose,
    MLP: MLP, Dense: Dense, EmbeddingModel: EmbeddingModel,
    softmaxRows: softmaxRows, pca2d: pca2d, cosine: cosine
  };
})(window);
