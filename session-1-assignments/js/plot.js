/* ============================================================================
 * plot.js — small canvas plotting helpers (no libraries)
 * Every function is theme-aware: it reads CSS custom properties off the page so
 * plots recolor with the light/dark toggle.
 * ========================================================================== */
(function (global) {
  'use strict';

  // shared, colour-blind-friendly palette
  const COLORS = {
    class0: '#2f6fed',   // blue
    class1: '#f2820a',   // orange
    cat: {               // token categories (S1-3)
      animal: '#2f6fed', fruit: '#e0447f', verb: '#12a594', det: '#8a7bd8', stop: '#9aa4b2'
    }
  };

  function themeInk() {
    const s = getComputedStyle(document.documentElement);
    return {
      ink: s.getPropertyValue('--ink').trim() || '#1a1d24',
      muted: s.getPropertyValue('--muted').trim() || '#6b7280',
      grid: s.getPropertyValue('--grid').trim() || '#e5e7eb',
      panel: s.getPropertyValue('--panel').trim() || '#ffffff'
    };
  }

  // set up a canvas for crisp hi-DPI drawing; returns {ctx,w,h}
  function setup(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width || canvas.width, h = rect.height || canvas.height;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx: ctx, w: w, h: h };
  }

  // data<->pixel mapping over a fixed data window
  function mapper(w, h, xmin, xmax, ymin, ymax, pad) {
    pad = pad == null ? 10 : pad;
    return {
      x: function (v) { return pad + (v - xmin) / (xmax - xmin) * (w - 2 * pad); },
      y: function (v) { return h - pad - (v - ymin) / (ymax - ymin) * (h - 2 * pad); }
    };
  }

  /* ---- decision-boundary heatmap + scatter ------------------------------ *
   * predictFn(x,y) -> probability of class 1. Fills a coarse grid then overlays
   * the data points.
   * -------------------------------------------------------------------------*/
  function boundary(canvas, predictFn, data, win, res) {
    const s = setup(canvas); const ctx = s.ctx;
    res = res || 60;
    const cw = s.w / res, ch = s.h / res;
    for (let i = 0; i < res; i++) {
      for (let j = 0; j < res; j++) {
        const dx = win.xmin + (i + 0.5) / res * (win.xmax - win.xmin);
        const dy = win.ymax - (j + 0.5) / res * (win.ymax - win.ymin);
        const p = predictFn(dx, dy);              // 0..1
        ctx.fillStyle = blend(p);
        ctx.fillRect(i * cw, j * ch, cw + 1, ch + 1);
      }
    }
    // points
    const m = mapper(s.w, s.h, win.xmin, win.xmax, win.ymin, win.ymax, 0);
    for (let k = 0; k < data.X.length; k++) {
      ctx.beginPath();
      ctx.arc(m.x(data.X[k][0]), m.y(data.X[k][1]), 3, 0, 2 * Math.PI);
      ctx.fillStyle = data.y[k] === 0 ? COLORS.class0 : COLORS.class1;
      ctx.globalAlpha = 0.95; ctx.fill(); ctx.globalAlpha = 1;
      ctx.lineWidth = 0.8; ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.stroke();
    }
  }
  // probability -> translucent class colour (blue<->orange, white at 0.5)
  function blend(p) {
    const a = Math.abs(p - 0.5) * 2 * 0.55;        // max ~0.55 alpha
    const c = p >= 0.5 ? '242,130,10' : '47,111,237';
    return 'rgba(' + c + ',' + a.toFixed(3) + ')';
  }

  /* ---- decision boundary from a PRECOMPUTED probability grid ------------ *
   * probs: Float array length res*res, ordered index = j*res + i, where i is the
   * column (x, left->right) and j is the row (y, top->bottom, top = ymax).
   * Much faster than per-cell: caller batches the grid through the model once.
   * -------------------------------------------------------------------------*/
  function decision(canvas, probs, res, data, win) {
    const s = setup(canvas); const ctx = s.ctx;
    const cw = s.w / res, ch = s.h / res;
    for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
      ctx.fillStyle = blend(probs[j * res + i]);
      ctx.fillRect(i * cw, j * ch, cw + 1, ch + 1);
    }
    const m = mapper(s.w, s.h, win.xmin, win.xmax, win.ymin, win.ymax, 0);
    for (let k = 0; k < data.X.length; k++) {
      ctx.beginPath();
      ctx.arc(m.x(data.X[k][0]), m.y(data.X[k][1]), 3, 0, 2 * Math.PI);
      ctx.fillStyle = data.y[k] === 0 ? COLORS.class0 : COLORS.class1;
      ctx.globalAlpha = 0.95; ctx.fill(); ctx.globalAlpha = 1;
      ctx.lineWidth = 0.8; ctx.strokeStyle = 'rgba(255,255,255,0.65)'; ctx.stroke();
    }
  }
  // build the grid points (same ordering as `decision`) for batched forward
  function gridPoints(res, win) {
    const pts = [];
    for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
      pts.push([win.xmin + (i + 0.5) / res * (win.xmax - win.xmin),
                win.ymax - (j + 0.5) / res * (win.ymax - win.ymin)]);
    }
    return pts;
  }

  /* ---- plain scatter (no boundary) -------------------------------------- */
  function scatter(canvas, data, win) {
    const s = setup(canvas); const ctx = s.ctx; const t = themeInk();
    ctx.strokeStyle = t.grid; ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, s.w - 1, s.h - 1);
    const m = mapper(s.w, s.h, win.xmin, win.xmax, win.ymin, win.ymax, 12);
    for (let k = 0; k < data.X.length; k++) {
      ctx.beginPath();
      ctx.arc(m.x(data.X[k][0]), m.y(data.X[k][1]), 3, 0, 2 * Math.PI);
      ctx.fillStyle = data.y[k] === 0 ? COLORS.class0 : COLORS.class1;
      ctx.fill();
    }
  }

  /* ---- loss / metric curves --------------------------------------------- *
   * series = [{data:[..], color, label}]; auto-scales y. yMax optional.
   * -------------------------------------------------------------------------*/
  function curves(canvas, series, opts) {
    opts = opts || {};
    const s = setup(canvas); const ctx = s.ctx; const t = themeInk();
    const pad = 34, padR = 12, padT = 12, padB = 22;
    let ymax = opts.yMax || 0, n = 0;
    series.forEach(function (se) { n = Math.max(n, se.data.length); se.data.forEach(function (v) { if (v > ymax) ymax = v; }); });
    if (!opts.yMax) ymax *= 1.05;
    ymax = ymax || 1; n = Math.max(n, 2);
    const ymin = opts.yMin || 0;
    // axes
    ctx.strokeStyle = t.grid; ctx.lineWidth = 1; ctx.fillStyle = t.muted; ctx.font = '11px system-ui,sans-serif';
    ctx.beginPath(); ctx.moveTo(pad, padT); ctx.lineTo(pad, s.h - padB); ctx.lineTo(s.w - padR, s.h - padB); ctx.stroke();
    for (let g = 0; g <= 4; g++) {
      const yy = padT + g / 4 * (s.h - padT - padB);
      ctx.strokeStyle = t.grid; ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.moveTo(pad, yy); ctx.lineTo(s.w - padR, yy); ctx.stroke(); ctx.globalAlpha = 1;
      const val = ymax - g / 4 * (ymax - ymin);
      ctx.fillText(val.toFixed(2), 3, yy + 3);
    }
    const X = function (i) { return pad + i / (n - 1) * (s.w - pad - padR); };
    const Y = function (v) { return padT + (1 - (v - ymin) / (ymax - ymin)) * (s.h - padT - padB); };
    series.forEach(function (se) {
      if (se.data.length < 2) return;
      ctx.strokeStyle = se.color; ctx.lineWidth = 2; ctx.beginPath();
      se.data.forEach(function (v, i) { const px = X(i), py = Y(Math.max(ymin, Math.min(ymax, v))); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
      ctx.stroke();
    });
    // legend
    let lx = pad + 6, ly = padT + 4;
    series.forEach(function (se) {
      ctx.fillStyle = se.color; ctx.fillRect(lx, ly - 8, 12, 3);
      ctx.fillStyle = t.ink; ctx.fillText(se.label, lx + 16, ly - 3);
      ly += 15;
    });
  }

  /* ---- render a matrix as a coloured number grid (S1-2) ----------------- */
  function matrixGrid(canvas, M, opts) {
    opts = opts || {};
    const s = setup(canvas); const ctx = s.ctx; const t = themeInk();
    const rows = M.rows, cols = M.cols;
    const pad = 4;
    const cw = (s.w - 2 * pad) / cols, ch = (s.h - 2 * pad) / rows;
    let amax = 1e-9;
    for (let i = 0; i < M.data.length; i++) amax = Math.max(amax, Math.abs(M.data[i]));
    ctx.font = Math.max(8, Math.min(13, ch * 0.42)) + 'px ui-monospace,monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const v = M.data[i * cols + j];
      const inten = Math.abs(v) / amax;
      const col = v >= 0 ? '47,111,237' : '242,130,10';
      ctx.fillStyle = 'rgba(' + col + ',' + (0.12 + 0.55 * inten).toFixed(3) + ')';
      ctx.fillRect(pad + j * cw, pad + i * ch, cw - 1, ch - 1);
      ctx.fillStyle = t.ink;
      ctx.fillText(v.toFixed(2), pad + (j + 0.5) * cw, pad + (i + 0.5) * ch);
    }
    ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
  }

  /* ---- embedding scatter with labels (S1-3) ----------------------------- */
  function embedding(canvas, pts, labels, cats) {
    const s = setup(canvas); const ctx = s.ctx; const t = themeInk();
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    pts.forEach(function (p) { xmin = Math.min(xmin, p[0]); xmax = Math.max(xmax, p[0]); ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]); });
    const padd = (xmax - xmin) * 0.12 + 1e-6, pady = (ymax - ymin) * 0.12 + 1e-6;
    const m = mapper(s.w, s.h, xmin - padd, xmax + padd, ymin - pady, ymax + pady, 30);
    ctx.font = '12px system-ui,sans-serif'; ctx.textAlign = 'center';
    pts.forEach(function (p, i) {
      const c = COLORS.cat[cats[i]] || t.muted;
      const px = m.x(p[0]), py = m.y(p[1]);
      ctx.beginPath(); ctx.arc(px, py, 5, 0, 2 * Math.PI); ctx.fillStyle = c; ctx.fill();
      ctx.fillStyle = t.ink; ctx.fillText(labels[i], px, py - 9);
    });
    ctx.textAlign = 'start';
  }

  /* ---- cosine-similarity heatmap (S1-3) --------------------------------- */
  function heatmap(canvas, M, labels) {
    const s = setup(canvas); const ctx = s.ctx; const t = themeInk();
    const n = labels.length;
    const left = 46, top = 46;
    const cw = (s.w - left - 6) / n, ch = (s.h - top - 6) / n;
    ctx.font = '10px system-ui,sans-serif';
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = M[i][j];                    // -1..1
      const pos = (v + 1) / 2;
      ctx.fillStyle = 'rgba(18,165,148,' + (0.08 + 0.85 * pos).toFixed(3) + ')';
      ctx.fillRect(left + j * cw, top + i * ch, cw - 1, ch - 1);
    }
    ctx.fillStyle = t.muted; ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) { ctx.textAlign = 'right'; ctx.fillText(labels[i], left - 4, top + (i + 0.5) * ch); }
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.save();
    for (let j = 0; j < n; j++) { ctx.save(); ctx.translate(left + (j + 0.5) * cw, top - 4); ctx.rotate(-Math.PI / 4); ctx.fillText(labels[j], 0, 0); ctx.restore(); }
    ctx.restore();
    ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
  }

  global.Plot = {
    COLORS: COLORS, setup: setup, boundary: boundary, decision: decision,
    gridPoints: gridPoints, scatter: scatter,
    curves: curves, matrixGrid: matrixGrid, embedding: embedding, heatmap: heatmap
  };
})(window);
