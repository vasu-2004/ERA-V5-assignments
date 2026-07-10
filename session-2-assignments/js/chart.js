/* Minimal dependency-free line chart for the fertility-vs-merges trend. */
(function (global) {
  'use strict';
  function themeInk() {
    const s = getComputedStyle(document.documentElement);
    return {
      ink: s.getPropertyValue('--ink').trim() || '#1a1d24',
      muted: s.getPropertyValue('--muted').trim() || '#6b7280',
      grid: s.getPropertyValue('--grid').trim() || '#e5e7eb'
    };
  }
  function lineChart(canvas, xs, series) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width || canvas.width, h = rect.height || canvas.height;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const t = themeInk();
    const padL = 42, padR = 14, padT = 14, padB = 26;
    let ymin = Infinity, ymax = -Infinity;
    series.forEach(function (s) { s.data.forEach(function (v) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); }); });
    ymin = Math.floor(ymin * 10) / 10 - 0.1; ymax = Math.ceil(ymax * 10) / 10 + 0.1;
    const X = function (i) { return padL + i / (xs.length - 1) * (w - padL - padR); };
    const Y = function (v) { return padT + (1 - (v - ymin) / (ymax - ymin)) * (h - padT - padB); };
    ctx.strokeStyle = t.grid; ctx.fillStyle = t.muted; ctx.font = '11px system-ui,sans-serif';
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const yy = padT + g / 4 * (h - padT - padB);
      ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke(); ctx.globalAlpha = 1;
      const val = ymax - g / 4 * (ymax - ymin);
      ctx.fillText(val.toFixed(1), 4, yy + 3);
    }
    ctx.strokeStyle = t.grid;
    ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();
    // x labels (sparse)
    ctx.fillStyle = t.muted;
    xs.forEach(function (v, i) {
      if (i === 0 || i === xs.length - 1 || i % 3 === 0) {
        ctx.textAlign = 'center';
        ctx.fillText(String(v), X(i), h - padB + 14);
      }
    });
    ctx.textAlign = 'start';
    // 1.2 target line
    if (1.2 >= ymin && 1.2 <= ymax) {
      ctx.strokeStyle = t.muted; ctx.setLineDash([4, 4]); ctx.globalAlpha = 0.8;
      ctx.beginPath(); ctx.moveTo(padL, Y(1.2)); ctx.lineTo(w - padR, Y(1.2)); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
      ctx.fillText('target X ≤ 1.2', padL + 4, Y(1.2) - 4);
    }
    series.forEach(function (s) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 2.2; ctx.beginPath();
      s.data.forEach(function (v, i) { const px = X(i), py = Y(v); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
      ctx.stroke();
      s.data.forEach(function (v, i) { ctx.beginPath(); ctx.arc(X(i), Y(v), 2.4, 0, 7); ctx.fillStyle = s.color; ctx.fill(); });
    });
    // legend
    let lx = padL + 6, ly = padT + 4;
    series.forEach(function (s) {
      ctx.fillStyle = s.color; ctx.fillRect(lx, ly - 8, 12, 3);
      ctx.fillStyle = t.ink; ctx.fillText(s.label, lx + 16, ly - 3);
      lx += 16 + ctx.measureText(s.label).width + 18;
    });
  }
  global.Chart = { lineChart: lineChart };
})(window);
