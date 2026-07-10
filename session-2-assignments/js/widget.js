(function () {
  'use strict';
  var D = window.SESSION2_DATA;
  var LANGS = [
    { code: 'en', name: 'English', color: '#2f6fed' },
    { code: 'hi', name: 'Hindi', color: '#e0447f' },
    { code: 'te', name: 'Telugu', color: '#12a594' },
    { code: 'mr', name: 'Marathi', color: '#f2820a' }
  ];
  function $(id) { return document.getElementById(id); }
  function fmt(n) { return n.toLocaleString('en-US'); }

  function renderStats() {
    var rows = LANGS.map(function (L) {
      var s = D.stats[L.code];
      return '<tr><td>' + L.name + '</td>' +
        '<td>' + fmt(s.total_characters) + '</td>' +
        '<td>' + fmt(s.total_running_words) + '</td>' +
        '<td>' + fmt(s.total_unique_words_raw) + '</td>' +
        '<td>' + fmt(s.total_unique_words_stripped) + '</td>' +
        '<td>' + fmt(s.num_distinct_unicode_chars) + '</td></tr>';
    }).join('');
    $('stats-table').innerHTML =
      '<tr><th>Language</th><th>Total characters</th><th>Total running words</th>' +
      '<th>Unique words (raw)</th><th>Unique words (stripped)</th><th>Distinct Unicode chars</th></tr>' + rows;
  }

  function renderFertilityTable() {
    var cps = D.checkpoints;
    var rows = cps.map(function (cp, i) {
      var cells = LANGS.map(function (L) {
        return '<td>' + D.fertility_by_language[L.code][cp].fertility.toFixed(4) + '</td>';
      }).join('');
      var isLast = i === cps.length - 1;
      return '<tr' + (isLast ? ' class="hl"' : '') + '><td>' + fmt(cp) + '</td>' + cells + '</tr>';
    }).join('');
    $('fert-table').innerHTML =
      '<tr><th>Merge #</th><th>En</th><th>Hi</th><th>Te</th><th>Mr</th></tr>' + rows;
  }

  function renderChart() {
    var series = LANGS.map(function (L) {
      return { label: L.name, color: L.color, data: D.checkpoints.map(function (cp) { return D.fertility_by_language[L.code][cp].fertility; }) };
    });
    Chart.lineChart($('fert-chart'), D.checkpoints, series);
  }

  function renderSummary() {
    $('summary-merges').textContent = fmt(D.total_merges_trained);
    $('summary-time').textContent = D.training_seconds.toFixed(1) + 's';
    $('summary-vocab').textContent = fmt(D.combined_unique_word_types);
    $('summary-running').textContent = fmt(D.combined_total_running_words);
    var final = D.checkpoints[D.checkpoints.length - 1];
    var vals = LANGS.map(function (L) { return { name: L.name, v: D.fertility_by_language[L.code][final].fertility }; });
    vals.sort(function (a, b) { return a.v - b.v; });
    $('summary-best').textContent = vals[0].name + ' (' + vals[0].v.toFixed(3) + ')';
    $('summary-worst').textContent = vals[vals.length - 1].name + ' (' + vals[vals.length - 1].v.toFixed(3) + ')';
    $('summary-spread').textContent = (vals[vals.length - 1].v - vals[0].v).toFixed(3);
  }

  function init() {
    renderStats();
    renderFertilityTable();
    renderChart();
    renderSummary();
    var themeBtn = $('theme-btn');
    if (themeBtn) themeBtn.addEventListener('click', function () {
      var root = document.documentElement;
      var cur = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', cur);
      themeBtn.textContent = cur === 'dark' ? '☀︎' : '☾';
      renderChart();
    });
    window.addEventListener('resize', renderChart);
  }
  window.addEventListener('DOMContentLoaded', init);
})();
