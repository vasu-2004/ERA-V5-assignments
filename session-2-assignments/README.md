# Session 2 — Multilingual BPE Tokenizer (India Wikipedia: EN / HI / TE / MR)

**Status: phase 1 (corpus stats + baseline fertility research).** See
[`report.md`](report.md) for the full write-up, or open [`index.html`](index.html) for an
interactive version of the same tables and a live chart.

## What's here

- `data/` — the four source corpora (English/Hindi/Telugu/Marathi "India" Wikipedia page
  text), copied byte-for-byte as supplied, no cleaning.
- `scripts/bpe.py` — a from-scratch byte-level BPE trainer + encoder (no external
  tokenizer libraries), validated against the textbook `low/lower/newest/widest` example.
- `scripts/run_analysis.py` — reproduces every number in `report.md` / `index.html`.
  Run with `cd scripts && python3 run_analysis.py` (standard library only, ~45s).
- `artifacts/` — machine-readable outputs: per-language stats, distinct-character lists,
  the full fertility-checkpoint table, and the trained baseline vocab (`vocab.json`,
  10,256 tokens = 256 base bytes + 10,000 merges) + `merges.txt`.
- `report.md` — the two requested tables (corpus stats; fertility at merge checkpoints
  500/1000/2000/.../10000) plus full methodology notes.
- `index.html` + `css/`, `js/` — an interactive widget presenting the same data.

## Headline result (baseline, equal-weight, undifferentiated BPE)

At a flat 10,000-merge budget shared equally across all four languages with no
per-language tuning, fertility (avg. tokens per unique word) lands at:

| Language | Fertility @ 10,000 merges |
|---|---:|
| Hindi   | 2.1886 |
| Marathi | 2.3310 |
| English | 2.4688 |
| Telugu  | 2.6793 |

None reach the assignment's `X ≤ 1.2` target — expected, since this is the *naive*
control condition. Closing that gap (and minimizing the score-relevant spread across
languages) requires a deliberately unequal per-language merge allocation, which is the
next phase of this work, not attempted here.

## How to view

```bash
cd session-2-assignments
python -m http.server 8000   # then open http://localhost:8000
```
