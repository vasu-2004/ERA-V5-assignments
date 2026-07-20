# Session 2 (resubmission) — Faithful-Markdown Multilingual BPE

Rebuilt for the instructor's **corrected** evaluation. A single **10,000-token** BPE tokenizer
for the India Wikipedia pages in **English, Hindi, Telugu, Maithili**, trained on the
**wiki-faithful Markdown** corpus and scored with the exact faithful-unit metric.

## Result (faithful-unit metric, all four < 1.2 → no penalty)

| Language | BPE tokens | Faithful units | Fertility |
|---|---:|---:|---:|
| English | 114,363 | 186,367 | 0.6136 |
| Hindi | 51,102 | 88,359 | 0.5783 (min) |
| Telugu | 21,880 | 36,292 | 0.6029 |
| Maithili | 3,703 | 5,808 | 0.6376 (max) |

```
spread = 0.6376 − 0.5783 = 0.05922
raw score = 1000 / 0.05922 = 16,885
Hindi penalty factor = exp(max(0, 0.5783/1.2 − 1)) = 1.0  (all < 1.2)
```

Weights `{en:2, hi:3, te:4, mai:3}`. (The instructor's reference solution scores 6,502 with the
same recipe; a weight search over the faithful corpus tightened the spread.)

## Recipe (exactly the reference's)

- **Model:** HuggingFace `BPE(unk_token="[UNK]")`, vocab 10,000, `min_frequency=1`
- **Normalizer:** NFKC
- **Pre-tokenizer / decoder:** `Metaspace(replacement="▁", prepend_scheme="never")`
  (Metaspace keeps Indic characters intact instead of exploding them into UTF-8 bytes as
  ByteLevel does)
- **Weights:** per-language corpus repetition

## Metric

```
faithful_unit  = one contiguous [\p{L}\p{M}\p{N}]+ run  OR  one visible non-space char [^\s\p{L}\p{M}\p{N}]
fertility(lang) = tokens(lang) / faithful_units(lang)
score           = 1000 / (max_fertility − min_fertility)
```

My `faithful_metric.py` reproduces the instructor's `metrics.json` **bit-for-bit** on their
corpus (validated: en 0.597692, hi 0.579341, te 0.673096, mai 0.733127, score 6502.56).

## Faithfulness

The tokenizer preserves visible text — punctuation, brackets, URL characters, apostrophes,
number separators all survive `decode(encode(text))` (nothing stripped). It merges whole
Markdown artifacts (e.g. `](https://en.wikipedia.org/wiki/India#cite_…`) into single tokens,
which is what drives fertility below 1 — while remaining lossless. (Like the reference, NFKC may
substitute a few compatibility characters; it never drops visible ones.)

## Deliverables

- **`artifacts/tokenizer.json`** — the tokenizer (standard HF format, load with `Tokenizer.from_file`).
- **`artifacts/metrics.json`** — the numbers above.
- **`index.html` / `standalone.html`** — self-contained widget: score, per-language table,
  searchable 10,000-token vocab browser, and a download button (no external requests).
- **`corpus/*.faithful.txt`** — the faithful-Markdown corpus (en/hi/te from the reference build,
  Maithili as the 4th language).
- **`scripts/`** — `build_wiki_faithful_markdown.py` (fetch), `train_tokenizer.py` (recipe +
  weight search), `evaluate_tokenizer.py`, `faithful_metric.py`, `search_weights.py`, `build_widget.py`.

## Reproduce

```bash
cd scripts && pip install tokenizers regex requests markdownify beautifulsoup4 lxml
python build_wiki_faithful_markdown.py   # fetch India pages -> corpus/*.faithful.txt
python train_tokenizer.py --search       # train + tune -> artifacts/tokenizer.json
python evaluate_tokenizer.py             # print fertilities + score
```

## View / host

Open `index.html`, or host the folder / `standalone.html` on Netlify (drag-and-drop).
