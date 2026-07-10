# Session 2 — Multilingual BPE Tokenizer (India: EN / HI / TE / MR)

A single **10,000-token byte-level BPE tokenizer** for the *India* Wikipedia page in English,
Hindi, Telugu and Marathi. **Zero UNK** by construction, **English fertility ≤ 1.2**,
cross-language fertility spread minimized.

- **Live widget:** open [`index.html`](index.html) — it re-tokenizes the corpora *in your
  browser* from the shipped `tokenizer.json` and shows the fertilities and self-score. You can
  paste your own cleaned page text and re-score.
- **Tokenizer:** [`artifacts/tokenizer.json`](artifacts/tokenizer.json) (standard HuggingFace
  format). Load with `Tokenizer.from_file(...)`.
- **Full write-up:** [`report.md`](report.md).

## Result (on the supplied corpora — see report for the honesty caveat)

| Language | Fertility X = tokens / `\w+` words |
|---|---:|
| English | 1.1793 ✓ (≤ 1.2) |
| Hindi | 1.1796 |
| Marathi | 1.1919 |
| Telugu | 1.2014 |

Spread = 0.0222 → **self-score ≈ 45,108**. Weights: `En 7, Hi 1, Te 2, Mr 1`.

## Why it satisfies the rules

- **Zero UNK:** byte-level base (all 256 byte values are tokens) ⇒ every character encodable
  (exact roundtrip verified on all four corpora + a stress string).
- **Handles markup ("secret sauce"):** a `use_regex=False` byte-level pre-tokenizer lets merges
  cross punctuation/spaces, so `India](/wiki/India)` compresses instead of exploding the count —
  solved by design, nothing deleted.
- **Correct metric:** fertility denominator is `len(re.findall(r"\w+", text))`; our JS
  word-counter matches Python's exactly (including that `\w` splits Devanagari/Telugu at vowel
  signs).
- **Exactly 10,000 tokens**, shared across all four languages.

## Reproduce / verify

```bash
cd scripts && pip install tokenizers
python3 optimize.py   # trains + saves artifacts/tokenizer.json
python3 verify.py     # reloads the file, proves zero-UNK, confirms the numbers
```

The in-browser encoder [`js/bpe_encoder.js`](js/bpe_encoder.js) is verified **bit-exact**
against HuggingFace `tokenizers` on all four full corpora.

## View locally

```bash
python -m http.server 8000   # then open http://localhost:8000
```
