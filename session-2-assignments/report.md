# Session 2 — Multilingual BPE Tokenizer: full report

A single **10,000-token byte-level BPE tokenizer** for the *India* Wikipedia page in
**English, Hindi, Telugu, Marathi**, optimized so that **English fertility ≤ 1.2** and the
cross-language fertility **spread is minimized** (the assignment scores on
`1000 / (X_max − X_min)`). The tokenizer has **zero UNK by construction**.

> **Scope / honesty.** All numbers below are computed on the *supplied copy-paste* of each
> India page (in [`data/`](data/)), read with universal newlines (`\r\n → \n`). The grader
> evaluates on their own *cleaned* pages, so the exact spread/score will differ. The
> **method** is what transfers; the interactive widget (`index.html`) re-computes everything
> live and lets anyone paste the real eval text to get the true number.

## Fertility metric (as clarified)

```
X_lang = (total BPE tokens for the whole page text) / len(re.findall(r"\w+", text))
```

i.e. tokens per **running** word (not unique). Score = `1000 / (X_max − X_min)` over the four
languages, subject to the hard constraint **X_english ≤ 1.2**, and **any UNK ⇒ score 0**.

## 1–4. Corpus statistics (metric-correct)

| Language | Total characters | Running words (`\w+`) | Unique words (`\w+`) |
|---|---:|---:|---:|
| English | 196,491 | 31,641 | 6,518 |
| Hindi   | 71,883  | 23,002 | 1,872 |
| Telugu  | 31,858  | 10,490 | 1,063 |
| Marathi | 38,925  | 14,207 | 1,165 |

**Key finding — the `\w+` denominator splits Indic words.** Python's `\w` excludes Devanagari/
Telugu vowel signs and viramas (Unicode categories `Mn`/`Mc`), so `re.findall(r"\w+", …)`
shatters words at every matra: `भारत → ['भ','रत']`, `क्षेत्र → ['क','ष','त','र']`. This roughly
**doubles** the Indic running-word counts (Hindi 12,186 whitespace tokens → 23,002 `\w+`
matches) and materially lowers Indic fertility. Our JS word-counter reproduces Python's counts
exactly, so the widget's denominators match the grader's formula.

## 5. Design & the "secret sauce"

1. **Byte-level base (256 tokens).** Every byte value is a base token ⇒ any character is
   encodable ⇒ **UNK is structurally impossible** (verified: exact roundtrip on all four full
   corpora + a markup/rare-char stress string, 0 UNK).
2. **`use_regex=False` pre-tokenizer (the dominant lever).** BPE merges may cross spaces and
   punctuation, so markup like `India](/wiki/India)` compresses into few tokens instead of
   exploding the count. This is how the "markdown problem" is solved *by tokenizer design*,
   not by deleting characters — encoding remains lossless. This lever, not weighting, is what
   lets English get under 1.2.
   - *Why it's needed:* with a standard GPT-2-style byte-level tokenizer (punctuation split
     off), English floors at **1.50** using the whole 10k vocab, and **1.37** even with
     unlimited vocab — because each comma/paren/period costs a token while `\w+` ignores it.
3. **Weighted training (fine-tuning).** One shared 10k vocab trained on all four languages
   with per-language weights (= corpus repetition = weighted-pair-frequency BPE). A search
   (English-weight sweep + coordinate descent) found **w = (En 7, Hi 1, Te 2, Mr 1)**:
   English up-weighted to pass under 1.2, Telugu up-weighted because it is hardest to compress.

## Result (on the supplied corpus)

| Language | Words | BPE tokens | Fertility X |
|---|---:|---:|---:|
| English | 31,641 | 37,313 | **1.1793** ✓ (≤ 1.2) |
| Hindi   | 23,002 | 27,133 | 1.1796 |
| Telugu  | 10,490 | 12,603 | 1.2014 (X_max) |
| Marathi | 14,207 | 16,934 | 1.1919 |

**X_max − X_min = 1.2014 − 1.1793 = 0.0222 → self-score = 1000/0.0222 ≈ 45,108.**
(Naive baseline — standard GPT-2-style byte-level BPE, equal weights — scores ~2,477 on the
same corpus, so the design + tuning is a ~18× improvement.) The 0.022 spread is razor-thin and
**overfit to this exact corpus**; on the grader's cleaned pages it will differ, which is why
the widget recomputes live.

## Verification (reproducibility is the point — the grader re-runs it)

- **Reload check:** `Tokenizer.from_file("tokenizer.json")` (the grader's exact code path)
  reproduces the table above — numbers are not asserted, they come out of the saved file.
- **Zero-UNK:** all 256 byte-level base chars present in vocab; exact roundtrip on all four
  corpora and a stress string of markup/rare characters.
- **Independent JS encoder:** `js/bpe_encoder.js` re-implements byte-level BPE and is verified
  **bit-exact** against HuggingFace on all four full corpora (en 37,313 · hi 27,133 · te 12,603
  · mr 16,934). The widget uses it to score live in your browser.
- **Exact vocab:** 10,000 tokens (256 base bytes + 9,744 merges).

## Reproduce

```bash
cd session-2-assignments/scripts
pip install tokenizers
python3 optimize.py     # search weights + train + save artifacts/tokenizer.json
python3 verify.py       # reload the saved file, prove zero-UNK, confirm numbers
```

## Files

```
data/                     the 4 India-page corpora, as supplied
scripts/tokenizer_lib.py   byte-level BPE build + fertility helpers (HF tokenizers)
scripts/optimize.py        weight search (line-train) + final whole-doc train
scripts/verify.py          reload tokenizer.json, zero-UNK proof, neighbour check
scripts/make_widget_data.py emits js/opt_data.js from the shipped tokenizer
artifacts/tokenizer.json    the deliverable (HF format, 10,000 tokens, no UNK)
artifacts/best_config.json  winning weights + fertilities + score
js/bpe_encoder.js          in-browser byte-level BPE (bit-exact vs HuggingFace)
js/widget2.js, index.html   the live widget
```
