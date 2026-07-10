# Session 2 — Corpus Stats & Baseline BPE Fertility Report

Source: the "India" Wikipedia article, as supplied (raw copy/paste, including page
navigation/reference chrome — **no content was cleaned or edited**), in four languages:
English, Hindi, Telugu, Marathi. Files live in [`data/`](data/).

Line endings were normalized `\r\n → \n` on read (standard universal-newline handling,
not a content change). Every number below is reproducible by running
[`scripts/run_analysis.py`](scripts/run_analysis.py).

## Methodology (so every number is auditable)

- **Total characters** — `len(text)`, i.e. every Unicode code point in the file (letters,
  punctuation, digits, whitespace, newlines — everything).
- **Total running words** — `text.split()`: split on any Unicode whitespace. Cross-checked
  independently against `wc -w` under a correct UTF-8 locale (`LC_ALL=C.UTF-8`) — this
  environment's default locale is `POSIX`/`C`, under which `wc -w` silently miscounts
  multi-byte UTF-8 text (it gave 3,509 for Hindi instead of the correct 12,186); forcing
  a UTF-8 locale makes `wc -w` match our Python count exactly for all four files. This is
  flagged explicitly because it's an easy way to get "actual figures" quietly wrong.
- **Total unique words** — reported two ways: **raw** (distinct whitespace tokens, as-is)
  and **stripped** (same, with leading/trailing Unicode punctuation removed from each
  token — e.g. trailing commas/periods/danda `।` don't create a spurious extra "word").
- **Number of Unicode characters** — count of *distinct* Unicode code points appearing in
  the file (`len(set(text))`) — i.e. the size of the character alphabet actually used
  (full lists in [`artifacts/distinct_unicode_chars.json`](artifacts/distinct_unicode_chars.json)).

## 1–4. Per-language corpus statistics

| Language | Total characters | Total running words | Total unique words (raw) | Total unique words (punct-stripped) | Distinct Unicode characters |
|---|---:|---:|---:|---:|---:|
| English | 196,491 | 29,348 | 9,454 | 7,660 | 141 |
| Hindi   | 71,883  | 12,186 | 4,017 | 3,504 | 171 |
| Telugu  | 31,858  | 4,082  | 2,458 | 2,245 | 155 |
| Marathi | 38,925  | 5,490  | 2,890 | 2,725 | 162 |

Full machine-readable version: [`artifacts/corpus_stats.json`](artifacts/corpus_stats.json).

## 5. Baseline BPE fertility (single tokenizer, equal weights, checkpointed every 500+ merges)

**Setup:** one byte-level BPE tokenizer trained on the **combined** corpus — all four
languages' word-frequency counts summed directly with no per-language re-weighting
("equal weights" = each language's real occurrence counts contribute as-is; the training
corpus is 51,106 total running words / 17,352 unique word types). Standard algorithm:
base alphabet = raw UTF-8 bytes (256 tokens), then greedily merge the most frequent
adjacent byte-pair, 10,000 times. **Trainer verified** against the textbook
`low/lower/newest/widest` BPE example (reproduces the exact canonical merge order:
`es → est → lo → low → ew → new → newest → dest → idest → widest`) before running on
real data — see [`scripts/bpe.py`](scripts/bpe.py).

**Fertility** = average BPE tokens per unique word, computed by tokenizing each
language's own unique (punctuation-stripped) word list and averaging — i.e. `(Σ tokens
across that language's unique words) / (count of unique words)`. This is the metric the
full assignment's `X = tokens/vocab ≤ 1.2` target refers to (see note below).

| Merge # | En | Hi | Te | Mr |
|---:|---:|---:|---:|---:|
| 100   | 6.2294 | 6.2038 | 8.6049 | 6.9442 |
| 250   | 5.3636 | 5.2737 | 7.1301 | 5.8415 |
| 500   | 4.8333 | 4.6356 | 5.9412 | 5.0932 |
| 1,000  | 4.2685 | 4.0140 | 5.2303 | 4.3479 |
| 1,500  | 3.9572 | 3.6841 | 4.7590 | 3.9266 |
| 2,000  | 3.7262 | 3.4652 | 4.4276 | 3.6738 |
| 3,000  | 3.4102 | 3.1276 | 3.9768 | 3.3031 |
| 4,000  | 3.1979 | 2.8947 | 3.6588 | 3.0884 |
| 5,000  | 3.0402 | 2.7203 | 3.4018 | 2.8771 |
| 6,000  | 2.8952 | 2.5939 | 3.2209 | 2.7402 |
| 7,500  | 2.6987 | 2.4235 | 2.9938 | 2.5666 |
| 8,000  | 2.6510 | 2.3587 | 2.9185 | 2.4778 |
| 9,000  | 2.5693 | 2.2554 | 2.7644 | 2.3758 |
| **10,000** | **2.4688** | **2.1886** | **2.6793** | **2.3310** |

Full machine-readable version (includes total-token and unique-word counts behind every
cell): [`artifacts/fertility_checkpoints.json`](artifacts/fertility_checkpoints.json).

Training took 43.2s for the full 10,000 merges on this machine; it had not exhausted the
corpus (the least-frequent merged pair at step 10,000 still occurred twice), so more
merges would continue to reduce fertility further.

### Reading the table honestly

At a flat, undifferentiated 10,000-merge budget shared equally across all four languages,
**none reach the assignment's `X ≤ 1.2` target** — fertility is still 2.19–2.68 tokens/word
at 10,000 merges. Two things stand out and both matter for the next phase of this project:

1. **Telugu is consistently the hardest to compress** (highest fertility at every
   checkpoint) — expected, given its complex conjunct-consonant orthography and smaller
   corpus (fewer repeated word forms to learn from).
2. **Hindi consistently compresses *better* than English** despite a much smaller corpus —
   its word forms apparently share more high-frequency internal substrings under this
   equal-weight joint training.

This "naive, undifferentiated" baseline is expected to fall well short of `X ≤ 1.2` — it's
the control condition. Reaching the target (and minimizing the `X_max − X_min` spread the
assignment scores on) requires *deliberately unequal* per-language merge budgets, biased
toward whichever language needs more help — a per-language allocation-equalization step
planned as the next phase of this work, not attempted in this baseline run.

### Note on the ratio's denominator ("Total Vocab")

The assignment defines `X = tokens / vocab, target ≤ 1.2`. Read literally as *tokens over
the whole running article* divided by unique-word count, that ratio is structurally
incapable of reaching ≤1.2 for real article-length text (Heaps' law: running-word count is
normally 3–5× unique-word count, independent of tokenizer quality — e.g. English here is
29,348 running / 7,660 unique ≈ 3.83, before any tokenization at all). The only reading
under which "≤1.2" is a meaningful, achievable design target is the one used above:
average tokens **per unique word type** (fertility) — this is what's reported throughout.

## Raw (non-punctuation-stripped) fertility, for transparency

At the full 10,000-merge checkpoint, fertility computed against each language's **raw**
(unstripped) unique-word list instead:

| Language | Fertility (raw words) | Total tokens | Unique words |
|---|---:|---:|---:|
| English | 2.4273 | 22,948 | 9,454 |
| Hindi   | 2.2245 | 8,936  | 4,017 |
| Telugu  | 2.6697 | 6,562  | 2,458 |
| Marathi | 2.3543 | 6,804  | 2,890 |

Nearly identical to the punctuation-stripped numbers above (as expected — trailing
punctuation affects only a minority of tokens).

## Deliverables in this folder

```
session-2-assignments/
  data/                        the 4 raw corpus files, byte-for-byte as supplied
  scripts/bpe.py                from-scratch byte-level BPE trainer + encoder (validated against the textbook example)
  scripts/run_analysis.py       reproduces every number in this report
  artifacts/corpus_stats.json          per-language stats (machine-readable)
  artifacts/distinct_unicode_chars.json  full distinct-character lists per language
  artifacts/fertility_checkpoints.json   full checkpoint table + raw-word fertility (machine-readable)
  artifacts/vocab.json                  the trained 10,256-token vocabulary (256 base bytes + 10,000 merges)
  artifacts/merges.txt                   the ordered merge rules (standard format: "a b new_id" per line)
  report.md                     this file
```

To reproduce from scratch: `cd scripts && python3 run_analysis.py` (no dependencies beyond
the Python standard library; ~45 seconds).
