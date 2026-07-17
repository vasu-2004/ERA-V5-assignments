# Session 2 (resubmission) — Faithful-Markdown Multilingual BPE

Rebuilt to match the instructor's **corrected** evaluation:

- **Corpus:** wiki-**faithful Markdown** of the India page (links, URLs, tables, references,
  navboxes, categories preserved) — not clipped prose.
- **Metric:** `fertility = token_count / faithful_unit_count`, where a *faithful unit* is one
  contiguous letter/mark/number run **or** one visible non-space punctuation/symbol character.
  `score = 1000 / (max_fertility − min_fertility)`, with an `exp(max(0, X/1.2−1))` penalty that
  is **1.0 when every language is < 1.2** (the goal).
- **Recipe (reference):** HuggingFace BPE, vocab 10,000, `min_frequency=1`, **NFKC** normalizer,
  **Metaspace** pre-tokenizer/decoder (`▁`) — Metaspace keeps Indic characters intact instead of
  spending many tokens on UTF-8 bytes the way ByteLevel does.
- **Faithfulness gate:** `decode(encode(text))` preserves every non-whitespace character
  (no stripping of punctuation / brackets / URL chars / number separators).

## ⚠️ One input needed from you (the sandbox can't reach Wikipedia)

The faithful-unit metric only means anything on the faithful-Markdown corpus, and that corpus
must be fetched from Wikipedia — which this environment blocks. So the corpus has to be generated
on a machine with internet:

```bash
cd session-2-assignments-resubmitted/scripts
pip install requests markdownify beautifulsoup4 lxml regex tokenizers
python build_wiki_faithful_markdown.py      # writes corpus/{en,hi,te,mr}.faithful.md
```

Then upload the four `corpus/*.faithful.md` files. After that, everything else is automatic:

```bash
python train_tokenizer.py --search          # trains + tunes weights, saves artifacts/tokenizer.json
python evaluate_tokenizer.py                # prints the faithful-unit fertilities + score
```

## Files

```
scripts/build_wiki_faithful_markdown.py   fetch + HTML->Markdown (run locally, needs internet)
scripts/faithful_metric.py                 the exact faithful-unit metric + penalty + faithfulness gate
scripts/train_tokenizer.py                 reference recipe (Metaspace/NFKC) + weight search
scripts/evaluate_tokenizer.py              evaluate a tokenizer.json on the corpus
corpus/                                     the faithful-Markdown corpus (added once fetched)
artifacts/tokenizer.json                   the deliverable (added after training)
index.html                                 live widget (added after training)
```

Status: pipeline ready; **waiting on the faithful-Markdown corpus** to train + produce final numbers.
