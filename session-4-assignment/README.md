# Session 4 — Data Cleaning & Deduplication, applied

The session lists **8 cleaning strategies** (confirmed live: a student counted
them and the instructor confirmed "eight sections... this is the minimum
that we have to do"). This folder actually runs all 8 over a real corpus,
rather than describing them.

**Widget:** `widget/index.html` — strategy count, dataset picked, what was
cleaned and why, other findings, final statistics. Host it on Netlify (drag
`widget/index.html` onto app.netlify.com/drop) or open it directly.

## Why the dataset isn't the obvious HF download

This environment has no route to HuggingFace, Wikipedia, Common Crawl,
Kaggle, or Zenodo (verified directly and via Anthropic's own fetch
infrastructure — a real network constraint, not a shortcut). So the corpus
is assembled from every source that *is* reachable:

| Bucket | Real source | License |
|---|---|---|
| Code (4,922 files) | 18 permissively-licensed Python packages via `pip download` (Django, Flask, requests, pydantic, pytest, FastAPI, Black, …) | MIT/BSD/Apache |
| GSM8K reasoning (8,791 rows) | `openai/grade-school-math` on GitHub | MIT |
| India-Wikipedia Indic text (233 paragraphs) | Reused from **this course's own Session 3** tokenizer work (en/hi/te/mai faithful-Markdown) | CC BY-SA 3.0 |
| Constructed demo set (17 of 31 survive) | Hand-built, <0.005% of tokens — near-dup PR articles, PII chat, ghost tags, SEO/list spam, mislabelled-language trap, benchmark canary | n/a, explicitly labelled everywhere |

Total ingested: 15,003 documents / 11,652,905 tokens (inside the assignment's
10–100M range). Final, cleaned: 13,963 documents / 10,727,775 tokens (92.1%
survival — much higher than the session's raw-Common-Crawl example because
the bulk of this corpus was already-curated source data, not raw HTML; the
low-survival cases are exactly the constructed junk built to fail).

## Reproduce

```bash
cd scripts
pip install datasketch py3langid regex beautifulsoup4 lxml
python build_synthetic_demo_bucket.py     # small labelled demo set
# (re-download code/GSM8K into corpus/raw/ — see extract_code_corpus.py
#  and the pip download commands in the session notes)
python extract_code_corpus.py
python ingest.py                          # unify everything -> corpus/docs.jsonl
python pipeline.py                        # run all 8 stages -> artifacts/
```

`artifacts/stats.json` and `artifacts/manifest.json` are committed (small);
the raw/intermediate corpora are `.gitignore`d (too large for git, and
fully reproducible from the scripts above with a hash-stamped manifest).

## What the pipeline found (beyond the happy path)

Six real bugs were found and fixed while running this on real data — script
vs. language confusion (Maithili/Hindi), a quality rule miscalibrated for
math word-problems, bold-markdown mistaken for a bullet list, a
decontamination check that over-flagged shared boilerplate phrasing as
leakage, PII regexes firing on source code, and a fixed shingle size that
under-measured similarity between short, reworded documents. All six are
detailed in the widget's "Other findings" section — that's the actual
substance of this assignment, not just the final numbers.
