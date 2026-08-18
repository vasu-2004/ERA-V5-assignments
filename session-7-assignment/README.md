# Morphological segmentation as a training-free prior for the Kronecker codec

**No training. No gradients. No GPU.** This is an encoding-level study: everything is
Kronecker-codec arithmetic and cosine similarity, computed analytically in ~40 seconds.

```bash
pip install -r requirements.txt     # numpy + pytest
python run_experiment.py            # ~40s -> results/results.json + results/report/index.html
python -m pytest tests -q           # 40 tests
```

Open **`results/report/index.html`** for the visual report (charts, an interactive
byte-level explorer for all 54 words, and the full limitations section).

---

## The problem I'm solving

The codec maps `byte -> one-hot(256)`, `position -> one-hot(dp)`, takes their Kronecker
product (`index = byte×dp + position`), sums over the string, and scales by `1/√L`.
With `dp = 32` this has two failure modes on Indic text:

1. **Truncation.** A Devanagari character costs 3 UTF-8 bytes, so `dp=32` holds only
   ~10 characters. Longer compounds lose their tail entirely.
2. **Positional rigidity.** A morpheme is bound to the absolute byte offset it happens
   to occupy. `आलय` alone sits at byte positions 0–8; inside `देवालय` the same morpheme
   sits at 9–17. The codec has no mechanism to notice they are the same thing.

**Hypothesis.** Splitting a compound at morpheme boundaries and encoding each piece from
position 0 should fix both: shorter pieces fit the window, and a morpheme always starts
at position 0 regardless of where it appeared in the compound.

## How I prove it (and how I try to break it)

Four conditions over the same 54 compounds:

| condition | what it does | role |
|---|---|---|
| `raw` | whole word, truncated at 32 bytes | baseline |
| `sandhi` | inverse-sandhi analyzer (training-free) | **the method** |
| `gold` | hand-annotated boundaries | ceiling |
| `midpoint` | cuts every word in half at a meaningless seam | **the control** |

The headline metric is deliberately **not** "is a compound close to its own morphemes" —
that is circular, because a segmented vector *is* the mean of its parts. Instead I score
**pairs of different compounds**: 141 pairs that share a morpheme vs 1300 that share
nothing, by ROC-AUC (rank-based, so it survives the fact that segmentation inflates all
cosines). Neither vector in a pair is built from the other.

Crucially the pairs are split by **where** the shared morpheme sits — initial (which the
rigid codec can already align) vs non-initial (which it cannot). That contrast is the
experiment.

## What I found

| condition | bytes lost | words colliding | AUC initial | **AUC non-initial** | Δ vs raw (95% CI) |
|---|---:|---:|---:|---:|---|
| raw | 4.10% | 5 | 0.860 | **0.665** | — |
| midpoint *(control)* | 0.00% | 0 | 0.791 | **0.687** | +0.022 [−0.017, +0.064] **n.s.** |
| sandhi *(method)* | 0.00% | 0 | 0.963 | **0.983** | **+0.318 [+0.265, +0.374]** ✓ |
| gold *(ceiling)* | 0.00% | 0 | 0.978 | **0.980** | +0.315 [+0.261, +0.371] ✓ |

*95% percentile bootstrap, 2000 resamples, paired over the same word pairs.*

**1. The gain is morphological, not just "shorter pieces."** The midpoint control gets
*identical* truncation relief (0.00% bytes lost, 0 collisions) yet its AUC improvement is
statistically indistinguishable from zero. The morphological split moves it +0.318. So the
gain comes from **where** the cut falls, not from the fact that a cut was made.

**2. The retrieval failure is not a truncation problem.** In the dp sweep, at `dp=48`
nothing is truncated at all (0.00%, zero collisions) — and raw AUC on non-initial
morphemes is *still* 0.656 vs 0.983 segmented. Enlarging the window does not fix
alignment. Truncation and rigidity are two independent defects; segmentation addresses
both, a bigger `dp` only the first.

**3. Real collisions, eliminated.** At `dp=32`, `विश्वविद्यालय` / `विश्वविद्यालयों` /
`विश्वविद्यार्थी` share all 32 surviving bytes and map to **one identical vector** —
indistinguishable to anything downstream. Segmentation: 0 collisions.

**4. An unplanned finding — the UTF-8 script floor.** A test I wrote asserting
`cos(देवालय, आलय) == 0` failed at 0.393. The reason: every Devanagari codepoint is
`E0 A4 xx` / `E0 A5 xx`, so **68.6% of corpus bytes are lead bytes** and any two words of
the script agree in two of every three slots. The consequence is sharper than my original
hypothesis:

```
cos(देवालय, आलय)        = 0.393   <- its own morpheme
cos(देवालय, कमलकमल)     = 0.611   <- a completely unrelated word
Latin control: cos(devalaya, alaya) = 0.000   <- rigidity in its pure form
```

Under raw encoding a word's own morpheme ranks **below** unrelated words. Raw Devanagari
similarity largely measures *script*, not content — which is why every claim here uses
rank-based AUC rather than a cosine threshold.

## The segmenter

I intended to call an off-the-shelf analyzer. Neither was usable:

- **Sanskrit Heritage Engine** — `sanskrit.inria.fr` unreachable from this environment.
- **indic-nlp-library** — installs, but `unsupervised_morph` needs the
  `indic_nlp_resources` Morfessor bundle, whose download returns 403.

So `kron/sandhi.py` implements the analyzer: an inverse vowel-sandhi rule table plus a
126-entry morpheme lexicon (65 of them **distractors** that are not constituents of
anything in the set, so it must discriminate rather than look up). Still training-free —
rules and a word list, no fitted parameters. It recovers **50/54 = 92.6%** of gold
decompositions and lands within 0.003 AUC of the gold ceiling.

The four misses are exactly the unmodelled linguistic classes, not noise:

| word | gold | why |
|---|---|---|
| रामायण | राम + अयन | retroflexion (natva) sandhi न→ण |
| अत्यन्त | अति + अन्त | yaṇ sandhi i+V→य् |
| इत्यादि | इति + आदि | yaṇ sandhi i+V→य् |
| स्वागत | सु + आगत | yaṇ sandhi u+V→व् |

`MorfessorSplitter` ships as a working adapter that activates automatically if the
resource bundle ever becomes available, so "pluggable" is real rather than rhetorical.

## Layout

```
kron/codec.py        the codec — ~20 lines of actual encoding, no parameters
kron/sandhi.py       inverse-sandhi rules + lexicon splitter, + Morfessor/gold/control adapters
kron/dataset.py      54 compounds with gold splits, morpheme families, pair classes
kron/experiments.py  truncation, collisions, retrieval, AUC, bootstrap CIs, dp sweep
kron/report.py       builds the visual report from the results bundle
run_experiment.py    one command, runs everything
tests/               40 tests: codec properties, sandhi rules, dataset sanity, conclusions
results/             generated: results.json + report/index.html
```

The codec's two structural claims are **proved as property tests**, not asserted:
injectivity below `dp` bytes, and "every collision is a truncation collision"
(`tests/test_codec.py`). The tests also assert the *control must fail* — if the
meaningless-seam split ever became significant, the argument would be broken and the
suite says so.

## What this does not show

- **Scale.** 54 curated compounds, one script. An encoding-level probe, not a corpus
  study.
- **Circularity where it exists.** The constituent-retrieval measure in `results.json` is
  favourable to segmented conditions *by construction*. It is reported, and is not the
  headline; the compound-to-compound comparison has no such defect.
- **Cosines are not comparable across conditions.** Segmentation raises all cosines
  (unrelated mean 0.509 → 0.666). Hence AUC everywhere.
- **Truncation relief is arithmetic.** Any split shortens the pieces; the midpoint control
  exists so this trivial part is not mistaken for the interesting part.
- **The analyzer is narrow.** Vowel sandhi only; small lexicon; declines to split
  out-of-lexicon words rather than guessing.
- **No downstream claim.** Nothing here shows that this improves a trained model. That
  needs the training run this study deliberately excludes.

### Natural next steps

**A.** Swap in a real wide-coverage analyzer (Heritage Engine / Morfessor) via the
existing adapter and rerun unchanged — the pipeline is already written for it.
**B.** Scale to a full Indic corpus and measure whether the collision reduction survives
at vocabulary scale, where near-prefix families are far denser than in 54 hand-picked
words.

_Built for ERA V5 (The School of AI), Session 7._
