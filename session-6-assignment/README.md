# TDES — a Training Data Execution System for V5

A small but complete data plane for LLM training. It runs the whole path

```
documents → tokenized shards → manifests → mixture schedule → packing → batches
→ training → consumption ledger → learning ledger → checkpoint → crash → resume
→ replay → audit
```

and, more importantly, **proves** each stage rather than asserting it. The model
is deliberately tiny; the point is that the data system is correct, reproducible,
auditable and efficient.

## Run it

```bash
pip install -r requirements.txt      # numpy only (pytest for the test suite)
python run_demo.py                   # ~30s, regenerates submission_artifacts/
python -m pytest tests -q            # 80 tests, ~10s
```

`run_demo.py` needs no arguments and no manual intervention. It wipes and
regenerates `submission_artifacts/` on every run.

## What the demo proves

| Claim | How it is proved (not merely asserted) |
|---|---|
| Tokenizer is frozen | Trained **twice** in one run; both must produce the same content hash. Every shard manifest records that hash and loading refuses a mismatch. |
| Shards are immutable | A byte is flipped in a scratch copy → load raises. Re-writing an existing shard → refused. |
| Eval firewall holds | Two layers: a structural gate that refuses non-`train` splits, plus a canary scan over **every** trainable pack. The canary detector is first given a *positive control* so that "no hits" is meaningful. |
| Packing is correct | Seven structural invariants checked over real packs: causal-only attention, no cross-document attention, no attention into padding, position ids restart per segment, no loss on padding, no loss across a document boundary, labels are exactly the next input token. |
| Mixture is respected | Planned vs actual shares recomputed from the consumption ledger, per curriculum stage. Max error this run: **0.005**. |
| Protected floors work | The Indic lane is the *most rejected* lane by OPUS; the floor overrides those rejections. Each override records the counterfactual (`would_have_been: REJECT`). |
| Crash recovery | A crash-free **reference run** is executed first. The main run crashes at step 27, resumes from the checkpoint at step 20, discards 7 uncommitted ledger records, and the next batch must match the reference's batch 20 **by hash**. All 40 steps then match the reference. |
| Replay | An interval is rebuilt from shards — re-deriving packs, lane order, OPUS decisions and batch assembly — and compared by hash. A test also mutates a consumed token and asserts replay **fails**, so the check is not vacuous. |
| Fork | A *same-config* fork from a checkpoint must reproduce the parent exactly; a *changed-mixture* fork must diverge. Both are asserted. |
| Throughput | Every figure in `performance.json` is a sum of per-step fields in the ledgers, and a test recomputes it from those ledgers. |

## Architecture

```
tdes/
  config.py      one hashable config; lanes, curriculum stages, floors, thresholds
  hashing.py     canonical JSON hashing + the append-only hash chain
  tokenizer.py   byte-level BPE, trained once then frozen and content-hashed
  shards.py      immutable token shards + manifests, verified on every load
  firewall.py    split gate (prevention) + canary scan (detection)
  mixture.py     curriculum stages, exact-rational deficit scheduler, floors
  packing.py     three packing policies; masks, labels, position ids, provenance
  opus.py        online admission control: ACCEPT / REJECT / DEFER / FORCED_ACCEPT
  ledger.py      consumption, learning and OPUS ledgers (hash-chained)
  model.py       small transformer in NumPy with hand-written backward
  checkpoint.py  weights + optimiser + scheduler + cursors + OPUS + ledger offsets
  trainer.py     DataPlane (what to feed) and Trainer (what happens to it)
  replay.py      rebuild a historical interval and compare
  audit.py       re-derive every claim from the artifacts on disk
  perf.py        throughput report, reconstructible from the ledgers
  evidence.py    turns computed audit values into evidence.json / evidence.md
```

`DataPlane` and `Trainer` are separated on purpose: the data plane touches the
model only through a `probe` callback, which is what allows the entire stream to
be replayed later **without a model**.

## Design decisions worth explaining

**NumPy with hand-written gradients, not a framework.** Every headline claim here
is bit-exact ("the next batch is *exactly* this batch"). Single-threaded NumPy
with an explicit operation order gives that; autograd kernel selection and
threaded BLAS reduction order do not. `run_demo.py` pins BLAS to one thread
*before* importing NumPy. The gradients are verified against central finite
differences in `tests/test_model_gradients.py`, so "hand-written" is not
"unchecked".

**Exact rational arithmetic in the scheduler.** Lane selection uses
`fractions.Fraction`, not floats. The lane sequence is part of the replayed byte
stream; float accumulation over many picks is exactly where a reproducibility
guarantee would quietly rot.

**Per-stage mixture counters.** "Stage B is reasoning-heavy" is a claim about the
mixture *during* stage B. With a single cumulative counter, entering a new stage
would spend its first steps repairing the previous stage's history instead of
serving the new plan — and the audit's planned-vs-actual comparison would not
mean what it appears to.

**OPUS scores are relative, not absolute.** A fresh model over a 1024-token vocab
starts near ln(1024) ≈ 6.9 and falls. Any fixed loss threshold either admits
everything early and rejects everything later, or the reverse. OPUS keeps an EMA
baseline and judges `loss / baseline`, which is scale-free and tracks the model
as it improves. (An earlier absolute-threshold version is what surfaced this: it
produced zero DEFERs across an entire run.)

**Checkpoints pin ledger offsets.** A weights-only checkpoint cannot support
exact resume — restoring parameters but re-deriving the data position skips or
repeats batches. Each checkpoint stores the scheduler state, lane cursors, OPUS
state and `(n_records, chain_head)` for all three ledgers. Recovery truncates the
ledgers back to those offsets and refuses to proceed if the chain head disagrees.

**A crash-free reference run.** Proving "resume produced a valid batch" is weak.
The demo runs the whole schedule cleanly first, then crashes a second run and
requires the resumed stream to match the reference **step for step by hash**.

**Ordering rule.** A batch is appended to the ledgers only after its optimiser
step completes, so the ledgers always describe work already reflected in the
weights, and truncating to a checkpoint is provably a return to a consistent
state.

## Packing policies

| Lane | Policy | Rationale | Utilisation |
|---|---|---|---|
| `web_en`, `indic` | `concat_split` | Concatenate and cut on a fixed stride; fragments become their own attention segments. Maximum utilisation. | ~99.9% |
| `code` | `whole_doc_bestfit` | Never interleave fragments of two files in one sequence. First-fit-decreasing bin packing with deterministic tie-breaks; over-length files are cut on sequence boundaries into standalone packs. Costs padding, buys coherent files. | ~95% |
| `math`, `eval_math` | `prompt_masked` | One reasoning trace per sequence, question tokens excluded from the loss. Lowest loss-bearing fraction of the three — and correct, because generating the question is not the task. | ~99% packed, ~20% loss-bearing |

Packs with no loss-bearing token are never emitted: they would consume a slot and
teach nothing. In this corpus that drops 56 of 200 math documents whose question
alone exceeds the 128-token sequence budget — a real, reported number rather than
silent waste.

## The corpus

Small but real (711 documents, ~426k tokens), carved from the sources used in
this course's Session 4 assignment — see `tools/build_corpus.py` for provenance:

| Lane | Split | Source | Licence |
|---|---|---|---|
| `web_en` | train | India Wikipedia (English) | CC BY-SA 3.0 |
| `indic` | train | India Wikipedia (Hindi, Telugu) | CC BY-SA 3.0 |
| `code` | train | 18 permissively-licensed PyPI packages | MIT / BSD / Apache-2.0 |
| `math` | train | GSM8K (`openai/grade-school-math`) | MIT |
| `eval_math` | **eval** | GSM8K test split, canary-marked | MIT |
| `val_web` | **val** | held-out Wikipedia slice, canary-marked | CC BY-SA 3.0 |

## Generated artifacts

```
submission_artifacts/
  run.log            full event sequence + every [PASS]/[FAIL] check
  evidence.json      machine-readable bundle (self-hashed)
  evidence.md        human-readable requirement table
  performance.json   throughput + packing efficiency
  manifests/         one manifest per shard, plus tokenizer.json and index.json
  ledgers/           consumption / learning / opus, hash-chained, per branch
  checkpoints/       per run-id, each pinning model + data-plane + ledger offsets
  shards/            immutable int32 token arrays
```

`[PASS]` lines are only ever emitted by `RunLog.check(name, passed, ...)`, which
takes a boolean the implementation computed — a hardcoded PASS is not expressible
through that API. `evidence.py` contains no verdicts of its own; it evaluates
predicates over values produced by `audit.py`, which re-reads the artifacts from
disk rather than trusting anything in memory.

## Results from the committed run

- 40 steps, 36/36 checks passed, 12/12 requirements PASS
- Packing utilisation **99.5%**, ~5,500 useful loss-bearing tokens/s
- Mixture max |planned − actual| **0.005**; protected floors respected in both stages
- OPUS: 156 ACCEPT / 143 DEFER / 31 REJECT / 4 FORCED_ACCEPT (all 4 genuine floor overrides)
- Loss 6.73 → 6.01; 77 distinct documents with attributed per-document loss
- Crash at step 27 → resumed from 20 → 7 records rolled back → all 40 steps match the reference

## Known limitations

- Replay takes the recorded OPUS *scores* from the ledger. Those are model
  outputs; reproducing them from scratch means re-running training, which is what
  the fork test covers separately. Everything else in replay — shards, packs,
  lane order, decision logic, batch assembly, hashes — is re-derived.
- Determinism is guaranteed for a fixed NumPy build on one machine with BLAS
  pinned to one thread. Cross-platform bit-exactness of float ops is not claimed.
- Global deduplication across shards is out of scope here (it was Session 4's
  subject); this system takes shard contents as given and proves what happens
  downstream of them.

_Built for ERA V5 (The School of AI), Session 6._
