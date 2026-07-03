# Session 1 — Neural Network Fundamentals (interactive proofs)

Four foundational deep-learning claims, each **proved live in the browser**. Every model
is trained client-side as you watch, on data generated on the spot — no pre-baked plots,
no libraries, no backend. Open the page, change a slider, flip a switch, hit re-run, and the
proof re-derives itself.

## The four claims

| # | Claim | What you see |
|---|-------|--------------|
| **S1-1** | *Activations exist for a reason.* A linear model can only draw a straight boundary, so it can't separate two concentric rings; one ReLU hidden layer can. | Linear+sigmoid stalls at a straight line (~55%); one ReLU layer wraps the ring (~99–100%). **Only the activation changed** — flip the switch to prove it. |
| **S1-2** | *Depth without nonlinearity is a lie.* Five stacked linear layers collapse to a single linear map. | 1-layer and 5-linear boundaries are identical lines at identical accuracy; 5+ReLU solves the ring. The five weight matrices are multiplied numerically into **one** 2×1 matrix. |
| **S1-3** | *Embeddings learn similarity from nothing but next-token.* | A tiny embedding→softmax model trained only to predict the next token in a toy grammar makes same-category tokens cluster in 2-D (PCA), with every token's nearest neighbour same-category. |
| **S1-4** | *Memorization vs generalization — data closes the gap.* | The same over-parameterized net memorizes at n=20 (train→100%, big gap) and generalizes at n=2000 (smooth boundary, tiny gap). The generalization-gap-vs-size curve trends down. |

## How to view

It's a static site — just open `index.html`.

```bash
cd session-1-assignments
python -m http.server 8000   # then visit http://localhost:8000
```

(Opening `index.html` directly via `file://` also works.) Runs offline; hosts as-is on GitHub Pages.

## How it works

Everything is plain vanilla JavaScript on `<canvas>` — no frameworks.

| File | Role |
|------|------|
| `js/nn.js` | A ~30-line autograd-free NN engine: `Dense` layers, ReLU/sigmoid/softmax, BCE/CE/MSE losses, an Adam optimizer, a seeded RNG, and a full-batch `EmbeddingModel`. |
| `js/data.js` | Dataset generators: concentric **rings**, a toy-grammar **next-token** corpus, and a noisy 2-D **blobs** classification with a held-out test split. |
| `js/plot.js` | Canvas helpers: decision-boundary heatmaps, scatter, loss curves, matrix grids, embedding scatter, cosine heatmap. Theme-aware. |
| `js/demos.js` | Wires each section's controls → a `requestAnimationFrame` training loop → the plots. |
| `css/style.css` | Design system with light/dark theme (follows OS preference, plus a manual toggle). |

Every demo takes a seed, so results are reproducible; the **↻ re-train** buttons draw a fresh
seed. Training animates over a few hundred epochs and stops automatically.

## Notes on the design choices

- **S1-3 grammar** uses an explicit `.` end-of-sentence marker. Without it, whichever
  category sat last in every sentence would never be a *previous* token, so its embeddings
  would receive no gradient and never cluster. The `.` gives every category token a
  consistent next-token signature — which is exactly why the clusters emerge cleanly.
- **S1-2 collapse proof** folds the weight matrices of the 5-linear network
  (`W₁·W₂·W₃·W₄·W₅`) into a single 2×1 matrix, demonstrating numerically that any depth of
  linear layers is *exactly* one linear layer.
