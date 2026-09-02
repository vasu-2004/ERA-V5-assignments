# How attention learned to pay less

**Live app:** _(Netlify link — see submission)_
**This repo:** `session-8-assignment/` in [vasu-2004/ERA-V5-assignments](https://github.com/vasu-2004/ERA-V5-assignments/tree/claude/neural-network-fundamentals-9xgeps/session-8-assignment)

A single-page, self-contained interactive timeline (`index.html` — no build step, no dependencies) covering
every attention mechanism from ERA V5 Session 8, in the order they actually shipped rather than grouped by
family, each with an honestly-written pros/cons/"pick it when" verdict and a source link.

```
session-8-assignment/
  index.html   the app — open directly in a browser, or deploy as-is to Netlify/Vercel
  README.md    this file
```

## Why chronological, not grouped

Grouping these by family (all the positional tricks together, all the sparse tricks together) is how most
references present this material, and it hides the thing that's actually interesting: the field arguing with
its own recent past, in public, and changing its mind about which cost it's willing to pay next. See the
**"What the timeline actually shows"** section in the app (also reproduced under Question 2, below) for what
that argument looks like once the mechanisms are in date order.

## Sources for every date (Question 2 bonus criterion: name it, date it, cite it)

Every date below was checked directly against a primary source page — the arXiv abstract page for the paper
(v1 submission date, not a later revision), a project/model release page, or, for the one mechanism that began
as a community post rather than a paper, the original public post — not pulled from memory. Where the exact
day couldn't be independently re-verified beyond the source's own month, that's stated.

| # | Mechanism | Date | Source |
|---|---|---|---|
| 1 | Standard (scaled dot-product) attention | 12 Jun 2017 | Vaswani et al., *"Attention Is All You Need"*, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| 2 | Sinusoidal positional encoding | 12 Jun 2017 | Same paper, §3.5 |
| 3 | Absolute learned position embeddings | Jun 2018 | Radford et al., *"Improving Language Understanding by Generative Pre-Training"* (GPT-1), [OpenAI technical report](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| 4 | Sparse & top-k attention | 23 Apr 2019 | Child, Gray, Radford, Sutskever, *"Generating Long Sequences with Sparse Transformers"*, [arXiv:1904.10509](https://arxiv.org/abs/1904.10509) |
| 5 | MQA (Multi-Query Attention) | 6 Nov 2019 | Shazeer, *"Fast Transformer Decoding: One Write-Head is All You Need"*, [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) |
| 6 | Sliding window attention | 10 Apr 2020 | Beltagy, Peters, Cohan, *"Longformer: The Long-Document Transformer"*, [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) |
| 7 | Linear attention | 29 Jun 2020 | Katharopoulos, Vyas, Pappas, Fleuret, *"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"*, [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) (ICML 2020) |
| 8 | RoPE (Rotary Position Embedding) | 20 Apr 2021 | Su, Lu, Pan, Murtadha, Wen, Liu, *"RoFormer: Enhanced Transformer with Rotary Position Embedding"*, [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| 9 | ALiBi (Attention with Linear Biases) | 27 Aug 2021 | Press, Smith, Lewis, *"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"*, [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) |
| 10 | **FlashAttention** *(bonus find — not on the instructor's list)* | 27 May 2022 | Dao, Fu, Ermon, Rudra, Ré, *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"*, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) (NeurIPS 2022) |
| 11 | GQA (Grouped-Query Attention) | 22 May 2023 | Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai, *"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"*, [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| 12 | NTK-aware RoPE scaling | 30 Jun 2023 | bloc97, *"NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning…"*, [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/) |
| 13 | YaRN | 31 Aug 2023 | Peng, Quesnelle, Fan, Shippole, *"YaRN: Efficient Context Window Extension of Large Language Models"*, [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) |
| 14 | Attention sinks (StreamingLLM) | 29 Sep 2023 | Xiao, Tian, Chen, Han, Lewis, *"Efficient Streaming Language Models with Attention Sinks"*, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) (ICLR 2024) |
| 15 | MLA (Multi-head Latent Attention) | 7 May 2024 | DeepSeek-AI, *"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"*, [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| 16 | The delta rule (DeltaNet) | 10 Jun 2024 | Yang, Wang, Zhang, Shen, Kim, *"Parallelizing Linear Transformers with the Delta Rule over Sequence Length"*, [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) (NeurIPS 2024) |
| 17 | Gated DeltaNet | 9 Dec 2024 | Yang, Kautz, Gu, *"Gated Delta Networks: Improving Mamba2 with Delta Rule"*, [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) |
| 18 | Compressed & sparse attention, DeepSeek's way (NSA) | 16 Feb 2025 | DeepSeek-AI, *"Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"*, [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) (ACL 2025) |
| 19 | DroPE | 13 Dec 2025 | Gelberg, Eguchi, Akiba, Cetin, *"Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings"*, [arXiv:2512.12167](https://arxiv.org/abs/2512.12167), Sakana AI |

**Two judgment calls made explicit** (also stated in-app, in the Sources & methodology section):

- **"Absolute learned positions"** is dated to GPT-1 (Jun 2018) — the first influential model to make trained
  per-slot position vectors the default — rather than to the original Transformer paper, whose appendix
  mentions testing a learned variant as a footnote to its actual proposal (sinusoidal).
- **"Sliding window attention"** is dated to Longformer (Apr 2020), which named and formalized the mechanism,
  rather than to the 2019 Sparse Transformer, whose strided/fixed patterns are closely related but distinct —
  and are credited separately as this timeline's sparse/top-k entry.
- **DeepSeek Sparse Attention (DSA)**, shipped in DeepSeek-V3.2-Exp (29 Sep 2025), is mentioned inside the NSA
  card as what came next rather than given its own card, since the assignment's "compressed and sparse
  attention as DeepSeek does it" item is one bullet, best matched by NSA specifically (it literally combines a
  *compression* branch with a *sparse selection* branch and a sliding-window branch — DSA drops the compression
  branch in favor of a lighter indexer).

Everything above was checked with live web search against the papers' own arXiv abstract pages, GitHub release
pages, or (for the NTK-aware entry) the original Reddit thread — arXiv's own HTML/PDF fetch was blocked by this
session's network egress policy, so dates were confirmed via search-engine snippets that quote the abstract
page directly (title, arXiv ID, and submission date), cross-checked against at least two independent sources
per entry (e.g. the arXiv page itself, a HuggingFace papers mirror, and/or the NeurIPS/ACL/ICLR proceedings
listing) rather than accepted from a single result.

## Question 2 — what the timeline actually shows

What the timeline shows that a list of the same 18 names never would is a field arguing with its own recent
past, in public, in real time — and the argument keeps flipping which resource it's willing to spend. Laid out
chronologically instead of by family, four separate swings show up:

**2017–2019, wants exactness.** The original Transformer keeps full softmax attention and pays whatever O(n²)
costs — the only fights are over how to tell it where tokens are (sinusoidal, then learned). Nobody is trying
to save compute yet; the architecture itself is the news.

**2019–2021, wants memory and reach back.** Sparse Transformers, MQA, Longformer, and linear attention all land
inside 18 months of each other, and every one of them is the same move: stop looking at, or storing,
everything. This is the field deciding exactness was a luxury it could partially sell off for length and cache
headroom — the cheapest four years on this list, and also the point where quality starts visibly leaking out of
the trade.

**2022, a genuine reversal.** FlashAttention doesn't belong to either camp. It proves the 2019–2021 premise
wrong for compute specifically — you don't have to approximate to go fast, you just have to stop moving data
you don't need to move. It's the one entry on this page that gets something back for free, and it resets what
every later entry has to beat: from 2022 on, an approximate method has to justify itself against an
already-fast *exact* baseline, not against the original 2017 paper.

**2023, wants length again, cheaply.** NTK-aware scaling, YaRN, and attention sinks are all a different kind of
cheap than 2019–2021: they're patches applied to an already-trained model, not architectural bets made at
pretraining time. Notably, the fix that started this wave (NTK-aware) came from a Reddit post before it came
from a lab — the community found the exploit before the literature formalized it, which the dates make visible
in a way a citation list wouldn't.

**2024–2025, wants memory back again, but this time without giving up correctness the way 2019's tools did.**
MLA compresses the KV cache with a jointly-learned latent instead of a fixed sharing rule. DeltaNet and Gated
DeltaNet go back and specifically patch linear attention's original 2020 flaw — that its running state could
only ever *add*, never *correct* — four years after that flaw first shipped. NSA fuses compression, selection,
and windowing into one natively-trained mechanism instead of picking exactly one lever the way 2019 did. The
tools of this era look like the tools of 2019–2021, but each one is visibly built to avoid that era's specific
failure mode.

**And then DroPE, at the very end, reframes the entire positional half of the story.** Every positional
mechanism from sinusoidal through YaRN treats position as something that has to be present at inference because
it was present at training. DroPE's finding — that RoPE is mainly a convergence aid you can remove after the
fact — is the first entry on this page to question that shared assumption rather than build a better version of
it.

**What you can guess from this, which the instructor asked for:** the pattern of the last three years is
mechanisms that get back something an earlier mechanism gave up, without re-paying that mechanism's original
cost — MLA vs. MQA/GQA, Gated DeltaNet vs. linear attention, NSA vs. plain sparse attention. If that pattern
holds, the next entries on this timeline are more likely to be a second-generation fix to a 2023–2024 idea's
specific weak point (a DSA-style successor to NSA that drops its compression branch already shipped in
September 2025, right on schedule) than a wholly new axis of attack. The other legible signal is that the gap
between "posted to a forum" and "in a frontier model's architecture" has been shrinking — NTK-aware scaling
went from Reddit to a formal paper (YaRN) in two months in 2023; whatever the community finds next may arrive
on this timeline faster than the labs do.

**Extra mechanism found, not on the instructor's list: FlashAttention** (Dao, Fu, Ermon, Rudra, Ré,
*"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"*, arXiv:2205.14135, submitted
27 May 2022, NeurIPS 2022) — included above with full pros/cons, marked as a bonus card in the app.

## What's in the app

Each of the 19 cards (18 required + FlashAttention) carries the same structure, honestly, not as marketing:

- **Why then** — what problem existed at that specific moment that made this worth building
- **How it works** — the mechanism, in plain terms
- **What it buys / What it gives up** — a real trade, both sides written down
- **Pick it when** — the concrete situation where this is actually the right call, not "always"
- **Source** — direct link to the paper or original post

A category filter (position / memory-cache / compute-sparsity / exact-fast / baseline) lets you re-slice the
same 19 entries by family if you want to compare that view against the chronological one — which is exactly
how you can see for yourself that the grouped view hides the back-and-forth the dated view reveals.

## Verification

- All dates cross-checked against primary sources as described above.
- The page was rendered headlessly (Chromium via Playwright) during development to confirm all 19 cards render
  in strict chronological order, the category filters and theme toggle work, and there are zero console errors.
- No build step, no external network calls, no CDN dependency — a single `index.html` that works identically
  opened locally, served from GitHub raw, or deployed to Netlify/Vercel with zero configuration.
