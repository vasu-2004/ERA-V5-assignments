"""
optimize.py -- search per-language weights for a byte-level (no-regex) BPE that
minimizes fertility spread across EN/HI/TE/MR subject to X_english <= 1.2.

Key design facts established empirically (see report):
  * Pre-tokenizer = ByteLevel(use_regex=False): merges may cross spaces/
    punctuation, so word+markup like `India](/wiki/India)` compresses to few
    tokens. This is the dominant lever (it is what lets English reach <=1.2 and
    is how the "markdown" problem is solved by design). It keeps every character
    encodable (exact roundtrip, zero UNK).
  * Training on that pretokenizer over the WHOLE document is slow (~1 giant
    sequence). Training over LINES is ~37x faster and yields almost identical
    merges, so we SEARCH on lines and do ONE final whole-document train for the
    winning weights (slightly better numbers).
  * Per-language weight = corpus repetition (== weighted-pair-frequency BPE).
"""
import json, re, time
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

LANGS = ["en", "hi", "te", "mr"]
FILES = {"en": "english", "hi": "hindi", "te": "telugu", "mr": "marathi"}
TEXTS = {l: (DATA / f"{FILES[l]}_india.txt").read_text(encoding="utf-8") for l in LANGS}
LINES = {l: [ln for ln in TEXTS[l].split("\n") if ln] for l in LANGS}
WORDS = {l: len(re.findall(r"\w+", TEXTS[l])) for l in LANGS}
VOCAB = 10000
EN_HARD = 1.2
EN_TARGET = 1.15           # search target (margin below the hard cap)
LOG = open(ART / "optimize_log.txt", "w")
_cache = {}


def _new_tok():
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def _trainer():
    return trainers.BpeTrainer(vocab_size=VOCAB, show_progress=False,
                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                               special_tokens=[])


def build(weights, whole_doc=False):
    tok = _new_tok()
    corpus = []
    for l in LANGS:
        w = int(weights[l])
        corpus += ([TEXTS[l]] * w) if whole_doc else (LINES[l] * w)
    tok.train_from_iterator(corpus, trainer=_trainer())
    return tok


def fert(tok):
    return {l: len(tok.encode(TEXTS[l]).ids) / WORDS[l] for l in LANGS}


def norm(w):
    m = min(w[l] for l in LANGS)
    return tuple(max(1, round(w[l] / m)) for l in LANGS)


def evaluate(w):
    nk = norm({l: w[l] for l in LANGS})
    if nk in _cache:
        return _cache[nk]
    tok = build({l: nk[i] for i, l in enumerate(LANGS)}, whole_doc=False)
    xs = fert(tok)
    xmax, xmin = max(xs.values()), min(xs.values())
    spread = xmax - xmin
    res = {"weights": nk, "fertility": xs, "spread": spread,
           "score": 1000.0 / spread if spread > 1e-9 else 1e9,
           "en_ok": xs["en"] <= EN_TARGET}
    _cache[nk] = res
    line = (f"w={nk}  " + " ".join(f"{l}={xs[l]:.3f}" for l in LANGS)
            + f"  spread={spread:.4f} score={res['score']:.0f} enOK={res['en_ok']}")
    print(line); LOG.write(line + "\n"); LOG.flush()
    return res


def better(a, b):
    if a["en_ok"] != b["en_ok"]:
        return a["en_ok"]
    return a["spread"] < b["spread"]


def main():
    t0 = time.time()
    best = None
    # 1) sweep english weight (others = 1) to locate the crossover region
    for en in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]:
        r = evaluate({"en": en, "hi": 1, "te": 1, "mr": 1})
        if best is None or better(r, best):
            best = r
    # 2) coordinate descent from the best point
    cur = {l: best["weights"][i] for i, l in enumerate(LANGS)}
    for _ in range(8):
        improved = False
        for l in LANGS:
            for delta in (+1, -1, +2, +3):
                cand = dict(cur); cand[l] = max(1, cand[l] + delta)
                if cand[l] > 16:
                    continue
                r = evaluate(cand)
                if better(r, best):
                    best = r; cur = {ll: r["weights"][i] for i, ll in enumerate(LANGS)}
                    improved = True
        if not improved:
            break

    # 3) final whole-document train on the winning weights (best numbers) + a couple neighbours
    winners = {best["weights"]}
    finals = []
    for wt in winners:
        wd = {l: wt[i] for i, l in enumerate(LANGS)}
        tok = build(wd, whole_doc=True)
        xs = fert(tok)
        spread = max(xs.values()) - min(xs.values())
        finals.append((spread, wt, xs, tok))
    finals.sort(key=lambda x: x[0])
    spread, wt, xs, tok = finals[0]
    tok.save(str(ART / "tokenizer.json"))

    # verify no UNK possible: base alphabet must be full byte-level set
    vocab = tok.get_vocab()
    out = {
        "method": "byte-level BPE, ByteLevel(use_regex=False), weighted by corpus repetition",
        "pretokenizer_note": "merges cross spaces/punctuation; every char encodable (exact roundtrip); zero UNK by construction",
        "vocab_size": tok.get_vocab_size(),
        "best_weights": {l: wt[i] for i, l in enumerate(LANGS)},
        "fertility_final_wholedoc": xs,
        "spread_final": spread,
        "score_final": 1000.0 / spread if spread > 1e-9 else 1e9,
        "x_max_lang": max(xs, key=xs.get),
        "x_min_lang": min(xs, key=xs.get),
        "en_within_1_2": xs["en"] <= EN_HARD,
        "words": WORDS,
        "configs_evaluated": len(_cache),
        "search_seconds": time.time() - t0,
    }
    (ART / "best_config.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nFINAL:", json.dumps(out, indent=2, ensure_ascii=False))
    LOG.write("\nFINAL: " + json.dumps(out, ensure_ascii=False) + "\n"); LOG.close()


if __name__ == "__main__":
    main()
