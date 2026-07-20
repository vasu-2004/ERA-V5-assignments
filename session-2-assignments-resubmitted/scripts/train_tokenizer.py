"""
train_tokenizer.py -- train the shared 10k multilingual BPE using the reference
recipe, then (optionally) search per-language weights to minimize the faithful-
unit fertility spread while keeping every language < 1.2.

Recipe (matches the instructor's reference solution):
  Model:        HuggingFace BPE
  Vocab size:   10,000   (min_frequency = 1)
  Normalizer:   NFKC
  Pretokenizer: Metaspace (marker ▁)   -- keeps Indic characters intact instead
                of exploding them into UTF-8 bytes like ByteLevel does
  Decoder:      Metaspace
  Weights:      per-language corpus repetition (default {en:3, hi:4, te:4, mai:2})

Run AFTER the corpus exists (see build_wiki_faithful_markdown.py):
    python train_tokenizer.py            # trains with default weights
    python train_tokenizer.py --search   # also searches weights for best score
"""
import argparse, itertools, json, pathlib
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from faithful_metric import evaluate

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
LANGS = ["en", "hi", "te", "mai"]
VOCAB = 10000
MARKER = "▁"  # ▁


def load_texts():
    texts = {}
    for l in LANGS:
        p = CORPUS / f"{l}.faithful.txt"
        if not p.exists():
            raise SystemExit(f"missing {p} -- run build_wiki_faithful_markdown.py first "
                             f"and upload the corpus/*.faithful.txt files")
        texts[l] = p.read_text(encoding="utf-8")
    return texts


def train(texts, weights, use_nfkc=True):
    # EXACT reference recipe: BPE(unk="[UNK]"), NFKC, Metaspace(prepend_scheme="never")
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    if use_nfkc:
        tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement=MARKER, prepend_scheme="never")
    tok.decoder = decoders.Metaspace(replacement=MARKER, prepend_scheme="never")
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, min_frequency=1,
                                  show_progress=False, special_tokens=["[UNK]"])
    corpus = []
    for l in LANGS:
        corpus += [texts[l]] * int(weights[l])
    tok.train_from_iterator(corpus, trainer=trainer)
    return tok


def score_weights(texts, weights, use_nfkc=True):
    tok = train(texts, weights, use_nfkc)
    m = evaluate(tok, texts)
    return tok, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--no-nfkc", action="store_true",
                    help="disable NFKC if it breaks faithfulness on the corpus")
    args = ap.parse_args()
    texts = load_texts()
    use_nfkc = not args.no_nfkc

    best_w = {"en": 3, "hi": 4, "te": 4, "mai": 2}
    tok, m = score_weights(texts, best_w, use_nfkc)
    print("default weights", best_w, "->",
          {l: round(m["per_language"][l]["fertility"], 4) for l in LANGS},
          f"spread={m['spread']:.4f} score={m['raw_score']:.1f}",
          f"allUnder1.2={m['all_under_1_2']} faithful={m['all_faithful']}")
    best = m

    if args.search:
        rng = [1, 2, 3, 4, 5, 6]
        for we, wh, wt, wm in itertools.product(rng, rng, rng, rng):
            w = {"en": we, "hi": wh, "te": wt, "mai": wm}
            t, mm = score_weights(texts, w, use_nfkc)
            better = (mm["all_under_1_2"], -mm["spread"]) > (best["all_under_1_2"], -best["spread"])
            if mm["all_faithful"] and better:
                best, best_w, tok = mm, w, t
                print("  new best", w, f"spread={mm['spread']:.4f} score={mm['raw_score']:.1f} "
                      f"allUnder1.2={mm['all_under_1_2']}")

    tok.save(str(ART / "tokenizer.json"))
    out = {"weights": best_w, "recipe": {
        "model": "BPE", "vocab_size": VOCAB, "min_frequency": 1,
        "normalizer": "NFKC" if use_nfkc else "none",
        "pretokenizer": "Metaspace(▁)", "decoder": "Metaspace(▁)"},
        "metrics": best}
    (ART / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsaved artifacts/tokenizer.json + artifacts/metrics.json")
    print(json.dumps({k: out["metrics"][k] for k in
                      ["x_max", "x_max_lang", "x_min", "x_min_lang", "spread",
                       "raw_score", "all_under_1_2", "hindi_penalty",
                       "hindi_adjusted_score", "all_faithful"]},
                     indent=2))


if __name__ == "__main__":
    main()
