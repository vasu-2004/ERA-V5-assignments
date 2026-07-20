"""Background weight search: maximize 1000/spread s.t. all languages < 1.2,
on the faithful corpus, reference recipe. Saves best tokenizer.json + metrics.json."""
import json, time, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from faithful_metric import evaluate
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORP = ROOT / "corpus"
ART = ROOT / "artifacts"
LANGS = ["en", "hi", "te", "mai"]
NAME = {"en": "English", "hi": "Hindi", "te": "Telugu", "mai": "Maithili"}
texts = {l: (CORP / f"{l}.faithful.txt").read_text(encoding="utf-8") for l in LANGS}
LOG = open(ART / "search_log.txt", "w")


def train(w):
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never")
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="never")
    tr = trainers.BpeTrainer(vocab_size=10000, min_frequency=1, show_progress=False, special_tokens=["[UNK]"])
    corp = []
    for l in LANGS:
        corp += [texts[l]] * w[l]
    tok.train_from_iterator(corp, trainer=tr)
    return tok


def main():
    best = None
    t0 = time.time()
    for en in [2, 3, 4]:
        for hi in [3, 4, 5, 6]:
            for te in [3, 4, 5, 6]:
                for ma in [1, 2, 3]:
                    w = {"en": en, "hi": hi, "te": te, "mai": ma}
                    tok = train(w)
                    m = evaluate(tok, texts)
                    line = (f"w={w} " + " ".join(f"{l}={m['per_language'][l]['fertility']:.4f}" for l in LANGS)
                            + f" spread={m['spread']:.5f} score={m['raw_score']:.0f} under1.2={m['all_under_1_2']}")
                    LOG.write(line + "\n"); LOG.flush()
                    if best is None or (m["all_under_1_2"], -m["spread"]) > (best[0]["all_under_1_2"], -best[0]["spread"]):
                        best = (m, w, tok)
                        tok.save(str(ART / "tokenizer.json"))
                        json.dump({"languages_trained": NAME, "weights": w,
                                   "recipe": {"model": "BPE", "unk_token": "[UNK]", "vocab_size": 10000,
                                              "min_frequency": 1, "normalizer": "NFKC",
                                              "pretokenizer": "Metaspace(prepend_scheme=never)",
                                              "decoder": "Metaspace(prepend_scheme=never)"},
                                   "metrics": m},
                                  open(ART / "metrics.json", "w"), ensure_ascii=False, indent=2)
                        LOG.write(f"  ** new best score={m['raw_score']:.0f}\n"); LOG.flush()
    m, w, tok = best
    LOG.write(f"\nDONE {time.time()-t0:.0f}s BEST {w} score={m['raw_score']:.0f}\n"); LOG.close()


if __name__ == "__main__":
    main()
