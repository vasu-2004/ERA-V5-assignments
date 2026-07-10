"""
tokenizer_lib.py -- shared helpers for the multilingual BPE optimization.

Uses HuggingFace `tokenizers` so the artifact we ship (`tokenizer.json`) is the
exact object the instructor will load and run -- our self-score is therefore
computed by the same code path that will grade us.

Design choices that satisfy the assignment's hard rules:
  - Byte-level BPE (models.BPE + ByteLevel pre-tokenizer/decoder, initial
    alphabet = all 256 byte-level chars). This makes UNK structurally
    impossible: every input byte is already a base token, so any character in
    any evaluation page is encodable -> zero-UNK guarantee.
  - vocab_size = 10000 EXACT, shared across all four languages.
  - Per-language weighting is done by corpus repetition (feeding a language's
    text w_lang times to the trainer), which is mathematically identical to the
    weighted-pair-frequency objective  score(pair)=sum_lang w_lang * f_lang(pair)
    described in the assignment KT.

Fertility uses the instructor's stated definition exactly:
    X = (total BPE tokens for the whole page text) / (len(re.findall(r"\\w+", text)))
"""
import re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

WORD_RE = re.compile(r"\w+", re.UNICODE)


def count_words(text):
    """Instructor's denominator: number of \\w+ regex matches (Unicode-aware)."""
    return len(WORD_RE.findall(text))


def build_tokenizer(texts_by_lang, weights, vocab_size=10000, min_frequency=1):
    """
    texts_by_lang: {lang_code: full_text_string}
    weights:       {lang_code: int repetition weight (>=1)}
    Returns a trained HF Tokenizer (byte-level BPE, no UNK, exact vocab_size).
    """
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 256 byte chars -> no UNK ever
        special_tokens=[],
    )
    corpus = []
    for lang, text in texts_by_lang.items():
        w = int(weights.get(lang, 1))
        for _ in range(max(1, w)):
            corpus.append(text)
    tok.train_from_iterator(corpus, trainer=trainer)
    return tok


def fertility(tok, texts_by_lang):
    """Return {lang: {'tokens':int,'words':int,'fertility':float}} using the
    instructor's metric: total encoded tokens / count of \\w+ words."""
    out = {}
    # encode_batch for speed
    langs = list(texts_by_lang.keys())
    encs = tok.encode_batch([texts_by_lang[l] for l in langs])
    for lang, enc in zip(langs, encs):
        text = texts_by_lang[lang]
        tokens = len(enc.ids)
        words = count_words(text)
        out[lang] = {
            "tokens": tokens,
            "words": words,
            "fertility": tokens / words if words else 0.0,
        }
    return out


def score_from_fertility(fert):
    """Assignment score = 1000 / (X_max - X_min) across the four languages."""
    xs = [v["fertility"] for v in fert.values()]
    xmax, xmin = max(xs), min(xs)
    spread = xmax - xmin
    return {
        "x_max": xmax,
        "x_min": xmin,
        "spread": spread,
        "score": (1000.0 / spread) if spread > 1e-12 else float("inf"),
    }
