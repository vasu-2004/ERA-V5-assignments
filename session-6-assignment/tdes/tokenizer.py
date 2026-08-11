"""A small byte-level BPE tokenizer, trained once and then FROZEN.

Byte-level means every possible input encodes -- there is no UNK and no silent
data loss, which matters because the corpus mixes English, Devanagari, Telugu
and source code. Once trained, the tokenizer is serialised and content-hashed;
every shard manifest records that hash, and loading a shard whose manifest
disagrees with the live tokenizer is a hard error. That is what stops the
classic "retokenised half the corpus and never noticed" failure.
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter
from typing import Iterable

from .hashing import hash_obj, sha256_bytes

# GPT-2 style pre-tokenisation: keeps leading spaces attached to words and
# never merges across a word boundary, which keeps merges interpretable.
_PRETOK = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _bytes_to_unicode() -> dict:
    """Reversible byte -> printable-codepoint map (GPT-2 trick), so BPE can work
    on strings while remaining exactly byte-level."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = _bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


class Tokenizer:
    def __init__(self, vocab: dict, merges: list, specials: list):
        self.vocab = vocab                      # token string -> id
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = [tuple(m) for m in merges]
        self.merge_rank = {m: i for i, m in enumerate(self.merges)}
        self.specials = specials
        self.pad_id = vocab["<pad>"]
        self.bos_id = vocab["<bos>"]
        self.eos_id = vocab["<eos>"]
        self.sep_id = vocab["<sep>"]
        self._cache: dict = {}

    # -- training ---------------------------------------------------------
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, specials: tuple) -> "Tokenizer":
        word_freq: Counter = Counter()
        for text in texts:
            for tok in _PRETOK.findall(text):
                encoded = "".join(BYTE_ENCODER[b] for b in tok.encode("utf-8"))
                word_freq[encoded] += 1

        # every word is a tuple of single-character (=byte) symbols to start
        words = {w: tuple(w) for w in word_freq}
        # ALL 256 byte symbols are reserved, not merely the ones observed during
        # training. Otherwise a byte absent from the training corpus (a tab, an
        # unusual control character, an unseen UTF-8 continuation byte) would
        # raise at encode time -- turning "byte level, so nothing can fail" into
        # a crash on the first document that happens to contain it.
        base_symbols = sorted(set(BYTE_ENCODER.values()))

        n_merges = vocab_size - len(specials) - len(base_symbols)
        merges: list = []
        if n_merges > 0:
            pair_freq: Counter = Counter()
            for w, syms in words.items():
                f = word_freq[w]
                for a, b in zip(syms, syms[1:]):
                    pair_freq[(a, b)] += f

            for _ in range(n_merges):
                if not pair_freq:
                    break
                # deterministic tie-break: highest count, then lexicographic
                best = max(pair_freq.items(), key=lambda kv: (kv[1], kv[0]))[0]
                if pair_freq[best] < 2:
                    break
                merges.append(best)
                merged = best[0] + best[1]
                affected = [w for w, syms in words.items()
                            if any(p == best for p in zip(syms, syms[1:]))]
                for w in affected:
                    syms = words[w]
                    f = word_freq[w]
                    for a, b in zip(syms, syms[1:]):     # retract old pair counts
                        pair_freq[(a, b)] -= f
                        if pair_freq[(a, b)] <= 0:
                            del pair_freq[(a, b)]
                    new_syms, i = [], 0
                    while i < len(syms):
                        if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best:
                            new_syms.append(merged)
                            i += 2
                        else:
                            new_syms.append(syms[i])
                            i += 1
                    new_syms = tuple(new_syms)
                    words[w] = new_syms
                    for a, b in zip(new_syms, new_syms[1:]):  # add new pair counts
                        pair_freq[(a, b)] += f

        vocab: dict = {}
        for s in specials:
            vocab[s] = len(vocab)
        for s in base_symbols:
            vocab[s] = len(vocab)
        for a, b in merges:
            tok = a + b
            if tok not in vocab:
                vocab[tok] = len(vocab)
        return cls(vocab, merges, list(specials))

    # -- encoding ---------------------------------------------------------
    def _bpe_word(self, word: str) -> list:
        if word in self._cache:
            return self._cache[word]
        syms = list(word)
        while len(syms) > 1:
            ranked = [(self.merge_rank.get((a, b), None), i)
                      for i, (a, b) in enumerate(zip(syms, syms[1:]))]
            ranked = [(r, i) for r, i in ranked if r is not None]
            if not ranked:
                break
            _, i = min(ranked)
            syms[i:i + 2] = [syms[i] + syms[i + 1]]
        self._cache[word] = syms
        return syms

    def encode(self, text: str) -> list:
        out = []
        for tok in _PRETOK.findall(text):
            encoded = "".join(BYTE_ENCODER[b] for b in tok.encode("utf-8"))
            for sym in self._bpe_word(encoded):
                # byte-level guarantees every symbol decomposes into known units
                if sym in self.vocab:
                    out.append(self.vocab[sym])
                else:
                    out.extend(self.vocab[c] for c in sym)
        return out

    def decode(self, ids: Iterable[int]) -> str:
        pieces = []
        for i in ids:
            tok = self.inv_vocab.get(int(i), "")
            if tok in self.specials:
                continue
            pieces.append(tok)
        text = "".join(pieces)
        try:
            return bytearray(BYTE_DECODER[c] for c in text).decode("utf-8", errors="replace")
        except KeyError:
            return text

    # -- freezing ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {"vocab": self.vocab, "merges": [list(m) for m in self.merges],
                "specials": self.specials}

    @property
    def content_hash(self) -> str:
        """Identity of the tokenizer. Any change to vocab or merge order
        changes this, and therefore invalidates every shard built with it."""
        return hash_obj(self.to_dict())

    def save(self, path: pathlib.Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")), encoding="utf-8")
        return self.content_hash

    @classmethod
    def load(cls, path: pathlib.Path) -> "Tokenizer":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["vocab"], d["merges"], d["specials"])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
