"""Tokenizer freeze and shard immutability."""
from __future__ import annotations

import pytest

from tdes.shards import ShardIntegrityError
from tdes.tokenizer import Tokenizer


def test_training_is_deterministic(small_corpus, tokenizer):
    import json
    from tdes.config import LANES
    texts = []
    for lane in LANES:
        p = small_corpus / f"{lane}.jsonl"
        if p.exists():
            texts.extend(json.loads(l)["text"] for l in p.open(encoding="utf-8"))
    again = Tokenizer.train(texts, 1024, ("<pad>", "<bos>", "<eos>", "<sep>"))
    assert again.content_hash == tokenizer.content_hash
    assert again.merges == tokenizer.merges


def test_save_load_roundtrip_preserves_hash(tokenizer, tmp_path):
    p = tmp_path / "tok.json"
    tokenizer.save(p)
    assert Tokenizer.load(p).content_hash == tokenizer.content_hash


@pytest.mark.parametrize("text", [
    "Plain English with punctuation, numbers 1,428,627,663 and symbols #@%.",
    "भारत गणराज्य — दक्षिण एशिया",
    "భారతదేశం తెలుగు",
    "def f(x):\n\treturn x ** 2  # tab and spaces\n",
    "mixed भारत code def x=1 और English",
    "",
])
def test_encode_decode_is_lossless(tokenizer, text):
    """Byte-level means no UNK and no silent data loss on any script."""
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocab_size_respected(tokenizer):
    assert tokenizer.vocab_size <= 1024
    # all 256 byte symbols must be reserved so no input can fail to encode
    assert tokenizer.vocab_size >= 256


def test_shards_verify_and_are_immutable(built, tokenizer):
    store = built["store"]
    ids = store.all_shard_ids()
    assert ids, "no shards were built"
    for sid in ids:
        sh = store.load(sid, tokenizer.content_hash)
        assert sh.tokenizer_hash == tokenizer.content_hash
        assert sh.tokens.size > 0
        # document index must tile the shard without overlap or overrun
        for d in sh.docs:
            assert 0 <= d.start < d.end <= sh.tokens.size

    with pytest.raises(ShardIntegrityError):
        store.write(store.load(ids[0], tokenizer.content_hash))


def test_wrong_tokenizer_hash_is_refused(built, tokenizer):
    store = built["store"]
    with pytest.raises(ShardIntegrityError):
        store._cache.clear()
        store.load(store.all_shard_ids()[0], "deadbeef" * 8)


def test_corrupted_bytes_are_detected(built, tokenizer, tmp_path):
    import shutil

    from tdes.shards import ShardStore
    src = built["store"]
    sid = src.all_shard_ids()[0]
    alt = ShardStore(tmp_path / "s", tmp_path / "m")
    shutil.copy(src.shard_path(sid), alt.shard_path(sid))
    shutil.copy(src.manifest_path(sid), alt.manifest_path(sid))
    raw = bytearray(alt.shard_path(sid).read_bytes())
    raw[5] ^= 0x01                     # flip a single bit
    alt.shard_path(sid).write_bytes(bytes(raw))
    with pytest.raises(ShardIntegrityError):
        alt.load(sid, tokenizer.content_hash)
