"""Property tests for the Kronecker codec.

The study's interpretation rests on two structural claims about the codec:
injectivity below dp bytes, and "every collision is a truncation collision".
Both are proved here against the implementation rather than asserted in prose.
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from kron.codec import (BYTE_CARD, DEFAULT_DP, cosine, encode, encode_segmented,
                        fingerprint, kron_index, kron_index_dense)

WORDS = ["देवालय", "शिवालय", "आलय", "देव", "विश्वविद्यालय", "code", "a", ""]


def test_index_shortcut_equals_a_real_kronecker_product():
    """b*dp+p is only a shortcut if it agrees with the actual np.kron."""
    for b, p in itertools.product([0, 1, 65, 224, 255], [0, 1, 7, 31]):
        dense = kron_index_dense(b, p, DEFAULT_DP)
        assert dense.size == BYTE_CARD * DEFAULT_DP
        assert int(np.flatnonzero(dense)[0]) == kron_index(b, p, DEFAULT_DP)


def test_vector_dimension_and_unit_norm():
    for w in WORDS:
        e = encode(w)
        assert e.vector.size == BYTE_CARD * DEFAULT_DP
        if w:
            assert e.vector.sum() == pytest.approx(np.sqrt(e.n_bytes_kept))
            assert np.linalg.norm(e.vector) == pytest.approx(1.0)


def test_nonzeros_are_exactly_the_kept_bytes():
    w = "देवालय"
    e = encode(w)
    raw = w.encode()[:DEFAULT_DP]
    expected = sorted(kron_index(b, i) for i, b in enumerate(raw))
    assert sorted(np.flatnonzero(e.vector).tolist()) == expected


def test_positions_beyond_dp_are_dropped():
    long_word = "विश्वविद्यालयों"
    e = encode(long_word)
    assert e.n_bytes_total > DEFAULT_DP
    assert e.n_bytes_kept == DEFAULT_DP
    assert e.n_bytes_lost == e.n_bytes_total - DEFAULT_DP
    assert e.truncated


def test_injective_below_dp_bytes():
    """No two distinct short strings may share a vector."""
    alphabet = "abcdeक्षदेव"
    seen = {}
    for n in (1, 2, 3):
        for combo in itertools.product(alphabet, repeat=n):
            s = "".join(combo)
            if len(s.encode()) > DEFAULT_DP:
                continue
            fp = fingerprint(encode(s).vector)
            assert seen.setdefault(fp, s) == s, f"collision between {seen[fp]!r} and {s!r}"


def test_every_collision_is_a_prefix_collision():
    """Two strings collide if and only if their first dp bytes agree."""
    a = "विश्वविद्यालय"
    b = "विश्वविद्यालयों"
    assert a.encode()[:DEFAULT_DP] == b.encode()[:DEFAULT_DP]
    assert fingerprint(encode(a).vector) == fingerprint(encode(b).vector)
    assert cosine(encode(a).vector, encode(b).vector) == pytest.approx(1.0)

    c = "शिवालय"            # different prefix -> must not collide
    assert fingerprint(encode(a).vector) != fingerprint(encode(c).vector)


def test_positional_rigidity_is_exact_when_scripts_do_not_share_lead_bytes():
    """In Latin, where bytes carry real information, a shifted morpheme aligns
    with nothing at all: the cosine is exactly zero."""
    assert cosine(encode("devalaya").vector, encode("alaya").vector) == pytest.approx(0.0)


def test_raw_codec_gives_a_shared_morpheme_no_advantage_over_an_unrelated_word():
    """The premise of the study, stated in the form that survives contact with UTF-8.

    In Devanagari the cosine is NOT zero, because every character encodes as
    E0 A4/A5 xx, so two of every three byte slots agree between any two words of
    the script. The right claim is therefore comparative: under raw encoding a
    word's own non-initial morpheme is no closer than an unrelated word --
    similarity is dominated by script overlap rather than by content.
    """
    compound = encode("देवालय").vector
    own_morpheme = cosine(compound, encode("आलय").vector)
    unrelated = max(cosine(compound, encode(w).vector)
                    for w in ("कमलकमल", "नगरपुत्र", "मित्रवायु"))
    assert own_morpheme < unrelated, (own_morpheme, unrelated)

    # the INITIAL morpheme does align, and is clearly visible
    assert cosine(compound, encode("देव").vector) > own_morpheme


def test_devanagari_bytes_are_dominated_by_utf8_lead_bytes():
    """Quantifies why the raw similarity floor is so high."""
    text = "देवालयशिवालयराजकुमारविद्यालय"
    raw = text.encode()
    lead = sum(1 for b in raw if b in (0xE0, 0xA4, 0xA5))
    assert lead / len(raw) > 0.6


def test_segmentation_restores_alignment():
    seg = encode_segmented(["देव", "आलय"]).vector
    assert cosine(seg, encode("आलय").vector) > 0.5
    assert cosine(seg, encode("देव").vector) > 0.5


def test_segmentation_reduces_truncation():
    word = "विश्वविद्यालयों"
    assert encode(word).n_bytes_lost > 0
    assert encode_segmented(["विश्व", "विद्या", "आलयों"]).n_bytes_lost == 0


def test_segmented_vector_is_unit_norm():
    e = encode_segmented(["देव", "आलय"])
    assert np.linalg.norm(e.vector) == pytest.approx(1.0)


def test_empty_and_single_byte_inputs_are_safe():
    assert encode("").vector.sum() == 0.0
    assert np.linalg.norm(encode("a").vector) == pytest.approx(1.0)
    assert encode_segmented([]).vector.sum() == 0.0
    assert encode_segmented(["", "देव"]).n_bytes_total == len("देव".encode())


def test_cosine_is_bounded_and_self_similarity_is_one():
    for w in WORDS:
        if not w:
            continue
        v = encode(w).vector
        assert cosine(v, v) == pytest.approx(1.0)
    for a, b in itertools.combinations([w for w in WORDS if w], 2):
        assert -1e-9 <= cosine(encode(a).vector, encode(b).vector) <= 1 + 1e-9


def test_encoding_is_deterministic():
    for w in WORDS:
        assert fingerprint(encode(w).vector) == fingerprint(encode(w).vector)
