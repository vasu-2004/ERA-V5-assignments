"""The Kronecker byte-position codec (paper Section 3.2), implemented exactly.

    byte      -> one-hot(256)
    position  -> one-hot(dp)          dp = 32
    Kronecker -> one-hot(256 * dp) at index (byte * dp + position)
    sum over the L encoded bytes
    scale by 1 / sqrt(L)

No training, no parameters, no gradients. The whole encoder is the function
`encode()` below; everything else in this repository is measurement.

Two properties follow directly from the construction and drive every result in
this study:

1.  Each position index appears at most once in a string, so the vector is
    exactly the set {(byte_i, i)} for i < min(L, dp). For strings of at most dp
    bytes the map is INJECTIVE -- distinct strings cannot collide.

2.  Therefore *every* collision is a truncation collision: two strings collide
    if and only if their first dp bytes agree (and both are >= dp bytes long).
    Under UTF-8 a Devanagari character costs 3 bytes, so dp=32 holds only ~10
    characters, and long compounds that share a stem collide with each other.

Both properties are asserted in tests/test_codec.py rather than assumed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

BYTE_CARD = 256
DEFAULT_DP = 32


def kron_index(byte_val: int, position: int, dp: int = DEFAULT_DP) -> int:
    """Index of the Kronecker product of one-hot(byte) and one-hot(position).

    kron(e_b, e_p) is a one-hot of length 256*dp whose single 1 sits at
    b*dp + p -- so the product never has to be materialised.
    """
    if not 0 <= byte_val < BYTE_CARD:
        raise ValueError(f"byte out of range: {byte_val}")
    if not 0 <= position < dp:
        raise ValueError(f"position out of range for dp={dp}: {position}")
    return byte_val * dp + position


def kron_index_dense(byte_val: int, position: int, dp: int = DEFAULT_DP) -> np.ndarray:
    """The same thing computed the slow, literal way: an actual Kronecker
    product of two one-hot vectors. Used only to prove the shortcut above is
    equivalent (see tests)."""
    b = np.zeros(BYTE_CARD)
    b[byte_val] = 1.0
    p = np.zeros(dp)
    p[position] = 1.0
    return np.kron(b, p)


@dataclass(frozen=True)
class Encoded:
    """A codec vector plus the bookkeeping needed to audit it."""
    vector: np.ndarray
    n_bytes_total: int      # bytes in the input
    n_bytes_kept: int       # bytes that fitted inside dp
    dp: int

    @property
    def n_bytes_lost(self) -> int:
        return self.n_bytes_total - self.n_bytes_kept

    @property
    def truncated(self) -> bool:
        return self.n_bytes_lost > 0


def encode(text: str, dp: int = DEFAULT_DP) -> Encoded:
    """Encode one string. Bytes at positions >= dp are DROPPED (truncation)."""
    raw = text.encode("utf-8")
    kept = raw[:dp]
    L = len(kept)
    v = np.zeros(BYTE_CARD * dp, dtype=np.float64)
    for pos, b in enumerate(kept):
        v[kron_index(b, pos, dp)] += 1.0
    if L > 0:
        v /= np.sqrt(L)                     # the paper's 1/sqrt(L) normalisation
    return Encoded(vector=v, n_bytes_total=len(raw), n_bytes_kept=L, dp=dp)


def encode_segmented(parts: list, dp: int = DEFAULT_DP) -> Encoded:
    """Encode a segmented word: each part is encoded from position 0 on its own
    positional budget, then the part vectors are averaged and re-normalised.

    Two things happen here, and they are worth separating because only one of
    them is interesting:

      * each part gets a fresh 0..dp-1 window, so far fewer bytes are truncated
        -- this is arithmetic, and would be true of ANY split;

      * a morpheme that occurs at a non-zero offset in the compound is
        re-encoded starting at position 0, which is the only way this codec can
        ever align it with that morpheme standing alone -- this requires the
        split to fall on a real morpheme boundary.

    The random-split ablation in the experiments exists precisely to separate
    those two effects.

    Mean (rather than concatenation) keeps the result in the same 256*dp space
    as a single morpheme's vector, so cosines against unsegmented words and
    against individual morphemes remain defined.
    """
    parts = [p for p in parts if p]
    if not parts:
        return encode("", dp)
    acc = np.zeros(BYTE_CARD * dp, dtype=np.float64)
    total, kept = 0, 0
    for p in parts:
        e = encode(p, dp)
        acc += e.vector
        total += e.n_bytes_total
        kept += e.n_bytes_kept
    n = np.linalg.norm(acc)
    if n > 0:
        acc /= n
    return Encoded(vector=acc, n_bytes_total=total, n_bytes_kept=kept, dp=dp)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def fingerprint(v: np.ndarray, decimals: int = 10) -> str:
    """Stable identity of a codec vector, for exact collision counting.

    Values are quantised before hashing so that two mathematically identical
    vectors cannot be separated by float representation noise.
    """
    nz = np.flatnonzero(v)
    payload = ",".join(f"{int(i)}:{round(float(v[i]), decimals)}" for i in nz)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]
