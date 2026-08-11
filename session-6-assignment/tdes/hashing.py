"""Canonical hashing + an append-only hash chain.

Everything the system claims later ("this is the same batch", "this shard was
not modified") reduces to one of these two primitives, so they are kept small
and boring on purpose.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_json(obj: Any) -> str:
    """Stable JSON: sorted keys, no incidental whitespace, numpy-aware."""
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        raise TypeError(f"not JSON serialisable: {type(o)}")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=default)


def hash_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def hash_array(a: np.ndarray) -> str:
    """Hash an array by dtype+shape+bytes, so an int32 and int64 view of the
    same numbers never collide."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


GENESIS = "0" * 64


def chain(prev_hash: str, record: Any) -> str:
    """Tamper-evident chaining: each record's hash commits to all before it."""
    return sha256_bytes((prev_hash + "|" + canonical_json(record)).encode("utf-8"))
