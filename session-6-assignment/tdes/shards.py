"""Immutable tokenized shards + their manifests.

A shard is a flat int32 token array plus an index of the documents inside it.
Once written it is never mutated: the manifest records a SHA-256 of the exact
token bytes and the hash of the tokenizer that produced them, and every load
re-verifies both. A shard that fails verification is refused, not repaired --
silently training on a half-rewritten shard is precisely the class of bug this
system exists to make impossible.
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field

import numpy as np

from .hashing import hash_obj, sha256_bytes

TOKEN_DTYPE = np.int32
MAX_TOKENS_PER_SHARD = 40_000


class ShardIntegrityError(RuntimeError):
    pass


@dataclass
class DocRecord:
    doc_id: str
    source: str
    license: str
    start: int          # inclusive offset into the shard token array
    end: int            # exclusive
    prompt_len: int = 0  # tokens of leading prompt that must NOT bear loss
    structured: bool = False

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "source": self.source, "license": self.license,
                "start": self.start, "end": self.end, "prompt_len": self.prompt_len,
                "structured": self.structured}

    @property
    def n_tokens(self) -> int:
        return self.end - self.start


@dataclass
class Shard:
    shard_id: str
    lane: str
    split: str
    policy: str
    tokenizer_hash: str
    docs: list = field(default_factory=list)
    tokens: np.ndarray = field(default=None, repr=False)

    # -- the immutable identity of this shard ------------------------------
    def content_fields(self) -> dict:
        """Only fields that define the data. Wall-clock time is deliberately
        excluded so two honest rebuilds hash identically."""
        return {
            "shard_id": self.shard_id, "lane": self.lane, "split": self.split,
            "policy": self.policy, "tokenizer_hash": self.tokenizer_hash,
            "n_docs": len(self.docs), "n_tokens": int(self.tokens.size),
            "token_sha256": sha256_bytes(np.ascontiguousarray(self.tokens, TOKEN_DTYPE).tobytes()),
            "docs": [d.to_dict() for d in self.docs],
        }

    @property
    def content_hash(self) -> str:
        return hash_obj(self.content_fields())

    def doc_at(self, pos: int) -> DocRecord | None:
        for d in self.docs:
            if d.start <= pos < d.end:
                return d
        return None


class ShardStore:
    """Writes shards + manifests, and re-verifies them on every load."""

    def __init__(self, root: pathlib.Path, manifest_dir: pathlib.Path):
        self.root = pathlib.Path(root)
        self.manifest_dir = pathlib.Path(manifest_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict = {}

    def shard_path(self, shard_id: str) -> pathlib.Path:
        return self.root / f"{shard_id}.bin"

    def manifest_path(self, shard_id: str) -> pathlib.Path:
        return self.manifest_dir / f"{shard_id}.manifest.json"

    def write(self, shard: Shard) -> dict:
        path = self.shard_path(shard.shard_id)
        if path.exists():
            raise ShardIntegrityError(f"refusing to overwrite immutable shard {shard.shard_id}")
        arr = np.ascontiguousarray(shard.tokens, TOKEN_DTYPE)
        path.write_bytes(arr.tobytes())
        manifest = {
            **shard.content_fields(),
            "content_hash": shard.content_hash,
            "file": path.name,
            "bytes": path.stat().st_size,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.manifest_path(shard.shard_id).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return manifest

    def load(self, shard_id: str, expected_tokenizer_hash: str) -> Shard:
        """Load a shard, verifying token bytes and tokenizer identity."""
        if shard_id in self._cache:
            return self._cache[shard_id]
        mpath = self.manifest_path(shard_id)
        if not mpath.exists():
            raise ShardIntegrityError(f"no manifest for shard {shard_id}")
        m = json.loads(mpath.read_text(encoding="utf-8"))

        if m["tokenizer_hash"] != expected_tokenizer_hash:
            raise ShardIntegrityError(
                f"shard {shard_id} was built with tokenizer {m['tokenizer_hash'][:12]} "
                f"but the live tokenizer is {expected_tokenizer_hash[:12]}")

        raw = self.shard_path(shard_id).read_bytes()
        if sha256_bytes(raw) != m["token_sha256"]:
            raise ShardIntegrityError(f"shard {shard_id} token bytes do not match manifest")

        tokens = np.frombuffer(raw, dtype=TOKEN_DTYPE)
        if tokens.size != m["n_tokens"]:
            raise ShardIntegrityError(f"shard {shard_id} length mismatch")

        shard = Shard(
            shard_id=m["shard_id"], lane=m["lane"], split=m["split"], policy=m["policy"],
            tokenizer_hash=m["tokenizer_hash"],
            docs=[DocRecord(**d) for d in m["docs"]], tokens=tokens,
        )
        if shard.content_hash != m["content_hash"]:
            raise ShardIntegrityError(f"shard {shard_id} manifest is internally inconsistent")
        self._cache[shard_id] = shard
        return shard

    def all_shard_ids(self) -> list:
        return sorted(p.name.replace(".manifest.json", "")
                      for p in self.manifest_dir.glob("*.manifest.json"))


def build_shards(corpus_dir: pathlib.Path, tokenizer, lanes: dict, store: ShardStore,
                 log=None) -> list:
    """Tokenise the corpus lane by lane into immutable shards."""
    manifests = []
    tok_hash = tokenizer.content_hash
    for lane, meta in lanes.items():
        path = pathlib.Path(corpus_dir) / f"{lane}.jsonl"
        if not path.exists():
            continue
        docs = [json.loads(l) for l in path.open(encoding="utf-8")]
        buf: list = []
        recs: list = []
        shard_idx = 0

        def flush():
            nonlocal buf, recs, shard_idx
            if not recs:
                return
            shard = Shard(
                shard_id=f"{lane}-{shard_idx:04d}", lane=lane, split=meta["split"],
                policy=meta["policy"], tokenizer_hash=tok_hash, docs=recs,
                tokens=np.array(buf, dtype=TOKEN_DTYPE),
            )
            m = store.write(shard)
            manifests.append(m)
            if log:
                log.event("shard_created", shard_id=shard.shard_id, lane=lane,
                          split=meta["split"], n_docs=len(recs), n_tokens=len(buf),
                          content_hash=shard.content_hash[:16])
            buf, recs = [], []
            shard_idx += 1

        for d in docs:
            ids = tokenizer.encode(d["text"])
            if not ids:
                continue
            prompt_len = 0
            if d.get("meta", {}).get("prompt"):
                # exact prompt length in tokens, so loss masking lands on the
                # real boundary rather than an estimate
                prompt_len = len(tokenizer.encode(d["meta"]["prompt"]))
                prompt_len = min(prompt_len, max(0, len(ids) - 1))
            if len(buf) + len(ids) > MAX_TOKENS_PER_SHARD and recs:
                flush()
            start = len(buf)
            buf.extend(ids)
            recs.append(DocRecord(
                doc_id=d["doc_id"], source=d["source"], license=d["license"],
                start=start, end=len(buf), prompt_len=prompt_len,
                structured=bool(d.get("meta", {}).get("structured", False)),
            ))
        flush()
    return manifests
