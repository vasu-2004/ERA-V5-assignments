"""Evaluation / validation firewall.

Two independent layers, because one is not enough:

  1. Admission gate  -- structural. Any request to route a shard whose split is
     not "train" into the loss-bearing path raises FirewallBlock. This is the
     layer that *prevents* leakage.

  2. Canary audit    -- empirical. Held-out documents carry canary strings whose
     token sequences are pre-computed. Every built batch is scanned for those
     token sequences. This layer *detects* leakage that slipped past layer 1 --
     including leakage via a path nobody thought to gate.

Layer 2 matters precisely because it does not trust layer 1. A firewall you
only assert is not a firewall; this one is tested by deliberately attacking it
during the demo run.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np


class FirewallBlock(RuntimeError):
    """Raised when held-out data is offered to the training path."""


class Firewall:
    def __init__(self, lanes: dict, tokenizer, corpus_dir: pathlib.Path):
        self.lanes = lanes
        self.trainable = {ln for ln, m in lanes.items() if m["split"] == "train"}
        self.blocked_events: list = []
        # canary token sequences, computed with the frozen tokenizer so the
        # comparison happens in token space (the space batches actually live in)
        idx_path = pathlib.Path(corpus_dir) / "corpus_index.json"
        self.canaries: dict = {}
        if idx_path.exists():
            for lane, text in json.loads(idx_path.read_text(encoding="utf-8"))["canaries"].items():
                self.canaries[lane] = np.array(tokenizer.encode(text), dtype=np.int32)

    # -- layer 1 ----------------------------------------------------------
    def assert_trainable(self, shard, purpose: str = "loss_bearing_batch"):
        if shard.split != "train":
            self.blocked_events.append({
                "shard_id": shard.shard_id, "lane": shard.lane, "split": shard.split,
                "purpose": purpose, "reason": "non_train_split_refused_for_loss_bearing_path",
            })
            raise FirewallBlock(
                f"shard {shard.shard_id} has split='{shard.split}' and may not enter {purpose}")
        return True

    def is_trainable_lane(self, lane: str) -> bool:
        return lane in self.trainable

    # -- layer 2 ----------------------------------------------------------
    @staticmethod
    def _contains_subsequence(hay: np.ndarray, needle: np.ndarray) -> bool:
        n, m = hay.size, needle.size
        if m == 0 or n < m:
            return False
        # match on the first token, then verify the full window
        starts = np.flatnonzero(hay[: n - m + 1] == needle[0])
        for s in starts:
            if np.array_equal(hay[s:s + m], needle):
                return True
        return False

    def scan_batch(self, input_ids: np.ndarray) -> list:
        """Return the names of any canaries present in this batch."""
        flat = np.ascontiguousarray(input_ids).reshape(-1)
        return [lane for lane, needle in self.canaries.items()
                if self._contains_subsequence(flat, needle)]

    def audit_batches(self, batches) -> dict:
        hits = []
        for b in batches:
            found = self.scan_batch(b.input_ids)
            if found:
                hits.append({"batch_id": b.batch_id, "canaries": found})
        return {"n_batches_scanned": len(batches), "canary_hits": hits,
                "clean": len(hits) == 0}
