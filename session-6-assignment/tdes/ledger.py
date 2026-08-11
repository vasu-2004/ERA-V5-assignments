"""Append-only, hash-chained ledgers.

Three ledgers make the run auditable:

  consumption  -- what data was fed to the model, in what order, with full
                  provenance (shard, document, token span) per batch
  learning     -- what the model did with it: total / per-lane / per-document
                  loss for each batch, linked by batch_id
  opus         -- every admission decision, its score, threshold and reason

Each record embeds the hash of the previous record, so a checkpoint can pin an
exact ledger prefix by (n_records, head_hash). Truncating to that prefix after a
crash is therefore provably a return to a known state, not a guess.
"""
from __future__ import annotations

import json
import pathlib

from .hashing import GENESIS, canonical_json, chain


class Ledger:
    def __init__(self, path: pathlib.Path, name: str):
        self.path = pathlib.Path(path)
        self.name = name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list = []
        self.head = GENESIS
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
            return
        # Adopt any existing content. A resumed process must continue the SAME
        # chain rather than starting a second one from genesis, otherwise the
        # checkpointed head would never match and recovery could not be verified.
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            w = json.loads(line)
            self.records.append(w)
            self.head = w["hash"]

    # -- writing ----------------------------------------------------------
    def append(self, record: dict) -> dict:
        h = chain(self.head, record)
        wrapped = {"seq": len(self.records), "prev": self.head, "hash": h, "record": record}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(wrapped) + "\n")
        self.records.append(wrapped)
        self.head = h
        return wrapped

    # -- reading ----------------------------------------------------------
    @classmethod
    def read(cls, path: pathlib.Path, name: str) -> "Ledger":
        led = cls.__new__(cls)
        led.path = pathlib.Path(path)
        led.name = name
        led.records = []
        led.head = GENESIS
        if led.path.exists():
            for line in led.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                w = json.loads(line)
                led.records.append(w)
                led.head = w["hash"]
        return led

    def verify_chain(self) -> tuple:
        """Recompute the whole chain. Returns (ok, first_bad_seq)."""
        prev = GENESIS
        for w in self.records:
            if w["prev"] != prev:
                return False, w["seq"]
            if chain(prev, w["record"]) != w["hash"]:
                return False, w["seq"]
            prev = w["hash"]
        return True, None

    # -- crash recovery ---------------------------------------------------
    def offset(self) -> dict:
        """The pin a checkpoint stores."""
        return {"ledger": self.name, "n_records": len(self.records), "head": self.head}

    def truncate_to(self, n_records: int, expected_head: str) -> int:
        """Roll back to a checkpointed prefix, dropping records written after the
        checkpoint but before the crash. Returns how many were discarded."""
        if n_records > len(self.records):
            raise RuntimeError(
                f"{self.name}: cannot truncate to {n_records}; ledger only has {len(self.records)}")
        dropped = len(self.records) - n_records
        self.records = self.records[:n_records]
        self.head = self.records[-1]["hash"] if self.records else GENESIS
        if self.head != expected_head:
            raise RuntimeError(f"{self.name}: head after truncation does not match checkpoint")
        with self.path.open("w", encoding="utf-8") as fh:
            for w in self.records:
                fh.write(canonical_json(w) + "\n")
        return dropped

    # -- convenience ------------------------------------------------------
    def rows(self) -> list:
        return [w["record"] for w in self.records]

    def by_step(self, step: int) -> list:
        return [w["record"] for w in self.records if w["record"].get("step") == step]

    def __len__(self):
        return len(self.records)


class LedgerSet:
    """The three ledgers, checkpointed and recovered as one unit."""

    def __init__(self, ledger_dir: pathlib.Path, suffix: str = ""):
        d = pathlib.Path(ledger_dir)
        sfx = f".{suffix}" if suffix else ""
        self.consumption = Ledger(d / f"consumption{sfx}.jsonl", "consumption")
        self.learning = Ledger(d / f"learning{sfx}.jsonl", "learning")
        self.opus = Ledger(d / f"opus{sfx}.jsonl", "opus")

    def offsets(self) -> dict:
        return {l.name: l.offset() for l in (self.consumption, self.learning, self.opus)}

    def truncate_to(self, offsets: dict) -> dict:
        return {l.name: l.truncate_to(offsets[l.name]["n_records"], offsets[l.name]["head"])
                for l in (self.consumption, self.learning, self.opus)}

    def verify_all(self) -> dict:
        return {l.name: l.verify_chain() for l in (self.consumption, self.learning, self.opus)}
