"""Replay: rebuild a historical interval of the data stream and prove it matches.

What makes this a real proof rather than a tautology matters, so it is worth
being explicit about what is *reconstructed* versus what is *read*:

  reconstructed from source   token shards (re-verified against their manifests),
                              every pack (re-derived by the packing code), the
                              lane order (re-derived by the mixture scheduler),
                              every OPUS decision (re-derived by the OPUS rules),
                              batch assembly and batch hashes

  read from the historical    only the raw OPUS *scores* -- these are outputs of
  record                      the model at that moment, and reproducing them
                              would require re-running training, which is what
                              the fork test covers separately

So replay re-executes the entire data plane and compares the result against the
consumption ledger. If the packer, the scheduler, the OPUS thresholds, the shard
bytes or the token spans had changed, the hashes would not match.

A stricter cross-check runs alongside: the replayed scan order must request the
same pack ids, in the same order, as the original run. A divergence there is
reported even if the final hashes happened to agree.
"""
from __future__ import annotations

from .trainer import DataPlane


class RecordedProbe:
    """Feeds the OPUS scores recorded for a step, in their original order.

    Also asserts that replay asks about the same packs in the same order; a
    mismatch means the reconstructed scan diverged from history.
    """

    def __init__(self, decisions_for_step: list):
        self.queue = [d for d in decisions_for_step
                      if d.get("reason") != "scan_budget_exhausted_progress_guarantee"]
        self.i = 0
        self.mismatches: list = []

    def __call__(self, pack):
        if self.i >= len(self.queue):
            self.mismatches.append({"pack_id": pack.pack_id, "issue": "more_probes_than_recorded"})
            return 1e9
        rec = self.queue[self.i]
        self.i += 1
        if rec["pack_id"] != pack.pack_id:
            self.mismatches.append({"expected": rec["pack_id"], "got": pack.pack_id,
                                    "probe_index": self.i - 1})
        return rec["score"]


def replay_interval(config, store, tokenizer, firewall, checkpoint, ledgers,
                    start_step: int, end_step: int, run_id: str, log=None) -> dict:
    """Rebuild steps [start_step, end_step) and compare to the consumption ledger."""
    payload = checkpoint.load(run_id, start_step)
    dp = DataPlane(config, store, tokenizer, firewall, log=None)
    dp.restore(payload["scheduler_state"], payload["cursors"], payload["opus_state"])

    original = {r["step"]: r for r in ledgers.consumption.rows()}
    opus_by_step: dict = {}
    for r in ledgers.opus.rows():
        opus_by_step.setdefault(r["step"], []).append(r)

    batch_id = int(payload["state"]["extra"]["batch_id"])
    comparisons: list = []
    probe_mismatches: list = []

    for step in range(start_step, end_step):
        orig = original.get(step)
        if orig is None:
            comparisons.append({"step": step, "status": "missing_in_original"})
            continue
        probe = RecordedProbe(opus_by_step.get(step, []))
        batch, _ = dp.next_batch(step, batch_id, probe)
        probe_mismatches.extend([{**m, "step": step} for m in probe.mismatches])

        replay_refs = [[l, i, pid, ph] for l, i, pid, ph in batch.pack_refs]
        replay_spans = [seg for segs in batch.provenance for seg in segs]
        comparisons.append({
            "step": step,
            "batch_id_match": batch.batch_id == orig["batch_id"],
            "hash_match": batch.content_hash == orig["batch_hash"],
            "pack_refs_match": replay_refs == orig["pack_refs"],
            "spans_match": replay_spans == orig["spans"],
            "loss_token_match": batch.n_loss_tokens == orig["n_loss_tokens"],
            "original_hash": orig["batch_hash"][:16],
            "replay_hash": batch.content_hash[:16],
        })
        batch_id += 1

    ok_rows = [c for c in comparisons if c.get("status") != "missing_in_original"]
    all_match = bool(ok_rows) and all(
        c["batch_id_match"] and c["hash_match"] and c["pack_refs_match"]
        and c["spans_match"] and c["loss_token_match"] for c in ok_rows)

    return {
        "interval": [start_step, end_step],
        "run_id": run_id,
        "n_steps_compared": len(ok_rows),
        "all_match": all_match,
        "probe_order_mismatches": probe_mismatches,
        "scan_order_reproduced": len(probe_mismatches) == 0,
        "comparisons": comparisons,
    }
