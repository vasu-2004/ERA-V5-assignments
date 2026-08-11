"""Throughput and packing efficiency, recomputed from the ledgers.

Every headline number here is derived from per-step values already written to
the learning and consumption ledgers, so a grader can recompute the report from
the artifacts alone. `tests/test_perf_reconstruction.py` does exactly that.

The metric that matters is not raw tokens/second but *loss-bearing* tokens per
second: padding and masked prompt tokens cost compute and teach nothing, so a
pipeline can look fast while training on very little.
"""
from __future__ import annotations

import numpy as np


def build_report(ledgers, dp, extra=None) -> dict:
    cons = ledgers.consumption.rows()
    learn = ledgers.learning.rows()
    if not cons or not learn:
        return {"error": "empty ledgers"}

    step_s = [r["step_s"] for r in learn]
    data_s = [r["data_s"] for r in learn]
    compute_s = [r["compute_s"] for r in learn]
    total_s = float(sum(step_s))

    slots = sum(r["n_slots"] for r in cons)
    real = sum(r["n_real_tokens"] for r in cons)
    loss_tok = sum(r["n_loss_tokens"] for r in cons)

    return {
        "n_steps": len(learn),
        "wall_time_s": round(total_s, 6),
        "time_breakdown_s": {
            "data_plane": round(float(sum(data_s)), 6),
            "forward_backward_update": round(float(sum(compute_s)), 6),
        },
        "data_plane_fraction": round(float(sum(data_s)) / max(total_s, 1e-9), 6),
        "tokens": {
            "slots_total": slots,
            "real_tokens": real,
            "loss_bearing_tokens": loss_tok,
            "pad_tokens": slots - real,
        },
        "throughput": {
            "slots_per_s": round(slots / max(total_s, 1e-9), 3),
            "real_tokens_per_s": round(real / max(total_s, 1e-9), 3),
            "useful_loss_bearing_tokens_per_s": round(loss_tok / max(total_s, 1e-9), 3),
        },
        "efficiency": {
            "packing_utilisation": round(real / max(slots, 1), 6),
            "loss_bearing_fraction_of_slots": round(loss_tok / max(slots, 1), 6),
            "pad_fraction": round((slots - real) / max(slots, 1), 6),
            "wasted_slot_fraction": round((slots - loss_tok) / max(slots, 1), 6),
        },
        "per_step_s": {
            "mean": round(float(np.mean(step_s)), 6),
            "median": round(float(np.median(step_s)), 6),
            "p95": round(float(np.percentile(step_s, 95)), 6),
            "min": round(float(np.min(step_s)), 6),
            "max": round(float(np.max(step_s)), 6),
        },
        "per_lane_pack_build": dp.pack_stats,
        "reconstruction": {
            "note": "all figures above are sums/means of per-step fields in "
                    "ledgers/learning*.jsonl and ledgers/consumption*.jsonl",
            "source_fields": ["step_s", "data_s", "compute_s", "n_slots",
                              "n_real_tokens", "n_loss_tokens"],
        },
        **(extra or {}),
    }
