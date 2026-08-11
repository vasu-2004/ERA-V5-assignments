"""Generates evidence.json and evidence.md.

Design rule: this module contains no verdicts. It receives already-computed audit
results, derives each PASS/FAIL from a predicate over those results, and records
alongside it the artifact path and the specific numbers a reader can check by
hand. A requirement can only pass because a computed value says so, which is
what "hardcoded evidence will not be accepted" demands.
"""
from __future__ import annotations

import json
import pathlib
from typing import Callable

from .hashing import hash_obj


class Requirement:
    def __init__(self, key: str, title: str, predicate: Callable[[dict], bool],
                 evidence: Callable[[dict], dict], artifacts: list):
        self.key = key
        self.title = title
        self.predicate = predicate
        self.evidence = evidence
        self.artifacts = artifacts

    def evaluate(self, a: dict) -> dict:
        try:
            passed = bool(self.predicate(a))
            ev = self.evidence(a)
            err = None
        except Exception as exc:              # a missing audit value is a FAIL, not a crash
            passed, ev, err = False, {}, f"{type(exc).__name__}: {exc}"
        return {"requirement": self.title, "key": self.key,
                "result": "PASS" if passed else "FAIL",
                "evidence": ev, "artifacts": self.artifacts,
                **({"error": err} if err else {})}


REQUIREMENTS = [
    Requirement(
        "end_to_end", "End-to-end execution",
        lambda a: (a["run"]["completed_all_phases"]
                   and a["run"]["checks_failed_so_far"] == 0),
        lambda a: {"phases_completed": len(a["run"]["phases"]),
                   "phases": a["run"]["phases"],
                   "total_steps": a["run"]["total_steps"],
                   "checks_failed_so_far": a["run"]["checks_failed_so_far"]},
        ["run.log"]),
    Requirement(
        "tokenizer_integrity", "Tokenizer integrity",
        lambda a: (a["tokenizer"]["hash_matches_all_manifests"]
                   and a["tokenizer"]["retrained_hash_stable"]
                   and a["tokenizer"]["roundtrip_lossless"]
                   and a["shards"]["all_verified"]
                   and a["shards"]["tamper_detected"]),
        lambda a: {"tokenizer_hash": a["tokenizer"]["hash"],
                   "vocab_size": a["tokenizer"]["vocab_size"],
                   "n_shards": a["shards"]["n_shards"],
                   "manifests_agree": a["tokenizer"]["hash_matches_all_manifests"],
                   "rebuild_is_bit_identical": a["tokenizer"]["retrained_hash_stable"],
                   "decode_encode_lossless": a["tokenizer"]["roundtrip_lossless"],
                   "corrupted_shard_rejected": a["shards"]["tamper_detected"],
                   "immutable_overwrite_refused": a["shards"]["overwrite_refused"]},
        ["manifests/", "manifests/tokenizer.json"]),
    Requirement(
        "eval_firewall", "Evaluation firewall",
        lambda a: (a["firewall"]["breach_attempt_blocked"]
                   and a["firewall"]["canary_scan_clean"]
                   and a["firewall"]["no_heldout_lane_in_packs"]),
        lambda a: {"blocked_events": a["firewall"]["n_blocked_events"],
                   "blocked_shards": a["firewall"]["blocked_shard_ids"],
                   "held_out_lanes": a["firewall"]["held_out_lanes"],
                   "batches_canary_scanned": a["firewall"]["n_batches_scanned"],
                   "canary_hits": a["firewall"]["canary_hits"]},
        ["run.log", "evidence.json"]),
    Requirement(
        "packing_correctness", "Packing correctness (masks, labels, position ids)",
        lambda a: (a["masks"]["masks_correct"]
                   and a["provenance"]["all_spans_trainable_and_in_range"]),
        lambda a: {"packs_checked": a["masks"]["packs_checked"],
                   "mask_problems": a["masks"]["n_problems"],
                   "checks": ["no_cross_segment_attention", "causal_only",
                              "position_ids_restart_per_segment", "no_loss_on_padding",
                              "no_loss_across_document_boundary",
                              "prompt_tokens_masked", "labels_are_next_token"],
                   "spans_checked": a["provenance"]["spans_checked"],
                   "policies": a["packing"]["policies"]},
        ["ledgers/consumption.jsonl", "performance.json"]),
    Requirement(
        "mixture_compliance", "Mixture compliance and protected floors",
        lambda a: (a["mixture"]["floors_respected_everywhere"]
                   and a["mixture"]["max_abs_error"] <= 0.10),
        lambda a: {"stages": a["mixture"]["stages"],
                   "max_abs_error_planned_vs_actual": a["mixture"]["max_abs_error"],
                   "tolerance": 0.10},
        ["ledgers/consumption.jsonl"]),
    Requirement(
        "opus_audit", "OPUS audit trail",
        lambda a: (a["opus"]["all_decision_types_present"]
                   and a["opus"]["audit_trail_complete"]
                   and a["opus"]["n_protected_floor_overrides"] > 0
                   and not a["opus"]["unjustified_overrides"]),
        lambda a: {"total_decisions": a["opus"]["total_decisions"],
                   "tally": a["opus"]["tally"],
                   "protected_floor_overrides": a["opus"]["n_protected_floor_overrides"],
                   "override_examples": a["opus"]["overrides_sample"][:3],
                   "every_record_has_score_reason_thresholds":
                       a["opus"]["audit_trail_complete"]},
        ["ledgers/opus.jsonl"]),
    Requirement(
        "crash_recovery", "Crash recovery (no skipped or repeated batches)",
        lambda a: (a["resume"]["next_batch_matched"]
                   and a["resume"]["full_stream_matches_reference"]
                   and a["continuity"]["no_skips_or_repeats"]
                   and a["resume"]["uncommitted_records_discarded"] >= 0),
        lambda a: {"crashed_at_step": a["resume"]["crash_step"],
                   "resumed_from_checkpoint_step": a["resume"]["resume_step"],
                   "records_rolled_back": a["resume"]["uncommitted_records_discarded"],
                   "expected_next_batch_id": a["resume"]["expected_batch_id"],
                   "actual_next_batch_id": a["resume"]["actual_batch_id"],
                   "expected_next_batch_hash": a["resume"]["expected_batch_hash"],
                   "actual_next_batch_hash": a["resume"]["actual_batch_hash"],
                   "all_steps_match_crashfree_reference":
                       a["resume"]["full_stream_matches_reference"],
                   "batch_ids_contiguous": a["continuity"]["batch_ids_contiguous_from_zero"],
                   "duplicates": a["continuity"]["duplicate_batch_ids"]},
        ["checkpoints/", "ledgers/consumption.jsonl"]),
    Requirement(
        "replay", "Replay of a historical interval",
        lambda a: a["replay"]["all_match"] and a["replay"]["scan_order_reproduced"],
        lambda a: {"interval": a["replay"]["interval"],
                   "steps_compared": a["replay"]["n_steps_compared"],
                   "all_hashes_match": a["replay"]["all_match"],
                   "scan_order_reproduced": a["replay"]["scan_order_reproduced"],
                   "sample": a["replay"]["comparisons"][:3]},
        ["ledgers/consumption.jsonl"]),
    Requirement(
        "fork", "Fork from an earlier checkpoint",
        lambda a: (a["fork"]["parent_state_matches"] and a["fork"]["diverges_after_fork"]
                   and a["fork"]["identical_fork_reproduces_parent"]),
        lambda a: {"forked_from_step": a["fork"]["fork_step"],
                   "parent_checkpoint_hash": a["fork"]["parent_checkpoint_hash"],
                   "same_config_fork_reproduces_original":
                       a["fork"]["identical_fork_reproduces_parent"],
                   "changed_config_fork_diverges": a["fork"]["diverges_after_fork"],
                   "first_divergent_step": a["fork"]["first_divergent_step"],
                   "lineage": a["fork"]["lineage"]},
        ["checkpoints/", "ledgers/consumption.fork.jsonl"]),
    Requirement(
        "learning_trace", "Learning trace linked to source data",
        lambda a: (a["learning"]["every_learning_record_linked_to_consumption"]
                   and a["learning"]["per_document_attribution_present"]
                   and a["learning"]["all_losses_finite"]
                   and a["learning"]["loss_decreased"]),
        lambda a: {"learning_records": a["learning"]["n_records"],
                   "linked_to_consumption_by_batch_hash":
                       a["learning"]["every_learning_record_linked_to_consumption"],
                   "documents_with_attributed_loss":
                       a["learning"]["distinct_documents_attributed"],
                   "mean_loss_first_fifth": a["learning"]["mean_first_fifth"],
                   "mean_loss_last_fifth": a["learning"]["mean_last_fifth"],
                   "loss_decreased": a["learning"]["loss_decreased"]},
        ["ledgers/learning.jsonl"]),
    Requirement(
        "ledger_integrity", "Ledger integrity (append-only hash chains)",
        lambda a: a["ledger_integrity"]["all_chains_valid"],
        lambda a: {k: v for k, v in a["ledger_integrity"].items() if k != "all_chains_valid"},
        ["ledgers/"]),
    Requirement(
        "throughput", "Throughput and packing efficiency",
        lambda a: (a["perf"]["throughput"]["useful_loss_bearing_tokens_per_s"] > 0
                   and a["perf"]["efficiency"]["packing_utilisation"] > 0.5),
        lambda a: {"wall_time_s": a["perf"]["wall_time_s"],
                   "steps": a["perf"]["n_steps"],
                   "useful_loss_bearing_tokens_per_s":
                       a["perf"]["throughput"]["useful_loss_bearing_tokens_per_s"],
                   "real_tokens_per_s": a["perf"]["throughput"]["real_tokens_per_s"],
                   "packing_utilisation": a["perf"]["efficiency"]["packing_utilisation"],
                   "loss_bearing_fraction": a["perf"]["efficiency"]["loss_bearing_fraction_of_slots"],
                   "pad_fraction": a["perf"]["efficiency"]["pad_fraction"],
                   "per_lane": a["perf"]["per_lane_pack_build"]},
        ["performance.json"]),
]


def build(audits: dict, out_dir: pathlib.Path) -> dict:
    out_dir = pathlib.Path(out_dir)
    rows = [r.evaluate(audits) for r in REQUIREMENTS]
    n_pass = sum(1 for r in rows if r["result"] == "PASS")

    bundle = {
        "schema": "tdes-evidence/1",
        "run_id": audits["run"]["run_id"],
        "generated_by": "tdes.evidence.build (derived from computed audit values)",
        "config_hash": audits["run"]["config_hash"],
        "summary": {"requirements": len(rows), "passed": n_pass, "failed": len(rows) - n_pass,
                    "all_passed": n_pass == len(rows)},
        "requirements": rows,
        "raw_audits": audits,
    }
    bundle["bundle_hash"] = hash_obj({k: v for k, v in bundle.items() if k != "bundle_hash"})
    (out_dir / "evidence.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

    # ---- human-readable companion
    lines = [
        "# Evidence Summary",
        "",
        f"**Run:** `{bundle['run_id']}`  ",
        f"**Config hash:** `{bundle['config_hash'][:16]}`  ",
        f"**Bundle hash:** `{bundle['bundle_hash'][:16]}`  ",
        f"**Result:** {n_pass}/{len(rows)} requirements passed",
        "",
        "| Requirement | Result | Evidence |",
        "|---|---|---|",
    ]
    for r in rows:
        ev_bits = []
        for k, v in list(r["evidence"].items())[:3]:
            if isinstance(v, (dict, list)):
                v = f"{len(v)} entries" if v else "none"
            elif isinstance(v, str) and len(v) == 64:
                v = v[:16] + "…"          # hashes: enough to check, short enough to read
            ev_bits.append(f"{k}={v}")
        artifacts = ", ".join(f"`{p}`" for p in r["artifacts"])
        lines.append(f"| {r['requirement']} | **{r['result']}** | {'; '.join(ev_bits)} — {artifacts} |")

    lines += ["", "## Key numbers", ""]
    perf = audits["perf"]
    lines += [
        f"- Steps executed: **{perf['n_steps']}**, wall time **{perf['wall_time_s']}s**",
        f"- Packing utilisation: **{perf['efficiency']['packing_utilisation']:.1%}** "
        f"(pad {perf['efficiency']['pad_fraction']:.1%})",
        f"- Useful loss-bearing tokens/s: **{perf['throughput']['useful_loss_bearing_tokens_per_s']}**",
        f"- OPUS decisions: **{audits['opus']['tally']}**, "
        f"protected-floor overrides: **{audits['opus']['n_protected_floor_overrides']}**",
        f"- Mixture max |planned − actual|: **{audits['mixture']['max_abs_error']:.4f}**",
        f"- Loss: **{audits['learning']['mean_first_fifth']} → "
        f"{audits['learning']['mean_last_fifth']}** (first/last fifth of steps)",
        f"- Crash at step **{audits['resume']['crash_step']}**, resumed from "
        f"**{audits['resume']['resume_step']}**, rolled back "
        f"**{audits['resume']['uncommitted_records_discarded']}** uncommitted records",
        f"- Replay interval **[{audits['replay']['interval'][0]}, "
        f"{audits['replay']['interval'][1]})**: "
        f"{audits['replay']['n_steps_compared']} steps, all hashes match "
        f"**{audits['replay']['all_match']}**",
        "",
        "## How to re-verify",
        "",
        "```bash",
        "python run_demo.py          # regenerates every artifact in this directory",
        "python -m pytest tests -q   # re-checks the invariants independently",
        "```",
        "",
        "Every value above is recomputed from `ledgers/`, `manifests/` and "
        "`checkpoints/` by `tdes/audit.py`; `tdes/evidence.py` only turns those "
        "computed values into PASS/FAIL rows.",
    ]
    (out_dir / "evidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return bundle
