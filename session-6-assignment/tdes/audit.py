"""Post-run audit: re-derive every claim from the ledgers and artifacts.

The audit never trusts an in-memory value. It re-reads the ledgers from disk,
re-verifies their hash chains, recomputes aggregates, and checks structural
invariants (batch ids contiguous, no duplicates, every consumed token traceable
to a train-split shard). Anything the evidence bundle asserts is produced here.
"""
from __future__ import annotations

import numpy as np

from .config import LANES
from .packing import IGNORE_INDEX, build_attention_mask


def audit_ledger_integrity(ledgers) -> dict:
    res = {}
    for name, (ok, bad) in ledgers.verify_all().items():
        res[name] = {"chain_valid": ok, "first_bad_seq": bad,
                     "n_records": len(getattr(ledgers, name))}
    res["all_chains_valid"] = all(v["chain_valid"] for v in res.values()
                                  if isinstance(v, dict))
    return res


def audit_batch_continuity(ledgers) -> dict:
    """No skipped and no repeated batches -- the core crash-recovery invariant."""
    rows = ledgers.consumption.rows()
    steps = [r["step"] for r in rows]
    ids = [r["batch_id"] for r in rows]
    dup_steps = sorted({s for s in steps if steps.count(s) > 1})
    dup_ids = sorted({i for i in ids if ids.count(i) > 1})
    gaps = [b for a, b in zip(steps, steps[1:]) if b != a + 1]
    return {
        "n_batches": len(rows),
        "steps_contiguous_from_zero": steps == list(range(len(steps))),
        "batch_ids_contiguous_from_zero": ids == list(range(len(ids))),
        "duplicate_steps": dup_steps,
        "duplicate_batch_ids": dup_ids,
        "unexpected_gaps": gaps,
        "no_skips_or_repeats": (not dup_steps and not dup_ids and not gaps
                                and steps == list(range(len(steps)))),
    }


def audit_provenance(ledgers, store, tokenizer) -> dict:
    """Every consumed span must resolve to a real, train-split, verified shard."""
    rows = ledgers.consumption.rows()
    bad_split, bad_range, checked = [], [], 0
    shard_cache = {}
    for r in rows:
        for seg in r["spans"]:
            sid = seg["shard_id"]
            if sid not in shard_cache:
                shard_cache[sid] = store.load(sid, tokenizer.content_hash)
            sh = shard_cache[sid]
            checked += 1
            if sh.split != "train":
                bad_split.append({"batch_id": r["batch_id"], "shard_id": sid,
                                  "split": sh.split})
            if not (0 <= seg["doc_start"] < seg["doc_end"] <= sh.tokens.size):
                bad_range.append({"batch_id": r["batch_id"], "shard_id": sid,
                                  "span": [seg["doc_start"], seg["doc_end"]]})
    return {
        "spans_checked": checked,
        "shards_touched": sorted(shard_cache),
        "non_train_spans": bad_split,
        "out_of_range_spans": bad_range,
        "all_spans_trainable_and_in_range": not bad_split and not bad_range,
    }


def audit_mixture(ledgers, config) -> dict:
    """Planned versus actual lane shares, per curriculum stage, plus floors."""
    rows = ledgers.consumption.rows()
    by_stage: dict = {}
    for r in rows:
        st = by_stage.setdefault(r["stage"], {"packs": {}, "tokens": {}, "n": 0})
        for lane, n in r["lane_counts"].items():
            st["packs"][lane] = st["packs"].get(lane, 0) + n
            st["n"] += n
        for seg in r["spans"]:
            t = seg["slot_end"] - seg["slot_start"]
            st["tokens"][seg["lane"]] = st["tokens"].get(seg["lane"], 0) + t

    out = {"stages": {}, "floors_respected_everywhere": True, "max_abs_error": 0.0}
    for stage in config.curriculum:
        if stage.name not in by_stage:
            continue
        agg = by_stage[stage.name]
        total_w = sum(stage.weights.values())
        planned = {ln: w / total_w for ln, w in stage.weights.items()}
        total_packs = max(agg["n"], 1)
        actual = {ln: agg["packs"].get(ln, 0) / total_packs for ln in planned}
        errors = {ln: abs(planned[ln] - actual[ln]) for ln in planned}
        floors = {ln: v for ln, v in stage.floors.items()}
        floor_ok = {ln: actual.get(ln, 0.0) >= v - 1e-9 for ln, v in floors.items()}
        out["stages"][stage.name] = {
            "n_packs": agg["n"],
            "planned": {k: round(v, 6) for k, v in planned.items()},
            "actual": {k: round(v, 6) for k, v in actual.items()},
            "abs_error": {k: round(v, 6) for k, v in errors.items()},
            "max_abs_error": round(max(errors.values()), 6),
            "floors": floors, "floors_respected": floor_ok,
            "token_share": {k: round(v / max(sum(agg["tokens"].values()), 1), 6)
                            for k, v in sorted(agg["tokens"].items())},
        }
        out["max_abs_error"] = max(out["max_abs_error"],
                                   out["stages"][stage.name]["max_abs_error"])
        if not all(floor_ok.values()):
            out["floors_respected_everywhere"] = False
    return out


def audit_opus(ledgers) -> dict:
    rows = ledgers.opus.rows()
    tally: dict = {}
    overrides = []
    for r in rows:
        tally[r["decision"]] = tally.get(r["decision"], 0) + 1
        if r.get("override"):
            overrides.append({"step": r["step"], "lane": r["lane"], "pack_id": r["pack_id"],
                              "score": r["score"], "would_have_been": r["would_have_been"],
                              "reason": r["reason"]})
    required = ("ACCEPT", "REJECT", "DEFER", "FORCED_ACCEPT")
    # an override must only ever happen for a lane that was actually starving
    bad_overrides = [o for o in overrides if "protected_floor_override" not in o["reason"]
                     and o["reason"] != "scan_budget_exhausted_progress_guarantee"]
    return {
        "total_decisions": len(rows),
        "tally": tally,
        "all_decision_types_present": all(tally.get(d, 0) > 0 for d in required),
        "missing_decision_types": [d for d in required if tally.get(d, 0) == 0],
        "n_protected_floor_overrides": len(overrides),
        "overrides_sample": overrides[:8],
        "unjustified_overrides": bad_overrides,
        "audit_trail_complete": all("score" in r and "reason" in r and "thresholds" in r
                                    for r in rows),
    }


def audit_learning(ledgers) -> dict:
    """Loss must be linked to the data that produced it, and must actually move."""
    rows = ledgers.learning.rows()
    cons = {r["batch_id"]: r for r in ledgers.consumption.rows()}
    losses = [r["loss"] for r in rows]
    linked = all(r["batch_id"] in cons
                 and cons[r["batch_id"]]["batch_hash"] == r["batch_hash"] for r in rows)
    has_doc_attr = all(len(r["per_doc_loss"]) > 0 for r in rows)
    docs = {d["doc_id"] for r in rows for d in r["per_doc_loss"]}
    first, last = (losses[0], losses[-1]) if losses else (None, None)
    k = max(1, len(losses) // 5)
    return {
        "n_records": len(rows),
        "every_learning_record_linked_to_consumption": linked,
        "per_document_attribution_present": has_doc_attr,
        "distinct_documents_attributed": len(docs),
        "first_loss": first, "last_loss": last,
        "mean_first_fifth": round(float(np.mean(losses[:k])), 6) if losses else None,
        "mean_last_fifth": round(float(np.mean(losses[-k:])), 6) if losses else None,
        "loss_decreased": bool(losses and np.mean(losses[-k:]) < np.mean(losses[:k])),
        "all_losses_finite": bool(all(np.isfinite(losses))),
    }


def audit_masks(dp, n_packs: int = 24) -> dict:
    """Structural checks on the packs the run actually used."""
    problems = []
    checked = 0
    for lane, packs in dp.packs.items():
        for p in packs[:max(1, n_packs // max(len(dp.packs), 1))]:
            checked += 1
            mask = build_attention_mask(p.segment_ids)
            seg = p.segment_ids

            # 1. no attention across documents, none into or out of padding
            for i in range(seg.size):
                row = mask[i]
                if seg[i] == 0:
                    if row.any() and not (row.sum() == 1 and row[0]):
                        problems.append((lane, p.pack_id, "pad_row_attends", i))
                    continue
                if (row & (seg != seg[i])).any():
                    problems.append((lane, p.pack_id, "cross_segment_attention", i))
                if row[i + 1:].any():
                    problems.append((lane, p.pack_id, "non_causal_attention", i))

            # 2. position ids restart at 0 and increase by 1 inside each segment
            for s in p.segments:
                pos = p.position_ids[s.slot_start:s.slot_end]
                if pos.size and (pos[0] != 0 or not np.array_equal(pos, np.arange(pos.size))):
                    problems.append((lane, p.pack_id, "bad_position_ids", s.slot_start))

            # 3. padding never bears loss
            if (p.labels[seg == 0] != IGNORE_INDEX).any():
                problems.append((lane, p.pack_id, "loss_on_padding", -1))

            # 4. a segment's final slot bears no loss (no cross-document target)
            for s in p.segments:
                if p.labels[s.slot_end - 1] != IGNORE_INDEX:
                    problems.append((lane, p.pack_id, "loss_on_segment_boundary", s.slot_end - 1))

            # 5. prompt tokens are excluded from the loss
            for s in p.segments:
                if s.prompt_tokens > 0:
                    upto = s.slot_start + s.prompt_tokens - 1
                    if (p.labels[s.slot_start:upto] != IGNORE_INDEX).any():
                        problems.append((lane, p.pack_id, "loss_inside_prompt", s.slot_start))

            # 6. labels really are the next input token where loss is taken
            for s in p.segments:
                for i in range(s.slot_start, s.slot_end - 1):
                    if p.labels[i] != IGNORE_INDEX and p.labels[i] != p.input_ids[i + 1]:
                        problems.append((lane, p.pack_id, "label_not_next_token", i))
                        break
    return {"packs_checked": checked, "n_problems": len(problems),
            "problems": [list(map(str, p)) for p in problems[:20]],
            "masks_correct": len(problems) == 0}


def audit_packing_efficiency(dp, ledgers) -> dict:
    """Per-policy utilisation, recomputed from the consumption ledger."""
    rows = ledgers.consumption.rows()
    slots = sum(r["n_slots"] for r in rows)
    real = sum(r["n_real_tokens"] for r in rows)
    loss = sum(r["n_loss_tokens"] for r in rows)
    return {
        "per_lane_pack_build": dp.pack_stats,
        "consumed_slots": slots, "consumed_real_tokens": real,
        "consumed_loss_tokens": loss,
        "utilisation": round(real / max(slots, 1), 6),
        "loss_bearing_fraction": round(loss / max(slots, 1), 6),
        "pad_fraction": round((slots - real) / max(slots, 1), 6),
        "policies": {ln: LANES[ln]["policy"] for ln in dp.packs},
    }
