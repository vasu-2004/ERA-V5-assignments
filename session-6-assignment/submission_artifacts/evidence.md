# Evidence Summary

**Run:** `v5-tdes-demo`  
**Config hash:** `64e6d552b44e38e9`  
**Bundle hash:** `e059b6a985958c92`  
**Result:** 12/12 requirements passed

| Requirement | Result | Evidence |
|---|---|---|
| End-to-end execution | **PASS** | phases_completed=9; phases=9 entries; total_steps=40 — `run.log` |
| Tokenizer integrity | **PASS** | tokenizer_hash=4bfc6e2141e28a1b…; vocab_size=1024; n_shards=14 — `manifests/`, `manifests/tokenizer.json` |
| Evaluation firewall | **PASS** | blocked_events=2; blocked_shards=2 entries; held_out_lanes=2 entries — `run.log`, `evidence.json` |
| Packing correctness (masks, labels, position ids) | **PASS** | packs_checked=24; mask_problems=0; checks=7 entries — `ledgers/consumption.jsonl`, `performance.json` |
| Mixture compliance and protected floors | **PASS** | stages=2 entries; max_abs_error_planned_vs_actual=0.005; tolerance=0.1 — `ledgers/consumption.jsonl` |
| OPUS audit trail | **PASS** | total_decisions=334; tally=4 entries; protected_floor_overrides=4 — `ledgers/opus.jsonl` |
| Crash recovery (no skipped or repeated batches) | **PASS** | crashed_at_step=27; resumed_from_checkpoint_step=20; records_rolled_back=7 — `checkpoints/`, `ledgers/consumption.jsonl` |
| Replay of a historical interval | **PASS** | interval=2 entries; steps_compared=10; all_hashes_match=True — `ledgers/consumption.jsonl` |
| Fork from an earlier checkpoint | **PASS** | forked_from_step=10; parent_checkpoint_hash=5f76ef4901b05de3…; same_config_fork_reproduces_original=True — `checkpoints/`, `ledgers/consumption.fork.jsonl` |
| Learning trace linked to source data | **PASS** | learning_records=40; linked_to_consumption_by_batch_hash=True; documents_with_attributed_loss=77 — `ledgers/learning.jsonl` |
| Ledger integrity (append-only hash chains) | **PASS** | consumption=3 entries; learning=3 entries; opus=3 entries — `ledgers/` |
| Throughput and packing efficiency | **PASS** | wall_time_s=2.921488; steps=40; useful_loss_bearing_tokens_per_s=5581.745 — `performance.json` |

## Key numbers

- Steps executed: **40**, wall time **2.921488s**
- Packing utilisation: **99.5%** (pad 0.5%)
- Useful loss-bearing tokens/s: **5581.745**
- OPUS decisions: **{'ACCEPT': 156, 'DEFER': 143, 'FORCED_ACCEPT': 4, 'REJECT': 31}**, protected-floor overrides: **4**
- Mixture max |planned − actual|: **0.0050**
- Loss: **6.730068 → 6.007355** (first/last fifth of steps)
- Crash at step **27**, resumed from **20**, rolled back **7** uncommitted records
- Replay interval **[10, 20)**: 10 steps, all hashes match **True**

## How to re-verify

```bash
python run_demo.py          # regenerates every artifact in this directory
python -m pytest tests -q   # re-checks the invariants independently
```

Every value above is recomputed from `ledgers/`, `manifests/` and `checkpoints/` by `tdes/audit.py`; `tdes/evidence.py` only turns those computed values into PASS/FAIL rows.
