"""The data plane and the training loop.

Separation of concerns that the rest of the system depends on:

  DataPlane  owns shards, packs, the mixture scheduler, lane cursors, OPUS and
             the firewall. It decides *what* the next batch is. It touches the
             model only through a `probe` callback, which is why the whole data
             stream can be replayed later without a model at all.

  Trainer    owns the model, the optimiser, the ledgers and the checkpoints. It
             decides *what happens* to a batch.

Ordering rule that makes crash recovery sound: a batch is appended to the
consumption ledger and the learning ledger only after its optimiser step has
completed. A checkpoint then pins both ledger heads. So at any instant the
ledgers describe exactly the work that is already reflected in the weights, and
truncating to a checkpoint's offsets restores a consistent state.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np

from .checkpoint import Checkpoint
from .config import LANES, TRAIN_LANES
from .firewall import Firewall
from .hashing import hash_obj
from .ledger import LedgerSet
from .mixture import LaneCursor, MixtureScheduler
from .model import Adam, Transformer
from .opus import Opus, FORCED_ACCEPT
from .packing import IGNORE_INDEX, assemble_batch, pack_lane


class SimulatedCrash(RuntimeError):
    """Raised to abort a run mid-flight, exactly like a killed process would."""


class DataPlane:
    def __init__(self, config, store, tokenizer, firewall: Firewall, log=None):
        self.config = config
        self.store = store
        self.tokenizer = tokenizer
        self.firewall = firewall
        self.log = log

        self.shards_by_lane: dict = {}
        for sid in store.all_shard_ids():
            sh = store.load(sid, tokenizer.content_hash)
            self.shards_by_lane.setdefault(sh.lane, []).append(sh)

        # Packs are built once, deterministically, from verified shards. Because
        # this is a pure function of (shards, policy, seq_len), replay can rebuild
        # byte-identical packs later without trusting any stored tensor.
        self.packs: dict = {}
        self.pack_stats: dict = {}
        for lane in TRAIN_LANES:
            shards = sorted(self.shards_by_lane.get(lane, []), key=lambda s: s.shard_id)
            if not shards:
                continue
            for sh in shards:
                firewall.assert_trainable(sh, purpose="pack_for_training")
            packs = pack_lane(shards, lane, LANES[lane]["policy"],
                              config.packing.seq_len, config.packing.min_segment_tokens)
            if not packs:
                # A configured lane that yields no servable pack (for example a
                # prompt-masked lane whose prompts all exceed seq_len) must be
                # dropped from the schedule explicitly and loudly. Leaving it in
                # would let the scheduler rank a lane it cannot serve, and the
                # mixture report would silently compare against a plan that was
                # never achievable.
                self.pack_stats[lane] = {"policy": LANES[lane]["policy"], "n_packs": 0,
                                         "slots": 0, "real_tokens": 0, "loss_tokens": 0,
                                         "utilisation": 0.0, "loss_token_fraction": 0.0,
                                         "excluded_reason": "no_servable_packs_at_this_seq_len"}
                if log:
                    log.warn(f"lane {lane} produced no servable packs at seq_len="
                             f"{config.packing.seq_len}; excluded from the schedule")
                continue
            self.packs[lane] = packs
            slots = sum(p.input_ids.size for p in packs)
            self.pack_stats[lane] = {
                "policy": LANES[lane]["policy"], "n_packs": len(packs), "slots": slots,
                "real_tokens": sum(p.n_real_tokens for p in packs),
                "loss_tokens": sum(p.n_loss_tokens for p in packs),
                "utilisation": round(sum(p.n_real_tokens for p in packs) / max(slots, 1), 6),
                "loss_token_fraction": round(sum(p.n_loss_tokens for p in packs) / max(slots, 1), 6),
            }
            if log:
                log.event("lane_packed", lane=lane, policy=LANES[lane]["policy"],
                          n_packs=len(packs),
                          utilisation=self.pack_stats[lane]["utilisation"],
                          loss_token_fraction=self.pack_stats[lane]["loss_token_fraction"])

        self.scheduler = MixtureScheduler(config)
        self.cursors = {lane: LaneCursor() for lane in self.packs}
        self.opus = Opus(config)

    # -- state (for checkpoint / restore) ---------------------------------
    def restore(self, scheduler_state, cursors, opus_state):
        self.scheduler = MixtureScheduler(self.config, state=scheduler_state)
        self.cursors = {lane: cursors.get(lane, LaneCursor()) for lane in self.packs}
        self.opus = Opus(self.config, state=opus_state)

    def pack_of(self, lane: str, index: int):
        return self.packs[lane][index]

    # -- batch construction ----------------------------------------------
    def next_batch(self, step: int, batch_id: int, probe):
        """Select batch_size admissible packs. Returns (Batch, [decision records])."""
        B = self.config.packing.batch_size
        chosen: list = []
        decisions: list = []
        scans = 0
        max_scans = self.config.opus.max_candidate_scans

        while len(chosen) < B and scans < max_scans:
            ranked = self.scheduler.rank_lanes(step)
            ranked = [ln for ln in ranked if ln in self.packs]
            if not ranked:
                raise RuntimeError("no trainable lanes with packs")
            lane = ranked[0]
            lane_packs = self.packs[lane]

            # walk past packs that are still on OPUS cooldown
            pack = None
            pack_idx = None
            for _ in range(min(len(lane_packs), 8)):
                idx = self.cursors[lane].index
                cand = lane_packs[idx]
                if self.opus.eligible(cand.pack_id, step):
                    pack, pack_idx = cand, idx
                    break
                self.cursors[lane].advance(len(lane_packs))
            if pack is None:
                pack_idx = self.cursors[lane].index
                pack = lane_packs[pack_idx]

            starving = lane in self.scheduler.starving_lanes(step)
            score = probe(pack)
            dec = self.opus.decide(pack.pack_id, score, step, lane, starving)
            dec["batch_id"] = batch_id
            dec["pack_index"] = pack_idx
            decisions.append(dec)
            self.cursors[lane].advance(len(lane_packs))
            scans += 1

            if Opus.admits(dec["decision"]):
                chosen.append((lane, pack_idx, pack))
                self.scheduler.commit(step, lane)

        # Progress guarantee: a step must always yield a full batch, or the run
        # could stall forever under an unlucky threshold setting. This path
        # deliberately does NOT call opus.decide: feeding it a sentinel score
        # would corrupt the EMA baseline that every later relative score depends
        # on. It is also recorded with override=False and its own reason, so it
        # can never be counted as a protected-floor override in the audit.
        while len(chosen) < B:
            lane = [ln for ln in self.scheduler.rank_lanes(step) if ln in self.packs][0]
            lane_packs = self.packs[lane]
            pack_idx = self.cursors[lane].index
            pack = lane_packs[pack_idx]
            decisions.append({
                "pack_id": pack.pack_id, "lane": lane, "step": step,
                "score": None, "baseline": self.opus.state.baseline,
                "relative_score": None,
                "decision": FORCED_ACCEPT, "would_have_been": None,
                "override": False,
                "reason": "scan_budget_exhausted_progress_guarantee",
                "lane_starving": False, "defers_so_far": 0,
                "thresholds": {"max_candidate_scans": max_scans},
                "batch_id": batch_id, "pack_index": pack_idx,
            })
            self.opus.state.tally[FORCED_ACCEPT] = \
                self.opus.state.tally.get(FORCED_ACCEPT, 0) + 1
            self.cursors[lane].advance(len(lane_packs))
            chosen.append((lane, pack_idx, pack))
            self.scheduler.commit(step, lane)

        for lane, _, _ in chosen:
            if not self.firewall.is_trainable_lane(lane):
                raise RuntimeError(f"firewall violation: lane {lane} reached a training batch")

        return assemble_batch(batch_id, step, chosen), decisions


class Trainer:
    def __init__(self, config, store, tokenizer, firewall, artifacts: pathlib.Path,
                 log, ledger_suffix: str = "", run_id: str | None = None,
                 log_packing: bool = False):
        self.config = config
        self.log = log
        self.run_id = run_id or config.run_id
        self.artifacts = pathlib.Path(artifacts)
        self.dp = DataPlane(config, store, tokenizer, firewall,
                            log if log_packing else None)
        self.model = Transformer(config.model, tokenizer.vocab_size)
        self.optim = Adam(self.model.p, config.optim)
        self.ledgers = LedgerSet(self.artifacts / "ledgers", suffix=ledger_suffix)
        self.ckpt = Checkpoint(self.artifacts / "checkpoints")
        self.rng = np.random.default_rng(config.seed)
        self.step = 0
        self.batch_id = 0
        self.tokens_seen = 0
        self.step_timings: list = []

    # -- OPUS probe -------------------------------------------------------
    def probe(self, pack):
        """Mean loss of the current model on this pack's loss-bearing tokens.

        This is a real forward pass on the live weights -- OPUS's signal is the
        model's actual predictive difficulty, not a proxy or a random number.
        """
        if int((pack.labels != IGNORE_INDEX).sum()) == 0:
            return 0.0
        _, out, _ = self.model.forward(pack.input_ids[None, :], pack.position_ids[None, :],
                                       pack.segment_ids[None, :], pack.labels[None, :])
        return out["loss"]

    # -- per-document loss attribution ------------------------------------
    @staticmethod
    def attribute_losses(batch, per_pos_loss):
        """Split a batch's loss back onto the documents that produced it."""
        per_lane: dict = {}
        per_doc: list = []
        for b, segs in enumerate(batch.provenance):
            for seg in segs:
                s, e = seg["slot_start"], seg["slot_end"]
                valid = batch.labels[b, s:e] != IGNORE_INDEX
                n = int(valid.sum())
                if n == 0:
                    continue
                tot = float(per_pos_loss[b, s:e][valid].sum())
                lane = seg["lane"]
                agg = per_lane.setdefault(lane, {"loss_sum": 0.0, "n": 0})
                agg["loss_sum"] += tot
                agg["n"] += n
                per_doc.append({"doc_id": seg["doc_id"], "shard_id": seg["shard_id"],
                                "lane": lane, "n_loss_tokens": n,
                                "mean_loss": round(tot / n, 6)})
        per_lane_mean = {ln: round(v["loss_sum"] / v["n"], 6) for ln, v in per_lane.items()}
        return per_lane_mean, per_doc

    # -- one step ---------------------------------------------------------
    def train_step(self):
        t0 = time.perf_counter()
        step = self.step
        batch, decisions = self.dp.next_batch(step, self.batch_id, self.probe)
        t_data = time.perf_counter()

        _, out, cache = self.model.forward(batch.input_ids, batch.position_ids,
                                           batch.segment_ids, batch.labels)
        grads = self.model.backward(cache)
        opt_info = self.optim.step(self.model.p, grads, step)
        t_end = time.perf_counter()

        stage = self.config.stage_for_step(step)
        n_slots = batch.n_slots
        n_real = batch.n_real_tokens
        n_loss = batch.n_loss_tokens
        self.tokens_seen += n_real

        # --- OPUS ledger (every decision, admitted or not)
        for dec in decisions:
            self.ledgers.opus.append(dec)

        # --- consumption ledger: what was consumed, with full provenance
        cons = {
            "step": step, "batch_id": batch.batch_id, "batch_hash": batch.content_hash,
            "stage": stage.name, "lane_counts": batch.lane_counts,
            "n_slots": n_slots, "n_real_tokens": n_real, "n_loss_tokens": n_loss,
            "pad_tokens": n_slots - n_real,
            "pack_refs": [[l, i, pid, ph] for l, i, pid, ph in batch.pack_refs],
            "spans": [seg for segs in batch.provenance for seg in segs],
            "utilisation": round(n_real / n_slots, 6),
            "loss_token_fraction": round(n_loss / n_slots, 6),
        }
        self.ledgers.consumption.append(cons)

        # --- learning ledger: what the model did with it
        per_lane, per_doc = self.attribute_losses(batch, out["per_pos_loss"])
        learn = {
            "step": step, "batch_id": batch.batch_id, "batch_hash": batch.content_hash,
            "loss": round(out["loss"], 8),
            "n_loss_tokens": out["n_valid"],
            "lr": opt_info["lr"], "grad_norm": round(opt_info["grad_norm"], 8),
            "clip_scale": round(opt_info["clip_scale"], 8),
            "per_lane_loss": per_lane,
            "per_doc_loss": per_doc,
            "tokens_seen_cum": self.tokens_seen,
            "data_s": round(t_data - t0, 6),
            "compute_s": round(t_end - t_data, 6),
            "step_s": round(t_end - t0, 6),
        }
        self.ledgers.learning.append(learn)
        self.step_timings.append(learn["step_s"])

        self.step += 1
        self.batch_id += 1
        return batch, out, cons, learn

    # -- checkpoint -------------------------------------------------------
    def save_checkpoint(self, extra=None) -> dict:
        st = self.ckpt.save(
            run_id=self.run_id, step=self.step, model=self.model, optim=self.optim,
            scheduler=self.dp.scheduler,
            cursors=self.dp.cursors, opus=self.dp.opus, ledgers=self.ledgers,
            config=self.config,
            rng_state=hash_obj(str(self.rng.bit_generator.state)),
            extra={**(extra or {}), "batch_id": self.batch_id,
                   "tokens_seen": self.tokens_seen},
        )
        self.log.event("checkpoint_saved", run_id=self.run_id, step=self.step,
                       checkpoint_hash=st["checkpoint_hash"][:16],
                       consumption_records=st["ledger_offsets"]["consumption"]["n_records"],
                       learning_records=st["ledger_offsets"]["learning"]["n_records"])
        self.log.check("checkpoint_saved", True, run_id=self.run_id, step=self.step,
                       hash=st["checkpoint_hash"][:16])
        return st

    def load_checkpoint(self, step: int, run_id: str | None = None) -> dict:
        payload = self.ckpt.load(run_id or self.run_id, step)
        self.model.load_state_dict(payload["model_sd"])
        self.optim.load_state_dict(payload["optim_sd"])
        self.dp.restore(payload["scheduler_state"], payload["cursors"], payload["opus_state"])
        self.step = payload["step"]
        self.batch_id = int(payload["state"]["extra"]["batch_id"])
        self.tokens_seen = int(payload["state"]["extra"]["tokens_seen"])
        return payload

    # -- run --------------------------------------------------------------
    def run(self, until_step: int, crash_at: int | None = None,
            checkpoint_every: int | None = None):
        ce = checkpoint_every or self.config.checkpoint_every
        while self.step < until_step:
            if crash_at is not None and self.step == crash_at:
                self.log.event("crash_injected", run_id=self.run_id, step=self.step,
                               reason="deliberate_fault_injection")
                raise SimulatedCrash(f"simulated crash at step {self.step}")
            self.train_step()
            if self.step % ce == 0:
                self.save_checkpoint()
