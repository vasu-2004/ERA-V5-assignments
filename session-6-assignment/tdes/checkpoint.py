"""Checkpoints that pin the data plane, not just the weights.

A weights-only checkpoint cannot support exact resume: restoring parameters but
re-deriving the data position will skip or repeat batches. So a checkpoint here
commits to all five pieces of state that determine what happens next:

  model / optimiser   the learner
  scheduler + cursors where the mixture had got to in every lane
  OPUS state          deferral counts and cooldowns
  ledger offsets      (n_records, chain head) for each ledger

The ledger offsets are what make recovery provable rather than hopeful. On
resume the ledgers are truncated back to exactly those offsets; the chain head
must match, or the checkpoint is rejected. Records written after the checkpoint
but before the crash are discarded and then regenerated identically.
"""
from __future__ import annotations

import json
import pathlib
import shutil

import numpy as np

from .hashing import hash_array, hash_obj
from .mixture import LaneCursor, SchedulerState
from .opus import OpusState


class Checkpoint:
    def __init__(self, root: pathlib.Path):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def dir_for(self, run_id: str, step: int) -> pathlib.Path:
        return self.root / run_id / f"step_{step:06d}"

    # ---------------------------------------------------------------- save
    def save(self, *, run_id, step, model, optim, scheduler, cursors, opus,
             ledgers, config, rng_state, extra=None) -> dict:
        d = self.dir_for(run_id, step)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

        msd = model.state_dict()
        np.savez(d / "model.npz", **msd)
        osd = optim.state_dict()
        np.savez(d / "optim.npz", t=np.array([osd["t"]]),
                 **{f"m__{k}": v for k, v in osd["m"].items()},
                 **{f"v__{k}": v for k, v in osd["v"].items()})

        state = {
            "run_id": run_id,
            "step": step,
            "config_hash": hash_obj(config.to_dict()),
            "scheduler": scheduler.state.to_dict(),
            "cursors": {ln: c.to_dict() for ln, c in cursors.items()},
            "opus": opus.state.to_dict(),
            "ledger_offsets": ledgers.offsets(),
            "rng_state": rng_state,
            "param_hashes": {k: hash_array(v) for k, v in msd.items()},
            "extra": extra or {},
        }
        state["checkpoint_hash"] = hash_obj(
            {k: v for k, v in state.items() if k != "checkpoint_hash"})
        (d / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True),
                                      encoding="utf-8")
        return state

    # ---------------------------------------------------------------- load
    def load(self, run_id: str, step: int) -> dict:
        d = self.dir_for(run_id, step)
        state = json.loads((d / "state.json").read_text(encoding="utf-8"))
        recomputed = hash_obj({k: v for k, v in state.items() if k != "checkpoint_hash"})
        if recomputed != state["checkpoint_hash"]:
            raise RuntimeError(f"checkpoint {run_id}@{step} state.json is corrupt")

        with np.load(d / "model.npz") as z:
            model_sd = {k: z[k] for k in z.files}
        for k, h in state["param_hashes"].items():
            if hash_array(model_sd[k]) != h:
                raise RuntimeError(f"checkpoint {run_id}@{step}: parameter {k} failed hash check")

        with np.load(d / "optim.npz") as z:
            t = int(z["t"][0])
            m = {k[len("m__"):]: z[k] for k in z.files if k.startswith("m__")}
            v = {k[len("v__"):]: z[k] for k in z.files if k.startswith("v__")}

        return {
            "state": state,
            "model_sd": model_sd,
            "optim_sd": {"t": t, "m": m, "v": v},
            "scheduler_state": SchedulerState.from_dict(state["scheduler"]),
            "cursors": {ln: LaneCursor.from_dict(c) for ln, c in state["cursors"].items()},
            "opus_state": OpusState.from_dict(state["opus"]),
            "ledger_offsets": state["ledger_offsets"],
            "step": state["step"],
        }

    def list_steps(self, run_id: str) -> list:
        p = self.root / run_id
        if not p.exists():
            return []
        return sorted(int(x.name.split("_")[1]) for x in p.glob("step_*") if x.is_dir())

    def latest_at_or_before(self, run_id: str, step: int) -> int | None:
        cands = [s for s in self.list_steps(run_id) if s <= step]
        return max(cands) if cands else None
