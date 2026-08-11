"""Mixture schedule: curriculum stages, lane weights, protected floors.

Lane selection is a deterministic largest-deficit scheduler evaluated in exact
rational arithmetic (`fractions.Fraction`), not floating point. That matters for
this system specifically: the lane sequence is part of the replayed byte stream,
so it must be reproducible bit-for-bit, and float accumulation across thousands
of picks is exactly where that guarantee would quietly rot.

Counters are kept PER CURRICULUM STAGE. A curriculum statement like "stage B is
reasoning-heavy" is a claim about the mixture *during* stage B; if the scheduler
tracked one cumulative total instead, entering a new stage would make it spend
the first part of that stage correcting the previous stage's history rather than
serving the new plan. Per-stage counters also make the audit's planned-vs-actual
comparison mean what it appears to mean.

Protected floors are enforced here rather than left to the quality filter. A
lane whose realised share within the current stage falls under its floor is
selected first and, if OPUS wants to drop the pack anyway, overridden -- the
mechanism that stops an online filter from starving a low-resource lane (Indic,
in this configuration) out of the run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction


def _normalise(weights: dict) -> dict:
    total = sum(Fraction(str(w)) for w in weights.values())
    return {k: Fraction(str(v)) / total for k, v in weights.items()}


@dataclass
class SchedulerState:
    # stage_name -> {lane -> packs admitted during that stage}
    served: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)   # stage_name -> packs admitted

    def to_dict(self) -> dict:
        return {"served": {s: dict(v) for s, v in self.served.items()},
                "totals": dict(self.totals)}

    @classmethod
    def from_dict(cls, d: dict) -> "SchedulerState":
        return cls(served={s: dict(v) for s, v in d["served"].items()},
                   totals={s: int(v) for s, v in d["totals"].items()})


class MixtureScheduler:
    def __init__(self, config, state: SchedulerState | None = None):
        self.config = config
        self.state = state or SchedulerState()
        for stage in config.curriculum:
            self.state.served.setdefault(stage.name, {})
            self.state.totals.setdefault(stage.name, 0)
            for lane in stage.weights:
                self.state.served[stage.name].setdefault(lane, 0)

    # -- planning ---------------------------------------------------------
    def stage(self, step: int):
        return self.config.stage_for_step(step)

    def planned(self, step: int) -> dict:
        return _normalise(self.stage(step).weights)

    def _served(self, stage_name: str) -> dict:
        return self.state.served.setdefault(stage_name, {})

    def total(self, stage_name: str) -> int:
        return self.state.totals.get(stage_name, 0)

    def realised(self, step: int) -> dict:
        st = self.stage(step).name
        total = self.total(st)
        served = self._served(st)
        if total == 0:
            return {ln: Fraction(0) for ln in self.stage(step).weights}
        return {ln: Fraction(served.get(ln, 0), total) for ln in self.stage(step).weights}

    # -- protected floors -------------------------------------------------
    def starving_lanes(self, step: int) -> list:
        """Lanes below their protected floor for the current stage, worst first.

        Floors need a little history before a share is measurable at all, so they
        do not engage during the first few picks of a stage.
        """
        stage = self.stage(step)
        if not stage.floors or self.total(stage.name) < 4:
            return []
        realised = self.realised(step)
        out = []
        for lane, floor in stage.floors.items():
            f = Fraction(str(floor))
            have = realised.get(lane, Fraction(0))
            if have < f:
                out.append((f - have, lane))
        out.sort(key=lambda t: (-t[0], t[1]))
        return [lane for _, lane in out]

    # -- selection --------------------------------------------------------
    def rank_lanes(self, step: int) -> list:
        """Lanes ordered by how far behind their planned share they are.

        Deficit is measured against the share the lane should hold after one more
        pick, so the sequence tracks the plan tightly instead of oscillating.
        Ties break on lane name, so the ordering is total and machine-independent.
        """
        stage = self.stage(step)
        planned = self.planned(step)
        served = self._served(stage.name)
        nxt = self.total(stage.name) + 1
        scored = []
        for lane, share in planned.items():
            deficit = share * nxt - Fraction(served.get(lane, 0))
            scored.append((-deficit, lane))
        scored.sort()
        ranked = [lane for _, lane in scored]
        for lane in reversed(self.starving_lanes(step)):
            if lane in ranked:
                ranked.remove(lane)
                ranked.insert(0, lane)
        return ranked

    def commit(self, step: int, lane: str):
        st = self.stage(step).name
        served = self._served(st)
        served[lane] = served.get(lane, 0) + 1
        self.state.totals[st] = self.total(st) + 1

    # -- reporting --------------------------------------------------------
    def compliance(self, step: int) -> dict:
        stage = self.stage(step)
        planned = self.planned(step)
        realised = self.realised(step)
        rows = {}
        for lane in planned:
            p = float(planned[lane])
            r = float(realised.get(lane, Fraction(0)))
            rows[lane] = {"planned": round(p, 6), "actual": round(r, 6),
                          "abs_error": round(abs(p - r), 6)}
        floors = {ln: float(Fraction(str(v))) for ln, v in stage.floors.items()}
        return {"stage": stage.name, "n_packs": self.total(stage.name), "lanes": rows,
                "floors": floors,
                "floors_respected": {ln: float(realised.get(ln, Fraction(0))) >= v - 1e-9
                                     for ln, v in floors.items()},
                "max_abs_error": max((v["abs_error"] for v in rows.values()), default=0.0)}


@dataclass
class LaneCursor:
    """Deterministic position in a lane's pack list, with epoch wrap-around."""
    index: int = 0
    epoch: int = 0

    def to_dict(self) -> dict:
        return {"index": self.index, "epoch": self.epoch}

    @classmethod
    def from_dict(cls, d: dict) -> "LaneCursor":
        return cls(index=int(d["index"]), epoch=int(d["epoch"]))

    def advance(self, n_packs: int):
        self.index += 1
        if self.index >= n_packs:
            self.index = 0
            self.epoch += 1
