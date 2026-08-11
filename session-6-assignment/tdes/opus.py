"""OPUS: online admission control for candidate packs.

The idea from the course: material the model already predicts confidently has
little left to teach it, so spending a step on it wastes compute. OPUS probes the
*live* model on a candidate pack and returns one of four decisions:

  ACCEPT           worth learning from now
  REJECT           already well predicted relative to the model's current ability
  DEFER            borderline -- cool down and reconsider shortly
  FORCED_ACCEPT    protected-floor override: the lane is below its guaranteed
                   share, so the pack is admitted even though the score alone
                   would have rejected or deferred it

Scoring is RELATIVE, not absolute. An absolute loss threshold is unusable here:
a fresh model over a 1024-token vocabulary starts near ln(1024) ~ 6.9 and falls
from there, so any fixed cut either admits everything early and rejects
everything later, or the reverse. Instead OPUS keeps an exponential moving
average of the scores it has seen and judges each pack by `loss / baseline`.
That is scale-free, tracks the model as it improves, and expresses the actual
intent: "is this pack easier than what this model currently finds typical?"

Two further design choices:

* The ratio is quantised before any comparison, so a decision never turns on
  float noise in the last bits -- decisions must be reproducible when a run is
  forked or resumed from a checkpoint.

* Every decision is written to a ledger with its raw score, its relative score,
  the thresholds in force, and what the decision *would* have been before an
  override. Without that counterfactual an override is indistinguishable from a
  plain acceptance, and the audit trail would not show the floor doing any work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ACCEPT = "ACCEPT"
REJECT = "REJECT"
DEFER = "DEFER"
FORCED_ACCEPT = "FORCED_ACCEPT"
DECISIONS = (ACCEPT, REJECT, DEFER, FORCED_ACCEPT)


@dataclass
class OpusState:
    defer_counts: dict = field(default_factory=dict)    # pack_id -> times deferred
    cooldown_until: dict = field(default_factory=dict)  # pack_id -> step
    tally: dict = field(default_factory=lambda: {d: 0 for d in DECISIONS})
    baseline: float = 0.0        # EMA of observed scores
    n_scored: int = 0

    def to_dict(self) -> dict:
        return {"defer_counts": dict(self.defer_counts),
                "cooldown_until": dict(self.cooldown_until),
                "tally": dict(self.tally),
                "baseline": self.baseline, "n_scored": self.n_scored}

    @classmethod
    def from_dict(cls, d: dict) -> "OpusState":
        st = cls(defer_counts=dict(d["defer_counts"]),
                 cooldown_until=dict(d["cooldown_until"]),
                 baseline=float(d["baseline"]), n_scored=int(d["n_scored"]))
        st.tally = {k: int(v) for k, v in d["tally"].items()}
        for k in DECISIONS:
            st.tally.setdefault(k, 0)
        return st


class Opus:
    def __init__(self, config, state: OpusState | None = None):
        self.cfg = config.opus
        self.state = state or OpusState()

    def eligible(self, pack_id: str, step: int) -> bool:
        """False while a deferred pack is still cooling down."""
        return step >= self.state.cooldown_until.get(pack_id, -1)

    def quantise(self, value: float) -> float:
        return round(float(value), self.cfg.quantise)

    def _update_baseline(self, score: float) -> float:
        """EMA over observed scores. Returns the baseline used for THIS decision
        (pre-update), so a pack is never judged against itself."""
        prior = self.state.baseline if self.state.n_scored > 0 else score
        a = self.cfg.baseline_ema
        self.state.baseline = score if self.state.n_scored == 0 else (
            a * self.state.baseline + (1.0 - a) * score)
        self.state.n_scored += 1
        return prior

    def decide(self, pack_id: str, score: float, step: int, lane: str,
               lane_starving: bool) -> dict:
        """Return a full decision record. Deterministic given (score, step, state)."""
        raw = self.quantise(score)
        baseline = self._update_baseline(raw)
        rel = self.quantise(raw / baseline) if baseline > 1e-12 else 1.0
        defers = self.state.defer_counts.get(pack_id, 0)

        if not self.cfg.enabled or step < self.cfg.warmup_steps:
            base, reason = ACCEPT, ("warmup" if self.cfg.enabled else "opus_disabled")
        elif rel < self.cfg.reject_below_rel:
            base, reason = REJECT, "relative_loss_below_reject_threshold_already_learned"
        elif rel < self.cfg.defer_below_rel:
            if defers >= self.cfg.max_defers:
                base, reason = ACCEPT, "defer_budget_exhausted_admitting"
            else:
                base, reason = DEFER, "relative_loss_in_defer_band"
        else:
            base, reason = ACCEPT, "relative_loss_above_defer_threshold"

        decision, override = base, False
        if lane_starving and base in (REJECT, DEFER):
            decision, override = FORCED_ACCEPT, True
            reason = f"protected_floor_override(was={base}:{reason})"

        if decision == DEFER:
            self.state.defer_counts[pack_id] = defers + 1
            self.state.cooldown_until[pack_id] = step + self.cfg.defer_cooldown
        self.state.tally[decision] = self.state.tally.get(decision, 0) + 1

        return {
            "pack_id": pack_id, "lane": lane, "step": step,
            "score": raw, "baseline": self.quantise(baseline), "relative_score": rel,
            "decision": decision, "would_have_been": base,
            "override": override, "reason": reason,
            "lane_starving": bool(lane_starving),
            "defers_so_far": defers,
            "thresholds": {"reject_below_rel": self.cfg.reject_below_rel,
                           "defer_below_rel": self.cfg.defer_below_rel,
                           "warmup_steps": self.cfg.warmup_steps,
                           "max_defers": self.cfg.max_defers,
                           "baseline_ema": self.cfg.baseline_ema},
        }

    @staticmethod
    def admits(decision: str) -> bool:
        return decision in (ACCEPT, FORCED_ACCEPT)

    def audit(self) -> dict:
        total = sum(self.state.tally.values())
        return {
            "tally": dict(self.state.tally),
            "total_decisions": total,
            "all_decision_types_exercised": all(self.state.tally.get(d, 0) > 0
                                                for d in DECISIONS),
            "baseline": self.state.baseline,
            "packs_currently_cooling": len(self.state.cooldown_until),
        }
