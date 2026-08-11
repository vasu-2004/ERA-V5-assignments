"""Central, hashable configuration for the Training Data Execution System.

Every number that can change the byte stream lives here. The config is hashed
into manifests, checkpoints and the evidence bundle, so a run can never be
confused with a run made under different settings.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Lanes: a "lane" is a data type with its own packing policy and mixture weight.
# split=train lanes are trainable; eval/val lanes are firewalled and may never
# contribute a loss-bearing token.
# --------------------------------------------------------------------------
LANES = {
    "web_en": {"split": "train", "policy": "concat_split"},
    "indic":  {"split": "train", "policy": "concat_split"},
    "code":   {"split": "train", "policy": "whole_doc_bestfit"},
    "math":   {"split": "train", "policy": "prompt_masked"},
    "eval_math": {"split": "eval", "policy": "prompt_masked"},
    "val_web":   {"split": "val",  "policy": "concat_split"},
}
TRAIN_LANES = [ln for ln, m in LANES.items() if m["split"] == "train"]
HELD_OUT_LANES = [ln for ln, m in LANES.items() if m["split"] != "train"]


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 1024
    special_tokens: tuple = ("<pad>", "<bos>", "<eos>", "<sep>")


@dataclass(frozen=True)
class PackingConfig:
    seq_len: int = 128
    batch_size: int = 4
    # a segment shorter than this is not worth its own slot in a bin-packed seq
    min_segment_tokens: int = 8


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    init_std: float = 0.02
    seed: int = 1234


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_steps: int = 5


@dataclass(frozen=True)
class CurriculumStage:
    """A curriculum stage: planned lane weights plus per-lane protected floors.

    `floors` are hard minimum shares. If a lane's realised share drops below its
    floor, OPUS is overridden and that lane is force-admitted -- this is what
    stops an online quality filter from silently starving a low-resource lane
    (the Indic-starvation failure mode discussed in the course).
    """
    name: str
    until_step: int              # exclusive upper bound
    weights: dict                # lane -> planned share (need not sum to 1; normalised)
    floors: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OpusConfig:
    """Online data-selection thresholds.

    Thresholds are on the RELATIVE score `pack_loss / EMA(baseline loss)`, not on
    an absolute loss. See tdes/opus.py for why: absolute cuts are unusable when
    the loss scale moves by design over the course of training.
    """
    enabled: bool = True
    quantise: int = 6              # decimal places for score comparison
    baseline_ema: float = 0.9      # EMA factor for the running baseline
    reject_below_rel: float = 0.93  # much easier than typical -> already learned
    defer_below_rel: float = 0.99   # borderline -> defer, reconsider after cooldown
    defer_cooldown: int = 3        # steps before a deferred pack is eligible again
    max_defers: int = 2            # after this many defers, admit it anyway
    warmup_steps: int = 3          # accept everything early; the baseline is unformed
    max_candidate_scans: int = 24  # bound the search for an admissible pack


@dataclass(frozen=True)
class RunConfig:
    run_id: str = "v5-tdes-demo"
    seed: int = 7
    total_steps: int = 40
    checkpoint_every: int = 10
    crash_at_step: int = 27      # deliberate crash, mid-interval
    resume_from_step: int = 20   # last checkpoint at or before the crash
    replay_interval: tuple = (10, 20)
    fork_from_step: int = 10
    fork_steps: int = 6

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    packing: PackingConfig = field(default_factory=PackingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    opus: OpusConfig = field(default_factory=OpusConfig)

    # Floors sit just under each protected lane's planned share. That is
    # deliberate: the realised share of a lane oscillates around its plan as
    # packs are admitted one at a time, so a floor set this way binds
    # *transiently* during the run -- which is what actually exercises the
    # override path -- while still being satisfiable at the end of the stage.
    # A floor set above the planned weight would be self-contradictory, and one
    # set far below would never fire and would prove nothing.
    curriculum: tuple = (
        CurriculumStage(
            name="stage_a_broad",
            until_step=20,
            weights={"web_en": 0.40, "code": 0.25, "math": 0.20, "indic": 0.15},
            floors={"indic": 0.14},
        ),
        CurriculumStage(
            name="stage_b_reasoning",
            until_step=10_000,
            weights={"web_en": 0.25, "code": 0.30, "math": 0.33, "indic": 0.12},
            floors={"indic": 0.11},
        ),
    )

    def stage_for_step(self, step: int) -> CurriculumStage:
        for st in self.curriculum:
            if step < st.until_step:
                return st
        return self.curriculum[-1]

    def to_dict(self) -> dict:
        def enc(o):
            if dataclasses.is_dataclass(o):
                return {k: enc(v) for k, v in dataclasses.asdict(o).items()}
            if isinstance(o, dict):
                return {k: enc(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [enc(v) for v in o]
            return o
        d = enc(self)
        d["lanes"] = LANES
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


# A fork uses the same machinery with different mixture weights, to prove that
# a branch diverges only where it is supposed to.
def forked_config(base: RunConfig) -> RunConfig:
    forked_curriculum = (
        CurriculumStage(
            name="fork_code_heavy",
            until_step=10_000,
            weights={"web_en": 0.15, "code": 0.50, "math": 0.25, "indic": 0.10},
            floors={"indic": 0.10},
        ),
    )
    return dataclasses.replace(base, run_id=base.run_id + "-fork",
                               curriculum=forked_curriculum)
