"""Mixture scheduling, protected floors, OPUS decisions and ledger integrity."""
from __future__ import annotations

import pytest

from tdes.ledger import Ledger
from tdes.mixture import MixtureScheduler
from tdes.opus import DECISIONS, Opus, OpusState
from tests.conftest import tiny_config


# ------------------------------------------------------------------ mixture
def test_lane_order_is_deterministic():
    cfg = tiny_config()
    a = MixtureScheduler(cfg)
    b = MixtureScheduler(cfg)
    for step in range(30):
        ra, rb = a.rank_lanes(step), b.rank_lanes(step)
        assert ra == rb
        a.commit(step, ra[0])
        b.commit(step, rb[0])


def test_realised_shares_converge_to_plan():
    cfg = tiny_config()
    s = MixtureScheduler(cfg)
    for _ in range(400):
        s.commit(0, s.rank_lanes(0)[0])
    comp = s.compliance(0)
    assert comp["max_abs_error"] < 0.02, comp


def test_stages_are_scheduled_independently():
    """Stage B's mixture must reflect stage B's weights, not repair stage A."""
    cfg = tiny_config()
    s = MixtureScheduler(cfg)
    a_step = 0
    b_step = cfg.curriculum[0].until_step + 1
    for _ in range(200):
        s.commit(a_step, s.rank_lanes(a_step)[0])
    for _ in range(200):
        s.commit(b_step, s.rank_lanes(b_step)[0])
    assert s.compliance(a_step)["max_abs_error"] < 0.02
    assert s.compliance(b_step)["max_abs_error"] < 0.02


def test_protected_floor_detects_starvation():
    cfg = tiny_config()
    s = MixtureScheduler(cfg)
    stage = cfg.stage_for_step(0)
    protected = next(iter(stage.floors))
    other = next(ln for ln in stage.weights if ln != protected)
    for _ in range(20):                    # starve the protected lane on purpose
        s.commit(0, other)
    assert protected in s.starving_lanes(0)
    assert s.rank_lanes(0)[0] == protected, "a starving lane must be served first"


# --------------------------------------------------------------------- OPUS
def _opus(**over):
    import dataclasses
    cfg = tiny_config()
    return Opus(dataclasses.replace(cfg, opus=dataclasses.replace(cfg.opus, **over)))


def test_all_four_decisions_are_reachable():
    """Drive each decision by targeting a relative score, since the thresholds
    are defined against the moving baseline rather than an absolute loss."""
    o = _opus(warmup_steps=0)
    cfg = o.cfg
    for _ in range(5):                       # establish a baseline
        o.decide("warm", 5.0, 10, "web_en", False)

    def at_relative(target, pack_id, lane="web_en", starving=False):
        return o.decide(pack_id, target * o.state.baseline, 10, lane, starving)

    seen = set()
    seen.add(at_relative(1.40, "hard")["decision"])                       # ACCEPT
    seen.add(at_relative(cfg.reject_below_rel - 0.10, "easy")["decision"])  # REJECT
    mid = (cfg.reject_below_rel + cfg.defer_below_rel) / 2
    seen.add(at_relative(mid, "edge")["decision"])                        # DEFER
    seen.add(at_relative(cfg.reject_below_rel - 0.10, "floor",
                         lane="indic", starving=True)["decision"])        # FORCED_ACCEPT
    assert set(DECISIONS) == seen, seen


def test_override_only_when_lane_is_starving():
    o = _opus(warmup_steps=0)
    for _ in range(5):
        o.decide("warm", 5.0, 10, "web_en", False)
    not_starving = o.decide("p1", 0.2, 10, "web_en", False)
    starving = o.decide("p2", 0.2, 10, "indic", True)
    assert not_starving["decision"] == "REJECT" and not not_starving["override"]
    assert starving["decision"] == "FORCED_ACCEPT" and starving["override"]
    assert starving["would_have_been"] == "REJECT"     # counterfactual recorded


def test_deferred_pack_is_on_cooldown_then_admitted():
    o = _opus(warmup_steps=0, defer_cooldown=3, max_defers=1)
    for _ in range(5):
        o.decide("warm", 5.0, 0, "web_en", False)
    d = o.decide("p", 4.85, 10, "web_en", False)
    assert d["decision"] == "DEFER"
    assert not o.eligible("p", 11) and o.eligible("p", 13)
    # once the defer budget is spent the pack must be admitted, not looped forever
    assert o.decide("p", 4.85, 13, "web_en", False)["decision"] == "ACCEPT"


def test_decisions_are_reproducible_from_restored_state():
    o1 = _opus(warmup_steps=0)
    scores = [5.0, 5.2, 4.7, 6.0, 1.0, 4.9]
    for i, s in enumerate(scores):
        o1.decide(f"p{i}", s, 10, "web_en", False)
    snapshot = o1.state.to_dict()

    o2 = _opus(warmup_steps=0)
    o2.state = OpusState.from_dict(snapshot)
    a = o1.decide("next", 4.88, 11, "web_en", False)
    b = o2.decide("next", 4.88, 11, "web_en", False)
    assert a["decision"] == b["decision"] and a["relative_score"] == b["relative_score"]


def test_every_record_carries_its_audit_fields():
    o = _opus(warmup_steps=0)
    r = o.decide("p", 3.0, 5, "code", False)
    for field in ("score", "baseline", "relative_score", "decision", "would_have_been",
                  "override", "reason", "thresholds", "lane", "step"):
        assert field in r


# ------------------------------------------------------------------ ledgers
def test_chain_is_valid_and_tamper_evident(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", "test")
    for i in range(10):
        led.append({"step": i, "payload": f"v{i}"})
    assert led.verify_chain() == (True, None)

    led.records[4]["record"]["payload"] = "tampered"
    ok, bad = led.verify_chain()
    assert not ok and bad == 4


def test_reload_continues_the_same_chain(tmp_path):
    p = tmp_path / "l.jsonl"
    a = Ledger(p, "test")
    for i in range(5):
        a.append({"step": i})
    head = a.head
    b = Ledger(p, "test")                 # simulates a restarted process
    assert len(b) == 5 and b.head == head
    b.append({"step": 5})
    assert b.verify_chain() == (True, None)


def test_truncation_restores_a_checkpointed_prefix(tmp_path):
    p = tmp_path / "l.jsonl"
    led = Ledger(p, "test")
    for i in range(6):
        led.append({"step": i})
    pinned = led.offset()
    for i in range(6, 10):
        led.append({"step": i})
    dropped = led.truncate_to(pinned["n_records"], pinned["head"])
    assert dropped == 4 and len(led) == 6
    assert Ledger(p, "test").head == pinned["head"]   # persisted, not just in memory


def test_truncation_rejects_a_head_mismatch(tmp_path):
    led = Ledger(tmp_path / "l.jsonl", "test")
    for i in range(4):
        led.append({"step": i})
    with pytest.raises(RuntimeError):
        led.truncate_to(2, "not-the-real-head")
