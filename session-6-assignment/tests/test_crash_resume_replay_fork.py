"""The recovery invariants, exercised on a self-contained miniature run.

These tests build their own trainers rather than reading submission_artifacts/,
so they verify the mechanism, not one recorded run.
"""
from __future__ import annotations

import dataclasses

import pytest

from tdes.audit import audit_batch_continuity
from tdes.checkpoint import Checkpoint
from tdes.config import forked_config
from tdes.replay import replay_interval
from tdes.trainer import SimulatedCrash
from tests.conftest import tiny_config


@pytest.fixture
def reference(trainer_factory, tmp_path):
    """A crash-free run: the ground truth every other run is compared against."""
    cfg = tiny_config()
    t = trainer_factory(cfg, suffix="ref", run_id="ref", art=tmp_path)
    t.run(cfg.total_steps)
    return cfg, t, {r["step"]: r for r in t.ledgers.consumption.rows()}


def test_reference_run_is_continuous(reference):
    _, t, _ = reference
    cont = audit_batch_continuity(t.ledgers)
    assert cont["no_skips_or_repeats"], cont


def test_two_clean_runs_are_bit_identical(trainer_factory, reference, tmp_path):
    """Determinism precondition: without it, no recovery proof means anything."""
    cfg, _, ref_rows = reference
    t2 = trainer_factory(cfg, suffix="ref2", run_id="ref2", art=tmp_path)
    t2.run(cfg.total_steps)
    rows2 = {r["step"]: r for r in t2.ledgers.consumption.rows()}
    assert len(rows2) == len(ref_rows)
    for s in ref_rows:
        assert rows2[s]["batch_hash"] == ref_rows[s]["batch_hash"]


def test_resume_produces_exactly_the_expected_next_batch(trainer_factory, reference,
                                                          tmp_path):
    cfg, _, ref_rows = reference
    art = tmp_path / "crashrun"
    art.mkdir()

    crashed = trainer_factory(cfg, suffix="", run_id="main", art=art)
    with pytest.raises(SimulatedCrash):
        crashed.run(cfg.total_steps, crash_at=cfg.crash_at_step)
    records_at_crash = len(crashed.ledgers.consumption)
    assert records_at_crash == cfg.crash_at_step
    del crashed

    ck = Checkpoint(art / "checkpoints")
    resume_step = ck.latest_at_or_before("main", cfg.crash_at_step)
    assert resume_step is not None and resume_step < cfg.crash_at_step

    resumed = trainer_factory(cfg, suffix="", run_id="main", art=art)
    payload = resumed.load_checkpoint(resume_step)
    dropped = resumed.ledgers.truncate_to(payload["ledger_offsets"])
    assert dropped["consumption"] == records_at_crash - resume_step

    batch, _, cons, _ = resumed.train_step()
    expected = ref_rows[resume_step]
    assert batch.batch_id == expected["batch_id"]
    assert batch.content_hash == expected["batch_hash"]
    assert cons["pack_refs"] == expected["pack_refs"]
    assert cons["spans"] == expected["spans"]


def test_resumed_run_matches_reference_and_skips_nothing(trainer_factory, reference,
                                                          tmp_path):
    cfg, _, ref_rows = reference
    art = tmp_path / "crashrun2"
    art.mkdir()

    crashed = trainer_factory(cfg, suffix="", run_id="main", art=art)
    with pytest.raises(SimulatedCrash):
        crashed.run(cfg.total_steps, crash_at=cfg.crash_at_step)
    del crashed

    ck = Checkpoint(art / "checkpoints")
    resumed = trainer_factory(cfg, suffix="", run_id="main", art=art)
    payload = resumed.load_checkpoint(ck.latest_at_or_before("main", cfg.crash_at_step))
    resumed.ledgers.truncate_to(payload["ledger_offsets"])
    resumed.run(cfg.total_steps)

    rows = {r["step"]: r for r in resumed.ledgers.consumption.rows()}
    assert len(rows) == len(ref_rows)
    for s in ref_rows:
        assert rows[s]["batch_hash"] == ref_rows[s]["batch_hash"], f"step {s} diverged"
    assert audit_batch_continuity(resumed.ledgers)["no_skips_or_repeats"]


def test_replay_reconstructs_the_same_batches(trainer_factory, built, tokenizer,
                                              reference, tmp_path):
    cfg, t, _ = reference
    a, b = cfg.replay_interval
    rep = replay_interval(cfg, built["store"], tokenizer, built["firewall"],
                          Checkpoint(tmp_path / "checkpoints"), t.ledgers,
                          a, b, "ref")
    assert rep["n_steps_compared"] == b - a
    assert rep["all_match"], rep["comparisons"]
    assert rep["scan_order_reproduced"], rep["probe_order_mismatches"]


def test_replay_detects_a_changed_shard(trainer_factory, built, tokenizer, reference,
                                        tmp_path, monkeypatch):
    """Replay must FAIL when the underlying data no longer matches history."""
    cfg, t, _ = reference
    a, b = cfg.replay_interval

    store = built["store"]
    # Alter a token that the replayed interval genuinely consumes; mutating an
    # arbitrary shard offset would prove nothing if that offset is never served.
    spans = [seg for r in t.ledgers.consumption.rows() if a <= r["step"] < b
             for seg in r["spans"]]
    assert spans, "no spans recorded in the replay interval"
    target = spans[0]
    shard = store.load(target["shard_id"], tokenizer.content_hash)
    original = shard.tokens.copy()
    mutated = original.copy()
    pos = target["doc_start"]
    mutated[pos] = (int(mutated[pos]) + 1) % 400
    object.__setattr__(shard, "tokens", mutated)     # bypass the store's guard
    try:
        rep = replay_interval(cfg, store, tokenizer, built["firewall"],
                              Checkpoint(tmp_path / "checkpoints"), t.ledgers,
                              a, b, "ref")
        assert not rep["all_match"], "replay passed despite altered shard bytes"
    finally:
        object.__setattr__(shard, "tokens", original)


def test_control_fork_reproduces_and_changed_fork_diverges(trainer_factory, reference,
                                                            tmp_path):
    cfg, _, ref_rows = reference
    fk = cfg.fork_from_step

    ctl = trainer_factory(cfg, suffix="ctl", run_id="ctl", art=tmp_path)
    ctl.load_checkpoint(fk, run_id="ref")
    ctl_hashes = [ctl.train_step()[0].content_hash for _ in range(cfg.fork_steps)]
    expected = [ref_rows[s]["batch_hash"] for s in range(fk, fk + cfg.fork_steps)]
    assert ctl_hashes == expected, "same-config fork failed to reproduce the parent"

    fcfg = dataclasses.replace(forked_config(cfg), run_id="fork")
    fork = trainer_factory(fcfg, suffix="fork", run_id="fork", art=tmp_path)
    fork.load_checkpoint(fk, run_id="ref")
    fork_hashes = [fork.train_step()[0].content_hash for _ in range(cfg.fork_steps)]
    assert fork_hashes != expected, "changed-config fork did not diverge"


def test_checkpoint_rejects_corrupted_state(reference, tmp_path):
    import json
    cfg, _, _ = reference
    ck = Checkpoint(tmp_path / "checkpoints")
    step = ck.list_steps("ref")[0]
    sp = ck.dir_for("ref", step) / "state.json"
    state = json.loads(sp.read_text())
    state["step"] = state["step"] + 1            # tamper without fixing the hash
    sp.write_text(json.dumps(state))
    with pytest.raises(RuntimeError):
        ck.load("ref", step)
