"""Grader's-eye checks: does the generated bundle agree with the raw artifacts?

These run against submission_artifacts/ and are skipped when it has not been
generated yet. They exist because the evidence bundle must be *checkable*, not
merely present: every headline number is recomputed here from the ledgers.
"""
from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
ART = ROOT / "submission_artifacts"

pytestmark = pytest.mark.skipif(
    not (ART / "evidence.json").exists(),
    reason="run `python run_demo.py` first to generate submission_artifacts/")


def read_ledger(name):
    p = ART / "ledgers" / name
    return [json.loads(l)["record"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="module")
def bundle():
    return json.loads((ART / "evidence.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def performance():
    return json.loads((ART / "performance.json").read_text(encoding="utf-8"))


def test_required_layout_exists():
    for rel in ["run.log", "evidence.json", "evidence.md", "performance.json",
                "manifests", "ledgers", "checkpoints"]:
        assert (ART / rel).exists(), f"missing required artifact: {rel}"


def test_every_requirement_passed(bundle):
    failed = [r["key"] for r in bundle["requirements"] if r["result"] != "PASS"]
    assert not failed, f"failing requirements: {failed}"


def test_bundle_hash_is_self_consistent(bundle):
    import sys
    sys.path.insert(0, str(ROOT))
    from tdes.hashing import hash_obj
    recomputed = hash_obj({k: v for k, v in bundle.items() if k != "bundle_hash"})
    assert recomputed == bundle["bundle_hash"]


def test_log_pass_lines_match_the_bundle():
    """Every [FAIL] in the log would contradict an all-pass bundle."""
    log = (ART / "run.log").read_text(encoding="utf-8")
    assert "[FAIL]" not in log, "run.log contains failing checks"
    for name in ["tokenizer_hash_verified", "eval_shard_blocked", "checkpoint_saved",
                 "resume_next_batch_matched", "replay_hash_matched"]:
        assert f"[PASS] {name}" in log, f"required check missing from run.log: {name}"


def test_performance_recomputes_from_the_ledgers(performance):
    """The throughput report must be reconstructible, or it earns no credit."""
    cons = read_ledger("consumption.jsonl")
    learn = read_ledger("learning.jsonl")

    assert performance["n_steps"] == len(learn)
    assert performance["tokens"]["slots_total"] == sum(r["n_slots"] for r in cons)
    assert performance["tokens"]["real_tokens"] == sum(r["n_real_tokens"] for r in cons)
    assert performance["tokens"]["loss_bearing_tokens"] == sum(r["n_loss_tokens"] for r in cons)

    wall = sum(r["step_s"] for r in learn)
    assert performance["wall_time_s"] == pytest.approx(wall, rel=1e-6)

    util = performance["tokens"]["real_tokens"] / performance["tokens"]["slots_total"]
    assert performance["efficiency"]["packing_utilisation"] == pytest.approx(util, rel=1e-6)

    tps = performance["tokens"]["loss_bearing_tokens"] / wall
    assert performance["throughput"]["useful_loss_bearing_tokens_per_s"] == \
        pytest.approx(tps, rel=1e-3)


def test_mixture_claim_recomputes_from_the_consumption_ledger(bundle):
    cons = read_ledger("consumption.jsonl")
    claimed = bundle["raw_audits"]["mixture"]["stages"]
    by_stage: dict = {}
    for r in cons:
        st = by_stage.setdefault(r["stage"], {})
        for lane, n in r["lane_counts"].items():
            st[lane] = st.get(lane, 0) + n
    for stage, counts in by_stage.items():
        total = sum(counts.values())
        for lane, n in counts.items():
            assert claimed[stage]["actual"][lane] == pytest.approx(n / total, abs=1e-6)


def test_learning_records_link_to_consumed_batches():
    cons = {r["batch_id"]: r for r in read_ledger("consumption.jsonl")}
    for r in read_ledger("learning.jsonl"):
        assert r["batch_id"] in cons
        assert cons[r["batch_id"]]["batch_hash"] == r["batch_hash"]
        assert r["per_doc_loss"], "no per-document loss attribution recorded"


def test_opus_overrides_are_all_justified():
    rows = read_ledger("opus.jsonl")
    overrides = [r for r in rows if r.get("override")]
    assert overrides, "no protected-floor override was ever recorded"
    for r in overrides:
        assert r["decision"] == "FORCED_ACCEPT"
        assert r["lane_starving"] is True
        assert r["would_have_been"] in ("REJECT", "DEFER")
        assert "protected_floor_override" in r["reason"]


def test_no_heldout_shard_appears_in_any_consumed_span():
    manifests = {p.name.replace(".manifest.json", ""):
                 json.loads(p.read_text(encoding="utf-8"))
                 for p in (ART / "manifests").glob("*.manifest.json")}
    for r in read_ledger("consumption.jsonl"):
        for seg in r["spans"]:
            assert manifests[seg["shard_id"]]["split"] == "train", \
                f"held-out shard {seg['shard_id']} was consumed in batch {r['batch_id']}"


def test_batch_ids_are_contiguous_with_no_repeats():
    ids = [r["batch_id"] for r in read_ledger("consumption.jsonl")]
    assert ids == list(range(len(ids)))
