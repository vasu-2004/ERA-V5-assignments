"""Packing correctness (masks, labels, position ids) and the evaluation firewall."""
from __future__ import annotations

import numpy as np
import pytest

from tdes.config import HELD_OUT_LANES, LANES, TRAIN_LANES
from tdes.firewall import FirewallBlock
from tdes.packing import IGNORE_INDEX, build_attention_mask, pack_lane
from tdes.trainer import DataPlane
from tests.conftest import tiny_config


@pytest.fixture(scope="module")
def planes(built, tokenizer):
    return DataPlane(tiny_config(), built["store"], tokenizer, built["firewall"])


def all_packs(planes):
    for lane, packs in planes.packs.items():
        for p in packs:
            yield lane, p


def test_only_trainable_lanes_are_packed(planes):
    assert set(planes.packs) <= set(TRAIN_LANES)
    assert not (set(planes.packs) & set(HELD_OUT_LANES))
    assert len(planes.packs) == len(TRAIN_LANES)


def test_every_policy_is_exercised(planes):
    used = {LANES[lane]["policy"] for lane in planes.packs}
    assert used == {"concat_split", "whole_doc_bestfit", "prompt_masked"}


def test_no_pack_is_empty_of_loss(planes):
    """A pack with no loss-bearing token costs a slot and teaches nothing."""
    for lane, p in all_packs(planes):
        assert p.n_loss_tokens > 0, f"{lane}/{p.pack_id} has no loss-bearing token"


def test_attention_is_causal_and_intra_segment(planes):
    for lane, p in list(all_packs(planes))[:60]:
        mask = build_attention_mask(p.segment_ids)
        seg = p.segment_ids
        for i in range(seg.size):
            row = mask[i]
            assert not row[i + 1:].any(), f"{p.pack_id}: attends to the future at {i}"
            if seg[i] == 0:
                continue
            assert not (row & (seg != seg[i])).any(), \
                f"{p.pack_id}: attends across a document boundary at {i}"
            assert not (row & (seg == 0)).any(), f"{p.pack_id}: attends into padding at {i}"


def test_position_ids_restart_per_segment(planes):
    for lane, p in list(all_packs(planes))[:60]:
        for s in p.segments:
            pos = p.position_ids[s.slot_start:s.slot_end]
            assert np.array_equal(pos, np.arange(pos.size)), \
                f"{p.pack_id}: position ids do not restart at 0 for segment {s.slot_start}"


def test_labels_are_next_token_and_never_cross_documents(planes):
    for lane, p in list(all_packs(planes))[:60]:
        for s in p.segments:
            # inside a segment, a label is either masked or exactly the next input
            for i in range(s.slot_start, s.slot_end - 1):
                if p.labels[i] != IGNORE_INDEX:
                    assert p.labels[i] == p.input_ids[i + 1]
            # the last slot of a segment must never predict into the next document
            assert p.labels[s.slot_end - 1] == IGNORE_INDEX


def test_padding_never_bears_loss(planes):
    for lane, p in all_packs(planes):
        assert (p.labels[p.segment_ids == 0] == IGNORE_INDEX).all()


def test_prompt_tokens_are_masked_out(planes):
    checked = 0
    for lane, p in all_packs(planes):
        for s in p.segments:
            if s.prompt_tokens <= 1:
                continue
            checked += 1
            region = p.labels[s.slot_start:s.slot_start + s.prompt_tokens - 1]
            assert (region == IGNORE_INDEX).all(), \
                f"{p.pack_id}: loss taken inside the prompt"
    assert checked > 0, "no prompt-masked segments were exercised"


def test_prompt_masked_lane_still_learns_the_answer(planes):
    """Masking the question must not mask the answer as well."""
    math_packs = planes.packs["math"]
    assert any(p.n_loss_tokens > 0 for p in math_packs)
    for p in math_packs[:20]:
        seg = p.segments[0]
        answer_region = p.labels[seg.slot_start + seg.prompt_tokens - 1:seg.slot_end - 1]
        assert (answer_region != IGNORE_INDEX).any()


def test_whole_doc_policy_never_splits_a_file_across_a_sequence(built, tokenizer):
    shards = sorted([s for s in (built["store"].load(i, tokenizer.content_hash)
                                 for i in built["store"].all_shard_ids())
                     if s.lane == "code"], key=lambda s: s.shard_id)
    packs = pack_lane(shards, "code", "whole_doc_bestfit", 96, 8)
    for p in packs:
        ids = [s.doc_id for s in p.segments]
        assert len(ids) == len(set(ids)), "a document appears twice in one sequence"


# ------------------------------------------------------------------ firewall
def test_held_out_shards_are_refused(built, tokenizer):
    store, fw = built["store"], built["firewall"]
    held = [s for s in store.all_shard_ids()
            if store.load(s, tokenizer.content_hash).split != "train"]
    assert held, "the corpus has no held-out shards to test with"
    for sid in held:
        with pytest.raises(FirewallBlock):
            fw.assert_trainable(store.load(sid, tokenizer.content_hash))


def test_canary_detector_actually_detects(built, tokenizer):
    """Negative results only mean something if the detector can fire at all."""
    store, fw = built["store"], built["firewall"]
    held = [s for s in store.all_shard_ids()
            if store.load(s, tokenizer.content_hash).split != "train"]
    hits = fw.scan_batch(store.load(held[0], tokenizer.content_hash).tokens[None, :])
    assert hits, "canary detector failed its positive control"


def test_no_canary_reaches_any_trainable_pack(planes, built):
    fw = built["firewall"]
    for lane, p in all_packs(planes):
        assert not fw.scan_batch(p.input_ids[None, :]), \
            f"held-out canary found in trainable pack {p.pack_id}"
