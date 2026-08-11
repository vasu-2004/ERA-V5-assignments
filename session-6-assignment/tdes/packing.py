"""Packing: shards -> fixed-length sequences -> batches.

Three policies, because different data types break in different ways if you
pack them identically:

  concat_split       (web_en, indic)  Concatenate and cut on a fixed stride.
                                     Maximum utilisation; documents may be split
                                     across packs, and each fragment becomes its
                                     own attention segment.

  whole_doc_bestfit  (code)          Never interleave a fragment of one file with
                                     another file mid-sequence. Whole files are
                                     bin-packed (first-fit-decreasing, deterministic
                                     tie-break); over-length files are cut on
                                     sequence boundaries into standalone packs.
                                     Costs some padding, buys coherent files.

  prompt_masked      (math, eval_*)  One reasoning trace per pack, never merged
                                     with anything else, with the question tokens
                                     excluded from the loss. Lowest utilisation of
                                     the three -- and correct, because training the
                                     model to generate the question is not the task.

Every pack carries four aligned arrays and full provenance:

  input_ids     the tokens
  labels        next-token targets, -100 wherever loss must not be taken
  segment_ids   0 = padding, 1..k = which document a slot belongs to
  position_ids  restart at 0 for every segment

Attention is causal AND intra-segment: a slot may never attend to a different
document sharing its sequence. Both are enforced in one mask built here and
asserted in tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hashing import hash_array, hash_obj

IGNORE_INDEX = -100


@dataclass
class Segment:
    """One document (or document fragment) occupying a slice of a pack."""
    shard_id: str
    doc_id: str
    lane: str
    doc_start: int      # offset within the shard's token array
    doc_end: int
    slot_start: int     # offset within the pack
    slot_end: int
    prompt_tokens: int = 0   # leading tokens of THIS segment that bear no loss

    def to_dict(self) -> dict:
        return {"shard_id": self.shard_id, "doc_id": self.doc_id, "lane": self.lane,
                "doc_start": self.doc_start, "doc_end": self.doc_end,
                "slot_start": self.slot_start, "slot_end": self.slot_end,
                "prompt_tokens": self.prompt_tokens}


@dataclass
class Pack:
    pack_id: str
    lane: str
    policy: str
    input_ids: np.ndarray
    labels: np.ndarray
    segment_ids: np.ndarray
    position_ids: np.ndarray
    segments: list = field(default_factory=list)

    @property
    def n_real_tokens(self) -> int:
        return int((self.segment_ids > 0).sum())

    @property
    def n_loss_tokens(self) -> int:
        return int((self.labels != IGNORE_INDEX).sum())

    @property
    def content_hash(self) -> str:
        return hash_obj({
            "pack_id": self.pack_id, "lane": self.lane, "policy": self.policy,
            "input_ids": hash_array(self.input_ids), "labels": hash_array(self.labels),
            "segment_ids": hash_array(self.segment_ids),
            "position_ids": hash_array(self.position_ids),
            "segments": [s.to_dict() for s in self.segments],
        })

    def provenance(self) -> list:
        return [s.to_dict() for s in self.segments]


def build_attention_mask(segment_ids: np.ndarray) -> np.ndarray:
    """[L,L] boolean: causal within a segment, nothing across segments or into pad."""
    L = segment_ids.shape[0]
    causal = np.tril(np.ones((L, L), dtype=bool))
    same_seg = segment_ids[:, None] == segment_ids[None, :]
    real = segment_ids > 0
    return causal & same_seg & real[:, None] & real[None, :]


def _finalise(pack_id, lane, policy, seq_len, placed, tokens_by_shard) -> Pack:
    """Turn placed segments into the four aligned arrays."""
    input_ids = np.zeros(seq_len, dtype=np.int32)
    labels = np.full(seq_len, IGNORE_INDEX, dtype=np.int32)
    segment_ids = np.zeros(seq_len, dtype=np.int32)
    position_ids = np.zeros(seq_len, dtype=np.int32)

    for k, seg in enumerate(placed, start=1):
        toks = tokens_by_shard[seg.shard_id][seg.doc_start:seg.doc_end]
        s, e = seg.slot_start, seg.slot_end
        input_ids[s:e] = toks
        segment_ids[s:e] = k
        position_ids[s:e] = np.arange(e - s, dtype=np.int32)
        # next-token targets, strictly inside the segment: the final slot of a
        # segment gets no label, which is what stops a document predicting the
        # first token of an unrelated document that follows it in the sequence.
        if e - s >= 2:
            labels[s:e - 1] = toks[1:]
        # prompt masking: drop loss wherever the *target* lies inside the prompt
        if seg.prompt_tokens > 0:
            for i in range(s, e - 1):
                target_pos_in_seg = (i - s) + 1
                if target_pos_in_seg < seg.prompt_tokens:
                    labels[i] = IGNORE_INDEX

    return Pack(pack_id=pack_id, lane=lane, policy=policy, input_ids=input_ids,
                labels=labels, segment_ids=segment_ids, position_ids=position_ids,
                segments=placed)


def pack_lane(shards: list, lane: str, policy: str, seq_len: int,
              min_segment_tokens: int) -> list:
    """Deterministically build every pack for one lane."""
    tokens_by_shard = {s.shard_id: s.tokens for s in shards}
    packs: list = []

    def new_id() -> str:
        return f"{lane}-p{len(packs):05d}"

    # ---------------- concat_split -------------------------------------
    if policy == "concat_split":
        # a flat stream of (shard_id, doc_id, offset) cut on a fixed stride
        stream: list = []   # (shard_id, doc_id, start, end) pieces in order
        for s in shards:
            for d in s.docs:
                stream.append((s.shard_id, d.doc_id, d.start, d.end))
        cursor = 0                     # slot cursor within the current pack
        placed: list = []
        for shard_id, doc_id, start, end in stream:
            pos = start
            while pos < end:
                room = seq_len - cursor
                take = min(room, end - pos)
                placed.append(Segment(shard_id, doc_id, lane, pos, pos + take,
                                      cursor, cursor + take))
                cursor += take
                pos += take
                if cursor >= seq_len:
                    packs.append(_finalise(new_id(), lane, policy, seq_len, placed,
                                           tokens_by_shard))
                    placed, cursor = [], 0
        if placed and cursor >= min_segment_tokens:
            packs.append(_finalise(new_id(), lane, policy, seq_len, placed, tokens_by_shard))

    # ---------------- whole_doc_bestfit --------------------------------
    elif policy == "whole_doc_bestfit":
        small: list = []
        for s in shards:
            for d in s.docs:
                n = d.n_tokens
                if n > seq_len:
                    # cut an over-length file on sequence boundaries; each piece
                    # is a standalone pack so no file fragment shares a sequence
                    for off in range(0, n, seq_len):
                        a = d.start + off
                        b = min(d.start + off + seq_len, d.end)
                        if b - a < min_segment_tokens:
                            continue
                        seg = Segment(s.shard_id, d.doc_id, lane, a, b, 0, b - a)
                        packs.append(_finalise(new_id(), lane, policy, seq_len, [seg],
                                               tokens_by_shard))
                elif n >= min_segment_tokens:
                    small.append((s.shard_id, d.doc_id, d.start, d.end, n))
        # first-fit-decreasing with a deterministic tie-break on (shard, doc)
        small.sort(key=lambda t: (-t[4], t[0], t[1]))
        bins: list = []   # each: [used_slots, [segments]]
        for shard_id, doc_id, start, end, n in small:
            target = None
            for b in bins:
                if b[0] + n <= seq_len:
                    target = b
                    break
            if target is None:
                target = [0, []]
                bins.append(target)
            target[1].append(Segment(shard_id, doc_id, lane, start, end,
                                     target[0], target[0] + n))
            target[0] += n
        for used, placed in bins:
            packs.append(_finalise(new_id(), lane, policy, seq_len, placed, tokens_by_shard))

    # ---------------- prompt_masked -----------------------------------
    elif policy == "prompt_masked":
        for s in shards:
            for d in s.docs:
                n = min(d.n_tokens, seq_len)
                if n < min_segment_tokens:
                    continue
                prompt = min(d.prompt_len, n)
                # If the question alone fills the sequence budget, the pack would
                # carry no loss-bearing target at all: it would consume a full
                # slot of compute and teach nothing. Such documents are dropped
                # here rather than silently trained on (they are counted in the
                # lane's packing statistics as dropped_no_loss_target).
                if prompt >= n - 1:
                    continue
                seg = Segment(s.shard_id, d.doc_id, lane, d.start, d.start + n, 0, n,
                              prompt_tokens=prompt)
                packs.append(_finalise(new_id(), lane, policy, seq_len, [seg],
                                       tokens_by_shard))
    else:
        raise ValueError(f"unknown packing policy: {policy}")

    # A pack with no loss-bearing token is pure waste under every policy; refuse
    # to emit one so the scheduler can never spend a batch slot on it.
    return [p for p in packs if p.n_loss_tokens > 0]


@dataclass
class Batch:
    batch_id: int
    step: int
    input_ids: np.ndarray        # [B, L]
    labels: np.ndarray
    segment_ids: np.ndarray
    position_ids: np.ndarray
    pack_refs: list              # [(lane, pack_index, pack_id, pack_hash)]
    provenance: list             # per-pack list of segment dicts

    @property
    def lane_counts(self) -> dict:
        out: dict = {}
        for lane, _, _, _ in self.pack_refs:
            out[lane] = out.get(lane, 0) + 1
        return out

    @property
    def n_real_tokens(self) -> int:
        return int((self.segment_ids > 0).sum())

    @property
    def n_loss_tokens(self) -> int:
        return int((self.labels != IGNORE_INDEX).sum())

    @property
    def n_slots(self) -> int:
        return int(self.input_ids.size)

    @property
    def content_hash(self) -> str:
        """The identity used by resume and replay proofs. Commits to the tokens,
        the masks, the position ids AND the provenance -- so a batch cannot match
        by coincidence while having come from different documents."""
        return hash_obj({
            "batch_id": self.batch_id,
            "input_ids": hash_array(self.input_ids),
            "labels": hash_array(self.labels),
            "segment_ids": hash_array(self.segment_ids),
            "position_ids": hash_array(self.position_ids),
            "pack_refs": [[l, i, pid, ph] for l, i, pid, ph in self.pack_refs],
            "provenance": self.provenance,
        })


def assemble_batch(batch_id: int, step: int, chosen: list) -> Batch:
    """Stack chosen packs into a batch. `chosen` is [(lane, pack_index, Pack)]."""
    input_ids = np.stack([p.input_ids for _, _, p in chosen])
    labels = np.stack([p.labels for _, _, p in chosen])
    segment_ids = np.stack([p.segment_ids for _, _, p in chosen])
    position_ids = np.stack([p.position_ids for _, _, p in chosen])
    return Batch(
        batch_id=batch_id, step=step, input_ids=input_ids, labels=labels,
        segment_ids=segment_ids, position_ids=position_ids,
        pack_refs=[(lane, idx, p.pack_id, p.content_hash) for lane, idx, p in chosen],
        provenance=[p.provenance() for _, _, p in chosen],
    )
