"""Finite-difference verification of the hand-written backward pass.

The recovery and fork proofs rest on the training step being deterministic and
correct. Hand-derived gradients are the most plausible place for a silent error,
so they are checked against central differences rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pytest

from tdes.config import ModelConfig
from tdes.model import Transformer
from tdes.packing import IGNORE_INDEX

PARAMS = ["Wte", "Wpe", "Whead", "ln_f_g", "ln_f_b",
          "l0_Wq", "l0_Wk", "l0_Wv", "l0_Wo", "l0_ln1_g", "l0_ln2_b",
          "l0_W1", "l0_b1", "l0_W2", "l0_b2", "l1_Wq", "l1_Wo", "l1_W1", "l1_W2"]


@pytest.fixture(scope="module")
def setup():
    cfg = ModelConfig(d_model=16, n_heads=2, n_layers=2, d_ff=24, seed=5)
    V = 32
    model = Transformer(cfg, V)
    rng = np.random.default_rng(0)
    B, L = 2, 8
    input_ids = rng.integers(0, V, (B, L))
    segment_ids = np.array([[1, 1, 1, 2, 2, 2, 0, 0],
                            [1, 1, 2, 2, 2, 2, 2, 0]])
    position_ids = np.zeros((B, L), dtype=int)
    for b in range(B):
        seen: dict = {}
        for i in range(L):
            s = int(segment_ids[b, i])
            if s == 0:
                continue
            position_ids[b, i] = seen.get(s, 0)
            seen[s] = seen.get(s, 0) + 1
    labels = np.full((B, L), IGNORE_INDEX)
    for b in range(B):
        for i in range(L - 1):
            if segment_ids[b, i] != 0 and segment_ids[b, i] == segment_ids[b, i + 1]:
                labels[b, i] = input_ids[b, i + 1]
    return model, input_ids, position_ids, segment_ids, labels


def test_some_positions_bear_loss(setup):
    _, _, _, _, labels = setup
    assert (labels != IGNORE_INDEX).sum() > 0


def test_padding_and_boundaries_are_excluded(setup):
    _, _, _, segment_ids, labels = setup
    assert (labels[segment_ids == 0] == IGNORE_INDEX).all()


@pytest.mark.parametrize("name", PARAMS)
def test_gradient_matches_finite_difference(setup, name):
    model, input_ids, position_ids, segment_ids, labels = setup

    def loss_of():
        _, out, _ = model.forward(input_ids, position_ids, segment_ids, labels)
        return out["loss"]

    _, _, cache = model.forward(input_ids, position_ids, segment_ids, labels)
    grads = model.backward(cache)

    rng = np.random.default_rng(abs(hash(name)) % (2 ** 31))
    flat = model.p[name].reshape(-1)
    gflat = grads[name].reshape(-1)
    idxs = rng.choice(flat.size, size=min(5, flat.size), replace=False)
    eps = 1e-6
    for idx in idxs:
        orig = flat[idx]
        flat[idx] = orig + eps
        lp = loss_of()
        flat[idx] = orig - eps
        lm = loss_of()
        flat[idx] = orig
        numeric = (lp - lm) / (2 * eps)
        analytic = gflat[idx]
        abs_err = abs(numeric - analytic)
        denom = max(abs(numeric) + abs(analytic), 1e-9)
        # A near-zero component is dominated by finite-difference noise, so an
        # absolute floor is required alongside the relative bound.
        assert abs_err / denom < 1e-4 or abs_err < 1e-7, (
            f"{name}[{idx}]: analytic {analytic:.3e} vs numeric {numeric:.3e}")


def test_forward_is_deterministic(setup):
    model, input_ids, position_ids, segment_ids, labels = setup
    a = model.forward(input_ids, position_ids, segment_ids, labels)[1]["loss"]
    b = model.forward(input_ids, position_ids, segment_ids, labels)[1]["loss"]
    assert a == b


def test_padding_cannot_change_the_loss(setup):
    """A padded slot must be inert: changing its token id must not move the loss."""
    model, input_ids, position_ids, segment_ids, labels = setup
    before = model.forward(input_ids, position_ids, segment_ids, labels)[1]["loss"]
    altered = input_ids.copy()
    pads = np.argwhere(segment_ids == 0)
    assert pads.size, "fixture has no padding to test with"
    for b, i in pads:
        altered[b, i] = (altered[b, i] + 7) % 32
    after = model.forward(altered, position_ids, segment_ids, labels)[1]["loss"]
    assert before == after


def test_a_document_cannot_influence_another_in_the_same_sequence(setup):
    """The block-diagonal mask must actually isolate documents."""
    model, input_ids, position_ids, segment_ids, labels = setup
    base = model.forward(input_ids, position_ids, segment_ids, labels)[1]["per_pos_loss"]
    altered = input_ids.copy()
    # perturb every token of segment 2 in row 0
    tgt = (segment_ids[0] == 2)
    altered[0, tgt] = (altered[0, tgt] + 3) % 32
    new = model.forward(altered, position_ids, segment_ids, labels)[1]["per_pos_loss"]
    seg1 = (segment_ids[0] == 1)
    assert np.allclose(base[0, seg1], new[0, seg1]), \
        "changing document 2 altered the loss of document 1 in the same sequence"
