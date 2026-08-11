"""Shared fixtures: a complete miniature system built in a temp directory.

The tests deliberately do NOT read submission_artifacts/. They stand up their
own tokenizer, shards, packs and trainer from the same corpus, so they verify
the implementation rather than the output of one particular demo run.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import dataclasses
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tdes.config import (LANES, ModelConfig, OpusConfig, PackingConfig,  # noqa: E402
                         RunConfig, TokenizerConfig)
from tdes.firewall import Firewall  # noqa: E402
from tdes.runlog import RunLog  # noqa: E402
from tdes.shards import ShardStore, build_shards  # noqa: E402
from tdes.tokenizer import Tokenizer  # noqa: E402
from tdes.trainer import Trainer  # noqa: E402

CORPUS = ROOT / "corpus_data"


def tiny_config(**over) -> RunConfig:
    base = RunConfig(
        run_id="test-run",
        total_steps=12,
        checkpoint_every=4,
        crash_at_step=9,
        replay_interval=(4, 8),
        fork_from_step=4,
        fork_steps=3,
        tokenizer=TokenizerConfig(vocab_size=1024),
        packing=PackingConfig(seq_len=96, batch_size=3, min_segment_tokens=8),
        model=ModelConfig(d_model=32, n_heads=2, n_layers=2, d_ff=48, seed=11),
        opus=OpusConfig(warmup_steps=2, max_candidate_scans=16),
    )
    return dataclasses.replace(base, **over) if over else base


@pytest.fixture(scope="session")
def small_corpus(tmp_path_factory):
    """A trimmed copy of the real corpus: same structure, far fewer documents."""
    d = tmp_path_factory.mktemp("corpus")
    limits = {"web_en": 24, "indic": 20, "code": 20, "math": 30,
              "eval_math": 8, "val_web": 6}
    for lane, n in limits.items():
        src = CORPUS / f"{lane}.jsonl"
        rows = [l for l in src.open(encoding="utf-8")][:n]
        (d / f"{lane}.jsonl").write_text("".join(rows), encoding="utf-8")
    idx = json.loads((CORPUS / "corpus_index.json").read_text(encoding="utf-8"))
    (d / "corpus_index.json").write_text(json.dumps(idx), encoding="utf-8")
    return d


@pytest.fixture(scope="session")
def tokenizer(small_corpus):
    texts = []
    for lane in LANES:
        p = small_corpus / f"{lane}.jsonl"
        if p.exists():
            texts.extend(json.loads(l)["text"] for l in p.open(encoding="utf-8"))
    return Tokenizer.train(texts, 1024, ("<pad>", "<bos>", "<eos>", "<sep>"))


@pytest.fixture(scope="session")
def built(tmp_path_factory, small_corpus, tokenizer):
    """Shards + manifests + firewall, built once for the whole test session."""
    art = tmp_path_factory.mktemp("artifacts")
    store = ShardStore(art / "shards", art / "manifests")
    manifests = build_shards(small_corpus, tokenizer, LANES, store)
    firewall = Firewall(LANES, tokenizer, small_corpus)
    return {"artifacts": art, "store": store, "manifests": manifests,
            "firewall": firewall, "corpus": small_corpus}


@pytest.fixture
def trainer_factory(built, tokenizer, tmp_path):
    """Builds fresh trainers writing into a per-test directory."""
    def make(cfg=None, suffix="", run_id=None, art=None):
        cfg = cfg or tiny_config()
        art = art or tmp_path
        log = RunLog(art / f"test{suffix or '_main'}.log", echo=False)
        return Trainer(cfg, built["store"], tokenizer, built["firewall"],
                       art, log, ledger_suffix=suffix, run_id=run_id or cfg.run_id)
    return make
