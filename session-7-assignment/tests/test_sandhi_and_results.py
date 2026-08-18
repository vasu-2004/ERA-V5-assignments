"""Tests for the sandhi engine, the dataset, and the reported conclusions."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kron.dataset import (COMPOUNDS, build_lexicon, related_pairs, summary,
                          unrelated_pairs)
from kron.experiments import (measure_splitter, pair_cosines, build_vectors,
                              paired_auc_delta, roc_auc, run_condition)
from kron.sandhi import (FixedPointSplitter, GoldSplitter, MorfessorSplitter,
                         SandhiSplitter, apply_sandhi)


@pytest.fixture(scope="module")
def splitter():
    return SandhiSplitter(build_lexicon())


# ------------------------------------------------------------------- sandhi
@pytest.mark.parametrize("s1,s2,expected", [
    ("देव", "आलय", "देवालय"),      # a + ā -> ā
    ("शिव", "आलय", "शिवालय"),
    ("महा", "ईश", "महेश"),          # ā + ī -> e
    ("गण", "ईश", "गणेश"),           # a + ī -> e
    ("सूर्य", "उदय", "सूर्योदय"),   # a + u -> o
    ("महा", "उत्सव", "महोत्सव"),    # ā + u -> o
    ("महा", "आत्मा", "महात्मा"),    # ā + ā -> ā
    ("राज", "कुमार", "राजकुमार"),   # no initial vowel -> plain join
])
def test_forward_sandhi_rules(s1, s2, expected):
    assert apply_sandhi(s1, s2) == expected


def test_splitter_inverts_its_own_forward_rules(splitter):
    """Anything the rule table can build, it should be able to take apart."""
    for word, parts in COMPOUNDS.items():
        if len(parts) != 2:
            continue
        if apply_sandhi(parts[0], parts[1]) != word:
            continue                      # unmodelled sandhi class; covered below
        assert splitter.split(word).parts == parts, word


def test_splitter_accuracy_is_reported_honestly(splitter):
    ev = measure_splitter(splitter)
    assert 0.85 <= ev["exact_match"] < 1.0, \
        "a perfect score would suggest the lexicon is leaking the answer"
    assert ev["failures"], "the known unmodelled sandhi classes must still fail"


def test_known_failures_are_the_unmodelled_sandhi_classes(splitter):
    """The misses must be the linguistically expected ones, not random."""
    expected_failures = {"रामायण", "अत्यन्त", "इत्यादि", "स्वागत"}
    got = {f["word"] for f in measure_splitter(splitter)["failures"]}
    assert got == expected_failures


def test_splitter_declines_rather_than_guessing_out_of_lexicon(splitter):
    for junk in ["ज़्ज़्ज़्", "qqqq", "कखगघङचछ"]:
        assert splitter.split(junk).parts == [junk]


def test_recursive_decomposition_handles_three_member_compounds(splitter):
    assert splitter.split("महाविद्यालय").parts == ["महा", "विद्या", "आलय"]
    assert splitter.split("उपराष्ट्रपति").parts == ["उप", "राष्ट्र", "पति"]


def test_lexicon_contains_real_distractors():
    ds = summary()
    assert ds["n_distractors"] >= 50
    lex = build_lexicon()
    # no compound of the evaluation set may sit in the lexicon as a whole word,
    # or the splitter could "solve" it by refusing to split
    assert not (set(COMPOUNDS) & lex)


def test_morfessor_adapter_reports_unavailability_rather_than_faking_it():
    assert MorfessorSplitter.available() is False


# ------------------------------------------------------------------ dataset
def test_pair_classes_are_disjoint_and_populated():
    rel = {(p["w1"], p["w2"]) for p in related_pairs()}
    unrel = {(p["w1"], p["w2"]) for p in unrelated_pairs()}
    assert not (rel & unrel)
    ds = summary()
    assert ds["n_related_initial"] >= 20 and ds["n_related_non_initial"] >= 50


def test_related_pairs_really_share_a_morpheme():
    for p in related_pairs():
        assert p["morpheme"] in COMPOUNDS[p["w1"]]
        assert p["morpheme"] in COMPOUNDS[p["w2"]]


def test_unrelated_pairs_share_nothing():
    for p in unrelated_pairs():
        assert not set(COMPOUNDS[p["w1"]]) & set(COMPOUNDS[p["w2"]])


# ------------------------------------------------------------------- metrics
def test_roc_auc_matches_known_values():
    assert roc_auc([1, 2, 3], [0, 0, 0]) == pytest.approx(1.0)
    assert roc_auc([0, 0, 0], [1, 2, 3]) == pytest.approx(0.0)
    assert roc_auc([1, 1], [1, 1]) == pytest.approx(0.5)      # all ties


# --------------------------------------------------------------- conclusions
@pytest.fixture(scope="module")
def conditions(splitter):
    out = {}
    for name, sp in {"raw": None, "sandhi": splitter,
                     "gold": GoldSplitter(COMPOUNDS),
                     "midpoint": FixedPointSplitter()}.items():
        out[name] = {"result": run_condition(name, sp),
                     "cos": pair_cosines(build_vectors(sp))}
    return out


def test_raw_codec_is_much_weaker_on_non_initial_morphemes(conditions):
    r = conditions["raw"]["result"]["relatedness"]["by_position"]
    assert r["initial"]["auc_vs_unrelated"] - r["non_initial"]["auc_vs_unrelated"] > 0.1


def test_segmentation_improves_non_initial_retrieval_significantly(conditions):
    d = paired_auc_delta(conditions["raw"]["cos"]["non_initial"],
                         conditions["raw"]["cos"]["unrelated"],
                         conditions["sandhi"]["cos"]["non_initial"],
                         conditions["sandhi"]["cos"]["unrelated"], n_boot=600)
    assert d["delta"] > 0.15 and d["excludes_zero"]


def test_midpoint_control_does_not_reproduce_the_gain(conditions):
    """The heart of the argument: matched truncation relief, no real gain."""
    ctl = conditions["midpoint"]["result"]
    san = conditions["sandhi"]["result"]
    assert ctl["truncation"]["truncation_rate"] == san["truncation"]["truncation_rate"] == 0.0
    d = paired_auc_delta(conditions["raw"]["cos"]["non_initial"],
                         conditions["raw"]["cos"]["unrelated"],
                         conditions["midpoint"]["cos"]["non_initial"],
                         conditions["midpoint"]["cos"]["unrelated"], n_boot=600)
    assert not d["excludes_zero"], "the meaningless-seam control should not be significant"


def test_segmentation_removes_truncation_and_collisions(conditions):
    raw, san = conditions["raw"]["result"], conditions["sandhi"]["result"]
    assert raw["truncation"]["bytes_lost"] > 0 and san["truncation"]["bytes_lost"] == 0
    assert raw["collisions"]["n_words_in_collision"] > 0
    assert san["collisions"]["n_words_in_collision"] == 0


def test_rule_splitter_approaches_the_gold_ceiling(conditions):
    g = conditions["gold"]["result"]["relatedness"]["by_position"]["non_initial"]["auc_vs_unrelated"]
    s = conditions["sandhi"]["result"]["relatedness"]["by_position"]["non_initial"]["auc_vs_unrelated"]
    assert abs(g - s) < 0.05


# ------------------------------------------------------------- artifact check
@pytest.mark.skipif(not (ROOT / "results" / "results.json").exists(),
                    reason="run `python run_experiment.py` first")
def test_published_results_are_internally_consistent():
    b = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    assert b["meta"]["training_performed"] is False
    raw = b["conditions"]["raw"]
    san = b["conditions"]["sandhi"]
    assert raw["collisions"]["n_words_in_collision"] > san["collisions"]["n_words_in_collision"]
    assert (san["relatedness"]["by_position"]["non_initial"]["auc_vs_unrelated"]
            > raw["relatedness"]["by_position"]["non_initial"]["auc_vs_unrelated"])
    # the report page must exist and embed the same run
    html = (ROOT / "results" / "report" / "index.html").read_text(encoding="utf-8")
    assert "__DATA__" not in html and "Kronecker" in html
