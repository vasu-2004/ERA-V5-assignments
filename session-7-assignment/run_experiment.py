#!/usr/bin/env python3
"""
Kronecker codec x morphological segmentation -- complete analytical study.

    python run_experiment.py

No training, no gradients, no GPU. Everything below is codec arithmetic and
cosine similarity. Writes:

    results/results.json     every number, machine-readable
    results/report/index.html  the visual report
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from kron import report as report_mod
from kron.codec import DEFAULT_DP
from kron.dataset import COMPOUNDS, build_lexicon, summary
from kron.experiments import (bootstrap_auc_ci, build_vectors, dp_sweep,
                              measure_script_floor, measure_splitter, pair_cosines,
                              paired_auc_delta, run_condition)
from kron.sandhi import (FixedPointSplitter, GoldSplitter, MorfessorSplitter,
                         SandhiSplitter)

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results"


def main() -> int:
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    lex = build_lexicon()

    sandhi = SandhiSplitter(lex)
    conditions = {
        "raw": None,
        "sandhi": sandhi,
        "gold": GoldSplitter(COMPOUNDS),
        "midpoint": FixedPointSplitter(),
    }

    print("=" * 74)
    print("Kronecker codec (dp=%d) x training-free morphological segmentation" % DEFAULT_DP)
    print("=" * 74)

    ds = summary()
    print(f"\ndataset: {ds['n_compounds']} compounds, {ds['n_families']} morpheme families, "
          f"lexicon {ds['n_lexicon']} ({ds['n_distractors']} distractors)")
    print(f"pairs:   {ds['n_related_initial']} shared-initial, "
          f"{ds['n_related_non_initial']} shared-non-initial, "
          f"{ds['n_unrelated_pairs']} unrelated")

    # -- which analyzers were actually available -------------------------
    availability = {
        "sanskrit_heritage_engine": {"available": False,
                                     "reason": "sanskrit.inria.fr unreachable from this environment"},
        "indic_nlp_library_morfessor": {"available": MorfessorSplitter.available(),
                                        "reason": "installs, but the indic_nlp_resources "
                                                  "morfessor bundle download returns 403"},
        "inverse_sandhi_rules": {"available": True,
                                 "reason": "implemented here; rule table + lexicon, no training"},
    }
    print("\nsegmenter availability:")
    for k, v in availability.items():
        print(f"  {'OK ' if v['available'] else 'NO '} {k}: {v['reason']}")

    # -- why raw cosines sit so high in this script -----------------------
    floor = measure_script_floor()
    ex = floor["example"]
    print(f"\nUTF-8 script floor: {floor['lead_byte_fraction']:.1%} of corpus bytes are "
          f"E0/A4/A5 lead bytes")
    print(f"  cos({ex['compound']}, {ex['own_non_initial_morpheme']}) = {ex['cos_to_own_morpheme']:.4f}  "
          f"vs best unrelated = {ex['best_unrelated']:.4f}  ->  own morpheme wins: "
          f"{ex['own_morpheme_beats_unrelated']}")
    print(f"  latin control cos(devalaya, alaya) = {floor['latin_control']['cos_devalaya_alaya']:.4f}")

    # -- E5 splitter quality ---------------------------------------------
    splitter_eval = measure_splitter(sandhi)
    print(f"\nsplitter exact-match vs gold: {splitter_eval['n_correct']}/"
          f"{splitter_eval['n_words']} = {splitter_eval['exact_match']:.1%}")
    for f in splitter_eval["failures"]:
        print(f"    miss: {f['word']}  gold={'|'.join(f['gold'])}  got={'|'.join(f['predicted'])}")

    # -- E1..E4 per condition ---------------------------------------------
    results, cosines = {}, {}
    for name, sp in conditions.items():
        results[name] = run_condition(name, sp)
        cosines[name] = pair_cosines(build_vectors(sp))

    print(f"\n{'condition':10s} {'trunc%':>7s} {'wtrunc':>7s} {'collide':>8s} "
          f"{'AUC all':>8s} {'AUC init':>9s} {'AUC non-init':>13s}")
    for name in conditions:
        r, rel = results[name], results[name]["relatedness"]
        print(f"{name:10s} {r['truncation']['truncation_rate']*100:6.2f}% "
              f"{r['truncation']['words_truncated']:7d} "
              f"{r['collisions']['n_words_in_collision']:8d} "
              f"{rel['auc_overall']:8.4f} "
              f"{rel['by_position']['initial']['auc_vs_unrelated']:9.4f} "
              f"{rel['by_position']['non_initial']['auc_vs_unrelated']:13.4f}")

    # -- confidence intervals + paired deltas against the raw baseline ----
    print("\nbootstrap 95% CIs (2000 resamples) and paired deltas vs raw:")
    stats = {}
    for name in conditions:
        c = cosines[name]
        entry = {
            "auc_overall_ci": bootstrap_auc_ci(c["related_all"], c["unrelated"]),
            "auc_non_initial_ci": bootstrap_auc_ci(c["non_initial"], c["unrelated"]),
            "auc_initial_ci": bootstrap_auc_ci(c["initial"], c["unrelated"]),
        }
        if name != "raw":
            entry["delta_vs_raw_non_initial"] = paired_auc_delta(
                cosines["raw"]["non_initial"], cosines["raw"]["unrelated"],
                c["non_initial"], c["unrelated"])
            entry["delta_vs_raw_overall"] = paired_auc_delta(
                cosines["raw"]["related_all"], cosines["raw"]["unrelated"],
                c["related_all"], c["unrelated"])
        stats[name] = entry
        ci = entry["auc_non_initial_ci"]
        line = f"  {name:10s} AUC(non-initial) = " \
               f"{results[name]['relatedness']['by_position']['non_initial']['auc_vs_unrelated']:.4f} " \
               f"[{ci['lo']:.4f}, {ci['hi']:.4f}]"
        if name != "raw":
            d = entry["delta_vs_raw_non_initial"]
            line += f"   delta vs raw = {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]" \
                    f" {'significant' if d['excludes_zero'] else 'NOT significant'}"
        print(line)

    # -- dp sweep ----------------------------------------------------------
    print("\ndp sweep (positional budget):")
    sweep = dp_sweep({"raw": None, "sandhi": sandhi, "midpoint": FixedPointSplitter()},
                     [16, 24, 32, 48, 64])
    print(f"  {'dp':>4s} {'condition':10s} {'trunc%':>7s} {'collide':>8s} {'AUC non-init':>13s}")
    for row in sweep["rows"]:
        print(f"  {row['dp']:4d} {row['condition']:10s} {row['truncation_rate']*100:6.2f}% "
              f"{row['words_in_collision']:8d} {row['auc_non_initial']:13.4f}")

    bundle = {
        "meta": {
            "title": "Morphological segmentation as a training-free prior for the "
                     "Kronecker byte-position codec",
            "dp": DEFAULT_DP,
            "training_performed": False,
            "repo_url": "https://github.com/vasu-2004/ERA-V5-assignments/tree/"
                        "claude/neural-network-fundamentals-9xgeps/session-7-assignment",
            "elapsed_s": round(time.time() - t0, 3),
        },
        "dataset": ds,
        "segmenter_availability": availability,
        "script_floor": floor,
        "splitter_evaluation": splitter_eval,
        "conditions": results,
        "statistics": stats,
        "dp_sweep": sweep,
        "pair_cosines": {k: {kk: [round(x, 6) for x in vv] for kk, vv in v.items()}
                         for k, v in cosines.items()},
    }
    (OUT / "results.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    report_mod.build(bundle, OUT / "report")

    print(f"\nwrote {OUT/'results.json'}")
    print(f"wrote {OUT/'report'/'index.html'}")
    print(f"done in {bundle['meta']['elapsed_s']}s -- no training was performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
