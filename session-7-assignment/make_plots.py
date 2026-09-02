#!/usr/bin/env python3
"""Render static PNG charts from results/results.json.

The HTML report draws its charts with inline SVG + JavaScript, which GitHub
strips when it renders a page. These PNGs are the same figures in a form that
survives GitHub's markdown renderer, so the repository shows the evidence
without anyone having to download and open the report.

    python make_plots.py        # -> results/plots/*.png

Devanagari is romanised in the labels: matplotlib has no Indic font here, and a
row of tofu boxes would defeat the point of the picture.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results" / "plots"

INK = "#1b1b1f"
MUTED = "#6b6b76"
COLOR = {
    "raw": "#b3261e",        # the baseline being attacked
    "midpoint": "#e08b00",   # the control
    "sandhi": "#1f6f4a",     # the method
    "gold": "#4a5fb8",       # the ceiling
}
LABEL = {
    "raw": "raw\n(baseline)",
    "midpoint": "midpoint\n(control)",
    "sandhi": "sandhi\n(method)",
    "gold": "gold\n(ceiling)",
}
ORDER = ["raw", "midpoint", "sandhi", "gold"]


def style(ax, title, sub=None):
    # the subtitle grows upward from the axes, so the title has to step back by
    # one line height for every extra line of it
    lines = sub.count("\n") + 1 if sub else 0
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, loc="left",
                 pad=14 + 12 * max(0, lines - 1))
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8.5, color=MUTED,
                va="bottom", linespacing=1.35)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c9d1")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color="#e8e8ee", lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote results/plots/{name}")


# ---------------------------------------------------------------- figure 1
def plot_auc_by_position(d):
    """The headline. Non-initial is the column the whole study is about."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    w, x = 0.36, np.arange(len(ORDER))
    for off, key, ci_key, alpha in ((-w / 2, "initial", "auc_initial_ci", 0.40),
                                    (+w / 2, "non_initial", "auc_non_initial_ci", 1.0)):
        vals = [d["conditions"][c]["relatedness"]["by_position"][key]["auc_vs_unrelated"]
                for c in ORDER]
        ci = [d["statistics"][c][ci_key] for c in ORDER]
        err = np.array([[v - c["lo"] for v, c in zip(vals, ci)],
                        [c["hi"] - v for v, c in zip(vals, ci)]])
        ax.bar(x + off, vals, w, color=[COLOR[c] for c in ORDER], alpha=alpha,
               edgecolor="white", lw=1.2, zorder=3)
        ax.errorbar(x + off, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=1.2,
                    capsize=4, zorder=4)
        for xi, v, c in zip(x + off, vals, ci):          # label above the whisker cap
            ax.text(xi, c["hi"] + 0.018, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)

    ax.axhline(0.5, color=MUTED, ls=":", lw=1.2, zorder=2)
    ax.text(-0.62, 0.508, "chance", fontsize=8, color=MUTED, ha="left")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[c] for c in ORDER])
    ax.set_ylim(0.4, 1.10); ax.set_ylabel("ROC-AUC, related vs unrelated pairs")
    ax.set_xlim(-0.65, len(ORDER) - 0.35)
    handles = [Patch(facecolor=MUTED, alpha=0.40, edgecolor="white",
                     label="shared morpheme INITIAL  — the codec can already align these"),
               Patch(facecolor=MUTED, alpha=1.0, edgecolor="white",
                     label="shared morpheme NON-INITIAL  — it cannot")]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower left",
              bbox_to_anchor=(0.005, 0.005))
    style(ax, "Compound-to-compound retrieval AUC, split by morpheme position",
          "141 related vs 1300 unrelated compound pairs · whiskers = 95% percentile "
          "bootstrap, 2000 resamples · dp=32")
    save(fig, "auc_by_position.png")


# ---------------------------------------------------------------- figure 2
def plot_delta_forest(d):
    """The control. Midpoint's interval crosses zero; sandhi's does not."""
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    conds = ["midpoint", "sandhi", "gold"]
    y = np.arange(len(conds))[::-1]
    for yi, c in zip(y, conds):
        s = d["statistics"][c]["delta_vs_raw_non_initial"]
        ok = s["excludes_zero"]
        ax.plot([s["lo"], s["hi"]], [yi, yi], color=COLOR[c], lw=3,
                solid_capstyle="round", zorder=3)
        ax.plot([s["delta"]], [yi], "o", color=COLOR[c], ms=9, zorder=4,
                markeredgecolor="white", markeredgewidth=1.4)
        ax.text(s["hi"] + 0.012, yi, f"{s['delta']:+.3f}  [{s['lo']:+.3f}, {s['hi']:+.3f}]"
                f"   {'significant' if ok else 'NOT significant'}",
                va="center", fontsize=9, color=INK if ok else MUTED,
                fontweight="bold" if ok else "normal")
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.text(0.004, len(conds) - 0.55, "no effect", fontsize=8, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[c].replace("\n", " ") for c in conds])
    ax.set_xlim(-0.07, 0.62); ax.set_ylim(-0.6, len(conds) - 0.35)
    ax.set_xlabel("Δ AUC vs raw baseline (non-initial pairs)")
    ax.grid(axis="y", lw=0); ax.grid(axis="x", color="#e8e8ee", lw=0.8)
    style(ax, "The gain is morphological, not just “shorter pieces”",
          "The midpoint control gets identical truncation relief (0.00% bytes lost, 0 "
          "collisions) yet moves AUC by nothing.")
    ax.grid(axis="y", lw=0)
    save(fig, "delta_vs_raw.png")


# ---------------------------------------------------------------- figure 3
def plot_dp_sweep(d):
    """Truncation and misalignment are independent defects."""
    rows = d["dp_sweep"]["rows"]
    dps = d["dp_sweep"]["dps"]
    conds = ["raw", "midpoint", "sandhi"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.9))

    for c in conds:
        r = [x for x in rows if x["condition"] == c]
        a1.plot(dps, [x["truncation_rate"] * 100 for x in r], "o-", color=COLOR[c],
                lw=2, ms=5, label=LABEL[c].replace("\n", " "), zorder=3)
        a2.plot(dps, [x["auc_non_initial"] for x in r], "o-", color=COLOR[c],
                lw=2, ms=5, label=LABEL[c].replace("\n", " "), zorder=3)

    a1.set_xlabel("dp (positional budget, bytes)"); a1.set_ylabel("bytes lost to truncation (%)")
    a1.set_xticks(dps); a1.legend(frameon=False, fontsize=8.5)
    style(a1, "Truncation vanishes as dp grows", "arithmetic — a bigger window holds more bytes")

    a2.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    a2.set_xlabel("dp (positional budget, bytes)"); a2.set_ylabel("AUC, non-initial pairs")
    a2.set_xticks(dps); a2.set_ylim(0.45, 1.03); a2.legend(frameon=False, fontsize=8.5, loc="center right")
    raw48 = next(x for x in rows if x["dp"] == 48 and x["condition"] == "raw")
    a2.annotate(f"dp=48: 0.00% truncated,\nraw AUC still {raw48['auc_non_initial']:.3f}",
                xy=(48, raw48["auc_non_initial"]), xytext=(33, 0.53), fontsize=8.5,
                color=INK, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    style(a2, "Misalignment does not", "a bigger window never teaches the codec that a "
                                       "shifted morpheme is the same morpheme")
    fig.tight_layout()
    save(fig, "dp_sweep.png")


# ---------------------------------------------------------------- figure 4
def plot_roc(d):
    """The AUC numbers as the curves they actually are."""
    def roc(pos, neg):
        pos, neg = np.asarray(pos), np.asarray(neg)
        thr = np.unique(np.concatenate([pos, neg]))[::-1]
        tpr = [(pos >= t).mean() for t in thr]
        fpr = [(neg >= t).mean() for t in thr]
        return [0.0] + fpr + [1.0], [0.0] + tpr + [1.0]

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for c in ORDER:
        pc = d["pair_cosines"][c]
        f, t = roc(pc["non_initial"], pc["unrelated"])
        auc = d["conditions"][c]["relatedness"]["by_position"]["non_initial"]["auc_vs_unrelated"]
        ax.plot(f, t, color=COLOR[c], lw=2.2,
                label=f"{LABEL[c].replace(chr(10), ' ')}  AUC {auc:.3f}", zorder=3)
    ax.plot([0, 1], [0, 1], ls=":", color=MUTED, lw=1.2, zorder=2)
    ax.set_xlabel("false positive rate (unrelated pairs)")
    ax.set_ylabel("true positive rate (100 non-initial shared-morpheme pairs)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax, "ROC — non-initial shared morphemes",
          "rank-based, so it is immune to segmentation inflating every cosine")
    save(fig, "roc_non_initial.png")


# ---------------------------------------------------------------- figure 5
def plot_truncation_collisions(d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    x = np.arange(len(ORDER))
    tr = [d["conditions"][c]["truncation"]["truncation_rate"] * 100 for c in ORDER]
    co = [d["conditions"][c]["collisions"]["n_words_in_collision"] for c in ORDER]

    a1.bar(x, tr, 0.55, color=[COLOR[c] for c in ORDER], edgecolor="white", zorder=3)
    for xi, v in zip(x, tr):
        a1.text(xi, v + 0.12, f"{v:.2f}%", ha="center", fontsize=9, color=INK)
    a1.set_xticks(x); a1.set_xticklabels([LABEL[c] for c in ORDER])
    a1.set_ylim(0, 5.0); a1.set_ylabel("bytes lost (%)")
    style(a1, "Truncation at dp=32", "8 of 54 words lose their tail under raw encoding")

    a2.bar(x, co, 0.55, color=[COLOR[c] for c in ORDER], edgecolor="white", zorder=3)
    for xi, v in zip(x, co):
        a2.text(xi, v + 0.12, str(v), ha="center", fontsize=9, color=INK)
    a2.set_xticks(x); a2.set_xticklabels([LABEL[c] for c in ORDER])
    a2.set_ylim(0, 6.2); a2.set_ylabel("words mapping to a shared vector")
    style(a2, "Collisions at dp=32",
          "vishvavidyalaya / -yon / vishvavidyarthi → one identical vector")
    fig.tight_layout()
    save(fig, "truncation_collisions.png")


# ---------------------------------------------------------------- figure 6
def plot_script_floor(d):
    """The unplanned finding: raw Devanagari cosine measures script, not content."""
    fl = d["script_floor"]; ex = fl["example"]
    names = ["alaya\n(its OWN morpheme)"] + \
            [f"{t}\n(unrelated)" for t in ("kamalakamala", "sagaraparvata",
                                           "nagaraputra", "mitravayu")]
    keys = ["कमलकमल", "सागरपर्वत",
            "नगरपुत्र", "मित्रवायु"]
    vals = [ex["cos_to_own_morpheme"]] + [ex["cos_to_unrelated_words"][k] for k in keys]
    cols = [COLOR["raw"]] + [MUTED] * 4

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.55, color=cols, edgecolor="white", zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax.bar([len(vals)], [fl["latin_control"]["cos_devalaya_alaya"]], 0.55,
           color="#2f6f9f", edgecolor="white", zorder=3)
    ax.text(len(vals), 0.012, "0.000", ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x) + [len(vals)])
    ax.set_xticklabels(names + ["LATIN control\ncos(devalaya, alaya)"], fontsize=8.5)
    ax.set_ylim(0, 0.72); ax.set_ylabel("cosine with devalaya (raw, dp=32)")
    style(ax, "Raw Devanagari similarity measures script, not content",
          f"{fl['lead_byte_fraction']:.1%} of corpus bytes are E0/A4/A5 UTF-8 lead bytes, so any "
          "two words of the script\nagree in two of every three slots — and devalaya's own "
          "morpheme ranks BELOW unrelated words.")
    save(fig, "script_floor.png")


def main() -> int:
    d = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    print("rendering static plots from results/results.json")
    plot_auc_by_position(d)
    plot_delta_forest(d)
    plot_dp_sweep(d)
    plot_roc(d)
    plot_truncation_collisions(d)
    plot_script_floor(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
