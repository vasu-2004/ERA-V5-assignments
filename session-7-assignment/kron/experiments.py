"""The analytical comparison. No training, no gradients -- codec math only.

Five measurements, and one control that decides how to read them:

  E1 truncation      bytes lost at dp=32, raw vs segmented
  E2 collisions      distinct words mapping to an identical codec vector
  E3 constituents    does a compound retrieve its own morphemes?  (CIRCULAR for
                     segmented conditions -- reported, but never the headline)
  E4 relatedness     do two DIFFERENT compounds sharing a morpheme sit closer
                     than two unrelated ones?  Split by whether the shared
                     morpheme is initial or non-initial. This is the headline,
                     and nothing in it is built from the thing it is scored on.
  E5 splitter        exact-match accuracy of the training-free analyzer

  CONTROL            the midpoint splitter cuts every word in half at a
                     linguistically meaningless seam. It receives the same
                     truncation relief as a real segmentation, so any gain it
                     does NOT reproduce is attributable to morphology rather
                     than to merely using shorter pieces.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .codec import DEFAULT_DP, cosine, encode, encode_segmented, fingerprint
from .dataset import (COMPOUNDS, CONSTITUENTS, build_lexicon, related_pairs,
                      unrelated_pairs)


def roc_auc(pos: list, neg: list) -> float:
    """AUC via the rank statistic: P(a random positive scores above a random
    negative), ties counted as half. Threshold-free, so it compares conditions
    whose cosines live on different scales."""
    if not pos or not neg:
        return float("nan")
    scores = [(s, 1) for s in pos] + [(s, 0) for s in neg]
    scores.sort(key=lambda t: t[0])
    ranks, i, n = {}, 0, len(scores)
    while i < n:                                   # average ranks within ties
        j = i
        while j + 1 < n and scores[j + 1][0] == scores[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(ranks[k] for k in range(n) if scores[k][1] == 1)
    npos, nneg = len(pos), len(neg)
    return (rank_sum - npos * (npos + 1) / 2.0) / (npos * nneg)


def build_vectors(splitter, dp: int = DEFAULT_DP) -> dict:
    """Encode every compound under one segmentation policy.

    `splitter=None` means the raw baseline: encode the whole word and let the
    codec truncate it at dp bytes.
    """
    out = {}
    for word in COMPOUNDS:
        if splitter is None:
            enc = encode(word, dp)
            parts = [word]
        else:
            parts = splitter.split(word).parts
            enc = encode_segmented(parts, dp)
        out[word] = {"enc": enc, "parts": parts}
    return out


# --------------------------------------------------------------- E1 truncation
def measure_truncation(vecs: dict) -> dict:
    total = sum(v["enc"].n_bytes_total for v in vecs.values())
    lost = sum(v["enc"].n_bytes_lost for v in vecs.values())
    affected = [w for w, v in vecs.items() if v["enc"].truncated]
    per_word = {w: {"bytes": v["enc"].n_bytes_total, "lost": v["enc"].n_bytes_lost,
                    "parts": v["parts"]}
                for w, v in vecs.items()}
    return {
        "bytes_total": total, "bytes_lost": lost,
        "truncation_rate": round(lost / max(total, 1), 6),
        "words_truncated": len(affected), "n_words": len(vecs),
        "word_truncation_rate": round(len(affected) / max(len(vecs), 1), 6),
        "truncated_words": sorted(affected),
        "per_word": per_word,
    }


# ---------------------------------------------------------------- E2 collisions
def measure_collisions(vecs: dict) -> dict:
    groups = defaultdict(list)
    for w, v in vecs.items():
        groups[fingerprint(v["enc"].vector)].append(w)
    colliding = {k: sorted(ws) for k, ws in groups.items() if len(ws) > 1}
    n_in_collision = sum(len(ws) for ws in colliding.values())
    return {
        "n_words": len(vecs),
        "n_distinct_vectors": len(groups),
        "n_collision_groups": len(colliding),
        "n_words_in_collision": n_in_collision,
        "collision_rate": round(n_in_collision / max(len(vecs), 1), 6),
        "groups": sorted(colliding.values()),
    }


# ------------------------------------------------------------- E3 constituents
def measure_constituent_retrieval(vecs: dict, dp: int = DEFAULT_DP, k: int = 5) -> dict:
    """Rank the whole lexicon by cosine to each compound; is its own morphology
    near the top?

    NOTE the asymmetry, which is why this is not the headline: under a segmented
    condition the compound vector is a mean of its part vectors, so its
    constituents rank highly *by construction*. It is reported because the raw
    number is still informative (it shows how little of a compound's morphology
    the raw codec surfaces at all), not as evidence of a discovery.
    """
    lexicon = sorted(build_lexicon())
    lex_vecs = {m: encode(m, dp).vector for m in lexicon}

    recalls, rrs = [], []
    per_word = {}
    for word, v in vecs.items():
        gold = [m for m in COMPOUNDS[word] if m in lex_vecs]
        if not gold:
            continue
        sims = sorted(((cosine(v["enc"].vector, lv), m) for m, lv in lex_vecs.items()),
                      reverse=True)
        ranked = [m for _, m in sims]
        topk = set(ranked[:k])
        rec = len(topk & set(gold)) / len(gold)
        best_rank = min((ranked.index(m) + 1 for m in gold), default=None)
        recalls.append(rec)
        rrs.append(1.0 / best_rank if best_rank else 0.0)
        per_word[word] = {"gold": gold, "top5": ranked[:5],
                          "recall_at_k": round(rec, 4), "best_rank": best_rank}
    return {
        "k": k,
        "mean_recall_at_k": round(float(np.mean(recalls)), 6) if recalls else 0.0,
        "mrr": round(float(np.mean(rrs)), 6) if rrs else 0.0,
        "n_evaluated": len(recalls),
        "lexicon_size": len(lexicon),
        "per_word": per_word,
        "caveat": "segmented conditions are favoured by construction; see E4 for "
                  "the non-circular comparison",
    }


# -------------------------------------------------------------- E4 relatedness
def measure_relatedness(vecs: dict) -> dict:
    """Compound-to-compound similarity: the non-circular test.

    Neither vector in a pair is built from the other, so this measures whether
    the representation actually places morphologically related words together.
    The split by `position` is the crux: the codec ties every byte to an
    absolute offset, so a shared INITIAL morpheme already aligns, while a shared
    NON-INITIAL one sits at different offsets in the two words and cannot align
    without segmentation.
    """
    rel, unrel = related_pairs(), unrelated_pairs()

    def cos_of(p):
        return cosine(vecs[p["w1"]]["enc"].vector, vecs[p["w2"]]["enc"].vector)

    rel_scored = [{**p, "cos": cos_of(p)} for p in rel if p["w1"] in vecs and p["w2"] in vecs]
    unrel_scored = [{**p, "cos": cos_of(p)} for p in unrel if p["w1"] in vecs and p["w2"] in vecs]
    neg = [p["cos"] for p in unrel_scored]

    by_pos = {}
    for pos in ("initial", "non_initial", "mixed"):
        sel = [p["cos"] for p in rel_scored if p["position"] == pos]
        by_pos[pos] = {
            "n_pairs": len(sel),
            "mean_cos": round(float(np.mean(sel)), 6) if sel else None,
            "auc_vs_unrelated": round(roc_auc(sel, neg), 6) if sel else None,
            "separation": round(float(np.mean(sel)) - float(np.mean(neg)), 6) if sel and neg else None,
        }

    all_rel = [p["cos"] for p in rel_scored]
    return {
        "n_related": len(rel_scored), "n_unrelated": len(unrel_scored),
        "mean_cos_related": round(float(np.mean(all_rel)), 6) if all_rel else None,
        "mean_cos_unrelated": round(float(np.mean(neg)), 6) if neg else None,
        "auc_overall": round(roc_auc(all_rel, neg), 6) if all_rel else None,
        "separation_overall": round(float(np.mean(all_rel)) - float(np.mean(neg)), 6)
                              if all_rel and neg else None,
        "by_position": by_pos,
        "examples": sorted(rel_scored, key=lambda p: -p["cos"])[:8],
    }


def measure_script_floor(dp: int = DEFAULT_DP) -> dict:
    """Why raw cosines between Devanagari words never approach zero.

    Every Devanagari codepoint encodes as E0 A4 xx or E0 A5 xx, so two of every
    three byte slots agree between ANY two words of the script. That puts a high
    similarity floor under the raw codec which has nothing to do with meaning,
    and it is the reason every claim in this study uses rank-based AUC rather
    than a cosine threshold.

    The Latin control isolates the effect: with no shared lead bytes, a shifted
    morpheme aligns with nothing and the cosine is exactly zero.
    """
    corpus = "".join(COMPOUNDS)
    raw = corpus.encode("utf-8")
    lead = sum(1 for b in raw if b in (0xE0, 0xA4, 0xA5))

    compound, morpheme = "देवालय", "आलय"
    c_vec = encode(compound, dp).vector
    own = cosine(c_vec, encode(morpheme, dp).vector)
    distractors = ["कमलकमल", "नगरपुत्र", "मित्रवायु", "सागरपर्वत"]
    unrel = {w: round(cosine(c_vec, encode(w, dp).vector), 6) for w in distractors}

    return {
        "lead_byte_fraction": round(lead / max(len(raw), 1), 6),
        "n_bytes_scanned": len(raw),
        "example": {
            "compound": compound, "own_non_initial_morpheme": morpheme,
            "cos_to_own_morpheme": round(own, 6),
            "cos_to_unrelated_words": unrel,
            "best_unrelated": round(max(unrel.values()), 6),
            "own_morpheme_beats_unrelated": bool(own > max(unrel.values())),
        },
        "latin_control": {
            "cos_devalaya_alaya": round(
                cosine(encode("devalaya", dp).vector, encode("alaya", dp).vector), 6),
            "note": "exactly 0: with no shared lead bytes, a shifted morpheme aligns "
                    "with nothing -- positional rigidity in its pure form",
        },
    }


def bootstrap_auc_ci(pos: list, neg: list, n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for an AUC.

    With ~100 related pairs a point estimate alone cannot say whether two
    conditions really differ, so every headline AUC is reported with an interval.
    Resampling is stratified over the two classes, which is the standard scheme
    for a rank statistic like this.
    """
    if not pos or not neg:
        return {"lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    pos_a, neg_a = np.asarray(pos), np.asarray(neg)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        p = rng.choice(pos_a, size=pos_a.size, replace=True)
        n = rng.choice(neg_a, size=neg_a.size, replace=True)
        boots[i] = roc_auc(list(p), list(n))
    return {"lo": round(float(np.percentile(boots, 100 * alpha / 2)), 4),
            "hi": round(float(np.percentile(boots, 100 * (1 - alpha / 2))), 4),
            "n_boot": n_boot}


def paired_auc_delta(pos_a: list, neg_a: list, pos_b: list, neg_b: list,
                     n_boot: int = 2000, seed: int = 1) -> dict:
    """Bootstrap CI for (AUC_b - AUC_a) over the SAME pairs.

    The two conditions score identical word pairs, so the same resampled index
    set is applied to both. That pairing is what makes the interval a statement
    about the difference rather than about two independent estimates.
    """
    if not pos_a or not neg_a:
        return {"delta": None}
    rng = np.random.default_rng(seed)
    pa, na = np.asarray(pos_a), np.asarray(neg_a)
    pb, nb = np.asarray(pos_b), np.asarray(neg_b)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        pi = rng.integers(0, pa.size, pa.size)
        ni = rng.integers(0, na.size, na.size)
        deltas[i] = (roc_auc(list(pb[pi]), list(nb[ni]))
                     - roc_auc(list(pa[pi]), list(na[ni])))
    return {
        "delta": round(float(roc_auc(pos_b, neg_b) - roc_auc(pos_a, neg_a)), 4),
        "lo": round(float(np.percentile(deltas, 2.5)), 4),
        "hi": round(float(np.percentile(deltas, 97.5)), 4),
        "excludes_zero": bool(np.percentile(deltas, 2.5) > 0
                              or np.percentile(deltas, 97.5) < 0),
        "n_boot": n_boot,
    }


def pair_cosines(vecs: dict) -> dict:
    """Raw cosine lists per pair class, so CIs and deltas can be recomputed."""
    rel, unrel = related_pairs(), unrelated_pairs()

    def cos_of(p):
        return cosine(vecs[p["w1"]]["enc"].vector, vecs[p["w2"]]["enc"].vector)

    out = {"unrelated": [cos_of(p) for p in unrel]}
    for pos in ("initial", "non_initial", "mixed"):
        out[pos] = [cos_of(p) for p in rel if p["position"] == pos]
    out["related_all"] = [cos_of(p) for p in rel]
    return out


def dp_sweep(splitters: dict, dps: list) -> dict:
    """How the whole picture moves with the positional budget dp.

    dp is the codec's only knob. Sweeping it shows that the effect reported here
    is not an artefact of one setting: as dp shrinks, truncation and collisions
    rise for raw encoding while segmentation keeps each piece inside the window.
    """
    rows = []
    for dp in dps:
        for name, sp in splitters.items():
            vecs = build_vectors(sp, dp)
            cos = pair_cosines(vecs)
            trunc = measure_truncation(vecs)
            coll = measure_collisions(vecs)
            rows.append({
                "dp": dp, "condition": name,
                "truncation_rate": trunc["truncation_rate"],
                "words_truncated": trunc["words_truncated"],
                "words_in_collision": coll["n_words_in_collision"],
                "auc_overall": round(roc_auc(cos["related_all"], cos["unrelated"]), 4),
                "auc_non_initial": round(roc_auc(cos["non_initial"], cos["unrelated"]), 4),
                "auc_initial": round(roc_auc(cos["initial"], cos["unrelated"]), 4),
            })
    return {"dps": dps, "rows": rows}


# ------------------------------------------------------------------ E5 splitter
def measure_splitter(splitter) -> dict:
    correct, rows = 0, []
    for word, gold in COMPOUNDS.items():
        got = splitter.split(word).parts
        ok = got == gold
        correct += ok
        rows.append({"word": word, "gold": gold, "predicted": got, "correct": ok})
    return {
        "splitter": getattr(splitter, "name", type(splitter).__name__),
        "n_words": len(COMPOUNDS),
        "n_correct": correct,
        "exact_match": round(correct / max(len(COMPOUNDS), 1), 6),
        "failures": [r for r in rows if not r["correct"]],
        "rows": rows,
    }


# ---------------------------------------------------------------- orchestration
def run_condition(name: str, splitter, dp: int = DEFAULT_DP) -> dict:
    vecs = build_vectors(splitter, dp)
    out = {
        "condition": name,
        "splitter": getattr(splitter, "name", "none (raw utf-8, truncated)") if splitter else "none (raw utf-8, truncated)",
        "dp": dp,
        "truncation": measure_truncation(vecs),
        "collisions": measure_collisions(vecs),
        "constituent_retrieval": measure_constituent_retrieval(vecs, dp),
        "relatedness": measure_relatedness(vecs),
    }
    if splitter is not None:
        out["segmentation"] = {
            "mean_parts": round(float(np.mean([len(v["parts"]) for v in vecs.values()])), 4),
            "part_histogram": dict(Counter(len(v["parts"]) for v in vecs.values())),
        }
    return out
