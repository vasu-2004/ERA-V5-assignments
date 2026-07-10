"""
run_analysis.py -- computes the requested stats and trains a single, standard,
equal-weight byte-level BPE tokenizer over the combined EN+HI+TE+MR corpus,
reporting fertility (avg BPE tokens per unique word) at merge checkpoints.

No content cleaning is applied to the source files beyond standard universal-
newline normalization (\\r\\n -> \\n) when reading -- every figure below is
computed on the corpora exactly as supplied.
"""
import json
import time
import unicodedata
from collections import Counter
from pathlib import Path

from bpe import train_bpe, build_merge_rank, encode_word, word_freqs_from_text

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

LANGS = [
    ("English", "en", DATA / "english_india.txt"),
    ("Hindi", "hi", DATA / "hindi_india.txt"),
    ("Telugu", "te", DATA / "telugu_india.txt"),
    ("Marathi", "mr", DATA / "marathi_india.txt"),
]

PUNCT_CATS = {"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps"}


def strip_punct(word):
    start, end = 0, len(word)
    while start < end and unicodedata.category(word[start]) in PUNCT_CATS:
        start += 1
    while end > start and unicodedata.category(word[end - 1]) in PUNCT_CATS:
        end -= 1
    return word[start:end]


def main():
    corpora = {}
    stats = {}
    for name, code, path in LANGS:
        text = path.read_text(encoding="utf-8")  # universal newlines by default
        raw_words = text.split()
        stripped_words = [strip_punct(w) for w in raw_words]
        stripped_words = [w for w in stripped_words if w]
        unique_raw = set(raw_words)
        unique_stripped = set(stripped_words)
        unique_chars = set(text)
        corpora[code] = {
            "name": name,
            "text": text,
            "raw_words": raw_words,
            "stripped_words": stripped_words,
            "unique_raw_words": unique_raw,
            "unique_stripped_words": unique_stripped,
        }
        stats[code] = {
            "language": name,
            "total_characters": len(text),
            "total_characters_no_whitespace": sum(1 for c in text if not c.isspace()),
            "total_running_words": len(raw_words),
            "total_unique_words_raw": len(unique_raw),
            "total_unique_words_stripped": len(unique_stripped),
            "num_distinct_unicode_chars": len(unique_chars),
            "distinct_unicode_chars_sorted": sorted(unique_chars, key=lambda c: ord(c)),
            "distinct_unicode_codepoints": sorted(ord(c) for c in unique_chars),
        }
        print(f"[{code}] chars={stats[code]['total_characters']:>7}  "
              f"running_words={stats[code]['total_running_words']:>6}  "
              f"unique_raw={stats[code]['total_unique_words_raw']:>5}  "
              f"unique_stripped={stats[code]['total_unique_words_stripped']:>5}  "
              f"distinct_unicode_chars={stats[code]['num_distinct_unicode_chars']:>4}")

    # ---- combined, EQUAL-WEIGHT training corpus (raw whitespace words, unstripped,
    #      exactly as they occur -- "equal weight" = each language's real word
    #      frequencies are summed as-is, no re-weighting factor applied) ----
    combined = Counter()
    for code in corpora:
        combined.update(corpora[code]["raw_words"])
    total_combined_running_words = sum(combined.values())
    print(f"\nCombined corpus: {len(combined)} unique word types, "
          f"{total_combined_running_words} total running word occurrences")

    MAX_MERGES = 10000
    t0 = time.time()
    merges, id_to_bytes = train_bpe(combined, MAX_MERGES, log_every=1000)
    t1 = time.time()
    print(f"\nTrained {len(merges)} merges in {t1 - t0:.1f}s "
          f"(requested up to {MAX_MERGES}; stopped early if no pair count>=2 remained)")

    # ---- checkpoint fertility per language ----
    checkpoints = [c for c in [100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000,
                                6000, 7500, 8000, 9000, 10000] if c <= len(merges)]
    if len(merges) not in checkpoints:
        checkpoints.append(len(merges))
    checkpoints = sorted(set(checkpoints))

    fertility_table = {code: {} for code in corpora}
    for cp in checkpoints:
        mr = build_merge_rank(merges, cp)
        for code, data in corpora.items():
            words = sorted(data["unique_stripped_words"])
            total_tokens = 0
            for w in words:
                total_tokens += len(encode_word(w, mr))
            fert = total_tokens / len(words) if words else 0.0
            fertility_table[code][cp] = {
                "fertility": fert,
                "total_tokens": total_tokens,
                "unique_words": len(words),
            }
        print(f"checkpoint {cp:>5} merges -> "
              + "  ".join(f"{code}:{fertility_table[code][cp]['fertility']:.4f}" for code in corpora))

    # ---- also report fertility on RAW (unstripped) unique words at final checkpoint, for transparency ----
    mr_final = build_merge_rank(merges, len(merges))
    raw_fertility_final = {}
    for code, data in corpora.items():
        words = sorted(data["unique_raw_words"])
        total_tokens = sum(len(encode_word(w, mr_final)) for w in words)
        raw_fertility_final[code] = {
            "fertility": total_tokens / len(words) if words else 0.0,
            "total_tokens": total_tokens,
            "unique_words": len(words),
        }

    # ---- save artifacts ----
    out_stats = {code: {k: v for k, v in s.items() if k not in
                         ("distinct_unicode_chars_sorted",)}  # keep codepoints, drop raw glyph list from summary
                 for code, s in stats.items()}
    (ARTIFACTS / "corpus_stats.json").write_text(
        json.dumps(out_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # full char lists saved separately (can include astral/combining marks etc.)
    char_lists = {code: stats[code]["distinct_unicode_chars_sorted"] for code in stats}
    (ARTIFACTS / "distinct_unicode_chars.json").write_text(
        json.dumps(char_lists, ensure_ascii=False, indent=2), encoding="utf-8")

    (ARTIFACTS / "fertility_checkpoints.json").write_text(
        json.dumps({"checkpoints": checkpoints, "by_language": fertility_table,
                    "raw_word_fertility_final": raw_fertility_final,
                    "total_merges_trained": len(merges),
                    "training_seconds": t1 - t0,
                    "combined_unique_word_types": len(combined),
                    "combined_total_running_words": total_combined_running_words},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # vocab + merges (human-readable, standard format)
    vocab = {}
    for i in range(256):
        vocab[i] = id_to_bytes[i].hex()
    for a, b, nid in merges:
        vocab[nid] = id_to_bytes[nid].hex()
    (ARTIFACTS / "vocab.json").write_text(
        json.dumps({str(k): v for k, v in vocab.items()}, indent=2), encoding="utf-8")
    with open(ARTIFACTS / "merges.txt", "w", encoding="utf-8") as f:
        f.write("#version: baseline-equal-weight-bpe\n")
        for a, b, nid in merges:
            f.write(f"{a} {b} {nid}\n")

    print("\nSaved artifacts to", ARTIFACTS)


if __name__ == "__main__":
    main()
