"""
faithful_metric.py -- the instructor's exact evaluation metric.

faithful_unit = one contiguous Unicode letter/mark/number run
                OR one visible non-space punctuation/symbol character
fertility(lang) = token_count(lang) / faithful_unit_count(lang)
score           = 1000 / (max_fertility - min_fertility)

Penalty (all-under-1.2 makes it 1.0): penalty(X) = exp(max(0, X/1.2 - 1)).
The grader has applied this to the highest-fertility relevant language; we aim to
keep EVERY language < 1.2 so the factor is exactly 1.

Faithfulness gate: decode(encode(text)) must preserve every non-whitespace char.
"""
import math
import regex as re

# EXACT reference rule: contiguous letters/marks/numbers as one unit, OR any
# single non-space non-alphanumeric character (punctuation/symbol/other visible).
_UNIT_RE = re.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")
_WS_RE = re.compile(r"\s")


def faithful_units(text):
    return len(_UNIT_RE.findall(text))


def nonspace(text):
    return _WS_RE.sub("", text)


def penalty(fertility, threshold=1.2):
    return math.exp(max(0.0, fertility / threshold - 1.0))


def evaluate(tokenizer, texts_by_lang):
    """texts_by_lang: {lang: text}. Returns a full metrics dict."""
    per = {}
    for lang, text in texts_by_lang.items():
        enc = tokenizer.encode(text)
        tokens = len(enc.ids)
        units = faithful_units(text)
        dec = tokenizer.decode(enc.ids)
        per[lang] = {
            "tokens": tokens,
            "faithful_units": units,
            "fertility": tokens / units if units else 0.0,
            "faithful_roundtrip": nonspace(dec) == nonspace(text),
        }
    ferts = {l: per[l]["fertility"] for l in per}
    xmax_l = max(ferts, key=ferts.get)
    xmin_l = min(ferts, key=ferts.get)
    spread = ferts[xmax_l] - ferts[xmin_l]
    raw = 1000.0 / spread if spread > 1e-12 else float("inf")
    # the reference evaluator penalizes Hindi; an earlier grader run penalized
    # English. We report both; when every language is < 1.2 both factors are 1.0.
    hi_pen = penalty(ferts["hi"]) if "hi" in ferts else 1.0
    en_pen = penalty(ferts["en"]) if "en" in ferts else 1.0
    max_pen = penalty(ferts[xmax_l])
    return {
        "per_language": per,
        "x_max": ferts[xmax_l], "x_max_lang": xmax_l,
        "x_min": ferts[xmin_l], "x_min_lang": xmin_l,
        "spread": spread,
        "raw_score": raw,
        "all_under_1_2": ferts[xmax_l] <= 1.2,
        "hindi_penalty": hi_pen, "hindi_adjusted_score": raw / hi_pen,
        "english_penalty": en_pen, "english_adjusted_score": raw / en_pen,
        "worst_case_adjusted_score": raw / max_pen,
        "all_faithful": all(per[l]["faithful_roundtrip"] for l in per),
    }
