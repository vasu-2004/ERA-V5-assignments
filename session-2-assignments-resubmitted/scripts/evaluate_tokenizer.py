"""evaluate_tokenizer.py -- load artifacts/tokenizer.json and report the
faithful-unit fertilities, spread, raw score, penalty, and faithfulness gate on
the faithful-Markdown corpus. This is the grader's-eye view of our tokenizer."""
import json, pathlib
from tokenizers import Tokenizer
from faithful_metric import evaluate

ROOT = pathlib.Path(__file__).resolve().parent.parent
LANGS = ["en", "hi", "te", "mr"]
NAME = {"en": "English", "hi": "Hindi", "te": "Telugu", "mr": "Marathi"}


def main():
    tok = Tokenizer.from_file(str(ROOT / "artifacts" / "tokenizer.json"))
    texts = {}
    for l in LANGS:
        p = ROOT / "corpus" / f"{l}.faithful.md"
        if not p.exists():
            raise SystemExit(f"missing {p} -- fetch the corpus first")
        texts[l] = p.read_text(encoding="utf-8")
    m = evaluate(tok, texts)
    print(f"vocab size: {tok.get_vocab_size()}\n")
    print(f"{'Language':10}{'Tokens':>10}{'Faithful units':>16}{'Fertility':>12}{'roundtrip':>12}")
    for l in LANGS:
        p = m["per_language"][l]
        print(f"{NAME[l]:10}{p['tokens']:>10}{p['faithful_units']:>16}"
              f"{p['fertility']:>12.4f}{('OK' if p['faithful_roundtrip'] else 'FAIL'):>12}")
    print(f"\nX_max = {m['x_max']:.4f} ({NAME[m['x_max_lang']]})   "
          f"X_min = {m['x_min']:.4f} ({NAME[m['x_min_lang']]})")
    print(f"spread = {m['spread']:.6f}   raw score = 1000/spread = {m['raw_score']:.2f}")
    print(f"all languages < 1.2: {m['all_under_1_2']}   penalty(max) = {m['penalty_on_max']:.6f}   "
          f"adjusted score = {m['adjusted_score']:.2f}")
    print(f"faithful (decode preserves non-whitespace): {m['all_faithful']}")


if __name__ == "__main__":
    main()
