#!/usr/bin/env python3
"""Stage 0: ingest every raw source into one unified per-document JSONL.
This is deliberately BEFORE the 8 cleaning strategies -- it just gives every
source a common {doc_id, bucket, claimed_lang, source, license, text} shape
so the pipeline can treat GSM8K rows, real .py files, India-wiki paragraphs,
and the constructed demo docs uniformly."""
import json
import pathlib
import re

RAW = pathlib.Path(__file__).resolve().parent.parent / "corpus" / "raw"
OUT = pathlib.Path(__file__).resolve().parent.parent / "corpus" / "docs.jsonl"

LANG_FILE_MAP = {"en": "en", "hi": "hi", "te": "te", "mai": "mai"}


def split_paragraphs(text: str, min_len: int = 120):
    parts = re.split(r"\n\s*\n+", text)
    out, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        buf = (buf + "\n\n" + p) if buf else p
        if len(buf) >= min_len:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def main():
    n = 0
    with open(OUT, "w", encoding="utf-8") as out:

        def emit(bucket, claimed_lang, source, license_, text, extra=None):
            nonlocal n
            rec = {"doc_id": f"d{n:07d}", "bucket": bucket, "claimed_lang": claimed_lang,
                   "source": source, "license": license_, "text": text}
            if extra:
                rec.update(extra)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

        # --- real code, per file ---
        for line in open(RAW / "code_corpus.jsonl", encoding="utf-8"):
            r = json.loads(line)
            emit("code", "code", f"pip:{r['package']}", r["license"], r["text"],
                 {"path": r["path"], "package": r["package"]})

        # --- real GSM8K reasoning traces ---
        for split, fname in [("train", "gsm8k_train.jsonl"), ("test", "gsm8k_test.jsonl")]:
            for line in open(RAW / fname, encoding="utf-8"):
                r = json.loads(line)
                text = f"Q: {r['question']}\nA: {r['answer']}"
                emit("gsm8k_" + split, "en", "github:openai/grade-school-math", "MIT", text)

        # --- real India-Wikipedia faithful corpus (Session 3) ---
        for lang, fname in [("en", "indic_en.faithful.txt"), ("hi", "indic_hi.faithful.txt"),
                            ("te", "indic_te.faithful.txt"), ("mai", "indic_mai.faithful.txt")]:
            text = (RAW / fname).read_text(encoding="utf-8")
            for para in split_paragraphs(text):
                emit("indic_wiki", lang, f"wikipedia.org/wiki/India ({lang})", "CC-BY-SA-3.0", para)

        # --- constructed demo bucket (small, labelled, exercises the 8 stages) ---
        for line in open(RAW / "synthetic_demo_bucket.jsonl", encoding="utf-8"):
            r = json.loads(line)
            emit(r["bucket"], r["lang"], "constructed:session4_demo", "n/a", r["text"],
                 {"pr_group": r.get("pr_group")})

    print(f"ingested {n} documents -> {OUT}")


if __name__ == "__main__":
    main()
