#!/usr/bin/env python3
"""
One-time corpus preparation (provenance tool -- NOT part of the demo run).

Carves a small, real, committable corpus out of the larger sources already
used in this course's Session 4 cleaning assignment, so the Session 6 data
system runs on genuine text rather than generated filler:

  web_en   real English prose        India Wikipedia (faithful Markdown)   CC BY-SA 3.0
  indic    real Devanagari + Telugu  India Wikipedia hi/te                 CC BY-SA 3.0
  code     real Python source        permissively-licensed PyPI packages   MIT/BSD/Apache
  math     real reasoning traces     OpenAI grade-school-math (GSM8K)      MIT

  eval_math / val_web  held-out splits, never trainable (firewall targets)

The demo itself only ever reads corpus_data/*.jsonl -- this script is kept for
auditability of where those documents came from.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

SRC = pathlib.Path("/home/user/ERA-V5-assignments/session-4-assignment/corpus/raw")
OUT = pathlib.Path(__file__).resolve().parent.parent / "corpus_data"

# Canary strings live ONLY in held-out shards. If one of these ever shows up in
# a loss-bearing training batch, the evaluation firewall has failed.
CANARY_EVAL = "CANARY-EVAL-7f3a91c25e8b4d06"
CANARY_VAL = "CANARY-VAL-2b8e64af10c97d35"


def split_paragraphs(text: str, min_len: int, max_len: int):
    out, buf = [], ""
    for para in re.split(r"\n\s*\n+", text):
        para = para.strip()
        if not para:
            continue
        buf = (buf + "\n\n" + para) if buf else para
        if len(buf) >= min_len:
            out.append(buf[:max_len])
            buf = ""
    if len(buf) >= min_len // 2:
        out.append(buf[:max_len])
    return out


def doc(lane, split, source, license_, text, meta=None):
    return {
        "doc_id": hashlib.sha256(f"{lane}|{source}|{text[:200]}".encode()).hexdigest()[:16],
        "lane": lane,
        "split": split,
        "source": source,
        "license": license_,
        "text": text,
        "meta": meta or {},
    }


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    lanes: dict[str, list] = {}

    # ---- web_en : real English encyclopedic prose -------------------------
    en = (SRC / "indic_en.faithful.txt").read_text(encoding="utf-8")
    paras = split_paragraphs(en, 400, 3000)
    train_en = paras[: int(len(paras) * 0.85)]
    held_en = paras[int(len(paras) * 0.85):]
    lanes["web_en"] = [doc("web_en", "train", "wikipedia:India(en)", "CC-BY-SA-3.0", t)
                       for t in train_en[:170]]
    # held-out validation slice, canary-marked
    lanes["val_web"] = [doc("val_web", "val", "wikipedia:India(en)", "CC-BY-SA-3.0",
                            f"{CANARY_VAL}\n{t}") for t in held_en[:25]]

    # ---- indic : real Devanagari (Hindi) + Telugu -------------------------
    indic_docs = []
    for fname, src in [("indic_hi.faithful.txt", "wikipedia:भारत(hi)"),
                       ("indic_te.faithful.txt", "wikipedia:భారతదేశం(te)")]:
        txt = (SRC / fname).read_text(encoding="utf-8")
        for t in split_paragraphs(txt, 300, 2500)[:70]:
            indic_docs.append(doc("indic", "train", src, "CC-BY-SA-3.0", t))
    lanes["indic"] = indic_docs

    # ---- code : real permissively-licensed Python -------------------------
    code_docs, seen_pkg = [], {}
    for line in (SRC / "code_corpus.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if not (400 < len(r["text"]) < 6000):
            continue
        # spread across packages so the lane isn't one project's style
        if seen_pkg.get(r["package"], 0) >= 12:
            continue
        seen_pkg[r["package"]] = seen_pkg.get(r["package"], 0) + 1
        code_docs.append(doc("code", "train", f"pypi:{r['package']}", r["license"], r["text"],
                             {"path": r["path"]}))
        if len(code_docs) >= 150:
            break
    lanes["code"] = code_docs

    # ---- math : real GSM8K reasoning traces (prompt/answer structured) ----
    math_docs = []
    for i, line in enumerate((SRC / "gsm8k_train.jsonl").open(encoding="utf-8")):
        if i >= 200:
            break
        r = json.loads(line)
        # keep prompt and completion separate: the packer masks prompt tokens
        # out of the loss, which is only meaningful if the boundary is real.
        math_docs.append(doc("math", "train", "github:openai/grade-school-math", "MIT",
                             f"Question: {r['question']}\nAnswer: {r['answer']}",
                             {"prompt": f"Question: {r['question']}\nAnswer:",
                              "structured": True}))
    lanes["math"] = math_docs

    # ---- eval_math : held-out GSM8K test, canary-marked -------------------
    eval_docs = []
    for i, line in enumerate((SRC / "gsm8k_test.jsonl").open(encoding="utf-8")):
        if i >= 40:
            break
        r = json.loads(line)
        eval_docs.append(doc("eval_math", "eval", "github:openai/grade-school-math#test", "MIT",
                             f"{CANARY_EVAL}\nQuestion: {r['question']}\nAnswer: {r['answer']}",
                             {"prompt": f"Question: {r['question']}\nAnswer:", "structured": True}))
    lanes["eval_math"] = eval_docs

    total_docs = total_chars = 0
    index = []
    for lane, docs in lanes.items():
        path = OUT / f"{lane}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        chars = sum(len(d["text"]) for d in docs)
        total_docs += len(docs)
        total_chars += chars
        index.append({"lane": lane, "split": docs[0]["split"], "n_docs": len(docs),
                      "chars": chars, "file": path.name,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f"{lane:10s} split={docs[0]['split']:5s} docs={len(docs):4d} chars={chars:8d}")

    (OUT / "corpus_index.json").write_text(json.dumps({
        "description": "Small real corpus carved from Session 4 sources; see tools/build_corpus.py",
        "canaries": {"eval_math": CANARY_EVAL, "val_web": CANARY_VAL},
        "total_docs": total_docs, "total_chars": total_chars,
        "lanes": index,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTOTAL docs={total_docs} chars={total_chars} -> {OUT}")


if __name__ == "__main__":
    if not SRC.exists():
        sys.exit(f"source corpus not found at {SRC}; this tool only runs where Session 4 data exists")
    build()
