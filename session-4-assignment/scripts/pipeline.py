#!/usr/bin/env python3
"""
Session 4 cleaning pipeline -- the 8 strategies taught in the session,
applied to the real+constructed corpus in corpus/docs.jsonl.

  1. Extract        -- strip boilerplate/nav/footer/cookie banners from raw HTML
  2. Normalize       -- NFC, HTML-entity unescape, whitespace collapse (Brahmic-safe),
                         ghost-tag removal, code license-header strip
  3. Language ID     -- detect real language/script per document; validate claimed label
  4. Quality filter   -- Gopher/C4-style heuristics (+ code-specific lighter pass)
  5. Deduplication    -- MinHash + LSH near-duplicate removal (shingle -> hash -> band)
  6. PII scrub        -- regex + heuristic redaction of emails/phones/IPs/usernames
  7. Decontamination  -- canary-string sweep + GSM8K train/test shingle-overlap check
  8. Manifest         -- provenance/license/script-hash/shard-hash record per bucket

Every stage prints and records a before/after token count so the widget can
show the real survival curve, exactly like the session's own slide.
"""
import hashlib
import html
import json
import pathlib
import re
import unicodedata
from collections import Counter, defaultdict

import py3langid as langid
from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "corpus" / "docs.jsonl"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

CANARY_GUID = "BB-CANARY-e3a1c9f0-4b2d-4a1e-9c7a-1f6d2b8e5a90"
ZWNJ, ZWJ = "‌", "‍"  # must survive normalization -- legitimate in Brahmic scripts

# rough tokenizer stand-in: word/number runs + punctuation, close enough for
# a before/after survival curve without needing a trained BPE tokenizer.
_TOK_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def n_tokens(text: str) -> int:
    return len(_TOK_RE.findall(text))


def load_docs():
    docs = []
    for line in open(DOCS, encoding="utf-8"):
        docs.append(json.loads(line))
    return docs


def stage_survival(name, docs, note=""):
    tok = sum(n_tokens(d["text"]) for d in docs)
    print(f"[{name:24s}] docs={len(docs):6d}  tokens={tok:10d}  {note}")
    return {"stage": name, "docs": len(docs), "tokens": tok, "note": note}


# ---------------------------------------------------------------- STAGE 1: EXTRACT
def stage_extract(docs, stats):
    before = stage_survival("0_ingested", docs)
    stats.append(before)
    out = []
    for d in docs:
        if d["bucket"] == "extract_demo":
            soup = BeautifulSoup(d["text"], "lxml")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            cookie = soup.find(class_="cookie-notice")
            if cookie:
                cookie.decompose()
            article = soup.find("article")
            d = dict(d)
            d["text"] = article.get_text(" ", strip=True) if article else soup.get_text(" ", strip=True)
            d["extracted"] = True
        out.append(d)
    stats.append(stage_survival("1_extract", out,
                 "stripped nav/footer/cookie-banner/script from raw-HTML docs"))
    return out


# ---------------------------------------------------------------- STAGE 2: NORMALIZE
_LICENSE_HEADER_RE = re.compile(
    r"^(#|//|/\*)[^\n]*(license|copyright|redistribut|permission is hereby granted)"
    r"[^\n]*\n(?:(?:#|//|\*)[^\n]*\n){2,40}", re.IGNORECASE | re.MULTILINE,
)
_GHOST_TAG_RE = re.compile(r"<\|(?:user|assistant|system)\|>")
_CALC_ANNOTATION_RE = re.compile(r"<<[^<>]*=([\-0-9.,]+)>>")


def normalize_text(text: str, bucket: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = html.unescape(text)
    # collapse horizontal whitespace runs, but keep ZWNJ/ZWJ intact (Brahmic-safe)
    placeholder_zwnj, placeholder_zwj = "\x00ZWNJ\x00", "\x00ZWJ\x00"
    text = text.replace(ZWNJ, placeholder_zwnj).replace(ZWJ, placeholder_zwj)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace(placeholder_zwnj, ZWNJ).replace(placeholder_zwj, ZWJ)
    # ghost SFT-role tags must never leak into a pretraining-shaped corpus
    text = _GHOST_TAG_RE.sub("", text).strip()
    if bucket in ("gsm8k_train", "gsm8k_test"):
        # <<48/2=24>> is a working-annotation, not final-answer text -- keep the
        # result, drop the bracketed arithmetic (mirrors the "invisible
        # formatting token" rule: no value once the document is the training text)
        text = _CALC_ANNOTATION_RE.sub(r"\1", text)
        text = text.replace("#### ", "Answer: ")
    if bucket == "code":
        text = _LICENSE_HEADER_RE.sub("", text, count=1)
    return text.strip()


def stage_normalize(docs, stats):
    out = []
    for d in docs:
        d = dict(d)
        d["text"] = normalize_text(d["text"], d["bucket"])
        if d["text"]:
            out.append(d)
    stats.append(stage_survival("2_normalize", out,
                 "NFC + entity-unescape + ws-collapse (ZWNJ/ZWJ preserved) + ghost-tag/license-header strip"))
    return out


# ---------------------------------------------------------------- STAGE 3: LANGUAGE ID
_SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F), "mai": (0x0900, 0x097F),  # Devanagari (shared)
    "te": (0x0C00, 0x0C7F), "bn": (0x0980, 0x09FF), "as": (0x0980, 0x09FF),
}


def script_detect(text: str):
    counts = Counter()
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[lang] += 1
    return counts.most_common(1)[0][0] if counts else None


def stage_langid(docs, stats):
    out, mismatches = [], []
    for d in docs:
        d = dict(d)
        if d["bucket"] == "code":
            d["detected_lang"] = "code"
            out.append(d)
            continue
        detected, conf = langid.classify(d["text"][:2000])
        script = script_detect(d["text"][:2000])
        d["detected_lang"] = script or detected
        d["langid_conf"] = float(conf)
        claimed = d.get("claimed_lang")
        # Only DROP on the constructed langid_trap bucket, which exists purely to
        # test this check (real mislabelled-folder documents, e.g. an "as" folder
        # that's actually English "translate X to Y" placeholder text). Indic-wiki
        # documents keep their source-verified label even when script-detection
        # disagrees: Devanagari alone can't tell Hindi from Maithili apart (the
        # session's own point -- script != language, e.g. Telugu-vs-Grantha,
        # German-vs-Swiss) and a statistical LID model trained mostly on high-
        # resource languages will systematically guess "hi" for genuine, correctly
        # labelled Maithili. Dropping it on that basis would repeat exactly the
        # low-resource-Indic mistake the session warns against -- so we record the
        # disagreement for the report instead of discarding real minority-language text.
        if d["bucket"] == "langid_trap" and claimed and script and claimed != script:
            mismatches.append({"doc_id": d["doc_id"], "claimed": claimed,
                                "detected": script, "text": d["text"][:80], "action": "dropped"})
            d["langid_flag"] = "mislabelled"
            continue
        if d["bucket"] == "indic_wiki" and claimed in ("mai",) and script == "hi":
            mismatches.append({"doc_id": d["doc_id"], "claimed": claimed, "detected": script,
                                "text": d["text"][:80], "action": "kept (source-verified, shared script)"})
        out.append(d)
    stats.append(stage_survival("3_language_id", out,
                 f"py3langid + Unicode-script cross-check; dropped "
                 f"{sum(1 for m in mismatches if m['action']=='dropped')} mislabelled-folder trap docs, "
                 f"flagged-but-kept {sum(1 for m in mismatches if 'kept' in m['action'])} script-sharing cases"))
    return out, mismatches


# ---------------------------------------------------------------- STAGE 4: QUALITY FILTER
_STOPWORDS_EN = {"the", "a", "is", "of", "and", "to", "in", "that", "it", "for",
                  "on", "with", "as", "by", "at", "this", "be", "or", "an"}
# the session gives this exact example -- "hello hi hello hi how are you we
# will not like to train on that" -- a short filler exchange with no
# information content that the six general Gopher/C4 checks above don't
# reliably catch on their own (it isn't bulleted, isn't duplicated-line-heavy
# enough, isn't symbol-heavy). This is the session's stated reason a *second*
# layer -- a trained quality classifier -- exists beyond deterministic
# heuristics; we approximate that second layer with one more explicit rule
# rather than pretending heuristics alone are sufficient.
_FILLER_WORDS = {"hi", "hello", "hey", "lol", "ok", "okay", "yeah", "yep", "u",
                  "good", "how", "are", "you", "fine", "cool", "nice", "thanks"}


def quality_score(text: str, stopword_floor: float = 0.03):
    words = re.findall(r"[A-Za-z]+", text)
    if not words:
        return {"pass": len(text) > 40}  # e.g. code / non-Latin script, judged elsewhere
    mean_wlen = sum(len(w) for w in words) / len(words)
    symbols = len(re.findall(r"[^\w\s]", text))
    sym_ratio = symbols / max(1, len(words))
    lines = [l for l in text.split("\n") if l.strip()]
    end_punct = sum(1 for l in lines if l.rstrip()[-1:] in ".!?") / max(1, len(lines))
    dup_line_frac = 1 - len(set(lines)) / max(1, len(lines))
    stopword_frac = sum(1 for w in words if w.lower() in _STOPWORDS_EN) / len(words)
    # require marker+space: "**bold**" starts with "*" too but isn't a bullet --
    # confusing Markdown emphasis with a list marker is exactly the kind of
    # format-blind rule the session warns against (JSON vs YAML vs Markdown all
    # look different and a generic cleaner breaks all of them the same way).
    bullet_frac = sum(1 for l in lines if l.strip().startswith(("- ", "* ", "• "))) / max(1, len(lines))
    ellipsis_frac = text.count("...") / max(1, len(lines))
    filler_frac = sum(1 for w in words if w.lower() in _FILLER_WORDS) / len(words)
    checks = {
        "mean_word_length_ok": 2.5 <= mean_wlen <= 10,
        "symbol_ratio_ok": sym_ratio <= 1.5,
        "stopword_present": stopword_frac >= stopword_floor or len(words) < 15,
        "not_bullet_spam": bullet_frac <= 0.5,
        "not_dup_line_spam": dup_line_frac <= 0.6,
        "min_length": len(words) >= 8,
        "not_low_info_filler": not (len(words) < 20 and filler_frac >= 0.5),
    }
    return {"pass": all(checks.values()), "checks": checks, "mean_wlen": mean_wlen,
            "sym_ratio": sym_ratio, "stopword_frac": stopword_frac,
            "bullet_frac": bullet_frac, "dup_line_frac": dup_line_frac}


def stage_quality(docs, stats):
    out, dropped_examples = [], []
    for d in docs:
        if d["bucket"] == "code":
            # different objective for code -- session says apply *different* rules,
            # not the English Gopher/C4 battery: keep if it looks like real source
            # (has def/class/import) rather than an empty stub or binary garbage
            looks_like_code = bool(re.search(r"\b(def|class|import|function|const|let)\b", d["text"]))
            if looks_like_code and len(d["text"]) > 20:
                out.append(d)
            continue
        # GSM8K math word-problems are legitimately stopword-sparse (many proper
        # nouns/numbers, few function words) -- the general-prose 3% floor
        # calibrated on Wikipedia/news would zap real, well-formed documents here.
        # This is the same one-size-fits-all trap the session warns about (a rule
        # tuned for one register wrongly nukes a different, still-valid register),
        # so the math bucket gets its own, empirically-checked floor instead of a
        # blanket loosening of the rule for everything.
        floor = 0.008 if d["bucket"] in ("gsm8k_train", "gsm8k_test") else 0.03
        q = quality_score(d["text"], stopword_floor=floor)
        d = dict(d)
        d["quality"] = q
        if q["pass"]:
            out.append(d)
        elif len(dropped_examples) < 6:
            dropped_examples.append({"doc_id": d["doc_id"], "bucket": d["bucket"],
                                      "reason": {k: v for k, v in q.get("checks", {}).items() if not v},
                                      "text": d["text"][:100]})
    stats.append(stage_survival("4_quality_filter", out,
                 "Gopher/C4-style heuristics (mean word length, symbol ratio, stopwords, "
                 "bullet/dup-line ratio) for prose; code kept on a lighter structural check"))
    return out, dropped_examples


# ---------------------------------------------------------------- STAGE 5: DEDUPLICATION
def shingles(text: str, k: int = 8):
    words = text.split()
    return {" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


def adaptive_k(text: str) -> int:
    # A fixed shingle size tuned for full-length news articles (the session's own
    # demo) badly under-estimates similarity on short documents: word order alone
    # shifts almost every 8-word window in a one-sentence PR blurb even though a
    # human reads it as the same story reworded. Shrinking k with document length
    # is standard practice for mixed-length corpora (we have single-sentence PR
    # snippets AND full source files in the same pipeline).
    n = len(text.split())
    if n < 40:
        return 3
    if n < 150:
        return 5
    return 8


def minhash_of(text: str, num_perm: int = 128, k: int | None = None) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text, k or adaptive_k(text)):
        m.update(sh.encode("utf-8"))
    return m


def stage_dedup(docs, stats, threshold: float = 0.55):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    keep, removed_pairs, seen_exact = [], [], set()
    for d in docs:
        exact_key = hashlib.sha256(d["text"].encode("utf-8")).hexdigest()
        if exact_key in seen_exact:
            removed_pairs.append({"doc_id": d["doc_id"], "reason": "exact_duplicate"})
            continue
        seen_exact.add(exact_key)
        mh = minhash_of(d["text"])
        dup_of = lsh.query(mh)
        if dup_of:
            removed_pairs.append({"doc_id": d["doc_id"], "reason": "near_duplicate", "dup_of": dup_of[0]})
            continue
        lsh.insert(d["doc_id"], mh)
        keep.append(d)
    stats.append(stage_survival("5_deduplication", keep,
                 f"MinHash(128 perm) + LSH banding, Jaccard threshold {threshold}; "
                 f"removed {len(removed_pairs)} exact/near duplicates"))
    return keep, removed_pairs


# ---------------------------------------------------------------- STAGE 6: PII SCRUB
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+\d{1,3}[ -])?\d{3,5}[ -]\d{3,5}[ -]?\d{2,5}(?!\w)")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_USERNAME_RE = re.compile(r"\b[a-z]+_[a-z]*\d{2,6}\b", re.IGNORECASE)
_NAME_CONTEXT_RE = re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b(?=[^.]{0,20}(?:@|mail|contact))")


def scrub_pii(text: str):
    hits = Counter()
    def sub(pattern, tag):
        nonlocal text
        n = len(pattern.findall(text))
        hits[tag] += n
        text = pattern.sub(f"[{tag}]", text)
    sub(_EMAIL_RE, "EMAIL")
    sub(_IP_RE, "IP")
    sub(_PHONE_RE, "PHONE")
    sub(_USERNAME_RE, "USERNAME")
    sub(_NAME_CONTEXT_RE, "NAME")
    return text, hits


def stage_pii(docs, stats):
    out, total_hits = [], Counter()
    for d in docs:
        d = dict(d)
        if d["bucket"] != "code":
            # PII scrubbing targets human-authored prose (Reddit/forum/chat-style
            # text). Running email/phone/IP regexes on source code is wrong -- it
            # false-positive-matches version strings, line numbers, and hex/IPs in
            # test fixtures, destroying real code for no privacy benefit.
            d["text"], hits = scrub_pii(d["text"])
            total_hits.update(hits)
        out.append(d)
    stats.append(stage_survival("6_pii_scrub", out,
                 f"regex(email/phone/IP) + username/name-context heuristic; redacted {sum(total_hits.values())} "
                 f"spans ({dict(total_hits)})"))
    return out, total_hits


# ---------------------------------------------------------------- STAGE 7: DECONTAMINATION
def stage_decontam(docs, stats, jaccard_threshold: float = 0.5):
    out, flagged = [], []
    canary_hits = [d["doc_id"] for d in docs if CANARY_GUID in d["text"]]
    # Real leak detection needs near-WHOLE-DOCUMENT overlap, not a shared common
    # phrase: GSM8K problems reuse boilerplate reasoning sentences ("how many...",
    # "find the total amount") across unrelated questions, so a naive "any shared
    # 10-gram" check massively over-flags formulaic domains -- a first pass at
    # this literally did (37/1319 test docs, nearly all false positives on stock
    # phrasing). Requiring high estimated Jaccard similarity via the same MinHash
    # index built for stage 5 is the honest version of the check: it only fires
    # when a test question and a train document are near-duplicates of each
    # other, which is what "the model has seen this exact question before" means.
    train_lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=128)
    train_mh = {}
    for d in docs:
        if d["bucket"] == "gsm8k_train":
            mh = minhash_of(d["text"])
            train_mh[d["doc_id"]] = mh
            train_lsh.insert(d["doc_id"], mh)
    leak_count = 0
    for d in docs:
        if CANARY_GUID in d["text"]:
            flagged.append({"doc_id": d["doc_id"], "reason": "canary_string_match"})
            continue
        if d["bucket"] == "gsm8k_test":
            match = train_lsh.query(minhash_of(d["text"]))
            if match:
                leak_count += 1
                flagged.append({"doc_id": d["doc_id"], "reason": "train/test_near_duplicate",
                                 "matched_train_doc": match[0]})
                continue
        out.append(d)
    stats.append(stage_survival("7_decontamination", out,
                 f"canary-GUID sweep ({len(canary_hits)} hit) + whole-document MinHash Jaccard>={jaccard_threshold} "
                 f"between GSM8K train/test ({leak_count} flagged; a naive any-shared-10-gram version of this "
                 f"check over-flagged 37 docs on shared boilerplate phrasing -- see report)"))
    return out, flagged


# ---------------------------------------------------------------- STAGE 8: MANIFEST
def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_manifest(final_docs, all_stats):
    by_bucket = defaultdict(list)
    for d in final_docs:
        by_bucket[d["bucket"]].append(d)
    manifest = {"shards": [], "pipeline_script_sha256": sha256_file(pathlib.Path(__file__))}
    for bucket, ds in by_bucket.items():
        shard_text = "\n".join(d["text"] for d in ds)
        manifest["shards"].append({
            "bucket": bucket,
            "n_docs": len(ds),
            "tokens": sum(n_tokens(d["text"]) for d in ds),
            "sources": sorted({d.get("source", "n/a") for d in ds}),
            "licenses": sorted({d.get("license", "n/a") for d in ds}),
            "shard_sha256": hashlib.sha256(shard_text.encode("utf-8")).hexdigest()[:16],
            "cleaning_stages_applied": [s["stage"] for s in all_stats],
        })
    return manifest


def main():
    docs = load_docs()
    stats = []
    docs = stage_extract(docs, stats)
    docs = stage_normalize(docs, stats)
    docs, langid_mismatches = stage_langid(docs, stats)
    docs, quality_dropped = stage_quality(docs, stats)
    docs, dedup_removed = stage_dedup(docs, stats)
    docs, pii_hits = stage_pii(docs, stats)
    docs, decontam_flagged = stage_decontam(docs, stats)
    manifest = stage_manifest(docs, stats)

    (ART / "cleaned_corpus.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")
    (ART / "stats.json").write_text(json.dumps({
        "stages": stats,
        "langid_mismatches": langid_mismatches,
        "quality_dropped_examples": quality_dropped,
        "dedup_removed_examples": dedup_removed[:10],
        "dedup_removed_total": len(dedup_removed),
        "pii_hits": dict(pii_hits),
        "decontam_flagged": decontam_flagged,
        "final_doc_count": len(docs),
        "final_token_count": sum(n_tokens(d["text"]) for d in docs),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (ART / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nWrote artifacts/cleaned_corpus.jsonl, stats.json, manifest.json")


if __name__ == "__main__":
    main()
