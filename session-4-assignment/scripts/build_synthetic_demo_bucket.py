#!/usr/bin/env python3
"""
Build a SMALL, explicitly-labelled "raw web crawl simulation" bucket.

Session 4 teaches 8 cleaning strategies using hand-built illustrative examples
(dirty English scrape, chat logs, SEO junk, PR duplicates, PII, ghost tags,
mislabelled-language folders, benchmark canaries) -- the instructor built these
live to demonstrate each stage, because genuinely-dirty raw Common Crawl HTML
at throwaway scale isn't something a laptop/sandbox can scrape.

This script does the same thing at slightly larger scale so every one of the
8 strategies has real material to remove. It is a small minority of the total
corpus (see manifest) -- the bulk of the corpus is authentic, sourced text
(GSM8K reasoning, real permissively-licensed code, real India-Wikipedia
faithful Markdown). This bucket exists to exercise the pipeline, not to pad
token count, and is labelled as constructed in every output artifact.
"""
import json
import random
import pathlib

random.seed(42)
OUT = pathlib.Path(__file__).resolve().parent.parent / "corpus" / "raw"

# ---------------------------------------------------------------- 1. EXTRACT
# Raw-HTML-shaped documents with boilerplate that step 1 must strip.
BOILERPLATE_DOCS = []
ARTICLE_BODIES = [
    "Researchers this week described how the monsoon system arrived nine days "
    "early over the Konkan coast, a shift attributed to warmer Arabian Sea "
    "surface temperatures. Farmers in three districts have already adjusted "
    "sowing schedules in response to the revised forecast from the regional "
    "meteorological centre.",
    "The state transport corporation announced a new fleet of electric buses "
    "for the Mysuru-Bengaluru corridor starting next quarter, part of a wider "
    "push to cut emissions on high-traffic intercity routes across Karnataka.",
    "A team at the Indian Institute of Science published new results on "
    "low-cost water filtration membranes, reporting a 40 percent reduction in "
    "fluoride content using a locally sourced ceramic composite.",
]
for i, body in enumerate(ARTICLE_BODIES):
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>
<div class="cookie-notice">We use cookies &amp; other technologies. By using this site you agree. 47 partners use cookies. <a href="#">Accept all</a> <a href="#">Manage</a></div>
<nav>Home &gt; Category &gt; India &gt; National <a href="/about">About Us</a> <a href="/contact">Contact</a></nav>
<article><h1>Regional report {i+1}</h1><p>{body}</p></article>
<footer>&copy; 2026 Example News Network. All rights reserved. <a href="/privacy">Privacy</a> | <a href="/terms">Terms</a> | Board of Directors | Media Kit | Advertise with us | Careers</footer>
<script>gtag('config','UA-XXXX');</script></body></html>"""
    BOILERPLATE_DOCS.append({"id": f"extract_{i}", "html": html, "clean_body": body})

# --------------------------------------------------------- 2. NORMALIZE
# Ghost tags, HTML entities, ZWNJ/ZWJ (must survive for Brahmic!), 4-space
# runs from PDF export, chat-role markers that must not leak into pretraining.
NORMALIZE_DOCS = [
    "Cookie&nbsp;Notice: We use cookies &amp; other technologies.&nbsp;&nbsp;&nbsp;"
    "Click &quot;I agree&quot; to continue.​​",  # HTML entities + ZWSP
    "<|user|> hi\n<|assistant|> hello, how can I help?\n<|user|> what's 2+2\n"
    "<|assistant|> 4",  # ghost tags -- must be swapped for the real special tokens, not left in
    "This    line    has    four-space    runs    from    a    PDF    export.",
    "क्ष‍ल -- keeps a legitimate ZWJ for the Devanagari conjunct; "
    "do not strip this the way an English-only whitespace cleaner would.",
]

# ---------------------------------------------------- 3. LANGUAGE ID (mislabel trap)
# Mirrors the real v4 bug described in the session: a folder tagged "as" (Assamese)
# that actually contains Bengali/English "translate X to Y" placeholder rows.
LANG_TRAP_DOCS = [
    {"claimed_lang": "as", "text": "Translate English to Assamese: The monsoon arrived early this year."},
    {"claimed_lang": "as", "text": "অসমীয়া অনুবাদ পৃষ্ঠা: এই বছর বৰষা সোনকালে আহিল।"},  # actually Bengali-adjacent/Assamese script mix
    {"claimed_lang": "hi", "text": "This document is entirely in English despite the Hindi folder label."},
    {"claimed_lang": "mr", "text": "भारत हा दक्षिण आशियातील एक देश आहे."},  # genuinely Marathi
]

# --------------------------------------------------- 4. QUALITY FILTER
QUALITY_GOOD = (
    "Photosynthesis is the process by which a green plant converts light into "
    "chemical energy. In the leaves, chlorophyll absorbs sunlight and drives a "
    "reaction that turns carbon dioxide and water into glucose and oxygen."
)
QUALITY_SEO_SPAM = (
    "buy wireless dog soap best dog soap 2026 dog soap review dog soap price "
    "dog soap online dog soap for sale cheap dog soap dog soap discount dog "
    "soap offer dog soap deal buy dog soap now dog soap coupon dog soap free "
    "shipping dog soap india dog soap delivery dog soap bulk order dog soap "
    "wholesale dog soap manufacturer dog soap supplier dog soap distributor"
)
QUALITY_LIST_SPAM = "\n".join(f"- city_{i}, list item {i}, entry number {i}" for i in range(120))
QUALITY_REDDIT_NOISE = "hello\nhi\nhello\nhi\nhow are you\ngood\nu\nyeah\nlol\nok"

# ------------------------------------------------------------- 5. DEDUP
# Same PR story run through 6 "outlets" -- near-duplicate, not exact.
PR_BASE = ("School of AI trained a 120 billion parameter model, the company "
           "announced today, calling it a milestone for homegrown large "
           "language model development in the region.")
PR_VARIANTS = [
    PR_BASE,
    PR_BASE.replace("School of AI trained a 120 billion parameter model",
                     "A 120 billion parameter model was trained by School of AI"),
    PR_BASE.replace("the region", "India") + " The announcement was made in Bangalore.",
    PR_BASE.replace("today,", "on Thursday,"),
    PR_BASE + " Industry watchers called the move significant.",
    PR_BASE.replace("milestone", "landmark achievement"),
]
EXACT_DUP_DOC = "Our web crawler pulls millions of raw pages from the internet every single day and indexes new stories within minutes."
EXACT_DUP_COUNT = 5

# ------------------------------------------------------------- 6. PII SCRUB
PII_DOCS = [
    "Posting on the migration thread: reply to the set from Ananya Sharma "
    "(ananya.sharma94@mailexample.com). Call back requested: +91 98765 43210. "
    "Logged in from 203.0.113.45. Ticket raised by user rahul_wm88 while "
    "travelling through the Mumbai data centre.",
    "GitHub user devbot_2847 committed this patch; contact patch-author at "
    "devbot2847@codehost-example.com if the build breaks. IP allowlist: "
    "198.51.100.23.",
]

# ---------------------------------------------------- 7. DECONTAMINATION (canary)
CANARY_GUID = "BB-CANARY-e3a1c9f0-4b2d-4a1e-9c7a-1f6d2b8e5a90"
CANARY_DOC = (
    f"BENCHMARK CANARY STRING {CANARY_GUID} -- if this string appears verbatim "
    "in a trained model's output, the model was trained on this held-out "
    "benchmark and its scores on that benchmark are invalid."
)

# ---------------------------------------------------- 8. MANIFEST / SYNTHETIC AUGMENT
# One rephrase-augmentation example (Kimi K2-style), Jaccard-similar but not identical.
REPHRASE_PAIR = (
    "The green plant turns sunlight into chemical energy using chlorophyll to "
    "combine carbon dioxide and water into glucose while releasing oxygen into the air.",
    "How can be, through photosynthesis, green plant turns light into chemical energy?"
)


def main():
    docs = []
    for d in BOILERPLATE_DOCS:
        docs.append({"bucket": "extract_demo", "lang": "en", "text": d["html"]})
    for i, t in enumerate(NORMALIZE_DOCS):
        docs.append({"bucket": "normalize_demo", "lang": "en" if i != 3 else "hi", "text": t})
    for d in LANG_TRAP_DOCS:
        docs.append({"bucket": "langid_trap", "lang": d["claimed_lang"], "text": d["text"]})
    for t, lbl in [(QUALITY_GOOD, "quality_good"), (QUALITY_SEO_SPAM, "quality_seo_spam"),
                   (QUALITY_LIST_SPAM, "quality_list_spam"), (QUALITY_REDDIT_NOISE, "quality_reddit_noise")]:
        docs.append({"bucket": lbl, "lang": "en", "text": t})
    for i, t in enumerate(PR_VARIANTS):
        docs.append({"bucket": "dedup_near_pr", "lang": "en", "text": t, "pr_group": "pr1"})
    for i in range(EXACT_DUP_COUNT):
        docs.append({"bucket": "dedup_exact", "lang": "en", "text": EXACT_DUP_DOC, "pr_group": "exact1"})
    for t in PII_DOCS:
        docs.append({"bucket": "pii_demo", "lang": "en", "text": t})
    docs.append({"bucket": "decontam_canary", "lang": "en", "text": CANARY_DOC})
    docs.append({"bucket": "synthetic_rephrase_a", "lang": "en", "text": REPHRASE_PAIR[0]})
    docs.append({"bucket": "synthetic_rephrase_b", "lang": "en", "text": REPHRASE_PAIR[1]})

    out_path = OUT / "synthetic_demo_bucket.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    total_chars = sum(len(d["text"]) for d in docs)
    print(f"wrote {len(docs)} constructed demo documents, {total_chars} chars -> {out_path}")
    print("CANARY_GUID for decontamination check:", CANARY_GUID)


if __name__ == "__main__":
    main()
