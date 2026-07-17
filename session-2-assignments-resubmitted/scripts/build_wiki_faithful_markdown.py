"""
build_wiki_faithful_markdown.py
-------------------------------
Fetch the "India" Wikipedia page in English, Hindi, Telugu, Marathi as
*wiki-faithful Markdown* (links, URLs, tables, references, image links, navboxes
and categories preserved where the HTML->Markdown conversion emits them), exactly
in the spirit of the instructor's reference solution.

RUN THIS ON YOUR OWN MACHINE (it needs internet; the assistant's sandbox can't
reach Wikipedia). Then upload the four generated files in `corpus/`:
    corpus/en.faithful.md   corpus/hi.faithful.md
    corpus/te.faithful.md   corpus/mr.faithful.md

    pip install requests markdownify beautifulsoup4 lxml
    python build_wiki_faithful_markdown.py
"""
import json, pathlib, sys
import requests
from markdownify import markdownify as md

OUT = pathlib.Path(__file__).resolve().parent.parent / "corpus"
OUT.mkdir(exist_ok=True)

# (lang code, wiki subdomain, page title) for the India article in each language
PAGES = [
    ("en", "en", "India"),
    ("hi", "hi", "भारत"),
    ("te", "te", "భారతదేశం"),
    ("mr", "mr", "भारत"),
]
UA = "ERA-V5-Assignment2/1.0 (student resubmission; contact via course)"


def fetch_html(subdomain, title):
    # Wikipedia REST v1 Parsoid HTML (full article: links, refs, tables, navboxes)
    url = f"https://{subdomain}.wikipedia.org/api/rest_v1/page/html/{requests.utils.quote(title, safe='')}"
    r = requests.get(url, headers={"User-Agent": UA, "Accept": "text/html"}, timeout=60)
    r.raise_for_status()
    return r.text


def main():
    meta = {}
    for code, sub, title in PAGES:
        print(f"fetching {code}: {sub}.wikipedia.org / {title} ...", flush=True)
        html = fetch_html(sub, title)
        # faithful HTML -> Markdown: keep links/tables/etc. ATX headings.
        markdown = md(html, heading_style="ATX", strip=[])
        (OUT / f"{code}.faithful.md").write_text(markdown, encoding="utf-8")
        (OUT / f"{code}.faithful.txt").write_text(markdown, encoding="utf-8")  # same content, plain-text input
        meta[code] = {"subdomain": sub, "title": title, "chars": len(markdown)}
        print(f"  wrote corpus/{code}.faithful.md ({len(markdown)} chars)")
    (OUT / "corpus.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDone. Upload the four corpus/*.faithful.md files back to the assistant.")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print("HTTP error:", e, file=sys.stderr)
        print("If a title 404s, open the India page in that language on Wikipedia and copy the exact title.", file=sys.stderr)
        sys.exit(1)
