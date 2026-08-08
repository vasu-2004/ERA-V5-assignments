#!/usr/bin/env python3
"""Extract real .py source files from downloaded package sdists into a
per-file JSONL (preserves document boundaries for real dedup/quality/lang-ID,
unlike a single concatenated blob)."""
import json
import pathlib
import tarfile

RAW = pathlib.Path(__file__).resolve().parent.parent / "corpus" / "raw"
PKG_DIR = RAW / "code_pkgs_all"
OUT = RAW / "code_corpus.jsonl"

LICENSE_GUESS = {
    "requests": "Apache-2.0", "flask": "BSD-3-Clause", "click": "BSD-3-Clause",
    "jinja2": "BSD-3-Clause", "werkzeug": "BSD-3-Clause", "pytest": "MIT",
    "attrs": "MIT", "pydantic": "MIT", "httpx": "BSD-3-Clause", "rich": "MIT",
    "typer": "MIT", "markdown": "BSD-3-Clause", "pygments": "BSD-2-Clause",
    "black": "MIT", "django": "BSD-3-Clause", "fastapi": "MIT",
    "starlette": "BSD-3-Clause", "tornado": "Apache-2.0",
}


def pkg_name(fname: str) -> str:
    base = fname.replace(".tar.gz", "")
    for name in LICENSE_GUESS:
        if base.lower().startswith(name):
            return name
    return base.split("-")[0].lower()


def main():
    n = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for tgz in sorted(PKG_DIR.glob("*.tar.gz")):
            pkg = pkg_name(tgz.name)
            lic = LICENSE_GUESS.get(pkg, "unknown")
            with tarfile.open(tgz, "r:gz") as tf:
                for member in tf.getmembers():
                    if not member.isfile() or not member.name.endswith(".py"):
                        continue
                    if member.size == 0 or member.size > 2_000_000:
                        continue
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    try:
                        text = f.read().decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                    if not text.strip():
                        continue
                    out.write(json.dumps({
                        "bucket": "code", "package": pkg, "license": lic,
                        "path": member.name, "text": text,
                    }, ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n} real source files -> {OUT}")


if __name__ == "__main__":
    main()
