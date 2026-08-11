"""The execution log (submission_artifacts/run.log).

Two kinds of line:
  [EVENT] name key=value ...      -- narrative of what the system did
  [PASS]/[FAIL] check_name ...    -- an assertion the system actually evaluated

Nothing writes [PASS] directly; it is emitted by `check()`, which takes a
boolean that the implementation computed. A hardcoded PASS is therefore not
expressible through this API.
"""
from __future__ import annotations

import pathlib
import time


class RunLog:
    def __init__(self, path: pathlib.Path, echo: bool = True):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("w", encoding="utf-8")
        self.echo = echo
        self.checks: list = []
        self.events: list = []
        self._t0 = time.time()

    def _write(self, line: str):
        self.fh.write(line + "\n")
        self.fh.flush()
        if self.echo:
            print(line, flush=True)

    def _fmt(self, **kw) -> str:
        return " ".join(f"{k}={v}" for k, v in kw.items())

    def event(self, name: str, **kw):
        self.events.append({"event": name, "fields": kw, "t": round(time.time() - self._t0, 4)})
        self._write(f"[EVENT] {name} {self._fmt(**kw)}".rstrip())

    def section(self, title: str):
        self._write("")
        self._write(f"=== {title} ===")

    def info(self, msg: str):
        self._write(f"[INFO ] {msg}")

    def warn(self, msg: str):
        self._write(f"[WARN ] {msg}")

    def check(self, name: str, passed: bool, **evidence) -> bool:
        """Record a verified assertion. `passed` must be computed by the caller."""
        passed = bool(passed)
        self.checks.append({"check": name, "passed": passed, "evidence": evidence})
        tag = "[PASS]" if passed else "[FAIL]"
        self._write(f"{tag} {name} {self._fmt(**evidence)}".rstrip())
        return passed

    def close(self):
        n_pass = sum(1 for c in self.checks if c["passed"])
        self._write("")
        self._write(f"[INFO ] checks_passed={n_pass}/{len(self.checks)} "
                    f"elapsed_s={round(time.time() - self._t0, 2)}")
        self.fh.close()
