"""A training-free Devanagari morphological splitter: inverse vowel sandhi.

Why this exists rather than a call to an existing analyzer
----------------------------------------------------------
The plan was to call an off-the-shelf analyzer. Neither was usable here:

  * the Sanskrit Heritage Engine (sanskrit.inria.fr) is not reachable from this
    environment;
  * `indic-nlp-library` installs, but its `unsupervised_morph` needs the
    `indic_nlp_resources` Morfessor model bundle, whose download returns 403.

So this module implements the analyzer instead. It is still *training-free* --
a rule table plus a morpheme lexicon, no fitted parameters, no corpus statistics.
`MorfessorSplitter` below is a working adapter for the indic-nlp analyzer that
activates automatically if the resource bundle is ever present, so the pluggable
claim is real rather than rhetorical.

How it works
------------
In Devanagari a consonant carries an inherent /a/; a following vowel is written
as a matra on that consonant. When two morphemes join, the final vowel of the
first and the initial vowel of the second coalesce into one matra:

    deva + ālaya   ->  देव + आलय   ->  देवालय     (a + ā -> ā)
    mahā + īśa     ->  महा + ईश    ->  महेश       (ā + ī -> e)
    sūrya + udaya  ->  सूर्य + उदय  ->  सूर्योदय   (a + u -> o)

Splitting inverts that. For every character boundary the splitter proposes

  * a plain concatenation (no sandhi applied at the seam), and
  * for each (v1, v2) pair that could have produced the matra sitting at the
    seam, the reconstructed pair (stem1 + matra(v1), vowel(v2) + stem2);

and keeps a candidate only when BOTH reconstructed pieces are in the lexicon.
Splitting then recurses on the tail, so three-member compounds decompose too.

Known blind spot, stated up front: only vowel (svara) sandhi is modelled.
Yaṇ sandhi (ati + anta -> अत्यन्त) and consonant sandhi are not, and the
dataset deliberately includes such words so the reported accuracy is honest
rather than flattering.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------- characters
INDEPENDENT_VOWEL = {
    "a": "अ", "ā": "आ", "i": "इ", "ī": "ई", "u": "उ", "ū": "ऊ",
    "ṛ": "ऋ", "e": "ए", "ai": "ऐ", "o": "ओ", "au": "औ",
}
MATRA = {
    "a": "", "ā": "ा", "i": "ि", "ī": "ी", "u": "ु", "ū": "ू",
    "ṛ": "ृ", "e": "े", "ai": "ै", "o": "ो", "au": "ौ",
}
MATRA_TO_VOWEL = {m: v for v, m in MATRA.items() if m}
ALL_MATRAS = set(MATRA_TO_VOWEL)
VIRAMA = "्"

# (v1, v2) -> resulting vowel at the seam.  Savarna-dirgha, guṇa and vṛddhi.
SANDHI_RULES = {}
for _v1 in ("a", "ā"):
    for _v2 in ("a", "ā"):
        SANDHI_RULES[(_v1, _v2)] = "ā"
    for _v2 in ("i", "ī"):
        SANDHI_RULES[(_v1, _v2)] = "e"
    for _v2 in ("u", "ū"):
        SANDHI_RULES[(_v1, _v2)] = "o"
    for _v2 in ("e", "ai"):
        SANDHI_RULES[(_v1, _v2)] = "ai"
    for _v2 in ("o", "au"):
        SANDHI_RULES[(_v1, _v2)] = "au"
    SANDHI_RULES[(_v1, "ṛ")] = "ar"
for _v1 in ("i", "ī"):
    for _v2 in ("i", "ī"):
        SANDHI_RULES[(_v1, _v2)] = "ī"
for _v1 in ("u", "ū"):
    for _v2 in ("u", "ū"):
        SANDHI_RULES[(_v1, _v2)] = "ū"

# inverse index: seam vowel -> every (v1, v2) that could have produced it
INVERSE_SANDHI: dict = {}
for (_a, _b), _r in SANDHI_RULES.items():
    INVERSE_SANDHI.setdefault(_r, []).append((_a, _b))


def apply_sandhi(stem1: str, stem2: str) -> str:
    """Forward direction -- used by the tests to check the rules round-trip."""
    if not stem1 or not stem2:
        return stem1 + stem2
    v2 = None
    for vowel, ch in INDEPENDENT_VOWEL.items():
        if stem2.startswith(ch):
            # prefer the longest matching vowel name (ai/au before a/o)
            if v2 is None or len(INDEPENDENT_VOWEL[v2]) < len(ch):
                v2 = vowel
    if v2 is None:
        return stem1 + stem2                      # no initial vowel: plain join
    last = stem1[-1]
    v1 = MATRA_TO_VOWEL.get(last, "a" if last not in ALL_MATRAS else None)
    if v1 is None:
        return stem1 + stem2
    result = SANDHI_RULES.get((v1, v2))
    if result is None or result == "ar":
        return stem1 + stem2
    base = stem1[:-1] if last in ALL_MATRAS else stem1
    tail = stem2[len(INDEPENDENT_VOWEL[v2]):]
    return base + MATRA[result] + tail


# ------------------------------------------------------------------ splitter
@dataclass
class Split:
    word: str
    parts: list
    rule: str                      # "concat", "sandhi:a+ā->ā", or "none"
    score: float = 0.0
    alternatives: int = 0

    @property
    def is_split(self) -> bool:
        return len(self.parts) > 1


@dataclass
class SandhiSplitter:
    """Lexicon-driven inverse-sandhi splitter."""
    lexicon: set
    max_parts: int = 3
    min_part_chars: int = 2
    name: str = "inverse-sandhi+lexicon"
    stats: dict = field(default_factory=lambda: {"ambiguous": 0, "unsplit": 0})

    # -- candidate generation at one seam -------------------------------
    def _candidates_at(self, word: str, i: int) -> list:
        out = []
        a, b = word[:i], word[i:]
        if not a or not b:
            return out

        # 1. plain concatenation, no sandhi at the seam
        out.append((a, b, "concat"))

        # 2. the seam carries a matra produced by two coalescing vowels
        last = a[-1]
        if last in ALL_MATRAS:
            seam_vowel = MATRA_TO_VOWEL[last]
            for v1, v2 in INVERSE_SANDHI.get(seam_vowel, []):
                stem1 = a[:-1] + MATRA[v1]
                stem2 = INDEPENDENT_VOWEL[v2] + b
                if stem1 and not stem1.endswith(VIRAMA):
                    out.append((stem1, stem2, f"sandhi:{v1}+{v2}->{seam_vowel}"))
        return out

    @staticmethod
    def _rank(parts: list) -> tuple:
        """Preference order over complete decompositions.

        Fewer pieces first (an analysis that invents morphemes is worse than one
        that does not), then the one whose smallest piece is largest, which
        rejects the degenerate two-character fragments any lexicon will admit.
        Deliberately simple and fixed -- nothing here is fitted to the data.
        """
        return (-len(parts), min(len(p) for p in parts), sum(len(p) for p in parts))

    def _decompose(self, word: str, depth: int, memo: dict):
        """Best full decomposition of `word` into lexicon morphemes, or None.

        Recursive rather than a single cut, because महाविद्यालय only resolves as
        महा + (विद्या + आलय): the middle piece is itself a compound, so a
        one-shot binary split cannot reach the correct analysis.
        """
        key = (word, depth)
        if key in memo:
            return memo[key]

        best = None
        n_valid = 0
        if word in self.lexicon:
            best = ([word], "leaf")
            n_valid = 1

        if depth < self.max_parts:
            for i in range(1, len(word)):
                for stem1, stem2, rule in self._candidates_at(word, i):
                    if len(stem1) < self.min_part_chars or len(stem2) < self.min_part_chars:
                        continue
                    if stem1 not in self.lexicon:
                        continue
                    sub = self._decompose(stem2, depth + 1, memo)
                    if sub is None:
                        continue
                    n_valid += 1
                    cand = ([stem1] + sub[0], f"{rule}+{sub[1]}")
                    if best is None or self._rank(cand[0]) > self._rank(best[0]):
                        best = cand
        memo[key] = best
        memo[(word, "n")] = max(memo.get((word, "n"), 0), n_valid)
        return best

    def split(self, word: str) -> Split:
        memo: dict = {}
        best = self._decompose(word, 1, memo)
        if best is None or len(best[0]) == 1:
            self.stats["unsplit"] += 1
            return Split(word, [word], "none", 0.0)
        alts = max(0, memo.get((word, "n"), 1) - 1)
        if alts:
            self.stats["ambiguous"] += 1
        return Split(word, best[0], best[1], 0.0, alts)


class MorfessorSplitter:
    """Adapter for indic-nlp-library's unsupervised morph analyzer.

    Present so the pipeline is genuinely pluggable. `available()` is False in
    this environment because the resource bundle cannot be downloaded, and the
    experiment reports that fact instead of pretending the analyzer ran.
    """
    name = "indic-nlp-library/morfessor"

    def __init__(self, lang: str = "hi"):
        self.lang = lang
        self._analyzer = None

    @staticmethod
    def available() -> bool:
        res = os.environ.get("INDIC_RESOURCES_PATH")
        if not res:
            return False
        return os.path.exists(os.path.join(res, "morph", "morfessor", "hi.model"))

    def split(self, word: str) -> Split:
        if self._analyzer is None:
            from indicnlp.morph import unsupervised_morph
            self._analyzer = unsupervised_morph.UnsupervisedMorphAnalyzer(self.lang)
        parts = self._analyzer.morph_analyze(word)
        return Split(word, list(parts), "morfessor", float(len(parts)))


class GoldSplitter:
    """Uses the annotated boundaries. This is an upper bound, not a method --
    it answers 'how much is available if segmentation were perfect?'"""
    name = "gold-annotation"

    def __init__(self, gold: dict):
        self.gold = gold

    def split(self, word: str) -> Split:
        parts = self.gold.get(word)
        return Split(word, list(parts) if parts else [word],
                     "gold" if parts else "none", 0.0)


class FixedPointSplitter:
    """Ablation: split at a byte-balanced midpoint, ignoring morphology.

    This is the control that decides whether the effect is morphological or
    merely a consequence of using shorter pieces. It gets the same truncation
    relief as a real split while landing on a linguistically meaningless seam.
    """
    name = "midpoint-control"

    def __init__(self, min_part_chars: int = 2):
        self.min_part_chars = min_part_chars

    def split(self, word: str) -> Split:
        if len(word) < 2 * self.min_part_chars:
            return Split(word, [word], "none", 0.0)
        i = len(word) // 2
        return Split(word, [word[:i], word[i:]], "midpoint", 0.0)
