"""The evaluation set: Devanagari compounds with annotated morpheme boundaries.

Design notes, because the composition of this list is what makes the headline
experiment non-circular:

*  Compounds are grouped into FAMILIES that share a morpheme. Some families
   share their FIRST morpheme (देव-, राज-, महा-) and some share a LATER one
   (-आलय, -ईश्वर, -आत्मा, -इन्द्र). That contrast is the whole experiment: the
   codec is positionally rigid, so it can already see a shared prefix, but a
   shared non-initial morpheme sits at a different byte offset in each word and
   is therefore invisible to it. Segmentation is what re-aligns those.

*  Long inflected variants of long compounds are included on purpose
   (विश्वविद्यालय / विश्वविद्यालयों / विश्वविद्यार्थी). At dp=32 they share all
   32 surviving bytes and therefore collide to one identical vector.

*  A few words use yaṇ sandhi (अत्यन्त, इत्यादि, स्वागत), which the rule engine
   does NOT model. They are kept in so that reported splitter accuracy is real.

*  The lexicon carries ~60 morphemes that are not constituents of anything here,
   so the splitter has to choose among competing lexicon-valid seams rather than
   effectively looking up an answer.
"""
from __future__ import annotations

# word -> gold morpheme decomposition
COMPOUNDS: dict = {
    # ---- family: -आलय  (shared NON-INITIAL morpheme) ----
    "देवालय": ["देव", "आलय"],
    "शिवालय": ["शिव", "आलय"],
    "हिमालय": ["हिम", "आलय"],
    "विद्यालय": ["विद्या", "आलय"],
    "पुस्तकालय": ["पुस्तक", "आलय"],
    "भोजनालय": ["भोजन", "आलय"],
    "चिकित्सालय": ["चिकित्सा", "आलय"],
    "कार्यालय": ["कार्य", "आलय"],
    "न्यायालय": ["न्याय", "आलय"],
    "ग्रन्थालय": ["ग्रन्थ", "आलय"],
    "महाविद्यालय": ["महा", "विद्या", "आलय"],
    "विश्वविद्यालय": ["विश्व", "विद्या", "आलय"],

    # ---- family: -ईश्वर / -ईश ----
    "परमेश्वर": ["परम", "ईश्वर"],
    "रामेश्वर": ["राम", "ईश्वर"],
    "योगेश्वर": ["योग", "ईश्वर"],
    "महेश": ["महा", "ईश"],
    "गणेश": ["गण", "ईश"],
    "दिनेश": ["दिन", "ईश"],

    # ---- family: -इन्द्र ----
    "देवेन्द्र": ["देव", "इन्द्र"],
    "नरेन्द्र": ["नर", "इन्द्र"],
    "सुरेन्द्र": ["सुर", "इन्द्र"],
    "राजेन्द्र": ["राज", "इन्द्र"],

    # ---- family: -आत्मा ----
    "महात्मा": ["महा", "आत्मा"],
    "परमात्मा": ["परम", "आत्मा"],
    "धर्मात्मा": ["धर्म", "आत्मा"],
    "जीवात्मा": ["जीव", "आत्मा"],

    # ---- family: -उदय / -उत्तम / -उत्सव  (a/ā + u -> o) ----
    "सूर्योदय": ["सूर्य", "उदय"],
    "चन्द्रोदय": ["चन्द्र", "उदय"],
    "पुरुषोत्तम": ["पुरुष", "उत्तम"],
    "सर्वोत्तम": ["सर्व", "उत्तम"],
    "महोत्सव": ["महा", "उत्सव"],

    # ---- shared FIRST morpheme (the codec's easy case) ----
    "राजकुमार": ["राज", "कुमार"],
    "राजकुमारी": ["राज", "कुमारी"],
    "राजभवन": ["राज", "भवन"],
    "रामकृष्ण": ["राम", "कृष्ण"],
    "रामायण": ["राम", "अयन"],
    "देवदत्त": ["देव", "दत्त"],
    "जलाशय": ["जल", "आशय"],
    "जलधारा": ["जल", "धारा"],
    "हिमाचल": ["हिम", "अचल"],
    "सत्याग्रह": ["सत्य", "आग्रह"],
    "विद्यार्थी": ["विद्या", "अर्थी"],
    "भारतवर्ष": ["भारत", "वर्ष"],
    "सूर्यास्त": ["सूर्य", "अस्त"],

    # ---- long / inflected: the truncation-collision cases ----
    "विश्वविद्यालयों": ["विश्व", "विद्या", "आलयों"],
    "विश्वविद्यार्थी": ["विश्व", "विद्या", "अर्थी"],
    "महाविद्यालयों": ["महा", "विद्या", "आलयों"],
    "राष्ट्रपति": ["राष्ट्र", "पति"],
    "उपराष्ट्रपति": ["उप", "राष्ट्र", "पति"],
    "प्रधानमन्त्री": ["प्रधान", "मन्त्री"],
    "मुख्यमन्त्री": ["मुख्य", "मन्त्री"],

    # ---- yaṇ sandhi: the rule engine is expected to FAIL on these ----
    "अत्यन्त": ["अति", "अन्त"],
    "इत्यादि": ["इति", "आदि"],
    "स्वागत": ["सु", "आगत"],
}

# Morphemes that appear as constituents above.
CONSTITUENTS = sorted({m for parts in COMPOUNDS.values() for m in parts})

# Lexicon distractors: real words that are NOT constituents of any compound
# here, so the splitter must discriminate rather than recall.
DISTRACTORS = [
    "कमल", "नगर", "पुत्र", "मित्र", "वायु", "अग्नि", "पृथ्वी", "आकाश", "सागर",
    "पर्वत", "नदी", "वृक्ष", "पुष्प", "फल", "मूल", "पत्र", "शाखा", "बीज",
    "गृह", "द्वार", "मार्ग", "सेतु", "यान", "चक्र", "ध्वज", "शस्त्र", "कवच",
    "मुकुट", "रत्न", "स्वर्ण", "रजत", "ताम्र", "लोह", "काष्ठ", "वस्त्र",
    "अन्न", "जल", "दुग्ध", "मधु", "तैल", "लवण", "मिष्ट", "तिक्त",
    "प्रात", "सायं", "दिवस", "रात्रि", "मास", "वर्ष", "काल", "समय",
    "गुरु", "शिष्य", "विद्वान", "मूर्ख", "बालक", "बालिका", "वृद्ध", "युवा",
    "श्रम", "फलक", "यन्त्र", "वाहन", "स्थल", "क्षेत्र",
]


def build_lexicon() -> set:
    return set(CONSTITUENTS) | set(DISTRACTORS)


def morpheme_families() -> dict:
    """morpheme -> the compounds containing it (families of size >= 2)."""
    fam: dict = {}
    for word, parts in COMPOUNDS.items():
        for i, m in enumerate(parts):
            fam.setdefault(m, []).append((word, i))
    return {m: v for m, v in fam.items() if len(v) >= 2}


def related_pairs() -> list:
    """Compound pairs sharing a morpheme, tagged by WHERE the morpheme sits.

    `position` is "initial" when the shared morpheme is first in BOTH words,
    and "non_initial" when it is non-first in both. Mixed cases are labelled
    "mixed" and reported separately -- lumping them together would blur exactly
    the distinction the experiment is built to measure.
    """
    out = []
    for m, members in morpheme_families().items():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                (w1, i1), (w2, i2) = members[a], members[b]
                if i1 == 0 and i2 == 0:
                    pos = "initial"
                elif i1 > 0 and i2 > 0:
                    pos = "non_initial"
                else:
                    pos = "mixed"
                out.append({"w1": w1, "w2": w2, "morpheme": m, "position": pos})
    return out


def unrelated_pairs() -> list:
    """Compound pairs sharing no morpheme at all -- the negative class."""
    words = sorted(COMPOUNDS)
    out = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if not set(COMPOUNDS[words[i]]) & set(COMPOUNDS[words[j]]):
                out.append({"w1": words[i], "w2": words[j],
                            "morpheme": None, "position": "none"})
    return out


def summary() -> dict:
    rel = related_pairs()
    return {
        "n_compounds": len(COMPOUNDS),
        "n_constituents": len(CONSTITUENTS),
        "n_lexicon": len(build_lexicon()),
        "n_distractors": len(DISTRACTORS),
        "n_families": len(morpheme_families()),
        "n_related_pairs": len(rel),
        "n_related_initial": sum(1 for p in rel if p["position"] == "initial"),
        "n_related_non_initial": sum(1 for p in rel if p["position"] == "non_initial"),
        "n_related_mixed": sum(1 for p in rel if p["position"] == "mixed"),
        "n_unrelated_pairs": len(unrelated_pairs()),
    }
