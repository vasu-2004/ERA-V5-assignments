"""verify.py -- independent verification of the shipped tokenizer.json:
   1) reload the saved file (exactly as the instructor will) and recompute fertility
   2) prove zero-UNK on a stress string of rare/mixed characters
   3) confirm exact vocab size and roundtrip
   4) whole-doc train the top candidate weights to confirm the winner is robust
"""
import json, re
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; ART = ROOT / "artifacts"
LANGS = ["en", "hi", "te", "mr"]; FILES = {"en":"english","hi":"hindi","te":"telugu","mr":"marathi"}
TEXTS = {l:(DATA/f"{FILES[l]}_india.txt").read_text(encoding="utf-8") for l in LANGS}
WORDS = {l: len(re.findall(r"\w+", TEXTS[l])) for l in LANGS}

print("=== 1) reload shipped tokenizer.json and recompute fertility (grader's code path) ===")
tok = Tokenizer.from_file(str(ART / "tokenizer.json"))
print("vocab size:", tok.get_vocab_size())
xs = {}
for l in LANGS:
    ids = tok.encode(TEXTS[l]).ids
    xs[l] = len(ids) / WORDS[l]
    print(f"  {l}: tokens={len(ids):>6} words={WORDS[l]:>6} X={xs[l]:.4f}")
xmax, xmin = max(xs.values()), min(xs.values())
print(f"  X_max={xmax:.4f}({max(xs,key=xs.get)}) X_min={xmin:.4f}({min(xs,key=xs.get)}) spread={xmax-xmin:.4f} score={1000/(xmax-xmin):.0f}")
print("  English <= 1.2:", xs["en"] <= 1.2)

print("\n=== 2) zero-UNK stress test ===")
stress = ("India](/wiki/India) — भारत, గణతంత్ర, महाराष्ट्र! 42% ₹100 "
          "‌‍\t\n©®™ emoji-free ĀīŚṇ ✓ \"quotes\" 'x' <tag> [[wikilink]] {json:1}")
enc = tok.encode(stress)
unk = [t for t in enc.tokens if t == "[UNK]" or t is None]
print("  tokens:", len(enc.ids), " any UNK:", bool(unk))
# byte-level always roundtrips exactly:
dec = tok.decode(enc.ids)
print("  exact roundtrip on stress string:", dec == stress)
# also every language's full text must roundtrip exactly (=> every char encodable)
allrt = all(tok.decode(tok.encode(TEXTS[l]).ids) == TEXTS[l] for l in LANGS)
print("  exact roundtrip on ALL four full corpora:", allrt)

print("\n=== 3) base alphabet check (why UNK is impossible) ===")
vocab = tok.get_vocab()
alpha = pre_tokenizers.ByteLevel.alphabet()
missing = [c for c in alpha if c not in vocab]
print(f"  all 256 byte-level base chars present in vocab: {len(missing)==0} (missing {len(missing)})")

print("\n=== 4) whole-doc train top candidate weights, compare on whole-doc ===")
LINES = {l: [x for x in TEXTS[l].split('\n') if x] for l in LANGS}
def build(w):
    t = Tokenizer(models.BPE(unk_token=None))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    t.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=10000, show_progress=False,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
    corpus=[]
    for l in LANGS: corpus += [TEXTS[l]]*w[l]
    t.train_from_iterator(corpus, trainer=tr); return t
for w in [{"en":7,"hi":1,"te":2,"mr":1},{"en":8,"hi":1,"te":2,"mr":1},{"en":6,"hi":1,"te":2,"mr":1},
          {"en":7,"hi":1,"te":3,"mr":1},{"en":9,"hi":1,"te":2,"mr":1}]:
    t = build(w); fx={l:len(t.encode(TEXTS[l]).ids)/WORDS[l] for l in LANGS}
    sp=max(fx.values())-min(fx.values())
    print(f"  w={w}  "+" ".join(f'{l}={fx[l]:.4f}' for l in LANGS)
          +f"  spread={sp:.4f} score={1000/sp:.0f} enOK={fx['en']<=1.2}")
