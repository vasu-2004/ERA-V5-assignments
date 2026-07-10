"""
bpe.py -- a from-scratch, byte-level BPE trainer + encoder.

Design (documented so results are reproducible / auditable):
  - Pretokenization: split on Unicode whitespace only (Python's default str.split()).
    This never merges across word boundaries (standard GPT-2-style behaviour), and
    treats zero-width joiners/non-joiners as part of the word they sit inside (they
    are not whitespace), which is linguistically correct for Devanagari/Telugu text.
  - Base alphabet: raw UTF-8 bytes (0..255). This is "byte-level BPE" (GPT-2/RoBERTa
    style) -- it can represent any input with zero unknown-token risk, and the base
    256-token alphabet is shared for free across every language (not a per-language
    cost).
  - Training corpus weighting for the baseline run: each language's word-frequency
    counts (from its own raw file) are summed directly, unweighted ("equal weights"
    = no per-language boost/discount is applied; a language that appears more/less
    in its own article simply contributes that many real occurrences).
  - Merge selection: standard BPE -- at each step, merge the most frequent adjacent
    symbol pair across the whole corpus. Ties are broken deterministically (lowest
    pair value first) so re-running training on the same input is exactly
    reproducible.
  - Efficiency: uses the standard incremental algorithm (maintain global pair
    counts + a reverse index of which words contain each pair; after a merge, only
    the affected words are rescanned) rather than rescanning the whole corpus on
    every merge step -- this is what makes 10,000+ merges tractable in pure Python.
"""
from collections import Counter, defaultdict


def word_freqs_from_text(text):
    """Whitespace-split word counts (Counter[str, int]) for one corpus."""
    return Counter(text.split())


def bytes_tuple(word):
    return tuple(word.encode("utf-8"))


def train_bpe(word_freq_counter, max_merges, log_every=None):
    """
    word_freq_counter: Counter[str -> int] (word string -> occurrence count)
    Returns: (merges, id_to_bytes)
      merges: ordered list of (a, b, new_id) in the order they were learned
      id_to_bytes: dict new_id -> bytes represented by that id (256.. onward)
    """
    words = []      # list[list[int]] current symbol sequence per unique word
    freqs = []       # list[int] frequency of that unique word
    for w, f in word_freq_counter.items():
        words.append(list(bytes_tuple(w)))
        freqs.append(f)

    pair_counts = Counter()
    where = defaultdict(set)  # pair -> set of word-indices currently containing it
    for i, w in enumerate(words):
        f = freqs[i]
        for a, b in zip(w, w[1:]):
            pair_counts[(a, b)] += f
            where[(a, b)].add(i)

    id_to_bytes = {i: bytes([i]) for i in range(256)}
    next_id = 256
    merges = []

    for step in range(max_merges):
        if not pair_counts:
            break
        best = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
        (a, b), cnt = best
        if cnt < 2:
            break  # nothing left worth merging
        new_id = next_id
        next_id += 1
        merges.append((a, b, new_id))
        id_to_bytes[new_id] = id_to_bytes[a] + id_to_bytes[b]

        affected = list(where.get((a, b), ()))
        for i in affected:
            w = words[i]
            f = freqs[i]
            old_pairs = Counter(zip(w, w[1:]))
            new_w = []
            j = 0
            n = len(w)
            while j < n:
                if j < n - 1 and w[j] == a and w[j + 1] == b:
                    new_w.append(new_id)
                    j += 2
                else:
                    new_w.append(w[j])
                    j += 1
            words[i] = new_w
            new_pairs = Counter(zip(new_w, new_w[1:]))
            for p, c in old_pairs.items():
                pair_counts[p] -= c * f
                if pair_counts[p] <= 0:
                    del pair_counts[p]
            for p, c in new_pairs.items():
                pair_counts[p] += c * f
                where[p].add(i)
        pair_counts.pop((a, b), None)
        where.pop((a, b), None)

        if log_every and (step + 1) % log_every == 0:
            print(f"  merge {step + 1}/{max_merges}  best_pair_count={cnt}")

    return merges, id_to_bytes


def encode_word(word, merge_rank):
    """
    word: python str (one whitespace-delimited token)
    merge_rank: dict (a,b) -> (rank, new_id), rank = merge order (lower = earlier/higher priority)
    Returns: list of final token ids for this single word.
    """
    symbols = list(bytes_tuple(word))
    if len(symbols) <= 1:
        return symbols
    while True:
        best_rank = None
        best_pos = None
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            info = merge_rank.get(pair)
            if info is not None and (best_rank is None or info[0] < best_rank):
                best_rank = info[0]
                best_pos = i
        if best_pos is None:
            break
        new_id = merge_rank[(symbols[best_pos], symbols[best_pos + 1])][1]
        symbols = symbols[:best_pos] + [new_id] + symbols[best_pos + 2:]
    return symbols


def build_merge_rank(merges, upto):
    """merges[:upto] -> dict (a,b) -> (rank, new_id)"""
    return {(a, b): (rank, new_id) for rank, (a, b, new_id) in enumerate(merges[:upto])}


def fertility(unique_words, merge_rank):
    """Average tokens-per-unique-word (the assignment's X ratio) over a word list."""
    if not unique_words:
        return 0.0, 0, 0
    total_tokens = 0
    for w in unique_words:
        total_tokens += len(encode_word(w, merge_rank))
    return total_tokens / len(unique_words), total_tokens, len(unique_words)
