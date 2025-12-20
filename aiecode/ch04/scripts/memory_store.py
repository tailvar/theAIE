"""tiny local memory store with cosine retrieval.

Implements:
- MemoryEntry dataclass (text note)
- MemoryStore with add(), retrieve_topk(), summarize_last_n()
- Simple bag-of-words vectors and cosine similarity (NumPy only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from unicodedata import combining

import numpy as np  # Vector math for retrieval.


# tag::ch05_memory[]
@dataclass
class MemoryEntry:
    text: str  # Raw note, observation, or fact.
    tags: List[str] # Semantic hints e.g. ["numbers", "math"]


class MemoryStore:
    def __init__(self) -> None:
        self.entries: List[MemoryEntry] = []  # Append-only episodic log.
        self.vocab: Dict[str, int] = {}  # Token → index for bag-of-words.

    def _tokenize(self, text: str) -> List[str]:
        # Very small tokenizer: lowercase and split on whitespace/punct.
        import re

        toks = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return toks

    def _vectorize(self, text: str, *, update_vocab: bool = True) -> np.ndarray:
        # Build/update vocabulary and return a frequency vector.
        toks = self._tokenize(text)
        for t in toks:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
        vec = np.zeros(len(self.vocab), dtype=float)
        for t in toks:
            idx = self.vocab.get(t)
            if idx is not None:
                vec[self.vocab[t]] += 1.0

        # L2-normalize to turn dot product into cosine similarity.
        norm = np.linalg.norm(vec) or 1.0
        return vec / norm

    def add(self, *, text: str, tags: List[str] | None = None) -> None:
        # Append entry; update vocab lazily (vector created on demand).
        self.entries.append(MemoryEntry(text=text, tags=tags or []))

    def retrieve_topk(self, *, query: str, k: int = 1) -> List[Tuple[str, float]]:
        if not self.entries:
            return []
        # 1) Freeze vocab for this retrieval: ensure vocab contains queries + all entries.
        _ = self._vectorize(query, update_vocab=True)
        for e in self.entries:
            _ = self._vectorize(e.text, update_vocab=True)

        # 2) Now vectorize without changing vocab
        q = self._vectorize(query, update_vocab=False)
        q_tokens = set(self._tokenize(query))

        scores = []
        for e in self.entries:
            v = self._vectorize(e.text, update_vocab=False)
            base = float(np.dot(q, v))  # Cosine similarity in [0, 1].

            # --- tag bonus ---
            tag_overlap = len(q_tokens.intersection(set(e.tags)))
            tag_bonus = 0.2 * tag_overlap

            score = base + tag_bonus
            scores.append((e.text, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def summarize_last_n(self, *, n: int = 3, max_words: int = 20) -> str:
        # Very small summary: take last n entries and keep key tokens.
        last = [e.text for e in self.entries[-n:]]
        joined = " ".join(last)
        toks = self._tokenize(joined)
        # Keep numbers and capitalized words from original fragments.
        keep: List[str] = []
        for frag in last:
            for w in frag.split():
                if w.isdigit() or (w[:1].isupper() and w[1:].islower()):
                    keep.append(w)
        # Fallback to frequent tokens if keep is empty.
        if not keep:
            from collections import Counter

            counts = Counter(toks)
            keep = [w for w, _ in counts.most_common(max_words)]
        return " ".join(keep[:max_words])

    def semantic_summary(selftexts: List[str], *, max_words: int=12) -> str:
        """Keep only capitalized words and numbers from the given texts."""
        keep: List[str] = []

        for text in texts:
            for word in text.split():
                # Numbers
                if word.isdigit():
                    keep.append(word)
                # capitalised words (simple heuristic for proper nouns)
                elif word[:1].isupper() and word[1:].islower():
                    keep.append(word)

        # Duplicate while preserving order
        seen = set()
        filtered = []
        for w in keep:
            if w not in seen:
                seen.add(w)
                filtered.append(w)

        return " ".join(filtered[:max_words])



def _demo() -> None:
    store = MemoryStore()
    store.add(text="Calculate 2 and 3 today")
    store.add(text="Email Alice the PDF report")
    store.add(text="Team meeting at 10am")
    # print(store.retrieve_topk(query="add numbers", k=1))
    # print(store.summarize_last_n(n=2, max_words=6))
    #
    # last_two = [e.text for e in store.entries[-2:]]
    # combined = " ".join(last_two)
    # summary = store.summarize_last_n(n=2, max_words=6)
    #
    # print("last_two", last_two)
    # print("combined:", combined)
    # print("summary", summary)
    # print("shorter?", len(summary) < len(combined))

    # Query that matches a tag
    results = store.retrieve_topk(query="add_numbers, k=1")
    print(results)




if __name__ == "__main__":
    _demo()









