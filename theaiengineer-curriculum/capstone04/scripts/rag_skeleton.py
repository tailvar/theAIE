
"""Toy retrieval + answer skeleton without external dependencies.

Implements a tiny TF‑IDF‑like index and cosine similarity in pure Python over a
small in‑memory corpus, then synthesizes a placeholder answer by citing top
passages.
"""

from __future__ import annotations  # postpone annotations

import argparse  # parse CLI flags
import math  # idf and cosine helpers
from collections import Counter, defaultdict  # term counting
from dataclasses import dataclass  # compact outputs
from typing import Dict, Iterable, List, Tuple  # type hints
import json
from datetime import datetime
from pathlib import Path
from typing import List

LOG_PATH = Path("runs.json1")


CORPUS: List[str] = [  # small sample corpus
    "The mean of numbers is the sum divided by the count.",  # mean
    "The median is the middle value when data are sorted.",  # median
    "Standard deviation measures dispersion around the mean.",  # stdev
    "A vector database stores embeddings for fast similarity search.",  # vec db
    "Retrieval augmented generation uses search to ground outputs.",  # rag
]


def tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, split on spaces; trivial tokenizer."""
    table = str.maketrans({c: " " for c in ",.;:()[]{}!?"})  # punct to space
    return [t for t in text.lower().translate(table).split() if t]  # tokens


def build_index(corpus: List[str]) -> Tuple[List[Counter], Dict[str, float]]:
    """Return term frequencies per doc and idf per term."""
    docs = [Counter(tokenize(doc)) for doc in corpus]  # counts per doc
    df = defaultdict(int)  # document frequencies
    for d in docs:  # each doc
        for term in d:  # unique terms per doc
            df[term] += 1  # increment
    N = len(corpus)  # doc count
    idf = {t: math.log((1 + N) / (1 + c)) + 1.0 for t, c in df.items()}  # smoothed
    return docs, idf  # index parts


def tfidf_vec(counts: Counter, idf: Dict[str, float]) -> Dict[str, float]:
    """Compute a sparse TF‑IDF vector as a dict."""
    total = sum(counts.values()) or 1  # normalizer
    vec = {t: (c / total) * idf.get(t, 0.0) for t, c in counts.items()}  # weights
    return vec  # vector


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity for sparse dict vectors."""
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in set(a) | set(b))  # dot
    na = math.sqrt(sum(v * v for v in a.values()))  # norm a
    nb = math.sqrt(sum(v * v for v in b.values()))  # norm b
    if na == 0.0 or nb == 0.0:  # degenerate
        return 0.0  # zero sim
    return dot / (na * nb)  # cosine

def log_run(query:str, hits: List[Hit], answer: str) -> None:
    """Append a compact JSONL record for this run."""
    record = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "query": query,
        "retrieved_ids": [h.doc_id for h in hits],
        "answer": answer,
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass  # retrieval hit
class Hit:
    score: float  # similarity
    doc_id: int  # index in corpus
    text: str  # passage text


def retrieve(query: str, corpus: List[str], k: int = 2) -> List[Hit]:
    """Return top‑k passages for a query using the tiny TF‑IDF index."""
    docs, idf = build_index(corpus)  # index
    q_vec = tfidf_vec(Counter(tokenize(query)), idf)  # query vector
    hits = []  # collector
    for i, counts in enumerate(docs):  # each doc
        d_vec = tfidf_vec(counts, idf)  # doc vector
        score = cosine(q_vec, d_vec)  # similarity
        hits.append(Hit(score, i, corpus[i]))  # collect hit
    hits.sort(key=lambda h: h.score, reverse=True)  # rank
    return hits[:k]  # top-k


def synthesize_answer(query: str, hits: List[Hit]) -> str:
    """Produce a templated answer citing top passages."""
    citations = " ".join(f"[{h.doc_id}]" for h in hits)  # cites
    summary = " ".join(h.text for h in hits)  # stitched
    return (  # templated answer
        f"Q: {query}\n"
        f"A: Based on {citations}, {summary}"
    )


def build_parser() -> argparse.ArgumentParser:  # CLI builder
    p = argparse.ArgumentParser(description="Toy retrieval + answer demo.")  # parser
    p.add_argument("--query", required=True, help="User question.")  # input
    p.add_argument("--k", type=int, default=2, help="Top‑k passages")  # top-k
    return p  # return


def main() -> None:  # entry point
    args = build_parser().parse_args()  # parse flags
    hits = retrieve(args.query, CORPUS, k=args.k)  # retrieve docs
    for h in hits:  # show rows
        print(f"score={h.score:.3f} id={h.doc_id} text={h.text}")  # hit row
    print("---")  # separator
    answer=synthesize_answer(args.query, hits)  # templated answer
    print(answer)

    # log the run
    log_run(args.query, hits, answer)

if __name__ == "__main__":  # script entry point
    main()