
"""Chunk, index, and retrieve from local text files; synthesize cited answers.

Reads `.txt` files, splits into paragraph chunks, builds a tiny TF‑IDF‑like
index, retrieves top‑k passages for a query, and prints a templated answer with
inline citations `source#chunk_id`.
"""

from __future__ import annotations  # postpone annotations

import argparse  # CLI
import math  # idf/cosine
from collections import Counter, defaultdict  # counts
from dataclasses import dataclass  # result rows
from pathlib import Path  # filesystem
from typing import Dict, Iterable, List, Sequence, Tuple  # types


@dataclass
class Chunk:  # chunk metadata
    source: str  # file stem  # file stem
    cid: int  # chunk id  # chunk id within the file
    text: str  # text  # chunk content


def read_txt_files(path: Path) -> List[Path]:  # expand a path to files
    if path.is_file() and path.suffix.lower() == ".txt":
        return [path]  # list
    if path.is_dir():  # directory
        return sorted(p for p in path.rglob("*.txt") if p.is_file())  # glob
    raise SystemExit(f"no .txt file(s) at: {path}")  # error


def split_paragraphs(text: str) -> List[str]:  # naive paragraph split
    paras = [p.strip() for p in text.split("\n\n")]
    return [p for p in paras if p]  # drop empties


def load_chunks(path: Path) -> List[Chunk]:  # build chunk list
    files = read_txt_files(path)  # list files
    chunks: List[Chunk] = []  # accumulator
    for f in files:  # iterate files
        paras = split_paragraphs(f.read_text(encoding="utf-8"))  # read + split
        for i, p in enumerate(paras):  # enumerate paragraphs
            chunks.append(Chunk(source=f.stem, cid=i, text=p))  # add chunk
    return chunks  # result


def tokenize(text: str) -> List[str]:  # trivial tokenizer
    table = str.maketrans({c: " " for c in ",.;:()[]{}!?"})  # punct map
    lowered = text.lower().translate(table)  # lowercase + map
    return [t for t in lowered.split() if t]  # tokens


def build_index(  # index
    chunks: Sequence[Chunk],
) -> Tuple[List[Counter], Dict[str, float]]:
    docs = [Counter(tokenize(c.text)) for c in chunks]  # per-doc counts
    df = defaultdict(int)  # document frequencies
    for d in docs:  # each doc
        for term in d:  # unique terms
            df[term] += 1  # increment
    N = max(1, len(docs))  # doc count
    idf = {  # smoothed idf
        t: math.log((1 + N) / (1 + c)) + 1.0 for t, c in df.items()
    }
    return docs, idf  # outputs


def tfidf_vec(  # vectorize
    counts: Counter, idf: Dict[str, float]
) -> Dict[str, float]:
    total = sum(counts.values()) or 1  # normalization
    return {t: (c / total) * idf.get(t, 0.0) for t, c in counts.items()}  # weights


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:  # cosine sim
    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in set(a) | set(b))  # dot
    na = math.sqrt(sum(v * v for v in a.values()))  # norm a
    nb = math.sqrt(sum(v * v for v in b.values()))  # norm b
    if na == 0.0 or nb == 0.0:  # degenerate
        return 0.0  # zero sim
    return dot / (na * nb)  # cosine


@dataclass
class Hit:  # retrieval result
    score: float  # similarity
    idx: int  # chunk list index
    source: str  # file stem
    cid: int  # chunk id
    text: str  # text


def retrieve(chunks: Sequence[Chunk], query: str, k: int) -> List[Hit]:  # search
    docs, idf = build_index(chunks)  # index
    q_vec = tfidf_vec(Counter(tokenize(query)), idf)  # query vec
    hits: List[Hit] = []  # collector
    for i, counts in enumerate(docs):  # each doc
        d_vec = tfidf_vec(counts, idf)  # doc vec
        score = cosine(q_vec, d_vec)  # similarity
        ch = chunks[i]  # chunk metadata
        hits.append(Hit(score, i, ch.source, ch.cid, ch.text))  # append
    hits.sort(key=lambda h: h.score, reverse=True)  # rank
    return hits[:k]  # top-k


def synthesize_answer(query: str, hits: Sequence[Hit]) -> str:  # format
    cites = " ".join(f"[{h.source}#{h.cid}]" for h in hits)  # citations
    body = " ".join(h.text for h in hits)  # stitched text
    return f"Q: {query}\nA: Based on {cites}, {body}"


def build_parser() -> argparse.ArgumentParser:  # CLI builder
    p = argparse.ArgumentParser(  # parser
        description="Chunk, index, and retrieve with cites."
    )
    p.add_argument(
        "--path", type=Path, required=True, help=".txt file or directory"
    )  # input path
    p.add_argument("--query", required=True, help="User question")  # query text
    p.add_argument("--k", type=int, default=2, help="Top‑k passages")  # top-k
    return p  # return


def main() -> None:  # entry point
    args = build_parser().parse_args()  # parse args
    chunks = load_chunks(args.path)  # load chunks
    hits = retrieve(chunks, args.query, args.k)  # search
    for h in hits:  # print hits
        print(f"score={h.score:.3f} id={h.source}#{h.cid} text={h.text}")
    print("---")  # separator
    print(synthesize_answer(args.query, hits))  # answer


if __name__ == "__main__":  # script guard
    main()  # run