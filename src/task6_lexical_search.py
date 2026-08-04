"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def load_corpus() -> list[dict]:
    """Load toàn bộ file markdown từ data/standardized/ làm corpus."""
    corpus = []
    if not STANDARDIZED_DIR.exists():
        return corpus

    for filepath in STANDARDIZED_DIR.rglob("*.md"):
        if filepath.name.startswith("."):
            continue
        text = filepath.read_text(encoding="utf-8").strip()
        if text:
            # Chia nhỏ nội dung thành các paragraph/chunk đơn giản theo dòng trống
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for idx, p in enumerate(paragraphs):
                corpus.append({
                    "content": p,
                    "metadata": {
                        "source": str(filepath.relative_to(STANDARDIZED_DIR)),
                        "chunk_id": f"{filepath.stem}_{idx}"
                    }
                })
    return corpus


CORPUS: list[dict] = load_corpus()
_TOKENIZED_CORPUS = [doc["content"].lower().split() for doc in CORPUS]
_BM25_INDEX = BM25Okapi(_TOKENIZED_CORPUS) if _TOKENIZED_CORPUS else None


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not CORPUS or _BM25_INDEX is None:
        return []

    tokenized_query = query.lower().split()
    scores = _BM25_INDEX.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("doanh nghiệp", top_k=5)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"[{r['score']:.3f}] {r['metadata']['source']} -> {r['content'][:100]}...")
