"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")


def cosine_sim(a, b) -> float:
    """Cosine similarity giữa 2 vector."""
    vec_a, vec_b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []
    if not JINA_API_KEY:
        raise RuntimeError(
            "Thiếu JINA_API_KEY trong .env — dùng method='rrf' nếu chưa có API key"
        )

    import requests

    # jina-reranker-v2-base-multilingual: cross-encoder đa ngôn ngữ, xử lý tốt
    # tiếng Việt — cần thiết vì corpus là luật VN + bài Shopee tiếng Việt.
    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {JINA_API_KEY}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": min(top_k, len(candidates)),
        },
        timeout=30,
    )
    response.raise_for_status()

    reranked = response.json()["results"]
    return [
        {**candidates[r["index"]], "score": float(r["relevance_score"])}
        for r in reranked
    ]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []
    if any("embedding" not in c for c in candidates):
        raise ValueError("rerank_mmr yêu cầu mỗi candidate có key 'embedding'")

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Độ tương đồng lớn nhất với các chunk ĐÃ chọn → phạt trùng lặp
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(
                    candidates[idx]["embedding"], candidates[sel_idx]["embedding"]
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = (
                lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)
        # Ghi lại điểm MMR để tầng trên (Task 9/10) thấy được thứ tự đã chọn
        candidates[best_idx] = {**candidates[best_idx], "score": round(best_score, 6)}

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # MMR cần vector: embed query bằng đúng model đã index ở Task 4
        from .task4_chunking_indexing import get_embedding_model

        query_embedding = get_embedding_model().encode(query).tolist()
        return rerank_mmr(query_embedding, candidates, top_k=top_k)
    elif method == "rrf":
        # Wrap candidates as a single ranked list for RRF fusion
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Dummy data theo đúng chủ đề corpus: TMĐT / Shopee Seller + luật VN
    dummy_candidates = [
        {"content": "Quy trình Shopee thanh toán cho Người bán và cách sử dụng Số dư TK Shopee", "score": 0.8, "metadata": {}},
        {"content": "Chính sách Trả hàng/Hoàn tiền và quy trình bồi thường cho Người bán", "score": 0.6, "metadata": {}},
        {"content": "Luật Doanh nghiệp 2020 quy định về vốn điều lệ của công ty TNHH", "score": 0.5, "metadata": {}},
    ]
    results = rerank("Shopee thanh toán cho người bán", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
