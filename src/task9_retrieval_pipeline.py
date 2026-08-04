"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Đã calibrate bằng cách đo trực tiếp trên corpus này (BAAI/bge-m3, 4176 chunks):
#   Query liên quan  : 0.719 - 0.794  (thanh toán Shopee, trả hàng, vận chuyển, Luật DN)
#   Query lạc đề/rác : 0.317 - 0.437  (chuỗi ngẫu nhiên, nấu phở, train GPU)
# Hai nhóm tách bạch rõ → chọn ngưỡng nằm giữa (~0.58).
# Lưu ý: bge-m3 cho cosine nền khá cao (~0.3-0.44 kể cả với text hoàn toàn lạc đề),
# nên ngưỡng 0.3 như giá trị mẫu ban đầu nằm DƯỚI cả sàn của query rác → fallback
# không bao giờ kích hoạt. Đổi embedding model thì phải đo lại.
SCORE_THRESHOLD = 0.58   # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Run semantic + lexical search
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Step 2: Merge with RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Rerank (optional)
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    # Step 4: Check fallback using ORIGINAL cosine score (NOT RRF score)
    # Per LAB_GUIDE warning: RRF max ≈ 1/(k+1) ≈ 0.0164 regardless of relevance
    best_cosine = dense_results[0]["score"] if dense_results else 0.0
    if best_cosine < score_threshold:
        print(f"  Semantic best score ({best_cosine:.3f}) < threshold ({score_threshold}) -> PageIndex fallback")
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        # News (Shopee Seller Education)
        "Quy trình Shopee thanh toán cho Người bán",
        "Các phương thức vận chuyển Shopee hỗ trợ",
        # Legal (luật VN)
        "Điều kiện thành lập doanh nghiệp theo Luật Doanh nghiệp 2020",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
