"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # Import tương đối (có dấu chấm) để đồng nhất với các module khác trong src/
    # và để `from src.task5_semantic_search import ...` trong tests chạy được.
    from .task4_chunking_indexing import get_collection, get_embedding_model

    collection = get_collection()
    if collection.count() == 0:
        # Chưa index → trả list rỗng thay vì raise, để Task 9 fallback hoạt động
        return []

    model = get_embedding_model()
    query_vector = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB lưu cosine distance = 1 - cosine_similarity
        # distance=0 → identical (score=1), distance=1 → perpendicular (score=0)
        score = round(max(0.0, 1.0 - dist), 4)
        output.append({"content": doc, "score": score, "metadata": meta or {}})

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Corpus: 3 luật VN (Doanh nghiệp 2020, TMĐT 2025, Giáo dục ĐH 2018) + 7 bài Shopee Seller
    for q in [
        "Quy trình Shopee thanh toán cho Người bán",
        "Điều kiện thành lập doanh nghiệp theo Luật Doanh nghiệp 2020",
    ]:
        print(f"\nQuery: {q}")
        for r in semantic_search(q, top_k=5):
            print(f"  [{r['score']:.3f}] {r['content'][:100]}...")
