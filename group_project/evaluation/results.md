# 📊 Báo Cáo Đánh Giá RAG Pipeline (RAGAS)

*Sinh tự động bởi `eval_pipeline.py` — 2026-08-04 15:33*

## 1. Thiết lập

- **Golden dataset**: 23 cặp Q&A (`golden_dataset.json`)
- **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` (384d)
- **Chunking**: 800 ký tự / overlap 100
- **LLM (generation + judge)**: `openai/gpt-4o-mini` qua OpenRouter
- **Embeddings cho RAGAS**: local (OpenRouter không có endpoint embeddings)
- **Fallback threshold**: 0.52 (cosine gốc của semantic search)

## 2. So sánh A/B

| Metric | Hybrid (semantic + BM25) | Dense-only (chỉ semantic) | Chênh lệch |
|---|---|---|---|
| **faithfulness** | `0.7652` | `0.6400` | 🔺 `+0.1253` |
| **answer_relevancy** | `0.7396` | `0.6353` | 🔺 `+0.1044` |
| **context_recall** | `0.7391` | `0.6812` | 🔺 `+0.0580` |
| **context_precision** | `0.7129` | `0.6336` | 🔺 `+0.0793` |
| **TRUNG BÌNH** | `0.7392` | `0.6475` |🔺 `+0.0917` |

*Cột chênh lệch = config A − config B (dương nghĩa là A tốt hơn).*

## 3. Worst performers

### Hybrid (semantic + BM25)

| Câu hỏi | TB | faithfulness | answer_relevancy | context_recall | context_precision |
|---|---|---|---|---|---|
| Hệ thống điểm phạt áp dụng đối với người bán vi phạm chính sách của Sh | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| Người bán cần thực hiện những thủ tục gì để bảo vệ tài khoản và nhãn h | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| Quyền của doanh nghiệp bao gồm những điều gì cơ bản? | `0.252` | `0.222` | `0.785` | `0.000` | `0.000` |
| Các gói dịch vụ hỗ trợ vận hành người bán trên Sàn Shopee bao gồm nhữn | `0.473` | `1.000` | `0.892` | `0.000` | `0.000` |
| Phương thức thanh toán và khấu trừ chi phí vận hành gian hàng trên sàn | `0.570` | `0.714` | `0.565` | `0.000` | `1.000` |

### Dense-only (chỉ semantic)

| Câu hỏi | TB | faithfulness | answer_relevancy | context_recall | context_precision |
|---|---|---|---|---|---|
| Doanh nghiệp tư nhân có được phát hành bất kỳ loại chứng khoán nào khô | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| Hệ thống điểm phạt áp dụng đối với người bán vi phạm chính sách của Sh | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| Các hành vi gian lận như giao dịch ngoài sàn và vi phạm giá ảo trên Sh | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| Người bán cần thực hiện những thủ tục gì để bảo vệ tài khoản và nhãn h | `0.000` | `0.000` | `0.000` | `0.000` | `0.000` |
| Quy trình vận chuyển và đóng gói hàng hóa trên Shopee cần tuân thủ nhữ | `0.083` | `0.000` | `0.000` | `0.333` | `0.000` |

