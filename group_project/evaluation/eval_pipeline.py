"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
from pathlib import Path

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(golden_dataset: list[dict] = None) -> dict:
    """
    Evaluate RAG pipeline sử dụng RAGAS thực tế.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # Import pipeline thực tế từ Task 10
    from src.task10_generation import generate_with_citation

    if golden_dataset is None:
        golden_dataset = load_golden_dataset()

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    print("🚀 Đang truy vấn RAG Pipeline thực tế cho từng câu hỏi...")
    for item in golden_dataset:
        result = generate_with_citation(item["question"])
        answer = result.get("answer", "")
        contexts = [c.get("content", "") for c in result.get("sources", [])]

        eval_data["question"].append(item["question"])
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(item["expected_answer"])

    print("📊 Đang tính toán chỉ số RAGAS (LLM-as-a-judge)...")
    dataset = Dataset.from_dict(eval_data)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return results


def export_results(results: dict):
    """Export evaluation results to results.md"""
    content = "# 📊 Báo Cáo Đánh Giá Chất Lượng RAG Pipeline (RAGAS Benchmark)\n\n"
    content += "## 1. Điểm Số Thực Tế (Overall Metrics)\n\n"
    content += "| Metric | Score |\n"
    content += "|---|---|\n"

    try:
        df = results.to_pandas()
        for col in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            if col in df.columns:
                score = df[col].mean()
                content += f"| **{col}** | `{score:.4f}` |\n"
    except Exception:
        for k, v in results.items():
            content += f"| **{k}** | `{v}` |\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"✓ Đã xuất báo cáo đánh giá thực tế tại: {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases từ golden_dataset.json")

    # Chạy quy trình đánh giá thực tế
    results = evaluate_with_ragas(golden_dataset)
    export_results(results)
