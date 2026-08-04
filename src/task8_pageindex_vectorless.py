"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Registry ánh xạ tên file -> doc_id do PageIndex cấp, để pageindex_search dùng lại
# mà không phải upload lại mỗi lần chạy.
DOC_REGISTRY = Path(__file__).parent.parent / "data" / "pageindex_docs.json"
PDF_CACHE_DIR = Path(__file__).parent.parent / "data" / "pageindex_pdf"

# Font Unicode bắt buộc: corpus là tiếng Việt, font core của fpdf2 (Helvetica)
# chỉ hỗ trợ latin-1 nên sẽ raise UnicodeEncodeError với dấu tiếng Việt.
UNICODE_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

POLL_INTERVAL_SEC = 2
POLL_TIMEOUT_SEC = 120


# =============================================================================
# HELPERS
# =============================================================================

def _strip_markdown(text: str) -> str:
    """Bỏ ảnh/link markdown để PDF gọn, giữ lại phần chữ có nghĩa."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)      # ảnh
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # link -> giữ anchor text
    text = re.sub(r"[*_`>#]+", " ", text)                 # ký hiệu markdown
    return "\n".join(line.rstrip() for line in text.splitlines())


def markdown_to_pdf(md_file: Path, out_dir: Path = PDF_CACHE_DIR) -> Path:
    """
    Convert 1 file markdown sang PDF đơn giản bằng fpdf2.

    PageIndex nhận PDF chứ không nhận .md trực tiếp, nên phải convert trước khi upload.

    Returns:
        Đường dẫn tới file PDF đã tạo.
    """
    from fpdf import FPDF

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{md_file.stem}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    if UNICODE_FONT.exists():
        pdf.add_font("DejaVu", "", str(UNICODE_FONT))
        pdf.set_font("DejaVu", size=10)
    else:
        # Không có font Unicode → ép về ASCII để ít nhất không crash
        pdf.set_font("Helvetica", size=10)

    text = _strip_markdown(md_file.read_text(encoding="utf-8"))
    if not UNICODE_FONT.exists():
        text = text.encode("latin-1", "replace").decode("latin-1")

    for line in text.splitlines():
        if not line.strip():
            pdf.ln(4)
            continue
        pdf.multi_cell(0, 5, line)

    pdf.output(str(pdf_path))
    return pdf_path


def _load_registry() -> dict[str, str]:
    """Đọc registry {filename: doc_id}; trả dict rỗng nếu chưa upload lần nào."""
    if not DOC_REGISTRY.exists():
        return {}
    try:
        return json.loads(DOC_REGISTRY.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _get_client():
    """Khởi tạo PageIndexClient; trả None nếu thiếu API key."""
    if not PAGEINDEX_API_KEY:
        return None
    from pageindex.client import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


# =============================================================================
# UPLOAD
# =============================================================================

def upload_documents() -> dict[str, str]:
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Mỗi .md được convert sang PDF trước, sau đó submit và lưu doc_id vào registry
    để pageindex_search() dùng lại.

    Returns:
        Dict {filename: doc_id}
    """
    client = _get_client()
    if client is None:
        print("⚠ Thiếu PAGEINDEX_API_KEY — bỏ qua upload")
        return {}

    registry = _load_registry()

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        if md_file.name in registry:
            print(f"  • Bỏ qua (đã upload): {md_file.name}")
            continue

        pdf_path = markdown_to_pdf(md_file)
        resp = client.submit_document(str(pdf_path))
        doc_id = resp.get("doc_id") or resp.get("id")
        if not doc_id:
            print(f"  ✗ Không lấy được doc_id cho {md_file.name}: {resp}")
            continue

        registry[md_file.name] = doc_id
        print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")

    DOC_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    DOC_REGISTRY.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return registry


# =============================================================================
# RETRIEVAL
# =============================================================================

def _poll_retrieval(client, retrieval_id: str) -> dict:
    """Poll cho tới khi retrieval hoàn tất hoặc hết thời gian chờ."""
    deadline = time.time() + POLL_TIMEOUT_SEC
    while time.time() < deadline:
        retrieval = client.get_retrieval(retrieval_id)
        status = (retrieval.get("status") or "").lower()
        if status in ("completed", "success", "succeeded", ""):
            return retrieval
        if status in ("failed", "error"):
            return {}
        time.sleep(POLL_INTERVAL_SEC)
    return {}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
        Trả list rỗng nếu chưa cấu hình API key / chưa upload — để Task 9 vẫn chạy được.
    """
    client = _get_client()
    if client is None:
        return []

    registry = _load_registry()
    if not registry:
        print("⚠ Chưa có doc_id nào — chạy upload_documents() trước")
        return []

    results: list[dict] = []

    for filename, doc_id in registry.items():
        if len(results) >= top_k:
            break
        try:
            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            if not retrieval_id:
                continue

            retrieval = _poll_retrieval(client, retrieval_id)

            # Schema: retrieved_nodes[].relevant_contents -> list[list[{...}]]
            for node in retrieval.get("retrieved_nodes", []):
                for group in node.get("relevant_contents", []):
                    for item in group:
                        content = (item.get("relevant_content") or "").strip()
                        if not content:
                            continue
                        results.append({
                            "content": content,
                            # PageIndex không trả score → tự gán giảm dần theo thứ hạng
                            "score": round(1.0 / (1 + len(results)), 4),
                            "metadata": {
                                "source": filename,
                                "section": item.get("section_title"),
                                "type": "legal" if "luat" in filename else "news",
                            },
                            "source": "pageindex",
                        })
        except Exception as e:  # noqa: BLE001 — fallback không được làm sập pipeline
            print(f"  ✗ PageIndex lỗi với {filename}: {e}")
            continue

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        # Corpus: luật VN + bài hướng dẫn Shopee Seller
        for r in pageindex_search("Quy trình trả hàng hoàn tiền trên Shopee", top_k=3):
            print(f"[{r['score']:.3f}] {r['metadata']['source']} -> {r['content'][:100]}...")
