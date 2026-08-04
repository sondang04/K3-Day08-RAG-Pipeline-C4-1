"""
Task 1 — Thu thập văn bản pháp luật nền cho chủ đề Thương mại điện tử.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ nguồn công khai.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Corpus hiện tại của nhóm (đã thu thập, xem data/landing/legal/):
    - luat_doanh_nghiep_2020.pdf      — Luật Doanh nghiệp, số 59/2020/QH14
    - luat_tmdt.pdf                   — Luật Thương mại điện tử, số 122/2025/QH15
    - luat_giao_duc_dai_hoc_2018.pdf  — Luật Giáo dục Đại học, số 34/2018/QH14

Ba văn bản trên được tải thủ công (in ra PDF từ trang tra cứu văn bản công khai) nên
DOCUMENTS bên dưới để trống. Nếu có direct link tới file PDF, thêm vào DOCUMENTS rồi
chạy `python -m src.task1_collect_legal_docs` để tải tự động.

Lưu ý: một số trang chặn bot crawler mặc định (HTTP 403) — đó là cấu hình WAF/Cloudflare
phía server, không phải lỗi của bạn. Đổi sang nguồn khác thay vì cố vượt qua, và chỉ
dùng nguồn công khai/được phép chia sẻ.
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# (url, filename) — điền khi có direct link tới PDF/DOCX
DOCUMENTS: list[tuple[str, str]] = []

REQUEST_TIMEOUT = 30
# Một số server từ chối request không có User-Agent trình duyệt
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Lab/1.0)"}


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str) -> Path | None:
    """
    Tải 1 file từ direct link về DATA_DIR.

    Returns:
        Đường dẫn file đã lưu, hoặc None nếu tải thất bại.
    """
    filepath = DATA_DIR / filename
    if filepath.exists():
        print(f"  • Bỏ qua (đã có): {filename}")
        return filepath

    try:
        response = requests.get(
            url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ✗ Lỗi tải {filename}: {e}")
        return None

    filepath.write_bytes(response.content)
    print(f"  ✓ Đã tải: {filepath.name} ({len(response.content) // 1024} KB)")
    return filepath


def collect_all() -> list[Path]:
    """Tải toàn bộ tài liệu trong DOCUMENTS."""
    setup_directory()

    if not DOCUMENTS:
        existing = sorted(p for p in DATA_DIR.glob("*") if p.suffix.lower() in (".pdf", ".docx"))
        print("DOCUMENTS trống — dùng các file đã tải thủ công:")
        for p in existing:
            print(f"  • {p.name}")
        return existing

    saved = []
    for url, filename in DOCUMENTS:
        path = download_file(url, filename)
        if path:
            saved.append(path)
    return saved


if __name__ == "__main__":
    files = collect_all()
    print(f"\n✓ Tổng cộng {len(files)} văn bản trong {DATA_DIR}")
