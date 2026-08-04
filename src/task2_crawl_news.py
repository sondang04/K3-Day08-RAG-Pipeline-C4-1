"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL category/trang chứa bài viết cần crawl
CATEGORY_URLS = [
    "https://banhang.shopee.vn/edu/category?sub_cat_id=2197",
]


def extract_article_urls(markdown: str) -> list[str]:
    """Trích xuất tất cả URL bài viết từ markdown của trang category."""
    pattern = r"https://banhang\.shopee\.vn/edu/article/\d+"
    urls = re.findall(pattern, markdown)
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    return unique_urls


def clean_markdown(raw_md: str) -> str:
    """
    Làm sạch markdown: loại bỏ navigation, sidebar, footer,
    footer links, copyright, chat widget, app download section.
    """
    lines = raw_md.split("\n")

    skip_patterns = [
        r"^\s*\[.*\]\(https://banhang\.shopee\.vn/edu/category",
        r"^\s*\[.*\]\(https://banhang\.shopee\.vn/edu/article",
        r"^\s*\[.*\]\(https://shopee\.vn",
        r"^\s*\[.*\]\(https://www\.facebook",
        r"^\s*\[.*\]\(https://www\.youtube",
        r"^\s*\[.*\]\(https://deo\.shopeemobile",
        r"^\s*\[.*\]\(https://down-vn\.img",
        r"^\s*\[.*\]\(https://open\.shopee",
        r"^\s*\[.*\]\(https://help\.shopee",
        r"^\s*VỀ SHOPEE",
        r"^\s*THEO DÕI CHÚNG TÔI",
        r"^\s*PHẢN HỒI",
        r"^\s*CHĂM SÓC KHÁCH HÀNG",
        r"^\s*TẢI ỨNG DỤNG",
        r"^\s*Trò chuyện với Shopee",
        r"^\s*Copyright",
        r"^\s*!\[\]\(https://banhang\.shopee\.vn/edu",
        r"^\s*!\[\]\(https://deo\.shopeemobile",
        r"^\s*!\[\]\(https://down-vn\.img",
        r"^\s*!\[\]\(https://www\.facebook",
        r"^\s*!\[\]\(https://www\.youtube",
        r"^\s*!\[\]\(https://ads\.shopee",
        r"^Bài viết này có hữu ích",
        r"^Tham khảo thêm",
        r"^\s*\[.*\]\(#",  # anchor links
        r"^\s*Email$",
        r"^\s*_\(Email\)$",
        r"^\s*_\(Tải Ứng Dụng\)$",
    ]

    cleaned_lines = []
    skip_until_next_h1 = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines that follow footer sections
        if not stripped:
            continue

        # Skip lines matching skip patterns
        skip = False
        for pattern in skip_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                skip = True
                break
        if skip:
            continue

        # Skip lines that are just navigation badges (e.g. "Thông Báo Mới" sidebar)
        if re.match(r"^[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-ỹ\s]{1,30}$", stripped) and len(stripped) < 50:
            # Could be sidebar header
            if not any(c in stripped for c in ".:,;0123456789"):
                continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()

    # Remove consecutive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


async def crawl_page(url: str) -> dict:
    """Crawl một trang và trả về result với content_strategy tốt."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

    browser_config = BrowserConfig(
        headless=True,
        extra_args=["--disable-blink-features=AutomationControlled"],
    )

    # Target main article content using CSS selectors
    run_config = CrawlerRunConfig(
        wait_for="js:() => document.querySelectorAll('div, section, article').length > 5",
        wait_for_timeout=10000,
        delay_before_return_html=2.0,
        remove_overlay_elements=True,
        page_timeout=30000,
        css_selector="article, .article-content, .article-body, main, [class*='article'], [class*='content'], [class*='post'], [class*='detail']",
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        return result


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.
    """
    result = await crawl_page(url)

    raw_md = result.markdown if result.markdown else ""
    cleaned_md = clean_markdown(raw_md)

    return {
        "url": url,
        "title": result.metadata.get("title", "Unknown"),
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": cleaned_md,
    }


async def crawl_all():
    """Crawl toàn bộ bài viết từ các category URLs."""
    setup_directory()
    all_article_urls = []

    # Bước 1: Crawl từng category page để lấy danh sách article URLs
    for i, cat_url in enumerate(CATEGORY_URLS, 1):
        print(f"[{i}/{len(CATEGORY_URLS)}] Crawling category: {cat_url}")
        result = await crawl_page(cat_url)
        article_urls = extract_article_urls(result.markdown or "")
        print(f"  → Found {len(article_urls)} article URLs")
        all_article_urls.extend(article_urls)

    if not all_article_urls:
        print("⚠ Không tìm thấy bài viết nào để crawl!")
        return

    print(f"\n📰 Total: {len(all_article_urls)} articles to crawl\n")

    # Bước 2: Crawl từng bài viết và lưu
    for i, article_url in enumerate(all_article_urls, 1):
        print(f"[{i}/{len(all_article_urls)}] Crawling: {article_url}")
        try:
            article = await crawl_article(article_url)
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"  ✓ Saved: {filepath}")
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    if not CATEGORY_URLS:
        print("⚠ Hãy điền CATEGORY_URLS trước khi chạy!")
    else:
        asyncio.run(crawl_all())
