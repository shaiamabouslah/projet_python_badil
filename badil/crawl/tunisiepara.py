import json
import sys
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from badil.crawl.http import get
from badil.paths import DATA_RAW

SHOP_URL = "https://www.tunisiepara.com/shop/"
SHOP_PAGE_URL = "https://www.tunisiepara.com/shop/page/{page}/"
OUTPUT_FILE = DATA_RAW / "parapharmacy" / "tunisiepara" / "observations.jsonl"


def parse_price(price_tag) -> float | None:
    if price_tag is None:
        return None
    amounts = price_tag.select("span.woocommerce-Price-amount bdi")
    if not amounts:
        return None
    text = amounts[-1].get_text()
    digits = text.split("D.T")[0].strip()
    return float(digits.replace(",", ""))


def parse_product(card) -> dict:
    title_link = card.select_one("p.product-title a")
    return {
        "name": title_link.get_text(strip=True) if title_link else None,
        "url": title_link.get("href") if title_link else None,
        "price": parse_price(card.select_one("span.price")),
        "in_stock": card.select_one("div.out-of-stock-label") is None,
    }


def fetch_page(page: int) -> list[dict]:
    url = SHOP_URL if page == 1 else SHOP_PAGE_URL.format(page=page)
    response = get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    cards = soup.select("div.product-small")
    return [card for card in cards if card.select_one("div.product-small") is None]


def already_done() -> set[str]:
    if not OUTPUT_FILE.exists():
        return set()
    with OUTPUT_FILE.open(encoding="utf-8") as f:
        return {json.loads(line)["url"] for line in f if line.strip()}


def save_products(products: list[dict]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    crawled_at = datetime.now(timezone.utc).isoformat()
    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        for product in products:
            record = {**product, "crawled_at": crawled_at}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def crawl(max_pages: int) -> int:
    done = already_done()
    saved = 0
    for page in range(1, max_pages + 1):
        cards = fetch_page(page)
        if not cards:
            print(f"page {page}: no products, stopping", flush=True)
            break
        products = [parse_product(card) for card in cards]
        new_products = [p for p in products if p["url"] not in done]
        if new_products:
            save_products(new_products)
            saved += len(new_products)
            done.update(p["url"] for p in new_products)
        print(f"page {page}: {len(products)} products, {len(new_products)} new, {saved} saved total", flush=True)
    return saved


if __name__ == "__main__":
    pages = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    total_saved = crawl(max_pages=pages)
    print(f"done: {total_saved} products saved")
