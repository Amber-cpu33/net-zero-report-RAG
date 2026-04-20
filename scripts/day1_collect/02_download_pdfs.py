"""
Day 1 — Step 2：從 SustaiHub 下載 ESG 永續報告書 PDF → 上傳 GCS

執行方式：
  # 批次（依 TIER 篩選）
  TIER=1 python scripts/day1_collect/02_download_pdfs.py

  # 手動指定 ticker
  python scripts/day1_collect/02_download_pdfs.py 1216 2330 2412

SustaiHub 搜尋：https://www.sustaihub.com/reports/?keyword={公司簡稱}
PDF 直連格式：FileStream 或 FileDownLoad
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import requests
import urllib3
from bs4 import BeautifulSoup
from google.cloud import storage

sys.path.insert(0, str(Path(__file__).parents[2] / "setup"))
from config import PROJECT_ID, BUCKET_NAME, REPORT_YEAR

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TIER           = int(os.getenv("TIER", "0"))
SUSTAIHUB_BASE = "https://www.sustaihub.com/reports/"
REQUEST_DELAY  = 1.5
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
}


def normalize_name(name: str) -> str:
    for suffix in ["股份有限公司", "有限公司", "集團", "-KY", "-創"]:
        name = name.replace(suffix, "")
    return name.strip()


def search_sustaihub(short_name: str, year: int) -> str | None:
    keyword = normalize_name(short_name)
    url = f"{SUSTAIHUB_BASE}?keyword={requests.utils.quote(keyword)}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20, verify=False)
        resp.raise_for_status()
    except Exception as e:
        log.warning(f"  SustaiHub 請求失敗：{e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for card in soup.select("li.relative.rounded-3xl"):
        ga_link = card.select_one("a.ga-track[data-report-info]")
        if not ga_link:
            continue
        report_info = ga_link.get("data-report-info", "")
        if str(year) not in report_info:
            continue
        if normalize_name(short_name) not in report_info:
            continue
        dl = card.find("a", href=lambda h: h and ("FileStream" in h or "FileDownLoad" in h))
        if dl:
            log.info(f"  ✓ 找到：{report_info}")
            return dl.get("href")
    return None


def download_pdf(url: str) -> bytes | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, verify=False)
        resp.raise_for_status()
        if len(resp.content) < 10000:
            log.warning(f"  檔案過小 ({len(resp.content)} bytes)，跳過")
            return None
        return resp.content
    except Exception as e:
        log.warning(f"  下載失敗：{e}")
        return None


def main():
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    companies = json.loads(
        bucket.blob(f"company_data/company_list_{REPORT_YEAR}.json").download_as_text()
    )

    manual_tickers = set(sys.argv[1:])
    if manual_tickers:
        targets = [c for c in companies if str(c["ticker"]) in manual_tickers]
        log.info(f"手動模式：{len(targets)} 家")
    elif TIER:
        targets = [c for c in companies if c.get("priority") == TIER]
        log.info(f"TIER={TIER}：{len(targets)} 家")
    else:
        targets = companies
        log.info(f"全部：{len(targets)} 家")

    success, failed, skipped = [], [], []

    for i, company in enumerate(targets, 1):
        ticker = str(company["ticker"])
        short_name = company.get("short_name") or company.get("company", "")
        log.info(f"\n[{i}/{len(targets)}] {ticker} {short_name}")

        blob = bucket.blob(f"raw_pdfs/{ticker}_{REPORT_YEAR}.pdf")
        if blob.exists():
            log.info(f"  已存在，跳過")
            skipped.append(ticker)
            continue

        pdf_url = search_sustaihub(short_name, REPORT_YEAR)
        if not pdf_url:
            log.warning(f"  ✗ SustaiHub 找不到")
            failed.append({"ticker": ticker, "name": short_name, "reason": "not_found"})
            time.sleep(REQUEST_DELAY)
            continue

        pdf_bytes = download_pdf(pdf_url)
        if not pdf_bytes:
            failed.append({"ticker": ticker, "name": short_name, "reason": "download_failed"})
            time.sleep(REQUEST_DELAY)
            continue

        blob.upload_from_string(pdf_bytes, content_type="application/pdf")
        size_kb = len(pdf_bytes) // 1024
        log.info(f"  ✓ {size_kb} KB → raw_pdfs/{ticker}_{REPORT_YEAR}.pdf")
        success.append({"ticker": ticker, "name": short_name, "size_kb": size_kb})
        time.sleep(REQUEST_DELAY)

    log.info(f"\n=== 完成 ===")
    log.info(f"  ✓ 成功：{len(success)} 家")
    log.info(f"  - 跳過：{len(skipped)} 家（已存在）")
    log.info(f"  ✗ 失敗：{len(failed)} 家")

    report = {"success": success, "failed": failed, "skipped": skipped}
    bucket.blob(f"logs/download_report_{REPORT_YEAR}.json").upload_from_string(
        json.dumps(report, ensure_ascii=False, indent=2),
        content_type="application/json"
    )


if __name__ == "__main__":
    main()
