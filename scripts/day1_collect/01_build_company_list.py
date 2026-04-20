"""
Day 1 — Step 1：建立金管會強制申報公司清單
資料來源：
  - 公司基本資料（含資本額）：TWSE Open API  openapi.twse.com.tw/v1/opendata/t187ap03_L
  - 永續報告書連結：         TWSE 公司治理中心 cgc.twse.com.tw/front/chPage（requests offset分頁）
篩選條件：資本額 ≥ 20 億 + CGC 有登錄報告書連結
輸出：company_list.json，上傳至 Cloud Storage。
"""

import json
import logging
import time
import urllib3
import sys
from pathlib import Path

import requests
from google.cloud import storage
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parents[2] / "setup"))
from config import PROJECT_ID, BUCKET_NAME, REPORT_YEAR, MIN_CAPITAL_BILLION

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CGC_URL   = "https://cgc.twse.com.tw/front/chPage"
TWSE_API  = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
HEADERS   = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
PAGE_SIZE = 30

# ── 優先級設定 ───────────────────────────────────────────────────
# Tier 1：台灣 50（0050 ETF 成分股，2024 年末近似清單）
# 每季可能小幅調整，視需要更新
TWSE_50_TICKERS: set[str] = {
    "2330", "2317", "2454", "2308", "2881", "2882", "2412",
    "2886", "2891", "2884", "3008", "2885", "2892", "2002",
    "1216", "2883", "2303", "5880", "2880", "4938", "2382",
    "2357", "2912", "3711", "2395", "2379", "2327", "1301",
    "2887", "2890", "6505", "3045", "1303", "2301", "2324",
    "6669", "1326", "2345", "2408", "2207", "3034", "2377",
    "3231", "2474", "2353", "3017", "2347", "2337", "4904",
    "2409",
}

# Tier 2：高碳排 + 高電耗科技業（TWSE 產業別代碼）
# 高碳排：水泥='01', 塑膠='03', 化學='21', 玻璃陶瓷='08', 造紙='09', 鋼鐵='10'
# 高電耗：半導體='24', 電子零組件='28', 光電='26', 通信網路='27', 其他電子='31'
# 航運（海運燃油高碳）='15'
TIER2_INDUSTRIES: set[str] = {
    "01",  # 水泥工業
    "03",  # 塑膠工業
    "21",  # 化學工業
    "08",  # 玻璃陶瓷
    "09",  # 造紙工業
    "10",  # 鋼鐵工業
    "24",  # 半導體業
    "28",  # 電子零組件業
    "26",  # 光電業
    "27",  # 通信網路業
    "31",  # 其他電子業
    "15",  # 航運業
}

TIER3_MIN_CAPITAL_BIL = 20.0   # 資本額 ≥ 20 億列 Tier 3


def assign_priority(ticker: str, industry: str, capital_bil: float) -> int:
    """
    回傳 1–4 的優先處理等級：
      1 = 台灣 50（市值最大，ESG 揭露完整）
      2 = 高碳排 / 高電耗重點產業
      3 = 資本額 ≥ 20 億（中大型企業）
      4 = 其他（中小型）
    """
    if ticker in TWSE_50_TICKERS:
        return 1
    if industry in TIER2_INDUSTRIES:
        return 2
    if capital_bil >= TIER3_MIN_CAPITAL_BIL:
        return 3
    return 4


# ── Step 1：從 TWSE Open API 取得所有上市公司基本資料（含資本額）─────────

def fetch_twse_company_info() -> dict[str, dict]:
    """回傳 {ticker: {company, industry, capital_bil}} 的字典"""
    log.info("從 TWSE Open API 取得公司基本資料...")
    r = requests.get(TWSE_API, headers=HEADERS, verify=False, timeout=20)
    r.raise_for_status()
    data = r.json()
    log.info(f"  取得 {len(data)} 筆上市公司資料")

    result = {}
    for row in data:
        ticker = row.get("公司代號", "").strip()
        if not ticker:
            continue
        try:
            capital_bil = round(float(row.get("實收資本額", "0")) / 1e8, 2)
        except (ValueError, TypeError):
            capital_bil = 0.0
        result[ticker] = {
            "company":     row.get("公司名稱", "").strip(),
            "short_name":  row.get("公司簡稱", "").strip(),
            "industry":    row.get("產業別", "").strip(),
            "capital_bil": capital_bil,
        }
    return result


# ── Step 2：從 CGC 網站取得各公司永續報告書連結（offset 分頁）──────────────

def fetch_cgc_report_links(year: int) -> dict[str, str]:
    """
    爬取 CGC 上市公司永續報告書清單（全部分頁）。
    回傳 {ticker: report_url}
    使用 GET ?year=YYYY&type=Listed&offset=N 方式分頁，不需要 playwright。
    """
    log.info(f"從 CGC 爬取 {year} 年永續報告書連結（GET offset 分頁）...")

    report_links = {}
    offset = 0

    while True:
        params = {
            "stkNo": "",
            "stkName": "",
            "year": str(year),
            "type": "Listed",
            "_action_chPage": "搜尋",
            "format": "",
            "max": str(PAGE_SIZE),
            "offset": str(offset),
        }
        r = requests.get(CGC_URL, params=params, headers=HEADERS, verify=False, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("table tr")[1:]  # 跳過 header

        if not rows:
            log.info(f"  offset={offset}：無資料，結束")
            break

        # 取得總筆數（第一次）
        if offset == 0:
            counter = soup.select_one(".counter")
            total_str = counter.get_text(strip=True) if counter else ""
            log.info(f"  {total_str}")

        for row in rows:
            cells = row.find_all("td")
            links = [a.get("href", "") for a in row.find_all("a", href=True)]
            if len(cells) >= 2 and links:
                ticker = cells[1].get_text(strip=True)
                url    = links[0].strip()
                if ticker and url:
                    report_links[ticker] = url

        log.info(f"  offset={offset}：取得 {len(rows)} 筆，累計 {len(report_links)} 家")

        if len(rows) < PAGE_SIZE:
            break  # 最後一頁
        offset += PAGE_SIZE
        time.sleep(0.5)  # 禮貌性延遲

    log.info(f"CGC 共取得 {len(report_links)} 家公司報告書連結")
    return report_links


# ── Step 3：合併、篩選、輸出 ─────────────────────────────────────────────

def build_company_list(twse_info: dict, cgc_links: dict) -> list[dict]:
    """合併 TWSE 基本資料 + CGC 報告書連結，篩選資本額 ≥ 20 億"""
    companies = []
    for ticker, url in cgc_links.items():
        info = twse_info.get(ticker, {})
        capital = info.get("capital_bil", 0.0)
        if capital < MIN_CAPITAL_BILLION:
            continue
        industry = info.get("industry", "")
        priority = assign_priority(ticker, industry, capital)
        companies.append({
            "ticker":      ticker,
            "company":     info.get("company", ""),
            "short_name":  info.get("short_name", ""),
            "industry":    industry,
            "capital_bil": capital,
            "priority":    priority,
            "report_year": REPORT_YEAR,
            "report_url":  url,
            "has_report":  bool(url),
        })
    # 依優先級排序，同級內依 ticker 排序
    companies.sort(key=lambda x: (x["priority"], x["ticker"]))
    log.info(f"篩選後（資本額 ≥ {MIN_CAPITAL_BILLION} 億 + 有報告書）：{len(companies)} 家")
    return companies


def upload_to_gcs(data: list[dict], filename: str) -> str:
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob   = bucket.blob(f"company_data/{filename}")
    blob.upload_from_string(
        json.dumps(data, ensure_ascii=False, indent=2),
        content_type="application/json"
    )
    gcs_uri = f"gs://{BUCKET_NAME}/company_data/{filename}"
    log.info(f"已上傳至 {gcs_uri}")
    return gcs_uri


def main():
    log.info(f"=== 建立 {REPORT_YEAR} 年度 ESG 申報公司清單 ===")

    twse_info = fetch_twse_company_info()
    cgc_links = fetch_cgc_report_links(REPORT_YEAR)
    companies = build_company_list(twse_info, cgc_links)

    # 優先級分佈統計
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for c in companies:
        tier_counts[c["priority"]] += 1
    log.info("優先級分佈：")
    tier_names = {1: "台灣50", 2: "高碳排/高電耗", 3: "資本額≥20億", 4: "其他"}
    for t in [1, 2, 3, 4]:
        log.info(f"  Tier {t}（{tier_names[t]}）：{tier_counts[t]} 家")

    # 產業分佈摘要（前 10）
    industries = {}
    for c in companies:
        industries[c["industry"]] = industries.get(c["industry"], 0) + 1
    log.info("產業分佈（前 10）：")
    for ind, cnt in sorted(industries.items(), key=lambda x: -x[1])[:10]:
        log.info(f"  {ind}: {cnt} 家")

    # 本地暫存（Windows/Linux 相容）
    import tempfile
    output_path = Path(tempfile.gettempdir()) / "company_list.json"
    output_path.write_text(json.dumps(companies, ensure_ascii=False, indent=2), encoding="utf-8")

    gcs_uri = upload_to_gcs(companies, f"company_list_{REPORT_YEAR}.json")

    log.info(f"\n✅ 完成！共 {len(companies)} 家公司")
    log.info(f"   GCS URI：{gcs_uri}")
    log.info(f"   下一步：執行 02_download_pdfs.py")

    return {"company_count": len(companies), "gcs_uri": gcs_uri}


if __name__ == "__main__":
    main()
