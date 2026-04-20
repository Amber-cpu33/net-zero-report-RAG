"""
Day 3/5 — Step 7：Gemini 2.5 Flash 公司摘要生成
流程：
  1. 從 GCS embeddings/{ticker}.jsonl 讀取帶向量的 chunks
  2. 用 Gemini 2.5 Flash 生成每家公司的「ESG 顧問快速摘要」
  3. 摘要結構化為 JSON，同樣向量化後加入索引
  4. 輸出：summaries/{ticker}_{REPORT_YEAR}_summary.json 至 GCS

摘要設計說明：
  摘要作為「公司級別的 Overview Chunk」，當使用者問「台積電的淨零目標是什麼」時，
  API 同時從兩個層次檢索：
    a. 細粒度 chunk（具體數值）
    b. 摘要 chunk（整體策略與目標）
  兩層合併給 LLM 合成答案，提升回答的深度與準確性。
"""

import json
import logging
import os
import re
import time
import traceback

try:
    from json_repair import repair_json
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False
from datetime import datetime
from pathlib import Path
from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.cloud import storage

import sys
sys.path.append(str(Path(__file__).parents[2] / "setup"))
from config import (
    PROJECT_ID, REGION_GEN, BUCKET_NAME,
    SUMMARY_MODEL, REPORT_YEAR
)

TIER           = int(os.getenv("TIER", "0"))
TEST_MODE      = os.getenv("TEST_MODE", "0") == "1"
TEST_TICKERS   = set(os.getenv("TEST_TICKERS", "").split(",")) - {""}
REQUIRE_VISION = os.getenv("REQUIRE_VISION", "0") == "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# 摘要生成設定
TOP_CHUNKS_FOR_SUMMARY = 30    # 每家公司取前 N 個 chunks 作為摘要輸入
MAX_INPUT_CHARS = 20000        # 輸入上限（避免超出 context window 與費用控制）
REQUEST_DELAY_S = 1.0          # Flash 請求間隔（60 req/min 限制）


# ── Prompt ────────────────────────────────────────────────────

PROMPT_PASS1 = """
你是一位 ESG 數據分析師。請從以下 {company_name}（{ticker}）永續報告書摘錄中，
萃取【溫室氣體排放】相關數值，只回傳 JSON，不加任何說明。

注意規則：
1. 數值請填入純數字，不可包含千分位逗號（如 1,234.5 → 1234.5）。
2. 排放數值單位換算為「公噸 CO2e (tCO2e)」；若標示為「公噸 CO2」視同 tCO2e。
3. 數值通常出現在報告書末尾【附錄】、【ESG 績效數據表】、【環境績效摘要】、【GHG 排放統計表】。
4. Scope 1 別名：直接排放、自有排放源。
5. Scope 2 別名：電力間接排放、購買電力排放、能源間接排放。
6. Scope 3 別名：其他間接排放量、價值鏈排放、供應鏈排放；若真的沒有揭露才填 null。
7. 若同時出現市場基準法與位置基準法，優先取市場基準法（market-based）。
8. 若報告書只揭露 Scope 1+2 合計（未分開），填入 scope1_2_combined，scope1_tco2e 與 scope2_tco2e 填 null。

--- 報告書內容 ---
{content}
--- 結束 ---

重要：若文本中找不到明確數值，必須填 null，嚴禁推算、估算或虛構數值。

輸出 JSON（數值填數字，不確定填 null，不可填字串）：
{{
  "net_zero_target_year": null,
  "carbon_neutral_target_year": null,
  "science_based_targets": false,
  "scope1_tco2e": null,
  "scope2_tco2e": null,
  "scope1_2_combined": null,
  "scope3_tco2e": null,
  "units": {{
    "scope1_tco2e": null,
    "scope2_tco2e": null,
    "scope1_2_combined": null,
    "scope3_tco2e": null
  }}
}}
"""

PROMPT_PASS2 = """
你是一位 ESG 數據分析師。請從以下 {company_name}（{ticker}）永續報告書摘錄中，
萃取【能源、用水、廢棄物】相關數值，只回傳 JSON，不加任何說明。

嚴格規則：
1. 目標是找出代表「全公司/集團合併」的總數值。
2. 加總數值通常位於報告書末尾的【附錄】、【ESG 績效數據表】或【環境指標統整表】。
3. 若文本中同時出現多個數值，優先採用當年度（{report_year}年）全公司合併總計，而非廠區分項或歷史累計。
4. 數值請填入純數字，不可包含千分位逗號（如 1,234.5 → 1234.5）。
5. 單位換算規則（填入換算後的數字）：
   - 能源：1 kWh = 0.0036 GJ；1 MWh = 3.6 GJ；1 TJ = 1000 GJ；若報告以 kWh/MWh 揭露請換算為 GJ。
   - 用水：1 千公秉(KL) = 1 m³；1 千立方公尺 = 1000 m³；1 百萬公升 = 1000 m³。
   - 廢棄物：單位已為公噸(ton)則直接填入。
6. renewable_energy_pct 為百分比數值（如 15.3，不是 0.153）。
7. waste_recycling_rate 別名：資源化率、再利用率、回收再利用比例；為百分比數值。
8. 表格單位欄與數值欄可能被斷行分離（例如「項目 單位 2024」一列、「回收率 % 95.5」下一列）。若看到欄位名稱（如「回收率」）後接單位符號（如「%」）再接純數字，請直接讀取該數字。

--- 報告書內容 ---
{content}
--- 結束 ---

重要：若文本中找不到明確數值，必須填 null，嚴禁推算、估算或虛構數值。

輸出 JSON（數值填數字，不確定填 null，不可填字串）：
{{
  "total_energy_gj": null,
  "renewable_energy_pct": null,
  "water_withdrawal_m3": null,
  "waste_total_ton": null,
  "waste_recycling_rate": null,
  "units": {{
    "total_energy_gj": null,
    "water_withdrawal_m3": null,
    "waste_total_ton": null
  }}
}}
"""

PROMPT_PASS3 = """
你是一位資深 ESG 永續發展顧問。請根據以下 {company_name}（{ticker}，{industry}）
永續報告書摘錄，生成定性描述欄位，只回傳 JSON，不加任何說明。

--- 報告書內容 ---
{content}
--- 結束 ---

輸出 JSON（使用繁體中文）：
{{
  "key_initiatives": [],
  "certifications": [],
  "reporting_standards": [],
  "esg_strengths": [],
  "esg_risks": [],
  "analyst_summary": "",
  "data_completeness": 0.0,
  "confidence": 0.0
}}

注意：
- esg_risks 一律輸出字串陣列。若報告對風險有分類（如轉型風險/實體風險），請將分類資訊合併入字串，例如「氣候轉型風險：碳費法規可能增加營運成本」。
- data_completeness：0.0～1.0 的小數，代表本報告書環境數據的揭露完整程度（0=完全沒數據，1=數據非常完整），絕對不可超過 1.0。
- confidence：0.0～1.0 的小數，代表你對萃取結果的把握程度（0=完全不確定，1=非常確定），絕對不可超過 1.0。
"""


# ── 輔助函式 ──────────────────────────────────────────────────

def load_company_chunks(gcs_client: storage.Client, ticker: str) -> list[dict]:
    """從 GCS 載入 embeddings/{ticker}.jsonl，回傳 chunk list"""
    bucket = gcs_client.bucket(BUCKET_NAME)

    # 先嘗試帶向量的版本，再 fallback 到原始 chunks
    for prefix in ["embeddings", "chunks"]:
        blob_name = f"{prefix}/{ticker}_{REPORT_YEAR}.jsonl"
        blob = bucket.blob(blob_name)
        if blob.exists():
            content = blob.download_as_text(encoding="utf-8")
            chunks = [
                json.loads(line) for line in content.strip().split("\n")
                if line.strip()
            ]
            log.debug(f"  讀取 {blob_name}：{len(chunks)} chunks")
            return chunks

    log.warning(f"[{ticker}] 找不到 chunk 檔案")
    return []


def load_vision_chunks(gcs_client: storage.Client, ticker: str) -> list[dict]:
    """載入 Vision 萃取的圖表 chunks（若有）"""
    bucket = gcs_client.bucket(BUCKET_NAME)
    blob_name = f"vision_output/confirmed/{ticker}_{REPORT_YEAR}.jsonl"
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return []

    content = blob.download_as_text(encoding="utf-8")
    return [
        json.loads(line) for line in content.strip().split("\n")
        if line.strip()
    ]


# Pass 1：溫室氣體（vision +10，絕對值在 vision chunks）
PASS1_FIELD_KEYWORDS = {
    "net_zero_target_year":       ["淨零", "2050", "net zero"],
    "carbon_neutral_target_year": ["碳中和", "carbon neutral"],
    "scope1_tco2e":               ["Scope 1", "範疇一", "直接排放", "CO2e", "公噸"],
    "scope2_tco2e":               ["Scope 2", "範疇二", "間接排放", "CO2e", "公噸"],
    "scope1_2_combined":          ["Scope 1+2", "範疇一及二", "範疇一、二", "範疇一二合計", "直接及間接排放"],
    "scope3_tco2e":               ["Scope 3", "範疇三", "CO2e", "公噸"],
}

# Pass 2：能源與資源（vision 0，絕對值在 text chunks）
PASS2_FIELD_KEYWORDS = {
    "total_energy_gj":       ["GJ", "能源消耗", "總用電", "能源使用量"],
    "renewable_energy_pct":  ["再生能源", "renewable", "太陽能比例"],
    "water_withdrawal_m3":   ["取水量", "m³", "用水量", "百萬公升"],
    "waste_total_ton":       ["廢棄物總量", "廢棄物產生量", "一般廢棄物", "事業廢棄物"],
    "waste_recycling_rate":  ["回收率", "資源化率"],
}


_APPENDIX_TITLE_PATTERNS = [
    r"績效數據表",
    r"環境績效(表|指標|數據)",
    r"ESG\s*績效",
    r"GHG\s*排放.*[表統]",
    r"永續績效摘要",
    r"環境指標(統|彙|摘要)",
    r"量化(資訊|指標)",
]

def _numeric_density(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isdigit()) / len(text)

def _appendix_bonus(text: str) -> float:
    # 只看前 200 字判斷標題（避開每頁頁首的 breadcrumb）
    head = text[:200]
    if any(re.search(p, head) for p in _APPENDIX_TITLE_PATTERNS):
        return 5.0
    d = _numeric_density(text)
    if d > 0.15:
        return 3.0
    if d > 0.10:
        return 1.5
    return 0.0


# 欄位 → 單位 regex；用於交叉搜尋（關鍵字 + 年份 + 單位）加權
FIELD_UNIT_PATTERNS = {
    "scope1_tco2e":          r"(tCO2e|公噸\s*CO|噸\s*CO|二氧化碳當量)",
    "scope2_tco2e":          r"(tCO2e|公噸\s*CO|噸\s*CO|二氧化碳當量)",
    "scope1_2_combined":     r"(tCO2e|公噸\s*CO|噸\s*CO|二氧化碳當量)",
    "scope3_tco2e":          r"(tCO2e|公噸\s*CO|噸\s*CO|二氧化碳當量)",
    "total_energy_gj":       r"(GJ|MJ|TJ|kWh|MWh|度電)",
    "renewable_energy_pct":  r"(再生能源|綠電)\s*(佔比|比例|使用率)",
    "water_withdrawal_m3":   r"(m³|m3|公秉|千公秉|KL|百萬公升|千立方公尺)",
    "waste_total_ton":       r"廢棄物.{0,15}?[0-9,]+(\.\d+)?\s*(公噸|ton|噸)",
    "waste_recycling_rate":  r"(回收|資源化|再利用)\s*(率|比例|佔比)",
}

_YEAR_PAT = re.compile(rf"\b{REPORT_YEAR}\b")

def _cross_match_bonus(text: str, keywords: list[str], unit_pat: str) -> float:
    """關鍵字 + 報告年度 + 單位 三者都命中 → 高度確信是數據表列，加 8 分。"""
    if not unit_pat:
        return 0.0
    if not any(kw in text for kw in keywords):
        return 0.0
    if not _YEAR_PAT.search(text):
        return 0.0
    if not re.search(unit_pat, text):
        return 0.0
    return 8.0


def select_chunks_for_pass(chunks: list[dict],
                            field_keywords: dict,
                            field_vision_bonus: int,
                            max_chunks: int = TOP_CHUNKS_FOR_SUMMARY,
                            max_chars: int = MAX_INPUT_CHARS) -> list[dict]:
    """依欄位關鍵字與 vision bonus 選取 chunks。"""
    GENERAL_KEYWORDS = [
        "淨零", "碳中和", "排放", "Scope", "GHG", "再生能源",
        "節能", "用水", "廢棄物", "目標", "2030", "2050",
        "溫室氣體", "減碳", "碳排"
    ]

    def general_score(chunk: dict) -> float:
        text = chunk.get("text", "")
        s = sum(1 for kw in GENERAL_KEYWORDS if kw in text)
        if chunk.get("extraction_method") == "gemini_vision":
            s += 3
        s += chunk.get("confidence_score", 0.5)
        s += _appendix_bonus(text)
        return s

    def field_score(chunk: dict, field_name: str, keywords: list[str]) -> float:
        text = chunk.get("text", "")
        score = sum(1 for kw in keywords if kw in text)
        score += _appendix_bonus(text)
        score += _cross_match_bonus(text, keywords, FIELD_UNIT_PATTERNS.get(field_name, ""))
        if chunk.get("extraction_method") == "gemini_vision":
            score += field_vision_bonus
        return score

    selected = []
    selected_ids = set()
    total_chars = 0

    def try_add(chunk: dict) -> bool:
        nonlocal total_chars
        cid = id(chunk)
        if cid in selected_ids:
            return False
        text_len = len(chunk.get("text", ""))
        if total_chars + text_len > max_chars:
            return False
        selected.append(chunk)
        selected_ids.add(cid)
        total_chars += text_len
        return True

    for field_name, keywords in field_keywords.items():
        best = max(
            (c for c in chunks if any(kw in c.get("text", "") for kw in keywords)),
            key=lambda c: field_score(c, field_name, keywords),
            default=None,
        )
        if best:
            try_add(best)
        if len(selected) >= max_chunks:
            break

    for chunk in sorted(chunks, key=general_score, reverse=True):
        if len(selected) >= max_chunks:
            break
        try_add(chunk)

    return selected


def _call_model(model: GenerativeModel, prompt: str,
                max_output_tokens: int = 2048) -> Optional[dict]:
    """呼叫模型並解析 JSON，失敗重試一次。"""
    for attempt in range(2):
        if attempt > 0:
            log.warning("  JSON 解析失敗，重試一次...")
        time.sleep(REQUEST_DELAY_S)
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.3,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
            )
        )
        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        raw = match.group() if match else raw
        try:
            if _HAS_JSON_REPAIR:
                return json.loads(repair_json(raw))
            text = re.sub(r'(?<!\\)[\x00-\x08\x0b\x0c\x0e-\x1f\x0a\x0d]', ' ', raw)
            text = re.sub(r':\s*None\b',  ': null',  text)
            text = re.sub(r':\s*True\b',  ': true',  text)
            text = re.sub(r':\s*False\b', ': false', text)
            text = re.sub(r',\s*([}\]])', r'\1', text)
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt == 1:
                raise
    return None


def _chunks_to_content(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        prefix = "[圖表數據]" if chunk.get("extraction_method") == "gemini_vision" else "[段落]"
        text = re.sub(r'(\d)\s+,\s+(\d)', r'\1,\2', chunk.get('text', ''))
        parts.append(f"{prefix} {text}")
    return "\n\n".join(parts)


def generate_summary(model: GenerativeModel,
                      company: dict,
                      all_chunks: list[dict]) -> Optional[dict]:
    """三段式萃取：Pass1 溫室氣體 / Pass2 能源資源 / Pass3 文字敘述"""
    if not all_chunks:
        log.warning(f"[{company['ticker']}] 無 chunks，跳過摘要生成")
        return None

    ticker      = company["ticker"]
    name        = company.get("company", ticker)
    industry    = company.get("industry", "")

    chunks1 = select_chunks_for_pass(all_chunks, PASS1_FIELD_KEYWORDS, field_vision_bonus=10)
    chunks2 = select_chunks_for_pass(all_chunks, PASS2_FIELD_KEYWORDS, field_vision_bonus=0)
    chunks3 = select_chunks_for_pass(all_chunks, {}, field_vision_bonus=3)

    try:
        r1 = _call_model(model, PROMPT_PASS1.format(
            company_name=name, ticker=ticker, content=_chunks_to_content(chunks1)),
            max_output_tokens=4096)
        log.info(f"  Pass1 scope1={r1.get('scope1_tco2e')} scope2={r1.get('scope2_tco2e')} scope3={r1.get('scope3_tco2e')}")

        r2 = _call_model(model, PROMPT_PASS2.format(
            company_name=name, ticker=ticker, report_year=REPORT_YEAR, content=_chunks_to_content(chunks2)),
            max_output_tokens=4096)
        log.info(f"  Pass2 energy={r2.get('total_energy_gj')} water={r2.get('water_withdrawal_m3')} waste={r2.get('waste_total_ton')}")

        r3 = _call_model(model, PROMPT_PASS3.format(
            company_name=name, ticker=ticker, industry=industry, content=_chunks_to_content(chunks3)),
            max_output_tokens=8192)

        units = {**r1.pop("units", {}), **r2.pop("units", {})}
        summary = {
            "company": name, "ticker": ticker,
            "industry": industry, "report_year": REPORT_YEAR,
            **r1, **r2, **r3,
            "units": units,
            "generated_at": datetime.utcnow().isoformat(),
            "source_chunks": len(all_chunks),
        }

        if TEST_MODE:
            test_dir = Path(__file__).parents[2] / "data" / "raw_response_test"
            test_dir.mkdir(parents=True, exist_ok=True)
            (test_dir / f"raw_response_{ticker}.txt").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info(f"  raw response 已寫入 data/raw_response_test/raw_response_{ticker}.txt")

        return summary

    except Exception as e:
        log.error(f"[{ticker}] 摘要生成失敗：{e}")
        log.error(traceback.format_exc())
        return None


def summary_to_chunk(summary: dict) -> dict:
    """將摘要轉為 Overview Chunk 格式，方便後續向量化"""
    ticker = summary.get("ticker", "")
    company = summary.get("company", ticker)

    # 組裝摘要文字（供 Embedding 使用）
    text_parts = [
        f"{company}（{ticker}）ESG 永續發展綜合摘要：",
        summary.get("analyst_summary", ""),
    ]

    if summary.get("net_zero_target_year"):
        text_parts.append(f"淨零目標年份：{summary['net_zero_target_year']} 年")
    if summary.get("scope1_tco2e") is not None:
        text_parts.append(f"Scope 1 排放：{summary['scope1_tco2e']} tCO2e")
    if summary.get("scope2_tco2e") is not None:
        text_parts.append(f"Scope 2 排放：{summary['scope2_tco2e']} tCO2e")
    if summary.get("scope1_tco2e") is None and summary.get("scope1_2_combined") is not None:
        text_parts.append(f"Scope 1+2 合計排放：{summary['scope1_2_combined']} tCO2e")
    if summary.get("renewable_energy_pct") is not None:
        text_parts.append(f"再生能源占比：{summary['renewable_energy_pct']}%")

    initiatives = [i if isinstance(i, str) else str(i) for i in summary.get("key_initiatives", [])]
    if initiatives:
        text_parts.append("主要減碳措施：" + "；".join(initiatives[:3]))

    text = " ".join(filter(None, text_parts))

    return {
        "chunk_id":               f"{ticker}_{summary.get('report_year', REPORT_YEAR)}_overview",
        "company":                company,
        "ticker":                 ticker,
        "industry":               summary.get("industry", ""),
        "report_year":            summary.get("report_year", REPORT_YEAR),
        "data_year":              summary.get("report_year", REPORT_YEAR),
        "reporting_standard":     summary.get("reporting_standards", []),
        "scope_boundary":         "",
        "unit":                   "mixed",
        "embedding_model_version": "text-embedding-004",
        "output_dimensionality":   768,
        "extraction_method":      "gemini_summary",
        "chunk_index":            0,
        "total_chunks":           1,
        "text":                   text,
        "text_preview":           text[:150],
        "confidence_score":       summary.get("confidence", 0.8),
        "is_overview":            True,   # 標記為 Overview Chunk，RAG 時優先返回
        "embedding":              [],
        "summary_metadata":       summary,  # 完整摘要附加在 metadata
    }


def save_summary(gcs_client: storage.Client,
                  ticker: str, summary: dict, overview_chunk: dict) -> None:
    """儲存摘要 JSON 與 Overview Chunk 至 GCS"""
    bucket = gcs_client.bucket(BUCKET_NAME)

    # 儲存完整摘要 JSON
    bucket.blob(
        f"summaries/{ticker}_{REPORT_YEAR}_summary.json"
    ).upload_from_string(
        json.dumps(summary, ensure_ascii=False, indent=2),
        content_type="application/json"
    )

    # 儲存 Overview Chunk（待 embedding 腳本向量化）
    bucket.blob(
        f"summaries/{ticker}_{REPORT_YEAR}_overview.jsonl"
    ).upload_from_string(
        json.dumps(overview_chunk, ensure_ascii=False),
        content_type="application/x-ndjson"
    )


# ── 主流程 ───────────────────────────────────────────────────

def main():
    log.info("=== Day 3/5：Gemini Flash 公司摘要生成 ===")

    vertexai.init(project=PROJECT_ID, location=REGION_GEN)
    gcs_client = storage.Client(project=PROJECT_ID)

    model = GenerativeModel(SUMMARY_MODEL)
    log.info(f"使用模型：{SUMMARY_MODEL}")

    # 讀取公司清單
    bucket = gcs_client.bucket(BUCKET_NAME)
    companies = json.loads(
        bucket.blob(f"company_data/company_list_{REPORT_YEAR}.json").download_as_text()
    )
    if TIER > 0:
        companies = [c for c in companies if c.get("priority", 4) == TIER]
        log.info(f"TIER={TIER}，篩選後：{len(companies)} 家")
    if TEST_TICKERS:
        companies = [c for c in companies if str(c["ticker"]) in TEST_TICKERS]
        log.info(f"TEST_TICKERS={TEST_TICKERS}，篩選後：{len(companies)} 家")
    elif TEST_MODE:
        companies = companies[:1]
        log.info(f"TEST_MODE=1，只跑第一家：{companies[0].get('ticker')}")

    if REQUIRE_VISION:
        chunks_t = {b.name.split("/")[-1].replace(f"_{REPORT_YEAR}.jsonl", "")
                    for b in gcs_client.list_blobs(BUCKET_NAME, prefix="chunks/")
                    if b.name.endswith(".jsonl")}
        vision_t = {b.name.split("/")[-1].replace(f"_{REPORT_YEAR}.jsonl", "")
                    for b in gcs_client.list_blobs(BUCKET_NAME, prefix="vision_output/confirmed/")
                    if b.name.endswith(".jsonl")}
        ready = chunks_t & vision_t
        before = len(companies)
        companies = [c for c in companies if str(c["ticker"]) in ready]
        log.info(f"REQUIRE_VISION=1，備齊 chunks+vision：{len(companies)}/{before} 家")

    # Checkpoint：跳過已完成的公司（TEST_MODE / TEST_TICKERS 不跳過）
    done = set() if (TEST_MODE or TEST_TICKERS) else {
        blob.name.split("/")[-1].replace(f"_{REPORT_YEAR}_summary.json", "")
        for blob in gcs_client.list_blobs(BUCKET_NAME, prefix="summaries/")
        if blob.name.endswith("_summary.json")
    }
    skipped = [c for c in companies if str(c["ticker"]) in done]
    companies = [c for c in companies if str(c["ticker"]) not in done]
    if skipped:
        log.info(f"  已完成（跳過）：{len(skipped)} 家")
    log.info(f"待處理：{len(companies)} 家")

    results = []

    for i, company in enumerate(companies, 1):
        ticker = company["ticker"]
        log.info(f"\n[{i}/{len(companies)}] {ticker} {company.get('company', '')}")

        # 載入 chunks（文字 + 圖表）
        text_chunks   = load_company_chunks(gcs_client, ticker)
        vision_chunks = load_vision_chunks(gcs_client, ticker)
        all_chunks    = text_chunks + vision_chunks

        if not text_chunks:
            log.info(f"  跳過：Step 03/04 尚未完成（無文字 chunks）")
            results.append({"ticker": ticker, "status": "skipped_no_text_chunks"})
            continue

        if not all_chunks:
            results.append({"ticker": ticker, "status": "no_chunks"})
            continue

        log.info(f"  chunks：{len(all_chunks)}（文字 {len(text_chunks)} + 圖表 {len(vision_chunks)}）")

        # 生成摘要（三段式，內部自行選取 chunks）
        summary = generate_summary(model, company, all_chunks)
        if summary is None:
            results.append({"ticker": ticker, "status": "generation_failed"})
            continue

        # 轉換為 Overview Chunk
        overview_chunk = summary_to_chunk(summary)

        # 儲存
        save_summary(gcs_client, ticker, summary, overview_chunk)

        log.info(
            f"  ✓ 摘要完成 "
            f"（淨零目標：{summary.get('net_zero_target_year', 'N/A')}，"
            f"Scope1：{summary.get('scope1_tco2e', 'N/A')} tCO2e，"
            f"信心：{summary.get('confidence', 0):.2f}）"
        )

        results.append({
            "ticker":   ticker,
            "status":   "success",
            "net_zero_year": summary.get("net_zero_target_year"),
            "confidence":    summary.get("confidence"),
        })

    success_count = sum(1 for r in results if r["status"] == "success")
    log.info(f"\n=== 摘要生成結果 ===")
    log.info(f"  ✓ 成功：{success_count} 家")

    # 儲存報告
    report = {
        "completed_at":    datetime.utcnow().isoformat(),
        "company_results": results,
        "summary": {
            "total":   len(companies),
            "success": success_count,
        }
    }
    bucket.blob(f"logs/summary_report_{REPORT_YEAR}.json").upload_from_string(
        json.dumps(report, ensure_ascii=False, indent=2),
        content_type="application/json"
    )

    log.info(f"\n✅ 完成！")
    log.info(f"   下一步：執行 08_build_faiss_index.py（建立本地向量索引）")
    return report


if __name__ == "__main__":
    main()
