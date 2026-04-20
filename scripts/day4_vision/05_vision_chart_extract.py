"""
Day 4 — Step 5：Gemini 多模態圖表萃取（動態 Thinking 策略）
流程：
  1. 從 GCS raw_pdfs/ 讀取 PDF，逐頁轉換為圖片
  2. Layer 1：gemini-2.5-flash 快速過濾（有無圖表/表格）
  3. Layer 2（動態）：
     - 簡單表格 → gemini-2.5-flash（無 thinking，$0.4/1M output）
     - 複雜圖表 → gemini-2.5-pro（thinking 預設，不支援 budget=0）
  4. 信心分數 ≥ 0.7 → vision_output/confirmed/
     信心分數 <  0.7 → vision_output/pending_review/ 待人工複核

費用估算（1041 家，2026 定價）：
  Layer 1（2.5-flash）：~$0.13 USD
  Layer 2a（2.5-flash 簡單表格）：~$0.60 USD
  Layer 2b（2.5-pro 複雜圖表）：~$12.40 USD
  合計：~$13 USD ≈ 416 TWD（比純 1.5-pro 便宜約 33%，品質更高）
"""

import base64
import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from google.cloud import storage
from pdf2image import convert_from_bytes

import sys
sys.path.append(str(Path(__file__).parents[2] / "setup"))
from config import (
    PROJECT_ID, REGION_GEN, BUCKET_NAME,
    VISION_MODEL, VISION_MODEL_FAST, FILTER_MODEL,
    GCS_VISION_OUT, GCS_EMBEDDINGS, GCS_LOGS,
    CONFIDENCE_THRESHOLD, REPORT_YEAR, EMBEDDING_DIM
)

TIER = int(os.getenv("TIER", "0"))  # 0 = 全跑，1-4 = 只跑指定 Tier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# 圖片設定
IMAGE_DPI       = 150    # 解析度：足夠 OCR，不要太高（省費用）
IMAGE_FORMAT    = "PNG"
MAX_IMAGE_SIZE  = 4 * 1024 * 1024   # 4MB，超過則降解析度

# 速率控制
FILTER_DELAY_S  = 1.0    # gemini-2.5-flash 有 QPM 限制，避免 429
FAST_DELAY_S    = 0.5    # 2.5-flash 簡單表格
PRO_DELAY_S     = 2.0    # 2.5-pro 複雜圖表（latency 本身 5-15s，2s 綽綽有餘）


# ── Prompt 設計 ───────────────────────────────────────────────

CHART_DETECTION_PROMPT = """
你是一個 ESG 資料分析師。請判斷此頁面是否包含「具體數值的圖表或表格」。

判斷標準：
- ✅ 含有：折線圖、長條圖、圓餅圖、數值表格，且包含 GHG 排放、能源、用水、廢棄物等 ESG 指標數值
- ❌ 不含：純文字說明、組織架構圖、照片、logo、空白頁

請回答 JSON 格式（不加任何註解）：
{
  "has_chart": true,
  "chart_types": ["bar", "line", "table", "pie"],
  "keywords": ["GHG", "碳排", "能源"],
  "confidence": 0.95
}
規則：confidence 為 0.0 到 1.0 的小數。只回傳 JSON，不加任何說明文字。
"""

CHART_EXTRACTION_PROMPT = """
你是一個專業的 ESG 資料萃取工程師。請從此 ESG 永續報告書圖表中，萃取所有具體的量化指標數值。

要求：
1. 識別所有數值（噸、千度、立方公尺、百分比等）
2. 對應到正確的年份（可能含多年比較）
3. 識別指標類別：
   - Scope 1/2/3 溫室氣體排放（tCO2e）
   - 能源消耗（GJ、MWh、千度）
   - 再生能源占比（%）
   - 用水量（m³、千公升）
   - 廢棄物（噸）
   - 員工人數與安全指標
   - 其他 ESG KPI

輸出 JSON 格式（嚴格遵守，不加任何註解）：
{
  "chart_title": "圖表標題",
  "reporting_standard": ["GRI", "TCFD"],
  "unit": "tCO2e",
  "data_points": [
    {
      "indicator": "Scope 1 溫室氣體排放",
      "year": 2023,
      "value": 12345.6,
      "unit": "tCO2e",
      "category": "emissions_scope1"
    }
  ],
  "scope_boundary": null,
  "notes": null,
  "confidence": 0.95
}
規則：
- 無法取得的欄位填 null，不得填 N/A、"-"、空字串
- value 必須為數字，無法辨識則填 null
- confidence 為 0.0 到 1.0 的小數
- 只回傳 JSON，不加任何說明文字或註解
"""


# ── 圖片處理 ──────────────────────────────────────────────────

def pdf_to_images(pdf_bytes: bytes, dpi: int = IMAGE_DPI) -> list[bytes]:
    """將 PDF 每頁轉換為 PNG bytes list"""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt=IMAGE_FORMAT)
        result = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format=IMAGE_FORMAT)
            img_bytes = buf.getvalue()

            # 若圖片過大則降解析度
            if len(img_bytes) > MAX_IMAGE_SIZE:
                buf = io.BytesIO()
                img_small = img.resize(
                    (img.width // 2, img.height // 2)
                )
                img_small.save(buf, format=IMAGE_FORMAT)
                img_bytes = buf.getvalue()

            result.append(img_bytes)
        return result
    except Exception as e:
        log.error(f"PDF 轉圖失敗：{e}")
        return []


def image_to_vertex_part(img_bytes: bytes) -> Part:
    """將 PNG bytes 轉為 Vertex AI Part（inline image）"""
    return Part.from_data(data=img_bytes, mime_type="image/png")


# ── Gemini 呼叫 ───────────────────────────────────────────────

def _parse_json_response(text: str) -> dict:
    """清理並解析模型回傳的 JSON（處理 markdown code block 與多餘文字）"""
    import re
    text = text.strip().replace("```json", "").replace("```", "").strip()
    # 移除 JSON 字串值以外的控制字元（換行、tab 等）
    text = re.sub(r'(?<!\\)[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def detect_chart_page(filter_model: GenerativeModel, img_bytes: bytes) -> dict:
    """
    Layer 1：用 gemini-2.5-flash 快速判斷此頁是否有圖表/表格。
    幾乎免費，適合全量過濾。
    """
    try:
        time.sleep(FILTER_DELAY_S)
        response = filter_model.generate_content(
            [image_to_vertex_part(img_bytes), CHART_DETECTION_PROMPT],
            generation_config={"temperature": 0.1, "max_output_tokens": 4096, "response_mime_type": "application/json", "thinking_config": {"thinking_budget": 0}},
        )
        result = _parse_json_response(response.text)
        log.info(f"偵測回覆：{response.text[:200]} → {result}")
        return result
    except Exception as e:
        log.warning(f"圖表偵測失敗：{e}")
        return {"has_chart": False, "confidence": 0.0}


def is_complex_chart(detection: dict) -> bool:
    """
    判斷是否為複雜圖表（需要 2.5-pro + thinking）。
    簡單表格（純 table）用 2.5-flash 即可，省 25 倍費用。
    """
    chart_types = detection.get("chart_types", [])
    # 只有 table，沒有其他圖表類型 → 簡單
    if chart_types == ["table"] or chart_types == []:
        return False
    # 含有折線/長條/圓餅 → 複雜
    return any(t in chart_types for t in ["bar", "line", "pie"])


def extract_chart_data(fast_model: GenerativeModel,
                        pro_model: GenerativeModel,
                        img_bytes: bytes,
                        ticker: str, page_num: int,
                        complex_chart: bool) -> Optional[dict]:
    """
    Layer 2（動態）：
    - 簡單表格 → gemini-2.5-flash（無 thinking）
    - 複雜圖表 → gemini-2.5-pro（thinking_budget=1024）
    """
    try:
        if complex_chart:
            time.sleep(PRO_DELAY_S)
            model = pro_model
            gen_config = {"temperature": 0.2, "max_output_tokens": 4096, "response_mime_type": "application/json"}
            model_tag = "2.5-pro"
        else:
            time.sleep(FAST_DELAY_S)
            model = fast_model
            gen_config = {"temperature": 0.2, "max_output_tokens": 4096, "response_mime_type": "application/json", "thinking_config": {"thinking_budget": 0}}
            model_tag = "2.5-flash"

        response = model.generate_content(
            [image_to_vertex_part(img_bytes), CHART_EXTRACTION_PROMPT],
            generation_config=gen_config,
        )
        result = _parse_json_response(response.text)
        result["source_ticker"] = ticker
        result["source_page"]   = page_num
        result["extracted_at"]  = datetime.utcnow().isoformat()
        result["model_used"]    = model_tag
        return result
    except Exception as e:
        try:
            raw = response.text if "response" in dir() else "（未呼叫）"
        except Exception:
            finish = ""
            if "response" in dir() and response.candidates:
                finish = str(response.candidates[0].finish_reason)
            raw = f"（response.text 不可讀，finish_reason={finish}）"
        log.warning(f"[{ticker}] 頁 {page_num} 萃取失敗：{e}\n  raw={raw[:200]}")
        return None


# ── 結果轉換為 Chunk 格式 ─────────────────────────────────────

def chart_data_to_chunks(chart_data: dict, company_info: dict) -> list[dict]:
    """
    將結構化圖表資料轉換為可向量化的 chunk 格式。
    每個 data_point 生成一個 chunk，確保細粒度的語意搜尋。
    """
    chunks = []
    ticker = chart_data.get("source_ticker", "")
    page   = chart_data.get("source_page", 0)
    title  = chart_data.get("chart_title", "ESG 圖表")
    unit   = chart_data.get("unit", "")
    standard = chart_data.get("reporting_standard", [])
    boundary = chart_data.get("scope_boundary", "")
    confidence = chart_data.get("confidence", 0.0)

    for dp_idx, dp in enumerate(chart_data.get("data_points", [])):
        indicator = dp.get("indicator", "")
        year      = dp.get("year", REPORT_YEAR)
        value     = dp.get("value")
        dp_unit   = dp.get("unit", unit)
        category  = dp.get("category", "other")

        if value is None:
            continue

        # 生成自然語言描述（向量化的文字）
        text = (
            f"{company_info.get('company', ticker)} "
            f"在 {year} 年度的「{indicator}」為 {value} {dp_unit}。"
            f"（圖表標題：{title}，報告書第 {page} 頁）"
        )
        if standard:
            text += f" 揭露依據：{', '.join(standard)}。"
        if boundary:
            text += f" 報告邊界：{boundary}。"

        chunk_id = f"{ticker}_{REPORT_YEAR}_vis_p{page:03d}_{dp_idx:02d}"

        chunk = {
            "chunk_id":               chunk_id,
            "company":                company_info.get("company", ticker),
            "ticker":                 ticker,
            "industry":               company_info.get("industry", ""),
            "report_year":            REPORT_YEAR,
            "data_year":              year,
            "reporting_standard":     standard,
            "scope_boundary":         boundary,
            "unit":                   dp_unit,
            "embedding_model_version": "text-embedding-004",
            "extraction_method":      "gemini_vision",
            "source_page":            page,
            "category":               category,
            "indicator":              indicator,
            "value":                  value,
            "chunk_index":            dp_idx,
            "total_chunks":           len(chart_data.get("data_points", [])),
            "text":                   text,
            "text_preview":           text[:150],
            "confidence_score":       confidence,
            "embedding":              [],  # 稍後由 Embedding 腳本填入
        }
        chunks.append(chunk)

    return chunks


# ── 主流程 ───────────────────────────────────────────────────

def process_one_company(ticker: str,
                         company_info: dict,
                         gcs_client: storage.Client,
                         filter_model: GenerativeModel,
                         fast_model: GenerativeModel,
                         pro_model: GenerativeModel) -> dict:
    """處理單一公司的所有圖表頁面"""
    bucket = gcs_client.bucket(BUCKET_NAME)

    # 讀取 PDF
    pdf_blob = bucket.blob(f"raw_pdfs/{ticker}_{REPORT_YEAR}.pdf")
    if not pdf_blob.exists():
        log.warning(f"[{ticker}] PDF 不存在，跳過")
        return {"ticker": ticker, "status": "pdf_not_found"}

    pdf_bytes = pdf_blob.download_as_bytes()
    log.info(f"[{ticker}] PDF 讀取完成（{len(pdf_bytes)//1024} KB）")

    # PDF → 圖片
    images = pdf_to_images(pdf_bytes)
    if not images:
        return {"ticker": ticker, "status": "image_conversion_failed"}
    log.info(f"[{ticker}] 共 {len(images)} 頁")

    # 逐頁處理
    all_chart_data  = []
    pending_review  = []
    flash_calls = 0
    pro_calls   = 0

    fast_calls = 0

    for page_num, img_bytes in enumerate(images, 1):
        # Layer 1: 2.0-flash 快速過濾
        detection = detect_chart_page(filter_model, img_bytes)
        flash_calls += 1

        if not detection.get("has_chart", False):
            if page_num <= 5:
                log.info(f"  頁 {page_num}：無圖表（confidence={detection.get('confidence', 0):.2f}，raw={detection}）")
            continue

        complex_chart = is_complex_chart(detection)
        model_tag = "2.5-pro" if complex_chart else "2.5-flash"
        log.info(
            f"  頁 {page_num}：{detection.get('chart_types', [])} "
            f"→ {model_tag}（信心 {detection.get('confidence', 0):.2f}）"
        )

        # Layer 2: 動態選模型萃取
        chart_data = extract_chart_data(
            fast_model, pro_model, img_bytes, ticker, page_num, complex_chart
        )
        if not complex_chart:
            fast_calls += 1
        else:
            pro_calls += 1

        if chart_data is None:
            continue

        confidence = chart_data.get("confidence", 0.0)

        if confidence >= CONFIDENCE_THRESHOLD:
            all_chart_data.append(chart_data)
            log.info(f"  頁 {page_num}：✓ 萃取成功（信心 {confidence:.2f}）")
        else:
            pending_review.append(chart_data)
            log.info(f"  頁 {page_num}：⚠ 信心不足（{confidence:.2f}），移至待複核")

    # 轉換為 chunk 格式
    all_chunks = []
    for chart_data in all_chart_data:
        chunks = chart_data_to_chunks(chart_data, company_info)
        all_chunks.extend(chunks)

    # 儲存已確認的 chunks（附加至現有 embeddings JSONL）
    if all_chunks:
        blob_name = f"vision_output/confirmed/{ticker}_{REPORT_YEAR}.jsonl"
        jsonl = "\n".join(json.dumps(c, ensure_ascii=False) for c in all_chunks)
        bucket.blob(blob_name).upload_from_string(
            jsonl, content_type="application/x-ndjson"
        )
        log.info(f"[{ticker}] ✓ {len(all_chunks)} 個圖表 chunks → GCS")

    # 儲存待複核的資料
    if pending_review:
        blob_name = f"vision_output/pending_review/{ticker}_{REPORT_YEAR}.json"
        bucket.blob(blob_name).upload_from_string(
            json.dumps(pending_review, ensure_ascii=False, indent=2),
            content_type="application/json"
        )
        log.warning(f"[{ticker}] ⚠ {len(pending_review)} 頁待人工複核")

    return {
        "ticker":           ticker,
        "status":           "success",
        "total_pages":      len(images),
        "chart_pages":      len(all_chart_data) + len(pending_review),
        "confirmed_chunks": len(all_chunks),
        "pending_review":   len(pending_review),
        "filter_calls":     flash_calls,
        "fast_calls":       fast_calls,
        "pro_calls":        pro_calls,
    }


def main():
    log.info("=== Day 4：Gemini Vision 圖表萃取 ===")

    vertexai.init(project=PROJECT_ID, location=REGION_GEN)
    gcs_client = storage.Client(project=PROJECT_ID)

    # 初始化三層模型
    filter_model = GenerativeModel(FILTER_MODEL)       # Layer 1 過濾（2.0-flash）
    fast_model   = GenerativeModel(VISION_MODEL_FAST)  # Layer 2a 簡單表格（2.5-flash）
    pro_model    = GenerativeModel(VISION_MODEL)       # Layer 2b 複雜圖表（2.5-pro）
    log.info(f"Filter 模型：{FILTER_MODEL}")
    log.info(f"Fast 模型：{VISION_MODEL_FAST}")
    log.info(f"Pro 模型：{VISION_MODEL}")

    # 讀取公司清單
    bucket = gcs_client.bucket(BUCKET_NAME)
    companies = json.loads(
        bucket.blob(f"company_data/company_list_{REPORT_YEAR}.json").download_as_text()
    )
    company_map = {c["ticker"]: c for c in companies}

    if TIER > 0:
        companies = [c for c in companies if c.get("priority", 4) == TIER]
        log.info(f"TIER={TIER}，篩選後：{len(companies)} 家")

    industry_codes_env = os.getenv("INDUSTRY_CODES", "")
    if industry_codes_env:
        codes = set(industry_codes_env.split(","))
        companies = [c for c in companies if c.get("industry", "") in codes]
        log.info(f"INDUSTRY_CODES={industry_codes_env}，篩選後：{len(companies)} 家")

    # 依優先級排序（priority 欄位由 01 腳本寫入；舊版清單若無此欄位則全部視為 Tier 4）
    companies.sort(key=lambda c: (c.get("priority", 4), c["ticker"]))

    shard_index = int(os.getenv("SHARD_INDEX", "0"))
    shard_total = int(os.getenv("SHARD_TOTAL", "1"))
    if shard_total > 1:
        companies = companies[shard_index::shard_total]
        log.info(f"SHARD {shard_index}/{shard_total}，分配：{len(companies)} 家")
    test_ticker = os.getenv("TEST_TICKER", "")
    if test_ticker:
        companies = [c for c in companies if c["ticker"] == test_ticker]
        log.info(f"TEST_TICKER={test_ticker}，只跑 1 家")
    tier_counts = {}
    for c in companies:
        t = c.get("priority", 4)
        tier_counts[t] = tier_counts.get(t, 0) + 1
    log.info(f"待處理：{len(companies)} 家（" +
             ", ".join(f"T{t}:{n}" for t, n in sorted(tier_counts.items())) + "）")

    results = []
    total_filter = 0
    total_fast   = 0
    total_pro    = 0

    for i, company in enumerate(companies, 1):
        ticker = company["ticker"]

        # 斷點保護：已有 confirmed 輸出則跳過（Spot VM 重啟後續跑用）
        already_done = bucket.blob(
            f"vision_output/confirmed/{ticker}_{REPORT_YEAR}.jsonl"
        ).exists()
        if already_done:
            log.info(f"[{i}/{len(companies)}] {ticker} 已處理，跳過")
            results.append({"ticker": ticker, "status": "skipped"})
            continue

        log.info(f"\n[{i}/{len(companies)}] {ticker} {company['company']}")

        result = process_one_company(
            ticker, company, gcs_client, filter_model, fast_model, pro_model
        )
        results.append(result)

        if result.get("status") == "success":
            total_filter += result.get("filter_calls", 0)
            total_fast   += result.get("fast_calls", 0)
            total_pro    += result.get("pro_calls", 0)

    # 費用估算（2026 定價）
    # 2.0-flash filter: input $0.10/1M × 258 tokens + output $0.40/1M × 50 tokens
    # 2.5-flash fast:   input $0.15/1M × 2000 tokens + output $0.60/1M × 512 tokens
    # 2.5-pro complex:  input $1.25/1M × 2000 tokens + output $10/1M × 512 tokens
    #                   + thinking $3.50/1M × thinking_budget
    filter_cost_usd = total_filter * (0.10 * 258 + 0.40 * 50)   / 1e6
    fast_cost_usd   = total_fast   * (0.15 * 2000 + 0.60 * 512) / 1e6
    pro_cost_usd    = total_pro    * (1.25 * 2000 + 10.0 * 512) / 1e6
    total_cost_usd  = filter_cost_usd + fast_cost_usd + pro_cost_usd
    total_cost_twd  = total_cost_usd * 32

    # 統計摘要
    success = [r for r in results if r.get("status") == "success"]
    total_confirmed = sum(r.get("confirmed_chunks", 0) for r in success)
    total_pending   = sum(r.get("pending_review", 0) for r in success)

    log.info(f"\n=== 圖表萃取結果 ===")
    log.info(f"  ✓ 成功：{len(success)} 家")
    log.info(f"  確認 chunks：{total_confirmed}")
    log.info(f"  待人工複核：{total_pending} 頁")
    log.info(f"\n=== API 呼叫費用估算 ===")
    log.info(f"  2.0-flash filter：{total_filter} 次 ≈ ${filter_cost_usd:.4f} USD")
    log.info(f"  2.5-flash fast：  {total_fast} 次 ≈ ${fast_cost_usd:.4f} USD")
    log.info(f"  2.5-pro thinking：{total_pro} 次 ≈ ${pro_cost_usd:.4f} USD")
    log.info(f"  總計：≈ ${total_cost_usd:.4f} USD ≈ {total_cost_twd:.1f} TWD")

    # 儲存報告
    report = {
        "completed_at":   datetime.utcnow().isoformat(),
        "company_results": results,
        "summary": {
            "total_companies":  len(companies),
            "success":          len(success),
            "confirmed_chunks": total_confirmed,
            "pending_review":   total_pending,
            "filter_calls":     total_filter,
            "fast_calls":       total_fast,
            "pro_calls":        total_pro,
            "est_cost_usd":     round(total_cost_usd, 4),
            "est_cost_twd":     round(total_cost_twd, 1),
        }
    }
    bucket.blob(f"logs/vision_report_{REPORT_YEAR}.json").upload_from_string(
        json.dumps(report, ensure_ascii=False, indent=2),
        content_type="application/json"
    )

    log.info(f"\n✅ 完成！")
    log.info(f"   下一步：執行 07_generate_summaries.py（公司摘要生成）")
    return report


if __name__ == "__main__":
    main()
