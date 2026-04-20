"""
Day 2 — Step 3：PDF 文字萃取 + 語言感知 Chunking
流程：
  1. 從 GCS 讀取 PDF
  2. 先用 pdfplumber 嘗試文字萃取（數位 PDF）
  3. 若萃取率 < 30%，改送 Document AI OCR（掃描版）
  4. 語言感知 Chunking（中英混合自適應）
  5. 結果存為 JSONL 上傳 GCS
"""

import gc
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Iterator

import pdfplumber
from google.cloud import documentai, storage

PDF_PARSE_TIMEOUT = 120  # 每個 PDF 最多處理 120 秒，避免損壞 PDF 卡住

import sys
sys.path.append(str(Path(__file__).parents[2] / "setup"))
from config import (
    PROJECT_ID, BUCKET_NAME, GCS_OCR_TEXT, GCS_CHUNKS,
    REPORT_YEAR, CHUNK_MAX_TOKENS, CHUNK_OVERLAP
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)
logging.getLogger("pdfminer").setLevel(logging.ERROR)  # 抑制 FontBBox 等字型解析雜訊

# Document AI 設定（OCR Processor，需在 Console 建立一次）
DOCAI_LOCATION   = "us"   # Document AI 目前只有 us / eu
DOCAI_PROCESSOR_ID = "your-processor-id"  # 建立後填入


# ── PDF 萃取 ─────────────────────────────────────────────────

def extract_with_pdfplumber(pdf_bytes: bytes) -> tuple[list[tuple[int, str]], float]:
    """
    使用 pdfplumber 萃取文字，保留頁碼資訊。
    回傳：([(page_num, text), ...], 萃取成功率)
    萃取成功率 = 有文字的頁數 / 總頁數
    """
    pages_text  = []
    total_pages = 0
    text_pages  = 0

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, 1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            finally:
                page.flush_cache()  # 強制釋放頁面物件，防止向量路徑殘留累積
            lines = [
                l.strip() for l in page_text.split("\n")
                if len(l.strip()) > 10
            ]
            if lines:
                text_pages += 1
                pages_text.append((page_num, "\n".join(lines)))

    success_rate = text_pages / total_pages if total_pages > 0 else 0
    return pages_text, success_rate


def extract_with_docai(pdf_bytes: bytes) -> str:
    """
    使用 GCP Document AI 進行 OCR。
    適用於掃描版 PDF（pdfplumber 萃取率 < 30%）。
    """
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID)

    raw_doc = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
    request = documentai.ProcessRequest(name=name, raw_document=raw_doc)

    result = client.process_document(request=request)
    return result.document.text


def extract_text(pdf_bytes: bytes, ticker: str) -> tuple[list[tuple[int, str]], str]:
    """
    智慧萃取：先用 pdfplumber（加逾時保護），若失敗改用 Document AI。
    回傳：([(page_num, text), ...], 萃取方法)
    Document AI 不含頁碼資訊，回傳 [(0, full_text)]。
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(extract_with_pdfplumber, pdf_bytes)
            pages_text, rate = future.result(timeout=PDF_PARSE_TIMEOUT)
    except FuturesTimeoutError:
        log.warning(f"[{ticker}] pdfplumber 逾時（>{PDF_PARSE_TIMEOUT}s），跳過此 PDF")
        return [], "timeout"
    except Exception as e:
        log.warning(f"[{ticker}] pdfplumber 發生例外：{e}，跳過")
        return [], "pdfplumber_error"

    if rate >= 0.3:
        log.info(f"[{ticker}] pdfplumber 萃取成功（{rate:.0%}）")
        return pages_text, "pdfplumber"

    if DOCAI_PROCESSOR_ID == "your-processor-id":
        log.warning(
            f"[{ticker}] pdfplumber 萃取率低（{rate:.0%}），"
            f"但 DOCAI_PROCESSOR_ID 尚未設定，維持 pdfplumber-only 模式"
        )
        return pages_text, "pdfplumber_low_quality"

    log.info(f"[{ticker}] pdfplumber 萃取率低（{rate:.0%}），改用 Document AI OCR")
    full_text = extract_with_docai(pdf_bytes)
    return [(0, full_text)], "document_ai"


# ── 報告標準偵測 ─────────────────────────────────────────────

_STANDARD_PATTERNS = [
    (r"\bGRI\b",                          "GRI"),
    (r"\bTCFD\b",                         "TCFD"),
    (r"\bSASB\b",                         "SASB"),
    (r"\bIFRS\s*S1\b",                    "IFRS S1"),
    (r"\bIFRS\s*S2\b",                    "IFRS S2"),
    (r"\bISO\s*26000\b",                  "ISO 26000"),
    (r"\bUN\s*SDGs?\b|永續發展目標",       "SDGs"),
    (r"\bUNGC\b|聯合國全球盟約",           "UNGC"),
    (r"\bCDP\b",                          "CDP"),
    (r"\bSBTi\b|科學基礎減量目標",         "SBTi"),
]

def detect_reporting_standards(text: str) -> list[str]:
    """從報告書全文偵測引用的永續報告標準/框架。"""
    found = []
    for pattern, label in _STANDARD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE) and label not in found:
            found.append(label)
    return found


# ── 文字清理 ─────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """移除雜訊字元，標準化格式"""
    # 移除多餘空白行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 移除頁碼（獨立的數字行）
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
    # 統一全形數字→半形
    text = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    # 移除控制字元
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


# ── 語言感知 Chunking ────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    估算 token 數（不安裝完整 tokenizer 的輕量版）：
    - 中文字元：每字 ≈ 1.3 tokens
    - 英文 / 數字：每個空白分隔的詞 ≈ 1.3 tokens
    """
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    others  = len(re.findall(r"[a-zA-Z0-9]+", text))
    return int((chinese + others) * 1.3)


def smart_chunk(pages_text: list[tuple[int, str]],
                max_tokens: int = CHUNK_MAX_TOKENS,
                overlap_tokens: int = CHUNK_OVERLAP) -> list[dict]:
    """
    語意感知分段，保留每個 chunk 的來源頁碼。
    輸入：[(page_num, text), ...]
    回傳：[{"text": str, "source_pages": [int, ...]}, ...]
    """
    # 展開成帶頁碼的段落清單
    tagged_paras: list[tuple[int, str]] = []
    for page_num, text in pages_text:
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                tagged_paras.append((page_num, para))

    chunks: list[dict]       = []
    current_paras: list[str] = []
    current_pages: list[int] = []
    current_tok              = 0

    def flush(paras, pages):
        if paras:
            chunks.append({
                "text":         "\n\n".join(paras),
                "source_pages": sorted(set(pages)),
            })

    for page_num, para in tagged_paras:
        tok = estimate_tokens(para)

        if tok > max_tokens:
            flush(current_paras, current_pages)
            sentences = re.split(r"(?<=[。！？\.\!\?])\s*", para)
            sub, sub_tok, sub_pages = [], 0, []
            for sent in sentences:
                s_tok = estimate_tokens(sent)
                if sub_tok + s_tok > max_tokens and sub:
                    chunks.append({"text": " ".join(sub), "source_pages": sorted(set(sub_pages))})
                    sub, sub_tok, sub_pages = sub[-2:], sum(estimate_tokens(s) for s in sub[-2:]), sub_pages[-2:]
                sub.append(sent)
                sub_tok += s_tok
                sub_pages.append(page_num)
            if sub:
                chunks.append({"text": " ".join(sub), "source_pages": sorted(set(sub_pages))})
            current_paras, current_pages, current_tok = [], [], 0
            continue

        if current_tok + tok > max_tokens and current_paras:
            flush(current_paras, current_pages)
            # Overlap：保留最後 2 個段落
            overlap_paras, overlap_pages, overlap_tok = [], [], 0
            for p, pg in zip(reversed(current_paras), reversed(current_pages)):
                p_tok = estimate_tokens(p)
                if overlap_tok + p_tok <= overlap_tokens:
                    overlap_paras.insert(0, p)
                    overlap_pages.insert(0, pg)
                    overlap_tok += p_tok
                else:
                    break
            current_paras, current_pages, current_tok = overlap_paras, overlap_pages, overlap_tok

        current_paras.append(para)
        current_pages.append(page_num)
        current_tok += tok

    flush(current_paras, current_pages)
    return chunks


# ── 主流程 ───────────────────────────────────────────────────

def process_one_company(company: dict, gcs_client: storage.Client) -> dict:
    """處理單一公司：萃取 → 清理 → Chunking → 上傳"""
    ticker = company["ticker"]
    bucket = gcs_client.bucket(BUCKET_NAME)

    # 已有 chunks 則跳過（重啟保護）
    if bucket.blob(f"chunks/{ticker}_{REPORT_YEAR}.jsonl").exists():
        log.info(f"[{ticker}] chunks 已存在，跳過")
        return {"ticker": ticker, "status": "success", "chunk_count": 0, "method": "cached"}

    # 讀取 PDF
    pdf_blob = bucket.blob(f"raw_pdfs/{ticker}_{REPORT_YEAR}.pdf")
    if not pdf_blob.exists():
        return {"ticker": ticker, "status": "pdf_not_found"}

    # 圖片密集型 PDF 門檻：> 10MB 以 pdfplumber 處理風險極高（嵌入圖片展開記憶體暴增）
    pdf_size_mb = pdf_blob.size / (1024 * 1024)
    if pdf_size_mb > 10:
        log.warning(f"[{ticker}] PDF 過大（{pdf_size_mb:.1f}MB），跳過（待 PyMuPDF fallback）")
        return {"ticker": ticker, "status": "skipped_large_pdf", "size_mb": round(pdf_size_mb, 1)}

    pdf_bytes = pdf_blob.download_as_bytes()

    # 萃取文字（含頁碼）
    pages_text, method = extract_text(pdf_bytes, ticker)
    del pdf_bytes  # 明確釋放大物件，避免跨公司記憶體累積
    if not pages_text:
        return {"ticker": ticker, "status": "extraction_failed", "method": method}

    # 清理每頁文字
    pages_text = [(pn, clean_text(t)) for pn, t in pages_text if clean_text(t)]

    full_text = "\n\n".join(t for _, t in pages_text)
    if len(full_text) < 500:
        log.warning(f"[{ticker}] 文字過短（{len(full_text)} 字元），可能萃取失敗")
        return {"ticker": ticker, "status": "extraction_failed", "method": method}

    # 偵測報告標準
    standards = detect_reporting_standards(full_text)
    if standards:
        log.info(f"[{ticker}] 偵測到報告標準：{standards}")

    # Chunking（含頁碼追蹤）
    chunks = smart_chunk(pages_text)

    # 組裝 JSONL（每行一個 chunk）
    jsonl_lines = []
    for i, chunk in enumerate(chunks):
        record = {
            "chunk_id":               f"{ticker}_{REPORT_YEAR}_c{i:04d}",
            "company":                company["company"],
            "ticker":                 ticker,
            "industry":               company.get("industry", ""),
            "report_year":            company["report_year"],
            "data_year":              company["report_year"],
            "reporting_standard":     standards,
            "scope_boundary":         "",
            "unit":                   "",
            "embedding_model_version": "text-embedding-004",
            "extraction_method":      method,
            "chunk_index":            i,
            "total_chunks":           len(chunks),
            "source_pages":           chunk["source_pages"],
            "text":                   chunk["text"],
            "text_preview":           chunk["text"][:150],
            "confidence_score":       1.0,
        }
        jsonl_lines.append(json.dumps(record, ensure_ascii=False))

    jsonl_content = "\n".join(jsonl_lines)

    # 上傳 JSONL
    chunk_blob = bucket.blob(f"chunks/{ticker}_{REPORT_YEAR}.jsonl")
    chunk_blob.upload_from_string(jsonl_content, content_type="application/x-ndjson")

    log.info(f"[{ticker}] ✓ {len(chunks)} chunks → GCS（方法：{method}）")
    return {
        "ticker":              ticker,
        "status":              "success",
        "chunk_count":         len(chunks),
        "method":              method,
        "char_count":          len(full_text),
        "reporting_standards": standards,
    }


CHECKPOINT_BLOB = f"logs/parse_checkpoint_{REPORT_YEAR}.json"
CHECKPOINT_EVERY = 10  # 每處理 N 家存一次 checkpoint


def load_checkpoint(bucket) -> tuple[int, set]:
    """
    讀取 GCS checkpoint，回傳 (start_index, processed_set)。
    start_index：上次跑到的位置，重啟時直接從此處繼續，不重掃 chunks/。
    """
    blob = bucket.blob(CHECKPOINT_BLOB)
    if not blob.exists():
        return 0, set()
    data = json.loads(blob.download_as_text())
    processed = set(data.get("processed", []))
    start_index = data.get("last_index", 0)
    log.info(f"從 checkpoint 恢復：已處理 {len(processed)} 家，從第 {start_index + 1} 家繼續")
    return start_index, processed


def save_checkpoint(bucket, last_index: int, processed_list: list):
    """儲存進度至 GCS，每 CHECKPOINT_EVERY 家或結束時呼叫。"""
    bucket.blob(CHECKPOINT_BLOB).upload_from_string(
        json.dumps({"last_index": last_index, "processed": processed_list}, ensure_ascii=False),
        content_type="application/json"
    )


def main():
    log.info("=== Day 2：PDF 萃取 + Chunking ===")

    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    # 讀取公司清單
    companies = json.loads(
        bucket.blob(f"company_data/company_list_{REPORT_YEAR}.json").download_as_text()
    )
    log.info(f"總公司數：{len(companies)} 家")

    # 從 checkpoint 快速恢復（取代掃描 chunks/ 目錄）
    start_index, processed = load_checkpoint(bucket)
    processed_list = list(processed)

    results = []
    for i, company in enumerate(companies, 1):
        if i <= start_index:
            continue  # 靜默跳過已確認處理的範圍，不產生 log

        ticker = company["ticker"]
        if ticker in processed:
            continue  # checkpoint 之前有失敗又重跑的保險

        log.info(f"[{i}/{len(companies)}] {ticker} {company['company']}")
        try:
            result = process_one_company(company, gcs_client)
        except Exception as e:
            log.error(f"[{ticker}] 未預期錯誤：{e}")
            result = {"ticker": ticker, "status": "error", "error": str(e)}

        results.append(result)

        if result["status"] == "success":
            processed.add(ticker)
            processed_list.append(ticker)

        gc.collect()  # 每家處理完強制 GC，防止記憶體累積

        if i % CHECKPOINT_EVERY == 0:
            save_checkpoint(bucket, i, processed_list)

        time.sleep(0.2)

    # 最終 checkpoint
    save_checkpoint(bucket, len(companies), processed_list)

    # 統計
    success = [r for r in results if r["status"] == "success"]
    failed  = [r for r in results if r["status"] not in ("success",)]
    total_chunks = sum(r.get("chunk_count", 0) for r in success)

    log.info(f"\n=== 萃取結果 ===")
    log.info(f"  ✓ 成功：{len(success)} 家，共 {total_chunks:,} chunks")
    log.info(f"  ✗ 失敗/跳過：{len(failed)} 家")
    log.info(f"  使用 Document AI：{sum(1 for r in success if r.get('method') == 'document_ai')} 家")

    bucket.blob(f"logs/parse_report_{REPORT_YEAR}.json").upload_from_string(
        json.dumps(results, ensure_ascii=False, indent=2),
        content_type="application/json"
    )

    log.info(f"\n✅ 完成！下一步：執行 Day 3 Embedding Batch 送出")
    return {"success": len(success), "total_chunks": total_chunks}


if __name__ == "__main__":
    main()
