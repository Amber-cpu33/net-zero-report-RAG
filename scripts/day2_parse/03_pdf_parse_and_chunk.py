"""
Day 2 — Step 3：PDF 文字萃取 + 語意感知 Chunking (v3)
流程：
  1. 從 GCS 讀取 PDF
  2. PyMuPDF CropBox 裁切（去除 nav bar / 頁尾）+ pymupdf4llm Markdown 萃取
  3. 若萃取率 < 30%，改送 Document AI OCR（掃描版）
  4. Markdown 標題感知 Chunking（H2+ 麵包屑注入 + token 上限補充切割）
  5. 結果存為 JSONL 上傳 GCS

執行方式：
  全量：  python 03_pdf_parse_and_chunk.py
  Tier：  python 03_pdf_parse_and_chunk.py --tier 1
  產業：  python 03_pdf_parse_and_chunk.py --industry 24 27
  組合：  python 03_pdf_parse_and_chunk.py --tier 2 --industry 24
  指定：  python 03_pdf_parse_and_chunk.py --ticker 2330 2317
  強制：  python 03_pdf_parse_and_chunk.py --ticker 2330 --force
"""

import argparse
import csv
import gc
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import fitz  # PyMuPDF
import pymupdf4llm
from google.cloud import documentai, storage

import sys
sys.path.append(str(Path(__file__).parents[2] / "setup"))
from config import (
    PROJECT_ID, BUCKET_NAME,
    REPORT_YEAR, CHUNK_MAX_TOKENS, CHUNK_OVERLAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

PDF_PARSE_TIMEOUT = 300

NAV_FILTER_TOP    = 0.05   # 頁面頂端 5% 以內視為 nav bar
NAV_FILTER_BOTTOM = 0.92   # 頁面底端 8% 以內視為頁尾
NAV_LOG_PATH = Path(__file__).parents[2] / "logs" / "job03_nav_filtered.csv"

DOCAI_LOCATION     = "us"
DOCAI_PROCESSOR_ID = "your-processor-id"


# ── PDF 萃取 ─────────────────────────────────────────────────

def extract_with_pymupdf(
    pdf_bytes: bytes,
    ticker: str = "",
    nav_log_writer: csv.DictWriter | None = None,
) -> tuple[list[dict], float]:
    """
    CropBox 裁切 nav bar，pymupdf4llm 轉 Markdown（page_chunks=True）。
    單一迴圈同時記錄 nav_log 與設定 CropBox，不重複遍歷。
    回傳：(page_chunks, 萃取成功率)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    for page in doc:
        h = page.rect.height
        w = page.rect.width

        if nav_log_writer and h > 0:
            try:
                blocks = page.get_text("blocks")
            except Exception:
                blocks = []
            for b in blocks:
                _, y0, _, _, text, *_ = b
                y_pct = y0 / h
                text = (text or "").strip()
                if text and (y_pct < NAV_FILTER_TOP or y_pct > NAV_FILTER_BOTTOM):
                    nav_log_writer.writerow({
                        "ticker": ticker,
                        "page":   page.number + 1,
                        "y_pct":  f"{y_pct:.3f}",
                        "text":   text.replace("\n", " ")[:200],
                    })

        crop = fitz.Rect(0, h * NAV_FILTER_TOP, w, h * NAV_FILTER_BOTTOM) & page.mediabox
        if not crop.is_empty:
            page.set_cropbox(crop)

    page_chunks = pymupdf4llm.to_markdown(doc, page_chunks=True, show_progress=False)
    doc.close()

    text_pages = sum(1 for c in page_chunks if c.get("text", "").strip())
    success_rate = text_pages / total_pages if total_pages > 0 else 0
    return page_chunks, success_rate


def extract_with_docai(pdf_bytes: bytes) -> str:
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID)
    raw_doc = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")
    result = client.process_document(
        request=documentai.ProcessRequest(name=name, raw_document=raw_doc)
    )
    return result.document.text


def extract_text(
    pdf_bytes: bytes,
    ticker: str,
    nav_log_writer: csv.DictWriter | None = None,
) -> tuple[list[dict], str]:
    """
    智慧萃取：先 PyMuPDF + pymupdf4llm，萃取率低改用 Document AI。
    回傳：(page_chunks, 萃取方法)
    Document AI 路徑回傳 [{"metadata": {"page": 1}, "text": full_text}]。
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                extract_with_pymupdf, pdf_bytes, ticker, nav_log_writer
            )
            page_chunks, rate = future.result(timeout=PDF_PARSE_TIMEOUT)
    except FuturesTimeoutError:
        log.warning(f"[{ticker}] PyMuPDF 逾時（>{PDF_PARSE_TIMEOUT}s），跳過")
        return [], "timeout"
    except Exception as e:
        log.warning(f"[{ticker}] PyMuPDF 發生例外：{e}，跳過")
        return [], "pymupdf_error"

    if rate >= 0.3:
        log.info(f"[{ticker}] PyMuPDF 萃取成功（{rate:.0%}）")
        return page_chunks, "pymupdf"

    if DOCAI_PROCESSOR_ID == "your-processor-id":
        log.warning(
            f"[{ticker}] PyMuPDF 萃取率低（{rate:.0%}），DOCAI 未設定，維持 pymupdf-only"
        )
        return page_chunks, "pymupdf_low_quality"

    log.info(f"[{ticker}] PyMuPDF 萃取率低（{rate:.0%}），改用 Document AI OCR")
    full_text = extract_with_docai(pdf_bytes)
    return [{"metadata": {"page": 1}, "text": full_text}], "document_ai"


# ── 文字清理 ─────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


# ── Token 估算 ────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    chinese = len(re.findall(r"[一-鿿]", text))
    others  = len(re.findall(r"[a-zA-Z0-9]+", text))
    return int((chinese + others) * 1.3)


# ── Markdown 標題感知 Chunking ────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+")


def _is_real_heading(content: str, level: int) -> bool:
    """H4-H6 需驗證：避免 pymupdf4llm 把加粗數字（如 '9,000 噸'）誤判為標題。"""
    if level <= 4:
        return True
    cjk = len(re.findall(r"[一-鿿]", content))
    alpha_words = re.findall(r"[a-zA-Z]{3,}", content)
    return cjk >= 2 or len(alpha_words) >= 1


TOC_PAGE_THRESHOLD = 5   # source_pages max 在此頁以內才考慮是目錄
TOC_LINE_RATIO     = 0.6  # 非空行中以數字結尾的比例超過此值視為目錄


def _looks_like_toc(text: str, source_pages: list[int]) -> bool:
    """目錄偵測：(1) 位於前 5 頁；(2) 超過 60% 的非空行以數字結尾（頁碼）。"""
    if not source_pages or max(source_pages) > TOC_PAGE_THRESHOLD:
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    page_num_lines = sum(1 for l in lines if re.search(r"\d+\s*$", l.rstrip()))
    return page_num_lines / len(lines) >= TOC_LINE_RATIO


def smart_chunk(
    page_chunks: list[dict],
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Markdown 標題感知斷句：
    - 遇 H1-H6 強制斷開，更新 heading_stack
    - 每個 chunk 開頭注入 H2+ 麵包屑（特徵注入，確保 embedding 看到語意脈絡）
    - 超過 max_tokens 補充切割，同標題內保留少量 overlap
    輸入：pymupdf4llm page_chunks [{"metadata": {"page": N}, "text": str}, ...]
    回傳：[{"text": str, "source_pages": [int]}, ...]
    """
    # 展開成帶頁碼的段落清單
    tagged_paras: list[tuple[int, str]] = []
    for pc in page_chunks:
        page_num = pc.get("metadata", {}).get("page", 0)
        text = clean_text(pc.get("text", ""))
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                tagged_paras.append((page_num, para))

    heading_stack: dict[int, str] = {}  # level -> heading line（H1 存但不注入）
    chunks: list[dict] = []
    current_paras: list[str] = []
    current_pages: list[int] = []
    current_tok = 0

    def _flush(paras: list[str], pages: list[int]) -> None:
        if not paras:
            return
        body = "\n\n".join(paras)
        breadcrumb = "\n".join(
            v for k, v in sorted(heading_stack.items()) if k >= 2
        )
        text = f"{breadcrumb}\n\n{body}" if breadcrumb else body
        sp = sorted(set(pages))
        chunks.append({"text": text, "source_pages": sp, "is_toc": _looks_like_toc(text, sp)})

    for page_num, para in tagged_paras:
        first_line = para.split("\n")[0]
        m = _HEADING_RE.match(first_line)
        if m and _is_real_heading(first_line[m.end():], len(m.group(1))):
            level = len(m.group(1))
            _flush(current_paras, current_pages)
            current_paras, current_pages, current_tok = [], [], 0
            heading_stack = {k: v for k, v in heading_stack.items() if k < level}
            heading_stack[level] = first_line
            # 標題行已進入 heading_stack（麵包屑），body 只保留標題後的內文
            rest = "\n".join(para.split("\n")[1:]).strip()
            if not rest:
                continue  # 純標題行，不加入 body
            para = rest

        tok = estimate_tokens(para)
        if current_tok + tok > max_tokens and current_paras:
            _flush(current_paras, current_pages)
            # 同標題內 overlap：保留最後幾段不超過 overlap_tokens
            overlap_paras: list[str] = []
            overlap_pages: list[int] = []
            overlap_tok = 0
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

    _flush(current_paras, current_pages)
    return chunks


# ── 主流程 ───────────────────────────────────────────────────

def process_one_company(
    company: dict,
    gcs_client: storage.Client,
    nav_log_writer: csv.DictWriter | None = None,
    force: bool = False,
) -> dict:
    ticker = company["ticker"]
    bucket = gcs_client.bucket(BUCKET_NAME)

    if not force and bucket.blob(f"chunks/{ticker}_{REPORT_YEAR}.jsonl").exists():
        log.info(f"[{ticker}] chunks 已存在，跳過")
        return {"ticker": ticker, "status": "success", "chunk_count": 0, "method": "cached"}

    pdf_blob = bucket.blob(f"raw_pdfs/{ticker}_{REPORT_YEAR}.pdf")
    if not pdf_blob.exists():
        return {"ticker": ticker, "status": "pdf_not_found"}

    pdf_bytes = pdf_blob.download_as_bytes(timeout=120)
    page_chunks, method = extract_text(pdf_bytes, ticker, nav_log_writer)
    del pdf_bytes

    if not page_chunks:
        return {"ticker": ticker, "status": "extraction_failed", "method": method}

    full_text = " ".join(pc.get("text", "") for pc in page_chunks)
    if len(full_text) < 500:
        log.warning(f"[{ticker}] 文字過短（{len(full_text)} 字元），可能萃取失敗")
        return {"ticker": ticker, "status": "extraction_failed", "method": method}

    chunks = smart_chunk(page_chunks)

    jsonl_lines = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"].encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
        record = {
            "chunk_id":          f"{ticker}_{REPORT_YEAR}_c{i:04d}",
            "company":           company["company"],
            "ticker":            ticker,
            "industry":          company.get("industry", ""),
            "report_year":       company["report_year"],
            "extraction_method": method,
            "chunk_index":       i,
            "source_pages":      chunk["source_pages"],
            "is_toc":            chunk.get("is_toc", False),
            "text":              text,
        }
        jsonl_lines.append(json.dumps(record, ensure_ascii=False))

    bucket.blob(f"chunks/{ticker}_{REPORT_YEAR}.jsonl").upload_from_string(
        "\n".join(jsonl_lines), content_type="application/x-ndjson"
    )
    log.info(f"[{ticker}] ✓ {len(chunks)} chunks → GCS（方法：{method}）")
    return {
        "ticker":      ticker,
        "status":      "success",
        "chunk_count": len(chunks),
        "method":      method,
        "char_count":  len(full_text),
    }


# ── Checkpoint ───────────────────────────────────────────────

CHECKPOINT_BLOB  = f"logs/parse_checkpoint_{REPORT_YEAR}.json"
CHECKPOINT_EVERY = 10


def load_checkpoint(bucket) -> tuple[int, set]:
    blob = bucket.blob(CHECKPOINT_BLOB)
    if not blob.exists():
        return 0, set()
    data = json.loads(blob.download_as_text())
    processed = set(data.get("processed", []))
    start_index = data.get("last_index", 0)
    log.info(f"從 checkpoint 恢復：已處理 {len(processed)} 家，從第 {start_index + 1} 家繼續")
    return start_index, processed


def save_checkpoint(bucket, last_index: int, processed_list: list) -> None:
    bucket.blob(CHECKPOINT_BLOB).upload_from_string(
        json.dumps(
            {"last_index": last_index, "processed": processed_list},
            ensure_ascii=False,
        ),
        content_type="application/json",
    )


# ── CLI ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 03: PDF parse and chunk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""範例：
  全量：          python 03_pdf_parse_and_chunk.py
  Tier 1 only：  python 03_pdf_parse_and_chunk.py --tier 1
  Tier 1+2：     python 03_pdf_parse_and_chunk.py --tier 2
  指定產業：     python 03_pdf_parse_and_chunk.py --industry 24 27
  指定 ticker：  python 03_pdf_parse_and_chunk.py --ticker 2330 2317
  強制重跑：     python 03_pdf_parse_and_chunk.py --ticker 2330 --force
""",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ticker", nargs="+", metavar="TICKER",
        help="指定 ticker（可多個）；預設強制重跑，不用加 --force",
    )
    group.add_argument(
        "--tier", type=int, choices=[1, 2, 3, 4],
        help="執行 priority <= N 的公司（1=Tier1，2=Tier1+2，依此類推）",
    )
    group.add_argument(
        "--industry", nargs="+", metavar="CODE",
        help="產業代碼過濾（兩位數，可多個，如 --industry 24 27）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="強制重跑（忽略已存在的 chunks 與 checkpoint）",
    )
    return parser.parse_args()


def filter_companies(companies: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.ticker:
        result = [c for c in companies if str(c["ticker"]) in set(args.ticker)]
        log.info(f"ticker 模式：{len(result)} 家")
    elif args.tier:
        result = [c for c in companies if c.get("priority", 99) <= args.tier]
        log.info(f"Tier 模式（≤{args.tier}）：{len(result)} 家")
    elif args.industry:
        codes = set(args.industry)
        result = [c for c in companies if str(c.get("industry", "")) in codes]
        log.info(f"產業模式（{', '.join(sorted(codes))}）：{len(result)} 家")
    else:
        result = companies
        log.info(f"全量模式：{len(result)} 家")
    return result


# ── 主程式 ───────────────────────────────────────────────────

def main() -> dict:
    log.info("=== Day 2：PDF 萃取 + Chunking (v3) ===")
    args = parse_args()

    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    all_companies = json.loads(
        bucket.blob(f"company_data/company_list_{REPORT_YEAR}.json").download_as_text()
    )
    log.info(f"公司清單載入：{len(all_companies)} 家")

    companies = filter_companies(all_companies, args)
    if not companies:
        log.warning("篩選後無公司，請確認 --ticker / --tier / --industry 參數")
        return {"success": 0, "total_chunks": 0}

    # ticker 模式與 --force 都跳過 checkpoint
    use_checkpoint = not args.ticker and not args.force
    if use_checkpoint:
        start_index, processed = load_checkpoint(bucket)
    else:
        start_index, processed = 0, set()
    processed_list = list(processed)

    force = args.force or bool(args.ticker)

    NAV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    nav_log_file = open(NAV_LOG_PATH, "w", newline="", encoding="utf-8")
    nav_log_writer = csv.DictWriter(
        nav_log_file, fieldnames=["ticker", "page", "y_pct", "text"]
    )
    nav_log_writer.writeheader()

    results = []
    for i, company in enumerate(companies, 1):
        if use_checkpoint and i <= start_index:
            continue
        ticker = company["ticker"]
        if use_checkpoint and ticker in processed:
            continue

        log.info(f"[{i}/{len(companies)}] {ticker} {company['company']}")
        try:
            result = process_one_company(company, gcs_client, nav_log_writer, force=force)
        except Exception as e:
            log.error(f"[{ticker}] 未預期錯誤：{e}")
            result = {"ticker": ticker, "status": "error", "error": str(e)}

        results.append(result)

        if use_checkpoint and result["status"] == "success":
            processed.add(ticker)
            processed_list.append(ticker)

        gc.collect()

        if use_checkpoint and i % CHECKPOINT_EVERY == 0:
            save_checkpoint(bucket, i, processed_list)

        time.sleep(0.2)

    if use_checkpoint:
        save_checkpoint(bucket, len(companies), processed_list)

    success      = [r for r in results if r["status"] == "success"]
    failed       = [r for r in results if r["status"] != "success"]
    total_chunks = sum(r.get("chunk_count", 0) for r in success)

    log.info("\n=== 萃取結果 ===")
    log.info(f"  ✓ 成功：{len(success)} 家，共 {total_chunks:,} chunks")
    log.info(f"  ✗ 失敗/跳過：{len(failed)} 家")
    for r in failed:
        log.info(f"    [{r['ticker']}] {r['status']} {r.get('error', '')}")

    bucket.blob(f"logs/parse_report_{REPORT_YEAR}.json").upload_from_string(
        json.dumps(results, ensure_ascii=False, indent=2),
        content_type="application/json",
    )

    nav_log_file.close()
    log.info(f"  導覽列過濾紀錄：{NAV_LOG_PATH}")
    log.info("\n✅ 完成！下一步：執行 Day 3 Embedding Batch 送出")
    return {"success": len(success), "total_chunks": total_chunks}


if __name__ == "__main__":
    main()
