"""
Day 3 — Step 4b：合併 Embedding 輸出，依 ticker 分組存回 GCS
流程：
  1. 讀取 logs/embed_job_{REPORT_YEAR}.json 取得 output_prefix
  2. 掃描 raw_outputs/{job_id}/ 下所有 JSONL（Vertex 輸出）
  3. 依 metadata.ticker 分組，與原始 chunks 合併（補齊完整欄位）
  4. 存至 embeddings/{ticker}_{REPORT_YEAR}.jsonl

設計說明：
  - 分兩步（04 submit + 04b merge）避免單一 Batch 跨 ticker 覆蓋問題
  - 04b 可重跑：embeddings/ 已存在的 ticker 會跳過
  - Vertex 輸出格式：
    {"instance": {"content": ..., "metadata": {...}},
     "prediction": {"embeddings": {"values": [...]}}}
"""

import json
import logging
from pathlib import Path

from google.cloud import storage

import sys
sys.path.append(str(Path(__file__).parents[2] / "setup"))
from config import PROJECT_ID, BUCKET_NAME, EMBEDDING_DIM, REPORT_YEAR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


def load_job_info(gcs_client: storage.Client) -> dict:
    blob = gcs_client.bucket(BUCKET_NAME).blob(f"logs/embed_job_{REPORT_YEAR}.json")
    return json.loads(blob.download_as_text())


def load_original_chunks(gcs_client: storage.Client) -> dict[str, dict]:
    """載入所有原始 chunks，建立 chunk_id → chunk 映射"""
    bucket = gcs_client.bucket(BUCKET_NAME)
    chunk_map = {}
    for blob in bucket.list_blobs(prefix="chunks/"):
        if not blob.name.endswith(".jsonl"):
            continue
        for line in blob.download_as_text().strip().split("\n"):
            if line.strip():
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    log.warning(f"跳過損壞行：{blob.name}")
                    continue
                if "chunk_id" not in c:
                    log.warning(f"跳過無 chunk_id 的行：{blob.name} → {list(c.keys())}")
                    continue
                chunk_map[c["chunk_id"]] = c
    log.info(f"載入原始 chunks：{len(chunk_map):,} 筆")
    return chunk_map


def load_embedding_outputs(gcs_client: storage.Client,
                            output_prefix: str) -> list[dict]:
    """讀取 Vertex Batch 輸出，回傳 [{chunk_id, vector}, ...]"""
    bucket   = gcs_client.bucket(BUCKET_NAME)
    prefix   = output_prefix.replace(f"gs://{BUCKET_NAME}/", "")
    blobs    = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".jsonl")]
    log.info(f"找到 {len(blobs)} 個輸出檔案")

    results = []
    for blob in blobs:
        for line in blob.download_as_text().strip().split("\n"):
            if not line.strip():
                continue
            record     = json.loads(line)
            instance   = record.get("instance", {})
            prediction = record.get("prediction") or (record.get("predictions") or [{}])[0]
            chunk_id   = instance.get("metadata", {}).get("chunk_id", "")
            vector     = prediction.get("embeddings", {}).get("values", [])
            if not chunk_id or not vector:
                continue
            if len(vector) != EMBEDDING_DIM:
                log.warning(f"向量維度異常：{chunk_id}（{len(vector)} 維，預期 {EMBEDDING_DIM}）")
                continue
            results.append({"chunk_id": chunk_id, "vector": vector})

    log.info(f"讀取向量：{len(results):,} 筆")
    return results


def already_processed(gcs_client: storage.Client) -> set[str]:
    """回傳 embeddings/ 已存在的 ticker 集合（支援重跑跳過）"""
    blobs = gcs_client.bucket(BUCKET_NAME).list_blobs(prefix="embeddings/")
    return {
        b.name.split("/")[1].split(f"_{REPORT_YEAR}")[0]
        for b in blobs if b.name.endswith(".jsonl")
    }


def main():
    log.info("=== Day 3 Step 4b：合併 Embedding 輸出 ===")

    gcs_client  = storage.Client(project=PROJECT_ID)
    job_info    = load_job_info(gcs_client)
    output_prefix = job_info["output_prefix"]
    log.info(f"Job ID：{job_info['job_id']}，輸出位置：{output_prefix}")

    if job_info.get("status") != "success":
        log.error("Batch Job 未成功，請先確認 04 的執行結果")
        return

    chunk_map    = load_original_chunks(gcs_client)
    embed_results = load_embedding_outputs(gcs_client, output_prefix)
    done_tickers = already_processed(gcs_client)
    log.info(f"已處理（跳過）：{len(done_tickers)} 家")

    # 依 ticker 分組
    ticker_groups: dict[str, list[dict]] = {}
    missing = 0
    for item in embed_results:
        chunk_id = item["chunk_id"]
        if chunk_id not in chunk_map:
            log.warning(f"找不到原始 chunk：{chunk_id}")
            missing += 1
            continue
        ticker = chunk_map[chunk_id]["ticker"]
        if ticker in done_tickers:
            continue
        enriched = dict(chunk_map[chunk_id])
        enriched["embedding"] = item["vector"]
        ticker_groups.setdefault(ticker, []).append(enriched)

    log.info(f"待寫入：{len(ticker_groups)} 家（遺失原始 chunk：{missing} 筆）")

    # 依 ticker 寫入 GCS
    bucket = gcs_client.bucket(BUCKET_NAME)
    success_count = 0
    for ticker, chunks in ticker_groups.items():
        jsonl = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks)
        blob_name = f"embeddings/{ticker}_{REPORT_YEAR}.jsonl"
        bucket.blob(blob_name).upload_from_string(jsonl, content_type="application/x-ndjson")
        log.info(f"  [{ticker}] {len(chunks)} chunks → {blob_name}")
        success_count += 1

    log.info(f"\n✅ 完成！{success_count} 家寫入 embeddings/")
    log.info("   下一步：執行 05_vision_chart_extract.py")
    return {"success": success_count, "missing_chunks": missing}


if __name__ == "__main__":
    main()
