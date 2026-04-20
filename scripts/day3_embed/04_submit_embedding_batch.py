"""
Day 3 — Step 4：批次提交 Vertex AI Embedding（text-embedding-004）
流程：
  1. 從 GCS 讀取所有 chunks/*.jsonl
  2. 合併成單一 JSONL，一次性提交 Batch Prediction Job
  3. 輸出至 gs://{BUCKET}/raw_outputs/{job_id}/
  4. 後處理（ticker 分組）由 04b_merge_embeddings.py 負責

架構說明：
  - 模型：text-embedding-004，output_dimensionality=768
    → 保留 95%+ 精度，索引大小比 3072 維縮減 75%
    → Cloud Run FAISS 服務可跑在 2GB RAM，冷啟動 2-3 秒
  - 單一 Batch Job 可處理高達 100 萬筆，無需分批（省冷啟動延遲）
  - metadata 放在每筆 instance 中，Vertex 輸出會原樣回傳
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from google.cloud import storage, aiplatform
from google.cloud.aiplatform import BatchPredictionJob

import sys
sys.path.append(str(Path(__file__).parents[2] / "setup"))
from config import (
    PROJECT_ID, REGION, BUCKET_NAME,
    GCS_CHUNKS, GCS_LOGS,
    EMBEDDING_MODEL, EMBEDDING_DIM, REPORT_YEAR
)

TIER = int(os.getenv("TIER", "0"))  # 0 = 全跑，1-4 = 只跑指定 Tier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

EMBEDDING_MODEL_RESOURCE = f"publishers/google/models/{EMBEDDING_MODEL}"
POLL_INTERVAL_S  = 60
MAX_WAIT_HOURS   = 6


# ── 輔助函式 ──────────────────────────────────────────────────

def list_chunk_files(gcs_client: storage.Client) -> list[str]:
    blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix="chunks/"))
    names = [b.name for b in blobs if b.name.endswith(".jsonl")]
    log.info(f"找到 {len(names)} 個 chunk 檔案")
    return names


def load_chunks_from_gcs(gcs_client: storage.Client, blob_name: str) -> list[dict]:
    content = gcs_client.bucket(BUCKET_NAME).blob(blob_name).download_as_text(encoding="utf-8")
    return [json.loads(line) for line in content.strip().split("\n") if line.strip()]


def build_batch_input_jsonl(chunks: list[dict]) -> str:
    """
    每行格式（2026 標準）：
    {"content": "...", "task_type": "RETRIEVAL_DOCUMENT",
     "output_dimensionality": 768, "metadata": {"chunk_id": "...", ...}}
    Vertex 輸出會原樣回傳 metadata，供 04b 對齊使用。
    """
    lines = []
    for chunk in chunks:
        record = {
            "content":              chunk["text"],
            "task_type":            "RETRIEVAL_DOCUMENT",
            "output_dimensionality": EMBEDDING_DIM,      # 768（截斷 3072 維，省 75% 記憶體）
            "metadata": {
                "chunk_id":     chunk["chunk_id"],
                "ticker":       chunk["ticker"],
                "chunk_index":  chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1),
            }
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines)


def upload_batch_input(gcs_client: storage.Client,
                       jsonl_content: str, job_id: str) -> str:
    blob_name = f"batch_inputs/embed_input_{job_id}.jsonl"
    gcs_client.bucket(BUCKET_NAME).blob(blob_name).upload_from_string(
        jsonl_content, content_type="application/x-ndjson"
    )
    gcs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
    log.info(f"Batch input 已上傳：{gcs_uri}（{len(jsonl_content.encode())//1024//1024} MB）")
    return gcs_uri


# ── Batch Job 提交 ────────────────────────────────────────────

def submit_batch_embedding_job(input_gcs_uri: str,
                                output_gcs_prefix: str,
                                job_display_name: str) -> BatchPredictionJob:
    log.info(f"提交 Batch Job：{job_display_name}")
    job = aiplatform.BatchPredictionJob.create(
        job_display_name=job_display_name,
        model_name=EMBEDDING_MODEL_RESOURCE,
        instances_format="jsonl",
        gcs_source=[input_gcs_uri],
        predictions_format="jsonl",
        gcs_destination_prefix=output_gcs_prefix,
    )
    log.info(f"  Job 名稱：{job.resource_name}")
    log.info(f"  狀態：{job.state.name}")
    return job


def poll_until_done(job: BatchPredictionJob,
                    poll_interval: int = POLL_INTERVAL_S,
                    max_hours: float = MAX_WAIT_HOURS) -> bool:
    max_polls = int(max_hours * 3600 / poll_interval)
    resource_name = job.resource_name
    for i in range(max_polls):
        job = BatchPredictionJob(resource_name)
        state = job.state.name
        log.info(f"  [{i+1}/{max_polls}] 狀態：{state}")
        if state == "JOB_STATE_SUCCEEDED":
            log.info("  ✓ Batch Job 完成")
            return True
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            log.error(f"  ✗ Job 失敗：{job.error}")
            return False
        time.sleep(poll_interval)
    log.error(f"  ✗ 等待逾時（{max_hours} 小時）")
    return False


# ── 主流程 ───────────────────────────────────────────────────

def main():
    log.info(f"=== Day 3 Step 4：Batch Embedding 提交（{EMBEDDING_MODEL}）===")

    aiplatform.init(project=PROJECT_ID, location=REGION)
    gcs_client = storage.Client(project=PROJECT_ID)

    # 讀取所有 chunks
    chunk_blob_names = list_chunk_files(gcs_client)
    if not chunk_blob_names:
        log.error("找不到任何 chunk 檔案，請先執行 Step 3")
        return

    if TIER > 0:
        company_list = json.loads(
            gcs_client.bucket(BUCKET_NAME).blob(
                f"company_data/company_list_{REPORT_YEAR}.json"
            ).download_as_text()
        )
        allowed = {c["ticker"] for c in company_list if c.get("priority", 4) == TIER}
        chunk_blob_names = [b for b in chunk_blob_names if b.split("/")[-1].split("_")[0] in allowed]
        log.info(f"TIER={TIER}，篩選後：{len(chunk_blob_names)} 個 chunk 檔案")

    all_chunks = []
    for blob_name in chunk_blob_names:
        chunks = load_chunks_from_gcs(gcs_client, blob_name)
        all_chunks.extend(chunks)
    log.info(f"總計：{len(all_chunks):,} chunks")

    # 費用估算
    total_chars = sum(len(c.get("text", "")) for c in all_chunks)
    est_usd = (total_chars / 1000) * 0.00002
    log.info(f"預估費用：${est_usd:.2f} USD ≈ {est_usd*32:.0f} TWD（{total_chars:,} 字元）")

    # 一次性提交單一 Batch Job
    job_id = f"esg-embed-{REPORT_YEAR}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    output_prefix = f"gs://{BUCKET_NAME}/raw_outputs/{job_id}"

    input_jsonl = build_batch_input_jsonl(all_chunks)
    input_uri   = upload_batch_input(gcs_client, input_jsonl, job_id)

    job = submit_batch_embedding_job(
        input_gcs_uri=input_uri,
        output_gcs_prefix=output_prefix,
        job_display_name=job_id,
    )

    # 等待完成
    success = poll_until_done(job)

    # 儲存 job 資訊供 04b 使用
    job_info = {
        "job_id":          job_id,
        "job_resource":    job.resource_name,
        "output_prefix":   output_prefix,
        "chunk_count":     len(all_chunks),
        "total_chars":     total_chars,
        "est_cost_usd":    round(est_usd, 4),
        "status":          "success" if success else "failed",
        "completed_at":    datetime.utcnow().isoformat(),
    }
    gcs_client.bucket(BUCKET_NAME).blob(
        f"logs/embed_job_{REPORT_YEAR}.json"
    ).upload_from_string(
        json.dumps(job_info, ensure_ascii=False, indent=2),
        content_type="application/json"
    )

    if success:
        log.info(f"\n✅ 完成！下一步：執行 04b_merge_embeddings.py 合併向量")
        log.info(f"   output_prefix：{output_prefix}")
    else:
        log.error("✗ Batch Job 失敗，請查看 Cloud Logging")

    return job_info


if __name__ == "__main__":
    main()
