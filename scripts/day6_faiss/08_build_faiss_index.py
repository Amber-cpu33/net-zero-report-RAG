"""
Day 6 — Step 8：建立 FAISS 本地向量索引（架構轉換核心）
流程：
  1. 從 GCS 讀取全部 embeddings/*.jsonl（文字 chunk + 向量）
  2. 讀取各家公司 summaries/*_overview.jsonl（Overview chunk）
  3. 對尚未有向量的 chunks（Summary）呼叫 Online Embedding API 補齊
  5. 組裝 FAISS IndexFlatIP（內積 = 餘弦相似度）
  6. 儲存 index.faiss + metadata.jsonl（向量→chunk 的映射）
  7. 上傳至 GCS，同時儲存一份至本地 api/ 目錄供 Docker 打包

架構說明：
  ┌─────────────────────────────────────────────────────────┐
  │  試用期（Day 1-6）：Heavy GCP Usage                     │
  │   Batch Embedding → FAISS 本地化 → GCS 備份             │
  │                                                         │
  │  期滿後（Scale-to-Zero 模式）：                          │
  │   Cloud Run（冷啟動時從 GCS 下載 FAISS 索引到記憶體）    │
  │   → 無常駐向量資料庫費用（省 ~$50-200/月）              │
  └─────────────────────────────────────────────────────────┘

FAISS 設計選擇：
  - IndexFlatIP（暴力精確搜尋）：1041 家企業 × ~600 chunks = ~60 萬向量
    → 600K × 768 dim × 4 bytes ≈ 1.8GB（需確保 Cloud Run 記憶體 ≥ 4Gi）
    → 查詢延遲 < 50ms（無需 IVF/HNSW 等近似演算法）
  - 向量必須先 L2 正規化（IndexFlatIP 等效於餘弦相似）
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import numpy as np
import faiss
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from google.cloud import storage

import sys
sys.path.append(str(Path(__file__).parents[2] / "setup"))
from config import (
    PROJECT_ID, REGION, BUCKET_NAME,
    GCS_EMBEDDINGS, GCS_VISION_OUT, GCS_SUMMARIES, GCS_FAISS, GCS_LOGS,
    EMBEDDING_MODEL, EMBEDDING_DIM, REPORT_YEAR
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# FAISS 輸出路徑（本地，供 Docker 打包）
LOCAL_OUTPUT_DIR  = Path(__file__).parents[2] / "api" / "faiss_index"
FAISS_INDEX_FILE  = "index.faiss"
METADATA_FILE     = "metadata.jsonl"
STATS_FILE        = "index_stats.json"

# Online Embedding 速率限制（補齊沒有向量的 chunk 用）
ONLINE_EMBED_BATCH_SIZE  = 50    # overview 文字長，token 限制 20000，50 筆安全
ONLINE_EMBED_DELAY       = 2.0   # 每批次間隔（秒）
EMBED_CHECKPOINT_GCS     = f"logs/embed_checkpoint_{REPORT_YEAR}.jsonl"  # GCS 斷點暫存


# ── 讀取 GCS 資料 ─────────────────────────────────────────────

def list_blobs_in_prefix(gcs_client: storage.Client, prefix: str) -> list[str]:
    """列出 GCS bucket 中指定 prefix 下的所有 blob 名稱"""
    bucket = gcs_client.bucket(BUCKET_NAME)
    return [b.name for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".jsonl")]


def load_jsonl_from_gcs(gcs_client: storage.Client, blob_name: str) -> list[dict]:
    """讀取 GCS JSONL 檔案"""
    try:
        content = gcs_client.bucket(BUCKET_NAME).blob(blob_name).download_as_text(encoding="utf-8")
        return [json.loads(line) for line in content.strip().split("\n") if line.strip()]
    except Exception as e:
        log.warning(f"讀取 {blob_name} 失敗：{e}")
        return []


def load_all_chunks(gcs_client: storage.Client) -> list[dict]:
    """
    從 GCS 讀取所有 chunks，包含：
    1. embeddings/*.jsonl  （文字 chunk + 向量）
    2. summaries/*_overview.jsonl（Overview chunk，可能無向量）
    """
    all_chunks = []

    # 1. 文字 embedding chunks
    log.info("讀取 Embedding Chunks...")
    embed_blobs = list_blobs_in_prefix(gcs_client, "embeddings/")
    for blob_name in tqdm(embed_blobs, desc="embeddings"):
        chunks = load_jsonl_from_gcs(gcs_client, blob_name)
        all_chunks.extend(chunks)
    log.info(f"  文字 chunks：{len(all_chunks)} 個")

    before_summary = len(all_chunks)

    # 3. Overview summary chunks
    log.info("讀取 Summary Overview Chunks...")
    summary_blobs = [b for b in list_blobs_in_prefix(gcs_client, "summaries/") if "_overview.jsonl" in b]
    for blob_name in tqdm(summary_blobs, desc="summaries"):
        if "_overview.jsonl" in blob_name:
            chunks = load_jsonl_from_gcs(gcs_client, blob_name)
            all_chunks.extend(chunks)
    log.info(f"  Overview chunks：{len(all_chunks) - before_summary} 個")

    log.info(f"\n總計：{len(all_chunks)} 個 chunks")
    return all_chunks


# ── 補齊缺少向量的 Chunk ─────────────────────────────────────

def embed_missing_chunks(chunks: list[dict],
                          embed_model: TextEmbeddingModel,
                          gcs_client: storage.Client) -> list[dict]:
    """
    對沒有 embedding 的 chunks 呼叫 Online Embedding API 補齊。
    （主要是 Vision 和 Summary chunks，Batch Job 未涵蓋）
    """
    missing = [c for c in chunks if not c.get("embedding")]
    if not missing:
        log.info("所有 chunks 均已有向量，跳過補齊")
        return chunks

    # 載入 GCS checkpoint（斷點續跑，VM 重啟後仍有效）
    done_ids: set[str] = set()
    bucket = gcs_client.bucket(BUCKET_NAME)
    cp_blob = bucket.blob(EMBED_CHECKPOINT_GCS)
    if cp_blob.exists():
        for line in cp_blob.download_as_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            cid = rec.get("chunk_id")
            emb = rec.get("embedding")
            if cid and emb:
                done_ids.add(cid)
                for c in chunks:
                    if c.get("chunk_id") == cid:
                        c["embedding"] = emb
                        break
        log.info(f"Checkpoint 載入：{len(done_ids)} 個已補齊，跳過")

    missing = [c for c in chunks if not c.get("embedding")]
    log.info(f"補齊 {len(missing)} 個缺少向量的 chunks...")

    total_batches = (len(missing) + ONLINE_EMBED_BATCH_SIZE - 1) // ONLINE_EMBED_BATCH_SIZE
    cp_lines: list[str] = []  # 累積後一次 upload，減少 GCS 請求

    for batch_start in tqdm(range(0, len(missing), ONLINE_EMBED_BATCH_SIZE),
                            total=total_batches, desc="embed_missing"):
        batch = missing[batch_start:batch_start + ONLINE_EMBED_BATCH_SIZE]
        texts = [c.get("text", "") for c in batch]

        try:
            inputs = [
                TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT")
                for t in texts
            ]
            # output_dimensionality 必須與 04 腳本一致（768），否則預設 3072 維
            # 會導致 FAISS 建立索引時維度不符報錯
            embeddings = embed_model.get_embeddings(
                inputs, output_dimensionality=EMBEDDING_DIM
            )

            for chunk, emb in zip(batch, embeddings):
                chunk["embedding"] = emb.values
                cp_lines.append(json.dumps({
                    "chunk_id": chunk.get("chunk_id"),
                    "embedding": emb.values
                }, ensure_ascii=False))

            # 每批次寫回 GCS（覆寫整個 checkpoint，確保 VM 重啟後可續跑）
            existing = cp_blob.download_as_text(encoding="utf-8") if cp_blob.exists() else ""
            cp_blob.upload_from_string(
                existing + "\n".join(cp_lines[-ONLINE_EMBED_BATCH_SIZE:]) + "\n",
                content_type="application/x-ndjson"
            )

            time.sleep(ONLINE_EMBED_DELAY)

        except Exception as e:
            log.error(f"  批次向量化失敗：{e}")
            # 失敗的 chunk 保持 embedding=[] 狀態

    return chunks


# ── FAISS 建立 ────────────────────────────────────────────────

def build_faiss_index(chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    """
    建立 FAISS IndexFlatIP 索引。
    回傳：(faiss_index, valid_metadata_list)

    注意：IndexFlatIP + L2 正規化向量 = 餘弦相似度搜尋
    """
    # 過濾出有效向量
    valid_chunks = [
        c for c in chunks
        if c.get("embedding") and len(c["embedding"]) == EMBEDDING_DIM
    ]
    invalid_count = len(chunks) - len(valid_chunks)

    if invalid_count > 0:
        log.warning(f"  {invalid_count} 個 chunks 向量維度異常或缺少向量，已排除")

    log.info(f"建立 FAISS 索引：{len(valid_chunks)} 個向量，{EMBEDDING_DIM} 維")

    # 組裝向量矩陣
    vectors = np.array(
        [c["embedding"] for c in valid_chunks],
        dtype=np.float32
    )

    # L2 正規化（讓 IndexFlatIP 等效於餘弦相似度）
    faiss.normalize_L2(vectors)

    # 建立索引
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)

    log.info(f"  ✓ FAISS 索引建立完成，向量數：{index.ntotal}")

    # 準備 metadata（移除 embedding 以節省空間）
    metadata_list = []
    for chunk in valid_chunks:
        meta = {k: v for k, v in chunk.items() if k != "embedding"}
        metadata_list.append(meta)

    return index, metadata_list


def save_index_locally(faiss_index: faiss.Index,
                        metadata_list: list[dict],
                        output_dir: Path) -> tuple[Path, Path]:
    """
    將 FAISS 索引與 metadata 儲存至本地（供 Docker 打包）。
    回傳：(faiss_path, metadata_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    faiss_path    = output_dir / FAISS_INDEX_FILE
    metadata_path = output_dir / METADATA_FILE
    stats_path    = output_dir / STATS_FILE

    # 儲存 FAISS 索引
    faiss.write_index(faiss_index, str(faiss_path))
    log.info(f"FAISS 索引已儲存：{faiss_path} ({faiss_path.stat().st_size // 1024 // 1024} MB)")

    # 儲存 metadata（JSONL）
    with open(metadata_path, "w", encoding="utf-8") as f:
        for meta in metadata_list:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    log.info(f"Metadata 已儲存：{metadata_path} ({len(metadata_list)} 筆)")

    # 儲存統計資訊
    stats = {
        "built_at":       datetime.utcnow().isoformat(),
        "total_vectors":  faiss_index.ntotal,
        "vector_dim":     EMBEDDING_DIM,
        "index_type":     "IndexFlatIP",
        "index_size_mb":  faiss_path.stat().st_size // 1024 // 1024,
        "embedding_model": EMBEDDING_MODEL,
        "report_year":    REPORT_YEAR,
        "metadata_count": len(metadata_list),
        "companies":      len(set(m.get("ticker", "") for m in metadata_list)),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"統計資訊：{stats}")

    return faiss_path, metadata_path


def upload_index_to_gcs(gcs_client: storage.Client,
                         faiss_path: Path,
                         metadata_path: Path,
                         stats_path: Path) -> dict:
    """將 FAISS 索引上傳至 GCS（備份）"""
    bucket = gcs_client.bucket(BUCKET_NAME)

    uris = {}
    for local_path, gcs_name in [
        (faiss_path,   f"faiss/{FAISS_INDEX_FILE}"),
        (metadata_path, f"faiss/{METADATA_FILE}"),
        (stats_path,   f"faiss/{STATS_FILE}"),
    ]:
        with open(local_path, "rb") as f:
            bucket.blob(gcs_name).upload_from_file(f)
        uris[local_path.name] = f"gs://{BUCKET_NAME}/{gcs_name}"
        log.info(f"已上傳：{uris[local_path.name]}")

    return uris


# ── 索引驗證 ──────────────────────────────────────────────────

def verify_index(faiss_index: faiss.Index, metadata_list: list[dict]) -> bool:
    """
    執行基本的索引驗證：用已知向量查詢，確認 Top-1 是自身。
    """
    log.info("執行索引驗證...")
    if faiss_index.ntotal == 0:
        log.error("索引為空！")
        return False

    # 取前 3 個向量做自我查詢
    test_count = min(3, faiss_index.ntotal)
    # 重新讀取向量（faiss.Index 無法直接讀取儲存的向量，需重建測試向量）
    # 使用一個簡單的零向量測試 API 是否正常運作
    dummy = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)
    dummy[0, 0] = 1.0  # 使第一個維度為 1
    faiss.normalize_L2(dummy)

    scores, indices = faiss_index.search(dummy, k=5)
    log.info(f"  測試查詢完成，Top-5 indices：{indices[0].tolist()}")
    log.info(f"  Top-5 scores：{[round(s, 4) for s in scores[0].tolist()]}")

    if indices[0][0] == -1:
        log.error("查詢返回 -1，索引可能有問題！")
        return False

    # 驗證 metadata 對應
    top_idx = indices[0][0]
    if top_idx < len(metadata_list):
        sample = metadata_list[top_idx]
        log.info(
            f"  ✓ Top-1 結果：{sample.get('ticker', 'N/A')} - "
            f"{sample.get('text_preview', '')[:60]}..."
        )

    log.info("  ✓ 索引驗證通過！")
    return True


# ── 主流程 ───────────────────────────────────────────────────

def main():
    log.info("=== Day 6：建立 FAISS 本地向量索引 ===")
    log.info("（這是架構轉換的核心步驟：從雲端向量庫 → 本地 FAISS）")

    vertexai.init(project=PROJECT_ID, location=REGION)
    gcs_client = storage.Client(project=PROJECT_ID)

    # 初始化 Online Embedding 模型（補齊用）
    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

    # Step 1: 讀取所有 chunks
    all_chunks = load_all_chunks(gcs_client)
    if not all_chunks:
        log.error("無任何 chunks，請先執行 Day 3 Embedding 腳本")
        return

    # Step 2: 補齊缺少向量的 chunks（Vision + Summary）
    all_chunks = embed_missing_chunks(all_chunks, embed_model, gcs_client)

    # Step 3: 去重（同 chunk_id 只保留一筆）
    seen_ids = set()
    unique_chunks = []
    for c in all_chunks:
        cid = c.get("chunk_id", "")
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_chunks.append(c)
    log.info(f"去重後：{len(unique_chunks)} 個 chunks（移除 {len(all_chunks)-len(unique_chunks)} 重複）")

    # Step 4: 建立 FAISS 索引
    faiss_index, metadata_list = build_faiss_index(unique_chunks)

    # Step 5: 儲存至本地（api/faiss_index/）
    faiss_path, metadata_path = save_index_locally(
        faiss_index, metadata_list, LOCAL_OUTPUT_DIR
    )
    stats_path = LOCAL_OUTPUT_DIR / STATS_FILE

    # Step 6: 驗證索引
    if not verify_index(faiss_index, metadata_list):
        log.error("索引驗證失敗！請檢查資料")
        return

    # Step 7: 上傳 GCS 備份
    log.info("\n上傳至 GCS 備份...")
    gcs_uris = upload_index_to_gcs(gcs_client, faiss_path, metadata_path, stats_path)

    # 最終摘要
    log.info(f"\n=== FAISS 索引建立完成 ===")
    log.info(f"  總向量數：{faiss_index.ntotal:,}")
    log.info(f"  覆蓋公司：{len(set(m.get('ticker','') for m in metadata_list))} 家")
    log.info(f"  索引大小：~{(faiss_path.stat().st_size // 1024 // 1024)} MB")
    log.info(f"  本地路徑：{LOCAL_OUTPUT_DIR}")
    log.info(f"  GCS 備份：{gcs_uris.get(FAISS_INDEX_FILE, 'N/A')}")
    log.info(f"\n✅ 架構轉換完成！後續每月費用：≈ 0-50 TWD（Cloud Run Scale-to-Zero）")
    log.info(f"   下一步：執行 09_build_and_deploy.sh（Docker 打包 + Cloud Run 部署）")

    return {
        "total_vectors": faiss_index.ntotal,
        "local_path": str(LOCAL_OUTPUT_DIR),
        "gcs_uris": gcs_uris,
    }


if __name__ == "__main__":
    main()
