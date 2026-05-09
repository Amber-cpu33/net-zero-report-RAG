"""
底層：全域狀態、常數、FAISS 載入。
不依賴任何專案內部模組。
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import faiss
import jieba
from google import genai
from google.genai import types
from google.cloud import storage
from rank_bm25 import BM25Okapi
from vertexai.language_models import TextEmbeddingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 環境設定 ──────────────────────────────────────────────────
PROJECT_ID       = os.getenv("GCP_PROJECT_ID", "net-zero-report-rag")
REGION           = os.getenv("GCP_REGION", "asia-east1")
BUCKET_NAME      = os.getenv("GCS_BUCKET", f"{PROJECT_ID}-esg-pipeline")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL  = "text-embedding-004"
GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_DIM    = 768

FAISS_DIR     = Path(__file__).parent / "faiss_index"
FAISS_PATH    = FAISS_DIR / "index.faiss"
METADATA_PATH = FAISS_DIR / "metadata.jsonl"
STATS_PATH    = FAISS_DIR / "index_stats.json"

# 300 → 600：原設定截斷了「前段無關、後段才是答案」的混合主題 chunk（如 c0016 CCUS），
# 導致 synthesis 看不到答案而拒答。Gemini 2.5 Flash context 1M tokens，8 chunks × 600 字 ≈ 1,600 tokens，無上限風險。
SNIPPET_MAX_CHARS   = int(os.getenv("SNIPPET_MAX_CHARS", "600"))
USE_BM25            = os.getenv("USE_BM25", "true").lower() == "true"
BM25_WEIGHT         = float(os.getenv("BM25_WEIGHT", "0.4"))   # RRF 融合時 BM25 佔比
RRF_K               = 60                                         # RRF 常數

TOP_K_RESULTS       = 12
OVERVIEW_BOOST      = 0.1
MIN_RELEVANCE_SCORE = 0.5
MIN_CHUNK_TEXT_LEN  = 50
MAX_INDUSTRY_EXPAND = 10


# ── 全域狀態 Singleton ────────────────────────────────────────

class AppState:
    """Application-level shared state（避免 FastAPI 重複初始化）"""
    faiss_index:             Optional[faiss.Index]       = None
    metadata:                Optional[list[dict]]        = None
    stats:                   Optional[dict]              = None
    embed_model:             Optional[TextEmbeddingModel] = None
    gen_client:              Optional[genai.Client]      = None
    esg_tools:               Optional[types.Tool]        = None
    company_index:           Optional[dict]              = None
    overview_index:          Optional[dict]              = None
    ticker_counts:           Optional[dict]              = None
    ticker_chunk_indices:    Optional[dict]              = None
    industry_chunk_indices:  Optional[dict]              = None
    bm25_index:              Optional[BM25Okapi]         = None
    bm25_corpus_tokens:      Optional[list[list[str]]]   = None
    loaded_at:               Optional[str]               = None


state = AppState()


# ── FAISS 載入 ────────────────────────────────────────────────

async def load_faiss_index():
    """載入 FAISS 索引（本地優先，fallback 到 GCS）"""
    if FAISS_PATH.exists():
        log.info(f"  從本地載入 FAISS 索引：{FAISS_PATH}")
    else:
        log.info("  本地 FAISS 不存在，從 GCS 下載...")
        await download_faiss_from_gcs()

    if not FAISS_PATH.exists():
        raise RuntimeError("無法載入 FAISS 索引！請確認 Day 6 腳本已執行")

    state.faiss_index = faiss.read_index(str(FAISS_PATH))
    log.info(f"  FAISS 索引已載入：{state.faiss_index.ntotal} 個向量")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        state.metadata = [json.loads(line) for line in f if line.strip()]
    log.info(f"  Metadata 已載入：{len(state.metadata)} 筆")

    if STATS_PATH.exists():
        state.stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))

    state.company_index          = {}
    state.overview_index         = {}
    state.ticker_counts          = {}
    state.ticker_chunk_indices   = {}
    state.industry_chunk_indices = {}
    for i, meta in enumerate(state.metadata):
        ticker   = meta.get("ticker", "")
        industry = meta.get("industry", "")
        if not ticker:
            continue
        if ticker not in state.company_index:
            state.company_index[ticker] = {
                "ticker":   ticker,
                "company":  meta.get("company", ticker),
                "industry": industry,
            }
        if meta.get("is_overview") and ticker not in state.overview_index:
            state.overview_index[ticker] = meta
        state.ticker_counts[ticker] = state.ticker_counts.get(ticker, 0) + 1
        state.ticker_chunk_indices.setdefault(ticker, []).append(i)
        if industry:
            state.industry_chunk_indices.setdefault(industry, []).append(i)

    if USE_BM25:
        import pickle
        bm25_index_path  = FAISS_DIR / "bm25_index.pkl"
        bm25_tokens_path = FAISS_DIR / "bm25_tokens.pkl"
        if bm25_index_path.exists():
            log.info("  載入 BM25 倒排索引（pickle）...")
            with open(bm25_index_path, "rb") as f:
                state.bm25_index = pickle.load(f)
            log.info(f"  BM25 索引已載入：{state.bm25_index.corpus_size} 筆")
        elif bm25_tokens_path.exists():
            log.info("  載入 BM25 corpus tokens（pickle）...")
            with open(bm25_tokens_path, "rb") as f:
                state.bm25_corpus_tokens = pickle.load(f)
            log.info("  建立 BM25 倒排索引...")
            state.bm25_index = BM25Okapi(state.bm25_corpus_tokens)
            log.info(f"  BM25 索引已建立：{len(state.bm25_corpus_tokens)} 筆")
        else:
            log.info("  建立 BM25 corpus tokens（jieba，首次建立）...")
            jieba.setLogLevel(logging.WARNING)
            state.bm25_corpus_tokens = [
                list(jieba.cut(meta.get("text", ""))) for meta in state.metadata
            ]
            log.info("  建立 BM25 倒排索引...")
            state.bm25_index = BM25Okapi(state.bm25_corpus_tokens)
            log.info(f"  BM25 索引已建立：{len(state.bm25_corpus_tokens)} 筆")

    try:
        gcs_client = storage.Client(project=PROJECT_ID)
        blob = gcs_client.bucket(BUCKET_NAME).blob("company_data/company_list_2024.json")
        company_list = json.loads(blob.download_as_text(encoding="utf-8"))
        for item in company_list:
            t = item.get("ticker", "")
            if t not in state.company_index:
                continue
            if item.get("short_name"):
                state.company_index[t]["short_name"] = item["short_name"]
            if item.get("priority"):
                state.company_index[t]["priority"] = item["priority"]
        log.info("  公司短名已補入 company_index")
    except Exception as e:
        log.warning(f"  載入 company_list 失敗（short_name 不可用）：{e}")


async def download_faiss_from_gcs():
    """從 GCS 下載 FAISS 索引至本地"""
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET_NAME)

    for filename in ["index.faiss", "metadata.jsonl", "index_stats.json", "bm25_tokens.pkl", "bm25_index.pkl"]:
        local_path = FAISS_DIR / filename
        gcs_blob   = bucket.blob(f"faiss/{filename}")
        if gcs_blob.exists():
            gcs_blob.download_to_filename(str(local_path))
            log.info(f"    已下載：{filename}")
        else:
            log.warning(f"    GCS 找不到：{filename}")
