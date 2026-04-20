"""
ESG RAG API — 入口層
=====================
只負責：FastAPI app 建立、Middleware、Lifespan、路由定義。
業務邏輯全部委派給 rag.py / search.py。
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import vertexai
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from vertexai.language_models import TextEmbeddingModel

from state import (
    state, load_faiss_index,
    PROJECT_ID, GEMINI_API_KEY, EMBEDDING_MODEL, GENERATION_MODEL, EMBEDDING_DIM,
)
from search import compare_companies, _metric_to_chinese, _get_metric_unit
from rag import agentic_rag, parse_query, build_esg_tools
from line_bot import register_line_bot

log = logging.getLogger(__name__)

TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "8"))


# ── Pydantic 模型 ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=5, max_length=500,
        description="ESG 相關問題（繁體中文或英文）",
        examples=["台積電的淨零目標是哪年？", "半導體產業平均 Scope 1 排放量是多少？"]
    )
    top_k: int = Field(default=TOP_K_RESULTS, ge=1, le=20)
    include_sources: bool = Field(default=True, description="是否回傳來源 chunks")


class CompareRequest(BaseModel):
    tickers: list[str] = Field(
        ..., min_length=2, max_length=10,
        description="公司代碼清單（如 [\"2330\", \"2317\"]）"
    )
    metric: str = Field(
        default="scope1_tco2e",
        description="比較指標（scope1_tco2e / scope2_tco2e / renewable_energy_pct 等）"
    )


class QueryResponse(BaseModel):
    question:   str
    answer:     str
    sources:    list[dict] = []
    tool_calls: list[str]  = []
    latency_ms: int        = 0


# ── Lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== ESG RAG API 啟動中 ===")
    startup_start = time.time()

    vertexai.init(project=PROJECT_ID, location="us-central1")
    log.info(f"  Vertex AI 初始化：project={PROJECT_ID}（Embedding 用）")

    await load_faiss_index()

    state.embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    state.gen_client  = genai.Client(api_key=GEMINI_API_KEY)
    state.esg_tools   = build_esg_tools()
    state.loaded_at   = datetime.utcnow().isoformat()

    log.info(f"  Embedding 模型：{EMBEDDING_MODEL}（Vertex AI）")
    log.info(f"  生成模型：{GENERATION_MODEL}（AI Studio）")
    log.info(f"=== 啟動完成（{time.time() - startup_start:.2f}s） ===")

    yield

    log.info("=== ESG RAG API 關閉 ===")


# ── FastAPI App ────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="ESG RAG API",
    description="台灣百大企業 ESG 永續知識庫 — 基於 FAISS + Vertex AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── 路由 ──────────────────────────────────────────────────────

@app.post("/debug/parse")
async def debug_parse(request: QueryRequest):
    result = parse_query(request.question)
    return result.model_dump()


@app.get("/health")
async def health_check():
    if state.faiss_index is None:
        raise HTTPException(status_code=503, detail="FAISS 索引未載入")
    return {
        "status":    "healthy",
        "vectors":   state.faiss_index.ntotal,
        "companies": len(state.company_index) if state.company_index else 0,
        "loaded_at": state.loaded_at,
        "model":     GENERATION_MODEL,
    }


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query_esg(request: Request, req: QueryRequest):
    if state.faiss_index is None:
        raise HTTPException(status_code=503, detail="服務尚未就緒，FAISS 索引載入中")

    log.info(f"收到查詢：{req.question[:80]}")
    result = agentic_rag(req.question)

    sources = []
    if req.include_sources:
        sources = [
            {
                "company":    s.get("company", ""),
                "ticker":     s.get("ticker", ""),
                "text":       s.get("text", "")[:200],
                "score":      s.get("score", 0),
                "data_year":  s.get("data_year"),
                "source_page": s.get("source_page"),
                "category":   s.get("category", ""),
            }
            for s in result.get("sources", [])
        ]

    return QueryResponse(
        question   = req.question,
        answer     = result["answer"],
        sources    = sources,
        tool_calls = result.get("tool_calls", []),
        latency_ms = result.get("latency_ms", 0),
    )


@app.get("/companies")
async def list_companies(industry: Optional[str] = None):
    if not state.company_index:
        raise HTTPException(status_code=503, detail="公司索引尚未載入")

    companies = list(state.company_index.values())
    if industry:
        companies = [c for c in companies if industry in c.get("industry", "")]

    result = [
        {**c, "chunk_count": state.ticker_counts.get(c["ticker"], 0)}
        for c in sorted(companies, key=lambda x: x.get("ticker", ""))
    ]
    return {"total": len(result), "companies": result}


@app.get("/stats")
async def get_stats():
    if state.faiss_index is None:
        raise HTTPException(status_code=503, detail="FAISS 索引未載入")

    method_counts   = {}
    category_counts = {}
    for meta in (state.metadata or []):
        method = meta.get("extraction_method", "unknown")
        cat    = meta.get("category", "text")
        method_counts[method]   = method_counts.get(method, 0) + 1
        category_counts[cat]    = category_counts.get(cat, 0) + 1

    return {
        "index_info": {
            "total_vectors": state.faiss_index.ntotal,
            "vector_dim":    EMBEDDING_DIM,
            "index_type":    "IndexFlatIP (Cosine Similarity)",
            "companies":     len(state.company_index) if state.company_index else 0,
        },
        "extraction_methods": method_counts,
        "categories":         category_counts,
        "models": {
            "embedding":  EMBEDDING_MODEL,
            "generation": GENERATION_MODEL,
        },
        "built_at":  state.stats.get("built_at") if state.stats else None,
        "loaded_at": state.loaded_at,
    }


@app.post("/compare")
async def compare_esg(req: CompareRequest):
    if state.faiss_index is None:
        raise HTTPException(status_code=503, detail="服務尚未就緒")

    valid_tickers = [t for t in req.tickers if t in state.company_index]
    if not valid_tickers:
        raise HTTPException(status_code=404, detail="找不到任何指定的公司")

    results      = compare_companies(valid_tickers, req.metric)
    metric_label = _metric_to_chinese(req.metric)
    unit         = _get_metric_unit(req.metric)

    return {
        "metric":       req.metric,
        "metric_label": metric_label,
        "unit":         unit,
        "results":      results,
        "note":         f"比較 {len(results)} 家公司的「{metric_label}」指標",
    }


# ── LINE Bot ──────────────────────────────────────────────────
register_line_bot(app, agentic_rag)


# ── 開發模式入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
        log_level="info",
    )
