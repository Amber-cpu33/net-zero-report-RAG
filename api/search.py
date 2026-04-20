"""
檢索層：FAISS 搜尋、公司比較、資料查詢。
只依賴 state.py，不牽涉 LLM 生成。
"""

import logging
import re
from typing import Optional

import faiss
import numpy as np
from vertexai.language_models import TextEmbeddingInput

from state import (
    state,
    EMBEDDING_DIM, OVERVIEW_BOOST, MIN_RELEVANCE_SCORE,
    MIN_CHUNK_TEXT_LEN, TOP_K_RESULTS,
)

log = logging.getLogger(__name__)

INDUSTRY_CODE_MAP = {
    "01": "水泥工業",      "02": "食品工業",      "03": "塑膠工業",
    "08": "玻璃陶瓷",      "09": "造紙工業",      "10": "鋼鐵工業",
    "12": "橡膠工業",      "15": "汽車工業",      "17": "電機機械",
    "18": "電器電纜",      "21": "化學工業",      "23": "油電燃氣業",
    "24": "半導體業",      "25": "電腦及週邊設備業", "26": "光電業",
    "27": "通信網路業",    "28": "電子零組件業",  "29": "電子通路業",
    "31": "其他電子業",
}


def clean_context_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\n{2,}', '<PARAGRAPH>', text)
    text = re.sub(r'\n\s*([-•]|\d+\.)', r'<LIST>\1', text)
    text = re.sub(r'([a-zA-Z0-9])\n\s*([a-zA-Z0-9])', r'\1 \2', text)
    text = text.replace('\n', '')
    text = text.replace('<PARAGRAPH>', '\n\n')
    text = text.replace('<LIST>', '\n')
    return text.strip()


def _get_metric_unit(metric: str) -> str:
    units = {
        "scope1_tco2e": "tCO2e",  "scope2_tco2e": "tCO2e", "scope3_tco2e": "tCO2e",
        "renewable_energy_pct": "%", "total_energy_gj": "GJ",
        "water_withdrawal_m3": "m³", "waste_total_ton": "噸",
    }
    return units.get(metric, "")


def _metric_to_chinese(metric: str) -> str:
    labels = {
        "scope1_tco2e": "Scope 1 溫室氣體排放",
        "scope2_tco2e": "Scope 2 溫室氣體排放",
        "scope3_tco2e": "Scope 3 溫室氣體排放",
        "renewable_energy_pct": "再生能源占比",
        "total_energy_gj": "能源消耗",
        "water_withdrawal_m3": "用水量",
        "waste_total_ton": "廢棄物",
    }
    return labels.get(metric, metric)


def embed_query(text: str) -> np.ndarray:
    """將查詢文字轉為 L2 正規化的向量"""
    inputs  = [TextEmbeddingInput(text=text, task_type="RETRIEVAL_QUERY")]
    results = state.embed_model.get_embeddings(inputs)
    vector  = np.array(results[0].values, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vector)
    return vector


def search_esg_knowledge_base(
    query: str,
    top_k: int = TOP_K_RESULTS,
    ticker_filter: Optional[str] = None,
    tickers_filter: Optional[list[str]] = None,
    industry_filter: Optional[str] = None,
    min_score: Optional[float] = None,
) -> list[dict]:
    """
    語意搜尋 FAISS 向量庫，回傳最相關的 chunks。
    ticker_filter：單一 ticker（Function Calling 用）。
    tickers_filter：多 ticker pre-filter（parse_query 路由用）。
    industry_filter：產業代碼 pre-filter（parse_query 路由用）。
    """
    if state.faiss_index is None:
        raise RuntimeError("FAISS 索引未載入")

    query_vector = embed_query(query)
    keywords = [kw for kw in query.split() if len(kw) >= 2]

    candidate_indices: Optional[list[int]] = None
    if tickers_filter and state.ticker_chunk_indices:
        candidate_indices = []
        for t in tickers_filter:
            candidate_indices.extend(state.ticker_chunk_indices.get(t, []))
    elif industry_filter and state.industry_chunk_indices:
        candidate_indices = list(state.industry_chunk_indices.get(industry_filter, []))
    elif ticker_filter and state.ticker_chunk_indices:
        candidate_indices = state.ticker_chunk_indices.get(ticker_filter, [])

    if candidate_indices is not None:
        if not candidate_indices:
            return []
        candidate_idx_arr = np.array(candidate_indices, dtype=np.int64)
        vectors = np.zeros((len(candidate_indices), EMBEDDING_DIM), dtype=np.float32)
        state.faiss_index.reconstruct_batch(candidate_idx_arr, vectors)
        raw_scores = (vectors @ query_vector.T).flatten()
        order = np.argsort(raw_scores)[::-1][:top_k * 3]
        pairs = [(float(raw_scores[j]), candidate_indices[j]) for j in order]
    else:
        search_k = min(top_k * 3, state.faiss_index.ntotal)
        raw_scores, raw_indices = state.faiss_index.search(query_vector, k=search_k)
        pairs = list(zip(raw_scores[0], raw_indices[0]))

    results = []
    for score, idx in pairs:
        if idx < 0 or idx >= len(state.metadata):
            continue
        meta = state.metadata[idx]

        text = meta.get("text", "")
        if keywords:
            matched = sum(1 for kw in keywords if kw in text)
            keyword_boost = 0.05 * matched / len(keywords)
        else:
            keyword_boost = 0.0

        adjusted_score = float(score) + keyword_boost
        if meta.get("is_overview"):
            adjusted_score += OVERVIEW_BOOST

        results.append({
            "score":            round(adjusted_score, 4),
            "chunk_id":         meta.get("chunk_id", ""),
            "company":          meta.get("company", ""),
            "ticker":           meta.get("ticker", ""),
            "industry":         meta.get("industry", ""),
            "text":             meta.get("text", ""),
            "data_year":        meta.get("data_year"),
            "category":         meta.get("category", "text"),
            "indicator":        meta.get("indicator", ""),
            "value":            meta.get("value"),
            "unit":             meta.get("unit", ""),
            "source_page":      (meta.get("source_pages") or [None])[0],
            "is_overview":      meta.get("is_overview", False),
            "confidence_score": meta.get("confidence_score", 1.0),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    threshold = min_score if min_score is not None else MIN_RELEVANCE_SCORE
    results = [r for r in results if r["score"] >= threshold
               and len(r.get("text", "")) >= MIN_CHUNK_TEXT_LEN]
    return results[:top_k]


def compare_companies(tickers: list[str], metric: str) -> list[dict]:
    """精確比較多家公司的特定指標。"""
    results = []
    for ticker in tickers:
        overview = state.overview_index.get(ticker) if state.overview_index else None
        if overview and overview.get("summary_metadata"):
            summary = overview["summary_metadata"]
            results.append({
                "ticker":        ticker,
                "company":       overview.get("company", ticker),
                "industry":      overview.get("industry", ""),
                "metric":        metric,
                "value":         summary.get(metric),
                "unit":          _get_metric_unit(metric),
                "net_zero_year": summary.get("net_zero_target_year"),
                "data_year":     summary.get("report_year"),
            })
        else:
            chunks = search_esg_knowledge_base(
                query=f"{ticker} {_metric_to_chinese(metric)}",
                top_k=3,
                ticker_filter=ticker,
            )
            results.append({
                "ticker":  ticker,
                "company": state.company_index.get(ticker, {}).get("company", ticker),
                "metric":  metric,
                "value":   chunks[0].get("value") if chunks else None,
                "source":  "semantic_search",
            })

    results.sort(key=lambda x: (x.get("value") is None, x.get("value") or 0))
    return results


def get_company_overview(ticker: str) -> Optional[dict]:
    """取得單一公司的 ESG 概況（summary_metadata）；無 overview chunk 時回傳 None"""
    for meta in state.metadata:
        if meta.get("ticker") == ticker and meta.get("is_overview"):
            return meta.get("summary_metadata") or {}
    return None


def lookup_company(name: str) -> list[dict]:
    """用公司名稱關鍵字查詢股票代號與產業"""
    if not state.company_index:
        return []
    results = []
    name_lower = name.lower()
    for ticker, info in state.company_index.items():
        company    = info.get("company", "")
        short_name = info.get("short_name", "")
        if (name_lower in company.lower() or name in company
                or (short_name and name_lower in short_name.lower())):
            industry_code = info.get("industry", "")
            results.append({
                "ticker":        ticker,
                "company":       company,
                "industry_code": industry_code,
                "industry_name": INDUSTRY_CODE_MAP.get(industry_code, industry_code),
            })

    def match_priority(r):
        info = state.company_index.get(r["ticker"], {})
        sn   = info.get("short_name", "")
        co   = info.get("company", "")
        if name == sn or name == co:
            return 0
        if sn and name in sn:
            return 1
        return 2

    results.sort(key=lambda x: (match_priority(x), x["ticker"]))
    return results[:10]
