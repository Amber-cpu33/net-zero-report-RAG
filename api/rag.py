"""
大腦層：意圖解析、Agentic RAG。
依賴 state.py、search.py、prompts.py、tools.py。
"""

import json
import logging
import re
import time
from typing import Optional

from google.genai import types
from pydantic import BaseModel, Field

from state import state, GENERATION_MODEL, MAX_INDUSTRY_EXPAND, SNIPPET_MAX_CHARS
from search import (
    search_esg_knowledge_base, compare_companies, get_company_overview,
    lookup_company, clean_context_text, INDUSTRY_CODE_MAP,
)
from prompts import (
    _PARSE_SYSTEM, _SYNTHESIS_SYSTEM, _METRIC_SEARCH,
    _JAILBREAK_PATTERNS,
    _OFF_TOPIC_REPLY, _MALICIOUS_REPLY, _TOO_LONG_REPLY,
)

log = logging.getLogger(__name__)


# ── 意圖解析 ──────────────────────────────────────────────────

class QueryIntent(BaseModel):
    intents:          list[str] = Field(default_factory=list)
    tickers:          list[str] = Field(default_factory=list)
    company_names:    list[str] = Field(default_factory=list)
    industry_code:    Optional[str] = None
    metrics:          list[str] = Field(default_factory=list)
    search_query:     str = ""
    want_source_page: bool = False


def _is_malicious_input(text: str) -> bool:
    return any(re.search(p, text) for p in _JAILBREAK_PATTERNS)


def parse_query(question: str, history: list[dict] | None = None) -> QueryIntent:
    try:
        contents = []
        for turn in (history or []):
            contents.append({"role": turn["role"], "parts": [{"text": turn["content"]}]})
        contents.append({"role": "user", "parts": [{"text": f"問句：{question}"}]})
        resp = state.gen_client.models.generate_content(
            model=GENERATION_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_PARSE_SYSTEM,
                response_mime_type="application/json",
            )
        )
        data = json.loads(resp.text or "")
        return QueryIntent.model_validate(data)
    except Exception as e:
        log.warning(f"parse_query 失敗，fallback 到 general：{e}")
        return QueryIntent(intents=["general"], search_query=question)


def _route_intents(
    parsed: QueryIntent,
    question: str,
) -> tuple[list[str], list[dict], list[str]]:
    """依 parsed.intents 執行各路由，回傳 (context_parts, all_sources, tool_calls_log)"""
    context_parts: list[str] = []
    all_sources:   list[dict] = []
    tool_calls_log: list[str] = []

    for intent in parsed.intents:
        if intent == "company_list" and parsed.industry_code:
            companies = sorted(
                [{"ticker": v["ticker"], "name": v.get("short_name") or v["company"]}
                 for v in state.company_index.values()
                 if v.get("industry") == parsed.industry_code],
                key=lambda x: x["ticker"]
            )
            industry_name = INDUSTRY_CODE_MAP.get(parsed.industry_code, parsed.industry_code)
            lines = "\n".join(f"{c['ticker']} {c['name']}" for c in companies)
            context_parts.append(f"【{industry_name}（{parsed.industry_code}）公司清單】\n{lines}")
            tool_calls_log.append(f"list_companies_by_industry(industry_code={parsed.industry_code!r})")

        elif intent == "metric_lookup" and parsed.tickers:
            for ticker in parsed.tickers:
                overview = get_company_overview(ticker)
                if overview:
                    context_parts.append(
                        f"【{ticker} ESG 概況】\n{json.dumps(overview, ensure_ascii=False)}"
                    )
                    tool_calls_log.append(f"get_company_overview(ticker={ticker!r})")

        elif intent == "comparison":
            comparison_tickers = list(parsed.tickers)
            if not comparison_tickers and parsed.industry_code:
                comparison_tickers = sorted(
                    v["ticker"] for v in state.company_index.values()
                    if v.get("industry") == parsed.industry_code
                )
            if not parsed.metrics:
                q = question
                if any(kw in q for kw in ["再生能源", "綠電", "綠能"]):
                    parsed.metrics = ["renewable_energy_pct"]
                elif any(kw in q for kw in ["用水", "耗水", "取水"]):
                    parsed.metrics = ["water_withdrawal_m3"]
                elif any(kw in q for kw in ["廢棄物", "廢棄"]):
                    parsed.metrics = ["waste_total_ton"]
                elif any(kw in q for kw in ["能源", "用電"]):
                    parsed.metrics = ["total_energy_gj"]
                else:
                    parsed.metrics = ["scope1_tco2e"]
            if len(comparison_tickers) >= 2:
                for metric in parsed.metrics:
                    result = compare_companies(comparison_tickers, metric)
                    top_n     = 5
                    has_value = [r for r in result if r.get("value") is not None]
                    no_value  = [r for r in result if r.get("value") is None]
                    if len(has_value) > top_n * 2:
                        trimmed = has_value[:top_n] + has_value[-top_n:]
                        note = f"（共 {len(has_value)} 家有資料，僅列出前 {top_n} 名與後 {top_n} 名；{len(no_value)} 家無揭露）"
                    else:
                        trimmed = has_value
                        note = f"（{len(no_value)} 家無揭露數據）" if no_value else ""
                    context_parts.append(
                        f"【指標比較：{metric}{note}】\n{json.dumps(trimmed, ensure_ascii=False)}"
                    )
                    tool_calls_log.append(
                        f"compare_companies(tickers={comparison_tickers}, metric={metric!r})"
                    )

        elif intent == "general":
            results = search_esg_knowledge_base(
                parsed.search_query,
                tickers_filter=parsed.tickers or None,
                industry_filter=parsed.industry_code or None,
            )
            all_sources.extend(results)
            snippets = "\n---\n".join(
                f"[CID:{r['chunk_id']}] {r['company']}（{r['ticker']}"
                f"{', p.' + str(r['source_pages'][0]) if r.get('source_pages') else ''}）："
                f"{clean_context_text(r['text'][:SNIPPET_MAX_CHARS])}"
                for r in results
            )
            context_parts.append(f"【語意搜尋結果】\n{snippets}")
            tool_calls_log.append(f"search_esg_knowledge_base(query={parsed.search_query!r})")

    return context_parts, all_sources, tool_calls_log


def _find_source_pages(
    parsed: QueryIntent,
    question: str,
) -> list[tuple[str, str]]:
    """當 want_source_page 時，回查各 ticker 的來源頁碼。回傳 [(context_part, tool_call_entry)]"""
    if not (parsed.want_source_page and parsed.tickers):
        return []

    _METRIC_UNIT = {
        "renewable_energy_pct": "%",
        "scope1_tco2e": "tCO2e", "scope2_tco2e": "tCO2e", "scope3_tco2e": "tCO2e",
        "total_energy_gj": "GJ", "water_withdrawal_m3": "m³", "waste_total_ton": "噸",
    }
    page_query = (
        " ".join(_METRIC_SEARCH.get(m, m) for m in parsed.metrics)
        or parsed.search_query
        or question
    )

    results: list[tuple[str, str]] = []
    for ticker in parsed.tickers:
        summary = (state.overview_index.get(ticker) or {}).get("summary_metadata") or {}
        value_hints: list[str] = []
        unit_hints:  list[str] = []
        for m in parsed.metrics:
            v = summary.get(m)
            if v is None:
                continue
            s = str(v).rstrip("0").rstrip(".") if "." in str(v) else str(v)
            value_hints.append(s)
            unit = _METRIC_UNIT.get(m, "")
            if unit:
                unit_hints.append(s + unit)
        value_query = page_query + (" " + " ".join(unit_hints or value_hints) if (unit_hints or value_hints) else "")

        raw_chunks = search_esg_knowledge_base(value_query, tickers_filter=[ticker], top_k=20, min_score=0.0)
        page_chunks = [r for r in raw_chunks if not r.get("is_overview") and r.get("source_pages")]
        log.info(f"  [source_page] {ticker}: raw={len(raw_chunks)} filtered={len(page_chunks)}")
        if not page_chunks:
            continue

        def _match_level(chunk: dict) -> int:
            text = chunk.get("text", "")
            if unit_hints and any(re.search(rf"(?<!\d){re.escape(h)}(?!\d)", text) for h in unit_hints):
                return 0
            if value_hints and any(re.search(rf"(?<!\d){re.escape(h)}(?!\d)", text) for h in value_hints):
                return 1
            return 2

        page_chunks.sort(key=lambda c: (_match_level(c), -c["score"]))
        tier0 = [c for c in page_chunks if _match_level(c) == 0]
        tier1 = [c for c in page_chunks if _match_level(c) == 1]
        tier2 = [c for c in page_chunks if _match_level(c) == 2]

        company_name = page_chunks[0]["company"]
        pages = sorted({p for r in page_chunks for p in r.get("source_pages", [])})
        lines = []
        if tier0:
            lines.append("▶ 含數值+單位的頁面（最優先）：")
            lines += [f"p.{r['source_page']}：{clean_context_text(r['text'][:150])}" for r in tier0[:3]]
        if tier1:
            lines.append("▶ 含數值的頁面：")
            lines += [f"p.{r['source_page']}：{clean_context_text(r['text'][:120])}" for r in tier1[:2]]
        if tier2:
            lines.append("其他相關頁面：")
            lines += [f"p.{r['source_page']}：{clean_context_text(r['text'][:80])}" for r in tier2[:2]]
        lines.append(f"（相關頁碼：{', '.join(str(p) for p in pages[:10])}）")

        results.append((
            f"【{company_name}（{ticker}）頁碼資訊】\n" + "\n".join(lines),
            f"find_source_pages(ticker={ticker!r})",
        ))

    return results


def agentic_rag(question: str, history: list[dict] | None = None) -> dict:
    """
    Query Understanding → Pre-fetch Context → Synthesis

    流程：
    1. parse_query：結構化解析意圖、ticker、產業、指標
    2. 依 intents 路由，各自取得 context block（pre-filter FAISS / 直查 index）
    3. 單次 Gemini 呼叫合成最終答案
    """
    t_start = time.time()
    tool_calls_log: list[str] = []
    all_sources:    list[dict] = []

    # Layer 1: 輸入過濾
    if len(question) > 500:
        return {"answer": _TOO_LONG_REPLY, "sources": [], "latency_ms": int((time.time() - t_start) * 1000)}
    if _is_malicious_input(question):
        log.warning(f"[injection] 偵測到潛在惡意注入: {question!r}")
        return {"answer": _MALICIOUS_REPLY, "sources": [], "latency_ms": int((time.time() - t_start) * 1000)}

    # Step 1: 解析意圖
    parsed = parse_query(question, history)

    # Sanitize: 防 LLM 偷塞 tickers（Q08 routing bug 修補）。只保留問句字面出現的代號
    if parsed.tickers:
        question_tickers = set(re.findall(r"\b\d{4}\b", question))
        sanitized = [t for t in parsed.tickers if t in question_tickers]
        if sanitized != parsed.tickers:
            log.info(f"  [sanitize] LLM 偷塞 tickers，移除 {set(parsed.tickers) - set(sanitized)}")
            parsed.tickers = sanitized

    log.info(f"  [parse] intents={parsed.intents} tickers={parsed.tickers} "
             f"names={parsed.company_names} industry={parsed.industry_code}")

    # Layer 2: off_topic early return
    if "off_topic" in parsed.intents:
        log.info("[off_topic] 問題與 ESG 無關，提早結束")
        return {"answer": _OFF_TOPIC_REPLY, "sources": [], "latency_ms": int((time.time() - t_start) * 1000)}

    # 公司名稱 → ticker 解析
    for name in parsed.company_names:
        matches = lookup_company(name)
        if matches:
            ticker = matches[0]["ticker"]
            if ticker not in parsed.tickers:
                parsed.tickers.append(ticker)
            tool_calls_log.append(f"lookup_company({name!r}) → {ticker}")

    # Step 2: 依 intent 執行，收集 context blocks

    # metric_lookup + 指定產業但無具體公司 → 超過上限提早回傳清單
    if "metric_lookup" in parsed.intents and parsed.industry_code and not parsed.tickers:
        industry_companies = sorted(
            [{"ticker": v["ticker"], "name": v.get("short_name") or v["company"]}
             for v in state.company_index.values()
             if v.get("industry") == parsed.industry_code],
            key=lambda x: x["ticker"]
        )
        if len(industry_companies) > MAX_INDUSTRY_EXPAND:
            display = industry_companies[:20]
            lines   = "\n".join(f"- {c['name']}（{c['ticker']}）" for c in display)
            suffix  = f"\n（共 {len(industry_companies)} 家，僅顯示前 20 家）" if len(industry_companies) > 20 else ""
            industry_name = INDUSTRY_CODE_MAP.get(parsed.industry_code, parsed.industry_code)
            return {
                "answer": (
                    f"您查詢的{industry_name}產業共有 {len(industry_companies)} 家公司，資料量較大。\n"
                    f"為提供更精準的數據，請從以下清單指定您想查詢哪幾家公司（或輸入公司代號）：\n\n"
                    f"{lines}{suffix}"
                ),
                "sources":     [],
                "tool_calls":  [f"list_companies_by_industry(industry_code={parsed.industry_code!r})"],
                "latency_ms":  int((time.time() - t_start) * 1000),
            }
        else:
            parsed.tickers = [c["ticker"] for c in industry_companies]

    context_parts, route_sources, route_calls = _route_intents(parsed, question)
    all_sources.extend(route_sources)
    tool_calls_log.extend(route_calls)

    for ctx, call in _find_source_pages(parsed, question):
        context_parts.append(ctx)
        tool_calls_log.append(call)

    # fallback：所有 intent 都沒產生 context → 全庫搜尋
    if not context_parts:
        results = search_esg_knowledge_base(question)
        all_sources.extend(results)
        snippets = "\n---\n".join(
            f"[CID:{r['chunk_id']}] {r['company']}（{r['ticker']}）：{clean_context_text(r['text'][:SNIPPET_MAX_CHARS])}"
            for r in results
        )
        context_parts.append(f"【語意搜尋結果】\n{snippets}")
        tool_calls_log.append(f"search_esg_knowledge_base(query={question!r})")

    # Step 3: 合成答案
    context_text = "\n\n".join(context_parts)
    synthesis_contents = []
    for turn in (history or []):
        synthesis_contents.append({"role": turn["role"], "parts": [{"text": turn["content"]}]})
    synthesis_contents.append({
        "role": "user",
        "parts": [{"text": f"<context>\n{context_text}\n</context>\n\n<question>{question}</question>"}]
    })

    synthesis_resp = state.gen_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=synthesis_contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            system_instruction=_SYNTHESIS_SYSTEM,
        )
    )

    try:
        synthesis_data = json.loads(synthesis_resp.text or "{}")
        answer    = synthesis_data.get("answer") or "抱歉，無法根據現有知識庫資料回答此問題。"
        cited_ids = set(
            cid.removeprefix("CID:") for cid in (synthesis_data.get("cited_chunk_ids") or [])
        )
    except (json.JSONDecodeError, AttributeError):
        answer    = synthesis_resp.text or "抱歉，無法根據現有知識庫資料回答此問題。"
        cited_ids = set()

    answer = re.sub(r"\s*\(CID:[^)]+\)", "", answer).strip()

    cited_sources = [s for s in all_sources if s.get("chunk_id") in cited_ids] if cited_ids else []

    return {
        "answer":     answer,
        "sources":    cited_sources,
        "tool_calls": tool_calls_log,
        "latency_ms": int((time.time() - t_start) * 1000),
    }
