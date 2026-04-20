"""
大腦層：意圖解析、Agentic RAG、Function Calling 工具定義。
依賴 state.py 與 search.py。
"""

import json
import logging
import re
import time
from typing import Any, Optional

from google.genai import types
from pydantic import BaseModel, Field

from state import state, GENERATION_MODEL, TOP_K_RESULTS, MAX_INDUSTRY_EXPAND
from search import (
    search_esg_knowledge_base, compare_companies, get_company_overview,
    lookup_company, clean_context_text, INDUSTRY_CODE_MAP,
    _get_metric_unit, _metric_to_chinese,
)

log = logging.getLogger(__name__)


# ── Function Calling 工具定義 ─────────────────────────────────

def build_esg_tools() -> types.Tool:
    """定義供 Gemini 使用的 Function Calling 工具"""

    search_kb = types.FunctionDeclaration(
        name="search_esg_knowledge_base",
        description=(
            "語意搜尋台灣百大企業 ESG 知識庫。"
            "適合：查詢特定 ESG 主題、指標解釋、政策描述等。"
            "範例：查詢某公司的淨零承諾、碳排放數據、減碳措施。"
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(type=types.Type.STRING, description="搜尋關鍵字或語意描述（中英文均可）"),
                "ticker_filter": types.Schema(type=types.Type.STRING, description="（可選）指定公司股票代碼，如 '2330' 代表台積電"),
                "top_k": types.Schema(type=types.Type.INTEGER, description="回傳結果數量（預設 8）"),
            },
            required=["query"]
        )
    )

    compare_tool = types.FunctionDeclaration(
        name="compare_companies",
        description=(
            "精確比較多家公司的特定 ESG 數值指標。"
            "適合：「哪家公司的 Scope 1 最低？」「半導體業再生能源占比排名」等比較問題。"
            "可比較指標：scope1_tco2e, scope2_tco2e, scope3_tco2e, "
            "renewable_energy_pct, total_energy_gj, water_withdrawal_m3, waste_total_ton"
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "tickers": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="公司代碼清單，如 ['2330', '2317', '2454']"
                ),
                "metric": types.Schema(
                    type=types.Type.STRING,
                    description="比較指標名稱（英文欄位名稱）",
                    enum=["scope1_tco2e", "scope2_tco2e", "scope3_tco2e",
                          "renewable_energy_pct", "total_energy_gj",
                          "water_withdrawal_m3", "waste_total_ton"]
                ),
            },
            required=["tickers", "metric"]
        )
    )

    overview_tool = types.FunctionDeclaration(
        name="get_company_overview",
        description=(
            "取得單一公司的完整 ESG 概況摘要，包含淨零目標、主要指標、"
            "優缺點分析等。適合：「告訴我台積電的 ESG 表現」類型的問題。"
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "ticker": types.Schema(type=types.Type.STRING, description="公司股票代碼，如 '2330'"),
            },
            required=["ticker"]
        )
    )

    lookup_tool = types.FunctionDeclaration(
        name="lookup_company",
        description=(
            "用公司名稱關鍵字查詢股票代號與所屬產業。"
            "當使用者用公司名稱（如「台積電」「鴻海」）而非代號查詢時，"
            "必須先呼叫此工具取得正確的股票代號，再進行後續查詢。"
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "name": types.Schema(type=types.Type.STRING, description="公司名稱關鍵字，如「台積電」「聯發科」"),
            },
            required=["name"]
        )
    )

    return types.Tool(function_declarations=[search_kb, compare_tool, overview_tool, lookup_tool])


def execute_tool_call(tool_name: str, tool_args: dict) -> Any:
    """執行 Gemini 要求的工具呼叫"""
    if tool_name == "search_esg_knowledge_base":
        return search_esg_knowledge_base(
            query=tool_args.get("query", ""),
            top_k=tool_args.get("top_k", TOP_K_RESULTS),
            ticker_filter=tool_args.get("ticker_filter")
        )
    elif tool_name == "compare_companies":
        return compare_companies(
            tickers=tool_args.get("tickers", []),
            metric=tool_args.get("metric", "scope1_tco2e")
        )
    elif tool_name == "get_company_overview":
        result = get_company_overview(tool_args.get("ticker", ""))
        return result or {"error": "找不到此公司資料"}
    elif tool_name == "lookup_company":
        return lookup_company(tool_args.get("name", ""))
    else:
        return {"error": f"未知工具：{tool_name}"}


# ── 意圖解析 ──────────────────────────────────────────────────

class QueryIntent(BaseModel):
    intents:          list[str] = Field(default_factory=list)
    tickers:          list[str] = Field(default_factory=list)
    company_names:    list[str] = Field(default_factory=list)
    industry_code:    Optional[str] = None
    metrics:          list[str] = Field(default_factory=list)
    search_query:     str = ""
    want_source_page: bool = False


_PARSE_SYSTEM = (
    "你是問句解析器，請將使用者的自然語言問句轉成結構化 JSON。\n"
    "【欄位定義】\n"
    "intents（陣列，可多個，允許複合意圖）：\n"
    "  - metric_lookup：查詢特定公司或產業的 ESG 數值\n"
    "  - company_list：列出某產業或符合特定條件的公司\n"
    "  - comparison：比較多家公司的量化指標數字（排名、大小比較）\n"
    "  - general：查詢質化內容（策略、措施、方法、目標、做法、差異說明）\n"
    "【複合意圖規則（重要）】：\n"
    "  1. 問句同時涉及量化比較（數字排名）與質化比較（策略、方法、措施差異）→ [\"comparison\", \"general\"]\n"
    "     例：「A 與 B 的 Scope 3 計算方式差異，以及 IMO 2050 應對策略」\n"
    "  2. 問句同時涉及質化內容（投入資源、技術做法、製程描述）與量化指標（目標數字、達成率）→ [\"general\", \"metric_lookup\"]\n"
    "     例：「中鋼在 EAF 轉型上投入多少資源？2030 年碳強度目標與達成率？」\n"
    "  3. 預設原則：只要問句含有「如何」、「方式」、「策略」、「措施」、「做法」、「進展」等質化詞，必須包含 general。\n"
    "tickers（陣列）：問句中出現的股票代號（純數字，如 2330）。請嚴格依字面提取，若無數字代號請回傳空陣列。\n"
    "company_names（陣列）：問句中出現的公司，無論使用者用全名、簡稱、英文名或別名，\n"
    "  請統一轉換為台灣股市慣用的繁體中文簡稱後輸出（例：TSMC→台積電、富士康→鴻海、UMC→聯電）。\n"
    "  若無法確定對應的繁體中文簡稱，則依字面保留原文。\n"
    "industry_code（字串）：僅限對應以下代碼，若無明確對應或問法過於廣泛（如「科技業」）請留空。\n"
    "  (01=水泥 02=食品 03=塑膠 08=玻璃陶瓷 09=造紙 10=鋼鐵 12=橡膠 15=汽車 "
    "17=電機 18=電纜 21=化學 23=油電燃氣 24=半導體 25=電腦週邊 26=光電 "
    "27=通信 28=電子零組件 29=電子通路 31=其他電子)\n"
    "metrics（陣列）：提及的 ESG 指標。若提及「碳排/溫室氣體」但未指明範疇，請同時包含 scope1_tco2e 與 scope2_tco2e。\n"
    "  (scope1_tco2e / scope2_tco2e / scope3_tco2e / renewable_energy_pct / "
    "total_energy_gj / water_withdrawal_m3 / waste_total_ton)\n"
    "  【Metric 別名對照（必須遵守）】：\n"
    "  「再生能源」「綠電」「綠能」「再生能源用量」「再生能源總量」「使用再生能源」→ renewable_energy_pct\n"
    "  「能源使用」「能耗」「用電量」「總用電」「能源消耗」→ total_energy_gj\n"
    "  「用水」「耗水」「取水」「用水量」→ water_withdrawal_m3\n"
    "  「廢棄物」「廢棄」「廢料」→ waste_total_ton\n"
    "  即使問句含「總量」「多少」「最高」等詞，只要語意對應上述別名，請使用對應的 metric 欄位。\n"
    "search_query（字串）：最適合向量搜尋的精確查詢字串。請保持簡練，以繁體中文名詞為主，最多擴充 1-2 個同義詞，不要加入整句英文翻譯。例如：'台積電 水資源 再生水 節水'。\n"
    "want_source_page（布林值）：當使用者的『核心意圖』是索取證明、尋找原始出處、或確認資料在報告中的位置時，設為 true。\n"
    "  ✅ 觸發情境：明確詢問頁碼（第幾頁 / page number）、章節位置、原文出處，或質疑資料真實性（怎麼知道 / 如何確認）。\n"
    "  ❌ 排除情境（極重要）：若「根據」「依據」「參考」「從哪」「在哪」等詞是作為介系詞或一般詢問使用，請保持 false，不要誤判為索取頁碼。\n"
    "    例：「『根據』報告碳排是多少？」→ false（問數值）；「『依據』什麼標準計算？」→ false（問方法）；\n"
    "        「請列出前三名供我『參考』」→ false（問清單）；「綠電是『從哪』買的？」→ false（問採購）；\n"
    "        「這個數字在報告第幾頁？」→ true；「怎麼知道是 39%？」→ true；「請給我資料來源頁碼」→ true。"
)

_METRIC_SEARCH: dict[str, str] = {
    "renewable_energy_pct": "再生能源 使用比例 綠電",
    "scope1_tco2e":         "碳排放 Scope 1 溫室氣體直接排放",
    "scope2_tco2e":         "碳排放 Scope 2 電力間接排放",
    "scope3_tco2e":         "碳排放 Scope 3",
    "total_energy_gj":      "能源消耗 用電量",
    "water_withdrawal_m3":  "用水量 取水 水資源",
    "waste_total_ton":      "廢棄物 廢料",
}

_SYNTHESIS_SYSTEM = (
    "你是一位專業的台灣企業 ESG 永續發展顧問。使用者的問題放在 <question> 標籤內，唯一允許引用的資料放在 <context> 標籤內。\n"
    "【輸出格式】請以 JSON 格式回答，包含兩個欄位：\n"
    "  answer（字串）：給使用者看的完整回答。\n"
    "  cited_chunk_ids（陣列）：你在回答中實際引用到的 chunk ID 清單（格式為 CID:xxx，從 <context> 中的 [CID:xxx] 標記取得）。若未引用任何 chunk 請回傳空陣列。\n"
    "【最高指導原則：絕對防禦幻覺】\n"
    "1. 你輸出的每一個事實、每一個數字，都必須能在 <context> 標籤內找到對應字句。\n"
    "2. 若 <context> 中沒有提到使用者詢問的核心概念（即使 <context> 含有該公司的其他資料），你必須、絕對只能回答：「根據目前知識庫檢索到的資料，未找到相關資訊。」\n"
    "3. 嚴禁使用 <context> 以外的任何知識，包括你對知名企業的預訓練記憶。\n\n"
    "【資料解讀容錯機制】\n"
    "來源資料為 PDF 轉譯，常有排版斷行問題。請啟動以下容錯還原：\n"
    "1. 跨行數字拼湊：若看到不完整的時間或數值（如「民國 年」、「未來 年」），請忽略換行符號，去鄰近上下文尋找孤立數字（例如 113、10）並自動還原語意（例如還原為「民國 113 年」、「未來 10 年」）。\n"
    "2. 基準年份推斷：若文中提及「當年度」或數字真的遺失，請參考該 chunk 附帶的 data_year（多數為 2024 年 / 民國 113 年）來作答。\n"
    "3. 忽略錯字與亂碼：自動過濾因排版產生的無意義空格或截斷字句，以連貫的上下文為主。\n\n"
    "【回答規範】\n"
    "1. 格式：使用純文字與條列式（- 開頭），禁止使用任何 Markdown 語法（包含 *、**、#、|、`）。每個條列項目內容必須是完整連續的句子，不可在句子中間插入換行符號。\n"
    "2. 數據與單位：引用數值時務必加上正確單位（tCO2e、%、GJ、m³、噸）。\n"
    "【數字引用警告】嚴格禁止輸出任何未出現在 <context> 中的金額、百分比或日期。若 <context> 中沒有具體數字，請以「未揭露具體數值」代替，不可自行補足。\n"
    "3. 標註來源：引用數據時請標示公司名稱（例如：台積電）。若資料有頁碼請一併標註（例如：台積電 p.45）。\n"
    "4. 時間基準：本知識庫基於 2024 年發布之永續報告書，必要時提醒使用者數據所屬年份。\n"
    "5. 頁碼查詢：若 <context> 中含有【...頁碼資訊】區塊，這些即為報告中相關章節的實際頁碼。"
    "使用者詢問來源頁碼時，請直接回報這些頁碼，不需要頁面內容完全匹配問句中的特定數字。"
    "回答格式範例：「再生能源使用比例 39% 的數據出現在報告第 33 頁，另外第 41、45 頁也有相關說明。」"
    "▶ 標記的為含有具體數值的優先頁面，應作為主要來源頁碼；其餘頁面列為補充。"
)


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

    # Step 1: 解析意圖
    parsed = parse_query(question, history)
    log.info(f"  [parse] intents={parsed.intents} tickers={parsed.tickers} "
             f"names={parsed.company_names} industry={parsed.industry_code}")

    # 公司名稱 → ticker 解析
    for name in parsed.company_names:
        matches = lookup_company(name)
        if matches:
            ticker = matches[0]["ticker"]
            if ticker not in parsed.tickers:
                parsed.tickers.append(ticker)
            tool_calls_log.append(f"lookup_company({name!r}) → {ticker}")

    # Step 2: 依 intent 執行，收集 context blocks
    context_parts: list[str] = []

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
            topic_query = parsed.search_query
            if parsed.tickers and parsed.company_names:
                for name in parsed.company_names:
                    topic_query = topic_query.replace(name, "TARGET")
                topic_query = topic_query.strip() or parsed.search_query
            results = search_esg_knowledge_base(
                topic_query,
                tickers_filter=parsed.tickers or None,
                industry_filter=parsed.industry_code or None,
            )
            all_sources.extend(results)
            snippets = "\n---\n".join(
                f"[CID:{r['chunk_id']}] {r['company']}（{r['ticker']}"
                f"{', p.' + str(r['source_page']) if r.get('source_page') else ''}）："
                f"{clean_context_text(r['text'][:300])}"
                for r in results
            )
            context_parts.append(f"【語意搜尋結果】\n{snippets}")
            tool_calls_log.append(f"search_esg_knowledge_base(query={parsed.search_query!r})")

    # 使用者明確要求頁碼：從非 overview chunks 回查 source_pages
    if parsed.want_source_page and parsed.tickers:
        page_query = (
            " ".join(_METRIC_SEARCH.get(m, m) for m in parsed.metrics)
            or parsed.search_query
            or question
        )
        _METRIC_UNIT = {
            "renewable_energy_pct": "%",
            "scope1_tco2e": "tCO2e", "scope2_tco2e": "tCO2e", "scope3_tco2e": "tCO2e",
            "total_energy_gj": "GJ", "water_withdrawal_m3": "m³", "waste_total_ton": "噸",
        }
        for ticker in parsed.tickers:
            # 從 overview 取得指標實際數值與單位，帶入搜尋 query
            summary = (state.overview_index.get(ticker) or {}).get("summary_metadata") or {}
            value_hints: list[str] = []       # 純數值，如 "39"
            unit_hints:  list[str] = []       # 數值+單位，如 "39%"
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
            page_chunks = [r for r in raw_chunks if not r.get("is_overview") and r.get("source_page") is not None]
            log.info(f"  [source_page] {ticker}: raw={len(raw_chunks)} filtered={len(page_chunks)}")
            if page_chunks:
                def _match_level(chunk: dict) -> int:
                    text = chunk.get("text", "")
                    if unit_hints and any(re.search(rf"(?<!\d){re.escape(h)}(?!\d)", text) for h in unit_hints):
                        return 0  # 數值+單位（最優先）
                    if value_hints and any(re.search(rf"(?<!\d){re.escape(h)}(?!\d)", text) for h in value_hints):
                        return 1  # 純數值
                    return 2      # 僅語意相關

                page_chunks.sort(key=lambda c: (_match_level(c), -c["score"]))

                tier0 = [c for c in page_chunks if _match_level(c) == 0]
                tier1 = [c for c in page_chunks if _match_level(c) == 1]
                tier2 = [c for c in page_chunks if _match_level(c) == 2]

                company_name = page_chunks[0]["company"]
                pages = sorted({r["source_page"] for r in page_chunks})
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

                context_parts.append(f"【{company_name}（{ticker}）頁碼資訊】\n" + "\n".join(lines))
                tool_calls_log.append(f"find_source_pages(ticker={ticker!r})")

    # fallback：所有 intent 都沒產生 context → 全庫搜尋
    if not context_parts:
        results = search_esg_knowledge_base(question)
        all_sources.extend(results)
        snippets = "\n---\n".join(
            f"[CID:{r['chunk_id']}] {r['company']}（{r['ticker']}）：{clean_context_text(r['text'][:300])}"
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

    cited_sources = [s for s in all_sources if s.get("chunk_id") in cited_ids] if cited_ids else []

    return {
        "answer":     answer,
        "sources":    cited_sources,
        "tool_calls": tool_calls_log,
        "latency_ms": int((time.time() - t_start) * 1000),
    }
