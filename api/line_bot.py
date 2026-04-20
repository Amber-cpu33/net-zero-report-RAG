import os
import re
import logging

from fastapi import FastAPI, HTTPException, Request

try:
    from linebot.v3 import WebhookHandler
    from linebot.v3.messaging import (
        ApiClient, Configuration, MessagingApi,
        ReplyMessageRequest, TextMessage as LineTextMessage,
    )
    from linebot.v3.webhooks import MessageEvent, TextMessageContent
    from linebot.v3.exceptions import InvalidSignatureError
    LINE_SDK_AVAILABLE = True
except ImportError:
    LINE_SDK_AVAILABLE = False

try:
    import time as _time
    from cachetools import TTLCache
    _session_cache = TTLCache(maxsize=1000, ttl=1800, timer=_time.time)
    _rate_cache = TTLCache(maxsize=1000, ttl=30, timer=_time.time)  # 每 30 秒重置計數（wall clock）
    _seen_events = TTLCache(maxsize=5000, ttl=60, timer=_time.time)  # LINE retry 去重
except ImportError:
    _session_cache = {}
    _rate_cache = {}
    _seen_events = {}

RATE_LIMIT_PER_MINUTE = 5

log = logging.getLogger(__name__)

LINE_CHANNEL_SECRET       = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

MAX_HISTORY = 3  # 保留最近 3 輪（6 則訊息）

WELCOME_MESSAGE = (
    "👋 歡迎使用台灣企業 ESG 知識庫！\n\n"
    "我可以回答：\n"
    "• 特定公司碳排放、再生能源、用水等數據\n"
    "  例：台積電的 Scope 1 排放量？\n"
    "• 跨公司比較\n"
    "  例：半導體業哪家再生能源比例最高？\n"
    "• ESG 政策與目標\n"
    "  例：鴻海的淨零目標是哪年？\n\n"
    "📊 資料來源：493 家台灣上市公司 2024 年永續報告書\n\n"
    "📋 涵蓋範圍：\n"
    "• Tier 1（50 家）：台灣 50 成分股（0050 ETF）\n"
    "• Tier 2（443 家）：高碳排產業＋高電耗科技業\n"
    "  包含：水泥、鋼鐵、塑化、石化、航運、半導體、電子等\n\n"
    "輸入問題即可開始查詢！\n\n"
    "⚠️ 本系統回答僅供參考，數據請以原公司報告為準：\n"
    "https://cgc.twse.com.tw/front/chPage"
)

TRIGGER_WORDS = {"你好", "hi", "hello", "開始", "help", "說明", "使用說明"}

_UNSUPPORTED_PATTERNS = [
    r"0050|ETF|指數.*成分|成分股",
    r"(?i)tier\s*[1-4]|第[一二三四]級",
    r"台灣\s*50\s*成分",
]

_UNSUPPORTED_REPLY = (
    "目前系統以「單一公司」或「特定產業」為查詢單位，暫不支援「ETF 成分股」或「特定 Tier」的群組查詢。\n\n"
    "請直接輸入您想查詢的「公司名稱」或「股票代號」（例如：台積電、2330），"
    "或指定「特定產業」（例如：列出半導體業的碳排）。"
)


def _is_unsupported_query(text: str) -> bool:
    for pattern in _UNSUPPORTED_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def _is_rate_limited(user_id: str) -> bool:
    count = _rate_cache.get(user_id, 0)
    if count >= RATE_LIMIT_PER_MINUTE:
        return True
    _rate_cache[user_id] = count + 1
    return False


def _get_history(user_id: str) -> list[dict]:
    return _session_cache.get(user_id, [])


def _append_history(user_id: str, question: str, answer: str):
    history = _session_cache.get(user_id, [])
    history.append({"role": "user", "content": question})
    history.append({"role": "model", "content": answer})
    if len(history) > MAX_HISTORY * 2:
        history = history[-MAX_HISTORY * 2:]
    _session_cache[user_id] = history


def register_line_bot(app: FastAPI, rag_fn):
    """掛載 LINE Webhook route 到 FastAPI app。rag_fn(question, history) -> dict"""
    if not (LINE_SDK_AVAILABLE and LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN):
        log.info("LINE Bot 未設定（LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN 未提供）")
        return

    handler = WebhookHandler(LINE_CHANNEL_SECRET)
    api_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

    @handler.add(MessageEvent, message=TextMessageContent)
    def _handle_message(event):
        event_id = event.message.id
        if event_id in _seen_events:
            return
        _seen_events[event_id] = True

        user_id = event.source.user_id
        user_text = event.message.text.strip()

        if user_text.lower() in TRIGGER_WORDS:
            answer = WELCOME_MESSAGE
        elif _is_rate_limited(user_id):
            answer = f"您的發問頻率過高，請稍候再試（30 秒內上限 {RATE_LIMIT_PER_MINUTE} 次）。"
        elif _is_unsupported_query(user_text):
            answer = _UNSUPPORTED_REPLY
        else:
            history = _get_history(user_id)
            try:
                result = rag_fn(user_text, history)
                answer = result["answer"]
                _append_history(user_id, user_text, answer)
            except Exception as e:
                log.error(f"rag_fn 執行失敗（user_id={user_id}）：{e}")
                answer = "抱歉，系統暫時忙碌，請稍後再試。"

        with ApiClient(api_config) as api_client:
            MessagingApi(api_client).reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[LineTextMessage(text=answer)]
                )
            )

    @app.post("/webhook")
    async def line_webhook(request: Request):
        signature = request.headers.get("X-Line-Signature", "")
        body = await request.body()
        try:
            handler.handle(body.decode(), signature)
        except InvalidSignatureError:
            raise HTTPException(status_code=400, detail="Invalid signature")
        return "OK"

    log.info("LINE Bot webhook 已啟用（/webhook）")
