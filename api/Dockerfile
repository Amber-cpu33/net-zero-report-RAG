# ── ESG RAG API Dockerfile ──────────────────────────────────────
# 多階段建置：減少最終映像大小
# ──────────────────────────────────────────────────────────────

# Stage 1: 依賴安裝（使用 slim 底圖）
FROM python:3.11-slim AS builder

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 複製並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/app/packages -r requirements.txt

# Stage 2: 最終映像
FROM python:3.11-slim

WORKDIR /app

# 安裝執行期系統依賴（FAISS 需要 OpenMP）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴
COPY --from=builder /app/packages /app/packages
ENV PYTHONPATH="/app/packages:${PYTHONPATH:-}"

# 複製應用程式碼
COPY main.py state.py search.py rag.py line_bot.py ./
COPY faiss_index/ ./faiss_index/

# Cloud Run 使用 PORT 環境變數
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# 非 root 使用者（安全最佳實踐）
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 啟動指令（使用 uvicorn，適合 Cloud Run）
CMD exec python -m uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 30
