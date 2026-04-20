# net-zero-report-RAG｜台灣上市公司 ESG 報告書 RAG 問答系統

透過 LINE 聊天機器人，以自然語言查詢 493 家台灣上市公司的 ESG 永續報告書資料。

---

![LINE Bot QR Code](LINE_BOT.png)

加入 LINE Bot，直接用中文提問：「台積電的碳排放數據？」、「比較鴻海與廣達的再生能源使用率」

| | URL |
|---|---|
| **LINE Bot** | 掃描上方 QR Code |
| **REST API** | https://esg-rag-api-4fonghyxqa-de.a.run.app |

---

## 一、發想

台灣上市公司每年發布永續報告書，但報告書動輒數百頁、格式不一，一般人難以快速查閱特定數據。本專案將 493 家公司的報告書轉化為可查詢的知識庫，讓使用者以對話方式取得 ESG 數據、跨公司比較、甚至追溯原始頁碼。

---

## 二、使用的技術

| 層次 | 技術 |
|------|------|
| 向量搜尋 | FAISS（123,865 向量，362 MB） |
| Embedding | Vertex AI `text-embedding-004`（dim=768） |
| 生成式 AI | Gemini 2.5 Flash（問答）/ Pro（圖表萃取） |
| API 框架 | FastAPI + Cloud Run |
| 對話介面 | LINE Bot SDK |
| 雲端基礎設施 | GCP（Cloud Run、GCS、Vertex AI、Artifact Registry） |
| PDF 解析 | pdfplumber + Gemini Vision（圖表）|

---

## 三、思考流程

### Pipeline 設計

報告書資料分兩軌處理：**文字 chunks**（pdfplumber 解析）與 **Vision chunks**（Gemini Pro 解析圖表與表格）。兩軌合併後透過 Vertex AI Batch Embedding 轉為向量，建立 FAISS 索引。

查詢時採三段式流程：
1. **parse_query**：Gemini 解析問句意圖（公司、指標、是否需頁碼）
2. **Pre-fetch**：依意圖路由至 FAISS 向量搜尋或精確指標查詢
3. **Synthesis**：Gemini 整合 context 生成回答

### Vibe Coding 實踐

本專案作為個人學習與作品集，全程搭配 **Claude Code** 輔助開發。開發者專注於 ESG 領域知識（報告書格式、指標定義、產業分類）與架構決策，繁瑣的系統實作由 AI 協助生成與迭代。

### 開發時程（7 天）

| 天 | 里程碑 |
|----|--------|
| Day 1 | 公司清單建立、PDF 批次下載（493 家） |
| Day 2 | PDF 解析與文字分塊 |
| Day 3 | Vertex AI Batch Embedding + FAISS 索引建立 |
| Day 4 | Gemini Vision 圖表萃取（高碳排產業優先） |
| Day 5-6 | 公司摘要生成（Step 07，三段式萃取） |
| Day 7 | FastAPI 部署 Cloud Run + LINE Bot 上線 |

---

## 四、主要功能

- **自然語言問答**：查詢單一公司的 ESG 指標（碳排、能源、用水、廢棄物）
- **跨公司比較**：「比較台積電與聯發科的 Scope 1 排放」
- **產業篩選**：「列出半導體業碳排前五名」
- **頁碼查詢**：「台積電再生能源數據在報告第幾頁？」
- **多輪對話**：LINE Bot 支援上下文記憶（30 分鐘內）

---

## 五、專案結構

```
esg-pipeline/
├── api/
│   ├── state.py          # 全域狀態與常數（FAISS、Embedding model）
│   ├── search.py         # 向量搜尋、公司比較、資料查詢
│   ├── rag.py            # Query Understanding、RAG 生成
│   ├── main.py           # FastAPI 路由
│   ├── line_bot.py       # LINE Webhook + 多輪對話管理
│   ├── Dockerfile        # 完整建置（含 FAISS index）
│   ├── Dockerfile.fast   # 快速部署（從 :latest 複製 FAISS）
│   └── requirements.txt
├── scripts/
│   ├── day1_collect/     # 01 公司清單、02 PDF 下載
│   ├── day2_parse/       # 03 PDF 解析與分塊
│   ├── day3_embed/       # 04 Embedding batch、07 摘要生成
│   ├── day4_vision/      # 05 Gemini Vision 圖表萃取
│   ├── day6_faiss/       # 08 FAISS 索引建立
│   └── day7_deploy/      # 09 Cloud Run 部署腳本
├── company_list_2024.json
└── pipeline_rebuild_guide.md
```

---

## 六、部署與使用

### Pipeline 步驟

| Step | 腳本 | 說明 |
|------|------|------|
| 01 | `day1_collect/01_build_company_list.py` | 建立公司清單（ticker、產業、優先級） |
| 02 | `day1_collect/02_download_pdfs.py` | 批次下載 PDF 至 GCS |
| 03 | `day2_parse/03_pdf_parse_and_chunk.py` | PDF 解析、文字分塊 |
| 04 | `day3_embed/04_submit_embedding_batch.py` | Vertex AI Batch Embedding |
| 05 | `day4_vision/05_vision_chart_extract.py` | Gemini Vision 圖表萃取 |
| 07 | `day3_embed/07_generate_summaries.py` | 公司 ESG 摘要生成 |
| 08 | `day6_faiss/08_build_faiss_index.py` | FAISS 索引建立 |
| 09 | `day7_deploy/09_build_and_deploy.sh` | Docker build + Cloud Run 部署 |

### 本地啟動 API

```bash
cd api
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

export GEMINI_API_KEY="your_key"
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### API 端點

| 端點 | 說明 |
|------|------|
| `GET /health` | 健康檢查 |
| `GET /companies?industry=24` | 公司清單（可依產業代碼篩選） |
| `POST /query` | ESG 問答 `{"question": "..."}` |
| `POST /compare` | 跨公司指標比較 `{"tickers": [...], "metric": "..."}` |
| `POST /webhook` | LINE Bot Webhook |

**可比較指標**：`scope1_tco2e` / `scope2_tco2e` / `scope3_tco2e` / `renewable_energy_pct` / `total_energy_gj` / `water_withdrawal_m3` / `waste_total_ton`
