# RAGAS 評估紀錄

## 問題集

`evaluation/RAGAS_questions_v2.yaml` — 共 32 題（Q12/Q18 excluded、Q24 excluded = 有效 29 題）

| Part | 題號 | 測試面向 |
|------|------|---------|
| 1 | Q01-Q05 | Faithfulness 壓力測試（防幻覺）|
| 2 | Q06-Q10 | Agentic Routing（意圖解析 + 攔截器）|
| 3 | Q11-Q15 | Context Precision（技術名詞 / 資料品質）|
| 4 | Q16-Q20 | Answer Relevancy（複雜意圖）|
| 5 | Q21-Q25 | 系統行為（頁碼 / off_topic / 注入 / multi-turn / 長度）|
| 6 | Q26-Q27 | 補充覆蓋（compare / metric_lookup）|
| 7 | Q28-Q32 | 數值正確性（CSV ground truth 精確比對）|

### 特殊題狀態

| 題號 | 狀態 | 原因 |
|------|------|------|
| Q12、Q18 | excluded（原始） | 重複或設計問題 |
| Q24 | excluded（2026-05-09） | history 記憶僅 LINE Bot 實作，/query endpoint 不支援 |
| Q08 | 測試方向改版（2026-05-09 × 2） | 第一版改無界查詢測截斷；第二版確認 MAX_INDUSTRY_EXPAND 未觸發但 token 防護已達，改測 industry routing correctness，expect_abstain=false |

### YAML 欄位說明

```yaml
gt_csv_ticker: "2330"        # 有此欄位時，run_ragas.py 從答案萃取數字對比 CSV
gt_csv_metric: scope1_tco2e  # scope1_tco2e / scope2_tco2e / scope3_tco2e
excluded: true               # 跳過此題
```

---

## 評分指標

| 指標 | 說明 |
|------|------|
| `faithfulness` | 答案可從 context 推得（0-1，Gemini judge）|
| `correctness` | 答案與 YAML ground_truth 語意符合（0/0.3/0.5/0.7/1.0，Gemini judge）|
| `csv_correctness` | 答案數值與政府 CSV 差異 ≤5% → 1.0，超出 → 0.0（無 LLM，純數值比對）|
| `abstain_pass` | expect_abstain 題的拒答正確率（regex 比對，詳見 run_ragas.py `ABSTAIN_PATTERN`）|

### abstain_pass 題目（新版，共 6 題）

Q02、Q03、Q08、Q22、Q23、Q25（Q01 改為 expect_abstain=false；Q24 excluded）

---

## Ground Truth 來源

- **YAML ground_truth**：手動驗證（2026-04-30 初版；2026-05-09 修正 Q01/Q07/Q30）
- **政府 CSV**：`evaluation/ground_truth_answer/ESG數位平台彙總資料-溫室氣體排放.csv`
  - 來源：金管會 ESG 數位平台，2024 年度，1,041 家上市櫃公司
  - 涵蓋：Scope 1 / Scope 2 / Scope 3（公噸 CO₂e）
  - 覆蓋率：我們索引的 508 家公司，507/508 有對應資料（僅 6272 缺失）

### Ground Truth 修正紀錄（2026-05-09）

| 題號 | 修正內容 | 原因 |
|------|---------|------|
| Q01 | expect_abstain false；GT 改為「3個月 / 2年」 | faiss_v3 重構後 chunk `2330_2024_c0353`（p.132）完整保留此數據，Run 5 retrieval miss，Run 7 答案正確但 faithfulness judge 誤評 0.0 |
| Q07 | GT 加入中鋼 2030 減碳 25% 中期目標 | faiss_v3 BM25 找到 chunk `2002_2024_c0108`，確認中鋼有此目標。Run 5/4 答「未找到」為 retrieval miss；Run 7 答案正確 |
| Q08 | 問題改為無界查詢；expect_abstain 恢復 true | 原「前三名」為有界查詢不應攔截；改成「哪些公司再生能源占比較高」才真正觸發 MAX_INDUSTRY_EXPAND 攔截 |
| Q22 | `ABSTAIN_PATTERN` 加入 `請針對.*ESG.*提問` | 系統拒答措辭未在 pattern 裡，導致誤判失敗 |
| Q24 | excluded | history 僅 LINE Bot 實作 |
| Q28 | GT 改為 `1,825,872 tCO2e` | 原 GT 為「依政府公開資料」，judge 無法比對 → 改為具體數值 |
| Q29 | GT 改為 `2,272,277 tCO2e` | 原 GT 為「依政府公開資料」，judge 評 0.0 → 改為具體數值 |
| Q30 | 移除 gt_csv_metric；GT 改為 17,587,087 tCO2e | CSV 為合併邊界（27,213,835），報告書為母公司邊界，永遠無法比對通過 |
| Q31 | GT 改為 `10,957,397 tCO2e` | 原 GT 為「依政府公開資料」，judge 評 0.0 → 改為具體數值 |
| Q32 | GT 改為 `5,518.2602 tCO2e` | 原 GT 為「依政府公開資料」，judge 評 0.0 → 改為具體數值 |

---

## 執行方式

```powershell
# 對 Cloud Run（預設）
$env:GEMINI_API_KEY = "..."
cd D:\Data\Claude\net-zero-report\esg-pipeline
python evaluation/run_ragas.py

# 對本地 API
$env:API_BASE = "http://localhost:8080"
$env:GEMINI_API_KEY = "..."
python evaluation/run_ragas.py
```

結果輸出：`evaluation/RAGAS_log/baseline_<ts>.csv` + `.jsonl`

---

## 歷史 Baseline

> Run 4\* / Run 5\* 為套用新版 YAML 的**估算值**（correctness 未重跑 LLM judge）
> **Run 8–10 為 RAGAS 框架校準**（GT 修正、YAML 調整），API 版本不變（Cloud Run v3 BM25Okapi），數值變化反映評估框架改善，非系統改善。

| Run | 時間 | 檔案 | API | faith | correctness | csv_correctness | abstain_pass | 備註 |
|-----|------|------|-----|-------|-------------|-----------------|--------------|------|
| Run 1 | 2026-04-30 18:56 | baseline_20260430_185655 | Cloud Run v1 | 0.49 | 0.694 | — | **100%** | YAML v1；Q08 expect_abstain=False |
| Run 2 | 2026-04-30 21:54 | baseline_20260430_215421 | Cloud Run v1 | 0.70 | 0.689 | — | 57% | Q08 改 true；Q23 洩漏系統提示；Q25 無長度攔截 |
| Run 3 | 2026-05-02 16:27 | baseline_20260502_162701 | Cloud Run v2 | 0.55 | 0.478 | — | 57% | text-only FAISS 重建；Q05/Q06/Q27 退步 |
| Run 4 | 2026-05-03 14:21 | baseline_20260503_142137 | Cloud Run v2（無 BM25） | 0.50 | 0.717 | 66.7%（4/6） | 57% | v2 YAML；Q28-Q32 新增 |
| Run 4\* | — | — | 舊答案套新版 YAML | ≈0.51 | ≈0.642 | **100%（4/4）** | 80%（4/5） | correctness 人工估算（n=24）；主要扣分：Q07 miss 2030→0.0、Q19 未找到→0.0、Q20 未找到→0.3 |
| Run 5 | 2026-05-03 14:53 | baseline_20260503_145315 | localhost:8080（BM25） | 0.48 | **0.752** | 66.7%（4/6） | 71% | BM25 + SNIPPET_MAX_CHARS=600 |
| Run 5\* | — | — | 舊答案套新版 YAML | ≈0.49 | ≈0.721 | **100%（4/4）** | 80%（4/5） | correctness 人工估算（n=24）；主要扣分：Q01/Q07 未找到→0.0；BM25 使 Q05/Q09/Q20 明顯進步 |
| Run 6 | 2026-05-05 13:47 | baseline_20260505_134254 | Cloud Run v3（BM25Okapi） | 0.55 | 0.645 | — | 57.1% | all_sources fallback 修改；3 題 504 timeout；Q24 CID 外洩 |
| Run 7 | 2026-05-05 14:32 | baseline_20260505_143232 | Cloud Run v3（BM25Okapi） | 0.496 | 0.722 | 66.7%（4/6） | 57.1% | REQUEST_TIMEOUT=120；Q24 CID 修補；Q07 實際正確（GT 有誤） |
| Run 8 | 2026-05-09 14:33 | baseline_20260509_143351 | Cloud Run v3（BM25Okapi） | 0.409 | **0.796** | **100%（4/4）** | 80%（4/5） | RAGAS v2 YAML 定版首跑；abstain：Q25 ❌（長度攔截非確定性，Run 8/9 曾通過）；correctness 為各 Run 最高 |

