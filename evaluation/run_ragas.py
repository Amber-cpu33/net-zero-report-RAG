"""
RAGAS-style 評分腳本（自寫 Gemini Flash judge 版）

骨架參考：evaluation/RAGAS/define_evaluation_metrics.ipynb
  - 把 LangChain + GPT-4o → google-genai + Gemini 2.5 Flash
  - 把 OpenAI structured output → Gemini response_schema
新增：abstain_accuracy（自訂指標，配合 v2 yaml 的 expect_abstain 欄位）

使用：
  set GEMINI_API_KEY=你的金鑰
  python evaluation/run_ragas.py
  # 結果：evaluation/baseline_<ts>.csv + evaluation/baseline_<ts>.jsonl
"""
import os, sys, json, time, csv, re, math, urllib.request, urllib.error
import concurrent.futures as cf
from datetime import datetime
from pathlib import Path

import yaml
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ── 設定 ──────────────────────────────────────────────────
API_BASE       = os.getenv("API_BASE", "https://esg-rag-api-4fonghyxqa-de.a.run.app")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
JUDGE_MODEL    = "gemini-2.5-flash"
YAML_PATH      = "evaluation/RAGAS_questions_v2.yaml"
GT_CSV_PATH    = Path("evaluation/ground_truth_answer/ESG數位平台彙總資料-溫室氣體排放.csv")
OUT_DIR        = Path("evaluation/RAGAS_log")
N_PARALLEL     = 4              # 平行度（API + judge）
CSV_TOLERANCE  = 0.05           # ±5%

if not GEMINI_API_KEY:
    sys.exit("❌ 請先設 GEMINI_API_KEY 環境變數")

client = genai.Client(api_key=GEMINI_API_KEY)


# ── CSV ground truth ──────────────────────────────────────
def _load_gt_csv() -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    if not GT_CSV_PATH.exists():
        return result
    with open(GT_CSV_PATH, encoding="utf-8-sig") as f:
        lines = f.readlines()
    reader = csv.reader(lines[1:])
    headers = next(reader)
    col_s1 = next(i for i, h in enumerate(headers) if "範疇一" in h and "數據" in h)
    col_s2 = next(i for i, h in enumerate(headers) if "範疇二" in h and "數據" in h)
    col_s3 = next(i for i, h in enumerate(headers) if "範疇三" in h and "數據" in h)
    for row in reader:
        if not row:
            continue
        raw = row[0].strip().strip('=').strip('"')
        if not raw.isdigit():
            continue
        def _parse(s: str) -> float | None:
            s = s.strip().replace(",", "")
            try:
                return float(s) if s else None
            except ValueError:
                return None
        result[raw] = {
            "scope1_tco2e": _parse(row[col_s1]),
            "scope2_tco2e": _parse(row[col_s2]),
            "scope3_tco2e": _parse(row[col_s3]),
        }
    return result

GT_CSV_DATA = _load_gt_csv()
print(f"CSV ground truth 載入：{len(GT_CSV_DATA)} 家公司")


# ── CSV 數值比對 ──────────────────────────────────────────
_NUMBER_RE = re.compile(r"[\d,]+(?:\.\d+)?")

def _extract_best_number(text: str, expected: float) -> float | None:
    """從文字萃取最接近 expected 的數值（log 尺度），處理千分位逗號。"""
    candidates = []
    for m in _NUMBER_RE.finditer(text):
        s = m.group().replace(",", "")
        try:
            v = float(s)
            if v > 0:
                candidates.append(v)
        except ValueError:
            pass
    if not candidates:
        return None
    log_exp = math.log10(max(expected, 1))
    return min(candidates, key=lambda v: abs(math.log10(max(v, 1)) - log_exp))


def evaluate_csv_correctness(answer: str, ticker: str, metric: str) -> float | None:
    """對比答案中萃取的數值與 CSV ground truth（±5% → 1.0，超出 → 0.0）。"""
    gt_val = GT_CSV_DATA.get(ticker, {}).get(metric)
    if gt_val is None:
        return None
    extracted = _extract_best_number(answer, gt_val)
    if extracted is None:
        return 0.0
    rel_diff = abs(extracted - gt_val) / max(abs(gt_val), 1e-9)
    return 1.0 if rel_diff <= CSV_TOLERANCE else 0.0


# ── Schema ───────────────────────────────────────────────
class Score(BaseModel):
    score: float = Field(..., ge=0, le=1, description="0=完全不符，1=完美符合")


# ── Judge prompts（譯自 define_evaluation_metrics.ipynb）──
_FAITHFULNESS_PROMPT = """Question: {question}
Context: {context}
Generated Answer: {answer}

評估生成的答案是否能從 Context 推導出來（不在意答案是否正確，只看能否從 Context 推得）。
- 0：答案內容無法從 Context 推得，或答案使用 Context 以外的知識
- 1：答案完全可從 Context 推得
- 中間值反映部分可推得的程度

範例 1：Q「法國首都？」Context「巴黎是法國首都，馬德里是西班牙首都」Answer「巴黎」→ 1
範例 2：Q「法國首都？」Context「猴子愛吃香蕉」Answer「巴黎」→ 0
範例 3：Q「2+2=?」Context「4」Answer「4」→ 0（Context 沒有「2+2」這個前提無法推得）
"""

_CORRECTNESS_PROMPT = """Question: {question}
Ground Truth: {ground_truth}
Generated Answer: {answer}

評估 Generated Answer 與 Ground Truth 在語意上的符合程度（允許用詞不同、單位等價）。
分級給分（不只 0/0.5/1，必要時可給中間值）：
- 1.0：完全符合 ground truth 的核心事實與數值
- 0.7：核心事實對，但細節缺漏（例：答出數值但漏頁碼／標記；答出主要策略但漏次要項）
- 0.5：部分符合（例：兩個子問題答對一個，漏另一個）
- 0.3：方向對但具體內容缺漏（例：知道在某段，但沒給數字）
- 0.0：與 ground truth 矛盾、跑題、或缺漏關鍵資訊

判斷規則：
- 數值類問題：允許 ±5% 誤差視為符合
- 「應拒答」類問題（ground truth 含「未揭露/未找到」）：answer 也誠實拒答視為 1.0
- 「部分可答」類問題（ground truth 含「以原文校正」「具體數值待補」）：answer 答出主要事實即給 0.7+
"""

ABSTAIN_PATTERN = re.compile(
    r"未找到|未揭露|未提及|沒有揭露|抱歉|無法.{0,10}回答|無法.{0,10}提供|無相關資訊|查無"
    r"|安全政策|操作規範|系統提示"     # prompt injection 安全拒答（Q23）
    r"|請針對.{0,20}ESG.{0,20}提問"   # off-topic 早期返回（Q22）
    r"|資料量較大"                      # 產業過多攔截早期返回（Q08）
    r"|資料量較大|請從.{0,5}清單"     # 產業排名類攔截訊息（Q08）
)


# ── 評分器 ────────────────────────────────────────────────
def _judge(prompt: str) -> float:
    # google-genai SDK：Flash 直接省略 thinking_config（與 api/rag.py 一致）
    try:
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Score,
            )
        )
        return Score.model_validate_json(resp.text).score
    except Exception as e:
        print(f"  judge error: {e}", flush=True)
        return None


_CORRECTNESS_STEPS = [0.0, 0.3, 0.5, 0.7, 1.0]


def evaluate_faithfulness(question: str, context: str, answer: str) -> float | None:
    return _judge(_FAITHFULNESS_PROMPT.format(question=question, context=context[:50000], answer=answer))


def evaluate_correctness(question: str, ground_truth: str, answer: str) -> float | None:
    raw = _judge(_CORRECTNESS_PROMPT.format(question=question, ground_truth=ground_truth, answer=answer))
    if raw is None:
        return None
    return min(_CORRECTNESS_STEPS, key=lambda s: abs(s - raw))


def evaluate_abstain(answer: str, expect_abstain: bool) -> bool:
    """expect_abstain=True 且 answer 含拒答關鍵字 → 通過"""
    answered_abstain = bool(ABSTAIN_PATTERN.search(answer))
    return answered_abstain == expect_abstain


# ── API ──────────────────────────────────────────────────
def ask_api(question: str, history: list | None = None) -> dict:
    body = json.dumps({"question": question, "history": history or []}).encode("utf-8")
    req  = urllib.request.Request(f"{API_BASE}/query", data=body,
                                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        return {**d, "elapsed_ms": int((time.time() - t0) * 1000), "error": None}
    except Exception as e:
        return {"answer": "", "sources": [], "error": str(e),
                "elapsed_ms": int((time.time() - t0) * 1000)}


# ── 主流程 ────────────────────────────────────────────────
def _prepare_question(q: dict) -> str:
    """若 yaml 含 question_padding 且 question < 500 字，自動補滿至 501+ 字以觸發長度 guard"""
    question = q["question"]
    if q.get("question_padding") and len(question) < 500:
        padding = "（請涵蓋環境、社會、治理三面向，包含碳排放、能源、水資源、廢棄物、供應鏈、員工福利、董事會結構等所有 ESG 議題之歷年數據趨勢與未來規劃）" * 5
        question = (question + padding)[:520]
    return question


def evaluate_one(q: dict) -> dict:
    qid = q["id"]
    print(f"  [{qid}] querying API...", flush=True)
    api = ask_api(_prepare_question(q), q.get("history"))

    answer  = api.get("answer", "")
    sources = api.get("sources", [])
    context = "\n---\n".join(s.get("text", "") for s in sources) if sources else ""

    metrics = q.get("metrics_applicable", [])
    out = {
        "id": qid,
        "category": q["category"],
        "retrieval_known_issue": q.get("retrieval_known_issue", False),
        "expect_abstain": q.get("expect_abstain", False),
        "answer": answer,
        "answer_truncated": (answer[:200] + "...") if len(answer) > 200 else answer,
        "n_sources": len(sources),
        "elapsed_ms": api["elapsed_ms"],
        "api_error": api.get("error"),
    }

    if api.get("error") or not answer:
        out.update({"faithfulness": None, "correctness": None, "abstain_pass": None, "csv_correctness": None})
        return out

    # 評分（按 metrics_applicable 篩選）
    out["faithfulness"] = evaluate_faithfulness(q["question"], context, answer) if "faithfulness" in metrics and context else None
    out["correctness"]  = evaluate_correctness(q["question"], q["ground_truth"], answer) if ("answer_relevancy" in metrics or "context_recall" in metrics) else None
    out["abstain_pass"] = evaluate_abstain(answer, q["expect_abstain"]) if "abstain_accuracy" in metrics else None

    # CSV 數值比對（精確 ground truth）
    csv_ticker = q.get("gt_csv_ticker")
    csv_metric = q.get("gt_csv_metric")
    out["csv_correctness"] = evaluate_csv_correctness(answer, csv_ticker, csv_metric) if csv_ticker and csv_metric else None

    return out


def summarize(results: list[dict]):
    def avg(key, filter_fn=lambda r: True):
        vals = [r[key] for r in results if r.get(key) is not None and filter_fn(r)]
        return sum(vals) / len(vals) if vals else None

    def pct(key, filter_fn=lambda r: True):
        vals = [r[key] for r in results if r.get(key) is not None and filter_fn(r)]
        return sum(1 for v in vals if v) / len(vals) if vals else None

    normal_filter = lambda r: not r["retrieval_known_issue"] and not r["expect_abstain"]
    issue_filter  = lambda r: r["retrieval_known_issue"]
    abstain_filter = lambda r: r["expect_abstain"]

    print("\n" + "═" * 60)
    print(f"總題數: {len(results)}（含 API error: {sum(1 for r in results if r.get('api_error'))}）")
    print("─" * 60)
    print(f"全體平均  faithfulness: {avg('faithfulness')}")
    print(f"全體平均  correctness:  {avg('correctness')}")
    print(f"全體平均  abstain_pass: {pct('abstain_pass')}")
    print("─" * 60)
    print("【正常題（非 abstain、非 retrieval issue）】")
    print(f"  faithfulness: {avg('faithfulness', normal_filter)}")
    print(f"  correctness:  {avg('correctness', normal_filter)}")
    print(f"  平均延遲:     {avg('elapsed_ms', normal_filter):.0f} ms" if avg('elapsed_ms', normal_filter) else "  N/A")
    print("【retrieval_known_issue（baseline 預期 abstain，重構後應轉好）】")
    print(f"  abstain_pass: {pct('abstain_pass', issue_filter)}（含此分組）")
    print(f"  faithfulness: {avg('faithfulness', issue_filter)}")
    print(f"  correctness:  {avg('correctness', issue_filter)}")
    print("【expect_abstain（系統行為測試）】")
    print(f"  abstain_pass: {pct('abstain_pass', abstain_filter)}")
    print("═" * 60)


def main():
    yaml_path = Path(YAML_PATH)
    if not yaml_path.exists():
        sys.exit(f"❌ 找不到 {yaml_path}")

    questions = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    questions = [q for q in questions if not q.get("excluded")]
    print(f"載入 {len(questions)} 題（已排除 excluded）")

    OUT_DIR.mkdir(exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_out = OUT_DIR / f"baseline_{ts}.jsonl"
    csv_out   = OUT_DIR / f"baseline_{ts}.csv"

    results = []
    with cf.ThreadPoolExecutor(max_workers=N_PARALLEL) as ex:
        futures = {ex.submit(evaluate_one, q): q["id"] for q in questions}
        for fut in cf.as_completed(futures):
            r = fut.result()
            results.append(r)
            jsonl_out.open("a", encoding="utf-8").write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  ✓ {r['id']}: faith={r.get('faithfulness')} correct={r.get('correctness')} "
                  f"abstain_pass={r.get('abstain_pass')} ({r['elapsed_ms']}ms)", flush=True)

    results.sort(key=lambda r: r["id"])

    # CSV output
    fields = ["id", "category", "retrieval_known_issue", "expect_abstain",
              "faithfulness", "correctness", "csv_correctness", "abstain_pass",
              "n_sources", "elapsed_ms", "api_error", "answer_truncated"]
    with csv_out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    results.sort(key=lambda r: r["id"])

    # CSV 數值正確率摘要
    csv_vals = [r["csv_correctness"] for r in results if r.get("csv_correctness") is not None]
    if csv_vals:
        pct = sum(csv_vals) / len(csv_vals)
        print(f"\nCSV 數值正確率：{pct:.1%}（{int(sum(csv_vals))}/{len(csv_vals)} 題）")

    print(f"\n→ {jsonl_out}")
    print(f"→ {csv_out}")
    summarize(results)


if __name__ == "__main__":
    main()
