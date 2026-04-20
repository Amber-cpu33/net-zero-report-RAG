#!/usr/bin/env bash
# =============================================================================
# Day 7 — Step 9：Docker 打包 + Cloud Run 部署
# =============================================================================
# 功能：
#   1. 確認本地 FAISS 索引已存在（api/faiss_index/）
#   2. 從 GCP Artifact Registry 建立 Docker Image
#   3. 部署至 Cloud Run（min-instances=0，Scale-to-Zero）
#   4. 設定環境變數與 Service Account
#   5. 執行健康檢查驗證部署成功
#
# 使用方式：
#   chmod +x 09_build_and_deploy.sh
#   ./09_build_and_deploy.sh
#
# 前置條件：
#   - gcloud 已認證（gcloud auth login）
#   - Docker 已安裝
#   - Artifact Registry 已啟用
#   - api/faiss_index/index.faiss 已存在
# =============================================================================

set -euo pipefail

# ── 設定變數 ──────────────────────────────────────────────────
PROJECT_ID="net-zero-report-rag"
REGION="asia-east1"
SERVICE_NAME="esg-rag-api"
SERVICE_ACCOUNT="esg-api-sa@${PROJECT_ID}.iam.gserviceaccount.com"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/esg-pipeline-repo/${SERVICE_NAME}"
IMAGE_TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE_FULL="${IMAGE_NAME}:${IMAGE_TAG}"

# Cloud Run 設定（Scale-to-Zero 是省錢關鍵）
MIN_INSTANCES=0           # 無流量時縮放至零（不收費）
MAX_INSTANCES=3           # 最大實例數（防止意外暴增費用）
MEMORY="2Gi"              # FAISS 索引 493 家 ~362MB，2Gi 足夠（Tier 1+2：台灣50+高碳排/高電耗科技業）
CPU="1"                   # 1 vCPU 足夠輕量推論
CONCURRENCY=5             # 每個實例同時處理請求數（sync threadpool 1 vCPU ≈ 5）
REQUEST_TIMEOUT=60        # 請求逾時（秒，Agentic RAG 多輪 function calling 需較長時間）

# 路徑設定
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
API_DIR="${PROJECT_ROOT}/api"
FAISS_DIR="${API_DIR}/faiss_index"

# ── 顏色輸出 ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC} $*"; exit 1; }

# ── Step 0: 前置檢查 ──────────────────────────────────────────
info "=== Day 7：Docker 打包 + Cloud Run 部署 ==="
info "專案：${PROJECT_ID}  區域：${REGION}"
info "映像：${IMAGE_FULL}"

# 確認 GCS FAISS 索引存在
GCS_BUCKET="${PROJECT_ID}-esg-pipeline"
if ! gsutil -q stat "gs://${GCS_BUCKET}/faiss/index.faiss" 2>/dev/null; then
    error "GCS 找不到 FAISS 索引！請先執行 Step 08：08_build_faiss_index.py"
fi
success "GCS FAISS 索引確認：gs://${GCS_BUCKET}/faiss/"

# 確認 GEMINI_API_KEY 已設
if [ -z "${GEMINI_API_KEY:-}" ]; then
    error "請先設定 GEMINI_API_KEY：export GEMINI_API_KEY=your_api_key"
fi

# LINE Bot（選填，未設定時 /webhook 不啟用）
if [ -z "${LINE_CHANNEL_SECRET:-}" ] || [ -z "${LINE_CHANNEL_ACCESS_TOKEN:-}" ]; then
    warning "LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN 未設定，/webhook 端點不會啟用"
fi

# 確認 gcloud 已認證
if ! gcloud auth list --filter="status:ACTIVE" --format="value(account)" 2>/dev/null | grep -q "@"; then
    error "請先執行 gcloud auth login"
fi

# ── Step 1: 設定 GCP 專案 ─────────────────────────────────────
info "Step 1：設定 GCP 專案..."
gcloud config set project "${PROJECT_ID}" --quiet
gcloud config set run/region "${REGION}" --quiet
success "GCP 專案設定完成"

# ── Step 2: 確保 Artifact Registry 存在 ──────────────────────
info "Step 2：確認 Artifact Registry..."
if ! gcloud artifacts repositories describe esg-pipeline-repo \
    --location="${REGION}" --quiet 2>/dev/null; then
    info "建立 Artifact Registry esg-pipeline-repo..."
    gcloud artifacts repositories create esg-pipeline-repo \
        --repository-format=docker \
        --location="${REGION}" \
        --description="ESG RAG Pipeline Docker Images"
    success "Artifact Registry 建立完成"
else
    success "Artifact Registry 已存在"
fi

# ── Step 3: Cloud Build（比較 faiss 版本，選擇快速或完整路徑） ─
GCS_CURRENT=$(gsutil cat "gs://${GCS_BUCKET}/faiss/index_stats.json" 2>/dev/null \
    | grep -oP '"built_at":\s*"\K[^"]*' || echo "")
GCS_DEPLOYED=$(gsutil cat "gs://${GCS_BUCKET}/faiss/last_deployed_built_at.txt" 2>/dev/null || echo "")

if [ -n "${GCS_CURRENT}" ] && [ "${GCS_CURRENT}" = "${GCS_DEPLOYED}" ]; then
    info "Step 3：FAISS 版本相同（${GCS_CURRENT}），使用快速更新（~3-5 分鐘）..."
    CLOUDBUILD_CONFIG="${API_DIR}/cloudbuild_fast.yaml"
else
    if [ -z "${GCS_DEPLOYED}" ]; then
        info "Step 3：首次部署，使用完整建置（~10-15 分鐘）..."
    else
        info "Step 3：FAISS 版本變更（${GCS_DEPLOYED} → ${GCS_CURRENT}），使用完整重建..."
    fi
    CLOUDBUILD_CONFIG="${API_DIR}/cloudbuild.yaml"
fi

gcloud builds submit "${API_DIR}" \
    --config="${CLOUDBUILD_CONFIG}" \
    --substitutions="_IMAGE_FULL=${IMAGE_FULL},_IMAGE_NAME=${IMAGE_NAME}" \
    --project="${PROJECT_ID}"
success "Cloud Build 完成：${IMAGE_FULL}"
if [ -n "${GCS_CURRENT}" ]; then
    echo "${GCS_CURRENT}" | gsutil cp - "gs://${GCS_BUCKET}/faiss/last_deployed_built_at.txt"
fi

# ── Step 4: 部署至 Cloud Run ────────────────────────────────────
info "Step 8：部署至 Cloud Run（Scale-to-Zero 模式）..."

gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE_FULL}" \
    --platform=managed \
    --region="${REGION}" \
    --service-account="${SERVICE_ACCOUNT}" \
    --min-instances="${MIN_INSTANCES}" \
    --max-instances="${MAX_INSTANCES}" \
    --memory="${MEMORY}" \
    --cpu="${CPU}" \
    --concurrency="${CONCURRENCY}" \
    --timeout="${REQUEST_TIMEOUT}" \
    --allow-unauthenticated \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION},GCS_BUCKET=${GCS_BUCKET},GEMINI_API_KEY=${GEMINI_API_KEY},LINE_CHANNEL_SECRET=${LINE_CHANNEL_SECRET:-},LINE_CHANNEL_ACCESS_TOKEN=${LINE_CHANNEL_ACCESS_TOKEN:-}" \
    --quiet

success "Cloud Run 部署完成！"

# ── Step 5: 取得服務 URL ───────────────────────────────────────
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region="${REGION}" \
    --format="value(status.url)")

info "服務 URL：${SERVICE_URL}"

# ── Step 6: 健康檢查 ─────────────────────────────────────────
info "Step 6：執行健康檢查..."

# 取得存取 Token
ACCESS_TOKEN=$(gcloud auth print-identity-token)

# 等待服務就緒（冷啟動 + FAISS 從 GCS 下載約需 20-30 秒）
sleep 30

# 測試健康端點
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    "${SERVICE_URL}/health")

if [ "${HTTP_CODE}" = "200" ]; then
    success "健康檢查通過（HTTP ${HTTP_CODE}）"

    # 執行一個測試查詢
    RESPONSE=$(curl -s \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{"question": "台積電的淨零目標是什麼年？"}' \
        "${SERVICE_URL}/query" 2>/dev/null || echo "查詢測試失敗")

    if echo "${RESPONSE}" | grep -q "answer"; then
        success "查詢測試通過"
    else
        warning "查詢測試回應異常（可能是冷啟動，稍後再試）：${RESPONSE:0:100}"
    fi
else
    warning "健康檢查回應 HTTP ${HTTP_CODE}（可能是冷啟動，稍後手動測試）"
fi

# ── Step 7: 輸出部署摘要 ─────────────────────────────────────
echo ""
echo "=================================================================="
echo "${GREEN}ESG RAG API 部署完成！${NC}"
echo "=================================================================="
echo ""
echo "  服務名稱  ：${SERVICE_NAME}"
echo "  服務 URL  ：${SERVICE_URL}"
echo "  映像標籤  ：${IMAGE_TAG}"
echo "  最小實例  ：${MIN_INSTANCES}（Scale-to-Zero，無流量不收費）"
echo "  最大實例  ：${MAX_INSTANCES}"
echo "  記憶體    ：${MEMORY}"
echo ""
echo "  API 端點："
echo "    GET  ${SERVICE_URL}/health           → 健康檢查"
echo "    POST ${SERVICE_URL}/query            → ESG 問答"
echo "    GET  ${SERVICE_URL}/companies        → 公司清單"
echo "    GET  ${SERVICE_URL}/stats            → 索引統計"
echo ""
echo "  測試指令："
echo "    TOKEN=\$(gcloud auth print-identity-token)"
echo "    curl -H \"Authorization: Bearer \$TOKEN\" \\"
echo "         -H \"Content-Type: application/json\" \\"
echo "         -d '{\"question\": \"台積電的碳排放量是多少？\"}' \\"
echo "         ${SERVICE_URL}/query"
echo ""
echo "  後續月費估算（實測：125 題問答 ≈ NT\$9）："
echo "    ✓ 無流量時：\$0（Scale-to-Zero）"
echo "    ✓ Gemini 2.5 Flash：~NT\$0.07／題（parse_query + synthesis 各一次）"
echo "    ✓ Vertex AI Embedding：~NT\$0.001／題（text-embedding-004）"
echo "    ✓ Cloud Run 請求費：免費額度 200 萬次／月，一般用量不計費"
echo "    ✓ 預估每月：< 100 TWD（依使用量）"
echo "=================================================================="
