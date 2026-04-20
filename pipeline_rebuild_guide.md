# Pipeline 重構指南

> 此文件供**重新執行 Pipeline** 時參考，日常 API 維護不需要。

## 建立 Spot VM

```bash
gcloud compute instances create esg-pipeline-vm \
  --project=net-zero-report-rag --zone=asia-east1-b \
  --machine-type=e2-standard-4 --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=50GB --scopes=cloud-platform \
  --service-account=esg-pipeline-sa@net-zero-report-rag.iam.gserviceaccount.com

# VM 初始化（SSH 後）
sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo apt-get update && sudo apt-get install -y python3-pip git

# 從 GCS 還原完整套件列表
gsutil cp gs://net-zero-report-rag-esg-pipeline/scripts/vm_requirements.txt /tmp/
pip3 install --break-system-packages -r /tmp/vm_requirements.txt
```

## Tier 執行流程（VM）

```bash
gcloud compute ssh esg-pipeline-vm --zone=asia-east1-b --project=net-zero-report-rag
tmux new -s esg
export TIER=1
python3 scripts/day3_embed/04_submit_embedding_batch.py
python3 scripts/day3_embed/04b_merge_embeddings.py
python3 scripts/day4_vision/05_vision_chart_extract.py
python3 scripts/day3_embed/07_generate_summaries.py
python3 scripts/day6_faiss/08_build_faiss_index.py
```

## Step 07 環境變數

- `TIER`：Tier 過濾（1/2/3/4，0 或未設為全跑）
- `TEST_TICKERS`：逗號分隔指定公司（測試或針對性重跑）
- `REQUIRE_VISION=1`：只跑 `chunks/` + `vision_output/confirmed/` 都備齊的公司

## Step 05 Multi-VM Sharding

```bash
# 上傳腳本到 GCS，再從 VM 下載
gcloud storage cp scripts/day4_vision/05_vision_chart_extract.py \
  gs://net-zero-report-rag-esg-pipeline/scripts/05_vision_chart_extract.py

gcloud compute ssh esg-pipeline-vm --zone=asia-east1-b --project=net-zero-report-rag \
  --command="gsutil cp gs://net-zero-report-rag-esg-pipeline/scripts/05_vision_chart_extract.py \
  ~/scripts/day4_vision/05_vision_chart_extract.py"

# SSH 進每台 VM，tmux 內啟動（nohup 在 plink SSH 環境會被 kill）
tmux new -s esg
mkdir -p ~/logs
export TIER=2 INDUSTRY_CODES="01,03,08,09,10,15,21" SHARD_INDEX=0 SHARD_TOTAL=5
python3 ~/scripts/day4_vision/05_vision_chart_extract.py 2>&1 | tee ~/logs/05_tier2_shard_0.log
```

## VM log 查看

```bash
# 單台
gcloud compute ssh esg-pipeline-vm --zone=asia-east1-b --project=net-zero-report-rag \
  --command="tail -20 ~/logs/05_tier2_shard_0.log"

# 5 台同時查（shard 0-4）
for i in 0 1 2 3 4; do
  VM=$([ $i -eq 0 ] && echo "esg-pipeline-vm" || echo "esg-vm-$i")
  gcloud compute ssh $VM --zone=asia-east1-b --project=net-zero-report-rag \
    --command="tail -3 ~/logs/05_tier2_shard_${i}.log"
done
```

## PDF 補救流程

```bash
# 1. 從 SustaiHub 下載（手動指定 ticker）
python scripts/day1_collect/02_download_pdfs.py 1234 5678 ...

# 2. 本地解析並上傳 GCS
python scripts/local/03_check_parse_pdf_local.py
```

## 腳本上傳到 GCS（更新版本時）

```bash
gsutil cp scripts/day4_vision/05_vision_chart_extract.py \
  gs://net-zero-report-rag-esg-pipeline/scripts/05_vision_chart_extract.py
gsutil cp scripts/day3_embed/07_generate_summaries.py \
  gs://net-zero-report-rag-esg-pipeline/scripts/07_generate_summaries.py
gsutil cp scripts/day6_faiss/08_build_faiss_index.py \
  gs://net-zero-report-rag-esg-pipeline/scripts/08_build_faiss_index.py
```
