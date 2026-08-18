# This is a run note, not an executable script.
# It runs local 3D backfill, latest local NMS/NMS-AI, and API issue publishing.

# install dependencies
requirements.txt
npm install
npx playwright install chromium


# Terminal 1: log in and keep the API port-forward running.
aws sso login --profile Production-app-developer-937745326287

kubectl -n sitelens port-forward \
  service/conxai-sitelens-var-api-conxai-helm-api \
  18000:8000

# Terminal 2: run the local workflow.
cd /Users/kaicheng/Documents/Work/code/tmp/conxai-qaqc-quality-checker
source .venv/bin/activate
cd tmp/local_3d_backfill_issues
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.playwright-browsers"
export AWS_PROFILE=Production-app-developer-937745326287
export QAQC_API_BASE_URL=http://127.0.0.1:18000
export QAQC_API_KEY_HEADER=X-API-Key-s2s
export MATTERPORT_SDK_KEY="$(kubectl -n qaqc get secret conxai-qaqc-quality-checker -o jsonpath='{.data.matterportSdkKey}' | base64 --decode)"
export OPENAI_API_KEY="$(kubectl -n qaqc get secret conxai-qaqc-quality-checker -o jsonpath='{.data.openaiApiKey}' | base64 --decode)"
export QAQC_INTERNAL_API_KEY="$(kubectl -n qaqc get secret conxai-qaqc-quality-checker -o jsonpath='{.data.qaqcInternalApiKey}' | base64 --decode)"



# Q845
python3 run.py \
  s3://prod-sitelens-var/90d0df4d-1539-42a2-b68c-5c3248951848/use_cases/88ed835e-ab0c-4dab-99f5-7f5f705011d5/images/ \
  --publish-issues

# Q844
python3 run.py \
  s3://prod-sitelens-var/b85a88b9-a424-42ac-9e1c-d618fc2528e8/use_cases/c77d816f-dbe7-4747-b137-89da80454805/images/ \
  --publish-issues

# Q545
python3 run.py \
  s3://prod-sitelens-var/efff0d1a-8d26-4db8-a615-5aad5e32634d/use_cases/04c587a8-93f4-475e-8f9d-895d03d1d4c6/images/ \
  --publish-issues


# Q545 - 3
python3 run.py \
  s3://prod-sitelens-var/4219ca25-2c3b-4a61-b872-74475de127fa/use_cases/8a739b89-ac87-48b3-8ce1-4c02e6424b07/images/ \
  --publish-issues


python3 run.py \
  s3://prod-sitelens-var/4219ca25-2c3b-4a61-b872-74475de127fa/use_cases/8a739b89-ac87-48b3-8ce1-4c02e6424b07/images/ \
  --force \
  --publish-issues

# Do not add --no-s3-report when you want to verify the S3 report and finalization files.
