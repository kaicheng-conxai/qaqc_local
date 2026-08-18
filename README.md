# Local 3D backfill and NMS

This workflow:

1. Reads QAQC results from S3.
2. Creates local 3D/NMS input files.
3. Runs local NMS/NMS-AI.
4. Optionally publishes issues to the API.

`command.sh` is a run note. Run its commands manually in two terminals.
For the standard setup, simply follow `command.sh` step by step.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
npm install
npx playwright install chromium
```

Playwright installs the browser in its default cache unless
`PLAYWRIGHT_BROWSERS_PATH` is set. If you use a custom directory, set the same
value both when installing and when running:

```bash
export PLAYWRIGHT_BROWSERS_PATH=/path/to/playwright-browsers
npx playwright install chromium
```

If you use Playwright's default cache, omit this variable and the export in
Terminal 2.

Terminal 1:

```bash
aws sso login --profile Production-app-developer-937745326287

kubectl -n sitelens port-forward \
  service/conxai-sitelens-var-api-conxai-helm-api \
  18000:8000
```

Terminal 2:

```bash
cd /path/to/qaqc_local
source /path/to/conxai-qaqc-quality-checker/.venv/bin/activate

# Use the same directory configured during Playwright installation.
export PLAYWRIGHT_BROWSERS_PATH=/path/to/playwright-browsers
export AWS_PROFILE=Production-app-developer-937745326287
export QAQC_API_BASE_URL=http://127.0.0.1:18000
export QAQC_API_KEY_HEADER=X-API-Key-s2s
export MATTERPORT_SDK_KEY="$(kubectl -n qaqc get secret conxai-qaqc-quality-checker -o jsonpath='{.data.matterportSdkKey}' | base64 --decode)"
export OPENAI_API_KEY="$(kubectl -n qaqc get secret conxai-qaqc-quality-checker -o jsonpath='{.data.openaiApiKey}' | base64 --decode)"
export QAQC_INTERNAL_API_KEY="$(kubectl -n qaqc get secret conxai-qaqc-quality-checker -o jsonpath='{.data.qaqcInternalApiKey}' | base64 --decode)"
```

## Run

Run one or more use cases:

```bash
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
```

To regenerate existing inputs, add `--force`:

```bash
python3 run.py \
  s3://prod-sitelens-var/4219ca25-2c3b-4a61-b872-74475de127fa/use_cases/8a739b89-ac87-48b3-8ce1-4c02e6424b07/images/ \
  --force \
  --publish-issues
```

Remove `--publish-issues` for a local-only run. Do not use `--no-s3-report` if
you need to verify the S3 report and finalization files.

If `nms_input.json` files already exist, run only NMS:

```bash
python3 run_nms.py \
  s3://prod-sitelens-var/4219ca25-2c3b-4a61-b872-74475de127fa/use_cases/8a739b89-ac87-48b3-8ce1-4c02e6424b07/images/ \
  --publish-issues
```

## NMS configuration

There are two config layers:

- `config.json` controls Matterport/3D capture (`workers`, `chunkSize`,
  `settleMs`, `headed`, and session settings).
- NMS settings are in the remote S3 `qaqc_config.json`, under `nms`.

The remote config must contain:

```json
{"processingMode": "perImage"}
```

For `run.py`, edit the remote `qaqc_config.json` before running. The
`run.py --config` option is only for 3D capture, not NMS.

For `run_nms.py`, create a local override file and pass it with `--config`:

Save this as `nms-overrides.json`:

```json
{
  "nms": {
    "distance": 0.75,
    "veryCloseDistance": 0.10,
    "groupBy": "task_object_type",
    "clusterMode": "greedy_direct",
    "keeperSelection": "cluster_middle",
    "taskDistances": {
      "A1": 0.75
    },
    "faceFilter": {
      "skipFaceDirections": ["top"]
    }
  },
  "nmsAi": {
    "enabled": false
  }
}
```

Run it with:

```bash
python3 run_nms.py \
  s3://prod-sitelens-var/project/use_cases/use_case/images/ \
  --config nms-overrides.json \
  --publish-issues
```

The override is merged into the remote config and is not written back to S3.

Useful NMS fields:

- `distance`: normal 3D duplicate distance; default `0.5`.
- `veryCloseDistance`: automatic same-task merge distance; default `0.1`.
- `taskDistances`: exact `task_name` distance overrides.
- `groupBy`: `task`, `trade`, `object_type`, `task_object_type`, or `all`.
- `clusterMode`: `greedy_direct` or `union`.
- `keeperSelection`: `cluster_middle` or `severity`.
- `faceFilter.skipFaceDirections`: faces to ignore, such as `top`.

NMS-AI is disabled by default. To enable it, set `nmsAi.enabled` to `true`,
configure its model and prompts, and provide `OPENAI_API_KEY`.

The local runner forces local NMS on and leaves remote QAQC disabled after the
run.

## Outputs

Local files:

```text
output/<project>/<use_case>/<run_id>/
  nms_report.json
  finalization.json
  issues/issue_0001.json
  publish_receipts.json
```

S3 files:

```text
<project>/use_cases/<use_case>/qaqc/local_nms/nms_report.json
<project>/use_cases/<use_case>/qaqc/local_nms/finalization.json
```
