from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


LOCAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOCAL_ROOT))

from local_nms.publisher import (  # noqa: E402
    publish_issue_report,
    validate_issue_report,
    validate_publish_environment,
)
from publish_issues import load_report  # noqa: E402


class LocalIssuePublisherTest(unittest.TestCase):
    def test_dry_run_validation_checks_payload_and_environment_without_network(self):
        report = {
            "issues": [
                {
                    "issueId": "issue-1",
                    "issueInfo": {"fe_issue_id": "", "location": {"id": 1}},
                }
            ]
        }
        with ExitStack() as stack:
            stack.enter_context(patch.dict("os.environ", {
                "QAQC_API_BASE_URL": "http://127.0.0.1:18000",
                "QAQC_INTERNAL_API_KEY": "secret",
            }, clear=True))
            settings = validate_publish_environment("project-1", "use-case-1")

        self.assertEqual(validate_issue_report(report), 1)
        self.assertEqual(
            settings["endpoint"],
            "http://127.0.0.1:18000/internal/qaqc/projects/project-1/use-cases/use-case-1/issues",
        )
        self.assertEqual(settings["apiKeyHeader"], "X-API-Key-s2s")

    def test_load_report_accepts_local_run_directory_and_s3_wrapper(self):
        report = {"issues": []}
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            (run_dir / "nms_report.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(load_report(run_dir), report)

            wrapped = Path(temp_dir) / "wrapped.json"
            wrapped.write_text(json.dumps({"status": "ready", "report": report}), encoding="utf-8")
            self.assertEqual(load_report(wrapped), report)

    @patch("local_nms.publisher._create_issue")
    def test_publish_uses_remote_idempotency_key(self, create_issue):
        issue_info = {
            "fe_issue_id": "frontend-only",
            "location": {"id": 1},
            "images": ["image-1"],
        }
        report = {"issues": [{"issueId": "issue-1", "issueInfo": issue_info}]}
        finalization_id = "finalization-1"

        created = publish_issue_report(
            report,
            project_id="project-1",
            use_case_id="use-case-1",
            finalization_id=finalization_id,
        )

        self.assertEqual(created, 1)
        expected_key = hashlib.sha256(f"{finalization_id}:issue-1".encode("utf-8")).hexdigest()
        create_issue.assert_called_once_with(
            issue_info,
            project_id="project-1",
            use_case_id="use-case-1",
            idempotency_key=expected_key,
        )

    @patch.dict("os.environ", {
        "QAQC_API_BASE_URL": "http://127.0.0.1:18000",
        "QAQC_INTERNAL_API_KEY": "secret",
        "QAQC_API_KEY_HEADER": "X-API-Key-s2s",
    }, clear=True)
    @patch("local_nms.publisher.urlopen")
    def test_publish_records_successful_api_receipt(self, urlopen):
        response = urlopen.return_value.__enter__.return_value
        response.getcode.return_value = 201
        response.read.return_value = b'{"id":"backend-1"}'
        report = {
            "issueCount": 1,
            "issues": [{
                "issueId": "project-image-A1-0001",
                "issueInfo": {
                    "fe_issue_id": "",
                    "location": {"id": 1},
                    "images": ["image-1"],
                },
            }],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            receipt_path = Path(temp_dir) / "publish_receipts.json"
            created = publish_issue_report(
                report,
                project_id="project-1",
                use_case_id="use-case-1",
                finalization_id="finalization-1",
                receipts_path=receipt_path,
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

        self.assertEqual(created, 1)
        self.assertEqual(receipt["publishedCount"], 1)
        self.assertEqual(receipt["receipts"][0]["issueId"], "project-image-A1-0001")
        self.assertEqual(receipt["receipts"][0]["status"], 201)


if __name__ == "__main__":
    unittest.main()
