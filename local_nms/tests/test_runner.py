from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


LOCAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOCAL_ROOT))

from local_nms import engine  # noqa: E402
from local_nms.ai import reasoning as nms_ai_reasoning  # noqa: E402
from local_nms.control import disabled_remote_config  # noqa: E402
from local_nms.runner import effective_config, run_s3_nms  # noqa: E402
from run import completion_target_names  # noqa: E402


class _Paginator:
    def __init__(self, client: "FakeS3"):
        self.client = client

    def paginate(self, *, Bucket: str, Prefix: str):
        keys = sorted(
            key
            for (bucket, key) in self.client.objects
            if bucket == Bucket and key.startswith(Prefix)
        )
        return [{"Contents": [{"Key": key} for key in keys]}]


class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def add_json(self, bucket: str, key: str, payload: dict) -> None:
        self.objects[(bucket, key)] = json.dumps(payload).encode("utf-8")

    def get_paginator(self, operation: str) -> _Paginator:
        if operation != "list_objects_v2":
            raise ValueError(operation)
        return _Paginator(self)

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_kwargs) -> dict:
        self.objects[(Bucket, Key)] = Body
        return {}

    def json_at(self, bucket: str, key: str) -> dict:
        return json.loads(self.objects[(bucket, key)].decode("utf-8"))


def _cube_faces() -> dict:
    return {
        "panos": {
            "pano1": {
                "roomId": "room1",
                "floorId": "floor1",
                "images": [{"fileName": "pano1_1_front.jpg", "faceIndex": 1}],
            }
        }
    }


def _result() -> dict:
    return {
        "prediction": {
            "3d_bounding_boxes": [
                {
                    "id": 1,
                    "panoId": "pano1",
                    "rotation": {"x": 0.1, "y": 0.2},
                    "anchorPosition": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "stemVector": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "surfaceNormal": {"x": 0.0, "y": 1.0, "z": 0.0},
                }
            ],
            "tasks": [
                {
                    "task_id": 7,
                    "task_name": "A1: Surface defect",
                    "task_result": [
                        {
                            "instance_id": 1,
                            "2d_bbox_id": 1,
                            "3d_bbox_id": 1,
                            "instance_class": "minor",
                            "instance_reasoning": "Minor defect.",
                            "attributes": {"trade": "Finishes"},
                        }
                    ],
                }
            ],
        }
    }


class LocalNmsRunnerTest(unittest.TestCase):
    def test_completion_uses_submitted_results_unless_strict_mode_is_requested(self):
        cube_faces = {"face-1.jpg", "face-2.jpg", "face-3.jpg"}
        submitted_results = {"face-2.jpg"}

        self.assertEqual(
            completion_target_names(
                cube_faces,
                submitted_results,
                require_all_cube_faces=False,
            ),
            submitted_results,
        )
        self.assertEqual(
            completion_target_names(
                cube_faces,
                submitted_results,
                require_all_cube_faces=True,
            ),
            cube_faces,
        )

    def test_remote_control_disables_processing_and_nms_without_mutating_source(self):
        source = {"enabled": True, "nms": {"enabled": True, "distance": 0.75}}

        disabled = disabled_remote_config(source)

        self.assertTrue(source["enabled"])
        self.assertTrue(source["nms"]["enabled"])
        self.assertFalse(disabled["enabled"])
        self.assertFalse(disabled["nms"]["enabled"])
        self.assertEqual(disabled["nms"]["distance"], 0.75)

    def test_effective_config_deep_merges_remote_defaults(self):
        config = effective_config(
            {
                "nms": {"enabled": False, "taskDistances": {"A1": 0.75}},
                "nmsAi": {"model": {"version": "custom"}},
            },
            "project-1",
        )

        self.assertTrue(config["nms"]["enabled"])
        self.assertEqual(config["nms"]["taskDistances"]["A1"], 0.75)
        self.assertEqual(config["nms"]["groupBy"], engine.DEFAULT_CONFIG["groupBy"])
        self.assertEqual(config["nmsAi"]["model"]["version"], "custom")
        self.assertEqual(
            config["nmsAi"]["model"]["name"],
            nms_ai_reasoning.DEFAULT_CONFIG["model"]["name"],
        )
        self.assertEqual(config["projectName"], "project-1")

    def test_run_writes_local_and_s3_reports_without_publishing(self):
        bucket = "bucket"
        project_id = "project-1"
        use_case_id = "use-case-1"
        images_prefix = f"{project_id}/use_cases/{use_case_id}/images/"
        input_key = f"{images_prefix}image-1/nms_input.json"
        s3 = FakeS3()
        s3.add_json(
            bucket,
            input_key,
            {
                "status": "ready",
                "imageId": "image-1",
                "source": {
                    "rawImageName": "pano1_1_front.jpg",
                    "resultKey": f"{images_prefix}image-1/result.json",
                },
                "result": _result(),
            },
        )
        s3.add_json(bucket, f"{images_prefix}image-2/nms_input.json", {"status": "skipped"})

        with tempfile.TemporaryDirectory() as temp_dir:
            run = run_s3_nms(
                s3,
                bucket=bucket,
                project_id=project_id,
                use_case_id=use_case_id,
                images_prefix=images_prefix,
                cube_faces=_cube_faces(),
                remote_config={"projectName": "Test Project", "nmsAi": {"enabled": False}},
                run_id="20260817T120000Z",
                output_root=Path(temp_dir),
            )

            self.assertEqual(run.report["tagCount"], 1)
            self.assertEqual(run.report["issueCount"], 1)
            self.assertEqual(run.marker["inputCount"], 1)
            self.assertFalse(run.marker["backendIssues"]["published"])
            self.assertTrue((run.output_dir / "nms_report.json").is_file())
            issue_payload = json.loads(
                (run.output_dir / "issues" / "issue_0001.json").read_text(encoding="utf-8")
            )

        self.assertNotIn("fe_issue_id", issue_payload)
        self.assertIsInstance(issue_payload["location"], list)
        expected_finalization_id = hashlib.sha256(
            f"{project_id}/{use_case_id}/per-image-nms".encode("utf-8")
        ).hexdigest()
        self.assertEqual(run.marker["finalizationId"], expected_finalization_id)
        report_key = f"{project_id}/use_cases/{use_case_id}/qaqc/local_nms/nms_report.json"
        marker_key = f"{project_id}/use_cases/{use_case_id}/qaqc/local_nms/finalization.json"
        self.assertEqual(run.report_key, report_key)
        self.assertEqual(s3.json_at(bucket, report_key)["report"], run.report)
        self.assertEqual(s3.json_at(bucket, marker_key), run.marker)


if __name__ == "__main__":
    unittest.main()
