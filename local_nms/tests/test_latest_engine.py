from __future__ import annotations

import sys
import unittest
from pathlib import Path


LOCAL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOCAL_ROOT))

from local_nms import engine  # noqa: E402


class LatestEngineTest(unittest.TestCase):
    def test_latest_room_metadata_and_object_type_are_in_issue_output(self):
        cube_faces = {
            "panos": {
                "pano1": {
                    "roomId": "room-1",
                    "floorId": "floor-1",
                    "locationId": "location-1",
                    "room_name": "Kitchen",
                    "images": [{"fileName": "pano1_1_front.jpg", "faceIndex": 1}],
                }
            }
        }
        results = {
            "image-1": {
                "prediction": {
                    "3d_bounding_boxes": [
                        {
                            "id": 1,
                            "panoId": "pano1",
                            "rotation": {"x": 0.1, "y": 0.2},
                            "anchorPosition": {"x": 0.0, "y": 0.0, "z": 1.0},
                            "stemVector": {"x": 0.0, "y": 0.0, "z": 1.0},
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
                                    "instance_reasoning": "Wall-crack; visible evidence",
                                    "attributes": {"trade": "Finishes"},
                                }
                            ],
                        }
                    ],
                }
            }
        }
        sources = {"image-1": {"rawImageName": "pano1_1_front.jpg"}}

        _results, report = engine.run_nms(
            results,
            cube_faces,
            sources,
            {
                "projectName": "project-1",
                "nms": {"enabled": True},
                "nmsAi": {"enabled": False},
                "issueId": {"numberPadding": 4, "issueTypeSource": "task_code"},
            },
        )

        issue = report["issues"][0]
        self.assertEqual(issue["keeper"]["objectType"], "wall")
        self.assertEqual(issue["issueInfo"]["location"]["label"], "Kitchen")


if __name__ == "__main__":
    unittest.main()
