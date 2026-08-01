import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sgd.scene_graph.global_builder import build_global_event_graph, group_candidate_regions
from sgd.scene_graph.schema import (
    safe_load_global_event_graph,
    safe_load_stage1_graph,
    safe_load_stage2_segment_graph,
)
from sgd.scene_graph.segment_builder import build_stage2_segment_graph
from sgd.scene_graph.stage1_builder import (
    _align_entity,
    _build_prompt,
    _normalize_frame_parse,
    _parse_single_frame_with_vlm,
    build_stage1_scene_graph,
)


class SceneGraphPipelineTest(unittest.TestCase):
    def test_frame_normalization_alignment_and_context(self):
        entities, relations = _normalize_frame_parse(
            {
                "entities": [
                    {
                        "local_id": "e1",
                        "name": "Person",
                        "type": "PERSON",
                        "attributes": ["Standing"],
                        "desc_text": "adult in blue",
                        "confidence": 2,
                    },
                    {"local_id": "e2", "name": "camera", "type": "object"},
                ],
                "relations": [{"s": "e1", "p": "Looking At", "o": "e2", "confidence": -1}],
            }
        )
        self.assertEqual([item["name"] for item in entities], ["person"])
        self.assertEqual(entities[0]["confidence"], 1.0)
        self.assertEqual(relations[0]["p"], "looking_at")
        self.assertEqual(relations[0]["confidence"], 0.0)

        bank = []
        first = _align_entity(entities[0], bank, 0.0, 0.78)
        second = _align_entity(dict(entities[0], confidence=0.4), bank, 5.0, 0.78)
        self.assertEqual(first, second)
        self.assertEqual(bank[0]["last_seen"], 5.0)

        prompt = _build_prompt(
            previous_graphs=[{"frame_id": "f_1"}],
            entity_bank=bank,
            caption_context="[0-10] a person enters",
        )[1]["content"]
        self.assertIn("preceding_frame_graphs", prompt)
        self.assertIn("video_entity_registry", prompt)
        self.assertIn("auxiliary retry context", prompt)

    def test_frame_vlm_path_builds_prompt_and_propagates_transport_failure(self):
        fake_utils = types.ModuleType("sgd.utils")
        fake_call = Mock(
            return_value={
                "content": json.dumps(
                    {
                        "entities": [
                            {
                                "local_id": "e1", "name": "person", "type": "person",
                                "attributes": [], "desc_text": "adult", "confidence": 0.9,
                            }
                        ],
                        "relations": [],
                    }
                )
            }
        )
        fake_utils.call_openai_model_with_tools = fake_call
        with patch.dict(sys.modules, {"sgd.utils": fake_utils}):
            entities, _ = _parse_single_frame_with_vlm(
                "/tmp/frame.jpg",
                query_hint="who enters",
                previous_graphs=[{"frame_id": "f_1"}],
                entity_bank=[],
            )
            self.assertEqual(entities[0]["name"], "person")
            prompt = fake_call.call_args.kwargs["messages"][1]["content"]
            self.assertIn("who enters", prompt)
            self.assertIn("preceding_frame_graphs", prompt)

            fake_call.return_value = None
            with self.assertRaisesRegex(RuntimeError, "Frame-graph VLM call failed"):
                _parse_single_frame_with_vlm("/tmp/frame.jpg")

    def test_candidate_region_thresholds(self):
        segments = [
            {
                "segment_id": "s1", "t_start": 0, "t_end": 10,
                "active_entities": [{"entity_id": "e1"}],
                "temporal_relations": [{"s": "e1", "p": "holds", "o": "e2"}],
            },
            {
                "segment_id": "s2", "t_start": 15, "t_end": 25,
                "active_entities": [{"entity_id": "e1"}],
                "temporal_relations": [{"s": "e1", "p": "walks_to", "o": "e3"}],
            },
            {
                "segment_id": "s3", "t_start": 50, "t_end": 60,
                "active_entities": [{"entity_id": "e4"}],
                "temporal_relations": [],
            },
        ]
        groups = group_candidate_regions(segments, 12.0, 0.18, 0.22)
        self.assertEqual([[s["segment_id"] for s in group] for group in groups], [["s1", "s2"], ["s3"]])

    def test_ubiquitous_generic_entity_does_not_merge_unrelated_regions(self):
        segments = [
            {
                "segment_id": f"s{index}",
                "t_start": index * 10,
                "t_end": index * 10 + 10,
                "active_entities": [
                    {"entity_id": "e_person", "canonical_name": "person"}
                ],
                "temporal_relations": [
                    {"s": "e_person", "p": f"action_{index}", "o": f"e{index}"}
                ],
            }
            for index in range(4)
        ]
        groups = group_candidate_regions(segments, 12.0, 0.18, 0.22)
        self.assertEqual(len(groups), 4)

    def test_empty_frame_parse_retries_with_caption_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_root = Path(tmp) / "retry_video"
            frames_dir = video_root / "frames"
            captions_dir = video_root / "captions"
            frames_dir.mkdir(parents=True)
            captions_dir.mkdir(parents=True)
            (frames_dir / "frame_n000000.jpg").write_bytes(b"jpeg")
            captions_path = captions_dir / "captions.json"
            captions_path.write_text(
                json.dumps({"0_10": {"caption": "A person opens a door."}, "subject_registry": {}})
            )
            recovered = (
                [{
                    "local_id": "e1", "name": "person", "type": "person",
                    "attributes": [], "desc_text": "person", "confidence": 0.8,
                }],
                [],
            )
            output_path = captions_dir / "scene_graph_stage1.json"
            with patch(
                "sgd.scene_graph.stage1_builder._parse_single_frame_with_vlm",
                side_effect=[([], []), recovered],
            ) as mocked_parse:
                graph = build_stage1_scene_graph(
                    str(captions_path), str(output_path), frame_dir_override=str(frames_dir)
                )
            self.assertEqual(mocked_parse.call_count, 2)
            self.assertEqual(graph["stats"]["caption_retry_frames"], 1)
            self.assertEqual(graph["stats"]["num_entities"], 1)

    def test_mocked_three_level_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_root = Path(tmp) / "sample"
            frames_dir = video_root / "frames"
            captions_dir = video_root / "captions"
            frames_dir.mkdir(parents=True)
            captions_dir.mkdir(parents=True)
            for index in (0, 10, 20):
                (frames_dir / f"frame_n{index:06d}.jpg").write_bytes(b"jpeg")
            captions_path = captions_dir / "captions.json"
            captions_path.write_text(
                json.dumps(
                    {
                        "0_10": {"caption": "A person carries a bag."},
                        "10_20": {"caption": "The person enters a room."},
                        "subject_registry": {},
                    }
                )
            )
            stage1_path = captions_dir / "scene_graph_stage1.json"
            frame_parse = (
                [
                    {
                        "local_id": "e1", "name": "person", "type": "person",
                        "attributes": [], "desc_text": "same person", "confidence": 0.9,
                    },
                    {
                        "local_id": "e2", "name": "bag", "type": "object",
                        "attributes": [], "desc_text": "a bag", "confidence": 0.8,
                    },
                ],
                [{"s": "e1", "p": "carrying", "o": "e2", "confidence": 0.8}],
            )
            with patch(
                "sgd.scene_graph.stage1_builder._parse_single_frame_with_vlm",
                return_value=frame_parse,
            ):
                build_stage1_scene_graph(str(captions_path), str(stage1_path), frame_dir_override=str(frames_dir))
            stage1 = safe_load_stage1_graph(str(stage1_path))
            self.assertIsNotNone(stage1)
            person_id = stage1["entities"][0]["entity_id"]
            bag_id = stage1["entities"][1]["entity_id"]

            stage2_path = captions_dir / "scene_graph_stage2_segment.json"
            segment_result = {
                "relations": [
                    {
                        "s": person_id, "p": "enters_with", "o": bag_id,
                        "relation_type": "action", "t_start": 0, "t_end": 10,
                        "support_frame_ids": [stage1["frames"][0]["frame_id"]],
                        "confidence": 0.9,
                    }
                ]
            }
            with patch("sgd.scene_graph.segment_builder._call_segment_vlm", return_value=segment_result), patch(
                "sgd.scene_graph.segment_builder.config.SCENE_GRAPH_STAGE2_EMBED_SEGMENTS", False
            ):
                build_stage2_segment_graph(str(stage1_path), str(stage2_path))
            stage2 = safe_load_stage2_segment_graph(str(stage2_path))
            self.assertIsNotNone(stage2)
            self.assertTrue(stage2["segments"][0]["caption_snippets"])

            global_path = captions_dir / "scene_graph_global.json"
            global_result = {
                "relations": [
                    {
                        "relation_type": "event", "predicate": "enters_room",
                        "participant_entity_ids": [person_id, bag_id],
                        "description": "The person enters while carrying the bag.",
                        "t_start": 0, "t_end": 10,
                        "support_segment_ids": [stage2["segments"][0]["segment_id"]],
                        "confidence": 0.9, "uncertainty": 0.1,
                    }
                ]
            }
            with patch("sgd.scene_graph.global_builder._call_global_vlm", return_value=global_result):
                build_global_event_graph(str(stage1_path), str(stage2_path), str(global_path))
            global_graph = safe_load_global_event_graph(str(global_path))
            self.assertIsNotNone(global_graph)
            self.assertEqual(global_graph["relations"][0]["relation_type"], "event")
            self.assertTrue(global_graph["contains_links"])


if __name__ == "__main__":
    unittest.main()
