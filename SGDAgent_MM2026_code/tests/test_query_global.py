from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SGD_ROOT = PACKAGE_ROOT / "sgd"
SCENE_GRAPH_ROOT = SGD_ROOT / "scene_graph"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load only the standard-library scene-graph modules under test.  This avoids
# importing optional video/API dependencies from the package-level __init__.
sgd_package = types.ModuleType("sgd")
sgd_package.__path__ = [str(SGD_ROOT)]
scene_graph_package = types.ModuleType("sgd.scene_graph")
scene_graph_package.__path__ = [str(SCENE_GRAPH_ROOT)]
sys.modules.setdefault("sgd", sgd_package)
sys.modules.setdefault("sgd.scene_graph", scene_graph_package)

_load_module("sgd.scene_graph.schema", SCENE_GRAPH_ROOT / "schema.py")
global_event_index = _load_module(
    "sgd.scene_graph.global_event_index",
    SCENE_GRAPH_ROOT / "global_event_index.py",
)
query_global = _load_module(
    "sgd.scene_graph.query_global",
    SCENE_GRAPH_ROOT / "query_global.py",
)


EVENT_SPECS = [
    ("bicycle", "riding"),
    ("kitchen", "cooking"),
    ("red car", "driving"),
    ("dog", "running"),
    ("concert", "singing"),
    ("book", "reading"),
    ("telescope", "observing"),
]


def _stage2_graph(reverse: bool = False) -> dict:
    segments = []
    for index, (entity_name, predicate) in enumerate(EVENT_SPECS, start=1):
        start = float((index - 1) * 100)
        entity_id = f"ent_{index:06d}"
        segments.append(
            {
                "segment_id": f"seg_{index:04d}",
                "t_start": start,
                "t_end": start + 30.0,
                "active_entities": [
                    {
                        "entity_id": entity_id,
                        "canonical_name": entity_name,
                        "type": "object",
                        "confidence": 0.9,
                    }
                ],
                "temporal_relations": [
                    {
                        "s": entity_id,
                        "p": predicate,
                        "o": entity_id,
                        "confidence": 0.8,
                    }
                ],
                "support_frame_ids": [f"f_{index:06d}"],
                "summary": f"{entity_name} {predicate}",
            }
        )
    if reverse:
        segments.reverse()
    return {
        "version": "stage2_segment_v1",
        "video_id": "video-test",
        "segments": segments,
        "stats": {},
    }


def _index(reverse: bool = False) -> dict:
    return global_event_index.build_global_event_index(
        _stage2_graph(reverse=reverse),
        max_events=12,
        max_chars=10000,
        merge_gap_sec=0.0,
    )


class GlobalEventIndexTests(unittest.TestCase):
    def test_metadata_event_ids_and_evidence_links_are_stable(self):
        forward = _index()
        reverse = _index(reverse=True)

        self.assertEqual(forward["version"], "gei_event_timeline_v1")
        self.assertEqual(forward["video_id"], "video-test")
        self.assertFalse(forward["backend"]["is_vlm_global_graph"])
        self.assertFalse(forward["backend"]["supports_causal_edges"])

        forward_events = forward["event_timeline"]
        reverse_events = reverse["event_timeline"]
        self.assertEqual(
            [event["event_id"] for event in forward_events],
            [f"gei_evt_{index:04d}" for index in range(1, 8)],
        )
        self.assertEqual(forward_events, reverse_events)
        self.assertEqual(
            forward_events[2]["support_segment_ids"],
            ["seg_0003"],
        )


class GlobalEventQueryTests(unittest.TestCase):
    def setUp(self):
        self.index = _index()

    def test_scores_are_direct_sum_and_semantic_match_ranks_first(self):
        result = query_global.query_global_event_index(
            self.index,
            "red car driving",
            top_k=3,
        )

        top = result["matched_events"][0]
        components = top["score_components"]
        self.assertEqual(top["event_id"], "gei_evt_0003")
        self.assertEqual(
            top["score"],
            round(
                components["lexical"]
                + components["soft"]
                + components["temporal"],
                4,
            ),
        )
        self.assertEqual(
            top["evidence_links"]["segment_ids"],
            ["seg_0003"],
        )
        self.assertFalse(result["fallback_used"])

    def test_scope_is_a_hard_filter_and_recommended_scope_is_intersection(self):
        result = query_global.query_global_event_index(
            self.index,
            "car",
            scope="00:03:10-00:04:10",
            top_k=5,
        )

        self.assertEqual(
            [event["event_id"] for event in result["matched_events"]],
            ["gei_evt_0003"],
        )
        self.assertEqual(result["scope"], "00:03:10-00:04:10")
        signals = result["controller_signals"]
        self.assertTrue(signals["scope_applied"])
        self.assertFalse(signals["scope_miss"])
        self.assertEqual(
            signals["recommended_scope"],
            "00:03:20-00:03:50",
        )

    def test_scope_miss_does_not_fall_back_to_full_video(self):
        result = query_global.query_global_event_index(
            self.index,
            "car",
            scope="00:20:00-00:21:00",
        )

        self.assertEqual(result["matched_events"], [])
        signals = result["controller_signals"]
        self.assertTrue(signals["scope_miss"])
        self.assertTrue(signals["weak_match"])
        self.assertEqual(signals["recommended_next_tool"], "clip_search_tool")

    def test_scope_requires_positive_interval_overlap(self):
        result = query_global.query_global_event_index(
            self.index,
            "bicycle",
            scope="00:00:30-00:01:30",
        )

        self.assertEqual(result["matched_events"], [])
        self.assertTrue(result["controller_signals"]["scope_miss"])

    def test_overview_fallback_is_time_distributed_and_weak(self):
        result = query_global.query_global_event_index(
            self.index,
            "quantum zephyr",
            top_k=5,
        )

        event_ids = [event["event_id"] for event in result["matched_events"]]
        self.assertEqual(len(event_ids), 5)
        self.assertEqual(event_ids[0], "gei_evt_0001")
        self.assertEqual(event_ids[-1], "gei_evt_0007")
        signals = result["controller_signals"]
        self.assertTrue(result["fallback_used"])
        self.assertTrue(signals["overview_fallback"])
        self.assertTrue(signals["weak_match"])
        self.assertTrue(signals["localize_only"])
        self.assertTrue(signals["verification_required"])

    def test_temporal_fallback_ranks_end_event_without_claiming_semantics(self):
        result = query_global.query_global_event_index(
            self.index,
            "What happens at the end?",
            top_k=2,
        )

        self.assertEqual(result["matched_events"][0]["event_id"], "gei_evt_0007")
        self.assertTrue(result["fallback_used"])
        self.assertFalse(result["controller_signals"]["overview_fallback"])
        self.assertFalse(result["controller_signals"]["no_temporal_hit"])

    def test_temporal_top_without_semantic_hit_is_weak(self):
        result = query_global.query_global_event_index(
            self.index,
            "car at 00:00:10",
            top_k=2,
        )

        top = result["matched_events"][0]
        self.assertEqual(top["event_id"], "gei_evt_0001")
        self.assertEqual(top["score_components"]["lexical"], 0.0)
        self.assertEqual(top["score_components"]["soft"], 0.0)
        self.assertTrue(result["controller_signals"]["weak_match"])

    def test_first_and_last_are_sequence_cues_without_substring_false_hits(self):
        self.assertEqual(
            query_global._infer_temporal_hint("What happens first?")["anchor"],
            "start",
        )
        self.assertEqual(
            query_global._infer_temporal_hint("What does the person do last?")["anchor"],
            "end",
        )
        self.assertIsNone(
            query_global._infer_temporal_hint("How do they apply first aid?")["anchor"]
        )
        self.assertIsNone(
            query_global._infer_temporal_hint("What does the first person do?")[
                "anchor"
            ]
        )
        self.assertIsNone(
            query_global._infer_temporal_hint("What does the last person carry?")[
                "anchor"
            ]
        )
        self.assertFalse(
            query_global._infer_temporal_hint("a lasting performance")[
                "has_temporal_cue"
            ]
        )

    def test_top_k_is_capped_at_five(self):
        result = query_global.query_global_event_index(
            self.index,
            "quantum zephyr",
            top_k=99,
        )

        self.assertEqual(result["top_k"], 5)
        self.assertEqual(len(result["matched_events"]), 5)
        self.assertTrue(result["controller_signals"]["top_k_capped"])

    def test_score_ties_are_ordered_by_start_time(self):
        tied_index = {
            "version": "gei_event_timeline_v1",
            "video_id": "tie-test",
            "event_timeline": [
                {
                    "event_id": "later",
                    "t_start": 50.0,
                    "t_end": 60.0,
                    "entities": "shared object",
                    "actions": "moving",
                    "keywords": [],
                    "support_segment_ids": ["seg_0002"],
                },
                {
                    "event_id": "earlier",
                    "t_start": 10.0,
                    "t_end": 20.0,
                    "entities": "shared object",
                    "actions": "moving",
                    "keywords": [],
                    "support_segment_ids": ["seg_0001"],
                },
            ],
        }
        result = query_global.query_global_event_index(
            tied_index,
            "shared object moving",
            top_k=2,
        )

        self.assertEqual(
            [event["event_id"] for event in result["matched_events"]],
            ["earlier", "later"],
        )

    def test_causal_query_is_flagged_as_unsupported(self):
        result = query_global.query_global_event_index(
            self.index,
            "Why did the red car move?",
        )

        signals = result["controller_signals"]
        self.assertTrue(signals["causal_query"])
        self.assertFalse(signals["causal_relation_available"])
        self.assertEqual(
            signals["recommended_next_tool"],
            "segment_graph_query_tool",
        )

    def test_invalid_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            query_global.query_global_event_index(
                self.index,
                "car",
                scope="00:05:00-00:04:00",
            )


if __name__ == "__main__":
    unittest.main()
