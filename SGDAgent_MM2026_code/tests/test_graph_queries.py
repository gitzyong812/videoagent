import unittest

from sgd.scene_graph.query import query_stage1_graph
from sgd.scene_graph.query_global import query_global_event_graph
from sgd.scene_graph.query_segment import query_stage2_segment_graph


class GraphQueryTest(unittest.TestCase):
    def test_parent_scope_hard_filters_frame_and_segment(self):
        stage1 = {
            "version": "stage1_v1", "video_id": "v", "stats": {},
            "entities": [{"entity_id": "e1", "canonical_name": "person", "type": "person", "attributes": []}],
            "frames": [
                {"frame_id": "f1", "t_sec": 5, "entities": [{"entity_id": "e1", "confidence": 1}], "relations": [], "caption_context": "person enters"},
                {"frame_id": "f2", "t_sec": 55, "entities": [{"entity_id": "e1", "confidence": 1}], "relations": [], "caption_context": "person leaves"},
            ],
        }
        frame_result = query_stage1_graph(stage1, "person", scope=[("00:00:50", "00:01:00")])
        self.assertEqual([item["frame_id"] for item in frame_result["matched_frames"]], ["f2"])
        self.assertIn("recommended_next_action", frame_result["controller_signals"])

        stage2 = {
            "version": "stage2_segment_v1", "video_id": "v", "stats": {},
            "segments": [
                {"segment_id": "s1", "t_start": 0, "t_end": 30, "active_entities": [], "temporal_relations": [], "summary": "person enters"},
                {"segment_id": "s2", "t_start": 50, "t_end": 80, "active_entities": [], "temporal_relations": [], "summary": "person leaves"},
            ],
        }
        segment_result = query_stage2_segment_graph(
            stage2, "person", scope=[("00:00:45", "00:01:30")], top_k_segments=2
        )
        self.assertEqual([item["segment_id"] for item in segment_result["matched_segments"]], ["s2"])
        self.assertIn("recommended_next_action", segment_result["controller_signals"])

    def test_global_query_returns_evidence_and_signals(self):
        graph = {
            "version": "global_event_v1", "video_id": "v", "stats": {},
            "entities": [{"entity_id": "e1", "canonical_name": "chef", "type": "person"}],
            "regions": [
                {"region_id": "r1", "t_start": 0, "t_end": 30, "support_segment_ids": ["s1"], "embedding": []},
                {"region_id": "r2", "t_start": 60, "t_end": 90, "support_segment_ids": ["s2"], "embedding": []},
            ],
            "relations": [
                {
                    "relation_id": "g1", "region_id": "r1", "relation_type": "causal",
                    "predicate": "causes_smoke", "participant_entity_ids": ["e1"],
                    "description": "The chef overheats the pan, causing smoke.",
                    "t_start": 0, "t_end": 30, "support_segment_ids": ["s1"],
                    "confidence": 0.9, "uncertainty": 0.1,
                }
            ],
            "contains_links": [{"relation_id": "g1", "segment_id": "s1"}],
        }
        result = query_global_event_graph(graph, "why is there smoke?", top_k_regions=1)
        self.assertEqual(result["matched_regions"][0]["region_id"], "r1")
        self.assertEqual(result["top_relations"][0]["relation_id"], "g1")
        self.assertEqual(result["controller_signals"]["recommended_next_action"], "segment_graph_query_tool")


if __name__ == "__main__":
    unittest.main()
