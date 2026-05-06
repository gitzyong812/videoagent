"""Tests for the Global Event Index (GEI) module."""

import json
import os
import tempfile

import pytest

from sgd.scene_graph.global_event_index import (
    build_global_event_index,
    build_global_event_index_from_path,
    _build_entity_timeline,
    _extract_event_timeline,
    _merge_entity_spans,
    _format_ts,
    _tokenize_for_keywords,
    _cluster_similarity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_segment(seg_id, t_start, t_end, entities, relations=None, caption_snippets=None):
    """Helper to build a minimal Stage 2 segment dict."""
    seg = {
        "segment_id": seg_id,
        "t_start": t_start,
        "t_end": t_end,
        "summary": f"segment {seg_id}",
        "active_entities": [
            {"entity_id": eid, "canonical_name": name, "type": etype, "confidence": 0.9}
            for eid, name, etype in entities
        ],
        "temporal_relations": relations or [],
        "support_frame_ids": [],
    }
    if caption_snippets:
        seg["caption_snippets"] = caption_snippets
    return seg


def _make_relation(subj, pred, obj, t_start=0, t_end=0):
    return {
        "s": subj,
        "s_name": subj,
        "p": pred,
        "o": obj,
        "o_name": obj,
        "t_start": t_start,
        "t_end": t_end,
        "confidence": 0.8,
    }


SAMPLE_GRAPH = {
    "version": "stage2_segment_v1",
    "video_id": "test_video",
    "created_at": "2026-01-01T00:00:00+00:00",
    "config": {},
    "segments": [
        _make_segment(
            "seg_0", 0, 30,
            [("e1", "Chef", "person"), ("e2", "Kitchen", "location")],
            [_make_relation("Chef", "cooking_in", "Kitchen", 0, 30)],
            caption_snippets=["The chef prepares a delicious pasta dish in the kitchen"],
        ),
        _make_segment(
            "seg_1", 25, 55,
            [("e1", "Chef", "person"), ("e3", "Sous Chef", "person")],
            [_make_relation("Chef", "talking_to", "Sous Chef", 25, 55)],
            caption_snippets=["Chef discusses the recipe with the sous chef"],
        ),
        _make_segment(
            "seg_2", 55, 85,
            [("e3", "Sous Chef", "person"), ("e4", "Oven", "object")],
            [_make_relation("Sous Chef", "opening", "Oven", 55, 85)],
            caption_snippets=["The sous chef opens the oven to check the roast"],
        ),
        _make_segment(
            "seg_3", 120, 150,
            [("e5", "Waiter", "person"), ("e6", "Dining Room", "location")],
            [_make_relation("Waiter", "serving_in", "Dining Room", 120, 150)],
            caption_snippets=["A waiter serves plates in the elegant dining room"],
        ),
    ],
    "stats": {
        "num_segments": 4,
        "num_temporal_relations": 4,
        "num_segment_entities": 6,
    },
}


def _make_large_graph(num_segments=120):
    """Build a graph with many segments to test compression behavior."""
    segments = []
    for i in range(num_segments):
        t_start = i * 10
        t_end = t_start + 30  # overlapping windows
        # Rotate through 5 entity groups to create natural clusters
        group = i % 20 // 4  # 5 groups of 4 segments each, repeating
        entities = [
            (f"e{group * 2}", f"Person_{group}", "person"),
            (f"e{group * 2 + 1}", f"Object_{group}", "object"),
        ]
        relations = [
            _make_relation(f"Person_{group}", f"action_{group}", f"Object_{group}", t_start, t_end),
        ]
        segments.append(_make_segment(
            f"seg_{i}", t_start, t_end, entities, relations,
            caption_snippets=[f"Person {group} interacts with object {group} in scene {i}"],
        ))
    return {
        "version": "stage2_segment_v1",
        "video_id": "large_video",
        "created_at": "2026-01-01T00:00:00+00:00",
        "config": {},
        "segments": segments,
        "stats": {"num_segments": num_segments, "num_temporal_relations": num_segments, "num_segment_entities": num_segments * 2},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFormatTs:
    def test_seconds_only(self):
        assert _format_ts(45) == "00:45"

    def test_minutes(self):
        assert _format_ts(125) == "02:05"

    def test_hours(self):
        assert _format_ts(3661) == "01:01:01"

    def test_zero(self):
        assert _format_ts(0) == "00:00"


class TestTokenize:
    def test_filters_stop_words(self):
        tokens = _tokenize_for_keywords("the person is walking in a park")
        assert "the" not in tokens
        assert "person" not in tokens  # in stop list
        assert "walking" in tokens
        assert "park" in tokens

    def test_min_length(self):
        tokens = _tokenize_for_keywords("a big red car on road")
        assert "big" in tokens
        assert "red" in tokens
        assert "car" in tokens


class TestClusterSimilarity:
    def test_identical_segments(self):
        seg = _make_segment("s1", 0, 10, [("e1", "X", "person")], [_make_relation("X", "holds", "Y")])
        assert _cluster_similarity(seg, seg) == pytest.approx(1.0)

    def test_disjoint_segments(self):
        seg_a = _make_segment("s1", 0, 10, [("e1", "X", "person")], [_make_relation("X", "holds", "Y")])
        seg_b = _make_segment("s2", 100, 110, [("e2", "Z", "person")], [_make_relation("Z", "runs", "W")])
        sim = _cluster_similarity(seg_a, seg_b)
        assert sim == pytest.approx(0.0)


class TestEntityTimeline:
    def test_basic_timeline(self):
        segments = SAMPLE_GRAPH["segments"]
        timeline = _build_entity_timeline(segments)
        assert "e1" in timeline
        assert len(timeline["e1"]) == 2
        assert timeline["e1"][0]["t_start"] == 0
        assert timeline["e1"][1]["t_start"] == 25

    def test_entity_sorted_by_time(self):
        segments = SAMPLE_GRAPH["segments"]
        timeline = _build_entity_timeline(segments)
        for eid, appearances in timeline.items():
            for i in range(len(appearances) - 1):
                assert appearances[i]["t_start"] <= appearances[i + 1]["t_start"]


class TestMergeEntitySpans:
    def test_merge_overlapping(self):
        appearances = [
            {"t_start": 0, "t_end": 30},
            {"t_start": 25, "t_end": 55},
        ]
        spans = _merge_entity_spans(appearances, merge_gap_sec=5.0)
        assert len(spans) == 1
        assert spans[0] == (0, 55)

    def test_separate_spans(self):
        appearances = [
            {"t_start": 0, "t_end": 30},
            {"t_start": 120, "t_end": 150},
        ]
        spans = _merge_entity_spans(appearances, merge_gap_sec=5.0)
        assert len(spans) == 2

    def test_empty(self):
        assert _merge_entity_spans([], 5.0) == []


class TestExtractEventTimeline:
    def test_clusters_adjacent_segments(self):
        segments = SAMPLE_GRAPH["segments"]
        events = _extract_event_timeline(segments, merge_gap_sec=5.0, max_events=20)
        # seg_0, seg_1, seg_2 are temporally close and share entities
        # seg_3 is far away (120s) — should be a separate event
        assert len(events) >= 2
        assert events[-1]["t_start"] >= 100

    def test_max_events_merges(self):
        segments = SAMPLE_GRAPH["segments"]
        events = _extract_event_timeline(segments, merge_gap_sec=5.0, max_events=1)
        assert len(events) == 1
        assert events[0]["t_start"] == 0
        assert events[0]["t_end"] == 150

    def test_events_have_keywords(self):
        segments = SAMPLE_GRAPH["segments"]
        events = _extract_event_timeline(segments, merge_gap_sec=5.0, max_events=20)
        for ev in events:
            assert "keywords" in ev
            assert isinstance(ev["keywords"], list)

    def test_large_graph_compression(self):
        """120 segments should compress to <= max_events."""
        graph = _make_large_graph(120)
        events = _extract_event_timeline(graph["segments"], merge_gap_sec=5.0, max_events=12)
        assert len(events) <= 12
        # Verify events cover the full time range
        assert events[0]["t_start"] == 0
        assert events[-1]["t_end"] >= 1190  # 119 * 10 + 30

    def test_compression_preserves_ordering(self):
        """Events must be in chronological order after compression."""
        graph = _make_large_graph(80)
        events = _extract_event_timeline(graph["segments"], merge_gap_sec=5.0, max_events=10)
        for i in range(len(events) - 1):
            assert events[i]["t_start"] <= events[i + 1]["t_start"]


class TestBuildGlobalEventIndex:
    def test_basic_output_structure(self):
        gei = build_global_event_index(SAMPLE_GRAPH)
        assert "event_timeline" in gei
        assert "entity_continuity" in gei
        assert "prompt_text" in gei
        assert isinstance(gei["prompt_text"], str)

    def test_prompt_text_contains_timeline(self):
        gei = build_global_event_index(SAMPLE_GRAPH)
        text = gei["prompt_text"]
        assert "VIDEO TIMELINE" in text
        assert "RECURRING ENTITIES" in text
        assert "Chef" in text

    def test_prompt_text_contains_seg_counts(self):
        gei = build_global_event_index(SAMPLE_GRAPH)
        text = gei["prompt_text"]
        assert "seg)" in text  # e.g. "(3seg)"

    def test_prompt_text_respects_char_limit(self):
        gei = build_global_event_index(SAMPLE_GRAPH, max_chars=100)
        assert len(gei["prompt_text"]) <= 100

    def test_empty_graph(self):
        empty = {
            "version": "stage2_segment_v1",
            "video_id": "empty",
            "created_at": "",
            "config": {},
            "segments": [],
            "stats": {"num_segments": 0, "num_temporal_relations": 0, "num_segment_entities": 0},
        }
        gei = build_global_event_index(empty)
        assert gei["prompt_text"] == ""
        assert gei["event_timeline"] == []

    def test_entity_continuity_includes_major_entities(self):
        gei = build_global_event_index(SAMPLE_GRAPH)
        names = [ent["name"] for ent in gei["entity_continuity"]]
        assert "Chef" in names
        assert "Sous Chef" in names

    def test_large_graph_prompt_within_budget(self):
        """120 segment graph should produce a prompt within char limit."""
        graph = _make_large_graph(120)
        gei = build_global_event_index(graph, max_chars=2400)
        assert len(gei["prompt_text"]) <= 2400
        assert len(gei["event_timeline"]) <= 12

    def test_discriminative_keywords_in_prompt(self):
        """Keywords from captions should appear in prompt text."""
        gei = build_global_event_index(SAMPLE_GRAPH)
        text = gei["prompt_text"].lower()
        # At least some caption-derived keywords should surface
        caption_words = {"pasta", "recipe", "oven", "roast", "waiter", "dining", "elegant"}
        found = {w for w in caption_words if w in text}
        assert len(found) >= 2, f"Expected >=2 caption keywords in prompt, found: {found}"


class TestBuildFromPath:
    def test_load_from_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(SAMPLE_GRAPH, f)
            path = f.name
        try:
            gei = build_global_event_index_from_path(path)
            assert "VIDEO TIMELINE" in gei["prompt_text"]
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty(self):
        gei = build_global_event_index_from_path("/nonexistent/path.json")
        assert gei["prompt_text"] == ""

    def test_invalid_json_returns_empty(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json")
            path = f.name
        try:
            gei = build_global_event_index_from_path(path)
            assert gei["prompt_text"] == ""
        finally:
            os.unlink(path)
