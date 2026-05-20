"""Tests for the v2 frame-fact extractor (M3)."""

from __future__ import annotations

from sgd.scene_graph.frame_facts import build_frame_facts, build_frame_facts_dict
from sgd.scene_graph.records import GraphFact


def _stage1_graph_with_frames() -> dict:
    return {
        "version": "stage1_v1",
        "video_id": "vid_test",
        "entities": [
            {"entity_id": "ent_p", "canonical_name": "person", "type": "person"},
            {"entity_id": "ent_c", "canonical_name": "cup", "type": "object"},
        ],
        "frames": [
            {
                "frame_id": "f_a",
                "t_sec": 10.0,
                "entities": [
                    {"entity_id": "ent_p", "name": "person", "type": "person", "confidence": 0.9},
                    {"entity_id": "ent_c", "name": "cup", "type": "object", "confidence": 0.8},
                ],
                "relations": [
                    {"s": "ent_p", "p": "holds", "o": "ent_c", "confidence": 0.85},
                ],
            },
            {
                "frame_id": "f_b",
                "t_sec": 12.5,
                "entities": [
                    {"entity_id": "ent_p", "name": "person", "type": "person", "confidence": 0.91},
                ],
                "relations": [],
            },
        ],
        "stats": {},
    }


def test_emits_relation_and_presence_facts():
    facts = build_frame_facts(_stage1_graph_with_frames())
    types = [f.fact_type for f in facts]
    assert types.count("entity_presence") == 3  # 2 in f_a + 1 in f_b
    assert types.count("relation") == 1


def test_relation_fact_carries_all_required_fields():
    facts = build_frame_facts(_stage1_graph_with_frames())
    relation_facts = [f for f in facts if f.fact_type == "relation"]
    assert len(relation_facts) == 1
    fact = relation_facts[0]
    assert isinstance(fact, GraphFact)
    assert fact.subject == "ent_p"
    assert fact.predicate == "holds"
    assert fact.object == "ent_c"
    assert fact.time_span == ("00:00:10.000", "00:00:10.000")
    assert fact.evidence == ["f_a"]
    assert fact.confidence == 0.85
    assert fact.source_provenance == "vlm_structured"
    assert fact.status == "unverified"
    assert fact.fact_id.startswith("fact_")


def test_facts_are_temporally_ordered_and_have_unique_ids():
    graph = _stage1_graph_with_frames()
    # Add a frame earlier in time to ensure sort happens.
    graph["frames"].insert(
        0,
        {
            "frame_id": "f_pre",
            "t_sec": 1.0,
            "entities": [
                {"entity_id": "ent_p", "name": "person", "type": "person", "confidence": 0.7}
            ],
            "relations": [],
        },
    )
    facts = build_frame_facts(graph)
    timestamps = [f.time_span[0] for f in facts]
    assert timestamps == sorted(timestamps)
    ids = [f.fact_id for f in facts]
    assert len(ids) == len(set(ids))


def test_include_presence_false_drops_presence_facts():
    facts = build_frame_facts(_stage1_graph_with_frames(), include_presence=False)
    assert all(f.fact_type == "relation" for f in facts)


def test_drops_invalid_relations():
    graph = _stage1_graph_with_frames()
    # Add several malformed relations that must be dropped silently.
    graph["frames"][0]["relations"].extend(
        [
            {"s": "", "p": "holds", "o": "ent_c", "confidence": 0.5},
            {"s": "ent_p", "p": "", "o": "ent_c", "confidence": 0.5},
            {"s": "ent_p", "p": "holds", "o": "", "confidence": 0.5},
        ]
    )
    facts = build_frame_facts(graph, include_presence=False)
    assert len(facts) == 1


def test_frame_facts_dict_top_level_shape():
    out = build_frame_facts_dict(_stage1_graph_with_frames())
    assert out["version"] == "frame_facts_v1"
    assert out["video_id"] == "vid_test"
    assert out["stats"]["num_facts"] == len(out["facts"])
    assert "entity_presence" in out["stats"]["by_type"]
    assert "relation" in out["stats"]["by_type"]
    assert out["stats"]["by_provenance"]["vlm_structured"] >= 1
    assert out["stats"]["by_status"]["unverified"] == out["stats"]["num_facts"]


def test_each_emitted_fact_dict_has_required_keys():
    out = build_frame_facts_dict(_stage1_graph_with_frames())
    required = {
        "fact_id",
        "fact_type",
        "subject",
        "predicate",
        "object",
        "time_span",
        "evidence",
        "confidence",
        "source",
        "source_provenance",
        "status",
    }
    for fact in out["facts"]:
        assert required.issubset(fact.keys()), fact
