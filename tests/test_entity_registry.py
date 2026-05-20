"""Tests for the v2 entity registry builder (M2)."""

from __future__ import annotations

from sgd.scene_graph.entity_registry import (
    build_entity_record,
    build_entity_registry,
    build_entity_registry_dict,
    strict_safe_aliases,
    _is_narrative_phrase,
)
from sgd.scene_graph.records import EntityRecord


def _stage1_graph_with_one_entity() -> dict:
    return {
        "version": "stage1_v1",
        "video_id": "vid_test",
        "entities": [
            {
                "entity_id": "ent_000001",
                "canonical_name": "person",
                "aliases": ["man with mustache", "tall man"],
                "type": "person",
                "attributes": ["tall"],
                "first_seen": 12.5,
                "last_seen": 75.25,
                "desc_text": "tall man",
                "confidence": 0.83,
                "evidence": ["f_000050", "f_000125", "f_000301"],
            }
        ],
        "frames": [
            {"frame_id": "f_000050", "t_sec": 25.0, "entities": [], "relations": []},
            {"frame_id": "f_000125", "t_sec": 62.5, "entities": [], "relations": []},
            {"frame_id": "f_000301", "t_sec": 150.5, "entities": [], "relations": []},
        ],
        "stats": {},
    }


def test_canonical_category_word_is_graph_attribute_alias():
    graph = _stage1_graph_with_one_entity()
    records = build_entity_registry(graph)
    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, EntityRecord)
    assert rec.entity_id == "ent_000001"
    assert rec.category == "person"
    # Canonical "person" is a category noun -> graph_attribute provenance.
    canonical_alias = rec.aliases[0]
    assert canonical_alias.text == "person"
    assert canonical_alias.source_provenance == "graph_attribute"
    assert canonical_alias.strict_safe is True


def test_non_narrative_alias_is_strict_safe_vlm_structured():
    graph = _stage1_graph_with_one_entity()
    rec = build_entity_registry(graph)[0]
    # "man with mustache" is short, non-narrative.
    aliases_by_text = {a.text: a for a in rec.aliases}
    a = aliases_by_text["man with mustache"]
    assert a.source_provenance == "vlm_structured"
    assert a.strict_safe is True


def test_narrative_alias_is_marked_not_strict_safe():
    graph = _stage1_graph_with_one_entity()
    graph["entities"][0]["aliases"].append(
        "the man who argues with the woman after entering the room"
    )
    rec = build_entity_registry(graph)[0]
    aliases_by_text = {a.text: a for a in rec.aliases}
    narrative = aliases_by_text[
        "the man who argues with the woman after entering the room"
    ]
    assert narrative.strict_safe is False
    # It is still recorded (for offline / debugging), just not strict-safe.
    assert narrative.source_provenance == "vlm_structured"


def test_first_seen_last_seen_are_hhmmss():
    graph = _stage1_graph_with_one_entity()
    rec = build_entity_registry(graph)[0]
    assert rec.first_seen == "00:00:12.500"
    assert rec.last_seen == "00:01:15.250"


def test_support_frame_ids_are_sorted_by_timestamp():
    graph = _stage1_graph_with_one_entity()
    # Shuffle the evidence list to confirm sorting.
    graph["entities"][0]["evidence"] = ["f_000301", "f_000050", "f_000125"]
    rec = build_entity_registry(graph)[0]
    assert rec.support_frame_ids == ["f_000050", "f_000125", "f_000301"]


def test_strict_safe_aliases_filter():
    graph = _stage1_graph_with_one_entity()
    graph["entities"][0]["aliases"].append("the villain who later breaks the mirror")
    registry_dict = build_entity_registry_dict(graph)
    ent_dict = registry_dict["entities"][0]
    safe = strict_safe_aliases(ent_dict)
    texts = {a["text"] for a in safe}
    assert "the villain who later breaks the mirror" not in texts
    assert "man with mustache" in texts


def test_registry_dict_can_strip_unsafe_aliases_for_inference():
    graph = _stage1_graph_with_one_entity()
    graph["entities"][0]["aliases"].append("the villain who later breaks the mirror")
    registry_dict = build_entity_registry_dict(graph, include_unsafe_aliases=False)
    texts = {a["text"] for a in registry_dict["entities"][0]["aliases"]}
    assert "the villain who later breaks the mirror" not in texts
    assert "man with mustache" in texts


def test_narrative_phrase_detector_examples():
    # Allowed strict examples per dev-plan M2.
    assert _is_narrative_phrase("person") is False
    assert _is_narrative_phrase("man in black sweater") is False
    assert _is_narrative_phrase("man with mustache") is False
    assert _is_narrative_phrase("red car") is False
    # Forbidden strict examples.
    assert (
        _is_narrative_phrase(
            "the man who argues with the woman after entering the room"
        )
        is True
    )
    assert _is_narrative_phrase("the villain who later breaks the mirror") is True


def test_unknown_category_falls_back_to_object():
    graph = _stage1_graph_with_one_entity()
    graph["entities"][0]["type"] = "spaceship"  # not in vocabulary
    rec = build_entity_registry(graph)[0]
    assert rec.category == "object"


def test_registry_dict_top_level_shape():
    graph = _stage1_graph_with_one_entity()
    registry_dict = build_entity_registry_dict(graph)
    assert registry_dict["version"] == "entity_registry_v1"
    assert registry_dict["video_id"] == "vid_test"
    assert registry_dict["stats"]["num_entities"] == 1
    assert registry_dict["stats"]["num_strict_safe_aliases"] >= 1


def test_skips_entities_missing_required_fields():
    graph = _stage1_graph_with_one_entity()
    graph["entities"].append({"canonical_name": "ghost"})  # no entity_id
    graph["entities"].append({"entity_id": "ent_000002"})  # no canonical_name
    records = build_entity_registry(graph)
    assert len(records) == 1
