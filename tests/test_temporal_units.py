"""Tests for the temporal-unit manifest builder (M1)."""

from __future__ import annotations

from sgd.scene_graph.temporal_units import (
    build_temporal_units,
    iter_segment_windows,
)


def _stage1_graph_with_frames() -> dict:
    return {
        "version": "stage1_v1",
        "video_id": "vid_test",
        "entities": [],
        "frames": [
            {"frame_id": "f_a", "frame_idx": 10, "t_sec": 5.0, "path": "/p/f_a.jpg"},
            {"frame_id": "f_b", "frame_idx": 20, "t_sec": 10.0, "path": "/p/f_b.jpg"},
            {"frame_id": "f_c", "frame_idx": 40, "t_sec": 20.0, "path": "/p/f_c.jpg"},
        ],
        "stats": {},
    }


def test_frames_sorted_by_time_with_hhmmss():
    manifest = build_temporal_units(_stage1_graph_with_frames())
    times = [f["t"] for f in manifest["frames"]]
    assert times == ["00:00:05.000", "00:00:10.000", "00:00:20.000"]
    assert manifest["stats"]["num_frames"] == 3
    assert manifest["stats"]["max_t_sec"] == 20.0


def test_segment_windows_cover_full_duration():
    manifest = build_temporal_units(
        _stage1_graph_with_frames(), window_sec=10, overlap_sec=2
    )
    windows = manifest["segment_windows"]
    assert windows
    assert windows[0]["t_start_secs"] == 0.0
    assert windows[-1]["t_end_secs"] >= 20.0


def test_iter_segment_windows_handles_zero_duration():
    assert iter_segment_windows(0.0) == []


def test_iter_segment_windows_overlap_normalization():
    # overlap >= window must be normalized down so we make progress.
    windows = iter_segment_windows(100.0, window_sec=10, overlap_sec=15)
    assert len(windows) > 1


def test_manifest_no_caption_fields():
    manifest = build_temporal_units(_stage1_graph_with_frames())
    flat = repr(manifest).lower()
    for forbidden in ("caption", "transcript", "narration", "summary"):
        assert forbidden not in flat


def test_manifest_drops_paths_when_disabled():
    manifest = build_temporal_units(
        _stage1_graph_with_frames(), include_frame_paths=False
    )
    for frame in manifest["frames"]:
        assert "path" not in frame


def test_manifest_top_level_shape():
    manifest = build_temporal_units(_stage1_graph_with_frames())
    assert manifest["version"] == "temporal_units_v1"
    assert manifest["video_id"] == "vid_test"
    assert "frames" in manifest and "segment_windows" in manifest
    assert manifest["micro_clips"] == []
