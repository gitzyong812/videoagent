"""Temporal-unit manifest builder (M1).

Emits a ``temporal_units.json`` document that grounds every fact, event, and
evidence pointer in stable frame IDs and time spans, per Alg-plan §9.2 +
dev-plan M1.

MVP scope:
    - Frames: extracted from a Stage1 scene graph (keyframe records).
    - Segment windows: deterministic sliding window over the video duration,
      independent of caption signals.
    - Micro-clip units are deferred to v2.0 full (no current consumer in MVP).

The manifest must not embed inference-visible caption text. Frame paths are
kept because they are bookkeeping pointers, not natural-language signals;
audit (M0) does not flag them.
"""

from __future__ import annotations

from typing import Any

from sgd.scene_graph.records import secs_to_hhmmss
from sgd.scene_graph.schema import safe_load_stage1_graph, utc_now_iso


DEFAULT_SEGMENT_WINDOW_SEC = 30.0
DEFAULT_SEGMENT_OVERLAP_SEC = 5.0


def iter_segment_windows(
    max_t_sec: float,
    *,
    window_sec: float = DEFAULT_SEGMENT_WINDOW_SEC,
    overlap_sec: float = DEFAULT_SEGMENT_OVERLAP_SEC,
) -> list[dict[str, Any]]:
    """Enumerate deterministic sliding windows over [0, max_t_sec)."""
    if max_t_sec <= 0:
        return []

    window_sec = max(float(window_sec), 1.0)
    overlap_sec = max(0.0, float(overlap_sec))
    if overlap_sec >= window_sec:
        overlap_sec = window_sec * 0.5

    stride = window_sec - overlap_sec
    windows: list[dict[str, Any]] = []
    idx = 1
    start = 0.0
    while start < max_t_sec:
        end = min(start + window_sec, max_t_sec)
        windows.append(
            {
                "window_id": f"win_{idx:04d}",
                "t_start_secs": round(start, 3),
                "t_end_secs": round(end, 3),
                "t_start": secs_to_hhmmss(start),
                "t_end": secs_to_hhmmss(end),
            }
        )
        if end >= max_t_sec:
            break
        start += stride
        idx += 1
    return windows


def build_temporal_units(
    stage1_graph: dict[str, Any],
    *,
    window_sec: float = DEFAULT_SEGMENT_WINDOW_SEC,
    overlap_sec: float = DEFAULT_SEGMENT_OVERLAP_SEC,
    include_frame_paths: bool = True,
) -> dict[str, Any]:
    """Build a temporal-unit manifest from a Stage1 graph dict."""
    frames_in = list(stage1_graph.get("frames", []) or [])
    frames_in.sort(key=lambda f: float(f.get("t_sec", 0.0)))

    frames_out: list[dict[str, Any]] = []
    max_t_sec = 0.0
    for frame in frames_in:
        frame_id = str(frame.get("frame_id", ""))
        if not frame_id:
            continue
        t_sec = float(frame.get("t_sec", 0.0))
        max_t_sec = max(max_t_sec, t_sec)
        entry: dict[str, Any] = {
            "frame_id": frame_id,
            "frame_idx": int(frame.get("frame_idx", 0)),
            "t_sec": round(t_sec, 3),
            "t": secs_to_hhmmss(t_sec),
        }
        if include_frame_paths and frame.get("path"):
            entry["path"] = str(frame["path"])
        frames_out.append(entry)

    windows = iter_segment_windows(
        max_t_sec, window_sec=window_sec, overlap_sec=overlap_sec
    )

    return {
        "version": "temporal_units_v1",
        "video_id": str(stage1_graph.get("video_id", "")),
        "created_at": utc_now_iso(),
        "config": {
            "window_sec": window_sec,
            "overlap_sec": overlap_sec,
            "include_frame_paths": include_frame_paths,
        },
        "frames": frames_out,
        "micro_clips": [],  # deferred to v2.0 full
        "segment_windows": windows,
        "stats": {
            "num_frames": len(frames_out),
            "num_segment_windows": len(windows),
            "max_t_sec": round(max_t_sec, 3),
        },
    }


def build_temporal_units_from_path(stage1_path: str) -> dict[str, Any] | None:
    graph = safe_load_stage1_graph(stage1_path)
    if graph is None:
        return None
    return build_temporal_units(graph)
