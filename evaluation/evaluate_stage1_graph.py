#!/usr/bin/env python3
"""
Evaluate Stage-1/Stage-2 scene graph quality and retrieval metrics.

This script supports:
1) Building Stage-1 graphs from assets database.json + artifact captions (caption fallback mode).
2) Building Stage-2 segment graphs from Stage-1 graphs.
3) Stage-1 graph quality + retrieval metrics against QA time_reference.
4) Stage-2 segment quality + retrieval metrics against QA time_reference.
5) Optional baseline QA accuracy parsing from trace logs (answer_check events).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sgd.config as config
from sgd.path_config import artifacts_root, badcases_root, eval_output_root, lvbench_root
from sgd.scene_graph.query import query_stage1_graph
from sgd.scene_graph.query_segment import query_stage2_segment_graph
from sgd.scene_graph.schema import (safe_load_stage1_graph,
                                    safe_load_stage2_segment_graph)
from sgd.scene_graph.segment_builder import build_stage2_segment_graph
from sgd.scene_graph.stage1_builder import build_stage1_scene_graph


DEFAULT_ARTIFACTS_ROOT = artifacts_root()
DEFAULT_ASSETS_ROOT = lvbench_root()
DEFAULT_META_PATH = DEFAULT_ASSETS_ROOT / "video_info.meta.jsonl"
DEFAULT_STAGE1_REPORT_PATH = eval_output_root() / "stage12_eval_report.json"


def _clean_question_text(question: str) -> str:
    if not question:
        return ""
    # Remove options to keep the semantic query concise.
    return re.sub(r"\n\s*\(A\).*", "", question, flags=re.DOTALL).strip()


def _hhmmss_to_seconds(value: str) -> float:
    value = value.strip()
    parts = value.split(":")
    if len(parts) == 2:
        h = 0
        m = int(parts[0])
        s = float(parts[1])
    elif len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
    else:
        raise ValueError(f"Invalid time value: {value}")
    return h * 3600 + m * 60 + s


def _parse_time_reference_ranges(time_ref: str) -> list[tuple[float, float]]:
    if not time_ref:
        return []
    matches = re.findall(
        r"(\d{1,2}:\d{2}(?::\d{1,2}(?:\.\d+)?)?)\s*-\s*(\d{1,2}:\d{2}(?::\d{1,2}(?:\.\d+)?)?)",
        time_ref,
    )
    ranges: list[tuple[float, float]] = []
    for start, end in matches:
        try:
            s = _hhmmss_to_seconds(start)
            e = _hhmmss_to_seconds(end)
        except Exception:
            continue
        if e < s:
            s, e = e, s
        ranges.append((s, e))
    return ranges


def _distance_to_ranges(t_sec: float, ranges: list[tuple[float, float]]) -> float:
    if not ranges:
        return float("inf")
    best = float("inf")
    for start, end in ranges:
        if start <= t_sec <= end:
            return 0.0
        if t_sec < start:
            best = min(best, start - t_sec)
        else:
            best = min(best, t_sec - end)
    return best


def _load_meta(meta_path: str) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("key")
            if key:
                meta[key] = row
    return meta


def _strip_caption_prefix(caption: str) -> str:
    caption = caption or ""
    # Remove prefix like: [From 00:00:0.0 to 00:00:5.0 seconds]
    return re.sub(r"^\[From .*?seconds\]\s*", "", caption, flags=re.IGNORECASE).strip()


def _build_caption_json_from_database_json(database_json_path: str, output_caption_json_path: str) -> None:
    with open(database_json_path, "r", encoding="utf-8") as f:
        db = json.load(f)
    rows = db.get("data", [])
    captions: dict[str, dict[str, str]] = {}
    for row in rows:
        start = row.get("time_start_secs")
        end = row.get("time_end_secs")
        caption = _strip_caption_prefix(str(row.get("caption", "")))
        if start is None or end is None:
            continue
        key = f"{float(start)}_{float(end)}"
        captions[key] = {"caption": caption}
    os.makedirs(os.path.dirname(output_caption_json_path), exist_ok=True)
    with open(output_caption_json_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)


def _find_trace_accuracy(trace_dir: str, video_key: str, qa_uids: set[str]) -> dict[str, bool]:
    """
    Parse answer_check from trace jsonl:
      trace filename pattern: <video_key>__<uid>.jsonl
    """
    result: dict[str, bool] = {}
    if not trace_dir or not os.path.isdir(trace_dir):
        return result
    pattern = os.path.join(trace_dir, f"{video_key}__*.jsonl")
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        uid_match = re.match(rf"{re.escape(video_key)}__(.+)\.jsonl$", base)
        if not uid_match:
            continue
        uid = uid_match.group(1)
        if uid not in qa_uids:
            continue
        is_correct = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue
                if event.get("event_type") == "answer_check":
                    is_correct = bool(event.get("is_correct"))
        if is_correct is not None:
            result[uid] = is_correct
    return result


@dataclass
class VideoEvalResult:
    video_key: str
    stage1_graph_path: str | None
    stage1_graph_stats: dict[str, Any]
    stage1_retrieval_stats: dict[str, Any]
    stage2_graph_path: str | None
    stage2_graph_stats: dict[str, Any]
    stage2_retrieval_stats: dict[str, Any]
    baseline_stats: dict[str, Any]


def _compute_graph_quality(graph: dict[str, Any]) -> dict[str, Any]:
    entities = graph.get("entities", [])
    frames = graph.get("frames", [])
    stats = graph.get("stats", {})

    num_entities = len(entities)
    num_frames = len(frames)
    num_relations = int(stats.get("num_relations", 0))

    by_name: defaultdict[str, int] = defaultdict(int)
    for ent in entities:
        name = str(ent.get("canonical_name", "")).strip().lower()
        if name:
            by_name[name] += 1
    duplicate_entities = sum(max(0, cnt - 1) for cnt in by_name.values())
    entity_duplication_rate = (duplicate_entities / num_entities) if num_entities else 0.0

    entity_ids = {str(ent.get("entity_id")) for ent in entities}
    allowed_predicates = set(getattr(config, "SCENE_GRAPH_STAGE1_ALLOWED_PREDICATES", []))
    invalid_rel_count = 0
    total_rel_count = 0
    empty_frame_count = 0
    total_entities_in_frames = 0

    for frame in frames:
        frame_entities = frame.get("entities", [])
        frame_relations = frame.get("relations", [])
        total_entities_in_frames += len(frame_entities)
        if not frame_entities:
            empty_frame_count += 1

        for rel in frame_relations:
            total_rel_count += 1
            s = str(rel.get("s", ""))
            o = str(rel.get("o", ""))
            p = str(rel.get("p", ""))
            bad_endpoint = (s not in entity_ids) or (o not in entity_ids)
            bad_predicate = bool(allowed_predicates) and (p not in allowed_predicates)
            if bad_endpoint or bad_predicate:
                invalid_rel_count += 1

    avg_entities_per_frame = (total_entities_in_frames / num_frames) if num_frames else 0.0
    empty_frame_rate = (empty_frame_count / num_frames) if num_frames else 0.0
    invalid_relation_rate = (invalid_rel_count / total_rel_count) if total_rel_count else 0.0
    frame_times = [float(frame.get("t_sec", 0.0)) for frame in frames]

    return {
        "num_entities": num_entities,
        "num_frames": num_frames,
        "num_relations": num_relations,
        "entity_duplication_rate": entity_duplication_rate,
        "invalid_relation_rate": invalid_relation_rate,
        "avg_entities_per_frame": avg_entities_per_frame,
        "empty_frame_rate": empty_frame_rate,
        "frame_time_min_sec": min(frame_times) if frame_times else None,
        "frame_time_max_sec": max(frame_times) if frame_times else None,
    }


def _compute_retrieval_metrics(
    graph: dict[str, Any],
    qa_list: list[dict[str, Any]],
    top_k: int,
    tolerance_sec: float,
) -> dict[str, Any]:
    evaluated = 0
    hit_at_1 = 0
    hit_at_k = 0
    non_empty_results = 0
    distances: list[float] = []

    per_question: list[dict[str, Any]] = []

    for qa in qa_list:
        uid = str(qa.get("uid", ""))
        question = str(qa.get("question", ""))
        query = _clean_question_text(question)
        ranges = _parse_time_reference_ranges(str(qa.get("time_reference", "")))
        if not query or not ranges:
            continue

        result = query_stage1_graph(graph=graph, query=query, top_k_frames=top_k, max_triples=24)
        matched_frames = result.get("matched_frames", [])

        evaluated += 1
        if matched_frames:
            non_empty_results += 1

        best_dist = float("inf")
        for item in matched_frames:
            t = float(item.get("t_sec", 0.0))
            d = _distance_to_ranges(t, ranges)
            best_dist = min(best_dist, d)

        if best_dist < float("inf"):
            distances.append(best_dist)

        hit1 = False
        if matched_frames:
            top1_t = float(matched_frames[0].get("t_sec", 0.0))
            hit1 = _distance_to_ranges(top1_t, ranges) <= tolerance_sec

        hitk = best_dist <= tolerance_sec

        if hit1:
            hit_at_1 += 1
        if hitk:
            hit_at_k += 1

        per_question.append(
            {
                "uid": uid,
                "question": query,
                "num_matches": len(matched_frames),
                "best_temporal_distance_sec": None if best_dist == float("inf") else best_dist,
                "hit_at_1": hit1,
                "hit_at_k": hitk,
            }
        )

    mean_distance = (sum(distances) / len(distances)) if distances else None
    return {
        "num_questions_evaluated": evaluated,
        "hit_at_1": (hit_at_1 / evaluated) if evaluated else 0.0,
        "hit_at_k": (hit_at_k / evaluated) if evaluated else 0.0,
        "non_empty_retrieval_rate": (non_empty_results / evaluated) if evaluated else 0.0,
        "mean_best_temporal_distance_sec": mean_distance,
        "per_question": per_question,
    }


def _distance_between_spans(
    a_start: float,
    a_end: float,
    ranges: list[tuple[float, float]],
) -> float:
    if not ranges:
        return float("inf")
    best = float("inf")
    for b_start, b_end in ranges:
        if not (a_end < b_start or b_end < a_start):
            return 0.0
        if a_end < b_start:
            best = min(best, b_start - a_end)
        elif b_end < a_start:
            best = min(best, a_start - b_end)
    return best


def _compute_segment_graph_quality(graph: dict[str, Any]) -> dict[str, Any]:
    segments = graph.get("segments", [])
    stats = graph.get("stats", {})
    num_segments = len(segments)
    num_temporal_relations = int(stats.get("num_temporal_relations", 0))
    num_segment_entities = int(stats.get("num_segment_entities", 0))

    invalid_temporal_rel_count = 0
    total_temporal_rel_count = 0
    empty_segment_count = 0

    sum_active_entities = 0
    sum_segment_relations = 0
    segment_time_bounds: list[tuple[float, float]] = []

    for seg in segments:
        t_start = float(seg.get("t_start", 0.0))
        t_end = float(seg.get("t_end", 0.0))
        segment_time_bounds.append((t_start, t_end))
        active_entities = seg.get("active_entities", [])
        temporal_relations = seg.get("temporal_relations", [])
        entity_ids = {str(e.get("entity_id", "")) for e in active_entities}

        if not active_entities:
            empty_segment_count += 1

        sum_active_entities += len(active_entities)
        sum_segment_relations += len(temporal_relations)

        for rel in temporal_relations:
            total_temporal_rel_count += 1
            s = str(rel.get("s", ""))
            o = str(rel.get("o", ""))
            rs = float(rel.get("t_start", t_start))
            re = float(rel.get("t_end", t_end))
            bad_endpoint = (s not in entity_ids) or (o not in entity_ids)
            bad_time = rs < t_start or re > t_end or re < rs
            if bad_endpoint or bad_time:
                invalid_temporal_rel_count += 1

    return {
        "num_segments": num_segments,
        "num_temporal_relations": num_temporal_relations,
        "num_segment_entities": num_segment_entities,
        "avg_active_entities_per_segment": (sum_active_entities / num_segments) if num_segments else 0.0,
        "avg_relations_per_segment": (sum_segment_relations / num_segments) if num_segments else 0.0,
        "empty_segment_rate": (empty_segment_count / num_segments) if num_segments else 0.0,
        "invalid_temporal_relation_rate": (
            invalid_temporal_rel_count / total_temporal_rel_count
            if total_temporal_rel_count
            else 0.0
        ),
        "segment_time_min_sec": min((x[0] for x in segment_time_bounds), default=None),
        "segment_time_max_sec": max((x[1] for x in segment_time_bounds), default=None),
    }


def _compute_segment_retrieval_metrics(
    graph: dict[str, Any],
    qa_list: list[dict[str, Any]],
    top_k: int,
    tolerance_sec: float,
) -> dict[str, Any]:
    evaluated = 0
    hit_at_1 = 0
    hit_at_k = 0
    non_empty_results = 0
    distances: list[float] = []
    per_question: list[dict[str, Any]] = []

    for qa in qa_list:
        uid = str(qa.get("uid", ""))
        question = str(qa.get("question", ""))
        query = _clean_question_text(question)
        ranges = _parse_time_reference_ranges(str(qa.get("time_reference", "")))
        if not query or not ranges:
            continue

        result = query_stage2_segment_graph(
            graph=graph,
            query=query,
            top_k_segments=top_k,
            max_triples=24,
        )
        matched_segments = result.get("matched_segments", [])
        evaluated += 1

        if matched_segments:
            non_empty_results += 1

        best_dist = float("inf")
        for item in matched_segments:
            s = float(item.get("t_start", 0.0))
            e = float(item.get("t_end", 0.0))
            d = _distance_between_spans(s, e, ranges)
            best_dist = min(best_dist, d)

        if best_dist < float("inf"):
            distances.append(best_dist)

        hit1 = False
        if matched_segments:
            top1 = matched_segments[0]
            hit1 = _distance_between_spans(
                float(top1.get("t_start", 0.0)),
                float(top1.get("t_end", 0.0)),
                ranges,
            ) <= tolerance_sec
        hitk = best_dist <= tolerance_sec

        if hit1:
            hit_at_1 += 1
        if hitk:
            hit_at_k += 1

        per_question.append(
            {
                "uid": uid,
                "question": query,
                "num_matches": len(matched_segments),
                "best_temporal_distance_sec": None if best_dist == float("inf") else best_dist,
                "hit_at_1": hit1,
                "hit_at_k": hitk,
            }
        )

    mean_distance = (sum(distances) / len(distances)) if distances else None
    return {
        "num_questions_evaluated": evaluated,
        "hit_at_1": (hit_at_1 / evaluated) if evaluated else 0.0,
        "hit_at_k": (hit_at_k / evaluated) if evaluated else 0.0,
        "non_empty_retrieval_rate": (non_empty_results / evaluated) if evaluated else 0.0,
        "mean_best_temporal_distance_sec": mean_distance,
        "per_question": per_question,
    }


def _evaluate_one_video(
    video_key: str,
    meta_row: dict[str, Any],
    assets_root: str,
    artifacts_root: str,
    build_if_missing: bool,
    force_rebuild: bool,
    use_vlm: bool | None,
    top_k: int,
    tolerance_sec: float,
    trace_dir: str | None,
    stage1_sample_secs: int | None,
    stage1_max_keyframes: int | None,
    enable_stage2: bool,
    top_k_segments: int,
    stage2_window_secs: int | None,
    stage2_overlap_secs: int | None,
) -> VideoEvalResult:
    asset_video_dir = os.path.join(assets_root, video_key)
    artifact_video_dir = os.path.join(artifacts_root, video_key)
    database_json_path = os.path.join(asset_video_dir, "database.json")
    frame_dir = os.path.join(asset_video_dir, "frames")
    captions_dir = os.path.join(artifact_video_dir, "captions")
    generated_caption_json_path = os.path.join(captions_dir, "captions.stage1_eval.json")
    base_caption_json_path = os.path.join(captions_dir, "captions.json")
    graph_path = os.path.join(captions_dir, "scene_graph_stage1.eval.json")
    segment_graph_path = os.path.join(captions_dir, "scene_graph_stage2.eval.json")

    if build_if_missing:
        caption_source_path = None
        if os.path.exists(generated_caption_json_path):
            caption_source_path = generated_caption_json_path
        elif os.path.exists(base_caption_json_path):
            caption_source_path = base_caption_json_path

        if caption_source_path is None and os.path.exists(database_json_path):
            os.makedirs(captions_dir, exist_ok=True)
            _build_caption_json_from_database_json(database_json_path, generated_caption_json_path)
            caption_source_path = generated_caption_json_path

        if force_rebuild or (not os.path.exists(graph_path)):
            if caption_source_path is None:
                return VideoEvalResult(
                    video_key=video_key,
                    stage1_graph_path=None,
                    stage1_graph_stats={"error": "missing_caption_and_database_for_build"},
                    stage1_retrieval_stats={"error": "missing_caption_and_database_for_build"},
                    stage2_graph_path=None,
                    stage2_graph_stats={"error": "missing_caption_and_database_for_build"},
                    stage2_retrieval_stats={"error": "missing_caption_and_database_for_build"},
                    baseline_stats={"error": "missing_caption_and_database_for_build"},
                )

            old_use_vlm = config.SCENE_GRAPH_STAGE1_USE_VLM
            old_fallback = config.SCENE_GRAPH_STAGE1_CAPTION_FALLBACK
            old_sample_secs = config.SCENE_GRAPH_STAGE1_SAMPLE_SECS
            old_max_keyframes = config.SCENE_GRAPH_STAGE1_MAX_KEYFRAMES
            try:
                if use_vlm is not None:
                    config.SCENE_GRAPH_STAGE1_USE_VLM = bool(use_vlm)
                config.SCENE_GRAPH_STAGE1_CAPTION_FALLBACK = True
                if stage1_sample_secs is not None and stage1_sample_secs > 0:
                    config.SCENE_GRAPH_STAGE1_SAMPLE_SECS = int(stage1_sample_secs)
                if stage1_max_keyframes is not None and stage1_max_keyframes > 0:
                    config.SCENE_GRAPH_STAGE1_MAX_KEYFRAMES = int(stage1_max_keyframes)
                build_stage1_scene_graph(
                    video_caption_json_path=caption_source_path,
                    output_path=graph_path,
                    force_rebuild=True,
                    frame_dir_override=(frame_dir if os.path.isdir(frame_dir) else None),
                )
            finally:
                config.SCENE_GRAPH_STAGE1_USE_VLM = old_use_vlm
                config.SCENE_GRAPH_STAGE1_CAPTION_FALLBACK = old_fallback
                config.SCENE_GRAPH_STAGE1_SAMPLE_SECS = old_sample_secs
                config.SCENE_GRAPH_STAGE1_MAX_KEYFRAMES = old_max_keyframes

    graph = safe_load_stage1_graph(graph_path) if os.path.exists(graph_path) else None
    if graph is None:
        return VideoEvalResult(
            video_key=video_key,
            stage1_graph_path=None,
            stage1_graph_stats={"error": "missing_or_invalid_graph"},
            stage1_retrieval_stats={"error": "missing_or_invalid_graph"},
            stage2_graph_path=None,
            stage2_graph_stats={"error": "missing_or_invalid_graph"},
            stage2_retrieval_stats={"error": "missing_or_invalid_graph"},
            baseline_stats={"error": "missing_or_invalid_graph"},
        )

    qa_list = list(meta_row.get("qa", []))
    graph_stats = _compute_graph_quality(graph)
    retrieval_stats = _compute_retrieval_metrics(
        graph=graph,
        qa_list=qa_list,
        top_k=top_k,
        tolerance_sec=tolerance_sec,
    )

    if enable_stage2 and (force_rebuild or not os.path.exists(segment_graph_path)):
        old_window = config.SCENE_GRAPH_STAGE2_WINDOW_SECS
        old_overlap = config.SCENE_GRAPH_STAGE2_OVERLAP_SECS
        try:
            if stage2_window_secs is not None and stage2_window_secs > 0:
                config.SCENE_GRAPH_STAGE2_WINDOW_SECS = int(stage2_window_secs)
            if stage2_overlap_secs is not None and stage2_overlap_secs >= 0:
                config.SCENE_GRAPH_STAGE2_OVERLAP_SECS = int(stage2_overlap_secs)
            build_stage2_segment_graph(
                stage1_graph_path=graph_path,
                output_path=segment_graph_path,
                force_rebuild=True,
            )
        finally:
            config.SCENE_GRAPH_STAGE2_WINDOW_SECS = old_window
            config.SCENE_GRAPH_STAGE2_OVERLAP_SECS = old_overlap

    segment_graph = (
        safe_load_stage2_segment_graph(segment_graph_path)
        if enable_stage2 and os.path.exists(segment_graph_path)
        else None
    )
    if enable_stage2 and segment_graph is not None:
        segment_graph_stats = _compute_segment_graph_quality(segment_graph)
        segment_retrieval_stats = _compute_segment_retrieval_metrics(
            graph=segment_graph,
            qa_list=qa_list,
            top_k=top_k_segments,
            tolerance_sec=tolerance_sec,
        )
    elif enable_stage2:
        segment_graph_stats = {"error": "missing_or_invalid_segment_graph"}
        segment_retrieval_stats = {"error": "missing_or_invalid_segment_graph"}
    else:
        segment_graph_stats = {"disabled": True}
        segment_retrieval_stats = {"disabled": True}

    qa_uids = {str(qa.get("uid", "")) for qa in qa_list}
    trace_acc_map = _find_trace_accuracy(trace_dir=trace_dir or "", video_key=video_key, qa_uids=qa_uids)
    num_trace = len(trace_acc_map)
    num_correct = sum(1 for v in trace_acc_map.values() if v)
    baseline_stats = {
        "num_trace_questions": num_trace,
        "num_trace_correct": num_correct,
        "trace_accuracy": (num_correct / num_trace) if num_trace else None,
    }

    return VideoEvalResult(
        video_key=video_key,
        stage1_graph_path=graph_path,
        stage1_graph_stats=graph_stats,
        stage1_retrieval_stats=retrieval_stats,
        stage2_graph_path=segment_graph_path if enable_stage2 else None,
        stage2_graph_stats=segment_graph_stats,
        stage2_retrieval_stats=segment_retrieval_stats,
        baseline_stats=baseline_stats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-1/Stage-2 scene graph metrics.")
    parser.add_argument(
        "--meta",
        default=str(DEFAULT_META_PATH),
        help="Meta jsonl path containing key/qa/time_reference.",
    )
    parser.add_argument(
        "--artifacts-root",
        default=str(DEFAULT_ARTIFACTS_ROOT),
        help="Artifact root folder containing per-video captions/graph folders.",
    )
    parser.add_argument(
        "--badcases-root",
        default="",
        help=(
            "Legacy alias for --artifacts-root. "
            "If both are set, --artifacts-root takes precedence."
        ),
    )
    parser.add_argument(
        "--assets-root",
        default=str(DEFAULT_ASSETS_ROOT),
        help="Asset root folder containing per-video frames/database folders.",
    )
    parser.add_argument(
        "--videos",
        default="",
        help="Comma-separated video keys. Empty means all available keys in meta.",
    )
    parser.add_argument("--max-videos", type=int, default=0, help="Limit evaluated videos (0 means all).")
    parser.add_argument("--build-if-missing", action="store_true", help="Build graph if missing.")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild graph.")
    vlm_group = parser.add_mutually_exclusive_group()
    vlm_group.add_argument(
        "--use-vlm",
        action="store_true",
        help="Force enable VLM in Stage-1 build (override config).",
    )
    vlm_group.add_argument(
        "--no-use-vlm",
        action="store_true",
        help="Force disable VLM in Stage-1 build (override config).",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Retrieval top-k frames.")
    parser.add_argument("--top-k-segments", type=int, default=6, help="Retrieval top-k segments for Stage-2.")
    parser.add_argument("--disable-stage2", action="store_true", help="Disable Stage-2 build/evaluation.")
    parser.add_argument(
        "--stage1-sample-secs",
        type=int,
        default=0,
        help="Override SCENE_GRAPH_STAGE1_SAMPLE_SECS for build (0 keeps config default).",
    )
    parser.add_argument(
        "--stage1-max-keyframes",
        type=int,
        default=0,
        help="Override SCENE_GRAPH_STAGE1_MAX_KEYFRAMES for build (0 keeps config default).",
    )
    parser.add_argument(
        "--stage2-window-secs",
        type=int,
        default=0,
        help="Override SCENE_GRAPH_STAGE2_WINDOW_SECS for build (0 keeps config default).",
    )
    parser.add_argument(
        "--stage2-overlap-secs",
        type=int,
        default=0,
        help="Override SCENE_GRAPH_STAGE2_OVERLAP_SECS for build (0 keeps config default).",
    )
    parser.add_argument(
        "--tolerance-sec",
        type=float,
        default=5.0,
        help="Temporal hit tolerance in seconds against time_reference.",
    )
    parser.add_argument(
        "--trace-dir",
        default="",
        help="Optional trace folder (jsonl) for baseline answer_check stats.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_STAGE1_REPORT_PATH),
        help="Output report json path.",
    )
    args = parser.parse_args()
    resolved_artifacts_root = args.artifacts_root
    if args.badcases_root:
        if args.artifacts_root != str(DEFAULT_ARTIFACTS_ROOT):
            print(
                "[WARN] Both --artifacts-root and legacy --badcases-root are set; "
                "using --artifacts-root."
            )
        else:
            resolved_artifacts_root = args.badcases_root

    if not resolved_artifacts_root:
        resolved_artifacts_root = str(badcases_root())

    use_vlm_override = None
    if args.use_vlm:
        use_vlm_override = True
    elif args.no_use_vlm:
        use_vlm_override = False

    meta = _load_meta(args.meta)
    if args.videos.strip():
        video_keys = [x.strip() for x in args.videos.split(",") if x.strip()]
    else:
        video_keys = sorted(meta.keys())

    if args.max_videos > 0:
        video_keys = video_keys[: args.max_videos]

    results: list[dict[str, Any]] = []
    agg = {
        "videos_attempted": 0,
        "videos_with_graph": 0,
        "sum_num_entities": 0.0,
        "sum_num_frames": 0.0,
        "sum_num_relations": 0.0,
        "sum_entity_duplication_rate": 0.0,
        "sum_invalid_relation_rate": 0.0,
        "sum_avg_entities_per_frame": 0.0,
        "sum_empty_frame_rate": 0.0,
        "sum_hit_at_1": 0.0,
        "sum_hit_at_k": 0.0,
        "sum_non_empty_retrieval_rate": 0.0,
        "sum_mean_distance": 0.0,
        "count_mean_distance": 0,
        "sum_trace_acc": 0.0,
        "count_trace_acc": 0,
        "questions_evaluated": 0,
        "videos_with_segment_graph": 0,
        "sum_num_segments": 0.0,
        "sum_num_temporal_relations": 0.0,
        "sum_num_segment_entities": 0.0,
        "sum_avg_active_entities_per_segment": 0.0,
        "sum_avg_relations_per_segment": 0.0,
        "sum_empty_segment_rate": 0.0,
        "sum_invalid_temporal_relation_rate": 0.0,
        "sum_segment_hit_at_1": 0.0,
        "sum_segment_hit_at_k": 0.0,
        "sum_segment_non_empty_retrieval_rate": 0.0,
        "sum_segment_mean_distance": 0.0,
        "count_segment_mean_distance": 0,
        "segment_questions_evaluated": 0,
    }

    for key in video_keys:
        meta_row = meta.get(key)
        if not meta_row:
            continue
        agg["videos_attempted"] += 1
        res = _evaluate_one_video(
            video_key=key,
            meta_row=meta_row,
            assets_root=args.assets_root,
            artifacts_root=resolved_artifacts_root,
            build_if_missing=args.build_if_missing,
            force_rebuild=args.force_rebuild,
            use_vlm=use_vlm_override,
            top_k=args.top_k,
            tolerance_sec=args.tolerance_sec,
            trace_dir=args.trace_dir or None,
            stage1_sample_secs=(args.stage1_sample_secs if args.stage1_sample_secs > 0 else None),
            stage1_max_keyframes=(args.stage1_max_keyframes if args.stage1_max_keyframes > 0 else None),
            enable_stage2=not args.disable_stage2,
            top_k_segments=args.top_k_segments,
            stage2_window_secs=(args.stage2_window_secs if args.stage2_window_secs > 0 else None),
            stage2_overlap_secs=(args.stage2_overlap_secs if args.stage2_overlap_secs > 0 else None),
        )
        out = {
            "video_key": res.video_key,
            "graph_path": res.stage1_graph_path,
            "graph_stats": res.stage1_graph_stats,
            "retrieval_stats": res.stage1_retrieval_stats,
            "segment_graph_path": res.stage2_graph_path,
            "segment_graph_stats": res.stage2_graph_stats,
            "segment_retrieval_stats": res.stage2_retrieval_stats,
            "baseline_stats": res.baseline_stats,
        }
        results.append(out)

        if "error" in res.stage1_graph_stats:
            continue

        agg["videos_with_graph"] += 1
        agg["sum_num_entities"] += float(res.stage1_graph_stats["num_entities"])
        agg["sum_num_frames"] += float(res.stage1_graph_stats["num_frames"])
        agg["sum_num_relations"] += float(res.stage1_graph_stats["num_relations"])
        agg["sum_entity_duplication_rate"] += float(res.stage1_graph_stats["entity_duplication_rate"])
        agg["sum_invalid_relation_rate"] += float(res.stage1_graph_stats["invalid_relation_rate"])
        agg["sum_avg_entities_per_frame"] += float(res.stage1_graph_stats["avg_entities_per_frame"])
        agg["sum_empty_frame_rate"] += float(res.stage1_graph_stats["empty_frame_rate"])

        if "error" not in res.stage1_retrieval_stats:
            agg["sum_hit_at_1"] += float(res.stage1_retrieval_stats["hit_at_1"])
            agg["sum_hit_at_k"] += float(res.stage1_retrieval_stats["hit_at_k"])
            agg["sum_non_empty_retrieval_rate"] += float(res.stage1_retrieval_stats["non_empty_retrieval_rate"])
            agg["questions_evaluated"] += int(res.stage1_retrieval_stats["num_questions_evaluated"])
            mean_dist = res.stage1_retrieval_stats.get("mean_best_temporal_distance_sec")
            if mean_dist is not None:
                agg["sum_mean_distance"] += float(mean_dist)
                agg["count_mean_distance"] += 1

        if (
            "error" not in res.stage2_graph_stats
            and "disabled" not in res.stage2_graph_stats
        ):
            agg["videos_with_segment_graph"] += 1
            agg["sum_num_segments"] += float(res.stage2_graph_stats["num_segments"])
            agg["sum_num_temporal_relations"] += float(res.stage2_graph_stats["num_temporal_relations"])
            agg["sum_num_segment_entities"] += float(res.stage2_graph_stats["num_segment_entities"])
            agg["sum_avg_active_entities_per_segment"] += float(
                res.stage2_graph_stats["avg_active_entities_per_segment"]
            )
            agg["sum_avg_relations_per_segment"] += float(
                res.stage2_graph_stats["avg_relations_per_segment"]
            )
            agg["sum_empty_segment_rate"] += float(res.stage2_graph_stats["empty_segment_rate"])
            agg["sum_invalid_temporal_relation_rate"] += float(
                res.stage2_graph_stats["invalid_temporal_relation_rate"]
            )

        if (
            "error" not in res.stage2_retrieval_stats
            and "disabled" not in res.stage2_retrieval_stats
        ):
            agg["sum_segment_hit_at_1"] += float(res.stage2_retrieval_stats["hit_at_1"])
            agg["sum_segment_hit_at_k"] += float(res.stage2_retrieval_stats["hit_at_k"])
            agg["sum_segment_non_empty_retrieval_rate"] += float(
                res.stage2_retrieval_stats["non_empty_retrieval_rate"]
            )
            agg["segment_questions_evaluated"] += int(
                res.stage2_retrieval_stats["num_questions_evaluated"]
            )
            segment_mean_dist = res.stage2_retrieval_stats.get("mean_best_temporal_distance_sec")
            if segment_mean_dist is not None:
                agg["sum_segment_mean_distance"] += float(segment_mean_dist)
                agg["count_segment_mean_distance"] += 1

        trace_acc = res.baseline_stats.get("trace_accuracy")
        if trace_acc is not None:
            agg["sum_trace_acc"] += float(trace_acc)
            agg["count_trace_acc"] += 1

    n = max(agg["videos_with_graph"], 1)
    summary = {
        "videos_attempted": agg["videos_attempted"],
        "videos_with_graph": agg["videos_with_graph"],
        "avg_num_entities": agg["sum_num_entities"] / n,
        "avg_num_frames": agg["sum_num_frames"] / n,
        "avg_num_relations": agg["sum_num_relations"] / n,
        "avg_entity_duplication_rate": agg["sum_entity_duplication_rate"] / n,
        "avg_invalid_relation_rate": agg["sum_invalid_relation_rate"] / n,
        "avg_entities_per_frame": agg["sum_avg_entities_per_frame"] / n,
        "avg_empty_frame_rate": agg["sum_empty_frame_rate"] / n,
        "avg_hit_at_1": agg["sum_hit_at_1"] / n,
        "avg_hit_at_k": agg["sum_hit_at_k"] / n,
        "avg_non_empty_retrieval_rate": agg["sum_non_empty_retrieval_rate"] / n,
        "avg_mean_best_temporal_distance_sec": (
            (agg["sum_mean_distance"] / agg["count_mean_distance"])
            if agg["count_mean_distance"] > 0
            else None
        ),
        "questions_evaluated_total": agg["questions_evaluated"],
        "avg_baseline_trace_accuracy": (
            (agg["sum_trace_acc"] / agg["count_trace_acc"])
            if agg["count_trace_acc"] > 0
            else None
        ),
        "baseline_trace_accuracy_video_count": agg["count_trace_acc"],
    }
    n_seg = max(agg["videos_with_segment_graph"], 1)
    summary.update(
        {
            "videos_with_segment_graph": agg["videos_with_segment_graph"],
            "segment_avg_num_segments": agg["sum_num_segments"] / n_seg,
            "segment_avg_num_temporal_relations": agg["sum_num_temporal_relations"] / n_seg,
            "segment_avg_num_segment_entities": agg["sum_num_segment_entities"] / n_seg,
            "segment_avg_active_entities_per_segment": agg["sum_avg_active_entities_per_segment"] / n_seg,
            "segment_avg_relations_per_segment": agg["sum_avg_relations_per_segment"] / n_seg,
            "segment_avg_empty_segment_rate": agg["sum_empty_segment_rate"] / n_seg,
            "segment_avg_invalid_temporal_relation_rate": agg["sum_invalid_temporal_relation_rate"] / n_seg,
            "segment_avg_hit_at_1": agg["sum_segment_hit_at_1"] / n_seg,
            "segment_avg_hit_at_k": agg["sum_segment_hit_at_k"] / n_seg,
            "segment_avg_non_empty_retrieval_rate": agg["sum_segment_non_empty_retrieval_rate"] / n_seg,
            "segment_avg_mean_best_temporal_distance_sec": (
                (agg["sum_segment_mean_distance"] / agg["count_segment_mean_distance"])
                if agg["count_segment_mean_distance"] > 0
                else None
            ),
            "segment_questions_evaluated_total": agg["segment_questions_evaluated"],
        }
    )

    report = {
        "config": {
            "meta": args.meta,
            "artifacts_root": resolved_artifacts_root,
            "assets_root": args.assets_root,
            "legacy_badcases_root_arg": args.badcases_root,
            "videos": video_keys,
            "build_if_missing": args.build_if_missing,
            "force_rebuild": args.force_rebuild,
            "use_vlm_override": use_vlm_override,
            "top_k": args.top_k,
            "top_k_segments": args.top_k_segments,
            "enable_stage2": not args.disable_stage2,
            "stage1_sample_secs": args.stage1_sample_secs,
            "stage1_max_keyframes": args.stage1_max_keyframes,
            "stage2_window_secs": args.stage2_window_secs,
            "stage2_overlap_secs": args.stage2_overlap_secs,
            "tolerance_sec": args.tolerance_sec,
            "trace_dir": args.trace_dir,
        },
        "summary": summary,
        "results": results,
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved report to: {args.output}")


if __name__ == "__main__":
    main()
