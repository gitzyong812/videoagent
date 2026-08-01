from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any

import sgd.config as config
from sgd.scene_graph.schema import (empty_stage2_segment_graph,
                                    safe_load_stage1_graph,
                                    safe_load_stage2_segment_graph,
                                    validate_stage2_segment_graph)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _iter_segment_spans(max_t_sec: float, window_sec: float, overlap_sec: float) -> list[tuple[float, float]]:
    if max_t_sec <= 0:
        return []
    window_sec = max(float(window_sec), 1.0)
    overlap_sec = max(0.0, float(overlap_sec))
    if overlap_sec >= window_sec:
        overlap_sec = window_sec * 0.5

    spans: list[tuple[float, float]] = []
    stride = window_sec - overlap_sec
    start = 0.0
    while start < max_t_sec:
        end = min(start + window_sec, max_t_sec)
        spans.append((round(start, 3), round(end, 3)))
        if end >= max_t_sec:
            break
        start += stride
    return spans


def _build_segment_summary(active_entities: list[dict[str, Any]], temporal_relations: list[dict[str, Any]]) -> str:
    ent_names = [ent.get("canonical_name", "") for ent in active_entities[:6] if ent.get("canonical_name")]
    rel_preds = [rel.get("p", "") for rel in temporal_relations[:6] if rel.get("p")]
    ent_part = ", ".join(ent_names) if ent_names else "none"
    rel_part = ", ".join(rel_preds) if rel_preds else "none"
    return f"entities: {ent_part}; predicates: {rel_part}"


def _build_segment_embed_text(seg: dict[str, Any]) -> str:
    """Build a compact text representation of a segment for embedding."""
    parts: list[str] = []
    summary = seg.get("summary", "")
    if summary:
        parts.append(summary)
    for ent in seg.get("active_entities", [])[:6]:
        name = ent.get("canonical_name", "")
        if name:
            parts.append(name)
    for rel in seg.get("temporal_relations", [])[:4]:
        p = rel.get("p", "")
        if p:
            parts.append(p)
    return " ".join(parts)


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    value = (text or "").strip()
    if not value:
        return None
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(value[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _segment_vlm_messages(
    seg_start: float,
    seg_end: float,
    seg_frames: list[dict[str, Any]],
    entity_bank: dict[str, dict[str, Any]],
    caption_snippets: list[str],
) -> list[dict[str, str]]:
    active_ids = {
        str(ent.get("entity_id", ""))
        for frame in seg_frames
        for ent in frame.get("entities", [])
        if ent.get("entity_id")
    }
    context = {
        "window": {"t_start": seg_start, "t_end": seg_end},
        "entity_registry": [entity_bank[eid] for eid in sorted(active_ids) if eid in entity_bank],
        "frame_graphs": [
            {
                "frame_id": frame.get("frame_id"),
                "t_sec": frame.get("t_sec"),
                "entities": frame.get("entities", []),
                "relations": frame.get("relations", []),
            }
            for frame in seg_frames
        ],
        "caption_snippets": caption_snippets,
    }
    schema = {
        "relations": [
            {
                "s": "ent_000001",
                "p": "interacting_with",
                "o": "ent_000002",
                "relation_type": "action",
                "t_start": seg_start,
                "t_end": seg_end,
                "support_frame_ids": ["f_000001"],
                "confidence": 0.0,
            }
        ]
    }
    return [
        {"role": "system", "content": "You extract segment-level video relations and return strict JSON only."},
        {
            "role": "user",
            "content": (
                "Analyze the video frames and frame-level graphs in this window. Extract entity "
                "trajectories, actions, interactions, state changes, and temporal relations. Reuse "
                "only entity IDs and frame IDs present in the supplied context. Use snake_case "
                "predicates and ground every relation with a time interval and supporting frames. "
                f"Return JSON matching this schema: {json.dumps(schema, separators=(',', ':'))}\n"
                f"Context: {json.dumps(context, ensure_ascii=False, separators=(',', ':'))}"
            ),
        },
    ]


def _call_segment_vlm(
    seg_start: float,
    seg_end: float,
    seg_frames: list[dict[str, Any]],
    entity_bank: dict[str, dict[str, Any]],
    caption_snippets: list[str],
) -> dict[str, Any]:
    if not bool(getattr(config, "SCENE_GRAPH_STAGE2_USE_VLM", True)):
        raise RuntimeError("Strict segment construction requires SCENE_GRAPH_STAGE2_USE_VLM=True.")

    from sgd.utils import call_openai_model_with_tools

    image_paths = [str(frame.get("path")) for frame in seg_frames if frame.get("path")]
    attempts = max(1, int(getattr(config, "SCENE_GRAPH_STAGE2_VLM_RETRIES", 3)))
    last_error = "empty response"
    for _ in range(attempts):
        response = call_openai_model_with_tools(
            messages=_segment_vlm_messages(
                seg_start=seg_start,
                seg_end=seg_end,
                seg_frames=seg_frames,
                entity_bank=entity_bank,
                caption_snippets=caption_snippets,
            ),
            endpoints=config.SCENE_GRAPH_STAGE2_VLM_ENDPOINT_LIST,
            model_name=config.SCENE_GRAPH_STAGE2_VLM_MODEL_NAME,
            api_key=config.OPENAI_API_KEY,
            image_paths=image_paths,
            temperature=0.0,
            max_tokens=int(getattr(config, "SCENE_GRAPH_STAGE2_VLM_MAX_TOKENS", 1600)),
            return_json=True,
        )
        parsed = _extract_json_obj((response or {}).get("content", ""))
        if parsed is not None and isinstance(parsed.get("relations", []), list):
            return parsed
        last_error = "response did not contain a valid `relations` list"
    raise RuntimeError(
        f"Segment VLM extraction failed for {seg_start:.1f}-{seg_end:.1f}s: {last_error}."
    )


def _normalize_vlm_relations(
    raw_relations: list[dict[str, Any]],
    seg_start: float,
    seg_end: float,
    seg_frames: list[dict[str, Any]],
    entity_bank: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    valid_frame_ids = {
        str(frame.get("frame_id"))
        for frame in seg_frames
        if frame.get("frame_id")
    }
    name_to_id = {
        str(ent.get("canonical_name", "")).strip().lower(): eid
        for eid, ent in entity_bank.items()
        if ent.get("canonical_name")
    }
    normalized: list[dict[str, Any]] = []
    for rel in raw_relations:
        if not isinstance(rel, dict):
            continue
        s_raw = str(rel.get("s", "")).strip()
        o_raw = str(rel.get("o", "")).strip()
        s = s_raw if s_raw in entity_bank else name_to_id.get(s_raw.lower(), "")
        o = o_raw if o_raw in entity_bank else name_to_id.get(o_raw.lower(), "")
        predicate = str(rel.get("p", "")).strip().lower().replace(" ", "_")
        if not s or not o or not predicate:
            continue
        try:
            t_start = max(seg_start, float(rel.get("t_start", seg_start)))
            t_end = min(seg_end, float(rel.get("t_end", seg_end)))
        except (TypeError, ValueError):
            t_start, t_end = seg_start, seg_end
        if t_end < t_start:
            t_start, t_end = t_end, t_start
        try:
            confidence = min(1.0, max(0.0, float(rel.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        support_ids = [
            str(frame_id)
            for frame_id in rel.get("support_frame_ids", [])
            if str(frame_id) in valid_frame_ids
        ]
        if not support_ids:
            support_ids = sorted(valid_frame_ids)
        normalized.append(
            {
                "s": s,
                "p": predicate,
                "o": o,
                "relation_type": str(rel.get("relation_type", "action")).strip().lower() or "action",
                "t_start": round(t_start, 3),
                "t_end": round(t_end, 3),
                "confidence": confidence,
                "support_frame_ids": support_ids,
                "count": 1,
                "source": "vlm",
            }
        )
    return normalized


def _embed_segments(segments: list[dict[str, Any]]) -> None:
    """Batch-embed all segments and store the vector in each segment dict."""
    if not segments:
        return
    batch_size = int(getattr(config, "SCENE_GRAPH_STAGE2_EMBED_BATCH_SIZE", 64))
    texts = [_build_segment_embed_text(seg) for seg in segments]

    from sgd.utils import AzureOpenAIEmbeddingService  # lazy import

    all_embeddings: list[list[float]] = [[] for _ in segments]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        try:
            results = AzureOpenAIEmbeddingService.get_embeddings(
                endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
                model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
                input_text=batch_texts,
                api_key=config.OPENAI_API_KEY,
            )
            for j, item in enumerate(results):
                all_embeddings[i + j] = item["embedding"]
        except Exception as exc:
            print(f"[WARN] Segment embedding batch {i}..{i+len(batch_texts)} failed: {exc}")

    for seg, emb in zip(segments, all_embeddings):
        if emb:
            seg["embedding"] = emb


def build_stage2_segment_graph(
    stage1_graph_path: str,
    output_path: str,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    if (not force_rebuild) and os.path.exists(output_path):
        cached = safe_load_stage2_segment_graph(output_path)
        if cached is not None:
            return cached

    stage1_graph = safe_load_stage1_graph(stage1_graph_path)
    if stage1_graph is None:
        raise ValueError(f"Invalid or missing Stage-1 graph: {stage1_graph_path}")

    frames = sorted(stage1_graph.get("frames", []), key=lambda x: float(x.get("t_sec", 0.0)))
    if not frames:
        graph = empty_stage2_segment_graph(
            video_id=stage1_graph.get("video_id", "unknown_video"),
            config_snapshot={
                "window_sec": float(getattr(config, "SCENE_GRAPH_STAGE2_WINDOW_SECS", 30)),
                "overlap_sec": float(getattr(config, "SCENE_GRAPH_STAGE2_OVERLAP_SECS", 5)),
                "max_relations_per_segment": int(getattr(config, "SCENE_GRAPH_STAGE2_MAX_RELATIONS_PER_SEGMENT", 64)),
            },
        )
        _ensure_parent_dir(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        return graph

    entity_bank = {
        ent["entity_id"]: ent
        for ent in stage1_graph.get("entities", [])
        if isinstance(ent, dict) and ent.get("entity_id")
    }
    max_t_sec = max(float(frame.get("t_sec", 0.0)) for frame in frames)
    spans = _iter_segment_spans(
        max_t_sec=max_t_sec,
        window_sec=float(getattr(config, "SCENE_GRAPH_STAGE2_WINDOW_SECS", 30)),
        overlap_sec=float(getattr(config, "SCENE_GRAPH_STAGE2_OVERLAP_SECS", 5)),
    )
    max_relations_per_segment = int(getattr(config, "SCENE_GRAPH_STAGE2_MAX_RELATIONS_PER_SEGMENT", 64))

    graph = empty_stage2_segment_graph(
        video_id=stage1_graph.get("video_id", "unknown_video"),
        config_snapshot={
            "window_sec": float(getattr(config, "SCENE_GRAPH_STAGE2_WINDOW_SECS", 30)),
            "overlap_sec": float(getattr(config, "SCENE_GRAPH_STAGE2_OVERLAP_SECS", 5)),
            "max_relations_per_segment": max_relations_per_segment,
        },
    )

    total_temporal_rel = 0
    total_segment_entities = 0

    for seg_idx, (seg_start, seg_end) in enumerate(spans, start=1):
        seg_frames = [
            frame
            for frame in frames
            if (seg_start <= float(frame.get("t_sec", 0.0)) < seg_end)
            or (seg_end == max_t_sec and seg_start <= float(frame.get("t_sec", 0.0)) <= seg_end)
        ]
        if not seg_frames:
            continue

        entity_acc: dict[str, dict[str, Any]] = {}
        relation_acc: dict[tuple[str, str, str], dict[str, Any]] = {}
        support_frame_ids: list[str] = []

        for frame in seg_frames:
            frame_id = str(frame.get("frame_id", ""))
            frame_t = float(frame.get("t_sec", 0.0))
            if frame_id:
                support_frame_ids.append(frame_id)

            for ent in frame.get("entities", []):
                ent_id = str(ent.get("entity_id", ""))
                if not ent_id:
                    continue
                bank_ent = entity_bank.get(ent_id, {})
                if ent_id not in entity_acc:
                    entity_acc[ent_id] = {
                        "entity_id": ent_id,
                        "canonical_name": bank_ent.get("canonical_name", ent.get("name", "")),
                        "type": bank_ent.get("type", ent.get("type", "object")),
                        "confidence": float(ent.get("confidence", 0.0)),
                        "frame_count": 1,
                    }
                else:
                    entity_acc[ent_id]["confidence"] = max(
                        float(entity_acc[ent_id]["confidence"]),
                        float(ent.get("confidence", 0.0)),
                    )
                    entity_acc[ent_id]["frame_count"] += 1

            for rel in frame.get("relations", []):
                s = str(rel.get("s", ""))
                p = str(rel.get("p", ""))
                o = str(rel.get("o", ""))
                if not s or not p or not o:
                    continue
                key = (s, p, o)
                if key not in relation_acc:
                    relation_acc[key] = {
                        "s": s,
                        "p": p,
                        "o": o,
                        "t_start": frame_t,
                        "t_end": frame_t,
                        "confidence": float(rel.get("confidence", 0.0)),
                        "support_frame_ids": [frame_id] if frame_id else [],
                        "count": 1,
                        "source": "frame",
                    }
                else:
                    relation_acc[key]["t_start"] = min(float(relation_acc[key]["t_start"]), frame_t)
                    relation_acc[key]["t_end"] = max(float(relation_acc[key]["t_end"]), frame_t)
                    relation_acc[key]["confidence"] = max(
                        float(relation_acc[key]["confidence"]),
                        float(rel.get("confidence", 0.0)),
                    )
                    if frame_id and frame_id not in relation_acc[key]["support_frame_ids"]:
                        relation_acc[key]["support_frame_ids"].append(frame_id)
                    relation_acc[key]["count"] += 1

        caption_snippets: list[str] = []
        if bool(getattr(config, "SCENE_GRAPH_STAGE2_INCLUDE_CAPTION_SNIPPETS", True)):
            for frame in seg_frames:
                caption = str(frame.get("caption_context", "")).strip()
                if caption and caption not in caption_snippets:
                    caption_snippets.append(caption)
            caption_snippets = caption_snippets[
                : int(getattr(config, "SCENE_GRAPH_STAGE2_MAX_CAPTION_SNIPPETS", 4))
            ]

        vlm_output = _call_segment_vlm(
            seg_start=seg_start,
            seg_end=seg_end,
            seg_frames=seg_frames,
            entity_bank=entity_bank,
            caption_snippets=caption_snippets,
        )
        vlm_relations = _normalize_vlm_relations(
            raw_relations=vlm_output.get("relations", []),
            seg_start=seg_start,
            seg_end=seg_end,
            seg_frames=seg_frames,
            entity_bank=entity_bank,
        )
        for relation in vlm_relations:
            key = (relation["s"], relation["p"], relation["o"])
            if key not in relation_acc:
                relation_acc[key] = relation
                continue
            existing = relation_acc[key]
            existing["t_start"] = min(float(existing["t_start"]), float(relation["t_start"]))
            existing["t_end"] = max(float(existing["t_end"]), float(relation["t_end"]))
            existing["confidence"] = max(
                float(existing.get("confidence", 0.0)),
                float(relation.get("confidence", 0.0)),
            )
            existing["relation_type"] = relation.get("relation_type", existing.get("relation_type", "action"))
            existing["source"] = "frame+vlm"
            for frame_id in relation.get("support_frame_ids", []):
                if frame_id not in existing["support_frame_ids"]:
                    existing["support_frame_ids"].append(frame_id)

        active_entities = sorted(
            entity_acc.values(),
            key=lambda x: (x["frame_count"], x["confidence"]),
            reverse=True,
        )
        temporal_relations = sorted(
            relation_acc.values(),
            key=lambda x: (
                1 if "vlm" in str(x.get("source", "")) else 0,
                x["count"],
                x["confidence"],
            ),
            reverse=True,
        )[:max_relations_per_segment]

        segment_id = f"seg_{seg_idx:04d}"
        total_temporal_rel += len(temporal_relations)
        total_segment_entities += len(active_entities)
        graph["segments"].append(
            {
                "segment_id": segment_id,
                "t_start": seg_start,
                "t_end": seg_end,
                "active_entities": active_entities,
                "temporal_relations": temporal_relations,
                "support_frame_ids": support_frame_ids,
                "caption_snippets": caption_snippets,
                "summary": _build_segment_summary(active_entities, temporal_relations),
                "overlap_link": {},
            }
        )

    # Build light overlap links between neighboring segments to support temporal expansion.
    for idx in range(1, len(graph["segments"])):
        prev_seg = graph["segments"][idx - 1]
        cur_seg = graph["segments"][idx]
        prev_entities = {e["entity_id"] for e in prev_seg.get("active_entities", [])}
        cur_entities = {e["entity_id"] for e in cur_seg.get("active_entities", [])}
        shared = sorted(prev_entities & cur_entities)
        if shared:
            cur_seg["overlap_link"] = {
                "prev_segment_id": prev_seg["segment_id"],
                "shared_entity_ids": shared[:16],
                "shared_entity_count": len(shared),
            }

    # ── optional: embed segment texts for semantic retrieval fallback ──────────
    if bool(getattr(config, "SCENE_GRAPH_STAGE2_EMBED_SEGMENTS", False)):
        _embed_segments(graph["segments"])
        embedded_count = sum(1 for seg in graph["segments"] if seg.get("embedding"))
        graph["stats"]["embedded_segments"] = embedded_count
    else:
        graph["stats"]["embedded_segments"] = 0

    graph["stats"]["num_segments"] = len(graph["segments"])
    graph["stats"]["num_temporal_relations"] = total_temporal_rel
    graph["stats"]["num_segment_entities"] = total_segment_entities

    ok, reason = validate_stage2_segment_graph(graph)
    if not ok:
        raise ValueError(f"Stage2 segment graph validation failed: {reason}")

    _ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    return graph


def ensure_stage2_segment_graph(
    stage1_graph_path: str,
    output_path: str,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    return build_stage2_segment_graph(
        stage1_graph_path=stage1_graph_path,
        output_path=output_path,
        force_rebuild=force_rebuild,
    )
