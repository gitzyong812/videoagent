from __future__ import annotations

import bisect
import json
import os
import re
from difflib import SequenceMatcher
from typing import Any

import sgd.config as config
from sgd.scene_graph.schema import (empty_stage1_graph, safe_load_stage1_graph,
                                    validate_stage1_graph)


ENTITY_BLACKLIST: set[str] = {
    "video", "clip", "scene", "frame", "shot", "footage", "camera",
    "cinematic", "view", "angle", "sequence", "image", "photo", "picture",
    "thumbnail", "narrator", "voiceover", "subtitle", "caption",
    "timestamp", "overlay", "background", "foreground",
    "begins", "starts", "ends", "shows", "appears", "continues", "features",
    "moving", "walking", "talking", "standing", "sitting", "running",
    "playing", "looking", "wearing", "holding", "showing", "speaking",
    "area", "side", "part", "moment",
    "something", "thing", "stuff", "nothing", "everything", "place",
    "content", "section", "segment", "portion", "event", "action",
    "activity", "transition", "setting", "environment",
}
ALLOWED_ENTITY_TYPES: set[str] = {
    "person", "object", "scene", "animal", "vehicle", "food", "text",
    "location", "building", "clothing", "instrument", "plant",
}


def _is_blacklisted_entity(name: str) -> bool:
    """Return True if the entity name should be filtered out."""
    n = name.strip().lower()
    if not n or len(n) <= 1:
        return True
    if n in ENTITY_BLACKLIST:
        return True
    return False


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _extract_video_id(video_caption_json_path: str) -> str:
    # .../<video_id>/captions/captions.json
    parent = os.path.dirname(video_caption_json_path)
    video_root = os.path.dirname(parent)
    return os.path.basename(video_root)


def _load_caption_segments(video_caption_json_path: str) -> tuple[list[dict[str, Any]], int]:
    with open(video_caption_json_path, "r", encoding="utf-8") as f:
        captions = json.load(f)
    captions.pop("subject_registry", None)
    captions.pop("character_registry", None)

    segments = []
    max_end = 0.0
    for key, val in captions.items():
        try:
            start, end = map(float, key.split("_"))
        except Exception:
            continue
        max_end = max(max_end, end)
        text = val.get("caption", "") if isinstance(val, dict) else ""
        if isinstance(text, list):
            text = text[0] if text else ""
        if not isinstance(text, str):
            text = str(text)
        segments.append(
            {
                "start": float(start),
                "end": float(end),
                "caption": text.strip(),
            }
        )
    segments.sort(key=lambda x: x["start"])
    return segments, int(round(max_end))


def _extract_frame_index(filename: str) -> int | None:
    match = re.search(r"(\d+)(?=\.[^.]+$)", filename)
    if not match:
        return None
    return int(match.group(1))


def _sample_keyframes(
    frame_dir: str,
    fps: float,
    sample_secs: int,
    max_keyframes: int,
) -> list[dict[str, Any]]:
    if not os.path.isdir(frame_dir):
        return []

    records = []
    for fn in os.listdir(frame_dir):
        if not fn.lower().endswith(".jpg"):
            continue
        frame_idx = _extract_frame_index(fn)
        if frame_idx is None:
            continue
        records.append((frame_idx, fn))
    if not records:
        return []

    records.sort(key=lambda x: x[0])
    frame_indices = [idx for idx, _ in records]
    frame_lookup = {idx: fn for idx, fn in records}

    index_offset = 1 if frame_indices[0] == 1 else 0
    stride = max(1, int(round(sample_secs * fps)))

    selected_indices: list[int] = []
    cur = frame_indices[0]
    while cur <= frame_indices[-1]:
        pos = bisect.bisect_left(frame_indices, cur)
        candidate = None
        if pos == 0:
            candidate = frame_indices[0]
        elif pos >= len(frame_indices):
            candidate = frame_indices[-1]
        else:
            left = frame_indices[pos - 1]
            right = frame_indices[pos]
            candidate = left if abs(cur - left) <= abs(right - cur) else right
        selected_indices.append(candidate)
        cur += stride

    selected_indices = sorted(set(selected_indices))
    if max_keyframes > 0 and len(selected_indices) > max_keyframes:
        # Keep global temporal coverage instead of truncating to early frames.
        n = len(selected_indices)
        picks = []
        for i in range(max_keyframes):
            idx = int(round(i * (n - 1) / max(max_keyframes - 1, 1)))
            picks.append(selected_indices[idx])
        selected_indices = sorted(set(picks))

    results = []
    for frame_idx in selected_indices:
        t_sec = max(0.0, (frame_idx - index_offset) / max(fps, 1e-6))
        results.append(
            {
                "frame_id": f"f_{frame_idx:06d}",
                "frame_idx": frame_idx,
                "path": os.path.join(frame_dir, frame_lookup[frame_idx]),
                "t_sec": round(t_sec, 3),
            }
        )
    return results


def _find_caption_for_time(caption_segments: list[dict[str, Any]], t_sec: float) -> str:
    for seg in caption_segments:
        if seg["start"] <= t_sec <= seg["end"]:
            return seg["caption"]
    if not caption_segments:
        return ""
    nearest = min(
        caption_segments,
        key=lambda x: min(abs(t_sec - x["start"]), abs(t_sec - x["end"])),
    )
    return nearest["caption"]


def _neighbor_caption_context(
    caption_segments: list[dict[str, Any]],
    t_sec: float,
) -> str:
    if not caption_segments:
        return ""
    nearest_idx = min(
        range(len(caption_segments)),
        key=lambda idx: min(
            abs(t_sec - caption_segments[idx]["start"]),
            abs(t_sec - caption_segments[idx]["end"]),
        ),
    )
    selected = caption_segments[max(0, nearest_idx - 1) : nearest_idx + 2]
    return "\n".join(
        f"[{item['start']:.1f}-{item['end']:.1f}] {item['caption']}"
        for item in selected
        if item.get("caption")
    )


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None

    # Try raw JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try ```json ... ```
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    for frag in fenced:
        try:
            obj = json.loads(frag)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    # Fallback: first {...} span
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _normalize_frame_parse(raw_obj: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    raw_entities = raw_obj.get("entities", [])
    if isinstance(raw_entities, list):
        for idx, ent in enumerate(raw_entities, start=1):
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name", "")).strip().lower()
            if not name:
                continue
            if _is_blacklisted_entity(name):
                continue
            local_id = str(ent.get("local_id", f"e{idx}")).strip() or f"e{idx}"
            etype = str(ent.get("type", "object")).strip().lower() or "object"
            if etype not in ALLOWED_ENTITY_TYPES:
                etype = "object"
            attrs = ent.get("attributes", [])
            if not isinstance(attrs, list):
                attrs = [str(attrs)]
            attrs = [str(a).strip().lower() for a in attrs if str(a).strip()]
            desc = str(ent.get("desc_text", "")).strip()
            conf = ent.get("confidence", 0.5)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.5
            conf = min(max(conf, 0.0), 1.0)
            entities.append(
                {
                    "local_id": local_id,
                    "name": name,
                    "type": etype,
                    "attributes": attrs,
                    "desc_text": desc,
                    "confidence": conf,
                }
            )

    raw_relations = raw_obj.get("relations", [])
    if isinstance(raw_relations, list):
        for rel in raw_relations:
            if not isinstance(rel, dict):
                continue
            s = str(rel.get("s", "")).strip()
            p = str(rel.get("p", "")).strip().lower().replace(" ", "_")
            o = str(rel.get("o", "")).strip()
            if not s or not p or not o:
                continue
            conf = rel.get("confidence", 0.5)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.5
            conf = min(max(conf, 0.0), 1.0)
            relations.append({"s": s, "p": p, "o": o, "confidence": conf})

    return entities, relations


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _entity_match_score(new_ent: dict[str, Any], old_ent: dict[str, Any]) -> float:
    name_score = _name_similarity(new_ent["name"], old_ent.get("canonical_name", ""))
    desc_new = new_ent.get("desc_text", "")
    desc_old = old_ent.get("desc_text", "")
    desc_score = _name_similarity(desc_new, desc_old) if desc_new and desc_old else 0.0
    type_score = 1.0 if new_ent.get("type") == old_ent.get("type") else 0.0
    return 0.7 * name_score + 0.2 * desc_score + 0.1 * type_score


def _align_entity(
    frame_ent: dict[str, Any],
    entity_bank: list[dict[str, Any]],
    t_sec: float,
    merge_threshold: float,
) -> str:
    best_idx = -1
    best_score = -1.0
    for i, old_ent in enumerate(entity_bank):
        score = _entity_match_score(frame_ent, old_ent)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx >= 0 and best_score >= merge_threshold:
        matched = entity_bank[best_idx]
        matched["last_seen"] = t_sec
        matched["confidence"] = max(float(matched.get("confidence", 0.0)), frame_ent["confidence"])
        if frame_ent["name"] not in matched["aliases"] and frame_ent["name"] != matched["canonical_name"]:
            matched["aliases"].append(frame_ent["name"])
        for attr in frame_ent.get("attributes", []):
            if attr not in matched["attributes"]:
                matched["attributes"].append(attr)
        return matched["entity_id"]

    entity_id = f"ent_{len(entity_bank) + 1:06d}"
    entity_bank.append(
        {
            "entity_id": entity_id,
            "canonical_name": frame_ent["name"],
            "aliases": [],
            "type": frame_ent.get("type", "object"),
            "attributes": list(frame_ent.get("attributes", [])),
            "first_seen": t_sec,
            "last_seen": t_sec,
            "desc_text": frame_ent.get("desc_text", ""),
            "confidence": frame_ent.get("confidence", 0.5),
            "evidence": [],
        }
    )
    return entity_id


def _build_prompt(
    question_hint: str = "",
    previous_graphs: list[dict[str, Any]] | None = None,
    entity_bank: list[dict[str, Any]] | None = None,
    caption_context: str = "",
) -> list[dict[str, str]]:
    user_msg = (
        "Extract a compact frame-level scene graph from this image. "
        "Return strict JSON with schema: "
        "{\"entities\":[{\"local_id\":\"e1\",\"name\":\"person\",\"type\":\"person\","
        "\"attributes\":[\"standing\"],\"desc_text\":\"short description\",\"confidence\":0.0}],"
        "\"relations\":[{\"s\":\"e1\",\"p\":\"holding\",\"o\":\"e2\",\"confidence\":0.0}]}. "
        "Rules: keep 1-8 salient entities, 0-12 relations, relation predicate in snake_case."
    )
    if question_hint:
        user_msg += f" Focus on details relevant to this query hint: {question_hint}"
    context = {
        "preceding_frame_graphs": [
            {
                "frame_id": frame.get("frame_id"),
                "t_sec": frame.get("t_sec"),
                "entities": frame.get("entities", []),
                "relations": frame.get("relations", []),
            }
            for frame in (previous_graphs or [])
        ],
        "video_entity_registry": [
            {
                "entity_id": ent.get("entity_id"),
                "canonical_name": ent.get("canonical_name"),
                "type": ent.get("type"),
                "attributes": ent.get("attributes", []),
                "desc_text": ent.get("desc_text", ""),
            }
            for ent in (entity_bank or [])
        ],
    }
    user_msg += (
        " Use the following context only for cross-frame identity consistency; "
        "still extract only what is visible in the current image:\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )
    if caption_context:
        user_msg += (
            "\nThe first visual parse was empty. Use these neighboring captions only as "
            f"auxiliary retry context, and do not invent invisible entities:\n{caption_context}"
        )
    return [
        {"role": "system", "content": "You produce strict JSON only."},
        {"role": "user", "content": user_msg},
    ]


def _parse_single_frame_with_vlm(
    image_path: str,
    query_hint: str = "",
    previous_graphs: list[dict[str, Any]] | None = None,
    entity_bank: list[dict[str, Any]] | None = None,
    caption_context: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not config.SCENE_GRAPH_STAGE1_USE_VLM:
        return [], []

    from sgd.utils import call_openai_model_with_tools

    msgs = call_openai_model_with_tools(
        messages=_build_prompt(
            query_hint=query_hint,
            previous_graphs=previous_graphs,
            entity_bank=entity_bank,
            caption_context=caption_context,
        ),
        endpoints=config.SCENE_GRAPH_STAGE1_VLM_ENDPOINT_LIST,
        model_name=config.SCENE_GRAPH_STAGE1_VLM_MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        image_paths=[image_path],
        temperature=0.0,
        max_tokens=800,
    )
    if msgs is None:
        return [], []
    raw_obj = _extract_json_obj(msgs.get("content", ""))
    if not raw_obj:
        return [], []
    return _normalize_frame_parse(raw_obj)


def build_stage1_scene_graph(
    video_caption_json_path: str,
    output_path: str,
    force_rebuild: bool = False,
    query_hint: str = "",
    frame_dir_override: str | None = None,
) -> dict[str, Any]:
    if (not force_rebuild) and os.path.exists(output_path):
        cached = safe_load_stage1_graph(output_path)
        if cached is not None:
            return cached

    caption_segments, _ = _load_caption_segments(video_caption_json_path)
    video_id = _extract_video_id(video_caption_json_path)
    video_root = os.path.dirname(os.path.dirname(video_caption_json_path))
    frame_dir = (
        os.path.abspath(frame_dir_override)
        if frame_dir_override
        else os.path.join(video_root, "frames")
    )
    keyframes = _sample_keyframes(
        frame_dir=frame_dir,
        fps=float(getattr(config, "VIDEO_FPS", 2)),
        sample_secs=int(getattr(config, "SCENE_GRAPH_STAGE1_SAMPLE_SECS", 5)),
        max_keyframes=int(getattr(config, "SCENE_GRAPH_STAGE1_MAX_KEYFRAMES", 120)),
    )

    graph = empty_stage1_graph(
        video_id=video_id,
        config_snapshot={
            "sample_secs": int(getattr(config, "SCENE_GRAPH_STAGE1_SAMPLE_SECS", 5)),
            "max_keyframes": int(getattr(config, "SCENE_GRAPH_STAGE1_MAX_KEYFRAMES", 120)),
            "entity_match_threshold": float(
                getattr(config, "SCENE_GRAPH_STAGE1_ENTITY_MATCH_THRESHOLD", 0.78)
            ),
            "use_vlm": bool(getattr(config, "SCENE_GRAPH_STAGE1_USE_VLM", True)),
            "frame_dir": frame_dir,
        },
    )
    entity_bank = graph["entities"]
    total_relations = 0
    vlm_success_frames = 0
    vlm_empty_frames = 0
    caption_retry_frames = 0
    merge_threshold = float(getattr(config, "SCENE_GRAPH_STAGE1_ENTITY_MATCH_THRESHOLD", 0.78))
    allowed_preds = set(getattr(config, "SCENE_GRAPH_STAGE1_ALLOWED_PREDICATES", []))

    for item in keyframes:
        caption_text = _find_caption_for_time(caption_segments, item["t_sec"])
        previous_graphs = graph["frames"][-2:]
        frame_entities, frame_relations = _parse_single_frame_with_vlm(
            item["path"],
            query_hint=query_hint,
            previous_graphs=previous_graphs,
            entity_bank=entity_bank,
        )
        if bool(getattr(config, "SCENE_GRAPH_STAGE1_USE_VLM", True)):
            if frame_entities:
                vlm_success_frames += 1
            else:
                vlm_empty_frames += 1
        if (
            not frame_entities
            and bool(getattr(config, "SCENE_GRAPH_STAGE1_CAPTION_RETRY", True))
            and caption_text
        ):
            frame_entities, frame_relations = _parse_single_frame_with_vlm(
                item["path"],
                query_hint=query_hint,
                previous_graphs=previous_graphs,
                entity_bank=entity_bank,
                caption_context=_neighbor_caption_context(caption_segments, item["t_sec"]),
            )
            caption_retry_frames += 1
            if frame_entities:
                vlm_success_frames += 1

        local_to_global: dict[str, str] = {}
        name_to_global: dict[str, str] = {}
        normalized_entities = []
        for ent in frame_entities:
            entity_id = _align_entity(ent, entity_bank, item["t_sec"], merge_threshold)
            local_to_global[ent["local_id"]] = entity_id
            name_to_global[ent["name"]] = entity_id
            normalized_entities.append(
                {
                    "entity_id": entity_id,
                    "name": ent["name"],
                    "type": ent["type"],
                    "confidence": ent["confidence"],
                }
            )

        dedup_rel = set()
        normalized_relations = []
        for rel in frame_relations:
            if allowed_preds and rel["p"] not in allowed_preds:
                continue
            s = local_to_global.get(rel["s"]) or name_to_global.get(rel["s"].lower())
            o = local_to_global.get(rel["o"]) or name_to_global.get(rel["o"].lower())
            if not s or not o:
                continue
            key = (s, rel["p"], o)
            if key in dedup_rel:
                continue
            dedup_rel.add(key)
            normalized_relations.append(
                {
                    "s": s,
                    "p": rel["p"],
                    "o": o,
                    "confidence": rel["confidence"],
                }
            )

        for ent in normalized_entities:
            ent_id = ent["entity_id"]
            for global_ent in entity_bank:
                if global_ent["entity_id"] == ent_id:
                    global_ent["evidence"].append(item["frame_id"])
                    break

        total_relations += len(normalized_relations)
        graph["frames"].append(
            {
                "frame_id": item["frame_id"],
                "frame_idx": item["frame_idx"],
                "t_sec": item["t_sec"],
                "path": item["path"],
                "caption_context": caption_text,
                "entities": normalized_entities,
                "relations": normalized_relations,
            }
        )

    graph["stats"]["num_entities"] = len(entity_bank)
    graph["stats"]["num_frames"] = len(graph["frames"])
    graph["stats"]["num_relations"] = total_relations
    graph["stats"]["vlm_success_frames"] = vlm_success_frames
    graph["stats"]["vlm_empty_frames"] = vlm_empty_frames
    graph["stats"]["caption_retry_frames"] = caption_retry_frames

    ok, reason = validate_stage1_graph(graph)
    if not ok:
        raise ValueError(f"Stage1 graph validation failed: {reason}")

    _ensure_parent_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    return graph


def ensure_stage1_scene_graph(
    video_caption_json_path: str,
    output_path: str,
    force_rebuild: bool = False,
    query_hint: str = "",
    frame_dir_override: str | None = None,
) -> dict[str, Any]:
    return build_stage1_scene_graph(
        video_caption_json_path=video_caption_json_path,
        output_path=output_path,
        force_rebuild=force_rebuild,
        query_hint=query_hint,
        frame_dir_override=frame_dir_override,
    )
