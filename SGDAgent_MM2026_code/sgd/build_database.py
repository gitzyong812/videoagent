from __future__ import annotations

import json
import multiprocessing
import os
from typing import Annotated as A
from typing import List, Optional, Tuple

import numpy as np
from nano_vectordb import NanoVectorDB
from tqdm import tqdm

import sgd.config as config
from sgd.func_call_shema import doc as D
from sgd.scene_graph import (ensure_stage1_scene_graph,
                             ensure_stage2_segment_graph,
                             ensure_global_event_graph,
                             query_global_event_graph_by_path,
                             query_stage1_graph_by_path,
                             query_stage2_segment_graph_by_path)
from sgd.utils import AzureOpenAIEmbeddingService, call_openai_model_with_tools


def frame_inspect_tool(
    database: A[NanoVectorDB, D("The database containing video metadata. Must be an instance of NanoVectorDB.")], 
    question: A[str, D("The specific detailed question to ask about the video content during the specified time ranges. No need to add time ranges in the question.")], 
    time_ranges_hhmmss: A[list[tuple], D("A list of tuples containing start and end times in HH:MM:SS format. If the time range is longer than 50 seconds, the function samples 50 evenly distributed frames.  Otherwise, it uses all frames within the specified range.")], 
) -> str:
    """
    Crop the video frames based on the time ranges and ask the model a detailed question about the cropped video clips.
    Returns:
        str: The model's response to the question. If no relevant content is found within the time range,
             returns an error message: "Error: Cannot find corresponding result in the given time range."
    """

    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    video_length_secs = convert_hhmmss_to_seconds(database.get_additional_data()['video_length'])
    video_meta = database.get_additional_data()
    video_file_root = video_meta['video_file_root']
    fps = video_meta.get('fps', config.VIDEO_FPS)
    time_ranges_secs = []
    for time_range in time_ranges_hhmmss:
        start_secs = convert_hhmmss_to_seconds(time_range[0])
        end_secs = convert_hhmmss_to_seconds(time_range[1])
        if start_secs > video_length_secs:
            raise ValueError(f"One of start time {time_range[0]} exceeds video length {video_length_secs}")
        end_secs = min(end_secs, video_length_secs)
        time_ranges_secs.append((start_secs, end_secs))

    time_ranges_secs.sort(key=lambda x: x[0])  # Sort by start time
    # Calculate total time across all ranges
    total_time = sum(end - start for start, end in time_ranges_secs)
    
    # Maximum number of timepoints to sample
    max_timepoints = config.AOAI_TOOL_VLM_MAX_FRAME_NUM
    timepoints = []

    assert total_time > 0 and max_timepoints > 0 

    # ① Uniformly sample on the flattened timeline
    #    endpoint=False ensures the last sample point < total_time
    offsets = np.linspace(
        0, total_time,
        num=max_timepoints,
        endpoint=False,
        dtype=float
    )

    # ② Calculate prefix sums for each segment, used to map offsets back to actual timestamps
    prefix_len = []          # (cumulative length, segment start, segment length)
    acc = 0
    for start, end in time_ranges_secs:
        seg_len   = end - start
        prefix_len.append((acc, start, seg_len))
        acc += seg_len

    # ③ Complete the mapping
    for off in offsets:
        # off = int(round(off))          # Ensure it is an integer
        for base, seg_start, seg_len in prefix_len:
            if off < base + seg_len:   # Find the corresponding segment
                timepoints.append(seg_start + (off - base))
                break
    # ④ Convert the sampled timestamps (seconds) to frame indices according to VIDEO_FPS
    max_frame_idx = int(video_length_secs * fps) - 1   # last valid frame index

    framepoints = [
        min(max(int(round(ts * fps)), 0), max_frame_idx)  # clamp to [0, max_frame_idx]
        for ts in timepoints
    ]
    framepoints = sorted(set(framepoints))[:max_timepoints]


    input_msgs = [
        {
            "role": "system",
            "content": "You are a helpful assistant to answer questions."
        },
        {
            "role": "user",
            "content": "Carefully watch the video frames. Pay attention to the cause and sequence of events, the detail and movement of objects and the action and pose of persons.\n\nBased on your observations, if you find content that can answer the question. If no relevant content is found within the given time range, return: `Error: Cannot find corresponding result in the given time range.`. \nQuestion: {question}\n",
        },
    ]

    input_msgs[1]['content'] = input_msgs[1]['content'].format(question=question)

    frame_dir = os.path.join(video_file_root, "frames")
    frame_prefix = "frames_n"
    index_offset = 0
    if framepoints:
        # Detect prefix and whether indices are 0- or 1-based.
        for candidate in ("frames_n", "frame_n"):
            if os.path.exists(os.path.join(frame_dir, f"{candidate}000001.jpg")) or os.path.exists(
                os.path.join(frame_dir, f"{candidate}000000.jpg")
            ):
                frame_prefix = candidate
                break
        if os.path.exists(os.path.join(frame_dir, f"{frame_prefix}000001.jpg")) and not os.path.exists(
            os.path.join(frame_dir, f"{frame_prefix}000000.jpg")
        ):
            index_offset = 1
    files = [
        os.path.join(frame_dir, f"{frame_prefix}{fn + index_offset:06d}.jpg") for fn in framepoints
    ]
    msgs = call_openai_model_with_tools(
        messages=input_msgs,
        endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
        model_name=config.AOAI_TOOL_VLM_MODEL_NAME,
        api_key=config.OPENAI_API_KEY,
        image_paths=files,
        temperature=0,
        max_tokens=512,
    )
    if msgs is None:
        raise ValueError("No response from the model")
    return msgs["content"]

def caption_backoff_tool(
    database: A[NanoVectorDB, D("The database object that supports caption embedding retrieval.")],
    query: A[str, D("The question or surface-form event description to search in the caption index.")],
    top_k: A[int, D("Maximum number of caption matches to return.")] = 16,
) -> str:
    """Search the embedding-based caption index as a backoff retrieval channel."""
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    query_emb = AzureOpenAIEmbeddingService.get_embeddings(
        endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
        model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
        input_text=[query],
        api_key=config.OPENAI_API_KEY,
    )[0]['embedding']
    results = database.query(query_emb, top_k=top_k)
    captions = [
        (data['time_start_secs'], data['caption'])
        for data in results
    ]
    captions = sorted(captions, key=lambda x: x[0])
    captions = "\n".join([cap[1] for cap in captions])
    return "Here is the searched video clip scripts:\n\n" + captions


def frame_graph_query_tool(
        database: A[NanoVectorDB, D("The database containing scene graph metadata.")],
        query: A[str, D("Question or keyword query for frame-level graph retrieval.")],
        time_reference: A[Optional[str], D("Optional explicit time reference used to bias graph ranking.")] = None,
        scope: A[Optional[List[Tuple]], D("Optional parent-level time ranges that hard-filter frame candidates.")] = None,
        top_k_frames: A[int, D("Top-k matched frames to return.")] = 8,
        max_triples: A[int, D("Maximum matched triples to include in the response.")] = 24,
) -> str:
    """
    Retrieve evidence from the Stage-1 frame-level scene graph.

    Returns:
        str: JSON string containing matched frames and top triples.
    """
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    add_data = database.get_additional_data()
    graph_path = add_data.get("stage1_scene_graph_path")
    if not graph_path or not os.path.exists(graph_path):
        return json.dumps(
            {
                "error": "Stage1 scene graph is unavailable for this video.",
                "hint": "Enable SCENE_GRAPH_STAGE1_ENABLED and rebuild or initialize database.",
            }
        )
    result = query_stage1_graph_by_path(
        graph_path=graph_path,
        query=query,
        time_reference=time_reference,
        scope=scope,
        top_k_frames=top_k_frames,
        max_triples=max_triples,
    )
    return json.dumps(result, ensure_ascii=False)


def segment_graph_query_tool(
        database: A[NanoVectorDB, D("The database containing scene graph metadata.")],
        query: A[str, D("Question or keyword query for segment-level graph retrieval.")],
        time_reference: A[Optional[str], D("Optional explicit time reference used to bias graph ranking.")] = None,
        scope: A[Optional[List[Tuple]], D("Optional global-level time ranges that hard-filter segment candidates.")] = None,
        top_k_segments: A[int, D("Top-k matched segments to return.")] = 6,
        max_triples: A[int, D("Maximum matched temporal triples to include in the response.")] = 24,
) -> str:
    """
    Retrieve evidence from the Stage-2 segment-level temporal graph.

    Returns:
        str: JSON string containing matched segments and temporal triples.
    """
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    add_data = database.get_additional_data()
    graph_path = add_data.get("stage2_segment_graph_path")
    if not graph_path or not os.path.exists(graph_path):
        return json.dumps(
            {
                "error": "Stage2 segment graph is unavailable for this video.",
                "hint": "Enable SCENE_GRAPH_STAGE2_ENABLED and rebuild or initialize database.",
            }
        )
    result = query_stage2_segment_graph_by_path(
        graph_path=graph_path,
        query=query,
        time_reference=time_reference,
        scope=scope,
        top_k_segments=top_k_segments,
        max_triples=max_triples,
    )
    return json.dumps(result, ensure_ascii=False)


def global_graph_query_tool(
    database: A[NanoVectorDB, D("The database containing global event graph metadata.")],
    query: A[str, D("Question or keyword query for global event graph retrieval.")],
    scope: A[Optional[List[Tuple]], D("Optional time ranges that hard-filter global candidate regions.")] = None,
    time_reference: A[Optional[str], D("Optional explicit question time reference used in ranking.")] = None,
    top_k_regions: A[int, D("Top-k matched global regions to return.")] = 5,
    max_relations: A[int, D("Maximum high-order relations to include.")] = 16,
) -> str:
    """Retrieve global event, state-evolution, temporal, and causal evidence."""
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    graph_path = database.get_additional_data().get("global_event_graph_path")
    if not graph_path or not os.path.exists(graph_path):
        return json.dumps(
            {
                "error": "Global event graph is unavailable for this video.",
                "hint": "Enable SCENE_GRAPH_GLOBAL_ENABLED and rebuild the index.",
            }
        )
    result = query_global_event_graph_by_path(
        graph_path=graph_path,
        query=query,
        scope=scope,
        time_reference=time_reference,
        top_k_regions=top_k_regions,
        max_relations=max_relations,
    )
    return json.dumps(result, ensure_ascii=False)


def convert_seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def convert_hhmmss_to_seconds(hhmmss):
    hhmmss = hhmmss.split('.')[0]
    parts = hhmmss.split(":")
    if len(parts) < 2:
        raise ValueError("Invalid time format. Expected HH:MM:SS.")
    elif len(parts) == 2:
        parts = ["00"] + parts
    hours, minutes, seconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds

def _default_stage1_graph_path(video_caption_json_path: str) -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(video_caption_json_path),
        getattr(config, "SCENE_GRAPH_STAGE1_FILE_NAME", "scene_graph_stage1.json"),
    ))


def _default_stage2_graph_path(stage1_graph_path: str) -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(stage1_graph_path),
        getattr(config, "SCENE_GRAPH_STAGE2_FILE_NAME", "scene_graph_stage2_segment.json"),
    ))


def _default_global_graph_path(stage2_graph_path: str) -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(stage2_graph_path),
        getattr(config, "SCENE_GRAPH_GLOBAL_FILE_NAME", "scene_graph_global.json"),
    ))


def _stage1_graph_candidates(video_caption_json_path: str, existing_path: str | None = None) -> list[str]:
    candidates = []
    if existing_path:
        candidates.append(os.path.abspath(existing_path))

    captions_dir = os.path.dirname(video_caption_json_path)
    candidates.append(_default_stage1_graph_path(video_caption_json_path))

    deduped = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _stage2_graph_candidates(stage1_graph_path: str, existing_path: str | None = None) -> list[str]:
    candidates = []
    if existing_path:
        candidates.append(os.path.abspath(existing_path))

    graph_dir = os.path.dirname(stage1_graph_path)
    candidates.append(_default_stage2_graph_path(stage1_graph_path))

    deduped = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _reuse_existing_graph_path(candidates: list[str], graph_label: str, force_rebuild: bool) -> str | None:
    if force_rebuild:
        return None
    for path in candidates:
        if os.path.exists(path):
            print(f"[SCENE_GRAPH] Reusing existing {graph_label}: {path}")
            return path
    return None


def _ensure_stage1_graph_file(
    video_caption_json_path: str,
    existing_path: str | None = None,
    frame_dir_override: str | None = None,
) -> str | None:
    if not getattr(config, "SCENE_GRAPH_STAGE1_ENABLED", False):
        return existing_path
    force_rebuild = bool(getattr(config, "SCENE_GRAPH_STAGE1_FORCE_REBUILD", False))
    graph_candidates = _stage1_graph_candidates(
        video_caption_json_path=video_caption_json_path,
        existing_path=existing_path,
    )
    reused_path = _reuse_existing_graph_path(graph_candidates, "Stage1 graph", force_rebuild)
    if reused_path is not None:
        return reused_path
    graph_path = _default_stage1_graph_path(video_caption_json_path)
    ensure_stage1_scene_graph(
        video_caption_json_path=video_caption_json_path,
        output_path=graph_path,
        force_rebuild=force_rebuild,
        frame_dir_override=frame_dir_override,
    )
    return graph_path


def _ensure_stage2_graph_file(stage1_graph_path: str, existing_path: str | None = None) -> str | None:
    if not getattr(config, "SCENE_GRAPH_STAGE2_ENABLED", False):
        return existing_path
    force_rebuild = bool(getattr(config, "SCENE_GRAPH_STAGE2_FORCE_REBUILD", False))
    graph_candidates = _stage2_graph_candidates(
        stage1_graph_path=stage1_graph_path,
        existing_path=existing_path,
    )
    reused_path = _reuse_existing_graph_path(graph_candidates, "Stage2 graph", force_rebuild)
    if reused_path is not None:
        return reused_path
    graph_path = _default_stage2_graph_path(stage1_graph_path)
    ensure_stage2_segment_graph(
        stage1_graph_path=stage1_graph_path,
        output_path=graph_path,
        force_rebuild=force_rebuild,
    )
    return graph_path


def _ensure_global_graph_file(
    stage1_graph_path: str,
    stage2_graph_path: str,
    existing_path: str | None = None,
) -> str | None:
    if not getattr(config, "SCENE_GRAPH_GLOBAL_ENABLED", True):
        return existing_path
    force_rebuild = bool(getattr(config, "SCENE_GRAPH_GLOBAL_FORCE_REBUILD", False))
    if existing_path and not force_rebuild and os.path.exists(existing_path):
        print(f"[SCENE_GRAPH] Reusing existing global graph: {existing_path}")
        return os.path.abspath(existing_path)
    graph_path = _default_global_graph_path(stage2_graph_path)
    if os.path.exists(graph_path) and not force_rebuild:
        print(f"[SCENE_GRAPH] Reusing existing global graph: {graph_path}")
        return graph_path
    ensure_global_event_graph(
        stage1_graph_path=stage1_graph_path,
        stage2_graph_path=stage2_graph_path,
        output_path=graph_path,
        force_rebuild=force_rebuild,
    )
    return graph_path

def init_single_video_db(
    video_caption_json_path,
    output_video_db_path,
    emb_dim,
    video_file_root_override: str | None = None,
):
    vdb = NanoVectorDB(emb_dim, storage_file=output_video_db_path)
    if os.path.exists(output_video_db_path):
        print(f"Database {output_video_db_path} already exists.")
        additional_data = vdb.get_additional_data() or {}
        if video_file_root_override:
            additional_data["video_file_root"] = os.path.abspath(video_file_root_override)
        elif "video_file_root" not in additional_data and video_caption_json_path:
            additional_data["video_file_root"] = os.path.abspath(
                os.path.dirname(os.path.dirname(video_caption_json_path))
            )
        if "fps" not in additional_data:
            additional_data["fps"] = getattr(config, "VIDEO_FPS", 2)
        if "video_file_root" not in additional_data:
            if video_file_root_override:
                additional_data["video_file_root"] = os.path.abspath(video_file_root_override)
            elif video_caption_json_path:
                additional_data["video_file_root"] = os.path.abspath(os.path.dirname(os.path.dirname(video_caption_json_path)))
            else:
                additional_data["video_file_root"] = os.path.abspath(os.path.dirname(output_video_db_path))

        needs_caption_metadata = any(
            key not in additional_data
            for key in ("subject_registry", "video_length")
        )
        if needs_caption_metadata:
            if not video_caption_json_path:
                raise ValueError(
                    "Existing database is missing subject_registry/video_length metadata, "
                    "but `video_caption_json_path` is None."
                )
            with open(video_caption_json_path, "r") as f:
                captions = json.load(f)
            subject_registry = captions.pop('subject_registry', captions.pop('character_registry', None))
            video_length = max([float(k.split("_")[1]) for k in captions.keys()])
            additional_data.update(
                {
                    "subject_registry": subject_registry,
                    "video_length": convert_seconds_to_hhmmss(video_length),
                }
            )
    else:
        cap2emb_list = preprocess_captions(video_caption_json_path)
        data = []
        for idx, (timestamp, cap, emb) in enumerate(cap2emb_list):
            t1 = convert_seconds_to_hhmmss(timestamp[0])
            t2 = convert_seconds_to_hhmmss(timestamp[1])
            prefix = f"[From {t1} to {t2} seconds]\n"
            data.append(
                {
                    "__vector__": np.array(emb),
                    "time_start_secs": timestamp[0],
                    "time_end_secs": timestamp[1],
                    "caption": prefix + cap['caption'],
                }
            )
        _ = vdb.upsert(data)
        with open(video_caption_json_path, "r") as f:
            captions = json.load(f)
        subject_registry = captions.pop('subject_registry', captions.pop('character_registry', None))          
        video_length = max([float(k.split("_")[1]) for k in captions.keys()])
        video_length = convert_seconds_to_hhmmss(video_length)
        addtional_data = {
            'subject_registry': subject_registry,
            'video_length': video_length,
            'video_file_root': (
                os.path.abspath(video_file_root_override)
                if video_file_root_override
                else os.path.abspath(os.path.dirname(os.path.dirname(video_caption_json_path)))
            ),
            'fps': getattr(config, "VIDEO_FPS", 2),
        }
        additional_data = addtional_data

    resolved_caption_path = None
    if video_caption_json_path:
        resolved_caption_path = os.path.abspath(video_caption_json_path)
    elif additional_data.get("video_caption_json_path"):
        resolved_caption_path = os.path.abspath(additional_data["video_caption_json_path"])

    if resolved_caption_path:
        additional_data["video_caption_json_path"] = resolved_caption_path
        frame_dir_override = None
        if additional_data.get("video_file_root"):
            frame_dir_override = os.path.join(str(additional_data["video_file_root"]), "frames")
        existing_stage1_graph = additional_data.get("stage1_scene_graph_path")
        if existing_stage1_graph:
            try:
                if os.path.dirname(os.path.abspath(existing_stage1_graph)) != os.path.dirname(resolved_caption_path):
                    existing_stage1_graph = None
            except Exception:
                existing_stage1_graph = None
        graph_path = _ensure_stage1_graph_file(
            video_caption_json_path=resolved_caption_path,
            existing_path=existing_stage1_graph,
            frame_dir_override=frame_dir_override,
        )
        if graph_path:
            additional_data["stage1_scene_graph_path"] = graph_path
            existing_stage2_graph = additional_data.get("stage2_segment_graph_path")
            if existing_stage2_graph:
                try:
                    if os.path.dirname(os.path.abspath(existing_stage2_graph)) != os.path.dirname(os.path.abspath(graph_path)):
                        existing_stage2_graph = None
                except Exception:
                    existing_stage2_graph = None
            segment_graph_path = _ensure_stage2_graph_file(
                stage1_graph_path=graph_path,
                existing_path=existing_stage2_graph,
            )
            if segment_graph_path:
                additional_data["stage2_segment_graph_path"] = segment_graph_path
                global_graph_path = _ensure_global_graph_file(
                    stage1_graph_path=graph_path,
                    stage2_graph_path=segment_graph_path,
                    existing_path=additional_data.get("global_event_graph_path"),
                )
                if global_graph_path:
                    additional_data["global_event_graph_path"] = global_graph_path
    elif additional_data.get("stage1_scene_graph_path"):
        segment_graph_path = _ensure_stage2_graph_file(
            stage1_graph_path=additional_data["stage1_scene_graph_path"],
            existing_path=additional_data.get("stage2_segment_graph_path"),
        )
        if segment_graph_path:
            additional_data["stage2_segment_graph_path"] = segment_graph_path
            global_graph_path = _ensure_global_graph_file(
                stage1_graph_path=additional_data["stage1_scene_graph_path"],
                stage2_graph_path=segment_graph_path,
                existing_path=additional_data.get("global_event_graph_path"),
            )
            if global_graph_path:
                additional_data["global_event_graph_path"] = global_graph_path
    vdb.store_additional_data(**additional_data)
    vdb.save()
    return vdb

def preprocess_captions(caption_json_path):
    with open(caption_json_path, "r") as f:
        captions = json.load(f)
    scripts = []
    captions.pop('subject_registry', None)
    captions.pop('character_registry', None)
    for idx, (timestamp, cap_info) in enumerate(captions.items()):
        if cap_info.get('caption') is None or len(cap_info['caption']) == 0:
            print(f"Empty caption information for {timestamp} in {caption_json_path}")
            continue
        elif isinstance(cap_info['caption'], list):
            cap_info['caption'] = cap_info['caption'][0]
        elif not isinstance(cap_info['caption'], str):
            print(f"Invalid caption type for {cap_info['caption']}")
            cap_info['caption'] = str(cap_info['caption'])
        timestamp = list(map(float, timestamp.split("_")))
        scripts.append((timestamp, cap_info['caption'], cap_info))

    # batchify
    batch_size = 128
    batched_scripts = []
    print(f"Embedding {len(scripts)} captions...")
    for i in range(0, len(scripts), batch_size):
        batch = scripts[i:i+batch_size]
        batched_scripts.append(batch)
    cap2emb_list = []
    with multiprocessing.Pool(int(getattr(config, "BUILD_WORKERS", 4))) as pool:
        with tqdm(total=len(scripts), desc="Embedding captions...") as pbar:
            for result in pool.imap_unordered(
                single_batch_embedding_task,
                batched_scripts,
            ):
                cap2emb_list.extend(result)
                pbar.update(len(result))
    return cap2emb_list

def single_batch_embedding_task(data):
    timestamps, captions, cap_infos = map(list, (zip(*data)))
    embs = AzureOpenAIEmbeddingService.get_embeddings(
        endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
        model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
        input_text=captions,
        api_key=config.OPENAI_API_KEY,
    )
    max_tries = 3
    while embs is None or len(embs) != len(captions):
        max_tries -= 1
        if max_tries < 0:
            raise ValueError(f"Failed to get embeddings for {timestamps} {captions}")
        print(f"Error in embedding {timestamps} {captions} retrying...")
        embs = AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
            model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            input_text=captions,
            api_key=config.OPENAI_API_KEY,
        )
    return list(zip(timestamps, cap_infos, [d['embedding'] for d in embs]))
