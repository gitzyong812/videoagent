from __future__ import annotations

import sgd.config as config
import argparse
import os
from pathlib import Path

from sgd.sgd_core import SGDCoreAgent
from sgd.video_utils import load_video, decode_video_to_frames
from sgd.frame_caption import process_video
from sgd.utils import extract_answer


def _infer_video_id(video_path: str, resolved_video_path: str | None = None) -> str:
    source = resolved_video_path or video_path
    video_id = Path(source).stem
    if not video_id:
        raise ValueError(f"Could not infer a stable video id from '{video_path}'.")
    return video_id


def _video_paths(video_id: str) -> dict[str, str]:
    video_root = os.path.join(config.VIDEO_DATABASE_FOLDER, video_id)
    return {
        "video_root": video_root,
        "frames_dir": os.path.join(video_root, "frames"),
        "captions_dir": os.path.join(video_root, "captions"),
        "video_db_path": os.path.join(video_root, "database.json"),
    }


def _find_existing_raw_video(video_id: str) -> str | None:
    raw_dir = Path(config.VIDEO_DATABASE_FOLDER) / "raw"
    if not raw_dir.exists():
        return None

    for candidate in sorted(raw_dir.iterdir()):
        if candidate.is_file() and candidate.stem == video_id:
            return str(candidate.resolve())
    return None


def _answer_from_messages(messages) -> str | None:
    if not messages:
        return None
    terminal = messages[-1]
    if not isinstance(terminal, dict) or terminal.get("role") != "assistant":
        return None
    tool_calls = terminal.get("tool_calls") or []
    if tool_calls and any(
        call.get("function", {}).get("name") != "finish"
        for call in tool_calls
    ):
        return None
    try:
        answer = extract_answer(terminal)
    except (KeyError, TypeError, ValueError):
        return None
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return None


def main():
    parser = argparse.ArgumentParser(description="Run SGDCoreAgent on a video.")
    parser.add_argument("video_path", help="A local video path.")
    parser.add_argument("question", help="The question to ask about the video.")
    args = parser.parse_args()

    video_path_input = args.video_path
    question = args.question
    if not os.path.isfile(video_path_input):
        parser.error(f"video_path must be an existing local file: {video_path_input}")

    os.makedirs(config.VIDEO_DATABASE_FOLDER, exist_ok=True)
    video_id = _infer_video_id(video_path_input)
    paths = _video_paths(video_id)
    captions_dir = paths["captions_dir"]
    video_db_path = paths["video_db_path"]
    video_path = _find_existing_raw_video(video_id)
    if video_path is None:
        print(f"Copying local video from {video_path_input}...")
        video_path = load_video(video_path_input)
        video_id = _infer_video_id(video_path_input, resolved_video_path=video_path)
        paths = _video_paths(video_id)
        captions_dir = paths["captions_dir"]
        video_db_path = paths["video_db_path"]
        print("Video copied.")
    else:
        print(f"Video already exists at {video_path}.")

    frames_dir = paths["frames_dir"]
    if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
        print(f"Decoding video to frames in {frames_dir}...")
        decode_video_to_frames(video_path)
        print("Video decoded.")
    else:
        print(f"Frames already exist in {frames_dir}.")

    caption_file = os.path.join(captions_dir, "captions.json")
    if not os.path.exists(caption_file):
        print("Processing video to get captions...")
        process_video(frames_dir, captions_dir)
        print("Captions generated.")
    else:
        print(f"Captions already exist at {caption_file}.")

    # Initialize SGDCoreAgent
    print("Initializing SGDCoreAgent...")
    agent = SGDCoreAgent(video_db_path, caption_file, config.MAX_ITERATIONS)
    print("Agent initialized.")

    # Run with question
    print(f"Running agent with question: '{question}'")
    msgs = agent.run(question)
    answer = _answer_from_messages(msgs)
    if answer is None:
        raise RuntimeError(
            "The agent returned no usable answer. Review the preceding API and tool errors."
        )
    print(answer)

if __name__ == "__main__":
    main()
