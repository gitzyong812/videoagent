from __future__ import annotations

import os
import shutil

import cv2

import sgd.config as config


def load_video(video_path: str) -> str:
    """
    Copies a local video file into the repository's raw video folder.

    Args:
        video_path: Local video file path.

    Returns:
        Absolute path to the video file stored in the database.

    Raises:
        ValueError, FileNotFoundError: On invalid inputs.
    """
    raw_video_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'raw')
    os.makedirs(raw_video_dir, exist_ok=True)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path '{video_path}' does not exist.")
    if not os.path.isfile(video_path):
        raise ValueError(f"Source path '{video_path}' is a directory, not a file.")

    filename = os.path.basename(video_path)
    destination_path = os.path.join(raw_video_dir, filename)
    shutil.copy2(video_path, destination_path)
    return os.path.abspath(destination_path)


def decode_video_to_frames(video_path: str) -> str:
    """
    Decodes a video into JPEG frames at the frame rate specified by config.VIDEO_FPS.
    Frames are saved in config.VIDEO_DATABASE_PATH/video_names/frames/.

    Args:
        video_path: The absolute path to the video file.

    Returns:
        The absolute path to the directory containing the extracted frames.

    Raises:
        FileNotFoundError: If the video file does not exist.
        Exception: If frame extraction fails.
    """

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' does not exist.")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, video_name, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Failed to open video file '{video_path}'.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = getattr(config, 'VIDEO_FPS', fps)
    frame_interval = int(round(fps / target_fps)) if target_fps < fps else 1

    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(frames_dir, f"frame_n{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    return os.path.abspath(frames_dir)
