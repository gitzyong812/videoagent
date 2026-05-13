from __future__ import annotations

import os
import shutil
from urllib.parse import urlparse

import cv2

import sgd.config as config


def _is_youtube_url(url: str) -> bool:
    parsed_url = urlparse(url)
    return parsed_url.netloc.lower().endswith(("youtube.com", "youtu.be"))


def _find_downloaded_srt(directory: str, video_id: str) -> str | None:
    for filename in os.listdir(directory):
        if filename.startswith(video_id) and filename.endswith(".srt"):
            return os.path.join(directory, filename)
    return None


def load_video(
    video_source: str,
    output_path: str | None = None,
    with_subtitle: bool = False,
    subtitle_source: str | None = None,
) -> str:
    """
    Load a YouTube URL or local video file into the raw video folder.

    Args:
        video_source: YouTube URL or local video file path.
        output_path: Optional exact destination path, kept for legacy callers.
        with_subtitle: If True, also download/copy an SRT subtitle.
        subtitle_source: Local SRT path for local videos.

    Returns:
        Absolute path to the video file stored in the database.
    """
    raw_video_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, "raw")
    os.makedirs(raw_video_dir, exist_ok=True)

    if video_source.startswith(("http://", "https://")):
        if not _is_youtube_url(video_source):
            raise ValueError("Provided URL is not a valid YouTube link.")
        import yt_dlp

        outtmpl = os.path.join(raw_video_dir, "%(id)s.%(ext)s")
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            outtmpl = os.path.splitext(output_path)[0] + ".%(ext)s"

        ydl_opts = {
            "format": (
                f"bestvideo[height<={config.VIDEO_RESOLUTION}][ext=mp4]+bestaudio[ext=m4a]/"
                f"best[height<={config.VIDEO_RESOLUTION}][ext=mp4]/best"
            ),
            "outtmpl": outtmpl,
            "merge_output_format": "mp4",
        }
        if with_subtitle:
            ydl_opts.update({
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitlesformat": "srt",
                "overwritesubtitles": True,
            })

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_source, download=True)
            downloaded_path = ydl.prepare_filename(info)
            if not downloaded_path.endswith(".mp4"):
                downloaded_path = os.path.splitext(downloaded_path)[0] + ".mp4"

        if output_path and os.path.abspath(downloaded_path) != os.path.abspath(output_path):
            if os.path.exists(downloaded_path):
                shutil.move(downloaded_path, output_path)
                downloaded_path = output_path

        if with_subtitle:
            subtitle_path = f"{os.path.splitext(downloaded_path)[0]}.srt"
            downloaded_subtitle = _find_downloaded_srt(os.path.dirname(downloaded_path), info["id"])
            if downloaded_subtitle and downloaded_subtitle != subtitle_path:
                shutil.move(downloaded_subtitle, subtitle_path)

        return os.path.abspath(downloaded_path)

    if not os.path.exists(video_source):
        raise FileNotFoundError(f"Video path '{video_source}' does not exist.")
    if not os.path.isfile(video_source):
        raise ValueError(f"Source path '{video_source}' is a directory, not a file.")

    destination_path = output_path or os.path.join(raw_video_dir, os.path.basename(video_source))
    os.makedirs(os.path.dirname(os.path.abspath(destination_path)), exist_ok=True)
    shutil.copy2(video_source, destination_path)

    if with_subtitle:
        if not subtitle_source:
            raise ValueError("subtitle_source must be provided for local videos.")
        if not subtitle_source.lower().endswith(".srt"):
            raise ValueError("Only SRT subtitle files are supported for local videos.")
        if not os.path.isfile(subtitle_source):
            raise FileNotFoundError(f"Subtitle file '{subtitle_source}' not found.")
        subtitle_destination = f"{os.path.splitext(destination_path)[0]}.srt"
        shutil.copy2(subtitle_source, subtitle_destination)

    return os.path.abspath(destination_path)


def download_srt_subtitle(video_url: str, output_path: str):
    """Download an SRT subtitle from a YouTube URL."""
    if not _is_youtube_url(video_url):
        raise ValueError("Provided URL is not a valid YouTube link.")
    import yt_dlp

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "writesubtitles": True,
        "subtitlesformat": "srt",
        "skip_download": True,
        "writeautomaticsub": True,
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        video_id = info["id"]
        ydl.download([video_url])

    downloaded_subtitle_path = _find_downloaded_srt(output_dir, video_id)
    if not downloaded_subtitle_path:
        raise FileNotFoundError(f"Could not find SRT subtitle for {video_url}")
    shutil.move(downloaded_subtitle_path, output_path)


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
