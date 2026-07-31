from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser()
    return default


def data_root() -> Path:
    return _env_path("SGD_DATA_ROOT", repo_root() / "data")


def lvbench_root() -> Path:
    return _env_path("SGD_LVBENCH_ROOT", data_root() / "LVBench" / "LVBench_4.1")


def artifacts_root() -> Path:
    """
    Writable root for experiment artifacts (captions/scene_graph/reports by run).
    """
    return _env_path("SGD_ARTIFACTS_ROOT", data_root() / "eval_artifacts")


def caption_root() -> Path:
    """
    Caption root used by benchmark runs.
    Defaults to SGD_CAPTION_ROOT; fallback to assets root for backward compatibility.
    """
    return _env_path("SGD_CAPTION_ROOT", lvbench_root())


def badcases_root() -> Path:
    """
    Backward-compatible alias:
    - prefer SGD_BADCASES_ROOT if provided
    - otherwise fallback to artifacts_root()/badcases
    """
    return _env_path("SGD_BADCASES_ROOT", artifacts_root() / "badcases")


def logs_root() -> Path:
    return _env_path("SGD_LOGS_ROOT", data_root() / "logs")


def eval_output_root() -> Path:
    return _env_path("SGD_EVAL_OUTPUT_ROOT", data_root() / "evaluation")
