# SGDAgent

Cross-scale Spatio-Temporal Scene Graph representation for long video understanding. This repository contains the core source code for the scene graph construction pipeline and the agent-based VQA reasoning loop.

## Requirements

- Python 3.9+
- An OpenAI-compatible API key in `OPENAI_API_KEY`, or Azure CLI login for the Azure OpenAI path
- A local video file path as input

## Structure

```
local_run.py                  # Entry point: run agent on a video with a question
sgd/
  sgd_core.py                 # Agent loop with tool-calling orchestration
  build_database.py            # Tool implementations (frame inspect, clip search, graph query, etc.)
  frame_caption.py             # Frame-level VLM captioning with subject registry
  video_utils.py               # Video decoding and frame extraction
  utils.py                     # OpenAI / Azure API helpers and embedding utilities
  config.py                    # Global configuration (models, policy gates, graph parameters)
  path_config.py               # Environment-aware path resolution
  func_call_shema.py           # JSON schema generation for function-calling tools
  scene_graph/
    stage1_builder.py          # Stage-1: frame-level scene graph construction
    segment_builder.py         # Stage-2: segment-level graph aggregation with embeddings
    global_event_index.py      # Global event index compression for prompt injection
    query.py                   # Frame-level graph querying with temporal reasoning
    query_segment.py           # Segment-level semantic retrieval and temporal grounding
    schema.py                  # Graph node / edge data structures and validation
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the OpenAI-compatible API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Usage

```bash
python local_run.py "<local_video_path>" "<question>"
```

Examples:

```bash
python local_run.py "/path/to/video.mp4" "What is the person holding at the end?"
```

Notes:

- Only local video files are supported in this supplementary-material version.
- Intermediate artifacts are written under `video_database/`.

## Submission Packaging

To create a sanitized archive without `.git/`, caches, or generated artifacts:

```bash
bash package_submission.sh
```

The archive will be written to `dist/`.
