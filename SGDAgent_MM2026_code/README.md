# SGDAgent

SGDAgent is a single-video question-answering pipeline built around cross-scale spatio-temporal scene graphs. Given one local video and one question, it constructs reusable video artifacts, retrieves candidate time regions at global, segment, and frame scales, and verifies fine visual details with sampled frames.

## Capability and scope

The included entry point runs one question against one local video:

```bash
python local_run.py "<local_video_path>" "<question>"
```

The representation has three scales:

1. **Frame scale.** Stage 1 samples keyframes, asks a visual model for entities and relations, aligns recurring entities, and stores the clip caption associated with each sampled time as context.
2. **Segment scale.** Stage 2 uses fixed 30-second windows with 5-second overlap. Given the same Stage-1 graph and configuration, window creation, entity aggregation, relation aggregation, support-frame links, and summaries are deterministic. Stage 2 does not make another VLM call to abstract segment events. Optional embeddings affect retrieval but not segment membership or aggregation.
3. **Global scale.** The Global Event Index deterministically clusters adjacent Stage-2 segments using temporal proximity and entity or predicate overlap. Each region contains a stable event identifier, time range, entity and action labels, keywords, and supporting segment identifiers.

### Basic global retrieval

`global_event_query_tool` performs lightweight retrieval over the Global Event Index. It combines exact token overlap, fuzzy string matching, and temporal cues, optionally applies a hard time scope, and returns at most five coarse event regions. Results include score components, segment provenance links, a recommended scope, and controller signals for the next retrieval step.

This is heuristic event-region retrieval, not a learned or VLM-abstracted global event graph. It has no causal edges, state-transition edges, event-relation paths, or learned global ranking. A global result only indicates where to continue searching. The agent must query the segment graph in the recommended scope and use frame-level retrieval or inspection when visual confirmation is needed.

### Caption-context limitations

- `local_run.py` does not accept a subtitle file. Its captioning calls therefore use sampled frames and the placeholder `No transcript.` unless `process_video` is called separately with an SRT path.
- Captions describe fixed 10-second clips. A Stage-1 frame receives the caption covering that timestamp, or the nearest caption when no interval covers it. The caption is clip-level context, not an observation tied exactly to that frame.
- If Stage-1 VLM parsing returns no entities, the builder can derive a limited fallback from caption text. Caption-derived fields and caption retrieval should be treated as localization context rather than independent visual evidence.
- Global Event Index keywords may inherit Stage-2 summaries and caption snippets. This package is therefore not suitable for a strict no-caption ablation.
- `frame_inspect_tool` samples at most 50 frame positions across all requested ranges. It does not inspect every source frame in a long interval.

## Reproduction boundary

The local command demonstrates the single-video construction and reasoning path. It is not a benchmark reproduction command. This package does not include benchmark videos, annotations, dataset splits, batch runners, metric computation, or the experiment-specific inputs needed to regenerate reported benchmark tables. Results from the local command should therefore be interpreted as method-level execution on a user-provided video, not as benchmark-score reproduction.

## Requirements

- Python 3.9 or newer
- A local video readable by OpenCV
- Access to the configured chat, vision, and embedding models
- Either an OpenAI-compatible API key or Azure CLI authentication with deployment endpoints

Install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### OpenAI-compatible API

When `OPENAI_API_KEY` is set, requests use `OPENAI_BASE_URL`, whose default is `https://api.openai.com/v1`.

```bash
export OPENAI_API_KEY="your-api-key"
# Optional for another compatible endpoint:
export OPENAI_BASE_URL="https://api.example.com/v1"
```

The configured model names must be available at that endpoint.

### Azure OpenAI

When `OPENAI_API_KEY` is unset, the code uses `AzureCliCredential`. Sign in with `az login` and provide the required endpoint lists as comma-separated environment variables:

```bash
export SGD_CAPTION_VLM_ENDPOINTS="https://<resource>.openai.azure.com"
export SGD_ORCHESTRATOR_ENDPOINTS="https://<resource>.openai.azure.com"
export SGD_TOOL_VLM_ENDPOINTS="https://<resource>.openai.azure.com"
export SGD_EMBEDDING_ENDPOINTS="https://<resource>.openai.azure.com"
```

The default endpoint lists are empty, so Azure execution requires these variables.

## Default models and processing settings

| Role | Default | Override |
| --- | --- | --- |
| Clip captioning and subject-registry merge | `gpt-4.1-mini` | `SGD_CAPTION_VLM_MODEL` |
| Agent orchestration | `o4-mini` | `SGD_ORCHESTRATOR_MODEL` |
| Frame inspection | `o3` | `SGD_TOOL_VLM_MODEL` |
| Stage-1 frame-graph extraction | `gpt-4.1` | `SGD_STAGE1_VLM_MODEL` |
| Caption and segment embeddings | `text-embedding-3-large`, 3072 dimensions | `SGD_EMBEDDING_MODEL`, `SGD_EMBEDDING_DIM` |

Other relevant defaults are 2 decoded frames per second, 10-second caption clips, 16 caption workers, Stage-1 sampling every 5 seconds with at most 120 keyframes, 30-second Stage-2 windows with 5-second overlap, at most 12 global event regions, at most 50 frame-inspection samples, and 20 agent iterations. GR, SR, and FR return at most 5, 6, and 8 regions by default. See `sgd/config.py` for the complete configuration.

## Single-video workflow

Run the command from this directory:

```bash
python local_run.py "/path/to/video.mp4" \
  "What is the person holding at the end?"
```

The command performs the following steps:

1. Copies the local video to `video_database/raw/`.
2. Decodes frames into `video_database/<video_id>/frames/`.
3. Generates 10-second clip captions and a subject registry.
4. Embeds captions and creates a local vector database.
5. Builds or reuses the Stage-1 frame graph.
6. Deterministically aggregates the Stage-2 segment graph.
7. Runs the coarse-to-fine agent loop, building and querying the Global Event Index when GR is called.
8. Prints the final answer.

Artifacts are cached under `video_database/<video_id>/` and reused on later runs. The video identifier is derived from the input filename stem, so different videos intended to remain independent should use distinct stems. Model calls can be numerous for long videos because captioning and Stage-1 extraction cover the full video.

## Source layout

```text
local_run.py                     Single-video command-line entry point
sgd/
  sgd_core.py                    Agent loop and coarse-to-fine control
  build_database.py              Retrieval and frame-inspection tools
  frame_caption.py               Clip captioning and subject registry
  video_utils.py                 Local video copy and frame decoding
  utils.py                       API, image, and embedding helpers
  config.py                      Models, processing defaults, and policy gates
  func_call_shema.py             Function-calling schema generation
  scene_graph/
    stage1_builder.py            Frame-level graph construction
    segment_builder.py           Deterministic segment aggregation
    global_event_index.py        Heuristic global event-region index
    query_global.py              Basic global event-region retrieval
    query.py                     Frame-graph retrieval
    query_segment.py             Segment-graph retrieval
    schema.py                    Graph schemas and validation
tests/                           Focused unit and smoke tests
```

## Submission packaging

Create a sanitized archive from this directory:

```bash
bash package_submission.sh
```

The script writes `dist/SGDAgent_submission_<date>.tar.gz`. The inner archive uses USTAR format, records members as `root/root`, omits macOS extended attributes, and excludes caches, local environments, generated video data, run outputs, logs, secrets, and existing archives.
