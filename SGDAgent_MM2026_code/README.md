# SGDAgent

This supplementary repository contains the core implementation of SGDAgent for structured long-form video understanding. It covers the offline three-level scene-graph index and the online graph-driven reasoning loop described in the paper.

## Core components

- Caption index for lexical backoff retrieval.
- Frame-level scene graphs with typed entities, spatial/state relations, and a shared video entity registry.
- Segment-level graphs built over overlapping 30-second windows with VLM-extracted actions, interactions, trajectories, and temporal grounding.
- A persistent global event graph with event, state-evolution, temporal, and causal relations linked to supporting segments.
- The `GR`, `SR`, `FR`, `FI`, `CB`, and `finish` tool chain with query-type routing, confidence gating, and terminal convergence.

The supplied code focuses on the method needed to inspect and reproduce the core pipeline.

## Requirements

- Python 3.10+
- An OpenAI-compatible API key in `OPENAI_API_KEY`, or an authenticated Azure CLI session with configured Azure endpoints
- A local video file

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## Model configuration

All model roles can be changed without editing source code:

| Environment variable | Default | Role |
| --- | --- | --- |
| `SGD_ORCHESTRATOR_MODEL` | `o4-mini` | Agent reasoning and tool selection |
| `SGD_CAPTION_MODEL` | `gpt-4.1-mini` | Clip caption index |
| `SGD_FRAME_GRAPH_MODEL` | `gpt-4.1` | Frame-level graph extraction |
| `SGD_SEGMENT_GRAPH_MODEL` | `gpt-4.1` | Segment actions and temporal relations |
| `SGD_GLOBAL_GRAPH_MODEL` | `gpt-4.1` | Global event and causal abstraction |
| `SGD_FRAME_INSPECT_MODEL` | `o3` | Raw-frame visual verification |
| `SGD_EMBEDDING_MODEL` | `text-embedding-3-large` | Caption and graph semantic retrieval |
| `SGD_BUILD_WORKERS` | `4` | Caption/embedding build workers |

The default reasoning, frame-graph, and frame-inspection roles match the reported paper configuration; the remaining roles are configurable. Frame transport failures and invalid segment/global outputs are reported as build errors rather than silently replaced by heuristic relations.

The paper-aligned defaults decode at 2 FPS, sample frame-graph candidates every 5 seconds with at most 120 keyframes, use 30-second segments with 5-second overlap, and group global candidates with the reported 12-second, 0.18, and 0.22 thresholds. `GR`, `SR`, and `FR` return at most 5, 6, and 8 candidates, and `FI` samples at most 50 frames. Ubiquitous generic labels such as `person` are ignored only as standalone global continuity cues to prevent degenerate all-video regions.

## Usage

Build the caption and three-level graph index without running QA:

```bash
python local_run.py "/path/to/video.mp4" --build-only
```

Build or reuse the index and answer a question:

```bash
python local_run.py "/path/to/video.mp4" "What is the person holding at the end?"
```

Generated artifacts are stored under `video_database/<video_id>/`:

```text
captions/captions.json
captions/scene_graph_stage1.json
captions/scene_graph_stage2_segment.json
captions/scene_graph_global.json
database.json
frames/
```

The global query returns time-bounded candidate regions and segment evidence links. These can be passed as `scope` to segment retrieval. Segment ranges can likewise scope frame retrieval before raw-frame inspection.


## Source layout

```text
local_run.py                       Local build and QA entry point
sgd/sgd_core.py                    Agent orchestration and control strategies
sgd/build_database.py              GR/SR/FR/FI/CB tools and index initialization
sgd/frame_caption.py               Caption index construction
sgd/scene_graph/stage1_builder.py  Frame graph and entity registry
sgd/scene_graph/segment_builder.py Segment VLM construction
sgd/scene_graph/global_builder.py  Global event graph construction
sgd/scene_graph/query*.py          Level-specific graph retrieval
sgd/scene_graph/schema.py          Graph schemas and validation
```
