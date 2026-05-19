# [Archived] app-plan.md

Archived on: 2026-05-20

This document is retained for historical context. It is no longer the canonical
demo, visualization, or development plan.

Canonical replacements:
  - `docs/plans/SGD-Alg-plan.md` for graph-native Agent semantics,
    VisualVerify boundaries, and evidence-chain requirements.
  - `docs/plans/dev-plan.md` for the M11 evaluation/demo development track.

Reason:
  The original demo design contains caption-centric and free-form display ideas
  that are no longer valid for strict graph-only inference. Non-conflicting demo
  requirements have been consolidated into `dev-plan.md`.

---

## 1. 先定展示目标：SGDAgent 页面应该展示什么

根据你当前 SGDAgent 的设定，它不是普通 caption-retrieval agent，而是把长视频转换成“scene-graph-augmented video database”，包含 global、segment、frame 三层图以及 caption database，然后在 observe-reason-act 循环里走 propose-verify-answer 范式。 因此页面至少应该分成四块：

| 展示区                | 作用                    | 页面上应该看到什么                                                      |
| ------------------ | --------------------- | -------------------------------------------------------------- |
| Video / Query      | 输入与原视频定位              | 视频、问题、候选答案、参数                                                  |
| Graph Construction | 建图过程                  | frame-level graph、segment-level graph、global event graph 的生成进度 |
| Agent Trace        | observe-reason-act 循环 | 每一步 Observation、Reason 摘要、Action、Tool、候选节点/边、验证结果              |
| Evidence & Answer  | 最终结果                  | 最终答案、被采用的证据链、时间区间、关键帧/片段、图路径                                   |

DVD 的研究思路是基于多粒度数据库和工具集做 agentic search：Global Browse、Clip Search、Frame Inspect 分别覆盖全局理解、语义检索、细粒度视觉检查。 你的 SGDAgent 可以保留这种“多粒度工具”思想，但把核心展示对象从“caption/tool 文本日志”换成“**图节点、边、时间段、候选路径、验证结果**”。

## 2. 推荐技术栈

最直接的版本：**Gradio + NetworkX/PyVis 或 Plotly + JSONL trajectory**。

Gradio 合适，因为 DeepVideoDiscovery 本身也用 Gradio Blocks 构建 demo；Gradio Blocks 官方定位就是比 Interface 更灵活，可以自定义布局、组件和事件流，适合你这种多面板展示页。([Gradio][2]) 另外，Gradio 支持 generator streaming，也就是后端函数可以逐步 `yield` 输出，这正好对应 agent 的每个 observe-reason-act step。([Gradio][3])

建议栈：

```text
UI:        gradio.Blocks
图展示:     pyvis 或 plotly + gr.HTML
时间轴:     plotly timeline 或自定义 HTML
状态日志:   gr.Markdown / gr.JSON
证据帧:     gr.Gallery / gr.Image / gr.Video
后端:       SGDAgent.stream_run(...)
轨迹存储:   trajectory.jsonl
图数据:     graph_bundle.json / graph_nodes.json / graph_edges.json
```

如果只是内部 demo，Gradio 足够。若你想做论文项目主页级别的交互式展示，再考虑 React + Cytoscape.js/D3 + FastAPI。

## 3. 你需要先把 SGDAgent 后端改成“可观测”的

这是最重要的准备工作。不要一开始写前端。先让 SGDAgent 每一步都能输出标准化事件。

### 3.1 定义统一事件格式

建议你让 agent 不只是返回 final answer，而是流式返回 `AgentEvent`：

```python
from dataclasses import dataclass, asdict
from typing import Any, Literal

@dataclass
class AgentEvent:
    type: Literal[
        "status",
        "graph_build_start",
        "graph_build_update",
        "graph_ready",
        "agent_step_start",
        "observation",
        "reason_summary",
        "action",
        "tool_call",
        "tool_result",
        "graph_highlight",
        "evidence",
        "verification",
        "final_answer",
        "error",
    ]
    step: int | None
    payload: dict[str, Any]
```

注意：页面上展示的 `reason_summary` 应该是**可公开的推理摘要**，不要直接展示模型原始隐藏 chain-of-thought。你可以让模型额外输出一个简短 rationale，比如：

```json
{
  "reason_summary": "The question asks who interacts with the suitcase after the chase. I should search the graph for suitcase-related event nodes, then verify the most relevant temporal segment visually."
}
```

### 3.2 让 SGDAgent 提供两个稳定接口

你的 app 最好只依赖两个函数：

```python
bundle = prepare_video_assets(video_path_or_url, cache_dir)

for event in agent.stream_run(
    graph_bundle=bundle,
    question=question,
    options=options,
    max_steps=max_steps,
):
    yield event
```

其中：

```python
prepare_video_assets(...)
```

负责：

```text
1. 下载或读取视频
2. 抽帧 / keyframe sampling
3. caption / transcript 可选
4. frame-level scene graph extraction
5. entity alignment / entity registry
6. segment-level graph aggregation
7. global event graph construction
8. embedding / graph index 构建
9. 写入可视化所需 JSON 文件
```

`agent.stream_run(...)` 负责：

```text
1. 读取问题
2. 生成检索/验证计划
3. 图上候选搜索
4. visual verification / frame inspect
5. 更新 observation
6. 重复 reason-act
7. 输出 final answer 和 evidence chain
```

这跟 DVD 的结构一致：DVD 的 `app.py` 先 `_prepare_video_assets(...)`，再构造 `SGDCoreAgent(...)`，然后对 `agent.stream_run(question)` 逐条消费消息并更新页面。([GitHub][1])

## 4. 准备 SGDAgent 的可视化数据文件

建议每个视频处理后落盘成一个目录：

```text
outputs/
  video_id/
    raw.mp4
    frames/
      000001.jpg
      000002.jpg
    captions.json
    entity_registry.json
    graphs/
      frame_graphs.json
      segment_graphs.json
      global_graph.json
      merged_graph.json
    indexes/
      caption.index
      graph_node.index
      graph_edge.index
    runs/
      run_2026xxxx_xxxxxx/
        trajectory.jsonl
        answer.json
        evidence.json
```

### 4.1 图节点 schema

```json
{
  "id": "seg_0012_event_03",
  "label": "boy gives food to rabbits",
  "level": "segment",
  "type": "event",
  "start": 125.0,
  "end": 142.0,
  "score": 0.87,
  "entities": ["boy_01", "rabbit_group_01", "food_01"],
  "source_frames": ["000250.jpg", "000260.jpg"],
  "caption": "The boy sits beside the table and feeds the rabbits."
}
```

### 4.2 图边 schema

```json
{
  "source": "boy_01",
  "target": "rabbit_group_01",
  "relation": "feeds",
  "level": "segment",
  "start": 125.0,
  "end": 142.0,
  "confidence": 0.82
}
```

### 4.3 agent trajectory schema

每一行一个事件，便于实时写入和离线 replay：

```json
{"type":"agent_step_start","step":1,"payload":{"query":"Why are the rabbits gathering on the table?"}}
{"type":"action","step":1,"payload":{"tool":"graph_propose","args":{"query":"rabbits gathering table","levels":["global","segment"]}}}
{"type":"tool_result","step":1,"payload":{"candidates":["seg_0012_event_03","seg_0013_event_01"]}}
{"type":"graph_highlight","step":1,"payload":{"nodes":["seg_0012_event_03"],"edges":["boy_01->rabbit_group_01:feeds"]}}
{"type":"verification","step":2,"payload":{"time_range":[125.0,142.0],"result":"verified","score":0.91}}
{"type":"final_answer","step":3,"payload":{"answer":"Because a boy is feeding them.","evidence_nodes":["seg_0012_event_03"]}}
```

这个准备工作比 UI 更关键。只要这个 trajectory 标准化，页面、论文图、case study、debug 工具都能复用。

## 5. 页面布局建议

建议做成 4 个 Tab。

### Tab 1: Overview

左侧输入：

```text
- Video upload / URL
- Question
- Options, 可选
- Max steps
- Use cached graph
- Run button
```

右侧输出：

```text
- 当前状态
- 最终答案
- 证据摘要
```

DVD 的 demo 也是左侧视频输入和问题输入，右侧 analysis results 输出；它还支持 YouTube URL 或本地路径，并提供示例问题。([GitHub][1]) 你可以沿用这个布局，但右侧不要只放 Markdown，应该增加图和证据面板。

### Tab 2: Graph Construction

展示三层图：

```text
Frame-level graph:
  每个关键帧的 entity-relation snapshot

Segment-level graph:
  30s 或滑动窗口内聚合的 interactions

Global event graph:
  高层 event nodes + temporal/event edges
```

建议显示：

```text
- 建图进度条
- 已处理帧数 / segment 数
- 当前抽取到的 entity registry
- 当前图节点数、边数
- 可交互图
```

### Tab 3: Agent Observe-Reason-Act

做成 step cards：

```text
Step 1
Observation:
  global graph contains event nodes related to "rabbits", "table", "food"

Reason summary:
  Need to verify whether rabbits gather because of feeding.

Action:
  graph_propose(query="rabbits gathering table food")

Tool result:
  candidates: seg_0012_event_03, seg_0013_event_01

Graph highlight:
  高亮相关节点/边

Evidence:
  00:02:05 - 00:02:22
```

DVD 的 `app.py` 已经把 assistant message、tool call、tool result、final answer 分别追加到 `accumulated_text`，并在每次更新后 `yield`。([GitHub][1]) 你要做的是把 `accumulated_text` 扩展成多输出：日志 Markdown、graph HTML、timeline HTML、evidence gallery、final answer。

### Tab 4: Evidence & Replay

展示：

```text
- 最终答案
- evidence chain
- 被采用的 graph path
- 关键帧 gallery
- 时间轴定位
- trajectory.jsonl 下载
```

VideoMind 的 Planner→Grounder→Verifier→Answerer 工作流很适合借鉴到你的展示逻辑里：先规划，再产生候选 moment，再验证，最后回答。 对 SGDAgent 来说，可以展示成：

```text
Planner / Reasoner
  ↓
Graph Proposer
  ↓
Visual Verifier
  ↓
Answerer
```

## 6. 最小可用 `app.py` 结构

你可以从这个骨架开始：

```python
# app.py
import argparse
import json
import gradio as gr

from sgdagent.config import DemoConfig
from sgdagent.pipeline import prepare_video_assets
from sgdagent.agent import SGDAgent
from sgdagent.viz.graph import render_graph_html
from sgdagent.viz.timeline import render_timeline_html
from sgdagent.viz.trace import render_trace_markdown


def empty_ui_state():
    return {
        "status": "Idle",
        "trace": [],
        "graph_nodes": [],
        "graph_edges": [],
        "active_nodes": [],
        "active_edges": [],
        "evidence": [],
        "answer": "",
    }


def to_outputs(state):
    graph_html = render_graph_html(
        nodes=state["graph_nodes"],
        edges=state["graph_edges"],
        active_nodes=state["active_nodes"],
        active_edges=state["active_edges"],
    )

    timeline_html = render_timeline_html(
        evidence=state["evidence"],
        active_nodes=state["active_nodes"],
    )

    trace_md = render_trace_markdown(state["trace"])

    answer_md = state["answer"] or "*Final answer will appear here.*"

    return (
        state["status"],
        graph_html,
        timeline_html,
        trace_md,
        state["evidence"],
        answer_md,
    )


def apply_event(state, event):
    event_type = event.get("type")
    payload = event.get("payload", {})
    step = event.get("step")

    if event_type == "status":
        state["status"] = payload.get("message", "")

    elif event_type == "graph_ready":
        state["graph_nodes"] = payload.get("nodes", [])
        state["graph_edges"] = payload.get("edges", [])
        state["status"] = "Graph ready."

    elif event_type in {"observation", "reason_summary", "action", "tool_call", "tool_result", "verification"}:
        state["trace"].append({
            "step": step,
            "type": event_type,
            "payload": payload,
        })

    elif event_type == "graph_highlight":
        state["active_nodes"] = payload.get("nodes", [])
        state["active_edges"] = payload.get("edges", [])

    elif event_type == "evidence":
        state["evidence"].extend(payload.get("items", []))

    elif event_type == "final_answer":
        state["answer"] = payload.get("answer", "")
        state["trace"].append({
            "step": step,
            "type": "final_answer",
            "payload": payload,
        })
        state["status"] = "Done."

    elif event_type == "error":
        state["status"] = f"Error: {payload.get('message', 'Unknown error')}"

    return state


def run_demo(video, question, options, max_steps, use_cache):
    state = empty_ui_state()

    if not video or not question:
        state["status"] = "Please provide both a video and a question."
        yield to_outputs(state)
        return

    try:
        state["status"] = "Preparing video assets..."
        yield to_outputs(state)

        bundle = prepare_video_assets(
            video_path_or_url=video,
            use_cache=use_cache,
        )

        # 先展示完整图或 global graph
        state["graph_nodes"] = bundle.to_visual_nodes()
        state["graph_edges"] = bundle.to_visual_edges()
        state["status"] = "Graph constructed. Starting agent..."
        yield to_outputs(state)

        agent = SGDAgent(
            graph_bundle=bundle,
            max_steps=int(max_steps),
        )

        for event in agent.stream_run(
            question=question,
            options=options,
        ):
            state = apply_event(state, event)
            yield to_outputs(state)

    except Exception as e:
        state["status"] = f"Error: {e}"
        yield to_outputs(state)


def launch(args):
    with gr.Blocks(title="SGDAgent Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# SGDAgent: Scene Graph-Driven Video Agent")
        gr.Markdown(
            "Visualize graph construction, graph-grounded observe-reason-act steps, "
            "visual verification, and final answer."
        )

        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Video", format="mp4")
                question = gr.Textbox(label="Question", lines=3)
                options = gr.Textbox(
                    label="Options, optional",
                    lines=5,
                    placeholder="A. ...\nB. ...\nC. ...\nD. ..."
                )
                max_steps = gr.Slider(1, 20, value=15, step=1, label="Max agent steps")
                use_cache = gr.Checkbox(value=True, label="Use cached graph if available")
                run_btn = gr.Button("Run SGDAgent", variant="primary")

            with gr.Column(scale=2):
                status = gr.Markdown(label="Status")
                answer = gr.Markdown(label="Final Answer")

        with gr.Tabs():
            with gr.Tab("Graph"):
                graph_html = gr.HTML(label="Scene Graph")

            with gr.Tab("Timeline"):
                timeline_html = gr.HTML(label="Evidence Timeline")

            with gr.Tab("Agent Trace"):
                trace_md = gr.Markdown(label="Observe-Reason-Act Trace")

            with gr.Tab("Evidence"):
                evidence_json = gr.JSON(label="Evidence Items")

        run_btn.click(
            fn=run_demo,
            inputs=[video, question, options, max_steps, use_cache],
            outputs=[status, graph_html, timeline_html, trace_md, evidence_json, answer],
        )

    demo.launch(share=args.share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    launch(args)
```

这个骨架的思想和 DVD `app.py` 一致：后端函数是 generator，页面不断接收更新。DVD 里也有 `click_kwargs = dict(fn=solve, inputs=[video_url, question], outputs=answer_box)`，并在可用时开启 stream 参数。([GitHub][1]) 你的版本只是把单一 `answer_box` 拆成多个输出面板。

## 7. 图可视化渲染函数

你可以先用 PyVis 生成 HTML，然后塞进 `gr.HTML`。伪代码如下：

```python
# sgdagent/viz/graph.py
import tempfile
from pyvis.network import Network


def render_graph_html(nodes, edges, active_nodes=None, active_edges=None):
    active_nodes = set(active_nodes or [])
    active_edges = set(active_edges or [])

    net = Network(
        height="650px",
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )

    for node in nodes:
        node_id = node["id"]
        level = node.get("level", "unknown")
        title = (
            f"ID: {node_id}<br>"
            f"Level: {level}<br>"
            f"Type: {node.get('type', '')}<br>"
            f"Time: {node.get('start', '')} - {node.get('end', '')}<br>"
            f"Caption: {node.get('caption', '')}"
        )

        net.add_node(
            node_id,
            label=node.get("label", node_id)[:40],
            title=title,
            size=25 if node_id in active_nodes else 12,
        )

    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        relation = edge.get("relation", "")
        edge_id = f"{src}->{tgt}:{relation}"

        net.add_edge(
            src,
            tgt,
            label=relation,
            title=json_safe(edge),
            width=4 if edge_id in active_edges else 1,
        )

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        path = f.name

    net.write_html(path)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def json_safe(obj):
    import json
    return "<pre>" + json.dumps(obj, ensure_ascii=False, indent=2) + "</pre>"
```

第一版不用做复杂交互。只要每个 agent step 能高亮相关节点/边，展示效果就会明显优于纯日志。

## 8. 时间轴渲染

建议每个 evidence item 长这样：

```json
{
  "id": "ev_0001",
  "step": 2,
  "start": 125.0,
  "end": 142.0,
  "source": "frame_inspect",
  "nodes": ["seg_0012_event_03"],
  "summary": "The boy feeds the rabbits at the table.",
  "frames": ["outputs/demo/frames/000250.jpg", "outputs/demo/frames/000260.jpg"],
  "verified": true,
  "score": 0.91
}
```

页面上显示：

```text
00:02:05 ━━━━━━━━━ 00:02:22
Step 2 | frame_inspect | verified | score 0.91
Evidence: The boy feeds the rabbits at the table.
```

这个设计能把 VideoMind 的“localize → verify → answer”可解释性迁移到 SGDAgent。VideoMind 强调复杂视频推理需要分解任务、定位片段、验证片段、再合成答案。

## 9. Agent 工具也要改成适合展示的形式

SGDAgent 建议至少暴露这些工具：

```text
global_graph_browse(query)
  返回 global event graph 中相关事件节点、全局实体、长程关系

segment_graph_search(query, entities=None, top_k=5)
  返回相关 segment-level event nodes

frame_graph_search(query, segment_id=None, top_k=8)
  返回 frame-level graph snapshots

graph_path_search(source_entity, target_entity, relation_hint=None)
  返回实体间路径或事件链

frame_inspect(time_range, question)
  对候选时间段做视觉验证

answer(answer, evidence_nodes, evidence_ranges)
  停止并输出答案
```

对应的事件输出：

```python
yield {
    "type": "tool_call",
    "step": step,
    "payload": {
        "tool": "segment_graph_search",
        "args": {"query": "boy feeding rabbits", "top_k": 5}
    }
}

yield {
    "type": "tool_result",
    "step": step,
    "payload": {
        "candidates": [
            {
                "node_id": "seg_0012_event_03",
                "score": 0.87,
                "time_range": [125.0, 142.0],
                "summary": "boy feeds rabbits at table"
            }
        ]
    }
}

yield {
    "type": "graph_highlight",
    "step": step,
    "payload": {
        "nodes": ["seg_0012_event_03", "boy_01", "rabbit_group_01"],
        "edges": ["boy_01->rabbit_group_01:feeds"]
    }
}
```

DVD 的经验说明，工具整合很重要：去掉 Global Browse、Frame Inspect、Clip Search 都会导致性能下降，尤其 Clip Search 与 Frame Inspect 对定位和细粒度验证作用明显。 SGDAgent 的等价物就是：图浏览、图检索、视觉验证都要能在页面里明确展示。

## 10. 最小开发路线

### Phase 1：离线 replay 页面

先不要做实时 streaming。先跑一次 SGDAgent，生成：

```text
graph_bundle.json
trajectory.jsonl
answer.json
evidence.json
```

然后页面只负责读取这些文件并展示。这样最快能得到可展示版本。

命令：

```bash
python -m sgdagent.run \
  --video examples/demo.mp4 \
  --question "Why are the rabbits gathering on the table?" \
  --options examples/options.txt \
  --dump outputs/demo_run/
```

页面：

```bash
python app_replay.py --run_dir outputs/demo_run/
```

### Phase 2：实时 streaming demo

把 `agent.run(...)` 改成 `agent.stream_run(...)`，每一步 yield event，然后 Gradio 实时刷新。

### Phase 3：交互式图调试

增加：

```text
- 点击节点显示原始 caption / scene graph / frame
- 点击边显示 relation 来源
- 点击 evidence 跳到视频时间段
- 支持过滤 global / segment / frame graph
- 支持显示当前 step 的 active subgraph
```

### Phase 4：论文展示版

增加：

```text
- 预置 demo videos
- 预置 questions
- 轨迹回放速度控制
- export case study
- download trajectory
```

## 11. 准备工作清单

你需要准备以下内容，按优先级排序。

### 必须准备

```text
1. 一个稳定的 SGDAgent CLI
   输入 video + question + options，输出 answer。

2. 一个 graph bundle 标准格式
   至少包含 nodes、edges、level、time_range、source_frames。

3. 一个 trajectory.jsonl 标准格式
   记录每一步 observation、reason_summary、action、tool_result、verification、answer。

4. 一个 stream_run 接口
   agent 每一步 yield AgentEvent。

5. 一个 evidence schema
   最终答案必须能追溯到 graph nodes、time ranges、frames。

6. 缓存机制
   长视频建图很慢，demo 必须优先支持 use_cache=True。

7. Demo 视频和问题
   选择 2-3 个能明显体现“图检索优于 caption 检索”的样例。
```

### 强烈建议准备

```text
1. 失败案例 replay
   展示 Clip Search Trap / Frame Inspect Trap 类似的 agent 行为分析。

2. 图层过滤器
   global / segment / frame 三层分开显示。

3. 证据对齐
   每个答案句子对应 evidence node 和时间段。

4. 参数面板
   max_steps、top_k_event、top_k_segment、top_k_frame、verification_budget。

5. Debug JSON tab
   方便你检查 event schema 和后端输出。
```

DVD 的 agent 行为分析里专门区分了 Global Browse Only、Simple Action、Iterative Search、Frame Inspect Trap、Clip Search Trap 等行为类型。 你也可以为 SGDAgent 设计类似的 trajectory classification，用于展示“为什么图结构减少无效搜索”。

## 12. 推荐项目结构

```text
SGDAgent/
  sgdagent/
    __init__.py
    config.py

    pipeline.py
      prepare_video_assets()

    graph/
      builder.py
      schema.py
      serialize.py
      retrieval.py

    agent/
      core.py
      tools.py
      prompts.py
      events.py

    viz/
      graph.py
      timeline.py
      trace.py
      evidence.py

  app.py
  app_replay.py
  examples/
    demo.mp4
    demo_questions.json

  outputs/
    ...
```

核心原则：**后端不应该知道 Gradio，Gradio 也不应该知道模型细节**。中间只通过 `AgentEvent` 和 `GraphBundle` 通信。

## 13. 你可以直接从 DVD 的 app.py 借鉴的部分

可以借鉴：

```text
1. Gradio Blocks 布局
2. 输入区 + 输出区
3. prepare assets 函数
4. solve 函数使用 generator
5. 每收到 agent 消息就 yield 更新
6. tool_call / tool_result / final_answer 的分段展示
7. --share CLI 参数
```

DVD 的 `app.py` 中 `_prepare_video_assets` 会处理下载/抽帧/caption/lite mode 资产准备；`solve` 会流式输出 “Processing video...”、“Initializing DVD agent...”，然后循环读取 `agent.stream_run(question)` 的消息。([GitHub][1]) UI 部分则用 `gr.Blocks`、`gr.Row`、`gr.Column`、输入框、按钮、示例问题、Markdown answer box 组成。([GitHub][1])

不建议照搬：

```text
1. 只用 Markdown 展示全部过程
2. 只截断 tool_result 文本
3. 不展示 evidence grounding
4. 不展示图节点/边/时间区间
5. 不区分建图阶段和推理阶段
```

SGDAgent 的核心卖点是 scene graph-driven reasoning。如果页面只像 DVD 一样显示文本日志，会削弱你的方法特色。

## 14. 最终目标效果

理想运行过程应该是这样：

```text
用户上传视频 + 问题
↓
页面显示：Extracting keyframes...
↓
页面显示：Building frame-level scene graphs...
↓
页面显示：Aligning entities...
↓
页面显示：Aggregating segment-level graphs...
↓
页面显示：Constructing global event graph...
↓
图面板出现完整 hierarchical scene graph
↓
Agent Step 1:
  Observation: global graph contains candidate events
  Action: global_graph_browse(...)
  页面高亮 global event nodes
↓
Agent Step 2:
  Action: segment_graph_search(...)
  页面高亮 segment nodes + timeline ranges
↓
Agent Step 3:
  Action: frame_inspect(...)
  页面展示关键帧和视觉验证结果
↓
Final:
  答案 + evidence chain + highlighted graph path + 时间轴
```

这会比 DVD demo 更能体现 SGDAgent 的研究贡献：不是只展示“Agent 调了什么工具”，而是展示“Agent 如何在图结构上定位、验证并合成答案”。

[1]: https://github.com/microsoft/DeepVideoDiscovery/blob/main/app.py "DeepVideoDiscovery/app.py at main · microsoft/DeepVideoDiscovery · GitHub"
[2]: https://www.gradio.app/docs/gradio/blocks?utm_source=chatgpt.com "Blocks"
[3]: https://www.gradio.app/guides/streaming-outputs?utm_source=chatgpt.com "Streaming Outputs"
