# SGDAgent Development Plan

> 本仓库是 [DeepVideoDiscovery](https://github.com/microsoft/DeepVideoDiscovery)（DVD）提交版的迁移版本。提交时移除了大部分监控点、调试入口与统计接口；本文件作为后续所有新功能开发与测试的总指引。完成的项请直接 `[x]` 勾选。

---

## 0. 迁移完整性 & 恢复"最简洁开发版"

**与 DVD 对比的实际差异**（仅列代码层面，已忽略 `__pycache__`、`.DS_Store`、命名 `dvd→sgd`）：

| 项 | DVD 中位置 | SGDAgent 中状态 | 备注 |
|---|---|---|---|
| `trace_logger.py` | `dvd/trace_logger.py`（147 行） | **完全删除** | sgd_core 中所有 callsite (`trace_logger.log_*`) 仍保留 ~30 处，仅靠 `_build_trace_logger() → None` 短路 |
| `TRACE_*` 配置 | `config.py` 末尾 5 行 (`TRACE_ENABLED / TRACE_LOG_DIR / TRACE_MAX_FIELD_CHARS`) | **删除** | 同时删了 `from sgd.path_config import logs_root` 顶部 import |
| `_build_trace_logger` 函数体 | `sgd_core.py:303-314` | **掏空，return None** | uuid import 还在 sgd_core.py:9 |
| `LITE_MODE` 开关 | `config.py` + `sgd_core.py:142-143` 的 `tools.remove(frame_inspect_tool)` | **删除** | 提交版强制完整模式 |
| `process_video_lite` | `frame_caption.py:332-367` | **删除** | 仅字幕、无像素的轻量路径 |
| `_is_youtube_url` + `yt_dlp` 依赖 | `video_utils.py` 顶部 | **删除** | 提交版仅支持本地视频 |
| `if __name__ == "__main__":` 调试入口 | `utils.py` / `build_database.py` / `sgd_core.py` 末尾 | **删除** | 含写死路径的旧测试 main、`pdb.set_trace()` |
| `evaluation/` | `evaluate_results.py / evaluate_results_enhanced.py / evaluate_stage1_graph.py` | **整目录移除** | 评测脚本 |
| `scripts/` | `analyze_regression_failure_modes.py / build_run_summary.py / classify_improvement_causes.py / compare_badcase_runs.py / render_trace_report.py / trace_index.py / smoke_test_embedding.py / monitor_progress.sh / resume_benchmark.sh / run_baseline.sh / run_graph_stage12_smoke.sh / run_stage12_parallel_build.sh` | **整目录移除** | 监控、bad-case 分析、trace 渲染 |
| `reproduce/` | `decode_frames.py / download_lvbench.sh / prepare_database.py / run_benchmark.py / transcribe.py` | **整目录移除** | LVBench 复现 |
| `tests/` | `test_global_event_index.py` | **整目录移除** | |
| `app.py` | DVD Gradio demo 主入口 | **整文件移除** | 见 `app-plan.md` |
| `mcp_server.py` | MCP 接口 | **整文件移除** | |
| `docs/paper/` | 论文与图表 | **整目录移除** | |

### 0.1 恢复开发版最小动作（从最低成本开始）

- [x] **a. 恢复 trace 子系统**（最优先；callsite 已经在 `sgd_core.py` 全部就位，恢复 1 个文件 + 5 行 config 即可解锁完整执行轨迹日志）
  - [x] 新建 `sgd/trace_logger.py`：从 DVD 直接迁移，无需改动逻辑
  - [x] `sgd/config.py` 末尾追加 `TRACE_ENABLED / TRACE_LOG_DIR / TRACE_MAX_FIELD_CHARS`，环境变量前缀使用 `SGD_TRACE_*`
  - [x] `sgd/path_config.py` 已经有 `logs_root()`，确保 import 链可用
  - [x] `sgd/sgd_core.py` 顶部恢复 `from sgd.trace_logger import TraceLogger`，恢复 `_build_trace_logger` 函数体（DVD `sgd_core.py:303-314`）
  - [ ] 跑一次 `local_run.py`，确认 `data/logs/traces/trace_*.jsonl` 落盘（待你在 venv 环境验证）
- [x] **b. 恢复 `tests/test_global_event_index.py`** 作为最小回归基线（29/29 passed in venv）
- [ ] **c. 恢复 `reproduce/`** 的 LVBench 数据准备脚本（实验前置依赖，不再恢复就重写）
- [x] **d. 恢复/重写 `evaluation/`** —— 已建包：`eval_agent.py`（QA 正确率，从 `evaluate_results.py` 迁移）+ `eval_graph.py`（stage1+stage2 图质量与检索，从 `evaluate_stage1_graph.py` 迁移）+ `eval_stage3_graph.py`（stub，§1 落地后填充）+ `run_graph_eval.py`（统一入口，子命令 `agent / graph / stage3 / all`）。CLI smoke 通过
- [ ] **e. 恢复 `scripts/render_trace_report.py` + `trace_index.py`** —— 调试与论文 case study 都依赖
- [ ] **f.（可选）恢复 `LITE_MODE` 与 `process_video_lite`** —— 长视频快速冒烟时有用，没需求可不恢复
- [ ] **g.（可选）恢复 `mcp_server.py`** —— 暂不在路线图内，仅当需要外部接入再做

> 注：`app.py`、`evaluation/`、`docs/` 不直接照搬，会按 §2、§3 的目标重新设计；DVD 版本只作参考。

---

## 1. 目标一 · Stage3 全局图：构建真正的三层图三元组表征

当前实现仅有 stage1（frame-level scene graph）+ stage2（segment-level aggregated graph + global event index 压缩）。`global_event_index.py` 实际只是把 segment 摘要拼接进 prompt，**没有独立的 global graph 数据结构**。Stage3 的目标是把"全局事件链 + 跨片段实体绑定"变成真正可查询的图层。

### 1.1 设计

- [ ] **三层图 schema 统一**（`sgd/scene_graph/schema.py`）
  - [ ] Frame-level: `FrameSceneGraph { frame_id, timestamp, objects:[VisualObject], relations:[Triplet] }`
  - [ ] Segment-level: `SegmentSceneGraph { segment_id, [start,end], entities, events:[EventNode], relations }`
  - [ ] Global-level: `GlobalSceneGraph { entity_registry, event_chain:[EventNode], cross_segment_relations }`
  - [ ] 跨层引用约定：每个上层节点持有下层节点的 id 列表（`source_segments`、`source_frames`）
- [ ] **视觉目标集合（visual object set）独立成第四类资产**
  - [ ] 在 frame 层抽取并持久化 `objects.json`（可考虑接入 detector / open-vocab 模型）
  - [ ] 与 frame graph 中的 entity 对齐到 `entity_registry`

### 1.2 实现

- [ ] **`sgd/scene_graph/stage3_builder.py`**（新建）
  - [ ] 输入：所有 segment graph + segment 摘要 + entity_registry
  - [ ] 输出：单个 `global_graph.json`，含 event chain（事件因果/时序边）+ 跨片段实体合并结果
  - [ ] 函数：`ensure_stage3_global_graph(video_db_path) -> GlobalSceneGraph`
- [ ] **重构 `global_event_index.py`** —— 从 stage3 graph 物化得到压缩 prompt，而不是独立计算
- [ ] **检索接口 `sgd/scene_graph/query_global.py`**（新建）
  - [ ] `query_global_event(query, top_k)`：在 global event 节点上做语义检索
  - [ ] `query_global_path(entity_a, entity_b)`：跨片段实体关系路径搜索
- [ ] **Agent 工具暴露**（`sgd/build_database.py` + `sgd/func_call_shema.py`）
  - [ ] 新增 `global_graph_browse` / `global_path_search` 两个 tool
  - [ ] 调整 `sgd_core.py` 工具列表与 system prompt

### 1.3 验收

- [ ] 在已建库视频上跑 stage3，生成 `global_graph.json` 与可视化导出
- [ ] Agent 在 multiple-choice 集合上启用全局工具后正确率不降，全局类问题正确率上升

---

## 2. 目标二 · 三层图测评 pipeline + 图驱动 Agent + Agent 能力测评

### 2.1 三层图本身的测评

> §0.1.d 已落地：`evaluation/eval_graph.py` 同时覆盖 stage1+stage2（DVD 原文件就是合一的，强行拆分会让 ~150 行共享工具重复，留作未来文件膨胀时的拆分点）；`evaluation/eval_stage3_graph.py` 为 stub。下面是要补的功能。

- [x] **`evaluation/eval_graph.py`** 已迁移（含 stage1 三元组质量 + stage1 检索 + stage2 segment 质量 + stage2 检索）
- [ ] 在 `eval_graph.py` 中补 frame-level 三元组准确率：基于人工标注子集（当前是 `time_reference` 距离指标，缺真值三元组）
- [ ] 在 `eval_graph.py` 中补 segment 事件边界 IoU 指标（当前只有距离）
- [ ] **`evaluation/eval_stage3_graph.py`**（filling stub，依赖 §1）：
  - [ ] event chain 拓扑正确率（时序、因果）
  - [ ] cross-segment entity binding 正确率
  - [ ] global retrieval recall@k
- [x] **`evaluation/run_graph_eval.py`** 统一入口已建（子命令 `agent / graph / stage3 / all`）
- [ ] 让 `run_graph_eval.py all` 产出 markdown 汇总报告（目前只是顺序调用）

### 2.2 图驱动 Agent 优化

- [ ] **Agent 行为分类器** `scripts/classify_trajectory.py`
  - [ ] 把每条 trace 标成：`Global Browse Only / Iterative Search / Frame Inspect Trap / Clip Search Trap` 等（沿用 DVD 思路）
  - [ ] 依赖 §0.1.a 的 trace 系统
- [ ] **system prompt / tool 描述迭代**：基于 trajectory 分类找出失败模式，针对性修改 `sgd_core.py` 中的 prompt
- [ ] **图层选择策略**：明确什么问题走哪一层（局部细节→frame、片段事件→segment、全局因果→global）
- [ ] **缓存 / 资源预算**：`max_iterations` 与每层 `top_k` 的合理默认（写入 `config.py`）

### 2.3 Agent 能力评测 pipeline

- [ ] **`evaluation/eval_agent.py`**：
  - [ ] 基础正确率（LVBench 全集 / 子集）
  - [ ] 按问题类型分桶（temporal / causal / counting / detail / global）
  - [ ] Trace 级指标：平均步数、tool 使用分布、verification 命中率
- [ ] **回归对比 `scripts/compare_runs.py`**（沿用 DVD `compare_badcase_runs.py` 思路重写）
  - [ ] 找出版本 A→B 的 win/lose case，附 trace diff
- [ ] **数据准备 `reproduce/`**（恢复或重写）
  - [ ] LVBench 下载 / 抽帧 / 字幕的脚本化
  - [ ] 一键 `prepare_database.py`

---

## 3. 目标三 · 可视化工具

### 3.1 工具 1 · 三层场景图可视化（建图侧）

定位：研究/调试/绘图用，输入是已生成的 `*.json`，**离线**渲染，不与 Agent 推理耦合。

- [ ] **`viz/graph_viewer.py`**（新模块；建议放 `sgd/viz/` 或顶层 `viz/`）
  - [ ] 读取 `frame_graphs.json / segment_graphs.json / global_graph.json`
  - [ ] 提供 layer filter（global / segment / frame）
  - [ ] 节点点击 → 弹出 caption / 关键帧 / 原始三元组
  - [ ] 边点击 → 来源 frame / segment 引用
  - [ ] 时间轴联动（hover segment 高亮对应 global event）
- [ ] **导出**：单文件 HTML（PyVis）+ 静态 PNG（用于论文图）
- [ ] **CLI**：`python -m sgd.viz.graph_viewer <video_db_path>`

技术栈建议：PyVis（快速）/ Cytoscape.js（可控）+ 简单 Flask/Gradio 壳。

### 3.2 工具 2 · SGDAgent 推理过程可视化（Agent 侧）

定位：Demo + 调试。输入是 `local_run.py` 一次执行，**实时流式**展示 observe-reason-act。

参考：[DVD `app.py`](https://github.com/microsoft/DeepVideoDiscovery/blob/main/app.py)。详细设计已沉淀在 `app-plan.md`，本节只列里程碑。

- [ ] **Phase 1 · 离线 replay 版**（先做这个，最快得到可演示物）
  - [ ] `sgd_core.py` 落盘标准化 `trajectory.jsonl`（依赖 §0.1.a 已恢复的 trace 系统；可直接复用其 schema 或在其上加 `agent_step_start / observation / reason_summary / graph_highlight / evidence` 等事件类型）
  - [ ] `app_replay.py` 读取 trace 与 graph bundle 渲染四个 Tab：Overview / Graph / Trace / Evidence
- [ ] **Phase 2 · 实时 streaming**
  - [ ] 改 `agent.run` 为 `agent.stream_run`，generator 模式
  - [ ] Gradio Blocks 多输出面板逐步刷新
- [ ] **Phase 3 · 交互调试**
  - [ ] 节点/边点击联动证据 / 视频时间段
  - [ ] 当前 step 的 active subgraph 高亮
  - [ ] 三层 filter
- [ ] **Phase 4 · 论文展示版**
  - [ ] 预置 demo videos + questions
  - [ ] 轨迹回放速度控制 + case study 导出

### 3.3 公共基础

- [ ] **统一事件 schema** `sgd/agent/events.py`（新建 `AgentEvent` dataclass，§0.1.a 的 trace 文件以及 §3.2 的实时流共用同一份）
- [ ] **公共渲染层** `sgd/viz/graph_render.py`（PyVis 封装，工具 1 / 2 共用）

---

## 4. 工作顺序建议

> 严格依赖关系：§0.1.a → §1 → §2 → §3.2。可并行：§3.1 与 §1 并行；§3.2 Phase 1 与 §2.1 并行。

1. **本周**：§0.1.a 恢复 trace 子系统 + §0.1.b 测试基线
2. **接下来 1–2 周**：§1.1 schema 设计 → §1.2 stage3_builder
3. **接下来 1–2 周**：§2.1 三层图测评 + §3.1 工具 1 离线版
4. **之后**：§2.2 / §2.3 Agent 能力优化与评测，§3.2 工具 2 分阶段推进

---

## 5. 仓库目标结构（含未来增量）

```
SGDAgent/
  local_run.py
  app_replay.py            # 新增 §3.2 P1
  app.py                   # 新增 §3.2 P2
  sgd/
    sgd_core.py
    build_database.py
    config.py
    frame_caption.py
    func_call_shema.py
    path_config.py
    trace_logger.py        # 恢复 §0.1.a
    utils.py
    video_utils.py
    scene_graph/
      schema.py
      stage1_builder.py
      segment_builder.py
      stage3_builder.py    # 新增 §1.2
      global_event_index.py
      query.py
      query_segment.py
      query_global.py      # 新增 §1.2
    agent/
      events.py            # 新增 §3.3
    viz/
      graph_render.py      # 新增 §3.3
      graph_viewer.py      # 新增 §3.1
      trace_render.py      # 新增 §3.2
  evaluation/
    __init__.py
    eval_agent.py            # 已迁移 §0.1.d
    eval_graph.py            # 已迁移 §0.1.d（stage1+stage2 合一）
    eval_stage3_graph.py     # stub，§1 落地后填充
    run_graph_eval.py        # 已迁移 §0.1.d
  reproduce/
    decode_frames.py
    prepare_database.py
    run_benchmark.py
  scripts/
    render_trace_report.py
    classify_trajectory.py
    compare_runs.py
  tests/
    test_global_event_index.py
    test_stage3_builder.py
  data/                    # 数据/缓存（gitignored）
  dist/                    # 提交版打包产物
  app-plan.md              # 工具 2 的详细设计稿
  dev-plan.md              # 本文件
```
