下面这份计划的核心不是“怎么把分数跑高”，而是把实验组织成一个明确的证明链：

> **SGDAgent 的 inference-time memory 完全由三层 Scene Graph 承担；captions 只允许作为离线建图脚手架，不能作为 Agent 检索、观察、回退或最终证据。**

DVD 的数据库设计明确包含 global subjects、clip captions、indexed frames，并且 Clip Search 依赖 caption embedding 返回 top-k captions；这正是你现在要替换的部分。 VideoAgent 也依赖 CLIP 检索帧后再用 VLM caption 更新 state；这也不是新 SGDAgent 要沿用的推理路线。 VideoMind 的 Planner–Grounder–Verifier–Answerer 思路可以借鉴，但你的关键差异是：候选 moment 不来自 learned grounder 或 caption query，而来自 graph-native query。

# SGDAgent 下一阶段详细实验计划

## 0. 总实验目标

本阶段实验应围绕一个主命题展开：

> **Can a hierarchical scene-graph database replace caption-based video memory for long-video agent reasoning?**

对应中文表述：

> **三层 Scene Graph 能否作为长视频 Agent 的主数据库和主检索接口，而不是作为 caption memory 的辅助索引？**

实验要证明四件事：

1. **可运行性**：删除 captions 与 caption embeddings 后，Agent 仍能完成 observe–reason–act。
2. **可检索性**：问题可以被转化为 entity / triple / event / temporal / path 查询，而不是 caption 相似度搜索。
3. **可验证性**：最终答案能追溯到 graph fact、event、path、support frame / track。
4. **有效性**：graph-only inference 在核心问题类型上优于 caption-only baseline，或者至少在 UNKNOWN、证据可追溯性、长程时序题上显著优于 DVD 式 caption search。

---

# 1. 实验红线与系统审计

这一部分要先做，不然后面所有实验都可能被 reviewer 质疑成“隐形 caption search”。

## 1.1 Caption-free inference hard audit

建立一个自动审计脚本，每次运行前后检查：

| 检查项                                                                              | 目标        |
| -------------------------------------------------------------------------------- | --------- |
| Inference DB 中是否存在 `caption`, `summary`, `description`, `narration`, `script` 字段 | 必须不存在或不可读 |
| Tool observation 中是否出现自然语言 clip/frame narration                                  | 必须为 0     |
| Agent action space 中是否存在 `clip_search`, `global_browse`, `caption_search`        | 必须为 0     |
| 是否调用 caption embedding index                                                     | 必须为 0     |
| 最终 answer evidence 是否包含 graph fact/event/path/support frame                      | 必须为 100%  |
| failure 是否 fallback 到 captions                                                   | 必须为 0     |

建议把这个实验命名为：

> **E0: Caption-Free Inference Sanity Check**

输出一个表：

| Setting       | Captions visible? | Caption embeddings? | Caption fallback? | Graph evidence trace? | Pass |
| ------------- | ----------------: | ------------------: | ----------------: | --------------------: | ---: |
| DVD           |               Yes |                 Yes |               Yes |               Partial |   No |
| Old GraphDVD  |           Partial |                 Yes |     Yes / Partial |               Partial |   No |
| SGDAgent-Ours |                No |                  No |                No |                   Yes |  Yes |

这张表非常重要。它直接定义你的工作边界。

---

# 2. 数据集与实验划分

## 2.1 主数据集

主实验仍建议放在 **LVBench**，原因是你已有 DVD 复现、GraphDVD 实验、badcase 分析和路径依赖。DVD 也把 LVBench 作为主要 benchmark，并报告了 1,549 道题、103 个 hour-long videos。

建议划分为三层：

| Split | 用途                    | 规模                               |
| ----- | --------------------- | -------------------------------- |
| Smoke | 快速验证系统可运行             | 2–3 videos，约 30–50 questions     |
| Dev   | 调图数据库、调 tool、调 prompt | 10–15 videos，约 150–250 questions |
| Full  | 最终主结果                 | LVBench full 1,549 questions     |

其中 Smoke 可以直接沿用你之前的 2-video / bad38 案例，因为里面有很多已知失败类型，适合测试 graph-only 是否真的能替代 caption search。

## 2.2 诊断子集

除了 full benchmark，要额外构建一个 **SGD-Diagnostic subset**。这个子集不是为了刷总分，而是为了证明三层图的价值。

建议从 LVBench 中人工筛 200–300 题，按问题类型标注：

| Question type      | 说明                                                      | 主要依赖图层           |
| ------------------ | ------------------------------------------------------- | ---------------- |
| Entity attribute   | “What color / which object / who is …”                  | Frame SG         |
| Spatial relation   | “next to / behind / holding / looking at …”             | Frame SG         |
| Event occurrence   | “What happens when …”                                   | Segment SG       |
| Event participant  | “Who gives / receives / opens …”                        | Segment SG       |
| Temporal order     | “before / after / first / last …”                       | Global SG        |
| State change       | “who finally has / what changes / where does it end up” | Global SG        |
| Multi-hop relation | “person → object → event → result”                      | Graph path       |
| Negative / absence | “which did not happen / what is not shown”              | Graph + verifier |

这个子集要做更细的 evidence annotation，用来评估 graph retrieval 与 evidence faithfulness。

---

# 3. 系统版本与 baseline matrix

主实验不能只比较 “DVD vs SGDAgent”。必须构造一组能隔离变量的系统版本。

## 3.1 Caption-based baselines

| Method                    | 作用                                          |
| ------------------------- | ------------------------------------------- |
| DVD reproduction          | caption-centric long-video agent baseline   |
| DVD w/o Global Browse     | 测试 clip caption search 的独立贡献                |
| DVD w/o Clip Search       | 证明 DVD 失去 caption search 后性能会明显下降           |
| Old GraphDVD / Caption+SG | 证明你旧方案是 graph-enhanced caption memory，不是新方案 |

DVD 的 ablation 已显示 Clip Search 在其体系里是关键工具，移除 Clip Search 会造成最大性能下降；这也说明你的新工作要替换的是 DVD 的核心检索机制，而不是一个边缘模块。

## 3.2 Graph-only SGDAgent variants

| Method                           | Frame SG | Segment SG | Global SG | Verifier | Captions at inference |
| -------------------------------- | -------: | ---------: | --------: | -------: | --------------------: |
| SGDAgent-F                       |        ✓ |          ✗ |         ✗ |        ✓ |                     ✗ |
| SGDAgent-FS                      |        ✓ |          ✓ |         ✗ |        ✓ |                     ✗ |
| SGDAgent-FSG                     |        ✓ |          ✓ |         ✓ |        ✓ |                     ✗ |
| SGDAgent-FSG w/o verifier        |        ✓ |          ✓ |         ✓ |        ✗ |                     ✗ |
| SGDAgent-FSG w/o path query      |        ✓ |          ✓ |         ✓ |        ✓ |                     ✗ |
| SGDAgent-FSG w/o temporal query  |        ✓ |          ✓ |         ✓ |        ✓ |                     ✗ |
| SGDAgent-FSG w/o entity resolver |        ✓ |          ✓ |         ✓ |        ✓ |                     ✗ |

主方法应命名为：

> **Caption-free SGDAgent-FSG**

其中 FSG 表示：

```text
Frame-level Visual Fact Graph
Segment-level Event Graph
Global-level Temporal Graph
```

## 3.3 Caption-assisted construction ablations

因为你允许 captions 作为离线建图脚手架，所以必须单独做 construction-time caption ablation：

| Builder setting           | Captions used during construction? | Inference captions? | 目的               |
| ------------------------- | ---------------------------------: | ------------------: | ---------------- |
| Visual-only builder       |                                 No |                  No | 测试完全视觉建图下限       |
| Caption-assisted builder  |                                Yes |                  No | 主方法              |
| Caption-rich builder      |  Yes + transcript / dense captions |                  No | 测试更强 teacher 的上限 |
| Caption-visible inference |                                Yes |                 Yes | 只作为违规上限，不作为主方法   |

核心表述是：

> Captions improve graph construction quality, but they are never exposed as inference-time memory.

---

# 4. 三层图数据库实验

这一组实验不是问答准确率，而是图数据库本身是否合格。你现在的目标是“图作为数据库”，所以必须有 intrinsic graph quality evaluation。

## 4.1 Frame-level SG 质量评估

### 目标

评估 frame facts 是否能作为视觉事实数据库。

### 样本

从 dev split 中抽：

```text
20 videos × 每个视频 20 frames = 400 frames
```

对每帧采样 top facts：

```text
entities
attributes
relations
bbox / crop
confidence
```

### 指标

| Metric                      | 定义                                          |
| --------------------------- | ------------------------------------------- |
| Entity precision            | 预测 entity 是否真实可见                            |
| Attribute precision         | color / clothing / object type 是否正确         |
| Relation precision          | holds / looks_at / next_to / enters 等关系是否正确 |
| BBox support validity       | support bbox 是否覆盖对应实体                       |
| Fact confidence calibration | 高置信 fact 的正确率是否更高                           |
| Duplicate entity rate       | 同一帧内重复实体 ID 比例                              |
| Missing key object rate     | 问题相关关键实体未入图比例                               |

### 结果表

| Builder          | Entity P | Attr P | Relation P | BBox Valid | Missing Key Obj ↓ |
| ---------------- | -------: | -----: | ---------: | ---------: | ----------------: |
| Visual-only      |          |        |            |            |                   |
| Caption-assisted |          |        |            |            |                   |
| Caption-rich     |          |        |            |            |                   |

你要证明 caption-assisted builder 不是把 caption 存进库，而是提高结构化视觉事实质量。

---

## 4.2 Segment-level SG 质量评估

### 目标

评估 frame facts 聚合成 event 是否可靠。

### 样本

从 dev split 中选择 100–150 个问题相关事件，人工或 VLM-assisted 标注：

```text
event_type
participants
start_time
end_time
support_frames
```

### 指标

| Metric                  | 定义                                  |
| ----------------------- | ----------------------------------- |
| Event type precision    | give / chase / open / break 等谓词是否正确 |
| Participant accuracy    | subject / object / recipient 是否正确   |
| Boundary IoU            | 预测 event span 与 gold span 的 IoU     |
| Support fact coverage   | event 是否由足够 frame facts 支撑          |
| Over-fragmentation rate | 同一事件被切成多个 event                     |
| Over-merge rate         | 多个事件被合并成一个 event                    |

### 结果表

| Aggregation method         | Event P | Participant Acc | mIoU | Over-frag ↓ | Over-merge ↓ |
| -------------------------- | ------: | --------------: | ---: | ----------: | -----------: |
| Heuristic window           |         |                 |      |             |              |
| Predicate clustering       |         |                 |      |             |              |
| Entity-aware event builder |         |                 |      |             |              |
| Final builder              |         |                 |      |             |              |

这组实验决定你的 segment graph 是否真的能替代 clip captions。

---

## 4.3 Global-level SG 质量评估

### 目标

评估 global graph 是否真正承担 long-range temporal reasoning。

### 样本

从 diagnostic subset 中选：

```text
first / last / before / after / finally / state_change 类型问题 100–150 道
```

### 标注

每题标注：

```text
anchor_event
target_event
temporal_relation
involved_entities
gold_answer_entity / object / event
support_event_chain
```

### 指标

| Metric                      | 定义                                          |
| --------------------------- | ------------------------------------------- |
| Temporal relation accuracy  | before / after / during 是否正确                |
| First/last accuracy         | first holder / last seen / final state 是否正确 |
| State transition accuracy   | from_state → to_state 是否正确                  |
| Entity continuity accuracy  | 跨 segment same_as 是否正确                      |
| Path validity               | global path 中每条边是否有支持证据                     |
| Temporal contradiction rate | 图中是否出现互相矛盾的时序边                              |

### 结果表

| Global graph variant | Before/After Acc | First/Last Acc | State Acc | Same-as Acc | Contradiction ↓ |
| -------------------- | ---------------: | -------------: | --------: | ----------: | --------------: |
| No global graph      |                  |                |           |             |                 |
| Temporal order only  |                  |                |           |             |                 |
| + state change       |                  |                |           |             |                 |
| + entity continuity  |                  |                |           |             |                 |
| Final global graph   |                  |                |           |             |                 |

这部分要尤其谨慎。你之前 Stage3 Global Graph 加入后反而降分，所以这次要先证明它在哪些问题类型上有用，而不是默认“层级越高越好”。

---

# 5. Graph-native retrieval 实验

这部分是核心。它要回答：

> Agent 是不是通过图查询找到了候选证据，而不是通过 caption 语义相似度找到了 clip？

## 5.1 Retrieval recall on gold evidence subset

对 diagnostic subset 标注 gold support span / frames，然后评估不同检索方式能否召回。

### 方法对比

| Retrieval method         | 说明                       |
| ------------------------ | ------------------------ |
| Caption embedding search | DVD Clip Search baseline |
| Frame triple query       | 只用 frame facts           |
| Segment event query      | 只用 event graph           |
| Global temporal query    | 只用 global graph          |
| Graph path query         | 多跳图路径                    |
| Full graph planner       | Agent 自动组合上述工具           |

### 指标

| Metric                    | 定义                                   |
| ------------------------- | ------------------------------------ |
| Recall@1 / @3 / @5        | top-k candidate spans 是否覆盖 gold span |
| mIoU                      | candidate span 与 gold span 的时间 IoU   |
| MRR                       | gold evidence 排名                     |
| Candidate reduction ratio | 从全视频帧数压缩到候选 frames 的比例               |
| Query success rate        | query 返回非空且可验证结果的比例                  |
| Wrong-anchor rate         | 找到了相关实体但锚定到错误事件的比例                   |

### 结果表

| Method             | R@1 | R@3 | R@5 | mIoU | Candidate frames ↓ | Wrong-anchor ↓ |
| ------------------ | --: | --: | --: | ---: | -----------------: | -------------: |
| Caption Search     |     |     |     |      |                    |                |
| Triple Query       |     |     |     |      |                    |                |
| Event Query        |     |     |     |      |                    |                |
| Temporal Query     |     |     |     |      |                    |                |
| Graph Path Query   |     |     |     |      |                    |                |
| Full Graph Planner |     |     |     |      |                    |                |

这张表比单纯 accuracy 更能证明“graph-native retrieval 有效”。

---

## 5.2 Query type routing experiment

问题解析器需要把自然语言问题转成结构化查询计划。

### Parser 输出格式

```json
{
  "question_type": "temporal_order",
  "entities": [
    {"mention": "the boy", "constraints": ["red shirt"]},
    {"mention": "the lollipop", "constraints": []}
  ],
  "relations": ["hold", "give"],
  "events": ["transfer"],
  "temporal_constraints": ["last"],
  "required_tools": ["resolve_entity", "query_events", "query_temporal"]
}
```

### 实验设置

对 diagnostic subset 手工标注 gold query plan，然后比较 parser 输出。

### 指标

| Metric                       | 定义                                   |
| ---------------------------- | ------------------------------------ |
| Question type accuracy       | 问题类型分类准确率                            |
| Entity mention recall        | 是否抽出关键实体                             |
| Relation recall              | 是否抽出关键谓词                             |
| Temporal constraint accuracy | before / after / first / last 识别是否正确 |
| Tool plan accuracy           | 选择工具是否合理                             |

### 对比

| Parser                         | Type Acc | Entity R | Relation R | Temporal Acc | Tool Plan Acc |
| ------------------------------ | -------: | -------: | ---------: | -----------: | ------------: |
| Rule-based                     |          |          |            |              |               |
| LLM JSON parser                |          |          |            |              |               |
| LLM parser + schema validation |          |          |            |              |               |
| Oracle parser                  |          |          |            |              |               |

Oracle parser 很重要。它可以区分：

```text
分数低是因为图不好？
还是因为 question parser 没把问题转成正确图查询？
```

---

# 6. Graph-only Agent 主实验

## 6.1 Main result table

主表建议这样组织：

| Method                  | Inference memory               | LVBench Acc | Answered Acc | UNKNOWN ↓ | Evidence Trace ↑ | Caption-free? |
| ----------------------- | ------------------------------ | ----------: | -----------: | --------: | ---------------: | ------------: |
| DVD reproduction        | Captions + frames              |             |              |           |                  |            No |
| DVD w/o Clip Search     | Global + frames                |             |              |           |                  |  No / Partial |
| Old GraphDVD            | Captions + SG                  |             |              |           |                  |            No |
| SGDAgent-F              | Frame graph                    |             |              |           |                  |           Yes |
| SGDAgent-FS             | Frame + segment graph          |             |              |           |                  |           Yes |
| SGDAgent-FSG            | Frame + segment + global graph |             |              |           |                  |           Yes |
| SGDAgent-FSG + verifier | Graph + visual support         |             |              |           |                  |           Yes |

这里的关键不是一开始就超过 DVD 官方 74%，而是要证明：

1. **caption-free SGDAgent 能跑完整 benchmark**；
2. **相对 caption-only 或 DVD reproduction 有明确优势**；
3. **在 temporal / relation / state-change / multi-hop 子类上提升更明显**；
4. **UNKNOWN 明显下降或 evidence trace 更完整**。

如果 full score 暂时不如 Caption+SG 旧方案，也可以接受；论文 claim 可以从 “new SOTA” 改成 “graph-native memory replacement with faithful evidence”，但前提是 evidence 和 retrieval 指标必须强。

---

## 6.2 Per-category result table

按问题类型拆分：

| Method       | Entity Attr | Spatial Rel | Event | Temporal | State Change | Multi-hop | Overall |
| ------------ | ----------: | ----------: | ----: | -------: | -----------: | --------: | ------: |
| DVD          |             |             |       |          |              |           |         |
| Old GraphDVD |             |             |       |          |              |           |         |
| SGDAgent-F   |             |             |       |          |              |           |         |
| SGDAgent-FS  |             |             |       |          |              |           |         |
| SGDAgent-FSG |             |             |       |          |              |           |         |

预期结论应是：

```text
Frame graph improves entity/attribute/spatial questions.
Segment graph improves event and participant questions.
Global graph improves first/last/before/after/state-change questions.
Graph path query improves multi-hop questions.
```

如果 Global Graph 总体降分，但在 first/last/state-change 上提升，也可以保留它，但要通过 router 控制使用范围，而不是让所有问题都默认用 global graph。

---

# 7. Tool ablation 实验

## 7.1 Action space ablation

主方法 action space：

```text
parse_question
resolve_entity
query_triples
query_events
query_temporal
query_graph_path
verify_visual_evidence
answer
```

消融设置：

| Variant             | 移除内容                        | 预期影响                              |
| ------------------- | --------------------------- | --------------------------------- |
| w/o Entity Resolver | 不做实体绑定                      | entity mention drift 上升           |
| w/o Triple Query    | 不能查 frame facts             | 属性/关系题下降                          |
| w/o Event Query     | 不能查 segment events          | event 题下降                         |
| w/o Temporal Query  | 不能查 first/last/before/after | 时序题下降                             |
| w/o Graph Path      | 不能多跳                        | multi-hop 题下降                     |
| w/o Visual Verifier | 不做视觉确认                      | hallucination / wrong evidence 上升 |
| fixed workflow      | 固定工具顺序                      | 效率下降，错用工具上升                       |

结果表：

| Variant            | Acc | UNKNOWN ↓ | Evidence Validity | Avg Steps ↓ | Wrong Tool Rate ↓ |
| ------------------ | --: | --------: | ----------------: | ----------: | ----------------: |
| Full SGDAgent      |     |           |                   |             |                   |
| w/o Resolver       |     |           |                   |             |                   |
| w/o Triple Query   |     |           |                   |             |                   |
| w/o Event Query    |     |           |                   |             |                   |
| w/o Temporal Query |     |           |                   |             |                   |
| w/o Path Query     |     |           |                   |             |                   |
| w/o Verifier       |     |           |                   |             |                   |
| fixed workflow     |     |           |                   |             |                   |

DVD 已经分析过不同 tool behavior，包括 Simple Action、Iterative Search、Frame Inspect Trap、Clip Search Trap；你可以沿用这个分析框架，但把 Clip Search Trap 改成 Graph Query Failure / Temporal Drift / Over-verification Loop。

---

# 8. Visual Evidence Verifier 实验

Verifier 是防止 graph hallucination 的关键。VideoMind 的经验说明，先定位候选 moment，再通过 verifier 判断候选 moment 是否正确，是 long-video reasoning 的有效机制。

但你的 verifier 不能输出 caption。建议输出结构化 JSON：

```json
{
  "claim_id": "claim_17",
  "verified": true,
  "support_frames": ["frame_11504", "frame_11512"],
  "support_tracks": ["track_boy_01", "track_lollipop_03"],
  "verified_facts": [
    ["boy_01", "holds", "lollipop_03"],
    ["boy_01", "faces", "girl_02"]
  ],
  "confidence": 0.82
}
```

## 8.1 Verifier ablation

| Verifier style         | 输出形式                            | 是否允许自然语言 |
| ---------------------- | ------------------------------- | -------: |
| No verifier            | 直接用 graph result                |       No |
| Binary verifier        | true / false                    |       No |
| Structured verifier    | verified_facts + support_frames |       No |
| Free-form VQA verifier | natural language answer         | Yes，违规上限 |

结果表：

| Verifier      | Acc | Evidence Validity | False Positive ↓ | False Negative ↓ | Caption-free Pass |
| ------------- | --: | ----------------: | ---------------: | ---------------: | ----------------: |
| None          |     |                   |                  |                  |               Yes |
| Binary        |     |                   |                  |                  |               Yes |
| Structured    |     |                   |                  |                  |               Yes |
| Free-form VQA |     |                   |                  |                  |                No |

主方法只能用 structured verifier。Free-form VQA 只能作为 upper bound，不可以作为正式 SGDAgent。

---

# 9. Caption leakage 对照实验

这一组实验专门回答 reviewer 可能提出的质疑：

> 你的图是不是只是 caption 的另一种序列化形式？

## 9.1 Graph serialization ablation

同一个 graph database，用不同 observation 格式给 Agent：

| Setting           | Observation form            | 示例                                                |
| ----------------- | --------------------------- | ------------------------------------------------- |
| Naturalized graph | 类 caption 句子                | “The boy gives a lollipop to the girl.”           |
| Tuple graph       | 结构化三元组                      | `(boy_01, gives, lollipop_03, recipient=girl_02)` |
| ID-heavy graph    | 强结构化                        | `event_24: type=give, subj=e12, obj=e31, rec=e08` |
| Minimal graph     | 只给 ID + predicate + support | 最少自然语言                                            |

主方法应该用 **Tuple graph** 或 **ID-heavy graph**。Naturalized graph 只能作为分析项，因为它接近 caption。

结果表：

| Observation format | Acc | Evidence Validity | Caption-like leakage score ↓ |
| ------------------ | --: | ----------------: | ---------------------------: |
| Naturalized graph  |     |                   |                         High |
| Tuple graph        |     |                   |                          Low |
| ID-heavy graph     |     |                   |                     Very low |
| Minimal graph      |     |                   |                     Very low |

## 9.2 Embedding leakage ablation

你允许 node / predicate embedding 用于实体消歧，但必须证明它不是 caption search。

| Retrieval index          | Corpus                                               | 合法性 |
| ------------------------ | ---------------------------------------------------- | --- |
| Caption embedding        | clip captions                                        | 不合法 |
| Event sentence embedding | “boy gives lollipop to girl”                         | 风险高 |
| Triple label embedding   | `give(subject=boy, object=lollipop, recipient=girl)` | 可接受 |
| Entity label embedding   | entity type + attributes                             | 可接受 |
| Pure symbolic query      | inverted index + graph traversal                     | 最严格 |

结果表：

| Matching method                  | Acc | Retrieval R@5 | Caption-free Pass | Notes               |
| -------------------------------- | --: | ------------: | ----------------: | ------------------- |
| Caption embedding                |     |               |                No | DVD-style           |
| Event sentence embedding         |     |               |              Weak | hidden caption risk |
| Triple label embedding           |     |               |               Yes | allowed             |
| Pure symbolic                    |     |               |               Yes | strict lower bound  |
| Hybrid symbolic + node embedding |     |               |               Yes | main setting        |

主方法建议用：

```text
symbolic filtering first,
embedding only for canonical entity / predicate normalization,
never over caption corpus.
```

---

# 10. Agent behavior 分析

你需要建立新的行为类型，不要完全沿用 DVD 的 Clip Search Trap / Frame Inspect Trap。

建议定义：

| Behavior type          | 定义                                  | 代表意义                |
| ---------------------- | ----------------------------------- | ------------------- |
| Direct Triple Answer   | resolve → triple → verify → answer  | 简单事实题               |
| Event-grounded Answer  | resolve → event → verify → answer   | 事件题                 |
| Temporal Chain Answer  | event → temporal → path → verify    | 长程时序题               |
| Multi-hop Path Answer  | entity → path → event/fact → verify | 多跳题                 |
| Entity Drift           | resolver 绑定错实体                      | 图实体问题               |
| Empty Graph Query      | 查询不到候选                              | 图 recall 问题         |
| Over-broad Query       | 返回过多候选                              | query constraint 不足 |
| Over-verification Loop | verifier 反复否定但 planner 不收敛          | Agent 控制问题          |
| Premature Answer       | 未验证就回答                              | reasoning policy 问题 |

输出分析表：

| Behavior          | Ratio | Acc | Avg Steps | Evidence Validity | Main failure |
| ----------------- | ----: | --: | --------: | ----------------: | ------------ |
| Direct Triple     |       |     |           |                   |              |
| Event-grounded    |       |     |           |                   |              |
| Temporal Chain    |       |     |           |                   |              |
| Multi-hop Path    |       |     |           |                   |              |
| Entity Drift      |       |     |           |                   |              |
| Empty Query       |       |     |           |                   |              |
| Over-verification |       |     |           |                   |              |

这部分可以直接支撑论文里的 Analysis section。

---

# 11. Failure taxonomy

每个错误样本都要归到一个失败类别。不要只报 wrong / unknown。

| Failure type            | 说明                        | 对应修复                      |
| ----------------------- | ------------------------- | ------------------------- |
| Missing entity          | 关键实体没入图                   | 改 frame SG builder        |
| Entity ID switch        | 跨帧实体对齐错                   | 改 track / same_as         |
| Wrong predicate         | relation / action 识别错     | 改 predicate normalization |
| Event missing           | frame facts 有，但没聚合成 event | 改 event builder           |
| Event boundary wrong    | 找到事件但时间窗偏                 | 改 segment aggregation     |
| Temporal relation wrong | before / after / last 错   | 改 global graph            |
| Query parser wrong      | 问题转图查询错                   | 改 parser / schema         |
| Resolver wrong          | mention 绑定错 entity        | 改 resolve_entity          |
| Path search wrong       | 找到无效多跳链                   | 加 path constraints        |
| Verifier false positive | 错误证据被确认                   | 改 verifier                |
| Verifier false negative | 正确证据被否定                   | 改 verifier                |
| Answer composer wrong   | 证据正确但选项映射错                | 改 answer prompt           |

最终要有一张：

| Failure type   | Count | Ratio | Most affected question type | Next fix |
| -------------- | ----: | ----: | --------------------------- | -------- |
| Missing entity |       |       | Entity / attr               |          |
| Event missing  |       |       | Event                       |          |
| Temporal wrong |       |       | First / last                |          |
| Resolver wrong |       |       | Multi-entity                |          |

这比泛泛写“模型理解错误”有价值。

