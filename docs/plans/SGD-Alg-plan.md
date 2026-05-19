SGD-Agent v2.0 算法设计详解

Canonical status:
  This is the highest-level algorithm specification for SGDAgent development.
  `dev-plan.md` is the only canonical implementation progress document.
  Historical plans (`SG-plan.md`, `app-plan.md`) are archived and must not
  override decisions in this file.

Last consolidated:
  2026-05-20

1.1 定位与核心目标
SGD-Agent v2.0 的目标是将现有 SGD-Agent 从：
Scene Graph as Retrieval Index
升级为：
Scene Graph as Native Video Memory and Reasoning Substrate
也就是说，scene graph 不再只是帮助定位 frame / segment / caption 的索引，而是成为 Agent 的主要视频记忆、推理对象和证据追溯载体。
与 VideoAgent 类方法相比，VideoAgent 的状态更新仍然依赖 VLM 将检索帧转换成 captions，再由 LLM 基于这些文本状态继续推理；其核心 loop 是“预测答案—自反思—寻找缺失信息—CLIP 检索帧—VLM caption 更新状态”。 与 DVD 相比，DVD 虽然有多粒度 database 和 Global Browse / Clip Search / Frame Inspect 工具，但 clip level 仍显式依赖 caption corpus 和 caption embedding retrieval。 SGD-Agent v2.0 的区别是：Agent 推理时不再读取 caption database，而只读取结构化图事实、事件节点、时间/因果边和视觉证据指针。
VideoMind 的 Planner / Grounder / Verifier / Answerer 角色化设计说明，长视频推理需要“规划—定位—验证—回答”的闭环，而不是一次性文本 CoT。 SGD-Agent v2.0 继承这个思想，但把 Grounder 从 temporal segment grounding 升级为 subgraph grounding，把 Verifier 从 moment verifier 升级为 graph fact / event / evidence chain verifier。

---
0. 固化决策与边界
0.1 文档层级
本文件是算法与系统语义的最高指示。所有开发任务、进度状态、验收标准统一落在 `dev-plan.md`，并按本文件的模块边界组织。历史实验计划和 demo 计划只保留上下文，不再作为开发指令来源。

0.2 MVP 与 strict graph-only 的边界
MVP 可以复用现有代码能力，但必须通过 wrapper / adapter 约束成本文定义的语义。strict graph-only 实验更严格：不能暴露 captions、caption embeddings、free-form summaries，也不能让 Agent 通过自由视觉检查绕回 caption-like observation。

0.3 Run Mode Taxonomy
Strict Graph-only:
  Agent may access only stripped graph memory, graph-native tool outputs,
  evidence pointers, and structured VisualVerify results.
  Captions, caption embeddings, free-form summaries, transcript snippets,
  GEI prompt timelines, and arbitrary raw-frame inspection are forbidden.

MVP Graph-guided:
  Temporary navigation aids such as GEI-MVP may be used to locate candidate
  intervals, but they cannot be used as answer evidence.

Builder-assisted:
  Captions, transcripts, VLM narrations, and OCR may be used during offline
  construction, but all such text must be stripped from inference artifacts.

Hybrid:
  Caption and graph tools may both be used. This is not a strict graph-only
  setting and must be reported separately.

0.4 Strict Inference Boundary
Allowed:
  stripped graph memory
  entity records
  graph facts
  event nodes
  temporal edges
  state carry-over records
  evidence pointers
  structured verification results

Forbidden:
  captions
  caption snippets
  transcript snippets
  caption embeddings
  free-form summaries
  GEI prompt timelines not materialized from G_global_v1
  arbitrary raw-frame inspection
  direct frame_inspect_tool
  clip_search_tool
  global_browse_tool

Temporary exception:
  frame_inspect_tool may be used only internally by VisualVerify, and only for
  structured claims grounded by graph-derived evidence pointers.

0.5 VisualVerify 与 frame_inspect_tool
MVP 中保留 `frame_inspect_tool`，但它只能作为内部 `VisualVerify` adapter，不能直接作为 Agent-facing action。

Agent-facing:
  VisualVerify(claim, evidence_pointers) -> VerifyResult

Internal MVP implementation:
  VisualVerifyAdapter may call frame_inspect_tool on bounded evidence intervals.

禁止：
  Agent freely calls frame_inspect_tool to describe arbitrary clips.
  Agent uses frame_inspect_tool to generate open-ended narration.
  Agent chooses arbitrary long timestamp ranges without graph support.

允许：
  Graph query -> candidate fact/event/path -> EvidenceTrace -> VisualVerify.
  VisualVerify may expand a graph-derived evidence interval by a small delta
  such as +/-2s or +/-5s, but cannot expand into unrelated long segments.

0.6 GEI status
GEI-MVP is retained only as a navigation aid in MVP Graph-guided mode. It can suggest candidate time regions, but it is not answer evidence and cannot appear in the final evidence chain. It should not be exposed in Strict Graph-only unless it is stripped into structured graph signatures. After Stage3 lands, GEI must be materialized from G_global_v1.

GEI-MVP:
  prompt/text timeline from Stage-2 graph facts
  navigation only
  excluded from evidence chain
  not valid for Strict Graph-only unless stripped into structured graph signatures

GEI-v2:
  materialized from G_global_v1
  event nodes + temporal edges + entity links
  usable as global graph index
  final evidence still traces to segment events and frame facts

Strict graph-only setting should either disable non-materialized GEI or label the run as MVP graph-guided rather than strict graph-only.

0.7 Stage3 v1 scope
Stage3 v1 is intentionally minimal:
  entity_lifecycle_index
  global_event_nodes
  temporal_edges
  first_last_index
  simple_state_carry_over

Stage3 v1 supports:
  before / after / overlap
  first / last
  holder(object)=entity carry-over
  location(entity/object)=place carry-over
  visible / disappeared / reappeared
  last_known_state

Stage3 v1 does not include:
  full causal edges
  episodes
  narrative graph
  global repair loop
  complex causal path reasoning

0.8 Agent-facing tool contract
The Agent sees Alg-plan semantic tools. Existing tools are low-level adapters.

Agent-facing MVP tools:
  QuestionDecompose
  EntityResolve
  RelationQuery
  EventSearch
  TemporalOrderQuery
  EvidenceTrace
  VisualVerify
  AnswerWithEvidence

Internal adapters:
  frame_graph_query_tool
  segment_graph_query_tool
  frame_inspect_tool
  current GEI / Stage1 / Stage2 graph files

0.9 Final answer contract
MVP answers must be produced through AnswerWithEvidence. A benchmark runner may extract only the option letter or short answer, but the trace must preserve the full structured evidence chain.

Minimum schema:
{
  "answer": "...",
  "confidence": 0.0,
  "evidence_chain": {
    "global": [],
    "segment_events": [],
    "frame_facts": [],
    "support_frames": [],
    "visual_evidence": [],
    "verification": []
  }
}

0.10 Construction artifacts vs inference artifacts
Caption-assisted construction is allowed, but inference artifacts must be stripped.

Allowed construction artifact:
  construction_graph_raw.json may contain caption_context, caption_snippets,
  VLM narration, ASR snippets, and builder debug fields.

Required inference artifact:
  inference_graph_stripped.json must remove caption-like fields.

Agent tools can only read inference-safe graph memory. Tool outputs, observations, traces, and evidence chains cannot expose captions.

0.11 Summary and graph signatures
Free-form `summary` is high-risk leakage in strict mode. It must be removed from inference observations or replaced by graph-derived structured signatures.

Forbidden in strict mode:
{
  "summary": "The man gives a cup to another person in the room."
}

Allowed:
{
  "event_signature": {
    "event_type": "give",
    "participants": {
      "agent": "person_03",
      "patient": "cup_01",
      "recipient": "person_05"
    },
    "preconditions": ["holder(cup_01)=person_03"],
    "effects": ["holder(cup_01)=person_05"],
    "time_span": ["00:12:04", "00:12:13"]
  }
}

0.12 Embedding policy
Allowed:
  symbolic retrieval
  tuple-label retrieval
  graph-signature embedding

Forbidden:
  caption embedding
  clip narration embedding
  VLM-generated free-form summary embedding

Every embedding entry must record `embedding_source`.

Allowed strict sources:
  tuple_label
  structured_graph_signature

Forbidden strict sources:
  caption
  transcript
  free_form_summary
  event_sentence_summary

Natural-language event sentence embedding is allowed only as an ablation and must be labeled as non-strict / leakage-risk.

0.13 MVP evidence granularity
MVP requires frame-level evidence tracing:
  support_frame_ids
  timestamps

v2.0 full adds region-level evidence:
  bbox
  crop
  mask
  visual embedding
  tracklet

0.14 Causal graph policy
Causal graph is not in MVP. MVP supports temporal and simple state carry-over. Why-questions may retrieve temporal/state-change predecessors, but the answer must be labeled as a causal hypothesis, not a verified causal edge.

---
1. 总体架构
1.1 系统输入输出
输入
V: long-form video
Q: user question / VQA query / evidence tracing request
Optional:
  A: audio transcript
  M: metadata, e.g., shot boundary, subtitle, chapter, scene label
离线输出
G_video: cross-scale spatio-temporal scene hypergraph
GraphStore: graph-native database
EvidenceStore: frame / bbox / mask / crop / tracklet evidence store
IndexStore: entity / relation / event / temporal / causal indexes
在线输出
Answer:
  answer_text or answer_option

Evidence Chain:
  global path
  segment event nodes
  frame-level facts
  visual evidence pointers

Trace:
  tool calls
  queried subgraphs
  verified facts
  confidence / uncertainty

---
1.2 两阶段系统
Stage A. Offline Graph Memory Construction
  long video → graph-native memory

Stage B. Online Graph-driven Agent Reasoning
  question + graph memory → answer + evidence chain
Stage A：离线构建
Video
 → temporal unit construction
 → entity-tube construction
 → frame-level atomic fact graph
 → temporal fact consolidation
 → segment-level event/state-transition graph
 → global entity-event-causal graph
 → graph verification and repair
 → graph store + indexes + evidence store
Stage B：在线推理
Question
 → question-to-graph plan
 → entity resolution
 → graph-native querying
 → temporal / causal / state reasoning
 → evidence tracing
 → graph / visual verification
 → answer with evidence chain

---
2. 设计原则
2.1 Caption-free inference
SGD-Agent v2.0 的关键约束：
During inference, the agent cannot retrieve or read captions.
允许：
Captions / transcripts / VLM narrations / ASR
  → used as offline graph construction signals.
禁止：
caption snippets
caption embedding retrieval
caption database as observation
free-form video narration as evidence
在线阶段 Agent 的 observation 必须是：
graph facts
entity records
event nodes
state transitions
temporal / causal edges
evidence pointers
verification results
而不是：
natural-language clip descriptions

---
2.2 Graph fact is the minimum memory unit
最小记忆单元不是 caption，而是：
temporally grounded graph fact
标准格式：
{
  "fact_id": "fact_000827",
  "level": "frame_or_moment",
  "subject": "person_03",
  "predicate": "holds",
  "object": "cup_01",
  "time_span": ["00:12:04.200", "00:12:09.600"],
  "evidence": [
    {
      "frame_id": "frame_03612",
      "bbox_subject": [318, 102, 491, 542],
      "bbox_object": [502, 331, 547, 381],
      "crop_id": "crop_03612_07"
    }
  ],
  "confidence": 0.82,
  "source": ["detector", "tracker", "vlm_relation_verifier"],
  "status": "verified"
}

---
2.3 Every high-level node must be supported
所有高层 event / causal edge / answer 必须可以回溯：
answer
 → global event / causal path
 → segment event node
 → frame-level facts
 → frame / bbox / mask / crop evidence
如果某个高层节点无法回溯到低层事实，必须标记为：
unsupported
low_confidence
needs_visual_verification

---
2.4 Three-level hierarchy is retained, but retrieval is upgraded
SGD-Agent v2.0 继续保留三层结构：
Frame-level Graph Memory
Segment-level Event Graph Memory
Global-level Video Graph Memory
但工具不再只是：
Frame Retrieval
Segment Retrieval
Global Retrieval
而是升级为：
EntityResolve
RelationQuery
AttributeQuery
EventSearch
StateChangeQuery
TemporalOrderQuery
CausalPathQuery
GraphExpand
EvidenceTrace
GraphVerify
VisualVerify
AnswerWithEvidence

---
3. 核心表征：Cross-scale Spatio-temporal Scene Hypergraph
3.1 形式化定义
导师原始定义可保留：
G_video = (E_video, G_frame, G_segment, G_global)
SGD-Agent v2.0 建议扩展为：
G_video = (E_video, G_F, G_S, G_G, Π, Ω, I)
其中：
E_video:
  全局持久实体集合，即 entity lifecycle registry。

G_F:
  frame / moment-level atomic fact graph sequence。

G_S:
  segment-level event and state-transition graph sequence。

G_G:
  global entity-event-temporal-causal graph。

Π:
  cross-level support mapping。
  frame facts → segment events → global paths。

Ω:
  evidence and provenance store。
  fact / event / path → frame / bbox / mask / crop / source / confidence。

I:
  graph-native indexes。
  entity index, relation index, event index, temporal index, causal index, evidence index。

---
3.2 树状超图骨架 + 非树状长程边
三层结构构成树状超图骨架：
Frame-level subgraphs
  → grouped into Segment-level event subgraphs
    → abstracted into Global-level event / causal graph
但长视频语义不能是严格树，因此必须加入横向边：
same_entity
reappears_as
state_carries_over
before / after / overlaps
causes / enables / prevents
object_transferred_to
location_changed_to
因此最终结构是：
tree-structured hypergraph backbone
+
cross-segment temporal / causal / entity-continuity links

---
4. Entity Lifecycle Registry
4.1 设计目标
E_video 不只是实体词典，而是全视频级实体生命周期表。它负责：
跨帧身份一致性
跨片段实体重识别
遮挡 / 离屏后的 object permanence
referential expression resolution
状态随时间变化
关系随时间变化
证据追溯

---
4.2 Entity schema
{
  "entity_id": "person_03",
  "category": "person",
  "aliases": [
    "man in black sweater",
    "the tall man",
    "the man with mustache"
  ],
  "appearance": {
    "clothing": ["black sweater"],
    "hair": ["short dark hair"],
    "face": ["mustache"],
    "body": ["tall", "thin"]
  },
  "tracklets": [
    {
      "tracklet_id": "trk_018",
      "time_span": ["00:03:20.000", "00:04:12.000"],
      "frames": ["frame_01000", "frame_01001"],
      "mean_embedding_id": "emb_trk_018",
      "confidence": 0.89
    }
  ],
  "states_over_time": [
    {
      "attribute": "location",
      "value": "kitchen_01",
      "time_span": ["00:03:20.000", "00:04:12.000"],
      "support_facts": ["fact_00231", "fact_00244"]
    }
  ],
  "relations_over_time": [
    {
      "predicate": "holds",
      "object": "cup_01",
      "time_span": ["00:12:04.200", "00:12:09.600"],
      "support_facts": ["fact_000827"]
    }
  ],
  "first_seen": "00:03:20.000",
  "last_seen": "00:15:42.000",
  "disappeared_intervals": [
    ["00:04:12.000", "00:07:40.000"]
  ],
  "reappeared_intervals": [
    ["00:07:40.000", "00:08:15.000"]
  ],
  "confidence": 0.78
}

---
4.3 Entity lifecycle 构建逻辑
Input:
  object detections
  instance masks
  visual embeddings
  local tracklets
  VLM descriptions
  historical entity registry

Process:
  1. Local tracking within shot.
  2. Cross-shot re-identification.
  3. Semantic alias generation.
  4. Entity merge / split verification.
  5. State and relation timeline construction.

Output:
  persistent entities with tracklets and temporal states.

Alias provenance policy:
  Strict inference aliases are allowed only when they are entity-local referring
  expressions derived from graph-safe attributes, categories, or repeated entity
  labels, e.g. `person`, `man in black sweater`, `man with mustache`, `red car`.

  Free-form narrative aliases copied from captions are construction-only and
  must be stripped or blocked in strict inference, e.g. `the man who argues with
  the woman after entering the room` or `the villain who later breaks the mirror`.

开发优先级
第一版可以先实现：
entity_id
category
graph-safe aliases
time spans
support_frame_ids
第二版再加入：
bbox-level entity registry
local tracklet
cross-clip semantic matching
mask-level grounding
object permanence
reappearing entity detection
state timeline smoothing

---
5. Frame / Moment-level Atomic Fact Graph
5.1 功能定位
Frame-level graph 负责存储原子视觉事实：
实体是否出现
实体属性
实体空间位置
实体间空间关系
接触关系
持有关系
局部动作
姿态 / 表情
文字 / OCR
数量
这一层服务于：
attribute question
counting question
fine-grained relation question
local evidence tracing
visual verification

---
5.2 Frame graph schema
G_F^i = (E_F^i, A_F^i, R_F^i, Ω_F^i)
其中：
E_F^i ⊂ E_video:
  当前帧 / 当前短 moment 中可见实体。

A_F^i:
  attribute facts。

R_F^i:
  relation facts。

Ω_F^i:
  visual evidence pointers。

---
5.3 Atomic fact types
Entity presence fact
{
  "fact_type": "entity_presence",
  "entity": "cup_01",
  "time": "00:12:04.200",
  "bbox": [502, 331, 547, 381],
  "mask_id": "mask_03612_04",
  "confidence": 0.91
}
Attribute fact
{
  "fact_type": "attribute",
  "subject": "cup_01",
  "attribute": "color",
  "value": "blue",
  "time_span": ["00:12:04.200", "00:12:09.600"],
  "evidence": ["frame_03612:bbox_04"],
  "confidence": 0.76
}
Relation fact
{
  "fact_type": "relation",
  "subject": "person_03",
  "predicate": "holds",
  "object": "cup_01",
  "relation_class": "contact_possession",
  "time_span": ["00:12:04.200", "00:12:09.600"],
  "evidence": ["frame_03612", "frame_03615"],
  "confidence": 0.82
}
Local action fact
{
  "fact_type": "local_action",
  "subject": "person_03",
  "predicate": "extends_arm_toward",
  "object": "person_05",
  "time_span": ["00:12:08.000", "00:12:11.000"],
  "evidence": ["frame_03618", "frame_03621"],
  "confidence": 0.71
}

---
5.4 Predicate ontology
建议使用“受控谓词集合 + open-vocabulary alias”。
spatial:
  left_of, right_of, above, below, in_front_of, behind, inside, near

contact:
  touching, holding, carrying, wearing, sitting_on, standing_on, leaning_on

attention:
  looking_at, facing, pointing_at

possession:
  holds, takes, gives, receives, drops, puts_down

motion:
  moves_toward, moves_away_from, moves_upward, moves_downward, exits, enters

state:
  open, closed, broken, intact, lit, off, empty, full

social/action:
  talking_to, helping, chasing, fighting, feeding, following
开放谓词保留 alias：
{
  "canonical_predicate": "holds",
  "aliases": ["grabs", "clutches", "holds tightly"]
}

---
6. Segment-level Event and State-transition Graph
6.1 功能定位
Segment graph 不应只是“片段内动态三元组”。它应显式建模：
event
participants
precondition
effect
state transition
temporal relation
supporting frame facts
这是 SGD-Agent v2.0 替代 captions 的关键层。

---
6.2 Segment graph schema
G_S^j = (E_S^j, F_S^j, V_event^j, R_temporal^j, R_state^j, Π_S^j)
其中：
E_S^j:
  片段内活跃实体。

F_S^j:
  从 frame-level facts 聚合出的 interval facts。

V_event^j:
  event nodes。

R_temporal^j:
  event-event temporal relations。

R_state^j:
  state transition relations。

Π_S^j:
  event → supporting frame facts。

---
6.3 Event node schema
{
  "event_id": "event_0027",
  "event_type": "give",
  "time_span": ["00:12:04.200", "00:12:13.400"],
  "participants": {
    "agent": "person_03",
    "patient": "cup_01",
    "recipient": "person_05",
    "location": "room_01"
  },
  "preconditions": [
    {
      "fact": "holder(cup_01)=person_03",
      "support_facts": ["fact_000827"]
    }
  ],
  "core_facts": [
    "fact_000827",
    "fact_000834",
    "fact_000839"
  ],
  "effects": [
    {
      "state": "holder(cup_01)=person_05",
      "support_facts": ["fact_000845"]
    }
  ],
  "temporal_edges": [
    {
      "relation": "before",
      "target_event": "event_0028"
    }
  ],
  "evidence": [
    "frame_03612:bbox_person_03,bbox_cup_01",
    "frame_03621:bbox_person_05,bbox_cup_01"
  ],
  "confidence": 0.74,
  "status": "verified"
}

---
6.4 State transition schema
{
  "transition_id": "trans_0044",
  "entity": "mirror_01",
  "attribute": "integrity",
  "from": "intact",
  "to": "broken",
  "trigger_event": "event_0044",
  "time_span": ["00:09:46.000", "00:10:06.000"],
  "support_facts": [
    "fact_0203",
    "fact_0204",
    "fact_0205"
  ],
  "confidence": 0.81
}

---
6.5 Event induction examples
Possession transfer
Input facts:
person_03 holds cup_01 [t1, t2]
person_03 extends_arm_toward person_05 [t2, t3]
cup_01 near person_05 hand [t3, t4]
person_05 holds cup_01 [t4, t5]
Induced event:
event_type = give
agent = person_03
patient = cup_01
recipient = person_05
effect = holder(cup_01)=person_05
Object breaking
Input facts:
mirror_01 intact [t1]
villain_01 strikes mirror_01 [t2]
mirror_01 broken [t3]
fragment_group_01 moves_upward [t4]
Induced events:
event_44 = villain breaks mirror
trans_44 = mirror integrity intact → broken
event_45 = fragments fly upward
temporal_edge = event_44 before event_45
Door opening
Input facts:
door_01 closed [t1]
person_03 near door_01 [t1]
person_03 touches handle_01 [t2]
door_01 open [t3]
Induced event:
event_type = open
agent = person_03
patient = door_01
precondition = door_01.closed
effect = door_01.open

---
7. Global-level Entity-Event-Causal Graph
7.1 功能定位
Global graph 负责长程推理：
first / last
before / after
why
consequence
who finally
object reappearance
state carry-over
multi-hop temporal reasoning
multi-hop causal reasoning

---
7.2 Global graph schema
G_G = (E_video, V_event, V_episode, R_temporal, R_causal, R_entity, R_narrative)
其中：
V_event:
  全视频事件节点，来自 segment-level events。

V_episode:
  多事件 episode / scene / narrative unit。

R_temporal:
  before, after, overlaps, first_occurrence, last_occurrence。

R_causal:
  causes, enables, prevents, triggers, resolves。

R_entity:
  same_entity, reappears, carries_state, transfers_object。

R_narrative:
  goal, conflict, outcome, repeated_motif。

---
7.3 Global edge schema
Temporal edge
{
  "edge_id": "gedge_0012",
  "edge_type": "temporal",
  "predicate": "before",
  "source": "event_0027",
  "target": "event_0031",
  "time_gap": 18.4,
  "confidence": 0.96,
  "support": ["time_span_order"]
}
Causal edge
{
  "edge_id": "gedge_0048",
  "edge_type": "causal",
  "predicate": "enables",
  "source": "event_0012",
  "target": "event_0013",
  "causal_variables": ["food_01", "table_01", "bunny_group_01"],
  "evidence": [
    "event_0012",
    "event_0013",
    "state(food_01.location=table_01)"
  ],
  "confidence": 0.68,
  "status": "needs_verification"
}
Entity carry-over edge
{
  "edge_id": "gedge_0091",
  "edge_type": "entity_continuity",
  "predicate": "reappears_as",
  "source": "person_03@trk_018",
  "target": "person_03@trk_029",
  "support": [
    "appearance_similarity=0.83",
    "alias_match=man_with_mustache"
  ],
  "confidence": 0.78
}

---
8. Cross-level Support Mapping and Evidence Store
8.1 Support mapping
定义：
Π: lower-level evidence supports higher-level semantics
具体包括：
Π_FS:
  frame facts → segment events

Π_SG:
  segment events → global event / causal paths

Π_FA:
  frame facts → answer evidence chain
示例：
{
  "support_id": "support_0027",
  "high_level_node": "event_0027",
  "supported_by": [
    "fact_000827",
    "fact_000834",
    "fact_000839",
    "fact_000845"
  ],
  "support_type": "event_induction",
  "support_score": 0.74
}

---
8.2 Evidence pointer
所有事实、事件、答案都必须绑定 evidence pointer。
{
  "evidence_id": "ev_03612_07",
  "video_id": "video_0001",
  "timestamp": "00:12:04.200",
  "frame_id": "frame_03612",
  "entity_ids": ["person_03", "cup_01"],
  "bbox": {
    "person_03": [318, 102, 491, 542],
    "cup_01": [502, 331, 547, 381]
  },
  "mask_ids": {
    "cup_01": "mask_03612_04"
  },
  "crop_ids": [
    "crop_03612_person_03",
    "crop_03612_cup_01"
  ],
  "source_fact_ids": ["fact_000827"]
}

---
8.3 Evidence chain schema
最终回答必须包含：
{
  "answer": "person_05",
  "answer_confidence": 0.76,
  "evidence_chain": {
    "global_reasoning": [
      {
        "query": "last holder of cup_01",
        "result": "holder(cup_01)=person_05",
        "source": "TemporalOrderQuery"
      }
    ],
    "segment_events": [
      {
        "event_id": "event_0027",
        "event_type": "give",
        "time_span": ["00:12:04.200", "00:12:13.400"]
      }
    ],
    "frame_facts": [
      "fact_000827",
      "fact_000845"
    ],
    "visual_evidence": [
      "frame_03612:bbox_person_03,bbox_cup_01",
      "frame_03621:bbox_person_05,bbox_cup_01"
    ]
  }
}

---
9. Graph Memory Construction Algorithm
9.1 Overview
Algorithm 1: BuildGraphMemory(V)

Input:
  V: long video

Output:
  G_video
  GraphStore
  EvidenceStore
  IndexStore

Steps:
  1. Construct temporal units.
  2. Build entity tubes.
  3. Generate frame-level atomic facts.
  4. Consolidate temporal facts.
  5. Induce segment-level events and state transitions.
  6. Build global temporal / causal graph.
  7. Verify and repair graph.
  8. Build graph-native indexes.

---
9.2 Temporal unit construction
目标
构建三种时间单位：
Frame unit:
  key frame or sampled frame。

Micro-clip unit:
  short window, e.g., 2–5 seconds。

Event unit:
  adaptive segment based on shot / action / relation / state boundary。
实现策略
1. Uniform sampling:
   每秒 1–2 帧，用于基础 coverage。

2. Content-adaptive sampling:
   shot boundary
   motion peak
   object appearance / disappearance
   relation change
   OCR / subtitle change

3. Sliding window:
   默认 30s window, 5s overlap。
   用于 segment-level event induction。

4. Event boundary refinement:
   根据 state change / action onset-offset / relation persistence 合并或切分。

---
9.3 Entity-tube builder
Input:
  sampled frames / micro-clips

Process:
  1. Detection / segmentation.
  2. Local tracking.
  3. Visual embedding extraction.
  4. Cross-shot re-ID.
  5. VLM appearance description.
  6. Entity registry update.
  7. Merge / split verification.

Output:
  E_video
  tracklets
  entity evidence pointers
Pseudocode
def build_entity_registry(frames):
    E_video = {}
    local_tracklets = run_tracker(frames)

    for trk in local_tracklets:
        visual_emb = compute_tracklet_embedding(trk)
        appearance_text = describe_tracklet(trk)
        candidate = match_existing_entity(E_video, visual_emb, appearance_text)

        if candidate.score >= MERGE_THRESHOLD:
            update_entity(candidate.entity_id, trk, appearance_text)
        else:
            entity_id = create_new_entity(trk, appearance_text)
            E_video[entity_id] = init_entity_record(entity_id, trk)

    E_video = verify_entity_consistency(E_video)
    return E_video

---
9.4 Frame-level graph generation
输入
frame
visible entities
historical entity registry
local tracklets
optional caption / transcript only for offline construction
输出
G_F^i:
  entity presence facts
  attribute facts
  relation facts
  local action facts
  evidence pointers
建议组合模型
Detector / Segmenter:
  object, bbox, mask

Tracker:
  tracklet id, temporal continuity

VSGG / relation model:
  object-object relation

VLM structured extraction:
  open-vocabulary attribute
  rare relation
  OCR / screen text
  fine-grained state

LLM canonicalizer:
  predicate canonicalization
  entity alias normalization
Pseudocode
def build_frame_graph(frame, E_video):
    detections = detect_objects(frame)
    masks = segment_objects(frame)
    aligned_entities = align_to_entity_registry(detections, masks, E_video)

    attributes = extract_attributes(frame, aligned_entities)
    relations = extract_relations(frame, aligned_entities)
    local_actions = extract_local_actions(frame, aligned_entities)

    facts = []
    facts += make_presence_facts(aligned_entities)
    facts += make_attribute_facts(attributes)
    facts += make_relation_facts(relations)
    facts += make_local_action_facts(local_actions)

    facts = canonicalize_predicates(facts)
    facts = bind_visual_evidence(facts, frame)
    facts = score_fact_confidence(facts)

    return FrameGraph(frame_id=frame.id, facts=facts)

---
9.5 Temporal fact consolidation
目标
把离散 frame facts 聚合成 interval facts。
输入：
person_03 holds cup_01 at frame 3612
person_03 holds cup_01 at frame 3615
person_03 holds cup_01 at frame 3618
输出：
person_03 holds cup_01 [00:12:04.200, 00:12:09.600]
操作
temporal smoothing
minimum duration filtering
relation hysteresis
confidence aggregation
contradiction suppression
Pseudocode
def consolidate_temporal_facts(frame_graphs):
    grouped = group_by_subject_predicate_object(frame_graphs)

    interval_facts = []
    for key, observations in grouped.items():
        segments = temporal_smooth(observations)
        for seg in segments:
            if duration(seg) < MIN_FACT_DURATION:
                continue
            fact = aggregate_observations_to_interval_fact(key, seg)
            interval_facts.append(fact)

    interval_facts = suppress_temporal_conflicts(interval_facts)
    return interval_facts

---
9.6 Segment-level event induction
输入
segment video
frame-level facts
interval facts
active entities
overlap facts from neighboring segments
输出
segment event graph
state transitions
event support mappings
事件归纳方式
组合三类方法：
Pattern-based induction:
  possession transfer
  open / close
  enter / exit
  break / repair
  pick up / put down
  give / receive

LLM / VLM structured induction:
  infer event type from facts
  identify participants
  infer preconditions and effects

Verifier-assisted induction:
  check whether event has enough support
  adjust time span
  remove unsupported events
Pseudocode
def build_segment_event_graph(segment, interval_facts, E_video):
    active_entities = get_active_entities(segment, E_video)
    candidate_events = []

    candidate_events += induce_events_by_patterns(interval_facts)
    candidate_events += induce_events_by_structured_llm(segment, interval_facts)

    verified_events = []
    for event in candidate_events:
        support = collect_supporting_facts(event, interval_facts)
        event.support_facts = support
        event.confidence = score_event_support(event, support)

        if event.confidence >= EVENT_KEEP_THRESHOLD:
            verified_events.append(event)

    state_transitions = infer_state_transitions(verified_events, interval_facts)
    temporal_edges = infer_intra_segment_temporal_edges(verified_events)

    return SegmentGraph(
        segment_id=segment.id,
        active_entities=active_entities,
        events=verified_events,
        state_transitions=state_transitions,
        temporal_edges=temporal_edges
    )

---
9.7 Global graph construction
输入
all segment graphs
entity lifecycle registry
state transition records
temporal order
optional transcript only as offline construction signal
输出
global event DAG
entity lifecycle graph
causal graph
narrative episode graph
Global graph 构建原则
1. 所有 segment event 升级为 global event node。
2. 根据 time_span 建 temporal edges。
3. 根据 shared entity / state variable 建 causal candidate edges。
4. 根据 entity lifecycle 建 state carry-over edges。
5. 根据 repeated participants / locations / goals 聚合 episode nodes。
6. 对 causal edges 执行 verification。
Pseudocode
def build_global_graph(segment_graphs, E_video):
    events = collect_all_events(segment_graphs)

    temporal_edges = build_temporal_edges(events)
    entity_edges = build_entity_continuity_edges(E_video, events)

    causal_candidates = propose_causal_edges(events, E_video)
    causal_edges = []
    for edge in causal_candidates:
        if verify_causal_edge(edge, events, E_video):
            causal_edges.append(edge)

    episodes = cluster_events_into_episodes(events, temporal_edges, causal_edges)

    return GlobalGraph(
        entities=E_video,
        events=events,
        temporal_edges=temporal_edges,
        causal_edges=causal_edges,
        entity_edges=entity_edges,
        episodes=episodes
    )

---
10. Quality Assurance: Generate–Verify–Repair Loop
10.1 核心思想
图数据库比 caption database 更需要质量控制。因为一旦错误边进入图，Agent 会把它当作结构化事实推理。
因此每层都需要：
generate
verify
repair
re-index

---
10.2 Verifier 类型
EntityVerifier
检查：
同一实体是否被错误分裂
不同实体是否被错误合并
tracklet 是否漂移
appearance 是否冲突
alias 是否冲突
修正：
merge_entity(e1, e2)
split_entity(e)
remove_tracklet(e, trk)
lower_entity_confidence(e)

---
RelationVerifier
检查：
relation 是否有视觉证据
relation 是否违反空间 / 接触约束
relation 是否与其他高置信 relation 冲突
修正：
delete_fact(f)
update_predicate(f, new_predicate)
lower_confidence(f)
adjust_time_span(f)

---
StateVerifier
检查：
同一实体同一时刻是否存在状态冲突
例如 open and closed, intact and broken, empty and full
修正：
resolve_state_conflict(entity, attribute, time_span)
keep higher-confidence state
trigger visual verification

---
TemporalVerifier
检查：
event time_span 是否重叠异常
before / after 是否与时间戳冲突
state transition 是否满足 before-state → after-state
修正：
adjust_event_boundary(event)
remove_invalid_temporal_edge(edge)
split_event(event)
merge_adjacent_events(event_a, event_b)

---
CausalVerifier
检查 causal edge 是否满足：
temporal precedence:
  cause earlier than effect

state change:
  effect has observable state transition

shared variable:
  cause and effect share entity / object / location / state

commonsense plausibility:
  relation does not violate physical / social commonsense
修正：
reject_causal_edge(edge)
downgrade_to_temporal_edge(edge)
mark_as_hypothesis(edge)

---
EvidenceVerifier
检查：
每个 event 是否有足够 frame facts 支持
每个 high-level answer 是否能回溯到 evidence pointer
每个 evidence pointer 是否仍可访问
修正：
mark_event_unsupported(event)
trigger VisualVerify
request additional frame-level extraction

---
10.3 Repair loop pseudocode
def verify_and_repair_graph(G_video):
    for _ in range(MAX_REPAIR_ROUNDS):
        issues = []

        issues += EntityVerifier.check(G_video)
        issues += RelationVerifier.check(G_video)
        issues += StateVerifier.check(G_video)
        issues += TemporalVerifier.check(G_video)
        issues += CausalVerifier.check(G_video)
        issues += EvidenceVerifier.check(G_video)

        if not issues:
            break

        for issue in issues:
            repair_action = propose_repair(issue)
            apply_repair(G_video, repair_action)

        reindex_graph(G_video)

    return G_video

---
11. Graph Store and Index Design
11.1 推荐存储结构
GraphStore:
  entities
  frame_facts
  interval_facts
  segment_events
  state_transitions
  global_events
  temporal_edges
  causal_edges
  support_mappings

EvidenceStore:
  frames
  bboxes
  masks
  crops
  tracklets
  visual embeddings

IndexStore:
  entity index
  relation index
  event index
  temporal index
  causal index
  evidence index
  optional graph-signature vector index

---
11.2 Graph-native indexes
Entity index
by entity_id
by category
by alias
by appearance descriptor
by tracklet
by time_span
by embedding
Relation index
by subject
by predicate
by object
by relation_class
by time_span
Event index
by event_type
by participant
by location
by precondition
by effect
by time_span
Temporal index
interval tree:
  before
  after
  overlap
  during
  first
  last
Causal index
event DAG adjacency:
  causes
  enables
  prevents
  triggers
  resolves
Evidence index
fact_id → evidence pointers
event_id → support facts → evidence pointers
answer_id → evidence chain

---
12. Graph-native Agent Toolset
12.1 工具设计原则
工具返回结构化 observation，而不是 caption。
返回格式统一为：
{
  "tool_name": "...",
  "query": {},
  "result": {
    "nodes": [],
    "edges": [],
    "facts": [],
    "events": [],
    "time_spans": [],
    "evidence_pointers": [],
    "confidence": 0.0,
    "missing_slots": [],
    "warnings": []
  }
}

---
12.2 QuestionDecompose
功能
将自然语言问题编译为图查询意图。
输入
{
  "question": "Who gets the lollipop last?"
}
输出
{
  "intent": "temporal_order_query",
  "entities": [
    {
      "mention": "lollipop",
      "type": "object",
      "slot": "target_object"
    }
  ],
  "relations": [
    {
      "predicate": "gets_or_holds",
      "subject": "?person",
      "object": "target_object"
    }
  ],
  "temporal_operator": "last",
  "missing_slots": ["final_holder"],
  "requires_evidence": true
}

---
12.3 EntityResolve
功能
解析实体引用。
输入
{
  "referring_expression": "the man with the mustache",
  "time_constraint": null
}
输出
{
  "candidates": [
    {
      "entity_id": "person_03",
      "score": 0.82,
      "aliases": ["man with mustache", "tall man"],
      "evidence": ["frame_01822:bbox_person_03"]
    },
    {
      "entity_id": "person_07",
      "score": 0.43,
      "aliases": ["old man"],
      "evidence": ["frame_02110:bbox_person_07"]
    }
  ]
}

---
12.4 RelationQuery
功能
查询 frame / moment-level 原子事实。
输入
{
  "subject": "person_03",
  "predicate": "holds",
  "object": "cup_01",
  "time_constraint": ["00:12:00", "00:12:15"]
}
输出
{
  "facts": [
    {
      "fact_id": "fact_000827",
      "subject": "person_03",
      "predicate": "holds",
      "object": "cup_01",
      "time_span": ["00:12:04.200", "00:12:09.600"],
      "confidence": 0.82,
      "evidence": ["frame_03612:bbox_person_03,bbox_cup_01"]
    }
  ]
}

---
12.5 AttributeQuery
功能
查询颜色、数量、姿态、状态等属性。
{
  "entity": "cup_01",
  "attribute": "color",
  "time_constraint": ["00:12:04", "00:12:09"]
}

---
12.6 EventSearch
功能
查询 segment-level event nodes。
输入
{
  "event_type": "break",
  "participants": {
    "patient": "mirror_01"
  },
  "time_constraint": null,
  "top_k": 5
}
输出
{
  "events": [
    {
      "event_id": "event_0044",
      "event_type": "break",
      "participants": {
        "agent": "villain_01",
        "patient": "mirror_01"
      },
      "time_span": ["00:09:46", "00:10:06"],
      "preconditions": ["mirror_01.integrity=intact"],
      "effects": ["mirror_01.integrity=broken"],
      "support_facts": ["fact_0203", "fact_0204", "fact_0205"],
      "confidence": 0.81
    }
  ]
}

---
12.7 StateChangeQuery
功能
查询状态变化。
输入
{
  "entity": "mirror_01",
  "attribute": "integrity",
  "from": "intact",
  "to": "broken",
  "time_constraint": null
}

---
12.8 TemporalOrderQuery
功能
处理 first / last / before / after / immediately after。
输入
{
  "facts_or_events": "holder(lollipop_01, ?entity)",
  "operator": "last",
  "time_constraint": "whole_video"
}
输出
{
  "result": {
    "entity": "zombie_02",
    "support_event": "event_0081",
    "support_fact": "fact_0912",
    "time_span": ["00:32:14", "00:32:28"],
    "confidence": 0.79
  }
}

---
12.9 CausalPathQuery
功能
处理 why / cause / consequence 问题。
输入
{
  "target_event": "event_0013",
  "relation_filter": ["causes", "enables", "triggers"],
  "max_hops": 3
}
输出
{
  "paths": [
    {
      "path_id": "path_0004",
      "nodes": ["event_0012", "event_0013"],
      "edges": ["gedge_0048"],
      "path_type": "enabling",
      "confidence": 0.68,
      "support": [
        "event_0012: boy puts food on table",
        "event_0013: bunnies gather on table"
      ]
    }
  ]
}

---
12.10 GraphExpand
功能
从 seed node 扩展局部子图。
输入
{
  "seed": "event_0044",
  "radius": 2,
  "filters": {
    "edge_types": ["before", "after", "causes", "state_transition"],
    "entity_types": ["person", "object"]
  }
}

---
12.11 EvidenceTrace
功能
追溯 fact / event / path 的视觉证据。
输入
{
  "target_id": "event_0044",
  "target_type": "event"
}
输出
{
  "event_id": "event_0044",
  "support_facts": ["fact_0203", "fact_0204", "fact_0205"],
  "visual_evidence": [
    {
      "frame_id": "frame_02910",
      "timestamp": "00:09:46.000",
      "bboxes": {
        "villain_01": [120, 80, 420, 640],
        "mirror_01": [460, 100, 760, 620]
      },
      "crop_ids": ["crop_02910_villain", "crop_02910_mirror"]
    }
  ],
  "support_score": 0.81
}

---
12.12 GraphVerify
功能
结构化校验 fact / event / path。
输入
{
  "target_id": "event_0044",
  "verification_type": "event_support"
}
输出
{
  "status": "confirm",
  "confidence": 0.78,
  "issues": [],
  "recommended_patch": null
}

---
12.13 VisualVerify
功能
只对指定 claim 和指定 evidence pointers 做视觉验证。
禁止开放式 VQA：
What is happening here?
允许结构化验证：
Does fact_0205 show fragment_group_01 moving upward after mirror_01 breaks?
输入
{
  "claim": {
    "subject": "fragment_group_01",
    "predicate": "moves_upward",
    "object": null,
    "time_span": ["00:09:50", "00:10:06"]
  },
  "evidence_pointers": [
    "frame_02915",
    "frame_02920",
    "frame_02925"
  ]
}
输出
{
  "status": "confirm",
  "confidence": 0.74,
  "verified_fact": {
    "subject": "fragment_group_01",
    "predicate": "moves_upward",
    "time_span": ["00:09:50", "00:10:06"]
  },
  "evidence_used": ["frame_02915", "frame_02920"]
}

---
12.14 AnswerWithEvidence
功能
生成最终答案和证据链。
输入
{
  "candidate_answer": "the mirror fragments fly into the sky",
  "supporting_subgraph": {
    "events": ["event_0044", "event_0045"],
    "facts": ["fact_0203", "fact_0205"],
    "evidence": ["frame_02915", "frame_02920"]
  }
}
输出
{
  "answer": "The mirror fragments fly into the sky.",
  "confidence": 0.76,
  "evidence_chain": {
    "global": ["event_0044 before event_0045"],
    "segment": [
      "event_0044: villain breaks mirror",
      "event_0045: fragments move upward"
    ],
    "frame_facts": [
      "fact_0203: mirror_01 becomes broken",
      "fact_0205: fragment_group_01 moves_upward"
    ],
    "visual_evidence": [
      "frame_02915",
      "frame_02920"
    ]
  }
}

---
13. Graph-driven Agent Loop
13.1 Agent state
{
  "question": "...",
  "parsed_intent": {},
  "resolved_entities": {},
  "candidate_events": [],
  "candidate_facts": [],
  "candidate_answers": [],
  "supporting_subgraph": {},
  "uncertain_items": [],
  "conflicts": [],
  "evidence_chain": {},
  "tool_history": []
}

---
13.2 High-level loop
Algorithm 2: SGD-Agent v2.0 Graph-driven Reasoning

Input:
  Q
  G_video
  Toolset T

Output:
  answer
  evidence_chain

1. Parse Q into graph query plan.
2. Initialize belief state H.
3. Resolve mentioned entities.
4. While not solved and step < max_steps:
     a. Select next graph-native tool.
     b. Execute tool.
     c. Update belief state.
     d. Check conflicts and uncertainty.
     e. If key fact / event / path uncertain, verify it.
     f. If enough support exists, answer with evidence.
5. Return final answer and evidence chain.

---
13.3 Pseudocode
def graph_agent_reason(Q, G_video, tools, max_steps=12):
    H = init_belief_state(Q)

    plan = tools.QuestionDecompose(Q)
    H.parsed_intent = plan

    for mention in plan.entities:
        candidates = tools.EntityResolve(
            referring_expression=mention["mention"],
            time_constraint=plan.get("time_constraint")
        )
        H.update_entities(mention, candidates)

    for step in range(max_steps):
        action = select_next_action(H)

        if action.name == "RelationQuery":
            obs = tools.RelationQuery(**action.params)

        elif action.name == "AttributeQuery":
            obs = tools.AttributeQuery(**action.params)

        elif action.name == "EventSearch":
            obs = tools.EventSearch(**action.params)

        elif action.name == "StateChangeQuery":
            obs = tools.StateChangeQuery(**action.params)

        elif action.name == "TemporalOrderQuery":
            obs = tools.TemporalOrderQuery(**action.params)

        elif action.name == "CausalPathQuery":
            obs = tools.CausalPathQuery(**action.params)

        elif action.name == "GraphExpand":
            obs = tools.GraphExpand(**action.params)

        elif action.name == "EvidenceTrace":
            obs = tools.EvidenceTrace(**action.params)

        elif action.name == "GraphVerify":
            obs = tools.GraphVerify(**action.params)

        elif action.name == "VisualVerify":
            obs = tools.VisualVerify(**action.params)

        elif action.name == "AnswerWithEvidence":
            return tools.AnswerWithEvidence(**action.params)

        H.update(action, obs)

        conflicts = detect_conflicts(H)
        if conflicts:
            H.conflicts.extend(conflicts)
            continue

        if has_sufficient_evidence(H):
            answer_params = build_answer_params(H)
            return tools.AnswerWithEvidence(**answer_params)

    return fallback_answer_with_uncertainty(H)

---
13.4 Action selection policy
Attribute question
EntityResolve
→ AttributeQuery / RelationQuery
→ EvidenceTrace
→ VisualVerify if low confidence
→ AnswerWithEvidence
Temporal order question
EntityResolve
→ RelationQuery / EventSearch
→ TemporalOrderQuery
→ EvidenceTrace
→ AnswerWithEvidence
Why / causal question
EventSearch(target event)
→ CausalPathQuery
→ GraphExpand(path)
→ EvidenceTrace
→ GraphVerify / VisualVerify
→ AnswerWithEvidence
State change question
EntityResolve
→ StateChangeQuery
→ EventSearch(trigger event)
→ RelationQuery(after event)
→ EvidenceTrace
→ AnswerWithEvidence
Counting question
EntityResolve(category / condition)
→ RelationQuery / AttributeQuery over time span
→ deduplicate by entity_id / tracklet
→ EvidenceTrace
→ AnswerWithEvidence

---
14. Query-type Examples
Version note:
  Attribute, relation, event, and temporal-order examples are MVP targets.
  Causal examples are v2.0 full targets unless explicitly labeled as MVP
  temporal/state-change hypotheses. In MVP, why-questions may retrieve temporal
  or state-change predecessors but must not claim verified causal edges.

14.1 “Who gets the lollipop last?”
QuestionDecompose:
  intent = temporal_order_query
  target = lollipop
  relation = holder/getter
  operator = last
Tool chain:
EntityResolve("lollipop")
RelationQuery(?entity, holds_or_gets, lollipop_01, whole_video)
TemporalOrderQuery(facts, operator="last")
EvidenceTrace(final_fact)
AnswerWithEvidence
Expected evidence chain:
answer: zombie_02
global: last holder(lollipop_01)=zombie_02
segment: event_081 zombie takes lollipop
facts: zombie_02 holds lollipop_01
evidence: frames / boxes

---
14.2 “What happens when the villain’s mirror breaks?”
Tool chain:
EntityResolve("villain")
EntityResolve("mirror")
EventSearch(event_type="break", patient=mirror_01)
StateChangeQuery(mirror_01, integrity, intact, broken)
GraphExpand(event_0044, after, radius=1)
RelationQuery(fragment_group_01, motion_direction, ?, after event_0044)
EvidenceTrace(fragment_motion_fact)
VisualVerify(fragment_motion_fact)
AnswerWithEvidence
Expected graph answer:
mirror_01 becomes broken
fragment_group_01 moves upward / flies into sky

---
14.3 “Why are the bunnies gathering on the table?”
Full v2.0 tool chain:
Tool chain:
EntityResolve("bunnies")
EntityResolve("table")
EventSearch(event_type="gather", participants=bunny_group_01, location=table_01)
CausalPathQuery(target_event=gather_event)
GraphExpand(causal_path)
EvidenceTrace(causal_path)
GraphVerify(causal_path)
AnswerWithEvidence
Expected reasoning:
event_12: boy puts / gives food on table
event_13: bunnies gather on table
edge: event_12 enables event_13
answer: because food is placed / handed out there

MVP fallback:
  EventSearch(gather_event)
  TemporalOrderQuery(predecessor events before gather_event)
  EvidenceTrace(predecessor + gather event)
  AnswerWithEvidence with:
    answer_type = temporal_explanation_hypothesis
    warning = causal graph is not enabled in MVP

---
15. Caption-free Inference Protocol
15.1 Strict mode
用于核心实验：
Agent cannot access:
  captions
  transcript text
  VLM-generated clip descriptions
  raw video frames except through VisualVerify with evidence pointers
  free-form summaries
  caption embeddings
  free-form event sentence embeddings
  GEI prompt timelines not materialized from G_global_v1
  direct frame_inspect_tool / clip_search_tool / global_browse_tool

Agent can access:
  graph facts
  event nodes
  entity registry
  temporal / causal edges
  evidence pointers
  verification results

Strict graph-only legal path:
  Question
  -> QuestionDecompose / EntityResolve
  -> RelationQuery / EventSearch / TemporalOrderQuery
  -> EvidenceTrace
  -> VisualVerify
  -> AnswerWithEvidence

Strict graph-only illegal path:
  Question
  -> arbitrary timestamp / arbitrary segment
  -> free-form frame or clip inspection
  -> answer

MVP note:
  `frame_inspect_tool` may exist only behind VisualVerifyAdapter.
  It is not an Agent-facing graph-native action.

---
15.2 Builder-assisted mode
用于离线构图：
Offline builder can use:
  captions
  transcripts
  VLM descriptions
  ASR
  OCR
  detector outputs
  tracker outputs
但所有文本信号必须被蒸馏成：
entities
attributes
relations
events
states
temporal edges
causal edges
evidence pointers
不能直接进入在线 Agent memory。

Recommended artifact split:
  construction_graph_raw.json:
    may contain caption_context, caption_snippets, ASR, VLM narration,
    builder debug summaries, and other construction-only signals.

  inference_graph_stripped.json:
    removes caption-like fields and exposes only graph facts, event nodes,
    entity records, structured signatures, support mappings, and evidence pointers.

Tool outputs, trace observations, and final evidence chains must use the stripped graph.

---
15.3 Diagnostic ablations
建议评估三种模式：
A. Caption Memory Agent:
   caption database + caption retrieval

B. SG-as-Index Agent:
   scene graph helps retrieval, captions remain main memory

C. Graph-native SGD-Agent v2.0:
   no caption retrieval, only graph facts + evidence pointers
核心 claim 应该是：
Removing caption memory does not collapse long-video VQA.
The agent can answer and trace evidence through graph facts only.

Additional leakage ablations:
  Naturalized graph observation:
    close to captions; analysis only, not main method.

  Tuple / ID-heavy graph observation:
    main method.

  Caption embedding:
    forbidden in strict setting; baseline / upper-bound only.

  Event natural-language sentence embedding:
    leakage-risk ablation only.

  Tuple-label / graph-signature embedding:
    allowed if derived from structured graph fields.

---
16. Evaluation Protocol
16.1 Task metrics
VQA accuracy:
  multiple-choice / open-ended answer correctness

Temporal grounding:
  event time span IoU
  evidence frame recall
  supporting moment localization

Evidence tracing:
  whether returned evidence supports answer
  fact-to-frame correctness
  answer-to-event correctness

Graph quality:
  entity precision / recall
  relation precision / recall
  event precision / recall
  causal edge plausibility
  state transition correctness

Agent efficiency:
  number of tool calls
  graph query latency
  visual verification calls
  tokens consumed

---
16.2 Evidence-specific metrics
Answer support rate
percentage of answers with at least one verified evidence chain
Faithful evidence rate
percentage of evidence chains where:
  answer ← event ← facts ← visual evidence
is valid
Unsupported answer rate
percentage of answers generated without sufficient graph support
Contradiction rate
percentage of answers whose support facts conflict with other high-confidence facts

---
16.3 Ablation studies
Remove Frame Graph:
  tests fine-grained details.

Remove Segment Event Graph:
  tests event/action understanding.

Remove Global Graph:
  tests long-range temporal / causal reasoning.

Remove Entity Lifecycle:
  tests cross-segment identity consistency.

Remove EvidenceTrace:
  tests evidence grounding.

Remove GraphVerify:
  tests robustness to noisy graph facts.

Allow captions at inference:
  measures performance upper bound and caption leakage effect.

No VisualVerify:
  measures graph-only reliability.

---
17. Development Roadmap
17.1 MVP version
目标：先证明 graph-native inference 可行。
实现：
0. Caption-free audit:
   strip construction-only text fields from inference artifacts.
   assert no caption tools, caption embeddings, free-form summaries, or caption-like
   payloads appear in Agent action space, tool outputs, traces, or final evidence.

1. Entity registry:
   entity_id
   category
   aliases
   time spans
   support_frame_ids
   confidence.
   MVP does not require bbox / tracklet / mask, but support frames are mandatory.

2. Frame facts:
   entity presence
   basic attributes
   spatial/contact/holding relations.
   facts must carry fact_id, time_span, confidence, and support_frame_ids.

3. Segment events:
   pattern-based event induction for common events:
     holds, gives, takes, opens, closes, breaks, enters, exits.
   free-form segment summary is forbidden in strict inference;
   use graph_signature / event_signature instead.

4. Global graph v1:
   temporal event DAG
   entity lifecycle
   first / last / before / after queries
   simple state carry-over:
     holder(object)=entity
     location(entity/object)=place
     visible / disappeared / reappeared
     last_known_state.

5. Tools:
   QuestionDecompose
   EntityResolve
   RelationQuery
   EventSearch
   TemporalOrderQuery
   EvidenceTrace
   VisualVerify
   AnswerWithEvidence.
   Existing frame/segment/inspect tools are internal adapters only.

6. Caption-free inference:
   Agent observation only graph JSON.
   Visual verification only through graph-derived evidence pointers or candidate intervals.
   Final answer must be structured AnswerWithEvidence JSON.

---
17.2 v2.0 full version
新增：
state transition graph
causal graph
GraphVerify
CausalPathQuery
GraphExpand
object permanence
mask-level evidence
bbox / crop / mask evidence
tracklet-level entity lifecycle
confidence-aware repair
global repair loop
materialized GEI from G_global_v1

---
17.3 v2.1 research version
新增：
learned graph builder
learned event induction
learned graph verifier
graph RL / preference optimization
subgraph planning policy
uncertainty-aware active visual verification

---
18. Recommended Module Interfaces
18.1 Python-style core classes
@dataclass
class EntityRecord:
    entity_id: str
    category: str
    aliases: list[str]
    appearance: dict
    tracklets: list[dict]
    states_over_time: list[dict]
    relations_over_time: list[dict]
    first_seen: str
    last_seen: str
    confidence: float


@dataclass
class GraphFact:
    fact_id: str
    fact_type: str
    subject: str
    predicate: str
    object: str | None
    time_span: tuple[str, str]
    evidence: list[str]
    confidence: float
    source: list[str]
    status: str


@dataclass
class EventNode:
    event_id: str
    event_type: str
    participants: dict
    time_span: tuple[str, str]
    preconditions: list[dict]
    core_facts: list[str]
    effects: list[dict]
    support_facts: list[str]
    evidence: list[str]
    confidence: float
    status: str


@dataclass
class EvidencePointer:
    evidence_id: str
    video_id: str
    timestamp: str
    frame_id: str
    entity_ids: list[str]
    bboxes: dict
    masks: dict
    crop_ids: list[str]
    source_fact_ids: list[str]

---
18.2 Tool interface
Existing implementation tools are low-level adapters. The Agent-facing API must
use the semantic names below. During MVP, GraphTools may wrap existing
`frame_graph_query_tool`, `segment_graph_query_tool`, GEI, and
`frame_inspect_tool`, but the wrapped outputs must satisfy caption-free and
evidence-chain constraints.

class GraphTools:
    def question_decompose(self, question: str) -> dict: ...
    def entity_resolve(self, referring_expression: str, time_constraint=None) -> dict: ...
    def relation_query(self, subject=None, predicate=None, object=None, time_constraint=None) -> dict: ...
    def attribute_query(self, entity: str, attribute: str, time_constraint=None) -> dict: ...
    def event_search(self, event_type=None, participants=None, time_constraint=None, top_k=5) -> dict: ...
    def state_change_query(self, entity: str, attribute: str, from_value=None, to_value=None) -> dict: ...
    def temporal_order_query(self, query: dict, operator: str) -> dict: ...
    def causal_path_query(self, target_event: str, relation_filter=None, max_hops=3) -> dict: ...
    def graph_expand(self, seed: str, radius: int, filters=None) -> dict: ...
    def evidence_trace(self, target_id: str, target_type: str) -> dict: ...
    def graph_verify(self, target_id: str, verification_type: str) -> dict: ...
    def visual_verify(self, claim: dict, evidence_pointers: list[str]) -> dict: ...
    def answer_with_evidence(self, candidate_answer: str, supporting_subgraph: dict) -> dict: ...

MVP adapter mapping:
  QuestionDecompose:
    compiles the question into a graph query plan and run-mode-safe tool route.

  EntityResolve:
    entity registry / frame graph / segment graph entity fields.

  RelationQuery:
    wraps frame_graph_query_tool and returns fact-like graph records.

  EventSearch:
    wraps segment_graph_query_tool and returns event-like graph records.

  TemporalOrderQuery:
    uses Stage3 v1 temporal DAG when available; otherwise GEI-MVP can propose
    candidate intervals as navigation only.

  EvidenceTrace:
    maps fact_id / event_id / path_id to support_frame_ids, timestamps, and
    later bbox/crop/mask evidence.

  VisualVerify:
    wraps frame_inspect_tool internally in MVP, bounded to graph-derived
    evidence pointers or candidate intervals.

  AnswerWithEvidence:
    replaces finish(answer) as the semantic final action. Benchmark runners may
    extract answer text / option letters, but traces must keep evidence JSON.

---
19. Failure Modes and Controls
19.1 Graph hallucination
问题：
VLM / LLM 生成不存在的实体或关系。
控制：
must bind to evidence pointer
low evidence support → low confidence
GraphVerify before answer
VisualVerify for critical facts

---
19.2 Entity drift
问题：
person_03 被错误匹配成 person_07。
控制：
tracklet confidence
appearance similarity threshold
alias consistency
manual split / auto split
EntityVerifier

---
19.3 Temporal fragmentation
问题：
一个连续事件被切成多个孤立片段。
控制：
overlap windows
event boundary smoothing
same event merge
state transition continuity

---
19.4 Over-verification
DVD 的分析表明，Agent 可能陷入反复 Frame Inspect 或 Clip Search 的低效循环；DVD 将其称为 Frame Inspect Trap 和 Clip Search Trap。 SGD-Agent v2.0 应避免开放式视觉问答式检查，而采用结构化 GraphVerify / VisualVerify，并设置：
max_verify_calls
max_same_tool_repeats
confidence convergence condition
answerability check

---
19.5 Premature answer
问题：
Agent 在证据不足时过早回答。
控制：
minimum evidence chain requirement
answer confidence threshold
unsupported answer penalty
mandatory EvidenceTrace before final answer

---
20. 最终方法主张
SGD-Agent v2.0 可以写成下面这段方法主张：
SGD-Agent v2.0 upgrades scene graphs from auxiliary indexes to native long-video memory. It represents videos as a cross-scale spatio-temporal scene hypergraph consisting of persistent entity lifecycles, frame-level atomic visual facts, segment-level event/state-transition graphs, and global entity-event-causal graphs. During inference, captions are removed from the agent’s accessible memory. The agent answers questions by executing graph-native operations over entities, relations, events, temporal orders, causal paths, and evidence pointers. Each answer is required to be supported by an explicit evidence chain from global reasoning paths to segment events, frame-level facts, and visual grounding evidence.
中文版本：
SGD-Agent v2.0 将场景图从辅助索引升级为长视频原生记忆。系统以跨尺度融合时空场景超图表示视频，包括全局实体生命周期、帧级原子视觉事实、片段级事件/状态转移图以及全局实体-事件-因果图。推理阶段，Agent 不再访问 caption 数据库，而是通过图原生工具查询实体、关系、事件、时间顺序、因果路径和视觉证据指针。最终答案必须由显式证据链支撑，从全局推理路径回溯到片段事件、帧级事实和具体视觉证据。

---
21. 可以直接放进开发任务列表的模块拆分
这些模块是 `dev-plan.md` 的唯一进度组织方式。

M0. Caption-free Audit
  input: inference artifacts, tool outputs, traces, benchmark results
  output: pass/fail audit report and stripped inference graph

M1. Temporal Unit Builder
  input: video
  output: frames, micro-clips, segment windows, shot boundaries

M2. Entity Registry / Entity Lifecycle MVP
  input: frames / detections
  output: E_video, aliases, time spans, support_frame_ids
  full output later: tracklets, masks, crops, visual embeddings

M3. Frame Fact Extractor
  input: frame, entity registry
  output: frame-level facts with support frames

M4. Temporal Fact Consolidator
  input: frame-level facts
  output: interval facts

M5. Segment Event Inducer
  input: interval facts, segment windows
  output: event nodes, state transitions, event_signature

M6. Global Graph v1 Builder
  input: segment events, entity lifecycle
  output: temporal graph, first_last_index, simple state carry-over,
  entity continuity graph
  full output later: causal graph, episodes, narrative graph

M7. Graph Verifier and Repairer
  input: full graph
  output: verified graph + repair logs
  MVP scope: evidence support checks and strict-mode stripping
  full scope: causal verifier and global repair loop

M8. Graph Store and Index Store
  input: verified graph
  output: queryable graph-native memory

M9. Graph Toolset Wrapper
  QuestionDecompose, EntityResolve, RelationQuery, EventSearch,
  TemporalOrderQuery, EvidenceTrace, VisualVerify, AnswerWithEvidence
  full later: StateChangeQuery, CausalPathQuery, GraphVerify, GraphExpand

M10. Graph Agent
  input: question, graph tools
  output: answer, evidence chain, trace

M11. Evaluation / Demo Harness
  answer accuracy
  evidence correctness
  temporal grounding
  graph quality
  caption-free ablation
  trace replay / graph visualization demo

---
22. 最小可运行 Demo
Input:
  10–30 min video
  20–50 questions

Offline:
  build entity registry
  extract frame facts
  induce basic events
  build temporal event graph

Online:
  disable captions
  answer using graph tools only

Required tools:
  EntityResolve
  RelationQuery
  EventSearch
  TemporalOrderQuery
  EvidenceTrace
  AnswerWithEvidence

Required question types:
  What is X holding?
  Who gets X last?
  What happens after event Y?
  What happens when object Z breaks?
  Where is object X after person Y leaves?
如果这个 demo 能跑通，SGD-Agent v2.0 的核心就成立：
scene graph is no longer an index;
it is the agent’s native video memory.
