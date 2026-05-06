## Abstract
Long-form video question answering requires locating verifiable evidence across extended timelines with sparse, multi-entity interactions. Existing caption-centric video agents convert long videos into searchable text databases, yet their retrieval still operates over open-ended semantic spaces, leading to query drift, redundant searches, and premature or failed trajectory termination. We propose SGDAgent, which converts open-ended caption search into graph-anchored candidate verification. Rather than replacing caption memory, SGDAgent augments it with a three-tier structured scene graph index — frame-level, segment-level, and global event-level — forming a hybrid memory that preserves open semantic coverage while introducing entity-, relation-, and temporally-grounded candidate proposal. Online reasoning follows a propose–verify–answer paradigm: graph queries first narrow the search to a small set of spatiotemporally consistent candidate windows; a local visual module then verifies these candidates on raw frames; and an uncertainty-aware controller governs when evidence is sufficient to commit an answer. Under strictly matched settings on LVBench (same backbone, caption source, and budget), the full SGDAgent configuration raises end-to-end accuracy from 54.8% to 73.0% and reduces the UNKNOWN rate from 16.5% to 1.6% compared with the caption-only baseline. Behavioral analysis further confirms that the gains stem not from the graph directly producing answers, but from restructuring the search paradigm itself — shifting from open retrieval to structured candidate verification with stronger local visual confirmation.

## 1. Intro 介绍

长视频问答的核心挑战并不只在于视频时长本身，更在于如何在长时序、多实体交互和稀疏证据分布下，稳定地找到足以支撑最终作答的可验证证据。随着多模态大模型上下文长度不断扩展，直接将更多帧或更多视觉 token 输入模型，确实在一定程度上缓解了长视频理解中的信息截断问题；但对于一小时级视频中的复杂问题，模型仍需同时处理跨时段事件关系、局部视觉细节以及问题相关证据的稀疏分布，因此仅依赖长上下文建模或视觉压缩，往往难以稳定回答多跳关系题、显式时序题和细粒度属性题。DVD 进一步将长视频理解建模为多粒度数据库上的 agentic search，说明“如何搜”与“在哪里搜”同样重要；VideoAgent 则强调交互式多轮搜证的重要性；VideoMind 进一步表明，视频推理要取得可靠结果，往往需要显式的 grounding 与 verification。
尽管如此，现有方法仍存在一个共同局限：候选证据空间通常仍然过于开放。以 caption-centric video agents 为例，系统虽然避免了直接在原始视频流上做全局建模，但其主搜索空间仍是开放式语义空间：agent 需要不断改写查询、反复召回相似描述，再尝试从这些文本片段中拼出答案。这样的流程对开放语义问题是有效的，但对依赖实体绑定、关系约束和时间一致性的复杂问题并不稳定，容易出现语义漂移、重复搜索、弱命中被误当作证据，以及最终轨迹无法闭合等现象。VideoAgent 通过多轮 frame retrieval 提升了搜证效率，但其核心证据组织仍以帧选择为中心；VideoMind 通过 planner–grounder–verifier–answerer 的角色链显式增强 temporal-grounded reasoning，但其重点更多落在统一 backbone 下的 learned grounding 与 verification，而非对视频记忆结构和检索空间本身进行重构。
本文提出 SGDAgent，其核心思想可以概括为：SGDAgent converts open-ended caption search into graph-anchored candidate verification. 我们并不试图用 scene graph 替代 caption memory，而是将长视频问答重新组织为一种分层证据搜索过程：caption memory 保留开放语义覆盖，scene graph 则把开放式语义搜索收缩为少量时空一致的候选区域，再由局部视觉模块对这些候选执行最终验证。为此，我们在原始 DVD 的基础上，将视频记忆从单一 caption database 扩展为由 caption memory、frame graph、segment graph 和 global event graph 组成的 structured hybrid memory；在此基础上，我们进一步将在线推理组织为 propose–verify–answer 的分层策略，其中 graph 负责候选提议，局部视觉核查负责证据验证，策略层控制负责稳定终止与答案收束。这样的设计并非简单地“给 DVD 加几个图工具”，而是将其从 caption-centric agent search 重构为一个以图为候选提议层、以视觉为验证层、以控制策略保证收束的分层检索系统。
这种重构带来两个直接收益。其一，scene graph 将实体、关系、时间范围和支持帧显式绑定，使得系统不再只能在“相似描述”之间游走，而是可以优先在结构化候选空间中定位问题相关区域。其二，候选先行、验证随后这一流程，使 frame_inspect_tool 不再承担对整段视频的盲目搜索，而是转化为对少量候选区域的局部验证器，从而显著降低 UNKNOWN、减少搜索轨迹中的重复循环，并提高已作答样本上的答案质量。实验表明，在严格 matched-setting comparison 下，SGDAgent 相比 caption-only DVD 显著提升端到端准确率，并明显降低 UNKNOWN rate；进一步的行为分析与案例分析表明，这种提升并不主要来自 scene graph 直接输出答案，而是来自搜索范式本身从 open-ended search 到 candidate verification 的转变。
本文的贡献主要体现在三个方面。
（1） 

（2）

（3）

## 2. 相关工作
### 2.1 Long-video understanding
长视频理解的主要困难在于：模型既要处理长时间跨度下的时空依赖，又要在稀疏且问题相关的证据分布中完成高效检索。现有方法大致可以分为两类。第一类侧重于扩展上下文容量或压缩视觉冗余，通过更长上下文窗口、token compression 或层次化视频表示来缓解长序列建模压力；这类方法能够提升输入覆盖范围，但在面对复杂问题时，仍然需要在压缩后的表示中可靠地找出真正相关的证据。第二类则将长视频理解视为检索或搜索问题，通过选择式采样、外部记忆或 agentic search 等方式，把问题相关的信息从长视频中逐步挖掘出来。DVD 就属于后者：它构建多粒度视频数据库，并让 reasoning LLM 在 Global Browse、Clip Search 和 Frame Inspect 三类工具之间自适应切换，从而实现 agentic search over a multi-granular database。
与这两类工作相比，本文关注的并不是单纯“让模型看到更多内容”，也不是只在 caption memory 上做更强的 query reformulation，而是进一步追问：候选证据空间本身能否被重构得更适合长视频问答？ 我们的答案是以结构化图记忆扩展 caption memory，并将长视频问答中的主要检索步骤改写为基于实体、关系和时间约束的候选提议过程。因而，本文与传统长上下文建模方法不同，也不同于仅在 caption 或帧级检索空间中进行搜索的 long-video agents。


### 2.2 Agentic video reasoning and tool use
随着大语言模型在推理、规划和工具使用方面能力增强，越来越多工作开始将长视频理解建模为 agentic reasoning 问题。VideoAgent 将视频理解过程表述为一系列 state、action 和 observation，并用 LLM 控制多轮 frame retrieval、caption 更新和 self-reflection；其关键主张是，相比直接处理长视觉上下文，交互式搜索与迭代式信息聚合更接近人类理解长视频的方式。DVD 则进一步将 agentic search 建立在多粒度视频数据库之上，强调 LLM 可以根据当前 observation state 自适应调用不同粒度的搜索工具，而不是遵循预定义的固定 workflow。
本文与这些 agent-based 方法一脉相承，但着力点不同。我们并不满足于“让 agent 更灵活地搜”，而是进一步修改了 agent 所作用的记忆结构、动作空间和证据闭合方式。具体地，我们将原本以 caption 检索为主的记忆扩展为 caption + graph 的混合记忆，并新增事件级、片段级和帧级图查询接口，使 agent 的动作空间从“caption search + local inspection”扩展为“graph proposal + visual verification + controlled fallback”的级联流程。换言之，本文的贡献不是单纯的工具重排，而是将 video agent 的核心搜索空间从 open-ended caption space 改造为 graph-anchored candidate space。
### 2.3 Temporal-grounded reasoning and verification
另一条与本文密切相关的路线，是显式强调 temporal grounding 与 verification 的视频推理方法。VideoMind 认为，视频推理不同于静态图像上的纯文本 CoT；有效的视频 reasoning 需要像人类一样先定位相关 moment，再验证边界和语义是否正确，最后给出答案。为此，VideoMind 设计了 Planner、Grounder、Verifier 和 Answerer 四种角色，并通过 Chain-of-LoRA 机制在统一 backbone 上实现角色切换；其实验也表明，Grounder 和 Verifier 对于提升 temporal reasoning 的可靠性至关重要。
本文与 VideoMind 的共同点在于，都不把长视频问答视为“一次性看完再答”的问题，而是强调 候选定位—局部验证—最终作答 这一分阶段证据闭合过程。不同之处在于，VideoMind 的核心是学习式 temporal grounding backbone 与 role-based reasoning pipeline，而本文的重点是对视频记忆与检索空间进行结构化重构：我们并不训练一个专门的 Grounder 去直接预测 moment，而是利用多层 scene graph 在事件级、片段级和帧级上逐步提出候选，再把最终确认交给局部视觉核查。因而，本文可以被看作是在 long-video agent 框架中引入显式的 graph-guided proposal layer，使 grounding 不再仅由 learned module 隐式承担，而是由结构化记忆与在线策略共同完成。
### 2.4 Our position
综合来看，现有工作分别从长上下文建模、交互式 agent 搜索以及 temporal-grounded reasoning 三个方向推进了长视频理解。然而，这三条路线之间仍存在一个尚未充分解决的空白：如何让 agent 在开放语义覆盖与结构化候选定位之间取得平衡，并把这种平衡转化为更稳定的证据闭合过程。 SGDAgent 的定位正是在这一空白上：它保留 caption memory 对开放语义的覆盖能力，同时引入 frame / segment / global event graph 作为结构化候选提议层，再通过局部视觉验证和不确定性感知控制将弱命中、错误终止和重复搜索降到更低。也正因此，本文既不同于纯 caption-centric 的 video agents，也不同于纯 grounding-centric 的统一大模型，而是处于两者之间的 graph-guided hybrid evidence search 框架。


## 3. 方法 


## 3.1 总览

原始 DVD 的优势在于把长视频问答转化为数据库上的 agentic search，但其主搜索空间仍然是开放式 caption 语义空间。对于多实体关系题、时序题和细粒度属性题，agent 往往需要不断改写查询、重复召回相似描述，再靠局部观察确认，因而容易出现语义漂移、重复搜索和轨迹中断。

SGDAgent 的核心思想是：**用 scene graph 将 open-ended caption search 改写为 graph-anchored candidate verification。** 我们不移除原始 DVD 的 caption memory，而是在其上增补分层结构化图记忆，使在线搜索先在事件级或片段级图上提出少量时空一致的候选，再对这些候选做局部视觉验证。为此，我们将视频 $V$ 的检索记忆统一写为

$$
M(V)=\{C(V),G^f,G^s,G^g\},
$$

其中 $C(V)$ 为 caption memory，$G^f$ 为帧级图序列，$G^s$ 为片段级图序列，$G^g$ 为由 Stage2 片段进一步抽象得到的全局事件图。完整 SGDAgent 使用三层图记忆；在消融实验中，我们通过关闭上层图来得到 `stage1`、`stage1+2` 和 `stage1+2+3` 等不同变体。

给定问题 $q$，SGDAgent 的在线推理被压缩为一个统一的 propose-verify-answer 链：

$$
\mathcal{W}_q=\mathrm{Propose}(M(V),q),\qquad
Z_q=\mathrm{Verify}(\mathcal{W}_q,q),\qquad
\hat a=\mathrm{Answer}(q,Z_q).
$$

这里，$\mathcal{W}_q$ 表示候选时间窗集合，$Z_q$ 表示通过验证的证据集合，$\hat a$ 表示最终答案。在线阶段沿用 DVD 的 observe-reason-act 循环，但动作空间被扩展为事件级图查询、片段级图查询、帧级图查询、caption 查询、局部视觉核查和最终作答这几类操作。具体而言，`global_graph_query_tool` 负责在 $G^g$ 上返回事件级候选、event chain 和支持片段，`segment_graph_query_tool` 负责在 $G^s$ 上提出和重排候选时间窗，`frame_graph_query_tool` 负责在 $G^f$ 上补充对象级锚点，`clip_search_tool` 与 `global_browse_tool` 作为 caption backoff，`frame_inspect_tool` 则对候选区域做最终视觉验证。换言之，SGDAgent 的在线动作空间可以概括为 `T={GG, SG, FG, CS, GB, FI, ANSWER}`。

**Algorithm 1: Graph-guided Hybrid Search**

```text
Input: memory M(V)={C(V), G^f, G^s, G^g}, question q
Output: answer \hat a

1: parse q into lightweight constraints phi(q)
2: if q contains long-range temporal or event-summary cues then
3:     propose event candidates on G^g and descend to support segments on G^s
4: else
5:     propose candidate windows on G^s
6: end if
7: repeat
8:     if object-level anchoring is needed then
9:         refine top candidates on G^f
10:    end if
11:    Z_q <- verify candidate windows with frame_inspect_tool
12:    if evidence is sufficient then
13:        return Answer(q, Z_q)
14:    else if evidence is weak then
15:        backoff to clip_search_tool or global_browse_tool
16:    end if
17: until no new evidence or budget exhausted
18: return UNKNOWN
```

## 3.2 混合记忆构建

### 3.2.1 记忆表示

SGDAgent 的视频级记忆由 caption memory 和三层图索引组成：

$$
M(V)=\{C(V),G^f,G^s,G^g\},\qquad
G^f=\{G_i^f\}_{i=1}^{N},\qquad
G^s=\{G_j^s\}_{j=1}^{M},\qquad
G^g=\{G_k^g\}_{k=1}^{L}.
$$

$C(V)$ 存储与视频时间轴对齐的 caption 片段，用于保留开放语义描述和 caption-based fallback；$G^f$、$G^s$ 与 $G^g$ 共享同一视频级实体注册表 $E^{\text{video}}$。其中，$G^f$ 保留关键帧上的局部实体关系，$G^s$ 组织短时片段内的时序交互，$G^g$ 则在更长时间尺度上组织事件节点与事件边。对应到系统实现中，Stage1 指帧级图 $G^f$，Stage2 指片段级图 $G^s$，Stage3 指全局事件图 $G^g$。

### 3.2.2 帧级图

对于第 $i$ 个关键帧，帧级图定义为

$$
G_i^f=(E_i^f,R_i^f,t_i,c_i),
$$

其中 $E_i^f\subseteq E^{\text{video}}$ 为该帧中的实体集合，$R_i^f$ 为实体关系集合，$t_i$ 为时间戳，$c_i$ 为与该时间点对齐的 caption context。每条关系边写为三元组 $(s,p,o)$，其中 $s,o\in E_i^f$，$p$ 为关系谓词。帧级图只回答一个具体问题：当片段级候选已经给出局部时间范围后，系统如何在该范围内获得对象级锚点。因而，$G^f$ 主要服务于颜色、数量、相对方位、局部动作主体等细粒度定位任务，而不是承担全视频级粗搜索。

### 3.2.3 片段级图

对于第 $j$ 个时间片段，片段级图定义为

$$
G_j^s=(E_j^s,R_j^s,[u_j,v_j],F_j,s_j),
$$

其中 $E_j^s$ 为活跃实体集合，$R_j^s$ 为片段内的时序关系集合，$[u_j,v_j]$ 为片段时间范围，$F_j$ 为支撑帧集合，$s_j$ 为该片段的简洁文本摘要。片段级关系边写为

$$
r=(s,p,o,t_{\text{start}},t_{\text{end}},\mathcal{F}),
$$

其中 $\mathcal{F}$ 是支持该关系的帧 ID 集合。与帧级图相比，片段级图显式绑定了关系参与者、持续时间和支持证据，因此它是 SGDAgent 的主候选提议层。当前实现中，每个片段还维护可选的 segment embedding，用于图检索失败时的软匹配回退。

### 3.2.4 全局事件图

Stage3 在 Stage2 之上进一步构造全局事件图。对第 $k$ 个事件节点，我们定义

$$
G_k^g=(P_k^g,R_k^g,[a_k,b_k],S_k,F_k,\sigma_k),
$$

其中 $P_k^g$ 为参与实体集合，$R_k^g$ 为事件的显著关系集合，$[a_k,b_k]$ 为事件时间范围，$S_k$ 为支持该事件的片段 ID 集合，$F_k$ 为支持帧集合，$\sigma_k$ 为事件摘要。当前 v2 版本中，Stage3 同时支持 `unit event` 与 `group event` 两类事件节点：前者对应较紧凑的局部事件单元，后者用于组织由多个时间单元组成的更长事件段。

除事件节点外，$G^g$ 还显式维护事件边，用于编码 `before`、`overlaps`、`entity-carryover`、`same-thread` 和 `parent-child` 等关系。因而，$G^g$ 不只是一个更粗粒度的时间索引，而是一个能够把跨片段事件链、事件线程延续和事件层级结构统一起来的事件级检索图。

## 3.3 离线图构建

SGDAgent 采用离线构建、在线查询的范式。离线阶段依次构建 Stage1、Stage2 和 Stage3 图索引。这样，在线搜索无需始终在原始视频流上做全局无约束探索，而是可以在不同粒度的结构化记忆之间做级联检索。

### 3.3.1 关键帧提取与实体对齐

我们首先从原始视频中抽取关键帧。当前实现采用 2 FPS 的默认视频采样率，并每隔 5 秒选择 1 个关键帧，总数最多保留 120 帧。对每个关键帧，系统调用多模态模型生成紧凑的帧级 scene graph，并将解析结果写入帧级图 $G_i^f$。

为了在跨帧范围内保持实体一致性，我们维护视频级实体注册表 $E^{\text{video}}$，并对当前帧实体 $e$ 与历史实体 $\tilde e$ 计算对齐分数

$$
R_{\text{align}}(e,\tilde e)=
0.7\,\mathrm{Sim}_{\text{name}}+
0.2\,\mathrm{Sim}_{\text{desc}}+
0.1\,\mathbf{1}[\mathrm{type}(e)=\mathrm{type}(\tilde e)].
$$

当 $R_{\text{align}}(e,\tilde e)\ge\tau$ 时，将 $e$ 合并到已有实体 ID；否则创建新实体。当前实现中 $\tau=0.78$。若某帧的 VLM 解析为空，则从邻近 caption 中提取粗粒度实体作为 caption fallback，以避免长视频上出现大段空图区域。

### 3.3.2 由帧图到片段图的归并

在得到帧级图序列后，我们沿时间轴构造重叠滑动窗口，并将窗口内的帧级事实归并为片段级图。当前默认窗口长度为 30 秒，重叠长度为 5 秒。对每个窗口 $[u_j,v_j]$，系统统计窗口内活跃实体，并将重复出现的帧级关系 $(s,p,o)$ 聚合为片段级时序关系。关系时间边界由

$$
t_{\text{start}}=\min_i t_i,\qquad
t_{\text{end}}=\max_i t_i
$$

确定，同时保留 `support_frame_ids`、片段摘要和相邻窗口之间的 overlap links。若启用语义回退，系统进一步对片段摘要计算 embedding，用于后续的 soft matching。

### 3.3.3 全局事件图构建

Stage3 以 Stage2 片段图为输入，将时间上相近且语义上连续的片段聚合为事件簇。当前 v2 实现同时考虑时间间隔、共享实体数量、关系 Jaccard 相似度和谓词 Jaccard 相似度来决定是否合并相邻片段；对跨度过长或内部连续性不足的簇，系统进一步执行 temporal-unit split，将其细化为多个 unit events。

在得到 unit events 后，系统会在满足最小单元数约束时生成 group events，并为 unit/group 之间建立 `parent-child` 边；同时，Stage3 还会在事件层上补充 `before`、`overlaps`、`entity-carryover` 和 `same-thread` 等事件边。最终得到的 $G^g$ 中，每个事件节点都显式保留 `participant_entity_ids`、`salient_relations`、`support_segment_ids`、`support_frame_ids`、`confidence` 和 `uncertainty` 等字段，从而为后续事件级检索、事件链构造和向下级联到 Stage2 的细化提供直接支持。

## 3.4 图引导的候选提议

在线推理的第一步是从问题中提取轻量结构约束，并在图记忆上提出少量候选时间窗。这个阶段只回答两个问题：系统从问题中读出什么约束，以及它如何据此产生候选。

### 3.4.1 问题到轻量结构约束的解析

给定问题 $q$，系统将其映射为轻量结构约束

$$
\phi(q)=(\mathcal{E},\mathcal{R},\mathcal{T},\mathcal{O}),
$$

其中 $\mathcal{E}$ 表示实体约束，$\mathcal{R}$ 表示关系或动作约束，$\mathcal{T}$ 表示时间约束，$\mathcal{O}$ 表示是否需要局部视觉核查。$\mathcal{T}$ 既可以来自显式时间跨度，也可以来自 “first / before / after / then / beginning / end” 等自然语言时间锚；$\mathcal{O}$ 用于标记颜色、数量、服饰、文本、精确身份等细粒度观测型问题，因为这类问题的图命中默认只能作为定位线索，不能直接当作终局证据。

### 3.4.2 候选扩展与排序

候选提议采用“事件级粗提议 + 片段级细化”的级联方式：

$$
\mathcal{W}_q=\mathrm{Expand}(G^g,G^s,\phi(q)).
$$

当问题包含长程时间约束、事件摘要需求、`before/after` 结构或多事件链式推理线索时，系统首先调用 `global_graph_query_tool` 在 $G^g$ 上检索事件节点，并返回 `matched_events`、`event_chain`、`candidate_support_segments` 与 `controller_signals`。随后，Stage3 会把候选事件的支持片段级联到 $G^s$，得到更窄的 `refined_segments` 和 `refined_primary_span`；在需要对象级锚定时，再由 `frame_graph_query_tool` 在 $G^f$ 上继续细化。对于本身就是局部关系或短程时序的问题，系统也可以直接从 $G^s$ 开始提议候选时间窗。

对任一候选 $w$，我们采用如下综合评分：

$$
S(w\mid q)=S_{\text{lex}}(w,q)+S_{\text{soft}}(w,q)+S_{\text{temp}}(w,q),
$$

其中 $S_{\text{lex}}$ 衡量查询与实体名、关系谓词、事件/片段摘要和 caption context 的直接词项匹配，$S_{\text{soft}}$ 衡量语义近邻程度，$S_{\text{temp}}$ 衡量候选与时间跨度或时间锚的一致性。对于 Stage3，事件级排序还会进一步结合时间锚、event chain 和 anchor-first rerank；对于 Stage2，则使用 support segments 和 temporal triples 做局部细化。当前实现默认返回 5 个事件级候选、6 个片段级候选和 8 个帧级候选；当直接 lexical hit 不足时，再启用 soft matching 回退。

## 3.5 局部验证与答案选择

图模块只负责提出候选，不直接给出最终答案。对候选时间窗集合 $\mathcal{W}_q$，系统使用 `frame_inspect_tool` 在原始视频帧上做局部视觉验证，并将通过验证的候选写入证据集合

$$
Z_q=\{w\in\mathcal{W}_q\mid \mathrm{Verify}(w,q)=1\}.
$$

若候选时间窗较长，系统会在该范围内均匀采样帧；当前实现中单次 `frame_inspect_tool` 最多使用 50 帧。若多个候选都通过验证，系统进一步结合时间一致性、最近一次局部核查结果以及候选答案间的区分度做最终选择，并输出

$$
\hat a=\mathrm{Answer}(q,Z_q).
$$

当没有任何候选获得足够支持时，系统可以继续搜索，也可以在预算耗尽后返回 `UNKNOWN`。

## 3.6 不确定性感知的搜索控制

SGDAgent 的控制层承担两个相互关联的职责：一是在搜索开始时根据问题特征定制检索策略（前向路由），二是在搜索过程中根据累积证据的质量决定何时可以安全地给出最终答案（终局门控）。两者共同保证 agent 不会因图命中过弱而过早终止，也不会在细粒度题上依赖单次不可靠的局部观察。

### 3.6.1 问题类型感知路由

在推理开始前，系统对问题 $q$ 执行轻量分类，将其映射为一组策略标志：

$$
\pi(q) = \bigl(\text{requires\_targeted\_fi},\ \text{requires\_repeated\_fi},\ \text{requires\_time\_valid\_terminal},\ \tau_\text{label}\bigr),
$$

其中 $\tau_\text{label} \in \{\text{count, color, identity, appearance, on-screen text, time-localized event}\}$ 标记需要视觉确认的目标属性类型。分类基于问题文本中的词法模式：涉及数量（”how many”、”number of”）、颜色（”what color”）、服饰外貌（”wearing”、”outfit”）、身份（”who is”）或屏显文字（”text”、”sign”、”license plate”）的问题，被识别为需要有针对性的局部视觉核查（$\text{requires\_targeted\_fi} = \text{true}$）；对于歧义度较高的属性（颜色、数量、身份），系统进一步要求重复核查一致性验证（$\text{requires\_repeated\_fi} = \text{true}$）。若问题含有显式时间参考（如 “after X”、”at timestamp Y”），则终局证据必须落在该时间范围内（$\text{requires\_time\_valid\_terminal} = \text{true}$）。

这一路由策略与图查询的局限性直接对应：帧级图和片段级图擅长提议”在哪里看”的候选区域，但对”看到什么具体细节”仍依赖视觉核查。路由层通过 $\pi(q)$ 明确标记哪些问题属于后者，从而阻止 agent 把图命中直接当作细粒度题的终局答案。这也解释了为什么消融实验中 Stage3 在不加路由的“裸”配置下收益有限，事件级候选提议带来的搜索路径压缩有时会跳过本应进行的视觉验证步骤。

### 3.6.2 终局证据门控

仅有候选窗口和粗定位不足以保证答案可靠。为此，系统在 finish 动作触发前实施分级证据门控：

**Targeted FI 约束**：若 $\text{requires\_targeted\_fi} = \text{true}$，finish 触发条件要求至少存在一条合格的 `frame_inspect_tool` 记录，即满足（1）该次核查的目标属性类型与 $\tau_\text{label}$ 一致；（2）核查时间窗口足够窄（默认上限 18 秒）。该约束阻止系统在图命中候选片段之后直接作答，而不做任何视觉确认。

**重复核查一致性约束**：若 $\text{requires\_repeated\_fi} = \text{true}$，finish 触发额外要求至少两条合格 FI 记录满足：（1）两次核查时间窗口重叠比例 $\geq 0.5$；（2）两次输出签名一致（对数字类答案比对数值，对颜色和身份类答案提取关键词后做 Jaccard 相似度检验，阈值 0.55）。这一设计源于以下观察：VLM 在单次低分辨率抽帧下容易对颜色、数量等细节给出措辞不稳定的描述；通过要求多次局部核查收敛，系统可在大多数情况下过滤偶发错误。§4.5 案例 3 直接展示了这一机制的作用。

**时间有效终局约束**：若 $\text{requires\_time\_valid\_terminal} = \text{true}$，终局证据必须来自显式时间参考范围内的核查记录，防止 agent 用时间范围外的视觉观察错误作答。

当上述约束未满足时，系统拒绝当前 finish 请求并向 agent 发送针对性指导；最多拒绝 $K_{\max}=3$ 次，超过上限后改用强制收束机制从当前消息历史中提取最可靠的候选选项。

### 3.6.3 图命中不确定性信号

在图查询阶段，`segment_graph_query_tool` 在返回候选片段的同时生成 `controller_signals` 字段，量化当前命中的可信程度并指导后续动作选择。这些信号由 top-1 候选的 lexical score、关系命中情况、时间窗宽度、是否依赖回退召回以及查询文本类型共同决定，被收缩为以下三类进入在线控制逻辑：

- **`localization-only`**：触发条件为（1）问题本身要求细粒度视觉细节（词法模式匹配，如颜色、数量、身份、屏显文字），或（2）候选时间窗口过宽（宽度 $\geq 25$ 秒）。此时图结果只被视为定位线索，系统必须调用 `frame_inspect_tool` 做局部视觉确认，不允许直接作答。

- **`weak-evidence`**：触发条件为词汇相似度过低（$< 0.15$）、关系命中为空、时间一致性差、top-1 与次优候选分差过小（$< 0.12$），或检索回落至嵌入向量软匹配（`fallback_only`）。此时系统触发 graph backoff，转而调用 `clip_search_tool` 或 `global_browse_tool`，并在后续若干步内禁止继续调用图查询工具。

- **`stagnation`**：若在滑动窗口（默认 3 步）内，每一步的 FI、clip_search 和图查询均未产生新的合格证据，系统判定当前轨迹停滞，同时激活 graph backoff 并向 agent 发送停滞指导，阻止其沿同一路径反复搜索。

这三类信号与 3.6.1–3.6.2 的路由与门控机制共同构成 SGDAgent 的控制层：它们不改变 propose–verify–answer 的主流程，而是决定搜索何时继续提议、何时切换工具、何时满足终止条件。

## 3.7 实现鲁棒性

以下几项机制不属于主方法贡献，但对 SGDAgent 在实际推理中稳定运行是必要的，此处简要列出以备复现参考。

**工具重试**：对涉及外部 VLM 调用的工具，系统在检测到超时、限速或服务不可用等错误时自动重试，采用指数退避策略（最多 3 次，基础延迟 2 秒），防止偶发 API 错误导致推理轨迹中断。

**弱命中拒绝**：图查询结果在返回给 agent 前，由内部分析模块检测多类拒绝条件（词汇分过低、时间范围未命中、仅含泛化实体且无关系证据等）。命中被拒绝时，系统直接触发 graph backoff，而非将低质量结果传递给 agent，避免 agent 将其误当作有效候选。

**强制选项收束**：对于四选一问答任务，系统在 finish 阶段严格校验答案格式，要求输出恰好为一个大写选项字母（A/B/C/D）。若 agent 输出冗长文本或未明确选项，系统通过分级强制机制（软提示 → 硬提示 → 从消息历史中恢复）最终提取合法选项，防止推理结果正确但因格式问题被计为 UNKNOWN。

**轨迹进度追踪**：每个推理步骤的工具调用、LLM 调用和策略决策均被写入结构化轨迹日志（JSONL 格式），系统同时在内存中维护滑动窗口进度向量（记录每步是否产生合格新证据），供停滞检测和 §4 中搜索行为分析使用。

## 3.8 方法小结

综上，SGDAgent 将长视频问答统一为一个混合记忆上的结构化证据搜索问题，并提出了由层次化场景图索引、自下而上的图构建、图引导候选提议、局部视觉验证以及不确定性感知控制组成的完整框架。与仅依赖 caption 库进行开放式语义检索的方案不同，SGDAgent 在保留 caption 检索灵活性的同时，引入 scene graph 作为实体、关系和时间约束下的结构化定位层，从而显著减小搜索歧义并提升证据可验证性。当前系统的主要性能提升来自帧级图与片段级图的稳定配合，而全局事件层则为进一步扩展长程事件建模提供了自然方向。




## 4. 实验

### 4.1 实验设置与评测协议

#### 4.1.1 数据集与任务设定

除特别说明外，本文的主 benchmark 为 **LVBench full1549**。该 benchmark 共包含 1549 个问题，任务形式为带显式候选答案的四选一视频问答。也就是说，系统输入包括长视频、自然语言问题以及候选答案集合，输出为候选集合中的一个选项；若系统未能返回合法选项，则记为 `UNKNOWN`。

为了确保 matched-setting comparison 的公平性，所有本文内部比较方法都共享同一视频数据库与同一 caption source。换言之，baseline DVD 与 SGDAgent 的差异不在于底层视频资产、caption 库或候选答案信息不同，而在于是否引入 scene graph 作为结构化候选发现层，以及是否启用相应的控制策略。除 full1549 主 benchmark 外，我们还使用 bad38 诊断切分做轨迹级分析，包括 UNKNOWN 行为、工具调用模式和 case study。

#### 4.1.2 比较方法

本文中的比较对象分为两类。

第一类是 **matched-setting methods**，即在同一代码框架、相同主干模型和相同预算约束下的严格对比，主要包括：

- `DVD caption-only`：仅使用 caption database 和原始工具链的 baseline。
- `SG only`：移除 caption 检索主导路径，仅依赖 scene graph 检索与局部验证。
- `SG stage1`：只启用帧级图。
- `SG stage1+2`：启用帧级图与片段级图。
- `SG stage1+2+3`：进一步加入全局图导航层。
- 若干控制策略变体：例如题型路由、额外回答轮次等。

第二类是 **reference numbers from prior work / external systems**，例如 VideoMind、MRVideo、Seed1.5-VL-Thinking、Gemini 以及原始 DeepVideoDiscovery 的公开结果。这些数字仅用于提供外部参考位置，而不用于支撑本文最核心的公平性结论。本文的主 claim 仅建立在 matched-setting comparison 之上。

#### 4.1.3 实现细节

除特别说明外，本文采用如下稳定配置。

| 组件 | 设置 |
|---|---|
| 主 benchmark | LVBench full1549 |
| 任务形式 | 四选一多选题，候选答案显式给定 |
| caption source | 所有 matched-setting 方法共享同一 caption 数据库；若本地重新生成 caption，则使用 `gpt-4.1-mini` |
| reasoning backbone | `o4-mini` |
| frame inspect VLM | `gpt-4.1-mini` |
| Stage-1 graph builder VLM | `gpt-4.1` |
| 视频抽帧帧率 | 2 FPS |
| 关键帧采样 | 每 5 秒采样 1 个关键帧 |
| 最多关键帧数 | 120 |
| Stage-2 segment window | 30 秒 |
| Stage-2 overlap | 5 秒 |
| top-k event candidates | 5 |
| Stage-3 merge max gap | 12 秒 |
| Stage-3 relation/predicate Jaccard threshold | 0.18 / 0.22 |
| Stage-3 thread max gap | 90 秒 |
| entity alignment threshold | 0.78 |
| top-k segment candidates | 6 |
| top-k frame candidates | 8 |
| max triples returned | 24 |
| frame inspect max frames | 50 |
| max reasoning steps | 15 |
| decoding / temperature | temperature = 0 |

#### 4.1.4 评测指标

我们至少报告以下四个主指标：

$$
\mathrm{e2e\_acc}=\frac{\#\mathrm{correct}}{N},
\qquad
\mathrm{answered\_acc}=\frac{\#\mathrm{correct}}{\#\mathrm{correct}+\#\mathrm{wrong}},
$$

$$
\mathrm{answer\ rate}=\frac{\#\mathrm{correct}+\#\mathrm{wrong}}{N},
\qquad
\mathrm{UNKNOWN\ rate}=\frac{\#\mathrm{unknown}}{N}.
$$

其中，`UNKNOWN` 指系统最终未返回合法候选选项的情形，包括空输出、运行时失败、轨迹中断或未能形成终局作答。`answered_acc` 只在已作答样本上计算，因此它反映的是“答了之后的质量”；`answer rate` 与 `UNKNOWN rate` 则反映系统能否形成稳定终局决策。

考虑到本文的核心 claim 不只是“答对更多”，还包括“改变搜索动力学”，我们进一步在有完整 trace 的切分上报告行为指标，包括：平均工具调用数、平均 LLM 调用数、平均 `frame_inspect_tool` 次数，以及不同图工具的调用频率。这些指标用于刻画结构化候选发现是否改变了搜索深度、验证强度和终止方式。

#### 4.1.5 公平性与预算控制

为了确保论文主结论可信，我们在 matched-setting comparison 中统一遵循以下原则：

- baseline DVD 与 SGDAgent 共享相同的 reasoning backbone。
- baseline DVD 与 SGDAgent 共享相同的 `frame_inspect_tool` VLM。
- 所有方法共享相同的 caption source、相同视频数据库以及相同候选答案集合。
- 所有方法使用相同的最大推理步数上限（20）和相同的单次 frame inspect 帧预算（最多 50 帧）。
- 所有方法都允许输出 `UNKNOWN`，因此不存在“某个方法被迫回答、另一个方法可以放弃”的不公平设置。

本文结果均来自固定 API 配置下的单次运行，temperature 设为 0。由于 API 推理仍可能带来轻微非确定性，我们目前不报告多随机种子平均，而是将 full1549 单次跑分作为主结果，并用 bad38 traces 补充搜索行为分析。

### 4.2 匹配设置下的结果与消融（Ablation under matched setting）

除特别说明外，本节所有表格均基于 matched-setting comparison：也就是说，各方法共享相同 backbone、相同 caption source 与相同预算控制。本节实验围绕一个统一主 claim 组织：**SGDAgent converts open-ended caption search into graph-anchored candidate verification.** 因而，我们依次验证三件事：第一，在严格匹配设置下，这种搜索范式比 caption-only DVD 更强；第二，增益主要来自 Stage1 和 Stage2 提供的结构化候选发现；第三，这种范式改变不仅提升准确率，也会改变 UNKNOWN、搜索轨迹稳定性和验证成本。


#### 实验1：主结论，在 matched setting 下 SGDAgent 显著优于 caption-only DVD

| 方法 | e2e_acc | answered_acc |
|---|---|---|
| Only caption | 54.81% (849/1549) | 65.66% (849/1293) |
| Only scene graph | 57.97% (898/1549) | 69.13% (898/1299) |
| Caption + SG | 64.42% (998/1549) | 72.21% (998/1382) |

实验 1 直接对应本文最核心的主 claim。结果表明，在相同 backbone、caption source 与预算约束下，把 caption 检索与 scene graph 候选发现结合起来，能够显著优于 caption-only DVD。与此同时，`Only scene graph` 明显弱于 `Caption + SG`，说明我们的优势并不是“用 graph 替代 caption”，而是通过 hybrid memory 把开放式语义搜索改写为更容易闭合的候选验证过程。

#### 实验2：增益来源，主要来自 Stage1 到 Stage2 的结构化候选发现

| 方法 | e2e_acc | answered_acc |
|---|---|---|
| SG stage1 | 59.20% (917/1549) | 69.00% (917/1329) |
| SG stage1 + stage2 | 64.62% (1001/1549) | 72.01% (1001/1390) |
| SG stage1 + stage2 + stage3 | 64.42% (998/1549) | 72.21% (998/1382) |

实验 2 用于拆解增益来源。从现有存档结果看，最大的性能跃升来自 `Stage1 -> Stage2`，这说明 SGDAgent 的主要收益来自“帧级定位 + 片段级组织”所形成的结构化候选发现链路，而不是某一个后处理 trick。在此基础上，Stage3 进一步把候选提议扩展到全局事件层：它显式建模事件节点、事件链和支持片段，并通过 event-to-segment cascade 将事件级提议回落到 Stage2 的细粒度时间窗上。其工作机制与搜索行为变化在 4.4 节中进一步分析。

#### 实验3：控制策略进一步稳定搜索轨迹

| 方法                                    | e2e_acc | answered_acc | answer rate | UNKNOWN rate |
| ------------------------------------- | ------: | -----------: | ----------: | -----------: |
| SGDAgent-base (Caption + SG)          | 64.43% (998/1549) | 72.21% (998/1382) | 89.22% (1382/1549) | 10.78% (167/1549) |
| + policy S1            | 71.47% (1107/1549) | 77.96% (1107/1420) | 91.67% (1420/1549) | 8.33% (129/1549) |
| + policy S2              | 64.62% (1001/1549) | 65.99% (1001/1517) | 97.93% (1517/1549) | 2.07% (32/1549) |
| + Full policy                         | 72.95% (1130/1549) | 74.15% (1130/1524) | 98.39% (1524/1549) | 1.61% (25/1549) |

实验 3 说明，控制策略的作用主要体现在“如何更稳地结束搜索”。SGDAgent 并非在 caption-centric 的 DVD 上附加若干图工具，而是将其重构为一个分层检索系统，scene graph 负责候选提议，局部视觉核查负责证据验证，策略层控制负责稳定收束。在 `SGDAgent-base` 上加入 `Query-type-aware routing` 后，`answered_acc` 从 72.21% 提升到 77.96%，同时 `UNKNOWN rate` 从 10.78% 降到 8.33%，说明题型路由首先改善的是工具选择与已作答样本上的答案质量。相比之下，`Answer refinement pass` 将 `answer rate` 大幅拉高到 97.93%，并把 `UNKNOWN rate` 压到 2.07%，但 `answered_acc` 下降到 65.99%，说明它主要解决的是“答不出来”的问题，同时也会把一部分样本从 `UNKNOWN` 推向错误答案。两者联合后的 `Full policy` 在四个指标上取得了最均衡的结果：`e2e_acc=72.95%`、`answered_acc=74.15%`、`answer rate=98.39%`、`UNKNOWN rate=1.61%`。因此，控制策略更适合被理解为对搜索终止条件、回答时机和验证节奏的系统性修正。

> 目的：找到一个最佳配置，也就是说把所有的有效的部分叠加起来，能得到一个很高的性能，显著超越 SOTA，同时保证这种提升是在 matched setting 下得到的，而不是由 backbone、budget 或 caption source 的变化造成的。

### 4.3 和 SOTA 的参考对比

表中的外部方法数字仅用于参考定位，不用于严格公平比较。本文的严格主结论仍以 4.2 节中的 matched-setting comparison 为准。

| 方法 | LVBench |
|---|---|
| VideoMind-7B | 40.8 |
| MRVideo | 60.8 |
| Seed1.5-VL-Thinking | 64.6 |
| Gemini-2.5-Pro | 67.4 |
| DeepVideoDiscovery(o4-mini reasoning) | 70.2 |
| DeepVideoDiscovery(o3 reasoning) | 74.2 |
| 我们方法(o4-mini reasoning) | 73.0 |

### 4.4 搜索行为与验证质量分析

如果本文的主 claim 成立，那么 SGDAgent 的收益就不应只表现为“回答更多题”，而应表现为搜索范式的整体变化：graph 先把开放式搜索锚定到少量候选，再由局部视觉做确认，因此系统既会更少停留在 UNKNOWN，也应在已作答样本上给出更稳定的答案。基于这一判断，我们进一步在具备完整 trace 的 bad38 切分上分析搜索行为。表 4 给出了 caption-only baseline 与 性能最强的两个阶段 变体的对比。可以看到，SGDAgent 不仅把 `answer rate` 从 75.84% 提升到 98.32%，把 `UNKNOWN rate` 从 24.16% 降到 1.81%，同时还把 `answered_acc` 从 65.66% 提升到 74.15%。这意味着增益并非单纯来自把 UNKNOWN 强行改成某个答案；在已经作答的样本上，答案质量本身也有所提升。与此同时，平均 LLM 调用数、工具调用数和 FI 次数均显著上升，说明 Stage1+2 的收益来自更深的结构化候选发现与更强的局部验证，而不是在原有轨迹上做轻量重排序。

| 方法  | answer rate | UNKNOWN rate | answered_acc | avg LLM calls | avg tool calls | avg frame inspect calls |
|---|---:|---:|---:|---:|---:|---:|
| DVD caption-only | 75.84% | 24.16% | 65.66% | 3.69 | 3.53 | 1.10 |
| SGDAgent Stage1+2 | 98.19% | 1.81% | 73.64%（1120/1521） | 10.63 | 10.34 | 5.34 |
| SGDAgent Stage1+2+3 | 98.39% | 1.61% | 74.15% | 5.34 | 5.21 | 1.01 |

进一步地，`compare_bad38_baseline_vs_g5g6` 的变更分析显示：Stage1+2 相对 baseline 共带来 168 个 `improved-to-correct` 样例，同时只有 70 个 `correct-to-wrong` 回退；其最常见的提升标签包括 `new_segment_graph_retrieval`、`graph_then_visual_confirm`、`replaced_broad_search_with_graph` 和 `more_visual_confirmation`。这些现象共同支持我们的主 claim：性能提升主要来自结构化候选发现和更稳定的局部验证，而不是简单的“多答题”。

对于 Stage3，我们进一步用一个具有完整 trace 的代表性运行来分析其工作方式。相较于 Stage1+2，加入 Stage3 后平均 LLM 调用数从 10.63 降到 5.34，平均工具调用数从 10.34 降到 5.21，平均 `frame_inspect_tool` 次数从 5.34 降到 1.01，使用 FI 的样本比例也从 99.83% 降到 61.91%。这些现象说明，global graph 并不是一个附属的“成本技巧”，而是一个有效的事件级候选提议层：它先在全局事件图上返回 matched events、event chain 和 candidate support segments，再把搜索级联回 Stage2/Stage1 的局部验证路径上，从而同时改变事件级定位方式和后续验证负担。

### 4.5 案例分析 Case study

为理解 SGDAgent 为何能够提升端到端准确率并显著减少 UNKNOWN，我们对 `dvd0_badcases_20260302_180904` 与`graph_vqa_bad38_g5g6_20260316_231130` 的全量运行进行了定性比较。下面前三个案例来自 improved-to-correct 样例中的代表性 `graph_driven` 轨迹，最后一个案例则选取自 UNKNOWN 消失但最终仍答错的反例，用于说明 SGDAgent 的能力边界。所有案例都服务于同一个主命题：SGDAgent 的核心收益并不在于 scene graph 直接输出答案，而在于它把 open-ended caption search 改写为 graph-anchored candidate verification。片段图（segment graph, SG）先给出时间上更集中的候选窗口，帧图（frame graph, FG）在需要时继续把问题绑定到更具体的局部对象，随后 frame inspection（FI）只在被图结构缩小后的证据空间内做视觉确认。这样，FI 不再是对整段视频的盲目搜索，而是对少量候选区域的局部验证器，因而整条推理轨迹更容易闭合。

案例 1：时间约束下的事件问答从开放搜索变为候选验证  
视频 ID： `gXnhqF0TqqI__349`

该问题询问：“mole 和 water vole 第一次上岸之后发生了什么？”并带有显式时间参考 `08:07-09:23`。在基线运行中，agent 只进行了粗粒度的 `global_browse_tool` 调用，随后即未能继续收敛，最终没有给出有效答案。失败的关键不在于视频中缺少相关画面，而在于基线缺乏把“上岸”这一事件与后续动作在同一时间链上稳定组织起来的机制。

SGDAgent 的轨迹则为 `SG → CS → FG → FI → FIN`。其中，SG 直接在目标时间窗内返回了一个高分片段 `seg_0021`（`500s-530s`），将搜索从全局 caption 空间压缩到少数几个时间一致的候选区域；随后，CS 提供补充 caption 语义，FG 在同一时间窗内进一步回溯到支持帧；最后，FI 在 `09:00-09:20` 的局部窗口内确认他们上岸后“坐下来野餐”，从而正确选择选项 D。这个案例表明，对于带有 after / first 等时序约束的问题，scene graph 的价值主要不在于直接完成推理，而在于把时序问答从开放检索问题改写为有限候选的验证问题。

案例 2：局部场景关系的判定受益于图先验定位  
视频 ID： `q01CUy_gwdU__132`

该问题询问：女人吃梨时，男女二人坐在什么地方。基线轨迹为 `GB → CSx5 → FI → FIN`。虽然基线多次调用 caption 搜索，但召回到的文本片段始终围绕“樱花树下的男女”“户外桌椅”等泛化描述，最终在一次 FI 后将场景误判为 “park bench”，输出了错误选项 D。

SGDAgent 的轨迹显著更短：`SG → FI → FI → FIN`。SG 先用 “woman eating a pear” 将时间定位到目标片段，随后两次 FI 都集中在同一局部场景上，分别给出“circular stone wall that surrounds a large globe” 和 “circular brick or stone well or planter” 这类更具体的空间描述。虽然图本身没有直接回答“是 fountain 还是 bench”，但它提供了更可靠的局部锚点，使得后续视觉核查不再在多个相似 outdoor scene 之间漂移，最终稳定映射到正确选项 B。这说明 SGDAgent 不仅改善时序定位，也能减少空间关系题中“语义相近场景”的误匹配。

案例 3：细粒度局部属性识别需要帧级锚定与重复确认  
视频 ID： `Z4HGQL_McDQ__1137`

该问题询问汽车方向盘的颜色。基线运行采用 `GB → FI → FIN`，虽然成功找到与车辆相关的场景，但只做了一次局部检查，就把答案定为 “black”，忽略了方向盘外部包覆的白色保护套，最终作答错误。

SGDAgent 在这个例子中首先调用 `frame_graph_query_tool` 检索 “steering wheel”，直接把问题锚定到 `06:09` 附近的关键帧；随后多次 FI 不再搜索整车，而是只围绕这一局部对象反复确认。第一次 FI 给出 “white”，之后又出现 “black with a white cover” 这样的混合描述。此时系统并未立即结束，而是由策略门控多次阻止 `finish`，要求继续执行窄窗、重复的 targeted FI，直到后续观察稳定收敛为 “white cover / protective cover is white”，最终才输出正确答案 A。这个案例说明，FG 的作用不是替代底层识别，而是把高歧义的全局视觉搜索变成对目标对象的局部验证；同时，不确定性感知控制可以避免 agent 在第一次看似合理的局部观察后过早终止。

案例 4：SGDAgent 可以消除 UNKNOWN，但并不自动保证最终语义判断正确  
视频 ID： `qAIRFyR6NyQ__20`

该问题询问男孩第一次亲吻女孩的是哪个部位。基线轨迹为 `GB → CSx2 → FI`，虽然已经搜到与 “boy kisses girl” 相关的全局线索，但始终未形成可终止的答案，最终停留在 UNKNOWN。相比之下，SGDAgent 的轨迹被压缩为 `SG → FI → FIN`：SG 迅速命中了与 “boy–girl–kiss” 对应的候选片段，FI 也在显式时间参考 `01:06:44-01:06:46` 内给出了一个明确判断，因此系统不再犹豫而是直接完成作答。

然而，这个答案仍然是错误的。日志显示，FI 将第一次亲吻部位判断为 “cheek”，而真实标签是 “forehead”。这说明图模块首先解决的是“去哪里看”和“是否已经有足够证据结束搜索”的问题，它能显著减少因搜索失败带来的 UNKNOWN；但如果最终的局部视觉判断本身出错，SGDAgent 仍可能从 “不会答” 变成 “答错”。因此，SGDAgent 的主要收益应理解为证据获取与轨迹闭合能力的提升，而不是对最终高层语义判断的绝对替代。



# Case Study: Repeated-Event Chain + Motive Inference

**Video ID:** `q01CUy_gwdU__121`  
**Video length:** 00:55:00  
**SGDAgent trajectory:** GG → SG → SG → SG → GB → FI → FIN  
**Baseline trajectory:** GB → FI → FI → FI → FI → FI → FIN  
**Baseline answer:** (A) ✗ | **SGDAgent answer:** (D) ✓

---

## Question

Question: Why does the man follow the woman in red after they meet for the third time?  
(A) He falls in love with the woman  
(B) He is a spy  
(C) The woman has his phone  
(D) The woman has his book

---

## SGDAgent Trajectory

---

### Agent (Step 1)

Calling tool **Global Graph Query** with query "Why does the man follow the woman in red after they meet for the third time?"

> **Global Graph Query**
>
> **Matched Events:**
>
> evt_0055 [23:45–24:40] score=2.93, confidence=0.84  
> *person, building, woman; actions: in_front_of; support: 2 segment(s)*
>
> evt_0025 [10:50–11:20] score=0.36, confidence=0.76  
> *fairy, flute; actions: holding; support: 1 segment(s)*
>
> evt_0117 [52:30–53:00] score=0.38, confidence=0.59  
> *text; actions: activity unresolved; support: 1 segment(s)*
>
> **Controller:** descend_recommended = **true**

**Interpretation for figure annotation:**  
GG anchors the "third meeting" to evt_0055 (23:45–24:40) with high confidence. The remaining events are irrelevant (low score). The controller signals that the agent must descend to segment-level for verification.

---

### Agent (Step 2)

Calling tool **Segment Graph Query** with query "Why does the man follow the woman in red after they meet for the third time?"

> **Segment Graph Query**
>
> **Matched Segments:**
>
> seg_0058 [23:45–24:15] score=2.61  
> *entities: woman, shoe, building, person; predicates: in_front_of*
>
> seg_0020 [07:55–08:25] score=0.38  
> *entities: person, tree, ledge, water; predicates: sitting_on*
>
> seg_0016 [06:15–06:45] score=0.38  
> *entities: brick structure, water puddles; predicates: none*

**Interpretation for figure annotation:**  
SG confirms the woman–person co-occurrence at 23:45, but the segment contains no causal information about *why* the man follows. The agent proceeds to test candidate hypotheses.

---

### Agent (Step 3)

Calling tool **Segment Graph Query** with query "woman book"

> **Segment Graph Query**
>
> **Matched Segments:**
>
> seg_0086 [35:25–35:55] score=**3.62**  
> *entities: person, book, tree; predicates: holding*  
> *Relation: person* **—holding→** *book (confidence=0.98)*
>
> seg_0010 [03:45–04:15] score=3.62  
> *entities: person, book; predicates: holding*  
> *Relation: person* **—holding→** *book (confidence=0.96)*
>
> seg_0011 [04:10–04:40] score=3.52  
> *entities: roads, person, building, water_pool, book, parking_lot; predicates: holding*  
> *Relation: person* **—holding→** *book (confidence=0.96)*

**Interpretation for figure annotation:**  
Strong structural evidence for "book": three segments across different time points all contain the `person–holding–book` relation with confidence ≥ 0.96. The book entity recurs throughout the video.

---

### Agent (Step 4)

Calling tool **Segment Graph Query** with query "phone"

> **Segment Graph Query**
>
> **Matched Segments:**
>
> seg_0026 [10:25–10:55] score=0.09  
> *entities: person; predicates: none*
>
> seg_0123 [50:50–51:20] score=0.10  
> *entities: map; predicates: none*
>
> seg_0127 [52:30–53:00] score=0.10  
> *entities: text; predicates: none*
>
> **fallback_used: true** (weak retrieval, no phone entity found)

**Interpretation for figure annotation:**  
No phone-related entity or relation exists anywhere in the scene graph. All scores are near zero, and fallback was triggered. This eliminates option (C).

---

### Agent (Step 5)

Calling tool **Global Browse** with query "man follows woman in red after their third meeting"

> **Global Browse**
>
> **Subjects Information:**
>
> ...  
> "subject_14": {  
>   "appearance": "Oversized illustrated **books** with colorful covers ... shown mid-air or tumbling onto a tiled floor."  
>   "identity": "These books spill out dramatically, suggesting they have been dropped or **scattered** in a moment of haste."  
>   "first_seen": "00:03:36.0"  
> }  
> ...  
> "subject_18": {  
>   "appearance": "Tall, slender woman ... Wearing a long-sleeved, vivid **red dress** or top"  
>   "identity": "She stands or sits in profile ... exuding an air of calm and introspective melancholy."  
>   "first_seen": "00:05:10.0"  
> }  
> ...  
> "subject_19": {  
>   "appearance": "Female ... Wearing a bright **red**, long-sleeved top ... Carries a white rectangular object (likely a **sketchbook**, folder, or document)"  
>   "first_seen": "00:05:55.0"  
> }  
> ...  
> "subject_20": {  
>   "appearance": "Male ... Carries a blue **folder, book, or portfolio** in one hand"  
>   "identity": "Walks closely with the female companion, sometimes gesturing with the folder"  
>   "first_seen": "00:05:55.0"  
> }

**Interpretation for figure annotation:**  
GB reveals the narrative context: books scatter on the ground (subject_14 at 03:36), the woman in red carries a sketchbook-like object (subject_19), and the male companion carries a book/portfolio (subject_20). Combined with the SG evidence from Step 3, the "book" hypothesis is strongly supported.

---

### Agent (Step 6)

Calling tool **Frame Inspect** at ["00:05:10", "00:05:30"]:  
"Why does the man follow the woman in red after they meet for the third time? What object does she have that belongs to him?"

> **Frame Inspect**
>
> **Fine-grained Response:**
>
> The man follows the woman in red after they meet for the third time because she has a scarf that belongs to him. This indicates that he is likely trying to retrieve it or is concerned about its whereabouts, which motivates him to follow her.

**Interpretation for figure annotation:**  
FI misidentifies the specific object as "scarf" rather than "book." However, the agent had already accumulated strong book-related evidence from SG (Step 3: three high-confidence `person–holding–book` triples) and GB (Step 5: scattered books, woman carrying sketchbook). The overall evidence chain supports option (D).

---

### Agent (Step 7)

Executing **Answer**: **(D) The woman has his book.** → **Right** ✓

---

## Key Takeaway for Paper

This case illustrates the **hypothesis-driven candidate verification** capability of SGDAgent. The question requires resolving both a **repeated-event chain** ("the third time they meet") and a **motive inference** ("why does he follow her").

1. **GG** first anchors the temporal window of the third encounter (evt_0055, 23:45–24:40).
2. **SG** then performs **contrastive hypothesis testing**: querying "woman book" yields strong structural hits (score=3.62, three `person–holding–book` triples with conf ≥ 0.96), while querying "phone" returns near-zero scores with fallback triggered — effectively eliminating option (C) before any visual inspection.
3. **GB** supplements narrative context (scattered books, red-dressed woman holding a sketchbook).
4. **FI** is invoked only once on a narrowed time window for final confirmation.

The baseline, lacking this graph-anchored hypothesis structure, repeats FI five times across broad and overlapping time ranges, never isolating the "book" motive, and ultimately defaults to the superficially plausible answer (A) "He falls in love."


## 5. 总结

本文提出 SGDAgent，用于在长视频问答中把 caption memory 与层次化 scene graph memory 结合起来。本文最核心的结论并不是“graph 模块带来了一些额外提升”，而是一个更强的搜索范式命题：**SGDAgent converts open-ended caption search into graph-anchored candidate verification.** 换言之，SGDAgent 的贡献在于把原本依赖开放式 caption 召回和查询改写的搜索过程，重构为“graph 锚定候选、视觉验证作答”的两阶段证据闭合过程。

这一主命题统一解释了本文的主要方法设计和实验结果。首先，它解释了为什么 hybrid memory 是必要的：只有 caption 会保留过大的开放搜索空间，只有 graph 又不足以覆盖开放语义描述，因此两者必须互补。其次，它解释了为什么 SGDAgent 能显著降低 UNKNOWN：当搜索从开放式召回转为候选验证后，agent 更容易形成稳定轨迹并给出终局判断。最后，它也解释了为什么准确率能够提升：SGDAgent 不只是回答了更多问题，而且在已作答样本上的验证质量也更高。

从当前结果看，SGDAgent 的主体收益来自 Stage1 和 Stage2 提供的局部结构化候选发现，而 Stage3 则在其上进一步补充了事件级候选提议、事件链建模和 event-to-segment cascade。未来工作可以继续沿两个方向推进：一是提升局部视觉验证本身的可靠性，以减少“从 UNKNOWN 变为错误答案”的残余误差；二是继续增强全局事件图的事件抽象与链式检索能力，使其在长程事件推理上发挥更稳定的作用。

## 6. 四张 Case Study 图绘制指引

下面这四个案例，对应附录里最后总结的两个成功模式和两个失败模式。这里不是给论文正文直接贴的文案，而是给画图用的“拆解说明”。建议最终四张图都保持和正文 Figure 3 类似的风格：左到右按 `Question -> Tool Path -> Structured Evidence -> Visual Confirmation -> Outcome` 展开。图里可以保留 `evt_xxxx / seg_xxxx / ent_xxxxxx` 作为内部工作标注，但正式版如果觉得太杂，可以只在图注里写 ID，在图中显示自然语言标签。

### 6.1 成功案例一：先局部看错，再被图结构拉回正确时间点

- 模式：`graph-guided recovery after an incorrect local probe`
- qid / video：`2256 / 16Z-XQh9jhk`
- 问题：`How many white lights were on at 45:08?`
- 正确答案：`(A) 24`
- trace：`/Users/toniekule/Dev/dvd_db/logs/graph_vqa_stage3_bad38_joint_20260310_232010/traces/16Z-XQh9jhk__2256.jsonl`
- 工具路径：`FI -> GR -> SR -> FI -> FI -> FI -> Answer`

建议把这张图画成“错误局部观察 -> 图结构纠偏 -> 窄窗精查”的过程，而不是“图直接给出答案”。

#### Panel 1：问题与第一次误判

画面重点：

1. 左上角放问题卡片：`How many white lights were on at 45:08?`
2. 紧接着画第一次 `FI`，时间窗是 `00:45:08-00:45:10`。
3. 在这个 panel 里明确标注第一次视觉回答是错的：`34 white lights`。

这一格的核心不是展示场景内容，而是展示“只看局部视觉，count 容易漂”。可以把舞台灯画成高亮白点很多、且画面稍微拥挤，强调直接数数容易把背景补光、边缘光、重复曝光都算进去。

#### Panel 2：Global Graph 把搜索支路拉回稳定舞台事件

这一格要突出 Stage-3 的作用是“重建搜索分支”，不是“替代计数”。

建议在图中画一个 event card：

- `evt_0053`
- 时间：`2675-2730s`
- summary：`stage, lights`
- 关键实体：`ent_000023 = lights`
- support segments：`seg_0108`, `seg_0109`

建议在 event card 下方画两条向下的箭头，指向两个 segment 卡片：

- `seg_0108 (2675-2705s)`，summary：`stage, lights`
- `seg_0109 (2700-2730s)`，summary：`stage, lights`

这一格里不要把 graph 画得太复杂。重点是：

1. `GR` 返回的不是“24 盏灯”，而是一个稳定的舞台灯光事件簇。
2. 这个事件簇把 agent 从第一次错误的 `FI` 结果里拽回来，让它知道“应该回到同一个灯光配置段继续核查”。

图像上可以把 `evt_0053` 画成蓝色大框，把 `seg_0108/0109` 画成两个浅蓝小框，再在框边上写一句短注释：`graph gives localization, not counting`.

#### Panel 3：Segment 证据与逐步收窄的视觉核查

这格最好画成三层逐步缩窗：

1. `SR` 结果卡片，强调 top hit 是 `seg_0108`，里面只有 `stage + lights`，几乎没有更细关系。
2. 第一次扩大窗口的 `FI`：`00:45:00-00:45:30`，回答 `30`。
3. 第二次缩小窗口的 `FI`：`00:45:05-00:45:12`，回答 `30`。
4. 最终精确到 `00:45:08-00:45:09` 的 `FI`，回答 `24`。

最适合画图的地方是最后一次 `FI` 的结构化解释：

- 左侧 10 个白灯
- 右侧 10 个白灯
- 中间额外 4 个白灯
- 总计 24

如果你想让图更直观，可以把最终 panel 里的舞台直接抽象成“三簇灯阵”：

- 左灯阵 `10`
- 中灯阵 `4`
- 右灯阵 `10`

并在旁边标一句：`Colored lights disregarded`.

#### Panel 4：这张图要传达的结论

这张图想表达的不是“graph 帮忙数灯”，而是：

1. 图结构负责把 agent 锚回同一组稳定舞台灯光事件。
2. 真正的计数仍然依赖 `FI`。
3. 如果没有 `GR/SR`，agent 会停留在第一次错误的局部观察里。

所以最后一格建议用一句很短的话收束：

`SGDAgent succeeds because the graph repairs temporal localization before the final visual count.`

#### 完整 Agent Loop（可直接照着正文 Figure 3 风格画）

##### Agent (Step 1)

Calling tool **Frame Inspect (FI)** at `["00:45:08", "00:45:10"]` with query:

`How many white lights are on?`

> **Frame Inspect**
>
> There are **34** white lights on.

**Interpretation for figure annotation:**  
这一轮故意保留成“先看错”的起点。agent 直接在目标时刻做局部计数，但没有任何结构化锚点，因此把舞台上的强白光、边缘高亮和密集灯阵一起算进去了。图里可以把这一步画成一个橙色 `FI` 框，旁边打一个红色叉号，标注 `34 (wrong)`.

##### Agent (Step 2)

Calling tool **Global Graph Query (GR)** with query:

`white lights`

> **Global Graph Query**
>
> **Matched Events:**
>
> `evt_0053 [2675-2730s]` score=`2.6053`, confidence=`0.5975`  
> *stage, lights; support: 2 segment(s)*  
> matched participant: `ent_000023 = lights` (confidence=`0.95`)  
> support segments: `seg_0108`, `seg_0109`
>
> `evt_0022 [1000-1030s]`  
> *darkness, lights*
>
> `evt_0056 [2850-2880s]`  
> *lights, screens, stage*

**Interpretation for figure annotation:**  
这一轮不是让 graph 直接告诉 agent “答案是 24”，而是把搜索重新锚回一个稳定的灯光事件簇。图里最值得高亮的是 `evt_0053` 以及它的核心实体 `ent_000023 = lights`。可以把 `evt_0022`、`evt_0056` 画成浅色备选框，表示它们是语义相近但非主分支的灯光事件。

##### Agent (Step 3)

Calling tool **Segment Graph Query (SR)** with query:

`white lights`

> **Segment Graph Query**
>
> **Matched Segments:**
>
> `seg_0108 [2675-2705s]` score=`2.4863`  
> *entities: stage, lights; predicates: none*  
> matched entity: `ent_000023 = lights` (confidence=`0.95`)  
> support frame: `f_005401`
>
> `seg_0109 [2700-2730s]` score=`2.4863`  
> *entities: stage, lights; predicates: none*
>
> `seg_0184 [4575-4605s]` score=`2.4316`  
> *entities: person, stage, crowd, lights*

**Interpretation for figure annotation:**  
这里最重要的不是 relation，而是 `seg_0108` 和 `seg_0109` 维持了同一组舞台灯阵配置。也就是说，graph 提供的是“稳定灯光场景”的定位线索，而不是可直接读出的数值答案。图里可以把 `evt_0053` 向下连到两个 segment 框：`seg_0108`、`seg_0109`，每个框里只保留 `stage` 和 `lights` 两个节点。

##### Agent (Step 4)

Calling tool **Frame Inspect (FI)** at `["00:45:00", "00:45:30"]` with query:

`How many white lights are on?`

> **Frame Inspect**
>
> There are a total of **30** white lights on.  
> `12 left + 6 middle + 12 right`

**Interpretation for figure annotation:**  
这一步说明：哪怕有了 graph 锚定，如果 `FI` 的时间窗还太宽，count 仍然会漂。图中可以把这一轮画成“回到正确事件后第一次重看”，但时间窗仍覆盖了过多近邻帧，所以仍数到 `30`。

##### Agent (Step 5)

Calling tool **Frame Inspect (FI)** at `["00:45:05", "00:45:12"]` with query:

`How many white lights are on? Count the individual white lights and disregard colored lights.`

> **Frame Inspect**
>
> There are **30** individual white lights on.

**Interpretation for figure annotation:**  
这一轮对应重复确认但仍未收敛。图里可以把两个 `FI=30` 结果并排画出来，强调 agent 已经开始缩窗，但还没有真正对齐题目要求的瞬时刻 `45:08`。

##### Agent (Step 6)

Calling tool **Frame Inspect (FI)** at `["00:45:08", "00:45:09"]` with query:

`Count individual white lights and disregard colored lights.`

> **Frame Inspect**
>
> The images show **24** individual white lights.  
> `10 left + 10 right + 4 center`

**Interpretation for figure annotation:**  
这是最终可直接落图的关键一格。建议把舞台灯阵抽象成三组：

- 左侧 `10`
- 中间 `4`
- 右侧 `10`

并在旁边写 `colored lights ignored`。这一轮才真正完成了“对瞬时时刻的 targeted verification”。

##### Agent (Step 7)

Executing **Answer**: **(A) 24** → **Right** ✓

**Interpretation for figure annotation:**  
这张图最后要强调的是：`GR/SR` 的价值在于把 agent 从第一次错误局部观察里拉回到正确的结构化搜索支路，最终由窄窗 `FI` 完成计数闭环。

---

### 6.2 成功案例二：角色名可以泛化，但关系骨架足以把场景锁在正确地点

- 模式：`relation-aware anchoring around stable scene structure`
- qid / video：`412 / HfEVEGf1A8Q`
- 问题：`Where does the girl take the vampire after the exam?`
- 正确答案：`(D) A park`
- trace：`/Users/toniekule/Dev/dvd_db/logs/graph_vqa_stage3_bad38_joint_20260310_232010/traces/HfEVEGf1A8Q__412.jsonl`
- 工具路径：`GR -> SR -> FI -> Answer`

这张图的重点不是“大动作剧情”，而是“graph 用关系骨架把 park-like scene 锁住”。

#### Panel 1：Question 与 Global Graph 命中事件簇

这一格直接画 `GR` 的 top-1 结果：

- `evt_0071`
- 时间：`2950-3030s`
- summary：`bench, vampire, girl`
- actions：`sitting_on, next_to`

同时把两个关键 participant 明确写出来：

- `ent_000032 = vampire`
- `ent_000049 = girl`
- `ent_000104 = bench`

最值得在图里画出来的一条关系是：

- `girl --sitting_on--> bench`

附加说明可以写：

`top event already contains the correct participants and a stable sitting scene after the exam`

视觉上建议把 `girl` 和 `vampire` 画成两个人物剪影坐在长椅附近，但不要太写实，因为这格主要展示“事件级锚定”。

#### Panel 2：Segment Graph 把地点语义补完整

这一格画 `SR` 的两个关键 segment：

- `seg_0116 (2875-2905s)`  
  `tree, person, bench`  
  `person --sitting_on--> bench`  
  `bench --under--> tree`

- `seg_0119 / seg_0120 (2950-3005s)`  
  `vampire, girl, bench`  
  `girl --sitting_on--> bench`

最适合画图的结构是：

1. 一个“场景骨架图”，中间是 `bench`
2. 上方是 `tree`
3. 两侧是 `girl / vampire`
4. 箭头关系分别是 `sitting_on` 和 `under`

这里要刻意强调一个方法点：

虽然图里并不总是把人物都规范成非常可靠的实名角色，有时会退成 generic `person`，但 `bench + tree + sit` 这个组合已经足够强地指向“park / outdoor resting place”，从而把候选地点压缩到很小。

#### Panel 3：FI 只做最后一跳的地点判别

这一格不要画得像在重新搜整个视频，而要像“在图锚定好的地点上做轻量确认”。

`FI` 时间窗：

- `00:48:10-00:48:25`

FI 的自然语言证据可以压成三点：

1. `bench is outdoors`
2. `trees and shrubs surround the bench`
3. `no playground equipment`

最后从视觉上落到：

- 不是 `playground`
- 不是 `home`
- 不是 `principal room`
- 最合理是 `park`

如果画图空间够，可以把四个选项画成小气泡，让 `FI` 逐个排除，最后把 `park` 高亮。

#### Panel 4：这张图要传达的结论

这张图最想传达的优势是：

1. 图结构不需要完整复述剧情。
2. 只要抓住稳定的关系骨架，地点推断就会变得很稳。
3. `FI` 只需要确认“park-like vs playground-like”，而不是从零定位人在哪。

可以用一句总结：

`SGDAgent succeeds because relational structure (bench + tree + sit) is already enough to anchor the location category.`

#### 完整 Agent Loop（可直接照着正文 Figure 3 风格画）

##### Agent (Step 1)

Calling tool **Global Graph Query (GR)** with query:

`girl take vampire after exam`

> **Global Graph Query**
>
> **Matched Events:**
>
> `evt_0071 [2950-3030s]` score=`6.2420`, confidence=`0.8605`  
> *bench, vampire, girl; actions: sitting_on, next_to; support: 3 segment(s)*  
> matched participants:  
> `ent_000032 = vampire` (confidence=`0.98`)  
> `ent_000049 = girl` (confidence=`0.95`)  
> `ent_000104 = bench`
>
> key relation:  
> `girl --sitting_on--> bench` (confidence=`0.97`)  
> support segments: `seg_0119`, `seg_0120`, `seg_0121`
>
> `evt_0052 [2125-2205s]`  
> *desk, girl, vampire*  
> earlier indoor event, lower relevance

**Interpretation for figure annotation:**  
这一轮已经把问题从“exam 后去了哪里”压成了一个非常稳定的 outdoor sitting event。图里最值得画的是 `evt_0071` 里的三元骨架：

- `girl`
- `vampire`
- `bench`

以及 `girl --sitting_on--> bench`。这就是后续地点判断的主锚点。

##### Agent (Step 2)

Calling tool **Segment Graph Query (SR)** with query:

`bench vampire girl under tree`

> **Segment Graph Query**
>
> **Matched Segments:**
>
> `seg_0116 [2875-2905s]` score=`10.3411`  
> *entities: tree, person, bench; predicates: sitting_on, under*  
> relations:  
> `person --sitting_on--> bench` (confidence=`0.91`)  
> `bench --under--> tree` (confidence=`0.88`)  
> support frame: `f_005781`
>
> `seg_0120 [2975-3005s]` score=`9.4169`  
> *entities: vampire, girl, bench; predicates: sitting_on*  
> relation: `girl --sitting_on--> bench` (confidence=`0.97`)  
> support frame: `f_005951`
>
> `seg_0119 [2950-2980s]` score=`9.4169`  
> *entities: vampire, girl, bench; predicates: sitting_on*

**Interpretation for figure annotation:**  
如果这张图要画得像正文 Figure 3，这一步最适合做成“从 event 卡片向下拆成两个 segment 卡片”：

- 一个 segment 强调 `bench --under--> tree`
- 一个 segment 强调 `girl/vampire --around--> bench`

这里的关键是：角色是否被完全实名识别并不是决定性的，真正决定地点语义的是 `bench + tree + sit` 这个稳定关系骨架。

##### Agent (Step 3)

Calling tool **Frame Inspect (FI)** at `["00:48:10", "00:48:25"]` with query:

`Describe the scene around the bench where the girl and vampire sit. Are there playground equipment, open lawn, or other indicators of park or playground?`

> **Frame Inspect**
>
> The bench is outdoors, flanked by trees, with shrubs and distant greenery.  
> There is **no playground equipment**.  
> The scene looks like a **park or garden area**.

**Interpretation for figure annotation:**  
这一步非常适合画成“最后一跳判别”。前面的 graph 已经把人物和 bench 锁住，这里的 `FI` 不再负责重新定位剧情，而只是对 location category 做最后确认：

- 不是 `playground`
- 不是 `home`
- 不是 `principal room`
- 最合理是 `park`

图里可以把四个选项画成小框，前三个打叉，最后把 `park` 高亮。

##### Agent (Step 4)

Executing **Answer**: **(D) A park** → **Right** ✓

**Interpretation for figure annotation:**  
这张图最终要让读者看到：SGDAgent 的优势不在于 graph 直接回答 “park”，而在于 graph 先用 `bench + tree + sit` 把候选地点缩到一个几乎可以单跳视觉确认的范围。

---

### 6.3 失败案例一：event 找对了表演段，但时间粒度太粗，抓不到开场瞬时灯光属性

- 模式：`event windows too coarse for transient visual attributes`
- qid / video：`4315 / 2LH3JCGkEBU`
- 问题：`At the start of the gala performance at the Vancouver Winter Olympics, what colour were the stage lights?`
- 正确答案：`(B) Aubergine`
- 模型错误答案：`(C) White`
- trace：`/Users/toniekule/Dev/dvd_db/logs/graph_vqa_stage3_bad38_joint_20260310_232010/traces/2LH3JCGkEBU__4315.jsonl`
- 工具路径：`GR -> FI -> Answer`

这张图的重点是：graph 没有彻底检索错，它是“找到对的大表演事件，但没有把开场那一个瞬间单独切出来”。

#### Panel 1：Global Graph 返回的是大事件，而不是开场瞬间

这一格画 `GR` 的 top-1：

- `evt_0012`
- 时间：`475-655s`
- summary：`person, ice_rink, audience`
- support segments：`seg_0020` 到 `seg_0026`

关键参与实体：

- `ent_000001 = person`
- `ent_000002 = ice_rink`
- `ent_000003 = audience`
- `ent_000007 = crowd`

但最关键的是这条“缺失信息”：

- `missing_slots = salient_relations`
- 没有任何 relation 直接描述 `opening instant`
- 没有任何 entity/relation 直接编码 `light color`

所以图里最好把 `evt_0012` 画成一个很长的时间条，横跨整个 gala performance 开场段，并在上方标一个红色批注：

`event is on-topic but too wide`

#### Panel 2：为什么这是 graph 的结构性短板

这一格建议不要画人物动作，而是直接画“时间跨度与问题需求不匹配”。

可以画一个长时间轴：

- 左端：真正要问的是 `start of performance`
- 中间大片：整段节目都被统一吸收到 `evt_0012`
- 右端：中后段白色主照明已经占主导

然后在时间轴上标出 support segments：

- `seg_0020`
- `seg_0021`
- `seg_0022`
- `seg_0023`
- `seg_0024`
- `seg_0025`
- `seg_0026`

图上最值得强调的结论是：

1. 图知道“这是温哥华冬奥 gala performance on ice”
2. 但图不知道“开场第一瞬间的灯光颜色”
3. 因为 transient attribute 没有被 event graph 显式表示

#### Panel 3：FI 落在过宽时间窗，只看到表演主照明

FI 时间窗：

- `00:09:35-00:10:05`

FI 的回答是：

- `white`

建议这格画成一个明显的误差来源示意：

1. 画面左边用小框标注“真正该看的瞬间：开场刚起灯”
2. 画面右边用大框标注“实际送进 FI 的 30 秒窗口”
3. 其中大多数帧都已经是明亮白光照明，因此 FI 自然答成 `white`

这张图里，错误不需要画成“模型看花眼”，而要画成：

`coarse localization -> wrong visual window -> wrong attribute answer`

#### Panel 4：这张图要传达的结论

这张图想强调的短板是：

1. 当前 event window 更适合稳定事件，不适合短时颜色属性。
2. graph 在“事件级定位”上并没错，但在“瞬时属性切片”上还不够细。
3. 真正缺的不是更多 caption，而是更细的 temporal slicing 或 attribute-aware event nodes。

建议总结句写成：

`SGDAgent fails because the graph localizes the correct performance but cannot isolate the opening instant where the light color is distinctive.`

#### 完整 Agent Loop（可直接照着正文 Figure 3 风格画）

##### Agent (Step 1)

Calling tool **Global Graph Query (GR)** with query:

`start of the gala performance Vancouver Winter Olympics stage lights`

> **Global Graph Query**
>
> **Matched Events:**
>
> `evt_0012 [475-655s]` score=`0.3032`, confidence=`0.6632`  
> *person, ice_rink, audience; support: 7 segment(s)*  
> participant entities:  
> `ent_000001 = person`  
> `ent_000002 = ice_rink`  
> `ent_000003 = audience`  
> `ent_000007 = crowd`  
> `ent_000009 = olympic_logo`
>
> support segments:  
> `seg_0020`, `seg_0021`, `seg_0022`, `seg_0023`, `seg_0024`, `seg_0025`, `seg_0026`
>
> missing slots:  
> `salient_relations`
>
> controller:  
> `descend_recommended = true`

**Interpretation for figure annotation:**  
这一步最重要的不是“命中错了”，而是“命中得太宽”。`evt_0012` 的确是 Vancouver gala performance 的开场表演段，但它把整段冰场表演都统一吸收到同一个 event 节点里。图里最适合画成长时间条：

- 左端是题目真正关心的 `start`
- 中间是整个开场表演段 `evt_0012`
- 下方接 7 个 support segments

并在 event 卡片旁标红：`missing salient_relations`, `no light-color relation`.

##### Agent (Step 2)

Calling tool **Frame Inspect (FI)** at `["00:09:35", "00:10:05"]` with query:

`What color are the stage lights at the start of the performance?`

> **Frame Inspect**
>
> The stage lights are **white**.  
> The ice rink and skaters are illuminated by clear bright lighting.

**Interpretation for figure annotation:**  
这一格要把错误来源画清楚：agent 没有继续用 `SR` 去切出更窄的 opening instant，而是直接对一个 30 秒宽窗做 `FI`。结果是，FI 看到的主要是已经稳定下来的明亮白色主照明，而不是开场最初那一瞬间的 aubergine tone。  

如果你要画得更明确，建议这格直接画成：

- 小红框：`true opening instant`
- 大橙框：`actual FI window = 00:09:35-00:10:05`

然后说明：大窗口里的主导视觉是白光，所以模型答成 `white`.

##### Agent (Step 3)

Executing **Answer**: **(C) White** → **Wrong** ✗  
Ground-truth: **(B) Aubergine**

**Interpretation for figure annotation:**  
这张图最好把“没有继续下钻”本身画成失败点。因为这不是 graph 完全失灵，而是 event-level localization 已经大体对了，但没有把 `opening-light` 这种瞬时属性切成更细的时间单元。最终错误链条是：

`coarse event -> broad FI window -> dominant white illumination -> wrong answer`

---

### 6.4 失败案例二：流程定位是对的，但图里没有“模具占位数”这种 count/state relation

- 模式：`correct process localization fails without count/state relations`
- qid / video：`3322 / Va_9Q6ekm60`
- 问题：`What is the maximum number of Taiyaki can the machine cook at the same time?`
- 正确答案：`(C) 4`
- 模型错误答案：`(D) 6`
- trace：`/Users/toniekule/Dev/dvd_db/logs/graph_vqa_stage3_bad38_joint_20260310_232010/traces/Va_9Q6ekm60__3322.jsonl`
- 工具路径：`GR -> SR -> FI -> Answer`

这张图的重点是：graph 已经把“做 Taiyaki 的那一步”找得很准了，真正缺的是对“capacity / occupancy”这种状态关系的编码。

#### Panel 1：Global Graph 准确找到 cooking phase

这一格画 `GR` 的 top event：

- `evt_0038`
- 时间：`1750-1780s`
- summary：`batter, waffle_iron, machine`

同时可在旁边补一个相邻事件，显示 graph 已经把 cooking process 的邻近结构都找到了：

- `evt_0040`：`taiyaki_mold, batter, taiyaki`
- `evt_0041`：`person, taiyaki, fish-shaped pastry`

这一格要传达：

1. agent 已经进入正确的制作阶段
2. 并不是在视频别处乱搜

所以图像可以把事件条画成连续流程：

`pour batter -> mold cooking -> finished taiyaki shown`

#### Panel 2：Segment Graph 已经把“模具、鱼形饼、机器”全部召回

这一格重点画 `SR` 的 top hit：

- `seg_0073 (1800-1830s)`
- summary：`taiyaki_mold, batter, taiyaki, taiyaki_maker, pouring_tool`

以及相邻 segment：

- `seg_0071 (1750-1780s)`：`batter, waffle_iron, machine`
- `seg_0074 (1825-1855s)`：`person, taiyaki, fish-shaped pastry, taiyaki_maker`

这格最适合画成一个结构图：

1. 中间是 `machine / taiyaki_maker`
2. 上面是 `batter`
3. 模具区域标 `taiyaki_mold`
4. 出料侧标 `taiyaki`

但要刻意在图里标一个“缺失项”：

- graph has entities
- graph lacks `slot_count` / `occupied_slot` / `max_capacity`

也就是说，图知道“这是一台 Taiyaki machine，里面有 mold 和 pastry”，但不知道“此时同时有几个 mold 槽位在工作”。

#### Panel 3：FI 看到了 6 个鱼形轮廓，但把可见形状误当成真实容量

FI 时间窗：

- `00:30:00-00:30:30`

FI 回答：

- `6 taiyaki`

这格建议画成一个非常典型的“视觉上数到了 6 个形状，但问题实际问 capacity=4”的冲突图。

最适合的画法是：

1. 机器俯视图或斜视图里画出多个鱼形轮廓
2. 用红色标出 agent 实际数到的 6 个 visible fish shapes
3. 用绿色标出真正对应 machine simultaneous cooking capacity 的 4 个有效槽位

如果你画得出来，最好把“部分已出模 / 展示品 / 相邻已成品”跟“正在机器里同时烤的槽位”区分开来。因为这正是这个问题的陷阱。

#### Panel 4：这张图要传达的结论

这张图想展示的不是“graph 不会数数”这么简单，而是：

1. graph 已经正确定位了 cooking process
2. graph 也召回了 machine / mold / taiyaki 这些核心实体
3. 但图里没有 state/count relation，所以最终仍依赖 FI 去数可见形状
4. 一旦视觉把可见轮廓和有效槽位混淆，就会从对的流程定位走向错的最终答案

最适合的总结句是：

`SGDAgent fails because the graph captures the cooking event but does not encode the machine's occupancy state or true slot capacity.`

#### 完整 Agent Loop（可直接照着正文 Figure 3 风格画）

##### Agent (Step 1)

Calling tool **Global Graph Query (GR)** with query:

`Taiyaki machine cook at the same time`

> **Global Graph Query**
>
> **Matched Events:**
>
> `evt_0038 [1750-1780s]` score=`2.6499`, confidence=`0.5736`  
> *batter, waffle_iron, machine; support: 1 segment(s)*  
> matched participant: `ent_000107 = machine`
>
> `evt_0040 [1800-1830s]` score=`2.6286`  
> *taiyaki_mold, batter, taiyaki*
>
> `evt_0041 [1825-1855s]` score=`2.6386`, confidence=`0.7781`  
> *person, taiyaki, fish-shaped pastry; actions: holding*

**Interpretation for figure annotation:**  
这一轮的关键不是精确 capacity，而是 graph 已经把搜索正确压进了 “Taiyaki 制作流程” 的邻近事件链里。图里可以把这三段 event 画成一条短流程：

`evt_0038: batter + machine`  
`evt_0040: mold + batter + taiyaki`  
`evt_0041: finished taiyaki shown`

这说明 agent 并没有在别的料理步骤里乱搜。

##### Agent (Step 2)

Calling tool **Segment Graph Query (SR)** with query:

`taiyaki machine number of taiyaki can cook at the same time`

> **Segment Graph Query**
>
> **Matched Segments:**
>
> `seg_0073 [1800-1830s]` score=`2.5423`  
> *entities: taiyaki_mold, batter, taiyaki, taiyaki_maker, pouring_tool; predicates: none*  
> matched entity: `ent_000112 = taiyaki`
>
> `seg_0071 [1750-1780s]` score=`2.4755`  
> *entities: batter, waffle_iron, machine, person*  
> matched entity: `ent_000107 = machine`
>
> `seg_0074 [1825-1855s]` score=`2.5401`  
> *entities: person, taiyaki, fish-shaped pastry, taiyaki_maker; predicates: holding*

**Interpretation for figure annotation:**  
这一步最值得画的是“实体已经很全，但关系缺了一层”。图里可以把 `machine / taiyaki_mold / batter / taiyaki` 四类实体都放进同一个 segment 框里，但同时标一个红字注释：

- `no slot_count relation`
- `no occupied_slot / max_capacity state`

也就是说，graph 已经知道“这是一台 Taiyaki machine，里面有 mold 和 fish-shaped pastry”，但它不知道“当前同时工作的有效模具槽位到底有几个”。

##### Agent (Step 3)

Calling tool **Frame Inspect (FI)** at `["00:30:00", "00:30:30"]` with query:

`How many taiyaki can the machine cook at the same time?`

> **Frame Inspect**
>
> The machine can cook **6** taiyaki at the same time.  
> The video frames show **6** fish shapes cooking simultaneously.

**Interpretation for figure annotation:**  
这一步就是最终误差发生点。最适合的画法是把机器画成一个带多个鱼形轮廓的俯视或斜视结构，并区分：

- agent 实际数到的 `6 visible fish shapes`
- 真正对应 machine capacity 的 `4 effective slots`

你也可以直接用红绿两色：

- 红色标出视觉上被数进去的 6 个鱼形轮廓
- 绿色标出真实问题想问的 4 个同时烹饪槽位

##### Agent (Step 4)

Executing **Answer**: **(D) 6** → **Wrong** ✗  
Ground-truth: **(C) 4**

**Interpretation for figure annotation:**  
这张图最后要强调：SGDAgent 并不是没找到 cooking stage，而是缺少 `count/state relation` 这种更结构化的图表示，所以最终仍把正确流程定位交给了一个会把“可见形状数”和“真实容量”混淆的视觉计数。



### 6.5 四张图统一的排版建议

如果你准备把这四个 case study 画成一组，建议继续保持和正文 Figure 3 一致的左到右布局：`Question / Tool Path -> Graph Evidence -> FI Confirmation -> Final Answer / Failure Point`。这样读者在视觉上能立刻把这四张图理解成同一个分析模板下的不同实例。

1. 成功例子里，graph 的真正价值是“把搜索空间压成一个稳定的结构化候选簇”。
2. 失败例子里，短板也同样是结构性的，主要体现在 temporal granularity 和 count/state relation 的缺失，而不是 agent 完全不会找视频内容。
