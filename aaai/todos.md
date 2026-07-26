# AAAI-27 Deferred Tasks

The following tasks track deferred and recently completed work in the current revision.

- [x] Reassess the novelty claim for the hierarchical representation, the paper title, and the ordering of contributions.
  - 核心定位已统一为 temporally hierarchical scene-graph video index、level-specific retrieval tools 与 adaptive agentic reasoning 的协同设计。
  - 标题保持不变，仅修正 `Long-Form`；贡献顺序保持 hierarchy index、coupled retrieval and reasoning、evaluation。
- [ ] Compress the main paper to the AAAI-27 limit of seven technical-content pages, with pages eight and nine reserved for references.
  - 当前 PDF 共 9 页，但 Conclusion 的最后一句仍占用第 8 页顶部两行，References 随后才开始。
- [ ] Merge or redraw Figures 1 and 2 with consistent vector typography and replace the sports-specific example with a more general one.
  - Figure 1 已更新为高分辨率 PNG，但仍使用体育案例，且部分层级命名和结构表达尚未与正文统一。
  - Figure 2 尚未重绘。
- [ ] Decide whether to add sampling-interval sensitivity, graph-perturbation robustness, and intrinsic graph-quality experiments.
  - 当前仅解释了缺少标准 benchmark 的原因，尚无正式实验，人工检查也未量化。

# 论文逻辑与表述一致性审查结果

## 审查状态

最后审查时间：2026-07-26。

- `[x]` 表示当前主稿已完成。
- `[ ]` 表示仍需处理。
- 本轮只更新问题状态，没有修改论文正文和实验数字。
- Supplementary 暂不纳入本轮审查范围。

## 重点审查内容

### P0：逻辑、数据和投稿硬伤

- [x] DVD 的 Video-MME `67.3` 和 EgoSchema `76.6` 已修正，旧的 `68.6 / 78.0` 不再错误用于这两列。
- [x] Abstract、Main Results 和 Conclusion 已将 graph index 与 control strategies 的端到端收益拆分为 `54.8 → 64.4 → 73.0`。
- [x] **Table 1 中 Gemini-1.5-Pro、GPT-4o、OpenAI o3 和 InternVL2.5 的 Video-MME 数据已修复。**
  - Gemini-1.5-Pro、GPT-4o 和 OpenAI o3 已修正为 Video-MME `Long, w/o sub` 的 `67.4 / 65.3 / 64.7`。
  - InternVL2.5 已补入同一来源中的 `62.6`。
  - 来源：[DVD 原论文 Table 3](https://papers.neurips.cc/paper_files/paper/2025/file/8190b210e9808e54ee16263b673a847d-Paper-Conference.pdf)。
- [x] **Table 1 的 Qwen2.5-VL 与 Seed1.5-VL 数据和分组已修复。**
  - Video-MME `63.9` 已移至 Qwen2.5-VL-72B，Seed 的 Video-MME Long 单元已留空。
  - Seed1.5-VL-Thinking 已移入 Commercial VLMs，其同一 LongVideoBench 口径的 `74.0` 已补入，并与 SGDAgent 的并列最佳值一同加粗。
- [x] **Table 1 的 `DVD, o4-mini, 70.2` 配置已明确标注。**
  - 该行已标明 `w/ transcripts`，并将模型角色写为 o4-mini reasoning 与 o3 frame inspection。
- [x] **两张表的字号和 caption 已按 AAAI 规则修复。**
  - 两张表均使用 `\small`，并已移除整体缩放和小于 9pt 的表头字号。
  - Table 2 使用缩写表头，并在表格下方的 caption 中统一解释。
  - 两张表的 caption 均已移至表格下方。
- [x] **帧采样参数的矛盾表述已删除。**
  - 正文不再同时声称“每 5 秒一帧”和“每视频最多 120 帧”，仅保留 frame-level graph extraction 的模型分工。
- [x] **Algorithm 1 按高层伪代码审查。**
  - 非关键执行细节不再单独列为待办，仅保留与正文规则直接冲突的问题。
- [x] **两个空 Section 引用已修复。**
  - 当前主稿不再渲染 `Section evaluates ...` 或 `in Section.`。
- [x] **Introduction 中控制策略的比较基线已统一。**
  - 控制策略现统一比较 `SGDAgent^0 → full SGDAgent`，对应 answer rate `89.2 → 98.4` 和 FTT `10.8 → 1.6`。
- [x] **Figure 3 已明确为机制示意，无需与原始轨迹逐项对应。**
  - 主路径用于说明 `GR → SR → FR → FI → Answer`，其他帧仅表示备选候选，不属于 scoped path 的后续步骤。
  - 案例正文已压缩为定性说明，性能依据仍来自 matched ablation。
- [x] **Terminal convergence 与视觉验证规则已统一。**
  - 正文现仅在需要视觉确认时调用 `FI`，不再要求所有 graph-derived claims 必须经过 `FI`。
  - `TerminalReadout(H)` 继续基于当前轨迹中支持最充分的选项完成预算耗尽时的收敛。


- [ ] **三张图的字体不符合 AAAI 要求。**
  - Figure 1 已改为 `4487 × 2662` 的高分辨率 PNG，不再涉及图内字体嵌入，但按 `0.92\textwidth` 缩放后，多数内部文字仍低于 9pt。
  - Figure 2 的 Helvetica、Times 和 Courier 字体没有嵌入最终 PDF。
  - Figure 3 的绝大部分内部文字缩放后低于 9pt。
  - Algorithm 1 的策略标签使用 `\scriptsize`，实际为 7pt，也应恢复到至少 9pt。

### P1：正文、图表和术语一致性

- [x] `CB` 已加入工具集合和算法执行分支。
- [x] 消融配置名、reference comparison 定位和四 benchmark 评测范围已基本统一。
- [x] HIG、VideoMindPalace、GraphVideoAgent、EGAgent 和 Vgent 已实际引用，正文 53 个 citation key 均存在。
- [x] 原 graph-depth、control-strategy 和 efficiency 表已合并，并显式区分 C、F、S、G、R、CG 和 T。
- [x] 旧的 R+G 算术错误已删除，当前 Table 2 的 9 行数字内部一致。
- [x] Table 2 所有行均满足 `answer rate + FTT = 100.0%`。
- [x] Table 2 所有行均满足 `e2e_acc ≈ ans_acc × answer rate`，差异仅来自独立舍入。
- [x] Table 2 的箭头方向和每列粗体最佳值均正确。
- [x] Entity alignment、segment/event construction 和粗略构建时间已加入 Implementation Details，低层参数已按篇幅要求移除。
- [x] 正文已停止使用 Stage 1–3，并明确 `GR / SR / FR` 与三个图层的对应关系。
- [x] Caption index 已说明沿用 DVD 的构建和 embedding-based retrieval 设置，案例中的 `GB` 已统一为 `CB`。
- [x] 指标定义、FTT 关系和基于原始计数的一位小数舍入规则已补齐。
- [x] Caption backoff、三个正式图层名称及 visual verification 与 frame inspection 的层级关系已统一。
- [x] `Base^{F+S+G}` 已改称不使用独立 caption index 的配置，不再称为 `graph-only`。
- [x] `54.8 → 64.4` 已统一归因于在 caption-only baseline 上加入 hierarchical graph retrieval channels。
- [x] 单案例已统一用于说明搜索机制，性能归因仅来自 matched ablations。
- [x] 检索评分的尺度与权重、entity registry 的 canonical name，以及 event node 的 confidence 和 uncertainty 字段已补齐。
- [x] hierarchy novelty 已收紧为多粒度表征、层级检索与自适应 Agent reasoning 的协同设计，并明确区分 HIG、VideoMindPalace、GraphVideoAgent、EGAgent 和 Vgent。
- [x] Related Work 已在保留关键引用和差异维度的前提下进一步压缩。

- [x] **与最接近的 DVD 工作已完成直接定位区分。**
  - Introduction 不再否定已有方法对视频索引和多粒度工具的组织。
  - Related Work 已明确 DVD 组织 global subjects、clip captions 和 indexed frames，而 SGDAgent 将三层场景图与对应 graph retrieval 紧密耦合。

- [ ] **默认配置和 o3 runs 的模型分工仍不可复现。**
  - Table 1 使用 `o4-mini (reasoning) / o3 (tool)`，但 Implementation Details 没有定义默认 frame-inspection 模型，`tool` 也不是精确角色名。
  - 还需明确 o3 行中 reasoning、frame inspection 和 frame-level graph extraction 是否全部替换为 o3，并说明其余设置是否不变。

- [ ] **补回直接影响 FTT 和效率结果的核心运行参数。**
  - 当前没有给出实际 keyframe sampling rule，也没有给出 Algorithm 1 的 step budget `B`。
  - 删除相互矛盾的采样描述是正确的，但最终仍需提供经确认的全视频采样方案；`B` 直接影响 terminal convergence、FTT 和平均调用次数，不能作为低层细节省略。

- [ ] **统一 FI 的窗口来源。**
  - Tool interface 将 `FI` 定义为检查 graph-proposed window，但 Strategy R 可从初始问题直接路由到 `FI`，caption-only trajectory 也执行 `CB → FI`。
  - 最小修复是将 `FI` 定义为检查 retrieved or agent-specified window，无需展开算法细节。

- [ ] **`uncertainty-aware control` 缺少对应机制。**
  - 方法只在 global event node 中保存 uncertainty，并笼统称其用于 downstream gating，没有说明 R、CG 或 T 如何使用该值。
  - 若不补充一句具体使用方式，应将贡献中的该表述收紧为已有消融直接支持的 `agent control` 或 `confidence-aware control`。

- [ ] **进一步收紧 caption channel 的消融归因。**
  - `Base^{F+S+G}` 仅关闭独立 caption index，图节点仍包含 aligned captions，空图也会借助 neighboring captions。
  - 因此应将“captions provide broad lexical coverage”改为“the standalone caption index provides additional lexical coverage”。

- [ ] **明确构建时间和 caption 复用的统计范围。**
  - `7.94 hours` 和 `121 seconds` 尚未说明是全部 103 个 LVBench 视频的总 wall-clock time，还是单视频或其他统计口径。
  - 还需说明 caption index 是复用 DVD captions 还是重新构建，以及该成本是否计入 7.94 小时。

- [ ] **收紧不可核验的 scene-graph 质量陈述。**
  - “inspect a small subset” 没有样本量、抽样方式或判定标准，却得出 “generally preserved” 的经验性结论。
  - 若已有记录，应补最小样本量和检查准则；否则应降格为 qualitative spot check，并用一句话承认稀疏采样遗漏和图错误传播。

- [ ] **补齐关键引用元数据并修正模型引用。**
  - EGAgent 当前仍缺 pages `46575--46602`、完整 booktitle、publisher、URL 和 DOI `10.18653/v1/2026.acl-long.2161`。
  - Gemini-2.0-Flash 当前错误引用 Gemini 1.5；应补正确来源，或明确其数值转引自 DVD Table 3。
  - HIG、VideoMindPalace 和 GraphVideoAgent 的标题缩写与专名还需用 BibTeX 花括号保护，避免最终参考文献显示为 `Hig`、`llms` 和 `llm-powered`。

### P1：图文问题清单

已完成项排前，未完成项排后：

- [x] Figure 1 中 `Praising Prompt Library`、`Blude jersey`、`Pereson 1` 和 `Scene Graph Constrion` 等拼写错误已修复。
- [x] Figure 1 已使用 `Contains` 连接展示 frame、segment 和 global 三层之间的跨层关联。
- [ ] Figure 1 的顶层仍是普通 entity-relation graph，未展示正文定义的 unit/group event nodes 和 event-to-event edges。
- [ ] Figure 1 应将层级统一为 `Frame-Level Graph`、`Segment-Level Graph` 和 `Global Event Graph`，并将 `Entity Reference`、`Entity Register` 和 `Video Entity Registry` 收敛为 `Registry-Based Entity Alignment` 与 `Shared Entity Registry`。
- [ ] Figure 2 使用 `Global-Level Graph`、`Hierarchical Graph Database` 和 `Caption Database`，与正文的 video index 术语不一致。
- [ ] Figure 2 将 Control Strategy 画成 verify tool，没有展示 R、CG 和 T 的真实作用位置。
- [ ] Figure 2 没有展示 `CB`，且 `FI` 的路径看起来连接 Caption Database 而不是原始视频帧。
- [ ] Figure 2 使用 MLLM，正文则区分 reasoning LLM 与 frame-inspection VLM。

### P2：阅读友好度和最终验证

- [x] 当前主稿可以完整编译，无 undefined citation、undefined reference、overfull box 或 fatal error。
- [x] 标题中的 `Long-form` 已修正为符合 Title Case 的 `Long-Form`。
- [x] Intro 当前已保持一行一个完整句子。
- [x] Table 2 caption 当前已保持一行一个完整句子。

- [ ] 完成 supplementary、reproducibility checklist 和最终投稿合规检查。
- [ ] 最终逐项核对 Table 1 的外部数据和 benchmark split。
- [ ] 最终确认技术内容不进入第 8 页，表格和图内文字不低于 9pt，且所有字体已嵌入。
- [ ] 最终确认无空 Section 引用、undefined citation、undefined reference、overfull box 和 fatal error。
- [ ] 当前仍有两处 Underfull vbox 和一处 bibliography Underfull hbox，内容稳定后再处理。
