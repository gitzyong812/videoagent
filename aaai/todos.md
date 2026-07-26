# AAAI-27 Deferred Tasks

The following tasks remain deferred or incomplete in the current revision.

- [ ] Compress the main paper to the AAAI-27 limit of seven technical-content pages, with pages eight and nine reserved for references.
  - 当前第 8 页顶部仍有 Analysis 的最后一句和完整 Conclusion。
- [ ] Reassess the novelty claim for the hierarchical representation, the paper title, and the ordering of contributions.
  - 与 HIG、VideoMindPalace、EGAgent 和 Vgent 的差异仍需收紧。
- [ ] Merge or redraw Figures 1 and 2 with consistent vector typography and replace the sports-specific example with a more general one.
  - Figure 1–2 尚未重绘，Figure 3 也需要根据真实轨迹重做。
- [ ] Decide whether to add sampling-interval sensitivity, graph-perturbation robustness, and intrinsic graph-quality experiments.
  - 当前仅解释了缺少标准 benchmark 的原因，尚无正式实验，人工检查也未量化。
- [ ] Decide whether the caption-free result is sufficiently verified and documented for inclusion.
  - 当前 `Base^{F+S+G}` 只关闭独立 caption index，不能视为严格 caption-free。
- [ ] Complete the supplementary material, reproducibility checklist, and final submission-compliance audit.
  - Supplementary 暂不纳入当前主稿审查，待主稿定稿后统一处理。

# 论文逻辑与表述一致性审查结果

## 审查状态

最后审查时间：2026-07-25。

- `[x]` 表示当前主稿已完成。
- `[ ]` 表示仍需处理。
- 本轮只更新问题状态，没有修改论文正文和实验数字。
- Supplementary 暂不纳入本轮审查范围。

## 重点审查内容

### P0：逻辑、数据和投稿硬伤

- [x] DVD 的 Video-MME `67.3` 和 EgoSchema `76.6` 已修正，旧的 `68.6 / 78.0` 不再错误用于这两列。
- [x] Abstract、Main Results 和 Conclusion 已将 graph index 与 control strategies 的端到端收益拆分为 `54.8 → 64.4 → 73.0`。

- [x] **Table 1 的 Video-MME 基线数据错位已修复。**
  - Gemini-1.5-Pro、GPT-4o 和 OpenAI o3 已修正为 Video-MME `Long, w/o sub` 的 `67.4 / 65.3 / 64.7`。
  - InternVL2.5 和 Seed1.5-VL-Thinking 已补入同一来源中的 `62.6 / 63.9`，其余未报告结果统一留空。
  - 来源：[DVD 原论文 Table 3](https://papers.neurips.cc/paper_files/paper/2025/file/8190b210e9808e54ee16263b673a847d-Paper-Conference.pdf)。

- [x] **Table 1 的 `DVD, o4-mini, 70.2` 配置标注已修复。**
  - 该行已标明 `w/ transcripts`，并将模型配置写为 o4-mini reasoning 与 o3 frame inspection。

- [x] **两张表的字号和 caption 已按 AAAI 规则修复。**
  - 两张表均使用 `\small`，并已移除整体缩放和小于 9pt 的表头字号。
  - Table 2 使用缩写表头，并在表格下方的 caption 中统一解释。
  - 两张表的 caption 均已移至表格下方。

- [x] **帧采样参数的矛盾表述已删除。**
  - 正文不再同时声称“每 5 秒一帧”和“每视频最多 120 帧”，仅保留 frame-level graph extraction 的模型分工。

- [x] **Algorithm 1 按高层伪代码审查。**
  - 非关键执行细节不再单独列为待办，仅保留与正文规则直接冲突的问题。

- [x] **Terminal convergence 和 CG 与视觉验证的硬约束冲突已消除。**
  - 正文改为在需要视觉确认时优先调用 `FI`，不再要求所有 graph-derived claims 必须经过 `FI`。
  - `TerminalReadout(H)` 作为预算耗尽时的高层终止操作保留。

- [x] **存在两个不触发 undefined warning 的空 Section 引用。**
  - `AnonymousSubmission2027.tex:406` 渲染为 `Section evaluates ...`。
  - `AnonymousSubmission2027.tex:578` 渲染为 `in Section.`。
  - 应直接写 `the Ablation Study`，不要修改 AAAI 模板设置。

- [x] **Introduction 中控制策略的比较基线已统一。**
  - 控制策略现统一比较 `SGDAgent^0 → full SGDAgent`，对应 answer rate `89.2 → 98.4` 和 FTT `10.8 → 1.6`。

- [ ] **Figure 3 和正文案例与现有轨迹记录不一致。**
  - 当前论文写成 `GR → SR → FR → FI → Answer`，并称 `FI` 在 `[24:00–24:15]` 确认了 book。
  - `paper/sg_based_dvd.md:468–651` 记录的是 `GG → SG → SG → SG → GB → FI → FIN`。
  - 该记录中的 `FI` 窗口为 `[00:05:10–00:05:30]`，返回对象为 scarf，并未确认 book。
  - 当前图中的 `fra_0086` 在现有记录中实际是 `seg_0086`，需要回到原始日志核验或更换案例。

- [ ] **Figure 3 违反层级时间范围约束。**
  - 当前 `GR [23:45–24:40] → SR [23:45–24:15]` 后仍返回 `[35:25–35:30]` 的帧。
  - 这与 `scope` 硬约束和 `FR within a narrow SR scope` 的正文定义冲突。

- [ ] **技术内容仍进入第 8 页。**
  - 当前 PDF 共 9 页，第 8 页同时包含 Conclusion 和 References。

- [ ] **三张图的字体不符合 AAAI 要求。**
  - Figure 1–2 的 Helvetica、Times 和 Courier 字体没有嵌入最终 PDF。
  - Figure 3 的绝大部分内部文字缩放后低于 9pt。


### P1：正文、图表和术语一致性

- [x] `CB` 已加入工具集合和算法执行分支。
- [x] 消融配置名、reference comparison 定位和四 benchmark 评测范围已基本统一。
- [x] HIG、VideoMindPalace、GraphVideoAgent、EGAgent 和 Vgent 已实际引用，正文 53 个 citation key 均存在。
- [x] 原 graph-depth、control-strategy 和 efficiency 表已合并，并显式区分 C、F、S、G、R、CG 和 T。
- [x] 旧的 R+G 算术错误已删除，当前 Table 2 的 9 行数字内部一致。
- [x] Table 2 所有行均满足 `answer rate + FTT = 100.0%`。
- [x] Table 2 所有行均满足 `e2e_acc ≈ ans_acc × answer rate`，差异仅来自独立舍入。
- [x] Table 2 的箭头方向和每列粗体最佳值均正确。
- [x] 关键模型分工和构建时间已加入 Implementation Details，低层参数已按篇幅要求移除。
- [x] 正文已停止使用 Stage 1–3，并明确 `GR / SR / FR` 与三个图层的对应关系。
- [x] Caption index 已说明沿用 DVD 的构建和 embedding-based retrieval 设置，案例中的 `GB` 已统一为 `CB`。
- [x] 指标定义、FTT 关系和基于原始计数的一位小数舍入规则已补齐。
- [x] Caption backoff、三个正式图层名称及 visual verification 与 frame inspection 的层级关系已统一。
- [x] EGAgent BibTeX 已按 ACL Anthology 补齐页码、完整会议名、publisher、URL 和 DOI。

- [ ] **补齐 o3 runs 的实现配置。**
  - Implementation Details 只定义了 o4-mini。
  - 需要说明 o3 是否仅替换 reasoning model，以及其他组件是否完全相同。

- [ ] **澄清 `7.94 hours` 和 `121 seconds` 的统计范围。**
  - 当前主文没有说明它们是 103 个 LVBench 视频的总 wall-clock time，还是单视频时间。

- [ ] **将 `graph-only configuration` 改为更准确的表述。**
  - Frame-level graph 包含 aligned caption `c_i`，空图还会使用 neighboring captions。
  - `Base^{F+S+G}` 只能称为 `without the standalone caption index` 或 `graph-index-only retrieval setting`。

- [ ] **相应收紧 `54.8 → 64.4` 的归因边界。**
  - 该对比说明加入 graph retrieval channels 的收益。
  - 它不能证明图构建过程完全不依赖 caption semantics。

- [ ] **收紧 hierarchy novelty 和最相关工作对比。**
  - Intro 已承认 HIG、VideoMindPalace、GraphVideoAgent、EGAgent 和 Vgent。
  - 随后又笼统声称现有工作只关注 frame-level 或 segment-level local relations，前后存在张力。
  - 应重点比较 index organization、retrieval granularity、online update、visual verification 和任务范围。

- [ ] **补足 limitations。**
  - 当前只解释了缺少标准 intrinsic benchmark，没有承认 sparse sampling miss 和 graph-error propagation。
  - `a small subset` 和 `generally preserved` 没有样本量、检查规则或比例，应量化或明确降级为非定量 sanity check。

- [ ] **降低单案例的证据强度。**
  - 当前只有一个 case study，不能证明 gains 的来源。
  - 性能归因应来自 matched ablation，trajectory case 只能 illustrate 机制。

- [ ] **补齐正文中的关键评分和数据结构定义。**
  - 说明 `S_lex + S_soft + S_temp` 的归一化或权重。
  - Entity registry 的正式字段缺少 alignment 使用的 `name`。
  - Global event node tuple 缺少 Quality Control 中声称保留的 confidence 和 uncertainty。

### P1：图文问题清单

只记录问题，不在本轮修改图片：

- [ ] Figure 1 的顶层仍是普通 entity-relation graph，不是正文定义的 global event graph。
- [ ] Figure 1 没有展示 unit/group event nodes、event-to-event edges 和 provenance。
- [ ] Figure 1 仍有 `Praising Prompt Library`、`Blude jersey`、`Pereson 1` 和 `Scene Graph Constrion` 等拼写错误。
- [ ] Figure 1 的 `Global-level`、`Entity Index`、`Entity Register` 和 `Video Entity Registry` 命名不统一。
- [ ] Figure 2 使用 `Global-Level Graph`、`Hierarchical Graph Database` 和 `Caption Database`，与正文的 video index 术语不一致。
- [ ] Figure 2 将 Control Strategy 画成 verify tool，没有展示 R、CG 和 T 的真实作用位置。
- [ ] Figure 2 没有展示 `CB`，且 `FI` 的路径看起来连接 Caption Database 而不是原始视频帧。
- [ ] Figure 2 使用 MLLM，正文则区分 reasoning LLM 与 frame-inspection VLM。
- [ ] Figure 3 的轨迹、时间戳、节点 ID 和 FI 输出需要根据原始日志整体重做。

### P2：阅读友好度和最终验证

- [x] 当前主稿可以完整编译，无 undefined citation、undefined reference、overfull box 或 fatal error。

- [ ] 将标题中的 `Long-form` 改为符合 Title Case 的 `Long-Form`。
- [ ] 将 Intro 中同一源码行的 `Firstly / Then / Finally` 三句拆成一行一个完整句子。
- [ ] 将 Table 2 caption 中的多个句子拆成一行一个完整句子。
- [ ] 完成 supplementary、reproducibility checklist 和最终投稿合规检查。
- [ ] 最终逐项核对 Table 1 的外部数据和 benchmark split。
- [ ] 最终逐项核对 Figure 3 的时间戳、节点 ID、工具调用和 FI 输出。
- [ ] 最终确认技术内容不进入第 8 页，表格和图内文字不低于 9pt，且所有字体已嵌入。
- [ ] 最终确认无空 Section 引用、undefined citation、undefined reference、overfull box 和 fatal error。
- [ ] 当前仍有若干 Underfull vbox 和一处 bibliography Underfull hbox，内容稳定后再处理。
