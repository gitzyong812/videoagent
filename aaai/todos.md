# AAAI-27 Deferred Tasks

The following tasks are intentionally deferred from the current revision.

- [ ] Compress the main paper to the AAAI-27 limit of seven technical-content pages, with pages eight and nine reserved for references.
- [ ] Reassess the novelty claim for the hierarchical representation, the paper title, and the ordering of contributions.
- [ ] Merge or redraw Figures 1 and 2 with consistent vector typography and replace the sports-specific example with a more general one.
- [ ] Decide whether to add sampling-interval sensitivity, graph-perturbation robustness, and intrinsic graph-quality experiments.
- [ ] Decide whether the caption-free result is sufficiently verified and documented for inclusion.
- [ ] Complete the supplementary material, reproducibility checklist, and final submission-compliance audit.

# 论文逻辑与表述一致性审查计划

## 审查目标

只审查 [AnonymousSubmission2027.tex](/Users/zhangyong/projects/ResearchProjects/VideoAgent/SGDAgent/aaai/AnonymousSubmission2027.tex)，不检查实现源码，不修改任何文件。

最终输出一份按优先级排列的问题清单。每项包含位置、冲突内容、阅读或可信度风险，以及建议统一方向。

## 重点审查内容

### P0：逻辑和数据自洽性

重点检查并列出：

- Abstract 将完整系统的 `+18.2 pp` 基本归因于 graph-anchored verification，但该增益实际同时包含 graph index 和控制策略。
- Introduction 在描述控制策略收益时混用了 caption-only 和 SGDAgent-base 两个基线。
- Strategy R+G 行中，`75.6% × 91.0% = 68.8%`，与表中的 `70.2% e2e_acc` 不一致。
- Table 5 使用同时改变 graph level 和控制策略的配置，却将效率变化单独归因于 Stage 3。
- Algorithm 的 forced answer 可能绕过视觉验证 gate，也无法解释完整系统仍有 `1.6% FTT`。
- Algorithm 的动作集合前后不一致，CB 在正文中是工具，但没有对应执行分支。
- Strategy R 允许直接调用 FI，但算法中的 FI 依赖已有的 `Top(H)` 候选。
- “每 5 秒一帧，最多 120 帧”的采样描述无法解释小时级视频的完整覆盖，需要作者确认真实采样规则。
- 图层深度消融没有明确说明 caption database 是否始终保留，容易与 “SG only” 混淆。

### P1：正文、图表和术语一致性

重点检查并列出：

- 统一 graph level：frame-level graph、segment-level graph、global event graph。
- 明确 Stage 1–3、图层名称和 GR/SR/FR 之间的对应关系。
- 统一 caption fallback 与 caption backoff。
- 统一 visual verification、pixel verification、frame inspection 和 FI 的职责层级。
- 统一 Strategy R 的 entry routing 与 query-type routing。
- 统一 SGDAgent-base、SGDAgent full、full system 和 S1+2 等配置名称。
- Table 1 的 reference comparison 定位是否在表注、正文和 Abstract 中一致。
- Table 3 与 Table 5 的统计样本不同，但差异说明不足。
- “shorter trajectories” 与 Table 5 中 full system 更多 tool calls 的结果不完全一致。
- Related Work 对 HIG、VideoMindPalace、EGAgent 的概括与 Introduction 中“现有工作缺少跨尺度结构”的说法是否冲突。

### P1：图文问题清单

只记录问题，不修改图片：

- Figure 1 的 global-level entity graph 与正文定义的 global event graph 不一致。
- Figure 1 没有实际展示正文强调的 event-to-event edges。
- Figure 1 存在 `Constrtion`、`Praising Prompt Library`、`Pereson`、`Blude jersey` 等疑似拼写错误。
- Figure 1 中 Entity Index、Entity Register、Video Entity Registry 命名不统一。
- Figure 2 使用 MLLM，而正文分别使用 VLM 和 LLM。
- Figure 2 将 Control Strategy 画成 verify 工具，但正文将其定义为路由和门控策略。
- Figure 2 中 FI 的输入路径看起来来自 Caption Database，与“FI 返回原始像素”冲突。
- Figure 2 未清楚展示 CB，与算法和正文的工具定义不一致。
- Figure 3 在父级 `[23:45–24:15]` 范围下返回 `[35:25–35:30]` 的帧，违反已确认的硬过滤规则。
- Figure 3 的内部文字在最终双栏页面中过小，影响阅读。

### P2：阅读友好度

- Abstract 和 Introduction 开头高度重复。
- Abstract 连续两次使用 “imprecise”，信息层次不清。
- Introduction、Method Tool Chain 和 Results 中存在多次重复解释 propose-verify-answer。
- Figure 1 放置位置与首次正文引用之间的阅读顺序需要检查。
- 方法段落过密，定义、工具职责和控制策略之间缺少清晰过渡。
- 数值精度不统一，例如 `8.30%`、`9.00%` 与其他一位小数结果混用。
- 当前编译仍有一处方法小标题产生的 Underfull hbox，作为低优先级排版问题记录。

## 输出格式与假设

- 问题按 P0、P1、P2 排序，同一优先级内按对核心论点的影响排序。
- 仅基于当前 LaTeX 和编译后的 PDF，不依据尚未完成的实现源码。
- 将父级时间 scope 视为下一级检索的硬过滤条件。
- 不验证外部论文数据，不建议新增实验，也不处理页数限制。
- 图片问题只形成清单，不修改 Figure 1–3。
