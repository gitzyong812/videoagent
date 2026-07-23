# MM review

<!-- https://ccn5n72irr7t.feishu.cn/docx/E5VkdcrkKoVXMIxBEYhcm4odnKf -->
<!-- https://openreview.net/forum?id=FerKHN21Tz&referrer=%5Bthe%20profile%20of%20Yong%20Zhang%5D(%2Fprofile%3Fid%3D~Yong_Zhang11) -->

# Official Review by Reviewer 27Jx（1分）

**Weaknesses: **The manuscript contains **a large number of draft\-stage errors**, and most of the text appears to have been submitted without substantive human polishing\. Examples include a dangling "Figure ??" reference, repeated typos such as "multi\-granul," "Constrion," and "Stage," and generally unedited phrasing throughout\. In its current state the paper does not meet the presentation bar expected for review\. **The core distinction from VAgent is not adequately established\.** VAgent also uses a graph\-structured representation for video retrieval, and the paper differentiates itself only by claiming VAgent is "limited to frame\- or segment\-level local relations without cross\-scale unification\."* But the cross\-scale event layer \(Stage 3\) is exactly what is supposed to provide this advantage, and the ablation in Table 3 shows that adding Stage 3 does not improve accuracy at all \(64\.6% → 64\.4%\)*\. If the event layer contributes no accuracy gain, the central conceptual difference from VAgent largely collapses, and the paper is left differentiating itself mainly on an efficiency argument that it does not frame as its core contribution\.稿件中存在大量处于草稿阶段的错误，大部分文本似乎在未经过实质性人工润色的情况下提交。例如存在悬挂的“Figure ??”引用，反复出现的拼写错误如“multi\-granul”、“Constrion”和“Stage”，以及整体上未经编辑的措辞。目前状态下，论文未达到审稿所期望的表达水平。论文与 VAgent 的核心区别并未得到充分论证。VAgent 也使用图结构表示用于视频检索，而论文仅以声称 VAgent“仅限于帧级或片段级的局部关系，缺乏跨尺度的统一”为差异化点。但跨尺度事件层（Stage 3）恰恰应当提供这种优势，而表 3 中的消融实验显示加入 Stage 3 并未改善准确率（64\.6% → 64\.4%）。如果事件层没有带来准确率上的提升，那么与 VAgent 的核心概念性差异在很大程度上就崩塌了，论文主要剩下以效率作为差异点的论述，而该点并未被定位为其核心贡献。

**Review: **This paper proposes SGDAgent, a long\-form video understanding agent built on a three\-level \(frame/segment/event\) spatio\-temporal scene graph index, queried through a propose\-verify\-answer loop\. The direction is sensible and the matched\-setting evaluation methodology is appreciated\. However, the paper has two problems that I consider disqualifying in its current form\. First, the manuscript is clearly a draft: it contains an unresolved "Figure ??" reference, numerous typographical errors, and unpolished prose throughout\. This is not a matter of a few isolated mistakes; the writing as a whole has not been carefully revised, which makes the paper difficult to evaluate fairly and signals it was submitted prematurely\. Second, and more fundamentally, the paper's claimed novelty relative to VAgent rests on cross\-scale hierarchical unification, embodied by the event\-level graph \(Stage 3\)\. Yet the authors' own ablation \(Table 3\) shows Stage 3 yields no accuracy improvement — it functions only as an efficiency layer that reduces LLM and frame\-inspection calls\. If the component that supposedly distinguishes SGDAgent from VAgent does not improve answer quality, then the core conceptual contribution over VAgent is not substantiated\. The efficiency benefit is real and worth reporting, but it is not framed as the paper's main claim and is not, on its own, sufficient to establish the contribution\. Because of the combination of an unpolished manuscript and an insufficiently justified core novelty claim, I recommend rejection\. I would encourage the authors to thoroughly revise the writing and to either demonstrate an accuracy benefit from the hierarchical event layer or reframe the paper's contribution honestly around efficiency\.

本文提出了 SGDAgent，一种基于三层（帧/片段/事件）时空场景图索引、通过 propose\-verify\-answer 循环进行查询的长时视频理解代理。该方向合理，且匹配设定下的评估方法值得肯定。然而，本文目前稿件存在两个我认为足以使其不合格的问题。首先，手稿明显仍是草稿：文中存在未解析的“Figure ??”引用、大量印刷错误以及通篇不够打磨的行文。这并非几个孤立的错误；整体写作未经过认真修订，导致论文难以公平评估，也表明其提交过于仓促。其次，更根本的是，论文相对于 VAgent 所声称的新颖性依赖于跨尺度的层级统一，这由事件级图（阶段 3）体现。然而作者自己的消融实验（表 3）显示阶段 3 并未带来准确率提升——它仅作为一个降低 LLM 和帧检索调用次数的效率层。 如果据称将 SGDAgent 与 VAgent 区分开的组件并未提高答案质量，那么相较于 VAgent 的核心概念性贡献就没有立得住脚。效率方面的收益是真实且值得报告的，但论文并未将其作为主要论断来表述，而且单凭效率本身不足以确立贡献。由于手稿不够成熟同时核心新颖性主张也缺乏充分论证，我建议拒稿。我鼓励作者彻底修订写作，要么证明分层事件层能带来准确性提升，要么如实将论文的贡献围绕效率来重新表述。

# Official Review by Reviewer D8hx（3分）

**Weaknesses: **

\(1\) The offline graph construction cost is not sufficiently reported\. SGDAgent relies on GPT to generate frame\-level scene graphs from sampled keyframes, followed by segment/event graph construction\. Since this graph index is central to the method, the paper should report these\.

\(1\) 离线图构建成本未得到充分报告。SGDAgent 依赖 GPT 从采样的关键帧生成帧级场景图，随后构建片段/事件图。由于该图索引是方法的核心，论文应报告这些（成本/时间/资源等）信息。

\(2\) The cross\-benchmark comparison is not fully controlled\. Table 1 includes results from different backbones, caption sources, and evaluation protocols, and the authors acknowledge that these should be read only as reference positions\. 

\(2\) 跨基准比较控制不充分。表 1 包含来自不同骨干网络、字幕来源和评估协议的结果，作者也承认这些结果应仅作为参考位置阅读。

\(3\) The scene graph quality is not directly evaluated\. Since the method depends on VLM\-generated entities, relations, event clusters, and cross\-frame entity alignment\. End\-task QA accuracy alone is insufficient to validate the reliability of the graph index\.

\(3\) 场景图质量未被直接评估。鉴于该方法依赖于 VLM 生成的实体、关系、事件簇以及跨帧实体对齐，单凭末端任务的问答准确率不足以验证图索引的可靠性。

\(4\) Many implementation details are underspecified\. This limits reproducibility for an agentic system where prompts and heuristics can strongly affect results\. For example, entity alignment thresholds, similarity weights, and entity blacklists\.

\(4\) 许多实现细节描述不足。这限制了在一个代理式系统上的可复现性，因为提示和启发式规则可能强烈影响结果。例如，实体对齐阈值、相似性权重和实体黑名单等。

\(5\) There are also some presentation issues\. Section 3\.2 contains a Figure?? reference\. Some language issues also should be polished\.

\(5\) 另外还有一些呈现问题。第 3\.2 节包含一个 Figure?? 引用。一些语言表述也应当润色。

**Review:  **

This paper proposes SGDAgent, a scene\-graph\-driven agent for long\-form video understanding\. The main idea is to construct a hierarchical spatio\-temporal scene graph index over long videos and use it to guide an agent through a propose\-verify\-answer pipeline\. Instead of relying on open\-ended caption retrieval, the method retrieves candidate events, segments, and frames from structured graph representations, then verifies evidence through targeted frame inspection\. I find the problem setting important and the main idea promising\. Long\-form video QA often fails because the relevant evidence is sparse and difficult to locate, rather than because the final reasoning step is intrinsically hard\. A structured graph index with entity, relation, and temporal anchors is a natural way to improve evidence search\.

本文提出了 SGDAgent，一种用于长视频理解的场景图驱动代理。其主要思想是对长视频构建分层的时空场景图索引，并利用该索引通过提出\-验证\-回答的流程来引导代理。方法不是依赖开放式的字幕检索，而是从结构化图表示中检索候选事件、片段和帧，然后通过针对性的帧检查来验证证据。我认为该问题设置重要且主要思想有前景。长视频问答常常失败的原因在于相关证据稀疏且难以定位，而非最终的推理步骤本质上困难。具有实体、关系和时间锚点的结构化图索引是改进证据检索的自然方式。

However, I have several concerns that prevent a stronger recommendation\. First, the cost and quality of the offline scene graph construction are not sufficiently analyzed\. Since the graph is generated using chatgpt, the method may be benefiting from substantial closed\-source VLM preprocessing that is not counted in the online reasoning budget\. Second, the largest gains mix search improvement with terminal convergence and FTT reduction, making it hard to isolate the actual improvement in evidence localization\. Third, most controlled ablations are conducted on LVBench, while the broader benchmark comparison is not strictly matched\.

然而，我有若干顾虑使我无法给予更强烈的推荐。首先，离线场景图构建的成本和质量没有得到充分分析。由于图是使用 ChatGPT 生成的，该方法可能受益于大量未公开的 VLM 预处理，这些成本并未计入在线推理预算。其次，最大的提升将搜索改进与终端收敛和 FTT 降低混合在一起，使得很难隔离出证据定位的实际改进。第三，大多数受控消融实验是在 LVBench 上进行的，而更广泛的基准比较并未严格匹配。

Overall, I like the direction and believe graph\-guided candidate verification is promising for long video understanding\. However, the current paper needs stronger cost accounting, direct graph\-quality evaluation, and more controlled cross\-benchmark ablations to fully support its claims\. I therefore recommend borderline\.

总体而言，我喜欢这一方向，并认为以图引导的候选验证对于长视频理解是有前途的。然而，当前论文需要更严格的成本核算、直接的图质量评估以及更受控的跨基准消融实验来充分支持其论断。因此我建议给出边缘接受（borderline）评价。

# Official Review by Reviewer k56s（4分）

### Significant Writing and Formatting Oversights \(Poor Writing and Formatting\)
1\. 严重的写作与格式疏漏（写作与格式较差）

The paper appears somewhat rushed in terms of fine\-tuning details; it contains numerous obvious, elementary errors that severely compromise both the readability and the academic rigor of the work:

论文在微调细节方面显得有些仓促；存在大量明显的基础性错误，严重影响了论文的可读性和学术严谨性：

**Incorrect Baseline Data Citation:** In the table comparing existing methods \(Table 3\), the authors report the results for the DVD model on the Video\-MME dataset as 68\.6\. However, according to the original DVD paper, its result on Video\-MME should be 67\.3; the figure 68\.6 actually corresponds to DVD's metric on the LongVideoBench \(Long\) dataset\. The authors must carefully cross\-check all cited benchmark data to avoid transcription errors that could invalidate the comparative analysis\. Additionally, the reported results for EgoSchema do not match those in \[1\]; please verify this discrepancy as well\.
基线数据引用错误：在比较现有方法的表格（表 3）中，作者报告 DVD 模型在 Video\-MME 数据集上的结果为 68\.6。然而，根据原始 DVD 论文，其在 Video\-MME 上的结果应为 67\.3；数字 68\.6 实际上对应 DVD 在 LongVideoBench \(Long\) 数据集上的指标。作者必须仔细核对所有引用的基准数据以避免抄录错误，这类错误可能使比较分析无效。此外，报告的 EgoSchema 结果与 \[1\] 中的不符；请核实这一差异。

**Spelling Errors:** In Figure 1, the bottom\-most label "Frame\-level" is misspelled as "Fame\-level\." Furthermore, Figure 1 has certain limitations in terms of clarity; specifically, the use of basketball as an example may make it appear that the concept cannot be generalized to broader, more universal methods\. It might be worth considering a different example for illustrative purposes\.
拼写错误：在图 1 中，最底部的标签 “Frame\-level” 被误写为 “Fame\-level”。此外，图 1 在清晰性方面存在一定局限；具体而言，使用篮球作为示例可能会让人觉得该概念无法推广到更广泛、更通用的方法。建议考虑采用不同的示例以作说明。

**Broken References:** In Section 3\.2 \(Line 390\), an unresolved cross\-reference appears as "as shown in Figure ??"; this reference clearly intended to point to Figure 1\.
损坏的引用：在第 3\.2 节（第 390 行），出现了未解析的交叉引用，显示为“as shown in Figure ??”；该引用明显应指向图 1。

### Insufficient Citation of Related Works \(Missing Related Works\)
2\. 相关工作引用不足（缺失相关工作）

When discussing the compression of spatio\-temporal information in long/complex videos, hierarchical graph representations, and Graph\-based Agents, it is recommended that the following recent works be cited to broaden the scope of the paper and ensure a more comprehensive literature review:

在讨论长/复杂视频中时空信息压缩、分层图表示和基于图的代理时，建议引用以下近期工作以拓宽论文视角并确保更全面的文献综述：

\[1\] "Building a Mind Palace: Structuring Environment\-Grounded Semantic Graphs for Effective Long Video Analysis with LLMs", CVPR 2025\. This work also addresses the problem of long video understanding, proposing to represent long video content as a hierarchically organized topological semantic graph \(Hierarchical Graph\), utilizing nodes and edges to enhance an LLM's comprehension of the video\. This approach bears a strong resemblance to the construction logic of the "Hierarchical Spatio\-Temporal Scene Graph" proposed in the current paper; comparing against this work would serve to better highlight the uniqueness of the proposed toolchain design\.

\[1\] “Building a Mind Palace: Structuring Environment\-Grounded Semantic Graphs for Effective Long Video Analysis with LLMs”，CVPR 2025。该工作同样研究长视频理解问题，提出将长视频内容表示为分层组织的拓扑语义图（分层图），利用节点和边增强 LLM 对视频的理解。该方法与本文提出的“分层时空场景图”的构建逻辑高度相似；与此工作的对比将有助于更好地突出所提工具链设计的独特性。

\[2\] "Multivariate Gaussian Representation Learning for Medical Action Evaluation", arXiv:2511\.10060\. This work proposes compressing fine\-grained actions into adaptive 3D Gaussian Tokens via a multivariate Gaussian representation, thereby filtering out spatio\-temporal noise while preserving action semantics\. This approach shares a fundamental conceptual similarity with the core idea, specifically, the strategy of constructing hierarchical scene graphs to compress spatio\-temporal redundancy and extract key evidence from long videos\. It is recommended that a discussion of this work be included around Line 159 or Line 168\. 

\[2\] “用于医学动作评估的多元高斯表示学习”，arXiv:2511\.10060。该工作提出通过多元高斯表示将细粒度动作压缩为自适应的三维高斯令牌，从而在保留动作语义的同时滤除时空噪声。该方法在根本概念上与本文的核心思想存在相似性，特别是构建分层场景图以压缩时空冗余并从长视频中提取关键证据的策略。建议在第 159 行或第 168 行附近加入对该工作的讨论。

\[3\] "Understanding Long Videos via LLM\-Powered Entity Relation Graphs", arXiv:2501\.15953\. This work introduces GraphVideoAgent, an agent\-based framework for long video understanding that tracks the evolution of entities within long videos by explicitly modulating dynamic entity relation graphs and leveraging the reasoning capabilities of LLMs\. This aligns closely with the core concept of the current paper, which drives agents for retrieval and reasoning based on Scene Graphs\.

\[3\] “通过由 LLM 驱动的实体关系图理解长视频”，arXiv:2501\.15953。该工作提出了 GraphVideoAgent，一种基于代理的长视频理解框架，通过显式调制动态实体关系图并利用 LLMs 的推理能力来追踪长视频中实体的演化。这与本文的核心概念高度一致，本文同样基于场景图驱动代理进行检索与推理。

### Computational Overhead of Offline Construction
3\. 离线构建的计算开销

The bottom\-up construction of scene graphs relies on a VLM to perform graph extraction on densely sampled keyframes\. When processing long videos spanning hours, this approach—involving dense, repeated invocations—can incur substantial computational costs and API token overhead\. We recommend that the authors include an analysis within the paper detailing the specific time consumption and token costs associated with the scene graph construction phase \(the Offline Pipeline\)\.

基于自底向上的场景图构建依赖于视觉语言模型（VLM）对密集采样的关键帧执行图提取。当处理跨越数小时的长视频时，这种涉及密集、重复调用的方法会产生大量计算开销和 API token 开销。我们建议作者在论文中加入一节分析，详细说明场景图构建阶段（离线流水线）所消耗的具体时间和 token 成本。

### Error Propagation and System Robustness
4\. 错误传播与系统稳健性

The system's performance is highly contingent upon the quality of the underlying frame\-level scene graph extraction\. If the underlying VLM suffers from hallucinations, attribute misidentification, or fails to detect critical entities during the generation of frame\-level scene graphs, these errors will propagate upward through the aggregation process to the Global Event Graph\.

系统性能高度依赖于底层帧级场景图提取的质量。如果底层 VLM 在生成帧级场景图时出现幻觉、属性识别错误或未能检测到关键实体，这些错误将通过聚合过程向上传播至全局事件图。

How robust is the system to variations in the quality of the underlying scene graph generation? \[4\] To what extent do the strategies outlined in the paper mitigate the potentially fatal impact of such structural errors? We recommend that the authors include a relevant robustness analysis or a brief discussion to bolster the credibility of the proposed methodology\.

系统对底层场景图生成质量变化的鲁棒性如何？\[4\] 论文中所述的策略在多大程度上缓解了此类结构性错误可能造成的致命影响？我们建议作者补充相关的鲁棒性分析或简短讨论，以增强所提出方法的可信度。

\[4\] "Agentic Very Long Video Understanding", arXiv:2601\.18157v1\.



# Official Review by Reviewer SSkz（3分）

**Weaknesses: **Some claims are stronger than the method supports\. The method mainly improves structured evidence retrieval and localization, but it does not fundamentally solve incomplete sampling coverage, since the graph is still built from sparsely sampled keyframes\.
有些论断超出了该方法的支撑范围。该方法主要提升了结构化证据检索与定位，但并未从根本上解决采样覆盖不全的问题，因为图仍然是由稀疏采样的关键帧建立的。

1. The global event graph contributes more to efficiency than accuracy\. The ablation shows that adding the global event graph does not further improve accuracy, but mainly reduces LLM calls and frame\-inspection calls\. This weakens the claim that it substantially improves high\-level reasoning\.
全局事件图对效率的贡献大于对准确性的提升。消融实验表明，加入全局事件图并未进一步提高准确性，主要是减少了对 LLM 的调用和逐帧检查的调用。这削弱了其显著提升高层次推理能力的论断。

2. Scene graph construction is heuristic and under\-analyzed\. Entity alignment, segment merging, and event construction rely on hand\-designed similarity rules and thresholds\. The paper does not sufficiently analyze robustness to noisy entities, wrong relations, or alignment errors\.
场景图构建具有启发性且分析不足。实体对齐、片段合并与事件构建依赖于人工设计的相似性规则与阈值。论文未充分分析对噪声实体、错误关系或对齐错误的鲁棒性。

3. Writing and presentation require improvement\. The paper contains several presentation issues\. In particular, Figure 1 and Figure 2 have inconsistent font sizes and appear to be rasterized rather than vector graphics, which makes some labels blurry and difficult to read\. There is also an unresolved figure reference in Section 3\.2 \(“Figure ?? \(b\)”\), which should be corrected\. In addition, the manuscript contains several typos and grammatical issues, such as “multi\-granul,” “and convert,” “Section 3\.3 describe,” and “Scene graph is \.\.\. that encode\.”
论文的写作和呈现需要改进。稿件存在若干展示问题。特别是图 1 和图 2 的字体大小不一致，且似为光栅图而非矢量图，导致某些标签模糊且难以辨认。第 3\.2 节中还存在未解析的图表引用（“Figure ?? \(b\)”），应予以更正。此外，手稿中存在若干拼写和语法问题，例如“multi\-granul”、“and convert”、“Section 3\.3 describe”以及“Scene graph is \.\.\. that encode”。

**Review: **This paper proposes SGDAgent, a hierarchical spatio\-temporal scene\-graph\-driven agent for long\-form video understanding\. The key idea is to convert long videos into structured scene graph indices at frame, segment, and global event levels, and then use an agent to perform graph\-guided candidate proposal, visual verification, and final answering\. Questions: How accurate are the generated scene graphs themselves? Is there any evaluation of entity accuracy, relation accuracy, or temporal alignment quality? How sensitive is SGDAgent to the keyframe sampling interval? Would the performance drop significantly if the interval changes from 5 seconds to 10 seconds or 2 seconds? Since the global event graph mainly improves efficiency rather than accuracy, should the paper reposition it as an efficiency layer instead of a reasoning layer? How much does the full pipeline cost per video, including graph construction with gpt\-4\.1 and online reasoning with o4\-mini or o3?

本文提出了 SGDAgent，一种用于长视频理解的分层时空场景图驱动代理。其核心思想是将长视频在帧、片段和全局事件三个层级转化为结构化的场景图索引，然后使用代理执行基于图的候选提议、视觉验证和最终答题。问题包括：生成的场景图本身有多准确？是否对实体准确率、关系准确率或时间对齐质量进行了评估？SGDAgent 对关键帧采样间隔有多敏感？如果间隔从 5 秒变为 10 秒或 2 秒，性能会显著下降吗？鉴于全局事件图主要提升的是效率而非准确性，是否应将其重新定位为效率层而不是推理层？完整流水线每个视频的成本是多少，包括使用 gpt\-4\.1 进行图构建以及使用 o4\-mini 或 o3 进行在线推理？

# Official Review by Reviewer wGgF（3分）

**Weaknesses:  **

- The paper claims as a key contribution a hierarchical spatio\-temporal scene graph architecture and says prior work lacks unified cross\-scale hierarchical organization, but prior work such as HIG \(CVPR 2024\) already introduced a hierarchical graph approach for scene graph generation in video understanding
论文声称其关键贡献为分层时空场景图架构，并称先前工作缺乏统一的跨尺度分层组织，但已有工作如 HIG \(CVPR 2024\) 已在视频理解的场景图生成中引入了分层图方法。

- Several important components are described only at a high level: entity alignment uses a weighted combination of similarities with thresholding; event construction uses temporal gap/shared entities/relation overlap; retrieval uses a composite score; controller signals and routing rules are mentioned but not concretely specified\.
若干重要组件仅以高层次描述：实体对齐使用带阈值的相似度加权组合；事件构建采用时间间隔/共享实体/关系重叠；检索使用复合评分；控制器信号与路由规则被提及但未作具体说明。

- The graph\-depth ablation indicates that adding the global event graph does not further improve accuracy over Stage 1\+2, but mainly reduces LLM and frame\-inspection calls\. That is still useful, but it weakens the impression that the full three\-level hierarchy is uniformly necessary for performance\.
图深度消融表明，加入全局事件图并不会在准确率上比第 1\+2 阶段进一步提升，主要是减少了对 LLM 和帧检视调用的次数。这仍然是有用的，但削弱了三层完整层级在性能上普遍必要的印象。

- The paper argues that the graph structure makes search more efficient, but the reported efficiency evidence mainly concerns online call counts\.
论文主张图结构使搜索更高效，但所报告的效率证据主要涉及在线调用次数。

- There are noticeable writing and formatting issues, including awkward grammar and an unresolved “Figure ??” reference\.
存在明显的写作和格式问题，包括语法生硬以及未解决的“Figure ??”引用。

**Review:  评审：**

- The paper claims a "hierarchical spatio\-temporal scene graph architecture" as a central contribution and explicitly states that prior methods lack unified cross\-scale hierarchical organization\. However, HIG \(CVPR 2024\) already proposes a Hierarchical Interlacement Graph for scene graph generation in video understanding, and describes it as a hierarchical graph structure for modeling scene changes across tasks\. Because of this prior art, I do not think the hierarchy itself can be presented as strongly novel
论文声称“分层时空场景图架构”是核心贡献，并明确指出以往方法缺乏统一的跨尺度分层组织。然而，HIG（CVPR 2024）已提出用于视频理解中场景图生成的层级交错图（Hierarchical Interlacement Graph），并将其描述为用于跨任务建模场景变化的层级图结构。鉴于这一先前工作，我认为单凭层级结构本身难以被呈现为具有强烈的新颖性。

- After discounting the hierarchy claim somewhat, the paper’s more convincing contribution is the graph\-driven propose\-verify\-answer pipeline: the graph is elevated from auxiliary representation to primary retrieval index, and the agent uses graph\-localized proposals before visual verification\. I think this is a fair and interesting contribution, but the paper should position it more explicitly as the main novelty, instead of bundling it together with a representation\.
在对层级结构的主张有所保留之后，本文更具说服力的贡献是图驱动的“提出\-验证\-回答”流程：将图从辅助表示提升为主要检索索引，智能体在视觉验证之前使用图局部化的候选。我认为这是一个公平且有趣的贡献，但论文应更明确地将其定位为主要新颖点，而不是与表示一起捆绑说明。

- Many implementation choices are too abstractly described\. For example, entity alignment depends on a weighted combination of name similarity, description similarity, and type consistency, but the weighting and threshold are not given\. Event construction likewise depends on temporal and semantic heuristics, but the exact merging/splitting rule is unclear\. The composite retrieval score is specified conceptually but not operationally\. The controller signals and routing/gating strategies are important to the reported gains, yet their instantiation remains underspecified\.
许多实现细节描述得过于抽象。例如，实体对齐依赖于名称相似度、描述相似度和类型一致性的加权组合，但未给出权重和阈值。事件构建同样依赖于时间和语义启发式规则，但具体的合并/拆分规则不清楚。组合检索得分在概念上被说明，但在操作上未加以实现。控制信号及路由/门控策略对所报告的性能提升很重要，然而其具体实现仍然描述不足。

- The matched\-setting LVBench results are compelling and, in my view, are the main reason the paper is above the bar\. However, the paper also acknowledges that the cross\-benchmark comparisons are not controlled\. This makes the broader performance positioning less definitive than the narrative sometimes suggests\.
匹配设置下的 LVBench 结果令人信服，在我看来也是论文超出基准的主要原因。然而，论文也承认跨基准比较并未受控。这使得更广泛的性能定位不如叙述有时所暗示的那样确定。

- Segment\-level graphs seem to provide the main accuracy benefit, while the event\-level graph primarily reduces tool usage and search depth\. This is still a meaningful contribution, but it suggests the full hierarchy should be presented with more nuance\. In particular, the event\-level graph looks more like an efficiency layer than an accuracy\-critical component\.
分段级图似乎提供了主要的准确性提升，而事件级图主要减少了工具使用和搜索深度。这仍然是一个有意义的贡献，但这表明整个层级结构的呈现应更为细致。特别地，事件级图更像是一个效率层，而非对准确性至关重要的组件。

