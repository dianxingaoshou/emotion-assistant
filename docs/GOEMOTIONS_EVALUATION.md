# GoEmotions 英文发言倾向对照实验

同一批 180 条独立复验评论中，本机 Qwen 语义复核判对 **110 条（61.11%）**，桌面原有文字分类器判对 **76 条（42.22%）**。大模型多判对 34 条，负向和中性贡献了其中 33 条。其覆盖率略降，两个方法在这批评论上分别暂缓 7 条和 6 条。

在另取的 240 条无类别配额样本上，Qwen 比原桌面模型多判对 31 条，但比专门用 GoEmotions 训练的监督模型少判对 34 条。两项配对差异都达到 0.05 显著性水平。不同基线对应不同研究问题。[新样本的数据与结果](#新的无类别配额复验)

## 数据与任务

我们检验桌面助手能否从一条英文评论中识别作者表达的正向、中性或负向倾向。对照双方逐条读取完全相同的官方文本。桌面基线使用项目随附的英文 TF-IDF 线性分类器，大模型使用本机 Ollama 的 Qwen3 8B（Q4_K_M）及正式复核入口 `semantic-review-v2`，取其 `content_polarity`。GoEmotions 的标注来自阅读评论的标注员。对于“作者此刻的真实感受”、求助意愿或调节成效，这些标签无法提供验证依据。[Demszky 等，ACL 2020](https://aclanthology.org/2020.acl-main.372/)

Google Research 发布了 27 种细粒度情绪和中性标签，本实验读取其官方测试拆分。使用发布方的倾向映射将 12 种情绪映射为正向、11 种映射为负向，保留单独的 neutral 标签。另外 4 种为 ambiguous，即领悟、惊讶、好奇、困惑。含任一 ambiguous 标签的评论先排除，其余多标签评论只有在所有标签指向同一倾向时才保留。[官方测试集说明](https://github.com/google-research/google-research/blob/7951440944924ac61b6e2f9a2a3c715834005c40/goemotions/README.md) · [官方倾向映射](https://github.com/google-research/google-research/blob/7951440944924ac61b6e2f9a2a3c715834005c40/goemotions/data/sentiment_mapping.json)

官方测试集共 5,427 条，排除含 ambiguous 标签的 677 条、其余跨倾向冲突的 211 条，文本或标注缺失 0 条。剩余 4,539 条中，负向 1,070 条、中性 1,606 条、正向 1,863 条。我们在每类内部按 `SHA256("goemotions-text-review-v1|" + 官方评论ID)` 从小到大排序。先导实验选第 1–20 名，每类 20 条，共 60 条。看到先导结果后，冻结模型、提示、推理参数和指标，用第 21–80 名做独立复验，每类 60 条，共 180 条。两组 ID 已核对，交集为零。测试数据按固定提交下载并校验三个输入文件的 SHA-256。Google Research 声明其数据集采用 CC BY 4.0，源代码采用 Apache 2.0。[官方许可说明](https://github.com/google-research/google-research/blob/7951440944924ac61b6e2f9a2a3c715834005c40/README.md)

## 计分方法

准确率以整组选中的评论为分母。覆盖率是给出正向、中性或负向判断的比例。模型返回证据不足、原文校验失败、单条超时，或线性分类器未达到支持阈值，都保留在分母内，计为暂缓。宏 F1 对三类 F1 等权取平均，暂缓会减少该真实类别的召回。配对改善指同一条评论由基线未判对变为大模型判对，退化指反方向，双侧精确 McNemar 检验以这两种条数计算。服务整体不可用时实验中止。

## 实测结果

| 样本 | 方法 | 判对 | 判错 | 暂缓 | 准确率 | 宏 F1 | 覆盖率 |
|---|---|---:|---:|---:|---:|---:|---:|
| 先导 60 条 | 桌面英文分类器 | 23 | 33 | 4 | 38.33% | 0.3872 | 93.33% |
| 先导 60 条 | Qwen 语义复核 | 37 | 22 | 1 | 61.67% | 0.6008 | 98.33% |
| 独立复验 180 条 | 桌面英文分类器 | 76 | 98 | 6 | 42.22% | 0.4162 | 96.67% |
| 独立复验 180 条 | Qwen 语义复核 | 110 | 63 | 7 | 61.11% | 0.6178 | 96.11% |

| 样本 | 两者均对 | 仅基线对（退化） | 仅 Qwen 对（改善） | 两者均未判对 | 准确率提升 | 双侧精确 p |
|---|---:|---:|---:|---:|---:|---:|
| 先导 60 条 | 17 | 6 | 20 | 17 | 23.33 个百分点 | 0.00936 |
| 独立复验 180 条 | 54 | 22 | 56 | 48 | 18.89 个百分点 | 0.000149 |

| 人工倾向 | 先导基线判对 / 20 | 先导 Qwen 判对 / 20 | 复验基线判对 / 60 | 复验 Qwen 判对 / 60 | 复验 Qwen 判错 / 暂缓 |
|---|---:|---:|---:|---:|---:|
| 负向 | 9 | 19 | 27 | 47 | 11 / 2 |
| 中性 | 4 | 7 | 14 | 27 | 29 / 4 |
| 正向 | 10 | 11 | 35 | 36 | 23 / 1 |

中性是两组中最难识别的类别。先导组的 20 条人工中性里有 10 条被 Qwen 判为负向，复验组的 60 条中性里有 22 条被判为负向。复验中，176 个回复通过输出校验，其中 3 个报告倾向证据不足。另有 4 个回复未通过校验，合计暂缓 7 条。先导组 60 个回复均通过校验，其中 1 个报告倾向证据不足。服务版本为 Ollama 0.40.18，两组单条复核墙钟时间中位数分别为 1.547 秒和 0.781 秒，期间同机有其他推理任务，耗时仅供这两次运行记录使用。[先导汇总与混淆矩阵](results/goemotions-text-review.json) · [独立复验汇总与混淆矩阵](results/goemotions-text-review-replication.json)

相对现有桌面英文模型，两组均观察到发言倾向准确率和宏 F1 提升，覆盖率只在先导组提升。复验的 60 条正向评论只多判对 1 条，实际收益随类别而变。两组按倾向等额抽样，有意排除了难以确定单一倾向的评论，不能直接代表所有 Reddit 评论或真实上传视频。此处也未评估中文、转写错误、三路融合或作者本人当前情绪。项目未用这些测试评论训练或调整模型，公开语料在大模型预训练阶段的可见性无法核实。

## 复现

域内监督参照使用同一官方语料的训练拆分。只保留目标倾向明确的 36,172 条训练评论，以固定超参数拟合词级 TF-IDF 和逻辑回归。在先导 60 条中判对 42 条，在独立复验 180 条中判对 116 条，在全部 4,539 条可映射测试评论中判对 3,230 条。相同的完整 4,539 条没有全部交给 Qwen 评分。桌面还会检查英文词覆盖，并对两种判断一致的样本单独显示提示。[与 Qwen 的同输入比较](AI_ASSISTANT_STUDY.md#识别实验) · [训练、预测及门控结果](results/goemotions-linear.json)

从项目根目录在 PowerShell 执行。第一步会把三个官方输入文件缓存到被 Git 忽略的 `.runtime/goemotions/` 中，输出先导组样本指纹。模型服务启动后，两次调用各自运行一组，输出汇总统计。复验的 `--prepare-only` 可在调用模型前单独核对样本指纹。

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_goemotions.py --download --prepare-only
.\scripts\start_text_review.ps1 -PullModel
.\.venv\Scripts\python.exe scripts/evaluate_goemotions.py --per-class 20 --output reports/semantic-review/goemotions-pilot.json
.\.venv\Scripts\python.exe scripts/evaluate_goemotions.py --per-class 60 --offset-per-class 20 --output reports/semantic-review/goemotions-replication.json --predictions-output reports/semantic-review/goemotions-decisions.json
.\.venv\Scripts\python.exe scripts/evaluate_goemotions_linear.py --download --review-decisions docs/results/goemotions-replication-decisions.json --output reports/semantic-review/goemotions-linear.json
```

输入文件、样本顺序、本机模型清单、复核源码和基线权重的指纹均随结果记录。公开汇总保留混淆矩阵、各类计数和服务状态。180 条独立复验的公开逐项记录包含官方评论 ID、人工极性、原模型与千问决策，可重新核对配对检验。原始评论只在本地缓存中使用。[评测脚本](../scripts/evaluate_goemotions.py) · [180 条逐项决策](results/goemotions-replication-decisions.json)

### 新的无类别配额复验

从完整官方测试拆分开始，沿用三类倾向映射，排除先导与复验的 240 个 ID、与训练或开发集内容重复的条目以及测试集内的重复文本。比较重复时使用 Unicode NFKC、大小写折叠和字母数字串。在剩余 4,197 条中，按预先固定的 ID 哈希取前 240 条，负向 66、中性 80、正向 94。冻结名单后才调用本机千问，每种方法只读取同样的一条英文原文。四种方法分别为原桌面模型、域内监督模型按桌面实际门槛输出、同一监督模型直接取首选、本机 Qwen3 8B。初次推理结果可以缓存续跑。公开 JSON 中的逐条模型决策可以用官方数据核对，所有暂缓计入总分母。[冻结取样与评分脚本](../scripts/evaluate_goemotions_release.py)

这批评论首次用于本项目的千问逐条评测。此前曾在完整可映射测试集上统计监督模型成绩，此次沿用相同权重与门槛。由于语料公开，无法核实千问预训练时是否见过这些文本。样本中的类别比例由选定子集自然形成，对含义模糊、多倾向或中文评论的表现要另外测量。

| 同一批 240 条 | 判对 / 判错 / 暂缓 | 全组准确率 | 宏平均精确率 | 宏平均召回率 | 宏 F1 | 覆盖率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原桌面英文模型 | 107 / 127 / 6 | 44.58% | 0.4271 | 0.4279 | 0.4126 | 97.50% |
| 本机 Qwen3 8B | 138 / 96 / 6 | 57.50% | 0.6073 | 0.5923 | 0.5781 | 97.50% |
| 域内监督模型，按桌面门槛 | 172 / 68 / 0 | 71.67% | 0.7175 | 0.7121 | 0.7119 | 100% |
| 域内监督模型，直接取首选 | 172 / 68 / 0 | 71.67% | 0.7175 | 0.7121 | 0.7119 | 100% |

相对原桌面模型，Qwen 独自判对 75 条，对方独自判对 44 条，整组准确率提升 **12.92 个百分点**，双侧精确 McNemar p=0.00573。逐类看，负向由 26/66 提高到 54/66，中性由 15/80 提高到 35/80，而正向从 66/94 降至 49/94。与域内监督模型对比，Qwen 独自判对 35 条，监督模型独自判对 69 条，Qwen 的准确率**低 14.17 个百分点**，p=0.00111。两种监督输出在本次抽样上完全相同。以上 p 值均为逐项配对检验，未经多重比较校正。用户因此仍可同时查看语义复核和域内参照，遇到分歧可回看原话。[完整混淆矩阵、各项指标与 240 条决策](results/goemotions-release.json)

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_goemotions_release.py --download --prepare-only
.\.venv\Scripts\python.exe scripts/evaluate_goemotions_release.py --output reports/goemotions-fresh-inference.json
.\.venv\Scripts\python.exe scripts/evaluate_goemotions_release.py --replay docs/results/goemotions-release.json --output reports/goemotions-fresh-replay.json
```
