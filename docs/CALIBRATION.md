# 融合校准与离线对照

桌面程序默认使用 `quality_v2`。它按基础权重、输入质量和有效覆盖率融合。可选的 `calibrated_fusion` 使用独立标注拟合的温度和带 L2 正则的线性可靠性门控。两种策略共享采集时间对齐、原始近均匀预测过滤和模态冲突规则。界面中的“可询问”表示该观察可以进入持续证据累计，实际提示仍需满足时长和打扰策略。

## 标注任务

标注对象是一个观察区间内可见或可听的外显线索。四个标注类别为 `positive`、`neutral`、`negative` 和 `unmapped`（无法归入前三类的外显线索，如单独出现惊讶）。原有接口的 `scores.uncertain` 保存映射到第四类的质量。预测熵与类别间隔单独显示。个人困扰自评、活动效果和拒绝打扰均属于独立的反馈任务。

输入为 UTF-8 JSONL，每行一个已通过工程质量过滤的独立观察区间。按同一参与者分配 `group_id`，同一人的所有会话归入同一划分。相互重叠或由同一源片段修订的区间应先合并为一个标注样本。为每个区间分配全局唯一的 `sample_id`。下面是一条格式示例。

```json
{"target":"external_cue_positive_neutral_negative_unmapped_v1","group_id":"subject-01","sample_id":"session-02-interval-05","split":"fit","label":"negative","modalities":{"audio":{"model_version":"emotion2vec_plus_seed@7be9a30d52bd","scores":{"positive":0.1,"neutral":0.1,"negative":0.7,"uncertain":0.1},"quality":0.8,"coverage":1.0,"age_seconds":0.6,"speech_seconds":3.0}}}
```

可以使用融合输出中的 `per_modality_scores`、`per_modality_quality`、`per_modality_coverage`、`per_modality_age_seconds`、`speech_seconds` 和 `model_versions` 准备对应字段。`quality` 是有观测部分的质量，`coverage` 是这段观察区间内的覆盖比例。样本中只放实际可用的模态。视觉和声学均使用已映射的四类概率。温度应用于其对数概率，再按四类归一化。八类原始表情识别和声学模型的原始 logits 均不属于此次拟合目标。

复测使用同一尺度上的原始映射分布 `per_modality_scores`。另一个字段 `per_modality_output_scores` 保存此次融合实际采用的分布。校准成功时它包含缩放后的概率，与记录的 `weights` 一起可重现最终融合分数。

## 拟合和验证

从项目根目录运行以下命令。

```powershell
.\.venv\Scripts\python.exe scripts/calibrate_fusion.py --dataset data/cue-labels.jsonl --out reports/calibration
```

- `fit` 拟合每种模态的温度与门控。每种模态至少 24 条且四类分别至少 2 条。在门控中，监督目标是该模态判出的外显类别是否符合人工标注。输入包括工程质量、有效覆盖、概率分布特征、两模态分歧、可用模态、语段长度和观测发布延迟。
- `policy` 按联合、单视觉、单声学三种组合分别选择接受阈值。默认在经验错误率不高于 0.25 的条件下选择覆盖最多观察的阈值，接受集合至少 12 条。同分样本整体进入或退出。某组合有足够数据却找不到合格阈值时，该组合拒识。样本不足或模型版本不匹配时，该次输出明确回退到质量加权。可通过 `--max-policy-error` 修改验证目标。这一经验阈值不提供未来预测错误率的统计保证。
- `test` 使用同一批保留标注比较质量基线、仅温度、仅门控和完整策略。各策略分别使用 `policy` 划分确定其阈值，质量基线沿用原有规则。脚本拒绝跨划分的分组和重复样本 ID。项目随附的合成测试只验证算法与接口。

输出的 `evaluation.json` 含总体和分模态组合的固定四类 Macro-F1、对数损失、多类 Brier 分数、可靠性分箱、按置信阈值计算的覆盖率与错误率，以及实际策略的接受覆盖率和错误率。`reliability.svg` 将测试集中各箱的平均最高类别置信度与其中的实际正确比例绘制在同一张图上。遇到接受集合为空时，错误率记录为 null。

四种识别方案在真实同步音视频和独立观察者标签上的测试，另见[识别对照实验](RECOGNITION_EXPERIMENT.md)。该实验报告包含按演员重采样的区间、分母和相同覆盖下的错误数。

## 在桌面端启用

审核演算结果后，将输出的 `calibration.json` 复制到本地 `data/calibration.json`，在 `data/fusion-policy.json` 写入下列配置并开始新会话。

```json
{"strategy":"calibrated_fusion"}
```

将 `strategy` 设置为 `quality_v2` 可恢复默认策略。两份文件及标注数据均保存在项目忽略的本地目录。模型版本、标签映射和校准参数在加载时校验，当前会话使用开始时读取的配置。每次输出记录实际策略、校准版本、决策原因、原始未映射质量和算法指纹。个人基线版本进入完整指纹，移除该版本后的 `fusion_basis_hash` 用于判断学习依据是否兼容。
