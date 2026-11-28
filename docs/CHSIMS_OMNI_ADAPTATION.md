# 中文多模态教师的微调与替代模型对照

2026 年 11 月 28 日完成。CH-SIMS v2.0 使用全部 2,722 段训练、647 段验证和 1,034 段测试。候选在开发集按 `video_id` 或采集系列整组留出选参。原 190 段影视来源已经用于过往诊断，本轮用于定向复测。另用未参加训练或调参的 MCAE-SPPS 普通话表演录音，检验声音模型在独立语料上的表现。[前一轮跨来源诊断](CHSIMS_OMNI_TEACHER.md#声音跨来源校准与人脸近景复验)

## 声音的中性误判与修复

中性预测的数量接近标注数量时，仍需要检查命中的片段。旧两阶段模型在 190 段中给出 31 次中性，其中 4 次正确、27 次错误。22 段真实中性中，6 段判成负向、12 段判成正向。两阶段模型分别估计中性和非中性方向，再让三个类别分数竞争。新来源上，中性分支给许多真中性的分数低于两侧方向分数。上一轮先验校正版相对 emotion2vec+ 向量参照纠正 52 个错误，同时产生 48 个新错误。4 段净收益按 15 个来源重采样，准确率差的 95% 区间为 −7.21 至 +13.87 个百分点。[逐类统计](results/chsims-omni-audio-prior.json) · [两阶段混淆矩阵](results/chsims-omni-audio-factorized.json)

本轮以校准后的桌面声音分数作首判。只有先验校正后的千问对正负方向给出至少 0.55 的后验分数，或对中性给出至少 0.75，才采纳其复核。门槛由开发集六个采集系列轮流留出选出。保护中性的版本还保留桌面已判为中性的片段，从规则上保证相对于同一桌面参照，中性召回不下降。没有可用音轨时保留暂缓。[复核与按来源导出监督](../scripts/chsims_audio_review_gate.py)

| 190 段未见影视来源，仅声音 | 判对数 | 准确率 | Macro-F1 | 预测中性 | 中性召回 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 上一轮两阶段 | 108 | 56.84% | 0.4740 | 31 | 4/22 |
| 上一轮千问先验校正 | 105 | 55.26% | 0.5089 | 48 | 11/22 |
| emotion2vec+ 同训练向量参照 | 101 | 53.16% | 0.4547 | 26 | 5/22 |
| 桌面声音分数校准 | 112 | 58.95% | 0.5405 | 32 | 10/22 |
| 千问低秩微调 | 92 | 48.42% | 0.4428 | 62 | 13/22 |
| 千问有条件复核 | 123 | 64.74% | 0.5831 | 25 | 9/22 |
| **千问复核并保护中性** | **119** | **62.63%** | **0.5677** | **33** | **10/22** |

保护中性的复核方案相对桌面纠正 14 段，改错 7 段，提升 3.68 个百分点。按来源配对重采样，95% 区间为 −2.20 至 +10.24 个百分点。准确率更高的复核方案少认出 1 段真实中性。保护中性方案在全量 1,034 段纯声音测试中，从桌面 **577 段、Macro-F1 0.4850** 提到 **710 段、0.5862**。在声音与文字标注不同的 242 段上，判对数从 119 增至 127，中性从 14/81 增至 15/81。保护中性的三类概率已按来源五折导出，覆盖 2,722 段训练视频。[复核结果及配对区间](results/chsims-omni-audio-review-gate.json)

## 千问微调与替代大模型

千问微调在 Qwen2.5-Omni-3B 音频编码器最后一层的注意力中加入秩为 8 的 LoRA，共更新 **61,440 个适配参数**。分类同时学习三类极性和 `label_A` 的连续效价。相同分类头与训练流程下，冻结编码器、低秩微调后在官方测试全集分别判对 697 和 672 段，在 190 段复测中分别判对 98 和 92 段。低秩微调在训练来源留出验证上提高了 Macro-F1，但该增益没有延续到新影视来源。[模型与训练实现](../scripts/chsims_audio_lora.py) · [冻参与微调同片对照](results/chsims-omni-audio-lora.json)

替代候选 **MiniCPM-o 4.5** 是原生接收文字、音频和视频的 9B 多模态大模型。本轮从作者发布的检查点提取原生音频编码器的中间层与末层，用相同开发集和分类训练对照。三路替换实验使用 MiniCPM-o 的声音、千问的文字和画面特征，并结合桌面分数。MiniCPM-o 的文字和画面分支仍需各自验证，因此结果只支持比较替换声音编码器的效果。[MiniCPM-o 官方项目](https://github.com/OpenBMB/MiniCPM-o) · [原生声音特征提取](../scripts/chsims_minicpmo_audio.py)

| 仅声音 | CH-SIMS 1,034 段判对 / Macro-F1 | CH-SIMS 未见来源 190 段判对 / Macro-F1 | MCAE-SPPS 360 段判对 / Macro-F1 |
| --- | ---: | ---: | ---: |
| 桌面声音分数校准 | 577 / 0.4850 | 112 / 0.5405 | 174 / 0.3825 |
| 同训练桌面声学特征 | 593 / 0.5216 | 106 / 0.4869 | 236 / 0.6517 |
| 千问特征 | 689 / 0.6058 | 88 / 0.4431 | 249 / 0.6781 |
| MiniCPM-o 4.5 声音特征 | 648 / 0.5938 | 72 / 0.3817 | 279 / 0.7724 |
| MiniCPM-o 4.5 声音与桌面特征 | 688 / 0.6070 | 92 / 0.4614 | 285 / 0.7915 |
| **千问与桌面声学特征共同训练** | **716 / 0.6128** | 103 / 0.5038 | **293 / 0.8119** |
| 千问复核并保护中性 | 710 / 0.5862 | **119 / 0.5677** | 215 / 0.5472 |

MCAE-SPPS 抽取 6 位普通话演员各 60 段，正向、中性、负向各 120 段。正向为高兴，负向各半取愤怒和悲伤。作者发布的听评记录须有超过半数评分者同意录音类别，组内按片段编号哈希选取。两套系统均读取同一份 16 kHz 单声道录音。千问与桌面声学特征共同训练的准确率是 **81.39%**，同训练桌面参照为 **65.56%**，中性召回为 **111/120 对 58/120**。按演员重采样，准确率差的 95% 区间为 +1.94 至 +29.17 个百分点。表演录音与 CH-SIMS 影视片段的录音条件及标注方式有别。保护中性的复核依赖桌面的首判，在表演录音上低于同训练桌面参照。模型选择保留千问作为三路教师，MiniCPM-o 保留作声音替代对照。[MCAE-SPPS 论文](https://doi.org/10.1038/s41597-026-06976-z) · [作者公开数据](https://doi.org/10.17605/OSF.IO/9JYZC) · [独立语料逐类统计](results/mcae-spps-audio-omni.json)

## 三路融合的整段结果

后续缺失实验增加了单路可用、三路全缺和连续缺段，并用同一批测试视频比较原分类头与按证据组合训练的分类头。[教师缺失证据实验](CHSIMS_OMNI_MISSING_EVIDENCE.md)

每路特征独立提取。训练时同时使用完整输入及遮蔽文字、声音、画面的样本，遮蔽会同时移除该路特征与可用标记。桌面特征与千问特征使用相同训练、遮蔽与验证流程，对照整段人工标签。输入里自然缺失的 8 段转写仍按其余证据计算。缺文字时原音轨仍含可听字词。[融合代码](../scripts/chsims_omni_model_comparison.py) · [全部混淆矩阵与来源区间](results/chsims-omni-model-comparison.json)

| 输入条件 | 全量同训练桌面 | 全量千问特征 | 全量千问与桌面 | 190 段未见来源，同训练桌面 → 千问与桌面 |
| --- | ---: | ---: | ---: | ---: |
| 完整输入 | 742 / 0.6377 | 777 / 0.6724 | 775 / 0.6639 | 112 → 122 |
| 缺文字 | 662 / 0.5581 | 760 / 0.6456 | 761 / 0.6536 | 122 → 131 |
| 缺声音 | 709 / 0.5812 | 753 / 0.6280 | 748 / 0.6267 | 106 → 127 |
| 缺画面 | 711 / 0.6062 | 716 / 0.6289 | 714 / 0.6256 | 109 → 97 |

表中每格为判对数及 Macro-F1，最后一列为同批 190 段的判对数。全量完整输入，原桌面校准版判对 **736/1,034（71.18%，Macro-F1 0.6391）**，千问加桌面判对 **775/1,034（74.95%，0.6639）**。同训练桌面判对 **742/1,034（71.76%，0.6377）**。千问加桌面相对同训练桌面的准确率差为 +3.19 个百分点，按来源配对重采样的 95% 区间为 +1.02 至 +5.43 个百分点。190 段上的整段标签对照中，桌面固定版、桌面校准版和千问加桌面分别判对 122、120 和 122 段，Macro-F1 分别为 0.5435、0.5784 和 0.5603。缺画面时，同一批新来源的结果仍低于同训练桌面。[整段与缺一路的对照数据](results/chsims-omni-model-comparison.json)

## 复现与文件位置

沿用全量教师缓存和 CUDA 环境。MiniCPM-o 4.5 的音频编码器位于 ModelScope 发布的第一分片，将该分片、`config.json` 和检查点索引放入 `.runtime/chsims-teacher/minicpm-o-4_5`。提取脚本逐个核对发布 SHA-256。MCAE-SPPS 的四个情绪 ZIP 和六个 `raw_data_actor*.csv` 来自作者的 OSF 仓库，缓存放在 `.runtime/chsims-teacher/full/mcae-spps`。六个评分文件在目录中按演员编号命名为 `ratings-1.csv` 至 `ratings-6.csv`。[教师缓存和 CUDA 环境](CHSIMS_OMNI_TEACHER.md#复现全量监督) · [MiniCPM-o 发布权重](https://www.modelscope.cn/models/OpenBMB/MiniCPM-o-4_5) · [MCAE-SPPS 附件](https://osf.io/9jyzc/)

```powershell
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$data = 'D:\情绪数据集\ch-sims2s\ch-simsv2s'
$desktop = '.\.venv\Scripts\python.exe'
$teacher = '.\.runtime\chsims-teacher\venv\Scripts\python.exe'
& $teacher -B -X utf8 scripts/chsims_omni_audio_encoder.py --data $data
& $teacher -B -X utf8 scripts/chsims_audio_lora.py fit --data $data
& $teacher -B -X utf8 scripts/chsims_audio_lora.py evaluate --output docs/results/chsims-omni-audio-lora.json
& $teacher -B -X utf8 scripts/chsims_minicpmo_audio.py --data $data
& $desktop -B -X utf8 scripts/chsims_omni_model_comparison.py fit
& $desktop -B -X utf8 scripts/chsims_omni_model_comparison.py evaluate --output docs/results/chsims-omni-model-comparison.json
& $desktop -B -X utf8 scripts/chsims_audio_review_gate.py fit
& $desktop -B -X utf8 scripts/chsims_audio_review_gate.py evaluate --output docs/results/chsims-omni-audio-review-gate.json
& $desktop -B -X utf8 scripts/evaluate_audio_external.py prepare
& $teacher -B -X utf8 scripts/evaluate_audio_external.py encode
& $teacher -B -X utf8 scripts/evaluate_audio_external.py evaluate --output docs/results/mcae-spps-audio-omni.json
```

模型、逐片预测及监督概率留在 `.runtime/chsims-teacher/full`，报告只记录聚合统计。纯声音分组按单模态标签，三路实验按整段标签。软件离屏回归 648 项通过，桌面启动及模拟调节闭环验收通过。
