# 千问作为中文多模态教师的四路对照

千问音频编码器微调、保护中性的复核、MiniCPM-o 4.5 的替代对照和独立普通话录音验证见[中文多模态教师的微调与替代模型对照](CHSIMS_OMNI_ADAPTATION.md)。

## 声音跨来源校准与人脸近景复验

本轮在本地复用了 CH-SIMS v2.0 的 2,722 段训练、647 段验证及 1,034 段测试视频。此前查看过的 190 段新来源音频和 598 段桌面可判画面用于定向复测。训练、归一化、类别先验和融合选参使用开发集。复测说明这些改动对这两批片段的影响，进一步确认跨来源表现需要独立的新来源。[声音指标与混淆矩阵](results/chsims-omni-audio-prior.json) · [声画模型对照](results/chsims-omni-cross-source.json) · [中性与正负方向分别建模](results/chsims-omni-audio-factorized.json)

### 声音为什么偏向中性

旧版在 647 段验证视频中只遇到 1 段所属 `video_id` 未在训练集中出现的视频，而 190 段复测来自训练和验证中都未出现的 15 个 `video_id`。这 190 段波形的时长中位数为 4.21 秒，其他 844 段测试视频为 3.11 秒。波形 RMS 中位数分别为 0.128 和 0.086。这些差异提示来源变化也改变了输入分布。

旧版声音分类头按训练类频的倒数给样本加权。训练集负向、中性、正向分别为 1,241、453、1,028 段，因此每个中性样本的训练权重约是负向样本的 2.74 倍。复测时，千问把 57 段负向和 42 段正向判成了中性。这 99 个错误中，有 65 段在负向、正向两个分数之间已经排对了方向，分类头的中性分数压过了这个判断。

保留原有模型权重，按训练集各类别比例的对数抵消训练时的逆频率加权。修正系数由训练数据直接给出。验证时轮流留出六个按数据目录命名归组的来源系列，比较纯千问与混入 emotion2vec+ 的方案，按各系列等权的准确率选出纯千问。推理阶段只读音轨相关特征。另测了按单个来源留组训练的新分类头和专家融合，以及两阶段分类。两阶段方案用 emotion2vec+ 及声学质量估计中性概率，用千问和 emotion2vec+ 的特征估计非中性时的正负方向，最后合为三个分数。[先验校正](../scripts/chsims_audio_prior.py) · [按来源训练](../scripts/chsims_teacher_cross_source.py) · [两阶段分类](../scripts/chsims_audio_factorized.py)

| 190 段未见来源声音 | 判对 | 准确率 | Macro-F1 | 预测中性 |
| --- | ---: | ---: | ---: | ---: |
| 上一版千问 | 70 | 36.84% | 0.3750 | 115 |
| 同训练的 emotion2vec+ 向量与桌面分数 | 101 | 53.16% | 0.4547 | 26 |
| 只校准桌面声音分数 | 112 | 58.95% | 0.5405 | 32 |
| 按来源重训后融合千问与桌面声音 | 94 | 49.47% | 0.4718 | 72 |
| 千问恢复训练类频先验 | 105 | 55.26% | 0.5089 | 48 |
| 两阶段识别中性和正负 | 108 | 56.84% | 0.4740 | 31 |

先验校正纠正了 41 段被误判为中性的正负向音频，另有 5 段原先判对的中性被改错。负向召回 27/90 → 49/90，正向 27/78 → 45/78，中性 16/22 → 11/22。千问比 emotion2vec+ 向量对照多判对 4 段，准确率差按来源配对重采样的 95% 区间为 −7.21 至 +13.87 个百分点。两阶段模型多判对 7 段，中性召回为 4/22。保留先验校正的模型作为蒸馏监督候选，因为它在 190 段上取得了更高的 Macro-F1，也保留了更多正确的中性判断。

全量测试上，先验校正将千问判对数从 658 升至 714，Macro-F1 从 0.6010 变为 0.6051。声音与文字标注分歧的 242 段里，先验校正判对 108 段，少于旧版的 115 段，按来源训练的专家融合判对 130 段。该分组的人工文字标注仅用于事后核对。

### 人脸近景怎样补充全画面

桌面表情模型先找脸，在脸部裁剪上判断表情，并筛去看不清脸或人脸过多的帧。千问旧版把整个画面缩到 224 像素范围，背景和其他人物也参与编码。在共同覆盖的 598 段里，旧版千问比桌面多认对 30 段中性，少认对 32 段负向和 42 段正向。这个差异解释了 373 对 417 的准确率差。

改进用同批去字幕帧定位人脸。只在检测到单张且脸框边长达到门槛时裁剪近景，单独编码并与全画面表示结合。找不到可裁剪的人脸时使用全画面。开发集按 `video_id` 留整组训练和验证，选择全画面与近景的比例，再比较加入桌面有质量证据的表情分数后的结果。在相同训练流程下，从全画面单支到加入人脸近景，598 段中判对数从 391 提到 419。再结合桌面分数后判对 431 段。[人脸近景提取代码](../scripts/chsims_face_teacher.py) · [同覆盖和缺失证据对照](results/chsims-omni-cross-source.json)

| 仅画面输入 | 桌面固定版同法校准 | 旧版千问 | 千问全画面与人脸近景 | 千问近景与桌面分数融合 |
| --- | ---: | ---: | ---: | ---: |
| 同覆盖 598 段判对数 | 417 | 373 | 419 | **431** |
| 同覆盖准确率 | 69.73% | 62.37% | 70.07% | **72.07%** |
| 同覆盖 Macro-F1 | 0.5354 | 0.5669 | **0.5995** | 0.5809 |
| 桌面原先暂缓的 436 段判对数 | 0 | 248 | **266** | 264 |
| 全部 1,034 段判对数 | 417 | 621 | 685 | **695** |

融合在共同覆盖组纠正了桌面的 35 段错误，把 21 段原本正确的判断改错，准确率净增 2.34 个百分点，按来源重采样 95% 区间为 +0.17 至 +4.72 个百分点。共同覆盖组的负向、中性、正向召回，桌面分别为 241/292、5/88、171/218，融合为 251/292、11/88、169/218。全集上，融合判对数提高，但中性召回从旧版千问的 70/188 降为 28/188。

### 本地复现

使用下文完成的全量缓存。人脸编码用教师 CUDA 环境，后续训练和评测用项目原有环境。两个方案的训练监督保存在 `.runtime/chsims-teacher/full` 的 `audio-prior-training-targets.json` 和 `cross-source-visual-targets.json`。

用官方标签和逐片预测独立复算了 30 个分组与模型组合的指标，核对了两份各 2,722 段的训练监督。项目离屏回归 646 项通过，桌面启动检查及模拟调节闭环通过。

```powershell
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$data = 'D:\情绪数据集\ch-sims2s\ch-simsv2s'
$desktop = '.\.venv\Scripts\python.exe'
$teacher = '.\.runtime\chsims-teacher\venv\Scripts\python.exe'
& $teacher -B -X utf8 scripts/chsims_face_teacher.py --data $data
& $desktop -B -X utf8 scripts/chsims_teacher_cross_source.py fit
& $desktop -B -X utf8 scripts/chsims_teacher_cross_source.py evaluate --output docs/results/chsims-omni-cross-source.json
& $desktop -B -X utf8 scripts/chsims_audio_prior.py fit
& $desktop -B -X utf8 scripts/chsims_audio_prior.py evaluate --output docs/results/chsims-omni-audio-prior.json
& $desktop -B -X utf8 scripts/chsims_audio_factorized.py fit
& $desktop -B -X utf8 scripts/chsims_audio_factorized.py evaluate --output docs/results/chsims-omni-audio-factorized.json
```

## 1,034 段官方测试结果

CH-SIMS v2.0 全量复验于 2026 年 11 月 28 日完成。用千问内部表示训练分类头后，四路在准确率和 Macro-F1 上都超过了只校准千问选项分数的做法。纯声音也超过了同样训练的 emotion2vec+ 声音对照。画面的整体提升包含新增覆盖的贡献，完整输入的 Macro-F1 与同法校准的桌面版接近。[实测指标、混淆矩阵与配对区间](results/chsims-omni-supervision.json)

各行按该路人工标签计分。每格依次为判对数、准确率和 Macro-F1，分母均为 1,034，暂缓也计入分母。

| 输入条件 | 桌面固定版 | 桌面同法校准 | 千问直接判断 | 千问只校准分数 | 千问内部表示分类头 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 仅文字 | 710，68.67%，0.5959 | 718，69.44%，0.6618 | 554，53.58%，0.5408 | 708，68.47%，0.6344 | **740，71.57%，0.6868** |
| 仅声音 | 565，54.64%，0.4627 | 577，55.80%，0.4850 | 172，16.63%，0.0951 | 630，60.93%，0.5193 | **658，63.64%，0.6010** |
| 仅画面 | 375，36.27%，0.3455 | 417，40.33%，0.3990 | 188，18.18%，0.1026 | 530，51.26%，0.4888 | **621，60.06%，0.5570** |
| 完整输入 | 693，67.02%，0.5288 | 736，71.18%，0.6391 | 137，13.25%，0.0909 | 623，60.25%，0.5425 | 764，73.89%，0.6375 |

| 输入条件 | 桌面同法校准 → 优化千问的负向召回 | 中性召回 | 正向召回 | 暂缓数 |
| --- | ---: | ---: | ---: | ---: |
| 仅文字 | 351/460 → 361/460 | 100/193 → 112/193 | 267/381 → 267/381 | 12 → 8 |
| 仅声音 | 332/464 → 322/464 | 34/172 → 92/172 | 211/398 → 244/398 | 8 → 0 |
| 仅画面 | 241/447 → 318/447 | 5/188 → 70/188 | 171/399 → 233/399 | 436 → 0 |
| 完整输入 | 364/488 → 411/488 | 55/123 → 39/123 | 317/423 → 314/423 | 3 → 0 |

- **声音的全量增益集中在中性和正向。** 给桌面模型加入已有的 emotion2vec+ 语音向量并重新训练后，桌面与千问分别判对 590 和 658 段，Macro-F1 为 0.4919 和 0.6010。准确率差为 +6.58 个百分点，按来源配对重采样的 95% 区间为 +2.07 至 +11.02 个百分点。排除先导样本和全部 v1 测试编号后的 441 段上，两者分别判对 262 和 263 段，Macro-F1 为 0.5076 和 0.5608。[声音的同训练对照](results/chsims-omni-supervision.json)
- **声音在新来源上过度预测中性。** 在训练和验证中没有相同 `video_id` 的 190 段上，千问仅判对 70 段（36.84%），emotion2vec+ 对照判对 101 段（53.16%）。这组只有 22 段声音标为中性，千问判了 115 段中性。在声音与文字标注分歧的 242 段中，千问命中声音标签 115 段，桌面分数校准版命中 119 段，emotion2vec+ 向量对照命中 109 段。千问同时有 86 段命中了文字标签，说明对原音轨的总分仍无法单独量化语调贡献。[新来源及声音与文字分歧统计](results/chsims-omni-supervision.json)
- **画面总体提升主要来自补上桌面暂缓的片段。** 桌面暂缓的 436 段里，千问判对 248 段。在双方都有判断的 598 段中，桌面和千问分别判对 417 和 373 段，准确率为 69.73% 和 62.37%，Macro-F1 为 0.5354 和 0.5669。仅画面模型输入还包含动作和可见场景，指标对应整段画面标注。[共覆盖样本及逐类结果](results/chsims-omni-supervision.json)
- **同法校准后，文字与完整输入的总体差异较小。** 两路的准确率分别增加 2.13 和 2.71 个百分点，两项按来源的 95% 区间都跨零。完整输入多认出 47 段负向，中性少认出 16 段，Macro-F1 从 0.6391 变为 0.6375。测试集有 8 段缺少有效转写，文字条件暂缓，完整输入条件使用剩余的声画。[配对差值及三路均有证据的 1,026 段](results/chsims-omni-supervision.json)

## 全量监督的训练方法

全量实验使用 CH-SIMS v2.0 本地有标注版本的 **2,722 段训练、647 段验证和 1,034 段测试视频**。4,403 个编号和对应原视频的 SHA-256 均唯一。文字、声音、画面和完整输入各自匹配 `label_T`、`label_A`、`label_V` 和整段 `label`。文字只用 SenseVoiceSmall 自动转写，声音直接用 16 kHz 原音轨，画面只用静音并遮盖字幕的 RGB 帧。被遮蔽的通道不会送入对应编码器。声音条件中的语音仍含字词，因此另对比声音和文字标注分歧的片段。[数据与证据准备](../scripts/chsims_teacher_data.py)

先导实验的 128 段验证视频中，千问纯声音和纯画面均把中性排在第一位。四个选项合计已占到模型下一词概率的中位数 **99.81%** 和 **99.92%**，其中中性的选项内分数中位数为 **97.92%** 和 **96.00%**。全量实验比较直接取最高选项、只校准分类分数、以及利用千问内部表示学习分类。分数校准用负向、中性、正向之间的两个对数比。内部表示包括判断位置的上下文向量，以及各条件实际调用的音频或画面编码器的均值和标准差池化。千问权重固定，四个条件各自用官方训练集拟合带正则化的逻辑回归。验证集先按 Macro-F1，再按准确率选择特征组合、正则化和类别加权。训练和选参完成后冻结分类头，再评测测试集。[教师推理](../scripts/chsims_omni_teacher.py) · [训练、验证与评测](../scripts/chsims_teacher_supervision.py)

桌面固定权重版使用同批预处理证据，保留直接判断和按相同流程学习的分数校准两种参照。声音另加同训练对照，将桌面已有的 emotion2vec+ 语音向量与桌面分数一同输入分类头。暂缓计入总分母，Macro-F1 对三类等权。除官方测试全集外，同时统计先导实验以外的片段、训练和验证中没有相同 `video_id` 的片段，以及两边都有判断的片段。置信区间按 `video_id` 做配对重采样。

为后续蒸馏导出的训练监督采用五折分组。每次用四折训练分类头，对其余一折生成负向、中性、正向的三个概率，同一个 `video_id` 归到同一折。温度参数来自验证集，仅调整概率分布，保持类别排序。缺少该条件证据时保留空值。训练监督含片段编号、输入通道和概率，保存在本机实验目录。

验证集为文字选出对数比与上下文向量，为声音和画面选出各自编码器池化与上下文向量，为完整输入选出声画编码器池化与上下文向量。训练监督覆盖 129 个影视来源，文字有 2,695 段有效概率，声音、画面、完整输入各有 2,722 段。五折留出计算的 Macro-F1 分别为 0.7211、0.6407、0.5524 和 0.6662。这些是训练监督自身的质量读数，测试集成绩见上表。[验证集各组合及训练监督统计](results/chsims-omni-supervision.json)

用真实原片检查了四个条件实际调用的编码器，并验证单独替换音轨或帧会改变对应特征。用 scikit-learn 独立复算了官方测试集的 126 项指标。项目的离屏回归测试 **641 项通过**，启动检查和模拟调节闭环也通过。[实测与核验记录](results/chsims-omni-supervision.json)

### 复现全量监督

沿用下文的本机识别和独立 CUDA 环境。CUDA 环境额外安装 `torchaudio==2.8.0+cu128` 与 `onnxruntime-gpu==1.22.0`，以迁移音频模型和字幕检测到显卡。全量缓存放在 `.runtime/chsims-teacher/full`。其中 `teacher-supervision.json` 记录冻结的分类头，`teacher-training-targets.json` 保存按来源留出的训练监督，`supervision-test-predictions.json` 保存逐片测试判断。全量命令使用各官方划分的所有编号，输出按片段写入缓存，可以继续运行中断的步骤。

```powershell
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$env:CHSIMS_PREPARATION_THREADS = '1'
$data = 'D:\情绪数据集\ch-sims2s\ch-simsv2s'
$desktop = '.\.venv\Scripts\python.exe'
$teacher = '.\.runtime\chsims-teacher\venv\Scripts\python.exe'
$cache = '.runtime/chsims-teacher/full'
$benchmark = 'scripts/benchmark_chsims_teacher.py'
$supervision = 'scripts/chsims_teacher_supervision.py'
.\scripts\start_text_review.ps1 -PullEmbedding
foreach ($split in @('train', 'valid', 'test')) {
    & $teacher -B -X utf8 $benchmark prepare --data $data --cache $cache --split $split --prepare-device cuda
}
foreach ($split in @('train', 'valid')) {
    & $teacher -B -X utf8 $benchmark teacher --data $data --cache $cache --split $split --representations --batch-size 4
    & $desktop -B -X utf8 $benchmark baseline --data $data --cache $cache --split $split
    & $desktop -B -X utf8 $benchmark evaluate --data $data --cache $cache --split $split
}
& $desktop -B -X utf8 $supervision fit --cache $cache
& $teacher -B -X utf8 $benchmark teacher --data $data --cache $cache --split test --representations --batch-size 4
& $desktop -B -X utf8 $benchmark baseline --data $data --cache $cache --split test
& $desktop -B -X utf8 $benchmark evaluate --data $data --cache $cache --split test
& $desktop -B -X utf8 $supervision evaluate --cache $cache
```

若已有 CH-SIMS v1 的官方 `label.csv`，可在最后一条命令加上 `--v1-labels .runtime/chsims/label.csv`，同时统计排除旧版测试编号和先导样本后的结果。

## 256 段先导实验

在同一批 256 段测试视频上，双方使用相同的校准流程后，千问纯音频分支判对 **154 段**，桌面分支判对 **132 段**。Macro-F1 从 **0.4538** 升至 **0.5781**。画面分支覆盖了更多片段，但在两方都有判断的 150 段上，原桌面模型更准确。完整输入的两种校准模型各判对 162 段。[指标、混淆矩阵及参数](results/chsims-omni-teacher.json)

### 数据和输入

用户提供的 `D:\情绪数据集\ch-sims2s\ch-simsv2s` 是 CH-SIMS v2.0 有标注版本。本地 `meta.csv` 列出 2,722 段训练、647 段验证和 1,034 段测试，共 4,403 条唯一编号，均能找到对应的 MP4。论文统计为 4,402 段，本轮按实际文件统计。各官方划分按片段编号固定哈希排列，取 **256 段训练、128 段验证和 256 段测试**。三组片段编号互斥，影视来源有交集。测试含 54 个此前在 v1 实验中见过的编号，去掉它们后的 202 段另行统计。标注大于零映射到正向，等于零映射到中性，小于零映射到负向。[作者数据页](https://thuiar.github.io/sims.github.io/chsims) · [CH-SIMS v2.0 论文](https://arxiv.org/abs/2209.02604)

| 条件 | 实际送入千问的内容 | 对照的人工标签 |
| --- | --- | --- |
| 仅文字 | SenseVoiceSmall 本机自动转写 | 文字 `label_T` |
| 仅声音 | 原音轨解码为 16 kHz 单声道波形 | 声音 `label_A` |
| 仅画面 | 解码的 RGB 帧，遮盖嵌入画面的字幕区和检测到的文字 | 画面 `label_V` |
| 完整输入 | 同一份自动转写、音轨波形和处理后的画面 | 整段 `label` |

程序只解码 MP4 的视频和音频轨。画面每约 0.5 秒取样，长片均匀取至多 8 帧。固定遮盖底部 28%，再用本地文字检测器遮盖其他可见文字区域。抽查了预处理后的画面，这一步也可能遮住场景和人物的局部动作。教师读取按比例缩放后的帧，桌面视觉模型在同一批遮盖后帧上检测和裁剪人脸。桌面声音分支从同一音轨截取有效语音。测试集有 4 段没有有效转写，文字条件记暂缓，完整输入条件使用剩下的声画。模型只接收自动转写和媒体证据。`meta.csv` 的四路人工标签用于校准及计分。[证据准备代码](../scripts/chsims_teacher_data.py)

### 模型和校准

教师选择 **Qwen2.5-Omni-3B** 的 Thinker，能够直接编码声音波形和视频帧。固定发布权重，使用 RTX 5070 Ti、CUDA 12.8 和 BF16，在独立的项目环境中本地运行。模型对负向、中性、正向和证据不足四个选项给出相对分数。训练组生成 1,020 份有输入的教师分数，验证组生成 512 份，测试组生成 1,020 份。之后的蒸馏实验可以从训练组缓存读取这些分数。[千问官方模型](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) · [教师推理代码](../scripts/chsims_omni_teacher.py)

在验证集，原始千问把 128 段纯音频和 128 段纯画面全部判为中性，其他类别分数仍随输入变化。为检验这些分数是否包含可用于分类的信息，每个分支用两项对数比作为特征，分别是负向相对中性、正向相对中性。用训练标签拟合带正则化的多项逻辑回归，用验证集 Macro-F1 和准确率选参数。教师校准在读取测试得分之前冻结。为了核查域内校准本身能带来多少收益，对桌面固定权重 v1 的原始分数补做同样流程，训练及选参仍只用训练、验证组。校准对象是输出分数，千问权重保持冻结。有可用分数的片段由校准器在三类中选择，缺输入的片段保留暂缓。[校准实现](../scripts/calibrate_chsims_teacher.py)

### 同一批测试视频的结果

每格为「判对数 / 256，Macro-F1」。每种条件采用该路人工标签。暂缓计入总分母，Macro-F1 对三类等权。[完整的四路统计](results/chsims-omni-teacher.json)

| 条件 | 桌面固定版 | 千问直接判断 | 桌面同法校准 | 千问校准后 |
| --- | ---: | ---: | ---: | ---: |
| 仅文字 | 172，0.5887 | 134，0.5331 | 172，0.6238 | 172，0.6147 |
| 仅声音 | 124，0.4050 | 50，0.1089 | 132，0.4538 | **154，0.5781** |
| 仅画面 | 86，0.3320 | 49，0.1071 | 107，0.4005 | 135，0.3769 |
| 完整输入 | 171，0.5565 | 42，0.1066 | 162，0.5979 | 162，0.5666 |

| 条件 | 桌面与千问同法校准后负向召回 | 中性召回 | 正向召回 | 暂缓数 |
| --- | ---: | ---: | ---: | ---: |
| 仅文字 | 88/109 → 91/109 | 18/51 → 16/51 | 66/96 → 65/96 | 6 → 4 |
| 仅声音 | 75/107 → 72/107 | 9/50 → 26/50 | 48/99 → 56/99 | 4 → 0 |
| 仅画面 | 56/104 → 94/104 | 0/49 → 0/49 | 51/103 → 41/103 | 106 → 0 |
| 完整输入 | 70/111 → 84/111 | 21/39 → 13/39 | 71/106 → 65/106 | 2 → 0 |

- **声音的增益集中在中性和正向。** 同样校准后准确率从 51.56% 升到 60.16%，提升 8.59 个百分点。按影视来源配对重采样 2,000 次，差值的 95% 区间为 +0.43 至 +17.29 个百分点，Macro-F1 差值区间为 +0.0386 至 +0.2174。仅看未出现在历史测试中的 202 段，判对数为 101 → 118，Macro-F1 为 0.4348 → 0.5649，准确率差值区间跨零。[配对结果](results/chsims-omni-teacher.json)
- **纯音频仍包含字词。** 在声音和文字人工标签不同的 62 段中，校准后的桌面和千问分别命中声音标签 28、33 段，恰好命中文字标签的次数均为 18。这一压力测试提示模型利用了部分对音轨有用的信息，语速、语调和口语语义的各自贡献还需要控制实验。[分歧样本统计](results/chsims-omni-teacher.json)
- **画面增益主要体现在覆盖。** 千问对所有片段给出判断，桌面有 106 段暂缓。在两边都判断的 150 段上，桌面判对 107 段，千问判对 82 段。两种校准模型都没有认出中性，千问整体 Macro-F1 也低于同法校准桌面。[共覆盖样本和逐类结果](results/chsims-omni-teacher.json)
- **完整输入的优势未建立。** 与桌面固定版相比，校准千问少判对 9 段。双方同法校准后准确率持平，千问 Macro-F1 较低。仅保留转写、音轨和帧都存在的 252 段，双方判对 160 与 162 段，Macro-F1 为 0.5983 与 0.5751。[完整输入分组](results/chsims-omni-teacher.json)

### 运行核验与文件位置

用真实训练原片检查了模型的编码器调用。纯文字时声音和画面编码器均未执行，纯声音只调用声音编码器，纯画面只调用画面编码器，完整输入两者都调用。保持文字不变，换入其他片段的音轨或画面，含该证据的分支改变了分数。独立用 scikit-learn 复算了四路两种校准模型的 16 个准确率和 Macro-F1 数值。项目回归测试通过 **639 项**。[编码器核验及指标](results/chsims-omni-teacher.json)

实验的独立 CUDA 环境位于 `.runtime/chsims-teacher/venv`，权重位于 `omni-3b`，自动转写及处理后的证据位于 `media`，四路教师输出位于 `teacher`。三组逐片对照为 `evaluation-train.json`、`evaluation-valid.json`、`evaluation-test.json`。`teacher-calibration.json` 保存已冻结的校准参数。原片继续从 D 盘读取，实验过程仅在本机使用音视频。

### 复现先导对照

在项目根目录使用现成的两套环境。首次在其他 Windows 机器准备时，可用 PowerShell 7 运行 `scripts/download_chsims_teacher.ps1` 下载并校验权重。教师环境使用 Python 3.12、CUDA 版 PyTorch `2.8.0+cu128`、torchvision `0.23.0+cu128`、Transformers `4.57.6`、Accelerate `1.12.0` 和 NumPy `1.26.4`。在项目原有环境中准备 OCR 依赖到隔离目录 `--target .runtime/chsims-teacher/ocr`，版本为 `rapidocr-onnxruntime==1.4.4`、`pyclipper==1.3.0.post6`、`shapely==2.1.2`。对照使用项目自带真实模型和本机 Ollama，启动方式见下方命令。[CUDA 版本安装指令](https://pytorch.org/get-started/previous-versions/) · [权重下载脚本](../scripts/download_chsims_teacher.ps1)

```powershell
$data = 'D:\情绪数据集\ch-sims2s\ch-simsv2s'
$desktop = '.\.venv\Scripts\python.exe'
$teacher = '.\.runtime\chsims-teacher\venv\Scripts\python.exe'
$experiment = 'scripts/benchmark_chsims_teacher.py'
$counts = [ordered]@{ train = 256; valid = 128; test = 256 }
.\scripts\start_text_review.ps1 -PullEmbedding
foreach ($split in $counts.Keys) {
    & $desktop -B -X utf8 $experiment prepare --data $data --split $split --limit $counts[$split]
}
& $teacher -B -X utf8 $experiment teacher --data $data --split train --limit 4 --batch-size 2
foreach ($split in $counts.Keys) {
    & $teacher -B -X utf8 $experiment teacher --data $data --split $split --limit $counts[$split] --batch-size 4
    & $desktop -B -X utf8 $experiment baseline --data $data --split $split --limit $counts[$split]
}
foreach ($split in @('train', 'valid')) {
    & $desktop -B -X utf8 $experiment evaluate --data $data --split $split --limit $counts[$split]
}
& $desktop -B -X utf8 scripts/calibrate_chsims_teacher.py fit
& $desktop -B -X utf8 scripts/calibrate_chsims_teacher.py fit --arm baseline
& $desktop -B -X utf8 $experiment evaluate --data $data --split test --limit 256
& $desktop -B -X utf8 scripts/calibrate_chsims_teacher.py evaluate
& $desktop -B -X utf8 scripts/calibrate_chsims_teacher.py evaluate --arm baseline
```
