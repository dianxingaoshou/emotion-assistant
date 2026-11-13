# 本地模型与可复现配置

模型权重、运行环境和数据均保留在本项目中。项目 `.venv` 使用 Python 3.12；不要向系统 Python 安装这些依赖。应用运行时只读取 `models/` 的已下载文件，不按模型名称联网拉取权重。下载是单独执行的准备步骤。

## 安装与下载

在项目根目录的 PowerShell 中执行。安装脚本优先使用完整的 `requirements-lock.txt`，仅在项目内创建或复用 Python 3.12 `.venv`。

```powershell
.\setup.ps1
# 没有 Python Launcher 时，明确指定本机已有的 Python 3.12：
.\setup.ps1 -PythonExe 'D:\python\python.exe'
```

也可在已存在的项目 `.venv` 中用 `python -m pip install -r requirements-lock.txt` 复现完整依赖。`requirements-base.txt`、`requirements-models.txt` 记录分层的直接依赖，`requirements-lock.txt` 记录本次验证的传递依赖。此次实际安装的 PyTorch/torchaudio 均为 PyPI 2.6.0，`torch.version.cuda` 为 `None`；所有适配器明确使用 CPU。没有显卡也可以运行；本项目尚未用真实设备验证最低配置或实时性能。

下载器仅使用 Python 标准库，直接写入项目 `models/`，不用用户目录下的 Hugging Face 或 ModelScope 缓存。

```powershell
# 默认下载视觉、语音情绪及 VAD
& '.\.venv\Scripts\python.exe' scripts/download_models.py
# 可以单独准备一个模块
& '.\.venv\Scripts\python.exe' scripts/download_models.py --visual
& '.\.venv\Scripts\python.exe' scripts/download_models.py --audio
& '.\.venv\Scripts\python.exe' scripts/download_models.py --asr
```

全部权重约 2 GiB，emotion2vec+ seed 原始检查点约 1.12 GB，SenseVoiceSmall 约 0.94 GB。断线后重复原命令，下载器保留 `.part` 并尝试断点续传。完整文件只有通过 SHA256 才会成为正式权重；下载完成写入 `models/manifest.json`。文件已存在时重复执行会校验，无需重新下载。若报告校验不符，先核查并移走该文件再重试，不能跳过校验。

## 固定的官方来源

| 用途 | 模型及版本 | 本地位置 | 实现依据 |
|---|---|---|---|
| 人脸检测 | OpenCV YuNet 2023mar；仓库 `48b10fa505aeb51fbd127fc60a5c660040e823b7` | `models/visual/face_detection_yunet_2023mar.onnx` | [OpenCV 官方模型与许可证](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet) |
| M1 表情 | EmotiEffLib `enet_b0_8_best_afew`；`520a051c64cd191521e5934655314e769a319684` | `models/visual/enet_b0_8_best_afew.onnx` | [作者的模型及推理源码](https://github.com/sb-ai-lab/EmotiEffLib/tree/520a051c64cd191521e5934655314e769a319684) |
| M2 语音情绪 | emotion2vec+ seed；HF `7be9a30d52bdc793cc24e6f7d25358d6648927c1` | `models/emotion2vec_plus_seed/` | [作者模型卡、类别和 16 kHz 示例](https://huggingface.co/emotion2vec/emotion2vec_plus_seed/tree/7be9a30d52bdc793cc24e6f7d25358d6648927c1) |
| 可选中文转写 | SenseVoiceSmall；HF `3847d57b6bdf2dd8875cb1508d2af43d80a16bf7` | `models/SenseVoiceSmall/` | [作者模型卡与本地路径用法](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/3847d57b6bdf2dd8875cb1508d2af43d80a16bf7) |
| 语音活动检测 | Silero VAD v5.1.2；`6478567951ae5c9979ad7b234185b5515f4be7a1` | `models/vad/silero_vad.onnx` | [作者 ONNX 状态与上下文接口](https://github.com/snakers4/silero-vad/blob/6478567951ae5c9979ad7b234185b5515f4be7a1/src/silero_vad/utils_vad.py) |

下载脚本为每个文件固定 SHA256，而不仅固定名称。HF 的两个大权重在国内网络较慢，因此下载器优先尝试同一作者的 [ModelScope emotion2vec+ seed](https://modelscope.cn/models/iic/emotion2vec_plus_seed) 和 [ModelScope SenseVoiceSmall](https://modelscope.cn/models/iic/SenseVoiceSmall)；只有内容完全匹配已固定的 HF SHA256 才接受。ModelScope 的 `master` 本身不是固定版本，校验值才是此镜像的版本约束。失败后尝试 HF 固定提交。清单同时记录规范来源、镜像和实际下载地址。

主要大文件校验值：

| 文件 | SHA256 |
|---|---|
| YuNet | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| EmotiEffLib ONNX | `7aa2ea31c1311f4f8aa9d3fdb085d418dd4e7a48c4b9ed41df8c044f91d0213f` |
| emotion2vec+ model.pt | `327e861049c7105a3f8589edfd16543a7cdc0c1926f1b64de975d98e0650cdd3` |
| SenseVoiceSmall model.pt | `833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea` |
| Silero VAD | `2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f` |

各模型的再分发和使用条件以表中的官方模型仓库、模型卡及许可证为准。

## 推理约定

`emotion_assistant.models.ModelHub` 提供 `load_visual()`、`load_audio()`、`analyze_frame()` 和 `analyze_audio()`。导入模块不会加载 PyTorch 或 FunASR；缺失依赖、权重、校验记录或模型加载失败均返回明确不可用状态。没有模拟情绪分数的替代路径。

视觉输入是 `uint8` BGR 图像。YuNet 仅检测与提供关键点；没有人脸或存在多个人脸时拒识。表情模型按作者 ONNX 实现使用 RGB、224×224 线性缩放、`/255`、ImageNet 均值与标准差、NCHW。直接运行完整 ONNX 分类头，再对 8 个 logits 做一次 softmax；无需安装 EmotiEffLib 的训练框架。Windows 中文路径通过读取模型字节再调用 OpenCV 的缓冲区接口处理。

语音输入必须是已经真正重采样的 16 kHz 单声道 float32 数组，时间戳长度应与样本数一致。FunASR 的 emotion2vec+ 模型执行 utterance 分类，关闭特征文件导出。原始类别分数和统一分布同时保存。SenseVoice 的 SentencePiece 在此 Windows 中文路径下同样需要从字节加载；适配器注册独立的本地 tokenizer 子类实现此操作，不修改第三方包或官方 tokenizer 类。

统一标签映射：`happy/happiness` 为正向；`neutral` 为中性；愤怒、蔑视、厌恶、恐惧、悲伤为负向；惊讶、`other`、`unknown` 和未识别标签保留为不确定。映射保留全部概率质量，不会丢弃“不确定”后制造高置信度分数。分数是可观察线索，未经目标场景校准，不能等同于用户真实心理状态的概率。

当前视觉门控包括单人、最小脸框 70 像素、灰度均值 30–225、拉普拉斯方差至少 25、明显姿态异常。音频门控包括时长、时间戳一致性、有限值、削波比例和音量；少于 2 秒不输出声学情绪。质量指标仅是工程量，不是情绪准确率。阈值为首版初始配置，后续须用授权数据检验和调整。采集层另外使用 Silero VAD 分段；VAD 不验证说话人身份。

ASR 是辅助记录模块。只有 `enable_asr=True` 且调用时 `asr_allowed=True` 才执行转写；未授权时不调用 `generate`。转写中的情绪标签和声学事件标签被去除，不作为另一份独立情绪证据。ASR 失败不让有效的 M2 结果失效。

## 已核查的启动提示

FunASR 1.2.6 加载 emotion2vec+ seed 时会打印 10 项 `miss key in ckpt`，均以 `modality_encoders.AUDIO.decoder.` 开头，涉及 `blocks.0.0` 至 `blocks.3.0` 和 `proj` 的 weight/bias。这是预训练重建 decoder，并非情绪分类输出层 `Emotion2vec.proj`。本次对固定官方检查点逐项核查：185 个其余 state-dict 张量均与已加载模型完全相等，只有上述 10 个未参与推理的 decoder 张量缺失。

依据为本项目安装的官方 FunASR 1.2.6 源码：`models/emotion2vec/model.py:236` 从 `inference()` 调用 `extract_features()`；`:183–188` 使用 `features_only=True, mask=False`；`:249–255` 使用顶层分类 `self.proj` 和 softmax。`models/emotion2vec/base.py:282–304` 的特征前向只调用局部特征与上下文编码；`:419–421` 的 `remove_pretraining_modules()` 明确把 `self.decoder` 作为可删除的预训练模块。可对照 [FunASR 官方 emotion2vec 推理实现](https://github.com/modelscope/FunASR/blob/main/funasr/models/emotion2vec/model.py) 和 [官方预训练模块实现](https://github.com/modelscope/FunASR/blob/main/funasr/models/emotion2vec/base.py)。这里的行号以已锁定的本地 1.2.6 包为准，在线主分支行号可能变化。

另外实际给 decoder 注册了“只要被调用就抛错”的 forward pre-hook，再对官方语音完成一次完整 `generate`：decoder 调用数为 0，推理有效，与未加 hook 的原始类别分数最大绝对差为 0。没有更改模型、检查点或第三方包来隐藏提示。此结论仅针对这 10 项已核查的 decoder 缺项；其他缺项仍应另行检查。

`ffmpeg is not installed` 是 FunASR 导入时对可选文件解码工具的提示。本项目把已重采样的 NumPy 音频数组传入模型，官方 `utils/load_utils.py:122–123` 直接用 `torch.from_numpy` 处理此分支，不调用 FFmpeg；烟测素材由 soundfile 读取。因此当前路径无需另装系统 FFmpeg。相关分支可对照 [官方音频加载源码](https://github.com/modelscope/FunASR/blob/main/funasr/utils/load_utils.py)。

## 检查范围

开发时的模型检查使用官方示例素材，保存在 `.runtime/test_assets/`，结果写入 `reports/model-smoke.json`。该文件只证明列出的模型能完成本地推理及相应门控，不代表摄像头、麦克风真实使用验收，也不证明识别准确率、融合增益、干预有效性或长时间性能。单次延迟记录是烟测过程值，不能写成 P95 或实时性结论。

```powershell
# 默认只使用已存在并校验通过的素材，不联网
& '.\.venv\Scripts\python.exe' scripts/model_smoke.py
# 缺少素材时，显式允许下载三项官方示例（固定校验值）
& '.\.venv\Scripts\python.exe' scripts/model_smoke.py --download-samples
# 另外验证实际 spawn 推理进程、队列收发与进程退出，不打开设备
& '.\.venv\Scripts\python.exe' scripts/model_smoke.py --workers
```

检查包括 M1、M2、ASR 的实际推理，单人/多人/无人脸拒识，以及 Silero VAD 对静音与官方语音的实际分段区间。`--workers` 对视觉和音频分别使用不同的示例会话，不将不相关照片与声音融合。门控和转写授权的快速回归测试在 `tests/test_models.py`，不在每次单元测试中加载大模型。

真实设备验收仍需自愿用户完成一次采集、确认、自评、调节、复测，并检查关闭后设备释放。后续应另做独立会话的延迟统计、质量门控测试与四组消融对照。
