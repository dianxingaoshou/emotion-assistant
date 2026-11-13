# 核心审查补丁与现有界面的集成

本轮依据用户提供的审查说明、测试日志，以及随后提供的 `emotion_optimization_patch.zip`，在现有工程中增量合并。保留项目 Python 3.12 虚拟环境、模型权重与已有入口，不执行附件中的示例路径命令，也不将补丁包当成完整应用覆盖。

源码和数据库已在修改前备份到项目内 `.runtime/backups/`。补丁解包放在 `.runtime/incoming/`，原始 54 项补丁测试作为补丁自身基线单独保留；真实工程的测试从 `emotion_assistant` 导入，以免测试到解包目录里的另一份源码。

## 检测、融合与有效证据

| 位置 | 接口与处理 |
| --- | --- |
| `evidence.py` | `raw_prediction_diagnostics()` / 兼容别名 `distribution_diagnostics()` 保留归一化熵、top1、margin；原始近均匀输出不参加决策。缺失原始分布明确标记为不可诊断，不凭空补出诊断结果 |
| `core.py` | `FusionEngine` 保留原有方法，融合前检查原始分散度；保留正负冲突检查，并补入 JSD 与非 uncertain 主类别分歧。输出模型版本、原始证据 ID、各模态区间和融合配置摘要 |
| `evidence.py` | `SustainedEvidenceGate` 使用实际有效区间的并集；重叠与重复不增加时长。默认有效负向证据 20 秒、密度至少 0.8、允许缺口 1.5 秒、有限等待 3 秒。所有阈值都是待验证的工程参数 |
| `controller.py` | `_consider_prompt()` 消费证据门控结果；音频待融合与迟到视觉修订使用同一 `audio_event_id`，使原事件的结论可以修订而不变成一条新独立证据 |

门控保留有限历史中的迟到修订；最终链末端仍须满足新鲜度要求，单独到达的陈旧证据不能触发提示。暂停或旧采集世代标记的结果不进入台账。超过有限等待预算的结果无法撤回已经发出的提示。

`fuse_audio()` 不缓存声学结果。控制器持有有界待融合音频，先等待视觉水位或有限期限，再发布；后续视觉到达时，可用同一声学原始记录重算。原始 `event_id` 和 `audio_event_id` 用于关联修订，原始类别分布始终保留，ASR 标签不作为第三个情绪投票。

## 声学与转写的实际接线

`models.py` 新增 `load_audio_emotion()`、`load_asr()`、`analyze_audio_emotion()` 与 `transcribe_audio()`。旧 `analyze_audio()` 保留，作为兼容接口仍会等待其内部调用完成。

实际运行的 `capture.py` 改为两个独立推理通路：声学情绪和 ASR 分别使用有界队列及推理进程。声学结果完成后立即发布，转写完成后通过独立 utterance 事件发布。队列满、任务跳过及模型超时会产生明确状态；暂停世代、停止状态、转写授权及授权撤回时间共同阻止旧任务继续发布。记录排队、推理与端到端时间，避免用单模型耗时代替界面等待。

转写撤回入口接到现有 UI，数据库同步更新授权状态；撤回后不保存新的文本。已授权历史按原有数据管理规则保留，可通过会话删除处理。

## 独立前后观察窗

新增 `assessment.py`：

```text
确认需要后开始 pre 固定观察窗
    ↓ 开始活动（可主动跳过未完成的前窗）
during 仅保留过程观察
    ↓ 点击“结束练习，开始复测”
重新开始 post 固定观察窗
    ↓ 窗口结束，或主动跳过
保存用户自评、模型分布变化、前后可比性三个独立结果
```

`AssessmentWindowCollector.begin()/add()/snapshot()/finish()` 使用会话相对单调时钟。默认前后各 30 秒，覆盖门槛为 0.5，并检查每个参与模态的覆盖。跨窗口边界的整段模型输出不参与窗口统计：仅裁剪时间戳不能去除该输出已包含的活动期间内容。

窗口保存 `assessment_id`、所属活动与阶段、固定起止时间、来源 ID、有效秒数、各模态覆盖、模型版本、融合配置摘要和排除原因。缺失、冲突、系统播放、暂停或跨窗口证据不补成中性；模型或配置在窗内变化则标记数据不足。

`compare_assessments()` 要求完整且独立的 pre/post 窗、相同会话与活动、相同模态组合、模型版本和融合配置。返回 `comparable`、`reasons` 和四类 `model_delta`。该变化只是辅助模型分布的差异，不能代替用户感受，也不能推断治疗效果。

UI 的“结束练习，开始复测”接到 `Controller.begin_post_assessment()`；“返回练习”接到 `resume_activity_guidance()`；“跳过观察窗”接到 `skip_assessment()`。旧调用未显式开始独立 post 窗时，仍可保存自评，但模型比较必须返回不足，不能使用活动期间最近一次结果冒充后测。

## 反馈统计与可靠保存

`regulation.py` 先在每次独立活动内配对即时和延迟反馈，再跨活动平均；只有有效即时或延迟反馈的活动并集才进入独立样本数。没有数值后测时，明确主观帮助反馈仍可统计。不适跨情境保留，推荐优先参考相同需要的历史；换方法链保留过程而不归因到最后一种方法。缺少来源的历史保持 unknown，不迁成 real。

`InterventionFlow` 将 `numeric_evaluation` 与最终 `evaluation` 分开，不适优先，取消路径同样保留不适。界面结果卡、历史摘要和方法统计使用同一状态语义。

`Store.save_activity_bundle()` 将活动、前后自评、明确偏好和完成事件放入同一 SQLite 事务；`complete_followup()` 在一个事务中完成延迟记录、自评与事件，并校验档案和模式、避免重复提交。新增的窗口和诊断字段随活动／观察 JSON 完整持久化，旧记录继续兼容。

控制器写入完成回调通过 Qt 队列信号回到所属线程。界面显示待保存、保存成功、失败与重试，成功只在事务提交后出现；停止设备或结束记录异常时不发送“设备已释放”的成功结论。

## 复现与验证边界

在项目根目录使用现有解释器：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\verify.py
.\.venv\Scripts\python.exe scripts\regulation_smoke.py
.\.venv\Scripts\python.exe scripts\ui_smoke.py
.\.venv\Scripts\python.exe scripts\review_ui_smoke.py
.\.venv\Scripts\python.exe scripts\model_smoke.py --workers --report reports\review\model-workers.json
```

补丁原版的 54 项测试与真实工程集成结果分别记录。`tests/test_review_patch_integrated.py` 保留 50 项可直接运行在真实包上的补丁逻辑／模型替身测试；原补丁 4 项 AST 回调测试由真实 Qt 线程、数据库失败与事务验证替代。独立窗口、迟到修订、实际双路调度与 UI 新接口另有集成验证，不把旧补丁日志当作本轮运行结果。

尚未进行概率校准、运行中的个人基线学习、语言语义情绪理解，也未执行真人效果、识别准确率或正式消融实验。原始分布熵、JSD、有效覆盖与规则排序均不是正确率或调节成功率；未下载新模型或新增必须安装的第三方依赖。

`start_activity()` 目前保留有界历史和偏好的同步读取，保证旧公共调用的即时行为；尚未以大规模数据库测量该读取对主线程的影响。本轮不声称已经消除所有主线程数据库访问或完成正式延迟基准。

## 本轮实际验收结果

补丁 manifest 的四个原文件 SHA-256 均与修改前备份一致。备份位置为 `.runtime/backups/review-20261113-152525/`，其中 `source.zip` 保存原工程源码，`sessions.sqlite3` 通过 SQLite backup API 保存一致性快照；没有以文件复制替代正在使用数据库的快照。

| 检查 | 本轮结果 | 证据与范围 |
| --- | --- | --- |
| 解包后的原始补丁 | 54 项通过 | 使用项目 Python 3.12 执行原始 unittest；与工程集成回归分别验证 |
| 真实工程自动化回归 | 241 项通过 | `reports/acceptance.json`、`reports/tests.xml`；包含真实包的 50 项补丁用例、Qt 线程和数据库故障注入 |
| 新接口与 UI 契约 | 97 项通过，14 张截图 | `reports/review/ui-contracts.json`；1280／1040px、观察窗、保存失败重试、授权撤回、未知来源 |
| 既有 UI／闭环／布局 | 47／63／12 项通过 | `reports/ui-smoke-result.json`、`reports/regulation/result.json`、`reports/review/layout/layout.json` |
| 官方样例直接推理 | 通过 | `reports/review/model-smoke.json`；表情、拒识、声学、ASR、VAD，未打开设备 |
| 三条真实推理进程与 IPC | 通过，均正常退出 | `reports/review/model-workers.json`；visual／audio／asr 独立加载，声学不负责转写 |

原 UI 验收脚本增加了等待真实保存回调的检查，再允许下一次活动；实时视觉派发也记录真实入队时刻，避免将近似零的等待时间显示为实测值。以上检查不等同于真人音视频会话、准确率、调节疗效或正式端到端延迟基准。
