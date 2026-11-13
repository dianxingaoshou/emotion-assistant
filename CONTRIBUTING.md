# 参与维护

请使用 Windows x64 和项目的 Python 3.12 虚拟环境。首次安装见 [README](README.md)；Git 命令与分支操作见 [维护流程](docs/GIT_WORKFLOW.md)。

## 修改与提交

1. 从最新 `main` 建立 `feat/`、`fix/` 或 `docs/` 分支，每次提交围绕一个明确问题。
2. 修改检测、融合、调节或存储逻辑时，保留接口兼容，覆盖相关正常路径和失败路径。接口或字段变化应同时核对 UI、持久化和调用脚本。
3. 运行与改动相关的验证；合并应用代码前运行完整离屏回归：

   ```powershell
   .\.venv\Scripts\python.exe scripts/verify.py
   ```

   UI 改动还应运行：

   ```powershell
   .\.venv\Scripts\python.exe scripts/ui_smoke.py
   .\.venv\Scripts\python.exe scripts/regulation_smoke.py
   .\.venv\Scripts\python.exe scripts/review_ui_smoke.py
   ```

   只改文档时检查命令、链接与说明即可。真实模型检查按 [模型说明](docs/MODELS.md) 单独进行，不以模拟输出替代。

4. 用 `git status`、`git diff` 和 `git diff --cached` 检查本次内容，只暂存与改动有关的文件。
5. 在 Pull Request 中说明具体问题、修改后的行为、已运行的检查及其验证范围，等待自动检查通过后合并。

## 文件与数据

- 提交源码、必要资源、测试、文档、依赖清单和下载脚本。
- `.venv/`、`models/`、`.runtime/`、`data/`、`reports/` 保留在本机；不要为了提交临时文件使用 `git add -f`。
- 不提交密钥、访问令牌、个人会话、录音、视频、转写、数据库或带有个人信息的报告。复现问题优先使用明确标记的合成输入。
- 调整依赖时同步维护直接依赖和锁定清单，并在独立环境中验证。模型仍通过固定版本与 SHA256 下载，不把权重加入 Git 历史。

保持模型观察、用户自评和主观反馈分别记录。缺失数据不补为改善，不适不能被数值下降覆盖；测试报告应区分软件流程、真实模型样例和真人实验。

第三方模型与素材遵循各自来源和许可。仓库公开不代表模型权重或其中引用材料自动获得新的使用授权。
