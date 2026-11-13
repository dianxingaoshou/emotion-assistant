# Git 日常维护

仓库地址为 [dianxingaoshou/emotion-assistant](https://github.com/dianxingaoshou/emotion-assistant)。以下命令在项目根目录运行。本机现有项目可继续使用原目录，从 GitHub 新克隆时目录默认为 `emotion-assistant`。

`main` 保存可运行版本，日常修改使用功能分支，通过 Pull Request 合并。GitHub 自动检查运行离屏软件测试，不连接真人采集设备或下载大模型。

## 开始一次修改

先检查工作区，确认没有尚未处理的改动，再更新主分支：

```powershell
git status
git switch main
git pull --ff-only
git switch -c fix/evidence-window
```

分支名按本次任务替换，例如 `feat/new-feedback-view`、`fix/storage-retry`、`docs/setup-guide`。若工作区已有修改，先查看 `git diff` 并在当前分支妥善提交；不要通过重置或覆盖文件来清理它们。`pull --ff-only` 拒绝有分歧的合并时，先查看分支历史和状态，再处理已有提交。

## 验证并提交

修改后运行相关检查。应用代码的完整离屏回归使用项目解释器：

```powershell
.\.venv\Scripts\python.exe scripts/verify.py
git status
git diff
```

逐项暂存自己检查过的文件。下面以融合逻辑及其测试为例：

```powershell
git add emotion_assistant/core.py tests/test_core.py
git diff --cached
git commit -m "fix: correct effective evidence duration"
git push -u origin fix/evidence-window
```

将示例路径、分支名和提交说明替换成本次改动。提交说明应说清修改的行为；后续向同一分支推送时直接使用 `git push`。

## 合并并继续维护

在 GitHub 为刚推送的分支创建 Pull Request，目标选 `main`，写明问题、修改行为和验证结果。自动检查通过、确认改动可用后合并。

回到本机同步合并结果：

```powershell
git switch main
git pull --ff-only
git branch -d fix/evidence-window
git status
```

本地分支确认已合并后再删除。下一项工作重新建立分支，避免把无关修改持续堆在同一个分支中。

## 回看与修正

```powershell
git log --oneline --decorate -10
git show HEAD
git diff main...HEAD
```

已推送的错误修改优先通过新的修复提交处理。需要撤销整个已合并提交时，先查看目标提交，再在修复分支使用 `git revert <提交号>`，验证后通过 Pull Request 合并。不要强制推送覆盖 `main` 历史。

## 本机数据与环境

| 纳入 Git | 仅保留在本机 |
| --- | --- |
| `emotion_assistant/`、必要资源、`tests/`、`scripts/`、`docs/` | `.venv/`、`models/`、`.runtime/`、`data/`、`reports/` |
| README、依赖清单、安装与启动脚本、版本管理配置 | 模型权重、SQLite 数据库、会话导出、运行日志、私人素材与凭证 |

仓库只保存项目源码及重建方法。换电脑时重新克隆和安装环境，再按需下载模型；个人会话与实验数据需要单独备份。SQLite 数据库应在应用关闭后备份，或使用 SQLite backup API 创建一致性快照，避免只复制仍在写入的数据库文件。

若文件意外出现在 `git status` 中，先检查 `.gitignore` 和文件用途，不要用 `git add -f` 绕过排除规则。已经进入 Git 历史的私密内容即使后来删除文件也不会自动消失；应停止继续推送，先处理泄露范围与相应凭证。
