# 私有开发仓库与公开展示仓库同步方案

## 目标

`SGDAgent` 是唯一开发主仓库，`SGDAgent-release` 是未来公开展示面。两个仓库不做普通 merge。公开仓库只通过 allowlist 同步稳定、可公开、可复现的内容。

## 仓库角色

私有开发仓库：

- GitHub：`toniekule/SGDAgent`
- 本地：`/Users/toniekule/Dev/SGDAgent`
- 默认分支：`main`
- 用途：功能开发、实验、论文、benchmark、日志分析、内部计划

公开展示仓库：

- GitHub：`toniekule/SGDAgent-release`
- 本地：`/Users/toniekule/Dev/SGDAgent-public-clean`
- 默认分支：`main`
- 用途：公开 README、核心代码、最小测试、可复现实验说明

## 同步原则

- 公开仓库永远不从私有 `main` 直接 merge。
- 公开仓库只复制 allowlist 文件。
- 私有仓库可以包含论文、实验、日志、失败尝试；公开仓库不包含这些内容。
- 每次同步前先在私有仓库完成测试和 baseline 判断。
- 每次同步后在公开仓库独立跑测试、打包和敏感扫描。

## Allowlist

默认允许同步：

- `README.md`
- `BASELINE.md`
- `PUBLIC_MANIFEST.md`
- `.gitignore`
- `requirements.txt`
- `package_submission.sh`
- `local_run.py`
- `sgd/`
- `tests/`
- `evaluation/` 中可以独立运行且不依赖私有 artifacts 的文件

需要人工确认后同步：

- 新增公开 demo 脚本
- 新增公开 benchmark 文档
- 新增图片或小型示例资产
- LICENSE

禁止同步：

- `docs/plans/`
- `docs/baselines/` 中含本地绝对路径或私有 run id 的详细记录
- `paper/`
- `video_database/`
- `dist/`
- `.claude/`
- `.env`, `.env.*`
- benchmark logs, traces, run summaries, reports
- `/Users/toniekule/Dev/dvd_db` 下的任何文件

## 标准同步流程

1. 在私有仓库完成开发。

```bash
cd /Users/toniekule/Dev/SGDAgent
git status --short --branch
python -m pytest -q
rg "from dvd|import dvd|dvd\\."
find . -type f -size +10M -not -path './.git/*' -not -path './.venv/*' -print
```

2. 确认哪些改动可公开。

```bash
git diff --stat origin/main..HEAD
```

只选择 allowlist 中的稳定内容进入 release。

3. 同步到公开 worktree。

```bash
cd /Users/toniekule/Dev/SGDAgent-public-clean
git status --short --branch
```

用明确路径复制，不使用 merge：

```bash
rsync -a --delete /Users/toniekule/Dev/SGDAgent/sgd/ ./sgd/
rsync -a --delete /Users/toniekule/Dev/SGDAgent/tests/ ./tests/
rsync -a --delete /Users/toniekule/Dev/SGDAgent/evaluation/ ./evaluation/
cp /Users/toniekule/Dev/SGDAgent/local_run.py ./local_run.py
cp /Users/toniekule/Dev/SGDAgent/requirements.txt ./requirements.txt
```

只在确认文档适合公开时同步：

```bash
cp /Users/toniekule/Dev/SGDAgent/README.md ./README.md
```

4. 公开仓库审计。

```bash
python -m pytest -q
bash package_submission.sh /tmp/sgdagent-release-package-test
git grep -n -i -E "password|secret|private[_-]?key|access[_-]?token|refresh[_-]?token|sk-[A-Za-z0-9]|/Users/|dvd_db|DeepVideoDiscovery|VideoAgent" -- ':!PUBLIC_MANIFEST.md'
find . -type f -size +10M -not -path './.git/*' -print
git status --short
```

5. 提交并推送 release。

```bash
git add -A
git commit -m "chore: sync public release snapshot"
git push release clean-public-main:main
```

## 版本节奏

建议采用三层节奏：

- 私有开发提交：频繁提交到 `SGDAgent/main` 或 feature 分支。
- 私有稳定点：跑完 smoke/bad38/full 中至少一个后打内部 tag。
- 公开 release：只在 README、核心代码、测试和 baseline 说明都一致时同步。

内部 tag 示例：

```bash
git tag dev/sgdagent-YYYYMMDD
git push origin dev/sgdagent-YYYYMMDD
```

公开 tag 示例：

```bash
cd /Users/toniekule/Dev/SGDAgent-public-clean
git tag v0.1.0
git push release v0.1.0
```

## 发布前检查

在把 `SGDAgent-release` 从 private 切到 public 前，必须完成：

- LICENSE 已确定版权主体。
- README 不含内部路径、私有数据位置、未公开论文草稿链接。
- `PUBLIC_MANIFEST.md` 中的审计命令零输出。
- `BASELINE.md` 不暴露私有文件系统细节。
- GitHub 仓库只存在 `main` 和明确需要的 release tags。
- 没有私有开发分支被推到 release 远端。

## 常见错误

- 不要在 `SGDAgent-release` 执行 `git merge origin/main`。
- 不要把 `docs/plans/` 推到公开仓库。
- 不要把 full benchmark 运行结果、trace 或 graph artifacts 当作公开代码提交。
- 不要在公开仓库里保留私有开发远端的分支引用。
- 不要因为 release 仓库是 private 就跳过审计；它只是 public 前的暂存区。
