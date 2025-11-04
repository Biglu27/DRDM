## Setup

Please use `uv`. Then `uv sync` and good to go.

## Training quickstart

在本地具备网络环境时，可执行 `uv sync` 安装 `jax[cpu]`、`jaxlib`、`diffusionlab`
等依赖。随后可通过 `uv run python tests/test_gaussian_training.py` 触发最小化的
高斯混合训练流程，验证训练循环是否正常运行。

## 同步到 GitHub 仓库

在容器或本地完成改动后，可按以下步骤把提交推送到你自己的 GitHub 仓库：

1. 确认已经在本地生成提交（例如 `git status` 显示工作区干净，`git log -1` 为最新提交）。
2. 如果尚未绑定远端，在仓库根目录执行
   ```bash
   git remote add origin <你的GitHub仓库地址>
   ```
   若远端已存在，可用 `git remote -v` 检查。
3. 运行
   ```bash
   git push -u origin work
   ```
   将当前分支推送到 GitHub。若目标分支不同，请把 `work` 替换为对应名称。
4. 推送成功后，即可在 GitHub 上查看该分支的最新文件，也可以发起 Pull Request 合并到主分支。

如遇到权限或网络问题，请根据 Git 提示配置访问令牌或代理，然后重试上述命令。

## Structure

Experiment setup files and outputs are in `experiments`. Everything else is in `src`.
