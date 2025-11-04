## Setup

Please use `uv`. Then `uv sync` and good to go.

## Training quickstart

在本地具备网络环境时，可执行 `uv sync` 安装 `jax[cpu]`、`jaxlib`、`diffusionlab`
等依赖。随后可通过 `uv run python tests/test_gaussian_training.py` 触发最小化的
高斯混合训练流程，验证训练循环是否正常运行。

## Structure

Experiment setup files and outputs are in `experiments`. Everything else is in `src`. 