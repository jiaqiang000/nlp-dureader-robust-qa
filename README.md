# DuReader Robust QA

本仓库是自然语言处理结课作业的实验工作区。当前选题是：

> 基于 DuReader_robust 的中文抽取式问答模型构建，并在本地低算力环境下分析小型预训练语言模型的效果、速度和模型规模。

## 项目范围

- 任务：抽取式 QA / 机器阅读理解。
- 数据集：DuReader_robust。
- 主要指标：F1、EM、模型大小、训练耗时、推理耗时。
- 算力主线：优先保证实验能在本地 Mac 上运行。

## 项目规则

- 代码、说明文档、实验设计和结果摘要进入 Git。
- 原始数据、处理缓存、模型权重、预测文件、运行输出不进入 Git。
- 当前阶段优先使用 `scripts/` 放实验脚本。
- 只有当代码复用明显增加时，才考虑新增 `src/`、`configs/` 等更复杂结构。

## 环境配置

在项目目录内创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

只做数据检查时，官方评测脚本依赖 `six`：

```bash
pip install six
```

进入模型微调阶段前，安装完整依赖：

```bash
pip install -r requirements.txt
```

## 数据检查

下载 DuReader_robust、统计数据划分，并验证官方评测脚本：

```bash
python scripts/inspect_dureader.py
```

脚本会把官方 PaddleNLP 数据包下载到 `data/raw/`。该目录已被 Git 忽略。

官方数据包统计结果：

| 划分 | 段落数 | QA 数 | 答案数 | 平均上下文字数 | 平均问题字数 | 平均答案字数 |
|---|---:|---:|---:|---:|---:|---:|
| train | 14,520 | 14,520 | 14,520 | 282.30 | 9.26 | 5.50 |
| dev | 1,417 | 1,417 | 1,962 | 284.28 | 9.42 | 6.45 |
| test | 31,032 | 50,000 | 0 | 304.38 | 10.29 | 0.00 |

脚本还会把 dev 集第一个标准答案构造成预测文件，并调用官方 `evaluate.py`。这个检查只用于确认评测链路正确，不是模型结果。预期结果：

```json
{"F1": "100.000", "EM": "100.000", "TOTAL": 1417, "SKIP": 0}
```

运行单元测试：

```bash
python -m unittest discover -s tests -v
```

## 最小 QA 链路验证

运行一个极小的端到端模型训练/预测/评测链路：

```bash
python scripts/minimal_qa.py \
  --train-samples 80 \
  --eval-samples 30 \
  --epochs 1 \
  --batch-size 8 \
  --max-length 256 \
  --seed 42 \
  --output-dir outputs/minimal_qa
```

该命令会加载 `uer/chinese_roberta_L-2_H-128`，只训练 80 条样本，在 30 条 dev 样本上预测，并调用 DuReader_robust 官方评测脚本。这个结果只是链路验证，不是论文正式实验结果。由于 QA 输出层是随机初始化，小样本分数不能解释为稳定模型能力。

本地链路验证的一次结果：

```json
{
  "model_name": "uer/chinese_roberta_L-2_H-128",
  "device": "mps",
  "train_examples": 80,
  "eval_examples": 30,
  "epochs": 1,
  "seed": 42,
  "official_metrics": {
    "F1": "8.068",
    "EM": "0.000",
    "TOTAL": 30,
    "SKIP": 0
  }
}
```

`outputs/` 下生成的文件不进入 Git。

## 正式实验主线

当前正式实验主线是本地低算力小模型实验：

1. 先做 `train 1000 / dev 300 / epoch 1` 的模型筛选预实验，这一步不是最终主实验。
2. 对比三个小型中文 RoBERTa 模型：
   - `uer/chinese_roberta_L-2_H-128`
   - `uer/chinese_roberta_L-4_H-256`
   - `uer/chinese_roberta_L-4_H-512`
3. 根据模型筛选结果选择 1-2 个模型扩大训练规模，进入正式主实验。
4. 最终比较 F1、EM、训练耗时、推理耗时和模型大小。

详细实验设计见：

```text
docs/superpowers/specs/2026-05-31-local-small-model-experiment-design.md
```

## Pass 1 模型筛选命令

Pass 1 是模型筛选预实验，不是最终主实验。默认配置为 `train 1000 / dev 300 / epoch 1`。

```bash
python scripts/run_experiment.py --model-key A_L2_H128
python scripts/run_experiment.py --model-key B_L4_H256
python scripts/run_experiment.py --model-key C_L4_H512
```

每次运行会写入：

```text
outputs/experiments/<run_name>/
├── run_config.json
├── metrics.json
├── predictions.json
└── dev_subset.json
```

这些输出文件不进入 Git。完成 Pass 1 后，再把汇总指标写入 `docs/results/experiment_summary.md`。

## 常见误解澄清

当前不租 4090D、不做 LoRA/量化/RAG、不追 leaderboard，只是**当前阶段**为了保证本地小模型实验可复现、变量清晰而设定的范围边界，不代表整个 DuReader_robust 课题永久不允许做这些扩展。完成本地基线实验后，如果时间和算力允许，可以再评估是否把其中一项作为论文扩展实验。
