# Local Small-Model DuReader Experiment Design

## Purpose

This stage turns the current runnable QA chain into a formal local low-compute
experiment for the NLP course paper. The experiment should answer:

> Under local M1 Pro compute constraints, how do small Chinese pretrained
> language models trade off QA accuracy, training time, inference time, and model
> size on DuReader_robust?

The goal is not leaderboard performance. The goal is a reproducible, defensible
course experiment that fits the selected topic: local, low-compute Chinese QA.

## Confirmed Mainline

- Task: extractive question answering / machine reading comprehension.
- Dataset: DuReader_robust.
- Hardware mainline: local MacBook Pro with Apple M1 Pro and MPS.
- Evaluation: official DuReader_robust `evaluate.py` with F1 and EM.
- Submission boundary: source code, documentation, and result summaries in Git;
  raw data, model checkpoints, caches, and predictions excluded from Git.

## Experiment Matrix

| Group | Model | Role |
|---|---|---|
| A | `uer/chinese_roberta_L-2_H-128` | Tiny low-compute baseline |
| B | `uer/chinese_roberta_L-4_H-256` | Small model, expected quality/speed middle point |
| C | `uer/chinese_roberta_L-4_H-512` | Larger local candidate, expected upper local cost point |

`uer/chinese_roberta_L-12_H-768` is out of scope for the first formal local
experiment. It may be discussed as a future extension or optional 4090D follow-up
if local results justify more compute.

## Data Plan

The experiment proceeds in two passes.

### Pass 1: Controlled Small-Scale Pre-Experiment

- Train split: first 1,000 DuReader_robust training examples.
- Dev split: first 300 DuReader_robust dev examples.
- Epochs: 1.
- Purpose: confirm all three models run with the same pipeline and produce
  comparable metrics without spending full-training time.

### Pass 2: Main Local Experiment

The main experiment is chosen after Pass 1.

Default main experiment:

- Use the best balance model from Pass 1 and the tiny baseline model.
- Train on either 5,000 examples or full train, depending on observed runtime.
- Evaluate on full dev.

Decision rule:

- If Pass 1 runtime is manageable and memory is stable, use 5,000 train examples
  for the main experiment.
- If 5,000 examples is still fast enough, optionally run full train for the best
  balance model only.
- If the largest local candidate is too slow, report its Pass 1 result and keep
  main experiment to Groups A and B.

## Metrics

Each run records:

- F1 from official evaluator.
- EM from official evaluator.
- Train examples and dev examples.
- Number of generated train/eval features.
- Training wall-clock seconds.
- Prediction/evaluation wall-clock seconds if measured separately.
- Total run seconds.
- Device used: `mps`, `cpu`, or `cuda`.
- Approximate model artifact size, reported from HuggingFace cache files or saved
  model directory size when available.

## Output Layout

Generated artifacts stay outside Git:

```text
outputs/
└── experiments/
    ├── A_L2_H128_pre/
    ├── B_L4_H256_pre/
    └── C_L4_H512_pre/
```

Each run directory should contain:

- `metrics.json`
- `predictions.json`
- `dev_subset.json` or a reference to the dev source

Only summarized metrics should be committed, for example:

```text
docs/results/experiment_summary.md
```

`docs/results/experiment_summary.md` should be created only after at least Pass 1
has completed.

## Implementation Shape

The current `scripts/minimal_qa.py` already proves the end-to-end path. For the
formal experiment, avoid adding a full `src/` package unless the script becomes
hard to maintain.

Expected implementation changes:

- Extend `scripts/minimal_qa.py` or add one focused `scripts/run_experiment.py`.
- Add CLI options for model group, run name, train size, dev size, and output
  directory.
- Write a structured `metrics.json` for every run.
- Keep official evaluator invocation unchanged.
- Keep unit tests on pure helper functions.

## Non-Goals

- Do not optimize for leaderboard score.
- Do not rent 4090D in this stage.
- Do not add LoRA, quantization, or RAG yet.
- Do not compare against CMRC2018 in code.
- Do not commit model weights, downloaded datasets, or generated predictions.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| MPS runtime or memory instability | Start with Pass 1 and fall back to smaller train sizes |
| Scores too low after 1 epoch | Treat Pass 1 as pre-experiment; increase data/epoch only for main run |
| Large model too slow locally | Keep Group C as local upper-bound evidence, not mandatory main model |
| Metrics vary due random seed | Use fixed `--seed 42` for formal runs |
| Generated files accidentally enter Git | Keep `outputs/`, `data/raw/`, and checkpoints ignored |

## Paper Framing

The experiment supports this paper argument:

> For Chinese extractive QA in realistic search-style data, small pretrained
> language models can be evaluated reproducibly on local hardware. The central
> trade-off is not only F1/EM, but also runtime and model size under low-compute
> constraints.

Newer QA directions such as generative QA and temporal RAG should remain in the
related-work/background section, not in the implementation scope.

## Approval Gate

Implementation should not start until this design is reviewed and approved. The
next step after approval is a writing-plans implementation plan for Pass 1.
