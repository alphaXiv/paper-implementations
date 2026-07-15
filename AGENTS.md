# paper-implementations

A collection of independent paper reproductions. This file describes the repo's structure and the conventions every paper folder must follow.

## Repository structure

Each top-level folder is one paper. The code inside a folder is the reproduction of that paper — model code, training/eval scripts, and any analysis.

```
paper-implementations/
├── agent_r1/
├── attention_is_not_all_you_need/
├── just_rl/
├── retriever/
├── rope_imaginary/
├── spurious_rewards/
├── tiny_recursive_models/
└── ...one folder per paper
```

## Paper folders are independent research repos

Treat each paper folder as a self-contained research repository:

- **Own dependencies.** Each folder manages its own environment with `uv` — it has its own `pyproject.toml` and `uv.lock`. Never share or hoist dependencies to the repo root. Run commands from inside the paper folder (e.g. `uv run ...`, `uv sync`).
- **Own README.** Each folder has a `README.md` that explains:
  - What the paper is (title, authors, link to the paper).
  - What was reproduced — which claims, figures, or results from the paper were targeted, and how the reproduction turned out.
- **No cross-folder imports.** Code in one paper folder must not depend on code in another paper folder.

## README experiment listing (required)

Every paper folder's `README.md` must list **all experiments that were run** for that reproduction. For each experiment, include:

1. A short description of the experiment (what question it answers, what setup was used).
2. **A link to the git branch containing the code for that experiment.**

Example format:

```markdown
## Experiments

| Experiment | Description | Branch |
|---|---|---|
| Baseline reproduction | Reproduce Table 1 of the paper | [`paper-x/baseline`](https://github.com/alphaXiv/paper-implementations/tree/paper-x/baseline) |
| LR ablation | Sweep learning rate 1e-4 → 1e-3 | [`paper-x/lr-ablation`](https://github.com/alphaXiv/paper-implementations/tree/paper-x/lr-ablation) |
```

When you add or modify an experiment, update the corresponding README entry and make sure the branch link points to the branch that actually contains that experiment's code.

## Working in this repo

- When adding a new paper: create a new top-level folder, initialize it with `uv init` (its own `pyproject.toml`/`uv.lock`), and write the README per the conventions above before adding experiments.
- When working on an existing paper: stay inside that paper's folder and use its environment; don't touch other paper folders.
- Keep the root of the repo minimal — only the top-level `README.md`, this file, and one folder per paper.
