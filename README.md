# alphaXiv Implementations

This repo contains implementations of heavily-requested papers on alphaXiv. The goal of this repo is to provide well-documented, easy-to-follow implementations of popular research paper codebases.

**Request implementations:** Open an issue or click 'implement' on any paper on alphaXiv.

## Requirements for new implementation PRs

Each implementation must include:

1. **README with specs**: GPU count/type required, runtime estimates, dataset instructions, reproduction results
2. **Standard structure**: Use `pyproject.toml` for dependencies and `src/` layout for code
3. **Speedrun.sh**: Each project must have a clear Nanochat-style speedrun.sh script that sets up the environment and runs relevant scripts for training and evaluation.

## Structure
```
paper-name/
├── README.md
├── pyproject.toml
└── src/
    └── paper_name/
        ├── train.py
        └── eval.py
```
