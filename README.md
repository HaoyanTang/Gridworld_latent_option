# Gridworld Latent Option
This repository contains the code for the project **Can Learned Latent Actions Support Temporal Abstraction through Options?**

The project studies temporal abstraction in an image-based Gridworld. It first compares primitive actions with hand-defined options. It then replaces primitive actions with learned latent actions and tests whether options can still reduce the number of agent decisions.

## Method
The environment is an `8 x 8` Gridworld rendered as `64 x 64` RGB images. The number of walls increases through four curriculum stages: `0`, `3`, `6`, and `10`.

The learned control pipeline contains three models:
- **Tokenizer:** converts each RGB observation into an `8 x 8` grid of discrete tokens.
- **Latent action model (LAM):** infers one of four latent actions from two successive observations.
- **Dynamics model (DM):** predicts the next token grid from the current token grid and a latent action.

## Experiments
The repository runs five experiments:
1. **Action:** a baseline agent with four primitive actions.
2. **Option:** four primitive actions and four fixed-direction options. Each option can execute for at most two environment steps. Its first move must be valid.
3. **Latent Action:** an agent with four learned latent actions. The DM converts each selected latent action into the next predicted state.
4. **Masked Latent Action:** the same latent-action agent with a validity mask. The mask checks the four DM predictions before selection and removes predicted invalid moves.
5. **Latent Option:** four latent actions and four length-two latent options. The same validity check is used as the option initiation condition.

The Action experiment defines the curriculum step budget. Option uses the same environment-step budget. The three latent experiments use four times this budget, as set by `LATENT_BUDGET_MULTIPLIER` in `scripts/run_experiments.py`.

## Installation
Python 3.11 is recommended. The Conda setup below includes PyTorch 2.5.1 and CUDA 12.1 for an NVIDIA GPU.

```bash
conda env create -f environment.yml
conda activate final_project
```

## Run the full experiment suite

Run all preprocessing, training, and reinforcement learning experiments from the project root:

```bash
python scripts/run_experiments.py
```
