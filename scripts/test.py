import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib.ticker import PercentFormatter

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = PROJECT_ROOT / "experiment_runs"

TRAINING_SEEDS = (1, 2, 3)
TEST_WALLS = (0, 3, 6, 10)
TEST_SEED_START = 200000
TEST_EPISODES = 200
REUSE_EXISTING_RESULTS = True

RESULTS_PATH = (
    SUITE_ROOT
    / f"independent_test_seed{TEST_SEED_START}_n{TEST_EPISODES}.json"
)
OUTPUT_DIR = (
    SUITE_ROOT
    / f"report_results_seed{TEST_SEED_START}_n{TEST_EPISODES}"
)

EXPERIMENTS = (
    "action",
    "option",
    "latent_action",
    "latent_action_masked",
    "latent_option",
)

LABELS = {
    "action": "Action",
    "option": "Option",
    "latent_action": "Latent Action",
    "latent_action_masked": "Masked Latent Action",
    "latent_option": "Latent Option",
}

COLORS = {
    "action": "#4C78A8",
    "option": "#F58518",
    "latent_action": "#E45756",
    "latent_action_masked": "#72B7B2",
    "latent_option": "#54A24B",
}


if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault("GENIE_RUN_ROOT", str(SUITE_ROOT / "seed_1"))
os.environ.setdefault("GENIE_SEED", "1")
os.environ.setdefault("GENIE_LATENT_BUDGET_MULTIPLIER", "4")

from scripts import train_action_dqn
from scripts import train_latent_dqn
from scripts import train_latent_option
from scripts import train_option


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = train_action_dqn.TokenDQN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def make_tokenized_env(
    tokenizer,
    device: torch.device,
    config: dict,
    num_walls: int,
):
    return train_action_dqn.TokenizedGridWorldEnv(
        tokenizer=tokenizer,
        device=device,
        grid_size=config["grid_size"],
        image_size=config["image_size"],
        max_steps=config["max_steps"],
        num_walls=num_walls,
        seed=TEST_SEED_START,
    )


def make_latent_env(
    tokenizer,
    dynamics_model,
    device: torch.device,
    config: dict,
    num_walls: int,
):
    return train_latent_dqn.LatentDMGridWorldEnv(
        tokenizer=tokenizer,
        dynamics_model=dynamics_model,
        device=device,
        grid_size=config["grid_size"],
        image_size=config["image_size"],
        max_steps=config["max_steps"],
        num_walls=num_walls,
        seed=TEST_SEED_START,
        goal_reward=config["goal_reward"],
        movement_reward=config["movement_reward"],
        blocked_reward=config["blocked_reward"],
        enforce_valid_motion=config["enforce_valid_motion"],
        validate_reset_detection=config["validate_reset_detection"],
    )


def evaluate_experiment(
    experiment: str,
    model,
    checkpoint: dict,
    tokenizer,
    dynamics_model,
    device: torch.device,
    num_walls: int,
) -> dict:
    config = checkpoint["training_config"]

    if experiment in ("action", "option"):
        env = make_tokenized_env(tokenizer, device, config, num_walls)
    else:
        env = make_latent_env(
            tokenizer,
            dynamics_model,
            device,
            config,
            num_walls,
        )

    if experiment == "action":
        metrics = train_action_dqn.evaluate(
            model,
            env,
            TEST_EPISODES,
            TEST_SEED_START,
            device,
        )
    elif experiment == "option":
        metrics = train_option.evaluate(
            model,
            env,
            TEST_EPISODES,
            TEST_SEED_START,
            config["gamma"],
            device,
        )
    elif experiment in ("latent_action", "latent_action_masked"):
        metrics = train_latent_dqn.evaluate(
            model,
            env,
            TEST_EPISODES,
            TEST_SEED_START,
            device,
            experiment == "latent_action_masked",
        )
    else:
        metrics = train_latent_option.evaluate(
            model,
            env,
            TEST_EPISODES,
            TEST_SEED_START,
            config["gamma"],
            device,
        )

    if "mean_episode_length_low_level" in metrics:
        low_level_steps = float(metrics["mean_episode_length_low_level"])
    else:
        low_level_steps = float(metrics["mean_episode_length"])

    high_level_decisions = float(
        metrics.get("mean_high_level_decisions", low_level_steps)
    )

    return {
        "success_rate": float(metrics["success_rate"]),
        "mean_return": float(metrics["mean_return"]),
        "mean_low_level_steps": low_level_steps,
        "mean_high_level_decisions": high_level_decisions,
        "decision_reduction": 1.0 - high_level_decisions / low_level_steps,
    }


def new_results() -> dict:
    return {
        "training_seeds": list(TRAINING_SEEDS),
        "test_seed_start": TEST_SEED_START,
        "test_episodes": TEST_EPISODES,
        "test_walls": list(TEST_WALLS),
        "results": {},
    }


def save_results(results: dict) -> None:
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


def run_tests(device: torch.device) -> dict:
    if REUSE_EXISTING_RESULTS and RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        results = new_results()

    for training_seed in TRAINING_SEEDS:
        run_root = SUITE_ROOT / f"seed_{training_seed}"
        experiment_root = run_root / "outputs" / "experiments"
        seed_results = results["results"].setdefault(str(training_seed), {})

        tokenizer = train_action_dqn.load_tokenizer(
            run_root / "outputs" / "tokenizer" / "model" / "best.pt",
            device,
        )
        dynamics_model = None

        for experiment in EXPERIMENTS:
            experiment_results = seed_results.setdefault(experiment, {})
            wall_results = experiment_results.setdefault("walls", {})

            if all(str(walls) in wall_results for walls in TEST_WALLS):
                print(
                    f"seed={training_seed} experiment={experiment}: already tested"
                )
                continue

            if experiment.startswith("latent") and dynamics_model is None:
                dynamics_model = train_latent_dqn.load_dynamics_model(
                    run_root / "outputs" / "dm_hybrid" / "best.pt",
                    device,
                )

            model, checkpoint = load_checkpoint(
                experiment_root / experiment / "best.pt",
                device,
            )

            for num_walls in TEST_WALLS:
                wall_key = str(num_walls)
                if wall_key in wall_results:
                    continue

                print(
                    f"testing seed={training_seed} "
                    f"experiment={experiment} "
                    f"walls={num_walls}",
                    flush=True,
                )

                metrics = evaluate_experiment(
                    experiment,
                    model,
                    checkpoint,
                    tokenizer,
                    dynamics_model,
                    device,
                    num_walls,
                )
                wall_results[wall_key] = metrics
                save_results(results)

                print(
                    f"success={metrics['success_rate']:.3f} "
                    f"return={metrics['mean_return']:.3f} "
                    f"steps={metrics['mean_low_level_steps']:.2f} "
                    f"decisions={metrics['mean_high_level_decisions']:.2f}",
                    flush=True,
                )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del tokenizer
        if dynamics_model is not None:
            del dynamics_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_results(results)
    return results


def write_csv(results: dict) -> None:
    fields = (
        "training_seed",
        "experiment",
        "num_walls",
        "success_rate",
        "mean_return",
        "mean_low_level_steps",
        "mean_high_level_decisions",
        "decision_reduction",
    )

    with (OUTPUT_DIR / "independent_test_per_seed.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for seed in TRAINING_SEEDS:
            for experiment in EXPERIMENTS:
                for num_walls in TEST_WALLS:
                    row = {
                        "training_seed": seed,
                        "experiment": experiment,
                        "num_walls": num_walls,
                    }
                    row.update(
                        results["results"][str(seed)][experiment]["walls"][
                            str(num_walls)
                        ]
                    )
                    writer.writerow(row)


def metric_matrix(results: dict, experiment: str, metric: str) -> np.ndarray:
    return np.asarray(
        [
            [
                results["results"][str(seed)][experiment]["walls"][str(walls)][
                    metric
                ]
                for walls in TEST_WALLS
            ]
            for seed in TRAINING_SEEDS
        ],
        dtype=float,
    )


def style_axis(axis) -> None:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.set_xticks(TEST_WALLS)


def plot_range(
    axis,
    values: np.ndarray,
    color: str,
    label: str,
) -> None:
    x = np.asarray(TEST_WALLS)

    axis.fill_between(
        x,
        np.min(values, axis=0),
        np.max(values, axis=0),
        color=color,
        alpha=0.16,
        linewidth=0,
    )
    axis.plot(
        x,
        np.mean(values, axis=0),
        color=color,
        linewidth=2.6,
        marker="o",
        markersize=6,
        label=label,
    )


def save_figure(figure, filename: str) -> None:
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def plot_success_comparison(
    results: dict,
    first: str,
    second: str,
    filename: str,
) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.4))

    first_values = metric_matrix(results, first, "success_rate")
    second_values = metric_matrix(results, second, "success_rate")

    for experiment, values in (
        (first, first_values),
        (second, second_values),
    ):
        plot_range(
            axis,
            values,
            COLORS[experiment],
            LABELS[experiment],
        )

    all_values = np.concatenate((first_values.ravel(), second_values.ravel()))
    data_range = float(np.max(all_values) - np.min(all_values))
    padding = max(0.02, 0.08 * data_range)
    axis.set_ylim(
        max(0.0, float(np.min(all_values)) - padding),
        min(1.02, float(np.max(all_values)) + padding),
    )
    axis.set_xlabel("Number of walls")
    axis.set_ylabel("Test success rate")
    axis.legend(frameon=False, loc="lower left")
    style_axis(axis)
    save_figure(figure, filename)


def plot_temporal_abstraction(results: dict) -> None:
    option_success_change = 100 * (
        metric_matrix(results, "option", "success_rate")
        - metric_matrix(results, "action", "success_rate")
    )
    latent_success_change = 100 * (
        metric_matrix(results, "latent_option", "success_rate")
        - metric_matrix(results, "latent_action_masked", "success_rate")
    )
    option_decision_reduction = 100 * metric_matrix(
        results,
        "option",
        "decision_reduction",
    )
    latent_decision_reduction = 100 * metric_matrix(
        results,
        "latent_option",
        "decision_reduction",
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    plot_range(
        axes[0],
        option_success_change,
        COLORS["option"],
        "Option $-$ Action",
    )
    plot_range(
        axes[0],
        latent_success_change,
        COLORS["latent_option"],
        "Latent Option $-$ Masked Latent Action",
    )
    axes[0].axhline(0, color="#555555", linewidth=1.0, linestyle="--")
    minimum_change = min(
        float(np.min(option_success_change)),
        float(np.min(latent_success_change)),
    )
    maximum_change = max(
        float(np.max(option_success_change)),
        float(np.max(latent_success_change)),
    )
    axes[0].set_ylim(
        min(-5.0, np.floor(minimum_change / 10.0) * 10.0 - 2.0),
        max(5.0, np.ceil(maximum_change / 5.0) * 5.0 + 2.0),
    )
    axes[0].set_xlabel("Number of walls")
    axes[0].set_ylabel("Change in success (percentage points)")
    axes[0].set_title("(a) Performance after adding options")
    axes[0].legend(frameon=False, loc="lower right")
    style_axis(axes[0])

    plot_range(
        axes[1],
        option_decision_reduction,
        COLORS["option"],
        "Primitive option",
    )
    plot_range(
        axes[1],
        latent_decision_reduction,
        COLORS["latent_option"],
        "Latent option",
    )
    maximum_reduction = max(
        float(np.max(option_decision_reduction)),
        float(np.max(latent_decision_reduction)),
    )
    axes[1].set_ylim(
        0,
        max(25.0, np.ceil(maximum_reduction / 5.0) * 5.0 + 2.0),
    )
    axes[1].set_xlabel("Number of walls")
    axes[1].set_ylabel("Decision reduction")
    axes[1].yaxis.set_major_formatter(
        PercentFormatter(xmax=100, decimals=0)
    )
    axes[1].set_title("(b) Temporal abstraction")
    axes[1].legend(frameon=False, loc="upper left")
    style_axis(axes[1])

    save_figure(figure, "03_temporal_abstraction_comparison.png")


def make_outputs(results: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(results)

    plot_success_comparison(
        results,
        "action",
        "option",
        "01_action_option_success.png",
    )
    plot_success_comparison(
        results,
        "latent_action",
        "latent_action_masked",
        "02_validity_mask_success.png",
    )
    plot_temporal_abstraction(results)


def main() -> None:
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    results = run_tests(device)
    make_outputs(results)

    print("test data:", RESULTS_PATH)
    print("csv and figures:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
