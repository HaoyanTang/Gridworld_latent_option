from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3, 2, 1)
LATENT_BUDGET_MULTIPLIER = 4

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def stream_command(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
) -> None:
    
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("command:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def run_step(
    name: str,
    command: list[str],
    env: dict[str, str],
    run_root: Path,
    required_outputs: tuple[Path, ...],
    resume: bool,
) -> None:
    marker = run_root / "suite_markers" / f"{name}.json"
    if (
        resume
        and marker.exists()
        and all(path.exists() for path in required_outputs)
    ):
        print(f"\n===== skip completed step: {name} =====", flush=True)
        return

    print(f"\n===== run step: {name} =====", flush=True)
    started_at = utc_now()
    stream_command(command, env, run_root / "logs" / f"{name}.log")

    missing = [str(path) for path in required_outputs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Step {name} finished but required outputs are missing: {missing}"
        )

    write_json(
        marker,
        {
            "step": name,
            "seed": int(env["GENIE_SEED"]),
            "started_at": started_at,
            "completed_at": utc_now(),
            "command": command,
            "required_outputs": [str(path) for path in required_outputs],
        },
    )


def summarize_history(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("history", rows.get("evaluations", []))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No history rows in {path}")

    final_rows = [
        row
        for row in rows
        if int(row.get("num_walls", row.get("walls", -1))) == 10
    ]
    if not final_rows:
        raise ValueError(f"No 10-wall rows in {path}")

    def success(row: dict[str, Any]) -> float:
        return float(row["eval_success_rate"])

    def score(row: dict[str, Any]) -> tuple[float, float]:
        return success(row), float(row.get("eval_mean_return", float("-inf")))

    best = max(final_rows, key=score)
    final = final_rows[-1]
    return {
        "history": str(path),
        "total_low_level_steps": int(max(row["global_step"] for row in rows)),
        "best_10_wall_success": success(best),
        "best_10_wall_return": float(best.get("eval_mean_return", 0.0)),
        "best_10_wall_step": int(best["global_step"]),
        "final_10_wall_success": success(final),
        "final_10_wall_return": float(final.get("eval_mean_return", 0.0)),
        "final_10_wall_step": int(final["global_step"]),
        "high_level_decisions": (
            int(final["high_level_decisions"])
            if "high_level_decisions" in final
            else int(final["global_step"])
        ),
    }


def experiment_environment(
    base: dict[str, str],
    output_dir: Path,
    action_output_dir: Path,
    latent_budget: bool = False,
    latent_validity_mask: bool | None = None,
) -> dict[str, str]:
    env = dict(base)
    env["GENIE_OUTPUT_DIR"] = str(output_dir)
    env["GENIE_ACTION_OUTPUT_DIR"] = str(action_output_dir)
    if latent_budget:
        env["GENIE_LATENT_BUDGET_MULTIPLIER"] = str(
            LATENT_BUDGET_MULTIPLIER
        )
    if latent_validity_mask is not None:
        env["GENIE_LATENT_VALIDITY_MASK"] = "1" if latent_validity_mask else "0"
    return env


def run_seed(seed: int, suite_root: Path, resume: bool) -> dict[str, Any]:
    run_root = (suite_root / f"seed_{seed}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    base_env = dict(os.environ)
    base_env.update(
        {
            "GENIE_RUN_ROOT": str(run_root),
            "GENIE_SEED": str(seed),
            "PYTHONHASHSEED": str(seed),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
    )

    python = sys.executable
    gridworld_data = run_root / "data" / "gridworld"
    dm_data = run_root / "data" / "dm"
    outputs = run_root / "outputs"
    experiments = outputs / "experiments"

    action_dir = experiments / "action"
    experiment_dirs = {
        "action": action_dir,
        "option": experiments / "option",
        "latent_action": experiments / "latent_action",
        "latent_action_masked": experiments / "latent_action_masked",
        "latent_option": experiments / "latent_option",
    }

    status_path = run_root / "suite_status.json"
    status = {
        "seed": seed,
        "run_root": str(run_root),
        "status": "running",
        "started_at": utc_now(),
        "latent_budget_multiplier": LATENT_BUDGET_MULTIPLIER,
        "experiments": {
            name: str(path) for name, path in experiment_dirs.items()
        },
    }
    write_json(status_path, status)

    try:
        run_step(
            "generate_data",
            [
                python,
                "scripts/generate_data.py",
            ],
            base_env,
            run_root,
            (
                gridworld_data / "train.npz",
                gridworld_data / "eval.npz",
                gridworld_data / "train_dm.npz",
                gridworld_data / "eval_dm.npz",
                gridworld_data / "train_goal.npz",
                gridworld_data / "eval_goal.npz",
            ),
            resume,
        )
        run_step(
            "train_tokenizer",
            [python, "scripts/train_tokenizer.py"],
            base_env,
            run_root,
            (
                outputs / "tokenizer" / "model" / "best.pt",
                outputs / "tokenizer" / "loss_history.json",
            ),
            resume,
        )
        run_step(
            "train_lam",
            [python, "scripts/train_lam.py"],
            base_env,
            run_root,
            (
                outputs / "lam" / "model" / "best.pt",
                outputs / "lam" / "loss_history.json",
            ),
            resume,
        )
        run_step(
            "generate_dm_data",
            [python, "scripts/generate_data_dm.py"],
            base_env,
            run_root,
            (
                dm_data / "dynamics_train.npz",
                dm_data / "dynamics_eval.npz",
                dm_data / "dynamics_train_dm.npz",
                dm_data / "dynamics_eval_dm.npz",
            ),
            resume,
        )

        moving_env = dict(base_env)
        moving_env["GENIE_DM_PHASE"] = "moving"
        run_step(
            "train_dm_moving",
            [python, "scripts/train_dm.py"],
            moving_env,
            run_root,
            (
                outputs / "dm" / "best.pt",
                outputs / "dm" / "history.json",
            ),
            resume,
        )
        hybrid_env = dict(base_env)
        hybrid_env["GENIE_DM_PHASE"] = "hybrid"
        run_step(
            "train_dm_hybrid",
            [python, "scripts/train_dm.py"],
            hybrid_env,
            run_root,
            (
                outputs / "dm_hybrid" / "best.pt",
                outputs / "dm_hybrid" / "history.json",
            ),
            resume,
        )

        run_step(
            "experiment_action",
            [python, "scripts/train_action_dqn.py"],
            experiment_environment(base_env, action_dir, action_dir),
            run_root,
            (action_dir / "best.pt", action_dir / "history.json"),
            resume,
        )
        run_step(
            "experiment_option",
            [python, "scripts/train_option.py"],
            experiment_environment(
                base_env,
                experiment_dirs["option"],
                action_dir,
            ),
            run_root,
            (
                experiment_dirs["option"] / "best.pt",
                experiment_dirs["option"] / "history.json",
            ),
            resume,
        )
        run_step(
            "experiment_latent_action",
            [python, "scripts/train_latent_dqn.py"],
            experiment_environment(
                base_env,
                experiment_dirs["latent_action"],
                action_dir,
                latent_budget=True,
                latent_validity_mask=False,
            ),
            run_root,
            (
                experiment_dirs["latent_action"] / "best.pt",
                experiment_dirs["latent_action"] / "history.json",
            ),
            resume,
        )
        run_step(
            "experiment_latent_action_masked",
            [python, "scripts/train_latent_dqn.py"],
            experiment_environment(
                base_env,
                experiment_dirs["latent_action_masked"],
                action_dir,
                latent_budget=True,
                latent_validity_mask=True,
            ),
            run_root,
            (
                experiment_dirs["latent_action_masked"] / "best.pt",
                experiment_dirs["latent_action_masked"] / "history.json",
            ),
            resume,
        )
        run_step(
            "experiment_latent_option",
            [python, "scripts/train_latent_option.py"],
            experiment_environment(
                base_env,
                experiment_dirs["latent_option"],
                action_dir,
                latent_budget=True,
            ),
            run_root,
            (
                experiment_dirs["latent_option"] / "best.pt",
                experiment_dirs["latent_option"] / "history.json",
            ),
            resume,
        )

        summary = {
            "seed": seed,
            "action_budget_relation": {
                "action": 1,
                "option": 1,
                "latent_action": LATENT_BUDGET_MULTIPLIER,
                "latent_action_masked": LATENT_BUDGET_MULTIPLIER,
                "latent_option": LATENT_BUDGET_MULTIPLIER,
            },
            "experiments": {
                name: summarize_history(directory / "history.json")
                for name, directory in experiment_dirs.items()
            },
        }
        write_json(run_root / "result_summary.json", summary)
        status.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "summary": str(run_root / "result_summary.json"),
            }
        )
        write_json(status_path, status)
        return summary
    except Exception as error:
        status.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json(status_path, status)
        raise


def main() -> None:
    seeds = SEEDS
    suite_root = Path(
        os.environ.get(
            "GENIE_SUITE_ROOT",
            str(PROJECT_ROOT / "experiment_runs"),
        )
    ).resolve()
    resume = os.environ.get("GENIE_RESUME", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    suite_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "suite_root": str(suite_root),
        "seeds": list(seeds),
        "resume_completed_steps": resume,
        "action_budget_multiplier": 1,
        "latent_budget_multiplier": LATENT_BUDGET_MULTIPLIER,
        "evaluation_seed_start": 100_000,
        "notes": [
            "The same training seed is used for data, tokenizer, LAM, DM, and all five RL runs within a replicate.",
            "RL evaluation maps remain fixed across training seeds for paired comparison.",
            "Action Option uses an initiation condition that masks an option when its first primitive move is invalid; after initiation, blocked moves are observed through environment interaction.",
            "Latent Option uses its initiation validity check before option selection; no separate checked Latent Option experiment is run.",
        ],
    }
    write_json(suite_root / "suite_manifest.json", manifest)

    summaries = []
    for index, seed in enumerate(seeds, start=1):
        print(
            f"\n######## seed {seed} ({index}/{len(seeds)}) ########",
            flush=True,
        )
        summaries.append(run_seed(seed, suite_root, resume))

    write_json(
        suite_root / "all_seed_results.json",
        {
            "completed_at": utc_now(),
            "seeds": list(seeds),
            "results": summaries,
        },
    )
    print("\nAll seeds completed.")
    print("results:", suite_root / "all_seed_results.json")


if __name__ == "__main__":
    main()

