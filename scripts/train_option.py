from __future__ import annotations

import json
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

RUN_ROOT = Path(os.environ.get("GENIE_RUN_ROOT")).resolve()
EXPERIMENT_SEED = int(os.environ.get("GENIE_SEED"))

from src.env.gridworld import NUM_ACTIONS
from src.env.tokenized_gridworld import TokenizedGridWorldEnv
from src.models.dqn import TokenDQN
from src.models.vqvae import VQVAEImageTokenizer
from src.rl.replay_buffer_option import ReplayBuffer


OPTION_LENGTH = 2
NUM_HIGH_LEVEL_ACTIONS = NUM_ACTIONS * 2
ACTION_NAMES = ("up", "down", "left", "right")


@dataclass
class ChoiceResult:
    next_state: np.ndarray
    discounted_return: float
    raw_return: float
    duration: int
    terminated: bool
    truncated: bool
    observed_block_termination: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(path: Path, device: torch.device) -> VQVAEImageTokenizer:
    tokenizer = VQVAEImageTokenizer(
        num_codes=128,
        embedding_dim=64,
        commitment_beta=0.25,
    ).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    tokenizer.load_state_dict(checkpoint)
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    return tokenizer


def find_reference_history() -> Path:
    return Path(os.environ["GENIE_ACTION_OUTPUT_DIR"]) / "history.json"


def load_step_matched_curriculum(path: Path) -> list[dict[str, int]]:
    rows = json.loads(path.read_text(encoding="utf-8"))

    curriculum: list[dict[str, int]] = []
    previous_end = 0
    for row in rows:
        if row["transition_reason"] is None:
            continue
        end_step = int(row["global_step"])
        curriculum.append(
            {
                "num_walls": int(row["num_walls"]),
                "start_step": previous_end,
                "end_step": end_step,
                "step_budget": end_step - previous_end,
            }
        )
        previous_end = end_step

    return curriculum


def epsilon_at_step(
    step: int,
    start: float,
    end: float,
    decay_steps: int,
) -> float:
    fraction = min(max(step / max(decay_steps, 1), 0.0), 1.0)
    return start + fraction * (end - start)


@torch.no_grad()
def greedy_choice(
    model: TokenDQN,
    state: np.ndarray,
    device: torch.device,
    valid_choices: np.ndarray,
) -> int:
    tensor = torch.from_numpy(np.asarray(state)).long().unsqueeze(0).to(device)
    q_values = model(tensor)
    valid_tensor = torch.from_numpy(valid_choices).bool().unsqueeze(0).to(device)
    q_values = q_values.masked_fill(~valid_tensor, -torch.inf)
    return int(q_values.argmax(dim=1).item())


def select_choice(
    model: TokenDQN,
    state: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
    valid_choices: np.ndarray,
) -> int:
    if rng.random() < epsilon:
        valid_ids = np.flatnonzero(valid_choices)
        return int(rng.choice(valid_ids))
    return greedy_choice(model, state, device, valid_choices)


def agent_position(env: TokenizedGridWorldEnv) -> Any:
    return env.env.agent_pos


def can_move(env: TokenizedGridWorldEnv, action: int) -> bool:
    base_env = env.env
    new_position = base_env._move(base_env.agent_pos, action)
    return bool(base_env._is_valid_position(new_position))


def valid_choice_mask(
    env: TokenizedGridWorldEnv,
) -> np.ndarray:
    mask = np.ones(NUM_HIGH_LEVEL_ACTIONS, dtype=np.bool_)
    for action in range(NUM_ACTIONS):
        mask[NUM_ACTIONS + action] = can_move(env, action)
    return mask


def execute_choice(
    env: TokenizedGridWorldEnv,
    state: np.ndarray,
    choice: int,
    gamma: float,
    max_low_level_steps: int,
) -> ChoiceResult:
    is_option = choice >= NUM_ACTIONS
    primitive_action = choice if not is_option else choice - NUM_ACTIONS
    requested_duration = OPTION_LENGTH if is_option else 1
    allowed_duration = min(requested_duration, max_low_level_steps)

    current_state = state
    discounted_return = 0.0
    raw_return = 0.0
    terminated = False
    truncated = False
    observed_block_termination = False
    duration = 0

    for _ in range(allowed_duration):
        old_position = agent_position(env)
        next_state, reward, terminated, truncated, info = env.step(
            primitive_action
        )
        new_position = agent_position(env)

        moved = old_position != new_position

        blocked = not moved
        discounted_return += (gamma**duration) * float(reward)
        raw_return += float(reward)
        current_state = next_state
        duration += 1

        if (
            is_option
            and blocked
            and not terminated
            and not truncated
        ):
            observed_block_termination = True
            break
        if terminated or truncated:
            break

    return ChoiceResult(
        next_state=current_state,
        discounted_return=discounted_return,
        raw_return=raw_return,
        duration=duration,
        terminated=bool(terminated),
        truncated=bool(truncated),
        observed_block_termination=bool(observed_block_termination),
    )


def optimize_smdp_dqn(
    online_network: TokenDQN,
    target_network: TokenDQN,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    gradient_clip: float,
    device: torch.device,
) -> float:
    batch = replay_buffer.sample(batch_size)
    states = torch.from_numpy(batch.states).long().to(device)
    choices = torch.from_numpy(batch.choices).long().to(device)
    returns = torch.from_numpy(batch.discounted_returns).float().to(device)
    next_states = torch.from_numpy(batch.next_states).long().to(device)
    dones = torch.from_numpy(batch.dones).float().to(device)
    durations = torch.from_numpy(batch.durations).float().to(device)
    next_valid_choices = torch.from_numpy(
        batch.next_valid_choices
    ).bool().to(device)

    chosen_q = online_network(states).gather(1, choices.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_online_q = online_network(next_states).masked_fill(~next_valid_choices, -torch.inf)
        next_choices = next_online_q.argmax(dim=1, keepdim=True)
        next_q = target_network(next_states).gather(1, next_choices).squeeze(1)
        discounts = torch.pow(torch.full_like(durations, gamma), durations)
        targets = returns + discounts * (1.0 - dones) * next_q

    loss = F.smooth_l1_loss(chosen_q, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_network.parameters(), gradient_clip)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(
    model: TokenDQN,
    env: TokenizedGridWorldEnv,
    episodes: int,
    seed_start: int,
    gamma: float,
    device: torch.device,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()

    returns: list[float] = []
    lengths: list[int] = []
    decisions: list[int] = []
    successes = 0
    truncations = 0
    choice_counts = np.zeros(NUM_HIGH_LEVEL_ACTIONS, dtype=np.int64)
    option_duration_sum = 0
    option_execution_count = 0
    observed_block_option_terminations = 0
    valid_option_count_sum = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=seed_start + episode)
        episode_return = 0.0
        low_steps = 0
        high_decisions = 0

        while low_steps < env.env.max_steps:
            valid_choices = valid_choice_mask(env)
            valid_option_count_sum += int(valid_choices[NUM_ACTIONS:].sum())
            choice = greedy_choice(model, state, device, valid_choices)
            choice_counts[choice] += 1
            high_decisions += 1
            result = execute_choice(
                env,
                state,
                choice,
                gamma,
                env.env.max_steps - low_steps,
            )
            state = result.next_state
            episode_return += result.raw_return
            low_steps += result.duration

            if choice >= NUM_ACTIONS:
                option_execution_count += 1
                option_duration_sum += result.duration
                observed_block_option_terminations += int(result.observed_block_termination)

            if result.terminated or result.truncated:
                successes += int(result.terminated)
                truncations += int(result.truncated)
                break

        returns.append(episode_return)
        lengths.append(low_steps)
        decisions.append(high_decisions)

    if was_training:
        model.train()

    primitive_count = int(choice_counts[:NUM_ACTIONS].sum())
    option_count = int(choice_counts[NUM_ACTIONS:].sum())
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "truncation_rate": truncations / episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length_low_level": float(np.mean(lengths)),
        "mean_high_level_decisions": float(np.mean(decisions)),
        "choice_counts": choice_counts.tolist(),
        "primitive_choice_count": primitive_count,
        "option_choice_count": option_count,
        "option_choice_fraction": option_count / max(primitive_count + option_count, 1),
        "mean_executed_option_duration": (
            option_duration_sum / option_execution_count
            if option_execution_count
            else 0.0
        ),
        "observed_block_option_termination_rate": (
            observed_block_option_terminations / option_execution_count
            if option_execution_count
            else 0.0
        ),
        "mean_valid_option_count": (
            valid_option_count_sum / max(sum(decisions), 1)
        ),
    }


def save_checkpoint(
    path: Path,
    online_network: TokenDQN,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    global_episode: int,
    global_step: int,
    high_level_decisions: int,
    stage_index: int,
    stage_episode: int,
    eval_metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "experiment": "action_option",
            "model_state_dict": online_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "training_config": training_config,
            "option_mapping": {
                "0": "primitive_up",
                "1": "primitive_down",
                "2": "primitive_left",
                "3": "primitive_right",
                "4": "repeat_up_2",
                "5": "repeat_down_2",
                "6": "repeat_left_2",
                "7": "repeat_right_2",
            },
            "episode": global_episode,
            "global_step": global_step,
            "high_level_decisions": high_level_decisions,
            "stage_index": stage_index,
            "stage_episode": stage_episode,
            "num_walls": training_config["curriculum"][stage_index]["num_walls"],
            "eval_metrics": eval_metrics,
        },
        path,
    )


def make_env(
    tokenizer: VQVAEImageTokenizer,
    device: torch.device,
    config: dict[str, Any],
    num_walls: int,
    seed: int,
) -> TokenizedGridWorldEnv:
    return TokenizedGridWorldEnv(
        tokenizer=tokenizer,
        device=device,
        grid_size=config["grid_size"],
        image_size=config["image_size"],
        max_steps=config["max_steps"],
        num_walls=num_walls,
        seed=seed,
    )


def main() -> None:
    reference_history = find_reference_history()
    curriculum = load_step_matched_curriculum(reference_history)
    training_config: dict[str, Any] = {
        "batch_size": 64,
        "buffer_capacity": 50000,
        "learning_starts": 1000,
        "train_frequency": 1,
        "target_update_frequency": 1000,
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gradient_clip": 10.0,
        "epsilon_start": 1.0,
        "epsilon_restart": 0.6,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 30000,
        "eval_interval_episodes": 100,
        "eval_episodes": 100,
        "eval_seed_start": 100000,
        "grid_size": 8,
        "image_size": 64,
        "max_steps": 32,
        "option_length": OPTION_LENGTH,
        "option_initiation": "first move must be valid",
        "option_termination": "goal, timeout, max length, or after an observed blocked move",
        "num_high_level_actions": NUM_HIGH_LEVEL_ACTIONS,
        "reference_history": str(reference_history),
        "curriculum": curriculum,
        "seed": EXPERIMENT_SEED,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tokenizer": str(
            RUN_ROOT / "outputs" / "tokenizer" / "model" / "best.pt"
        ),
        "output_dir": os.environ.get(
            "GENIE_OUTPUT_DIR",
            str(RUN_ROOT / "outputs" / "dqn_options_2"),
        ),
    }

    set_seed(training_config["seed"])
    rng = np.random.default_rng(training_config["seed"])
    device = torch.device(training_config["device"])
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(Path(training_config["tokenizer"]), device)
    initial_walls = curriculum[0]["num_walls"]
    train_env = make_env(
        tokenizer, device, training_config, initial_walls, training_config["seed"]
    )
    eval_env = make_env(
        tokenizer,
        device,
        training_config,
        initial_walls,
        training_config["eval_seed_start"],
    )
    initial_state, _ = train_env.reset(seed=training_config["seed"])

    model_config = {
        "tokenizer_size": 128,
        "num_actions": NUM_HIGH_LEVEL_ACTIONS,
        "token_embedding_dim": 32,
        "hidden_channels": 64,
        "hidden_dim": 256,
        "token_grid_size": int(initial_state.shape[-1]),
        "num_residual_blocks": 3,
    }
    online_network = TokenDQN(**model_config).to(device)
    target_network = TokenDQN(**model_config).to(device)
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()
    for parameter in target_network.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(online_network.parameters(), lr=training_config["learning_rate"])
    replay_buffer = ReplayBuffer(
        capacity=training_config["buffer_capacity"],
        state_shape=tuple(initial_state.shape),
        seed=training_config["seed"],
    )

    global_step = 0
    global_episode = 0
    high_level_decisions = 0
    losses: deque[float] = deque(maxlen=1000)
    recent_returns: deque[float] = deque(maxlen=100)
    recent_lengths: deque[int] = deque(maxlen=100)
    recent_successes: deque[int] = deque(maxlen=100)
    history: list[dict[str, Any]] = []

    for stage_index, stage in enumerate(curriculum):
        num_walls = stage["num_walls"]
        target_step = stage["end_step"]
        stage_step = 0
        stage_episode = 0
        stage_best_score = (-1.0, float("-inf"))
        losses.clear()
        recent_returns.clear()
        recent_lengths.clear()
        recent_successes.clear()

        if stage_index > 0:
            train_env = make_env(
                tokenizer,
                device,
                training_config,
                num_walls,
                training_config["seed"] + global_episode,
            )
            eval_env = make_env(
                tokenizer,
                device,
                training_config,
                num_walls,
                training_config["eval_seed_start"],
            )

        print(
            f"\n===== step-matched stage: {num_walls} walls =====\n"
            f"low-level step range: {stage['start_step']} -> {target_step} "
            f"(budget={stage['step_budget']})"
        )

        while global_step < target_step:
            global_episode += 1
            stage_episode += 1
            state, _ = train_env.reset()
            episode_return = 0.0
            episode_low_steps = 0
            episode_success = False
            episode_finished = False

            while not episode_finished and global_step < target_step:
                epsilon = epsilon_at_step(
                    stage_step,
                    training_config["epsilon_start"]
                    if stage_index == 0
                    else training_config["epsilon_restart"],
                    training_config["epsilon_end"],
                    training_config["epsilon_decay_steps"],
                )
                valid_choices = valid_choice_mask(train_env)
                choice = select_choice(
                    online_network,
                    state,
                    epsilon,
                    rng,
                    device,
                    valid_choices,
                )
                old_global_step = global_step
                remaining_stage_steps = target_step - global_step
                result = execute_choice(
                    train_env,
                    state,
                    choice,
                    training_config["gamma"],
                    remaining_stage_steps,
                )
                global_step += result.duration
                stage_step += result.duration
                episode_low_steps += result.duration
                high_level_decisions += 1
                episode_return += result.raw_return

                hit_stage_boundary = global_step >= target_step
                replay_done = result.terminated or hit_stage_boundary
                next_valid_choices = valid_choice_mask(train_env)
                replay_buffer.add(
                    state,
                    choice,
                    result.discounted_return,
                    result.next_state,
                    replay_done,
                    result.duration,
                    next_valid_choices,
                )
                state = result.next_state

                for update_step in range(old_global_step + 1, global_step + 1):
                    if (
                        update_step >= training_config["learning_starts"]
                        and len(replay_buffer) >= training_config["batch_size"]
                        and update_step % training_config["train_frequency"] == 0
                    ):
                        losses.append(
                            optimize_smdp_dqn(
                                online_network,
                                target_network,
                                optimizer,
                                replay_buffer,
                                training_config["batch_size"],
                                training_config["gamma"],
                                training_config["gradient_clip"],
                                device,
                            )
                        )
                    if (
                        update_step
                        % training_config["target_update_frequency"]
                        == 0
                    ):
                        target_network.load_state_dict(
                            online_network.state_dict()
                        )

                episode_success = result.terminated
                episode_finished = (
                    result.terminated
                    or result.truncated
                    or hit_stage_boundary
                )

            recent_returns.append(episode_return)
            recent_lengths.append(episode_low_steps)
            recent_successes.append(int(episode_success))

            should_evaluate = (
                stage_episode % training_config["eval_interval_episodes"] == 0
                or global_step >= target_step
            )
            if not should_evaluate:
                continue

            eval_metrics = evaluate(
                online_network,
                eval_env,
                training_config["eval_episodes"],
                training_config["eval_seed_start"],
                training_config["gamma"],
                device,
            )
            epsilon = epsilon_at_step(
                stage_step,
                training_config["epsilon_start"]
                if stage_index == 0
                else training_config["epsilon_restart"],
                training_config["epsilon_end"],
                training_config["epsilon_decay_steps"],
            )
            transition_reason = ("reference_step_budget" if global_step >= target_step else None)
            score = (eval_metrics["success_rate"], eval_metrics["mean_return"])
            is_new_stage_best = score > stage_best_score
            if is_new_stage_best:
                stage_best_score = score

            row = {
                "episode": global_episode,
                "global_step": global_step,
                "high_level_decisions": high_level_decisions,
                "stage_index": stage_index,
                "stage_episode": stage_episode,
                "stage_low_level_step": stage_step,
                "stage_step_budget": stage["step_budget"],
                "num_walls": num_walls,
                "epsilon": epsilon,
                "mean_td_loss_1000": (float(np.mean(losses)) if losses else None),
                "train_success_rate_100": float(np.mean(recent_successes)),
                "train_mean_return_100": float(np.mean(recent_returns)),
                "train_mean_length_low_level_100": float(np.mean(recent_lengths)),
                "eval_success_rate": eval_metrics["success_rate"],
                "eval_mean_return": eval_metrics["mean_return"],
                "eval_mean_episode_length": eval_metrics["mean_episode_length_low_level"],
                "eval_mean_high_level_decisions": eval_metrics["mean_high_level_decisions"],
                "eval_choice_counts": eval_metrics["choice_counts"],
                "eval_option_choice_fraction": eval_metrics["option_choice_fraction"],
                "eval_mean_executed_option_duration": eval_metrics["mean_executed_option_duration"],
                "eval_observed_block_option_termination_rate": eval_metrics["observed_block_option_termination_rate"],
                "eval_mean_valid_option_count": eval_metrics["mean_valid_option_count"],
                "transition_reason": transition_reason,
            }
            history.append(row)

            print(
                f"episode={global_episode:05d} "
                f"stage_episode={stage_episode:04d} "
                f"walls={num_walls:02d} "
                f"step={global_step:06d}/{target_step:06d} "
                f"decision={high_level_decisions:06d} "
                f"epsilon={epsilon:.3f} "
                f"train_success={row['train_success_rate_100']:.3f} "
                f"eval_success={row['eval_success_rate']:.3f} "
                f"eval_return={row['eval_mean_return']:.3f} "
                f"option_use={row['eval_option_choice_fraction']:.3f} "
                f"option_k={row['eval_mean_executed_option_duration']:.2f} "
                f"postblock={row['eval_observed_block_option_termination_rate']:.3f}"
            )

            checkpoint_args = (
                online_network,
                optimizer,
                model_config,
                training_config,
                global_episode,
                global_step,
                high_level_decisions,
                stage_index,
                stage_episode,
                eval_metrics,
            )
            save_checkpoint(output_dir / "latest.pt", *checkpoint_args)
            save_checkpoint(
                output_dir / f"latest_walls_{num_walls}.pt", *checkpoint_args
            )
            if is_new_stage_best:
                save_checkpoint(
                    output_dir / f"best_walls_{num_walls}.pt", *checkpoint_args
                )
                print(f"saved new best checkpoint for {num_walls} walls")
                if stage_index == len(curriculum) - 1:
                    save_checkpoint(output_dir / "best.pt", *checkpoint_args)

            (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(
            f"stage finished exactly at low-level step {global_step}; "
            f"best_success={stage_best_score[0]:.3f}"
        )

    print("Training finished")
    print("low-level environment steps:", global_step)
    print("high-level decisions:", high_level_decisions)


if __name__ == "__main__":
    main()
