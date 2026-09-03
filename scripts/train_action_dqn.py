from __future__ import annotations

import json
import os
import random
import sys
from collections import deque
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
from src.rl.replay_buffer import ReplayBuffer


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
    tokenizer.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    return tokenizer

def epsilon_at_step(
    step: int,
    start: float,
    end: float,
    decay_steps: int,
) -> float:
    
    fraction = min(max(step / max(decay_steps, 1), 0.0), 1.0)
    return start + fraction * (end - start)

@torch.no_grad()
def greedy_action(
    model: TokenDQN,
    state: np.ndarray,
    device: torch.device,
) -> int:
    
    state_tensor = torch.from_numpy(state).long().unsqueeze(0).to(device)
    q_values = model(state_tensor)
    return int(q_values.argmax(dim=1).item())

def select_action(
    model: TokenDQN,
    state: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    
    if rng.random() < epsilon:
        return int(rng.integers(0, NUM_ACTIONS))
    return greedy_action(model, state, device)

def optimize_dqn(
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
    actions = torch.from_numpy(batch.actions).long().to(device)
    rewards = torch.from_numpy(batch.rewards).float().to(device)
    next_states = torch.from_numpy(batch.next_states).long().to(device)
    dones = torch.from_numpy(batch.dones).float().to(device)

    q_values = online_network(states)
    chosen_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_actions = online_network(next_states).argmax(dim=1, keepdim=True)
        next_q_values = target_network(next_states).gather(1, next_actions).squeeze(1)
        td_targets = rewards + gamma * (1.0 - dones) * next_q_values

    loss = F.smooth_l1_loss(chosen_q_values, td_targets)

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
    device: torch.device,
) -> dict[str, Any]:
    training = model.training
    model.eval()

    returns = []
    lengths = []
    successes = 0
    truncations = 0
    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)

    for episode in range(episodes):
        state, _ = env.reset(seed=seed_start + episode)
        episode_return = 0.0

        for step in range(1, env.env.max_steps + 1):
            action = greedy_action(model, state, device)
            action_counts[action] += 1

            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward

            if terminated or truncated:
                successes += int(terminated)
                truncations += int(truncated)
                lengths.append(step)
                break

        returns.append(episode_return)

    if training:
        model.train()

    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "truncation_rate": truncations / episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "action_counts": action_counts.tolist(),
    }


def save_checkpoint(
    path: Path,
    online_network: TokenDQN,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    episode: int,
    global_step: int,
    stage_index: int,
    stage_episode: int,
    num_walls: int,
    eval_metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": online_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "training_config": training_config,
            "episode": episode,
            "global_step": global_step,
            "stage_index": stage_index,
            "stage_episode": stage_episode,
            "num_walls": num_walls,
            "eval_metrics": eval_metrics,
        },
        path,
    )


def main() -> None:
    training_config = {
        "batch_size": 64,
        "buffer_capacity": 50000,
        "learning_starts": 1000,
        "target_update_frequency": 1000,
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gradient_clip": 10.0,
        "epsilon_start": 1.0,
        "epsilon_restart": 0.6,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 30000,
        "eval_interval": 100,
        "eval_episodes": 100,
        "eval_seed_start": 100000,
        "grid_size": 8,
        "image_size": 64,
        "max_steps": 32,
        "curriculum": [
            {
                "num_walls": 0,
                "min_episodes": 1000,
                "max_episodes": 2000,
                "success_threshold": 0.95,
                "required_passes": 3,
            },
            {
                "num_walls": 3,
                "min_episodes": 1500,
                "max_episodes": 3000,
                "success_threshold": 0.90,
                "required_passes": 3,
            },
            {
                "num_walls": 6,
                "min_episodes": 2000,
                "max_episodes": 4000,
                "success_threshold": 0.80,
                "required_passes": 3,
            },
            {
                "num_walls": 10,
                "min_episodes": 3000,
                "max_episodes": 6000,
                "success_threshold": 0.70,
                "required_passes": 3,
            },
        ],
        "seed": EXPERIMENT_SEED,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tokenizer": str(
            RUN_ROOT / "outputs" / "tokenizer" / "model" / "best.pt"
        ),
        "output_dir": os.environ.get(
            "GENIE_OUTPUT_DIR",
            str(RUN_ROOT / "outputs" / "dqn_action_improved"),
        ),
    }

    set_seed(training_config["seed"])
    rng = np.random.default_rng(training_config["seed"])
    device = torch.device(training_config["device"])
    tokenizer_path = Path(training_config["tokenizer"])
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_path, device)

    initial_num_walls = training_config["curriculum"][0]["num_walls"]

    train_env = TokenizedGridWorldEnv(
        tokenizer=tokenizer,
        device=device,
        grid_size=training_config["grid_size"],
        image_size=training_config["image_size"],
        max_steps=training_config["max_steps"],
        num_walls=initial_num_walls,
        seed=training_config["seed"],
    )
    eval_env = TokenizedGridWorldEnv(
        tokenizer=tokenizer,
        device=device,
        grid_size=training_config["grid_size"],
        image_size=training_config["image_size"],
        max_steps=training_config["max_steps"],
        num_walls=initial_num_walls,
        seed=training_config["eval_seed_start"],
    )

    model_config = {
        "tokenizer_size": 128,
        "num_actions": NUM_ACTIONS,
        "token_embedding_dim": 32,
        "hidden_channels": 64,
        "hidden_dim": 256,
        "token_grid_size": 8,
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
        state_shape=(8, 8),
        seed=training_config["seed"],
    )

    global_step = 0
    global_episode = 0
    losses: deque[float] = deque(maxlen=1_000)
    recent_returns: deque[float] = deque(maxlen=100)
    recent_lengths: deque[int] = deque(maxlen=100)
    recent_successes: deque[int] = deque(maxlen=100)
    history: list[dict[str, Any]] = []

    final_stage_best_success = -1.0
    final_stage_best_return = float("-inf")

    for stage_index, stage in enumerate(training_config["curriculum"]):
        current_num_walls = stage["num_walls"]
        epsilon_step = 0
        success_passes = 0
        stage_best_success = -1.0
        stage_best_return = float("-inf")

        losses.clear()
        recent_returns.clear()
        recent_lengths.clear()
        recent_successes.clear()

        if stage_index > 0:
            train_env = TokenizedGridWorldEnv(
                tokenizer=tokenizer,
                device=device,
                grid_size=training_config["grid_size"],
                image_size=training_config["image_size"],
                max_steps=training_config["max_steps"],
                num_walls=current_num_walls,
                seed=training_config["seed"] + global_episode,
            )
            eval_env = TokenizedGridWorldEnv(
                tokenizer=tokenizer,
                device=device,
                grid_size=training_config["grid_size"],
                image_size=training_config["image_size"],
                max_steps=training_config["max_steps"],
                num_walls=current_num_walls,
                seed=training_config["eval_seed_start"],
            )

        print(
            f"\n===== Curriculum stage {stage_index + 1}/"
            f"{len(training_config['curriculum'])} =====\n"
            f"num_walls={current_num_walls}, "
        )

        for stage_episode in range(1, stage["max_episodes"] + 1):
            global_episode += 1
            state, _ = train_env.reset()
            episode_return = 0.0

            for episode_step in range(1, training_config["max_steps"] + 1):
                epsilon = epsilon_at_step(
                    epsilon_step,
                    training_config["epsilon_start"]
                    if stage_index == 0
                    else training_config["epsilon_restart"],
                    training_config["epsilon_end"],
                    training_config["epsilon_decay_steps"],
                )
                action = select_action(
                    online_network,
                    state,
                    epsilon,
                    rng,
                    device,
                )

                next_state, reward, terminated, truncated, _ = train_env.step(action)
                episode_done = terminated or truncated
                replay_done = terminated
                replay_buffer.add(
                    state,
                    action,
                    reward,
                    next_state,
                    replay_done,
                )

                state = next_state
                episode_return += reward
                global_step += 1
                epsilon_step += 1

                if (
                    global_step >= training_config["learning_starts"]
                    and len(replay_buffer) >= training_config["batch_size"]
                ):
                    loss = optimize_dqn(
                        online_network,
                        target_network,
                        optimizer,
                        replay_buffer,
                        training_config["batch_size"],
                        training_config["gamma"],
                        training_config["gradient_clip"],
                        device,
                    )
                    losses.append(loss)

                if global_step % training_config["target_update_frequency"] == 0:
                    target_network.load_state_dict(online_network.state_dict())

                if episode_done:
                    recent_returns.append(episode_return)
                    recent_lengths.append(episode_step)
                    recent_successes.append(int(terminated))
                    break

            should_evaluate = (
                stage_episode % training_config["eval_interval"] == 0
                or stage_episode == stage["max_episodes"]
            )
            if not should_evaluate:
                continue

            eval_metrics = evaluate(
                online_network,
                eval_env,
                training_config["eval_episodes"],
                training_config["eval_seed_start"],
                device,
            )
            epsilon = epsilon_at_step(
                epsilon_step,
                training_config["epsilon_start"]
                if stage_index == 0
                else training_config["epsilon_restart"],
                training_config["epsilon_end"],
                training_config["epsilon_decay_steps"],
            )

            if (
                stage_episode >= stage["min_episodes"]
                and eval_metrics["success_rate"] >= stage["success_threshold"]
            ):
                success_passes += 1
            else:
                success_passes = 0

            current_score = (eval_metrics["success_rate"], eval_metrics["mean_return"])
            stage_best_score = (stage_best_success, stage_best_return)
            is_new_stage_best = current_score > stage_best_score
            if is_new_stage_best:
                stage_best_success, stage_best_return = current_score

            transition_reason = None
            if success_passes >= stage["required_passes"]:
                transition_reason = "success_threshold"
            elif stage_episode >= stage["max_episodes"]:
                transition_reason = "max_episodes"

            row = {
                "episode": global_episode,
                "global_step": global_step,
                "stage_index": stage_index,
                "stage_episode": stage_episode,
                "num_walls": current_num_walls,
                "epsilon": epsilon,
                "mean_td_loss_1000": float(np.mean(losses)) if losses else None,
                "train_success_rate_100": float(np.mean(recent_successes)),
                "train_mean_return_100": float(np.mean(recent_returns)),
                "train_mean_length_100": float(np.mean(recent_lengths)),
                "eval_success_rate": eval_metrics["success_rate"],
                "eval_mean_return": eval_metrics["mean_return"],
                "eval_mean_episode_length": eval_metrics["mean_episode_length"],
                "eval_truncation_rate": eval_metrics["truncation_rate"],
                "eval_action_counts": eval_metrics["action_counts"],
                "success_passes": success_passes,
                "required_passes": stage["required_passes"],
                "transition_reason": transition_reason,
            }
            history.append(row)

            print(
                f"episode={global_episode:05d} "
                f"stage_episode={stage_episode:04d} "
                f"walls={current_num_walls:02d} "
                f"step={global_step:06d} "
                f"epsilon={epsilon:.3f} "
                f"train_success_100={row['train_success_rate_100']:.3f} "
                f"eval_success={row['eval_success_rate']:.3f} "
                f"passes={success_passes}/{stage['required_passes']} "
                f"eval_return={row['eval_mean_return']:.3f} "
                f"eval_length={row['eval_mean_episode_length']:.3f}"
            )

            checkpoint_arguments = (
                online_network,
                optimizer,
                model_config,
                training_config,
                global_episode,
                global_step,
                stage_index,
                stage_episode,
                current_num_walls,
                eval_metrics,
            )
            save_checkpoint(output_dir / "latest.pt", *checkpoint_arguments)
            save_checkpoint(output_dir / f"latest_walls_{current_num_walls}.pt", *checkpoint_arguments)
            if is_new_stage_best:
                save_checkpoint(output_dir / f"best_walls_{current_num_walls}.pt", *checkpoint_arguments)
                print(f"saved new best checkpoint for {current_num_walls} walls")

                if stage_index == len(training_config["curriculum"]) - 1:
                    final_stage_best_success = stage_best_success
                    final_stage_best_return = stage_best_return
                    save_checkpoint(output_dir / "best.pt", *checkpoint_arguments)

            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )

            if transition_reason is not None:
                print(
                    f"stage finished: walls={current_num_walls}, "
                    f"reason={transition_reason}, "
                    f"stage_episodes={stage_episode}, "
                    f"best_success={stage_best_success:.3f}"
                )
                break

    print("Training finished")
    print("final-stage best success rate:", final_stage_best_success)
    print("final-stage best mean return:", final_stage_best_return)


if __name__ == "__main__":
    main()
