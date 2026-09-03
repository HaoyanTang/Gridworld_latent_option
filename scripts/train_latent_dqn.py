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
LATENT_BUDGET_MULTIPLIER = int(os.environ.get("GENIE_LATENT_BUDGET_MULTIPLIER"))
LATENT_VALIDITY_MASK = (os.environ.get("GENIE_LATENT_VALIDITY_MASK", "0").strip().lower() in {"1", "true", "yes", "on"})

from src.env.latent_dm_gridworld import LatentDMGridWorldEnv
from src.models.dm import LatentDynamicsModel
from src.models.dqn import TokenDQN
from src.models.vqvae import VQVAEImageTokenizer
from src.rl.replay_buffer import ReplayBuffer


NUM_LATENT_ACTIONS = 4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_tokenizer(
    path: Path,
    device: torch.device,
) -> VQVAEImageTokenizer:

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


def load_dynamics_model(
    path: Path,
    device: torch.device,
) -> LatentDynamicsModel:

    model = LatentDynamicsModel(
        tokenizer_size=128,
        num_actions=NUM_LATENT_ACTIONS,
        state_embedding_dim=64,
        action_embedding_dim=64,
        hidden_dim=128,
    ).to(device)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_dqn(
    model_config: dict[str, Any],
    device: torch.device,
) -> TokenDQN:
    return TokenDQN(**model_config).to(device)


def load_reference_dqn_config(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return dict(checkpoint["model_config"])


def epsilon_at_step(
    step: int,
    start: float,
    end: float,
    decay_steps: int,
) -> float:
    fraction = min(max(step / max(decay_steps, 1), 0.0), 1.0)
    return start + fraction * (end - start)


def load_reference_step_curriculum(
    history_path: Path,
    expected_walls: tuple[int, ...] = (0, 3, 6, 10),
    budget_multiplier: int = 4,
) -> list[dict[str, int]]:
    rows = json.loads(history_path.read_text(encoding="utf-8"))

    curriculum: list[dict[str, int]] = []
    previous_reference_end_step = 0
    previous_scaled_end_step = 0

    for stage_index, num_walls in enumerate(expected_walls):
        stage_rows = [row for row in rows if int(row["num_walls"]) == num_walls]
        transition_rows = [row for row in stage_rows if row["transition_reason"]]
        boundary_row = (
            max(
                transition_rows,
                key=lambda row: int(row["global_step"]),
            )
            if transition_rows
            else max(
                stage_rows,
                key=lambda row: int(row["global_step"]),
            )
        )
        reference_end_step = int(boundary_row["global_step"])
        reference_budget = reference_end_step - previous_reference_end_step
        scaled_budget = reference_budget * budget_multiplier
        scaled_end_step = previous_scaled_end_step + scaled_budget

        curriculum.append(
            {
                "stage_index": stage_index,
                "num_walls": num_walls,
                "start_global_step": previous_scaled_end_step,
                "end_global_step": scaled_end_step,
                "step_budget": scaled_budget,
                "reference_end_episode": int(boundary_row["episode"]),
            }
        )
        previous_reference_end_step = reference_end_step
        previous_scaled_end_step = scaled_end_step

    return curriculum


@torch.inference_mode()
def valid_latent_action_masks(
    env: LatentDMGridWorldEnv,
    states: np.ndarray,
) -> np.ndarray:

    state_array = np.asarray(states, dtype=np.int64)
    single_state = state_array.ndim == 2

    if single_state:
        state_array = state_array[None, ...]


    batch_size = state_array.shape[0]
    state_tensor = torch.from_numpy(state_array).long().to(env.device)

    current_quantized = env.tokenizer.quantizer.codebook(state_tensor)
    current_quantized = current_quantized.permute(0, 3, 1, 2).contiguous()
    current_decoded = env.tokenizer.decoder(current_quantized).clamp_(0.0, 1.0)
    current_decoded_arrays = current_decoded.cpu().numpy()

    current_positions = [
        env._decoded_agent_position(current_decoded_arrays[index])[0]
        for index in range(batch_size)
    ]

    expanded_states = (
        state_tensor.unsqueeze(1)
        .repeat(1, NUM_LATENT_ACTIONS, 1, 1)
        .reshape(
            batch_size * NUM_LATENT_ACTIONS,
            state_tensor.shape[-2],
            state_tensor.shape[-1],
        )
    )
    action_tensor = (
        torch.arange(
            NUM_LATENT_ACTIONS,
            dtype=torch.long,
            device=env.device,
        )
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .reshape(-1)
    )

    logits = env.dynamics_model(expanded_states, action_tensor)
    predictions = logits.argmax(dim=1)

    predicted_quantized = env.tokenizer.quantizer.codebook(predictions)
    predicted_quantized = predicted_quantized.permute(0, 3, 1, 2).contiguous()
    predicted_decoded = env.tokenizer.decoder(
        predicted_quantized
    ).clamp_(0.0, 1.0)
    predicted_decoded_arrays = predicted_decoded.cpu().numpy()

    masks = np.zeros(
        (batch_size, NUM_LATENT_ACTIONS),
        dtype=np.bool_,
    )

    for batch_index in range(batch_size):
        previous_pos = current_positions[batch_index]
        for latent_action in range(NUM_LATENT_ACTIONS):
            flat_index = (
                batch_index * NUM_LATENT_ACTIONS + latent_action
            )
            predicted_pos, _, _ = env._decoded_agent_position(
                predicted_decoded_arrays[flat_index]
            )
            distance = (
                abs(predicted_pos[0] - previous_pos[0])
                + abs(predicted_pos[1] - previous_pos[1])
            )
            masks[batch_index, latent_action] = distance == 1

    return masks[0] if single_state else masks


@torch.inference_mode()
def greedy_action(
    model: TokenDQN,
    state: np.ndarray,
    device: torch.device,
    valid_actions: np.ndarray | None = None,
) -> int:
    state_tensor = torch.from_numpy(state).long().unsqueeze(0).to(device)
    q_values = model(state_tensor)

    if valid_actions is not None:
        valid_tensor = (
            torch.from_numpy(valid_actions)
            .bool()
            .unsqueeze(0)
            .to(device)
        )
        q_values = q_values.masked_fill(~valid_tensor, -torch.inf)

    return int(q_values.argmax(dim=1).item())


def select_action(
    model: TokenDQN,
    state: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
    valid_actions: np.ndarray | None = None,
) -> int:
    if rng.random() < epsilon:
        if valid_actions is None:
            return int(rng.integers(0, NUM_LATENT_ACTIONS))
        return int(rng.choice(np.flatnonzero(valid_actions)))

    return greedy_action(
        model,
        state,
        device,
        valid_actions,
    )


def optimize_double_dqn(
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
    next_valid_actions = (torch.from_numpy(batch.next_valid_actions).bool().to(device))
    q_values = online_network(states)
    chosen_q_values = (q_values.gather(1, actions.unsqueeze(1)).squeeze(1))

    with torch.no_grad():
        next_online_q = online_network(next_states)
        next_online_q = next_online_q.masked_fill(~next_valid_actions, -torch.inf)
        next_actions = next_online_q.argmax(dim=1, keepdim=True)
        next_q_values = (target_network(next_states).gather(1, next_actions).squeeze(1))
        td_targets = (rewards + gamma * (1.0 - dones) * next_q_values)

    loss = F.smooth_l1_loss(chosen_q_values, td_targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_network.parameters(), gradient_clip)
    optimizer.step()

    return float(loss.item())


@torch.inference_mode()
def evaluate(
    model: TokenDQN,
    env: LatentDMGridWorldEnv,
    episodes: int,
    seed_start: int,
    device: torch.device,
    latent_validity_mask: bool,
) -> dict[str, Any]:
    
    was_training = model.training
    model.eval()

    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    truncations = 0
    blocked_steps = 0
    invalid_dm_motions = 0
    all_invalid_fallbacks = 0
    action_counts = np.zeros(NUM_LATENT_ACTIONS, dtype=np.int64)
    decoder_margins: list[float] = []
    valid_action_count_sum = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=seed_start + episode)
        episode_return = 0.0

        for step in range(1, env.max_steps + 1):
            valid_actions = (
                valid_latent_action_masks(env, state)
                if latent_validity_mask
                else None
            )
            valid_action_count_sum += (
                int(valid_actions.sum())
                if valid_actions is not None
                else NUM_LATENT_ACTIONS
            )

            if valid_actions is not None and not valid_actions.any():
                all_invalid_fallbacks += 1
                valid_actions[:] = True

            latent_action = greedy_action(
                model,
                state,
                device,
                valid_actions,
            )
            action_counts[latent_action] += 1

            state, reward, terminated, truncated, info = env.step(latent_action)
            episode_return += reward
            blocked_steps += int(info["blocked"])
            invalid_dm_motions += int(info["invalid_dm_motion"])
            decoder_margins.append(float(info["decoder_agent_margin"]))

            if terminated or truncated:
                successes += int(terminated)
                truncations += int(truncated)
                lengths.append(step)
                break

        returns.append(episode_return)

    if was_training:
        model.train()

    total_steps = int(sum(lengths))
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "truncation_rate": truncations / episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "total_steps": total_steps,
        "blocked_steps": blocked_steps,
        "blocked_step_rate": blocked_steps / max(total_steps, 1),
        "invalid_dm_motions": invalid_dm_motions,
        "invalid_dm_motion_rate": invalid_dm_motions / max(total_steps, 1),
        "mean_decoder_agent_margin": float(np.mean(decoder_margins)),
        "latent_action_counts": action_counts.tolist(),
        "mean_valid_latent_action_count": (valid_action_count_sum / max(total_steps, 1)),
        "all_invalid_fallbacks": all_invalid_fallbacks,
        "all_invalid_fallback_rate": (all_invalid_fallbacks / max(total_steps, 1)),
    }


def save_checkpoint(
    path: Path,
    online_network: TokenDQN,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    global_episode: int,
    global_step: int,
    stage_index: int,
    stage_episode: int,
    eval_metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "format_version": 1,
            "experiment": "latent_dqn_frozen_dm",
            "model_state_dict": online_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "training_config": training_config,
            "episode": global_episode,
            "global_step": global_step,
            "stage_index": stage_index,
            "stage_episode": stage_episode,
            "eval_metrics": eval_metrics,
        },
        path,
    )


def main() -> None:
    configured_action_dir = os.environ.get("GENIE_ACTION_OUTPUT_DIR")
    reference_action_dir = (
        Path(configured_action_dir)
        if configured_action_dir
        else RUN_ROOT / "outputs" / "dqn_action_improved"
    )
    default_output_dir = (
        RUN_ROOT
        / "outputs"
        / (
            "dqn_latent_dm_masked"
            if LATENT_VALIDITY_MASK
            else "dqn_latent_dm"
        )
    )

    training_config: dict[str, Any] = {
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
        "goal_reward": 1.0,
        "movement_reward": -0.01,
        "blocked_reward": -0.05,
        "enforce_valid_motion": True,
        "validate_reset_detection": True,
        "latent_validity_mask": LATENT_VALIDITY_MASK,
        "seed": EXPERIMENT_SEED,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tokenizer": str(
            RUN_ROOT / "outputs" / "tokenizer" / "model" / "best.pt"
        ),
        "dynamics_model": str(
            RUN_ROOT / "outputs" / "dm_hybrid" / "best.pt"
        ),
        "reference_action_dqn": str(
            reference_action_dir / "best.pt"
        ),
        "reference_action_history": str(
            reference_action_dir / "history.json"
        ),
        "output_dir": os.environ.get(
            "GENIE_OUTPUT_DIR",
            str(default_output_dir),
        ),
        "expected_curriculum_walls": [0, 3, 6, 10],
    }

    seed_everything(int(training_config["seed"]))
    rng = np.random.default_rng(training_config["seed"])
    device = torch.device(training_config["device"])
    tokenizer_path = Path(training_config["tokenizer"])
    dynamics_path = Path(training_config["dynamics_model"])
    reference_dqn_path = Path(training_config["reference_action_dqn"])
    reference_history_path = Path(training_config["reference_action_history"])
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    curriculum = load_reference_step_curriculum(
        reference_history_path,
        tuple(training_config["expected_curriculum_walls"]),
        LATENT_BUDGET_MULTIPLIER,
    )
    training_config["curriculum"] = curriculum
    training_config["max_global_steps"] = curriculum[-1]["end_global_step"]

    print(
        "latent validity mask:",
        "ON" if training_config["latent_validity_mask"] else "OFF",
    )
    for stage in curriculum:
        print(
            f"  {stage['num_walls']:2d} walls: "
            f"[{stage['start_global_step']}, "
            f"{stage['end_global_step']}) "
            f"budget={stage['step_budget']}"
        )
    print("total step budget:", training_config["max_global_steps"])
    print("output:", output_dir)

    tokenizer = load_tokenizer(tokenizer_path, device)
    dynamics_model = load_dynamics_model(dynamics_path, device)

    first_num_walls = curriculum[0]["num_walls"]
    env_kwargs = {
        "tokenizer": tokenizer,
        "dynamics_model": dynamics_model,
        "device": device,
        "grid_size": training_config["grid_size"],
        "image_size": training_config["image_size"],
        "max_steps": training_config["max_steps"],
        "num_walls": first_num_walls,
        "goal_reward": training_config["goal_reward"],
        "movement_reward": training_config["movement_reward"],
        "blocked_reward": training_config["blocked_reward"],
        "enforce_valid_motion": training_config["enforce_valid_motion"],
        "validate_reset_detection": training_config["validate_reset_detection"],
    }
    train_env = LatentDMGridWorldEnv(**env_kwargs, seed=training_config["seed"])
    eval_env = LatentDMGridWorldEnv(**env_kwargs, seed=training_config["eval_seed_start"])

    train_env.reset(seed=training_config["seed"])
    eval_env.reset(seed=training_config["eval_seed_start"])

    model_config = load_reference_dqn_config(reference_dqn_path)
    online_network = build_dqn(model_config, device)
    target_network = build_dqn(model_config, device)
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

    global_episode = 0
    global_step = 0
    losses: deque[float] = deque(maxlen=1_000)
    history: list[dict[str, Any]] = []

    for stage_index, stage in enumerate(curriculum):
        num_walls = int(stage["num_walls"])
        stage_start_step = int(stage["start_global_step"])
        stage_end_step = int(stage["end_global_step"])

        train_env.num_walls = num_walls
        eval_env.num_walls = num_walls

        recent_returns: deque[float] = deque(maxlen=100)
        recent_lengths: deque[int] = deque(maxlen=100)
        recent_successes: deque[int] = deque(maxlen=100)
        recent_blocked_rates: deque[float] = deque(maxlen=100)

        best_stage_score = (-1.0, float("-inf"))
        final_stage = stage_index == len(training_config["curriculum"]) - 1
        stage_episode = 0

        print(
            f"\n===== curriculum stage: {num_walls} walls "
            f"({stage_start_step} -> {stage_end_step} steps) ====="
        )

        while global_step < stage_end_step:
            stage_episode += 1
            global_episode += 1
            state, _ = train_env.reset()
            episode_return = 0.0
            episode_blocked = 0
            reached_stage_boundary = False

            valid_actions = (valid_latent_action_masks(train_env, state)if training_config["latent_validity_mask"] else None)
            if (valid_actions is not None and not valid_actions.any()):
                valid_actions[:] = True

            for episode_step in range(1, training_config["max_steps"] + 1):
                stage_step = global_step - stage_start_step
                epsilon = epsilon_at_step(
                    stage_step,
                    training_config["epsilon_start"]
                    if stage_index == 0
                    else training_config["epsilon_restart"],
                    training_config["epsilon_end"],
                    training_config["epsilon_decay_steps"],
                )

                latent_action = select_action(
                    online_network,
                    state,
                    epsilon,
                    rng,
                    device,
                    valid_actions,
                )

                next_state, reward, terminated, truncated, info = (
                    train_env.step(latent_action)
                )
                next_global_step = global_step + 1
                reached_stage_boundary = next_global_step >= stage_end_step
                episode_done = (terminated or truncated or reached_stage_boundary)
                replay_done = (terminated or reached_stage_boundary)

                if ( training_config["latent_validity_mask"] and not replay_done):
                    next_valid_actions = valid_latent_action_masks(train_env, next_state)
                    if not next_valid_actions.any():
                        next_valid_actions[:] = True
                else:
                    next_valid_actions = np.ones(NUM_LATENT_ACTIONS, dtype=np.bool_)

                replay_buffer.add(
                    state,
                    latent_action,
                    reward,
                    next_state,
                    replay_done,
                    next_valid_actions,
                )

                state = next_state
                valid_actions = (next_valid_actions if training_config["latent_validity_mask"] else None)
                episode_return += reward
                episode_blocked += int(info["blocked"])
                global_step = next_global_step

                if (
                    global_step >= training_config["learning_starts"]
                    and len(replay_buffer) >= training_config["batch_size"]
                ):
                    loss = optimize_double_dqn(
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
                    recent_blocked_rates.append(episode_blocked / episode_step)
                    break

            should_evaluate = (
                stage_episode % training_config["eval_interval"] == 0
                or reached_stage_boundary
            )
            if not should_evaluate:
                continue

            eval_metrics = evaluate(
                online_network,
                eval_env,
                training_config["eval_episodes"],
                training_config["eval_seed_start"],
                device,
                training_config["latent_validity_mask"],
            )
            stage_step = global_step - stage_start_step
            epsilon = epsilon_at_step(
                stage_step,
                training_config["epsilon_start"]
                if stage_index == 0
                else training_config["epsilon_restart"],
                training_config["epsilon_end"],
                training_config["epsilon_decay_steps"],
            )

            current_score = (
                eval_metrics["success_rate"],
                eval_metrics["mean_return"],
            )
            is_new_stage_best = current_score > best_stage_score
            if is_new_stage_best:
                best_stage_score = current_score

            transition_reason = (
                "reference_step_boundary"
                if reached_stage_boundary
                else None
            )

            row = {
                "episode": global_episode,
                "global_step": global_step,
                "stage_index": stage_index,
                "stage_episode": stage_episode,
                "num_walls": num_walls,
                "stage_start_global_step": stage_start_step,
                "stage_end_global_step": stage_end_step,
                "stage_step": stage_step,
                "stage_step_budget": int(stage["step_budget"]),
                "epsilon": epsilon,
                "mean_td_loss_1000": (float(np.mean(losses)) if losses else None),
                "train_success_rate_100": float(np.mean(recent_successes)),
                "train_mean_return_100": float(np.mean(recent_returns)),
                "train_mean_length_100": float(np.mean(recent_lengths)),
                "train_blocked_step_rate_100": float(np.mean(recent_blocked_rates)),
                "eval_success_rate": eval_metrics["success_rate"],
                "eval_mean_return": eval_metrics["mean_return"],
                "eval_mean_episode_length": eval_metrics["mean_episode_length"],
                "eval_truncation_rate": eval_metrics["truncation_rate"],
                "eval_blocked_step_rate": eval_metrics["blocked_step_rate"],
                "eval_invalid_dm_motion_rate": eval_metrics["invalid_dm_motion_rate"],
                "eval_mean_decoder_agent_margin": eval_metrics["mean_decoder_agent_margin"],
                "eval_latent_action_counts": eval_metrics["latent_action_counts"],
                "eval_mean_valid_latent_action_count": eval_metrics["mean_valid_latent_action_count"],
                "latent_validity_mask": training_config["latent_validity_mask"],
                "transition_reason": transition_reason,
                "eval_all_invalid_fallback_rate": eval_metrics["all_invalid_fallback_rate"],
            }
            history.append(row)

            print(
                f"episode={global_episode:05d} "
                f"stage_episode={stage_episode:04d} "
                f"walls={num_walls:02d} "
                f"step={global_step:06d} "
                f"stage_step={stage_step:05d}/{stage['step_budget']:05d} "
                f"epsilon={epsilon:.3f} "
                f"train_success={row['train_success_rate_100']:.3f} "
                f"eval_success={row['eval_success_rate']:.3f} "
                f"eval_return={row['eval_mean_return']:.3f} "
                f"eval_length={row['eval_mean_episode_length']:.2f} "
                f"valid_actions={row['eval_mean_valid_latent_action_count']:.2f} "
                f"blocked={row['eval_blocked_step_rate']:.3f} "
                f"invalid_dm={row['eval_invalid_dm_motion_rate']:.3f}"
            )

            save_checkpoint(
                output_dir / "latest.pt",
                online_network,
                optimizer,
                model_config,
                training_config,
                global_episode,
                global_step,
                stage_index,
                stage_episode,
                eval_metrics,
            )

            if is_new_stage_best:
                stage_checkpoint = (
                    output_dir / f"best_stage_{num_walls}walls.pt"
                )
                save_checkpoint(
                    stage_checkpoint,
                    online_network,
                    optimizer,
                    model_config,
                    training_config,
                    global_episode,
                    global_step,
                    stage_index,
                    stage_episode,
                    eval_metrics,
                )
                if final_stage:
                    save_checkpoint(
                        output_dir / "best.pt",
                        online_network,
                        optimizer,
                        model_config,
                        training_config,
                        global_episode,
                        global_step,
                        stage_index,
                        stage_episode,
                        eval_metrics,
                    )
                print(f"saved new best checkpoint for {num_walls} walls")

            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )

            if transition_reason is not None:
                print(
                    f"finished {num_walls}-wall stage: "
                    f"{transition_reason} at global_step={global_step}"
                )
                break

    print("\nTraining finished")


if __name__ == "__main__":
    main()
