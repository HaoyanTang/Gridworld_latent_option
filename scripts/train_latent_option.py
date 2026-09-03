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
LATENT_BUDGET_MULTIPLIER = int(os.environ.get("GENIE_LATENT_BUDGET_MULTIPLIER"))

from src.env.latent_dm_gridworld import LatentDMGridWorldEnv
from src.models.dm import LatentDynamicsModel
from src.models.dqn import TokenDQN
from src.models.vqvae import VQVAEImageTokenizer
from src.rl.replay_buffer_option import ReplayBuffer


NUM_LATENT_ACTIONS = 4
OPTION_LENGTH = 2
NUM_HIGH_LEVEL_CHOICES = NUM_LATENT_ACTIONS * 2


@dataclass
class LatentPreview:
    next_state: np.ndarray
    next_agent_pos: tuple[int, int]
    moved: bool
    invalid_motion: bool
    decoder_margin: float


@dataclass
class ChoiceResult:
    next_state: np.ndarray
    discounted_return: float
    raw_return: float
    duration: int
    terminated: bool
    truncated: bool
    observed_block_termination: bool
    executed_blocked_steps: int
    executed_invalid_dm_steps: int
    decoder_margins: list[float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_tokenizer(path: Path, device: torch.device) -> VQVAEImageTokenizer:
    tokenizer = VQVAEImageTokenizer(
        num_codes=128,
        embedding_dim=64,
        commitment_beta=0.25,
    ).to(device)
    tokenizer.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
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


def find_reference_action_files() -> tuple[Path, Path]:
    directory = Path(os.environ["GENIE_ACTION_OUTPUT_DIR"])
    return directory / "best.pt", directory / "history.json"


def load_reference_dqn_config(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["model_config"])
    config["num_actions"] = NUM_HIGH_LEVEL_CHOICES
    return config


def build_dqn(config: dict[str, Any], device: torch.device) -> TokenDQN:
    return TokenDQN(**config).to(device)


def load_step_curriculum(
    path: Path,
    expected_walls: tuple[int, ...] = (0, 3, 6, 10),
    budget_multiplier: int = 4,
) -> list[dict[str, int]]:
    rows = json.loads(path.read_text(encoding="utf-8"))

    curriculum: list[dict[str, int]] = []
    previous_reference_end = 0
    previous_scaled_end = 0
    for stage_index, walls in enumerate(expected_walls):
        stage_rows = [row for row in rows if int(row["num_walls"]) == walls]
        transition_rows = [row for row in stage_rows if row["transition_reason"]]
        source = transition_rows if transition_rows else stage_rows
        boundary = max(source, key=lambda row: int(row["global_step"]))
        reference_end_step = int(boundary["global_step"])
        reference_budget = reference_end_step - previous_reference_end
        scaled_budget = reference_budget * budget_multiplier
        scaled_end_step = previous_scaled_end + scaled_budget
        curriculum.append(
            {
                "stage_index": stage_index,
                "num_walls": walls,
                "start_step": previous_scaled_end,
                "end_step": scaled_end_step,
                "step_budget": scaled_budget,
                "reference_start_step": previous_reference_end,
                "reference_end_step": reference_end_step,
                "reference_step_budget": reference_budget,
                "budget_multiplier": budget_multiplier,
            }
        )
        previous_reference_end = reference_end_step
        previous_scaled_end = scaled_end_step
    return curriculum


def epsilon_at_step(
    step: int,
    start: float,
    end: float,
    decay_steps: int,
) -> float:
    fraction = min(max(step / max(decay_steps, 1), 0.0), 1.0)
    return start + fraction * (end - start)


def _classify_prediction(
    env: LatentDMGridWorldEnv,
    state: np.ndarray,
    raw_prediction: np.ndarray,
    decoded_frame: np.ndarray,
) -> LatentPreview:
    predicted_pos, _, margin = env._decoded_agent_position(decoded_frame)
    previous_pos = env.agent_pos
    distance = (abs(predicted_pos[0] - previous_pos[0]) + abs(predicted_pos[1] - previous_pos[1]))
    invalid_motion = distance > 1
    moved = distance == 1

    if env.enforce_valid_motion and (invalid_motion or not moved):
        next_state = np.array(state, copy=True)
        next_agent_pos = previous_pos
        moved = False
    else:
        next_state = np.array(raw_prediction, copy=True)
        next_agent_pos = predicted_pos

    return LatentPreview(
        next_state=next_state,
        next_agent_pos=next_agent_pos,
        moved=bool(moved),
        invalid_motion=bool(invalid_motion),
        decoder_margin=float(margin),
    )


@torch.inference_mode()
def preview_all_latent_actions(
    env: LatentDMGridWorldEnv,
    state: np.ndarray,
) -> list[LatentPreview]:
    state_array = np.array(state, copy=True)
    state_tensor = torch.from_numpy(state_array).long().unsqueeze(0)
    state_tensor = state_tensor.repeat(NUM_LATENT_ACTIONS, 1, 1).to(env.device)
    action_tensor = torch.arange(
        NUM_LATENT_ACTIONS,
        dtype=torch.long,
        device=env.device,
    )

    logits = env.dynamics_model(state_tensor, action_tensor)
    predictions = logits.argmax(dim=1)
    quantized = env.tokenizer.quantizer.codebook(predictions)
    quantized = quantized.permute(0, 3, 1, 2).contiguous()
    decoded = env.tokenizer.decoder(quantized).clamp_(0.0, 1.0)

    prediction_arrays = predictions.cpu().numpy().astype(np.int64)
    decoded_arrays = decoded.cpu().numpy()
    return [
        _classify_prediction(
            env,
            state_array,
            prediction_arrays[action],
            decoded_arrays[action],
        )
        for action in range(NUM_LATENT_ACTIONS)
    ]


def valid_choice_mask(
    previews: list[LatentPreview],
) -> np.ndarray:
    movement_mask = np.asarray([preview.moved for preview in previews], dtype=np.bool_)
    if movement_mask.any():
        return np.concatenate([movement_mask, movement_mask])
    return np.concatenate([np.ones(NUM_LATENT_ACTIONS, dtype=np.bool_), np.zeros(NUM_LATENT_ACTIONS, dtype=np.bool_)])


@torch.inference_mode()
def greedy_choice(
    model: TokenDQN,
    state: np.ndarray,
    device: torch.device,
    valid_choices: np.ndarray,
) -> int:
    state_tensor = torch.from_numpy(np.asarray(state)).long().unsqueeze(0).to(device)
    q_values = model(state_tensor)
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
        return int(rng.choice(np.flatnonzero(valid_choices)))
    return greedy_choice(model, state, device, valid_choices)


def execute_choice(
    env: LatentDMGridWorldEnv,
    state: np.ndarray,
    choice: int,
    gamma: float,
    max_low_level_steps: int,
) -> ChoiceResult:
    is_option = choice >= NUM_LATENT_ACTIONS
    latent_action = choice if not is_option else choice - NUM_LATENT_ACTIONS
    requested_duration = OPTION_LENGTH if is_option else 1
    allowed_duration = min(requested_duration, max_low_level_steps)

    current_state = state
    discounted_return = 0.0
    raw_return = 0.0
    duration = 0
    terminated = False
    truncated = False
    observed_block_termination = False
    executed_blocked_steps = 0
    executed_invalid_dm_steps = 0
    decoder_margins: list[float] = []

    for _ in range(allowed_duration):
        next_state, reward, terminated, truncated, info = env.step(latent_action)
        discounted_return += (gamma**duration) * float(reward)
        raw_return += float(reward)
        duration += 1
        current_state = next_state
        executed_blocked_steps += int(info["blocked"])
        executed_invalid_dm_steps += int(info["invalid_dm_motion"])
        decoder_margins.append(float(info["decoder_agent_margin"]))

        if (
            is_option
            and info["blocked"]
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
        executed_blocked_steps=executed_blocked_steps,
        executed_invalid_dm_steps=executed_invalid_dm_steps,
        decoder_margins=decoder_margins,
    )


def optimize_smdp_double_dqn(
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
    next_valid = torch.from_numpy(batch.next_valid_choices).bool().to(device)

    chosen_q = online_network(states).gather(1, choices.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_online_q = online_network(next_states).masked_fill(
            ~next_valid, -torch.inf
        )
        next_choices = next_online_q.argmax(dim=1, keepdim=True)
        next_q = target_network(next_states).gather(
            1, next_choices
        ).squeeze(1)
        discounts = torch.pow(torch.full_like(durations, gamma), durations)
        targets = returns + discounts * (1.0 - dones) * next_q

    loss = F.smooth_l1_loss(chosen_q, targets)
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
    choice_counts = np.zeros(NUM_HIGH_LEVEL_CHOICES, dtype=np.int64)
    option_executions = 0
    option_duration_sum = 0
    observed_block_terminations = 0
    valid_option_count_sum = 0
    blocked_steps = 0
    invalid_dm_steps = 0
    decoder_margins: list[float] = []
    all_invalid_fallbacks = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=seed_start + episode)
        episode_return = 0.0
        low_steps = 0
        high_decisions = 0

        while low_steps < env.max_steps:
            previews = preview_all_latent_actions(env, state)
            all_invalid_fallbacks += int(not any(preview.moved for preview in previews))
            valid_choices = valid_choice_mask(previews)
            valid_option_count_sum += int(
                valid_choices[NUM_LATENT_ACTIONS:].sum()
            )
            choice = greedy_choice(model, state, device, valid_choices)
            choice_counts[choice] += 1
            high_decisions += 1

            result = execute_choice(
                env,
                state,
                choice,
                gamma,
                env.max_steps - low_steps,
            )
            state = result.next_state
            episode_return += result.raw_return
            low_steps += result.duration
            blocked_steps += result.executed_blocked_steps
            invalid_dm_steps += result.executed_invalid_dm_steps
            decoder_margins.extend(result.decoder_margins)

            if choice >= NUM_LATENT_ACTIONS:
                option_executions += 1
                option_duration_sum += result.duration
                observed_block_terminations += int(
                    result.observed_block_termination
                )

            if result.terminated or result.truncated:
                successes += int(result.terminated)
                truncations += int(result.truncated)
                break

        returns.append(episode_return)
        lengths.append(low_steps)
        decisions.append(high_decisions)

    if was_training:
        model.train()

    total_steps = int(sum(lengths))
    primitive_count = int(choice_counts[:NUM_LATENT_ACTIONS].sum())
    option_count = int(choice_counts[NUM_LATENT_ACTIONS:].sum())
    total_choices = primitive_count + option_count
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
        "option_choice_fraction": option_count / max(total_choices, 1),
        "mean_executed_option_duration": (
            option_duration_sum / option_executions
            if option_executions
            else 0.0
        ),
        "observed_block_option_termination_rate": (
            observed_block_terminations / option_executions
            if option_executions
            else 0.0
        ),
        "mean_valid_option_count": (valid_option_count_sum / max(sum(decisions), 1)),
        "blocked_step_rate": blocked_steps / max(total_steps, 1),
        "invalid_dm_motion_rate": invalid_dm_steps / max(total_steps, 1),
        "mean_decoder_agent_margin": (float(np.mean(decoder_margins)) if decoder_margins else 0.0),
        "all_invalid_fallbacks": all_invalid_fallbacks,"all_invalid_fallback_rate": (all_invalid_fallbacks / max(sum(decisions), 1)),
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
            "experiment": "latent_option_smdp_dqn_frozen_dm",
            "model_state_dict": online_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "training_config": training_config,
            "choice_mapping": {
                "0": "primitive_latent_0",
                "1": "primitive_latent_1",
                "2": "primitive_latent_2",
                "3": "primitive_latent_3",
                "4": "repeat_latent_0_k2",
                "5": "repeat_latent_1_k2",
                "6": "repeat_latent_2_k2",
                "7": "repeat_latent_3_k2",
            },
            "episode": global_episode,
            "global_step": global_step,
            "high_level_decisions": high_level_decisions,
            "stage_index": stage_index,
            "stage_episode": stage_episode,
            "num_walls": training_config["curriculum"][stage_index][
                "num_walls"
            ],
            "eval_metrics": eval_metrics,
        },
        path,
    )


def make_env(
    tokenizer: VQVAEImageTokenizer,
    dynamics_model: LatentDynamicsModel,
    device: torch.device,
    config: dict[str, Any],
    num_walls: int,
    seed: int,
) -> LatentDMGridWorldEnv:
    return LatentDMGridWorldEnv(
        tokenizer=tokenizer,
        dynamics_model=dynamics_model,
        device=device,
        grid_size=config["grid_size"],
        image_size=config["image_size"],
        max_steps=config["max_steps"],
        num_walls=num_walls,
        seed=seed,
        goal_reward=config["goal_reward"],
        movement_reward=config["movement_reward"],
        blocked_reward=config["blocked_reward"],
        enforce_valid_motion=config["enforce_valid_motion"],
        validate_reset_detection=config["validate_reset_detection"],
    )


def main() -> None:
    reference_dqn, reference_history = find_reference_action_files()
    curriculum = load_step_curriculum(
        reference_history,
        budget_multiplier=LATENT_BUDGET_MULTIPLIER,
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
        "eval_interval_episodes": 100,
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
        "option_length": OPTION_LENGTH,
        "num_high_level_choices": NUM_HIGH_LEVEL_CHOICES,
        "budget_multiplier": LATENT_BUDGET_MULTIPLIER,
        "seed": EXPERIMENT_SEED,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tokenizer": str(
            RUN_ROOT / "outputs" / "tokenizer" / "model" / "best.pt"
        ),
        "dynamics_model": str(
            RUN_ROOT / "outputs" / "dm_hybrid" / "best.pt"
        ),
        "reference_action_dqn": str(reference_dqn),
        "reference_action_history": str(reference_history),
        "curriculum": curriculum,
        "max_global_steps": curriculum[-1]["end_step"],
        "output_dir": os.environ.get(
            "GENIE_OUTPUT_DIR",
            str(RUN_ROOT / "outputs" / "dqn_latent_options"),
        ),
    }

    seed_everything(training_config["seed"])
    rng = np.random.default_rng(training_config["seed"])
    device = torch.device(training_config["device"])
    tokenizer_path = Path(training_config["tokenizer"])
    dynamics_path = Path(training_config["dynamics_model"])
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(tokenizer_path, device)
    dynamics_model = load_dynamics_model(dynamics_path, device)
    initial_walls = curriculum[0]["num_walls"]
    train_env = make_env(
        tokenizer,
        dynamics_model,
        device,
        training_config,
        initial_walls,
        training_config["seed"],
    )
    eval_env = make_env(
        tokenizer,
        dynamics_model,
        device,
        training_config,
        initial_walls,
        training_config["eval_seed_start"],
    )
    initial_state, _ = train_env.reset(seed=training_config["seed"])
    eval_env.reset(seed=training_config["eval_seed_start"])

    model_config = load_reference_dqn_config(reference_dqn)
    online_network = build_dqn(model_config, device)
    target_network = build_dqn(model_config, device)
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
    history: list[dict[str, Any]] = []

    for stage_index, stage in enumerate(curriculum):
        num_walls = int(stage["num_walls"])
        stage_start = int(stage["start_step"])
        stage_end = int(stage["end_step"])

        if stage_index > 0:
            train_env = make_env(
                tokenizer,
                dynamics_model,
                device,
                training_config,
                num_walls,
                training_config["seed"] + global_episode,
            )
            eval_env = make_env(
                tokenizer,
                dynamics_model,
                device,
                training_config,
                num_walls,
                training_config["eval_seed_start"],
            )
            train_env.reset(seed=training_config["seed"] + global_episode)
            eval_env.reset(seed=training_config["eval_seed_start"])

        stage_episode = 0
        stage_step = 0
        best_stage_score = (-1.0, float("-inf"))
        recent_returns: deque[float] = deque(maxlen=100)
        recent_lengths: deque[int] = deque(maxlen=100)
        recent_successes: deque[int] = deque(maxlen=100)
        recent_blocked_rates: deque[float] = deque(maxlen=100)
        recent_invalid_rates: deque[float] = deque(maxlen=100)
        losses.clear()

        print(
            f"\n===== latent-option stage: {num_walls} walls =====\n"
            f"low-level step range: {stage_start} -> {stage_end} "
            f"(budget={stage['step_budget']})"
        )

        while global_step < stage_end:
            global_episode += 1
            stage_episode += 1
            state, _ = train_env.reset()
            episode_return = 0.0
            episode_steps = 0
            episode_blocked = 0
            episode_invalid = 0
            episode_success = False
            episode_finished = False
            previews = preview_all_latent_actions(train_env, state)

            while not episode_finished and global_step < stage_end:
                epsilon = epsilon_at_step(
                    stage_step,
                    training_config["epsilon_start"]
                    if stage_index == 0
                    else training_config["epsilon_restart"],
                    training_config["epsilon_end"],
                    training_config["epsilon_decay_steps"],
                )
                valid_choices = valid_choice_mask(previews)
                choice = select_choice(
                    online_network,
                    state,
                    epsilon,
                    rng,
                    device,
                    valid_choices,
                )

                old_global_step = global_step
                result = execute_choice(
                    train_env,
                    state,
                    choice,
                    training_config["gamma"],
                    stage_end - global_step,
                )
                global_step += result.duration
                stage_step += result.duration
                high_level_decisions += 1
                episode_steps += result.duration
                episode_return += result.raw_return
                episode_blocked += result.executed_blocked_steps
                episode_invalid += result.executed_invalid_dm_steps

                hit_stage_boundary = global_step >= stage_end
                replay_done = result.terminated or hit_stage_boundary

                next_previews = None
                if replay_done:
                    next_valid_choices = np.ones(NUM_HIGH_LEVEL_CHOICES, dtype=np.bool_)
                else:
                    next_previews = preview_all_latent_actions(train_env, result.next_state)
                    next_valid_choices = valid_choice_mask(next_previews)
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
                if next_previews is not None:
                    previews = next_previews
                    
                for update_step in range(old_global_step + 1, global_step + 1):
                    if (update_step >= training_config["learning_starts"] and len(replay_buffer) >= training_config["batch_size"]):
                        losses.append(
                            optimize_smdp_double_dqn(
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
            recent_lengths.append(episode_steps)
            recent_successes.append(int(episode_success))
            recent_blocked_rates.append(episode_blocked / max(episode_steps, 1))
            recent_invalid_rates.append(episode_invalid / max(episode_steps, 1))

            should_evaluate = (
                stage_episode % training_config["eval_interval_episodes"] == 0
                or global_step >= stage_end
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
            score = (
                eval_metrics["success_rate"],
                eval_metrics["mean_return"],
            )
            is_new_stage_best = score > best_stage_score
            if is_new_stage_best:
                best_stage_score = score
            transition_reason = (
                "reference_step_budget" if global_step >= stage_end else None
            )

            row = {
                "episode": global_episode,
                "global_step": global_step,
                "high_level_decisions": high_level_decisions,
                "stage_index": stage_index,
                "stage_episode": stage_episode,
                "stage_low_level_step": stage_step,
                "stage_step_budget": int(stage["step_budget"]),
                "num_walls": num_walls,
                "epsilon": epsilon,
                "mean_td_loss_1000": (float(np.mean(losses)) if losses else None),
                "train_success_rate_100": float(np.mean(recent_successes)),
                "train_mean_return_100": float(np.mean(recent_returns)),
                "train_mean_length_low_level_100": float(np.mean(recent_lengths)),
                "train_blocked_step_rate_100": float(np.mean(recent_blocked_rates)),
                "train_invalid_dm_motion_rate_100": float(np.mean(recent_invalid_rates)),
                "eval_success_rate": eval_metrics["success_rate"],
                "eval_mean_return": eval_metrics["mean_return"],
                "eval_mean_episode_length": eval_metrics["mean_episode_length_low_level"],
                "eval_mean_high_level_decisions": eval_metrics["mean_high_level_decisions"],
                "eval_choice_counts": eval_metrics["choice_counts"],
                "eval_option_choice_fraction": eval_metrics["option_choice_fraction"],
                "eval_mean_executed_option_duration": eval_metrics["mean_executed_option_duration"],
                "eval_observed_block_option_termination_rate": eval_metrics["observed_block_option_termination_rate"],
                "eval_mean_valid_option_count": eval_metrics["mean_valid_option_count"],
                "eval_blocked_step_rate": eval_metrics["blocked_step_rate"],
                "eval_invalid_dm_motion_rate": eval_metrics["invalid_dm_motion_rate"],
                "eval_mean_decoder_agent_margin": eval_metrics["mean_decoder_agent_margin"],
                "transition_reason": transition_reason,
                "eval_all_invalid_fallback_rate": eval_metrics["all_invalid_fallback_rate"],
            }
            history.append(row)

            print(
                f"episode={global_episode:05d} "
                f"stage_episode={stage_episode:04d} "
                f"walls={num_walls:02d} "
                f"step={global_step:06d}/{stage_end:06d} "
                f"decision={high_level_decisions:06d} "
                f"epsilon={epsilon:.3f} "
                f"train_success={row['train_success_rate_100']:.3f} "
                f"eval_success={row['eval_success_rate']:.3f} "
                f"eval_return={row['eval_mean_return']:.3f} "
                f"option_use={row['eval_option_choice_fraction']:.3f} "
                f"option_k={row['eval_mean_executed_option_duration']:.2f} "
                f"postblock={row['eval_observed_block_option_termination_rate']:.3f} "
                f"invalid_dm={row['eval_invalid_dm_motion_rate']:.3f}"
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

            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )

        print(
            f"stage finished exactly at low-level step {global_step}; "
            f"best_success={best_stage_score[0]:.3f}"
        )

    print("\nTraining finished")
    print("low-level DM steps:", global_step)
    print("high-level decisions:", high_level_decisions)
    print("history:", output_dir / "history.json")
    print("best final checkpoint:", output_dir / "best.pt")


if __name__ == "__main__":
    main()