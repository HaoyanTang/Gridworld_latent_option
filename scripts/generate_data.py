import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

RUN_ROOT = Path(os.environ.get("GENIE_RUN_ROOT")).resolve()
EXPERIMENT_SEED = int(os.environ.get("GENIE_SEED"))

from src.env.gridworld import GridWorldEnv, NUM_ACTIONS

def generate_split(
    num_sequences: int,
    sequence_length: int,
    grid_size: int,
    image_size: int,
    max_steps: int,
    num_walls: int,
    seed: int,
) -> dict[str, np.ndarray]:

    rng = np.random.default_rng(seed)
    env = GridWorldEnv(
        grid_size=grid_size,
        image_size=image_size,
        max_steps=max_steps,
        num_walls=num_walls,
        seed=seed,
    )
    frames = np.zeros(
        (
            num_sequences,
            sequence_length,
            3,
            image_size,
            image_size,
        ),
        dtype=np.uint8,
    )
    actions = np.zeros((num_sequences, sequence_length - 1), dtype=np.int64)
    rewards = np.zeros((num_sequences, sequence_length - 1), dtype=np.float32)
    dones = np.zeros((num_sequences, sequence_length - 1), dtype=np.bool_)
    seq_idx = 0

    while seq_idx < num_sequences:
        reset_seed = int(rng.integers(0, 2**31 - 1,))
        obs, _ = env.reset(seed=reset_seed)

        seq_frames = [obs]
        seq_actions = []
        seq_rewards = []
        seq_dones = []

        valid_sequence = True

        for _ in range(sequence_length - 1):
            action = int(rng.integers(0, NUM_ACTIONS))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            seq_actions.append(action)
            seq_rewards.append(float(reward))
            seq_dones.append(done)
            seq_frames.append(next_obs)

            if (done and len(seq_frames) < sequence_length):
                valid_sequence = False
                break

        if not valid_sequence:
            continue

        frames[seq_idx] = np.stack(seq_frames, axis=0)
        actions[seq_idx] = np.asarray(seq_actions, dtype=np.int64)
        rewards[seq_idx] = np.asarray(seq_rewards, dtype=np.float32)
        dones[seq_idx] = np.asarray(seq_dones, dtype=np.bool_)

        seq_idx += 1

        if (seq_idx % 100 == 0):
            print(
                f"Generated "
                f"{seq_idx}/"
                f"{num_sequences} "
                f"sequences"
            )

    return {
        "frames": frames,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


def generate_dm_split(
    num_sequences: int,
    sequence_length: int,
    grid_size: int,
    image_size: int,
    max_steps: int,
    num_walls: int,
    seed: int,
) -> dict[str, np.ndarray]:

    rng = np.random.default_rng(
        seed
    )

    env = GridWorldEnv(
        grid_size=grid_size,
        image_size=image_size,
        max_steps=max_steps,
        num_walls=num_walls,
        seed=seed,
    )

    frames = np.zeros(
        (
            num_sequences,
            sequence_length,
            3,
            image_size,
            image_size,
        ),
        dtype=np.uint8,
    )

    actions = np.zeros((num_sequences, sequence_length - 1), dtype=np.int64)
    rewards = np.zeros((num_sequences, sequence_length - 1), dtype=np.float32)
    dones = np.zeros((num_sequences, sequence_length - 1), dtype=np.bool_)
    moved = np.zeros((num_sequences, sequence_length - 1,),dtype=np.bool_)
    blocked = np.zeros((num_sequences, sequence_length - 1),dtype=np.bool_)
    seq_idx = 0

    while seq_idx < num_sequences:
        reset_seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=reset_seed)
        action = int(seq_idx % NUM_ACTIONS)

        seq_frames = [obs]
        seq_actions = []
        seq_rewards = []
        seq_dones = []
        seq_moved = []
        seq_blocked = []
        valid_sequence = True

        for _ in range(sequence_length - 1):
            current_obs = obs
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            did_move = bool(np.any(current_obs!= next_obs))
            did_block = (not did_move)
            seq_actions.append(action)
            seq_rewards.append(float(reward))
            seq_dones.append(done)
            seq_moved.append(did_move)
            seq_blocked.append(did_block)
            seq_frames.append(next_obs)
            obs = next_obs

            if (done and len(seq_frames) < sequence_length):
                valid_sequence = False
                break
        if not valid_sequence:
            continue

        seq_moved = np.asarray(seq_moved, dtype=np.bool_)
        seq_blocked = np.asarray(seq_blocked, dtype=np.bool_)

        if int(seq_moved.sum()) < 2:
            continue

        if not np.any(seq_blocked):
            continue

        frames[seq_idx] = np.stack(seq_frames, axis=0)
        actions[seq_idx] = np.asarray(seq_actions, dtype=np.int64)
        rewards[seq_idx] = np.asarray(seq_rewards, dtype=np.float32)
        dones[seq_idx] = np.asarray(seq_dones, dtype=np.bool_)
        moved[seq_idx] = seq_moved
        blocked[seq_idx] = seq_blocked

        seq_idx += 1

        if (seq_idx % 100== 0):
            print(
                f"Generated DM "
                f"{seq_idx}/"
                f"{num_sequences} "
                f"sequences"
            )

    return {
        "frames": frames,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "moved": moved,
        "blocked": blocked,
    }


def generate_goal_split(
    num_sequences: int,
    grid_size: int,
    image_size: int,
    max_steps: int,
    num_walls: int,
    seed: int,
    goal_distance: int = 3,
) -> dict[str, np.ndarray]:

    rng = np.random.default_rng(seed)
    env = GridWorldEnv(
        grid_size=grid_size,
        image_size=image_size,
        max_steps=max_steps,
        num_walls=num_walls,
        seed=seed,
    )

    sequence_length = goal_distance + 1
    num_transitions = goal_distance

    frames = np.zeros((num_sequences, sequence_length, 3, image_size, image_size), dtype=np.uint8)
    actions = np.zeros((num_sequences, num_transitions), dtype=np.int64)
    rewards = np.zeros((num_sequences, num_transitions), dtype=np.float32)
    dones = np.zeros((num_sequences, num_transitions), dtype=np.bool_)
    moved = np.ones((num_sequences, num_transitions), dtype=np.bool_)
    blocked = np.zeros((num_sequences, num_transitions), dtype=np.bool_)
    goal_transition = np.zeros((num_sequences, num_transitions), dtype=np.bool_)

    for seq_idx in range(num_sequences):
        action = int(seq_idx % NUM_ACTIONS)
        if action == 0:  # up
            start = (
                int(rng.integers(0, grid_size)),
                int(rng.integers(goal_distance, grid_size)),
            )
            goal = (start[0], start[1] - goal_distance)
            path = [(start[0], start[1] - step) for step in range(goal_distance + 1)]
        elif action == 1:  # down
            start = (
                int(rng.integers(0, grid_size)),
                int(rng.integers(0, grid_size - goal_distance)),
            )
            goal = (start[0], start[1] + goal_distance)
            path = [(start[0], start[1] + step) for step in range(goal_distance + 1)]
        elif action == 2:  # left
            start = (
                int(rng.integers(goal_distance, grid_size)),
                int(rng.integers(0, grid_size)),
            )
            goal = (start[0] - goal_distance, start[1])
            path = [(start[0] - step, start[1]) for step in range(goal_distance + 1)]
        else:  # right
            start = (
                int(rng.integers(0, grid_size - goal_distance)),
                int(rng.integers(0, grid_size)),
            )
            goal = (start[0] + goal_distance, start[1])
            path = [(start[0] + step, start[1]) for step in range(goal_distance + 1)]

        path_cells = set(path)
        wall_candidates = [
            (x, y)
            for x in range(grid_size)
            for y in range(grid_size)
            if (x, y) not in path_cells
        ]
        wall_indices = rng.choice(
            len(wall_candidates),
            size=num_walls,
            replace=False,
        )

        env.walls = {wall_candidates[int(index)] for index in wall_indices}
        env.agent_pos = start
        env.goal_pos = goal
        env.step_count = 0

        sequence_frames = [env.render()]
        sequence_rewards: list[float] = []
        sequence_dones: list[bool] = []

        for _ in range(num_transitions):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            sequence_frames.append(next_obs)
            sequence_rewards.append(float(reward))
            sequence_dones.append(done)

        frames[seq_idx] = np.stack(sequence_frames, axis=0)
        actions[seq_idx].fill(action)
        rewards[seq_idx] = np.asarray(sequence_rewards, dtype=np.float32)
        dones[seq_idx] = np.asarray(sequence_dones, dtype=np.bool_)
        goal_transition[seq_idx, -1] = True

        if (seq_idx + 1) % 100 == 0:
            print(f"Generated goal {seq_idx + 1}/{num_sequences} sequences")

    return {
        "frames": frames,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "moved": moved,
        "blocked": blocked,
        "goal_transition": goal_transition,
    }

def save_dataset(
    path: Path,
    data: dict[str, np.ndarray],
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=data["frames"],
        actions=data["actions"],
        rewards=data["rewards"],
        dones=data["dones"],
    )

    print(f"Saved dataset to: {path}")
    print(f"frames: {data['frames'].shape}, {data['frames'].dtype}")
    print(f"actions: {data['actions'].shape}, {data['actions'].dtype}")
    print(f"rewards: {data['rewards'].shape}, {data['rewards'].dtype}")
    print(f"dones: {data['dones'].shape}, {data['dones'].dtype}")

def save_dm_dataset(
    path: Path,
    data: dict[str, np.ndarray],
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=data["frames"],
        actions=data["actions"],
        rewards=data["rewards"],
        dones=data["dones"],
        moved=data["moved"],
        blocked=data["blocked"]
    )

    print(f"Saved dataset to: {path}")
    print(f"frames: {data['frames'].shape}, {data['frames'].dtype}")
    print(f"actions: {data['actions'].shape}, {data['actions'].dtype}")
    print(f"rewards: {data['rewards'].shape}, {data['rewards'].dtype}")
    print(f"dones: {data['dones'].shape}, {data['dones'].dtype}")
    print(f"moved: {data['moved'].shape}, {data['moved'].dtype}")
    print(f"blocked: {data['blocked'].shape}, {data['blocked'].dtype}")
    print("Moving transitions:", int(data["moved"].sum()))
    print("Blocked transitions:", int(data["blocked"].sum()))


def save_goal_dataset(
    path: Path,
    data: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        frames=data["frames"],
        actions=data["actions"],
        rewards=data["rewards"],
        dones=data["dones"],
        moved=data["moved"],
        blocked=data["blocked"],
        goal_transition=data["goal_transition"],
    )

    print(f"Saved dataset to: {path}")
    print(f"frames: {data['frames'].shape}, {data['frames'].dtype}")
    print(f"actions: {data['actions'].shape}, {data['actions'].dtype}")
    print("Goal transitions:", int(data["goal_transition"].sum()))

def main() -> None:
    output_dir = RUN_ROOT / "data" / "gridworld"
    train_size = 5000
    eval_size = 1000
    sequence_length = 16
    train_dm_size = 5000
    eval_dm_size = 1000
    dm_sequence_length = 8
    train_goal_size = 2000
    eval_goal_size = 500
    goal_distance = 3
    grid_size = 8
    image_size = 64
    max_steps = 64
    num_walls = 10
    seed = EXPERIMENT_SEED

    train_data = generate_split(
        num_sequences=train_size,
        sequence_length=sequence_length,
        grid_size=grid_size,
        image_size=image_size,
        max_steps=max_steps,
        num_walls=num_walls,
        seed=seed,
    )
    save_dataset(output_dir / "train.npz", train_data)

    eval_data = generate_split(
        num_sequences=eval_size,
        sequence_length=sequence_length,
        grid_size=grid_size,
        image_size=image_size,
        max_steps=max_steps,
        num_walls=num_walls,
        seed=seed + 10000,
    )
    save_dataset(output_dir / "eval.npz", eval_data)

    train_dm_data = generate_dm_split(
        num_sequences=train_dm_size,
        sequence_length=dm_sequence_length,
        grid_size=grid_size,
        image_size=image_size,
        max_steps=max_steps,
        num_walls=num_walls,
        seed=seed + 20000,
    )
    save_dm_dataset(output_dir / "train_dm.npz", train_dm_data)

    eval_dm_data = generate_dm_split(
        num_sequences=eval_dm_size,
        sequence_length=dm_sequence_length,
        grid_size=grid_size,
        image_size=image_size,
        max_steps=max_steps,
        num_walls=num_walls,
        seed=seed + 30000,
    )
    save_dm_dataset(output_dir / "eval_dm.npz", eval_dm_data)

    train_goal_data = generate_goal_split(
        num_sequences=train_goal_size,
        grid_size=grid_size,
        image_size=image_size,
        max_steps=max_steps,
        num_walls=num_walls,
        seed=seed + 40000,
        goal_distance=goal_distance,
    )
    save_goal_dataset(output_dir / "train_goal.npz", train_goal_data)

    eval_goal_data = generate_goal_split(
        num_sequences=eval_goal_size,
        grid_size=grid_size,
        image_size=image_size,
        max_steps=max_steps,
        num_walls=num_walls,
        seed=seed + 50000,
        goal_distance=goal_distance,
    )
    save_goal_dataset(output_dir / "eval_goal.npz", eval_goal_data)

if __name__ == "__main__":
    main()