from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

RUN_ROOT = Path(os.environ.get("GENIE_RUN_ROOT")).resolve()

from src.models.lam import LatentActionModel
from src.models.vqvae import VQVAEImageTokenizer

class TransitionDataset(Dataset):
    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)
        frames = data["frames"]

        frame_t = frames[:, :-1]
        frame_tp1 = frames[:, 1:]

        n, s, c, h, w = frame_t.shape

        frame_t = frame_t.reshape(n * s, c, h, w)
        frame_tp1 = frame_tp1.reshape(n * s, c, h, w)

        moved = np.any(frame_t != frame_tp1, axis=(1, 2, 3))

        frame_t = frame_t[moved]
        frame_tp1 = frame_tp1[moved]

        self.frame_t = frame_t
        self.frame_tp1 = frame_tp1

        print(f"{npz_path}: {len(self.frame_t)} transitions")

    def __len__(self) -> int:
        return len(self.frame_t)

    def __getitem__(self, index: int):
        frame_t = torch.from_numpy(self.frame_t[index]).float() / 255.0
        frame_tp1 = torch.from_numpy(self.frame_tp1[index]).float() / 255.0

        return frame_t, frame_tp1

def load_state_dict(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
) -> None:
    
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

@torch.no_grad()
def encode_tokens(
    tokenizer: VQVAEImageTokenizer,
    frames: torch.Tensor,
) -> torch.Tensor:
    
    z_e = tokenizer.encoder(frames)
    _, token_ids, _ = tokenizer.quantizer(z_e)
    return token_ids.long()

@torch.no_grad()
def build_moving_split(
    input_path: Path,
    output_path: Path,
    tokenizer: VQVAEImageTokenizer,
    lam: LatentActionModel,
    device: torch.device,
    batch_size: int = 128,
) -> None:
    
    dataset = TransitionDataset(str(input_path))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    tokenizer.eval()
    lam.eval()

    all_z_t = []
    all_z_tp1 = []
    all_latent_actions = []

    for frame_t, frame_tp1 in loader:
        frame_t = frame_t.to(device)
        frame_tp1 = frame_tp1.to(device)

        z_t = encode_tokens(tokenizer, frame_t)
        z_tp1 = encode_tokens(tokenizer, frame_tp1)

        outputs = lam(frame_t, frame_tp1)
        latent_actions = outputs["action_ids"]

        all_z_t.append(z_t.cpu())
        all_z_tp1.append(z_tp1.cpu())
        all_latent_actions.append(latent_actions.cpu())

    z_t = torch.cat(all_z_t, dim=0).numpy()
    z_tp1 = torch.cat(all_z_tp1, dim=0).numpy()
    latent_actions = torch.cat(all_latent_actions, dim=0).numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        z_t=z_t.astype(np.int64),
        latent_actions=latent_actions.astype(np.int64),
        z_tp1=z_tp1.astype(np.int64),
    )

@torch.no_grad()
def extract_goal_samples(
    input_path: Path,
    tokenizer: VQVAEImageTokenizer,
    lam: LatentActionModel,
    device: torch.device,
    min_context_transitions: int = 2,
    min_lam_agreement: float = 0.80,
) -> dict[str, np.ndarray]:
    
    data = np.load(input_path)
    frames = data["frames"]
    moved = data["moved"]
    goal_transition = data["goal_transition"]

    all_z_t = []
    all_z_tp1 = []
    all_latent_actions = []
    all_agreements = []

    skipped_short = 0
    skipped_no_goal = 0
    skipped_low_agreement = 0

    for seq_idx in range(frames.shape[0]):
        seq_frames = frames[seq_idx]
        seq_goal = goal_transition[seq_idx]
        goal_indices = np.where(seq_goal)[0]

        if len(goal_indices) != 1:
            skipped_no_goal += 1
            continue

        goal_index = int(goal_indices[0])
        context_indices = np.where(moved[seq_idx] & ~seq_goal)[0]
        context_indices = context_indices[context_indices < goal_index]

        if len(context_indices) < min_context_transitions:
            skipped_short += 1
            continue

        context_t = torch.from_numpy(seq_frames[context_indices]).float().div(255.0).to(device)
        context_tp1 = torch.from_numpy(seq_frames[context_indices + 1]).float().div(255.0).to(device)

        context_actions = lam(context_t, context_tp1)["action_ids"]
        latent_counts = torch.bincount(context_actions, minlength=4)
        sequence_latent_action = int(latent_counts.argmax().item())
        agreement = float(latent_counts.max().item() / len(context_indices))

        if agreement < min_lam_agreement:
            skipped_low_agreement += 1
            continue

        selected = np.asarray([goal_index], dtype=np.int64)
        frame_t = torch.from_numpy(seq_frames[selected]).float().div(255.0).to(device)
        frame_tp1 = torch.from_numpy(seq_frames[selected + 1]).float().div(255.0).to(device)

        all_z_t.append(encode_tokens(tokenizer, frame_t).cpu())
        all_z_tp1.append(encode_tokens(tokenizer, frame_tp1).cpu())
        all_latent_actions.append(torch.tensor([sequence_latent_action], dtype=torch.long))
        all_agreements.append(agreement)
    z_t = torch.cat(all_z_t, dim=0).numpy()
    z_tp1 = torch.cat(all_z_tp1, dim=0).numpy()
    latent_actions = torch.cat(all_latent_actions, dim=0).numpy()
    agreements = np.asarray(all_agreements, dtype=np.float32)

    print(f"goal transitions: {len(z_t)}")
    return {
        "z_t": z_t.astype(np.int64),
        "z_tp1": z_tp1.astype(np.int64),
        "latent_actions": latent_actions.astype(np.int64),
        "label_agreement": agreements,
    }


@torch.no_grad()
def build_dm_split(
    input_path: Path,
    goal_input_path: Path,
    moving_path: Path,
    output_path: Path,
    tokenizer: VQVAEImageTokenizer,
    lam: LatentActionModel,
    device: torch.device,
    min_moving_transitions: int = 2,
    min_lam_agreement: float = 0.80,
) -> None:
    
    data = np.load(input_path)

    frames = data["frames"]
    moved = data["moved"]
    blocked = data["blocked"]

    tokenizer.eval()
    lam.eval()

    all_z_t = []
    all_z_tp1 = []
    all_latent_actions = []
    all_agreements = []

    num_sequences = frames.shape[0]

    skipped_short = 0
    skipped_no_block = 0
    skipped_low_agreement = 0

    for seq_idx in range(num_sequences):
        seq_frames = frames[seq_idx]
        seq_moved = moved[seq_idx]
        seq_blocked = blocked[seq_idx]

        moving_indices = np.where(seq_moved)[0]

        if len(moving_indices) < min_moving_transitions:
            skipped_short += 1
            continue

        blocked_indices = np.where(seq_blocked)[0]

        if len(blocked_indices) == 0:
            skipped_no_block += 1
            continue

        moving_frame_t = torch.from_numpy(seq_frames[moving_indices]).float() / 255.0
        moving_frame_tp1 = torch.from_numpy(seq_frames[moving_indices + 1]).float() / 255.0
        moving_frame_t = moving_frame_t.to(device)
        moving_frame_tp1 = moving_frame_tp1.to(device)

        outputs = lam(moving_frame_t, moving_frame_tp1)
        moving_latent_actions = outputs["action_ids"]

        latent_counts = torch.bincount(moving_latent_actions, minlength=4)
        sequence_latent_action = int(torch.argmax(latent_counts).item())

        majority_count = int(latent_counts.max().item())
        agreement = majority_count / len(moving_indices)

        if agreement < min_lam_agreement:
            skipped_low_agreement += 1
            continue

        selected_indices = np.asarray(
            [int(blocked_indices[0])],
            dtype=np.int64,
        )

        frame_t = torch.from_numpy(seq_frames[selected_indices]).float() / 255.0
        frame_tp1 = torch.from_numpy(seq_frames[selected_indices + 1]).float() / 255.0

        frame_t = frame_t.to(device)
        frame_tp1 = frame_tp1.to(device)

        z_t = encode_tokens(tokenizer, frame_t)
        z_tp1 = encode_tokens(tokenizer, frame_tp1)

        latent_actions = torch.full(
            (1,),
            sequence_latent_action,
            dtype=torch.long,
            device=device,
        )

        all_z_t.append(z_t.cpu())
        all_z_tp1.append(z_tp1.cpu())
        all_latent_actions.append(latent_actions.cpu())
        all_agreements.append(agreement)

    blocked_z_t = torch.cat(all_z_t, dim=0).numpy()
    blocked_z_tp1 = torch.cat(all_z_tp1, dim=0).numpy()
    blocked_latent_actions = torch.cat(all_latent_actions, dim=0).numpy()
    blocked_agreements = np.asarray(all_agreements, dtype=np.float32)

    goal_data = extract_goal_samples(
        input_path=goal_input_path,
        tokenizer=tokenizer,
        lam=lam,
        device=device,
        min_context_transitions=min_moving_transitions,
        min_lam_agreement=min_lam_agreement,
    )
    goal_z_t = goal_data["z_t"]
    goal_z_tp1 = goal_data["z_tp1"]
    goal_latent_actions = goal_data["latent_actions"]
    goal_agreements = goal_data["label_agreement"]

    moving_data = np.load(moving_path)
    moving_z_t = moving_data["z_t"].astype(np.int64)
    moving_z_tp1 = moving_data["z_tp1"].astype(np.int64)
    moving_latent_actions = moving_data["latent_actions"].astype(np.int64)

    num_moving = len(moving_z_t)
    num_blocked = len(blocked_z_t)
    num_goal = len(goal_z_t)

    z_t = np.concatenate([moving_z_t, blocked_z_t, goal_z_t], axis=0)
    z_tp1 = np.concatenate([moving_z_tp1, blocked_z_tp1, goal_z_tp1], axis=0)
    latent_actions = np.concatenate([moving_latent_actions, blocked_latent_actions, goal_latent_actions], axis=0)
    blocked = np.concatenate(
        [
            np.zeros(num_moving, dtype=np.bool_),
            np.ones(num_blocked, dtype=np.bool_),
            np.zeros(num_goal, dtype=np.bool_),
        ],
        axis=0,
    )
    goal = np.concatenate(
        [
            np.zeros(num_moving, dtype=np.bool_),
            np.zeros(num_blocked, dtype=np.bool_),
            np.ones(num_goal, dtype=np.bool_),
        ],
        axis=0,
    )
    label_agreement = np.concatenate(
        [
            np.ones(num_moving, dtype=np.float32),
            blocked_agreements,
            goal_agreements,
        ],
        axis=0,
    )
    print(f"moving transitions: {num_moving}")
    print(f"blocked transitions: {num_blocked}")
    print(f"goal transitions: {num_goal}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        z_t=z_t.astype(np.int64),
        latent_actions=latent_actions.astype(np.int64),
        z_tp1=z_tp1.astype(np.int64),
        blocked=blocked.astype(np.bool_),
        goal=goal.astype(np.bool_),
        label_agreement=label_agreement,
    )

    print(f"Saved: {output_path}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    train_input = RUN_ROOT / "data" / "gridworld" / "train.npz"
    eval_input = RUN_ROOT / "data" / "gridworld" / "eval.npz"

    train_dm_input = RUN_ROOT / "data" / "gridworld" / "train_dm.npz"
    eval_dm_input = RUN_ROOT / "data" / "gridworld" / "eval_dm.npz"

    train_goal_input = RUN_ROOT / "data" / "gridworld" / "train_goal.npz"
    eval_goal_input = RUN_ROOT / "data" / "gridworld" / "eval_goal.npz"

    train_output = RUN_ROOT / "data" / "dm" / "dynamics_train.npz"
    eval_output = RUN_ROOT / "data" / "dm" / "dynamics_eval.npz"

    train_dm_output = RUN_ROOT / "data" / "dm" / "dynamics_train_dm.npz"
    eval_dm_output = RUN_ROOT / "data" / "dm" / "dynamics_eval_dm.npz"

    tokenizer_path = RUN_ROOT / "outputs" / "tokenizer" / "model" / "best.pt"
    lam_path = RUN_ROOT / "outputs" / "lam" / "model" / "best.pt"

    tokenizer = VQVAEImageTokenizer(num_codes=128, embedding_dim=64).to(device)
    load_state_dict(tokenizer, tokenizer_path,device,)
    tokenizer.eval()
    lam = LatentActionModel(
        num_codes=4,
        embedding_dim=64,
        commitment_beta=0.25,
    ).to(device)

    load_state_dict(
        lam,
        lam_path,
        device,
    )

    lam.eval()
    print("\nBuilding moving-only training data...")
    build_moving_split(
        input_path=train_input,
        output_path=train_output,
        tokenizer=tokenizer,
        lam=lam,
        device=device,
        batch_size=128,
    )

    build_moving_split(
        input_path=eval_input,
        output_path=eval_output,
        tokenizer=tokenizer,
        lam=lam,
        device=device,
        batch_size=128,
    )

    build_dm_split(
        input_path=train_dm_input,
        goal_input_path=train_goal_input,
        moving_path=train_output,
        output_path=train_dm_output,
        tokenizer=tokenizer,
        lam=lam,
        device=device,
    )

    build_dm_split(
        input_path=eval_dm_input,
        goal_input_path=eval_goal_input,
        moving_path=eval_output,
        output_path=eval_dm_output,
        tokenizer=tokenizer,
        lam=lam,
        device=device,
    )


if __name__ == "__main__":
    main()
