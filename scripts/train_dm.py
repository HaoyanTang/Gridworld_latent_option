from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

RUN_ROOT = Path(os.environ.get("GENIE_RUN_ROOT")).resolve()
EXPERIMENT_SEED = int(os.environ.get("GENIE_SEED"))

from src.models.dm import LatentDynamicsModel


class DynamicsDataset(Dataset):
    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)

        self.z_t = data["z_t"].astype(np.int64)
        self.latent_actions = data["latent_actions"].astype(np.int64)
        self.z_tp1 = data["z_tp1"].astype(np.int64)

        if "blocked" in data:
            self.blocked = data["blocked"].astype(np.bool_)
            self.goal = data["goal"].astype(np.bool_)
        else:
            self.blocked = None
            self.goal = None

        if self.blocked is not None:
            moving = ~(self.blocked | self.goal)
            print(f"moving: {moving.sum()}")
            print(f"blocked: {self.blocked.sum()}")
            print(f"goal: {self.goal.sum()}")

    def __len__(self) -> int:
        return len(self.z_t)

    def __getitem__(self, index: int):
        z_t = torch.from_numpy(self.z_t[index]).long()
        latent_action = torch.tensor(self.latent_actions[index], dtype=torch.long)
        z_tp1 = torch.from_numpy(self.z_tp1[index]).long()

        if self.blocked is None:
            return z_t, latent_action, z_tp1

        blocked = bool(self.blocked[index])
        goal = bool(self.goal[index])

        return z_t, latent_action, z_tp1, blocked, goal


@torch.no_grad()
def evaluate(
    model: LatentDynamicsModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    
    model.eval()

    total_loss = 0.0
    total_samples = 0

    total_tokens = 0
    correct_tokens = 0

    changed_total = 0
    changed_correct = 0

    unchanged_total = 0
    unchanged_correct = 0

    exact_total = 0
    exact_correct = 0

    moving_total = 0
    moving_exact_correct = 0

    blocked_total = 0
    blocked_exact_correct = 0

    goal_total = 0
    goal_exact_correct = 0

    for batch in loader:
        if len(batch) == 5:
            z_t, latent_action, z_tp1, blocked, goal = batch
            blocked = blocked.to(device)
            goal = goal.to(device)
        else:
            z_t, latent_action, z_tp1 = batch
            blocked = None
            goal = None
        z_t = z_t.to(device)
        latent_action = latent_action.to(device)
        z_tp1 = z_tp1.to(device)

        logits = model(z_t, latent_action)
        loss = F.cross_entropy(logits, z_tp1)
        prediction = torch.argmax(logits, dim=1)
        correct = prediction == z_tp1

        batch_size = z_t.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        total_tokens += z_tp1.numel()
        correct_tokens += correct.sum().item()

        changed_mask = z_t != z_tp1
        unchanged_mask = ~changed_mask

        changed_total += changed_mask.sum().item()
        changed_correct += (correct & changed_mask).sum().item()

        unchanged_total += unchanged_mask.sum().item()
        unchanged_correct += (correct & unchanged_mask).sum().item()

        exact = torch.all(correct, dim=(1, 2))

        exact_total += batch_size
        exact_correct += exact.sum().item()

        if blocked is not None:
            moving = ~(blocked | goal)

            moving_total += moving.sum().item()
            moving_exact_correct += (exact & moving).sum().item()

            blocked_total += blocked.sum().item()
            blocked_exact_correct += (exact & blocked).sum().item()

            goal_total += goal.sum().item()
            goal_exact_correct += (exact & goal).sum().item()

    metrics = {
        "loss": total_loss / total_samples,
        "token_accuracy": correct_tokens / total_tokens,
        "changed_accuracy": changed_correct / changed_total,
        "unchanged_accuracy": unchanged_correct / unchanged_total,
        "exact_accuracy": exact_correct / exact_total,
    }

    if moving_total > 0:
        metrics["moving_exact_accuracy"] = moving_exact_correct / moving_total
    if blocked_total > 0:
        metrics["blocked_exact_accuracy"] = blocked_exact_correct / blocked_total
    if goal_total > 0:
        metrics["goal_exact_accuracy"] = goal_exact_correct / goal_total
    return metrics


def main() -> None:
    phase = os.environ.get("GENIE_DM_PHASE").strip().lower()

    batch_size = 128
    epochs = 30
    seed = EXPERIMENT_SEED
    moving_fraction = 0.60
    blocked_fraction = 0.25
    goal_fraction = 0.15

    if phase == "moving":
        learning_rate = 1e-3

        train_path = RUN_ROOT / "data" / "dm" / "dynamics_train.npz"
        eval_path = RUN_ROOT / "data" / "dm" / "dynamics_eval.npz"

        output_dir = RUN_ROOT / "outputs" / "dm"

        pretrained_path = None

    elif phase == "hybrid":
        learning_rate = 1e-4

        train_path = RUN_ROOT / "data" / "dm" / "dynamics_train_dm.npz"
        eval_path = RUN_ROOT / "data" / "dm" / "dynamics_eval_dm.npz"

        output_dir = RUN_ROOT / "outputs" / "dm_hybrid"

        pretrained_path = RUN_ROOT / "outputs" / "dm" / "best.pt"

    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = DynamicsDataset(str(train_path))
    eval_dataset = DynamicsDataset(str(eval_path))

    if phase == "hybrid":
        moving_mask = ~(train_dataset.blocked | train_dataset.goal)
        moving_count = int(moving_mask.sum())
        blocked_count = int(train_dataset.blocked.sum())
        goal_count = int(train_dataset.goal.sum())

        sample_weights = np.empty(len(train_dataset), dtype=np.float64)
        sample_weights[moving_mask] = moving_fraction / moving_count
        sample_weights[train_dataset.blocked] = (blocked_fraction / blocked_count)
        sample_weights[train_dataset.goal] = goal_fraction / goal_count

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
        )

        print("hybrid sampling moving:blocked:goal = 60%:25%:15%")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = LatentDynamicsModel().to(device)

    if phase == "hybrid":
        state_dict = torch.load(
            pretrained_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_eval_loss = float("inf")
    best_hybrid_score = 0.0

    history = []

    if phase == "hybrid":
        initial_metrics = evaluate(
            model,
            eval_loader,
            device,
        )

        initial_score = (
            0.4 * initial_metrics["moving_exact_accuracy"]
            + 0.3 * initial_metrics["blocked_exact_accuracy"]
            + 0.3 * initial_metrics["goal_exact_accuracy"]
        )

        best_path = output_dir / "best.pt"
        best_hybrid_score = initial_score
        torch.save(model.state_dict(), best_path)

        print(
            f"Before fine-tuning "
            f"moving={initial_metrics['moving_exact_accuracy']:.4f} "
            f"blocked={initial_metrics['blocked_exact_accuracy']:.4f} "
            f"goal={initial_metrics['goal_exact_accuracy']:.4f} "
            f"score={initial_score:.4f}"
        )

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        total_tokens = 0
        correct_tokens = 0

        changed_total = 0
        changed_correct = 0

        sampled_moving = 0
        sampled_blocked = 0
        sampled_goal = 0

        for batch in train_loader:
            if len(batch) == 5:
                (
                    z_t,
                    latent_action,
                    z_tp1,
                    batch_blocked,
                    batch_goal,
                ) = batch
                sampled_blocked += int(batch_blocked.sum().item())
                sampled_goal += int(batch_goal.sum().item())
                sampled_moving += int(
                    (~(batch_blocked | batch_goal)).sum().item()
                )
            else:
                z_t, latent_action, z_tp1 = batch

            z_t = z_t.to(device)
            latent_action = latent_action.to(device)
            z_tp1 = z_tp1.to(device)

            optimizer.zero_grad()
            logits = model(z_t, latent_action)
            loss = F.cross_entropy(logits, z_tp1)
            loss.backward()
            optimizer.step()
            prediction = torch.argmax(logits, dim=1)
            correct = (prediction == z_tp1)

            current_batch_size = z_t.shape[0]

            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

            total_tokens += z_tp1.numel()
            correct_tokens += correct.sum().item()

            changed_mask = z_t != z_tp1

            changed_total += changed_mask.sum().item()
            changed_correct += (correct & changed_mask).sum().item()

        train_loss = total_loss / total_samples
        train_token_accuracy = correct_tokens / total_tokens
        train_changed_accuracy = changed_correct / changed_total if changed_total > 0 else 0.0

        eval_metrics = evaluate(
            model,
            eval_loader,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"eval_loss={eval_metrics['loss']:.6f}"
        )

        if phase == "moving":
            print(
                f"changed={eval_metrics['changed_accuracy']:.4f} "
                f"exact={eval_metrics['exact_accuracy']:.4f}"
            )
        else:
            hybrid_score = (
                0.4 * eval_metrics["moving_exact_accuracy"]
                + 0.3 * eval_metrics["blocked_exact_accuracy"]
                + 0.3 * eval_metrics["goal_exact_accuracy"]
            )

            print(
                f"moving={eval_metrics['moving_exact_accuracy']:.4f} "
                f"blocked={eval_metrics['blocked_exact_accuracy']:.4f} "
                f"goal={eval_metrics['goal_exact_accuracy']:.4f} "
                f"score={hybrid_score:.4f} "
                f"sampled={sampled_moving}:{sampled_blocked}:{sampled_goal}"
            )

        epoch_history = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_token_accuracy": train_token_accuracy,
            "train_changed_accuracy": train_changed_accuracy,
            "eval_loss": eval_metrics["loss"],
            "eval_token_accuracy": eval_metrics["token_accuracy"],
            "eval_changed_accuracy": eval_metrics["changed_accuracy"],
            "eval_unchanged_accuracy": eval_metrics["unchanged_accuracy"],
            "eval_exact_accuracy": eval_metrics["exact_accuracy"],
        }

        if phase == "hybrid":
            epoch_history["moving_exact_accuracy"] = eval_metrics["moving_exact_accuracy"]
            epoch_history["blocked_exact_accuracy"] = eval_metrics["blocked_exact_accuracy"]
            epoch_history["goal_exact_accuracy"] = eval_metrics["goal_exact_accuracy"]
            epoch_history["hybrid_score"] = hybrid_score
            epoch_history["sampled_moving"] = sampled_moving
            epoch_history["sampled_blocked"] = sampled_blocked
            epoch_history["sampled_goal"] = sampled_goal

        history.append(epoch_history)

        if phase == "moving":
            if eval_metrics["loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["loss"]
                torch.save(model.state_dict(), output_dir / "best.pt")
        else:
            if hybrid_score > best_hybrid_score:
                best_hybrid_score = hybrid_score
                torch.save(model.state_dict(), output_dir / "best.pt")

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=4)

    if phase == "moving":
        print(f"\nBest evaluation loss: {best_eval_loss:.6f}")

    else:
        print(f"\nBest hybrid score: {best_hybrid_score:.6f}")


if __name__ == "__main__":
    main()
