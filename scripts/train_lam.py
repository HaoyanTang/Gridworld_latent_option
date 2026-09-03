from pathlib import Path
import os
import sys
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

RUN_ROOT = Path(os.environ.get("GENIE_RUN_ROOT")).resolve()
EXPERIMENT_SEED = int(os.environ.get("GENIE_SEED"))

from src.models.lam import LatentActionModel
from sklearn.cluster import KMeans

class GridWorldTransitionDataset(Dataset):
    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)

        frames = data["frames"]

        frame_t = frames[:, :-1, :, :, :]
        frame_tp1 = frames[:, 1:, :, :, :]

        n, s, c, h, w = frame_t.shape

        frame_t = frame_t.reshape(n * s, c, h, w)
        frame_tp1 = frame_tp1.reshape(n * s, c, h, w)

        total_transitions = len(frame_t)
        moved_mask = np.any(
            frame_t != frame_tp1,
            axis=(1, 2, 3),
        )
        self.frame_t = frame_t[moved_mask]
        self.frame_tp1 = frame_tp1[moved_mask]

        print(
            f"{npz_path}: "
            f"total={total_transitions}, "
            f"moving={len(self.frame_t)}, "
            f"blocked={total_transitions - len(self.frame_t)}"
        )

    def __len__(self) -> int:
        return len(self.frame_t)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        frame_t = self.frame_t[idx]
        frame_tp1 = self.frame_tp1[idx]

        frame_t = torch.from_numpy(frame_t).float() / 255.0
        frame_tp1 = torch.from_numpy(frame_tp1).float() / 255.0

        return frame_t, frame_tp1


def delta_reconstruction_loss(
    pred_delta: torch.Tensor,
    frame_t: torch.Tensor,
    frame_tp1: torch.Tensor,
    change_weight: float = 10.0,
) -> torch.Tensor:

    target_delta = (frame_tp1 - frame_t)
    pixel_loss = (pred_delta - target_delta) ** 2
    changed_mask = (target_delta.abs().sum(dim=1, keepdim=True) > 1e-6)

    weights = torch.ones_like(pixel_loss)
    weights = torch.where(changed_mask, torch.full_like(weights, change_weight), weights)

    loss = (pixel_loss * weights).mean()
    return loss

def evaluate(
    model: LatentActionModel,
    dataloader: DataLoader,
    device: torch.device,
    num_codes: int,
    lambda_usage: float,
) -> tuple[float, float, float, list[int]]:

    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_samples = 0
    action_counts = [0] * num_codes

    with torch.no_grad():
        for frame_t, frame_tp1 in dataloader:
            frame_t = frame_t.to(device)
            frame_tp1 = frame_tp1.to(device)

            outputs = model(frame_t, frame_tp1)

            pred_delta = outputs["pred_delta"]
            action_ids = outputs["action_ids"]
            vq_loss = outputs["vq_loss"]
            recon_loss = delta_reconstruction_loss(pred_delta, frame_t, frame_tp1, change_weight=10.0)
            usage_loss = code_usage_loss(outputs["distances"])

            loss = (
                recon_loss
                + vq_loss
                + lambda_usage * usage_loss
            )

            batch_size = frame_t.size(0)
            total_loss += loss.item() * batch_size
            total_recon_loss += recon_loss.item() * batch_size
            total_vq_loss += vq_loss.item() * batch_size
            total_samples += batch_size

            for action_id in action_ids.cpu().tolist():
                action_counts[action_id] += 1

    return (
        total_loss / total_samples,
        total_recon_loss / total_samples,
        total_vq_loss / total_samples,
        action_counts,
    )

@torch.no_grad()
def initialize_codebook_from_h(
    model,
    train_loader,
    device,
    max_samples: int = 4096,
) -> None:

    model.eval()
    h_list = []
    num_collected = 0

    for frame_t, frame_tp1 in train_loader:
        frame_t = frame_t.to(device)
        frame_tp1 = frame_tp1.to(device)
        h = model.encoder(frame_t, frame_tp1,)
        h_list.append(h.cpu())
        num_collected += h.shape[0]
        if num_collected >= max_samples:
            break

    h_data = torch.cat(h_list, dim=0,)
    h_data = h_data[:max_samples]
    kmeans = KMeans(
        n_clusters=model.quantizer.num_codes,
        n_init=10,
        random_state=EXPERIMENT_SEED,
    )
    kmeans.fit(h_data.numpy())
    centers = torch.tensor(
        kmeans.cluster_centers_,
        dtype=model.quantizer.codebook.weight.dtype,
        device=device,
    )

    model.quantizer.codebook.weight.copy_(centers)
    model.train()

def code_usage_loss(
    distances: torch.Tensor,
) -> torch.Tensor:
    
    probs = torch.softmax(-distances / 0.1, dim=1,)
    mean_probs = probs.mean(dim=0)
    num_codes = mean_probs.numel()
    uniform = torch.full_like(mean_probs, 1.0 / num_codes)
    loss = (mean_probs * (torch.log(mean_probs + 1e-8) - torch.log(uniform))).sum()
    return loss


def main() -> None:
    seed = EXPERIMENT_SEED

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    lam_dir = RUN_ROOT / "outputs" / "lam"
    data_dir = RUN_ROOT / "data" / "gridworld"
    model_dir = lam_dir / "model"

    model_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = GridWorldTransitionDataset(data_dir / "train.npz")
    eval_dataset = GridWorldTransitionDataset(data_dir / "eval.npz")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_codes = 4
    model = LatentActionModel(
        num_codes=num_codes,
        embedding_dim=64,
        commitment_beta=0.25,
    ).to(device)

    initialize_codebook_from_h(
        model=model,
        train_loader=train_loader,
        device=device,
        max_samples=4096,
        )


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 30
    best_eval_loss = float("inf")

    history = []

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0.0
        total_train_recon_loss = 0.0
        total_train_vq_loss = 0.0
        total_samples = 0

        train_action_counts = [0] * num_codes

        for frame_t, frame_tp1 in train_loader:
            frame_t = frame_t.to(device)
            frame_tp1 = frame_tp1.to(device)
            optimizer.zero_grad()

            outputs = model(frame_t, frame_tp1)

            pred_delta = outputs["pred_delta"]
            action_ids = outputs["action_ids"]
            vq_loss = outputs["vq_loss"]

            recon_loss = delta_reconstruction_loss(pred_delta, frame_t, frame_tp1, change_weight=10.0)

            usage_loss = code_usage_loss(outputs["distances"])

            if epoch < 15:
                lambda_usage = 0.01
            else:
                lambda_usage = 0

            loss = (
                recon_loss
                + vq_loss
                + lambda_usage * usage_loss
            )

            loss.backward()
            optimizer.step()

            batch_size = frame_t.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_recon_loss += recon_loss.item() * batch_size
            total_train_vq_loss += vq_loss.item() * batch_size
            total_samples += batch_size

            for action_id in action_ids.cpu().tolist():
                train_action_counts[action_id] += 1

        train_loss = total_train_loss / total_samples
        train_recon_loss = total_train_recon_loss / total_samples
        train_vq_loss = total_train_vq_loss / total_samples

        eval_loss, eval_recon_loss, eval_vq_loss, eval_action_counts = evaluate(
            model,
            eval_loader,
            device,
            num_codes,
            lambda_usage,
        )

        history.append(
            {
                "epoch": epoch + 1,

                "train_total_loss": train_loss,
                "train_recon_loss": train_recon_loss,
                "train_vq_loss": train_vq_loss,
                "train_action_counts": train_action_counts,

                "eval_total_loss": eval_loss,
                "eval_recon_loss": eval_recon_loss,
                "eval_vq_loss": eval_vq_loss,
                "eval_action_counts": eval_action_counts,
            }
        )

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} "
            f"train_loss={train_loss:.6f} "
            f"eval_loss={eval_loss:.6f}"
        )

        print(
            f"train_actions={train_action_counts} "
            f"eval_actions={eval_action_counts}"
        )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), model_dir / "best.pt")

        with (lam_dir / "loss_history.json").open("w", encoding="utf-8") as f:
            json.dump(
                history,
                f,
                indent=2,
            )

if __name__ == "__main__":
    main()
