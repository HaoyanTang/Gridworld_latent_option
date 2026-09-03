from pathlib import Path
import os
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

RUN_ROOT = Path(os.environ.get("GENIE_RUN_ROOT")).resolve()
EXPERIMENT_SEED = int(os.environ.get("GENIE_SEED"))

from src.models.vqvae import VQVAEImageTokenizer


class GridWorldFrameDataset(Dataset):
    def __init__(self, npz_path: str) -> None:
        data = np.load(npz_path)

        frames = data["frames"]
        n, s, c, h, w = frames.shape
        self.frames = frames.reshape(n*s, c, h, w)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.frames[idx]
        #x~[0, 1] as output of decoder is sigmoid ~[0, 1]
        frame = torch.from_numpy(frame).float()/ 255.0
        return frame

def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    goal_weight: float = 5.0,
    agent_weight: float = 5.0,
) -> torch.Tensor:

    pixel_loss = (reconstruction - target) ** 2
    red = target[:, 0, :, :]
    green = target[:, 1, :, :]
    blue = target[:, 2, :, :]

    goal_mask = (green > red + 0.2)& (green > blue + 0.2)& (green > 0.4)
    goal_mask = goal_mask.unsqueeze(1)

    agent_mask = (blue > red + 0.2)& (blue > green + 0.2)& (blue > 0.4)
    agent_mask = agent_mask.unsqueeze(1)

    weights = torch.ones_like(pixel_loss)
    weights = torch.where(goal_mask, goal_weight, weights)
    weights = torch.where(agent_mask, agent_weight, weights)
    return (pixel_loss * weights).mean()

def evaluate(
    model: VQVAEImageTokenizer,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:

    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)

            outputs = model(images)

            reconstruction = outputs["reconstruction"]

            vq_loss = outputs["vq_loss"]
            recon_loss = reconstruction_loss(reconstruction, images, goal_weight=5.0, agent_weight=5.0)
            loss = recon_loss + vq_loss

            batch_size = images.size(0)
            total_recon_loss += recon_loss.item() * batch_size
            total_vq_loss += vq_loss.item() * batch_size
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss/ total_samples, total_recon_loss/ total_samples, total_vq_loss/ total_samples

def save_reconstruction_grid(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    path: Path,
    max_images: int = 8,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    originals = (
        originals[:max_images]
        .detach()
        .cpu()
        .clamp(0.0, 1.0)
    )
    reconstructions = (
        reconstructions[:max_images]
        .detach()
        .cpu()
        .clamp(0.0, 1.0)
    )

    b, c, h, w = originals.shape

    grid = torch.zeros(c, 2 * h, b * w)

    for i in range(b):
        grid[:, 0:h, i * w:(i + 1) * w] = originals[i]
        grid[:, h:2 * h, i * w:(i + 1) * w] = reconstructions[i]

    array = (grid.permute(1, 2, 0).numpy()* 255.0).astype(np.uint8)

    Image.fromarray(array).save(path)


def main() -> None:
    random.seed(EXPERIMENT_SEED)
    np.random.seed(EXPERIMENT_SEED)
    torch.manual_seed(EXPERIMENT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EXPERIMENT_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer_dir = RUN_ROOT / "outputs" / "tokenizer"
    data_dir = RUN_ROOT / "data" / "gridworld"
    model_dir = tokenizer_dir / "model"
    recon_dir = tokenizer_dir / "reconstructions"

    model_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = GridWorldFrameDataset(data_dir / "train.npz")
    eval_dataset = GridWorldFrameDataset(data_dir / "eval.npz")

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAEImageTokenizer(
        num_codes=128,
        embedding_dim=64,
        commitment_beta=0.25,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,)
    num_epochs = 20
    best_eval_loss = float("inf")

    history =[]

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_train_recon_loss = 0.0
        total_train_vq_loss = 0.0
        total_samples = 0

        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            reconstruction = outputs["reconstruction"]

            vq_loss = outputs["vq_loss"]
            recon_loss = reconstruction_loss(reconstruction, images, goal_weight=5.0, agent_weight=5.0)
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_recon_loss += recon_loss.item() * batch_size
            total_train_vq_loss += vq_loss.item() * batch_size
            total_samples += batch_size
        train_loss = total_train_loss/ total_samples
        train_recon_loss = total_train_recon_loss/ total_samples
        train_vq_loss = total_train_vq_loss/ total_samples

        eval_loss, eval_recon_loss, eval_vq_loss = evaluate(
            model,
            eval_loader,
            device,
        )

        model.eval()
        sample_images = next(iter(eval_loader))
        sample_images = sample_images.to(device)

        with torch.no_grad():
            sample_outputs = model(sample_images)
        
        sample_reconstruction = sample_outputs["reconstruction"]
        save_reconstruction_grid(
            sample_images,
            sample_reconstruction,
            recon_dir / f"epoch_{epoch + 1:03d}.png",
            max_images=8,
        )

        history.append(
            {
                "epoch": epoch + 1,

                "train_total_loss": train_loss,
                "train_recon_loss": train_recon_loss,
                "train_vq_loss": train_vq_loss,

                "eval_total_loss": eval_loss,
                "eval_recon_loss": eval_recon_loss,
                "eval_vq_loss": eval_vq_loss,
            }
        )

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs} "
            f"train_loss={train_loss:.6f} "
            f"eval_loss={eval_loss:.6f}"
        )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), model_dir / "best.pt")

        with (tokenizer_dir / "loss_history.json").open("w", encoding="utf-8",) as f:
            json.dump(history, f,indent=2)
    print("Training finished.")


if __name__ == "__main__":
    main()
