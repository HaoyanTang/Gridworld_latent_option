from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.env.gridworld import GridWorldEnv
from src.models.vqvae import VQVAEImageTokenizer


class TokenizedGridWorldEnv:
    def __init__(
        self,
        tokenizer: VQVAEImageTokenizer,
        device: torch.device,
        grid_size: int = 8,
        image_size: int = 64,
        max_steps: int = 32,
        num_walls: int = 10,
        seed: int = 1,
    ) -> None:
        self.tokenizer = tokenizer.to(device).eval()
        self.device = device
        self.grid_size = grid_size

        self.env = GridWorldEnv(
            grid_size=grid_size,
            image_size=image_size,
            max_steps=max_steps,
            num_walls=num_walls,
            seed=seed,
        )

        self.last_frame: np.ndarray | None = None

        for parameter in self.tokenizer.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:
        frame = torch.from_numpy(
            observation
        ).float().div(255.0)

        frame = frame.unsqueeze(0).to(self.device)
        z_e = self.tokenizer.encoder(frame)
        _, token_ids, _ = (self.tokenizer.quantizer(z_e))
        return (
            token_ids[0]
            .cpu()
            .numpy()
            .astype(np.int64)
        )

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(
            seed=seed
        )

        self.last_frame = observation

        info = dict(info)
        info["requested_seed"] = seed
        info["layout_seed"] = seed

        return self.encode(observation), info

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        self.last_frame = observation

        return (
            self.encode(observation),
            reward,
            terminated,
            truncated,
            info,
        )