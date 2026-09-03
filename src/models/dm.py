from __future__ import annotations

import torch
import torch.nn as nn


class LatentDynamicsModel(nn.Module):
    def __init__(
        self,
        tokenizer_size: int = 128,
        num_actions: int = 4,
        state_embedding_dim: int = 64,
        action_embedding_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.tokenizer_size = tokenizer_size
        self.num_actions = num_actions

        self.state_embedding = nn.Embedding(
            num_embeddings=tokenizer_size,
            embedding_dim=state_embedding_dim,
        )
        self.action_embedding = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=action_embedding_dim,
        )

        fusion_dim = (state_embedding_dim + action_embedding_dim)

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=fusion_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=tokenizer_size,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:
        
        if z_t.dtype != torch.long:
            z_t = z_t.long()

        if action_ids.dtype != torch.long:
            action_ids = action_ids.long()

        state_feature = self.state_embedding(z_t)
        state_feature = state_feature.permute(0, 3, 1, 2)

        action_feature = self.action_embedding(action_ids)
        action_feature = action_feature.unsqueeze(-1).unsqueeze(-1)
        action_feature = action_feature.expand(-1, -1, state_feature.shape[2], state_feature.shape[3])

        x = torch.cat([state_feature,action_feature], dim=1)
        logits = self.net(x)

        return logits
