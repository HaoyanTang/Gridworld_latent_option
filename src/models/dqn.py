from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.activation(self.conv1(features))
        features = self.conv2(features)
        return self.activation(features + residual)


class TokenDQN(nn.Module):
    def __init__(
        self,
        tokenizer_size: int = 128,
        num_actions: int = 4,
        token_embedding_dim: int =32,
        hidden_channels: int = 64,
        hidden_dim: int = 256,
        token_grid_size: int = 8,
        num_residual_blocks: int = 3,
    ) -> None:
        super().__init__()

        self.tokenizer_size = tokenizer_size
        self.num_actions = num_actions
        self.token_grid_size = token_grid_size

        self.token_embedding = nn.Embedding(
            num_embeddings=tokenizer_size,
            embedding_dim=token_embedding_dim,
        )

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=token_embedding_dim,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            *[ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)],
        )

        self.shared_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                hidden_channels * token_grid_size * token_grid_size,
                hidden_dim,
            ),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, token_state: torch.Tensor) -> torch.Tensor:
        if token_state.ndim == 2:
            token_state = token_state.unsqueeze(0)

        token_state = token_state.long()

        features = self.token_embedding(token_state)
        features = features.permute(0, 3, 1, 2).contiguous()
        features = self.feature_extractor(features)
        features = self.shared_head(features)

        state_value = self.value_head(features)
        action_advantages = self.advantage_head(features)

        return (state_value + action_advantages - action_advantages.mean(dim=1, keepdim=True))
