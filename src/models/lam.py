from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
    ) -> None:
        
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4,4)),
        )

        self.projection = nn.Sequential(
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(
        self,
        frame_t: torch.Tensor,
        frame_tp1: torch.Tensor,
    ) -> torch.Tensor:

        delta = frame_tp1 - frame_t
        x = self.net(delta)
        x = x.flatten(start_dim=1)
        h = self.projection(x)
        return h

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_codes: int = 4,
        embedding_dim: int = 64,
        commitment_beta: float = 0.25,
    ) -> None:
        
        super().__init__()

        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_beta = commitment_beta
        self.codebook = nn.Embedding(num_embeddings=num_codes, embedding_dim=embedding_dim)

        nn.init.uniform_(self.codebook.weight, -0.1, 0.1)

    def forward(
        self,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        embeddings = self.codebook.weight
        distances = (torch.sum(h ** 2, dim=1, keepdim=True) + torch.sum(embeddings ** 2, dim=1) - 2 * torch.matmul(h, embeddings.t()))
        action_ids = torch.argmin(distances, dim=1)
        h_q = self.codebook(action_ids)
        codebook_loss = F.mse_loss(h_q, h.detach())
        commitment_loss = F.mse_loss(h, h_q.detach())
        vq_loss = (codebook_loss + self.commitment_beta * commitment_loss)
        h_q_st = h + (h_q - h).detach()

        return h_q_st, action_ids, vq_loss, distances


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
    ) -> None:
        
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.action_projection = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        frame_t: torch.Tensor,
        action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        
        frame_feature = self.frame_encoder(frame_t)
        action_feature = self.action_projection(action_embedding)
        action_feature = action_feature.unsqueeze(-1).unsqueeze(-1)
        action_feature = action_feature.expand(-1, -1, frame_feature.shape[2], frame_feature.shape[3])
        x = torch.cat([frame_feature, action_feature], dim=1)
        x = self.fusion(x)
        pred_delta = self.decoder(x)

        return pred_delta


class LatentActionModel(nn.Module):
    def __init__(
        self,
        num_codes: int = 4,
        embedding_dim: int = 64,
        commitment_beta: float = 0.25,
    ) -> None:
        
        super().__init__()

        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.quantizer = VectorQuantizer(
            num_codes=num_codes,
            embedding_dim=embedding_dim,
            commitment_beta=commitment_beta,
        )
        self.decoder = Decoder(embedding_dim=embedding_dim)

    def forward(
        self,
        frame_t: torch.Tensor,
        frame_tp1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        
        h = self.encoder(frame_t, frame_tp1)
        h_q_st, action_ids, vq_loss, distances = self.quantizer(h)
        pred_delta = self.decoder(frame_t, h_q_st)

        return {
            "pred_delta": pred_delta,
            "action_ids": action_ids,
            "vq_loss": vq_loss,
            "distances": distances,
            "h" : h,
        }