from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()

        self.net = nn.Sequential(
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
                out_channels=embedding_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_codes: int = 128,
        embedding_dim: int = 64,
        commitment_beta: float = 0.25,
    ) -> None:
        
        super().__init__()

        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_beta = commitment_beta
        self.codebook = nn.Embedding(num_embeddings=num_codes, embedding_dim=embedding_dim)

        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(
        self,
        z_e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, channels, height, width = z_e.shape

        z_e_flat = (
            z_e
            .permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, self.embedding_dim)
        )

        embeddings = self.codebook.weight
        distances = (torch.sum(z_e_flat ** 2, dim=1, keepdim=True) + torch.sum(embeddings ** 2, dim=1) - 2 * torch.matmul(z_e_flat, embeddings.t()))
        flat_token_idx = torch.argmin(distances, dim=1)
        z_q_flat = self.codebook(flat_token_idx)

        z_q = (
            z_q_flat
            .view(
                batch_size,
                height,
                width,
                self.embedding_dim,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        token_idx = flat_token_idx.view(batch_size, height,width)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = (codebook_loss + self.commitment_beta * commitment_loss)
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, token_idx, vq_loss

class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
    ) -> None:
        
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                embedding_dim,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class VQVAEImageTokenizer(nn.Module):
    def __init__(
        self,
        num_codes: int = 128,
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
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        z_e = self.encoder(x)
        z_q_st, token_ids, vq_loss = self.quantizer(z_e)
        reconstruction = self.decoder(z_q_st)

        return {
            "reconstruction": reconstruction,
            "token_ids": token_ids,
            "vq_loss": vq_loss,
        }
    