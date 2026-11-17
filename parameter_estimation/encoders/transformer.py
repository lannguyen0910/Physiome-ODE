import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        max_period: float = 10000.0,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(
        self,
        T: torch.Tensor,
    ):

        half_dim = (
            self.embedding_dim // 2
        )

        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(
                half_dim,
                device=T.device,
                dtype=T.dtype,
            )
            / max(
                half_dim - 1,
                1,
            )
        )

        angles = (
            T.unsqueeze(-1)
            * frequencies
        )

        embedding = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=-1,
        )

        if self.embedding_dim % 2:

            embedding = torch.cat(
                [
                    embedding,
                    torch.zeros_like(
                        embedding[..., :1]
                    ),
                ],
                dim=-1,
            )

        return embedding


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        parameter_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.time_embedding = TimeEmbedding(
            hidden_dim
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
        )

        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        )

        self.theta_head = nn.Linear(
            hidden_dim,
            parameter_dim,
        )

        self.y0_head = nn.Linear(
            hidden_dim,
            state_dim,
        )

    def forward(
        self,
        T: torch.Tensor,
        X: torch.Tensor,
        M: torch.Tensor,
    ):

        T_feature = T.unsqueeze(-1)

        inputs = torch.cat(
            [
                T_feature,
                X,
                M,
            ],
            dim=-1,
        )

        h = self.input_projection(
            inputs
        )

        # Actual continuous timestamps
        h = (
            h
            + self.time_embedding(T)
        )

        h = self.transformer(h)

        h = h[:, -1]

        theta_pred = self.theta_head(h)

        y0_pred = self.y0_head(h)

        return theta_pred, y0_pred
