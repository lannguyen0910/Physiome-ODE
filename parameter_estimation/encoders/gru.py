import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        parameter_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
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

        # T: [B, L]
        # X: [B, L, D]
        # M: [B, L, D]

        T = T.unsqueeze(-1)

        inputs = torch.cat(
            [
                T,
                X,
                M,
            ],
            dim=-1,
        )

        _, hidden = self.gru(inputs)

        h = hidden[-1]

        theta_pred = self.theta_head(h)

        y0_pred = self.y0_head(h)

        return theta_pred, y0_pred
