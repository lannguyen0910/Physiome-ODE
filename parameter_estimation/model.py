import torch
import torch.nn as nn


class ParameterEncoder(nn.Module):
    """
    Input:
        T : [B, T, 1]
        X : [B, T, state_dim]
        M : [B, T, state_dim]

    Output:
        theta_pred : [B, parameter_dim]
        y0_pred    : [B, state_dim]
    """

    def __init__(
        self,
        state_dim,
        parameter_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.parameter_dim = parameter_dim

        input_dim = 1 + state_dim + state_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.parameter_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, parameter_dim),
        )

        self.initial_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, T, X, M):
        if T.dim() == 2:
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

        log_theta_pred = self.parameter_head(h)

        theta_pred = torch.exp(
            log_theta_pred
        )

        # Predict initial state
        y0_raw = self.initial_state_head(h)

        y0_pred = torch.nn.functional.softplus(
            y0_raw
        )

        return theta_pred, y0_pred


class ParameterEstimator(nn.Module):
    def __init__(
        self,
        state_dim,
        parameter_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = ParameterEncoder(
            state_dim=state_dim,
            parameter_dim=parameter_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        T,
        X,
        M,
    ):

        theta_pred, y0_pred = self.encoder(
            T=T,
            X=X,
            M=M,
        )

        return theta_pred, y0_pred
