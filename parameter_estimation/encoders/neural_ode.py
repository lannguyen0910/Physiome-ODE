import torch
import torch.nn as nn


class LatentDynamics(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        h: torch.Tensor,
    ):

        return self.network(h)


def rk4_step(
    func,
    h: torch.Tensor,
    dt: torch.Tensor,
):

    k1 = func(h)

    k2 = func(
        h + 0.5 * dt * k1
    )

    k3 = func(
        h + 0.5 * dt * k2
    )

    k4 = func(
        h + dt * k3
    )

    return h + (
        dt / 6.0
    ) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )


class NeuralODEEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        parameter_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.gru_cell = nn.GRUCell(
            hidden_dim,
            hidden_dim,
        )

        self.latent_dynamics = (
            LatentDynamics(
                hidden_dim
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

        batch_size = X.shape[0]
        sequence_length = X.shape[1]

        h = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=X.device,
            dtype=X.dtype,
        )

        for i in range(
            sequence_length
        ):

            observation = torch.cat(
                [
                    T[:, i:i + 1],
                    X[:, i],
                    M[:, i],
                ],
                dim=-1,
            )

            observation_embedding = (
                self.input_projection(
                    observation
                )
            )

            # Assimilate current observation
            h = self.gru_cell(
                observation_embedding,
                h,
            )

            if i < sequence_length - 1:

                dt = (
                    T[:, i + 1]
                    - T[:, i]
                )

                dt = dt.unsqueeze(-1)

                h = rk4_step(
                    self.latent_dynamics,
                    h,
                    dt,
                )

        theta_pred = self.theta_head(h)

        y0_pred = self.y0_head(h)

        return theta_pred, y0_pred
