import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        padding = (
            kernel_size - 1
        ) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.residual = (
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        length = x.size(-1)

        out = self.conv1(x)
        out = out[..., :length]
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[..., :length]
        out = F.relu(out)
        out = self.dropout(out)

        residual = self.residual(x)

        return F.relu(
            out + residual
        )


class TCNEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        parameter_dim: int,
        hidden_dim: int = 128,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        channels = (
            [input_dim]
            + [hidden_dim] * num_levels
        )

        layers = []

        for i in range(num_levels):

            layers.append(
                TemporalBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(
            *layers
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

        T = T.unsqueeze(-1)

        inputs = torch.cat(
            [
                T,
                X,
                M,
            ],
            dim=-1,
        )

        inputs = inputs.transpose(
            1,
            2,
        )

        features = self.network(
            inputs
        )

        h = features[:, :, -1]

        theta_pred = self.theta_head(h)

        y0_pred = self.y0_head(h)

        return theta_pred, y0_pred
