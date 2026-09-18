import torch

import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# TEMPORAL BLOCK
# ==========================================================

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

        # Normalize convolution outputs.
        self.norm1 = nn.BatchNorm1d(
            out_channels
        )

        self.norm2 = nn.BatchNorm1d(
            out_channels
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

        # --------------------------------------------------
        # First convolution
        # --------------------------------------------------

        out = self.conv1(x)

        out = out[
            ...,
            :length
        ]

        out = self.norm1(
            out
        )

        out = F.relu(
            out
        )

        out = self.dropout(
            out
        )

        # --------------------------------------------------
        # Second convolution
        # --------------------------------------------------

        out = self.conv2(
            out
        )

        out = out[
            ...,
            :length
        ]

        out = self.norm2(
            out
        )

        out = F.relu(
            out
        )

        out = self.dropout(
            out
        )

        # --------------------------------------------------
        # Residual connection
        # --------------------------------------------------

        residual = self.residual(
            x
        )

        return F.relu(
            out + residual
        )


# ==========================================================
# TCN ENCODER
# ==========================================================

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

        # --------------------------------------------------
        # TCN channels
        # --------------------------------------------------

        channels = (
            [input_dim]
            + [hidden_dim] * num_levels
        )

        layers = []

        for i in range(
            num_levels
        ):

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

        # --------------------------------------------------
        # Final feature normalization
        # --------------------------------------------------

        self.feature_norm = nn.LayerNorm(
            hidden_dim
        )

        # --------------------------------------------------
        # Prediction heads
        # --------------------------------------------------

        self.theta_head = nn.Linear(
            hidden_dim,
            parameter_dim,
        )

        self.y0_head = nn.Linear(
            hidden_dim,
            state_dim,
        )

        # --------------------------------------------------
        # Smaller head initialization
        # --------------------------------------------------

        nn.init.xavier_uniform_(
            self.theta_head.weight,
            gain=0.5,
        )

        nn.init.zeros_(
            self.theta_head.bias
        )

        nn.init.xavier_uniform_(
            self.y0_head.weight,
            gain=0.5,
        )

        nn.init.zeros_(
            self.y0_head.bias
        )

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(
        self,
        T: torch.Tensor,
        X: torch.Tensor,
        M: torch.Tensor,
    ):

        # --------------------------------------------------
        # Time feature
        # --------------------------------------------------

        T_input = T.unsqueeze(
            -1
        )

        # --------------------------------------------------
        # Concatenate:
        #
        # time
        # state values
        # observation mask
        #
        # [B, L, 1 + D + D]
        # --------------------------------------------------

        inputs = torch.cat(
            [
                T_input,
                X,
                M,
            ],
            dim=-1,
        )

        # --------------------------------------------------
        # Conv1d expects:
        #
        # [B, C, L]
        # --------------------------------------------------

        inputs = inputs.transpose(
            1,
            2,
        )

        # --------------------------------------------------
        # TCN
        # --------------------------------------------------

        features = self.network(
            inputs
        )

        # --------------------------------------------------
        # Select last VALID temporal position
        # instead of always using -1.
        # --------------------------------------------------

        valid_lengths = (
            (T > 0.0)
            .sum(dim=1)
        )

        last_indices = (
            valid_lengths - 1
        ).clamp(
            min=0
        )

        batch_indices = torch.arange(
            features.size(0),
            device=features.device,
        )

        h = features[
            batch_indices,
            :,
            last_indices,
        ]

        # --------------------------------------------------
        # Normalize representation
        # --------------------------------------------------

        h = self.feature_norm(
            h
        )

        # --------------------------------------------------
        # Predict raw values
        # --------------------------------------------------

        theta_raw = self.theta_head(
            h
        )

        y0_raw = self.y0_head(
            h
        )

        # --------------------------------------------------
        # Positive parameterization
        # --------------------------------------------------

        theta_pred = (
            F.softplus(
                theta_raw
            )
            + 1e-6
        )

        y0_pred = (
            F.softplus(
                y0_raw
            )
            + 1e-6
        )

        # --------------------------------------------------
        # Final safety check
        # --------------------------------------------------

        theta_pred = torch.nan_to_num(
            theta_pred,
            nan=1.0,
            posinf=1e6,
            neginf=1e-6,
        )

        y0_pred = torch.nan_to_num(
            y0_pred,
            nan=1.0,
            posinf=1e6,
            neginf=1e-6,
        )

        return (
            theta_pred,
            y0_pred,
        )
