"""
ContiFormer-based parameter encoder for the Physiome-ODE thesis pipeline.

This adapter reuses the continuous-time Transformer implementation from
PhysioPro's ``physiopro.network.contiformer`` module.

Expected thesis-pipeline interface
-----------------------------------
    theta_pred, y0_pred = encoder(T, X, M)

Input
-----
T : [B, L]
    Observation timestamps.
X : [B, L, state_dim]
    Observed state values. For the current Dupont model state_dim=2.
M : [B, L, state_dim]
    Observation mask (1 = observed, 0 = missing).

Output
------
theta_pred : [B, 14]
    Positive predicted Physiome-ODE parameters.
y0_pred : [B, 2]
    Predicted initial state [Z0, Y0].

Important design choice
-----------------------
Unlike the GRU/TCN encoder, time is NOT concatenated into the feature vector.
The PhysioPro ContiFormer already receives time separately and uses it in its
continuous-time positional encoding and ODE-based attention projections.
The feature vector is therefore [Z, Y, M_Z, M_Y].
"""

import torch
from torch import nn
import torch.nn.functional as F

from .contiformer import ContiFormer


class ContiFormerParameterEncoder(nn.Module):
    """Continuous-time Transformer encoder for per-sample ODE identification."""

    def __init__(
        self,
        state_dim: int = 2,
        hidden_dim: int = 128,
        d_inner: int = 256,
        n_layers: int = 2,
        n_head: int = 4,
        d_k: int = 32,
        d_v: int = 32,
        dropout: float = 0.1,
        actfn_ode: str = "softplus",
        layer_type_ode: str = "concatnorm",
        zero_init_ode: bool = True,
        atol_ode: float = 1e-2,
        rtol_ode: float = 1e-2,
        method_ode: str = "rk4",
        regularize: bool = False,
        approximate_method: str = "bilinear",
        nlinspace: int = 1,
        interpolate_ode: str = "linear",
        itol_ode: float = 1e-2,
        add_pe: bool = False,
        normalize_before: bool = True,
        theta_dim: int = 14,
        y0_dim: int = 2,
        pool: str = "mean_last",
    ):
        super().__init__()

        if hidden_dim != n_head * d_k:
            raise ValueError(
                "For the uploaded PhysioPro ContiFormer, hidden_dim must "
                "equal n_head * d_k."
            )

        if hidden_dim != n_head * d_v:
            raise ValueError(
                "For the uploaded PhysioPro ContiFormer, hidden_dim must "
                "equal n_head * d_v."
            )

        if pool not in {"mean", "last", "mean_last"}:
            raise ValueError(
                "pool must be one of: 'mean', 'last', 'mean_last'."
            )

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.theta_dim = theta_dim
        self.y0_dim = y0_dim
        self.pool = pool

        # The PhysioPro ContiFormer receives values/masks as x and timestamps
        # separately.  For Z/Y this is [Z, Y, M_Z, M_Y] -> 4 features.
        self.input_dim = 2 * state_dim

        self.backbone = ContiFormer(
            input_size=self.input_dim,
            d_model=hidden_dim,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            actfn_ode=actfn_ode,
            layer_type_ode=layer_type_ode,
            zero_init_ode=zero_init_ode,
            atol_ode=atol_ode,
            rtol_ode=rtol_ode,
            method_ode=method_ode,
            regularize=regularize,
            approximate_method=approximate_method,
            nlinspace=nlinspace,
            interpolate_ode=interpolate_ode,
            itol_ode=itol_ode,
            add_pe=add_pe,
            normalize_before=normalize_before,
        )

        # The backbone returns a sequence [B, L, hidden_dim].
        summary_dim = hidden_dim
        if pool == "mean_last":
            summary_dim = 2 * hidden_dim

        self.summary_norm = nn.LayerNorm(summary_dim)

        self.theta_head = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, theta_dim),
        )

        self.y0_head = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, y0_dim),
        )

    @staticmethod
    def _replace_nonfinite(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    @staticmethod
    def _last_valid_index(valid: torch.Tensor) -> torch.Tensor:
        """Return [B] index of the last valid event."""
        lengths = valid.long().sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError(
                "Every sample must contain at least one observed event."
            )
        return lengths - 1

    def _prepare_backbone_inputs(
        self,
        T: torch.Tensor,
        X: torch.Tensor,
        M: torch.Tensor,
    ):
        """
        Prepare padded IMTS batches for the uploaded PhysioPro ContiFormer.

        PhysioPro's interpolation module treats zero timestamps after the
        sequence start as padding.  It does not directly consume our M mask
        when constructing interpolation coefficients, so padded positions are
        copied from the last valid event before entering the backbone.
        Attention still receives a True padding mask and therefore ignores
        those copied tokens as attention keys.
        """
        T = self._replace_nonfinite(T)
        X = self._replace_nonfinite(X)
        M = self._replace_nonfinite(M).clamp(0.0, 1.0)

        # Missing values are zeroed; the mask tells the model they were absent.
        X = X * M

        # A sequence slot is a real event when at least one state channel is
        # observed. Shape: [B, L].
        valid = M.bool().any(dim=-1)
        last_idx = self._last_valid_index(valid)

        batch_index = torch.arange(
            T.size(0),
            device=T.device,
        )

        last_t = T[
            batch_index,
            last_idx,
        ]

        x_features = torch.cat(
            [X, M],
            dim=-1,
        )

        last_features = x_features[
            batch_index,
            last_idx,
        ]

        # PhysioPro interpolation expects nonzero timestamps in padded slots to
        # be treated as ordinary coordinates. Copy the final valid time/value
        # there, then rely on the attention mask to ignore the padded tokens.
        T_safe = torch.where(
            valid,
            T,
            last_t.unsqueeze(1),
        )

        X_safe = torch.where(
            valid.unsqueeze(-1),
            x_features,
            last_features.unsqueeze(1),
        )

        # True = ignore this key position in PhysioPro attention.
        padding_mask = ~valid

        return (
            X_safe,
            T_safe,
            padding_mask,
            valid,
        )

    def _pool_sequence(
        self,
        sequence: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Pool [B, L, H] into one representation per sample."""

        if self.pool == "last":
            last_idx = self._last_valid_index(valid)
            batch_index = torch.arange(
                sequence.size(0),
                device=sequence.device,
            )
            return sequence[batch_index, last_idx]

        valid_f = valid.unsqueeze(-1).to(
            dtype=sequence.dtype
        )
        mean = (
            sequence * valid_f
        ).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)

        if self.pool == "mean":
            return mean

        last_idx = self._last_valid_index(valid)
        batch_index = torch.arange(
            sequence.size(0),
            device=sequence.device,
        )
        last = sequence[batch_index, last_idx]

        return torch.cat(
            [mean, last],
            dim=-1,
        )

    def forward(
        self,
        T: torch.Tensor,
        X: torch.Tensor,
        M: torch.Tensor,
    ):
        """
        Encode an irregular observation history.

        Parameters
        ----------
        T : [B, L]
            Observation timestamps.
        X : [B, L, state_dim]
            Observation values.
        M : [B, L, state_dim]
            Observation mask.
        """
        if X.ndim != 3:
            raise ValueError(
                f"Expected X with shape [B, L, D], got {tuple(X.shape)}"
            )

        if T.ndim != 2:
            raise ValueError(
                f"Expected T with shape [B, L], got {tuple(T.shape)}"
            )

        if M.shape != X.shape:
            raise ValueError(
                "M must have the same shape as X."
            )

        if X.size(-1) != self.state_dim:
            raise ValueError(
                f"Expected state_dim={self.state_dim}, "
                f"got X.shape[-1]={X.size(-1)}."
            )

        if T.shape[:2] != X.shape[:2]:
            raise ValueError(
                "T and X must have the same batch and sequence dimensions."
            )

        backbone_x, backbone_t, padding_mask, valid = (
            self._prepare_backbone_inputs(T, X, M)
        )

        # --------------------------------------------------
        # PhysioPro ContiFormer
        # --------------------------------------------------
        # Important: future observations are never passed here.
        # Only T, X, M from the observation history are used.
        # --------------------------------------------------
        sequence, _ = self.backbone(
            backbone_x,
            t=backbone_t,
            mask=padding_mask,
        )

        summary = self._pool_sequence(
            sequence,
            valid,
        )

        summary = self.summary_norm(summary)

        theta_raw = self.theta_head(summary)
        y0_pred = self.y0_head(summary)

        # Physiome parameters are positive in the current Dupont setup.
        # Softplus avoids hard clipping while preserving differentiability.
        theta_pred = F.softplus(theta_raw) + 1e-6

        return theta_pred, y0_pred
