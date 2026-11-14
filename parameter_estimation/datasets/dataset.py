import sys

import torch
from torch.utils.data import Dataset


class IMTS_dataset:
    """
    Dummy compatibility class.

    This is needed because older .pt files may contain
    an object serialized as __main__.IMTS_dataset.
    """
    pass


sys.modules["__main__"].IMTS_dataset = IMTS_dataset


class PhysiomeDataset(Dataset):
    """
    Physiome IMTS dataset.

    Each sample contains:

        T       : observation times
        X       : observed states
        M       : observation mask

        TY      : future times
        MY      : future observation mask
        Y       : future states

        theta   : ground-truth ODE parameters
        y0      : ground-truth initial state

    theta_norm is added for parameter-loss training.
    The original physical theta is kept unchanged.
    """

    REQUIRED_FIELDS = [
        "T",
        "TY",
        "X",
        "M",
        "Y",
        "MY",
        "theta",
        "y0",
    ]

    def __init__(
        self,
        data,
        theta_mean=None,
        theta_std=None,
    ):
        super().__init__()

        if isinstance(
            data,
            torch.utils.data.Subset,
        ):
            self.data = data
            self.indices = list(data.indices)
            self.base_dataset = data.dataset

        else:
            self.data = data
            self.indices = list(range(len(data)))
            self.base_dataset = data

        for field in self.REQUIRED_FIELDS:
            if not hasattr(
                self.base_dataset,
                field,
            ):
                raise AttributeError(
                    "Stored dataset does not contain "
                    f"'{field}'."
                )

        # Parameter normalization statistics.
        # These must come ONLY from the training set.
        self.theta_mean = theta_mean
        self.theta_std = theta_std

        if (
            self.theta_mean is not None
            and self.theta_std is not None
        ):
            self.theta_mean = self.theta_mean.float()
            self.theta_std = self.theta_std.float()

            # Avoid division by zero for constant parameters.
            self.theta_std = torch.clamp(
                self.theta_std,
                min=1e-8,
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        T = self.base_dataset.T[
            real_idx
        ].float()

        X = self.base_dataset.X[
            real_idx
        ].float()

        M = self.base_dataset.M[
            real_idx
        ].float()

        TY = self.base_dataset.TY[
            real_idx
        ].float()

        MY = self.base_dataset.MY[
            real_idx
        ].float()

        Y = self.base_dataset.Y[
            real_idx
        ].float()

        theta = self.base_dataset.theta[
            real_idx
        ].float()

        y0 = self.base_dataset.y0[
            real_idx
        ].float()

        # Replace invalid state observations.
        X = torch.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        Y = torch.nan_to_num(
            Y,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        theta = torch.nan_to_num(
            theta,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        y0 = torch.nan_to_num(
            y0,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # --------------------------------------------------
        # Normalize theta ONLY for parameter-loss training.
        # --------------------------------------------------

        if (
            self.theta_mean is not None
            and self.theta_std is not None
        ):
            theta_norm = (
                theta - self.theta_mean
            ) / self.theta_std
        else:
            theta_norm = theta.clone()

        return {
            "key": real_idx,
            "T": T,
            "X": X,
            "M": M,
            "TY": TY,
            "MY": MY,
            "Y": Y,

            # Physical parameters.
            "theta": theta,

            # Normalized parameters for parameter loss.
            "theta_norm": theta_norm,

            "y0": y0,
        }
