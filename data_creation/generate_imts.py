import argparse
import glob
import os
import random
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence as pad
from torch.utils.data import DataLoader, Dataset


class Sample(NamedTuple):
    r"""A single sample of the data."""

    key: int
    inputs: tuple
    targets: Tensor

    # Ground-truth ODE information
    theta: Tensor
    y0: Tensor


class IMTS_dataset(Dataset):
    def __init__(
        self, files, ot, fh, fold, t_drop, c_drop, noise, forecasting_channels
    ):
        # load files
        torch.manual_seed(fold)
        random.seed(fold)

        T = []
        X = []
        TY = []
        Y = []

        # NEW:
        # Ground-truth ODE parameters and initial states
        THETA = []
        Y0 = []

        T_max = max(pd.read_parquet(files[0])["t"])

        # ------------------------------------------------------------
        # Determine time-series columns
        # ------------------------------------------------------------

        value_columns = list(
            pd.read_parquet(files[0]).columns
        )

        value_columns.remove("t")

        # NEW:
        # Remove ODE metadata from time-series variables
        theta_columns = [
            c for c in value_columns
            if c.startswith("theta_")
        ]

        y0_columns = [
            c for c in value_columns
            if c.startswith("y0_")
        ]

        value_columns = [
            c for c in value_columns
            if not c.startswith("theta_")
            and not c.startswith("y0_")
        ]

        observation_time = T_max * ot

        forecasting_horizon = (
            observation_time
            + (T_max * fh)
        )

        # ------------------------------------------------------------
        # Load all trajectories
        # ------------------------------------------------------------

        for f in files:

            raw_TS = pd.read_parquet(f)

            # --------------------------------------------------------
            # NEW:
            # Read ground-truth theta and y0
            #
            # They are repeated as metadata columns in the parquet
            # file, so we only need the first row.
            # --------------------------------------------------------

            theta = raw_TS[
                theta_columns
            ].iloc[0].values

            y0 = raw_TS[
                y0_columns
            ].iloc[0].values

            THETA.append(theta)
            Y0.append(y0)

            # --------------------------------------------------------
            # Existing time-series processing
            # --------------------------------------------------------

            T.append(
                raw_TS["t"]
                .loc[
                    raw_TS["t"] <= observation_time
                ]
                .values
            )

            X.append(
                raw_TS[value_columns]
                .loc[
                    raw_TS["t"] <= observation_time
                ]
                .values
            )

            TY.append(
                raw_TS["t"]
                .loc[
                    (raw_TS["t"] > observation_time)
                    & (
                        raw_TS["t"]
                        < forecasting_horizon
                    )
                ]
                .values
            )

            Y.append(
                raw_TS[value_columns]
                .loc[
                    (raw_TS["t"] > observation_time)
                    & (
                        raw_TS["t"]
                        < forecasting_horizon
                    )
                ]
                .values
            )

        # ------------------------------------------------------------
        # Convert to tensors
        # ------------------------------------------------------------

        T = torch.tensor(
            np.stack(T, axis=0)
        ).type(torch.float32)

        X = torch.tensor(
            np.stack(X, axis=0)
        ).type(torch.float32)

        TY = torch.tensor(
            np.stack(TY, axis=0)
        ).type(torch.float32)

        Y = torch.tensor(
            np.stack(Y, axis=0)
        ).type(torch.float32)

        # NEW
        THETA = torch.tensor(
            np.stack(THETA, axis=0)
        ).type(torch.float32)

        Y0 = torch.tensor(
            np.stack(Y0, axis=0)
        ).type(torch.float32)

        # ------------------------------------------------------------
        # Normalize time
        # ------------------------------------------------------------

        T = T / T_max
        TY = TY / T_max

        # ------------------------------------------------------------
        # Normalize state variables
        # ------------------------------------------------------------

        XY = torch.cat(
            [X, Y],
            axis=1
        )

        std_V = torch.std(
            XY.reshape(
                -1,
                XY.shape[-1]
            ),
            dim=0
        )

        mean_V = torch.mean(
            XY.reshape(
                -1,
                X.shape[-1]
            ),
            dim=0
        )

        X = (
            X - mean_V
        ) / std_V

        Y = (
            Y - mean_V
        ) / std_V

        # ------------------------------------------------------------
        # Drop too large samples
        # ------------------------------------------------------------

        XY_normed = torch.cat(
            [X, Y],
            axis=1
        )

        mask = (
            np.absolute(XY_normed)
            > 10
        ).any(
            dim=(1, 2)
        )

        X = X[~mask]
        Y = Y[~mask]

        T = T[~mask]
        TY = TY[~mask]

        # NEW:
        # Keep theta and y0 aligned with the remaining samples
        THETA = THETA[~mask]
        Y0 = Y0[~mask]

        # ------------------------------------------------------------
        # Apply noise
        # ------------------------------------------------------------

        X += (
            torch.randn(X.shape)
            * noise
        )

        Y += (
            torch.randn(Y.shape)
            * noise
        )

        # ------------------------------------------------------------
        # Apply masking
        # ------------------------------------------------------------

        M = (
            torch.rand(X.shape)
            > c_drop
        ).type(torch.bool)

        MY = (
            torch.rand(Y.shape)
            > c_drop
        ).type(torch.bool)

        # MY2 is created so only the relevant_channels can be MY

        MY2 = np.zeros_like(MY)

        MY2[
            :,
            :,
            forecasting_channels
        ] = 1

        MY = MY * MY2

        MY = MY.type(torch.bool)

        # ------------------------------------------------------------
        # Time masking
        # ------------------------------------------------------------

        T_MASK = (
            torch.rand(T.shape)
            > t_drop
        ).type(torch.bool) & (
            torch.sum(
                M,
                axis=-1
            ) > 0
        )

        TY_MASK = (
            torch.rand(TY.shape)
            > t_drop
        ).type(torch.bool) & (
            torch.sum(
                MY,
                axis=-1
            ) > 0
        )

        # ------------------------------------------------------------
        # Padding
        # ------------------------------------------------------------

        T = pad(
            [
                T[i, T_MASK[i]]
                for i in range(X.shape[0])
            ],
            batch_first=True
        )

        TY = pad(
            [
                TY[i, TY_MASK[i]]
                for i in range(X.shape[0])
            ],
            batch_first=True
        )

        X = pad(
            [
                X[i, T_MASK[i], :]
                for i in range(X.shape[0])
            ],
            batch_first=True
        )

        Y = pad(
            [
                Y[i, TY_MASK[i], :]
                for i in range(X.shape[0])
            ],
            batch_first=True
        )

        M = pad(
            [
                M[i, T_MASK[i], :]
                for i in range(X.shape[0])
            ],
            batch_first=True
        )

        MY = pad(
            [
                MY[i, TY_MASK[i], :]
                for i in range(X.shape[0])
            ],
            batch_first=True
        )

        # ------------------------------------------------------------
        # Apply NaN to missing observations
        # ------------------------------------------------------------

        X[~M] = torch.nan

        Y[~MY] = torch.nan

        # ------------------------------------------------------------
        # Store
        # ------------------------------------------------------------

        self.T = T
        self.TY = TY

        self.X = X
        self.Y = Y

        self.M = M
        self.MY = MY

        # NEW
        self.theta = THETA
        self.y0 = Y0

    def __len__(self):

        return self.X.shape[0]

    def __getitem__(self, idx):

        return Sample(
            key=idx,

            inputs=(
                self.T[idx],
                self.X[idx],
                self.M[idx],
                self.TY[idx],
                self.MY[idx],
            ),

            targets=self.Y[idx],

            # NEW
            theta=self.theta[idx],

            y0=self.y0[idx],
        )


def get_forecasting_channel(
    model,
    df_path="resources/Top50_final.csv"
):

    df = pd.read_csv(
        df_path
    )

    channel_scores_str = (
        df.loc[
            df["model"] == model
        ]["channel_scores"]
        .values[0]
    )

    channel_scores = np.fromstring(
        channel_scores_str
        .replace("[", "")
        .replace("]", ""),
        sep=","
    )

    top_10_ind = np.argsort(
        channel_scores
    )[-10:]

    return top_10_ind


def create_dataloaders(
    model,
    fold,
    observation_time,
    forecasting_horizon,
    t_drop,
    c_drop,
    out_path,
    raw_data_path,
    noise,
):

    files = glob.glob(
        f"{raw_data_path}/*.parquet"
    )

    forecasting_channels = (
        get_forecasting_channel(
            model
        )
    )

    random.seed(fold)

    random.shuffle(files)

    the_dataset = IMTS_dataset(
        files=files,

        ot=observation_time,

        fh=forecasting_horizon,

        t_drop=t_drop,

        c_drop=c_drop,

        fold=fold,

        forecasting_channels=forecasting_channels,

        noise=noise,
    )

    train_dataset, valid_dataset, test_dataset = (
        torch.utils.data.random_split(
            the_dataset,
            [0.7, 0.2, 0.1],
        )
    )

    torch.save(
        train_dataset,
        f"{out_path}/train.pt"
    )

    torch.save(
        valid_dataset,
        f"{out_path}/valid.pt"
    )

    torch.save(
        test_dataset,
        f"{out_path}/test.pt"
    )


raw_data_path = (
    "data/benchmark_datasets/"
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate IMTSBench"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True
    )

    args = parser.parse_args()

    print(args)

    # t_drop:
    # Chance to drop a complete timestep
    # (with all observations)

    t_drop = 0.0

    # c_drop:
    # Chance of dropping a channel
    # within a timestep

    c_drop = 0.8

    if not os.path.isdir(
        "data/final/"
    ):

        os.mkdir(
            "data/final/"
        )

    out_path = (
        f"data/final/{args.model}"
    )

    os.mkdir(
        out_path
    )

    for fold in range(5):

        os.mkdir(
            out_path
            + "/"
            + str(fold)
        )

        create_dataloaders(
            args.model,

            fold,

            observation_time=0.5,

            forecasting_horizon=0.5,

            t_drop=t_drop,

            c_drop=c_drop,

            out_path=out_path
            + "/"
            + str(fold),

            raw_data_path=f"{raw_data_path}/{args.model}",

            noise=0.05,
        )


print("DONE")
