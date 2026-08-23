import sys

import torch

from torch.utils.data import Dataset, DataLoader


class IMTS_dataset:
    """
    Dummy compatibility class.
    """
    pass


sys.modules["__main__"].IMTS_dataset = IMTS_dataset


class PhysiomeDataset(Dataset):
    """
    Each sample contains:
        T       : observation times
        X       : observed states
        M       : observation mask
        TY      : future times
        MY      : future observation mask
        Y       : future states
        theta   : ground-truth ODE parameters
        y0      : ground-truth initial state
    """

    def __init__(self, data):
        if isinstance(data, torch.utils.data.Subset):
            self.data = data
            base_dataset = data.dataset
            self.indices = data.indices
            self.base_dataset = base_dataset

        else:
            self.data = data
            self.indices = list(
                range(len(data))
            )
            self.base_dataset = data

        required = [
            "T",
            "TY",
            "X",
            "M",
            "Y",
            "MY",
            "theta",
            "y0",
        ]

        for name in required:
            if not hasattr(
                self.base_dataset,
                name,
            ):
                raise AttributeError(
                    f"Stored dataset does not contain "
                    f"'{name}'."
                )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        T = self.base_dataset.T[real_idx]
        X = self.base_dataset.X[real_idx]
        TY = self.base_dataset.TY[real_idx]
        Y = self.base_dataset.Y[real_idx]
        theta = self.base_dataset.theta[real_idx]
        y0 = self.base_dataset.y0[real_idx]
        M = self.base_dataset.M[real_idx].float()
        MY = self.base_dataset.MY[real_idx].float()

        return {
            "key": real_idx,
            "T": T,
            "X": X,
            "M": M,
            "TY": TY,
            "MY": MY,
            "Y": Y,
            "theta": theta,
            "y0": y0,
        }


def collate_fn(batch):
    """
    Collate samples into batch tensors.
    """

    keys = [
        item["key"]
        for item in batch
    ]

    T = torch.stack(
        [
            item["T"]
            for item in batch
        ]
    )

    X = torch.stack(
        [
            item["X"]
            for item in batch
        ]
    )

    M = torch.stack(
        [
            item["M"]
            for item in batch
        ]
    )

    TY = torch.stack(
        [
            item["TY"]
            for item in batch
        ]
    )

    MY = torch.stack(
        [
            item["MY"]
            for item in batch
        ]
    )

    Y = torch.stack(
        [
            item["Y"]
            for item in batch
        ]
    )

    theta = torch.stack(
        [
            item["theta"]
            for item in batch
        ]
    )

    y0 = torch.stack(
        [
            item["y0"]
            for item in batch
        ]
    )

    return {
        "key": keys,
        "T": T,
        "X": X,
        "M": M,
        "TY": TY,
        "MY": MY,
        "Y": Y,
        "theta": theta,
        "y0": y0,
    }


class PhysiomeDataModule:
    def __init__(
        self,
        data_path,
        batch_size=32,
        num_workers=0,
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def setup(self):
        print("Loading datasets...")

        train_data = torch.load(
            f"{self.data_path}/train.pt",
            weights_only=False,
        )

        valid_data = torch.load(
            f"{self.data_path}/valid.pt",
            weights_only=False,
        )

        test_data = torch.load(
            f"{self.data_path}/test.pt",
            weights_only=False,
        )

        self.train_dataset = PhysiomeDataset(train_data)
        self.valid_dataset = PhysiomeDataset(valid_data)
        self.test_dataset = PhysiomeDataset(test_data)

        print(f"Train samples: {len(self.train_dataset)}")

        print(
            f"Validation samples: "
            f"{len(self.valid_dataset)}"
        )

        print(
            f"Test samples: "
            f"{len(self.test_dataset)}"
        )

    def train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def valid_loader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
