import torch

from torch.utils.data import Dataset, DataLoader


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
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        T = sample.inputs[0]
        X = sample.inputs[1]
        M = sample.inputs[2]
        TY = sample.inputs[3]
        MY = sample.inputs[4]
        Y = sample.targets
        theta = sample.theta
        y0 = sample.y0

        return {
            "key": sample.key,

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
