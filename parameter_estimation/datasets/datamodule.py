import os

import torch
from torch.utils.data import DataLoader

from .collator import collate_fn
from .dataset import PhysiomeDataset


class PhysiomeDataModule:

    def __init__(
        self,
        data_path,
        batch_size=32,
        num_workers=0,
    ):

        self.data_path = data_path

        self.batch_size = (
            int(batch_size)
        )

        self.num_workers = (
            int(num_workers)
        )

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def _load(
        self,
        filename,
    ):

        path = os.path.join(
            self.data_path,
            filename,
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Dataset file not found: {path}"
            )

        return torch.load(
            path,
            weights_only=False,
        )

    def setup(self):

        print(
            "\nLoading datasets..."
        )

        train_data = self._load(
            "train.pt"
        )

        valid_data = self._load(
            "valid.pt"
        )

        test_data = self._load(
            "test.pt"
        )

        self.train_dataset = (
            PhysiomeDataset(
                train_data
            )
        )

        self.valid_dataset = (
            PhysiomeDataset(
                valid_data
            )
        )

        self.test_dataset = (
            PhysiomeDataset(
                test_data
            )
        )

        print(
            f"Train samples: "
            f"{len(self.train_dataset)}"
        )

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
            pin_memory=torch.cuda.is_available(),
        )

    def valid_loader(self):

        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def test_loader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
        )
