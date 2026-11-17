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

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.theta_mean = None
        self.theta_std = None

    def _load(self, filename):

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

    def _compute_theta_statistics(
        self,
        train_dataset,
    ):
        """
        Calculate theta mean/std using ONLY
        the training samples.
        """

        theta = train_dataset.base_dataset.theta[
            train_dataset.indices
        ].float()

        theta = torch.nan_to_num(
            theta,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        theta_mean = theta.mean(dim=0)

        theta_std = theta.std(
            dim=0,
            unbiased=False,
        )

        # Prevent division by zero for parameters
        # that do not vary in the training set.
        theta_std = torch.clamp(
            theta_std,
            min=1e-8,
        )

        return theta_mean, theta_std

    def setup(self):

        print("\nLoading datasets...")

        train_data = self._load("train.pt")
        valid_data = self._load("valid.pt")
        test_data = self._load("test.pt")

        # First create the training dataset WITHOUT
        # normalization statistics.
        train_dataset = PhysiomeDataset(
            train_data
        )

        # Calculate statistics ONLY from training data.
        self.theta_mean, self.theta_std = (
            self._compute_theta_statistics(
                train_dataset
            )
        )

        print("\nTheta normalization:")
        print("theta_mean =", self.theta_mean)
        print("theta_std  =", self.theta_std)

        # Now create all datasets using the SAME
        # training statistics.
        self.train_dataset = PhysiomeDataset(
            train_data,
            theta_mean=self.theta_mean,
            theta_std=self.theta_std,
        )

        self.valid_dataset = PhysiomeDataset(
            valid_data,
            theta_mean=self.theta_mean,
            theta_std=self.theta_std,
        )

        self.test_dataset = PhysiomeDataset(
            test_data,
            theta_mean=self.theta_mean,
            theta_std=self.theta_std,
        )

        print(
            f"\nTrain samples: "
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
