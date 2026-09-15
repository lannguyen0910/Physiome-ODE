import torch


def collate_fn(batch):
    """
    Collate Physiome samples into batch tensors.
    """

    return {
        "key": [
            item["key"]
            for item in batch
        ],

        "T": torch.stack(
            [
                item["T"]
                for item in batch
            ]
        ),

        "X": torch.stack(
            [
                item["X"]
                for item in batch
            ]
        ),

        "M": torch.stack(
            [
                item["M"]
                for item in batch
            ]
        ),

        "TY": torch.stack(
            [
                item["TY"]
                for item in batch
            ]
        ),

        "MY": torch.stack(
            [
                item["MY"]
                for item in batch
            ]
        ),

        "Y": torch.stack(
            [
                item["Y"]
                for item in batch
            ]
        ),

        "theta": torch.stack(
            [
                item["theta"]
                for item in batch
            ]
        ),

        "y0": torch.stack(
            [
                item["y0"]
                for item in batch
            ]
        ),
    }
