# ==========================================================
# utilities/optimizer.py
# ==========================================================

from typing import Iterable

import torch
from torch import nn


def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Build a PyTorch optimizer.

    No optimizer registry is used.

    Supported optimizers:
        - adamw
        - adam
        - sgd

    Parameters
    ----------
    model:
        Model whose trainable parameters will be optimized.

    name:
        Optimizer name.

    learning_rate:
        Learning rate.

    weight_decay:
        L2-style weight decay.

    kwargs:
        Additional optimizer-specific arguments.

    Returns
    -------
    torch.optim.Optimizer
    """

    # ------------------------------------------------------
    # Normalize optimizer name
    # ------------------------------------------------------

    name = str(name).lower().strip()

    # ------------------------------------------------------
    # Trainable parameters only
    # ------------------------------------------------------

    parameters: Iterable[nn.Parameter] = (
        p
        for p in model.parameters()
        if p.requires_grad
    )

    # ------------------------------------------------------
    # AdamW
    # ------------------------------------------------------

    if name == "adamw":

        return torch.optim.AdamW(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            **kwargs,
        )

    # ------------------------------------------------------
    # Adam
    # ------------------------------------------------------

    if name == "adam":

        return torch.optim.Adam(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            **kwargs,
        )

    # ------------------------------------------------------
    # SGD
    # ------------------------------------------------------

    if name == "sgd":

        return torch.optim.SGD(
            parameters,
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
            **kwargs,
        )

    # ------------------------------------------------------
    # Unknown optimizer
    # ------------------------------------------------------

    raise ValueError(
        f"Unknown optimizer '{name}'. "
        f"Supported optimizers: adamw, adam, sgd."
    )
