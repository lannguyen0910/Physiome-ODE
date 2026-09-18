import os

import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# PARAMETER RECOVERY
# ==========================================================

def plot_parameter_recovery(
    theta_true,
    theta_pred,
    parameter_names,
    output_dir="figures/parameters",
):
    """
    Plot true vs predicted values for every ODE parameter.

    Parameters
    ----------
    theta_true:
        Tensor [N, P]

    theta_pred:
        Tensor [N, P]

    parameter_names:
        List[str] of length P

    output_dir:
        Directory for output PNG files.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    theta_true = (
        theta_true
        .detach()
        .cpu()
        .numpy()
    )

    theta_pred = (
        theta_pred
        .detach()
        .cpu()
        .numpy()
    )

    if theta_true.ndim != 2:
        raise ValueError(
            "theta_true must have shape [N, P]. "
            f"Got {theta_true.shape}."
        )

    if theta_pred.shape != theta_true.shape:
        raise ValueError(
            "theta_pred and theta_true must have "
            "the same shape. "
            f"Got {theta_pred.shape} and "
            f"{theta_true.shape}."
        )

    n_parameters = theta_true.shape[1]

    if len(parameter_names) != n_parameters:
        raise ValueError(
            "Number of parameter names does not "
            "match parameter dimension. "
            f"Got {len(parameter_names)} names "
            f"for {n_parameters} parameters."
        )

    for j in range(n_parameters):

        true_values = theta_true[:, j]
        pred_values = theta_pred[:, j]

        # Remove non-finite points
        valid = (
            np.isfinite(true_values)
            &
            np.isfinite(pred_values)
        )

        true_values = true_values[valid]
        pred_values = pred_values[valid]

        if len(true_values) == 0:
            continue

        plt.figure(
            figsize=(6, 6)
        )

        plt.scatter(
            true_values,
            pred_values,
            alpha=0.6,
        )

        minimum = min(
            true_values.min(),
            pred_values.min(),
        )

        maximum = max(
            true_values.max(),
            pred_values.max(),
        )

        # Avoid zero-width axis
        if np.isclose(
            minimum,
            maximum,
        ):
            padding = (
                abs(minimum) * 0.05
                + 1e-3
            )

            minimum -= padding
            maximum += padding

        # Perfect recovery line
        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
            label="Perfect recovery",
        )

        plt.xlabel(
            f"True {parameter_names[j]}"
        )

        plt.ylabel(
            f"Estimated {parameter_names[j]}"
        )

        plt.title(
            f"Parameter recovery: "
            f"{parameter_names[j]}"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                f"{parameter_names[j]}.png",
            ),
            dpi=200,
        )

        plt.close()


# ==========================================================
# TRAINING HISTORY
# ==========================================================

def plot_training_history(
    history,
    output_dir="figures",
):
    """
    Plot training and validation losses.

    Expected keys:
        train_loss
        valid_loss

    Optional:
        train_parameter_loss
        valid_parameter_loss
        train_trajectory_loss
        valid_trajectory_loss
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Total loss
    # ------------------------------------------------------

    if (
        "train_loss" in history
        and "valid_loss" in history
    ):

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            history["train_loss"],
            label="Train",
        )

        plt.plot(
            history["valid_loss"],
            label="Validation",
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Loss"
        )

        plt.title(
            "Training history"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "training.png",
            ),
            dpi=200,
        )

        plt.close()

    # ------------------------------------------------------
    # Parameter loss
    # ------------------------------------------------------

    if (
        "train_parameter_loss" in history
        and
        "valid_parameter_loss" in history
    ):

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            history[
                "train_parameter_loss"
            ],
            label="Train",
        )

        plt.plot(
            history[
                "valid_parameter_loss"
            ],
            label="Validation",
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Parameter MSE"
        )

        plt.title(
            "Parameter loss"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "parameter_loss.png",
            ),
            dpi=200,
        )

        plt.close()

    # ------------------------------------------------------
    # Trajectory loss
    # ------------------------------------------------------

    if (
        "train_trajectory_loss" in history
        and
        "valid_trajectory_loss" in history
    ):

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            history[
                "train_trajectory_loss"
            ],
            label="Train",
        )

        plt.plot(
            history[
                "valid_trajectory_loss"
            ],
            label="Validation",
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Trajectory MSE"
        )

        plt.title(
            "Trajectory loss"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "trajectory_loss.png",
            ),
            dpi=200,
        )

        plt.close()


# ==========================================================
# TRAJECTORY
# ==========================================================

def plot_trajectory(
    t,
    trajectory_true,
    trajectory_pred,
    sample_idx=0,
    trajectory_mask=None,
    state_names=None,
    output_path="figures/trajectory.png",
):
    """
    Plot ODE trajectory reconstruction.

    The predicted trajectory is plotted as a continuous
    line.

    The target trajectory is plotted only where the target
    is actually observed. Missing values are represented
    by NaN so matplotlib leaves gaps instead of connecting
    through artificial zero-filled values.

    Parameters
    ----------
    t:
        Either:

        [T]
            shared time vector

        or:

        [B, T]
            sample-specific padded time vectors

    trajectory_true:
        Tensor [B, T, D]

    trajectory_pred:
        Tensor [B, T, D]

    trajectory_mask:
        Tensor [B, T, D], optional.
        1 = observed
        0 = missing

    sample_idx:
        Which sample to plot.

    state_names:
        Names such as ["Z", "Y"].

    output_path:
        Output PNG path.
    """

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    # ------------------------------------------------------
    # Check sample index
    # ------------------------------------------------------

    if sample_idx < 0:
        raise ValueError(
            "sample_idx must be non-negative."
        )

    if sample_idx >= trajectory_true.shape[0]:
        raise IndexError(
            f"sample_idx={sample_idx} is out of range. "
            f"Number of samples={trajectory_true.shape[0]}."
        )

    # ------------------------------------------------------
    # Convert trajectory tensors to numpy
    # ------------------------------------------------------

    trajectory_true_sample = (
        trajectory_true[
            sample_idx
        ]
        .detach()
        .cpu()
        .numpy()
    )

    trajectory_pred_sample = (
        trajectory_pred[
            sample_idx
        ]
        .detach()
        .cpu()
        .numpy()
    )

    # ------------------------------------------------------
    # Select the correct time vector
    # ------------------------------------------------------
    #
    # Old format:
    #     t.shape == [T]
    #
    # New format:
    #     t.shape == [B, T]
    #
    # The current trainer saves sample-specific
    # time vectors, so we must use t[sample_idx].
    # ------------------------------------------------------

    t = (
        t.detach()
        .cpu()
    )

    if t.ndim == 1:

        t_sample = (
            t
            .numpy()
        )

    elif t.ndim == 2:

        if sample_idx >= t.shape[0]:
            raise IndexError(
                f"sample_idx={sample_idx} is out of range "
                f"for time tensor with shape {tuple(t.shape)}."
            )

        t_sample = (
            t[
                sample_idx
            ]
            .numpy()
        )

    else:

        raise ValueError(
            "Time tensor must have shape [T] "
            "or [B, T]. "
            f"Got {tuple(t.shape)}."
        )

    # ------------------------------------------------------
    # Check true/predicted trajectory shapes
    # ------------------------------------------------------

    if trajectory_true_sample.shape != (
        trajectory_pred_sample.shape
    ):
        raise ValueError(
            "True and predicted trajectories "
            "must have the same shape. "
            f"Got {trajectory_true_sample.shape} "
            f"and {trajectory_pred_sample.shape}."
        )

    # ------------------------------------------------------
    # Remove padding from the time vector
    # ------------------------------------------------------
    #
    # The current trainer pads shorter trajectories
    # with NaN.
    # ------------------------------------------------------

    valid_time = np.isfinite(
        t_sample
    )

    t_sample = (
        t_sample[
            valid_time
        ]
    )

    trajectory_true_sample = (
        trajectory_true_sample[
            valid_time
        ]
    )

    trajectory_pred_sample = (
        trajectory_pred_sample[
            valid_time
        ]
    )

    # ------------------------------------------------------
    # Time/trajectory length check
    # ------------------------------------------------------

    if len(t_sample) != (
        trajectory_true_sample.shape[0]
    ):
        raise ValueError(
            "Time vector length does not match "
            "trajectory length. "
            f"t={len(t_sample)}, "
            f"trajectory={trajectory_true_sample.shape[0]}."
        )

    # ------------------------------------------------------
    # State dimension
    # ------------------------------------------------------

    state_dim = (
        trajectory_true_sample.shape[-1]
    )

    if state_names is None:
        state_names = [
            f"State {i}"
            for i in range(state_dim)
        ]

    if len(state_names) != state_dim:
        raise ValueError(
            "Number of state names does not "
            "match state dimension."
        )

    # ------------------------------------------------------
    # Mask
    # ------------------------------------------------------

    if trajectory_mask is not None:

        mask_sample = (
            trajectory_mask[
                sample_idx
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )

        original_trajectory_shape = (
            trajectory_true[
                sample_idx
            ]
            .shape
        )

        if mask_sample.shape != (
            original_trajectory_shape
        ):
            raise ValueError(
                "trajectory_mask must have the "
                "same shape as trajectory_true."
            )

        # Remove the same padded positions
        mask_sample = (
            mask_sample[
                valid_time
            ]
        )

    else:

        # If no mask is supplied, assume all target
        # values are valid.
        mask_sample = np.ones(
            trajectory_true_sample.shape,
            dtype=bool,
        )

    # ------------------------------------------------------
    # Check mask after padding removal
    # ------------------------------------------------------

    if mask_sample.shape != (
        trajectory_true_sample.shape
    ):
        raise ValueError(
            "Mask shape does not match the "
            "unpadded trajectory shape. "
            f"Mask={mask_sample.shape}, "
            f"trajectory={trajectory_true_sample.shape}."
        )

    # ------------------------------------------------------
    # One figure containing all states
    # ------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    for state_idx in range(
        state_dim
    ):

        # --------------------------------------------------
        # True values
        # --------------------------------------------------

        true_values = (
            trajectory_true_sample[
                :,
                state_idx,
            ]
            .astype(float)
        )

        # --------------------------------------------------
        # Predicted values
        # --------------------------------------------------

        pred_values = (
            trajectory_pred_sample[
                :,
                state_idx,
            ]
            .astype(float)
        )

        # --------------------------------------------------
        # State-specific observation mask
        # --------------------------------------------------

        valid_mask = (
            mask_sample[
                :,
                state_idx,
            ]
        )

        # --------------------------------------------------
        # Target
        #
        # Missing target values become NaN.
        # Matplotlib automatically breaks the line.
        # --------------------------------------------------

        true_plot = np.where(
            valid_mask,
            true_values,
            np.nan,
        )

        plt.plot(
            t_sample,
            true_plot,
            linewidth=1.8,
            label=(
                f"True "
                f"{state_names[state_idx]}"
            ),
        )

        # --------------------------------------------------
        # Emphasize actual observations
        # --------------------------------------------------

        valid_indices = np.where(
            valid_mask
        )[0]

        if len(valid_indices) > 0:

            plt.scatter(
                t_sample[
                    valid_indices
                ],
                true_values[
                    valid_indices
                ],
                s=18,
            )

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        plt.plot(
            t_sample,
            pred_values,
            linestyle="--",
            linewidth=2.0,
            label=(
                f"Predicted "
                f"{state_names[state_idx]}"
            ),
        )

    # ------------------------------------------------------
    # Initial state time
    # ------------------------------------------------------

    plt.axvline(
        0.0,
        linestyle=":",
        linewidth=1.2,
        label="Initial state t=0",
    )

    # ------------------------------------------------------
    # Labels
    # ------------------------------------------------------

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "State value"
    )

    plt.title(
        "ODE trajectory reconstruction "
        f"(sample {sample_idx})"
    )

    plt.legend(
        ncol=2
    )

    plt.tight_layout()

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ==========================================================
# MULTIPLE TRAJECTORY EXAMPLES
# ==========================================================

def plot_trajectory_examples(
    t,
    trajectory_true,
    trajectory_pred,
    trajectory_mask=None,
    state_names=None,
    output_dir="figures/trajectories",
    num_examples=5,
):
    """
    Save trajectory plots for several test samples.

    Parameters
    ----------
    t:
        [T] or [B, T]

    trajectory_true:
        [B, T, D]

    trajectory_pred:
        [B, T, D]

    trajectory_mask:
        [B, T, D], optional

    state_names:
        State names such as ["Z", "Y"]

    output_dir:
        Output directory

    num_examples:
        Number of samples to plot
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    n_samples = min(
        num_examples,
        trajectory_true.shape[0],
    )

    for sample_idx in range(
        n_samples
    ):

        output_path = os.path.join(
            output_dir,
            f"sample_{sample_idx}.png",
        )

        plot_trajectory(
            t=t,
            trajectory_true=trajectory_true,
            trajectory_pred=trajectory_pred,
            sample_idx=sample_idx,
            trajectory_mask=trajectory_mask,
            state_names=state_names,
            output_path=output_path,
        )
