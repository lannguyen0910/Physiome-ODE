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
    connect_true_observations=True,
    interpolate_true=False,
):
    """
    Plot ODE trajectory reconstruction.

    The predicted trajectory is plotted as a continuous line.

    Ground-truth observations are plotted as scatter points. For a
    cleaner visualization of irregular / sparse data, the observed
    points can also be connected directly (without inventing values at
    missing timestamps). Optionally, a linear interpolation can be
    drawn between observed points for visualization only.

    IMPORTANT
    ---------
    Connecting observed points does NOT mean the system was observed
    continuously between them. It is only a visual connection between
    successive valid measurements.

    Parameters
    ----------
    t:
        Either [T] or [B, T].

    trajectory_true:
        Tensor [B, T, D]. Ground-truth target values.

    trajectory_pred:
        Tensor [B, T, D]. ODE-predicted trajectory.

    trajectory_mask:
        Tensor [B, T, D], optional.
        1 = observed / valid target, 0 = missing / invalid.

    connect_true_observations:
        If True, connect the valid observed true points with a solid
        line. This removes the visual gaps caused by missing entries
        without fabricating missing values.

    interpolate_true:
        If True, linearly interpolate the observed true values over
        the valid time range for visualization. This should be used
        only when explicitly desired because the interpolated values
        are not actual measurements.
    """

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if sample_idx < 0:
        raise ValueError("sample_idx must be non-negative.")

    if sample_idx >= trajectory_true.shape[0]:
        raise IndexError(
            f"sample_idx={sample_idx} is out of range. "
            f"Number of samples={trajectory_true.shape[0]}."
        )

    # ------------------------------------------------------
    # Convert trajectories to numpy
    # ------------------------------------------------------

    trajectory_true_sample = (
        trajectory_true[sample_idx]
        .detach()
        .cpu()
        .numpy()
    )

    trajectory_pred_sample = (
        trajectory_pred[sample_idx]
        .detach()
        .cpu()
        .numpy()
    )

    # ------------------------------------------------------
    # Select the correct time vector
    # ------------------------------------------------------

    t_tensor = t.detach().cpu()

    if t_tensor.ndim == 1:
        t_sample = t_tensor.numpy()

    elif t_tensor.ndim == 2:
        if sample_idx >= t_tensor.shape[0]:
            raise IndexError(
                f"sample_idx={sample_idx} is out of range "
                f"for time tensor with shape {tuple(t_tensor.shape)}."
            )
        t_sample = t_tensor[sample_idx].numpy()

    else:
        raise ValueError(
            "Time tensor must have shape [T] or [B, T]. "
            f"Got {tuple(t_tensor.shape)}."
        )

    # ------------------------------------------------------
    # Prepare / validate mask
    # ------------------------------------------------------

    if trajectory_mask is not None:
        mask_sample = (
            trajectory_mask[sample_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )

        if mask_sample.shape != trajectory_true_sample.shape:
            raise ValueError(
                "trajectory_mask must have the same shape as "
                "trajectory_true for the selected sample. "
                f"Got mask={mask_sample.shape}, "
                f"trajectory={trajectory_true_sample.shape}."
            )

    else:
        # Without a mask, treat finite true values as valid observations.
        mask_sample = np.isfinite(trajectory_true_sample)

    # ------------------------------------------------------
    # Remove padded / invalid event slots
    # ------------------------------------------------------
    # A padded event is typically represented by t=0 and all mask values
    # equal to zero, or by NaN time in the trainer output.

    finite_time = np.isfinite(t_sample)
    event_has_target = np.any(mask_sample, axis=-1)
    event_valid = finite_time & event_has_target

    if not np.any(event_valid):
        raise ValueError(
            "No valid observation events remain after removing padding."
        )

    t_sample = t_sample[event_valid]
    trajectory_true_sample = trajectory_true_sample[event_valid]
    trajectory_pred_sample = trajectory_pred_sample[event_valid]
    mask_sample = mask_sample[event_valid]

    # ------------------------------------------------------
    # Ensure time is ordered for visualization
    # ------------------------------------------------------

    order = np.argsort(t_sample, kind="stable")

    t_sample = t_sample[order]
    trajectory_true_sample = trajectory_true_sample[order]
    trajectory_pred_sample = trajectory_pred_sample[order]
    mask_sample = mask_sample[order]

    # ------------------------------------------------------
    # State dimension
    # ------------------------------------------------------

    state_dim = trajectory_true_sample.shape[-1]

    if state_names is None:
        state_names = [f"State {i}" for i in range(state_dim)]

    if len(state_names) != state_dim:
        raise ValueError(
            "Number of state names does not match state dimension."
        )

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for state_idx in range(state_dim):

        true_values = trajectory_true_sample[:, state_idx].astype(float)
        pred_values = trajectory_pred_sample[:, state_idx].astype(float)
        valid_mask = mask_sample[:, state_idx]

        # Additional finite-value filtering for the selected state.
        valid_true = valid_mask & np.isfinite(true_values)

        # --------------------------------------------------
        # Ground truth: actual observed points
        # --------------------------------------------------

        t_true = t_sample[valid_true]
        y_true = true_values[valid_true]

        if len(t_true) > 0:
            plt.scatter(
                t_true,
                y_true,
                s=22,
                label=f"True {state_names[state_idx]} observations",
            )

        # --------------------------------------------------
        # Ground truth: connect observed points
        # --------------------------------------------------
        # This is the requested seamless visual. It connects only the
        # measurements that actually exist; no missing values are
        # invented.

        if connect_true_observations and len(t_true) >= 2:
            plt.plot(
                t_true,
                y_true,
                linewidth=1.8,
                label=f"True {state_names[state_idx]} connected",
            )

        # --------------------------------------------------
        # Optional linear interpolation for visualization only
        # --------------------------------------------------

        if interpolate_true and len(t_true) >= 2:
            finite_pred_time = np.isfinite(t_sample)
            interp_time = t_sample[finite_pred_time]

            # Only interpolate inside the range of observed values.
            inside = (
                (interp_time >= t_true.min())
                & (interp_time <= t_true.max())
            )
            interp_time = interp_time[inside]

            if len(interp_time) >= 2:
                interp_values = np.interp(
                    interp_time,
                    t_true,
                    y_true,
                )

                plt.plot(
                    interp_time,
                    interp_values,
                    linewidth=1.2,
                    linestyle=":",
                    alpha=0.8,
                    label=(
                        f"True {state_names[state_idx]} "
                        "linear interpolation"
                    ),
                )

        # --------------------------------------------------
        # ODE prediction: continuous reconstruction
        # --------------------------------------------------

        finite_pred = np.isfinite(pred_values)

        if np.count_nonzero(finite_pred) >= 2:
            plt.plot(
                t_sample[finite_pred],
                pred_values[finite_pred],
                linestyle="--",
                linewidth=2.0,
                label=f"Predicted {state_names[state_idx]}",
            )

    # ------------------------------------------------------
    # Initial reference time
    # ------------------------------------------------------

    if t_sample.min() <= 0.0 <= t_sample.max():
        plt.axvline(
            0.0,
            linestyle=":",
            linewidth=1.2,
            label="Initial state t=0",
        )

    plt.xlabel("Time")
    plt.ylabel("State value")
    plt.title(
        "ODE trajectory reconstruction "
        f"(sample {sample_idx})"
    )

    # Remove duplicate legend labels while preserving order.
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    plt.legend(
        unique.values(),
        unique.keys(),
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
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
