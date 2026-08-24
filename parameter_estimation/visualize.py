import os
import matplotlib.pyplot as plt


def plot_parameter_recovery(
    theta_true,
    theta_pred,
    parameter_names,
    output_dir="figures/parameters",
):
    os.makedirs(output_dir, exist_ok=True)

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

    n_parameters = theta_true.shape[1]

    for j in range(n_parameters):

        plt.figure(figsize=(6, 6))

        plt.scatter(
            theta_true[:, j],
            theta_pred[:, j],
            alpha=0.6,
        )

        minimum = min(
            theta_true[:, j].min(),
            theta_pred[:, j].min(),
        )

        maximum = max(
            theta_true[:, j].max(),
            theta_pred[:, j].max(),
        )

        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
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

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                f"{parameter_names[j]}.png",
            ),
            dpi=150,
        )

        plt.close()


def plot_training_history(
    history,
    output_path="figures/training.png",
):
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["train_loss"],
        label="Train",
    )

    plt.plot(
        history["valid_loss"],
        label="Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training history")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def plot_trajectory(
    t,
    trajectory_true,
    trajectory_pred,
    sample_idx=0,
    output_path="figures/trajectory.png",
):
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    # t = t.detach().cpu().numpy()
    true = (
        trajectory_true[sample_idx]
        .detach()
        .cpu()
        .numpy()
    )

    pred = trajectory_pred[sample_idx].detach().cpu().numpy()
    plt.figure(figsize=(10, 5))

    # Z
    plt.plot(
        t,
        true[:, 0],
        label="True Z",
    )

    plt.plot(
        t,
        pred[:, 0],
        linestyle="--",
        label="Predicted Z",
    )

    # Y
    plt.plot(
        t,
        true[:, 1],
        label="True Y",
    )

    plt.plot(
        t,
        pred[:, 1],
        linestyle="--",
        label="Predicted Y",
    )

    plt.xlabel("Time")
    plt.ylabel("State value")

    plt.title(
        f"ODE trajectory reconstruction "
        f"(sample {sample_idx})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
    )
    plt.close()
