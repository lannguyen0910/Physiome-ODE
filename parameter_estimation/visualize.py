import os

import matplotlib.pyplot as plt


def plot_parameter_recovery(
    theta_true,
    theta_pred,
    parameter_names,
    output_dir="figures",
):
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    theta_true = theta_true.numpy()
    theta_pred = theta_pred.numpy()

    n_parameters = theta_true.shape[1]

    for j in range(n_parameters):
        plt.figure(figsize=(6, 6))

        plt.scatter(
            theta_true[:, j],
            theta_pred[:, j],
            alpha=0.6
        )

        minimum = min(
            theta_true[:, j].min(),
            theta_pred[:, j].min()
        )

        maximum = max(
            theta_true[:, j].max(),
            theta_pred[:, j].max()
        )

        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--"
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
                f"{parameter_names[j]}.png"
            )
        )

        plt.close()


def plot_training_history(
    history,
    output_path="figures/training.png",
):
    plt.figure(figsize=(8, 5))

    plt.plot(
        history["train"],
        label="Train"
    )

    plt.plot(
        history["valid"],
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path
    )
    plt.close()
