import os

import random

import numpy as np

import torch

from config import Config

from dataset import PhysiomeDataModule

from model import ParameterEstimator, ParameterEncoder

from ode import PhysiomeODE, RK4Solver

from loss import ParameterEstimationLoss

from metrics import ParameterMetrics, TrajectoryMetrics, JGDMetrics

from eda import DatasetEDA

from trainer import Trainer

from visualize import (
    plot_training_history,
    plot_parameter_recovery,
    plot_trajectory,
)


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


def main():

    config = Config()

    set_seed(config.seed)

    device = config.device

    print(f"Device: {device}")

    # ==========================================================
    # CREATE OUTPUT DIRECTORIES
    # ==========================================================

    os.makedirs(

        os.path.dirname(
            config.checkpoint_path
        ),

        exist_ok=True,

    )

    os.makedirs(

        os.path.dirname(
            config.results_path
        ),

        exist_ok=True,

    )

    os.makedirs(

        config.figures_path,

        exist_ok=True,

    )

    # ==========================================================
    # LOAD DATA
    # ==========================================================

    data = PhysiomeDataModule(

        data_path=config.data_path,

        batch_size=config.batch_size,

        num_workers=config.num_workers,

    )

    data.setup()

    train_loader = data.train_loader()

    valid_loader = data.valid_loader()

    test_loader = data.test_loader()

    # ==========================================================
    # DATASET INFORMATION
    # ==========================================================

    batch = next(iter(train_loader))

    print("\nDataset information:")

    print("T:", batch["T"].shape)

    print("X:", batch["X"].shape)

    print("M:", batch["M"].shape)

    print("TY:", batch["TY"].shape)

    print("MY:", batch["MY"].shape)

    print("Y:", batch["Y"].shape)

    print("theta:", batch["theta"].shape)

    print("y0:", batch["y0"].shape)

    # ==========================================================
    # EXPLORATORY DATA ANALYSIS
    # ==========================================================

    print("\n" + "=" * 60)

    print("RUNNING EXPLORATORY DATA ANALYSIS")

    print("=" * 60)

    dataset_eda = DatasetEDA(

        output_dir=os.path.join(
            config.figures_path,
            "eda",
        )

    )

    print("\nRunning EDA on TRAIN dataset...")

    dataset_eda.analyze(

        dataset=data.train_dataset,

        dataset_name="train",

    )

    print("\nRunning EDA on VALIDATION dataset...")

    dataset_eda.analyze(

        dataset=data.valid_dataset,

        dataset_name="validation",

    )

    print("\nRunning EDA on TEST dataset...")

    dataset_eda.analyze(

        dataset=data.test_dataset,

        dataset_name="test",

    )

    print("\nEDA completed.")

    print("=" * 60)

    # ==========================================================
    # MODEL
    # ==========================================================

    encoder = ParameterEncoder(

        state_dim=config.state_dim,

        parameter_dim=config.parameter_dim,

        hidden_dim=config.hidden_dim,

        num_layers=config.num_gru_layers,

        dropout=config.drop_out,

    ).to(device)

    model = ParameterEstimator(
        encoder
    )

    num_parameters = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    print(

        f"\nTrainable parameters: "

        f"{num_parameters:,}"

    )

    # ==========================================================
    # ODE AND SOLVER
    # ==========================================================

    physiome_ode = PhysiomeODE()

    ode_solver = RK4Solver(

        physiome_ode.dynamics

    )

    # ==========================================================
    # LOSS FUNCTION
    # ==========================================================

    loss_fn = ParameterEstimationLoss(

        parameter_weight=(
            config.parameter_loss_weight
        ),

        trajectory_weight=(
            config.trajectory_loss_weight
        ),

    )

    # ==========================================================
    # OPTIMIZER
    # ==========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config.learning_rate,

        weight_decay=config.weight_decay,

    )

    # ==========================================================
    # TRAINER
    # ==========================================================

    trainer = Trainer(

        model=model,

        ode_solver=ode_solver,

        train_loader=train_loader,

        valid_loader=valid_loader,

        test_loader=test_loader,

        loss_fn=loss_fn,

        optimizer=optimizer,

        device=device,

        epochs=config.epochs,

        patience=config.patience,

        gradient_clip=config.gradient_clip,

        checkpoint_path=config.checkpoint_path,


        # ======================================================
        # METRICS
        # ======================================================

        parameter_metrics={

            "mae":
                ParameterMetrics.mae,

            "rmse":
                ParameterMetrics.rmse,

            "relative_error":
                ParameterMetrics.relative_error,

            "log_mae":
                ParameterMetrics.log_mae,

            "overall_accuracy":
                ParameterMetrics.overall_parameter_accuracy,

        },


        trajectory_metrics={

            "mae":
                TrajectoryMetrics.mae,

            "rmse":
                TrajectoryMetrics.rmse,

            "normalized_rmse":
                TrajectoryMetrics.normalized_rmse,

        },


        jgd_metrics={

            "distribution_error":
                JGDMetrics.trajectory_distribution_error,

            "temporal_dynamics_error":
                JGDMetrics.temporal_dynamics_error,

            "jgd_score":
                JGDMetrics.jgd_score,

        },

    )

    # ==========================================================
    # TRAINING
    # ==========================================================

    print("\nStarting training...")

    history = trainer.fit()

    print("Training finished.")

    # ==========================================================
    # TRAINING HISTORY PLOT
    # ==========================================================

    plot_training_history(

        history,

        output_path=os.path.join(

            config.figures_path,

            "training.png",

        ),

    )

    # ==========================================================
    # PARAMETER AND TRAJECTORY VISUALIZATION
    # ==========================================================

    trainer.model.eval()

    with torch.no_grad():

        batch = next(iter(test_loader))

        outputs = trainer._forward(batch)

        # ------------------------------------------------------
        # PARAMETER RECOVERY
        # ------------------------------------------------------

        plot_parameter_recovery(

            theta_true=outputs["theta_true"],

            theta_pred=outputs["theta_pred"],

            parameter_names=config.parameter_names,

            output_dir=os.path.join(

                config.figures_path,

                "parameters",

            ),

        )

        # ------------------------------------------------------
        # TRAJECTORY
        # ------------------------------------------------------

        plot_trajectory(

            t=outputs["t"],

            trajectory_true=(
                outputs["trajectory_true"]
            ),

            trajectory_pred=(
                outputs["trajectory_pred"]
            ),

            sample_idx=0,

            output_path=os.path.join(

                config.figures_path,

                "trajectory_sample_0.png",

            ),

        )

    # ==========================================================
    # TEST EVALUATION
    # ==========================================================

    print("\nEvaluating test set...")

    results = trainer.evaluate_test()

    torch.save(

        results,

        config.results_path,

    )

    torch.save(

        history,

        config.history_path,

    )

    # ==========================================================
    # PRINT FINAL RESULTS
    # ==========================================================

    print("\n" + "=" * 60)

    print("FINAL TEST RESULTS")

    print("=" * 60)

    for key, value in results.items():

        if torch.is_tensor(value):

            if value.numel() == 1:

                print(

                    f"{key}: "

                    f"{value.item():.6f}"

                )

            else:

                print(

                    f"{key}: "

                    f"{value.detach().cpu().numpy()}"

                )

        else:

            print(

                f"{key}: {value}"

            )

    print("=" * 60)

    print("\nExperiment completed.")


if __name__ == "__main__":

    main()
