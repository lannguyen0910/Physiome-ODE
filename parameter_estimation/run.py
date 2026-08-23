import os
import random

import numpy as np
import torch

from config import Config
from dataset import PhysiomeDataModule
from model import ParameterEstimator
from ode import PhysiomeODE, RK4Solver
from loss import ParameterEstimationLoss
from trainer import Trainer
from visualize import plot_training_history, plot_parameter_recovery, plot_trajectory


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

    data = PhysiomeDataModule(
        data_path=config.data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    data.setup()

    train_loader = data.train_loader()
    valid_loader = data.valid_loader()
    test_loader = data.test_loader()

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

    model = ParameterEstimator(
        state_dim=config.state_dim,
        parameter_dim=config.parameter_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_gru_layers,
        dropout=config.drop_out
    ).to(device)

    num_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{num_parameters:,}"
    )

    physiome_ode = PhysiomeODE()
    ode_solver = RK4Solver(physiome_ode.dynamics)

    loss_fn = ParameterEstimationLoss(
        parameter_weight=config.parameter_loss_weight,
        trajectory_weight=config.trajectory_loss_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

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
    )

    print("\nStarting training...")
    history = trainer.fit()

    print("Training finished.")

    plot_training_history(
        history,
        output_path="figures/training.png",
    )

    trainer.model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        outputs = trainer._forward(batch)

        plot_parameter_recovery(
            theta_true=outputs["theta_true"],
            theta_pred=outputs["theta_pred"],
            parameter_names=config.ode_parameters,
            output_dir="figures/parameters",
        )

        plot_trajectory(
            t=outputs["T"][0],
            trajectory_true=outputs["trajectory_true"],
            trajectory_pred=outputs["trajectory_pred"],
            sample_idx=0,
            output_path="figures/trajectory_sample_0.png",
        )

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

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
