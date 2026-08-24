import os
import torch


class Trainer:
    def __init__(
        self,
        model,
        ode_solver,
        train_loader,
        valid_loader,
        test_loader,
        loss_fn,
        optimizer,
        device,
        epochs,
        patience,
        gradient_clip,
        checkpoint_path,
    ):
        self.model = model
        self.ode_solver = ode_solver
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.gradient_clip = gradient_clip
        self.checkpoint_path = checkpoint_path

        self.history = {
            "train_loss": [],
            "valid_loss": [],
            "train_parameter_loss": [],
            "valid_parameter_loss": [],
            "train_trajectory_loss": [],
            "valid_trajectory_loss": [],
        }

    def _prepare_batch(
        self,
        batch,
    ):
        T = batch['T'].to(self.device)
        X = batch['X'].to(self.device)
        M = batch['M'].to(self.device)
        TY = batch['TY'].to(self.device)
        MY = batch['MY'].to(self.device)
        Y = batch['Y'].to(self.device)
        theta_true = batch['theta'].to(self.device)
        y0_true = batch['y0'].to(self.device)

        return (
            T,
            X,
            M,
            TY,
            Y,
            MY,
            theta_true,
            y0_true,
        )

    def _forward(
        self,
        batch,
    ):
        (
            T,
            X,
            M,
            TY,
            Y,
            MY,
            theta_true,
            y0_true,
        ) = self._prepare_batch(batch)

        theta_pred, y0_pred = self.model(
            T=T,
            X=X,
            M=M,
        )

        # Construct time grid for trajectory reconstruction:
        # observation + forecast

        if TY.shape[1] > 0:
            t = torch.cat(
                [
                    T[0],
                    TY[0],
                ],
                dim=0,
            )
        else:
            t = T[0]

        trajectory_pred = self.ode_solver(
            y0=y0_pred,
            t=t,
            theta=theta_pred,
        )

        # X = observed part
        # Y = forecasting part
        trajectory_true = torch.cat(
            [
                torch.nan_to_num(
                    X,
                    nan=0.0,
                ),
                torch.nan_to_num(
                    Y,
                    nan=0.0,
                ),
            ],
            dim=1,
        )

        return {
            "T": T,
            "X": X,
            "M": M,
            "TY": TY,
            "Y": Y,
            "MY": MY,
            "theta_true": theta_true,
            "y0_true": y0_true,
            "theta_pred": theta_pred,
            "y0_pred": y0_pred,
            "trajectory_true": trajectory_true,
            "trajectory_pred": trajectory_pred,
            "t": t,
        }

    def _compute_loss(
        self,
        outputs,
    ):
        return self.loss_fn(
            theta_pred=outputs["theta_pred"],
            theta_true=outputs["theta_true"],
            trajectory_pred=outputs["trajectory_pred"],
            trajectory_true=outputs["trajectory_true"],
        )

    def train_epoch(self):
        self.model.train()

        total_loss = 0.0
        total_parameter_loss = 0.0
        total_trajectory_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            # self.optimizer.zero_grad(set_to_none=True)

            outputs = self._forward(batch)
            losses = self._compute_loss(outputs)
            loss = losses["total"]

            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became NaN or Inf.")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip,
            )

            self.optimizer.step()

            total_loss += loss.item()
            total_parameter_loss += losses["parameter"].item()
            total_trajectory_loss += losses["trajectory"].item()

            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "parameter": (
                total_parameter_loss
                / num_batches
            ),
            "trajectory": (
                total_trajectory_loss
                / num_batches
            ),
        }

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_loss = 0.0
        total_parameter_loss = 0.0
        total_trajectory_loss = 0.0

        num_batches = 0

        for batch in self.valid_loader:
            outputs = self._forward(batch)
            losses = self._compute_loss(outputs)

            total_loss += losses["total"].item()
            total_parameter_loss += losses["parameter"].item()
            total_trajectory_loss += losses["trajectory"].item()

            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "parameter": (
                total_parameter_loss
                / num_batches
            ),
            "trajectory": (
                total_trajectory_loss
                / num_batches
            ),
        }

    def fit(self):
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            train_result = self.train_epoch()
            valid_result = self.validate()

            self.history[
                "train_loss"
            ].append(
                train_result["loss"]
            )

            self.history[
                "valid_loss"
            ].append(
                valid_result["loss"]
            )

            self.history[
                "train_parameter_loss"
            ].append(
                train_result["parameter"]
            )

            self.history[
                "valid_parameter_loss"
            ].append(
                valid_result["parameter"]
            )

            self.history[
                "train_trajectory_loss"
            ].append(
                train_result["trajectory"]
            )

            self.history[
                "valid_trajectory_loss"
            ].append(
                valid_result["trajectory"]
            )

            print(
                f"Epoch {epoch:04d} | "
                f"Train {train_result['loss']:.6f} | "
                f"Val {valid_result['loss']:.6f} | "
                f"Param {valid_result['parameter']:.6f} | "
                f"Traj {valid_result['trajectory']:.6f}"
            )

            # Save best model
            if valid_result["loss"] < best_val_loss:
                best_val_loss = valid_result["loss"]
                patience_counter = 0

                os.makedirs(
                    os.path.dirname(self.checkpoint_path),
                    exist_ok=True,
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict":
                            self.model.state_dict(),
                        "optimizer_state_dict":
                            self.optimizer.state_dict(),
                        "val_loss":
                            best_val_loss,
                    },
                    self.checkpoint_path,
                )

            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping.")
                break

        # Load best model
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])

        return self.history

    @torch.no_grad()
    def evaluate_test(self):
        self.model.eval()

        theta_true_list = []
        theta_pred_list = []

        y0_true_list = []
        y0_pred_list = []

        trajectory_true_list = []
        trajectory_pred_list = []

        for batch in self.test_loader:
            outputs = self._forward(batch)

            theta_true_list.append(
                outputs[
                    "theta_true"
                ].cpu()
            )

            theta_pred_list.append(
                outputs[
                    "theta_pred"
                ].cpu()
            )

            y0_true_list.append(
                outputs[
                    "y0_true"
                ].cpu()
            )

            y0_pred_list.append(
                outputs[
                    "y0_pred"
                ].cpu()
            )

            trajectory_true_list.append(
                outputs[
                    "trajectory_true"
                ].cpu()
            )

            trajectory_pred_list.append(
                outputs[
                    "trajectory_pred"
                ].cpu()
            )

        theta_true = torch.cat(
            theta_true_list,
            dim=0,
        )

        theta_pred = torch.cat(
            theta_pred_list,
            dim=0,
        )

        y0_true = torch.cat(
            y0_true_list,
            dim=0,
        )

        y0_pred = torch.cat(
            y0_pred_list,
            dim=0,
        )

        trajectory_true = torch.cat(
            trajectory_true_list,
            dim=0,
        )

        trajectory_pred = torch.cat(
            trajectory_pred_list,
            dim=0,
        )

        # Parameter metrics
        parameter_error = theta_pred - theta_true

        parameter_mae = (
            parameter_error
            .abs()
            .mean()
            .item()
        )

        parameter_rmse = torch.sqrt(
            parameter_error.pow(2)
            .mean()
        ).item()

        # Initial state metrics
        y0_error = y0_pred - y0_true

        y0_mae = (
            y0_error
            .abs()
            .mean()
            .item()
        )

        y0_rmse = torch.sqrt(
            y0_error.pow(2)
            .mean()
        ).item()

        # Trajectory metrics
        trajectory_error = trajectory_pred - trajectory_true

        trajectory_mae = (
            trajectory_error
            .abs()
            .mean()
            .item()
        )

        trajectory_rmse = torch.sqrt(
            trajectory_error.pow(2)
            .mean()
        ).item()

        results = {
            "theta_true": theta_true,
            "theta_pred": theta_pred,
            "y0_true": y0_true,
            "y0_pred": y0_pred,
            "trajectory_true": trajectory_true,
            "trajectory_pred": trajectory_pred,
            "parameter_mae": parameter_mae,
            "parameter_rmse": parameter_rmse,
            "y0_mae": y0_mae,
            "y0_rmse": y0_rmse,
            "trajectory_mae": trajectory_mae,
            "trajectory_rmse": trajectory_rmse,
        }

        print("\n================ TEST RESULTS ================")

        print(f"Parameter MAE  : {parameter_mae:.6f}")
        print(f"Parameter RMSE : {parameter_rmse:.6f}")
        print(f"Initial y0 MAE : {y0_mae:.6f}")
        print(f"Initial y0 RMSE: {y0_rmse:.6f}")
        print(f"Trajectory MAE : {trajectory_mae:.6f}")
        print(f"Trajectory RMSE: {trajectory_rmse:.6f}")

        print("==============================================\n")

        return results
