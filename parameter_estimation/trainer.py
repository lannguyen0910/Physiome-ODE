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


        # ======================================================
        # NEW: METRICS
        # ======================================================

        parameter_metrics=None,

        trajectory_metrics=None,

        jgd_metrics=None,

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

        # ======================================================
        # STORE METRICS
        # ======================================================

        self.parameter_metrics = (

            parameter_metrics

            if parameter_metrics is not None

            else {}

        )

        self.trajectory_metrics = (

            trajectory_metrics

            if trajectory_metrics is not None

            else {}

        )

        self.jgd_metrics = (

            jgd_metrics

            if jgd_metrics is not None

            else {}

        )

        self.model.to(
            self.device
        )

    # ==========================================================
    # FORWARD
    # ==========================================================

    def _forward(self, batch):

        T = batch["T"].to(
            self.device
        )

        X = batch["X"].to(
            self.device
        )

        M = batch["M"].to(
            self.device
        )

        TY = batch["TY"].to(
            self.device
        )

        Y = batch["Y"].to(
            self.device
        )

        theta_true = batch["theta"].to(
            self.device
        )

        # ======================================================
        # MODEL PREDICTION
        # ======================================================

        theta_pred, y0_pred = self.model(

            T,

            X,

            M,

        )

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

        # ======================================================
        # CONCATENATE HISTORY AND FUTURE TRAJECTORY
        # ======================================================

        trajectory_true = torch.cat(

            [

                X,

                Y,

            ],

            dim=1,

        )

        # ======================================================
        # SOLVE ODE
        # ======================================================

        trajectory_pred = self.ode_solver(

            y0=y0_pred,

            t=t,

            theta=theta_pred,

        )

        return {

            "theta_pred": theta_pred,

            "theta_true": theta_true,

            "y0_pred": y0_pred,

            "t": t,

            "trajectory_pred": trajectory_pred,

            "trajectory_true": trajectory_true,

        }

    # ==========================================================
    # COMPUTE METRICS
    # ==========================================================

    def _compute_metrics(

        self,

        theta_pred,

        theta_true,

        trajectory_pred,

        trajectory_true,

    ):

        metrics = {}

        # ======================================================
        # PARAMETER METRICS
        # ======================================================

        for name, metric_fn in self.parameter_metrics.items():

            metrics[f"parameter_{name}"] = (

                metric_fn(

                    theta_pred,

                    theta_true,

                )

            )

        # ======================================================
        # TRAJECTORY METRICS
        # ======================================================

        for name, metric_fn in self.trajectory_metrics.items():

            metrics[f"trajectory_{name}"] = (

                metric_fn(

                    trajectory_pred,

                    trajectory_true,

                )

            )

        # ======================================================
        # JGD METRICS
        # ======================================================

        for name, metric_fn in self.jgd_metrics.items():

            metrics[f"jgd_{name}"] = (

                metric_fn(

                    trajectory_pred,

                    trajectory_true,

                )

            )

        return metrics

    # ==========================================================
    # TRAIN ONE EPOCH
    # ==========================================================

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0

        total_parameter_loss = 0.0

        total_trajectory_loss = 0.0

        metric_totals = {}

        num_batches = 0

        for batch in self.train_loader:

            self.optimizer.zero_grad()

            outputs = self._forward(
                batch
            )

            losses = self.loss_fn(

                theta_pred=(
                    outputs["theta_pred"]
                ),

                theta_true=(
                    outputs["theta_true"]
                ),

                trajectory_pred=(
                    outputs["trajectory_pred"]
                ),

                trajectory_true=(
                    outputs["trajectory_true"]
                ),

            )

            total = losses["total"]

            if not torch.isfinite(total):

                print(

                    "WARNING: Non-finite training loss. "

                    "Skipping batch."

                )

                continue

            total.backward()

            if self.gradient_clip is not None:

                torch.nn.utils.clip_grad_norm_(

                    self.model.parameters(),

                    max_norm=self.gradient_clip,

                )

            self.optimizer.step()

            # ==================================================
            # LOSSES
            # ==================================================

            total_loss += (
                losses["total"]
                .detach()
                .item()
            )

            total_parameter_loss += (
                losses["parameter"]
                .detach()
                .item()
            )

            total_trajectory_loss += (
                losses["trajectory"]
                .detach()
                .item()
            )

            # ==================================================
            # METRICS
            # ==================================================

            with torch.no_grad():

                batch_metrics = (

                    self._compute_metrics(

                        theta_pred=(
                            outputs["theta_pred"]
                        ),

                        theta_true=(
                            outputs["theta_true"]
                        ),

                        trajectory_pred=(
                            outputs["trajectory_pred"]
                        ),

                        trajectory_true=(
                            outputs["trajectory_true"]
                        ),

                    )

                )

                for name, value in batch_metrics.items():

                    if torch.isfinite(value):

                        if name not in metric_totals:

                            metric_totals[name] = 0.0

                        metric_totals[name] += (

                            value.detach().item()

                        )

            num_batches += 1

        if num_batches == 0:

            raise RuntimeError(

                "No valid training batches were processed."

            )

        results = {

            "loss": (
                total_loss / num_batches
            ),

            "parameter_loss": (
                total_parameter_loss / num_batches
            ),

            "trajectory_loss": (
                total_trajectory_loss / num_batches
            ),

        }

        for name, value in metric_totals.items():

            results[name] = (

                value / num_batches

            )

        return results

    # ==========================================================
    # VALIDATION
    # ==========================================================

    def validate(self):

        self.model.eval()

        total_loss = 0.0

        total_parameter_loss = 0.0

        total_trajectory_loss = 0.0

        metric_totals = {}

        all_theta_pred = []

        all_theta_true = []

        all_trajectory_pred = []

        all_trajectory_true = []

        num_batches = 0

        with torch.no_grad():

            for batch in self.valid_loader:

                outputs = self._forward(
                    batch
                )

                losses = self.loss_fn(

                    theta_pred=(
                        outputs["theta_pred"]
                    ),

                    theta_true=(
                        outputs["theta_true"]
                    ),

                    trajectory_pred=(
                        outputs["trajectory_pred"]
                    ),

                    trajectory_true=(
                        outputs["trajectory_true"]
                    ),

                )

                total_loss += (

                    losses["total"].item()

                )

                total_parameter_loss += (

                    losses["parameter"].item()

                )

                total_trajectory_loss += (

                    losses["trajectory"].item()

                )

                # ==============================================
                # STORE ALL PREDICTIONS
                # ==============================================

                all_theta_pred.append(

                    outputs["theta_pred"].detach()

                )

                all_theta_true.append(

                    outputs["theta_true"].detach()

                )

                all_trajectory_pred.append(

                    outputs[
                        "trajectory_pred"
                    ].detach()

                )

                all_trajectory_true.append(

                    outputs[
                        "trajectory_true"
                    ].detach()

                )

                num_batches += 1

        if num_batches == 0:

            raise RuntimeError(

                "Validation loader is empty."

            )

        # ======================================================
        # CONCATENATE ALL VALIDATION RESULTS
        # ======================================================

        theta_pred = torch.cat(

            all_theta_pred,

            dim=0,

        )

        theta_true = torch.cat(

            all_theta_true,

            dim=0,

        )

        trajectory_pred = torch.cat(

            all_trajectory_pred,

            dim=0,

        )

        trajectory_true = torch.cat(

            all_trajectory_true,

            dim=0,

        )

        # ======================================================
        # COMPUTE METRICS ON ENTIRE VALIDATION SET
        # ======================================================

        metrics = self._compute_metrics(

            theta_pred=theta_pred,

            theta_true=theta_true,

            trajectory_pred=trajectory_pred,

            trajectory_true=trajectory_true,

        )

        results = {

            "loss": (
                total_loss / num_batches
            ),

            "parameter_loss": (
                total_parameter_loss / num_batches
            ),

            "trajectory_loss": (
                total_trajectory_loss / num_batches
            ),

        }

        for name, value in metrics.items():

            results[name] = (

                value.item()

            )

        return results

    # ==========================================================
    # TRAINING LOOP
    # ==========================================================

    def fit(self):

        history = {

            "train_loss": [],

            "valid_loss": [],

            "train_parameter_loss": [],

            "valid_parameter_loss": [],

            "train_trajectory_loss": [],

            "valid_trajectory_loss": [],

        }

        best_valid_loss = float("inf")

        patience_counter = 0

        for epoch in range(

            1,

            self.epochs + 1,

        ):

            train_result = (

                self.train_epoch()

            )

            valid_result = (

                self.validate()

            )

            # ==================================================
            # STORE BASIC LOSSES
            # ==================================================

            history["train_loss"].append(

                train_result["loss"]

            )

            history["valid_loss"].append(

                valid_result["loss"]

            )

            history[
                "train_parameter_loss"
            ].append(

                train_result[
                    "parameter_loss"
                ]

            )

            history[
                "valid_parameter_loss"
            ].append(

                valid_result[
                    "parameter_loss"
                ]

            )

            history[
                "train_trajectory_loss"
            ].append(

                train_result[
                    "trajectory_loss"
                ]

            )

            history[
                "valid_trajectory_loss"
            ].append(

                valid_result[
                    "trajectory_loss"
                ]

            )

            # ==================================================
            # STORE ALL TRAIN METRICS
            # ==================================================

            for name, value in train_result.items():

                if name in [

                    "loss",

                    "parameter_loss",

                    "trajectory_loss",

                ]:

                    continue

                history_key = f"train_{name}"

                if history_key not in history:

                    history[history_key] = []

                history[history_key].append(
                    value
                )

            # ==================================================
            # STORE ALL VALIDATION METRICS
            # ==================================================

            for name, value in valid_result.items():

                if name in [

                    "loss",

                    "parameter_loss",

                    "trajectory_loss",

                ]:

                    continue

                history_key = f"valid_{name}"

                if history_key not in history:

                    history[history_key] = []

                history[history_key].append(
                    value
                )

            # ==================================================
            # PRINT EPOCH
            # ==================================================

            print(

                f"\nEpoch {epoch:04d}"

            )

            print(

                f"Train Loss: "

                f"{train_result['loss']:.6f}"

            )

            print(

                f"Valid Loss: "

                f"{valid_result['loss']:.6f}"

            )

            print(

                f"Train Parameter Loss: "

                f"{train_result['parameter_loss']:.6f}"

            )

            print(

                f"Valid Parameter Loss: "

                f"{valid_result['parameter_loss']:.6f}"

            )

            print(

                f"Train Trajectory Loss: "

                f"{train_result['trajectory_loss']:.6f}"

            )

            print(

                f"Valid Trajectory Loss: "

                f"{valid_result['trajectory_loss']:.6f}"

            )

            # ==================================================
            # PRINT VALIDATION METRICS
            # ==================================================

            print("\nValidation metrics:")

            for name, value in valid_result.items():

                if name in [

                    "loss",

                    "parameter_loss",

                    "trajectory_loss",

                ]:

                    continue

                print(

                    f"{name}: {value:.6f}"

                )

            # ==================================================
            # EARLY STOPPING
            # ==================================================

            if valid_result["loss"] < best_valid_loss:

                best_valid_loss = (

                    valid_result["loss"]

                )

                patience_counter = 0

                os.makedirs(

                    os.path.dirname(

                        self.checkpoint_path

                    ),

                    exist_ok=True,

                )

                torch.save(

                    {

                        "model_state_dict":

                            self.model.state_dict(),

                        "optimizer_state_dict":

                            self.optimizer.state_dict(),

                        "epoch":

                            epoch,

                        "valid_loss":

                            best_valid_loss,

                    },

                    self.checkpoint_path,

                )

            else:

                patience_counter += 1

            if patience_counter >= self.patience:

                print(

                    "\nEarly stopping triggered."

                )

                break

        # ======================================================
        # LOAD BEST MODEL
        # ======================================================

        checkpoint = torch.load(

            self.checkpoint_path,

            map_location=self.device,

            weights_only=False,

        )

        self.model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        print(

            f"\nBest validation loss: "

            f"{checkpoint['valid_loss']:.6f}"

        )

        return history

    # ==========================================================
    # TEST EVALUATION
    # ==========================================================

    def evaluate_test(self):

        self.model.eval()

        total_loss = 0.0

        total_parameter_loss = 0.0

        total_trajectory_loss = 0.0

        all_theta_pred = []

        all_theta_true = []

        all_trajectory_pred = []

        all_trajectory_true = []

        num_batches = 0

        with torch.no_grad():

            for batch in self.test_loader:

                outputs = self._forward(
                    batch
                )

                losses = self.loss_fn(

                    theta_pred=(
                        outputs["theta_pred"]
                    ),

                    theta_true=(
                        outputs["theta_true"]
                    ),

                    trajectory_pred=(
                        outputs["trajectory_pred"]
                    ),

                    trajectory_true=(
                        outputs["trajectory_true"]
                    ),

                )

                total_loss += (

                    losses["total"].item()

                )

                total_parameter_loss += (

                    losses["parameter"].item()

                )

                total_trajectory_loss += (

                    losses["trajectory"].item()

                )

                all_theta_pred.append(

                    outputs["theta_pred"].detach()

                )

                all_theta_true.append(

                    outputs["theta_true"].detach()

                )

                all_trajectory_pred.append(

                    outputs[
                        "trajectory_pred"
                    ].detach()

                )

                all_trajectory_true.append(

                    outputs[
                        "trajectory_true"
                    ].detach()

                )

                num_batches += 1

        # ======================================================
        # CONCATENATE ENTIRE TEST SET
        # ======================================================

        theta_pred = torch.cat(

            all_theta_pred,

            dim=0,

        )

        theta_true = torch.cat(

            all_theta_true,

            dim=0,

        )

        trajectory_pred = torch.cat(

            all_trajectory_pred,

            dim=0,

        )

        trajectory_true = torch.cat(

            all_trajectory_true,

            dim=0,

        )

        # ======================================================
        # COMPUTE METRICS
        # ======================================================

        metrics = self._compute_metrics(

            theta_pred=theta_pred,

            theta_true=theta_true,

            trajectory_pred=trajectory_pred,

            trajectory_true=trajectory_true,

        )

        results = {

            "test_loss":

                total_loss / num_batches,


            "test_parameter_loss":

                total_parameter_loss / num_batches,


            "test_trajectory_loss":

                total_trajectory_loss / num_batches,

        }

        # ======================================================
        # ADD METRICS
        # ======================================================

        for name, value in metrics.items():

            results[f"test_{name}"] = (

                value.detach().cpu()

            )

        return results
