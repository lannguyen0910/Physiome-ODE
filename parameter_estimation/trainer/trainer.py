import os

import torch


class Trainer:
    def __init__(
        self,
        model,
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
        history_path=None,
        results_path=None,
        parameter_metrics=None,
        trajectory_metrics=None,
        jgd_metrics=None,
    ):
        """
        Trainer for ODE parameter estimation.

        The model is expected to return a dictionary:

            {
                "theta_pred": ...,
                "y0_pred": ...,
                "trajectory_pred": ...,
                "trajectory_full": ...,
                "t": ...,
            }

        The Trainer is responsible for:

            - training
            - validation
            - testing
            - loss calculation
            - metric calculation
            - checkpointing
            - history/result saving

        Visualization is intentionally NOT handled here.
        Visualization is handled by pipeline.py.
        """

        self.model = model

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.device = torch.device(
            device
        )

        self.epochs = int(
            epochs
        )

        self.patience = int(
            patience
        )

        self.gradient_clip = (
            None
            if gradient_clip is None
            else float(
                gradient_clip
            )
        )

        self.checkpoint_path = (
            checkpoint_path
        )

        self.history_path = (
            history_path
        )

        self.results_path = (
            results_path
        )

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        self.parameter_metrics = (
            parameter_metrics
        )

        self.trajectory_metrics = (
            trajectory_metrics
        )

        self.jgd_metrics = (
            jgd_metrics
        )

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        self.model.to(
            self.device
        )

    # ======================================================
    # TIME GRID
    # ======================================================

    @staticmethod
    def _build_time_grid(
        T,
        TY,
        X,
        M,
        Y,
        MY,
    ):
        """
        Construct the evaluation and integration grids.

        y0 is interpreted as the state at t = 0.

        Therefore:

            integration_times:
                starts at t = 0

            t_eval:
                contains the actual observation/future
                timestamps at which the trajectory is
                compared against the target.

        Returns
        -------
        t_eval:
            [T_eval]

        integration_times:
            [T_integration]

        eval_indices:
            [T_eval]

        trajectory_true:
            [B, T_eval, D]

        trajectory_mask:
            [B, T_eval, D]
        """

        T_ref = T[0]

        TY_ref = TY[0]

        # --------------------------------------------------
        # Case: no future observations
        # --------------------------------------------------

        if TY_ref.numel() == 0:

            t_eval = T_ref

            trajectory_true = X

            trajectory_mask = M

        # --------------------------------------------------
        # History + future
        # --------------------------------------------------

        else:

            duplicated_boundary = (
                torch.isclose(
                    T_ref[-1],
                    TY_ref[0],
                    atol=1e-7,
                    rtol=1e-6,
                )
            )

            if duplicated_boundary:

                # The first future point is the same
                # timestamp as the final history point.

                t_eval = torch.cat(
                    [
                        T_ref,
                        TY_ref[1:],
                    ],
                    dim=0,
                )

                trajectory_true = torch.cat(
                    [
                        X,
                        Y[:, 1:],
                    ],
                    dim=1,
                )

                trajectory_mask = torch.cat(
                    [
                        M,
                        MY[:, 1:],
                    ],
                    dim=1,
                )

            else:

                t_eval = torch.cat(
                    [
                        T_ref,
                        TY_ref,
                    ],
                    dim=0,
                )

                trajectory_true = torch.cat(
                    [
                        X,
                        Y,
                    ],
                    dim=1,
                )

                trajectory_mask = torch.cat(
                    [
                        M,
                        MY,
                    ],
                    dim=1,
                )

        # --------------------------------------------------
        # Make sure timestamps are strictly increasing
        # --------------------------------------------------

        if t_eval.numel() > 1:

            dt = torch.diff(
                t_eval
            )

            if not torch.all(
                dt > 0
            ):

                raise RuntimeError(
                    "Evaluation timestamps are not "
                    "strictly increasing.\n"
                    f"t_eval = {t_eval}"
                )

        # --------------------------------------------------
        # RK4 must start from y0 at t = 0
        # --------------------------------------------------

        t_zero = torch.zeros(
            1,
            dtype=t_eval.dtype,
            device=t_eval.device,
        )

        integration_times = torch.cat(
            [
                t_zero,
                t_eval,
            ],
            dim=0,
        )

        # Remove t=0 duplicate if t_eval already begins at 0.
        integration_times = torch.unique(
            integration_times,
            sorted=True,
        )

        # --------------------------------------------------
        # Find evaluation indices
        # --------------------------------------------------

        eval_indices = []

        for time_value in t_eval:

            matches = torch.where(
                torch.isclose(
                    integration_times,
                    time_value,
                    atol=1e-7,
                    rtol=1e-6,
                )
            )[0]

            if matches.numel() == 0:

                raise RuntimeError(
                    "Could not find evaluation timestamp "
                    f"{time_value.item()} in "
                    "integration_times."
                )

            eval_indices.append(
                matches[0]
            )

        eval_indices = torch.stack(
            eval_indices
        )

        # --------------------------------------------------
        # Shape checks
        # --------------------------------------------------

        if trajectory_true.shape[1] != (
            t_eval.shape[0]
        ):

            raise RuntimeError(
                "Trajectory/time mismatch.\n"
                f"trajectory length: "
                f"{trajectory_true.shape[1]}\n"
                f"time length: "
                f"{t_eval.shape[0]}"
            )

        if trajectory_mask.shape != (
            trajectory_true.shape
        ):

            raise RuntimeError(
                "Trajectory mask shape mismatch.\n"
                f"mask: "
                f"{trajectory_mask.shape}\n"
                f"trajectory: "
                f"{trajectory_true.shape}"
            )

        return (
            t_eval,
            integration_times,
            eval_indices,
            trajectory_true,
            trajectory_mask,
        )

    # ======================================================
    # FORWARD
    # ======================================================

    def _forward(
        self,
        batch,
    ):
        """
        Prepare a batch and run the model.

        Expected model output:
            dictionary
        """

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

        MY = batch["MY"].to(
            self.device
        )

        Y = batch["Y"].to(
            self.device
        )

        theta_true = batch[
            "theta"
        ].to(
            self.device
        )

        y0_true = batch[
            "y0"
        ].to(
            self.device
        )

        # --------------------------------------------------
        # Build evaluation/integration times
        # --------------------------------------------------

        (
            t_eval,
            integration_times,
            eval_indices,
            trajectory_true,
            trajectory_mask,
        ) = self._build_time_grid(
            T=T,
            TY=TY,
            X=X,
            M=M,
            Y=Y,
            MY=MY,
        )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        outputs = self.model(
            T=T,
            X=X,
            M=M,
            integration_times=integration_times,
            evaluation_indices=eval_indices,
        )

        if not isinstance(
            outputs,
            dict,
        ):

            raise TypeError(
                "The model must return a dictionary. "
                f"Received: {type(outputs)}"
            )

        required_keys = [
            "theta_pred",
            "y0_pred",
            "trajectory_pred",
        ]

        for key in required_keys:

            if key not in outputs:

                raise KeyError(
                    f"Model output is missing '{key}'. "
                    f"Available keys: "
                    f"{list(outputs.keys())}"
                )

        # --------------------------------------------------
        # Add targets
        # --------------------------------------------------

        outputs[
            "theta_true"
        ] = theta_true

        outputs[
            "y0_true"
        ] = y0_true

        outputs[
            "trajectory_true"
        ] = trajectory_true

        outputs[
            "trajectory_mask"
        ] = trajectory_mask

        outputs[
            "t_eval"
        ] = t_eval

        outputs[
            "integration_times"
        ] = integration_times

        outputs[
            "M"
        ] = M

        outputs[
            "MY"
        ] = MY

        return outputs

    # ======================================================
    # LOSS
    # ======================================================

    def _compute_loss(
        self,
        outputs,
    ):

        return self.loss_fn(
            theta_pred=outputs[
                "theta_pred"
            ],

            theta_true=outputs[
                "theta_true"
            ],

            trajectory_pred=outputs[
                "trajectory_pred"
            ],

            trajectory_true=outputs[
                "trajectory_true"
            ],
        )

    # ======================================================
    # METRICS
    # ======================================================

    def _compute_parameter_metrics(
        self,
        theta_pred,
        theta_true,
    ):

        if (
            self.parameter_metrics
            is None
        ):

            return {}

        return self.parameter_metrics(
            prediction=theta_pred,
            target=theta_true,
        )

    def _compute_trajectory_metrics(
        self,
        trajectory_pred,
        trajectory_true,
    ):

        if (
            self.trajectory_metrics
            is None
        ):

            return {}

        return self.trajectory_metrics(
            prediction=trajectory_pred,
            target=trajectory_true,
        )

    def _compute_jgd_metrics(
        self,
        trajectory_pred,
        trajectory_true,
    ):

        if (
            self.jgd_metrics
            is None
        ):

            return {}

        return self.jgd_metrics(
            prediction=trajectory_pred,
            target=trajectory_true,
        )

    # ======================================================
    # TRAIN ONE EPOCH
    # ======================================================

    def train_epoch(
        self,
    ):

        self.model.train()

        total_loss = 0.0

        total_parameter_loss = 0.0

        total_trajectory_loss = 0.0

        metric_totals = {}

        num_batches = 0

        for batch in self.train_loader:

            self.optimizer.zero_grad()

            try:

                outputs = self._forward(
                    batch
                )

                losses = self._compute_loss(
                    outputs
                )

                total = losses[
                    "total"
                ]

                if not torch.isfinite(
                    total
                ):

                    print(
                        "WARNING: "
                        "Non-finite training loss. "
                        "Skipping batch."
                    )

                    continue

            except (
                FloatingPointError,
                RuntimeError,
                ValueError,
            ) as error:

                print(
                    "WARNING: "
                    f"Skipping invalid training batch: "
                    f"{error}"
                )

                self.optimizer.zero_grad()

                continue

            # --------------------------------------------------
            # Backpropagation
            # --------------------------------------------------

            total.backward()

            # --------------------------------------------------
            # Gradient clipping
            # --------------------------------------------------

            if (
                self.gradient_clip
                is not None
            ):

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip,
                )

            self.optimizer.step()

            # --------------------------------------------------
            # Accumulate losses
            # --------------------------------------------------

            total_loss += (
                losses[
                    "total"
                ]
                .detach()
                .item()
            )

            total_parameter_loss += (
                losses[
                    "parameter"
                ]
                .detach()
                .item()
            )

            total_trajectory_loss += (
                losses[
                    "trajectory"
                ]
                .detach()
                .item()
            )

            # --------------------------------------------------
            # Batch metrics
            # --------------------------------------------------

            with torch.no_grad():

                parameter_results = (
                    self._compute_parameter_metrics(
                        outputs[
                            "theta_pred"
                        ],
                        outputs[
                            "theta_true"
                        ],
                    )
                )

                trajectory_results = (
                    self._compute_trajectory_metrics(
                        outputs[
                            "trajectory_pred"
                        ],
                        outputs[
                            "trajectory_true"
                        ],
                    )
                )

                # JGD intentionally not calculated
                # on each training batch.

                for name, value in (
                    parameter_results.items()
                ):

                    if (
                        torch.is_tensor(value)
                        and value.numel() == 1
                        and torch.isfinite(value)
                    ):

                        metric_name = (
                            f"parameter_{name}"
                        )

                        metric_totals[
                            metric_name
                        ] = (
                            metric_totals.get(
                                metric_name,
                                0.0,
                            )
                            + value.detach().item()
                        )

                for name, value in (
                    trajectory_results.items()
                ):

                    if (
                        torch.is_tensor(value)
                        and value.numel() == 1
                        and torch.isfinite(value)
                    ):

                        metric_name = (
                            f"trajectory_{name}"
                        )

                        metric_totals[
                            metric_name
                        ] = (
                            metric_totals.get(
                                metric_name,
                                0.0,
                            )
                            + value.detach().item()
                        )

            num_batches += 1

        if num_batches == 0:

            raise RuntimeError(
                "No valid training batches "
                "were processed."
            )

        results = {

            "loss":
                total_loss / num_batches,

            "parameter_loss":
                total_parameter_loss
                / num_batches,

            "trajectory_loss":
                total_trajectory_loss
                / num_batches,
        }

        for name, value in (
            metric_totals.items()
        ):

            results[name] = (
                value / num_batches
            )

        return results

    # ======================================================
    # VALIDATION
    # ======================================================

    def validate(
        self,
    ):

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

            for batch in self.valid_loader:

                outputs = self._forward(
                    batch
                )

                losses = self._compute_loss(
                    outputs
                )

                total_loss += (
                    losses[
                        "total"
                    ]
                    .item()
                )

                total_parameter_loss += (
                    losses[
                        "parameter"
                    ]
                    .item()
                )

                total_trajectory_loss += (
                    losses[
                        "trajectory"
                    ]
                    .item()
                )

                all_theta_pred.append(
                    outputs[
                        "theta_pred"
                    ].detach()
                )

                all_theta_true.append(
                    outputs[
                        "theta_true"
                    ].detach()
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

        # --------------------------------------------------
        # Entire validation set
        # --------------------------------------------------

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

        results = {

            "loss":
                total_loss / num_batches,

            "parameter_loss":
                total_parameter_loss
                / num_batches,

            "trajectory_loss":
                total_trajectory_loss
                / num_batches,
        }

        # --------------------------------------------------
        # Parameter metrics
        # --------------------------------------------------

        parameter_results = (
            self._compute_parameter_metrics(
                theta_pred,
                theta_true,
            )
        )

        for name, value in (
            parameter_results.items()
        ):

            # Only scalar metrics go to epoch log.
            # Per-parameter vectors are not collapsed.
            if (
                torch.is_tensor(value)
                and value.numel() == 1
                and torch.isfinite(value)
            ):

                results[
                    f"parameter_{name}"
                ] = (
                    value.detach().item()
                )

        # --------------------------------------------------
        # Trajectory metrics
        # --------------------------------------------------

        trajectory_results = (
            self._compute_trajectory_metrics(
                trajectory_pred,
                trajectory_true,
            )
        )

        for name, value in (
            trajectory_results.items()
        ):

            if (
                torch.is_tensor(value)
                and value.numel() == 1
                and torch.isfinite(value)
            ):

                results[
                    f"trajectory_{name}"
                ] = (
                    value.detach().item()
                )

        # --------------------------------------------------
        # JGD on complete validation population
        # --------------------------------------------------

        jgd_results = (
            self._compute_jgd_metrics(
                trajectory_pred,
                trajectory_true,
            )
        )

        for name, value in (
            jgd_results.items()
        ):

            if (
                torch.is_tensor(value)
                and value.numel() == 1
                and torch.isfinite(value)
            ):

                results[
                    f"jgd_{name}"
                ] = (
                    value.detach().item()
                )

        return results

    # ======================================================
    # FIT
    # ======================================================

    def fit(
        self,
    ):

        history = {
            "train": [],
            "valid": [],

            # ----------------------------------------------
            # Backward-compatible flat keys
            # ----------------------------------------------

            "train_loss": [],
            "valid_loss": [],

            "train_parameter_loss": [],
            "valid_parameter_loss": [],

            "train_trajectory_loss": [],
            "valid_trajectory_loss": [],
        }

        best_valid_loss = float(
            "inf"
        )

        patience_counter = 0

        for epoch in range(
            1,
            self.epochs + 1,
        ):

            # --------------------------------------------------
            # Training
            # --------------------------------------------------

            train_result = (
                self.train_epoch()
            )

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------

            valid_result = (
                self.validate()
            )

            # --------------------------------------------------
            # Nested history
            # --------------------------------------------------

            history[
                "train"
            ].append(
                train_result
            )

            history[
                "valid"
            ].append(
                valid_result
            )

            # --------------------------------------------------
            # Basic loss history
            # --------------------------------------------------

            history[
                "train_loss"
            ].append(
                train_result[
                    "loss"
                ]
            )

            history[
                "valid_loss"
            ].append(
                valid_result[
                    "loss"
                ]
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

            # --------------------------------------------------
            # Other train metrics
            # --------------------------------------------------

            for name, value in (
                train_result.items()
            ):

                if name in [
                    "loss",
                    "parameter_loss",
                    "trajectory_loss",
                ]:

                    continue

                key = (
                    f"train_{name}"
                )

                history.setdefault(
                    key,
                    [],
                )

                history[
                    key
                ].append(
                    value
                )

            # --------------------------------------------------
            # Other validation metrics
            # --------------------------------------------------

            for name, value in (
                valid_result.items()
            ):

                if name in [
                    "loss",
                    "parameter_loss",
                    "trajectory_loss",
                ]:

                    continue

                key = (
                    f"valid_{name}"
                )

                history.setdefault(
                    key,
                    [],
                )

                history[
                    key
                ].append(
                    value
                )

            # --------------------------------------------------
            # Logging
            # --------------------------------------------------

            print(
                "\n"
                + "=" * 60
            )

            print(
                f"Epoch {epoch:04d}"
            )

            print(
                "=" * 60
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

            print(
                "\nValidation metrics:"
            )

            for name, value in (
                valid_result.items()
            ):

                if name in [
                    "loss",
                    "parameter_loss",
                    "trajectory_loss",
                ]:

                    continue

                print(
                    f"{name}: {value:.6f}"
                )

            # --------------------------------------------------
            # Early stopping
            # --------------------------------------------------

            valid_loss = (
                valid_result[
                    "loss"
                ]
            )

            if valid_loss < (
                best_valid_loss
            ):

                best_valid_loss = (
                    valid_loss
                )

                patience_counter = 0

                checkpoint_dir = (
                    os.path.dirname(
                        self.checkpoint_path
                    )
                )

                if checkpoint_dir:

                    os.makedirs(
                        checkpoint_dir,
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

                print(
                    "Best model saved."
                )

            else:

                patience_counter += 1

                print(
                    f"No improvement: "
                    f"{patience_counter}/"
                    f"{self.patience}"
                )

            if patience_counter >= (
                self.patience
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break

        # --------------------------------------------------
        # Restore best model
        # --------------------------------------------------

        if os.path.exists(
            self.checkpoint_path
        ):

            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            self.model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            print(
                "\nLoaded best model:"
            )

            print(
                f"Epoch: "
                f"{checkpoint['epoch']}"
            )

            print(
                f"Validation loss: "
                f"{checkpoint['valid_loss']:.6f}"
            )

        # --------------------------------------------------
        # Save history
        # --------------------------------------------------

        if self.history_path:

            history_dir = (
                os.path.dirname(
                    self.history_path
                )
            )

            if history_dir:

                os.makedirs(
                    history_dir,
                    exist_ok=True,
                )

            torch.save(
                history,
                self.history_path,
            )

            print(
                f"Training history saved to: "
                f"{self.history_path}"
            )

        return history

    # ======================================================
    # TEST
    # ======================================================

    def evaluate_test(
        self,
    ):

        self.model.eval()

        total_loss = 0.0

        total_parameter_loss = 0.0

        total_trajectory_loss = 0.0

        all_theta_pred = []

        all_theta_true = []

        all_y0_pred = []

        all_y0_true = []

        all_trajectory_pred = []

        all_trajectory_true = []

        all_trajectory_masks = []

        all_times = None

        num_batches = 0

        with torch.no_grad():

            for batch in self.test_loader:

                outputs = self._forward(
                    batch
                )

                losses = self._compute_loss(
                    outputs
                )

                total_loss += (
                    losses[
                        "total"
                    ]
                    .item()
                )

                total_parameter_loss += (
                    losses[
                        "parameter"
                    ]
                    .item()
                )

                total_trajectory_loss += (
                    losses[
                        "trajectory"
                    ]
                    .item()
                )

                all_theta_pred.append(
                    outputs[
                        "theta_pred"
                    ].detach()
                )

                all_theta_true.append(
                    outputs[
                        "theta_true"
                    ].detach()
                )

                all_y0_pred.append(
                    outputs[
                        "y0_pred"
                    ].detach()
                )

                all_y0_true.append(
                    outputs[
                        "y0_true"
                    ].detach()
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

                all_trajectory_masks.append(
                    outputs[
                        "trajectory_mask"
                    ].detach()
                )

                if all_times is None:

                    all_times = (
                        outputs[
                            "t_eval"
                        ].detach()
                    )

                num_batches += 1

        if num_batches == 0:

            raise RuntimeError(
                "Test loader is empty."
            )

        # --------------------------------------------------
        # Concatenate entire test set
        # --------------------------------------------------

        theta_pred = torch.cat(
            all_theta_pred,
            dim=0,
        )

        theta_true = torch.cat(
            all_theta_true,
            dim=0,
        )

        y0_pred = torch.cat(
            all_y0_pred,
            dim=0,
        )

        y0_true = torch.cat(
            all_y0_true,
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

        trajectory_mask = torch.cat(
            all_trajectory_masks,
            dim=0,
        )

        # --------------------------------------------------
        # Base results
        # --------------------------------------------------

        results = {

            "test_loss":
                total_loss / num_batches,

            "test_parameter_loss":
                total_parameter_loss
                / num_batches,

            "test_trajectory_loss":
                total_trajectory_loss
                / num_batches,

            "theta_pred":
                theta_pred.detach().cpu(),

            "theta_true":
                theta_true.detach().cpu(),

            "y0_pred":
                y0_pred.detach().cpu(),

            "y0_true":
                y0_true.detach().cpu(),

            "trajectory_pred":
                trajectory_pred.detach().cpu(),

            "trajectory_true":
                trajectory_true.detach().cpu(),

            "trajectory_mask":
                trajectory_mask.detach().cpu(),

            "t":
                all_times.detach().cpu(),
        }

        # --------------------------------------------------
        # Parameter metrics
        # --------------------------------------------------

        parameter_results = (
            self._compute_parameter_metrics(
                theta_pred,
                theta_true,
            )
        )

        for name, value in (
            parameter_results.items()
        ):

            if not torch.is_tensor(
                value
            ):

                value = torch.as_tensor(
                    value
                )

            results[
                f"test_parameter_{name}"
            ] = (
                value.detach().cpu()
            )

        # --------------------------------------------------
        # Trajectory metrics
        # --------------------------------------------------

        trajectory_results = (
            self._compute_trajectory_metrics(
                trajectory_pred,
                trajectory_true,
            )
        )

        for name, value in (
            trajectory_results.items()
        ):

            if not torch.is_tensor(
                value
            ):

                value = torch.as_tensor(
                    value
                )

            results[
                f"test_trajectory_{name}"
            ] = (
                value.detach().cpu()
            )

        # --------------------------------------------------
        # JGD
        # --------------------------------------------------

        jgd_results = (
            self._compute_jgd_metrics(
                trajectory_pred,
                trajectory_true,
            )
        )

        for name, value in (
            jgd_results.items()
        ):

            if not torch.is_tensor(
                value
            ):

                value = torch.as_tensor(
                    value
                )

            results[
                f"test_jgd_{name}"
            ] = (
                value.detach().cpu()
            )

        # --------------------------------------------------
        # Save results
        # --------------------------------------------------

        if self.results_path:

            results_dir = (
                os.path.dirname(
                    self.results_path
                )
            )

            if results_dir:

                os.makedirs(
                    results_dir,
                    exist_ok=True,
                )

            torch.save(
                results,
                self.results_path,
            )

            print(
                f"\nTest results saved to: "
                f"{self.results_path}"
            )

        return results

    # ======================================================
    # TEST ALIAS
    # ======================================================

    def test(self):

        return self.evaluate_test()
