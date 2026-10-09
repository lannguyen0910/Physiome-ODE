import os
from typing import Any, Dict, List, Optional, Tuple

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

        The important difference from the previous implementation is that
        irregular timestamps are handled per sample.  The dataset contains
        zero-padded timestamps, and different samples can have different
        valid timestamp positions.  Therefore a whole batch cannot share
        one ODE integration/evaluation grid.

        The trainer keeps the existing loss-function interface:

            loss_fn(
                theta_pred=...,
                theta_true=...,
                trajectory_pred=...,
                trajectory_true=...,
            )

        The current ParameterEstimator interface accepts:

            model(
                T=...,
                X=...,
                M=...,
                integration_times=...,
            )

        The trainer handles evaluation-index selection itself because the
        current model does not accept an ``evaluation_indices`` argument.

        Visualization is intentionally NOT handled here.  Visualization is
        handled by pipeline.py.
        """

        self.model = model

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.device = torch.device(device)

        self.epochs = int(epochs)
        self.patience = int(patience)

        self.gradient_clip = (
            None
            if gradient_clip is None
            else float(gradient_clip)
        )

        self.checkpoint_path = checkpoint_path
        self.history_path = history_path
        self.results_path = results_path

        self.parameter_metrics = parameter_metrics
        self.trajectory_metrics = trajectory_metrics
        self.jgd_metrics = jgd_metrics

        self.model.to(self.device)

        # Time-grid structure is checked once, then reused without repeated
        # CUDA synchronization checks inside every sample of every batch.
        self._time_grid_checked = False

    # ======================================================
    # TENSOR / BATCH HELPERS
    # ======================================================

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensor entries in a batch to the configured device."""

        return {
            key: value.to(self.device)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }

    @staticmethod
    def _valid_time_mask(t: torch.Tensor) -> torch.Tensor:
        """
        Return True for real timestamps and False for padding.

        The current dataset uses 0.0 as timestamp padding.  Real timestamps
        in the normalized Physiome data are positive.
        """

        return torch.isfinite(t) & (t > 0.0)

    @staticmethod
    def _validate_strictly_increasing(
        t: torch.Tensor,
        name: str,
        sample_idx: int,
    ) -> None:
        """Check that a 1-D timestamp tensor is strictly increasing."""

        if t.numel() <= 1:
            return

        dt = torch.diff(t)

        if not torch.all(dt > 0):
            raise RuntimeError(
                f"{name} timestamps are not strictly increasing "
                f"for sample {sample_idx}.\n"
                f"{name} = {t}"
            )

    # ======================================================
    # TIME GRID FOR ONE SAMPLE
    # ======================================================

    @classmethod
    def _build_sample_time_grid(
        cls,
        T: torch.Tensor,
        TY: torch.Tensor,
        X: torch.Tensor,
        M: torch.Tensor,
        Y: torch.Tensor,
        MY: torch.Tensor,
        sample_idx: int,
        integration_times: torch.Tensor,
        validate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Prepare one sample's ragged target trajectory and indices."""

        # --------------------------------------------------
        # Remove zero padding from history and forecast times.
        # --------------------------------------------------

        obs_valid = cls._valid_time_mask(T)
        T_obs = T[obs_valid]
        X_obs = X[obs_valid]
        M_obs = M[obs_valid]

        if T_obs.numel() == 0:
            raise RuntimeError(
                f"Sample {sample_idx} contains no valid observation timestamps."
            )

        future_valid = cls._valid_time_mask(TY)
        TY_valid = TY[future_valid]
        Y_valid = Y[future_valid]
        MY_valid = MY[future_valid]

        # --------------------------------------------------
        # Optional one-time validation. Do NOT synchronize the GPU for every
        # sample on every batch.
        # --------------------------------------------------

        if validate:
            cls._validate_strictly_increasing(
                T_obs,
                "Observation",
                sample_idx,
            )

            if TY_valid.numel() > 1:
                cls._validate_strictly_increasing(
                    TY_valid,
                    "Forecast",
                    sample_idx,
                )

        # --------------------------------------------------
        # Remove duplicated history/forecast boundary.
        # --------------------------------------------------

        if TY_valid.numel() > 0:
            duplicated_boundary = torch.isclose(
                T_obs[-1],
                TY_valid[0],
                atol=1e-7,
                rtol=1e-6,
            )

            if duplicated_boundary:
                TY_valid = TY_valid[1:]
                Y_valid = Y_valid[1:]
                MY_valid = MY_valid[1:]

        # --------------------------------------------------
        # Build sample target trajectory.
        # --------------------------------------------------

        if TY_valid.numel() > 0:
            t_eval = torch.cat([T_obs, TY_valid], dim=0)
            trajectory_true = torch.cat([X_obs, Y_valid], dim=0)
            trajectory_mask = torch.cat([M_obs, MY_valid], dim=0)
        else:
            t_eval = T_obs
            trajectory_true = X_obs
            trajectory_mask = M_obs

        if validate:
            cls._validate_strictly_increasing(
                t_eval,
                "Evaluation",
                sample_idx,
            )

        if trajectory_true.shape[0] != t_eval.shape[0]:
            raise RuntimeError(
                f"Trajectory/time mismatch for sample {sample_idx}: "
                f"trajectory length={trajectory_true.shape[0]}, "
                f"time length={t_eval.shape[0]}"
            )

        # --------------------------------------------------
        # Map this sample's timestamps into the batch integration grid.
        # --------------------------------------------------

        evaluation_indices = torch.searchsorted(
            integration_times,
            t_eval,
        )

        if validate:
            if torch.any(evaluation_indices >= integration_times.numel()):
                raise RuntimeError(
                    f"Could not map all evaluation timestamps for sample {sample_idx}."
                )

            matched_times = integration_times[evaluation_indices]

            if not torch.allclose(
                matched_times,
                t_eval,
                atol=1e-6,
                rtol=1e-5,
            ):
                raise RuntimeError(
                    f"Evaluation timestamps could not be matched to the shared "
                    f"integration grid for sample {sample_idx}."
                )

        return {
            "T_obs": T_obs,
            "X_obs": X_obs,
            "M_obs": M_obs,
            "TY": TY_valid,
            "Y": Y_valid,
            "MY": MY_valid,
            "t_eval": t_eval,
            "integration_times": integration_times,
            "evaluation_indices": evaluation_indices,
            "trajectory_true": trajectory_true,
            "trajectory_mask": trajectory_mask,
        }

    # ======================================================
    # FORWARD FOR A WHOLE PADDED BATCH
    # ======================================================

    def _forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fast forward pass for padded irregular-time data.

        The expensive operations are fully batched:

            1. GRU encoder: one call for the whole batch
            2. ODE/RK4 solver: one call for the whole batch

        The only per-sample loop that remains prepares ragged target slices and
        gathers those slices from the already computed batch trajectory.
        """

        batch = self._move_batch_to_device(batch)

        T = batch["T"]
        X = batch["X"]
        M = batch["M"]
        TY = batch["TY"]
        MY = batch["MY"]
        Y = batch["Y"]
        theta_true = batch["theta"]
        y0_true = batch["y0"]

        batch_size = T.shape[0]

        # --------------------------------------------------
        # One shared integration grid for this batch.
        # --------------------------------------------------

        valid_times = torch.cat(
            [T[T > 0.0], TY[TY > 0.0]],
            dim=0,
        )

        if valid_times.numel() == 0:
            raise RuntimeError("Batch contains no valid timestamps.")

        t_eval_shared = torch.unique(
            valid_times,
            sorted=True,
        )

        zero = torch.zeros(
            1,
            dtype=t_eval_shared.dtype,
            device=t_eval_shared.device,
        )

        if t_eval_shared[0] > 0.0:
            integration_times = torch.cat(
                [zero, t_eval_shared],
                dim=0,
            )
        else:
            integration_times = t_eval_shared

        # A single global monotonicity check for the current batch.
        if not self._time_grid_checked:
            self._validate_strictly_increasing(
                integration_times,
                "Batch integration",
                -1,
            )

        # --------------------------------------------------
        # CRITICAL PERFORMANCE POINT:
        # one batched ParameterEstimator call, not B separate calls.
        # --------------------------------------------------

        model_output = self.model(
            T=T,
            X=X,
            M=M,
            integration_times=integration_times,
        )

        if not isinstance(model_output, dict):
            raise TypeError(
                "The model must return a dictionary. "
                f"Received: {type(model_output)}"
            )

        required_keys = (
            "theta_pred",
            "y0_pred",
            "trajectory_pred",
        )

        for key in required_keys:
            if key not in model_output:
                raise KeyError(
                    f"Model output is missing '{key}'. "
                    f"Available keys: {list(model_output.keys())}"
                )

        theta_pred = model_output["theta_pred"]
        y0_pred = model_output["y0_pred"]
        trajectory_full = model_output["trajectory_pred"]

        if trajectory_full.ndim != 3:
            raise RuntimeError(
                "Expected trajectory_pred with shape [B,T,D], "
                f"received {tuple(trajectory_full.shape)}."
            )

        if trajectory_full.shape[0] != batch_size:
            raise RuntimeError(
                "Trajectory batch dimension mismatch: "
                f"{trajectory_full.shape[0]} != {batch_size}."
            )

        if trajectory_full.shape[1] != integration_times.numel():
            raise RuntimeError(
                "Trajectory/integration-time mismatch: "
                f"trajectory has {trajectory_full.shape[1]} points, "
                f"integration grid has {integration_times.numel()} points."
            )

        # --------------------------------------------------
        # Batch-wide finite checks: much cheaper than checking every sample.
        # --------------------------------------------------

        if not torch.isfinite(theta_pred).all():
            raise FloatingPointError("Non-finite theta prediction.")

        if not torch.isfinite(y0_pred).all():
            raise FloatingPointError("Non-finite y0 prediction.")

        if not torch.isfinite(trajectory_full).all():
            raise FloatingPointError("Non-finite trajectory prediction.")

        # --------------------------------------------------
        # Create sample-specific ragged targets and gather predictions.
        # No model/ODE call occurs inside this loop.
        # --------------------------------------------------

        sample_outputs: List[Dict[str, torch.Tensor]] = []

        validate_samples = not self._time_grid_checked

        for sample_idx in range(batch_size):

            sample = self._build_sample_time_grid(
                T=T[sample_idx],
                TY=TY[sample_idx],
                X=X[sample_idx],
                M=M[sample_idx],
                Y=Y[sample_idx],
                MY=MY[sample_idx],
                sample_idx=sample_idx,
                integration_times=integration_times,
                validate=validate_samples,
            )

            trajectory_pred_i = trajectory_full[
                sample_idx,
                sample["evaluation_indices"],
                :,
            ]

            sample_outputs.append(
                {
                    "theta_pred": theta_pred[sample_idx],
                    "theta_true": theta_true[sample_idx],
                    "y0_pred": y0_pred[sample_idx],
                    "y0_true": y0_true[sample_idx],
                    "trajectory_pred": trajectory_pred_i,
                    "trajectory_true": sample["trajectory_true"],
                    "trajectory_mask": sample["trajectory_mask"],
                    "t_eval": sample["t_eval"],
                    "integration_times": integration_times,
                    "evaluation_indices": sample["evaluation_indices"],
                    "T_obs": sample["T_obs"],
                    "X_obs": sample["X_obs"],
                    "M_obs": sample["M_obs"],
                    "TY": sample["TY"],
                    "Y": sample["Y"],
                    "MY": sample["MY"],
                }
            )

        self._time_grid_checked = True

        return {
            "theta_pred": theta_pred,
            "theta_true": theta_true,
            "y0_pred": y0_pred,
            "y0_true": y0_true,
            "trajectory_full": trajectory_full,
            "integration_times": integration_times,
            "t_eval_shared": t_eval_shared,
            "samples": sample_outputs,
        }

    # ======================================================
    # LOSS
    # ======================================================

    def _compute_loss(
        self,
        outputs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the existing loss function independently for every sample
        and average the losses across the batch.

        This keeps the original loss implementation unchanged while allowing
        every sample to have its own irregular trajectory length.
        """

        sample_losses: List[Dict[str, torch.Tensor]] = []

        for sample_idx, sample in enumerate(outputs["samples"]):

            losses = self.loss_fn(
                theta_pred=sample["theta_pred"].unsqueeze(0),
                theta_true=sample["theta_true"].unsqueeze(0),
                trajectory_pred=sample["trajectory_pred"].unsqueeze(0),
                trajectory_true=sample["trajectory_true"].unsqueeze(0),
            )

            if not isinstance(losses, dict):
                raise TypeError(
                    "The loss function must return a dictionary with "
                    "'total', 'parameter' and 'trajectory' keys. "
                    f"Received: {type(losses)}"
                )

            for key in ("total", "parameter", "trajectory"):
                if key not in losses:
                    raise KeyError(
                        f"Loss output is missing '{key}' for sample "
                        f"{sample_idx}. Available: {list(losses.keys())}"
                    )

                if not torch.isfinite(losses[key]).all():
                    raise FloatingPointError(
                        f"Non-finite {key} loss for sample {sample_idx}."
                    )

            sample_losses.append(losses)

        if not sample_losses:
            raise RuntimeError("No samples available for loss computation.")

        return {
            key: torch.stack(
                [losses[key] for losses in sample_losses]
            ).mean()
            for key in ("total", "parameter", "trajectory")
        }

    # ======================================================
    # METRICS
    # ======================================================

    def _compute_parameter_metrics(
        self,
        theta_pred: torch.Tensor,
        theta_true: torch.Tensor,
    ) -> Dict[str, Any]:
        if self.parameter_metrics is None:
            return {}

        return self.parameter_metrics(
            prediction=theta_pred,
            target=theta_true,
        )

    def _compute_trajectory_metrics_one_sample(
        self,
        trajectory_pred: torch.Tensor,
        trajectory_true: torch.Tensor,
    ) -> Dict[str, Any]:
        if self.trajectory_metrics is None:
            return {}

        return self.trajectory_metrics(
            prediction=trajectory_pred.unsqueeze(0),
            target=trajectory_true.unsqueeze(0),
        )

    @staticmethod
    def _average_scalar_metric_dicts(
        metric_dicts: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Average scalar metric values across ragged samples."""

        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for metric_dict in metric_dicts:
            for name, value in metric_dict.items():

                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)

                # Epoch history stores only scalar values here.
                if value.numel() != 1:
                    continue

                value = value.detach().float().reshape(())

                if not torch.isfinite(value):
                    continue

                totals[name] = totals.get(name, 0.0) + value.item()
                counts[name] = counts.get(name, 0) + 1

        return {
            name: totals[name] / counts[name]
            for name in totals
            if counts[name] > 0
        }

    def _compute_trajectory_metrics_ragged(
        self,
        samples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Compute trajectory metrics sample-by-sample because trajectory lengths
        differ after zero-padding is removed.
        """

        if self.trajectory_metrics is None:
            return {}

        metric_dicts = []

        for sample in samples:
            metric_dicts.append(
                self._compute_trajectory_metrics_one_sample(
                    sample["trajectory_pred"],
                    sample["trajectory_true"],
                )
            )

        return self._average_scalar_metric_dicts(metric_dicts)

    def _compute_jgd_metrics_ragged(
        self,
        samples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Compute the configured JGD metric sample-by-sample and average it.

        The current JGD implementation in the project expects a rectangular
        [B,T,D] tensor.  Once padding is removed, trajectories are ragged, so
        the trainer cannot concatenate them without inventing timestamps or
        values.  Sample-wise averaging is therefore used here.
        """

        if self.jgd_metrics is None:
            return {}

        metric_dicts = []

        for sample in samples:
            metric_dicts.append(
                self.jgd_metrics(
                    prediction=sample["trajectory_pred"].unsqueeze(0),
                    target=sample["trajectory_true"].unsqueeze(0),
                )
            )

        return self._average_scalar_metric_dicts(metric_dicts)

    # ======================================================
    # TRAIN ONE EPOCH
    # ======================================================

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total_parameter_loss = 0.0
        total_trajectory_loss = 0.0

        metric_totals: Dict[str, float] = {}

        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):

            self.optimizer.zero_grad(set_to_none=True)

            try:
                outputs = self._forward(batch)
                losses = self._compute_loss(outputs)

                total = losses["total"]

                if not torch.isfinite(total).all():
                    raise FloatingPointError(
                        "Non-finite training loss."
                    )

            except (
                FloatingPointError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
            ) as error:

                print(
                    "WARNING: "
                    f"Skipping invalid training batch {batch_idx}: "
                    f"{error}"
                )

                self.optimizer.zero_grad(set_to_none=True)
                continue

            # --------------------------------------------------
            # Backpropagation
            # --------------------------------------------------

            total.backward()

            # --------------------------------------------------
            # Gradient clipping
            # --------------------------------------------------

            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip,
                )

            self.optimizer.step()

            # --------------------------------------------------
            # Accumulate losses
            # --------------------------------------------------

            total_loss += total.detach().item()
            total_parameter_loss += (
                losses["parameter"].detach().item()
            )
            total_trajectory_loss += (
                losses["trajectory"].detach().item()
            )

            # --------------------------------------------------
            # Metrics
            # --------------------------------------------------

            with torch.no_grad():

                # Keep training-time metric overhead low. Parameter metrics
                # are batched; trajectory/JGD metrics are computed during
                # validation and test, where they are actually used for
                # model selection/reporting.
                parameter_results = self._compute_parameter_metrics(
                    outputs["theta_pred"],
                    outputs["theta_true"],
                )

                for name, value in parameter_results.items():
                    if not torch.is_tensor(value):
                        value = torch.as_tensor(value)

                    if value.numel() == 1 and torch.isfinite(value).all():
                        metric_name = f"parameter_{name}"
                        metric_totals[metric_name] = (
                            metric_totals.get(metric_name, 0.0)
                            + value.detach().float().item()
                        )

            num_batches += 1

            # Show progress during long ODE epochs.
            if (
                batch_idx == 0
                or (batch_idx + 1) % 10 == 0
                or (batch_idx + 1) == len(self.train_loader)
            ):
                print(
                    f"       Train batch {batch_idx + 1}/"
                    f"{len(self.train_loader)}",
                    flush=True,
                )

        if num_batches == 0:
            raise RuntimeError(
                "No valid training batches were processed."
            )

        results: Dict[str, float] = {
            "loss": total_loss / num_batches,
            "parameter_loss": total_parameter_loss / num_batches,
            "trajectory_loss": total_trajectory_loss / num_batches,
        }

        for name, value in metric_totals.items():
            results[name] = value / num_batches

        return results

    # ======================================================
    # VALIDATION
    # ======================================================

    def validate(self) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        total_parameter_loss = 0.0
        total_trajectory_loss = 0.0

        all_theta_pred: List[torch.Tensor] = []
        all_theta_true: List[torch.Tensor] = []

        all_samples: List[Dict[str, torch.Tensor]] = []

        num_batches = 0

        with torch.no_grad():

            for batch in self.valid_loader:

                outputs = self._forward(batch)
                losses = self._compute_loss(outputs)

                total_loss += losses["total"].item()
                total_parameter_loss += losses["parameter"].item()
                total_trajectory_loss += losses["trajectory"].item()

                all_theta_pred.append(
                    outputs["theta_pred"].detach()
                )

                all_theta_true.append(
                    outputs["theta_true"].detach()
                )

                for sample in outputs["samples"]:
                    # Keep detached tensors for validation metrics.
                    all_samples.append(
                        {
                            key: value.detach()
                            for key, value in sample.items()
                        }
                    )

                num_batches += 1

        if num_batches == 0:
            raise RuntimeError(
                "Validation loader is empty."
            )

        theta_pred = torch.cat(
            all_theta_pred,
            dim=0,
        )

        theta_true = torch.cat(
            all_theta_true,
            dim=0,
        )

        results: Dict[str, float] = {
            "loss": total_loss / num_batches,
            "parameter_loss": total_parameter_loss / num_batches,
            "trajectory_loss": total_trajectory_loss / num_batches,
        }

        # --------------------------------------------------
        # Parameter metrics over complete validation set
        # --------------------------------------------------

        parameter_results = self._compute_parameter_metrics(
            theta_pred,
            theta_true,
        )

        for name, value in parameter_results.items():
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)

            if value.numel() == 1 and torch.isfinite(value).all():
                results[f"parameter_{name}"] = (
                    value.detach().item()
                )

        # --------------------------------------------------
        # Trajectory metrics over ragged samples
        # --------------------------------------------------

        trajectory_results = self._compute_trajectory_metrics_ragged(
            all_samples
        )

        for name, value in trajectory_results.items():
            results[f"trajectory_{name}"] = float(value)

        # --------------------------------------------------
        # JGD over ragged samples
        # --------------------------------------------------

        jgd_results = self._compute_jgd_metrics_ragged(
            all_samples
        )

        for name, value in jgd_results.items():
            results[f"jgd_{name}"] = float(value)

        return results

    # ======================================================
    # RAGGED TRAJECTORY PADDING FOR SAVED RESULTS
    # ======================================================

    @staticmethod
    def _pad_trajectory_samples(
        samples: List[Dict[str, torch.Tensor]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Convert ragged trajectory results into padded tensors for saving.

        Returns
        -------
        t_padded:
            [N, L_max], NaN after each sample's valid length.

        trajectory_pred_padded:
            [N, L_max, D], NaN after valid length.

        trajectory_true_padded:
            [N, L_max, D], NaN after valid length.

        trajectory_mask_padded:
            [N, L_max, D].  Includes both the original state mask and the
            valid-length mask.
        """

        if not samples:
            raise RuntimeError(
                "Cannot pad an empty trajectory sample list."
            )

        max_length = max(
            sample["t_eval"].numel()
            for sample in samples
        )

        state_dim = samples[0]["trajectory_pred"].shape[-1]
        num_samples = len(samples)

        dtype = samples[0]["trajectory_pred"].dtype
        device = samples[0]["trajectory_pred"].device

        t_padded = torch.full(
            (num_samples, max_length),
            float("nan"),
            dtype=samples[0]["t_eval"].dtype,
            device=device,
        )

        trajectory_pred_padded = torch.full(
            (num_samples, max_length, state_dim),
            float("nan"),
            dtype=dtype,
            device=device,
        )

        trajectory_true_padded = torch.full(
            (num_samples, max_length, state_dim),
            float("nan"),
            dtype=dtype,
            device=device,
        )

        trajectory_mask_padded = torch.zeros(
            (num_samples, max_length, state_dim),
            dtype=samples[0]["trajectory_mask"].dtype,
            device=device,
        )

        for sample_idx, sample in enumerate(samples):
            length = sample["t_eval"].numel()

            t_padded[sample_idx, :length] = sample["t_eval"]

            trajectory_pred_padded[
                sample_idx,
                :length,
            ] = sample["trajectory_pred"]

            trajectory_true_padded[
                sample_idx,
                :length,
            ] = sample["trajectory_true"]

            trajectory_mask_padded[
                sample_idx,
                :length,
            ] = sample["trajectory_mask"]

        return (
            t_padded,
            trajectory_pred_padded,
            trajectory_true_padded,
            trajectory_mask_padded,
        )

    # ======================================================
    # FIT
    # ======================================================

    def fit(self):
        history = {
            "train": [],
            "valid": [],
            "train_loss": [],
            "valid_loss": [],
            "train_parameter_loss": [],
            "valid_parameter_loss": [],
            "train_trajectory_loss": [],
            "valid_trajectory_loss": [],
        }

        best_valid_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):

            # --------------------------------------------------
            # Training
            # --------------------------------------------------

            train_result = self.train_epoch()

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------

            valid_result = self.validate()

            # --------------------------------------------------
            # Nested history
            # --------------------------------------------------

            history["train"].append(train_result)
            history["valid"].append(valid_result)

            # --------------------------------------------------
            # Basic loss history
            # --------------------------------------------------

            history["train_loss"].append(
                train_result["loss"]
            )

            history["valid_loss"].append(
                valid_result["loss"]
            )

            history["train_parameter_loss"].append(
                train_result["parameter_loss"]
            )

            history["valid_parameter_loss"].append(
                valid_result["parameter_loss"]
            )

            history["train_trajectory_loss"].append(
                train_result["trajectory_loss"]
            )

            history["valid_trajectory_loss"].append(
                valid_result["trajectory_loss"]
            )

            # --------------------------------------------------
            # Other train metrics
            # --------------------------------------------------

            for name, value in train_result.items():
                if name in {
                    "loss",
                    "parameter_loss",
                    "trajectory_loss",
                }:
                    continue

                history.setdefault(
                    f"train_{name}",
                    [],
                ).append(value)

            # --------------------------------------------------
            # Other validation metrics
            # --------------------------------------------------

            for name, value in valid_result.items():
                if name in {
                    "loss",
                    "parameter_loss",
                    "trajectory_loss",
                }:
                    continue

                history.setdefault(
                    f"valid_{name}",
                    [],
                ).append(value)

            # --------------------------------------------------
            # Logging
            # --------------------------------------------------

            print("\n" + "=" * 60, flush=True)
            print(f"Epoch {epoch:04d}", flush=True)
            print("=" * 60)

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

            print("\nValidation metrics:")

            for name, value in valid_result.items():
                if name in {
                    "loss",
                    "parameter_loss",
                    "trajectory_loss",
                }:
                    continue

                print(f"{name}: {value:.6f}")

            # --------------------------------------------------
            # Early stopping
            # --------------------------------------------------

            valid_loss = valid_result["loss"]

            if valid_loss < best_valid_loss:

                best_valid_loss = valid_loss
                patience_counter = 0

                checkpoint_dir = os.path.dirname(
                    self.checkpoint_path
                )

                if checkpoint_dir:
                    os.makedirs(
                        checkpoint_dir,
                        exist_ok=True,
                    )

                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "valid_loss": best_valid_loss,
                    },
                    self.checkpoint_path,
                )

                print("Best model saved.")

            else:

                patience_counter += 1

                print(
                    f"No improvement: "
                    f"{patience_counter}/"
                    f"{self.patience}"
                )

            if patience_counter >= self.patience:
                print("\nEarly stopping triggered.")
                break

        # --------------------------------------------------
        # Restore best model
        # --------------------------------------------------

        if os.path.exists(self.checkpoint_path):

            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            print("\nLoaded best model:")
            print(
                f"Epoch: {checkpoint['epoch']}"
            )
            print(
                f"Validation loss: "
                f"{checkpoint['valid_loss']:.6f}"
            )

        # --------------------------------------------------
        # Save history
        # --------------------------------------------------

        if self.history_path:

            history_dir = os.path.dirname(
                self.history_path
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

    def evaluate_test(self):
        self.model.eval()

        total_loss = 0.0
        total_parameter_loss = 0.0
        total_trajectory_loss = 0.0

        all_theta_pred: List[torch.Tensor] = []
        all_theta_true: List[torch.Tensor] = []
        all_y0_pred: List[torch.Tensor] = []
        all_y0_true: List[torch.Tensor] = []
        all_samples: List[Dict[str, torch.Tensor]] = []

        num_batches = 0

        with torch.no_grad():

            for batch in self.test_loader:

                outputs = self._forward(batch)
                losses = self._compute_loss(outputs)

                total_loss += losses["total"].item()
                total_parameter_loss += losses["parameter"].item()
                total_trajectory_loss += losses["trajectory"].item()

                all_theta_pred.append(
                    outputs["theta_pred"].detach()
                )

                all_theta_true.append(
                    outputs["theta_true"].detach()
                )

                all_y0_pred.append(
                    outputs["y0_pred"].detach()
                )

                all_y0_true.append(
                    outputs["y0_true"].detach()
                )

                for sample in outputs["samples"]:
                    all_samples.append(
                        {
                            key: value.detach()
                            for key, value in sample.items()
                        }
                    )

                num_batches += 1

        if num_batches == 0:
            raise RuntimeError(
                "Test loader is empty."
            )

        # --------------------------------------------------
        # Parameter tensors are rectangular
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

        # --------------------------------------------------
        # Ragged trajectory tensors for saving/plotting
        # --------------------------------------------------

        (
            t_padded,
            trajectory_pred,
            trajectory_true,
            trajectory_mask,
        ) = self._pad_trajectory_samples(
            all_samples
        )

        # --------------------------------------------------
        # Base results
        # --------------------------------------------------

        results = {
            "test_loss": total_loss / num_batches,
            "test_parameter_loss": (
                total_parameter_loss / num_batches
            ),
            "test_trajectory_loss": (
                total_trajectory_loss / num_batches
            ),
            "theta_pred": theta_pred.cpu(),
            "theta_true": theta_true.cpu(),
            "y0_pred": y0_pred.cpu(),
            "y0_true": y0_true.cpu(),
            "trajectory_pred": trajectory_pred.cpu(),
            "trajectory_true": trajectory_true.cpu(),
            "trajectory_mask": trajectory_mask.cpu(),
            # 2-D because every sample can have its own irregular time grid.
            "t": t_padded.cpu(),
        }

        # --------------------------------------------------
        # Parameter metrics
        # --------------------------------------------------

        parameter_results = self._compute_parameter_metrics(
            theta_pred,
            theta_true,
        )

        for name, value in parameter_results.items():
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)

            results[
                f"test_parameter_{name}"
            ] = value.detach().cpu()

        # --------------------------------------------------
        # Trajectory metrics
        # --------------------------------------------------

        trajectory_results = (
            self._compute_trajectory_metrics_ragged(
                all_samples
            )
        )

        for name, value in trajectory_results.items():
            results[
                f"test_trajectory_{name}"
            ] = torch.as_tensor(value)

        # --------------------------------------------------
        # JGD
        # --------------------------------------------------

        jgd_results = self._compute_jgd_metrics_ragged(
            all_samples
        )

        for name, value in jgd_results.items():
            results[
                f"test_jgd_{name}"
            ] = torch.as_tensor(value)

        # --------------------------------------------------
        # Save results
        # --------------------------------------------------

        if self.results_path:

            results_dir = os.path.dirname(
                self.results_path
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
