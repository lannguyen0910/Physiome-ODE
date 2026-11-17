import torch
import torch.nn as nn


class ParameterEstimator(nn.Module):

    def __init__(
        self,
        encoder,
        ode,
        solver,
        theta_mean,
        theta_std,
    ):
        super().__init__()

        self.encoder = encoder
        self.ode = ode
        self.solver = solver

        # Registered as buffers so they automatically move
        # to CPU/GPU with the model.
        self.register_buffer(
            "theta_mean",
            theta_mean.float(),
        )

        self.register_buffer(
            "theta_std",
            torch.clamp(
                theta_std.float(),
                min=1e-8,
            ),
        )

    def denormalize_theta(
        self,
        theta_norm,
    ):
        return (
            theta_norm * self.theta_std
            + self.theta_mean
        )

    def normalize_theta(
        self,
        theta,
    ):
        return (
            theta - self.theta_mean
        ) / self.theta_std

    def forward(
        self,
        T,
        X,
        M,
        integration_times,
    ):

        # Encoder predicts normalized parameters.
        theta_pred_norm, y0_pred = (
            self.encoder(
                T,
                X,
                M,
            )
        )

        # Convert back to physical units BEFORE
        # passing parameters to the ODE.
        theta_pred = (
            self.denormalize_theta(
                theta_pred_norm
            )
        )

        trajectory_pred = self.solver(
            ode=self.ode,
            y0=y0_pred,
            theta=theta_pred,
            t=integration_times,
        )

        return {
            "theta_pred_norm": theta_pred_norm,
            "theta_pred": theta_pred,
            "y0_pred": y0_pred,
            "trajectory_pred": trajectory_pred,
            "t": integration_times,
        }
