import torch
import torch.nn as nn


class ParameterEstimationLoss(nn.Module):
    def __init__(
        self,
        parameter_loss_weight=0.1,
        trajectory_loss_weight=1.0,
    ):
        super().__init__()

        self.parameter_loss_weight = (
            float(parameter_loss_weight)
        )

        self.trajectory_loss_weight = (
            float(trajectory_loss_weight)
        )

    @staticmethod
    def _check_finite(
        name,
        tensor,
    ):

        if not torch.isfinite(tensor).all():

            raise FloatingPointError(
                f"Non-finite values detected in {name}."
            )

    def forward(
        self,
        theta_pred,
        theta_true,
        trajectory_pred,
        trajectory_true,
    ):

        self._check_finite(
            "theta_pred",
            theta_pred,
        )

        self._check_finite(
            "theta_true",
            theta_true,
        )

        self._check_finite(
            "trajectory_pred",
            trajectory_pred,
        )

        self._check_finite(
            "trajectory_true",
            trajectory_true,
        )

        parameter_loss = torch.mean(
            (
                theta_pred
                - theta_true
            ) ** 2
        )

        trajectory_loss = torch.mean(
            (
                trajectory_pred
                - trajectory_true
            ) ** 2
        )

        total_loss = (
            self.parameter_loss_weight
            * parameter_loss
            +
            self.trajectory_loss_weight
            * trajectory_loss
        )

        return {
            "total": total_loss,
            "parameter": parameter_loss,
            "trajectory": trajectory_loss,
        }
