import torch
import torch.nn as nn


class ParameterEstimationLoss(nn.Module):
    """
    Combined loss for ODE parameter estimation.

    Total loss:
        L = parameter_weight * L_parameter
          + trajectory_weight * L_trajectory

    where:
        L_parameter   = MSE(theta_pred, theta_true)
        L_trajectory  = MSE(trajectory_pred, trajectory_true)
    """

    def __init__(
        self,
        parameter_weight: float = 1.0,
        trajectory_weight: float = 1.0,
    ):
        super().__init__()

        self.parameter_weight = parameter_weight
        self.trajectory_weight = trajectory_weight

        self.parameter_loss = nn.MSELoss()
        self.trajectory_loss = nn.MSELoss()

    def forward(
        self,
        theta_pred: torch.Tensor,
        theta_true: torch.Tensor,
        trajectory_pred: torch.Tensor,
        trajectory_true: torch.Tensor,
    ):
        """
        theta_pred: [batch, parameter_dim]
        theta_true: [batch, parameter_dim]

        trajectory_pred: [batch, time, state_dim]
        trajectory_true: [batch, time, state_dim]
        """

        # 1. Parameter estimation loss
        parameter_loss = self.parameter_loss(
            theta_pred,
            theta_true,
        )

        print('Parameter loss: ', parameter_loss)

        # 2. Trajectory reconstruction loss
        trajectory_loss = self.trajectory_loss(
            trajectory_pred,
            trajectory_true,
        )

        print('Trajectory loss: ', trajectory_loss)

        # 3. Weighted total loss
        total_loss = self.parameter_weight * parameter_loss + \
            self.trajectory_weight * trajectory_loss

        return {
            "total": total_loss,
            "parameter": parameter_loss,
            "trajectory": trajectory_loss,
        }
