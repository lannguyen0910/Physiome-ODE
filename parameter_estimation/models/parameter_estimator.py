import torch.nn as nn


class ParameterEstimator(nn.Module):

    def __init__(
        self,
        encoder,
        ode,
        solver,
    ):
        super().__init__()

        self.encoder = encoder
        self.ode = ode
        self.solver = solver

    def forward(
        self,
        T,
        X,
        M,
        integration_times,
    ):

        theta_pred, y0_pred = self.encoder(T, X, M)

        trajectory_pred = self.solver(
            ode=self.ode,
            y0=y0_pred,
            theta=theta_pred,
            t=integration_times,
        )


        if trajectory_pred.ndim == 3:
            trajectory_pred = trajectory_pred.transpose(0, 1)

        return {
            "theta_pred": theta_pred,
            "y0_pred": y0_pred,
            "trajectory_pred": trajectory_pred,
            "t": integration_times,
        }
