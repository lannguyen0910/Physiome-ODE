import torch
import torch.nn as nn
import math


class CircadianRhythmsGoodwin1965AODE(nn.Module):
    """
    PyTorch translation of circadian_rhythms/goodwin_1965_a/model.py.
    Dynamic states: 2; independent constants: 6; derived constants used by rates: 0.
    Original constant slots: 6; algebraic slots: 0.
    """
    source_file = 'circadian_rhythms/goodwin_1965_a/model.py'
    source_state_dim = 2
    source_constant_dim = 6
    algebraic_dim = 0
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 2
        self.parameter_dim = 6

        self.parameters = (
            parameters
            if parameters is not None
            else {'ai': 72, 'bi': 2, 'Ai': 36, 'ki': 1, 'beta_i': 0, 'alpha_i': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Xi': 7, 'Yi': -10}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['ai', 'bi', 'Ai', 'ki', 'beta_i', 'alpha_i']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Xi', 'Yi']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        Xi = y[:, 0]
        Yi = y[:, 1]

        # Independent parameters (source constant order)
        ai = theta[:, 0]  # constants[0]
        bi = theta[:, 1]  # constants[1]
        Ai = theta[:, 2]  # constants[2]
        ki = theta[:, 3]  # constants[3]
        beta_i = theta[:, 4]  # constants[4]
        alpha_i = theta[:, 5]  # constants[5]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Xi = torch.zeros_like(y[:, 0])
        d_Yi = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Xi = torch.zeros_like(y[:, 0]) + (ai / (Ai + ki * Yi) - bi)
        d_Yi = torch.zeros_like(y[:, 0]) + (alpha_i * Xi - beta_i)

        return torch.stack(
            [d_Xi, d_Yi],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
