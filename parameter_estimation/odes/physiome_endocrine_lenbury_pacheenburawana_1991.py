import torch
import torch.nn as nn
import math


class EndocrineLenburyPacheenburawana1991ODE(nn.Module):
    """
    PyTorch translation of endocrine/lenbury_pacheenburawana_1991/model.py.
    Dynamic states: 3; independent constants: 7; derived constants used by rates: 0.
    Original constant slots: 7; algebraic slots: 0.
    """
    source_file = 'endocrine/lenbury_pacheenburawana_1991/model.py'
    source_state_dim = 3
    source_constant_dim = 7
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

        self.state_dim = 3
        self.parameter_dim = 7

        self.parameters = (
            parameters
            if parameters is not None
            else {'alpha': 0.5, 'omega': 2, 'D': 0.8228, 'a': 8.1252, 'b': 1.091, 'beta': 0.38, 'gamma': 0.6}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'x': 1.5, 'z': 1, 'y': 0.95}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['alpha', 'omega', 'D', 'a', 'b', 'beta', 'gamma']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['x', 'z', 'y']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        x = y[:, 0]
        z = y[:, 1]
        state_y = y[:, 2]

        # Independent parameters (source constant order)
        alpha = theta[:, 0]  # constants[0]
        omega = theta[:, 1]  # constants[1]
        D = theta[:, 2]  # constants[2]
        a = theta[:, 3]  # constants[3]
        b = theta[:, 4]  # constants[4]
        beta = theta[:, 5]  # constants[5]
        gamma = theta[:, 6]  # constants[6]

        # Initialize rates and algebraic variables, as in computeRates()
        d_x = torch.zeros_like(y[:, 0])
        d_z = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_x = torch.zeros_like(y[:, 0]) + (alpha * torch.exp(a * (1.0 - z ** 2.0) + b * (1.0 - state_y ** 2.0)) + D * torch.cos(omega * t) - alpha * x)
        d_state_y = torch.zeros_like(y[:, 0]) + (beta * x * torch.exp(b * (1.0 - z ** 2.0)) - beta * state_y)
        d_z = torch.zeros_like(y[:, 0]) + (gamma * state_y - gamma * z)

        return torch.stack(
            [d_x, d_z, d_state_y],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
