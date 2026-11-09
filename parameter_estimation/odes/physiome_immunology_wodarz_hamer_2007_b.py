import torch
import torch.nn as nn
import math


class ImmunologyWodarzHamer2007BODE(nn.Module):
    """
    PyTorch translation of immunology/wodarz_hamer_2007_b/model.py.
    Dynamic states: 3; independent constants: 7; derived constants used by rates: 0.
    Original constant slots: 7; algebraic slots: 0.
    """
    source_file = 'immunology/wodarz_hamer_2007_b/model.py'
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
            else {'r': 1, 'k': 10, 'd': 0.1, 'beta': 0.2, 'a': 0.2, 'u': 0.5, 'eta': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'x': 0.1, 'v': 1, 'y': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['r', 'k', 'd', 'beta', 'a', 'u', 'eta']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['x', 'v', 'y']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        x = y[:, 0]
        v = y[:, 1]
        state_y = y[:, 2]

        # Independent parameters (source constant order)
        r = theta[:, 0]  # constants[0]
        k = theta[:, 1]  # constants[1]
        d = theta[:, 2]  # constants[2]
        beta = theta[:, 3]  # constants[3]
        a = theta[:, 4]  # constants[4]
        u = theta[:, 5]  # constants[5]
        eta = theta[:, 6]  # constants[6]

        # Initialize rates and algebraic variables, as in computeRates()
        d_x = torch.zeros_like(y[:, 0])
        d_v = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_x = torch.zeros_like(y[:, 0]) + (r * x * v * (1.0 - (x + state_y) / k) - (d * x + beta * x * v))
        d_state_y = torch.zeros_like(y[:, 0]) + (beta * x * v + r * state_y * v * (1.0 - (x + state_y) / k) - a * state_y)
        d_v = torch.zeros_like(y[:, 0]) + (eta * state_y - u * v)

        return torch.stack(
            [d_x, d_v, d_state_y],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
