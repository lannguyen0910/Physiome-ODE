import torch
import torch.nn as nn
import math


class ImmunologyWodarzHamer2007AODE(nn.Module):
    """
    PyTorch translation of immunology/wodarz_hamer_2007_a/model.py.
    Dynamic states: 2; independent constants: 5; derived constants used by rates: 0.
    Original constant slots: 5; algebraic slots: 0.
    """
    source_file = 'immunology/wodarz_hamer_2007_a/model.py'
    source_state_dim = 2
    source_constant_dim = 5
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
        self.parameter_dim = 5

        self.parameters = (
            parameters
            if parameters is not None
            else {'r': 1, 'a': 0.2, 'k': 10, 'u': 0.5, 'eta': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'y': 0, 'v': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['r', 'a', 'k', 'u', 'eta']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['y', 'v']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        state_y = y[:, 0]
        v = y[:, 1]

        # Independent parameters (source constant order)
        r = theta[:, 0]  # constants[0]
        a = theta[:, 1]  # constants[1]
        k = theta[:, 2]  # constants[2]
        u = theta[:, 3]  # constants[3]
        eta = theta[:, 4]  # constants[4]

        # Initialize rates and algebraic variables, as in computeRates()
        d_state_y = torch.zeros_like(y[:, 0])
        d_v = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_state_y = torch.zeros_like(y[:, 0]) + (r * state_y * v * (1.0 - state_y / k) - a * state_y)
        d_v = torch.zeros_like(y[:, 0]) + (eta * state_y - u * v)

        return torch.stack(
            [d_state_y, d_v],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
