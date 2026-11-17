import torch
import torch.nn as nn
import math


class ImmunologyRevillaGarciaRamos2003ODE(nn.Module):
    """
    PyTorch translation of immunology/revilla_garcia-ramos_2003/model.py.
    Dynamic states: 5; independent constants: 10; derived constants used by rates: 0.
    Original constant slots: 10; algebraic slots: 0.
    """
    source_file = 'immunology/revilla_garcia-ramos_2003/model.py'
    source_state_dim = 5
    source_constant_dim = 10
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

        self.state_dim = 5
        self.parameter_dim = 10

        self.parameters = (
            parameters
            if parameters is not None
            else {'lamda': 2, 'd': 0.01, 'beta': 0.004, 'a': 0.33, 'alpha': 0.004, 'b': 2, 'k': 50, 'u': 2, 'c': 2000, 'q': 2}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'x': 3, 'v': 149, 'y': 6, 'w': 1, 'z': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['lamda', 'd', 'beta', 'a', 'alpha', 'b', 'k', 'u', 'c', 'q']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['x', 'v', 'y', 'w', 'z']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        x = y[:, 0]
        v = y[:, 1]
        state_y = y[:, 2]
        w = y[:, 3]
        z = y[:, 4]

        # Independent parameters (source constant order)
        lamda = theta[:, 0]  # constants[0]
        d = theta[:, 1]  # constants[1]
        beta = theta[:, 2]  # constants[2]
        a = theta[:, 3]  # constants[3]
        alpha = theta[:, 4]  # constants[4]
        b = theta[:, 5]  # constants[5]
        k = theta[:, 6]  # constants[6]
        u = theta[:, 7]  # constants[7]
        c = theta[:, 8]  # constants[8]
        q = theta[:, 9]  # constants[9]

        # Initialize rates and algebraic variables, as in computeRates()
        d_x = torch.zeros_like(y[:, 0])
        d_v = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])
        d_w = torch.zeros_like(y[:, 0])
        d_z = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_x = torch.zeros_like(y[:, 0]) + (1.0 * lamda - (d * x + 1.0 * beta * x * v))
        d_state_y = torch.zeros_like(y[:, 0]) + (1.0 * beta * x * v - (a * state_y + 1.0 * alpha * w * state_y))
        d_z = torch.zeros_like(y[:, 0]) + (1.0 * alpha * w * state_y - b * z)
        d_v = torch.zeros_like(y[:, 0]) + (k * state_y - u * v)
        d_w = torch.zeros_like(y[:, 0]) + (c * z - q * w)

        return torch.stack(
            [d_x, d_v, d_state_y, d_w, d_z],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
