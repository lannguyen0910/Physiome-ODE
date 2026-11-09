import torch
import torch.nn as nn
import math


class ImmunologyNeumannLamDahariGretchWileyLaydenPerelson1998ODE(nn.Module):
    """
    PyTorch translation of immunology/neumann_lam_dahari_gretch_wiley_layden_perelson_1998/model.py.
    Dynamic states: 3; independent constants: 8; derived constants used by rates: 0.
    Original constant slots: 8; algebraic slots: 0.
    """
    source_file = 'immunology/neumann_lam_dahari_gretch_wiley_layden_perelson_1998/model.py'
    source_state_dim = 3
    source_constant_dim = 8
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
        self.parameter_dim = 8

        self.parameters = (
            parameters
            if parameters is not None
            else {'s': 10, 'd': 0.02, 'eta': 0.8, 'beta': 3e-07, 'delta': 0.5, 'epsilon': 0.99, 'p': 100, 'c': 5}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'T': 1000, 'V': 9400000, 'I': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['s', 'd', 'eta', 'beta', 'delta', 'epsilon', 'p', 'c']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['T', 'V', 'I']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        T = y[:, 0]
        V = y[:, 1]
        I = y[:, 2]

        # Independent parameters (source constant order)
        s = theta[:, 0]  # constants[0]
        d = theta[:, 1]  # constants[1]
        eta = theta[:, 2]  # constants[2]
        beta = theta[:, 3]  # constants[3]
        delta = theta[:, 4]  # constants[4]
        epsilon = theta[:, 5]  # constants[5]
        p = theta[:, 6]  # constants[6]
        c = theta[:, 7]  # constants[7]

        # Initialize rates and algebraic variables, as in computeRates()
        d_T = torch.zeros_like(y[:, 0])
        d_V = torch.zeros_like(y[:, 0])
        d_I = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_T = torch.zeros_like(y[:, 0]) + (s - (d * T + (1.0 - eta) * beta * V * T))
        d_I = torch.zeros_like(y[:, 0]) + ((1.0 - eta) * beta * V * T - delta * I)
        d_V = torch.zeros_like(y[:, 0]) + ((1.0 - epsilon) * p * I - c * V)

        return torch.stack(
            [d_T, d_V, d_I],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
