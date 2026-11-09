import torch
import torch.nn as nn
import math


class ImmunologyBonhoefferRembiszewskiOrtizNixon2000AODE(nn.Module):
    """
    PyTorch translation of immunology/bonhoeffer_rembiszewski_ortiz_nixon_2000_a/model.py.
    Dynamic states: 3; independent constants: 8; derived constants used by rates: 0.
    Original constant slots: 8; algebraic slots: 0.
    """
    source_file = 'immunology/bonhoeffer_rembiszewski_ortiz_nixon_2000_a/model.py'
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
            else {'s': 10, 'dT': 0.01, 'b': 0.001, 'p': 0.05, 'dI': 0.3, 'c': 0.05, 'dE': 0.1, 'K': 0.1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'T': 1, 'I': 1, 'E': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['s', 'dT', 'b', 'p', 'dI', 'c', 'dE', 'K']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['T', 'I', 'E']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        T = y[:, 0]
        I = y[:, 1]
        E = y[:, 2]

        # Independent parameters (source constant order)
        s = theta[:, 0]  # constants[0]
        dT = theta[:, 1]  # constants[1]
        b = theta[:, 2]  # constants[2]
        p = theta[:, 3]  # constants[3]
        dI = theta[:, 4]  # constants[4]
        c = theta[:, 5]  # constants[5]
        dE = theta[:, 6]  # constants[6]
        K = theta[:, 7]  # constants[7]

        # Initialize rates and algebraic variables, as in computeRates()
        d_T = torch.zeros_like(y[:, 0])
        d_I = torch.zeros_like(y[:, 0])
        d_E = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_T = torch.zeros_like(y[:, 0]) + (s - (dT * T + b * T * I))
        d_I = torch.zeros_like(y[:, 0]) + (b * T * I - (dI * I + p * I * E))
        d_E = torch.zeros_like(y[:, 0]) + (c * I / ((I + K) * E) - dE * E)

        return torch.stack(
            [d_T, d_I, d_E],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
