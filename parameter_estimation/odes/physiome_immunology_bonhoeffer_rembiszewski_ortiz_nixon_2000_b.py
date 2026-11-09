import torch
import torch.nn as nn
import math


class ImmunologyBonhoefferRembiszewskiOrtizNixon2000BODE(nn.Module):
    """
    PyTorch translation of immunology/bonhoeffer_rembiszewski_ortiz_nixon_2000_b/model.py.
    Dynamic states: 4; independent constants: 13; derived constants used by rates: 0.
    Original constant slots: 13; algebraic slots: 0.
    """
    source_file = 'immunology/bonhoeffer_rembiszewski_ortiz_nixon_2000_b/model.py'
    source_state_dim = 4
    source_constant_dim = 13
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

        self.state_dim = 4
        self.parameter_dim = 13

        self.parameters = (
            parameters
            if parameters is not None
            else {'s': 10, 'dT': 0.01, 'b': 0.001, 'p': 0.05, 'dI': 0.3, 'ql': 0.001, 'qa': 0.001, 'al': 0.01, 'c': 0.3, 'dE': 0.1, 'K': 0.1, 'dE_2': 0.25, 'K_2': 5}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'T': 1, 'I': 1, 'E': 1, 'Il': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['s', 'dT', 'b', 'p', 'dI', 'ql', 'qa', 'al', 'c', 'dE', 'K', 'dE_2', 'K_2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['T', 'I', 'E', 'Il']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        T = y[:, 0]
        I = y[:, 1]
        E = y[:, 2]
        Il = y[:, 3]

        # Independent parameters (source constant order)
        s = theta[:, 0]  # constants[0]
        dT = theta[:, 1]  # constants[1]
        b = theta[:, 2]  # constants[2]
        p = theta[:, 3]  # constants[3]
        dI = theta[:, 4]  # constants[4]
        ql = theta[:, 5]  # constants[5]
        qa = theta[:, 6]  # constants[6]
        al = theta[:, 7]  # constants[7]
        c = theta[:, 8]  # constants[8]
        dE = theta[:, 9]  # constants[9]
        K = theta[:, 10]  # constants[10]
        dE_2 = theta[:, 11]  # constants[11]
        K_2 = theta[:, 12]  # constants[12]

        # Initialize rates and algebraic variables, as in computeRates()
        d_T = torch.zeros_like(y[:, 0])
        d_I = torch.zeros_like(y[:, 0])
        d_E = torch.zeros_like(y[:, 0])
        d_Il = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_T = torch.zeros_like(y[:, 0]) + (s - (dT * T + b * T * I))
        d_I = torch.zeros_like(y[:, 0]) + (b * T * I - (dI * I + p * E * I + ql * I) + qa * I)
        d_Il = torch.zeros_like(y[:, 0]) + (ql * I - (al * Il + qa * Il))
        d_E = torch.zeros_like(y[:, 0]) + (c * E * I / (I + K) - (dE * E + dE_2 * E * I / (K_2 + I)))

        return torch.stack(
            [d_T, d_I, d_E, d_Il],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
