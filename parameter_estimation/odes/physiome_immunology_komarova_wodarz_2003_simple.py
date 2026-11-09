import torch
import torch.nn as nn
import math


class ImmunologyKomarovaWodarz2003SimpleODE(nn.Module):
    """
    PyTorch translation of immunology/komarova_wodarz_2003_simple/model.py.
    Dynamic states: 2; independent constants: 8; derived constants used by rates: 0.
    Original constant slots: 9; algebraic slots: 1.
    """
    source_file = 'immunology/komarova_wodarz_2003_simple/model.py'
    source_state_dim = 2
    source_constant_dim = 9
    algebraic_dim = 1
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
        self.parameter_dim = 8

        self.parameters = (
            parameters
            if parameters is not None
            else {'rs': 1, 'epsilon_s': 0.99, 'alpha': 0.1, 'u': 0.07, 'beta': 0.2, 'rm': 1.3, 'epsilon_m': 0.1, 'a': 0.5}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'S': 0.5, 'M': 0.5}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['rs', 'epsilon_s', 'alpha', 'u', 'beta', 'rm', 'epsilon_m', 'a']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['S', 'M']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        S = y[:, 0]
        M = y[:, 1]

        # Independent parameters (source constant order)
        rs = theta[:, 0]  # constants[0]
        epsilon_s = theta[:, 1]  # constants[1]
        alpha = theta[:, 2]  # constants[2]
        u = theta[:, 3]  # constants[3]
        beta = theta[:, 4]  # constants[4]
        rm = theta[:, 5]  # constants[5]
        epsilon_m = theta[:, 6]  # constants[6]
        a = theta[:, 7]  # constants[7]

        # Initialize rates and algebraic variables, as in computeRates()
        d_S = torch.zeros_like(y[:, 0])
        d_M = torch.zeros_like(y[:, 0])
        alg_0_phi = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_phi = S * rs / 1.0 * (1.0 - u / 1.0 * (1.0 - beta * epsilon_s / 1.0 - alpha / 1.0 * (1.0 - epsilon_s / 1.0))) + M * rm / 1.0 * (1.0 - u / 1.0 * (1.0 - beta * epsilon_m / 1.0 - alpha / 1.0 * (1.0 - epsilon_m / 1.0)))
        d_S = torch.zeros_like(y[:, 0]) + (rs * S * (1.0 - u * 1.0 + beta * epsilon_s * u / 1.0) + alpha * u * rs * S / 1.0 * (1.0 - epsilon_s) - alg_0_phi * S * 1.0)
        d_M = torch.zeros_like(y[:, 0]) + (rm * M * (1.0 - u / 1.0 + beta * epsilon_m * u / 1.0) + alpha * u * rm * M / 1.0 * (1.0 - epsilon_m) - alg_0_phi * M * 1.0)

        return torch.stack(
            [d_S, d_M],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
