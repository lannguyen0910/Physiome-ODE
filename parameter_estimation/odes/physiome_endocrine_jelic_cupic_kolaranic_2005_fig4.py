import torch
import torch.nn as nn
import math


class EndocrineJelicCupicKolaranic2005Fig4ODE(nn.Module):
    """
    PyTorch translation of endocrine/jelic_cupic_kolaranic_2005_Fig4/model.py.
    Dynamic states: 2; independent constants: 12; derived constants used by rates: 0.
    Original constant slots: 15; algebraic slots: 3.
    """
    source_file = 'endocrine/jelic_cupic_kolaranic_2005_Fig4/model.py'
    source_state_dim = 2
    source_constant_dim = 15
    algebraic_dim = 3
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
        self.parameter_dim = 12

        self.parameters = (
            parameters
            if parameters is not None
            else {'alpha': 0.008, 'beta': 1.485, 'K': 30, 'gamma': 11.385, 'L': 0.1, 'k2': 0.0006, 'k3': 4.8e-06, 'k6': 0.000891, 'k7': 0.006831, 'k0': 8.7831e-11, 'k4': 2100000000000, 'km': 6.9001e-14}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'a': 4.39927, 'g': 1.96477}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['alpha', 'beta', 'K', 'gamma', 'L', 'k2', 'k3', 'k6', 'k7', 'k0', 'k4', 'km']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['a', 'g']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        a = y[:, 0]
        g = y[:, 1]

        # Independent parameters (source constant order)
        alpha = theta[:, 0]  # constants[0]
        beta = theta[:, 1]  # constants[1]
        K = theta[:, 2]  # constants[2]
        gamma = theta[:, 3]  # constants[3]
        L = theta[:, 4]  # constants[4]
        k2 = theta[:, 5]  # constants[5]
        k3 = theta[:, 6]  # constants[6]
        k6 = theta[:, 7]  # constants[7]
        k7 = theta[:, 8]  # constants[8]
        k0 = theta[:, 9]  # constants[9]
        k4 = theta[:, 10]  # constants[10]
        km = theta[:, 11]  # constants[11]

        # Initialize rates and algebraic variables, as in computeRates()
        d_a = torch.zeros_like(y[:, 0])
        d_g = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_a = torch.zeros_like(y[:, 0]) + (K - ((1.0 + alpha + beta) * a + a * g ** 2.0))
        d_g = torch.zeros_like(y[:, 0]) + ((1.0 - alpha) * a + a * g ** 2.0 - (L + gamma * g))

        return torch.stack(
            [d_a, d_g],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
