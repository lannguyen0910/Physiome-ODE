import torch
import torch.nn as nn
import math


class EndocrineJelicCupicKolaranic2005Fig5ODE(nn.Module):
    """
    PyTorch translation of endocrine/jelic_cupic_kolaranic_2005_Fig5/model.py.
    Dynamic states: 2; independent constants: 12; derived constants used by rates: 1.
    Original constant slots: 15; algebraic slots: 5.
    """
    source_file = 'endocrine/jelic_cupic_kolaranic_2005_Fig5/model.py'
    source_state_dim = 2
    source_constant_dim = 15
    algebraic_dim = 5
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
            else {'alpha': 0.008, 'beta': 1.485, 'gamma': 11.385, 'L': 0.1, 'k2': 0.0006, 'k3': 4.8e-06, 'k6': 0.000891, 'k7': 0.006831, 'k0': 8.7831e-11, 'k4': 2100000000000, 'km': 6.9001e-14, 'K': 30}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'a': 4.39927, 'g': 1.96477}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['alpha', 'beta', 'gamma', 'L', 'k2', 'k3', 'k6', 'k7', 'k0', 'k4', 'km', 'K']
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
        gamma = theta[:, 2]  # constants[2]
        L = theta[:, 3]  # constants[3]
        k2 = theta[:, 4]  # constants[4]
        k3 = theta[:, 5]  # constants[5]
        k6 = theta[:, 6]  # constants[6]
        k7 = theta[:, 7]  # constants[7]
        k0 = theta[:, 8]  # constants[8]
        k4 = theta[:, 9]  # constants[9]
        km = theta[:, 10]  # constants[10]
        K = theta[:, 11]  # constants[11]

        # Derived constants from the original initConsts()
        C_0 = k2  # constants[12]

        # Initialize rates and algebraic variables, as in computeRates()
        d_a = torch.zeros_like(y[:, 0])
        d_g = torch.zeros_like(y[:, 0])
        alg_1_time = torch.zeros_like(y[:, 0])
        alg_3_P = torch.zeros_like(y[:, 0])
        alg_4_KP = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_g = torch.zeros_like(y[:, 0]) + ((1.0 - alpha) * a + a * g ** 2.0 - (L + gamma * g))
        alg_1_time = t / C_0
        alg_3_P = 0.735 + 0.1563 * torch.sin(2.0 * 3.14159 * alg_1_time / (24.0 * 3600.0)) + 0.3127 * torch.abs(torch.sin(3.14159 * alg_1_time / (24.0 * 3600.0)))
        alg_4_KP = K * alg_3_P
        d_a = torch.zeros_like(y[:, 0]) + (alg_4_KP - ((1.0 + alpha + beta) * a + a * g ** 2.0))

        return torch.stack(
            [d_a, d_g],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
