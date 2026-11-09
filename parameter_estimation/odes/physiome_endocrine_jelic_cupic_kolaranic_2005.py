import torch
import torch.nn as nn
import math


class EndocrineJelicCupicKolaranic2005ODE(nn.Module):
    """
    PyTorch translation of endocrine/jelic_cupic_kolaranic_2005/model.py.
    Dynamic states: 2; independent constants: 7; derived constants used by rates: 5.
    Original constant slots: 15; algebraic slots: 3.
    """
    source_file = 'endocrine/jelic_cupic_kolaranic_2005/model.py'
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
        self.parameter_dim = 7

        self.parameters = (
            parameters
            if parameters is not None
            else {'k2': 0.0006, 'k3': 4.8e-06, 'k6': 0.000891, 'k7': 0.006831, 'k0': 8.7831e-11, 'k4': 2100000000000, 'km': 6.9001e-14}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'a': 7.68e-08, 'g': 3.43e-08}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k2', 'k3', 'k6', 'k7', 'k0', 'k4', 'km']
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
        k2 = theta[:, 0]  # constants[0]
        k3 = theta[:, 1]  # constants[1]
        k6 = theta[:, 2]  # constants[2]
        k7 = theta[:, 3]  # constants[3]
        k0 = theta[:, 4]  # constants[4]
        k4 = theta[:, 5]  # constants[5]
        km = theta[:, 6]  # constants[6]

        # Derived constants from the original initConsts()
        alpha = k3 / k2  # constants[8]
        beta = k6 / k2  # constants[9]
        K = (k0 ** 2.0 * k4 / k2 ** 3.0) ** (1.0 / 2)  # constants[10]
        gamma = k7 / k2  # constants[12]
        L = (km ** 2.0 * k4 / k2 ** 3.0) ** (1.0 / 2)  # constants[13]

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
