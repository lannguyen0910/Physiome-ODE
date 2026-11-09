import torch
import torch.nn as nn
import math


class EndocrineKomarova2003bODE(nn.Module):
    """
    PyTorch translation of endocrine/komarova_2003b/model.py.
    Dynamic states: 3; independent constants: 10; derived constants used by rates: 3.
    Original constant slots: 13; algebraic slots: 2.
    """
    source_file = 'endocrine/komarova_2003b/model.py'
    source_state_dim = 3
    source_constant_dim = 13
    algebraic_dim = 2
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
        self.parameter_dim = 10

        self.parameters = (
            parameters
            if parameters is not None
            else {'alpha1': 3, 'beta1': 0.2, 'g11': 1.1, 'g21': -0.5, 'alpha2': 4, 'beta2': 0.02, 'g12': 1, 'g22': 0, 'k1': 0.093, 'k2': 0.0008}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'x1': 10.06066, 'x2': 212.132, 'z': 100}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['alpha1', 'beta1', 'g11', 'g21', 'alpha2', 'beta2', 'g12', 'g22', 'k1', 'k2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['x1', 'x2', 'z']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        x1 = y[:, 0]
        x2 = y[:, 1]
        z = y[:, 2]

        # Independent parameters (source constant order)
        alpha1 = theta[:, 0]  # constants[0]
        beta1 = theta[:, 1]  # constants[1]
        g11 = theta[:, 2]  # constants[2]
        g21 = theta[:, 3]  # constants[3]
        alpha2 = theta[:, 4]  # constants[4]
        beta2 = theta[:, 5]  # constants[5]
        g12 = theta[:, 6]  # constants[6]
        g22 = theta[:, 7]  # constants[7]
        k1 = theta[:, 8]  # constants[8]
        k2 = theta[:, 9]  # constants[9]

        # Derived constants from the original initConsts()
        gamma = g12 * g21 - (1.0 - g11) * (1.0 - g22)  # constants[10]
        x1_bar = (beta1 / alpha1) ** ((1.0 - g22) / gamma) * (beta2 / alpha2) ** (g21 / gamma)  # constants[11]
        x2_bar = (beta1 / alpha1) ** (g12 / gamma) * (beta2 / alpha2) ** ((1.0 - g11) / gamma)  # constants[12]

        # Initialize rates and algebraic variables, as in computeRates()
        d_x1 = torch.zeros_like(y[:, 0])
        d_x2 = torch.zeros_like(y[:, 0])
        d_z = torch.zeros_like(y[:, 0])
        alg_0_y1 = torch.zeros_like(y[:, 0])
        alg_1_y2 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_x1 = torch.zeros_like(y[:, 0]) + (alpha1 * x1 ** g11 * x2 ** g21 - beta1 * x1)
        d_x2 = torch.zeros_like(y[:, 0]) + (alpha2 * x1 ** g12 * x2 ** g22 - beta2 * x2)
        alg_0_y1 = torch.where(x1 > x1_bar, x1 - x1_bar, 0.0)
        alg_1_y2 = torch.where(x2 > x2_bar, x2 - x2_bar, 0.0)
        d_z = torch.zeros_like(y[:, 0]) + (k2 * alg_1_y2 - k1 * alg_0_y1)

        return torch.stack(
            [d_x1, d_x2, d_z],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
