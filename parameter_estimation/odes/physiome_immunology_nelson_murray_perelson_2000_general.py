import torch
import torch.nn as nn
import math


class ImmunologyNelsonMurrayPerelson2000GeneralODE(nn.Module):
    """
    PyTorch translation of immunology/nelson_murray_perelson_2000_general/model.py.
    Dynamic states: 4; independent constants: 7; derived constants used by rates: 0.
    Original constant slots: 7; algebraic slots: 3.
    """
    source_file = 'immunology/nelson_murray_perelson_2000_general/model.py'
    source_state_dim = 4
    source_constant_dim = 7
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

        self.state_dim = 4
        self.parameter_dim = 7

        self.parameters = (
            parameters
            if parameters is not None
            else {'lambda_var': 5, 'delta_1': 0.03, 'delta': 0.5, 'np': 1, 'c': 3, 'k': 3.43e-05, 'N': 480}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'T': 180, 'VI': 134000, 'T_star': 3.6, 'VNI': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['lambda_var', 'delta_1', 'delta', 'np', 'c', 'k', 'N']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['T', 'VI', 'T_star', 'VNI']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        T = y[:, 0]
        VI = y[:, 1]
        T_star = y[:, 2]
        VNI = y[:, 3]

        # Independent parameters (source constant order)
        lambda_var = theta[:, 0]  # constants[0]
        delta_1 = theta[:, 1]  # constants[1]
        delta = theta[:, 2]  # constants[2]
        np = theta[:, 3]  # constants[3]
        c = theta[:, 4]  # constants[4]
        k = theta[:, 5]  # constants[5]
        N = theta[:, 6]  # constants[6]

        # Initialize rates and algebraic variables, as in computeRates()
        d_T = torch.zeros_like(y[:, 0])
        d_VI = torch.zeros_like(y[:, 0])
        d_T_star = torch.zeros_like(y[:, 0])
        d_VNI = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_T = torch.zeros_like(y[:, 0]) + (lambda_var * 1.0 - delta_1 * T / 1.0 - k * VI * T)
        d_T_star = torch.zeros_like(y[:, 0]) + (k * VI * T - delta / 1.0 * T_star)
        d_VI = torch.zeros_like(y[:, 0]) + ((1.0 - np) * N * delta * T_star / 1.0 - c * VI / 1.0)
        d_VNI = torch.zeros_like(y[:, 0]) + (np * N * delta * T_star / 1.0 - c * VNI / 1.0)

        return torch.stack(
            [d_T, d_VI, d_T_star, d_VNI],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
