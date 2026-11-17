import torch
import torch.nn as nn
import math


class ImmunologyDixitPerelson2004ODE(nn.Module):
    """
    PyTorch translation of immunology/dixit_perelson_2004/model.py.
    Dynamic states: 5; independent constants: 18; derived constants used by rates: 0.
    Original constant slots: 18; algebraic slots: 3.
    """
    source_file = 'immunology/dixit_perelson_2004/model.py'
    source_state_dim = 5
    source_constant_dim = 18
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

        self.state_dim = 5
        self.parameter_dim = 18

        self.parameters = (
            parameters
            if parameters is not None
            else {'lamda': 10000, 'd': 0.01, 'k': 2.4e-08, 'tau': 1.5, 'm': 0.01, 'delta': 0.01, 'N': 2500, 'c': 23, 'IC50': 9e-07, 'Vd': 28000, 'F': 1, 'D': 600, 'ka': 14.64, 'ke': 6.86, 'kacell': 24000, 'kecell': 1.1, 'H': 0.052, 'fb': 0.99}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'T': 1000000, 'VI': 1, 'T_2': 1, 'VNI': 2, 'Cc': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['lamda', 'd', 'k', 'tau', 'm', 'delta', 'N', 'c', 'IC50', 'Vd', 'F', 'D', 'ka', 'ke', 'kacell', 'kecell', 'H', 'fb']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['T', 'VI', 'T_2', 'VNI', 'Cc']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        T = y[:, 0]
        VI = y[:, 1]
        T_2 = y[:, 2]
        VNI = y[:, 3]
        Cc = y[:, 4]

        # Independent parameters (source constant order)
        lamda = theta[:, 0]  # constants[0]
        d = theta[:, 1]  # constants[1]
        k = theta[:, 2]  # constants[2]
        tau = theta[:, 3]  # constants[3]
        m = theta[:, 4]  # constants[4]
        delta = theta[:, 5]  # constants[5]
        N = theta[:, 6]  # constants[6]
        c = theta[:, 7]  # constants[7]
        IC50 = theta[:, 8]  # constants[8]
        Vd = theta[:, 9]  # constants[9]
        F = theta[:, 10]  # constants[10]
        D = theta[:, 11]  # constants[11]
        ka = theta[:, 12]  # constants[12]
        ke = theta[:, 13]  # constants[13]
        kacell = theta[:, 14]  # constants[14]
        kecell = theta[:, 15]  # constants[15]
        H = theta[:, 16]  # constants[16]
        fb = theta[:, 17]  # constants[17]

        # Initialize rates and algebraic variables, as in computeRates()
        d_T = torch.zeros_like(y[:, 0])
        d_VI = torch.zeros_like(y[:, 0])
        d_T_2 = torch.zeros_like(y[:, 0])
        d_VNI = torch.zeros_like(y[:, 0])
        d_Cc = torch.zeros_like(y[:, 0])
        alg_0_epsilon_PI = torch.zeros_like(y[:, 0])
        alg_1_Cb = torch.zeros_like(y[:, 0])
        alg_2_Cx = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_T = torch.zeros_like(y[:, 0]) + (lamda - (d * T + k * T * VI))
        d_T_2 = torch.zeros_like(y[:, 0]) + (k * T * (t - tau) * VI * (t - tau) * torch.exp(-m * tau) - delta * T_2)
        alg_0_epsilon_PI = Cc / (IC50 + Cc)
        d_VI = torch.zeros_like(y[:, 0]) + (N * delta * T_2 * (1.0 - alg_0_epsilon_PI) - c * VI)
        d_VNI = torch.zeros_like(y[:, 0]) + (N * delta * T_2 * alg_0_epsilon_PI - c * VNI)
        alg_1_Cb = F * D / Vd * (ka / (ke - ka)) * (torch.exp(-ka * t) - torch.exp(-ke * t))
        alg_2_Cx = torch.where((1.0 - fb) * H * alg_1_Cb - Cc > 0.0, (1.0 - fb) * H * alg_1_Cb - Cc, 0.0)
        d_Cc = torch.zeros_like(y[:, 0]) + (kacell * alg_2_Cx - kecell * Cc)

        return torch.stack(
            [d_T, d_VI, d_T_2, d_VNI, d_Cc],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
