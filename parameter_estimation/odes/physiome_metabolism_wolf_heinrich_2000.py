import torch
import torch.nn as nn
import math


class MetabolismWolfHeinrich2000ODE(nn.Module):
    """
    PyTorch translation of metabolism/wolf_heinrich_2000/model.py.
    Dynamic states: 7; independent constants: 14; derived constants used by rates: 0.
    Original constant slots: 14; algebraic slots: 11.
    """
    source_file = 'metabolism/wolf_heinrich_2000/model.py'
    source_state_dim = 7
    source_constant_dim = 14
    algebraic_dim = 11
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 7
        self.parameter_dim = 14

        self.parameters = (
            parameters
            if parameters is not None
            else {'Jo': 3, 'phi': 0.1, 'A': 4, 'N': 1, 'K_I': 0.52, 'k_1': 100, 'q': 4, 'k_2': 6, 'k_3': 16, 'k_4': 100, 'k_5': 1.28, 'k_6': 12, 'k': 1.3, 'kappa': 13}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'S1': 5.8, 'S2': 0.9, 'S3': 0.2, 'S4': 0.2, 'S4_ex': 0.1, 'A3': 2.4, 'N2': 0.1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Jo', 'phi', 'A', 'N', 'K_I', 'k_1', 'q', 'k_2', 'k_3', 'k_4', 'k_5', 'k_6', 'k', 'kappa']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['S1', 'S2', 'S3', 'S4', 'S4_ex', 'A3', 'N2']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (7)
        S1 = y[:, 0]
        S2 = y[:, 1]
        S3 = y[:, 2]
        S4 = y[:, 3]
        S4_ex = y[:, 4]
        A3 = y[:, 5]
        N2 = y[:, 6]

        # Independent parameters (source constant order)
        Jo = theta[:, 0]  # constants[0]
        phi = theta[:, 1]  # constants[1]
        A = theta[:, 2]  # constants[2]
        N = theta[:, 3]  # constants[3]
        K_I = theta[:, 4]  # constants[4]
        k_1 = theta[:, 5]  # constants[5]
        q = theta[:, 6]  # constants[6]
        k_2 = theta[:, 7]  # constants[7]
        k_3 = theta[:, 8]  # constants[8]
        k_4 = theta[:, 9]  # constants[9]
        k_5 = theta[:, 10]  # constants[10]
        k_6 = theta[:, 11]  # constants[11]
        k = theta[:, 12]  # constants[12]
        kappa = theta[:, 13]  # constants[13]

        # Initialize rates and algebraic variables, as in computeRates()
        d_S1 = torch.zeros_like(y[:, 0])
        d_S2 = torch.zeros_like(y[:, 0])
        d_S3 = torch.zeros_like(y[:, 0])
        d_S4 = torch.zeros_like(y[:, 0])
        d_S4_ex = torch.zeros_like(y[:, 0])
        d_A3 = torch.zeros_like(y[:, 0])
        d_N2 = torch.zeros_like(y[:, 0])
        alg_0_A2 = torch.zeros_like(y[:, 0])
        alg_1_N1 = torch.zeros_like(y[:, 0])
        alg_2_f_A3 = torch.zeros_like(y[:, 0])
        alg_3_v1 = torch.zeros_like(y[:, 0])
        alg_4_v2 = torch.zeros_like(y[:, 0])
        alg_5_v3 = torch.zeros_like(y[:, 0])
        alg_6_v4 = torch.zeros_like(y[:, 0])
        alg_7_v5 = torch.zeros_like(y[:, 0])
        alg_8_v6 = torch.zeros_like(y[:, 0])
        alg_9_v7 = torch.zeros_like(y[:, 0])
        alg_10_J = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_2_f_A3 = (1.0 + (A3 / K_I) ** q) ** (-1.0)
        alg_3_v1 = k_1 * S1 * A3 * alg_2_f_A3
        d_S1 = torch.zeros_like(y[:, 0]) + (Jo - alg_3_v1)
        alg_1_N1 = N - N2
        alg_4_v2 = k_2 * S2 * alg_1_N1
        alg_0_A2 = A - A3
        alg_5_v3 = k_3 * S3 * alg_0_A2
        d_S3 = torch.zeros_like(y[:, 0]) + (alg_4_v2 - alg_5_v3)
        alg_7_v5 = k_5 * A3
        d_A3 = torch.zeros_like(y[:, 0]) + (2.0 * alg_5_v3 - (2.0 * alg_3_v1 + alg_7_v5))
        alg_8_v6 = k_6 * S2 * N2
        d_S2 = torch.zeros_like(y[:, 0]) + (2.0 * alg_3_v1 - (alg_4_v2 + alg_8_v6))
        alg_6_v4 = k_4 * S4 * N2
        d_N2 = torch.zeros_like(y[:, 0]) + (alg_4_v2 - (alg_6_v4 + alg_8_v6))
        alg_10_J = kappa * (S4 - S4_ex)
        d_S4 = torch.zeros_like(y[:, 0]) + (alg_5_v3 - (alg_6_v4 + alg_10_J))
        alg_9_v7 = k * S4_ex
        d_S4_ex = torch.zeros_like(y[:, 0]) + (phi * alg_10_J - alg_9_v7)

        return torch.stack(
            [d_S1, d_S2, d_S3, d_S4, d_S4_ex, d_A3, d_N2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
