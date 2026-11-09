import torch
import torch.nn as nn
import math


class SignalTransductionWolfPassargeSomsenSnoepHeinrichWesterhoff2000ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/wolf_passarge_somsen_snoep_heinrich_westerhoff_2000/model.py.
    Dynamic states: 9; independent constants: 19; derived constants used by rates: 0.
    Original constant slots: 19; algebraic slots: 13.
    """
    source_file = 'signal_transduction/wolf_passarge_somsen_snoep_heinrich_westerhoff_2000/model.py'
    source_state_dim = 9
    source_constant_dim = 19
    algebraic_dim = 13
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 9
        self.parameter_dim = 19

        self.parameters = (
            parameters
            if parameters is not None
            else {'Jo': 50, 'phi': 0.1, 'A': 4, 'N': 1, 'K_i': 1, 'k_1': 550, 'n': 4, 'k_2': 9.8, 'k_GAPDH_plus': 323.8, 'k_GAPDH_minus': 57823.1, 'k_PGK_plus': 76411.1, 'k_PGK_minus': 23.7, 'k_4': 80, 'k_5': 9.7, 'k_6': 2000, 'k_7': 28, 'k_8': 85.7, 'k_9': 80, 'k': 375}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'S1': 1.57981839, 'S2': 4.8279999, 'S3': 0.468657507, 'S4': 0.589391932, 'S5': 8.210114438, 'S6': 0.078042624, 'S6_ex': 0.025277594, 'A3': 1.972814237, 'N2': 0.384873894}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Jo', 'phi', 'A', 'N', 'K_i', 'k_1', 'n', 'k_2', 'k_GAPDH_plus', 'k_GAPDH_minus', 'k_PGK_plus', 'k_PGK_minus', 'k_4', 'k_5', 'k_6', 'k_7', 'k_8', 'k_9', 'k']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S6_ex', 'A3', 'N2']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (9)
        S1 = y[:, 0]
        S2 = y[:, 1]
        S3 = y[:, 2]
        S4 = y[:, 3]
        S5 = y[:, 4]
        S6 = y[:, 5]
        S6_ex = y[:, 6]
        A3 = y[:, 7]
        N2 = y[:, 8]

        # Independent parameters (source constant order)
        Jo = theta[:, 0]  # constants[0]
        phi = theta[:, 1]  # constants[1]
        A = theta[:, 2]  # constants[2]
        N = theta[:, 3]  # constants[3]
        K_i = theta[:, 4]  # constants[4]
        k_1 = theta[:, 5]  # constants[5]
        n = theta[:, 6]  # constants[6]
        k_2 = theta[:, 7]  # constants[7]
        k_GAPDH_plus = theta[:, 8]  # constants[8]
        k_GAPDH_minus = theta[:, 9]  # constants[9]
        k_PGK_plus = theta[:, 10]  # constants[10]
        k_PGK_minus = theta[:, 11]  # constants[11]
        k_4 = theta[:, 12]  # constants[12]
        k_5 = theta[:, 13]  # constants[13]
        k_6 = theta[:, 14]  # constants[14]
        k_7 = theta[:, 15]  # constants[15]
        k_8 = theta[:, 16]  # constants[16]
        k_9 = theta[:, 17]  # constants[17]
        k = theta[:, 18]  # constants[18]

        # Initialize rates and algebraic variables, as in computeRates()
        d_S1 = torch.zeros_like(y[:, 0])
        d_S2 = torch.zeros_like(y[:, 0])
        d_S3 = torch.zeros_like(y[:, 0])
        d_S4 = torch.zeros_like(y[:, 0])
        d_S5 = torch.zeros_like(y[:, 0])
        d_S6 = torch.zeros_like(y[:, 0])
        d_S6_ex = torch.zeros_like(y[:, 0])
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
        alg_8_v7 = torch.zeros_like(y[:, 0])
        alg_9_v6 = torch.zeros_like(y[:, 0])
        alg_10_v8 = torch.zeros_like(y[:, 0])
        alg_11_v9 = torch.zeros_like(y[:, 0])
        alg_12_J = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_2_f_A3 = (1.0 + (A3 / K_i) ** n) ** (-1.0)
        alg_3_v1 = k_1 * S1 * A3 * alg_2_f_A3
        d_S1 = torch.zeros_like(y[:, 0]) + (Jo - alg_3_v1)
        alg_4_v2 = k_2 * S2
        d_S2 = torch.zeros_like(y[:, 0]) + (alg_3_v1 - alg_4_v2)
        alg_0_A2 = A - A3
        alg_1_N1 = N - N2
        alg_5_v3 = (k_GAPDH_plus * k_PGK_plus * S3 * alg_1_N1 * alg_0_A2 - k_GAPDH_minus * k_PGK_minus * S4 * A3 * N2) / (k_GAPDH_minus * N2 + k_PGK_plus * alg_0_A2)
        alg_6_v4 = k_4 * S4 * alg_0_A2
        d_S4 = torch.zeros_like(y[:, 0]) + (alg_5_v3 - alg_6_v4)
        alg_7_v5 = k_5 * S5
        d_S5 = torch.zeros_like(y[:, 0]) + (alg_6_v4 - alg_7_v5)
        alg_8_v7 = k_7 * A3
        d_A3 = torch.zeros_like(y[:, 0]) + (alg_5_v3 + alg_6_v4 - (2.0 * alg_3_v1 + alg_8_v7))
        alg_10_v8 = k_8 * S3 * N2
        d_S3 = torch.zeros_like(y[:, 0]) + (2.0 * alg_4_v2 - (alg_5_v3 + alg_10_v8))
        alg_9_v6 = k_6 * S6 * N2
        d_N2 = torch.zeros_like(y[:, 0]) + (alg_5_v3 - (alg_9_v6 + alg_10_v8))
        alg_12_J = k * (S6 - S6_ex)
        d_S6 = torch.zeros_like(y[:, 0]) + (alg_7_v5 - (alg_9_v6 + alg_12_J))
        alg_11_v9 = k_9 * S6_ex
        d_S6_ex = torch.zeros_like(y[:, 0]) + (phi * alg_12_J - alg_11_v9)

        return torch.stack(
            [d_S1, d_S2, d_S3, d_S4, d_S5, d_S6, d_S6_ex, d_A3, d_N2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
