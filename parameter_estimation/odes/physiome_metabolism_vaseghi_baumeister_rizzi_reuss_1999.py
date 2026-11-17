import torch
import torch.nn as nn
import math


class MetabolismVaseghiBaumeisterRizziReuss1999ODE(nn.Module):
    """
    PyTorch translation of metabolism/vaseghi_baumeister_rizzi_reuss_1999/model.py.
    Dynamic states: 10; independent constants: 19; derived constants used by rates: 0.
    Original constant slots: 19; algebraic slots: 17.
    """
    source_file = 'metabolism/vaseghi_baumeister_rizzi_reuss_1999/model.py'
    source_state_dim = 10
    source_constant_dim = 19
    algebraic_dim = 17
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 10
        self.parameter_dim = 19

        self.parameters = (
            parameters
            if parameters is not None
            else {'mu': 1, 'K_NADP_1': 0.116, 'Ki_NADPH_1': 1.702, 'Ki_MgATP_1': 0.33, 'rmax_G6PDH': 44.19, 'K_NADP_2': 1.848, 'Ki_NADPH_2': 0.055, 'Ki_MgATP_2': 0.109, 'rmax_6PGDH': 0.654, 'rmax_R5PI': 0.57, 'rmax_Ru5PE': 0.85, 'rmax_TKL1': 3.24, 'C_GAP': 0.064, 'rmax_TAL': 3, 'rmax_TKL2': 10.5, 'K_PKDA': 0.0032, 'rmax_PKDA': 0.004, 'K_RPPK': 0.0034, 'rmax_RPPK': 0.003}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'C_6PG': 0.25, 'C_Ru5P': 0.033, 'C_R5P': 0.118, 'C_X5P': 0.041, 'C_S7P': 0.082, 'C_E4P': 0.029, 'C_G6P': 0.9, 'C_NADP': 0.168, 'C_NADPH': 0.168, 'C_MgATP': 2.3}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['mu', 'K_NADP_1', 'Ki_NADPH_1', 'Ki_MgATP_1', 'rmax_G6PDH', 'K_NADP_2', 'Ki_NADPH_2', 'Ki_MgATP_2', 'rmax_6PGDH', 'rmax_R5PI', 'rmax_Ru5PE', 'rmax_TKL1', 'C_GAP', 'rmax_TAL', 'rmax_TKL2', 'K_PKDA', 'rmax_PKDA', 'K_RPPK', 'rmax_RPPK']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['C_6PG', 'C_Ru5P', 'C_R5P', 'C_X5P', 'C_S7P', 'C_E4P', 'C_G6P', 'C_NADP', 'C_NADPH', 'C_MgATP']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (10)
        C_6PG = y[:, 0]
        C_Ru5P = y[:, 1]
        C_R5P = y[:, 2]
        C_X5P = y[:, 3]
        C_S7P = y[:, 4]
        C_E4P = y[:, 5]
        C_G6P = y[:, 6]
        C_NADP = y[:, 7]
        C_NADPH = y[:, 8]
        C_MgATP = y[:, 9]

        # Independent parameters (source constant order)
        mu = theta[:, 0]  # constants[0]
        K_NADP_1 = theta[:, 1]  # constants[1]
        Ki_NADPH_1 = theta[:, 2]  # constants[2]
        Ki_MgATP_1 = theta[:, 3]  # constants[3]
        rmax_G6PDH = theta[:, 4]  # constants[4]
        K_NADP_2 = theta[:, 5]  # constants[5]
        Ki_NADPH_2 = theta[:, 6]  # constants[6]
        Ki_MgATP_2 = theta[:, 7]  # constants[7]
        rmax_6PGDH = theta[:, 8]  # constants[8]
        rmax_R5PI = theta[:, 9]  # constants[9]
        rmax_Ru5PE = theta[:, 10]  # constants[10]
        rmax_TKL1 = theta[:, 11]  # constants[11]
        C_GAP = theta[:, 12]  # constants[12]
        rmax_TAL = theta[:, 13]  # constants[13]
        rmax_TKL2 = theta[:, 14]  # constants[14]
        K_PKDA = theta[:, 15]  # constants[15]
        rmax_PKDA = theta[:, 16]  # constants[16]
        K_RPPK = theta[:, 17]  # constants[17]
        rmax_RPPK = theta[:, 18]  # constants[18]

        # Initialize rates and algebraic variables, as in computeRates()
        d_C_6PG = torch.zeros_like(y[:, 0])
        d_C_Ru5P = torch.zeros_like(y[:, 0])
        d_C_R5P = torch.zeros_like(y[:, 0])
        d_C_X5P = torch.zeros_like(y[:, 0])
        d_C_S7P = torch.zeros_like(y[:, 0])
        d_C_E4P = torch.zeros_like(y[:, 0])
        d_C_G6P = torch.zeros_like(y[:, 0])
        d_C_NADP = torch.zeros_like(y[:, 0])
        d_C_NADPH = torch.zeros_like(y[:, 0])
        d_C_MgATP = torch.zeros_like(y[:, 0])
        alg_0_dC_G6P_dt = torch.zeros_like(y[:, 0])
        alg_1_I_NADPH_1 = torch.zeros_like(y[:, 0])
        alg_2_I_MgATP_1 = torch.zeros_like(y[:, 0])
        alg_3_rG6PDH = torch.zeros_like(y[:, 0])
        alg_4_I_NADPH_2 = torch.zeros_like(y[:, 0])
        alg_6_I_MgATP_2 = torch.zeros_like(y[:, 0])
        alg_9_r6PGDH = torch.zeros_like(y[:, 0])
        alg_10_rR5PI = torch.zeros_like(y[:, 0])
        alg_11_rRu5PE = torch.zeros_like(y[:, 0])
        alg_12_rTKL1 = torch.zeros_like(y[:, 0])
        alg_13_rRPPK = torch.zeros_like(y[:, 0])
        alg_14_rTAL = torch.zeros_like(y[:, 0])
        alg_15_rTKL2 = torch.zeros_like(y[:, 0])
        alg_16_rPKDA = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_C_NADP = torch.zeros_like(y[:, 0]) + (-1.48 / (9.17 + 16.1 * t + 0.48 * t ** 2.0) + 1.48 * t / ((9.17 + 16.1 * t + 0.48 * t ** 2.0) * (9.17 + 16.1 * t + 0.48 * t ** 2.0)) * (16.1 + 0.96 * t))
        d_C_NADPH = torch.zeros_like(y[:, 0]) + (0.516 / (25.39 + 0.37 * t + 0.5 * t ** 2.0) - 0.516 * t / ((25.39 + 0.37 * t + 0.5 * t ** 2.0) * (25.39 + 0.37 * t + 0.5 * t ** 2.0)) * (0.37 + 1.0 * t))
        d_C_MgATP = torch.zeros_like(y[:, 0]) + (29.83 / (29.77 + 13.42 * t + 0.05 * t ** 2.0) - 29.83 * t / ((29.77 + 13.42 * t + 0.05 * t ** 2.0) * (29.77 + 13.42 * t + 0.05 * t ** 2.0)) * (13.42 + 0.1 * t))
        alg_0_dC_G6P_dt = 44.1 / (48.0 + 1.0 * t + 0.45 * t ** 2.0) + 44.1 * t / ((48.0 + 1.0 * t + 0.45 * t ** 2.0) * (48.0 + 1.0 * t + 0.45 * t ** 2.0)) * (1.0 + 0.9 * t)
        d_C_G6P = torch.zeros_like(y[:, 0]) + (alg_0_dC_G6P_dt)
        alg_1_I_NADPH_1 = 1.0 + C_NADPH / Ki_NADPH_1
        alg_2_I_MgATP_1 = 1.0 + C_MgATP / Ki_MgATP_1
        alg_3_rG6PDH = rmax_G6PDH * (C_NADP / ((C_NADP + K_NADP_1 * alg_1_I_NADPH_1) * alg_2_I_MgATP_1))
        alg_4_I_NADPH_2 = 1.0 + C_NADPH / Ki_NADPH_2
        alg_6_I_MgATP_2 = 1.0 + C_MgATP / Ki_MgATP_2
        alg_9_r6PGDH = rmax_6PGDH * (C_NADP / ((C_NADP + K_NADP_2 * alg_4_I_NADPH_2) * alg_6_I_MgATP_2))
        d_C_6PG = torch.zeros_like(y[:, 0]) + (alg_3_rG6PDH - (alg_9_r6PGDH + mu * C_6PG))
        alg_11_rRu5PE = 1.0 * rmax_Ru5PE * C_Ru5P
        alg_10_rR5PI = 1.0 * rmax_R5PI * C_Ru5P
        d_C_Ru5P = torch.zeros_like(y[:, 0]) + (alg_9_r6PGDH - (alg_10_rR5PI + alg_11_rRu5PE + mu * C_Ru5P))
        alg_12_rTKL1 = 1.0 * rmax_TKL1 * C_X5P * C_R5P
        alg_13_rRPPK = 1.0 * rmax_RPPK * (C_R5P / (C_R5P + K_RPPK))
        d_C_R5P = torch.zeros_like(y[:, 0]) + (alg_10_rR5PI - (alg_12_rTKL1 + alg_13_rRPPK + mu * C_R5P))
        alg_14_rTAL = 1.0 * rmax_TAL * C_GAP * C_S7P
        d_C_S7P = torch.zeros_like(y[:, 0]) + (alg_12_rTKL1 - (alg_14_rTAL + mu * C_S7P))
        alg_15_rTKL2 = 1.0 * rmax_TKL2 * C_E4P * C_X5P
        d_C_X5P = torch.zeros_like(y[:, 0]) + (alg_11_rRu5PE - (alg_12_rTKL1 + alg_15_rTKL2 + mu * C_X5P))
        alg_16_rPKDA = 1.0 * rmax_PKDA * (C_E4P / (C_E4P + K_PKDA))
        d_C_E4P = torch.zeros_like(y[:, 0]) + (alg_14_rTAL - (alg_15_rTKL2 + alg_16_rPKDA + mu * C_E4P))

        return torch.stack(
            [d_C_6PG, d_C_Ru5P, d_C_R5P, d_C_X5P, d_C_S7P, d_C_E4P, d_C_G6P, d_C_NADP, d_C_NADPH, d_C_MgATP],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
