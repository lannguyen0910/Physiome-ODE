import torch
import torch.nn as nn
import math


class CalciumDynamicsWang2006ODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/wang_2006/model.py.
    Dynamic states: 5; independent constants: 27; derived constants used by rates: 0.
    Original constant slots: 27; algebraic slots: 10.
    """
    source_file = 'calcium_dynamics/wang_2006/model.py'
    source_state_dim = 5
    source_constant_dim = 27
    algebraic_dim = 10
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
        self.parameter_dim = 27

        self.parameters = (
            parameters
            if parameters is not None
            else {'k_0': 0.0001, 'k_1': 3.4, 'k_2': 4, 'k_3': 4.5, 'k_4': 1.2, 'k_5': 0.12, 'k_6': 14, 'k_7': 2, 'k_8': 10.5, 'k_9': 0.6, 'k_10': 3, 'k_11': 0.26, 'C_PLC_T': 0.01, 'K_D': 0.01, 'K_P': 0.004, 'K_R': 0.2, 'K_G': 0.025, 'K_S': 0.025, 'K_ER': 0.075, 'K_C1': 1, 'K_C2': 2, 'beta': 0.05, 'lambda_var': 0.001, 'rho': 0.2, 'n': 4, 'm': 2, 'w': 3}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'APLC': 0.001, 'C_cyt': 0.2, 'G': 0.001, 'IP_3': 0.001, 'C_ER': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k_0', 'k_1', 'k_2', 'k_3', 'k_4', 'k_5', 'k_6', 'k_7', 'k_8', 'k_9', 'k_10', 'k_11', 'C_PLC_T', 'K_D', 'K_P', 'K_R', 'K_G', 'K_S', 'K_ER', 'K_C1', 'K_C2', 'beta', 'lambda_var', 'rho', 'n', 'm', 'w']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['APLC', 'C_cyt', 'G', 'IP_3', 'C_ER']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        APLC = y[:, 0]
        C_cyt = y[:, 1]
        G = y[:, 2]
        IP_3 = y[:, 3]
        C_ER = y[:, 4]

        # Independent parameters (source constant order)
        k_0 = theta[:, 0]  # constants[0]
        k_1 = theta[:, 1]  # constants[1]
        k_2 = theta[:, 2]  # constants[2]
        k_3 = theta[:, 3]  # constants[3]
        k_4 = theta[:, 4]  # constants[4]
        k_5 = theta[:, 5]  # constants[5]
        k_6 = theta[:, 6]  # constants[6]
        k_7 = theta[:, 7]  # constants[7]
        k_8 = theta[:, 8]  # constants[8]
        k_9 = theta[:, 9]  # constants[9]
        k_10 = theta[:, 10]  # constants[10]
        k_11 = theta[:, 11]  # constants[11]
        C_PLC_T = theta[:, 12]  # constants[12]
        K_D = theta[:, 13]  # constants[13]
        K_P = theta[:, 14]  # constants[14]
        K_R = theta[:, 15]  # constants[15]
        K_G = theta[:, 16]  # constants[16]
        K_S = theta[:, 17]  # constants[17]
        K_ER = theta[:, 18]  # constants[18]
        K_C1 = theta[:, 19]  # constants[19]
        K_C2 = theta[:, 20]  # constants[20]
        beta = theta[:, 21]  # constants[21]
        lambda_var = theta[:, 22]  # constants[22]
        rho = theta[:, 23]  # constants[23]
        n = theta[:, 24]  # constants[24]
        m = theta[:, 25]  # constants[25]
        w = theta[:, 26]  # constants[26]

        # Initialize rates and algebraic variables, as in computeRates()
        d_APLC = torch.zeros_like(y[:, 0])
        d_C_cyt = torch.zeros_like(y[:, 0])
        d_G = torch.zeros_like(y[:, 0])
        d_IP_3 = torch.zeros_like(y[:, 0])
        d_C_ER = torch.zeros_like(y[:, 0])
        alg_0_R_IP_3 = torch.zeros_like(y[:, 0])
        alg_1_R_APLC = torch.zeros_like(y[:, 0])
        alg_2_R_Cyt1 = torch.zeros_like(y[:, 0])
        alg_3_R_G = torch.zeros_like(y[:, 0])
        alg_4_R_Cyt2 = torch.zeros_like(y[:, 0])
        alg_5_PLC = torch.zeros_like(y[:, 0])
        alg_6_R_ER = torch.zeros_like(y[:, 0])
        alg_7_DG = torch.zeros_like(y[:, 0])
        alg_8_R_PKC = torch.zeros_like(y[:, 0])
        alg_9_R_DG = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_IP_3 = torch.zeros_like(y[:, 0]) + (k_6 * APLC - k_7 * IP_3)
        alg_0_R_IP_3 = IP_3 ** 3.0 / (K_S ** 3.0 + IP_3 ** 3.0)
        alg_2_R_Cyt1 = C_cyt / (K_C1 + C_cyt)
        alg_4_R_Cyt2 = C_cyt / (K_C2 + C_cyt)
        alg_6_R_ER = C_ER ** w / (K_ER ** w + C_ER ** w)
        d_C_cyt = torch.zeros_like(y[:, 0]) + (beta * (rho * (k_8 * alg_0_R_IP_3 * alg_6_R_ER - k_9 * alg_2_R_Cyt1) - k_10 * alg_4_R_Cyt2 + k_11))
        d_C_ER = torch.zeros_like(y[:, 0]) + (lambda_var * (-k_8 * alg_0_R_IP_3 * alg_6_R_ER + k_9 * alg_2_R_Cyt1))
        alg_1_R_APLC = APLC / (K_P + APLC)
        alg_7_DG = IP_3
        alg_8_R_PKC = alg_7_DG / (K_D + alg_7_DG) * C_cyt / (K_R + C_cyt)
        d_G = torch.zeros_like(y[:, 0]) + (k_0 + k_1 * G - k_2 * alg_1_R_APLC * G - k_3 * alg_8_R_PKC * G)
        alg_3_R_G = G ** n / (K_G ** n + G ** n)
        alg_9_R_DG = alg_7_DG ** m / (K_D ** m + alg_7_DG ** m)
        alg_5_PLC = C_PLC_T - APLC
        d_APLC = torch.zeros_like(y[:, 0]) + (k_4 * alg_3_R_G * alg_9_R_DG * alg_5_PLC - k_5 * APLC)

        return torch.stack(
            [d_APLC, d_C_cyt, d_G, d_IP_3, d_C_ER],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
