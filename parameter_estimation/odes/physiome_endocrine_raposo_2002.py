import torch
import torch.nn as nn
import math


class EndocrineRaposo2002ODE(nn.Module):
    """
    PyTorch translation of endocrine/raposo_2002/model.py.
    Dynamic states: 11; independent constants: 38; derived constants used by rates: 15.
    Original constant slots: 57; algebraic slots: 7.
    """
    source_file = 'endocrine/raposo_2002/model.py'
    source_state_dim = 11
    source_constant_dim = 57
    algebraic_dim = 7
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 11
        self.parameter_dim = 38

        self.parameters = (
            parameters
            if parameters is not None
            else {'J_bp_Ca': 3.3, 'J_pu_Ca': 0.21, 'J_pu_P': 1.27, 'C_PTH_p': 3.85, 'k_PTH': 100, 'S_E': 1, 'k_E': 0.05, 'k_D': 0.1, 'k2_P': 1, 'J_P_ing': 1, 'Stoic_Ca_P': 0.464, 'k_Ca_i': 1, 'C_Ca_i': 1, 'C_Ca_p': 2.4, 'C_P_p': 1.2, 'k_Ca_b': 1, 'k3_P': 1, 'k4_P': 51.8, 'C_D_p': 90, 'Y_Ca_i_2plus_Max': 1, 'Y_PTH_p_2plus_Max': 1, 'Y_PTH_p_2minus_Max': 1, 'Y_Ca_p_1plus_minus_Max': 0.02, 'Y_i_D_p_1plus_minus_Max': 0.02, 'Y_PT_D_p_1plus_minus_Max': 0.01, 'Y_k_Jext_1plus_minus_Max': 0.01, 'X_R_PTH_Ca': 1, 'X_R_i_Ca': 1, 'X_R_PTH_D': 90, 'X_R_E_PTH': 3.85, 'X_R_i_D': 90, 'X_R_P_k_PTH': 90, 'b_PTH_Ca': 0.05, 'b_PTH_D': 0.03, 'b_E_PTH': 0.55, 'b_i_D': 0.03, 'a': 0.85, 'd': 0.15}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Q_Ca_p': 1, 'Q_P_p': 1, 'Q_P_c': 3226, 'Q_PTH_p': 1, 'Q_Ca_b': 100, 'Q_P_b': 1, 'Q_E_k': 1, 'Q_D_p': 1, 'Q_TCa_i': 1, 'Q_C_PT': 1, 'Q_TP_k': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['J_bp_Ca', 'J_pu_Ca', 'J_pu_P', 'C_PTH_p', 'k_PTH', 'S_E', 'k_E', 'k_D', 'k2_P', 'J_P_ing', 'Stoic_Ca_P', 'k_Ca_i', 'C_Ca_i', 'C_Ca_p', 'C_P_p', 'k_Ca_b', 'k3_P', 'k4_P', 'C_D_p', 'Y_Ca_i_2plus_Max', 'Y_PTH_p_2plus_Max', 'Y_PTH_p_2minus_Max', 'Y_Ca_p_1plus_minus_Max', 'Y_i_D_p_1plus_minus_Max', 'Y_PT_D_p_1plus_minus_Max', 'Y_k_Jext_1plus_minus_Max', 'X_R_PTH_Ca', 'X_R_i_Ca', 'X_R_PTH_D', 'X_R_E_PTH', 'X_R_i_D', 'X_R_P_k_PTH', 'b_PTH_Ca', 'b_PTH_D', 'b_E_PTH', 'b_i_D', 'a', 'd']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Q_Ca_p', 'Q_P_p', 'Q_P_c', 'Q_PTH_p', 'Q_Ca_b', 'Q_P_b', 'Q_E_k', 'Q_D_p', 'Q_TCa_i', 'Q_C_PT', 'Q_TP_k']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (11)
        Q_Ca_p = y[:, 0]
        Q_P_p = y[:, 1]
        Q_P_c = y[:, 2]
        Q_PTH_p = y[:, 3]
        Q_Ca_b = y[:, 4]
        Q_P_b = y[:, 5]
        Q_E_k = y[:, 6]
        Q_D_p = y[:, 7]
        Q_TCa_i = y[:, 8]
        Q_C_PT = y[:, 9]
        Q_TP_k = y[:, 10]

        # Independent parameters (source constant order)
        J_bp_Ca = theta[:, 0]  # constants[0]
        J_pu_Ca = theta[:, 1]  # constants[1]
        J_pu_P = theta[:, 2]  # constants[2]
        C_PTH_p = theta[:, 3]  # constants[3]
        k_PTH = theta[:, 4]  # constants[4]
        S_E = theta[:, 5]  # constants[5]
        k_E = theta[:, 6]  # constants[6]
        k_D = theta[:, 7]  # constants[7]
        k2_P = theta[:, 8]  # constants[8]
        J_P_ing = theta[:, 9]  # constants[9]
        Stoic_Ca_P = theta[:, 10]  # constants[10]
        k_Ca_i = theta[:, 11]  # constants[11]
        C_Ca_i = theta[:, 12]  # constants[12]
        C_Ca_p = theta[:, 13]  # constants[13]
        C_P_p = theta[:, 14]  # constants[14]
        k_Ca_b = theta[:, 15]  # constants[15]
        k3_P = theta[:, 16]  # constants[16]
        k4_P = theta[:, 17]  # constants[17]
        C_D_p = theta[:, 18]  # constants[18]
        Y_Ca_i_2plus_Max = theta[:, 19]  # constants[19]
        Y_PTH_p_2plus_Max = theta[:, 20]  # constants[20]
        Y_PTH_p_2minus_Max = theta[:, 21]  # constants[21]
        Y_Ca_p_1plus_minus_Max = theta[:, 22]  # constants[22]
        Y_i_D_p_1plus_minus_Max = theta[:, 23]  # constants[23]
        Y_PT_D_p_1plus_minus_Max = theta[:, 24]  # constants[24]
        Y_k_Jext_1plus_minus_Max = theta[:, 25]  # constants[25]
        X_R_PTH_Ca = theta[:, 26]  # constants[26]
        X_R_i_Ca = theta[:, 27]  # constants[27]
        X_R_PTH_D = theta[:, 28]  # constants[28]
        X_R_E_PTH = theta[:, 29]  # constants[29]
        X_R_i_D = theta[:, 30]  # constants[30]
        X_R_P_k_PTH = theta[:, 31]  # constants[31]
        b_PTH_Ca = theta[:, 32]  # constants[32]
        b_PTH_D = theta[:, 33]  # constants[33]
        b_E_PTH = theta[:, 34]  # constants[34]
        b_i_D = theta[:, 35]  # constants[35]
        a = theta[:, 36]  # constants[36]
        d = theta[:, 37]  # constants[37]

        # Derived constants from the original initConsts()
        J_bp_P = Stoic_Ca_P * J_bp_Ca  # constants[38]
        Y_Ca_i_2plus = Y_Ca_i_2plus_Max * (C_Ca_i / (C_Ca_i + X_R_i_Ca))  # constants[39]
        Y_Ca_p_1minus = Y_Ca_p_1plus_minus_Max * (a * (1.0 - torch.tanh(b_PTH_Ca * (C_Ca_p - X_R_PTH_Ca))) + d)  # constants[40]
        Y_i_D_p_1plus = Y_i_D_p_1plus_minus_Max * (a * (1.0 + torch.tanh(b_i_D * (C_D_p - X_R_i_D))) + d)  # constants[41]
        Y_PT_D_p_1plus = Y_PT_D_p_1plus_minus_Max * (a * (1.0 + torch.tanh(b_PTH_D * (C_D_p - X_R_PTH_D))) + d)  # constants[42]
        Y_k_Jext_1plus = Y_k_Jext_1plus_minus_Max * (a * (1.0 + torch.tanh(b_E_PTH * (1.0 * C_P_p - X_R_E_PTH))) + d)  # constants[43]
        Y_i_D_p_1minus = Y_i_D_p_1plus_minus_Max * (a * (1.0 - torch.tanh(b_i_D * (C_D_p - X_R_i_D))) + d)  # constants[46]
        Y_PT_D_p_1minus = Y_PT_D_p_1plus_minus_Max * (a * (1.0 - torch.tanh(b_PTH_D * (C_D_p - X_R_PTH_D))) + d)  # constants[47]
        J_ip_P = J_P_ing  # constants[48]
        k1_D = Y_i_D_p_1plus  # constants[49]
        k2_D = Y_i_D_p_1minus  # constants[50]
        k3_D = Y_PT_D_p_1plus  # constants[51]
        k4_D = Y_PT_D_p_1minus  # constants[52]
        k1_P = Y_k_Jext_1plus  # constants[53]
        J_pc_P = k4_P * 1.0 * C_P_p  # constants[55]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Q_Ca_p = torch.zeros_like(y[:, 0])
        d_Q_P_p = torch.zeros_like(y[:, 0])
        d_Q_P_c = torch.zeros_like(y[:, 0])
        d_Q_PTH_p = torch.zeros_like(y[:, 0])
        d_Q_Ca_b = torch.zeros_like(y[:, 0])
        d_Q_P_b = torch.zeros_like(y[:, 0])
        d_Q_E_k = torch.zeros_like(y[:, 0])
        d_Q_D_p = torch.zeros_like(y[:, 0])
        d_Q_TCa_i = torch.zeros_like(y[:, 0])
        d_Q_C_PT = torch.zeros_like(y[:, 0])
        d_Q_TP_k = torch.zeros_like(y[:, 0])
        alg_0_J_pb_Ca = torch.zeros_like(y[:, 0])
        alg_1_S_PTH = torch.zeros_like(y[:, 0])
        alg_3_J_ip_Ca = torch.zeros_like(y[:, 0])
        alg_4_J_pb_P = torch.zeros_like(y[:, 0])
        alg_6_J_cp_P = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Q_E_k = torch.zeros_like(y[:, 0]) + (S_E - k_E * Q_E_k)
        d_Q_D_p = torch.zeros_like(y[:, 0]) + (1.0 * Q_E_k - k_D * Q_D_p)
        d_Q_TCa_i = torch.zeros_like(y[:, 0]) + ((1.0 - Q_TCa_i) * k1_D - Q_TCa_i * k2_D)
        d_Q_C_PT = torch.zeros_like(y[:, 0]) + ((1.0 - Q_C_PT) * k3_D - Q_C_PT * k4_D)
        d_Q_TP_k = torch.zeros_like(y[:, 0]) + ((1.0 - Q_TP_k) * k1_P - Q_TP_k * k2_P)
        alg_1_S_PTH = 1.0 * Y_Ca_p_1minus * (k3_D * (1.0 - Q_C_PT) - k4_D * Q_C_PT)
        d_Q_PTH_p = torch.zeros_like(y[:, 0]) + (alg_1_S_PTH - k_PTH * 1.0 * C_PTH_p)
        alg_0_J_pb_Ca = k_Ca_b * Q_Ca_b
        d_Q_Ca_b = torch.zeros_like(y[:, 0]) + (alg_0_J_pb_Ca - J_bp_Ca)
        alg_3_J_ip_Ca = Y_Ca_i_2plus * (1.0 - Q_TCa_i) * 1.0 * k1_D + k_Ca_i * 1.0 * (C_Ca_i - C_Ca_p) - Q_TCa_i * k2_D
        d_Q_Ca_p = torch.zeros_like(y[:, 0]) + (J_bp_Ca + alg_3_J_ip_Ca - (alg_0_J_pb_Ca + J_pu_Ca))
        alg_4_J_pb_P = Stoic_Ca_P * alg_0_J_pb_Ca
        d_Q_P_b = torch.zeros_like(y[:, 0]) + (alg_4_J_pb_P - J_bp_P)
        alg_6_J_cp_P = k3_P * Q_P_c
        d_Q_P_p = torch.zeros_like(y[:, 0]) + (J_bp_P + J_ip_P + alg_6_J_cp_P - (alg_4_J_pb_P + J_pu_P + J_pc_P))
        d_Q_P_c = torch.zeros_like(y[:, 0]) + (J_pc_P - alg_6_J_cp_P)

        return torch.stack(
            [d_Q_Ca_p, d_Q_P_p, d_Q_P_c, d_Q_PTH_p, d_Q_Ca_b, d_Q_P_b, d_Q_E_k, d_Q_D_p, d_Q_TCa_i, d_Q_C_PT, d_Q_TP_k],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
