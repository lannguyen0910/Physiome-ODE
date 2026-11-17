import torch
import torch.nn as nn
import math


class MechanicalConstitutiveLawsBental2006ODE(nn.Module):
    """
    PyTorch translation of mechanical_constitutive_laws/bental_2006/model.py.
    Dynamic states: 7; independent constants: 26; derived constants used by rates: 0.
    Original constant slots: 26; algebraic slots: 9.
    """
    source_file = 'mechanical_constitutive_laws/bental_2006/model.py'
    source_state_dim = 7
    source_constant_dim = 26
    algebraic_dim = 9
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
        self.parameter_dim = 26

        self.parameters = (
            parameters
            if parameters is not None
            else {'P_m': 760, 'omega': 1.256637, 'V_T': 0.41, 'E': 2.5, 'P_m_2': 760, 'E_2': 2.5, 'R': 1, 'D_o': 1.56e-05, 'D_c': 3.16e-05, 'P_w': 47, 'f_om': 0.21, 'f_cm': 0, 'V_D': 0.151, 'V_T_2': 0.41, 'L': 171200000, 'K_T': 10000, 'K_R': 3600000, 'sigma': 1.4e-06, 'V_c': 0.071, 'T_h': 0.002, 'l_2': 164000, 'r_2': 0.12, 'sigma_c': 3.3e-05, 'R_2': 1.2566370614359172, 'delta': 79.43282347242814, 'h': 3.981071705534969e-08}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V_A': 1, 'P_A': 760, 'f_o': 0.1368, 'f_c': 0.05263, 'p_o': 40, 'p_c': 46, 'z': 4.4219e-07}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['P_m', 'omega', 'V_T', 'E', 'P_m_2', 'E_2', 'R', 'D_o', 'D_c', 'P_w', 'f_om', 'f_cm', 'V_D', 'V_T_2', 'L', 'K_T', 'K_R', 'sigma', 'V_c', 'T_h', 'l_2', 'r_2', 'sigma_c', 'R_2', 'delta', 'h']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V_A', 'P_A', 'f_o', 'f_c', 'p_o', 'p_c', 'z']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (7)
        V_A = y[:, 0]
        P_A = y[:, 1]
        f_o = y[:, 2]
        f_c = y[:, 3]
        p_o = y[:, 4]
        p_c = y[:, 5]
        z = y[:, 6]

        # Independent parameters (source constant order)
        P_m = theta[:, 0]  # constants[0]
        omega = theta[:, 1]  # constants[1]
        V_T = theta[:, 2]  # constants[2]
        E = theta[:, 3]  # constants[3]
        P_m_2 = theta[:, 4]  # constants[4]
        E_2 = theta[:, 5]  # constants[5]
        R = theta[:, 6]  # constants[6]
        D_o = theta[:, 7]  # constants[7]
        D_c = theta[:, 8]  # constants[8]
        P_w = theta[:, 9]  # constants[9]
        f_om = theta[:, 10]  # constants[10]
        f_cm = theta[:, 11]  # constants[11]
        V_D = theta[:, 12]  # constants[12]
        V_T_2 = theta[:, 13]  # constants[13]
        L = theta[:, 14]  # constants[14]
        K_T = theta[:, 15]  # constants[15]
        K_R = theta[:, 16]  # constants[16]
        sigma = theta[:, 17]  # constants[17]
        V_c = theta[:, 18]  # constants[18]
        T_h = theta[:, 19]  # constants[19]
        l_2 = theta[:, 20]  # constants[20]
        r_2 = theta[:, 21]  # constants[21]
        sigma_c = theta[:, 22]  # constants[22]
        R_2 = theta[:, 23]  # constants[23]
        delta = theta[:, 24]  # constants[24]
        h = theta[:, 25]  # constants[25]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V_A = torch.zeros_like(y[:, 0])
        d_P_A = torch.zeros_like(y[:, 0])
        d_f_o = torch.zeros_like(y[:, 0])
        d_f_c = torch.zeros_like(y[:, 0])
        d_p_o = torch.zeros_like(y[:, 0])
        d_p_c = torch.zeros_like(y[:, 0])
        d_z = torch.zeros_like(y[:, 0])
        alg_0_P_L = torch.zeros_like(y[:, 0])
        alg_1_dP_Ldt = torch.zeros_like(y[:, 0])
        alg_2_df_satdp = torch.zeros_like(y[:, 0])
        alg_3_q = torch.zeros_like(y[:, 0])
        alg_4_p_ao = torch.zeros_like(y[:, 0])
        alg_5_p_ac = torch.zeros_like(y[:, 0])
        alg_6_Q_A = torch.zeros_like(y[:, 0])
        alg_7_f_oi = torch.zeros_like(y[:, 0])
        alg_8_f_ci = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_z = torch.zeros_like(y[:, 0]) + (delta * r_2 * sigma_c * p_c / 1.0 - delta * l_2 * h * z)
        alg_0_P_L = P_m - R_2 * omega * V_T / 2.0 * torch.sin(omega * t / 1.0) - E * (2.5 - V_T * 1.0 / 2.0 * torch.cos(omega * t / 1.0))
        d_V_A = torch.zeros_like(y[:, 0]) + ((P_m_2 - alg_0_P_L - V_A * E_2 / 1.0) / R)
        alg_2_df_satdp = ((L * (1.0 + K_T * sigma * p_o) ** 4.0 + (1.0 + K_R * sigma * p_o) ** 4.0) * (3.0 * L * K_T ** 2.0 * sigma ** 2.0 * p_o * 1.0 * (1.0 + K_T * sigma * p_o) ** 2.0 + L * K_T * sigma * 1.0 * (1.0 + K_T * sigma * p_o) ** 3.0 + 3.0 * K_R ** 2.0 * sigma ** 2.0 * p_o * 1.0 * (1.0 + K_R * sigma * p_o) ** 2.0 + K_R * sigma * 1.0 * (1.0 + K_R * sigma * p_o) ** 3.0) - (L * K_T * sigma * p_o * (1.0 + K_T * sigma * p_o) ** 3.0 + K_R * sigma * p_o * (1.0 + K_R * sigma * p_o) ** 3.0) * (4.0 * L * K_T * sigma * 1.0 * (1.0 + K_T * sigma * p_o) ** 3.0 + 4.0 * K_R * sigma * 1.0 * (1.0 + K_R * sigma * p_o) ** 3.0)) / (L * (1.0 + K_T * sigma * p_o) ** 4.0 + (1.0 + K_R * sigma * p_o) ** 4.0) ** 2.0
        d_p_o = torch.zeros_like(y[:, 0]) + (D_o / (sigma * V_c) * (1.0 + 4.0 * T_h / sigma * alg_2_df_satdp) ** (-1.0) * (f_o * (P_A - P_w) - p_o))
        alg_5_p_ac = f_c * (P_A - P_w)
        d_p_c = torch.zeros_like(y[:, 0]) + (D_c / (sigma_c * V_c) * (alg_5_p_ac - p_c) + 1.0 * delta * l_2 / sigma_c * h * z - delta * r_2 * p_c)
        alg_1_dP_Ldt = -R_2 * omega ** 2.0 * V_T / (2.0 * 1.0) * torch.cos(omega * t / 1.0) - E * (2.5 - V_T / 2.0 * torch.sin(omega * t / 1.0))
        alg_3_q = (P_m_2 - P_A) / R
        alg_4_p_ao = f_o * (P_A - P_w)
        alg_6_Q_A = alg_3_q + 1.0 * D_c * (p_c - alg_5_p_ac) + 1.0 * D_o * (p_o - alg_4_p_ao)
        d_P_A = torch.zeros_like(y[:, 0]) + (P_m_2 * E_2 * alg_6_Q_A / (P_A * 1.0) + alg_1_dP_Ldt)
        alg_7_f_oi = torch.where(V_T_2 >= V_D, (f_o * V_D + f_om * (V_T_2 - V_D)) / V_T_2, f_o)
        d_f_o = torch.zeros_like(y[:, 0]) + (1.0 / V_A * (1.0 * D_o * (p_o - alg_4_p_ao) + (alg_7_f_oi - f_o) * alg_3_q - f_o * (1.0 * D_c * (p_c - alg_5_p_ac) + 1.0 * D_o * (p_o - alg_4_p_ao))))
        alg_8_f_ci = torch.where(V_T_2 >= V_D, (f_c * V_D + f_cm * (V_T_2 - V_D)) / V_T_2, f_c)
        d_f_c = torch.zeros_like(y[:, 0]) + (1.0 / V_A * (1.0 * D_c * (p_c - alg_5_p_ac) + (alg_8_f_ci - f_c) * alg_3_q - f_c * (1.0 * D_o * (p_o - alg_4_p_ao) + 1.0 * D_c * (p_c - alg_5_p_ac))))

        return torch.stack(
            [d_V_A, d_P_A, d_f_o, d_f_c, d_p_o, d_p_c, d_z],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
