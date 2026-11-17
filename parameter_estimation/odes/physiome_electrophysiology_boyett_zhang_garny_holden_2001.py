import torch
import torch.nn as nn
import math


class ElectrophysiologyBoyettZhangGarnyHolden2001ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/boyett_zhang_garny_holden_2001/model.py.
    Dynamic states: 22; independent constants: 66; derived constants used by rates: 24.
    Original constant slots: 90; algebraic slots: 71.
    """
    source_file = 'electrophysiology/boyett_zhang_garny_holden_2001/model.py'
    source_state_dim = 22
    source_constant_dim = 90
    algebraic_dim = 71
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 22
        self.parameter_dim = 66

        self.parameters = (
            parameters
            if parameters is not None
            else {'dCell': 0, 'R': 8314, 'T': 310, 'F': 96845, 'CmCentre': 2e-05, 'CmPeriphery': 6.5e-05, 'g_Na_Centre': 0, 'g_Na_Periphery': 1.2e-06, 'Na_o': 140, 'g_Ca_L_Centre': 0.0058, 'g_Ca_L_Periphery': 0.0659, 'E_Ca_L': 46.4, 'g_Ca_T_Centre': 0.0043, 'g_Ca_T_Periphery': 0.0139, 'E_Ca_T': 45, 'g_to_Centre': 0.00491, 'g_to_Periphery': 0.03649, 'g_sus_Centre': 6.65e-05, 'g_sus_Periphery': 0.0114, 'g_K_r_Centre': 0.000797, 'g_K_r_Periphery': 0.016, 'tau_P_i': 0.002, 'g_K_s_Centre': 0.000518, 'g_K_s_Periphery': 0.0104, 'g_f_Na_Centre': 0.000548, 'g_f_Na_Periphery': 0.0069, 'g_f_K_Centre': 0.000548, 'g_f_K_Periphery': 0.0069, 'g_b_Na_Centre': 5.8e-05, 'g_b_Na_Periphery': 0.000189, 'g_b_K_Centre': 2.52e-05, 'g_b_K_Periphery': 8.19e-05, 'g_b_Ca_Centre': 1.32e-05, 'g_b_Ca_Periphery': 4.3e-05, 'k_NaCa_Centre': 2.7e-06, 'k_NaCa_Periphery': 8.8e-06, 'd_NaCa': 0.0001, 'gamma_NaCa': 0.5, 'Na_i': 8, 'Ca_i': 0.0001, 'Ca_o': 2, 'K_m_Na': 5.64, 'K_m_K': 0.621, 'i_p_max_Centre': 0.0478, 'i_p_max_Periphery': 0.16, 'K_o': 5.4, 'K_i': 140, 'g_st': 0.00214, 'i_Ca_P_max': 0.24, 'J_Ca_P_max': 9.6e-10, 'J_Ca_u_max': 9.6e-10, 'B_d': 0.4, 'B_s': 0.2, 'Vol_c': 8e-12, 'K_m_b': 0.002, 'K_m_r': 0.001, 'K_m_u': 0.005, 'f_d': 0.1, 'f_r': 0.3, 'f_u': 0.06, 'alpha_ds': 3300, 'alpha_fR': 6670, 'alpha_1': 0.172, 'alpha_r': 333.5, 'alpha_ur': 533.6, 'beta_fR': 2}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -39.013558536, 'm': 0.092361701692, 'h1': 0.015905380261, 'h2': 0.01445216109, 'd_L': 0.04804900895, 'f_L': 0.48779845203, 'd_T': 0.42074047435, 'f_T': 0.038968420558, 'q': 0.29760539675, 'r': 0.064402950262, 'P_af': 0.13034201158, 'P_as': 0.46960956028, 'P_i': 0.87993375273, 'xs': 0.082293827208, 'y': 0.03889291759, 'd_s': 0.23, 'f_s': 0.283, 'Ca_d': 0.001544, 'Ca_s': 0.05487, 'Ca_u': 6.983, 'Ca_r': 4.909, 'f_R': 0.188}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['dCell', 'R', 'T', 'F', 'CmCentre', 'CmPeriphery', 'g_Na_Centre', 'g_Na_Periphery', 'Na_o', 'g_Ca_L_Centre', 'g_Ca_L_Periphery', 'E_Ca_L', 'g_Ca_T_Centre', 'g_Ca_T_Periphery', 'E_Ca_T', 'g_to_Centre', 'g_to_Periphery', 'g_sus_Centre', 'g_sus_Periphery', 'g_K_r_Centre', 'g_K_r_Periphery', 'tau_P_i', 'g_K_s_Centre', 'g_K_s_Periphery', 'g_f_Na_Centre', 'g_f_Na_Periphery', 'g_f_K_Centre', 'g_f_K_Periphery', 'g_b_Na_Centre', 'g_b_Na_Periphery', 'g_b_K_Centre', 'g_b_K_Periphery', 'g_b_Ca_Centre', 'g_b_Ca_Periphery', 'k_NaCa_Centre', 'k_NaCa_Periphery', 'd_NaCa', 'gamma_NaCa', 'Na_i', 'Ca_i', 'Ca_o', 'K_m_Na', 'K_m_K', 'i_p_max_Centre', 'i_p_max_Periphery', 'K_o', 'K_i', 'g_st', 'i_Ca_P_max', 'J_Ca_P_max', 'J_Ca_u_max', 'B_d', 'B_s', 'Vol_c', 'K_m_b', 'K_m_r', 'K_m_u', 'f_d', 'f_r', 'f_u', 'alpha_ds', 'alpha_fR', 'alpha_1', 'alpha_r', 'alpha_ur', 'beta_fR']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'm', 'h1', 'h2', 'd_L', 'f_L', 'd_T', 'f_T', 'q', 'r', 'P_af', 'P_as', 'P_i', 'xs', 'y', 'd_s', 'f_s', 'Ca_d', 'Ca_s', 'Ca_u', 'Ca_r', 'f_R']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (22)
        V = y[:, 0]
        m = y[:, 1]
        h1 = y[:, 2]
        h2 = y[:, 3]
        d_L = y[:, 4]
        f_L = y[:, 5]
        d_T = y[:, 6]
        f_T = y[:, 7]
        q = y[:, 8]
        r = y[:, 9]
        P_af = y[:, 10]
        P_as = y[:, 11]
        P_i = y[:, 12]
        xs = y[:, 13]
        state_y = y[:, 14]
        d_s = y[:, 15]
        f_s = y[:, 16]
        Ca_d = y[:, 17]
        Ca_s = y[:, 18]
        Ca_u = y[:, 19]
        Ca_r = y[:, 20]
        f_R = y[:, 21]

        # Independent parameters (source constant order)
        dCell = theta[:, 0]  # constants[0]
        R = theta[:, 1]  # constants[1]
        T = theta[:, 2]  # constants[2]
        F = theta[:, 3]  # constants[3]
        CmCentre = theta[:, 4]  # constants[4]
        CmPeriphery = theta[:, 5]  # constants[5]
        g_Na_Centre = theta[:, 6]  # constants[6]
        g_Na_Periphery = theta[:, 7]  # constants[7]
        Na_o = theta[:, 8]  # constants[8]
        g_Ca_L_Centre = theta[:, 9]  # constants[9]
        g_Ca_L_Periphery = theta[:, 10]  # constants[10]
        E_Ca_L = theta[:, 11]  # constants[11]
        g_Ca_T_Centre = theta[:, 12]  # constants[12]
        g_Ca_T_Periphery = theta[:, 13]  # constants[13]
        E_Ca_T = theta[:, 14]  # constants[14]
        g_to_Centre = theta[:, 15]  # constants[15]
        g_to_Periphery = theta[:, 16]  # constants[16]
        g_sus_Centre = theta[:, 17]  # constants[17]
        g_sus_Periphery = theta[:, 18]  # constants[18]
        g_K_r_Centre = theta[:, 19]  # constants[19]
        g_K_r_Periphery = theta[:, 20]  # constants[20]
        tau_P_i = theta[:, 21]  # constants[21]
        g_K_s_Centre = theta[:, 22]  # constants[22]
        g_K_s_Periphery = theta[:, 23]  # constants[23]
        g_f_Na_Centre = theta[:, 24]  # constants[24]
        g_f_Na_Periphery = theta[:, 25]  # constants[25]
        g_f_K_Centre = theta[:, 26]  # constants[26]
        g_f_K_Periphery = theta[:, 27]  # constants[27]
        g_b_Na_Centre = theta[:, 28]  # constants[28]
        g_b_Na_Periphery = theta[:, 29]  # constants[29]
        g_b_K_Centre = theta[:, 30]  # constants[30]
        g_b_K_Periphery = theta[:, 31]  # constants[31]
        g_b_Ca_Centre = theta[:, 32]  # constants[32]
        g_b_Ca_Periphery = theta[:, 33]  # constants[33]
        k_NaCa_Centre = theta[:, 34]  # constants[34]
        k_NaCa_Periphery = theta[:, 35]  # constants[35]
        d_NaCa = theta[:, 36]  # constants[36]
        gamma_NaCa = theta[:, 37]  # constants[37]
        Na_i = theta[:, 38]  # constants[38]
        Ca_i = theta[:, 39]  # constants[39]
        Ca_o = theta[:, 40]  # constants[40]
        K_m_Na = theta[:, 41]  # constants[41]
        K_m_K = theta[:, 42]  # constants[42]
        i_p_max_Centre = theta[:, 43]  # constants[43]
        i_p_max_Periphery = theta[:, 44]  # constants[44]
        K_o = theta[:, 45]  # constants[45]
        K_i = theta[:, 46]  # constants[46]
        g_st = theta[:, 47]  # constants[47]
        i_Ca_P_max = theta[:, 48]  # constants[48]
        J_Ca_P_max = theta[:, 49]  # constants[49]
        J_Ca_u_max = theta[:, 50]  # constants[50]
        B_d = theta[:, 51]  # constants[51]
        B_s = theta[:, 52]  # constants[52]
        Vol_c = theta[:, 53]  # constants[53]
        K_m_b = theta[:, 54]  # constants[54]
        K_m_r = theta[:, 55]  # constants[55]
        K_m_u = theta[:, 56]  # constants[56]
        f_d = theta[:, 57]  # constants[57]
        f_r = theta[:, 58]  # constants[58]
        f_u = theta[:, 59]  # constants[59]
        alpha_ds = theta[:, 60]  # constants[60]
        alpha_fR = theta[:, 61]  # constants[61]
        alpha_1 = theta[:, 62]  # constants[62]
        alpha_r = theta[:, 63]  # constants[63]
        alpha_ur = theta[:, 64]  # constants[64]
        beta_fR = theta[:, 65]  # constants[65]

        # Derived constants from the original initConsts()
        FCell = 1.07 * (3.0 * dCell - 0.1) / (3.0 * (1.0 + 0.7745 * torch.exp(-(3.0 * dCell - 2.05) / 0.295)))  # constants[66]
        E_Na = R * T / F * torch.log(Na_o / Na_i)  # constants[67]
        E_K = R * T / F * torch.log(K_o / K_i)  # constants[68]
        E_K_s = R * T / F * torch.log((K_o + 0.12 * Na_o) / (K_i + 0.12 * Na_i))  # constants[69]
        E_Ca = R * T / (2.0 * F) * torch.log(Ca_o / Ca_i)  # constants[70]
        Cm = CmCentre + FCell * (CmPeriphery - CmCentre)  # constants[71]
        g_Na = g_Na_Centre + FCell * (g_Na_Periphery - g_Na_Centre)  # constants[72]
        g_Ca_L = g_Ca_L_Centre + FCell * (g_Ca_L_Periphery - g_Ca_L_Centre)  # constants[73]
        g_Ca_T = g_Ca_T_Centre + FCell * (g_Ca_T_Periphery - g_Ca_T_Centre)  # constants[74]
        g_to = g_to_Centre + FCell * (g_to_Periphery - g_to_Centre)  # constants[75]
        g_sus = g_sus_Centre + FCell * (g_sus_Periphery - g_sus_Centre)  # constants[76]
        g_K_r = g_K_r_Centre + FCell * (g_K_r_Periphery - g_K_r_Centre)  # constants[77]
        g_K_s = g_K_s_Centre + FCell * (g_K_s_Periphery - g_K_s_Centre)  # constants[78]
        g_f_Na = g_f_Na_Centre + FCell * (g_f_Na_Periphery - g_f_Na_Centre)  # constants[79]
        g_f_K = g_f_K_Centre + FCell * (g_f_K_Periphery - g_f_K_Centre)  # constants[80]
        g_b_Na = g_b_Na_Centre + FCell * (g_b_Na_Periphery - g_b_Na_Centre)  # constants[81]
        g_b_K = g_b_K_Centre + FCell * (g_b_K_Periphery - g_b_K_Centre)  # constants[82]
        g_b_Ca = g_b_Ca_Centre + FCell * (g_b_Ca_Periphery - g_b_Ca_Centre)  # constants[83]
        k_NaCa = k_NaCa_Centre + FCell * (k_NaCa_Periphery - k_NaCa_Centre)  # constants[84]
        i_p_max = i_p_max_Centre + FCell * (i_p_max_Periphery - i_p_max_Centre)  # constants[85]
        Vol_u = f_u * Vol_c  # constants[86]
        Vol_r = f_r * Vol_c  # constants[87]
        Vol_d = f_d * Vol_c  # constants[88]
        Vol_s = Vol_c - (Vol_u + Vol_d)  # constants[89]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h1 = torch.zeros_like(y[:, 0])
        d_h2 = torch.zeros_like(y[:, 0])
        d_d_L = torch.zeros_like(y[:, 0])
        d_f_L = torch.zeros_like(y[:, 0])
        d_d_T = torch.zeros_like(y[:, 0])
        d_f_T = torch.zeros_like(y[:, 0])
        d_q = torch.zeros_like(y[:, 0])
        d_r = torch.zeros_like(y[:, 0])
        d_P_af = torch.zeros_like(y[:, 0])
        d_P_as = torch.zeros_like(y[:, 0])
        d_P_i = torch.zeros_like(y[:, 0])
        d_xs = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])
        d_d_s = torch.zeros_like(y[:, 0])
        d_f_s = torch.zeros_like(y[:, 0])
        d_Ca_d = torch.zeros_like(y[:, 0])
        d_Ca_s = torch.zeros_like(y[:, 0])
        d_Ca_u = torch.zeros_like(y[:, 0])
        d_Ca_r = torch.zeros_like(y[:, 0])
        d_f_R = torch.zeros_like(y[:, 0])
        alg_0_F_Na = torch.zeros_like(y[:, 0])
        alg_2_m_infinity = torch.zeros_like(y[:, 0])
        alg_3_h1_infinity = torch.zeros_like(y[:, 0])
        alg_4_alpha_d_L = torch.zeros_like(y[:, 0])
        alg_5_alpha_f_L = torch.zeros_like(y[:, 0])
        alg_6_alpha_d_T = torch.zeros_like(y[:, 0])
        alg_7_alpha_f_T = torch.zeros_like(y[:, 0])
        alg_8_q_infinity = torch.zeros_like(y[:, 0])
        alg_9_r_infinity = torch.zeros_like(y[:, 0])
        alg_10_P_af_infinity = torch.zeros_like(y[:, 0])
        alg_11_P_i_infinity = torch.zeros_like(y[:, 0])
        alg_12_alpha_xs = torch.zeros_like(y[:, 0])
        alg_13_alpha_y = torch.zeros_like(y[:, 0])
        alg_14_alpha_d_s = torch.zeros_like(y[:, 0])
        alg_15_alpha_f_s = torch.zeros_like(y[:, 0])
        alg_17_h = torch.zeros_like(y[:, 0])
        alg_18_tau_m = torch.zeros_like(y[:, 0])
        alg_19_h2_infinity = torch.zeros_like(y[:, 0])
        alg_20_tau_h1 = torch.zeros_like(y[:, 0])
        alg_21_beta_d_L = torch.zeros_like(y[:, 0])
        alg_22_beta_f_L = torch.zeros_like(y[:, 0])
        alg_23_beta_d_T = torch.zeros_like(y[:, 0])
        alg_24_beta_f_T = torch.zeros_like(y[:, 0])
        alg_25_tau_q = torch.zeros_like(y[:, 0])
        alg_26_tau_r = torch.zeros_like(y[:, 0])
        alg_27_tau_P_af = torch.zeros_like(y[:, 0])
        alg_28_P_as_infinity = torch.zeros_like(y[:, 0])
        alg_29_beta_xs = torch.zeros_like(y[:, 0])
        alg_30_beta_y = torch.zeros_like(y[:, 0])
        alg_31_beta_d_s = torch.zeros_like(y[:, 0])
        alg_32_beta_f_s = torch.zeros_like(y[:, 0])
        alg_33_i_Na = torch.zeros_like(y[:, 0])
        alg_34_tau_h2 = torch.zeros_like(y[:, 0])
        alg_35_tau_d_L = torch.zeros_like(y[:, 0])
        alg_36_tau_f_L = torch.zeros_like(y[:, 0])
        alg_37_tau_d_T = torch.zeros_like(y[:, 0])
        alg_38_tau_f_T = torch.zeros_like(y[:, 0])
        alg_39_tau_P_as = torch.zeros_like(y[:, 0])
        alg_40_tau_d_s = torch.zeros_like(y[:, 0])
        alg_41_tau_f_s = torch.zeros_like(y[:, 0])
        alg_42_i_Ca_L = torch.zeros_like(y[:, 0])
        alg_43_d_L_infinity = torch.zeros_like(y[:, 0])
        alg_44_f_L_infinity = torch.zeros_like(y[:, 0])
        alg_45_d_T_infinity = torch.zeros_like(y[:, 0])
        alg_46_f_T_infinity = torch.zeros_like(y[:, 0])
        alg_47_d_s_infinity = torch.zeros_like(y[:, 0])
        alg_48_f_s_infinity = torch.zeros_like(y[:, 0])
        alg_49_i_Ca_T = torch.zeros_like(y[:, 0])
        alg_50_i_to = torch.zeros_like(y[:, 0])
        alg_51_i_sus = torch.zeros_like(y[:, 0])
        alg_52_P_a = torch.zeros_like(y[:, 0])
        alg_53_i_K_r = torch.zeros_like(y[:, 0])
        alg_54_i_K_s = torch.zeros_like(y[:, 0])
        alg_55_i_f_Na = torch.zeros_like(y[:, 0])
        alg_56_i_f_K = torch.zeros_like(y[:, 0])
        alg_57_i_b_Na = torch.zeros_like(y[:, 0])
        alg_58_i_b_K = torch.zeros_like(y[:, 0])
        alg_59_i_b_Ca = torch.zeros_like(y[:, 0])
        alg_60_i_NaCa = torch.zeros_like(y[:, 0])
        alg_61_i_p = torch.zeros_like(y[:, 0])
        alg_62_U_d = torch.zeros_like(y[:, 0])
        alg_63_i_Ca_P = torch.zeros_like(y[:, 0])
        alg_64_J_Ca_ds = torch.zeros_like(y[:, 0])
        alg_65_J_Ca_r = torch.zeros_like(y[:, 0])
        alg_66_U_s = torch.zeros_like(y[:, 0])
        alg_67_J_Ca_u = torch.zeros_like(y[:, 0])
        alg_68_J_Ca_ur = torch.zeros_like(y[:, 0])
        alg_69_J_Ca_1 = torch.zeros_like(y[:, 0])
        alg_70_i_Ca = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_f_R = torch.zeros_like(y[:, 0]) + (-alpha_fR * Ca_d * f_R + beta_fR * (1.0 - f_R))
        alg_11_P_i_infinity = 1.0 / (1.0 + torch.exp((V + 18.6) / 10.1))
        d_P_i = torch.zeros_like(y[:, 0]) + ((alg_11_P_i_infinity - P_i) / tau_P_i)
        alg_2_m_infinity = (1.0 / (1.0 + torch.exp(-V / 5.46))) ** (1.0 / 3.0)
        alg_18_tau_m = 0.0006247 / (0.832 * torch.exp(-0.335 * (V + 56.7)) + 0.627 * torch.exp(0.082 * (V + 65.01))) + 4e-05
        d_m = torch.zeros_like(y[:, 0]) + ((alg_2_m_infinity - m) / alg_18_tau_m)
        alg_3_h1_infinity = 1.0 / (1.0 + torch.exp((V + 66.1) / 6.4))
        alg_20_tau_h1 = 3.717e-06 * torch.exp(-0.2815 * (V + 17.11)) / (1.0 + 0.003732 * torch.exp(-0.3426 * (V + 37.76))) + 0.0005977
        d_h1 = torch.zeros_like(y[:, 0]) + ((alg_3_h1_infinity - h1) / alg_20_tau_h1)
        alg_8_q_infinity = 1.0 / (1.0 + torch.exp((V + 59.37) / 13.1))
        alg_25_tau_q = 0.0101 + 0.06517 / (0.57 * torch.exp(-0.08 * (V + 49.0))) + 2.4e-05 * torch.exp(0.1 * (V + 50.93))
        d_q = torch.zeros_like(y[:, 0]) + ((alg_8_q_infinity - q) / alg_25_tau_q)
        alg_9_r_infinity = 1.0 / (1.0 + torch.exp(-(V - 10.93) / 19.7))
        alg_26_tau_r = 0.001 * (2.98 + 15.59 / (1.037 * torch.exp(0.09 * (V + 30.61)) + 0.369 * torch.exp(-0.12 * (V + 23.84))))
        d_r = torch.zeros_like(y[:, 0]) + ((alg_9_r_infinity - r) / alg_26_tau_r)
        alg_10_P_af_infinity = 1.0 / (1.0 + torch.exp(-(V + 14.2) / 10.6))
        alg_27_tau_P_af = 1.0 / (37.2 * torch.exp((V - 9.0) / 15.9) + 0.96 * torch.exp(-(V - 9.0) / 22.5))
        d_P_af = torch.zeros_like(y[:, 0]) + ((alg_10_P_af_infinity - P_af) / alg_27_tau_P_af)
        alg_12_alpha_xs = 14.0 / (1.0 + torch.exp(-(V - 40.0) / 9.0))
        alg_29_beta_xs = 1.0 * torch.exp(-V / 45.0)
        d_xs = torch.zeros_like(y[:, 0]) + (alg_12_alpha_xs * (1.0 - xs) - alg_29_beta_xs * xs)
        alg_13_alpha_y = 1.0 * torch.exp(-(V + 78.91) / 26.62)
        alg_30_beta_y = 1.0 * torch.exp((V + 75.13) / 21.25)
        d_state_y = torch.zeros_like(y[:, 0]) + (alg_13_alpha_y * (1.0 - state_y) - alg_30_beta_y * state_y)
        alg_19_h2_infinity = alg_3_h1_infinity
        alg_34_tau_h2 = 3.186e-08 * torch.exp(-0.6219 * (V + 18.8)) / (1.0 + 7.189e-05 * torch.exp(-0.6683 * (V + 34.07))) + 0.003556
        d_h2 = torch.zeros_like(y[:, 0]) + ((alg_19_h2_infinity - h2) / alg_34_tau_h2)
        alg_28_P_as_infinity = alg_10_P_af_infinity
        alg_39_tau_P_as = 1.0 / (4.2 * torch.exp((V - 9.0) / 17.0) + 0.15 * torch.exp(-(V - 9.0) / 21.6))
        d_P_as = torch.zeros_like(y[:, 0]) + ((alg_28_P_as_infinity - P_as) / alg_39_tau_P_as)
        alg_43_d_L_infinity = 1.0 / (1.0 + torch.exp(-(V + 23.1) / 6.0))
        alg_4_alpha_d_L = -28.38 * (V + 35.0) / (torch.exp(-(V + 35.0) / 2.5) - 1.0) - 84.9 * V / (torch.exp(-0.208 * V) - 1.0)
        alg_21_beta_d_L = 11.42 * (V - 5.0) / (torch.exp(0.4 * (V - 5.0)) - 1.0)
        alg_35_tau_d_L = 2.0 / (alg_4_alpha_d_L + alg_21_beta_d_L)
        d_d_L = torch.zeros_like(y[:, 0]) + ((alg_43_d_L_infinity - d_L) / alg_35_tau_d_L)
        alg_44_f_L_infinity = 1.0 / (1.0 + torch.exp((V + 45.0) / 5.0))
        alg_5_alpha_f_L = 3.12 * (V + 28.0) / (torch.exp((V + 28.0) / 4.0) - 1.0)
        alg_22_beta_f_L = 25.0 / (1.0 + torch.exp(-(V + 28.0) / 4.0))
        alg_36_tau_f_L = 1.0 / (alg_5_alpha_f_L + alg_22_beta_f_L)
        d_f_L = torch.zeros_like(y[:, 0]) + ((alg_44_f_L_infinity - f_L) / alg_36_tau_f_L)
        alg_45_d_T_infinity = 1.0 / (1.0 + torch.exp(-(V + 37.0) / 6.8))
        alg_6_alpha_d_T = 1068.0 * torch.exp((V + 26.3) / 30.0)
        alg_23_beta_d_T = 1068.0 * torch.exp(-(V + 26.3) / 30.0)
        alg_37_tau_d_T = 1.0 / (alg_6_alpha_d_T + alg_23_beta_d_T)
        d_d_T = torch.zeros_like(y[:, 0]) + ((alg_45_d_T_infinity - d_T) / alg_37_tau_d_T)
        alg_46_f_T_infinity = 1.0 / (1.0 + torch.exp((V + 71.0) / 9.0))
        alg_7_alpha_f_T = 15.3 * torch.exp(-(V + 71.7) / 83.3)
        alg_24_beta_f_T = 15.0 * torch.exp((V + 71.7) / 15.38)
        alg_38_tau_f_T = 1.0 / (alg_7_alpha_f_T + alg_24_beta_f_T)
        d_f_T = torch.zeros_like(y[:, 0]) + ((alg_46_f_T_infinity - f_T) / alg_38_tau_f_T)
        alg_14_alpha_d_s = 1000.0 / (0.15 * torch.exp(-V / 11.0) + 0.2 * torch.exp(-V / 700.0))
        alg_31_beta_d_s = 1000.0 / (16.0 * torch.exp(V / 8.0) + 0.2 * torch.exp(V / 50.0))
        alg_47_d_s_infinity = alg_14_alpha_d_s / (alg_14_alpha_d_s + alg_31_beta_d_s)
        alg_40_tau_d_s = 1.0 / (alg_14_alpha_d_s + alg_31_beta_d_s)
        d_d_s = torch.zeros_like(y[:, 0]) + ((alg_47_d_s_infinity - d_s) / alg_40_tau_d_s)
        alg_15_alpha_f_s = 1000.0 / (3100.0 * torch.exp(-V / 13.0) + 700.0 * torch.exp(-V / 70.0))
        alg_32_beta_f_s = 1000.0 / (16.0 * torch.exp(V / 8.0) + 0.2 * torch.exp(V / 50.0))
        alg_48_f_s_infinity = alg_15_alpha_f_s / (alg_15_alpha_f_s + alg_32_beta_f_s)
        alg_41_tau_f_s = 1.0 / (alg_15_alpha_f_s + alg_32_beta_f_s)
        d_f_s = torch.zeros_like(y[:, 0]) + ((alg_48_f_s_infinity - f_s) / alg_41_tau_f_s)
        alg_0_F_Na = 0.0952 * torch.exp(-0.063 * (V + 34.4)) / (1.0 + 1.66 * torch.exp(-0.225 * (V + 63.7))) + 0.0869
        alg_17_h = (1.0 - alg_0_F_Na) * h1 + alg_0_F_Na * h2
        alg_33_i_Na = g_Na * m ** 3.0 * alg_17_h * Na_o * F ** 2.0 / (R * T) * (torch.exp((V - E_Na) * F / (R * T)) - 1.0) / (torch.exp(V * F / (R * T)) - 1.0) * V
        alg_42_i_Ca_L = g_Ca_L * (f_L * d_L + 0.006 / (1.0 + torch.exp(-(V + 14.1) / 6.0))) * (V - E_Ca_L)
        alg_49_i_Ca_T = g_Ca_T * d_T * f_T * (V - E_Ca_T)
        alg_50_i_to = g_to * q * r * (V - E_K)
        alg_51_i_sus = g_sus * r * (V - E_K)
        alg_52_P_a = 0.6 * P_af + 0.4 * P_as
        alg_53_i_K_r = g_K_r * alg_52_P_a * P_i * (V - E_K)
        alg_54_i_K_s = g_K_s * xs ** 2.0 * (V - E_K_s)
        alg_55_i_f_Na = g_f_Na * state_y * (V - E_Na)
        alg_56_i_f_K = g_f_K * state_y * (V - E_K)
        alg_57_i_b_Na = g_b_Na * (V - E_Na)
        alg_59_i_b_Ca = g_b_Ca * (V - E_Ca)
        alg_58_i_b_K = g_b_K * (V - E_K)
        alg_60_i_NaCa = k_NaCa * (Na_i ** 3.0 * Ca_o * torch.exp(0.03743 * V * gamma_NaCa) - Na_o ** 3.0 * Ca_i * torch.exp(0.0374 * V * (gamma_NaCa - 1.0))) / (1.0 + d_NaCa * (Ca_i * Na_o ** 3.0 + Ca_o * Na_i ** 3.0))
        alg_61_i_p = i_p_max * (Na_i / (K_m_Na + Na_i)) ** 3.0 * (K_o / (K_m_K + K_o)) ** 2.0 * 1.6 / (1.5 + torch.exp(-(V + 60.0) / 40.0))
        alg_63_i_Ca_P = i_Ca_P_max * (Ca_s / (Ca_s + 0.0004))
        d_V = torch.zeros_like(y[:, 0]) + (-1.0 / Cm * (alg_33_i_Na + alg_42_i_Ca_L + alg_49_i_Ca_T + alg_50_i_to + alg_51_i_sus + alg_53_i_K_r + alg_54_i_K_s + alg_55_i_f_Na + alg_56_i_f_K + alg_57_i_b_Na + alg_59_i_b_Ca + alg_58_i_b_K + alg_60_i_NaCa + alg_61_i_p + alg_63_i_Ca_P))
        alg_65_J_Ca_r = alpha_r * f_R * (Ca_d ** 2.0 / (K_m_r ** 2.0 + Ca_d ** 2.0)) * Vol_r * Ca_r
        alg_68_J_Ca_ur = alpha_ur * Vol_u * (Ca_u - Ca_r)
        d_Ca_r = torch.zeros_like(y[:, 0]) + ((alg_68_J_Ca_ur - alg_65_J_Ca_r) / Vol_r)
        alg_67_J_Ca_u = J_Ca_u_max * (Ca_s ** 2.0 / (K_m_u ** 2.0 + Ca_s ** 2.0))
        alg_69_J_Ca_1 = alpha_1 * Vol_u * Ca_u
        d_Ca_u = torch.zeros_like(y[:, 0]) + ((alg_67_J_Ca_u - (alg_69_J_Ca_1 + alg_68_J_Ca_ur)) / Vol_u)
        alg_62_U_d = 1.0 - B_d / (K_m_b + Ca_d + B_d)
        alg_64_J_Ca_ds = alpha_ds * Vol_d * (Ca_d - Ca_s)
        alg_70_i_Ca = alg_42_i_Ca_L + alg_49_i_Ca_T
        d_Ca_d = torch.zeros_like(y[:, 0]) + (alg_62_U_d / Vol_d * (alg_64_J_Ca_ds - 0.95 * alg_70_i_Ca / (2.0 * F)))
        alg_66_U_s = 1.0 - B_s / (K_m_b + Ca_s + B_s)
        d_Ca_s = torch.zeros_like(y[:, 0]) + (alg_66_U_s / Vol_s * (alg_64_J_Ca_ds - ((0.05 * alg_70_i_Ca - 2.0 * alg_60_i_NaCa + alg_59_i_b_Ca) / (2.0 * F) + alg_67_J_Ca_u) + alg_65_J_Ca_r + alg_69_J_Ca_1))

        return torch.stack(
            [d_V, d_m, d_h1, d_h2, d_d_L, d_f_L, d_d_T, d_f_T, d_q, d_r, d_P_af, d_P_as, d_P_i, d_xs, d_state_y, d_d_s, d_f_s, d_Ca_d, d_Ca_s, d_Ca_u, d_Ca_r, d_f_R],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
