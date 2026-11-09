import torch
import torch.nn as nn
import math


class CalciumDynamicsIribeKohlNoble2006ODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/iribe_kohl_noble_2006/model.py.
    Dynamic states: 23; independent constants: 53; derived constants used by rates: 16.
    Original constant slots: 69; algebraic slots: 51.
    """
    source_file = 'calcium_dynamics/iribe_kohl_noble_2006/model.py'
    source_state_dim = 23
    source_constant_dim = 69
    algebraic_dim = 51
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 23
        self.parameter_dim = 53

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8314.472, 'T': 310, 'F': 96485.3415, 'Cm': 9.5e-05, 'v_i': 1.6404e-05, 'v_SR': 3.3477e-06, 'Na_o': 140, 'K_o': 4, 'Ca_o': 2, 'stim_start': 0, 'stim_end': 1000, 'stim_period': 0.5, 'stim_duration': 0.002, 'stim_amplitude': -4, 'g_Na': 2.5, 'delta_m': 1e-05, 'shift_h': 0, 'i_Kmax': 1, 'K_mk1': 10, 'g_K1': 1, 'g_to': 0.005, 'P_Ca_L_Ca': 0.25, 'speed_d': 3, 'speed_f': 0.5, 'i_NaCa_max': 0.0005, 'gamma': 0.5, 'i_NaK_max': 1.36, 'K_mK': 1, 'K_mNa': 21.7, 'g_b_Ca': 0.00025, 'g_b_K': 0.0006, 'g_b_Na': 0.0006, 'Tau_CaMK': 0.8, 'K_rel_max': 500, 'Tau_SRCa_RyR': 0.05, 'gain_k1': 1, 'gain_k2': 1, 'gain_k3': 1, 'gain_k4': 1, 'K_leak_rate': 0, 'V_max_f': 0.292, 'V_max_r': 0.391, 'Cmdn_tot': 0.02, 'alpha_cmdn': 10000, 'beta_cmdn': 500, 'Trpn_tot': 0.07, 'alpha_trpn': 80000, 'beta_trpn': 200, 'zeta': 0.1, 'beta_tm': 40, 'SL': 2.15, 'f_XB': 10, 'g_XB': 30}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -92.849333, 'K_i': 138.22, 'Na_i': 5.8041, 'Ca_i': 9.91e-06, 'm': 0.0013809, 'h': 0.99569, 'x': 0.051127, 's': 0.95854, 'r': 1.5185e-08, 'd': 1.7908e-08, 'f': 1, 'F_CaMK': 1.028, 'Cmdn_Ca': 3.9636e-06, 'Ca_SR': 0.24886, 'F_SRCa_RyR': 0.25089, 'F_1': 0.5268, 'F_2': 8.7508e-06, 'Trpn_Ca': 0.00027661, 'N_0': 0.99917, 'P_0': 9.8593e-05, 'P_1': 0.00013331, 'P_2': 0.00023505, 'P_3': 0.00015349}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'Cm', 'v_i', 'v_SR', 'Na_o', 'K_o', 'Ca_o', 'stim_start', 'stim_end', 'stim_period', 'stim_duration', 'stim_amplitude', 'g_Na', 'delta_m', 'shift_h', 'i_Kmax', 'K_mk1', 'g_K1', 'g_to', 'P_Ca_L_Ca', 'speed_d', 'speed_f', 'i_NaCa_max', 'gamma', 'i_NaK_max', 'K_mK', 'K_mNa', 'g_b_Ca', 'g_b_K', 'g_b_Na', 'Tau_CaMK', 'K_rel_max', 'Tau_SRCa_RyR', 'gain_k1', 'gain_k2', 'gain_k3', 'gain_k4', 'K_leak_rate', 'V_max_f', 'V_max_r', 'Cmdn_tot', 'alpha_cmdn', 'beta_cmdn', 'Trpn_tot', 'alpha_trpn', 'beta_trpn', 'zeta', 'beta_tm', 'SL', 'f_XB', 'g_XB']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'K_i', 'Na_i', 'Ca_i', 'm', 'h', 'x', 's', 'r', 'd', 'f', 'F_CaMK', 'Cmdn_Ca', 'Ca_SR', 'F_SRCa_RyR', 'F_1', 'F_2', 'Trpn_Ca', 'N_0', 'P_0', 'P_1', 'P_2', 'P_3']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (23)
        V = y[:, 0]
        K_i = y[:, 1]
        Na_i = y[:, 2]
        Ca_i = y[:, 3]
        m = y[:, 4]
        h = y[:, 5]
        x = y[:, 6]
        s = y[:, 7]
        r = y[:, 8]
        d = y[:, 9]
        f = y[:, 10]
        F_CaMK = y[:, 11]
        Cmdn_Ca = y[:, 12]
        Ca_SR = y[:, 13]
        F_SRCa_RyR = y[:, 14]
        F_1 = y[:, 15]
        F_2 = y[:, 16]
        Trpn_Ca = y[:, 17]
        N_0 = y[:, 18]
        P_0 = y[:, 19]
        P_1 = y[:, 20]
        P_2 = y[:, 21]
        P_3 = y[:, 22]

        # Independent parameters (source constant order)
        R = theta[:, 0]  # constants[0]
        T = theta[:, 1]  # constants[1]
        F = theta[:, 2]  # constants[2]
        Cm = theta[:, 3]  # constants[3]
        v_i = theta[:, 4]  # constants[4]
        v_SR = theta[:, 5]  # constants[5]
        Na_o = theta[:, 6]  # constants[6]
        K_o = theta[:, 7]  # constants[7]
        Ca_o = theta[:, 8]  # constants[8]
        stim_start = theta[:, 9]  # constants[9]
        stim_end = theta[:, 10]  # constants[10]
        stim_period = theta[:, 11]  # constants[11]
        stim_duration = theta[:, 12]  # constants[12]
        stim_amplitude = theta[:, 13]  # constants[13]
        g_Na = theta[:, 14]  # constants[14]
        delta_m = theta[:, 15]  # constants[15]
        shift_h = theta[:, 16]  # constants[16]
        i_Kmax = theta[:, 17]  # constants[17]
        K_mk1 = theta[:, 18]  # constants[18]
        g_K1 = theta[:, 19]  # constants[19]
        g_to = theta[:, 20]  # constants[20]
        P_Ca_L_Ca = theta[:, 21]  # constants[21]
        speed_d = theta[:, 22]  # constants[22]
        speed_f = theta[:, 23]  # constants[23]
        i_NaCa_max = theta[:, 24]  # constants[24]
        gamma = theta[:, 25]  # constants[25]
        i_NaK_max = theta[:, 26]  # constants[26]
        K_mK = theta[:, 27]  # constants[27]
        K_mNa = theta[:, 28]  # constants[28]
        g_b_Ca = theta[:, 29]  # constants[29]
        g_b_K = theta[:, 30]  # constants[30]
        g_b_Na = theta[:, 31]  # constants[31]
        Tau_CaMK = theta[:, 32]  # constants[32]
        K_rel_max = theta[:, 33]  # constants[33]
        Tau_SRCa_RyR = theta[:, 34]  # constants[34]
        gain_k1 = theta[:, 35]  # constants[35]
        gain_k2 = theta[:, 36]  # constants[36]
        gain_k3 = theta[:, 37]  # constants[37]
        gain_k4 = theta[:, 38]  # constants[38]
        K_leak_rate = theta[:, 39]  # constants[39]
        V_max_f = theta[:, 40]  # constants[40]
        V_max_r = theta[:, 41]  # constants[41]
        Cmdn_tot = theta[:, 42]  # constants[42]
        alpha_cmdn = theta[:, 43]  # constants[43]
        beta_cmdn = theta[:, 44]  # constants[44]
        Trpn_tot = theta[:, 45]  # constants[45]
        alpha_trpn = theta[:, 46]  # constants[46]
        beta_trpn = theta[:, 47]  # constants[47]
        zeta = theta[:, 48]  # constants[48]
        beta_tm = theta[:, 49]  # constants[49]
        SL = theta[:, 50]  # constants[50]
        f_XB = theta[:, 51]  # constants[51]
        g_XB = theta[:, 52]  # constants[52]

        # Derived constants from the original initConsts()
        k_4 = gain_k4 * 1.8  # constants[53]
        phi_SL = torch.where((SL >= 1.7) & (SL <= 2.0), (SL - 0.6) / 1.4, torch.where((SL > 2.0) & (SL <= 2.2), 1.0, torch.where((SL > 2.2) & (SL <= 2.3), (3.6 - SL) / 1.4, (1e309-1e309))))  # constants[54]
        SL_norm = (SL - 1.7) / 0.7  # constants[55]
        f_01 = 3.0 * f_XB  # constants[56]
        K_tm = 1.0 / (1.0 + beta_trpn / alpha_trpn / (0.0017 - 0.0009 * SL_norm))  # constants[57]
        N_tm = 3.5 + 2.5 * SL_norm  # constants[58]
        f_12 = 10.0 * f_XB  # constants[59]
        g_01 = g_XB * (2.0 - SL_norm)  # constants[60]
        g_12 = 2.0 * g_XB * (2.0 - SL_norm)  # constants[61]
        g_23 = 3.0 * g_XB * (2.0 - SL_norm)  # constants[62]
        f_23 = 7.0 * f_XB  # constants[63]
        sigma_paths = 1.0 * g_XB * 2.0 * g_XB * 3.0 * g_XB + 1.0 * f_01 * 2.0 * g_XB * 3.0 * g_XB + 1.0 * f_01 * 1.0 * f_12 * 3.0 * g_XB + 1.0 * f_01 * 1.0 * f_12 * 1.0 * f_23  # constants[64]
        P_1_max = 1.0 * f_01 * 2.0 * g_XB * 3.0 * g_XB / sigma_paths  # constants[65]
        P_2_max = 1.0 * f_01 * 1.0 * f_12 * 3.0 * g_XB / sigma_paths  # constants[66]
        P_3_max = 1.0 * f_01 * 1.0 * f_12 * 1.0 * f_23 / sigma_paths  # constants[67]
        Force_max = P_1_max + 2.0 * P_2_max + 3.0 * P_3_max  # constants[68]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_x = torch.zeros_like(y[:, 0])
        d_s = torch.zeros_like(y[:, 0])
        d_r = torch.zeros_like(y[:, 0])
        d_d = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_F_CaMK = torch.zeros_like(y[:, 0])
        d_Cmdn_Ca = torch.zeros_like(y[:, 0])
        d_Ca_SR = torch.zeros_like(y[:, 0])
        d_F_SRCa_RyR = torch.zeros_like(y[:, 0])
        d_F_1 = torch.zeros_like(y[:, 0])
        d_F_2 = torch.zeros_like(y[:, 0])
        d_Trpn_Ca = torch.zeros_like(y[:, 0])
        d_N_0 = torch.zeros_like(y[:, 0])
        d_P_0 = torch.zeros_like(y[:, 0])
        d_P_1 = torch.zeros_like(y[:, 0])
        d_P_2 = torch.zeros_like(y[:, 0])
        d_P_3 = torch.zeros_like(y[:, 0])
        alg_0_i_Stim = torch.zeros_like(y[:, 0])
        alg_1_E0_m = torch.zeros_like(y[:, 0])
        alg_2_alpha_h = torch.zeros_like(y[:, 0])
        alg_3_alpha_x = torch.zeros_like(y[:, 0])
        alg_4_alpha_s = torch.zeros_like(y[:, 0])
        alg_5_E0_d = torch.zeros_like(y[:, 0])
        alg_6_E0_f = torch.zeros_like(y[:, 0])
        alg_7_Inf_CaMK = torch.zeros_like(y[:, 0])
        alg_8_E_Na = torch.zeros_like(y[:, 0])
        alg_9_alpha_m = torch.zeros_like(y[:, 0])
        alg_10_beta_h = torch.zeros_like(y[:, 0])
        alg_11_beta_x = torch.zeros_like(y[:, 0])
        alg_12_beta_s = torch.zeros_like(y[:, 0])
        alg_13_alpha_d = torch.zeros_like(y[:, 0])
        alg_14_alpha_f = torch.zeros_like(y[:, 0])
        alg_15_E_K = torch.zeros_like(y[:, 0])
        alg_16_beta_m = torch.zeros_like(y[:, 0])
        alg_17_beta_d = torch.zeros_like(y[:, 0])
        alg_18_beta_f = torch.zeros_like(y[:, 0])
        alg_19_E_Ca = torch.zeros_like(y[:, 0])
        alg_20_E_mh = torch.zeros_like(y[:, 0])
        alg_21_i_Na = torch.zeros_like(y[:, 0])
        alg_22_i_K = torch.zeros_like(y[:, 0])
        alg_23_i_K1 = torch.zeros_like(y[:, 0])
        alg_24_i_to = torch.zeros_like(y[:, 0])
        alg_25_i_Ca_L_Ca = torch.zeros_like(y[:, 0])
        alg_26_i_Ca_L_K = torch.zeros_like(y[:, 0])
        alg_27_i_Ca_L_Na = torch.zeros_like(y[:, 0])
        alg_28_i_Ca_L = torch.zeros_like(y[:, 0])
        alg_29_N_CaMK = torch.zeros_like(y[:, 0])
        alg_30_i_NaCa = torch.zeros_like(y[:, 0])
        alg_31_k_1 = torch.zeros_like(y[:, 0])
        alg_32_i_NaK = torch.zeros_like(y[:, 0])
        alg_33_k_2 = torch.zeros_like(y[:, 0])
        alg_34_k_3 = torch.zeros_like(y[:, 0])
        alg_35_i_b_Ca = torch.zeros_like(y[:, 0])
        alg_36_i_b_K = torch.zeros_like(y[:, 0])
        alg_37_F_rel = torch.zeros_like(y[:, 0])
        alg_38_F_3 = torch.zeros_like(y[:, 0])
        alg_39_i_b_Na = torch.zeros_like(y[:, 0])
        alg_40_K_rel = torch.zeros_like(y[:, 0])
        alg_41_j_rel = torch.zeros_like(y[:, 0])
        alg_42_f_b = torch.zeros_like(y[:, 0])
        alg_43_r_b = torch.zeros_like(y[:, 0])
        alg_44_j_up = torch.zeros_like(y[:, 0])
        alg_45_dCmdn_Ca_dtime = torch.zeros_like(y[:, 0])
        alg_46_alpha_tm = torch.zeros_like(y[:, 0])
        alg_47_N_1 = torch.zeros_like(y[:, 0])
        alg_48_Force_norm = torch.zeros_like(y[:, 0])
        alg_49_dTrpn_Ca_dtime = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_r = torch.zeros_like(y[:, 0]) + (333.0 * (1.0 / (1.0 + torch.exp(-(V + 4.0) / 5.0)) - r))
        d_F_SRCa_RyR = torch.zeros_like(y[:, 0]) + ((Ca_SR - F_SRCa_RyR) / Tau_SRCa_RyR)
        d_Cmdn_Ca = torch.zeros_like(y[:, 0]) + (alpha_cmdn * (Cmdn_tot - Cmdn_Ca) * Ca_i - beta_cmdn * Cmdn_Ca)
        d_P_2 = torch.zeros_like(y[:, 0]) + (-(f_23 + g_12) * P_2 + f_12 * P_1 + g_23 * P_3)
        d_P_3 = torch.zeros_like(y[:, 0]) + (-g_23 * P_3 + f_23 * P_2)
        alg_7_Inf_CaMK = Cmdn_Ca / 5e-05
        d_F_CaMK = torch.zeros_like(y[:, 0]) + ((alg_7_Inf_CaMK - F_CaMK) / Tau_CaMK)
        alg_2_alpha_h = 20.0 * torch.exp(-0.125 * (V + 75.0 - shift_h))
        alg_10_beta_h = 2000.0 / (1.0 + 320.0 * torch.exp(-0.1 * (V + 75.0 - shift_h)))
        d_h = torch.zeros_like(y[:, 0]) + (alg_2_alpha_h * (1.0 - h) - alg_10_beta_h * h)
        alg_3_alpha_x = 0.5 * torch.exp(0.0826 * (V + 50.0)) / (1.0 + torch.exp(0.057 * (V + 50.0)))
        alg_11_beta_x = 1.3 * torch.exp(-0.06 * (V + 20.0)) / (1.0 + torch.exp(-0.04 * (V + 20.0)))
        d_x = torch.zeros_like(y[:, 0]) + (alg_3_alpha_x * (1.0 - x) - alg_11_beta_x * x)
        alg_4_alpha_s = 0.033 * torch.exp(-V / 17.0)
        alg_12_beta_s = 33.0 / (1.0 + torch.exp(-0.125 * (V + 10.0)))
        d_s = torch.zeros_like(y[:, 0]) + (alg_4_alpha_s * (1.0 - s) - alg_12_beta_s * s)
        alg_1_E0_m = V + 41.0
        alg_9_alpha_m = torch.where(torch.abs(alg_1_E0_m) < delta_m, 2000.0, 200.0 * alg_1_E0_m / (1.0 - torch.exp(-0.1 * alg_1_E0_m)))
        alg_16_beta_m = 8000.0 * torch.exp(-0.056 * (V + 66.0))
        d_m = torch.zeros_like(y[:, 0]) + (alg_9_alpha_m * (1.0 - m) - alg_16_beta_m * m)
        alg_5_E0_d = V + 24.0 - 5.0
        alg_13_alpha_d = torch.where(torch.abs(alg_5_E0_d) < 1e-05, speed_d * 120.0, speed_d * 30.0 * alg_5_E0_d / (1.0 - torch.exp(-alg_5_E0_d / 4.0)))
        alg_17_beta_d = torch.where(torch.abs(alg_5_E0_d) < 1e-05, speed_d * 120.0, speed_d * -12.0 * alg_5_E0_d / (1.0 - torch.exp(alg_5_E0_d / 10.0)))
        d_d = torch.zeros_like(y[:, 0]) + (alg_13_alpha_d * (1.0 - d) - alg_17_beta_d * d)
        alg_6_E0_f = V + 34.0
        alg_14_alpha_f = torch.where(torch.abs(alg_6_E0_f) < 1e-05, speed_f * 25.0, speed_f * 6.25 * alg_6_E0_f / (-1.0 + torch.exp(alg_6_E0_f / 4.0)))
        alg_18_beta_f = speed_f * 50.0 / (1.0 + torch.exp(-alg_6_E0_f / 4.0))
        d_f = torch.zeros_like(y[:, 0]) + (alg_14_alpha_f * (1.0 - f) - alg_18_beta_f * f)
        alg_27_i_Ca_L_Na = 0.01 * d * f * P_Ca_L_Ca * (V - 50.0) * F / (R * T * (1.0 - torch.exp(-(V - 50.0) * F / (R * T)))) * (Na_i * torch.exp(50.0 * F / (R * T)) - Na_o * torch.exp(-(V - 50.0) * F / (R * T)))
        alg_26_i_Ca_L_K = 0.002 * d * f * P_Ca_L_Ca * (V - 50.0) * F / (R * T * (1.0 - torch.exp(-(V - 50.0) * F / (R * T)))) * (K_i * torch.exp(50.0 * F / (R * T)) - K_o * torch.exp(-(V - 50.0) * F / (R * T)))
        alg_25_i_Ca_L_Ca = 4.0 * d * f * P_Ca_L_Ca * (V - 50.0) * F / (R * T * (1.0 - torch.exp(-2.0 * (V - 50.0) * F / (R * T)))) * (Ca_i * torch.exp(100.0 * F / (R * T)) - Ca_o * torch.exp(-2.0 * (V - 50.0) * F / (R * T)))
        alg_28_i_Ca_L = alg_25_i_Ca_L_Ca + alg_26_i_Ca_L_K + alg_27_i_Ca_L_Na
        alg_31_k_1 = gain_k1 * (30625000.0 * Ca_i ** 2.0 - 245.0 * alg_28_i_Ca_L)
        alg_33_k_2 = gain_k2 * 450.0 / (1.0 + 0.36 / Ca_SR)
        d_F_2 = torch.zeros_like(y[:, 0]) + (alg_31_k_1 * F_1 - alg_33_k_2 * F_2)
        alg_29_N_CaMK = (F_CaMK / 0.7) ** 2.0
        alg_34_k_3 = gain_k3 * 1.885 * (F_SRCa_RyR / 0.22) ** alg_29_N_CaMK
        alg_38_F_3 = 1.0 - (F_1 + F_2)
        d_F_1 = torch.zeros_like(y[:, 0]) + (alg_34_k_3 * alg_38_F_3 - k_4 * F_1 - alg_31_k_1 * F_1)
        alg_15_E_K = R * T / F * torch.log(K_o / K_i)
        alg_23_i_K1 = g_K1 * K_o / (K_o + K_mk1) * (V - alg_15_E_K) / (1.0 + torch.exp(2.0 * F * (V - alg_15_E_K - 10.0) / (R * T)))
        alg_24_i_to = g_to * s * r * (V - alg_15_E_K)
        alg_22_i_K = i_Kmax * x * (K_i - K_o * torch.exp(-V * F / (R * T))) / 140.0
        alg_32_i_NaK = i_NaK_max * K_o / (K_mK + K_o) * Na_i / (K_mNa + Na_i) * 1.0 / (1.0 + 0.1245 * torch.exp(-0.1 * V * F / (R * T)) + 0.0353 * torch.exp(-V * F / (R * T)))
        alg_36_i_b_K = g_b_K * (V - alg_15_E_K)
        d_K_i = torch.zeros_like(y[:, 0]) + (-(alg_23_i_K1 + alg_22_i_K + alg_24_i_to + alg_36_i_b_K + alg_26_i_Ca_L_K - 2.0 * alg_32_i_NaK) / (v_i * F))
        alg_20_E_mh = R * T / F * torch.log((Na_o + 0.12 * K_o) / (Na_i + 0.12 * K_i))
        alg_21_i_Na = g_Na * m ** 3.0 * h * (V - alg_20_E_mh)
        alg_30_i_NaCa = i_NaCa_max * (torch.exp(gamma * V * F / (R * T)) * Na_i ** 3.0 * Ca_o - torch.exp((gamma - 1.0) * V * F / (R * T)) * Na_o ** 3.0 * Ca_i) / (1.0 + Ca_i / 0.0069)
        alg_8_E_Na = R * T / F * torch.log(Na_o / Na_i)
        alg_39_i_b_Na = g_b_Na * (V - alg_8_E_Na)
        alg_19_E_Ca = 0.5 * R * T / F * torch.log(Ca_o / Ca_i)
        alg_35_i_b_Ca = g_b_Ca * (V - alg_19_E_Ca)
        alg_0_i_Stim = torch.where((t >= stim_start) & (t <= stim_end) & (t - stim_start - torch.floor((t - stim_start) / stim_period) * stim_period <= stim_duration), stim_amplitude, 0.0)
        d_V = torch.zeros_like(y[:, 0]) + (-1.0 / Cm * (alg_21_i_Na + alg_39_i_b_Na + alg_23_i_K1 + alg_22_i_K + alg_24_i_to + alg_36_i_b_K + alg_28_i_Ca_L + alg_35_i_b_Ca + alg_30_i_NaCa + alg_32_i_NaK + alg_0_i_Stim))
        d_Na_i = torch.zeros_like(y[:, 0]) + (-(alg_21_i_Na + alg_39_i_b_Na + alg_27_i_Ca_L_Na + 3.0 * alg_30_i_NaCa + 3.0 * alg_32_i_NaK) / (v_i * F))
        alg_40_K_rel = K_rel_max * F_SRCa_RyR / (F_SRCa_RyR + 0.2)
        alg_37_F_rel = (F_2 / (F_2 + 0.25)) ** 2.0
        alg_41_j_rel = (alg_40_K_rel * alg_37_F_rel + K_leak_rate) * (Ca_SR - Ca_i)
        alg_42_f_b = (Ca_i / 0.00024) ** 2.0
        alg_43_r_b = (Ca_SR / 1.64) ** 2.0
        alg_44_j_up = (F_CaMK * V_max_f * alg_42_f_b - V_max_r * alg_43_r_b) / (1.0 + alg_42_f_b + alg_43_r_b)
        d_Ca_SR = torch.zeros_like(y[:, 0]) + (alg_44_j_up * v_i / v_SR - alg_41_j_rel)
        alg_46_alpha_tm = beta_tm * (Trpn_Ca / (Trpn_tot * K_tm)) ** N_tm
        d_P_0 = torch.zeros_like(y[:, 0]) + (-(beta_tm + f_01) * P_0 + alg_46_alpha_tm * N_0 + g_01 * P_1)
        alg_47_N_1 = 1.0 - (N_0 + P_0 + P_1 + P_2 + P_3)
        d_N_0 = torch.zeros_like(y[:, 0]) + (beta_tm * P_0 - alg_46_alpha_tm * N_0 + g_01 * alg_47_N_1)
        d_P_1 = torch.zeros_like(y[:, 0]) + (-(beta_tm + f_12 + g_01) * P_1 + alg_46_alpha_tm * alg_47_N_1 + f_01 * P_0 + g_12 * P_2)
        alg_48_Force_norm = phi_SL * (P_1 + alg_47_N_1 + 2.0 * P_2 + 3.0 * P_3) / Force_max
        d_Trpn_Ca = torch.zeros_like(y[:, 0]) + (alpha_trpn * (Trpn_tot - Trpn_Ca) * Ca_i - beta_trpn * (1.0 + 2.0 * (1.0 - alg_48_Force_norm)) / 3.0 * Trpn_Ca)
        alg_45_dCmdn_Ca_dtime = alpha_cmdn * (Cmdn_tot - Cmdn_Ca) * Ca_i - beta_cmdn * Cmdn_Ca
        alg_49_dTrpn_Ca_dtime = alpha_trpn * (Trpn_tot - Trpn_Ca) * Ca_i - beta_trpn * (1.0 + 2.0 * (1.0 - alg_48_Force_norm)) / 3.0 * Trpn_Ca
        d_Ca_i = torch.zeros_like(y[:, 0]) + (-(alg_25_i_Ca_L_Ca + alg_35_i_b_Ca - 2.0 * alg_30_i_NaCa) / (2.0 * v_i * F) - alg_44_j_up + alg_41_j_rel * v_SR / v_i - alg_45_dCmdn_Ca_dtime - alg_49_dTrpn_Ca_dtime)

        return torch.stack(
            [d_V, d_K_i, d_Na_i, d_Ca_i, d_m, d_h, d_x, d_s, d_r, d_d, d_f, d_F_CaMK, d_Cmdn_Ca, d_Ca_SR, d_F_SRCa_RyR, d_F_1, d_F_2, d_Trpn_Ca, d_N_0, d_P_0, d_P_1, d_P_2, d_P_3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
