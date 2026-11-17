import torch
import torch.nn as nn
import math


class ElectrophysiologyEspinosa1998HypertrophicODE(nn.Module):
    """
    PyTorch translation of electrophysiology/espinosa_1998_hypertrophic/model.py.
    Dynamic states: 21; independent constants: 65; derived constants used by rates: 5.
    Original constant slots: 70; algebraic slots: 52.
    """
    source_file = 'electrophysiology/espinosa_1998_hypertrophic/model.py'
    source_state_dim = 21
    source_constant_dim = 70
    algebraic_dim = 52
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 21
        self.parameter_dim = 65

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8314.472, 'T': 310, 'F': 96485.3415, 'Cm': 9.5e-05, 'stim_start': 0.02, 'stim_end': 9, 'stim_period': 0.5, 'stim_duration': 0.002, 'stim_amplitude': -1.8, 'Na_o': 148.5, 'Ca_o': 2.5, 'K_mk1': 10, 'g_K1': 0.12, 'i_K_max': 1.5, 'delta_x': 0.0001, 'g_Na': 0.6, 'delta_m': 1e-05, 'shift_h': 0, 'g_bna': 0.0001, 'g_bk': 0.004, 'P_Ca_L': 0.045, 'P_CaK': 0.003, 'P_CaNa': 0.01, 'Km_f2': 100000, 'Km_f2ds': 0.001, 'R_decay': 20, 'FrICa': 1, 'speed_d': 10, 'speed_f': 2, 'delta_f': 0.0001, 'g_bca': 0.00025, 'g_to': 0.048, 'g_tos': 0.15, 'i_NaK_max': 0.7, 'K_mK': 1, 'K_mNa': 40, 'k_NaCa': 0.0002, 'n_NaCa': 3, 'd_NaCa': 0.001, 'gamma': 0.5, 'FRiNaCa': 0.001, 'K_cyca': 0.0003, 'K_xcs': 0.4, 'K_srca': 0.5, 'alpha_up': 0.4, 'beta_up': 0.03, 'K_leak_rate': 0.05, 'K_m_rel': 250, 'K_m_Ca_cyt': 0.0005, 'K_m_Ca_ds': 0.01, 'K_b': 5.4, 'pf': 0.7, 'Calmod': 0.02, 'Trop': 0.05, 'alpha_Calmod': 100000, 'beta_Calmod': 50, 'alpha_Trop': 100000, 'beta_Trop': 200, 'radius': 12, 'length': 74, 'V_ds_ratio': 0.1, 'V_rel_ratio': 0.1, 'V_e_ratio': 0.4, 'V_up_ratio': 0.01, 'Kdecay': 10}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -85.35765, 'K_o': 5.3367, 'K_i': 140.096, 'Na_i': 6.9366, 'Ca_i': 3.792e-05, 'x': 3.5095e-06, 'm': 0.00448, 'h': 0.9704, 'Ca_ds': 0.00077, 'd': 4.171e-12, 'f': 0.999997, 'f2': 0.99279, 'f2ds': 0.459, 's': 0.9379, 'r': 2.6578e-05, 'Ca_up': 0.3342, 'Ca_rel': 0.31007, 'ActFrac': 0.0112, 'ProdFrac': 0.9059, 'Ca_Calmod': 0.001419, 'Ca_Trop': 0.000932}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'Cm', 'stim_start', 'stim_end', 'stim_period', 'stim_duration', 'stim_amplitude', 'Na_o', 'Ca_o', 'K_mk1', 'g_K1', 'i_K_max', 'delta_x', 'g_Na', 'delta_m', 'shift_h', 'g_bna', 'g_bk', 'P_Ca_L', 'P_CaK', 'P_CaNa', 'Km_f2', 'Km_f2ds', 'R_decay', 'FrICa', 'speed_d', 'speed_f', 'delta_f', 'g_bca', 'g_to', 'g_tos', 'i_NaK_max', 'K_mK', 'K_mNa', 'k_NaCa', 'n_NaCa', 'd_NaCa', 'gamma', 'FRiNaCa', 'K_cyca', 'K_xcs', 'K_srca', 'alpha_up', 'beta_up', 'K_leak_rate', 'K_m_rel', 'K_m_Ca_cyt', 'K_m_Ca_ds', 'K_b', 'pf', 'Calmod', 'Trop', 'alpha_Calmod', 'beta_Calmod', 'alpha_Trop', 'beta_Trop', 'radius', 'length', 'V_ds_ratio', 'V_rel_ratio', 'V_e_ratio', 'V_up_ratio', 'Kdecay']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'K_o', 'K_i', 'Na_i', 'Ca_i', 'x', 'm', 'h', 'Ca_ds', 'd', 'f', 'f2', 'f2ds', 's', 'r', 'Ca_up', 'Ca_rel', 'ActFrac', 'ProdFrac', 'Ca_Calmod', 'Ca_Trop']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (21)
        V = y[:, 0]
        K_o = y[:, 1]
        K_i = y[:, 2]
        Na_i = y[:, 3]
        Ca_i = y[:, 4]
        x = y[:, 5]
        m = y[:, 6]
        h = y[:, 7]
        Ca_ds = y[:, 8]
        d = y[:, 9]
        f = y[:, 10]
        f2 = y[:, 11]
        f2ds = y[:, 12]
        s = y[:, 13]
        r = y[:, 14]
        Ca_up = y[:, 15]
        Ca_rel = y[:, 16]
        ActFrac = y[:, 17]
        ProdFrac = y[:, 18]
        Ca_Calmod = y[:, 19]
        Ca_Trop = y[:, 20]

        # Independent parameters (source constant order)
        R = theta[:, 0]  # constants[0]
        T = theta[:, 1]  # constants[1]
        F = theta[:, 2]  # constants[2]
        Cm = theta[:, 3]  # constants[3]
        stim_start = theta[:, 4]  # constants[4]
        stim_end = theta[:, 5]  # constants[5]
        stim_period = theta[:, 6]  # constants[6]
        stim_duration = theta[:, 7]  # constants[7]
        stim_amplitude = theta[:, 8]  # constants[8]
        Na_o = theta[:, 9]  # constants[9]
        Ca_o = theta[:, 10]  # constants[10]
        K_mk1 = theta[:, 11]  # constants[11]
        g_K1 = theta[:, 12]  # constants[12]
        i_K_max = theta[:, 13]  # constants[13]
        delta_x = theta[:, 14]  # constants[14]
        g_Na = theta[:, 15]  # constants[15]
        delta_m = theta[:, 16]  # constants[16]
        shift_h = theta[:, 17]  # constants[17]
        g_bna = theta[:, 18]  # constants[18]
        g_bk = theta[:, 19]  # constants[19]
        P_Ca_L = theta[:, 20]  # constants[20]
        P_CaK = theta[:, 21]  # constants[21]
        P_CaNa = theta[:, 22]  # constants[22]
        Km_f2 = theta[:, 23]  # constants[23]
        Km_f2ds = theta[:, 24]  # constants[24]
        R_decay = theta[:, 25]  # constants[25]
        FrICa = theta[:, 26]  # constants[26]
        speed_d = theta[:, 27]  # constants[27]
        speed_f = theta[:, 28]  # constants[28]
        delta_f = theta[:, 29]  # constants[29]
        g_bca = theta[:, 30]  # constants[30]
        g_to = theta[:, 31]  # constants[31]
        g_tos = theta[:, 32]  # constants[32]
        i_NaK_max = theta[:, 33]  # constants[33]
        K_mK = theta[:, 34]  # constants[34]
        K_mNa = theta[:, 35]  # constants[35]
        k_NaCa = theta[:, 36]  # constants[36]
        n_NaCa = theta[:, 37]  # constants[37]
        d_NaCa = theta[:, 38]  # constants[38]
        gamma = theta[:, 39]  # constants[39]
        FRiNaCa = theta[:, 40]  # constants[40]
        K_cyca = theta[:, 41]  # constants[41]
        K_xcs = theta[:, 42]  # constants[42]
        K_srca = theta[:, 43]  # constants[43]
        alpha_up = theta[:, 44]  # constants[44]
        beta_up = theta[:, 45]  # constants[45]
        K_leak_rate = theta[:, 46]  # constants[46]
        K_m_rel = theta[:, 47]  # constants[47]
        K_m_Ca_cyt = theta[:, 48]  # constants[48]
        K_m_Ca_ds = theta[:, 49]  # constants[49]
        K_b = theta[:, 50]  # constants[50]
        pf = theta[:, 51]  # constants[51]
        Calmod = theta[:, 52]  # constants[52]
        Trop = theta[:, 53]  # constants[53]
        alpha_Calmod = theta[:, 54]  # constants[54]
        beta_Calmod = theta[:, 55]  # constants[55]
        alpha_Trop = theta[:, 56]  # constants[56]
        beta_Trop = theta[:, 57]  # constants[57]
        radius = theta[:, 58]  # constants[58]
        length = theta[:, 59]  # constants[59]
        V_ds_ratio = theta[:, 60]  # constants[60]
        V_rel_ratio = theta[:, 61]  # constants[61]
        V_e_ratio = theta[:, 62]  # constants[62]
        V_up_ratio = theta[:, 63]  # constants[63]
        Kdecay = theta[:, 64]  # constants[64]

        # Derived constants from the original initConsts()
        V_Cell = 3.14159 * (radius / 1000.0) ** 2.0 * length / 1000.0  # constants[65]
        K_1 = K_cyca * K_xcs / K_srca  # constants[66]
        V_i_ratio = 1.0 - V_e_ratio - V_up_ratio - V_rel_ratio  # constants[67]
        V_e = V_Cell * V_e_ratio  # constants[68]
        V_i = V_Cell * V_i_ratio  # constants[69]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_K_o = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_x = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_Ca_ds = torch.zeros_like(y[:, 0])
        d_d = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_f2 = torch.zeros_like(y[:, 0])
        d_f2ds = torch.zeros_like(y[:, 0])
        d_s = torch.zeros_like(y[:, 0])
        d_r = torch.zeros_like(y[:, 0])
        d_Ca_up = torch.zeros_like(y[:, 0])
        d_Ca_rel = torch.zeros_like(y[:, 0])
        d_ActFrac = torch.zeros_like(y[:, 0])
        d_ProdFrac = torch.zeros_like(y[:, 0])
        d_Ca_Calmod = torch.zeros_like(y[:, 0])
        d_Ca_Trop = torch.zeros_like(y[:, 0])
        alg_0_E0_x = torch.zeros_like(y[:, 0])
        alg_1_E0_m = torch.zeros_like(y[:, 0])
        alg_2_alpha_h = torch.zeros_like(y[:, 0])
        alg_3_E0_d = torch.zeros_like(y[:, 0])
        alg_4_E0_f = torch.zeros_like(y[:, 0])
        alg_5_alpha_s = torch.zeros_like(y[:, 0])
        alg_6_i_Stim = torch.zeros_like(y[:, 0])
        alg_7_VoltDep = torch.zeros_like(y[:, 0])
        alg_8_alpha_x = torch.zeros_like(y[:, 0])
        alg_9_alpha_m = torch.zeros_like(y[:, 0])
        alg_10_beta_h = torch.zeros_like(y[:, 0])
        alg_11_alpha_d = torch.zeros_like(y[:, 0])
        alg_12_alpha_f = torch.zeros_like(y[:, 0])
        alg_13_beta_s = torch.zeros_like(y[:, 0])
        alg_14_E_Na = torch.zeros_like(y[:, 0])
        alg_15_CaiReg = torch.zeros_like(y[:, 0])
        alg_16_beta_x = torch.zeros_like(y[:, 0])
        alg_17_beta_m = torch.zeros_like(y[:, 0])
        alg_18_beta_d = torch.zeros_like(y[:, 0])
        alg_19_beta_f = torch.zeros_like(y[:, 0])
        alg_20_E_K = torch.zeros_like(y[:, 0])
        alg_21_CadsReg = torch.zeros_like(y[:, 0])
        alg_22_E_Ca = torch.zeros_like(y[:, 0])
        alg_23_RegBindSite = torch.zeros_like(y[:, 0])
        alg_24_E_mh = torch.zeros_like(y[:, 0])
        alg_25_ActRate = torch.zeros_like(y[:, 0])
        alg_26_i_K1 = torch.zeros_like(y[:, 0])
        alg_27_InactRate = torch.zeros_like(y[:, 0])
        alg_28_I_K = torch.zeros_like(y[:, 0])
        alg_29_SpeedRel = torch.zeros_like(y[:, 0])
        alg_30_i_K = torch.zeros_like(y[:, 0])
        alg_31_PrecFrac = torch.zeros_like(y[:, 0])
        alg_32_i_Na = torch.zeros_like(y[:, 0])
        alg_33_i_b_Na = torch.zeros_like(y[:, 0])
        alg_34_i_b_K = torch.zeros_like(y[:, 0])
        alg_35_i_Ca_L_Ca_cyt = torch.zeros_like(y[:, 0])
        alg_36_i_Ca_L_K_cyt = torch.zeros_like(y[:, 0])
        alg_37_i_Ca_L_Na_cyt = torch.zeros_like(y[:, 0])
        alg_38_i_Ca_L_Ca_ds = torch.zeros_like(y[:, 0])
        alg_39_i_Ca_L_K_ds = torch.zeros_like(y[:, 0])
        alg_40_i_Ca_L_Na_ds = torch.zeros_like(y[:, 0])
        alg_42_i_b_Ca = torch.zeros_like(y[:, 0])
        alg_43_i_to = torch.zeros_like(y[:, 0])
        alg_44_i_NaK = torch.zeros_like(y[:, 0])
        alg_45_i_NaCa_cyt = torch.zeros_like(y[:, 0])
        alg_46_i_NaCa_ds = torch.zeros_like(y[:, 0])
        alg_47_K_2 = torch.zeros_like(y[:, 0])
        alg_49_i_up = torch.zeros_like(y[:, 0])
        alg_50_i_trans = torch.zeros_like(y[:, 0])
        alg_51_i_rel = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_f2 = torch.zeros_like(y[:, 0]) + (1.0 - 1.0 * (Ca_i / (Km_f2 + Ca_i) + f2))
        d_f2ds = torch.zeros_like(y[:, 0]) + (R_decay * (1.0 - (Ca_ds / (Km_f2ds + Ca_ds) + f2ds)))
        d_r = torch.zeros_like(y[:, 0]) + (333.0 * (1.0 / (1.0 + torch.exp(-(V + 4.0 - 24.0) / 10.0)) - r))
        alg_2_alpha_h = 20.0 * torch.exp(-0.125 * (V + 75.0 - shift_h))
        alg_10_beta_h = 2000.0 / (1.0 + 320.0 * torch.exp(-0.1 * (V + 75.0 - shift_h)))
        d_h = torch.zeros_like(y[:, 0]) + (alg_2_alpha_h * (1.0 - h) - alg_10_beta_h * h)
        alg_5_alpha_s = 0.26 * 0.033 * torch.exp(-V / 14.875)
        alg_13_beta_s = 0.26 * 33.0 / (1.0 + torch.exp(-(V + 10.0) / 7.0))
        d_s = torch.zeros_like(y[:, 0]) + (alg_5_alpha_s * (1.0 - s) - alg_13_beta_s * s)
        alg_0_E0_x = V + 22.0 - 40.0
        alg_8_alpha_x = torch.where(torch.abs(alg_0_E0_x) < delta_x, 2.5, 3.0 * 0.5 * alg_0_E0_x / (1.0 - torch.exp(-alg_0_E0_x / 5.0)))
        alg_16_beta_x = torch.where(torch.abs(alg_0_E0_x) < delta_x, 2.5, 3.0 * 0.178 * alg_0_E0_x / (torch.exp(alg_0_E0_x / 15.0) - 1.0))
        d_x = torch.zeros_like(y[:, 0]) + (alg_8_alpha_x * (1.0 - x) - alg_16_beta_x * x)
        alg_1_E0_m = V + 41.0
        alg_9_alpha_m = torch.where(torch.abs(alg_1_E0_m) < delta_m, 2000.0, 200.0 * alg_1_E0_m / (1.0 - torch.exp(-0.1 * alg_1_E0_m)))
        alg_17_beta_m = 8000.0 * torch.exp(-0.056 * (V + 66.0))
        d_m = torch.zeros_like(y[:, 0]) + (alg_9_alpha_m * (1.0 - m) - alg_17_beta_m * m)
        alg_3_E0_d = V + 24.0 - 20.0
        alg_11_alpha_d = torch.where(torch.abs(alg_3_E0_d) < 0.0001, 120.0, 30.0 * alg_3_E0_d / (1.0 - torch.exp(-alg_3_E0_d / 3.0)))
        alg_18_beta_d = torch.where(torch.abs(alg_3_E0_d) < 0.0001, 120.0, 12.0 * alg_3_E0_d / (torch.exp(alg_3_E0_d / 7.5) - 1.0))
        d_d = torch.zeros_like(y[:, 0]) + (speed_d * (alg_11_alpha_d * (1.0 - d) - alg_18_beta_d * d))
        alg_4_E0_f = V + 34.0 - 10.0
        alg_12_alpha_f = torch.where(torch.abs(alg_4_E0_f) < delta_f, 25.0, 6.25 * alg_4_E0_f / (torch.exp(alg_4_E0_f / 5.5) - 1.0))
        alg_19_beta_f = 12.0 / (1.0 + torch.exp(-1.0 * (V + 34.0) / 5.5))
        d_f = torch.zeros_like(y[:, 0]) + (speed_f * (alg_12_alpha_f * (1.0 - f) - alg_19_beta_f * f))
        alg_15_CaiReg = Ca_i / (Ca_i + K_m_Ca_cyt)
        alg_21_CadsReg = Ca_ds / (Ca_ds + K_m_Ca_ds)
        alg_23_RegBindSite = alg_15_CaiReg + (1.0 - alg_15_CaiReg) * alg_21_CadsReg
        alg_27_InactRate = 60.0 + 500.0 * alg_23_RegBindSite ** 2.0
        alg_29_SpeedRel = torch.where(V < -50.0, 5.0, 1.0)
        d_ProdFrac = torch.zeros_like(y[:, 0]) + (ActFrac * alg_29_SpeedRel * alg_27_InactRate - alg_29_SpeedRel * 1.0 * ProdFrac)
        alg_7_VoltDep = torch.exp(0.08 * (V - 40.0))
        alg_25_ActRate = 0.0 * alg_7_VoltDep + 500.0 * alg_23_RegBindSite ** 2.0
        alg_31_PrecFrac = 1.0 - ActFrac - ProdFrac
        d_ActFrac = torch.zeros_like(y[:, 0]) + (alg_31_PrecFrac * alg_29_SpeedRel * alg_25_ActRate - ActFrac * alg_29_SpeedRel * alg_27_InactRate)
        alg_38_i_Ca_L_Ca_ds = FrICa * 4.0 * P_Ca_L * d * f * f2ds * (V - 50.0) * F / (R * T) / (1.0 - torch.exp(-(V - 50.0) * F * 2.0 / (R * T))) * (Ca_i * torch.exp(100.0 * F / (R * T)) - Ca_o * torch.exp(-(V - 50.0) * F * 2.0 / (R * T)))
        d_Ca_ds = torch.zeros_like(y[:, 0]) + (-1.0 * alg_38_i_Ca_L_Ca_ds / (2.0 * V_ds_ratio * 1.0 * V_i * F) - Ca_ds * Kdecay)
        alg_20_E_K = R * T / F * torch.log(K_o / K_i)
        alg_26_i_K1 = g_K1 * K_o / (K_o + K_mk1) * (V - alg_20_E_K) / (1.0 + torch.exp((V - alg_20_E_K + 10.0) * F * 1.67 / (R * T)))
        alg_43_i_to = g_to * (g_tos + s * (1.0 - g_tos)) * r * (V - alg_20_E_K)
        alg_28_I_K = i_K_max * (K_i - K_o * torch.exp(-V * F / (R * T))) / 140.0
        alg_30_i_K = x * alg_28_I_K
        alg_36_i_Ca_L_K_cyt = (1.0 - FrICa) * P_CaK * P_Ca_L * d * f * f2 * (V - 50.0) * F / (R * T) / (1.0 - torch.exp(-(V - 50.0) * F / (R * T))) * (K_i * torch.exp(50.0 * F / (R * T)) - K_o * torch.exp(-(V - 50.0) * F / (R * T)))
        alg_39_i_Ca_L_K_ds = FrICa * P_CaK * P_Ca_L * d * f * f2ds * (V - 50.0) * F / (R * T) / (1.0 - torch.exp(-(V - 50.0) * F / (R * T))) * (K_i * torch.exp(50.0 * F / (R * T)) - K_o * torch.exp(-(V - 50.0) * F / (R * T)))
        alg_44_i_NaK = i_NaK_max * K_o / (K_mK + K_o) * Na_i / (K_mNa + Na_i)
        alg_34_i_b_K = g_bk * (V - alg_20_E_K)
        d_K_o = torch.zeros_like(y[:, 0]) + (1.0 * (alg_26_i_K1 + alg_43_i_to + alg_36_i_Ca_L_K_cyt + alg_39_i_Ca_L_K_ds + -2.0 * alg_44_i_NaK + alg_30_i_K + alg_34_i_b_K) / (1.0 * V_e * F) - pf * (K_o - K_b))
        d_K_i = torch.zeros_like(y[:, 0]) + (-1.0 / (1.0 * V_i * F) * (alg_26_i_K1 + alg_30_i_K + alg_36_i_Ca_L_K_cyt + alg_39_i_Ca_L_K_ds + alg_43_i_to + alg_34_i_b_K - 2.0 * alg_44_i_NaK))
        alg_24_E_mh = R * T / F * torch.log((Na_o + 0.12 * K_o) / (Na_i + 0.12 * K_i))
        alg_32_i_Na = g_Na * m ** 3.0 * h * (V - alg_24_E_mh)
        alg_14_E_Na = R * T / F * torch.log(Na_o / Na_i)
        alg_33_i_b_Na = g_bna * (V - alg_14_E_Na)
        alg_37_i_Ca_L_Na_cyt = (1.0 - FrICa) * P_CaNa * P_Ca_L * d * f * f2 * (V - 50.0) * F / (R * T) / (1.0 - torch.exp(-(V - 50.0) * F / (R * T))) * (Na_i * torch.exp(50.0 * F / (R * T)) - Na_o * torch.exp(-(V - 50.0) * F / (R * T)))
        alg_40_i_Ca_L_Na_ds = FrICa * P_CaNa * P_Ca_L * d * f * f2ds * (V - 50.0) * F / (R * T) / (1.0 - torch.exp(-(V - 50.0) * F / (R * T))) * (Na_i * torch.exp(50.0 * F / (R * T)) - Na_o * torch.exp(-(V - 50.0) * F / (R * T)))
        alg_45_i_NaCa_cyt = (1.0 - FRiNaCa) * k_NaCa * (torch.exp(gamma * (n_NaCa - 2.0) * V * F / (R * T)) * Na_i ** n_NaCa * Ca_o - torch.exp((gamma - 1.0) * (n_NaCa - 2.0) * V * F / (R * T)) * Na_o ** n_NaCa * Ca_i) / ((1.0 + d_NaCa * (Ca_i * Na_o ** n_NaCa + Ca_o * Na_i ** n_NaCa)) * (1.0 + Ca_i / 0.0069))
        d_Na_i = torch.zeros_like(y[:, 0]) + (-1.0 / (1.0 * V_i * F) * (alg_32_i_Na + alg_33_i_b_Na + 3.0 * alg_44_i_NaK + 3.0 * alg_45_i_NaCa_cyt + alg_37_i_Ca_L_Na_cyt + alg_40_i_Ca_L_Na_ds))
        alg_46_i_NaCa_ds = FRiNaCa * k_NaCa * (torch.exp(gamma * (n_NaCa - 2.0) * V * F / (R * T)) * Na_i ** n_NaCa * Ca_o - torch.exp((gamma - 1.0) * (n_NaCa - 2.0) * V * F / (R * T)) * Na_o ** n_NaCa * Ca_ds) / ((1.0 + d_NaCa * (Ca_ds * Na_o ** n_NaCa + Ca_o * Na_i ** n_NaCa)) * (1.0 + Ca_ds / 0.0069))
        alg_35_i_Ca_L_Ca_cyt = (1.0 - FrICa) * 4.0 * P_Ca_L * d * f * f2 * (V - 50.0) * F / (R * T) / (1.0 - torch.exp(-(V - 50.0) * F * 2.0 / (R * T))) * (Ca_i * torch.exp(100.0 * F / (R * T)) - Ca_o * torch.exp(-(V - 50.0) * F * 2.0 / (R * T)))
        alg_22_E_Ca = 0.5 * R * T / F * torch.log(Ca_o / Ca_i)
        alg_42_i_b_Ca = g_bca * (V - alg_22_E_Ca)
        alg_6_i_Stim = torch.where((t >= stim_start) & (t <= stim_end) & (t - stim_start - torch.floor((t - stim_start) / stim_period) * stim_period <= stim_duration), stim_amplitude, 0.0)
        d_V = torch.zeros_like(y[:, 0]) + (-1.0 / Cm * (alg_6_i_Stim + alg_26_i_K1 + alg_43_i_to + alg_30_i_K + alg_44_i_NaK + alg_32_i_Na + alg_33_i_b_Na + alg_37_i_Ca_L_Na_cyt + alg_40_i_Ca_L_Na_ds + alg_45_i_NaCa_cyt + alg_46_i_NaCa_ds + alg_35_i_Ca_L_Ca_cyt + alg_38_i_Ca_L_Ca_ds + alg_36_i_Ca_L_K_cyt + alg_39_i_Ca_L_K_ds + alg_42_i_b_Ca + alg_34_i_b_K))
        alg_47_K_2 = Ca_i + Ca_up * K_1 + K_cyca * K_xcs + K_cyca
        alg_49_i_up = Ca_i / alg_47_K_2 * alpha_up - Ca_up * K_1 / alg_47_K_2 * beta_up
        alg_50_i_trans = 50.0 * (Ca_up - Ca_rel)
        d_Ca_up = torch.zeros_like(y[:, 0]) + (V_i_ratio / V_up_ratio * alg_49_i_up - alg_50_i_trans)
        d_Ca_Calmod = torch.zeros_like(y[:, 0]) + (alpha_Calmod * Ca_i * (Calmod - Ca_Calmod) - beta_Calmod * Ca_Calmod)
        alg_51_i_rel = ((ActFrac / (ActFrac + 0.25)) ** 2.0 * K_m_rel + K_leak_rate) * Ca_rel
        d_Ca_rel = torch.zeros_like(y[:, 0]) + (V_up_ratio / V_rel_ratio * alg_50_i_trans - alg_51_i_rel)
        d_Ca_Trop = torch.zeros_like(y[:, 0]) + (alpha_Trop * Ca_i * (Trop - Ca_Trop) - beta_Trop * Ca_Trop)
        d_Ca_i = torch.zeros_like(y[:, 0]) + (-1.0 / (2.0 * 1.0 * V_i * F) * (alg_35_i_Ca_L_Ca_cyt + alg_42_i_b_Ca - 2.0 * alg_45_i_NaCa_cyt) + Ca_ds * V_ds_ratio * Kdecay + alg_51_i_rel * V_rel_ratio / V_i_ratio - d_Ca_Calmod - d_Ca_Trop - alg_49_i_up)

        return torch.stack(
            [d_V, d_K_o, d_K_i, d_Na_i, d_Ca_i, d_x, d_m, d_h, d_Ca_ds, d_d, d_f, d_f2, d_f2ds, d_s, d_r, d_Ca_up, d_Ca_rel, d_ActFrac, d_ProdFrac, d_Ca_Calmod, d_Ca_Trop],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
