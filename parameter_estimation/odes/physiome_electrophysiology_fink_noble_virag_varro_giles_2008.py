import torch
import torch.nn as nn
import math


class ElectrophysiologyFinkNobleViragVarroGiles2008ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/fink_noble_virag_varro_giles_2008/model.py.
    Dynamic states: 27; independent constants: 61; derived constants used by rates: 0.
    Original constant slots: 61; algebraic slots: 78.
    """
    source_file = 'electrophysiology/fink_noble_virag_varro_giles_2008/model.py'
    source_state_dim = 27
    source_constant_dim = 61
    algebraic_dim = 78
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 27
        self.parameter_dim = 61

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8314.472, 'T': 310, 'F': 96485.3415, 'K_o': 5.4, 'Ca_o': 2, 'Na_o': 140, 'Cm': 0.115, 'Vol_c': 0.016404, 'stim_Period': 1000, 'P_kna': 0.03, 'g_K1_0': 0.6821, 'Mg_Buf': 0.0356, 'SPM': 0.0014613, 'fac': 1.0648, 'phi': 0.8838, 'g_Kr_0': 0.024, 'T_Base': 310, 'Sotalol_mM': 0, 'kBinding': 0.005, 'kDiss': 0.00125, 'g_Ks': 0.0392, 'g_to': 0.2, 'g_Na': 11, 'shift_INa_inact': 0, 'g_bna': 0.00029, 'g_CaL': 2e-05, 'z': 2, 'd_inf_shift': 5, 'g_bca': 0.0004736, 'g_pCa': 0.0619, 'K_pCa': 0.0005, 'g_pK': 0.00973, 'P_NaK': 1.297, 'K_mk': 1, 'K_mNa': 40, 'K_NaCa': 200, 'K_sat': 0.1, 'alpha': 2.5, 'gamma': 0.35, 'Km_Ca': 1.38, 'Km_Nai': 87.5, 'V_sr': 0.001094, 'V_ss': 5.468e-05, 'Vol_xfer': 0.0038, 'K_up': 0.00025, 'Vol_leak': 0.00036, 'Vmax_up': 0.006375, 'k1_prime': 0.15, 'k2_prime': 0.045, 'EC': 1.5, 'max_sr': 2.5, 'min_sr': 1, 'k3': 0.06, 'k4': 0.005, 'Vol_rel': 0.306, 'Buf_c': 0.2, 'K_buf_c': 0.001, 'Buf_sr': 10, 'K_buf_sr': 0.3, 'Buf_ss': 0.4, 'K_buf_ss': 0.00025}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -86.45, 'K_i': 141.0167, 'Na_i': 7.940167, 'Ca_i': 0.0001092, 'Or4': 0.014, 'Cr1': 0.9786, 'Cr2': 0.0031, 'Cr3': 0.0029, 'Ir5': 0.0014, 'BCr1': 0, 'BCr2': 0, 'BCr3': 0, 'BOr4': 0, 'BIr5': 0, 'Xs': 0.00303, 's': 1, 'r': 2.11e-08, 'm': 0.00132, 'h': 0.7768, 'j': 0.7766, 'Ca_ss': 0.0001893, 'd': 5.06e-06, 'f': 0.9999, 'f2': 0.9995, 'fCass': 1, 'Ca_SR': 2.7656, 'R_prime': 0.9864}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'K_o', 'Ca_o', 'Na_o', 'Cm', 'Vol_c', 'stim_Period', 'P_kna', 'g_K1_0', 'Mg_Buf', 'SPM', 'fac', 'phi', 'g_Kr_0', 'T_Base', 'Sotalol_mM', 'kBinding', 'kDiss', 'g_Ks', 'g_to', 'g_Na', 'shift_INa_inact', 'g_bna', 'g_CaL', 'z', 'd_inf_shift', 'g_bca', 'g_pCa', 'K_pCa', 'g_pK', 'P_NaK', 'K_mk', 'K_mNa', 'K_NaCa', 'K_sat', 'alpha', 'gamma', 'Km_Ca', 'Km_Nai', 'V_sr', 'V_ss', 'Vol_xfer', 'K_up', 'Vol_leak', 'Vmax_up', 'k1_prime', 'k2_prime', 'EC', 'max_sr', 'min_sr', 'k3', 'k4', 'Vol_rel', 'Buf_c', 'K_buf_c', 'Buf_sr', 'K_buf_sr', 'Buf_ss', 'K_buf_ss']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'K_i', 'Na_i', 'Ca_i', 'Or4', 'Cr1', 'Cr2', 'Cr3', 'Ir5', 'BCr1', 'BCr2', 'BCr3', 'BOr4', 'BIr5', 'Xs', 's', 'r', 'm', 'h', 'j', 'Ca_ss', 'd', 'f', 'f2', 'fCass', 'Ca_SR', 'R_prime']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (27)
        V = y[:, 0]
        K_i = y[:, 1]
        Na_i = y[:, 2]
        Ca_i = y[:, 3]
        Or4 = y[:, 4]
        Cr1 = y[:, 5]
        Cr2 = y[:, 6]
        Cr3 = y[:, 7]
        Ir5 = y[:, 8]
        BCr1 = y[:, 9]
        BCr2 = y[:, 10]
        BCr3 = y[:, 11]
        BOr4 = y[:, 12]
        BIr5 = y[:, 13]
        Xs = y[:, 14]
        s = y[:, 15]
        r = y[:, 16]
        m = y[:, 17]
        h = y[:, 18]
        j = y[:, 19]
        Ca_ss = y[:, 20]
        d = y[:, 21]
        f = y[:, 22]
        f2 = y[:, 23]
        fCass = y[:, 24]
        Ca_SR = y[:, 25]
        R_prime = y[:, 26]

        # Independent parameters (source constant order)
        R = theta[:, 0]  # constants[0]
        T = theta[:, 1]  # constants[1]
        F = theta[:, 2]  # constants[2]
        K_o = theta[:, 3]  # constants[3]
        Ca_o = theta[:, 4]  # constants[4]
        Na_o = theta[:, 5]  # constants[5]
        Cm = theta[:, 6]  # constants[6]
        Vol_c = theta[:, 7]  # constants[7]
        stim_Period = theta[:, 8]  # constants[8]
        P_kna = theta[:, 9]  # constants[9]
        g_K1_0 = theta[:, 10]  # constants[10]
        Mg_Buf = theta[:, 11]  # constants[11]
        SPM = theta[:, 12]  # constants[12]
        fac = theta[:, 13]  # constants[13]
        phi = theta[:, 14]  # constants[14]
        g_Kr_0 = theta[:, 15]  # constants[15]
        T_Base = theta[:, 16]  # constants[16]
        Sotalol_mM = theta[:, 17]  # constants[17]
        kBinding = theta[:, 18]  # constants[18]
        kDiss = theta[:, 19]  # constants[19]
        g_Ks = theta[:, 20]  # constants[20]
        g_to = theta[:, 21]  # constants[21]
        g_Na = theta[:, 22]  # constants[22]
        shift_INa_inact = theta[:, 23]  # constants[23]
        g_bna = theta[:, 24]  # constants[24]
        g_CaL = theta[:, 25]  # constants[25]
        z = theta[:, 26]  # constants[26]
        d_inf_shift = theta[:, 27]  # constants[27]
        g_bca = theta[:, 28]  # constants[28]
        g_pCa = theta[:, 29]  # constants[29]
        K_pCa = theta[:, 30]  # constants[30]
        g_pK = theta[:, 31]  # constants[31]
        P_NaK = theta[:, 32]  # constants[32]
        K_mk = theta[:, 33]  # constants[33]
        K_mNa = theta[:, 34]  # constants[34]
        K_NaCa = theta[:, 35]  # constants[35]
        K_sat = theta[:, 36]  # constants[36]
        alpha = theta[:, 37]  # constants[37]
        gamma = theta[:, 38]  # constants[38]
        Km_Ca = theta[:, 39]  # constants[39]
        Km_Nai = theta[:, 40]  # constants[40]
        V_sr = theta[:, 41]  # constants[41]
        V_ss = theta[:, 42]  # constants[42]
        Vol_xfer = theta[:, 43]  # constants[43]
        K_up = theta[:, 44]  # constants[44]
        Vol_leak = theta[:, 45]  # constants[45]
        Vmax_up = theta[:, 46]  # constants[46]
        k1_prime = theta[:, 47]  # constants[47]
        k2_prime = theta[:, 48]  # constants[48]
        EC = theta[:, 49]  # constants[49]
        max_sr = theta[:, 50]  # constants[50]
        min_sr = theta[:, 51]  # constants[51]
        k3 = theta[:, 52]  # constants[52]
        k4 = theta[:, 53]  # constants[53]
        Vol_rel = theta[:, 54]  # constants[54]
        Buf_c = theta[:, 55]  # constants[55]
        K_buf_c = theta[:, 56]  # constants[56]
        Buf_sr = theta[:, 57]  # constants[57]
        K_buf_sr = theta[:, 58]  # constants[58]
        Buf_ss = theta[:, 59]  # constants[59]
        K_buf_ss = theta[:, 60]  # constants[60]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_Or4 = torch.zeros_like(y[:, 0])
        d_Cr1 = torch.zeros_like(y[:, 0])
        d_Cr2 = torch.zeros_like(y[:, 0])
        d_Cr3 = torch.zeros_like(y[:, 0])
        d_Ir5 = torch.zeros_like(y[:, 0])
        d_BCr1 = torch.zeros_like(y[:, 0])
        d_BCr2 = torch.zeros_like(y[:, 0])
        d_BCr3 = torch.zeros_like(y[:, 0])
        d_BOr4 = torch.zeros_like(y[:, 0])
        d_BIr5 = torch.zeros_like(y[:, 0])
        d_Xs = torch.zeros_like(y[:, 0])
        d_s = torch.zeros_like(y[:, 0])
        d_r = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_j = torch.zeros_like(y[:, 0])
        d_Ca_ss = torch.zeros_like(y[:, 0])
        d_d = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_f2 = torch.zeros_like(y[:, 0])
        d_fCass = torch.zeros_like(y[:, 0])
        d_Ca_SR = torch.zeros_like(y[:, 0])
        d_R_prime = torch.zeros_like(y[:, 0])
        alg_0_i_Stim = torch.zeros_like(y[:, 0])
        alg_1_alpha_xr1 = torch.zeros_like(y[:, 0])
        alg_2_xs_inf = torch.zeros_like(y[:, 0])
        alg_3_s_inf = torch.zeros_like(y[:, 0])
        alg_4_r_inf = torch.zeros_like(y[:, 0])
        alg_5_m_inf = torch.zeros_like(y[:, 0])
        alg_6_h_inf = torch.zeros_like(y[:, 0])
        alg_7_j_inf = torch.zeros_like(y[:, 0])
        alg_8_d_inf = torch.zeros_like(y[:, 0])
        alg_9_f_inf = torch.zeros_like(y[:, 0])
        alg_10_f2_inf = torch.zeros_like(y[:, 0])
        alg_11_fCass_inf = torch.zeros_like(y[:, 0])
        alg_12_E_Na = torch.zeros_like(y[:, 0])
        alg_13_beta_xr1 = torch.zeros_like(y[:, 0])
        alg_14_alpha_xs = torch.zeros_like(y[:, 0])
        alg_15_tau_s = torch.zeros_like(y[:, 0])
        alg_16_tau_r = torch.zeros_like(y[:, 0])
        alg_17_alpha_m = torch.zeros_like(y[:, 0])
        alg_18_alpha_h = torch.zeros_like(y[:, 0])
        alg_19_alpha_j = torch.zeros_like(y[:, 0])
        alg_20_alpha_d = torch.zeros_like(y[:, 0])
        alg_21_tau_f = torch.zeros_like(y[:, 0])
        alg_22_tau_f2 = torch.zeros_like(y[:, 0])
        alg_23_tau_fCass = torch.zeros_like(y[:, 0])
        alg_24_E_K = torch.zeros_like(y[:, 0])
        alg_25_alpha_xr2 = torch.zeros_like(y[:, 0])
        alg_26_beta_xs = torch.zeros_like(y[:, 0])
        alg_27_beta_m = torch.zeros_like(y[:, 0])
        alg_28_beta_h = torch.zeros_like(y[:, 0])
        alg_29_beta_j = torch.zeros_like(y[:, 0])
        alg_30_beta_d = torch.zeros_like(y[:, 0])
        alg_31_E_Ks = torch.zeros_like(y[:, 0])
        alg_32_beta_xr2 = torch.zeros_like(y[:, 0])
        alg_33_tau_xs = torch.zeros_like(y[:, 0])
        alg_34_tau_m = torch.zeros_like(y[:, 0])
        alg_35_tau_h = torch.zeros_like(y[:, 0])
        alg_36_tau_j = torch.zeros_like(y[:, 0])
        alg_37_gamma_d = torch.zeros_like(y[:, 0])
        alg_38_E_Ca = torch.zeros_like(y[:, 0])
        alg_39_alpha_xr3 = torch.zeros_like(y[:, 0])
        alg_40_tau_d = torch.zeros_like(y[:, 0])
        alg_41_KiMg = torch.zeros_like(y[:, 0])
        alg_42_beta_xr3 = torch.zeros_like(y[:, 0])
        alg_43_KbMg = torch.zeros_like(y[:, 0])
        alg_44_alpha_xr4 = torch.zeros_like(y[:, 0])
        alg_45_Kd1SPM = torch.zeros_like(y[:, 0])
        alg_46_beta_xr4 = torch.zeros_like(y[:, 0])
        alg_47_Kd2SPM = torch.zeros_like(y[:, 0])
        alg_48_OtoB = torch.zeros_like(y[:, 0])
        alg_49_temp = torch.zeros_like(y[:, 0])
        alg_50_BtoO = torch.zeros_like(y[:, 0])
        alg_51_rec1 = torch.zeros_like(y[:, 0])
        alg_52_rec2 = torch.zeros_like(y[:, 0])
        alg_53_xK1_inf = torch.zeros_like(y[:, 0])
        alg_54_i_K1 = torch.zeros_like(y[:, 0])
        alg_55_i_Kr = torch.zeros_like(y[:, 0])
        alg_56_i_Ks = torch.zeros_like(y[:, 0])
        alg_57_i_to = torch.zeros_like(y[:, 0])
        alg_58_i_Na = torch.zeros_like(y[:, 0])
        alg_59_i_b_Na = torch.zeros_like(y[:, 0])
        alg_60_i_CaL = torch.zeros_like(y[:, 0])
        alg_61_i_b_Ca = torch.zeros_like(y[:, 0])
        alg_62_i_p_Ca = torch.zeros_like(y[:, 0])
        alg_63_i_p_K = torch.zeros_like(y[:, 0])
        alg_64_i_NaK = torch.zeros_like(y[:, 0])
        alg_65_i_NaCa = torch.zeros_like(y[:, 0])
        alg_66_i_tot = torch.zeros_like(y[:, 0])
        alg_67_i_up = torch.zeros_like(y[:, 0])
        alg_68_i_leak = torch.zeros_like(y[:, 0])
        alg_69_i_xfer = torch.zeros_like(y[:, 0])
        alg_70_Ca_i_bufc = torch.zeros_like(y[:, 0])
        alg_71_kcasr = torch.zeros_like(y[:, 0])
        alg_72_k1 = torch.zeros_like(y[:, 0])
        alg_73_k2 = torch.zeros_like(y[:, 0])
        alg_74_O = torch.zeros_like(y[:, 0])
        alg_75_i_rel = torch.zeros_like(y[:, 0])
        alg_76_Ca_sr_bufsr = torch.zeros_like(y[:, 0])
        alg_77_Ca_ss_bufss = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_alpha_xr1 = 1.0 * T / T_Base * torch.exp(24.335 + T_Base / T * (0.0112 * V - 25.914))
        alg_13_beta_xr1 = 1.0 * T / T_Base * torch.exp(13.688 + T_Base / T * (-0.0603 * V - 15.707))
        d_Cr1 = torch.zeros_like(y[:, 0]) + (alg_13_beta_xr1 * Cr2 - alg_1_alpha_xr1 * Cr1)
        d_BCr1 = torch.zeros_like(y[:, 0]) + (alg_13_beta_xr1 * BCr2 - alg_1_alpha_xr1 * BCr1)
        alg_3_s_inf = 1.0 / (1.0 + torch.exp((V + 20.0) / 5.0))
        alg_15_tau_s = 85.0 * torch.exp(-(V + 45.0) ** 2.0 / 320.0) + 5.0 / (1.0 + torch.exp((V - 20.0) / 5.0)) + 3.0
        d_s = torch.zeros_like(y[:, 0]) + ((alg_3_s_inf - s) / alg_15_tau_s)
        alg_4_r_inf = 1.0 / (1.0 + torch.exp((20.0 - V) / 6.0))
        alg_16_tau_r = 9.5 * torch.exp(-(V + 40.0) ** 2.0 / 1800.0) + 0.8
        d_r = torch.zeros_like(y[:, 0]) + ((alg_4_r_inf - r) / alg_16_tau_r)
        alg_9_f_inf = 1.0 / (1.0 + torch.exp((V + 20.0) / 7.0))
        alg_21_tau_f = (1102.5 * torch.exp(-(V + 27.0) ** 2.0 / 225.0) + 200.0 / (1.0 + torch.exp((13.0 - V) / 10.0)) + 180.0 / (1.0 + torch.exp((V + 30.0) / 10.0)) + 20.0) / 4.0
        d_f = torch.zeros_like(y[:, 0]) + ((alg_9_f_inf - f) / alg_21_tau_f)
        alg_10_f2_inf = 0.75 / (1.0 + torch.exp((V + 35.0) / 7.0)) + 0.25
        alg_22_tau_f2 = (562.0 * torch.exp(-(V + 27.0) ** 2.0 / 240.0) + 31.0 / (1.0 + torch.exp((25.0 - V) / 10.0)) + 80.0 / (1.0 + torch.exp((V + 30.0) / 10.0))) / 2.0
        d_f2 = torch.zeros_like(y[:, 0]) + ((alg_10_f2_inf - f2) / alg_22_tau_f2)
        alg_11_fCass_inf = 0.4 / (1.0 + (Ca_ss / 0.05) ** 2.0) + 0.6
        alg_23_tau_fCass = 80.0 / (1.0 + (Ca_ss / 0.05) ** 2.0) + 2.0
        d_fCass = torch.zeros_like(y[:, 0]) + ((alg_11_fCass_inf - fCass) / alg_23_tau_fCass)
        alg_25_alpha_xr2 = 1.0 * T / T_Base * torch.exp(22.746 + T_Base / T * (0.0 * V - 25.914))
        alg_32_beta_xr2 = 1.0 * T / T_Base * torch.exp(13.193 + T_Base / T * (0.0 * V - 15.707))
        d_Cr2 = torch.zeros_like(y[:, 0]) + (alg_1_alpha_xr1 * Cr1 + alg_32_beta_xr2 * Cr3 - (alg_25_alpha_xr2 + alg_13_beta_xr1) * Cr2)
        d_BCr2 = torch.zeros_like(y[:, 0]) + (alg_1_alpha_xr1 * BCr1 + alg_32_beta_xr2 * BCr3 - (alg_25_alpha_xr2 + alg_13_beta_xr1) * BCr2)
        alg_2_xs_inf = 1.0 / (1.0 + torch.exp((-5.0 - V) / 14.0))
        alg_14_alpha_xs = 1400.0 / (1.0 + torch.exp((5.0 - V) / 6.0)) ** (1.0 / 2)
        alg_26_beta_xs = 1.0 / (1.0 + torch.exp((V - 35.0) / 15.0))
        alg_33_tau_xs = 1.0 * alg_14_alpha_xs * alg_26_beta_xs + 80.0
        d_Xs = torch.zeros_like(y[:, 0]) + ((alg_2_xs_inf - Xs) / alg_33_tau_xs)
        alg_5_m_inf = 1.0 / (1.0 + torch.exp((-56.86 - V) / 9.03)) ** 2.0
        alg_17_alpha_m = 1.0 / (1.0 + torch.exp((-60.0 - V) / 5.0))
        alg_27_beta_m = 0.1 / (1.0 + torch.exp((V + 35.0) / 5.0)) + 0.1 / (1.0 + torch.exp((V - 50.0) / 200.0))
        alg_34_tau_m = 1.0 * alg_17_alpha_m * alg_27_beta_m
        d_m = torch.zeros_like(y[:, 0]) + ((alg_5_m_inf - m) / alg_34_tau_m)
        alg_6_h_inf = 1.0 / (1.0 + torch.exp((V + 71.55 - shift_INa_inact) / 7.43)) ** 2.0
        alg_18_alpha_h = torch.where(V < -40.0 + shift_INa_inact, 0.057 * torch.exp(-(V + 80.0 - shift_INa_inact) / 6.8), 0.0)
        alg_28_beta_h = torch.where(V < -40.0 + shift_INa_inact, 2.7 * torch.exp(0.079 * (V - shift_INa_inact)) + 310000.0 * torch.exp(0.3485 * (V - shift_INa_inact)), 0.77 / (0.13 * (1.0 + torch.exp((V + 10.66 - shift_INa_inact) / -11.1))))
        alg_35_tau_h = 1.0 / (alg_18_alpha_h + alg_28_beta_h)
        d_h = torch.zeros_like(y[:, 0]) + ((alg_6_h_inf - h) / alg_35_tau_h)
        alg_7_j_inf = 1.0 / (1.0 + torch.exp((V + 71.55 - shift_INa_inact) / 7.43)) ** 2.0
        alg_19_alpha_j = torch.where(V < -40.0 + shift_INa_inact, (-25428.0 * torch.exp(0.2444 * (V - shift_INa_inact)) - 6.948e-06 * torch.exp(-0.04391 * (V - shift_INa_inact))) * (V + 37.78) / 1.0 / (1.0 + torch.exp(0.311 * (V + 79.23 - shift_INa_inact))), 0.0)
        alg_29_beta_j = torch.where(V < -40.0 + shift_INa_inact, 0.02424 * torch.exp(-0.01052 * (V - shift_INa_inact)) / (1.0 + torch.exp(-0.1378 * (V + 40.14 - shift_INa_inact))), 0.6 * torch.exp(0.057 * (V - shift_INa_inact)) / (1.0 + torch.exp(-0.1 * (V + 32.0 - shift_INa_inact))))
        alg_36_tau_j = 1.0 / (alg_19_alpha_j + alg_29_beta_j)
        d_j = torch.zeros_like(y[:, 0]) + ((alg_7_j_inf - j) / alg_36_tau_j)
        alg_8_d_inf = 1.0 / (1.0 + torch.exp((d_inf_shift - V) / 7.5))
        alg_20_alpha_d = 1.4 / (1.0 + torch.exp((-35.0 - V) / 13.0)) + 0.25
        alg_30_beta_d = 1.4 / (1.0 + torch.exp((V + 5.0) / 5.0))
        alg_37_gamma_d = 1.0 / (1.0 + torch.exp((50.0 - V) / 20.0))
        alg_40_tau_d = 1.0 * alg_20_alpha_d * alg_30_beta_d + alg_37_gamma_d
        d_d = torch.zeros_like(y[:, 0]) + ((alg_8_d_inf - d) / alg_40_tau_d)
        alg_39_alpha_xr3 = 1.0 * T / T_Base * torch.exp(22.098 + T_Base / T * (0.0365 * V - 25.914))
        alg_42_beta_xr3 = 1.0 * T / T_Base * torch.exp(7.313 + T_Base / T * (-0.0399 * V - 15.707))
        d_Cr3 = torch.zeros_like(y[:, 0]) + (alg_25_alpha_xr2 * Cr2 + alg_42_beta_xr3 * Or4 - (alg_39_alpha_xr3 + alg_32_beta_xr2) * Cr3)
        d_BCr3 = torch.zeros_like(y[:, 0]) + (alg_25_alpha_xr2 * BCr2 + alg_42_beta_xr3 * BOr4 - (alg_39_alpha_xr3 + alg_32_beta_xr2) * BCr3)
        alg_44_alpha_xr4 = 1.0 * T / T_Base * torch.exp(30.016 + T_Base / T * (0.0223 * V - 30.888)) * (5.4 / K_o) ** 0.4
        alg_46_beta_xr4 = 1.0 * T / T_Base * torch.exp(30.061 + T_Base / T * (-0.0312 * V - 33.243))
        d_Ir5 = torch.zeros_like(y[:, 0]) + (alg_44_alpha_xr4 * Or4 - alg_46_beta_xr4 * Ir5)
        d_BIr5 = torch.zeros_like(y[:, 0]) + (alg_44_alpha_xr4 * BOr4 - alg_46_beta_xr4 * BIr5)
        alg_48_OtoB = Or4 * Sotalol_mM * kBinding
        alg_50_BtoO = BOr4 * kDiss
        d_Or4 = torch.zeros_like(y[:, 0]) + (alg_39_alpha_xr3 * Cr3 + alg_46_beta_xr4 * Ir5 - (alg_44_alpha_xr4 + alg_42_beta_xr3) * Or4 - alg_48_OtoB + alg_50_BtoO)
        d_BOr4 = torch.zeros_like(y[:, 0]) + (alg_39_alpha_xr3 * BCr3 + alg_46_beta_xr4 * BIr5 - (alg_44_alpha_xr4 + alg_42_beta_xr3) * BOr4 + alg_48_OtoB - alg_50_BtoO)
        alg_24_E_K = R * T / F * torch.log(K_o / K_i)
        alg_43_KbMg = 0.45 * torch.exp(-(V - fac * alg_24_E_K) / 20.0)
        alg_49_temp = 1.0 + Mg_Buf / alg_43_KbMg
        alg_41_KiMg = 2.8 * torch.exp(-(V - fac * alg_24_E_K) / 180.0)
        alg_45_Kd1SPM = 0.0007 * torch.exp(-(V - fac * alg_24_E_K + 8.0 * Mg_Buf) / 4.8)
        alg_51_rec1 = alg_49_temp * alg_49_temp / (SPM / alg_45_Kd1SPM + Mg_Buf / alg_41_KiMg + alg_49_temp * alg_49_temp * alg_49_temp)
        alg_47_Kd2SPM = 0.04 * torch.exp(-(V - fac * alg_24_E_K) / 9.1)
        alg_52_rec2 = 1.0 / (1.0 + SPM / alg_47_Kd2SPM)
        alg_53_xK1_inf = phi * alg_51_rec1 + (1.0 - phi) * alg_52_rec2
        alg_54_i_K1 = g_K1_0 * (T / 35.0 - 55.0 / 7.0) * (K_o / 5.4) ** (1.0 / 2) * alg_53_xK1_inf * (V - alg_24_E_K)
        alg_57_i_to = g_to * r * s * (V - alg_24_E_K)
        alg_55_i_Kr = g_Kr_0 * (T / 35.0 - 55.0 / 7.0) * (K_o / 5.4) ** (1.0 / 2) * Or4 * (V - alg_24_E_K)
        alg_31_E_Ks = R * T / F * torch.log((K_o + P_kna * Na_o) / (K_i + P_kna * Na_i))
        alg_56_i_Ks = g_Ks * Xs ** 2.0 * (V - alg_31_E_Ks)
        alg_64_i_NaK = P_NaK * K_o / (K_o + K_mk) * Na_i / (Na_i + K_mNa) / (1.0 + 0.1245 * torch.exp(-0.1 * V * F / (R * T)) + 0.0353 * torch.exp(-V * F / (R * T)))
        alg_63_i_p_K = g_pK * (V - alg_24_E_K) / (1.0 + torch.exp((25.0 - V) / 5.98))
        alg_0_i_Stim = torch.where((t - torch.floor(t / stim_Period) * stim_Period >= 100.0) & (t - torch.floor(t / stim_Period) * stim_Period <= 103.0), -12.0, 0.0)
        d_K_i = torch.zeros_like(y[:, 0]) + (-(alg_54_i_K1 + alg_57_i_to + alg_55_i_Kr + alg_56_i_Ks + alg_63_i_p_K + alg_0_i_Stim - 2.0 * alg_64_i_NaK) / (Vol_c * F) * Cm)
        alg_12_E_Na = R * T / F * torch.log(Na_o / Na_i)
        alg_58_i_Na = g_Na * m ** 3.0 * h * j * (V - alg_12_E_Na)
        alg_59_i_b_Na = g_bna * (V - alg_12_E_Na)
        alg_65_i_NaCa = K_NaCa * (torch.exp(gamma * V * F / (R * T)) * Na_i ** 3.0 * Ca_o - torch.exp((gamma - 1.0) * V * F / (R * T)) * Na_o ** 3.0 * Ca_i * alpha) / ((Km_Nai ** 3.0 + Na_o ** 3.0) * (Km_Ca + Ca_o) * (1.0 + K_sat * torch.exp((gamma - 1.0) * V * F / (R * T))))
        d_Na_i = torch.zeros_like(y[:, 0]) + (-(alg_58_i_Na + alg_59_i_b_Na + 3.0 * alg_64_i_NaK + 3.0 * alg_65_i_NaCa) * Cm / (Vol_c * F))
        alg_60_i_CaL = g_CaL * d * f * f2 * fCass * z ** 2.0 * (V - 15.0) * F ** 2.0 / (R * T) * (0.25 * Ca_ss * torch.exp(2.0 * (V - 15.0) * F / (R * T)) - Ca_o) / (torch.exp(2.0 * (V - 15.0) * F / (R * T)) - 1.0)
        alg_38_E_Ca = 0.5 * R * T / F * torch.log(Ca_o / Ca_i)
        alg_61_i_b_Ca = g_bca * (V - alg_38_E_Ca)
        alg_62_i_p_Ca = g_pCa * Ca_i / (Ca_i + K_pCa)
        alg_66_i_tot = alg_54_i_K1 + alg_57_i_to + alg_55_i_Kr + alg_56_i_Ks + alg_60_i_CaL + alg_64_i_NaK + alg_58_i_Na + alg_59_i_b_Na + alg_65_i_NaCa + alg_61_i_b_Ca + alg_63_i_p_K + alg_62_i_p_Ca + alg_0_i_Stim
        d_V = torch.zeros_like(y[:, 0]) + (-alg_66_i_tot)
        alg_70_Ca_i_bufc = 1.0 / (1.0 + Buf_c * K_buf_c / (Ca_i + K_buf_c) ** 2.0)
        alg_67_i_up = Vmax_up / (1.0 + K_up ** 2.0 / Ca_i ** 2.0)
        alg_68_i_leak = Vol_leak * (Ca_SR - Ca_i)
        alg_69_i_xfer = Vol_xfer * (Ca_ss - Ca_i)
        d_Ca_i = torch.zeros_like(y[:, 0]) + (alg_70_Ca_i_bufc * ((alg_68_i_leak - alg_67_i_up) * V_sr / Vol_c + alg_69_i_xfer - (alg_61_i_b_Ca + alg_62_i_p_Ca - 2.0 * alg_65_i_NaCa) * Cm / (2.0 * Vol_c * F)))
        alg_71_kcasr = max_sr - (max_sr - min_sr) / (1.0 + (EC / Ca_SR) ** 2.0)
        alg_73_k2 = k2_prime * alg_71_kcasr
        d_R_prime = torch.zeros_like(y[:, 0]) + (-alg_73_k2 * Ca_ss * R_prime + k4 * (1.0 - R_prime))
        alg_76_Ca_sr_bufsr = 1.0 / (1.0 + Buf_sr * K_buf_sr / (Ca_SR + K_buf_sr) ** 2.0)
        alg_72_k1 = k1_prime / alg_71_kcasr
        alg_74_O = alg_72_k1 * Ca_ss ** 2.0 * R_prime / (k3 + alg_72_k1 * Ca_ss ** 2.0)
        alg_75_i_rel = Vol_rel * alg_74_O * (Ca_SR - Ca_ss)
        d_Ca_SR = torch.zeros_like(y[:, 0]) + (alg_76_Ca_sr_bufsr * (alg_67_i_up - (alg_75_i_rel + alg_68_i_leak)))
        alg_77_Ca_ss_bufss = 1.0 / (1.0 + Buf_ss * K_buf_ss / (Ca_ss + K_buf_ss) ** 2.0)
        d_Ca_ss = torch.zeros_like(y[:, 0]) + (alg_77_Ca_ss_bufss * (-alg_60_i_CaL * Cm / (2.0 * V_ss * F) + alg_75_i_rel * V_sr / V_ss - alg_69_i_xfer * Vol_c / V_ss))

        return torch.stack(
            [d_V, d_K_i, d_Na_i, d_Ca_i, d_Or4, d_Cr1, d_Cr2, d_Cr3, d_Ir5, d_BCr1, d_BCr2, d_BCr3, d_BOr4, d_BIr5, d_Xs, d_s, d_r, d_m, d_h, d_j, d_Ca_ss, d_d, d_f, d_f2, d_fCass, d_Ca_SR, d_R_prime],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
