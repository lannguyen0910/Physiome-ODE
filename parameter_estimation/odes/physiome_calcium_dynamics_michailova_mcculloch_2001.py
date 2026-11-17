import torch
import torch.nn as nn
import math


class CalciumDynamicsMichailovaMcculloch2001ODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/michailova_mcculloch_2001/model.py.
    Dynamic states: 43; independent constants: 95; derived constants used by rates: 2.
    Original constant slots: 97; algebraic slots: 72.
    """
    source_file = 'calcium_dynamics/michailova_mcculloch_2001/model.py'
    source_state_dim = 43
    source_constant_dim = 97
    algebraic_dim = 72
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 43
        self.parameter_dim = 95

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8.314472, 'T': 310, 'F': 96.4853415, 'C_sc': 0.001, 'stim_start': 0.1, 'stim_end': 100000000, 'stim_period': 1, 'stim_duration': 0.0005, 'stim_amplitude': -100, 'g_Na': 12.8, 'Na_o': 138, 'g_Kr': 0.0034, 'K_o': 4, 'tau_factor': 1, 'g_Ks': 0.0027134, 'g_to1': 0.23815, 'g_K1': 2.8, 'K_mK1': 13, 'g_Kp': 0.002216, 'K_mCa': 1.38, 'K_mNa': 87.5, 'K_NaCa': 0.3, 'K_sat': 0.2, 'eta': 0.35, 'Ca_o': 2, 'I_NaK': 0.693, 'K_mNa_i': 10, 'K_mK_o': 1.5, 'MgATP_i0': 2.888, 'K_mpCa': 5e-05, 'I_pCa': 0.05, 'g_Cab': 0.0003842, 'g_Nab': 0.0031, 'P_Ca': 0.0003125, 'P_K': 5.79e-07, 'i_Ca_half': -0.265, 'a': 2, 'b': 2, 'g': 2000, 'f': 300, 'gprime': 7000, 'fprime': 7, 'omega': 10, 'v1': 1800, 'k_a_plus': 12150000000000, 'k_a_minus': 576, 'k_b_plus': 4050000000, 'k_b_minus': 1930, 'k_c_plus': 100, 'k_c_minus': 0.8, 'n': 4, 'm': 3, 'K_fb': 0.000168, 'K_rb': 3.29, 'Vmaxf': 0.0813, 'Vmaxr': 0.318, 'K_SR': 1, 'N_fb': 1.2, 'N_rb': 1, 'tau_tr': 0.0005747, 'tau_xfer': 0.0267, 'HTRPN_tot': 0.14, 'LTRPN_tot': 0.07, 'k_htrpn_plus': 20000, 'k_htrpn_minus': 0.066, 'k_ltrpn_plus': 40000, 'k_ltrpn_minus': 40, 'A_cap': 0.0001534, 'V_myo': 2.584e-05, 'V_JSR': 1.6e-07, 'V_NSR': 2.1e-06, 'V_ss': 1.2e-09, 'K_mCMDN': 0.00238, 'K_mEGTA': 0.00015, 'K_mCSQN': 0.8, 'CMDN_tot': 0.05, 'EGTA_tot': 0, 'CSQN_tot': 15, 'k_plus_CaATP': 225000, 'k_minus_CaATP': 45000, 'k_plus_CaADP': 125000, 'k_minus_CaADP': 193500, 'ATP_tot': 7, 'k_plus_MgATP': 125000, 'k_minus_MgATP': 10875, 'tau_xfer_CaATP': 0.0534, 'tau_xfer_MgATP': 0.0534, 'tau_xfer_Mg': 0.0267, 'ADP_tot': 0.005, 'k_plus_MgADP': 125000, 'k_minus_MgADP': 84500, 'tau_xfer_CaADP': 0.0534, 'tau_xfer_MgADP': 0.0534, 'V_myo_2': 2.584e-05, 'V_ss_2': 1.2e-09}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -96.1638, 'Na_i': 10, 'm': 0.0328302, 'h': 0.988354, 'j': 0.99254, 'K_i': 159.48, 'X_kr': 0.51, 'X_ks': 0.264, 'X_to1': 2.63, 'Y_to1': 0.99, 'Ca_i': 8.464e-05, 'MgATP_i': 6.4395, 'O': 9.84546e-21, 'O_Ca': 0, 'C0': 0.997208, 'C1': 6.38897e-05, 'C2': 1.535e-09, 'C3': 1.63909e-14, 'C4': 6.56337e-20, 'C_Ca0': 0.00272826, 'C_Ca1': 6.99215e-07, 'C_Ca2': 6.71989e-11, 'C_Ca3': 2.87031e-15, 'C_Ca4': 4.59752e-20, 'Ca_ss': 0.0001315, 'y': 0.798, 'P_O1': 0, 'P_O2': 0, 'P_C1': 0.47, 'P_C2': 0.53, 'Ca_JSR': 0.2616, 'Ca_NSR': 0.262, 'HTRPNCa': 0.98, 'LTRPNCa': 0.078, 'CaADP_i': 1.1e-07, 'CaADP_ss': 1.3e-07, 'CaATP_ss': 0.00025, 'CaATP_i': 0.000237, 'Mg_ss': 1, 'Mg_i': 1, 'MgADP_i': 0.00298, 'MgADP_ss': 0.00298, 'MgATP_ss': 6.4395}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'C_sc', 'stim_start', 'stim_end', 'stim_period', 'stim_duration', 'stim_amplitude', 'g_Na', 'Na_o', 'g_Kr', 'K_o', 'tau_factor', 'g_Ks', 'g_to1', 'g_K1', 'K_mK1', 'g_Kp', 'K_mCa', 'K_mNa', 'K_NaCa', 'K_sat', 'eta', 'Ca_o', 'I_NaK', 'K_mNa_i', 'K_mK_o', 'MgATP_i0', 'K_mpCa', 'I_pCa', 'g_Cab', 'g_Nab', 'P_Ca', 'P_K', 'i_Ca_half', 'a', 'b', 'g', 'f', 'gprime', 'fprime', 'omega', 'v1', 'k_a_plus', 'k_a_minus', 'k_b_plus', 'k_b_minus', 'k_c_plus', 'k_c_minus', 'n', 'm', 'K_fb', 'K_rb', 'Vmaxf', 'Vmaxr', 'K_SR', 'N_fb', 'N_rb', 'tau_tr', 'tau_xfer', 'HTRPN_tot', 'LTRPN_tot', 'k_htrpn_plus', 'k_htrpn_minus', 'k_ltrpn_plus', 'k_ltrpn_minus', 'A_cap', 'V_myo', 'V_JSR', 'V_NSR', 'V_ss', 'K_mCMDN', 'K_mEGTA', 'K_mCSQN', 'CMDN_tot', 'EGTA_tot', 'CSQN_tot', 'k_plus_CaATP', 'k_minus_CaATP', 'k_plus_CaADP', 'k_minus_CaADP', 'ATP_tot', 'k_plus_MgATP', 'k_minus_MgATP', 'tau_xfer_CaATP', 'tau_xfer_MgATP', 'tau_xfer_Mg', 'ADP_tot', 'k_plus_MgADP', 'k_minus_MgADP', 'tau_xfer_CaADP', 'tau_xfer_MgADP', 'V_myo_2', 'V_ss_2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'Na_i', 'm', 'h', 'j', 'K_i', 'X_kr', 'X_ks', 'X_to1', 'Y_to1', 'Ca_i', 'MgATP_i', 'O', 'O_Ca', 'C0', 'C1', 'C2', 'C3', 'C4', 'C_Ca0', 'C_Ca1', 'C_Ca2', 'C_Ca3', 'C_Ca4', 'Ca_ss', 'y', 'P_O1', 'P_O2', 'P_C1', 'P_C2', 'Ca_JSR', 'Ca_NSR', 'HTRPNCa', 'LTRPNCa', 'CaADP_i', 'CaADP_ss', 'CaATP_ss', 'CaATP_i', 'Mg_ss', 'Mg_i', 'MgADP_i', 'MgADP_ss', 'MgATP_ss']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (43)
        V = y[:, 0]
        Na_i = y[:, 1]
        m = y[:, 2]
        h = y[:, 3]
        j = y[:, 4]
        K_i = y[:, 5]
        X_kr = y[:, 6]
        X_ks = y[:, 7]
        X_to1 = y[:, 8]
        Y_to1 = y[:, 9]
        Ca_i = y[:, 10]
        MgATP_i = y[:, 11]
        O = y[:, 12]
        O_Ca = y[:, 13]
        C0 = y[:, 14]
        C1 = y[:, 15]
        C2 = y[:, 16]
        C3 = y[:, 17]
        C4 = y[:, 18]
        C_Ca0 = y[:, 19]
        C_Ca1 = y[:, 20]
        C_Ca2 = y[:, 21]
        C_Ca3 = y[:, 22]
        C_Ca4 = y[:, 23]
        Ca_ss = y[:, 24]
        state_y = y[:, 25]
        P_O1 = y[:, 26]
        P_O2 = y[:, 27]
        P_C1 = y[:, 28]
        P_C2 = y[:, 29]
        Ca_JSR = y[:, 30]
        Ca_NSR = y[:, 31]
        HTRPNCa = y[:, 32]
        LTRPNCa = y[:, 33]
        CaADP_i = y[:, 34]
        CaADP_ss = y[:, 35]
        CaATP_ss = y[:, 36]
        CaATP_i = y[:, 37]
        Mg_ss = y[:, 38]
        Mg_i = y[:, 39]
        MgADP_i = y[:, 40]
        MgADP_ss = y[:, 41]
        MgATP_ss = y[:, 42]

        # Independent parameters (source constant order)
        R = theta[:, 0]  # constants[0]
        T = theta[:, 1]  # constants[1]
        F = theta[:, 2]  # constants[2]
        C_sc = theta[:, 3]  # constants[3]
        stim_start = theta[:, 4]  # constants[4]
        stim_end = theta[:, 5]  # constants[5]
        stim_period = theta[:, 6]  # constants[6]
        stim_duration = theta[:, 7]  # constants[7]
        stim_amplitude = theta[:, 8]  # constants[8]
        g_Na = theta[:, 9]  # constants[9]
        Na_o = theta[:, 10]  # constants[10]
        g_Kr = theta[:, 11]  # constants[11]
        K_o = theta[:, 12]  # constants[12]
        tau_factor = theta[:, 13]  # constants[13]
        g_Ks = theta[:, 14]  # constants[14]
        g_to1 = theta[:, 15]  # constants[15]
        g_K1 = theta[:, 16]  # constants[16]
        K_mK1 = theta[:, 17]  # constants[17]
        g_Kp = theta[:, 18]  # constants[18]
        K_mCa = theta[:, 19]  # constants[19]
        K_mNa = theta[:, 20]  # constants[20]
        K_NaCa = theta[:, 21]  # constants[21]
        K_sat = theta[:, 22]  # constants[22]
        eta = theta[:, 23]  # constants[23]
        Ca_o = theta[:, 24]  # constants[24]
        I_NaK = theta[:, 25]  # constants[25]
        K_mNa_i = theta[:, 26]  # constants[26]
        K_mK_o = theta[:, 27]  # constants[27]
        MgATP_i0 = theta[:, 28]  # constants[28]
        K_mpCa = theta[:, 29]  # constants[29]
        I_pCa = theta[:, 30]  # constants[30]
        g_Cab = theta[:, 31]  # constants[31]
        g_Nab = theta[:, 32]  # constants[32]
        P_Ca = theta[:, 33]  # constants[33]
        P_K = theta[:, 34]  # constants[34]
        i_Ca_half = theta[:, 35]  # constants[35]
        a = theta[:, 36]  # constants[36]
        b = theta[:, 37]  # constants[37]
        g = theta[:, 38]  # constants[38]
        f = theta[:, 39]  # constants[39]
        gprime = theta[:, 40]  # constants[40]
        fprime = theta[:, 41]  # constants[41]
        omega = theta[:, 42]  # constants[42]
        v1 = theta[:, 43]  # constants[43]
        k_a_plus = theta[:, 44]  # constants[44]
        k_a_minus = theta[:, 45]  # constants[45]
        k_b_plus = theta[:, 46]  # constants[46]
        k_b_minus = theta[:, 47]  # constants[47]
        k_c_plus = theta[:, 48]  # constants[48]
        k_c_minus = theta[:, 49]  # constants[49]
        n = theta[:, 50]  # constants[50]
        const_m = theta[:, 51]  # constants[51]
        K_fb = theta[:, 52]  # constants[52]
        K_rb = theta[:, 53]  # constants[53]
        Vmaxf = theta[:, 54]  # constants[54]
        Vmaxr = theta[:, 55]  # constants[55]
        K_SR = theta[:, 56]  # constants[56]
        N_fb = theta[:, 57]  # constants[57]
        N_rb = theta[:, 58]  # constants[58]
        tau_tr = theta[:, 59]  # constants[59]
        tau_xfer = theta[:, 60]  # constants[60]
        HTRPN_tot = theta[:, 61]  # constants[61]
        LTRPN_tot = theta[:, 62]  # constants[62]
        k_htrpn_plus = theta[:, 63]  # constants[63]
        k_htrpn_minus = theta[:, 64]  # constants[64]
        k_ltrpn_plus = theta[:, 65]  # constants[65]
        k_ltrpn_minus = theta[:, 66]  # constants[66]
        A_cap = theta[:, 67]  # constants[67]
        V_myo = theta[:, 68]  # constants[68]
        V_JSR = theta[:, 69]  # constants[69]
        V_NSR = theta[:, 70]  # constants[70]
        V_ss = theta[:, 71]  # constants[71]
        K_mCMDN = theta[:, 72]  # constants[72]
        K_mEGTA = theta[:, 73]  # constants[73]
        K_mCSQN = theta[:, 74]  # constants[74]
        CMDN_tot = theta[:, 75]  # constants[75]
        EGTA_tot = theta[:, 76]  # constants[76]
        CSQN_tot = theta[:, 77]  # constants[77]
        k_plus_CaATP = theta[:, 78]  # constants[78]
        k_minus_CaATP = theta[:, 79]  # constants[79]
        k_plus_CaADP = theta[:, 80]  # constants[80]
        k_minus_CaADP = theta[:, 81]  # constants[81]
        ATP_tot = theta[:, 82]  # constants[82]
        k_plus_MgATP = theta[:, 83]  # constants[83]
        k_minus_MgATP = theta[:, 84]  # constants[84]
        tau_xfer_CaATP = theta[:, 85]  # constants[85]
        tau_xfer_MgATP = theta[:, 86]  # constants[86]
        tau_xfer_Mg = theta[:, 87]  # constants[87]
        ADP_tot = theta[:, 88]  # constants[88]
        k_plus_MgADP = theta[:, 89]  # constants[89]
        k_minus_MgADP = theta[:, 90]  # constants[90]
        tau_xfer_CaADP = theta[:, 91]  # constants[91]
        tau_xfer_MgADP = theta[:, 92]  # constants[92]
        V_myo_2 = theta[:, 93]  # constants[93]
        V_ss_2 = theta[:, 94]  # constants[94]

        # Derived constants from the original initConsts()
        f_K_o = (K_o / 4.0) ** (1.0 / 2)  # constants[95]
        sigma = 1.0 / 7.0 * (torch.exp(Na_o / 67.3) - 1.0)  # constants[96]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_j = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_X_kr = torch.zeros_like(y[:, 0])
        d_X_ks = torch.zeros_like(y[:, 0])
        d_X_to1 = torch.zeros_like(y[:, 0])
        d_Y_to1 = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_MgATP_i = torch.zeros_like(y[:, 0])
        d_O = torch.zeros_like(y[:, 0])
        d_O_Ca = torch.zeros_like(y[:, 0])
        d_C0 = torch.zeros_like(y[:, 0])
        d_C1 = torch.zeros_like(y[:, 0])
        d_C2 = torch.zeros_like(y[:, 0])
        d_C3 = torch.zeros_like(y[:, 0])
        d_C4 = torch.zeros_like(y[:, 0])
        d_C_Ca0 = torch.zeros_like(y[:, 0])
        d_C_Ca1 = torch.zeros_like(y[:, 0])
        d_C_Ca2 = torch.zeros_like(y[:, 0])
        d_C_Ca3 = torch.zeros_like(y[:, 0])
        d_C_Ca4 = torch.zeros_like(y[:, 0])
        d_Ca_ss = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])
        d_P_O1 = torch.zeros_like(y[:, 0])
        d_P_O2 = torch.zeros_like(y[:, 0])
        d_P_C1 = torch.zeros_like(y[:, 0])
        d_P_C2 = torch.zeros_like(y[:, 0])
        d_Ca_JSR = torch.zeros_like(y[:, 0])
        d_Ca_NSR = torch.zeros_like(y[:, 0])
        d_HTRPNCa = torch.zeros_like(y[:, 0])
        d_LTRPNCa = torch.zeros_like(y[:, 0])
        d_CaADP_i = torch.zeros_like(y[:, 0])
        d_CaADP_ss = torch.zeros_like(y[:, 0])
        d_CaATP_ss = torch.zeros_like(y[:, 0])
        d_CaATP_i = torch.zeros_like(y[:, 0])
        d_Mg_ss = torch.zeros_like(y[:, 0])
        d_Mg_i = torch.zeros_like(y[:, 0])
        d_MgADP_i = torch.zeros_like(y[:, 0])
        d_MgADP_ss = torch.zeros_like(y[:, 0])
        d_MgATP_ss = torch.zeros_like(y[:, 0])
        alg_0_E0_m = torch.zeros_like(y[:, 0])
        alg_1_alpha_h = torch.zeros_like(y[:, 0])
        alg_2_alpha_j = torch.zeros_like(y[:, 0])
        alg_3_K12 = torch.zeros_like(y[:, 0])
        alg_4_X_ks_infinity = torch.zeros_like(y[:, 0])
        alg_5_alpha_X_to1 = torch.zeros_like(y[:, 0])
        alg_6_alpha_Y_to1 = torch.zeros_like(y[:, 0])
        alg_7_alpha = torch.zeros_like(y[:, 0])
        alg_8_y_infinity = torch.zeros_like(y[:, 0])
        alg_9_i_Stim = torch.zeros_like(y[:, 0])
        alg_10_alpha_m = torch.zeros_like(y[:, 0])
        alg_11_beta_h = torch.zeros_like(y[:, 0])
        alg_12_beta_j = torch.zeros_like(y[:, 0])
        alg_13_K21 = torch.zeros_like(y[:, 0])
        alg_14_tau_X_ks = torch.zeros_like(y[:, 0])
        alg_15_beta_X_to1 = torch.zeros_like(y[:, 0])
        alg_16_beta_Y_to1 = torch.zeros_like(y[:, 0])
        alg_17_beta = torch.zeros_like(y[:, 0])
        alg_18_tau_y = torch.zeros_like(y[:, 0])
        alg_19_E_Na = torch.zeros_like(y[:, 0])
        alg_20_beta_m = torch.zeros_like(y[:, 0])
        alg_21_X_kr_inf = torch.zeros_like(y[:, 0])
        alg_22_alpha_a = torch.zeros_like(y[:, 0])
        alg_23_i_Na = torch.zeros_like(y[:, 0])
        alg_24_tau_X_kr = torch.zeros_like(y[:, 0])
        alg_25_beta_b = torch.zeros_like(y[:, 0])
        alg_26_E_K = torch.zeros_like(y[:, 0])
        alg_27_gamma = torch.zeros_like(y[:, 0])
        alg_28_R_V = torch.zeros_like(y[:, 0])
        alg_29_i_Kr = torch.zeros_like(y[:, 0])
        alg_30_E_Ks = torch.zeros_like(y[:, 0])
        alg_31_i_Ks = torch.zeros_like(y[:, 0])
        alg_32_i_to1 = torch.zeros_like(y[:, 0])
        alg_33_K1_infinity_V = torch.zeros_like(y[:, 0])
        alg_34_i_K1 = torch.zeros_like(y[:, 0])
        alg_35_Kp_V = torch.zeros_like(y[:, 0])
        alg_36_i_Kp = torch.zeros_like(y[:, 0])
        alg_37_i_NaCa = torch.zeros_like(y[:, 0])
        alg_38_f_NaK = torch.zeros_like(y[:, 0])
        alg_39_i_NaK_winslow = torch.zeros_like(y[:, 0])
        alg_40_i_NaK = torch.zeros_like(y[:, 0])
        alg_41_i_p_Ca_winslow = torch.zeros_like(y[:, 0])
        alg_42_i_p_Ca = torch.zeros_like(y[:, 0])
        alg_43_E_Ca = torch.zeros_like(y[:, 0])
        alg_44_i_Ca_b = torch.zeros_like(y[:, 0])
        alg_45_i_Na_b = torch.zeros_like(y[:, 0])
        alg_46_i_Ca_max = torch.zeros_like(y[:, 0])
        alg_47_i_Ca = torch.zeros_like(y[:, 0])
        alg_48_p_prime_k = torch.zeros_like(y[:, 0])
        alg_49_J_rel = torch.zeros_like(y[:, 0])
        alg_50_i_Ca_K = torch.zeros_like(y[:, 0])
        alg_51_fb = torch.zeros_like(y[:, 0])
        alg_52_rb = torch.zeros_like(y[:, 0])
        alg_53_J_up_winslow = torch.zeros_like(y[:, 0])
        alg_54_J_up = torch.zeros_like(y[:, 0])
        alg_55_J_xfer = torch.zeros_like(y[:, 0])
        alg_56_J_tr = torch.zeros_like(y[:, 0])
        alg_57_beta_JSR = torch.zeros_like(y[:, 0])
        alg_58_J_HTRPNCa = torch.zeros_like(y[:, 0])
        alg_59_J_LTRPNCa = torch.zeros_like(y[:, 0])
        alg_60_J_trpn = torch.zeros_like(y[:, 0])
        alg_61_beta_i = torch.zeros_like(y[:, 0])
        alg_62_beta_SS = torch.zeros_like(y[:, 0])
        alg_63_ATP_ss = torch.zeros_like(y[:, 0])
        alg_64_ATP_i = torch.zeros_like(y[:, 0])
        alg_65_ADP_ss = torch.zeros_like(y[:, 0])
        alg_66_Jxfer_CaATP = torch.zeros_like(y[:, 0])
        alg_67_Jxfer_MgATP = torch.zeros_like(y[:, 0])
        alg_68_ADP_i = torch.zeros_like(y[:, 0])
        alg_69_Jxfer_Mg = torch.zeros_like(y[:, 0])
        alg_70_Jxfer_CaADP = torch.zeros_like(y[:, 0])
        alg_71_Jxfer_MgADP = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_O = torch.zeros_like(y[:, 0]) + (f * C4 - g * O)
        d_O_Ca = torch.zeros_like(y[:, 0]) + (fprime * C_Ca4 - gprime * O_Ca)
        d_P_C1 = torch.zeros_like(y[:, 0]) + (-k_a_plus * Ca_ss ** n * P_C1 + k_a_minus * P_O1)
        d_P_O1 = torch.zeros_like(y[:, 0]) + (k_a_plus * Ca_ss ** n * P_C1 - (k_a_minus * P_O1 + k_b_plus * Ca_ss ** const_m * P_O1 + k_c_plus * P_O1) + k_b_minus * P_O2 + k_c_minus * P_C2)
        d_P_O2 = torch.zeros_like(y[:, 0]) + (k_b_plus * Ca_ss ** const_m * P_O1 - k_b_minus * P_O2)
        d_P_C2 = torch.zeros_like(y[:, 0]) + (k_c_plus * P_O1 - k_c_minus * P_C2)
        alg_1_alpha_h = torch.where(V < -40.0, 135.0 * torch.exp((80.0 + V) / -6.8), 0.0)
        alg_11_beta_h = torch.where(V < -40.0, 3560.0 * torch.exp(0.079 * V) + 310000.0 * torch.exp(0.35 * V), 1000.0 / (0.13 * (1.0 + torch.exp((V + 10.66) / -11.1))))
        d_h = torch.zeros_like(y[:, 0]) + (alg_1_alpha_h * (1.0 - h) - alg_11_beta_h * h)
        alg_2_alpha_j = torch.where(V < -40.0, 1000.0 * -(127140.0 * torch.exp(0.2444 * V) + 3.474e-05 * torch.exp(-0.04391 * V)) * (V + 37.78) / (1.0 + torch.exp(0.311 * (V + 79.23))), 0.0)
        alg_12_beta_j = torch.where(V < -40.0, 121.2 * torch.exp(-0.01052 * V) / (1.0 + torch.exp(-0.1378 * (V + 40.14))), 300.0 * torch.exp(-2.535e-07 * V) / (1.0 + torch.exp(-0.1 * (V + 32.0))))
        d_j = torch.zeros_like(y[:, 0]) + (alg_2_alpha_j * (1.0 - j) - alg_12_beta_j * j)
        alg_14_tau_X_ks = 0.001 / (7.19e-05 * (V - 10.0) / (1.0 - torch.exp(-0.148 * (V - 10.0))) + 0.000131 * (V - 10.0) / (torch.exp(0.0687 * (V - 10.0)) - 1.0))
        alg_4_X_ks_infinity = 1.0 / (1.0 + torch.exp(-(V - 24.7) / 13.6))
        d_X_ks = torch.zeros_like(y[:, 0]) + ((alg_4_X_ks_infinity - X_ks) / alg_14_tau_X_ks)
        alg_5_alpha_X_to1 = 45.16 * torch.exp(0.03577 * V)
        alg_15_beta_X_to1 = 98.9 * torch.exp(-0.06237 * V)
        d_X_to1 = torch.zeros_like(y[:, 0]) + (alg_5_alpha_X_to1 * (1.0 - X_to1) - alg_15_beta_X_to1 * X_to1)
        alg_6_alpha_Y_to1 = 5.415 * torch.exp(-(V + 33.5) / 5.0) / (1.0 + 0.051335 * torch.exp(-(V + 33.5) / 5.0))
        alg_16_beta_Y_to1 = 5.415 * torch.exp((V + 33.5) / 5.0) / (1.0 + 0.051335 * torch.exp((V + 33.5) / 5.0))
        d_Y_to1 = torch.zeros_like(y[:, 0]) + (alg_6_alpha_Y_to1 * (1.0 - Y_to1) - alg_16_beta_Y_to1 * Y_to1)
        alg_8_y_infinity = 0.8 / (1.0 + torch.exp((V + 12.5) / 5.0)) + 0.2
        alg_18_tau_y = (20.0 + 600.0 / (1.0 + torch.exp((V + 20.0) / 9.5))) / 1000.0
        d_state_y = torch.zeros_like(y[:, 0]) + ((alg_8_y_infinity - state_y) / alg_18_tau_y)
        alg_0_E0_m = V + 47.13
        alg_10_alpha_m = torch.where(torch.abs(alg_0_E0_m) < 1e-05, 320.0 / (0.1 - 0.005 * alg_0_E0_m), 320.0 * alg_0_E0_m / (1.0 - torch.exp(-0.1 * alg_0_E0_m)))
        alg_20_beta_m = 80.0 * torch.exp(-V / 11.0)
        d_m = torch.zeros_like(y[:, 0]) + (torch.where(V >= -90.0, alg_10_alpha_m * (1.0 - m) - alg_20_beta_m * m, 0.0))
        alg_3_K12 = torch.exp(-5.495 + 0.1691 * V)
        alg_13_K21 = torch.exp(-7.677 - 0.0128 * V)
        alg_21_X_kr_inf = alg_3_K12 / (alg_3_K12 + alg_13_K21)
        alg_24_tau_X_kr = 0.001 / (alg_3_K12 + alg_13_K21) + tau_factor * 0.027
        d_X_kr = torch.zeros_like(y[:, 0]) + ((alg_21_X_kr_inf - X_kr) / alg_24_tau_X_kr)
        alg_7_alpha = 400.0 * torch.exp((V + 2.0) / 10.0)
        alg_17_beta = 50.0 * torch.exp(-(V + 2.0) / 13.0)
        alg_27_gamma = 103.75 * Ca_ss / 1.0
        d_C0 = torch.zeros_like(y[:, 0]) + (alg_17_beta * C1 + omega * C_Ca0 - (4.0 * alg_7_alpha + alg_27_gamma) * C0)
        d_C1 = torch.zeros_like(y[:, 0]) + (4.0 * alg_7_alpha * C0 + 2.0 * alg_17_beta * C2 + omega / b * C_Ca1 - (alg_17_beta + 3.0 * alg_7_alpha + alg_27_gamma * a) * C1)
        d_C2 = torch.zeros_like(y[:, 0]) + (3.0 * alg_7_alpha * C1 + 3.0 * alg_17_beta * C3 + omega / b ** 2.0 * C_Ca2 - (alg_17_beta * 2.0 + 2.0 * alg_7_alpha + alg_27_gamma * a ** 2.0) * C2)
        d_C3 = torch.zeros_like(y[:, 0]) + (2.0 * alg_7_alpha * C2 + 4.0 * alg_17_beta * C4 + omega / b ** 3.0 * C_Ca3 - (alg_17_beta * 3.0 + alg_7_alpha + alg_27_gamma * a ** 3.0) * C3)
        d_C4 = torch.zeros_like(y[:, 0]) + (alg_7_alpha * C3 + g * O + omega / b ** 4.0 * C_Ca4 - (alg_17_beta * 4.0 + f + alg_27_gamma * a ** 4.0) * C4)
        alg_22_alpha_a = alg_7_alpha * a
        alg_25_beta_b = alg_17_beta / b
        d_C_Ca0 = torch.zeros_like(y[:, 0]) + (alg_25_beta_b * C_Ca1 + alg_27_gamma * C0 - (4.0 * alg_22_alpha_a + omega) * C_Ca0)
        d_C_Ca1 = torch.zeros_like(y[:, 0]) + (4.0 * alg_22_alpha_a * C_Ca0 + 2.0 * alg_25_beta_b * C_Ca2 + alg_27_gamma * a * C1 - (alg_25_beta_b + 3.0 * alg_22_alpha_a + omega / b) * C_Ca1)
        d_C_Ca2 = torch.zeros_like(y[:, 0]) + (3.0 * alg_22_alpha_a * C_Ca1 + 3.0 * alg_25_beta_b * C_Ca3 + alg_27_gamma * a ** 2.0 * C2 - (alg_25_beta_b * 2.0 + 2.0 * alg_22_alpha_a + omega / b ** 2.0) * C_Ca2)
        d_C_Ca3 = torch.zeros_like(y[:, 0]) + (2.0 * alg_22_alpha_a * C_Ca2 + 4.0 * alg_25_beta_b * C_Ca4 + alg_27_gamma * a ** 3.0 * C3 - (alg_25_beta_b * 3.0 + alg_22_alpha_a + omega / b ** 3.0) * C_Ca3)
        d_C_Ca4 = torch.zeros_like(y[:, 0]) + (alg_22_alpha_a * C_Ca3 + gprime * O_Ca + alg_27_gamma * a ** 4.0 * C4 - (alg_25_beta_b * 4.0 + fprime + omega / b ** 4.0) * C_Ca4)
        alg_19_E_Na = R * T / F * torch.log(Na_o / Na_i)
        alg_23_i_Na = g_Na * m ** 3.0 * h * j * (V - alg_19_E_Na)
        alg_37_i_NaCa = K_NaCa * 5000.0 / ((K_mNa ** 3.0 + Na_o ** 3.0) * (K_mCa + Ca_o) * (1.0 + K_sat * torch.exp((eta - 1.0) * V * F / (R * T)))) * (torch.exp(eta * V * F / (R * T)) * Na_i ** 3.0 * Ca_o - torch.exp((eta - 1.0) * V * F / (R * T)) * Na_o ** 3.0 * Ca_i)
        alg_38_f_NaK = 1.0 / (1.0 + 0.1245 * torch.exp(-0.1 * V * F / (R * T)) + 0.0365 * sigma * torch.exp(-V * F / (R * T)))
        alg_39_i_NaK_winslow = I_NaK * alg_38_f_NaK / (1.0 + (K_mNa_i / Na_i) ** 1.5) * K_o / (K_o + K_mK_o)
        alg_40_i_NaK = MgATP_i / MgATP_i0 * alg_39_i_NaK_winslow
        alg_45_i_Na_b = g_Nab * (V - alg_19_E_Na)
        d_Na_i = torch.zeros_like(y[:, 0]) + (-0.0 * (alg_23_i_Na + alg_45_i_Na_b + alg_37_i_NaCa * 3.0 + alg_40_i_NaK * 3.0) * A_cap * 1.0 / (V_myo * F))
        alg_9_i_Stim = torch.where((t >= stim_start) & (t <= stim_end) & (t - stim_start - torch.floor((t - stim_start) / stim_period) * stim_period <= stim_duration), stim_amplitude, 0.0)
        alg_46_i_Ca_max = P_Ca / (1.0 * 1.0) * 4.0 * V * F ** 2.0 * 1000.0 / (R * T) * (0.001 * torch.exp(2.0 * V * F / (R * T)) - 0.341 * Ca_o) / (torch.exp(2.0 * V * F / (R * T)) - 1.0)
        alg_47_i_Ca = alg_46_i_Ca_max * state_y * (O + O_Ca)
        alg_48_p_prime_k = P_K / (1.0 + alg_46_i_Ca_max / i_Ca_half)
        alg_50_i_Ca_K = alg_48_p_prime_k / (1.0 * 1.0) * state_y * (O + O_Ca) * V * F ** 2.0 / (R * T) * (K_i * torch.exp(V * F / (R * T)) - K_o) / (torch.exp(V * F / (R * T)) - 1.0)
        alg_26_E_K = R * T / F * torch.log(K_o / K_i)
        alg_28_R_V = 1.0 / (1.0 + 1.4945 * torch.exp(0.0446 * V))
        alg_29_i_Kr = g_Kr * f_K_o * alg_28_R_V * X_kr * (V - alg_26_E_K)
        alg_30_E_Ks = R * T / F * torch.log((K_o + 0.01833 * Na_o) / (K_i + 0.01833 * Na_i))
        alg_31_i_Ks = g_Ks * X_ks ** 2.0 * (V - alg_30_E_Ks)
        alg_32_i_to1 = g_to1 * X_to1 * Y_to1 * (V - alg_26_E_K)
        alg_33_K1_infinity_V = 1.0 / (2.0 + torch.exp(1.5 * F / (R * T) * (V - alg_26_E_K)))
        alg_34_i_K1 = g_K1 * alg_33_K1_infinity_V * K_o / (K_o + K_mK1) * (V - alg_26_E_K)
        alg_35_Kp_V = 1.0 / (1.0 + torch.exp((7.488 - V) / 5.98))
        alg_36_i_Kp = g_Kp * alg_35_Kp_V * (V - alg_26_E_K)
        alg_41_i_p_Ca_winslow = I_pCa * Ca_i / (K_mpCa + Ca_i)
        alg_42_i_p_Ca = MgATP_i / MgATP_i0 * alg_41_i_p_Ca_winslow
        alg_43_E_Ca = R * T / (2.0 * F) * torch.log(Ca_o / Ca_i)
        alg_44_i_Ca_b = g_Cab * (V - alg_43_E_Ca)
        d_V = torch.zeros_like(y[:, 0]) + (-1.0 * 1.0 * (alg_23_i_Na + alg_47_i_Ca + alg_50_i_Ca_K + alg_29_i_Kr + alg_31_i_Ks + alg_32_i_to1 + alg_34_i_K1 + alg_36_i_Kp + alg_37_i_NaCa + alg_40_i_NaK + alg_42_i_p_Ca + alg_45_i_Na_b + alg_44_i_Ca_b + alg_9_i_Stim) / C_sc)
        d_K_i = torch.zeros_like(y[:, 0]) + (-0.0 * (alg_50_i_Ca_K + alg_29_i_Kr + alg_31_i_Ks + alg_34_i_K1 + alg_36_i_Kp + alg_32_i_to1 + alg_40_i_NaK * -2.0) * A_cap * 1.0 / (V_myo * F))
        d_HTRPNCa = torch.zeros_like(y[:, 0]) + (k_htrpn_plus * Ca_i * (1.0 - HTRPNCa) - k_htrpn_minus * HTRPNCa)
        alg_51_fb = (Ca_i / K_fb) ** N_fb
        alg_52_rb = (Ca_NSR / K_rb) ** N_rb
        alg_53_J_up_winslow = K_SR * (Vmaxf * alg_51_fb - Vmaxr * alg_52_rb) / (1.0 + alg_51_fb + alg_52_rb)
        alg_54_J_up = MgATP_i / MgATP_i0 * alg_53_J_up_winslow
        alg_56_J_tr = (Ca_NSR - Ca_JSR) / tau_tr
        d_Ca_NSR = torch.zeros_like(y[:, 0]) + (alg_54_J_up * V_myo / V_NSR - alg_56_J_tr * V_JSR / V_NSR)
        alg_49_J_rel = v1 * (P_O1 + P_O2) * (Ca_JSR - Ca_ss)
        alg_57_beta_JSR = 1.0 / (1.0 + CSQN_tot * K_mCSQN / (K_mCSQN + Ca_JSR) ** 2.0)
        d_Ca_JSR = torch.zeros_like(y[:, 0]) + (alg_57_beta_JSR * (alg_56_J_tr - alg_49_J_rel))
        d_LTRPNCa = torch.zeros_like(y[:, 0]) + (k_ltrpn_plus * Ca_i * (1.0 - LTRPNCa) - k_ltrpn_minus * LTRPNCa)
        alg_55_J_xfer = (Ca_ss - Ca_i) / tau_xfer
        alg_62_beta_SS = 1.0 / (1.0 + CMDN_tot * K_mCMDN / (K_mCMDN + Ca_ss) ** 2.0 + EGTA_tot * K_mEGTA / (K_mEGTA + Ca_ss) ** 2.0)
        alg_65_ADP_ss = ADP_tot - (CaADP_ss + MgADP_ss)
        alg_63_ATP_ss = ATP_tot - (CaATP_ss + MgATP_ss)
        d_Ca_ss = torch.zeros_like(y[:, 0]) + (alg_62_beta_SS * (alg_49_J_rel * V_JSR / V_ss + k_minus_CaATP * CaATP_ss + k_minus_CaADP * CaADP_ss - (alg_55_J_xfer * V_myo / V_ss + alg_47_i_Ca * (A_cap * 1.0 / (2.0 * V_ss * F)) + k_plus_CaATP * Ca_ss * alg_63_ATP_ss + k_plus_CaADP * Ca_ss * alg_65_ADP_ss)))
        alg_66_Jxfer_CaATP = (CaATP_ss - CaATP_i) / tau_xfer_CaATP
        d_CaATP_ss = torch.zeros_like(y[:, 0]) + (k_plus_CaATP * Ca_ss * alg_63_ATP_ss - (alg_66_Jxfer_CaATP * (V_myo_2 / V_ss_2) + k_minus_CaATP * CaATP_ss))
        alg_67_Jxfer_MgATP = (MgATP_ss - MgATP_i) / tau_xfer_MgATP
        d_MgATP_ss = torch.zeros_like(y[:, 0]) + (k_plus_MgATP * Mg_ss * alg_63_ATP_ss - (alg_67_Jxfer_MgATP * (V_myo_2 / V_ss_2) + k_minus_MgATP * MgATP_ss))
        alg_64_ATP_i = ATP_tot - (CaATP_i + MgATP_i)
        d_CaATP_i = torch.zeros_like(y[:, 0]) + (alg_66_Jxfer_CaATP + k_plus_CaATP * Ca_i * alg_64_ATP_i - k_minus_CaATP * CaATP_i)
        d_MgATP_i = torch.zeros_like(y[:, 0]) + (alg_67_Jxfer_MgATP + k_plus_MgATP * Mg_i * alg_64_ATP_i - k_minus_MgATP * MgATP_i)
        alg_58_J_HTRPNCa = d_HTRPNCa
        alg_59_J_LTRPNCa = d_LTRPNCa
        alg_60_J_trpn = HTRPN_tot * alg_58_J_HTRPNCa + LTRPN_tot * alg_59_J_LTRPNCa
        alg_61_beta_i = 1.0 / (1.0 + CMDN_tot * K_mCMDN / (K_mCMDN + Ca_i) ** 2.0 + EGTA_tot * K_mEGTA / (K_mEGTA + Ca_i) ** 2.0)
        alg_68_ADP_i = ADP_tot - (CaADP_i + MgADP_i)
        d_Ca_i = torch.zeros_like(y[:, 0]) + (alg_61_beta_i * (alg_55_J_xfer - (alg_54_J_up + alg_60_J_trpn) + -((alg_42_i_p_Ca + alg_44_i_Ca_b - 2.0 * alg_37_i_NaCa) * (A_cap * C_sc / (2.0 * V_myo * F))) + (k_minus_CaATP * CaATP_i + k_minus_CaADP * CaADP_i - (k_plus_CaATP * Ca_i * alg_64_ATP_i + k_plus_CaADP * Ca_i * alg_68_ADP_i))))
        alg_70_Jxfer_CaADP = (CaADP_ss - CaADP_i) / tau_xfer_CaADP
        d_CaADP_ss = torch.zeros_like(y[:, 0]) + (k_plus_CaADP * Ca_ss * alg_65_ADP_ss - (alg_70_Jxfer_CaADP * (V_myo_2 / V_ss_2) + k_minus_CaADP * CaADP_ss))
        alg_71_Jxfer_MgADP = (MgADP_ss - MgADP_i) / tau_xfer_MgADP
        d_MgADP_ss = torch.zeros_like(y[:, 0]) + (k_plus_MgADP * Mg_ss * alg_65_ADP_ss - (alg_71_Jxfer_MgADP * (V_myo_2 / V_ss_2) + k_minus_MgADP * MgADP_ss))
        d_CaADP_i = torch.zeros_like(y[:, 0]) + (alg_70_Jxfer_CaADP + k_plus_CaADP * Ca_i * alg_68_ADP_i - k_minus_CaADP * CaADP_i)
        d_MgADP_i = torch.zeros_like(y[:, 0]) + (alg_71_Jxfer_MgADP + k_plus_MgADP * Mg_i * alg_68_ADP_i - k_minus_MgADP * MgADP_i)
        alg_69_Jxfer_Mg = (Mg_ss - Mg_i) / tau_xfer_Mg
        d_Mg_ss = torch.zeros_like(y[:, 0]) + (k_minus_MgATP * MgATP_ss + k_minus_MgADP * MgADP_ss - (k_plus_MgATP * Mg_ss * alg_63_ATP_ss + k_plus_MgADP * Mg_ss * alg_65_ADP_ss + alg_69_Jxfer_Mg * (V_myo_2 / V_ss_2)))
        d_Mg_i = torch.zeros_like(y[:, 0]) + (alg_69_Jxfer_Mg + k_minus_MgATP * MgATP_i + k_minus_MgADP * MgADP_i - (k_plus_MgATP * Mg_i * alg_64_ATP_i + k_plus_MgADP * Mg_i * alg_68_ADP_i))

        return torch.stack(
            [d_V, d_Na_i, d_m, d_h, d_j, d_K_i, d_X_kr, d_X_ks, d_X_to1, d_Y_to1, d_Ca_i, d_MgATP_i, d_O, d_O_Ca, d_C0, d_C1, d_C2, d_C3, d_C4, d_C_Ca0, d_C_Ca1, d_C_Ca2, d_C_Ca3, d_C_Ca4, d_Ca_ss, d_state_y, d_P_O1, d_P_O2, d_P_C1, d_P_C2, d_Ca_JSR, d_Ca_NSR, d_HTRPNCa, d_LTRPNCa, d_CaADP_i, d_CaADP_ss, d_CaATP_ss, d_CaATP_i, d_Mg_ss, d_Mg_i, d_MgADP_i, d_MgADP_ss, d_MgATP_ss],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
