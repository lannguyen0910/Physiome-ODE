import torch
import torch.nn as nn
import math


class ElectrophysiologyViswanathanShawRudy1999CODE(nn.Module):
    """
    PyTorch translation of electrophysiology/viswanathan_shaw_rudy_1999_c/model.py.
    Dynamic states: 25; independent constants: 64; derived constants used by rates: 11.
    Original constant slots: 75; algebraic slots: 79.
    """
    source_file = 'electrophysiology/viswanathan_shaw_rudy_1999_c/model.py'
    source_state_dim = 25
    source_constant_dim = 75
    algebraic_dim = 79
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 25
        self.parameter_dim = 64

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8314, 'T': 310, 'F': 96485, 'stim_start': 100, 'stim_end': 99999000, 'stim_period': 1000, 'stim_duration': 2, 'stim_amplitude': -25.5, 'g_Na': 16, 'Nao': 132, 'delta_m': 1e-05, 'gamma_Nai': 0.75, 'gamma_Nao': 0.75, 'gamma_Ki': 0.75, 'gamma_Ko': 0.75, 'P_Ca': 0.00054, 'P_Na': 6.75e-07, 'P_K': 1.93e-07, 'gamma_Cai': 1, 'gamma_Cao': 0.341, 'Cao': 1.8, 'Ko': 4.5, 'Km_Ca': 0.0006, 'g_CaT': 0.05, 'PNaK': 0.01833, 'g_Kp': 0.00552, 'nATP': 0.24, 'nicholsarea': 5e-05, 'ATPi': 3, 'hATP': 2, 'kATP': 0.00025, 'K_mpCa': 0.0005, 'I_pCa': 1.15, 'g_Nab': 0.004, 'g_Cab': 0.003016, 'I_NaK': 2, 'K_mNai': 10, 'K_mKo': 1.5, 'K_m_ns_Ca': 0.0012, 'n_NaCa': 3, 'K_NaCa': 0.002, 'd_NaCa': 0.001, 'gamma': 0.5, 'G_rel_max': 60000, 'G_rel_overload': 4000, 'tau_tr': 0.18, 'K_mrel': 0.0008, 'delta_Ca_ith': 0.00018, 'CSQN_max': 10, 'K_mCSQN': 0.8, 'K_mup': 0.00092, 'I_up': 5, 'Ca_NSR_max': 15, 'A_cap': 1.434e-07, 'K_mTn': 0.0005, 'K_mCMDN': 0.00238, 'Tn_max': 0.07, 'CMDN_max': 0.05, 'CSQNthresh': 0.7, 'Logicthresh': 0.98, 'preplength': 0.001, 'radius': 0.00011, 'g_to': 0, 'P_ns_Ca': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -88.5004, 'Nai': 14.177596, 'm': 0.0008601, 'h': 0.992806, 'j': 0.99511469, 'Cai': 0.00034209, 'Ki': 138.37949, 'd': 3.538268e-06, 'f': 0.995259, 'b': 0.0010301, 'g': 0.9341828, 'xr': 0.0007500578, 'xs1': 0.0421517, 'xs2': 0.08866, 'zdv': 0.01198974, 'ydv': 0.97497, 'Ca_JSR': 0.8204291, 'Ca_NSR': 2.7872629, 'APtrack': 3.7458e-87, 'APtrack2': 2.07066e-86, 'APtrack3': 0.01924195, 'Cainfluxtrack': 1.660129e-87, 'OVRLDtrack': 2e-323, 'OVRLDtrack2': -2.5e-323, 'OVRLDtrack3': 2.2645e-282}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'stim_start', 'stim_end', 'stim_period', 'stim_duration', 'stim_amplitude', 'g_Na', 'Nao', 'delta_m', 'gamma_Nai', 'gamma_Nao', 'gamma_Ki', 'gamma_Ko', 'P_Ca', 'P_Na', 'P_K', 'gamma_Cai', 'gamma_Cao', 'Cao', 'Ko', 'Km_Ca', 'g_CaT', 'PNaK', 'g_Kp', 'nATP', 'nicholsarea', 'ATPi', 'hATP', 'kATP', 'K_mpCa', 'I_pCa', 'g_Nab', 'g_Cab', 'I_NaK', 'K_mNai', 'K_mKo', 'K_m_ns_Ca', 'n_NaCa', 'K_NaCa', 'd_NaCa', 'gamma', 'G_rel_max', 'G_rel_overload', 'tau_tr', 'K_mrel', 'delta_Ca_ith', 'CSQN_max', 'K_mCSQN', 'K_mup', 'I_up', 'Ca_NSR_max', 'A_cap', 'K_mTn', 'K_mCMDN', 'Tn_max', 'CMDN_max', 'CSQNthresh', 'Logicthresh', 'preplength', 'radius', 'g_to', 'P_ns_Ca']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'Nai', 'm', 'h', 'j', 'Cai', 'Ki', 'd', 'f', 'b', 'g', 'xr', 'xs1', 'xs2', 'zdv', 'ydv', 'Ca_JSR', 'Ca_NSR', 'APtrack', 'APtrack2', 'APtrack3', 'Cainfluxtrack', 'OVRLDtrack', 'OVRLDtrack2', 'OVRLDtrack3']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (25)
        V = y[:, 0]
        Nai = y[:, 1]
        m = y[:, 2]
        h = y[:, 3]
        j = y[:, 4]
        Cai = y[:, 5]
        Ki = y[:, 6]
        d = y[:, 7]
        f = y[:, 8]
        b = y[:, 9]
        g = y[:, 10]
        xr = y[:, 11]
        xs1 = y[:, 12]
        xs2 = y[:, 13]
        zdv = y[:, 14]
        ydv = y[:, 15]
        Ca_JSR = y[:, 16]
        Ca_NSR = y[:, 17]
        APtrack = y[:, 18]
        APtrack2 = y[:, 19]
        APtrack3 = y[:, 20]
        Cainfluxtrack = y[:, 21]
        OVRLDtrack = y[:, 22]
        OVRLDtrack2 = y[:, 23]
        OVRLDtrack3 = y[:, 24]

        # Independent parameters (source constant order)
        R = theta[:, 0]  # constants[0]
        T = theta[:, 1]  # constants[1]
        F = theta[:, 2]  # constants[2]
        stim_start = theta[:, 3]  # constants[3]
        stim_end = theta[:, 4]  # constants[4]
        stim_period = theta[:, 5]  # constants[5]
        stim_duration = theta[:, 6]  # constants[6]
        stim_amplitude = theta[:, 7]  # constants[7]
        g_Na = theta[:, 8]  # constants[8]
        Nao = theta[:, 9]  # constants[9]
        delta_m = theta[:, 10]  # constants[10]
        gamma_Nai = theta[:, 11]  # constants[11]
        gamma_Nao = theta[:, 12]  # constants[12]
        gamma_Ki = theta[:, 13]  # constants[13]
        gamma_Ko = theta[:, 14]  # constants[14]
        P_Ca = theta[:, 15]  # constants[15]
        P_Na = theta[:, 16]  # constants[16]
        P_K = theta[:, 17]  # constants[17]
        gamma_Cai = theta[:, 18]  # constants[18]
        gamma_Cao = theta[:, 19]  # constants[19]
        Cao = theta[:, 20]  # constants[20]
        Ko = theta[:, 21]  # constants[21]
        Km_Ca = theta[:, 22]  # constants[22]
        g_CaT = theta[:, 23]  # constants[23]
        PNaK = theta[:, 24]  # constants[24]
        g_Kp = theta[:, 25]  # constants[25]
        nATP = theta[:, 26]  # constants[26]
        nicholsarea = theta[:, 27]  # constants[27]
        ATPi = theta[:, 28]  # constants[28]
        hATP = theta[:, 29]  # constants[29]
        kATP = theta[:, 30]  # constants[30]
        K_mpCa = theta[:, 31]  # constants[31]
        I_pCa = theta[:, 32]  # constants[32]
        g_Nab = theta[:, 33]  # constants[33]
        g_Cab = theta[:, 34]  # constants[34]
        I_NaK = theta[:, 35]  # constants[35]
        K_mNai = theta[:, 36]  # constants[36]
        K_mKo = theta[:, 37]  # constants[37]
        K_m_ns_Ca = theta[:, 38]  # constants[38]
        n_NaCa = theta[:, 39]  # constants[39]
        K_NaCa = theta[:, 40]  # constants[40]
        d_NaCa = theta[:, 41]  # constants[41]
        gamma = theta[:, 42]  # constants[42]
        G_rel_max = theta[:, 43]  # constants[43]
        G_rel_overload = theta[:, 44]  # constants[44]
        tau_tr = theta[:, 45]  # constants[45]
        K_mrel = theta[:, 46]  # constants[46]
        delta_Ca_ith = theta[:, 47]  # constants[47]
        CSQN_max = theta[:, 48]  # constants[48]
        K_mCSQN = theta[:, 49]  # constants[49]
        K_mup = theta[:, 50]  # constants[50]
        I_up = theta[:, 51]  # constants[51]
        Ca_NSR_max = theta[:, 52]  # constants[52]
        A_cap = theta[:, 53]  # constants[53]
        K_mTn = theta[:, 54]  # constants[54]
        K_mCMDN = theta[:, 55]  # constants[55]
        Tn_max = theta[:, 56]  # constants[56]
        CMDN_max = theta[:, 57]  # constants[57]
        CSQNthresh = theta[:, 58]  # constants[58]
        Logicthresh = theta[:, 59]  # constants[59]
        preplength = theta[:, 60]  # constants[60]
        radius = theta[:, 61]  # constants[61]
        g_to = theta[:, 62]  # constants[65]
        P_ns_Ca = theta[:, 63]  # constants[67]

        # Derived constants from the original initConsts()
        g_Kr = 0.02614 * (Ko / 5.4) ** (1.0 / 2)  # constants[62]
        g_K1 = 0.75 * (Ko / 5.4) ** (1.0 / 2)  # constants[63]
        g_K_ATP = 0.000193 / nicholsarea  # constants[64]
        sigma = 1.0 / 7.0 * (torch.exp(Nao / 67.3) - 1.0)  # constants[66]
        K_leak = I_up / Ca_NSR_max  # constants[68]
        volume = math.pi * preplength * radius ** 2.0  # constants[69]
        pATP = 1.0 / (1.0 + (ATPi / kATP) ** hATP)  # constants[70]
        V_myo = 0.68 * volume  # constants[71]
        GKbaraATP = g_K_ATP * pATP * (Ko / 4.0) ** nATP  # constants[72]
        V_JSR = 0.0048 / 0.68 * V_myo  # constants[73]
        V_NSR = 0.0552 / 0.68 * V_myo  # constants[74]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_Nai = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_j = torch.zeros_like(y[:, 0])
        d_Cai = torch.zeros_like(y[:, 0])
        d_Ki = torch.zeros_like(y[:, 0])
        d_d = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_b = torch.zeros_like(y[:, 0])
        d_g = torch.zeros_like(y[:, 0])
        d_xr = torch.zeros_like(y[:, 0])
        d_xs1 = torch.zeros_like(y[:, 0])
        d_xs2 = torch.zeros_like(y[:, 0])
        d_zdv = torch.zeros_like(y[:, 0])
        d_ydv = torch.zeros_like(y[:, 0])
        d_Ca_JSR = torch.zeros_like(y[:, 0])
        d_Ca_NSR = torch.zeros_like(y[:, 0])
        d_APtrack = torch.zeros_like(y[:, 0])
        d_APtrack2 = torch.zeros_like(y[:, 0])
        d_APtrack3 = torch.zeros_like(y[:, 0])
        d_Cainfluxtrack = torch.zeros_like(y[:, 0])
        d_OVRLDtrack = torch.zeros_like(y[:, 0])
        d_OVRLDtrack2 = torch.zeros_like(y[:, 0])
        d_OVRLDtrack3 = torch.zeros_like(y[:, 0])
        alg_0_E0_m = torch.zeros_like(y[:, 0])
        alg_1_alpha_h = torch.zeros_like(y[:, 0])
        alg_2_alpha_j = torch.zeros_like(y[:, 0])
        alg_3_E0_d = torch.zeros_like(y[:, 0])
        alg_4_f_infinity = torch.zeros_like(y[:, 0])
        alg_5_b_inf = torch.zeros_like(y[:, 0])
        alg_6_g_inf = torch.zeros_like(y[:, 0])
        alg_7_xr_infinity = torch.zeros_like(y[:, 0])
        alg_8_xs1_infinity = torch.zeros_like(y[:, 0])
        alg_9_xs2_infinity = torch.zeros_like(y[:, 0])
        alg_10_alpha_zdv = torch.zeros_like(y[:, 0])
        alg_11_alpha_ydv = torch.zeros_like(y[:, 0])
        alg_12_I_st = torch.zeros_like(y[:, 0])
        alg_13_alpha_m = torch.zeros_like(y[:, 0])
        alg_14_beta_h = torch.zeros_like(y[:, 0])
        alg_15_beta_j = torch.zeros_like(y[:, 0])
        alg_16_d_infinity = torch.zeros_like(y[:, 0])
        alg_17_tau_f = torch.zeros_like(y[:, 0])
        alg_18_tau_b = torch.zeros_like(y[:, 0])
        alg_19_tau_g = torch.zeros_like(y[:, 0])
        alg_20_tau_xr = torch.zeros_like(y[:, 0])
        alg_21_tau_xs1 = torch.zeros_like(y[:, 0])
        alg_22_tau_xs2 = torch.zeros_like(y[:, 0])
        alg_23_beta_zdv = torch.zeros_like(y[:, 0])
        alg_24_beta_ydv = torch.zeros_like(y[:, 0])
        alg_25_E_Na = torch.zeros_like(y[:, 0])
        alg_26_beta_m = torch.zeros_like(y[:, 0])
        alg_27_tau_d = torch.zeros_like(y[:, 0])
        alg_28_alpha_f = torch.zeros_like(y[:, 0])
        alg_29_tau_zdv = torch.zeros_like(y[:, 0])
        alg_30_tau_ydv = torch.zeros_like(y[:, 0])
        alg_31_i_Na = torch.zeros_like(y[:, 0])
        alg_32_alpha_d = torch.zeros_like(y[:, 0])
        alg_33_beta_f = torch.zeros_like(y[:, 0])
        alg_34_zdv_ss = torch.zeros_like(y[:, 0])
        alg_35_ydv_ss = torch.zeros_like(y[:, 0])
        alg_36_I_CaCa = torch.zeros_like(y[:, 0])
        alg_37_beta_d = torch.zeros_like(y[:, 0])
        alg_38_I_CaNa = torch.zeros_like(y[:, 0])
        alg_39_I_CaK = torch.zeros_like(y[:, 0])
        alg_40_f_Ca = torch.zeros_like(y[:, 0])
        alg_41_i_CaCa = torch.zeros_like(y[:, 0])
        alg_42_i_CaNa = torch.zeros_like(y[:, 0])
        alg_43_i_CaK = torch.zeros_like(y[:, 0])
        alg_44_i_Ca_L = torch.zeros_like(y[:, 0])
        alg_45_Rect = torch.zeros_like(y[:, 0])
        alg_46_E_Ks = torch.zeros_like(y[:, 0])
        alg_47_g_Ks = torch.zeros_like(y[:, 0])
        alg_48_i_Ks = torch.zeros_like(y[:, 0])
        alg_49_E_K = torch.zeros_like(y[:, 0])
        alg_50_i_Kr = torch.zeros_like(y[:, 0])
        alg_51_alpha_K1 = torch.zeros_like(y[:, 0])
        alg_52_beta_K1 = torch.zeros_like(y[:, 0])
        alg_53_K1_infinity = torch.zeros_like(y[:, 0])
        alg_54_i_K1 = torch.zeros_like(y[:, 0])
        alg_55_Kp = torch.zeros_like(y[:, 0])
        alg_56_i_Kp = torch.zeros_like(y[:, 0])
        alg_57_i_K_ATP = torch.zeros_like(y[:, 0])
        alg_58_rvdv = torch.zeros_like(y[:, 0])
        alg_59_i_to = torch.zeros_like(y[:, 0])
        alg_60_i_p_Ca = torch.zeros_like(y[:, 0])
        alg_61_i_Na_b = torch.zeros_like(y[:, 0])
        alg_62_E_Ca = torch.zeros_like(y[:, 0])
        alg_63_i_Ca_T = torch.zeros_like(y[:, 0])
        alg_64_i_Ca_b = torch.zeros_like(y[:, 0])
        alg_65_f_NaK = torch.zeros_like(y[:, 0])
        alg_66_i_NaK = torch.zeros_like(y[:, 0])
        alg_67_I_ns_Na = torch.zeros_like(y[:, 0])
        alg_68_I_ns_K = torch.zeros_like(y[:, 0])
        alg_69_i_ns_Na = torch.zeros_like(y[:, 0])
        alg_70_i_ns_K = torch.zeros_like(y[:, 0])
        alg_71_i_ns_Ca = torch.zeros_like(y[:, 0])
        alg_72_i_NaCa = torch.zeros_like(y[:, 0])
        alg_73_dVdt = torch.zeros_like(y[:, 0])
        alg_74_G_rel = torch.zeros_like(y[:, 0])
        alg_75_i_rel = torch.zeros_like(y[:, 0])
        alg_76_i_up = torch.zeros_like(y[:, 0])
        alg_77_i_leak = torch.zeros_like(y[:, 0])
        alg_78_i_tr = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_APtrack2 = torch.zeros_like(y[:, 0]) + (torch.where((APtrack < 0.2) & (APtrack > 0.18), 100.0 * (1.0 - APtrack2) - 0.5 * APtrack2, -0.5 * APtrack2))
        d_APtrack3 = torch.zeros_like(y[:, 0]) + (torch.where((APtrack < 0.2) & (APtrack > 0.18), 100.0 * (1.0 - APtrack3) - 0.5 * APtrack3, -0.01 * APtrack3))
        d_OVRLDtrack = torch.zeros_like(y[:, 0]) + (torch.where((1.0 / (1.0 + K_mCSQN / Ca_JSR) > CSQNthresh) & (OVRLDtrack3 < 0.37) & (APtrack3 < 0.37), 50.0 * (1.0 - OVRLDtrack), -0.5 * OVRLDtrack))
        d_OVRLDtrack2 = torch.zeros_like(y[:, 0]) + (torch.where((OVRLDtrack > Logicthresh) & (OVRLDtrack2 < Logicthresh), 50.0 * (1.0 - OVRLDtrack2), -0.5 * OVRLDtrack2))
        d_OVRLDtrack3 = torch.zeros_like(y[:, 0]) + (torch.where((OVRLDtrack > Logicthresh) & (OVRLDtrack3 < Logicthresh), 50.0 * (1.0 - OVRLDtrack3), -0.01 * OVRLDtrack3))
        alg_1_alpha_h = torch.where(V < -40.0, 0.135 * torch.exp((80.0 + V) / -6.8), 0.0)
        alg_14_beta_h = torch.where(V < -40.0, 3.56 * torch.exp(0.079 * V) + 310000.0 * torch.exp(0.35 * V), 1.0 / (0.13 * (1.0 + torch.exp((V + 10.66) / -11.1))))
        d_h = torch.zeros_like(y[:, 0]) + (alg_1_alpha_h * (1.0 - h) - alg_14_beta_h * h)
        alg_2_alpha_j = torch.where(V < -40.0, -(127140.0 * torch.exp(0.2444 * V) + 3.474e-05 * torch.exp(-0.04391 * V)) * (V + 37.78) / (1.0 + torch.exp(0.311 * (V + 79.23))), 0.0)
        alg_15_beta_j = torch.where(V < -40.0, 0.1212 * torch.exp(-0.01052 * V) / (1.0 + torch.exp(-0.1378 * (V + 40.14))), 0.3 * torch.exp(-2.535e-07 * V) / (1.0 + torch.exp(-0.1 * (V + 32.0))))
        d_j = torch.zeros_like(y[:, 0]) + (alg_2_alpha_j * (1.0 - j) - alg_15_beta_j * j)
        alg_5_b_inf = 1.0 / (1.0 + torch.exp(-(V + 14.0) / 10.8))
        alg_18_tau_b = 3.7 + 6.1 / (1.0 + torch.exp((V + 25.0) / 4.5))
        d_b = torch.zeros_like(y[:, 0]) + ((alg_5_b_inf - b) / alg_18_tau_b)
        alg_6_g_inf = 1.0 / (1.0 + torch.exp((V + 60.0) / 5.6))
        alg_19_tau_g = torch.where(V <= 0.0, -0.875 * V + 12.0, 12.0)
        d_g = torch.zeros_like(y[:, 0]) + ((alg_6_g_inf - g) / alg_19_tau_g)
        alg_7_xr_infinity = 1.0 / (1.0 + torch.exp(-(V + 21.5) / 7.5))
        alg_20_tau_xr = 1.0 / (0.00138 * (V + 14.2) / (1.0 - torch.exp(-0.123 * (V + 14.2))) + 0.00061 * (V + 38.9) / (torch.exp(0.145 * (V + 38.9)) - 1.0))
        d_xr = torch.zeros_like(y[:, 0]) + ((alg_7_xr_infinity - xr) / alg_20_tau_xr)
        alg_8_xs1_infinity = 1.0 / (1.0 + torch.exp(-(V - 1.5) / 16.7))
        alg_21_tau_xs1 = 1.0 / (7.19e-05 * (V + 30.0) / (1.0 - torch.exp(-0.148 * (V + 30.0))) + 0.000131 * (V + 30.0) / (torch.exp(0.0687 * (V + 30.0)) - 1.0))
        d_xs1 = torch.zeros_like(y[:, 0]) + ((alg_8_xs1_infinity - xs1) / alg_21_tau_xs1)
        alg_9_xs2_infinity = 1.0 / (1.0 + torch.exp(-(V - 1.5) / 16.7))
        alg_22_tau_xs2 = 4.0 / (7.19e-05 * (V + 30.0) / (1.0 - torch.exp(-0.148 * (V + 30.0))) + 0.000131 * (V + 30.0) / (torch.exp(0.0687 * (V + 30.0)) - 1.0))
        d_xs2 = torch.zeros_like(y[:, 0]) + ((alg_9_xs2_infinity - xs2) / alg_22_tau_xs2)
        alg_0_E0_m = V + 47.13
        alg_13_alpha_m = torch.where(torch.abs(alg_0_E0_m) >= delta_m, 0.32 * alg_0_E0_m / (1.0 - torch.exp(-0.1 * alg_0_E0_m)), 3.2)
        alg_26_beta_m = 0.08 * torch.exp(-V / 11.0)
        d_m = torch.zeros_like(y[:, 0]) + (alg_13_alpha_m * (1.0 - m) - alg_26_beta_m * m)
        alg_4_f_infinity = 1.0 / (1.0 + torch.exp((V + 32.0) / 8.0)) + 0.6 / (1.0 + torch.exp((50.0 - V) / 20.0))
        alg_17_tau_f = 1.0 / (0.0197 * torch.exp(-(0.0337 * (V + 10.0)) ** 2.0) + 0.02)
        alg_28_alpha_f = alg_4_f_infinity / alg_17_tau_f
        alg_33_beta_f = (1.0 - alg_4_f_infinity) / alg_17_tau_f
        d_f = torch.zeros_like(y[:, 0]) + (alg_28_alpha_f * (1.0 - f) - alg_33_beta_f * f)
        alg_10_alpha_zdv = 10.0 * torch.exp((V - 40.0) / 25.0) / (1.0 + torch.exp((V - 40.0) / 25.0))
        alg_23_beta_zdv = 10.0 * torch.exp(-(V + 90.0) / 25.0) / (1.0 + torch.exp(-(V + 90.0) / 25.0))
        alg_29_tau_zdv = 1.0 / (alg_10_alpha_zdv + alg_23_beta_zdv)
        alg_34_zdv_ss = alg_10_alpha_zdv / (alg_10_alpha_zdv + alg_23_beta_zdv)
        d_zdv = torch.zeros_like(y[:, 0]) + ((alg_34_zdv_ss - zdv) / alg_29_tau_zdv)
        alg_11_alpha_ydv = 0.015 / (1.0 + torch.exp((V + 60.0) / 5.0))
        alg_24_beta_ydv = 0.1 * torch.exp((V + 25.0) / 5.0) / (1.0 + torch.exp((V + 25.0) / 5.0))
        alg_30_tau_ydv = 1.0 / (alg_11_alpha_ydv + alg_24_beta_ydv)
        alg_35_ydv_ss = alg_11_alpha_ydv / (alg_11_alpha_ydv + alg_24_beta_ydv)
        d_ydv = torch.zeros_like(y[:, 0]) + ((alg_35_ydv_ss - ydv) / alg_30_tau_ydv)
        alg_3_E0_d = V + 10.0
        alg_16_d_infinity = 1.0 / (1.0 + torch.exp(-alg_3_E0_d / 6.24))
        alg_27_tau_d = torch.where(torch.abs(alg_3_E0_d) < 1e-05, 1.0 / (0.035 * 6.24), alg_16_d_infinity * (1.0 - torch.exp(-alg_3_E0_d / 6.24)) / (0.035 * alg_3_E0_d))
        alg_32_alpha_d = alg_16_d_infinity / alg_27_tau_d
        alg_37_beta_d = (1.0 - alg_16_d_infinity) / alg_27_tau_d
        d_d = torch.zeros_like(y[:, 0]) + (alg_32_alpha_d * (1.0 - d) - alg_37_beta_d * d)
        alg_45_Rect = 1.0 / (1.0 + torch.exp((V + 9.0) / 22.4))
        alg_49_E_K = R * T / F * torch.log(Ko / Ki)
        alg_50_i_Kr = g_Kr * xr * alg_45_Rect * (V - alg_49_E_K)
        alg_47_g_Ks = 0.289 * (1.0 + 0.6 / (1.0 + (3.8e-05 / Cai) ** 1.4))
        alg_46_E_Ks = R * T / F * torch.log((Ko + PNaK * Nao) / (Ki + PNaK * Nai))
        alg_48_i_Ks = alg_47_g_Ks * xs1 * xs2 * (V - alg_46_E_Ks)
        alg_57_i_K_ATP = GKbaraATP * (V - alg_49_E_K)
        alg_58_rvdv = torch.exp(V / 100.0)
        alg_59_i_to = g_to * zdv ** 3.0 * ydv * alg_58_rvdv * (V - alg_49_E_K)
        alg_51_alpha_K1 = 1.02 / (1.0 + torch.exp(0.2385 * (V - alg_49_E_K - 59.215)))
        alg_52_beta_K1 = 1.0 * (0.49124 * torch.exp(0.08032 * (V - alg_49_E_K + 5.476)) + torch.exp(0.06175 * (V - alg_49_E_K - 594.31))) / (1.0 + torch.exp(-0.5143 * (V - alg_49_E_K + 4.753)))
        alg_53_K1_infinity = alg_51_alpha_K1 / (alg_51_alpha_K1 + alg_52_beta_K1)
        alg_54_i_K1 = g_K1 * alg_53_K1_infinity * (V - alg_49_E_K)
        alg_55_Kp = 1.0 / (1.0 + torch.exp((7.488 - V) / 5.98))
        alg_56_i_Kp = g_Kp * alg_55_Kp * (V - alg_49_E_K)
        alg_65_f_NaK = 1.0 / (1.0 + 0.1245 * torch.exp(-0.1 * V * F / (R * T)) + 0.0365 * sigma * torch.exp(-V * F / (R * T)))
        alg_66_i_NaK = I_NaK * alg_65_f_NaK * 1.0 / (1.0 + (K_mNai / Nai) ** 2.0) * Ko / (Ko + K_mKo)
        alg_39_I_CaK = P_K * 1.0 * V * F ** 2.0 / (R * T) * (gamma_Ki * Ki * torch.exp(1.0 * V * F / (R * T)) - gamma_Ko * Ko) / (torch.exp(1.0 * V * F / (R * T)) - 1.0)
        alg_40_f_Ca = 1.0 / (1.0 + Cai / Km_Ca)
        alg_43_i_CaK = d * f * alg_40_f_Ca * alg_39_I_CaK
        alg_68_I_ns_K = P_ns_Ca * 1.0 * V * F ** 2.0 / (R * T) * (gamma_Ki * Ki * torch.exp(1.0 * V * F / (R * T)) - gamma_Ko * Ko) / (torch.exp(1.0 * V * F / (R * T)) - 1.0)
        alg_70_i_ns_K = alg_68_I_ns_K * 1.0 / (1.0 + (K_m_ns_Ca / Cai) ** 3.0)
        d_Ki = torch.zeros_like(y[:, 0]) + (-0.001 * (alg_43_i_CaK + alg_50_i_Kr + alg_48_i_Ks + alg_54_i_K1 + alg_56_i_Kp + alg_57_i_K_ATP + alg_59_i_to + alg_70_i_ns_K + -alg_66_i_NaK * 2.0) * A_cap / (V_myo * F))
        alg_62_E_Ca = R * T / (2.0 * F) * torch.log(Cao / Cai)
        alg_63_i_Ca_T = g_CaT * b * b * g * (V - alg_62_E_Ca)
        alg_60_i_p_Ca = I_pCa * Cai / (K_mpCa + Cai)
        alg_64_i_Ca_b = g_Cab * (V - alg_62_E_Ca)
        alg_72_i_NaCa = K_NaCa * (torch.exp(gamma * (n_NaCa - 2.0) * V * F / (R * T)) * Nai ** n_NaCa * Cao - torch.exp((gamma - 1.0) * (n_NaCa - 2.0) * V * F / (R * T)) * Nao ** n_NaCa * Cai) / ((1.0 + d_NaCa * (Cai * Nao ** n_NaCa + Cao * Nai ** n_NaCa)) * (1.0 + Cai / 0.0069))
        alg_36_I_CaCa = P_Ca * 4.0 * V * F ** 2.0 / (R * T) * (gamma_Cai * Cai * torch.exp(2.0 * V * F / (R * T)) - gamma_Cao * Cao) / (torch.exp(2.0 * V * F / (R * T)) - 1.0)
        alg_41_i_CaCa = d * f * alg_40_f_Ca * alg_36_I_CaCa
        d_Cainfluxtrack = torch.zeros_like(y[:, 0]) + (torch.where(APtrack > 0.2, -1.0 * A_cap * (alg_41_i_CaCa + alg_63_i_Ca_T - alg_72_i_NaCa + alg_60_i_p_Ca + alg_64_i_Ca_b) / (2.0 * V_myo * F), torch.where((APtrack2 > 0.01) & (APtrack <= 0.2), 0.0, -0.5 * Cainfluxtrack)))
        alg_25_E_Na = R * T / F * torch.log(Nao / Nai)
        alg_31_i_Na = g_Na * m ** 3.0 * h * j * (V - alg_25_E_Na)
        alg_61_i_Na_b = g_Nab * (V - alg_25_E_Na)
        alg_38_I_CaNa = P_Na * 1.0 * V * F ** 2.0 / (R * T) * (gamma_Nai * Nai * torch.exp(1.0 * V * F / (R * T)) - gamma_Nao * Nao) / (torch.exp(1.0 * V * F / (R * T)) - 1.0)
        alg_42_i_CaNa = d * f * alg_40_f_Ca * alg_38_I_CaNa
        alg_67_I_ns_Na = P_ns_Ca * 1.0 * V * F ** 2.0 / (R * T) * (gamma_Nai * Nai * torch.exp(1.0 * V * F / (R * T)) - gamma_Nao * Nao) / (torch.exp(1.0 * V * F / (R * T)) - 1.0)
        alg_69_i_ns_Na = alg_67_I_ns_Na * 1.0 / (1.0 + (K_m_ns_Ca / Cai) ** 3.0)
        d_Nai = torch.zeros_like(y[:, 0]) + (-0.001 * (alg_31_i_Na + alg_42_i_CaNa + alg_61_i_Na_b + alg_69_i_ns_Na + alg_72_i_NaCa * 3.0 + alg_66_i_NaK * 3.0) * A_cap / (V_myo * F))
        alg_12_I_st = torch.where((t >= stim_start) & (t <= stim_end) & (t - stim_start - torch.floor((t - stim_start) / stim_period) * stim_period <= stim_duration), stim_amplitude, 0.0)
        alg_44_i_Ca_L = alg_41_i_CaCa + alg_43_i_CaK + alg_42_i_CaNa
        alg_71_i_ns_Ca = alg_69_i_ns_Na + alg_70_i_ns_K
        alg_73_dVdt = -(alg_31_i_Na + alg_44_i_Ca_L + alg_63_i_Ca_T + alg_50_i_Kr + alg_48_i_Ks + alg_57_i_K_ATP + alg_59_i_to + alg_54_i_K1 + alg_56_i_Kp + alg_60_i_p_Ca + alg_61_i_Na_b + alg_64_i_Ca_b + alg_66_i_NaK + alg_72_i_NaCa + alg_71_i_ns_Ca + alg_12_I_st)
        d_V = torch.zeros_like(y[:, 0]) + (alg_73_dVdt)
        d_APtrack = torch.zeros_like(y[:, 0]) + (torch.where(alg_73_dVdt > 150.0, 100.0 * (1.0 - APtrack) - 0.5 * APtrack, -0.5 * APtrack))
        alg_74_G_rel = torch.where(Cainfluxtrack > delta_Ca_ith, G_rel_max * (Cainfluxtrack - delta_Ca_ith) / (K_mrel + Cainfluxtrack - delta_Ca_ith) * (1.0 - APtrack2) * APtrack2, torch.where((Cainfluxtrack <= delta_Ca_ith) & (OVRLDtrack2 > 0.0), G_rel_overload * (1.0 - OVRLDtrack2) * OVRLDtrack2, 0.0))
        alg_75_i_rel = alg_74_G_rel * (Ca_JSR - Cai)
        alg_76_i_up = I_up * Cai / (Cai + K_mup)
        alg_77_i_leak = K_leak * Ca_NSR
        d_Cai = torch.zeros_like(y[:, 0]) + (0.001 / (1.0 + CMDN_max * K_mCMDN / (K_mCMDN + Cai) ** 2.0 + Tn_max * K_mTn / (K_mTn + Cai) ** 2.0) * (-1.0 * A_cap * (alg_41_i_CaCa + alg_63_i_Ca_T - alg_72_i_NaCa + alg_60_i_p_Ca + alg_64_i_Ca_b) / (2.0 * V_myo * F) + alg_75_i_rel * V_JSR / V_myo + (alg_77_i_leak - alg_76_i_up) * V_NSR / V_myo))
        alg_78_i_tr = (Ca_NSR - Ca_JSR) / tau_tr
        d_Ca_JSR = torch.zeros_like(y[:, 0]) + (0.001 / (1.0 + CSQN_max * K_mCSQN / (K_mCSQN + Ca_JSR) ** 2.0) * (alg_78_i_tr - alg_75_i_rel))
        d_Ca_NSR = torch.zeros_like(y[:, 0]) + (0.001 * (-alg_78_i_tr * V_JSR / V_NSR - alg_77_i_leak + alg_76_i_up))

        return torch.stack(
            [d_V, d_Nai, d_m, d_h, d_j, d_Cai, d_Ki, d_d, d_f, d_b, d_g, d_xr, d_xs1, d_xs2, d_zdv, d_ydv, d_Ca_JSR, d_Ca_NSR, d_APtrack, d_APtrack2, d_APtrack3, d_Cainfluxtrack, d_OVRLDtrack, d_OVRLDtrack2, d_OVRLDtrack3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
