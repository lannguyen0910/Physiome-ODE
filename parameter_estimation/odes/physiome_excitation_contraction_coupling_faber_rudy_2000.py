import torch
import torch.nn as nn
import math


class ExcitationContractionCouplingFaberRudy2000ODE(nn.Module):
    """
    PyTorch translation of excitation-contraction_coupling/faber_rudy_2000/model.py.
    Dynamic states: 25; independent constants: 70; derived constants used by rates: 11.
    Original constant slots: 81; algebraic slots: 85.
    """
    source_file = 'excitation-contraction_coupling/faber_rudy_2000/model.py'
    source_state_dim = 25
    source_constant_dim = 81
    algebraic_dim = 85
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
        self.parameter_dim = 70

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8314, 'T': 310, 'F': 96485, 'stim_start': 100, 'stim_end': 9000000, 'stim_period': 1000, 'stim_duration': 2, 'stim_amplitude': -25.5, 'g_Na': 16, 'Nao': 140, 'delta_m': 1e-05, 'gamma_Nai': 0.75, 'gamma_Nao': 0.75, 'gamma_Ki': 0.75, 'gamma_Ko': 0.75, 'gamma_Cai': 1, 'gamma_Cao': 0.341, 'P_Ca': 0.00054, 'P_Na': 6.75e-07, 'P_K': 1.93e-07, 'Cao': 1.8, 'Ko': 5.4, 'Km_Ca': 0.0006, 'g_CaT': 0.05, 'G_Kr': 0.02614, 'G_Ks': 0.433, 'PNaK': 0.01833, 'G_K1': 0.75, 'g_Kp': 0.00552, 'g_K_Na': 0.12848, 'nKNa': 2.8, 'kdKNa': 66, 'i_K_ATP_on': 1, 'nATP': 0.24, 'nicholsarea': 5e-05, 'ATPi': 3, 'hATP': 2, 'kATP': 0.00025, 'K_mpCa': 0.0005, 'I_pCa': 1.15, 'g_Nab': 0.004, 'g_Cab': 0.003016, 'I_NaK': 2.25, 'K_mNai': 10, 'K_mKo': 1.5, 'K_m_ns_Ca': 0.0012, 'c1': 0.00025, 'c2': 0.0001, 'gamma': 0.15, 'G_rel_max': 60, 'G_rel_overload': 4, 'tau_tr': 180, 'K_mrel': 0.0008, 'delta_Ca_ith': 0.00018, 'CSQN_max': 10, 'K_mCSQN': 0.8, 'K_mup': 0.00092, 'I_up': 0.00875, 'Ca_NSR_max': 15, 'A_cap': 0.0001534, 'K_mTn': 0.0005, 'K_mCMDN': 0.00238, 'Tn_max': 0.07, 'CMDN_max': 0.05, 'CSQNthresh': 0.7, 'Logicthresh': 0.98, 'preplength': 0.1, 'radius': 0.011, 'g_to': 0, 'P_ns_Ca': 1.75e-07}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -85.2119207874627, 'Nai': 13.3649235394859, 'm': 0.00149183115674257, 'h': 0.985596581239651, 'j': 0.990898461370389, 'Cai': 0.000117482029668194, 'Ki': 141.056872392446, 'd': 5.82597094505446e-06, 'f': 0.997765362821995, 'b': 0.00136737866785149, 'g': 0.98881442877378, 'xr': 0.000204700363126417, 'xs1': 0.00660746743356887, 'xs2': 0.0303768241233812, 'zdv': 0.0144622472219576, 'ydv': 0.999945568566232, 'Ca_JSR': 1.12791401197882, 'Ca_NSR': 1.76731003671612, 'APtrack': 9.65910542308504e-196, 'APtrack2': 5.33944967562997e-195, 'APtrack3': 0.000129515197402902, 'Cainfluxtrack': 2.69380318286645e-196, 'OVRLDtrack': 0, 'OVRLDtrack2': 0, 'OVRLDtrack3': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'stim_start', 'stim_end', 'stim_period', 'stim_duration', 'stim_amplitude', 'g_Na', 'Nao', 'delta_m', 'gamma_Nai', 'gamma_Nao', 'gamma_Ki', 'gamma_Ko', 'gamma_Cai', 'gamma_Cao', 'P_Ca', 'P_Na', 'P_K', 'Cao', 'Ko', 'Km_Ca', 'g_CaT', 'G_Kr', 'G_Ks', 'PNaK', 'G_K1', 'g_Kp', 'g_K_Na', 'nKNa', 'kdKNa', 'i_K_ATP_on', 'nATP', 'nicholsarea', 'ATPi', 'hATP', 'kATP', 'K_mpCa', 'I_pCa', 'g_Nab', 'g_Cab', 'I_NaK', 'K_mNai', 'K_mKo', 'K_m_ns_Ca', 'c1', 'c2', 'gamma', 'G_rel_max', 'G_rel_overload', 'tau_tr', 'K_mrel', 'delta_Ca_ith', 'CSQN_max', 'K_mCSQN', 'K_mup', 'I_up', 'Ca_NSR_max', 'A_cap', 'K_mTn', 'K_mCMDN', 'Tn_max', 'CMDN_max', 'CSQNthresh', 'Logicthresh', 'preplength', 'radius', 'g_to', 'P_ns_Ca']
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
        gamma_Cai = theta[:, 15]  # constants[15]
        gamma_Cao = theta[:, 16]  # constants[16]
        P_Ca = theta[:, 17]  # constants[17]
        P_Na = theta[:, 18]  # constants[18]
        P_K = theta[:, 19]  # constants[19]
        Cao = theta[:, 20]  # constants[20]
        Ko = theta[:, 21]  # constants[21]
        Km_Ca = theta[:, 22]  # constants[22]
        g_CaT = theta[:, 23]  # constants[23]
        G_Kr = theta[:, 24]  # constants[24]
        G_Ks = theta[:, 25]  # constants[25]
        PNaK = theta[:, 26]  # constants[26]
        G_K1 = theta[:, 27]  # constants[27]
        g_Kp = theta[:, 28]  # constants[28]
        g_K_Na = theta[:, 29]  # constants[29]
        nKNa = theta[:, 30]  # constants[30]
        kdKNa = theta[:, 31]  # constants[31]
        i_K_ATP_on = theta[:, 32]  # constants[32]
        nATP = theta[:, 33]  # constants[33]
        nicholsarea = theta[:, 34]  # constants[34]
        ATPi = theta[:, 35]  # constants[35]
        hATP = theta[:, 36]  # constants[36]
        kATP = theta[:, 37]  # constants[37]
        K_mpCa = theta[:, 38]  # constants[38]
        I_pCa = theta[:, 39]  # constants[39]
        g_Nab = theta[:, 40]  # constants[40]
        g_Cab = theta[:, 41]  # constants[41]
        I_NaK = theta[:, 42]  # constants[42]
        K_mNai = theta[:, 43]  # constants[43]
        K_mKo = theta[:, 44]  # constants[44]
        K_m_ns_Ca = theta[:, 45]  # constants[45]
        c1 = theta[:, 46]  # constants[46]
        c2 = theta[:, 47]  # constants[47]
        gamma = theta[:, 48]  # constants[48]
        G_rel_max = theta[:, 49]  # constants[49]
        G_rel_overload = theta[:, 50]  # constants[50]
        tau_tr = theta[:, 51]  # constants[51]
        K_mrel = theta[:, 52]  # constants[52]
        delta_Ca_ith = theta[:, 53]  # constants[53]
        CSQN_max = theta[:, 54]  # constants[54]
        K_mCSQN = theta[:, 55]  # constants[55]
        K_mup = theta[:, 56]  # constants[56]
        I_up = theta[:, 57]  # constants[57]
        Ca_NSR_max = theta[:, 58]  # constants[58]
        A_cap = theta[:, 59]  # constants[59]
        K_mTn = theta[:, 60]  # constants[60]
        K_mCMDN = theta[:, 61]  # constants[61]
        Tn_max = theta[:, 62]  # constants[62]
        CMDN_max = theta[:, 63]  # constants[63]
        CSQNthresh = theta[:, 64]  # constants[64]
        Logicthresh = theta[:, 65]  # constants[65]
        preplength = theta[:, 66]  # constants[66]
        radius = theta[:, 67]  # constants[67]
        g_to = theta[:, 68]  # constants[72]
        P_ns_Ca = theta[:, 69]  # constants[74]

        # Derived constants from the original initConsts()
        volume = math.pi * preplength * radius ** 2.0  # constants[68]
        g_Kr = G_Kr * (Ko / 5.4) ** (1.0 / 2)  # constants[69]
        g_K1 = G_K1 * (Ko / 5.4) ** (1.0 / 2)  # constants[70]
        g_K_ATP = i_K_ATP_on * 0.000193 / nicholsarea  # constants[71]
        sigma = 1.0 / 7.0 * (torch.exp(Nao / 67.3) - 1.0)  # constants[73]
        K_leak = I_up / Ca_NSR_max  # constants[75]
        V_myo = 0.68 * volume  # constants[76]
        pATP = 1.0 / (1.0 + (ATPi / kATP) ** hATP)  # constants[77]
        V_JSR = 0.0048 * volume  # constants[78]
        GKbaraATP = g_K_ATP * pATP * (Ko / 4.0) ** nATP  # constants[79]
        V_NSR = 0.0552 * volume  # constants[80]

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
        alg_3_I_st = torch.zeros_like(y[:, 0])
        alg_4_E0_d = torch.zeros_like(y[:, 0])
        alg_5_f_infinity = torch.zeros_like(y[:, 0])
        alg_6_b_inf = torch.zeros_like(y[:, 0])
        alg_7_g_inf = torch.zeros_like(y[:, 0])
        alg_8_xr_infinity = torch.zeros_like(y[:, 0])
        alg_9_xs1_infinity = torch.zeros_like(y[:, 0])
        alg_10_xs2_infinity = torch.zeros_like(y[:, 0])
        alg_11_alpha_zdv = torch.zeros_like(y[:, 0])
        alg_12_alpha_ydv = torch.zeros_like(y[:, 0])
        alg_14_alpha_m = torch.zeros_like(y[:, 0])
        alg_15_beta_h = torch.zeros_like(y[:, 0])
        alg_16_beta_j = torch.zeros_like(y[:, 0])
        alg_17_E_Na = torch.zeros_like(y[:, 0])
        alg_18_d_infinity = torch.zeros_like(y[:, 0])
        alg_19_tau_f = torch.zeros_like(y[:, 0])
        alg_20_tau_b = torch.zeros_like(y[:, 0])
        alg_21_tau_g = torch.zeros_like(y[:, 0])
        alg_22_tau_xr = torch.zeros_like(y[:, 0])
        alg_23_tau_xs1 = torch.zeros_like(y[:, 0])
        alg_24_tau_xs2 = torch.zeros_like(y[:, 0])
        alg_25_beta_zdv = torch.zeros_like(y[:, 0])
        alg_26_beta_ydv = torch.zeros_like(y[:, 0])
        alg_27_beta_m = torch.zeros_like(y[:, 0])
        alg_28_i_Na = torch.zeros_like(y[:, 0])
        alg_29_tau_d = torch.zeros_like(y[:, 0])
        alg_30_alpha_f = torch.zeros_like(y[:, 0])
        alg_31_tau_zdv = torch.zeros_like(y[:, 0])
        alg_32_tau_ydv = torch.zeros_like(y[:, 0])
        alg_33_I_CaCa = torch.zeros_like(y[:, 0])
        alg_34_alpha_d = torch.zeros_like(y[:, 0])
        alg_35_beta_f = torch.zeros_like(y[:, 0])
        alg_36_zdv_ss = torch.zeros_like(y[:, 0])
        alg_37_ydv_ss = torch.zeros_like(y[:, 0])
        alg_38_I_CaNa = torch.zeros_like(y[:, 0])
        alg_39_beta_d = torch.zeros_like(y[:, 0])
        alg_40_I_CaK = torch.zeros_like(y[:, 0])
        alg_41_f_Ca = torch.zeros_like(y[:, 0])
        alg_42_i_CaCa = torch.zeros_like(y[:, 0])
        alg_43_i_CaNa = torch.zeros_like(y[:, 0])
        alg_44_i_CaK = torch.zeros_like(y[:, 0])
        alg_45_i_Ca_L = torch.zeros_like(y[:, 0])
        alg_46_Rect = torch.zeros_like(y[:, 0])
        alg_47_E_Ks = torch.zeros_like(y[:, 0])
        alg_48_g_Ks = torch.zeros_like(y[:, 0])
        alg_49_i_Ks = torch.zeros_like(y[:, 0])
        alg_50_E_K = torch.zeros_like(y[:, 0])
        alg_51_i_Kr = torch.zeros_like(y[:, 0])
        alg_52_alpha_K1 = torch.zeros_like(y[:, 0])
        alg_53_beta_K1 = torch.zeros_like(y[:, 0])
        alg_54_K1_infinity = torch.zeros_like(y[:, 0])
        alg_55_i_K1 = torch.zeros_like(y[:, 0])
        alg_56_Kp = torch.zeros_like(y[:, 0])
        alg_57_i_Kp = torch.zeros_like(y[:, 0])
        alg_58_pona = torch.zeros_like(y[:, 0])
        alg_59_pov = torch.zeros_like(y[:, 0])
        alg_60_i_K_Na = torch.zeros_like(y[:, 0])
        alg_61_i_K_ATP = torch.zeros_like(y[:, 0])
        alg_62_rvdv = torch.zeros_like(y[:, 0])
        alg_63_i_to = torch.zeros_like(y[:, 0])
        alg_64_i_p_Ca = torch.zeros_like(y[:, 0])
        alg_65_i_Na_b = torch.zeros_like(y[:, 0])
        alg_66_E_Ca = torch.zeros_like(y[:, 0])
        alg_67_i_Ca_T = torch.zeros_like(y[:, 0])
        alg_68_i_Ca_b = torch.zeros_like(y[:, 0])
        alg_69_f_NaK = torch.zeros_like(y[:, 0])
        alg_70_i_NaK = torch.zeros_like(y[:, 0])
        alg_71_I_ns_Na = torch.zeros_like(y[:, 0])
        alg_72_I_ns_K = torch.zeros_like(y[:, 0])
        alg_73_i_ns_Na = torch.zeros_like(y[:, 0])
        alg_74_i_ns_K = torch.zeros_like(y[:, 0])
        alg_75_i_ns_Ca = torch.zeros_like(y[:, 0])
        alg_76_i_NaCa = torch.zeros_like(y[:, 0])
        alg_77_dVdt = torch.zeros_like(y[:, 0])
        alg_78_RyRopen = torch.zeros_like(y[:, 0])
        alg_79_RyRclose = torch.zeros_like(y[:, 0])
        alg_80_G_rel = torch.zeros_like(y[:, 0])
        alg_81_i_rel = torch.zeros_like(y[:, 0])
        alg_82_i_up = torch.zeros_like(y[:, 0])
        alg_83_i_leak = torch.zeros_like(y[:, 0])
        alg_84_i_tr = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_APtrack2 = torch.zeros_like(y[:, 0]) + (torch.where((APtrack < 0.2) & (APtrack > 0.18), 100.0 * (1.0 - APtrack2) - 0.5 * APtrack2, -0.5 * APtrack2))
        d_APtrack3 = torch.zeros_like(y[:, 0]) + (torch.where((APtrack < 0.2) & (APtrack > 0.18), 100.0 * (1.0 - APtrack3) - 0.5 * APtrack3, -0.01 * APtrack3))
        d_OVRLDtrack = torch.zeros_like(y[:, 0]) + (torch.where((1.0 / (1.0 + K_mCSQN / Ca_JSR) > CSQNthresh) & (OVRLDtrack3 < 0.37) & (APtrack3 < 0.37), 50.0 * (1.0 - OVRLDtrack), -0.5 * OVRLDtrack))
        d_OVRLDtrack2 = torch.zeros_like(y[:, 0]) + (torch.where((OVRLDtrack > Logicthresh) & (OVRLDtrack2 < Logicthresh), 50.0 * (1.0 - OVRLDtrack2), -0.5 * OVRLDtrack2))
        d_OVRLDtrack3 = torch.zeros_like(y[:, 0]) + (torch.where((OVRLDtrack > Logicthresh) & (OVRLDtrack3 < Logicthresh), 50.0 * (1.0 - OVRLDtrack3), -0.01 * OVRLDtrack3))
        alg_1_alpha_h = torch.where(V < -40.0, 0.135 * torch.exp((80.0 + V) / -6.8), 0.0)
        alg_15_beta_h = torch.where(V < -40.0, 3.56 * torch.exp(0.079 * V) + 310000.0 * torch.exp(0.35 * V), 1.0 / (0.13 * (1.0 + torch.exp((V + 10.66) / -11.1))))
        d_h = torch.zeros_like(y[:, 0]) + (alg_1_alpha_h * (1.0 - h) - alg_15_beta_h * h)
        alg_2_alpha_j = torch.where(V < -40.0, -(127140.0 * torch.exp(0.2444 * V) + 3.474e-05 * torch.exp(-0.04391 * V)) * (V + 37.78) / (1.0 + torch.exp(0.311 * (V + 79.23))), 0.0)
        alg_16_beta_j = torch.where(V < -40.0, 0.1212 * torch.exp(-0.01052 * V) / (1.0 + torch.exp(-0.1378 * (V + 40.14))), 0.3 * torch.exp(-2.535e-07 * V) / (1.0 + torch.exp(-0.1 * (V + 32.0))))
        d_j = torch.zeros_like(y[:, 0]) + (alg_2_alpha_j * (1.0 - j) - alg_16_beta_j * j)
        alg_6_b_inf = 1.0 / (1.0 + torch.exp(-(V + 14.0) / 10.8))
        alg_20_tau_b = 3.7 + 6.1 / (1.0 + torch.exp((V + 25.0) / 4.5))
        d_b = torch.zeros_like(y[:, 0]) + ((alg_6_b_inf - b) / alg_20_tau_b)
        alg_7_g_inf = 1.0 / (1.0 + torch.exp((V + 60.0) / 5.6))
        alg_21_tau_g = torch.where(V <= 0.0, -0.875 * V + 12.0, 12.0)
        d_g = torch.zeros_like(y[:, 0]) + ((alg_7_g_inf - g) / alg_21_tau_g)
        alg_8_xr_infinity = 1.0 / (1.0 + torch.exp(-(V + 21.5) / 7.5))
        alg_22_tau_xr = 1.0 / (0.00138 * (V + 14.2) / (1.0 - torch.exp(-0.123 * (V + 14.2))) + 0.00061 * (V + 38.9) / (torch.exp(0.145 * (V + 38.9)) - 1.0))
        d_xr = torch.zeros_like(y[:, 0]) + ((alg_8_xr_infinity - xr) / alg_22_tau_xr)
        alg_9_xs1_infinity = 1.0 / (1.0 + torch.exp(-(V - 1.5) / 16.7))
        alg_23_tau_xs1 = 1.0 / (7.19e-05 * (V + 30.0) / (1.0 - torch.exp(-0.148 * (V + 30.0))) + 0.000131 * (V + 30.0) / (torch.exp(0.0687 * (V + 30.0)) - 1.0))
        d_xs1 = torch.zeros_like(y[:, 0]) + ((alg_9_xs1_infinity - xs1) / alg_23_tau_xs1)
        alg_10_xs2_infinity = 1.0 / (1.0 + torch.exp(-(V - 1.5) / 16.7))
        alg_24_tau_xs2 = 4.0 / (7.19e-05 * (V + 30.0) / (1.0 - torch.exp(-0.148 * (V + 30.0))) + 0.000131 * (V + 30.0) / (torch.exp(0.0687 * (V + 30.0)) - 1.0))
        d_xs2 = torch.zeros_like(y[:, 0]) + ((alg_10_xs2_infinity - xs2) / alg_24_tau_xs2)
        alg_0_E0_m = V + 47.13
        alg_14_alpha_m = torch.where(torch.abs(alg_0_E0_m) >= delta_m, 0.32 * alg_0_E0_m / (1.0 - torch.exp(-0.1 * alg_0_E0_m)), 3.2)
        alg_27_beta_m = 0.08 * torch.exp(-V / 11.0)
        d_m = torch.zeros_like(y[:, 0]) + (alg_14_alpha_m * (1.0 - m) - alg_27_beta_m * m)
        alg_5_f_infinity = 1.0 / (1.0 + torch.exp((V + 35.06) / 8.6)) + 0.6 / (1.0 + torch.exp((50.0 - V) / 20.0))
        alg_19_tau_f = 1.0 / (0.0197 * torch.exp(-(0.0337 * (V + 10.0)) ** 2.0) + 0.02)
        alg_30_alpha_f = alg_5_f_infinity / alg_19_tau_f
        alg_35_beta_f = (1.0 - alg_5_f_infinity) / alg_19_tau_f
        d_f = torch.zeros_like(y[:, 0]) + (alg_30_alpha_f * (1.0 - f) - alg_35_beta_f * f)
        alg_11_alpha_zdv = 10.0 * torch.exp((V - 40.0) / 25.0) / (1.0 + torch.exp((V - 40.0) / 25.0))
        alg_25_beta_zdv = 10.0 * torch.exp(-(V + 90.0) / 25.0) / (1.0 + torch.exp(-(V + 90.0) / 25.0))
        alg_31_tau_zdv = 1.0 / (alg_11_alpha_zdv + alg_25_beta_zdv)
        alg_36_zdv_ss = alg_11_alpha_zdv / (alg_11_alpha_zdv + alg_25_beta_zdv)
        d_zdv = torch.zeros_like(y[:, 0]) + ((alg_36_zdv_ss - zdv) / alg_31_tau_zdv)
        alg_12_alpha_ydv = 0.015 / (1.0 + torch.exp((V + 60.0) / 5.0))
        alg_26_beta_ydv = 0.1 * torch.exp((V + 25.0) / 5.0) / (1.0 + torch.exp((V + 25.0) / 5.0))
        alg_32_tau_ydv = 1.0 / (alg_12_alpha_ydv + alg_26_beta_ydv)
        alg_37_ydv_ss = alg_12_alpha_ydv / (alg_12_alpha_ydv + alg_26_beta_ydv)
        d_ydv = torch.zeros_like(y[:, 0]) + ((alg_37_ydv_ss - ydv) / alg_32_tau_ydv)
        alg_4_E0_d = V + 10.0
        alg_18_d_infinity = 1.0 / (1.0 + torch.exp(-alg_4_E0_d / 6.24))
        alg_29_tau_d = torch.where(torch.abs(alg_4_E0_d) < 1e-05, 1.0 / (0.035 * 6.24 * 2.0), 1.0 * alg_18_d_infinity * (1.0 - torch.exp(-alg_4_E0_d / 6.24)) / (0.035 * alg_4_E0_d))
        alg_34_alpha_d = alg_18_d_infinity / alg_29_tau_d
        alg_39_beta_d = (1.0 - alg_18_d_infinity) / alg_29_tau_d
        d_d = torch.zeros_like(y[:, 0]) + (alg_34_alpha_d * (1.0 - d) - alg_39_beta_d * d)
        alg_3_I_st = torch.where((t >= stim_start) & (t <= stim_end) & (t - stim_start - torch.floor((t - stim_start) / stim_period) * stim_period <= stim_duration), stim_amplitude, 0.0)
        alg_46_Rect = 1.0 / (1.0 + torch.exp((V + 9.0) / 22.4))
        alg_50_E_K = R * T / F * torch.log(Ko / Ki)
        alg_51_i_Kr = g_Kr * xr * alg_46_Rect * (V - alg_50_E_K)
        alg_48_g_Ks = G_Ks * (1.0 + 0.6 / (1.0 + (3.8e-05 / Cai) ** 1.4))
        alg_47_E_Ks = R * T / F * torch.log((Ko + PNaK * Nao) / (Ki + PNaK * Nai))
        alg_49_i_Ks = alg_48_g_Ks * xs1 * xs2 * (V - alg_47_E_Ks)
        alg_58_pona = 0.85 / (1.0 + (kdKNa / Nai) ** nKNa)
        alg_59_pov = 0.8 - 0.65 / (1.0 + torch.exp((V + 125.0) / 15.0))
        alg_60_i_K_Na = g_K_Na * alg_58_pona * alg_59_pov * (V - alg_50_E_K)
        alg_61_i_K_ATP = GKbaraATP * (V - alg_50_E_K)
        alg_62_rvdv = torch.exp(V / 100.0)
        alg_63_i_to = g_to * zdv ** 3.0 * ydv * alg_62_rvdv * (V - alg_50_E_K)
        alg_52_alpha_K1 = 1.02 / (1.0 + torch.exp(0.2385 * (V - alg_50_E_K - 59.215)))
        alg_53_beta_K1 = 1.0 * (0.49124 * torch.exp(0.08032 * (V - alg_50_E_K + 5.476)) + torch.exp(0.06175 * (V - alg_50_E_K - 594.31))) / (1.0 + torch.exp(-0.5143 * (V - alg_50_E_K + 4.753)))
        alg_54_K1_infinity = alg_52_alpha_K1 / (alg_52_alpha_K1 + alg_53_beta_K1)
        alg_55_i_K1 = g_K1 * alg_54_K1_infinity * (V - alg_50_E_K)
        alg_56_Kp = 1.0 / (1.0 + torch.exp((7.488 - V) / 5.98))
        alg_57_i_Kp = g_Kp * alg_56_Kp * (V - alg_50_E_K)
        alg_69_f_NaK = 1.0 / (1.0 + 0.1245 * torch.exp(-0.1 * V * F / (R * T)) + 0.0365 * sigma * torch.exp(-V * F / (R * T)))
        alg_70_i_NaK = I_NaK * alg_69_f_NaK * 1.0 / (1.0 + (K_mNai / Nai) ** 2.0) * Ko / (Ko + K_mKo)
        alg_40_I_CaK = P_K * 1.0 * V * F ** 2.0 / (R * T) * (gamma_Ki * Ki * torch.exp(1.0 * V * F / (R * T)) - gamma_Ko * Ko) / (torch.exp(1.0 * V * F / (R * T)) - 1.0)
        alg_41_f_Ca = 1.0 / (1.0 + Cai / Km_Ca)
        alg_44_i_CaK = d * f * alg_41_f_Ca * alg_40_I_CaK
        alg_72_I_ns_K = P_ns_Ca * 1.0 * V * F ** 2.0 / (R * T) * (gamma_Ki * Ki * torch.exp(1.0 * V * F / (R * T)) - gamma_Ko * Ko) / (torch.exp(1.0 * V * F / (R * T)) - 1.0)
        alg_74_i_ns_K = alg_72_I_ns_K * 1.0 / (1.0 + (K_m_ns_Ca / Cai) ** 3.0)
        d_Ki = torch.zeros_like(y[:, 0]) + (-1.0 * (alg_3_I_st + alg_44_i_CaK + alg_51_i_Kr + alg_49_i_Ks + alg_55_i_K1 + alg_57_i_Kp + alg_60_i_K_Na + alg_61_i_K_ATP + alg_63_i_to + alg_74_i_ns_K + -alg_70_i_NaK * 2.0) * A_cap / (V_myo * F))
        alg_66_E_Ca = R * T / (2.0 * F) * torch.log(Cao / Cai)
        alg_67_i_Ca_T = g_CaT * b * b * g * (V - alg_66_E_Ca)
        alg_76_i_NaCa = c1 * torch.exp((gamma - 1.0) * V * F / (R * T)) * (torch.exp(V * F / (R * T)) * Nai ** 3.0 * Cao - Nao ** 3.0 * Cai) / (1.0 + c2 * torch.exp((gamma - 1.0) * V * F / (R * T)) * (torch.exp(V * F / (R * T)) * Nai ** 3.0 * Cao + Nao ** 3.0 * Cai))
        alg_64_i_p_Ca = I_pCa * Cai / (K_mpCa + Cai)
        alg_68_i_Ca_b = g_Cab * (V - alg_66_E_Ca)
        alg_33_I_CaCa = P_Ca * 4.0 * V * F ** 2.0 / (R * T) * (gamma_Cai * Cai * torch.exp(2.0 * V * F / (R * T)) - gamma_Cao * Cao) / (torch.exp(2.0 * V * F / (R * T)) - 1.0)
        alg_42_i_CaCa = d * f * alg_41_f_Ca * alg_33_I_CaCa
        d_Cainfluxtrack = torch.zeros_like(y[:, 0]) + (torch.where(APtrack > 0.2, -1.0 * A_cap * (alg_42_i_CaCa + alg_67_i_Ca_T - 2.0 * alg_76_i_NaCa + alg_64_i_p_Ca + alg_68_i_Ca_b) / (2.0 * V_myo * F), torch.where((APtrack2 > 0.01) & (APtrack <= 0.2), 0.0, -0.5 * Cainfluxtrack)))
        alg_17_E_Na = R * T / F * torch.log(Nao / Nai)
        alg_28_i_Na = g_Na * m ** 3.0 * h * j * (V - alg_17_E_Na)
        alg_65_i_Na_b = g_Nab * (V - alg_17_E_Na)
        alg_38_I_CaNa = P_Na * 1.0 * V * F ** 2.0 / (R * T) * (gamma_Nai * Nai * torch.exp(1.0 * V * F / (R * T)) - gamma_Nao * Nao) / (torch.exp(1.0 * V * F / (R * T)) - 1.0)
        alg_43_i_CaNa = d * f * alg_41_f_Ca * alg_38_I_CaNa
        alg_71_I_ns_Na = P_ns_Ca * 1.0 * V * F ** 2.0 / (R * T) * (gamma_Nai * Nai * torch.exp(1.0 * V * F / (R * T)) - gamma_Nao * Nao) / (torch.exp(1.0 * V * F / (R * T)) - 1.0)
        alg_73_i_ns_Na = alg_71_I_ns_Na * 1.0 / (1.0 + (K_m_ns_Ca / Cai) ** 3.0)
        d_Nai = torch.zeros_like(y[:, 0]) + (-1.0 * (alg_28_i_Na + alg_43_i_CaNa + alg_65_i_Na_b + alg_73_i_ns_Na + alg_76_i_NaCa * 3.0 + alg_70_i_NaK * 3.0) * A_cap / (V_myo * F))
        alg_45_i_Ca_L = alg_42_i_CaCa + alg_44_i_CaK + alg_43_i_CaNa
        alg_75_i_ns_Ca = alg_73_i_ns_Na + alg_74_i_ns_K
        alg_77_dVdt = -(alg_28_i_Na + alg_45_i_Ca_L + alg_67_i_Ca_T + alg_51_i_Kr + alg_49_i_Ks + alg_60_i_K_Na + alg_61_i_K_ATP + alg_63_i_to + alg_55_i_K1 + alg_57_i_Kp + alg_76_i_NaCa + alg_64_i_p_Ca + alg_65_i_Na_b + alg_68_i_Ca_b + alg_70_i_NaK + alg_75_i_ns_Ca + alg_3_I_st)
        d_V = torch.zeros_like(y[:, 0]) + (alg_77_dVdt)
        d_APtrack = torch.zeros_like(y[:, 0]) + (torch.where(alg_77_dVdt > 150.0, 100.0 * (1.0 - APtrack) - 0.5 * APtrack, -0.5 * APtrack))
        alg_78_RyRopen = 1.0 / (1.0 + 7.38905609893065 * APtrack2)
        alg_79_RyRclose = 1.0 - alg_78_RyRopen
        alg_80_G_rel = alg_78_RyRopen * alg_79_RyRclose * 150.0 / (1.0 + torch.exp((alg_42_i_CaCa + alg_67_i_Ca_T - 2.0 * alg_76_i_NaCa + alg_64_i_p_Ca + alg_68_i_Ca_b + 5.0) / 0.9))
        alg_81_i_rel = alg_80_G_rel * (Ca_JSR - Cai)
        alg_82_i_up = I_up * Cai / (Cai + K_mup)
        alg_83_i_leak = K_leak * Ca_NSR
        d_Cai = torch.zeros_like(y[:, 0]) + (1.0 / (1.0 + CMDN_max * K_mCMDN / (K_mCMDN + Cai) ** 2.0 + Tn_max * K_mTn / (K_mTn + Cai) ** 2.0) * (-1.0 * A_cap * (alg_42_i_CaCa + alg_67_i_Ca_T - 2.0 * alg_76_i_NaCa + alg_64_i_p_Ca + alg_68_i_Ca_b) / (2.0 * V_myo * F) + alg_81_i_rel * V_JSR / V_myo + (alg_83_i_leak - alg_82_i_up) * V_NSR / V_myo))
        alg_84_i_tr = (Ca_NSR - Ca_JSR) / tau_tr
        d_Ca_JSR = torch.zeros_like(y[:, 0]) + (1.0 / (1.0 + CSQN_max * K_mCSQN / (K_mCSQN + Ca_JSR) ** 2.0) * (alg_84_i_tr - alg_81_i_rel))
        d_Ca_NSR = torch.zeros_like(y[:, 0]) + (-alg_84_i_tr * V_JSR / V_NSR - alg_83_i_leak + alg_82_i_up)

        return torch.stack(
            [d_V, d_Nai, d_m, d_h, d_j, d_Cai, d_Ki, d_d, d_f, d_b, d_g, d_xr, d_xs1, d_xs2, d_zdv, d_ydv, d_Ca_JSR, d_Ca_NSR, d_APtrack, d_APtrack2, d_APtrack3, d_Cainfluxtrack, d_OVRLDtrack, d_OVRLDtrack2, d_OVRLDtrack3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
