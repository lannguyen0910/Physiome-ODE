import torch
import torch.nn as nn
import math


class ElectrophysiologyAslanidi2009ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/aslanidi_2009/model.py.
    Dynamic states: 30; independent constants: 74; derived constants used by rates: 9.
    Original constant slots: 83; algebraic slots: 115.
    """
    source_file = 'electrophysiology/aslanidi_2009/model.py'
    source_state_dim = 30
    source_constant_dim = 83
    algebraic_dim = 115
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 30
        self.parameter_dim = 74

        self.parameters = (
            parameters
            if parameters is not None
            else {'stim_start': 0, 'stim_end': 1, 'stim_amplitude': -80, 'r_NaK': 0.01833, 'Na_o': 140, 'Ca_o': 1.8, 'K_o': 5.4, 'Cl_o': 100, 'R': 8314, 'F': 96485, 'T': 310, 'g_Na': 8, 'g_Na_L': 0.037375, 'g_Ca_L': 0.3392328, 'p_Ca': 0.000243, 'z_Ca': 2, 'gamma_Cai': 1, 'gamma_Cao': 0.341, 'km_Ca_MK': 0.15, 'Cm': 1, 'g_Ca_T': 0.13, 'g_to_1': 0.14135944, 'g_K_p': 0.00276, 'p_Cl': 4e-07, 'z_Cl': -1, 'km_to_2': 0.1502, 'X_NaCa': 0.4, 'i_NaCa_max': 4.5, 'km_Na_i_1': 12.3, 'km_Na_o': 87.5, 'km_Ca_i': 0.0036, 'km_Ca_o': 1.3, 'km_Ca_act': 0.000125, 'k_sat': 0.27, 'g_NaK': 0.61875, 'km_Na_i_2': 10, 'km_K_o': 1.5, 'i_Ca_p_max': 0.0575, 'km_Ca_p': 0.0005, 'CT_K_Cl_max': 7.0756e-06, 'CT_Na_Cl_max': 9.8443e-06, 'g_Na_b': 0.0025, 'g_K_b': 0.005, 'p_Ca_b': 1.995084e-07, 'g_Cl_b': 0.000225, 'km_TRPN': 0.0005, 'km_CMDN': 0.00238, 'TRPN_max': 0.07, 'CMDN_max': 0.05, 'alpha_Ca_MK': 0.05, 'beta_Ca_MK': 0.00068, 'Ca_MK_0': 0.05, 'CSQN_max': 10, 'km_CSQN': 0.8, 'km_b_SR': 0.00087, 'km_b_SL': 0.0087, 'b_SR_max': 0.047, 'b_SL_max': 1.124, 'tau_ss': 0.2, 'tau_Ca_MK_max': 10, 'tau_ro': 3, 'q_leak_max': 0.004375, 'NSR_max': 15, 'X_q_up': 0.5, 'q_up_max': 0.004375, 'dq_up_Ca_MK_max': 0.75, 'dkm_plb_max': 0.00017, 'km_up': 0.00092, 'tau_tr': 120, 'Vol_cell': 0.3454, 'radius': 0.0011, 'length': 0.01, 'rcg': 2, 'tau_a': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -83.43812846286808, 'Na_i': 9.927155552932733, 'Ca_i': 0.00022355433459434943, 'K_i': 141.9670801746057, 'Cl_i': 18.904682470140408, 'm': 0.002003390432234504, 'h': 0.9786390933308567, 'j': 0.09866447258167589, 'm_L': 0.002003390432234504, 'h_L': 0.8946968372659203, 'Ca_r': 0.00022418117117903934, 'd': 2.322223865147363e-06, 'f': 0.9985607329462358, 'f2': 0.8173435436674658, 'f_Ca': 0.9610551285529658, 'f_Ca2': 0.868690796671854, 'b': 0.0002563937630984438, 'g': 0.9720432601848331, 'a': 0.0004238729429342389, 'i': 0.9990935802459496, 'i2': 0.9777368439681764, 'xr': 0.07084939408222911, 'xs1': 0.0011737654433043125, 'xs2': 0.001179442867470093, 'a_2': 0.0014909437525000811, 'Ca_MK_trap': 8.789168284782809e-06, 'Ca_NSR': 1.2132524695849454, 'Ca_JSR': 1.1433050636518596, 'ri': 0.7802870066567904, 'ro': 1.2785734760674763e-09}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['stim_start', 'stim_end', 'stim_amplitude', 'r_NaK', 'Na_o', 'Ca_o', 'K_o', 'Cl_o', 'R', 'F', 'T', 'g_Na', 'g_Na_L', 'g_Ca_L', 'p_Ca', 'z_Ca', 'gamma_Cai', 'gamma_Cao', 'km_Ca_MK', 'Cm', 'g_Ca_T', 'g_to_1', 'g_K_p', 'p_Cl', 'z_Cl', 'km_to_2', 'X_NaCa', 'i_NaCa_max', 'km_Na_i_1', 'km_Na_o', 'km_Ca_i', 'km_Ca_o', 'km_Ca_act', 'k_sat', 'g_NaK', 'km_Na_i_2', 'km_K_o', 'i_Ca_p_max', 'km_Ca_p', 'CT_K_Cl_max', 'CT_Na_Cl_max', 'g_Na_b', 'g_K_b', 'p_Ca_b', 'g_Cl_b', 'km_TRPN', 'km_CMDN', 'TRPN_max', 'CMDN_max', 'alpha_Ca_MK', 'beta_Ca_MK', 'Ca_MK_0', 'CSQN_max', 'km_CSQN', 'km_b_SR', 'km_b_SL', 'b_SR_max', 'b_SL_max', 'tau_ss', 'tau_Ca_MK_max', 'tau_ro', 'q_leak_max', 'NSR_max', 'X_q_up', 'q_up_max', 'dq_up_Ca_MK_max', 'dkm_plb_max', 'km_up', 'tau_tr', 'Vol_cell', 'radius', 'length', 'rcg', 'tau_a']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'Na_i', 'Ca_i', 'K_i', 'Cl_i', 'm', 'h', 'j', 'm_L', 'h_L', 'Ca_r', 'd', 'f', 'f2', 'f_Ca', 'f_Ca2', 'b', 'g', 'a', 'i', 'i2', 'xr', 'xs1', 'xs2', 'a_2', 'Ca_MK_trap', 'Ca_NSR', 'Ca_JSR', 'ri', 'ro']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (30)
        V = y[:, 0]
        Na_i = y[:, 1]
        Ca_i = y[:, 2]
        K_i = y[:, 3]
        Cl_i = y[:, 4]
        m = y[:, 5]
        h = y[:, 6]
        j = y[:, 7]
        m_L = y[:, 8]
        h_L = y[:, 9]
        Ca_r = y[:, 10]
        d = y[:, 11]
        f = y[:, 12]
        f2 = y[:, 13]
        f_Ca = y[:, 14]
        f_Ca2 = y[:, 15]
        b = y[:, 16]
        g = y[:, 17]
        a = y[:, 18]
        i = y[:, 19]
        i2 = y[:, 20]
        xr = y[:, 21]
        xs1 = y[:, 22]
        xs2 = y[:, 23]
        a_2 = y[:, 24]
        Ca_MK_trap = y[:, 25]
        Ca_NSR = y[:, 26]
        Ca_JSR = y[:, 27]
        ri = y[:, 28]
        ro = y[:, 29]

        # Independent parameters (source constant order)
        stim_start = theta[:, 0]  # constants[0]
        stim_end = theta[:, 1]  # constants[1]
        stim_amplitude = theta[:, 2]  # constants[2]
        r_NaK = theta[:, 3]  # constants[3]
        Na_o = theta[:, 4]  # constants[4]
        Ca_o = theta[:, 5]  # constants[5]
        K_o = theta[:, 6]  # constants[6]
        Cl_o = theta[:, 7]  # constants[7]
        R = theta[:, 8]  # constants[8]
        F = theta[:, 9]  # constants[9]
        T = theta[:, 10]  # constants[10]
        g_Na = theta[:, 11]  # constants[11]
        g_Na_L = theta[:, 12]  # constants[12]
        g_Ca_L = theta[:, 13]  # constants[13]
        p_Ca = theta[:, 14]  # constants[14]
        z_Ca = theta[:, 15]  # constants[15]
        gamma_Cai = theta[:, 16]  # constants[16]
        gamma_Cao = theta[:, 17]  # constants[17]
        km_Ca_MK = theta[:, 18]  # constants[18]
        Cm = theta[:, 19]  # constants[19]
        g_Ca_T = theta[:, 20]  # constants[20]
        g_to_1 = theta[:, 21]  # constants[21]
        g_K_p = theta[:, 22]  # constants[22]
        p_Cl = theta[:, 23]  # constants[23]
        z_Cl = theta[:, 24]  # constants[24]
        km_to_2 = theta[:, 25]  # constants[25]
        X_NaCa = theta[:, 26]  # constants[26]
        i_NaCa_max = theta[:, 27]  # constants[27]
        km_Na_i_1 = theta[:, 28]  # constants[28]
        km_Na_o = theta[:, 29]  # constants[29]
        km_Ca_i = theta[:, 30]  # constants[30]
        km_Ca_o = theta[:, 31]  # constants[31]
        km_Ca_act = theta[:, 32]  # constants[32]
        k_sat = theta[:, 33]  # constants[33]
        g_NaK = theta[:, 34]  # constants[34]
        km_Na_i_2 = theta[:, 35]  # constants[35]
        km_K_o = theta[:, 36]  # constants[36]
        i_Ca_p_max = theta[:, 37]  # constants[37]
        km_Ca_p = theta[:, 38]  # constants[38]
        CT_K_Cl_max = theta[:, 39]  # constants[39]
        CT_Na_Cl_max = theta[:, 40]  # constants[40]
        g_Na_b = theta[:, 41]  # constants[41]
        g_K_b = theta[:, 42]  # constants[42]
        p_Ca_b = theta[:, 43]  # constants[43]
        g_Cl_b = theta[:, 44]  # constants[44]
        km_TRPN = theta[:, 45]  # constants[45]
        km_CMDN = theta[:, 46]  # constants[46]
        TRPN_max = theta[:, 47]  # constants[47]
        CMDN_max = theta[:, 48]  # constants[48]
        alpha_Ca_MK = theta[:, 49]  # constants[49]
        beta_Ca_MK = theta[:, 50]  # constants[50]
        Ca_MK_0 = theta[:, 51]  # constants[51]
        CSQN_max = theta[:, 52]  # constants[52]
        km_CSQN = theta[:, 53]  # constants[53]
        km_b_SR = theta[:, 54]  # constants[54]
        km_b_SL = theta[:, 55]  # constants[55]
        b_SR_max = theta[:, 56]  # constants[56]
        b_SL_max = theta[:, 57]  # constants[57]
        tau_ss = theta[:, 58]  # constants[58]
        tau_Ca_MK_max = theta[:, 59]  # constants[59]
        tau_ro = theta[:, 60]  # constants[60]
        q_leak_max = theta[:, 61]  # constants[61]
        NSR_max = theta[:, 62]  # constants[62]
        X_q_up = theta[:, 63]  # constants[63]
        q_up_max = theta[:, 64]  # constants[64]
        dq_up_Ca_MK_max = theta[:, 65]  # constants[65]
        dkm_plb_max = theta[:, 66]  # constants[66]
        km_up = theta[:, 67]  # constants[67]
        tau_tr = theta[:, 68]  # constants[68]
        Vol_cell = theta[:, 69]  # constants[69]
        radius = theta[:, 70]  # constants[70]
        length = theta[:, 71]  # constants[71]
        rcg = theta[:, 72]  # constants[72]
        tau_a = theta[:, 73]  # constants[76]

        # Derived constants from the original initConsts()
        a_geo = 2.0 * 3.14 * radius ** 2.0 + 2.0 * 3.14 * radius * length  # constants[73]
        g_Kr = 0.0400085 * (K_o / 5.4) ** (1.0 / 2)  # constants[74]
        g_K1 = 0.25 * (K_o / 5.4) ** (1.0 / 2)  # constants[75]
        sigma = 1.0 / 7.0 * (torch.exp(Na_o / 67.3) - 1.0)  # constants[77]
        a_cap = rcg * a_geo  # constants[78]
        Vol_myo = Vol_cell * 0.68  # constants[79]
        Vol_nsr = Vol_cell * 0.0552  # constants[80]
        Vol_jsr = Vol_cell * 0.0048  # constants[81]
        Vol_ss = Vol_cell * 0.02  # constants[82]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_Cl_i = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_j = torch.zeros_like(y[:, 0])
        d_m_L = torch.zeros_like(y[:, 0])
        d_h_L = torch.zeros_like(y[:, 0])
        d_Ca_r = torch.zeros_like(y[:, 0])
        d_d = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_f2 = torch.zeros_like(y[:, 0])
        d_f_Ca = torch.zeros_like(y[:, 0])
        d_f_Ca2 = torch.zeros_like(y[:, 0])
        d_b = torch.zeros_like(y[:, 0])
        d_g = torch.zeros_like(y[:, 0])
        d_a = torch.zeros_like(y[:, 0])
        d_i = torch.zeros_like(y[:, 0])
        d_i2 = torch.zeros_like(y[:, 0])
        d_xr = torch.zeros_like(y[:, 0])
        d_xs1 = torch.zeros_like(y[:, 0])
        d_xs2 = torch.zeros_like(y[:, 0])
        d_a_2 = torch.zeros_like(y[:, 0])
        d_Ca_MK_trap = torch.zeros_like(y[:, 0])
        d_Ca_NSR = torch.zeros_like(y[:, 0])
        d_Ca_JSR = torch.zeros_like(y[:, 0])
        d_ri = torch.zeros_like(y[:, 0])
        d_ro = torch.zeros_like(y[:, 0])
        alg_0_i_stim = torch.zeros_like(y[:, 0])
        alg_2_alpha_m = torch.zeros_like(y[:, 0])
        alg_3_alpha_h = torch.zeros_like(y[:, 0])
        alg_4_alpha_j = torch.zeros_like(y[:, 0])
        alg_5_alpha_m_L = torch.zeros_like(y[:, 0])
        alg_6_h_L_infinity = torch.zeros_like(y[:, 0])
        alg_7_d_infinity = torch.zeros_like(y[:, 0])
        alg_8_f_infinity = torch.zeros_like(y[:, 0])
        alg_9_f2_infinity = torch.zeros_like(y[:, 0])
        alg_10_alpha_b = torch.zeros_like(y[:, 0])
        alg_11_alpha_g = torch.zeros_like(y[:, 0])
        alg_12_alpha_a = torch.zeros_like(y[:, 0])
        alg_13_alpha_i = torch.zeros_like(y[:, 0])
        alg_14_alpha_i2 = torch.zeros_like(y[:, 0])
        alg_15_tau_xr = torch.zeros_like(y[:, 0])
        alg_16_tau_xs1 = torch.zeros_like(y[:, 0])
        alg_17_tau_xs2 = torch.zeros_like(y[:, 0])
        alg_18_a_infinity = torch.zeros_like(y[:, 0])
        alg_19_E_Na = torch.zeros_like(y[:, 0])
        alg_20_beta_m = torch.zeros_like(y[:, 0])
        alg_21_beta_h = torch.zeros_like(y[:, 0])
        alg_22_beta_j = torch.zeros_like(y[:, 0])
        alg_23_beta_m_L = torch.zeros_like(y[:, 0])
        alg_24_tau_h_L = torch.zeros_like(y[:, 0])
        alg_25_tau_d = torch.zeros_like(y[:, 0])
        alg_26_tau_f = torch.zeros_like(y[:, 0])
        alg_27_tau_f2 = torch.zeros_like(y[:, 0])
        alg_28_beta_b = torch.zeros_like(y[:, 0])
        alg_29_beta_g = torch.zeros_like(y[:, 0])
        alg_30_beta_a = torch.zeros_like(y[:, 0])
        alg_31_beta_i = torch.zeros_like(y[:, 0])
        alg_32_beta_i2 = torch.zeros_like(y[:, 0])
        alg_33_xr_infinity = torch.zeros_like(y[:, 0])
        alg_34_xs1_infinity = torch.zeros_like(y[:, 0])
        alg_35_xs2_infinity = torch.zeros_like(y[:, 0])
        alg_36_E_K = torch.zeros_like(y[:, 0])
        alg_37_m_infinity = torch.zeros_like(y[:, 0])
        alg_38_h_infinity = torch.zeros_like(y[:, 0])
        alg_39_j_infinity = torch.zeros_like(y[:, 0])
        alg_40_m_L_infinity = torch.zeros_like(y[:, 0])
        alg_41_b_infinity = torch.zeros_like(y[:, 0])
        alg_42_g_infinity = torch.zeros_like(y[:, 0])
        alg_43_tau_a = torch.zeros_like(y[:, 0])
        alg_44_tau_i = torch.zeros_like(y[:, 0])
        alg_45_tau_i2 = torch.zeros_like(y[:, 0])
        alg_46_E_Cl = torch.zeros_like(y[:, 0])
        alg_47_tau_m = torch.zeros_like(y[:, 0])
        alg_48_tau_h = torch.zeros_like(y[:, 0])
        alg_49_tau_j = torch.zeros_like(y[:, 0])
        alg_50_tau_m_L = torch.zeros_like(y[:, 0])
        alg_51_tau_b = torch.zeros_like(y[:, 0])
        alg_52_tau_g = torch.zeros_like(y[:, 0])
        alg_53_a_infinity_2 = torch.zeros_like(y[:, 0])
        alg_54_i_infinity = torch.zeros_like(y[:, 0])
        alg_55_i2_infinity = torch.zeros_like(y[:, 0])
        alg_56_E_Ks = torch.zeros_like(y[:, 0])
        alg_57_i_Na = torch.zeros_like(y[:, 0])
        alg_58_i_Na_L = torch.zeros_like(y[:, 0])
        alg_59_i_Ca_L_max = torch.zeros_like(y[:, 0])
        alg_60_i_Ca_L = torch.zeros_like(y[:, 0])
        alg_61_f_Ca_infinity = torch.zeros_like(y[:, 0])
        alg_62_f_Ca2_infinity = torch.zeros_like(y[:, 0])
        alg_63_ro_infinity_JSR = torch.zeros_like(y[:, 0])
        alg_64_i_Ca_T = torch.zeros_like(y[:, 0])
        alg_65_tau_f_Ca2 = torch.zeros_like(y[:, 0])
        alg_66_ro_infinity = torch.zeros_like(y[:, 0])
        alg_67_i_to_1 = torch.zeros_like(y[:, 0])
        alg_68_rr_infinity = torch.zeros_like(y[:, 0])
        alg_69_i_Kr = torch.zeros_like(y[:, 0])
        alg_70_g_Ks = torch.zeros_like(y[:, 0])
        alg_71_i_Ks = torch.zeros_like(y[:, 0])
        alg_72_beta_xK1 = torch.zeros_like(y[:, 0])
        alg_73_alpha_xK1 = torch.zeros_like(y[:, 0])
        alg_74_xK1 = torch.zeros_like(y[:, 0])
        alg_75_i_K1 = torch.zeros_like(y[:, 0])
        alg_76_kp = torch.zeros_like(y[:, 0])
        alg_77_i_K_p = torch.zeros_like(y[:, 0])
        alg_78_i_to_2_max = torch.zeros_like(y[:, 0])
        alg_79_i_to_2 = torch.zeros_like(y[:, 0])
        alg_80_dNaCa_1 = torch.zeros_like(y[:, 0])
        alg_81_dNaCa_2 = torch.zeros_like(y[:, 0])
        alg_82_i_NaCa = torch.zeros_like(y[:, 0])
        alg_83_f_NaK = torch.zeros_like(y[:, 0])
        alg_84_i_NaK = torch.zeros_like(y[:, 0])
        alg_85_i_Ca_p = torch.zeros_like(y[:, 0])
        alg_86_CT_K_Cl = torch.zeros_like(y[:, 0])
        alg_87_CT_Na_Cl = torch.zeros_like(y[:, 0])
        alg_88_i_Na_b = torch.zeros_like(y[:, 0])
        alg_89_i_K_b = torch.zeros_like(y[:, 0])
        alg_90_i_Cl_b = torch.zeros_like(y[:, 0])
        alg_91_i_Ca_b = torch.zeros_like(y[:, 0])
        alg_92_i_tot = torch.zeros_like(y[:, 0])
        alg_93_CMDN = torch.zeros_like(y[:, 0])
        alg_94_TRPN = torch.zeros_like(y[:, 0])
        alg_95_b_myo = torch.zeros_like(y[:, 0])
        alg_96_Ca_MK_bound = torch.zeros_like(y[:, 0])
        alg_97_Ca_MK_act = torch.zeros_like(y[:, 0])
        alg_98_tau_f_Ca = torch.zeros_like(y[:, 0])
        alg_99_q_diff = torch.zeros_like(y[:, 0])
        alg_100_Ca_fac = torch.zeros_like(y[:, 0])
        alg_101_b_SL = torch.zeros_like(y[:, 0])
        alg_102_ri_infinity = torch.zeros_like(y[:, 0])
        alg_103_b_SR = torch.zeros_like(y[:, 0])
        alg_104_tau_Ca_MK = torch.zeros_like(y[:, 0])
        alg_105_Ca_r_tot = torch.zeros_like(y[:, 0])
        alg_106_tau_ri = torch.zeros_like(y[:, 0])
        alg_107_vg = torch.zeros_like(y[:, 0])
        alg_108_g_rel = torch.zeros_like(y[:, 0])
        alg_109_q_rel = torch.zeros_like(y[:, 0])
        alg_110_q_leak = torch.zeros_like(y[:, 0])
        alg_111_dq_up_Ca_MK = torch.zeros_like(y[:, 0])
        alg_112_dkm_plb = torch.zeros_like(y[:, 0])
        alg_113_q_up = torch.zeros_like(y[:, 0])
        alg_114_q_tr = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_18_a_infinity = 1.0 / (1.0 + km_to_2 / Ca_r)
        d_a_2 = torch.zeros_like(y[:, 0]) + ((alg_18_a_infinity - a_2) / tau_a)
        alg_24_tau_h_L = 175.0 + 125.0 / (1.0 + torch.exp(-(V + 25.0) / 6.0))
        alg_6_h_L_infinity = 1.0 / (1.0 + torch.exp((V + 69.0) / 6.1))
        d_h_L = torch.zeros_like(y[:, 0]) + ((alg_6_h_L_infinity - h_L) / alg_24_tau_h_L)
        alg_7_d_infinity = 1.0 / (1.0 + torch.exp(-(V - 4.0) / 6.74))
        alg_25_tau_d = 0.59 + 0.8 * torch.exp(0.052 * (V + 13.0)) / (1.0 + torch.exp(0.132 * (V + 13.0)))
        d_d = torch.zeros_like(y[:, 0]) + ((alg_7_d_infinity - d) / alg_25_tau_d)
        alg_8_f_infinity = 1.0 / (1.0 + torch.exp((V + 18.0) / 10.0))
        alg_26_tau_f = 4.0 + 0.005 * (V - 2.5) ** 2.0
        d_f = torch.zeros_like(y[:, 0]) + ((alg_8_f_infinity - f) / alg_26_tau_f)
        alg_9_f2_infinity = 1.0 / (1.0 + torch.exp((V + 18.0) / 10.0))
        alg_27_tau_f2 = 38.0 + 0.07 * (V - 18.6) ** 2.0
        d_f2 = torch.zeros_like(y[:, 0]) + ((alg_9_f2_infinity - f2) / alg_27_tau_f2)
        alg_15_tau_xr = 900.0 / (1.0 + torch.exp(V / 5.0)) + 100.0
        alg_33_xr_infinity = 1.0 / (1.0 + torch.exp(-(V + 0.085) / 12.25))
        d_xr = torch.zeros_like(y[:, 0]) + ((alg_33_xr_infinity - xr) / alg_15_tau_xr)
        alg_16_tau_xs1 = 1.0 / (7.61e-05 * (V + 44.6) / (1.0 - torch.exp(-9.97 * (V + 44.6))) + 0.00036 * (V - 0.55) / (torch.exp(0.128 * (V - 0.55)) - 1.0))
        alg_34_xs1_infinity = 1.0 / (1.0 + torch.exp(-(V - 9.0) / 13.7))
        d_xs1 = torch.zeros_like(y[:, 0]) + ((alg_34_xs1_infinity - xs1) / alg_16_tau_xs1)
        alg_17_tau_xs2 = 2.0 * 1.0 / (7.61e-05 * (V + 44.6) / (1.0 - torch.exp(-9.97 * (V + 44.6))) + 0.00036 * (V - 0.55) / (torch.exp(0.128 * (V - 0.55)) - 1.0))
        alg_35_xs2_infinity = 1.0 / (1.0 + torch.exp(-(V - 9.0) / 13.7))
        d_xs2 = torch.zeros_like(y[:, 0]) + ((alg_35_xs2_infinity - xs2) / alg_17_tau_xs2)
        alg_2_alpha_m = 0.32 * (V + 47.13) / (1.0 - torch.exp(-0.1 * (V + 47.13)))
        alg_20_beta_m = 0.08 * torch.exp(-V / 11.0)
        alg_47_tau_m = 1.0 / (alg_2_alpha_m + alg_20_beta_m)
        alg_37_m_infinity = alg_2_alpha_m / (alg_2_alpha_m + alg_20_beta_m)
        d_m = torch.zeros_like(y[:, 0]) + ((alg_37_m_infinity - m) / alg_47_tau_m)
        alg_3_alpha_h = torch.where(V < -40.0, 0.135 * torch.exp((V + 80.0) / -6.8), 0.0)
        alg_21_beta_h = torch.where(V < -40.0, 3.56 * torch.exp(0.079 * V) + 310000.0 * torch.exp(0.35 * V), 1.0 / (0.13 * (1.0 + torch.exp((V + 10.66) / -11.1))))
        alg_48_tau_h = 1.0 / (alg_3_alpha_h + alg_21_beta_h)
        alg_38_h_infinity = alg_3_alpha_h / (alg_3_alpha_h + alg_21_beta_h)
        d_h = torch.zeros_like(y[:, 0]) + ((alg_38_h_infinity - h) / alg_48_tau_h)
        alg_4_alpha_j = torch.where(V < -40.0, (-127140.0 * torch.exp(0.2444 * V) - 3.474e-05 * torch.exp(-0.04391 * V)) * (V + 37.78) / (1.0 + torch.exp(0.311 * (V + 79.23))), 0.0)
        alg_22_beta_j = torch.where(V < -40.0, 0.1212 * torch.exp(-0.01052 * V) / (1.0 + torch.exp(-0.1378 * (V + 40.14))), 0.3 * torch.exp(-2.535e-07 * V) / (1.0 + torch.exp(-0.1 * (V + 32.0))))
        alg_49_tau_j = 0.1 / (alg_4_alpha_j + alg_22_beta_j)
        alg_39_j_infinity = 0.1 * alg_4_alpha_j / (alg_4_alpha_j + alg_22_beta_j)
        d_j = torch.zeros_like(y[:, 0]) + ((alg_39_j_infinity - j) / alg_49_tau_j)
        alg_5_alpha_m_L = 0.32 * (V + 47.13) / (1.0 - torch.exp(-0.1 * (V + 47.13)))
        alg_23_beta_m_L = 0.08 * torch.exp(-V / 11.0)
        alg_50_tau_m_L = 1.0 / (alg_5_alpha_m_L + alg_23_beta_m_L)
        alg_40_m_L_infinity = alg_5_alpha_m_L / (alg_5_alpha_m_L + alg_23_beta_m_L)
        d_m_L = torch.zeros_like(y[:, 0]) + ((alg_40_m_L_infinity - m_L) / alg_50_tau_m_L)
        alg_10_alpha_b = 1.068 * torch.exp((V + 16.3) / 30.0)
        alg_28_beta_b = 1.068 * torch.exp(-(V + 16.3) / 30.0)
        alg_51_tau_b = 1.0 / (alg_10_alpha_b + alg_28_beta_b)
        alg_41_b_infinity = 1.0 / (1.0 + torch.exp(-(V + 33.0) / 6.1))
        d_b = torch.zeros_like(y[:, 0]) + ((alg_41_b_infinity - b) / alg_51_tau_b)
        alg_11_alpha_g = 0.015 * torch.exp(-(V + 71.7) / 83.3)
        alg_29_beta_g = 0.015 * torch.exp((V + 71.7) / 15.4)
        alg_52_tau_g = 1.0 / (alg_11_alpha_g + alg_29_beta_g)
        alg_42_g_infinity = 1.0 / (1.0 + torch.exp((V + 60.0) / 6.6))
        d_g = torch.zeros_like(y[:, 0]) + ((alg_42_g_infinity - g) / alg_52_tau_g)
        alg_12_alpha_a = 25.0 * torch.exp((V - 76.0) / 20.0) / (1.0 + torch.exp((V - 76.0) / 20.0))
        alg_30_beta_a = 25.0 * torch.exp(-(V + 54.0) / 20.0) / (1.0 + torch.exp(-(V + 54.0) / 20.0))
        alg_43_tau_a = 1.0 / (alg_12_alpha_a + alg_30_beta_a)
        alg_53_a_infinity_2 = alg_12_alpha_a / (alg_12_alpha_a + alg_30_beta_a)
        d_a = torch.zeros_like(y[:, 0]) + ((alg_53_a_infinity_2 - a) / alg_43_tau_a)
        alg_44_tau_i = 6.0 + 5.0 / (1.0 + torch.exp((V - 16.5) / 10.0))
        alg_13_alpha_i = 0.03 / (1.0 + torch.exp((V + 25.0) / 15.0))
        alg_31_beta_i = 0.1 * torch.exp((V - 40.0) / 15.0) / (1.0 + torch.exp((V - 40.0) / 15.0))
        alg_54_i_infinity = alg_13_alpha_i / (alg_13_alpha_i + alg_31_beta_i)
        d_i = torch.zeros_like(y[:, 0]) + ((alg_54_i_infinity - i) / alg_44_tau_i)
        alg_45_tau_i2 = 21.5 + 30.0 / (1.0 + torch.exp((V - 25.0) / 10.0))
        alg_14_alpha_i2 = 0.00442 / (1.0 + torch.exp((V + 26.0) / 15.0))
        alg_32_beta_i2 = 0.05 * torch.exp((V - 10.0) / 15.0) / (1.0 + torch.exp((V - 10.0) / 15.0))
        alg_55_i2_infinity = alg_14_alpha_i2 / (alg_14_alpha_i2 + alg_32_beta_i2)
        d_i2 = torch.zeros_like(y[:, 0]) + ((alg_55_i2_infinity - i2) / alg_45_tau_i2)
        alg_59_i_Ca_L_max = 1.0 * p_Ca / Cm * z_Ca ** 2.0 * (V - 15.0) * F ** 2.0 / (R * T) * (gamma_Cai * Ca_r * torch.exp(z_Ca * F * (V - 15.0) / (R * T)) - gamma_Cao * Ca_o) / (torch.exp(z_Ca * F * (V - 15.0) / (R * T)) - 1.0)
        alg_60_i_Ca_L = g_Ca_L * d * f * f2 * f_Ca * f_Ca2 * alg_59_i_Ca_L_max
        alg_62_f_Ca2_infinity = 1.0 / (1.0 - alg_60_i_Ca_L / 0.01)
        alg_65_tau_f_Ca2 = 125.0 + 300.0 / (1.0 + torch.exp((-alg_60_i_Ca_L - 0.175) / 0.04))
        d_f_Ca2 = torch.zeros_like(y[:, 0]) + ((alg_62_f_Ca2_infinity - f_Ca2) / alg_65_tau_f_Ca2)
        alg_63_ro_infinity_JSR = Ca_JSR ** 1.9 / (Ca_JSR ** 1.9 + (49.28 * Ca_r / (Ca_r + 0.0028)) ** 1.9)
        alg_66_ro_infinity = alg_63_ro_infinity_JSR * alg_60_i_Ca_L ** 2.0 / (alg_60_i_Ca_L ** 2.0 + 1.0)
        d_ro = torch.zeros_like(y[:, 0]) + ((alg_66_ro_infinity - ro) / tau_ro)
        alg_19_E_Na = R * T / F * torch.log(Na_o / Na_i)
        alg_57_i_Na = g_Na * m ** 3.0 * (0.8 * h + 0.2 * j) * (V - alg_19_E_Na)
        alg_58_i_Na_L = g_Na_L * m_L ** 3.0 * h_L * (V - alg_19_E_Na)
        alg_80_dNaCa_1 = km_Ca_o * Na_i ** 3.0 + 1.5 * km_Na_o ** 3.0 * Ca_i + km_Na_i_1 ** 3.0 * Ca_o * (1.0 + 1.5 * Ca_i / km_Ca_i)
        alg_81_dNaCa_2 = km_Ca_i * Na_o ** 3.0 * (1.0 + Na_i / km_Na_i_1) + Na_i ** 3.0 * Ca_o + 1.5 * Na_o ** 3.0 * Ca_i
        alg_82_i_NaCa = (X_NaCa * i_NaCa_max * Na_i ** 3.0 * Ca_o * torch.exp(0.35 * F * V / (R * T)) - 1.5 * Na_o ** 3.0 * Ca_i * torch.exp(-0.65 * F * V / (R * T))) / ((1.0 + (km_Ca_act / (1.5 * Ca_i)) ** 2.0) * (1.0 + k_sat * torch.exp(-0.65 * V * F / (R * T))) * (alg_80_dNaCa_1 + alg_81_dNaCa_2))
        alg_83_f_NaK = 1.0 / (1.0 + 0.1245 * torch.exp(-0.1 * F * V / (R * T)) + 0.0365 * sigma * torch.exp(-F * V / (R * T)))
        alg_84_i_NaK = g_NaK * alg_83_f_NaK * 1.0 / (1.0 + (km_Na_i_2 / Na_i) ** 2.0) * K_o / (K_o + km_K_o)
        alg_88_i_Na_b = g_Na_b * (V - alg_19_E_Na)
        alg_46_E_Cl = -R * T / F * torch.log(Cl_o / Cl_i)
        alg_87_CT_Na_Cl = CT_Na_Cl_max * (alg_19_E_Na - alg_46_E_Cl) ** 4.0 / ((alg_19_E_Na - alg_46_E_Cl) ** 4.0 + 59494197.640692554)
        d_Na_i = torch.zeros_like(y[:, 0]) + (-Cm * (alg_57_i_Na + alg_58_i_Na_L + alg_88_i_Na_b + 3.0 * alg_84_i_NaK + 3.0 * alg_82_i_NaCa) * a_cap / (Vol_myo * F) + alg_87_CT_Na_Cl)
        alg_36_E_K = R * T / F * torch.log(K_o / K_i)
        alg_67_i_to_1 = g_to_1 * a * (0.8 * i + 0.2 * i2) * (V - alg_36_E_K)
        alg_68_rr_infinity = 1.0 / (1.0 + torch.exp((V - 5.4) / 20.4))
        alg_69_i_Kr = g_Kr * xr * alg_68_rr_infinity * (V - alg_36_E_K)
        alg_56_E_Ks = R * T / F * torch.log((K_o + r_NaK * Na_o) / (K_i + r_NaK * Na_i))
        alg_70_g_Ks = 0.0525813 * (1.0 + 0.6 / (1.0 + (3.8e-05 / Ca_i) ** 1.4))
        alg_71_i_Ks = alg_70_g_Ks * xs1 * xs2 * (V - alg_56_E_Ks)
        alg_73_alpha_xK1 = 1.02 / (1.0 + torch.exp(0.2385 * (V - (alg_36_E_K + 59.215))))
        alg_72_beta_xK1 = (0.49124 * torch.exp(0.08032 * (V + 5.476 - alg_36_E_K)) + torch.exp(0.06175 * (V - (594.31 + alg_36_E_K)))) / (1.0 + torch.exp(-0.5143 * (V + 4.753 - alg_36_E_K)))
        alg_74_xK1 = alg_73_alpha_xK1 / (alg_73_alpha_xK1 + alg_72_beta_xK1)
        alg_75_i_K1 = (g_K1 * alg_74_xK1 + 0.004) * (V - alg_36_E_K)
        alg_89_i_K_b = g_K_b * (V - alg_36_E_K)
        alg_76_kp = 1.0 / (1.0 + torch.exp((7.488 - V) / 5.98))
        alg_77_i_K_p = g_K_p * alg_76_kp * (V - alg_36_E_K)
        alg_86_CT_K_Cl = CT_K_Cl_max * (alg_36_E_K - alg_46_E_Cl) / (alg_36_E_K + 87.8251 - alg_46_E_Cl)
        d_K_i = torch.zeros_like(y[:, 0]) + (-Cm * (alg_67_i_to_1 + alg_75_i_K1 + alg_69_i_Kr + alg_71_i_Ks + alg_77_i_K_p + alg_89_i_K_b - 2.0 * alg_84_i_NaK) * a_cap / (Vol_myo * F) + alg_86_CT_K_Cl)
        alg_78_i_to_2_max = 1.0 * p_Cl / Cm * z_Cl ** 2.0 * V * F ** 2.0 / (R * T) * (Cl_i - Cl_o * torch.exp(-z_Cl * V * F / (R * T))) / (1.0 - torch.exp(-z_Cl * V * F / (R * T)))
        alg_79_i_to_2 = 20.0 * alg_78_i_to_2_max * a_2
        alg_90_i_Cl_b = g_Cl_b * (V - alg_46_E_Cl)
        d_Cl_i = torch.zeros_like(y[:, 0]) + (-Cm * (alg_79_i_to_2 + alg_90_i_Cl_b) * a_cap / (Vol_myo * F) + alg_87_CT_Na_Cl + alg_86_CT_K_Cl)
        alg_0_i_stim = torch.where((t >= stim_start) & (t <= stim_end), stim_amplitude, 0.0)
        alg_64_i_Ca_T = g_Ca_T * b * g * (V - 50.0)
        alg_91_i_Ca_b = 1.0 * p_Ca_b / Cm * z_Ca ** 2.0 * V * F ** 2.0 / (R * T) * (gamma_Cai * Ca_i * torch.exp(z_Ca * V * F / (R * T)) - gamma_Cao * Ca_o) / (torch.exp(z_Ca * V * F / (R * T)) - 1.0)
        alg_85_i_Ca_p = i_Ca_p_max * Ca_i / (km_Ca_p + Ca_i)
        alg_92_i_tot = alg_57_i_Na + alg_58_i_Na_L + alg_60_i_Ca_L + alg_64_i_Ca_T + alg_67_i_to_1 + alg_79_i_to_2 + alg_69_i_Kr + alg_71_i_Ks + alg_75_i_K1 + alg_82_i_NaCa + alg_84_i_NaK + alg_88_i_Na_b + alg_89_i_K_b + alg_91_i_Ca_b + alg_90_i_Cl_b + alg_85_i_Ca_p + alg_77_i_K_p
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_92_i_tot + alg_0_i_stim))
        alg_96_Ca_MK_bound = Ca_MK_0 * (1.0 - Ca_MK_trap) / (1.0 + km_Ca_MK / Ca_r)
        d_Ca_MK_trap = torch.zeros_like(y[:, 0]) + (alpha_Ca_MK * alg_96_Ca_MK_bound * (alg_96_Ca_MK_bound + Ca_MK_trap) - beta_Ca_MK * Ca_MK_trap)
        alg_61_f_Ca_infinity = 0.3 / (1.0 - alg_60_i_Ca_L / 0.05) + 0.55 / (1.0 + Ca_r / 0.003) + 0.15
        alg_97_Ca_MK_act = alg_96_Ca_MK_bound + Ca_MK_trap
        alg_98_tau_f_Ca = 0.5 + 10.0 * 1.0 * alg_97_Ca_MK_act / (1.0 * alg_97_Ca_MK_act + km_Ca_MK) + 1.0 / (1.0 + Ca_r / 0.003)
        d_f_Ca = torch.zeros_like(y[:, 0]) + ((alg_61_f_Ca_infinity - f_Ca) / alg_98_tau_f_Ca)
        alg_104_tau_Ca_MK = tau_Ca_MK_max * 1.0 * alg_97_Ca_MK_act / (km_Ca_MK + 1.0 * alg_97_Ca_MK_act)
        alg_100_Ca_fac = 1.0 / (1.0 + torch.exp((alg_60_i_Ca_L + 0.05) / 0.015))
        alg_106_tau_ri = (350.0 - alg_104_tau_Ca_MK) / (1.0 + torch.exp((Ca_r - 0.003 + 0.003 * alg_100_Ca_fac) / 0.0002)) + 3.0 + alg_104_tau_Ca_MK
        alg_102_ri_infinity = 1.0 / (1.0 + torch.exp((Ca_r - 0.0004 + 0.002 * alg_100_Ca_fac) / 2.5e-05))
        d_ri = torch.zeros_like(y[:, 0]) + ((alg_102_ri_infinity - ri) / alg_106_tau_ri)
        alg_99_q_diff = (Ca_r - Ca_i) / tau_ss
        alg_107_vg = 1.0 / (1.0 + torch.exp((g_Ca_L * alg_59_i_Ca_L_max + 13.0) / 5.0))
        alg_108_g_rel = 3000.0 * alg_107_vg
        alg_109_q_rel = alg_108_g_rel * ro * ri * (Ca_JSR - Ca_r)
        alg_103_b_SR = 2.0 * b_SR_max * Ca_r / (Ca_r + km_b_SR) ** 2.0
        alg_101_b_SL = 2.0 * b_SL_max * Ca_r / (Ca_r + km_b_SL) ** 2.0
        alg_105_Ca_r_tot = 1.0 / (1.0 + alg_103_b_SR + alg_101_b_SL)
        d_Ca_r = torch.zeros_like(y[:, 0]) + (alg_105_Ca_r_tot * (-Cm * alg_60_i_Ca_L * a_cap / (Vol_ss * z_Ca * F) + alg_109_q_rel * Vol_jsr / Vol_ss - alg_99_q_diff))
        alg_94_TRPN = 2.0 * TRPN_max * Ca_i / (Ca_i + km_TRPN) ** 2.0
        alg_93_CMDN = 2.0 * CMDN_max * Ca_i / (Ca_i + km_CMDN) ** 2.0
        alg_95_b_myo = 1.0 / (1.0 + alg_94_TRPN + alg_93_CMDN)
        alg_111_dq_up_Ca_MK = dq_up_Ca_MK_max * alg_97_Ca_MK_act * 1.0 / (km_Ca_MK + alg_97_Ca_MK_act * 1.0)
        alg_112_dkm_plb = dkm_plb_max * alg_97_Ca_MK_act * 1.0 / (km_Ca_MK + alg_97_Ca_MK_act * 1.0)
        alg_113_q_up = X_q_up * (alg_111_dq_up_Ca_MK + 1.0) * q_up_max * Ca_i / (Ca_i + km_up - alg_112_dkm_plb)
        alg_110_q_leak = q_leak_max * Ca_NSR / NSR_max
        d_Ca_i = torch.zeros_like(y[:, 0]) + (-alg_95_b_myo * (Cm * (alg_91_i_Ca_b + alg_85_i_Ca_p - 2.0 * alg_82_i_NaCa) * a_cap / (2.0 * Vol_myo * F) + (alg_113_q_up - alg_110_q_leak) * Vol_nsr / Vol_myo - alg_99_q_diff * Vol_ss / Vol_myo))
        alg_114_q_tr = (Ca_NSR - Ca_JSR) / tau_tr
        d_Ca_NSR = torch.zeros_like(y[:, 0]) + (alg_113_q_up - (alg_110_q_leak + alg_114_q_tr * Vol_jsr / Vol_nsr))
        d_Ca_JSR = torch.zeros_like(y[:, 0]) + ((alg_114_q_tr - alg_109_q_rel) / (1.0 + CSQN_max * km_CSQN / (km_CSQN + Ca_JSR) ** 2.0))

        return torch.stack(
            [d_V, d_Na_i, d_Ca_i, d_K_i, d_Cl_i, d_m, d_h, d_j, d_m_L, d_h_L, d_Ca_r, d_d, d_f, d_f2, d_f_Ca, d_f_Ca2, d_b, d_g, d_a, d_i, d_i2, d_xr, d_xs1, d_xs2, d_a_2, d_Ca_MK_trap, d_Ca_NSR, d_Ca_JSR, d_ri, d_ro],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
