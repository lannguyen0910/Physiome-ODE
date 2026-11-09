import torch
import torch.nn as nn
import math


class ExcitationContractionCouplingPanditHinchNiedererODE(nn.Module):
    """
    PyTorch translation of excitation-contraction_coupling/Pandit_Hinch_Niederer/model.py.
    Dynamic states: 22; independent constants: 84; derived constants used by rates: 9.
    Original constant slots: 93; algebraic slots: 106.
    """
    source_file = 'excitation-contraction_coupling/Pandit_Hinch_Niederer/model.py'
    source_state_dim = 22
    source_constant_dim = 93
    algebraic_dim = 106
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
        self.parameter_dim = 84

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8314.5, 'T': 295, 'F': 96487, 'Cm': 0.0001, 'stim_period': 1000, 'stim_duration': 10, 'stim_amplitude': -0.0006, 'V_myo': 25850, 'V_SR': 2098, 'V_myo_uL': 2.585e-05, 'V_SR_uL': 2.098e-06, 'i_NaK_max': 9.5e-05, 'K_m_K': 1.5, 'K_m_Na': 10, 'K_o': 5.4, 'Na_o': 140, 'Ca_TRPN_Max': 0.07, 'g_Na': 0.8, 'g_t': 0.035, 'a_endo': 0.583, 'b_endo': 0.417, 'g_ss': 0.007, 'g_K1': 0.024, 'g_f': 0.00145, 'f_Na': 0.2, 'g_B_Na': 8.015e-05, 'g_B_Ca': 3.24e-05, 'g_B_K': 0.000138, 'E_Ca': 65, 'Ca_o': 1.2, 'g_D': 0.065, 'J_R': 0.02, 'J_L': 0.000913, 'N': 50000, 'K_mNa': 87.5, 'K_mCa': 1.38, 'eta': 0.35, 'k_sat': 0.1, 'g_NCX': 0.0385, 'g_SERCA': 0.00045, 'K_SERCA': 0.0005, 'g_pCa': 3.5e-06, 'K_mpCa': 0.0005, 'g_CaB': 2.6875e-08, 'g_SRl': 1.8951e-05, 'k_CMDN': 0.002382, 'B_CMDN': 0.05, 'k_on': 100, 'k_Ref_off': 0.2, 'gamma_trpn': 2, 'alpha_0': 0.008, 'alpha_r1': 0.002, 'alpha_r2': 0.00175, 'n_Rel': 3, 'K_z': 0.15, 'n_Hill': 3, 'Ca_50ref': 0.00105, 'z_p': 0.85, 'beta_1': -4, 'beta_0': 4.9, 'T_ref': 56.2, 'a': 0.35, 'A_1': -29, 'A_2': 138, 'A_3': 129, 'alpha_1': 0.03, 'alpha_2': 0.13, 'alpha_3': 0.625, 'V_L': -2, 'del_VL': 7, 'phi_L': 2.35, 't_L': 1, 'tau_L': 650, 'tau_R': 2.43, 'phi_R': 0.05, 'theta_R': 0.012, 'K_RyR': 0.041, 'K_L': 0.00022, 'a_2': 0.0625, 'b': 14, 'c': 0.01, 'd': 100, 'dExtensionRatiodt': 0, 'tau_s_ss': 2.1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -80.50146, 'Na_i': 10.73519, 'Ca_i': 7.901351e-05, 'Ca_SR': 0.7, 'K_i': 139.2751, 'TRPN': 0.067593139865, 'm': 0.004164108, 'h': 0.6735613, 'j': 0.6729362, 'r': 0.002191519, 's': 0.9842542, 's_slow': 0.6421196, 'r_ss': 0.002907171, 's_ss': 0.3142767, 'y': 0.003578708, 'z': 0.014417937837, 'Q_1': 0, 'Q_2': 0, 'Q_3': 0, 'z_1': 0.98859, 'z_2': 0.0087302, 'z_3': 0.0026566}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'Cm', 'stim_period', 'stim_duration', 'stim_amplitude', 'V_myo', 'V_SR', 'V_myo_uL', 'V_SR_uL', 'i_NaK_max', 'K_m_K', 'K_m_Na', 'K_o', 'Na_o', 'Ca_TRPN_Max', 'g_Na', 'g_t', 'a_endo', 'b_endo', 'g_ss', 'g_K1', 'g_f', 'f_Na', 'g_B_Na', 'g_B_Ca', 'g_B_K', 'E_Ca', 'Ca_o', 'g_D', 'J_R', 'J_L', 'N', 'K_mNa', 'K_mCa', 'eta', 'k_sat', 'g_NCX', 'g_SERCA', 'K_SERCA', 'g_pCa', 'K_mpCa', 'g_CaB', 'g_SRl', 'k_CMDN', 'B_CMDN', 'k_on', 'k_Ref_off', 'gamma_trpn', 'alpha_0', 'alpha_r1', 'alpha_r2', 'n_Rel', 'K_z', 'n_Hill', 'Ca_50ref', 'z_p', 'beta_1', 'beta_0', 'T_ref', 'a', 'A_1', 'A_2', 'A_3', 'alpha_1', 'alpha_2', 'alpha_3', 'V_L', 'del_VL', 'phi_L', 't_L', 'tau_L', 'tau_R', 'phi_R', 'theta_R', 'K_RyR', 'K_L', 'a_2', 'b', 'c', 'd', 'dExtensionRatiodt', 'tau_s_ss']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'Na_i', 'Ca_i', 'Ca_SR', 'K_i', 'TRPN', 'm', 'h', 'j', 'r', 's', 's_slow', 'r_ss', 's_ss', 'y', 'z', 'Q_1', 'Q_2', 'Q_3', 'z_1', 'z_2', 'z_3']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (22)
        V = y[:, 0]
        Na_i = y[:, 1]
        Ca_i = y[:, 2]
        Ca_SR = y[:, 3]
        K_i = y[:, 4]
        TRPN = y[:, 5]
        m = y[:, 6]
        h = y[:, 7]
        j = y[:, 8]
        r = y[:, 9]
        s = y[:, 10]
        s_slow = y[:, 11]
        r_ss = y[:, 12]
        s_ss = y[:, 13]
        state_y = y[:, 14]
        z = y[:, 15]
        Q_1 = y[:, 16]
        Q_2 = y[:, 17]
        Q_3 = y[:, 18]
        z_1 = y[:, 19]
        z_2 = y[:, 20]
        z_3 = y[:, 21]

        # Independent parameters (source constant order)
        R = theta[:, 0]  # constants[0]
        T = theta[:, 1]  # constants[1]
        F = theta[:, 2]  # constants[2]
        Cm = theta[:, 3]  # constants[3]
        stim_period = theta[:, 4]  # constants[4]
        stim_duration = theta[:, 5]  # constants[5]
        stim_amplitude = theta[:, 6]  # constants[6]
        V_myo = theta[:, 7]  # constants[7]
        V_SR = theta[:, 8]  # constants[8]
        V_myo_uL = theta[:, 9]  # constants[9]
        V_SR_uL = theta[:, 10]  # constants[10]
        i_NaK_max = theta[:, 11]  # constants[11]
        K_m_K = theta[:, 12]  # constants[12]
        K_m_Na = theta[:, 13]  # constants[13]
        K_o = theta[:, 14]  # constants[14]
        Na_o = theta[:, 15]  # constants[15]
        Ca_TRPN_Max = theta[:, 16]  # constants[16]
        g_Na = theta[:, 17]  # constants[17]
        g_t = theta[:, 18]  # constants[18]
        a_endo = theta[:, 19]  # constants[19]
        b_endo = theta[:, 20]  # constants[20]
        g_ss = theta[:, 21]  # constants[21]
        g_K1 = theta[:, 22]  # constants[22]
        g_f = theta[:, 23]  # constants[23]
        f_Na = theta[:, 24]  # constants[24]
        g_B_Na = theta[:, 25]  # constants[25]
        g_B_Ca = theta[:, 26]  # constants[26]
        g_B_K = theta[:, 27]  # constants[27]
        E_Ca = theta[:, 28]  # constants[28]
        Ca_o = theta[:, 29]  # constants[29]
        g_D = theta[:, 30]  # constants[30]
        J_R = theta[:, 31]  # constants[31]
        J_L = theta[:, 32]  # constants[32]
        N = theta[:, 33]  # constants[33]
        K_mNa = theta[:, 34]  # constants[34]
        K_mCa = theta[:, 35]  # constants[35]
        eta = theta[:, 36]  # constants[36]
        k_sat = theta[:, 37]  # constants[37]
        g_NCX = theta[:, 38]  # constants[38]
        g_SERCA = theta[:, 39]  # constants[39]
        K_SERCA = theta[:, 40]  # constants[40]
        g_pCa = theta[:, 41]  # constants[41]
        K_mpCa = theta[:, 42]  # constants[42]
        g_CaB = theta[:, 43]  # constants[43]
        g_SRl = theta[:, 44]  # constants[44]
        k_CMDN = theta[:, 45]  # constants[45]
        B_CMDN = theta[:, 46]  # constants[46]
        k_on = theta[:, 47]  # constants[47]
        k_Ref_off = theta[:, 48]  # constants[48]
        gamma_trpn = theta[:, 49]  # constants[49]
        alpha_0 = theta[:, 50]  # constants[50]
        alpha_r1 = theta[:, 51]  # constants[51]
        alpha_r2 = theta[:, 52]  # constants[52]
        n_Rel = theta[:, 53]  # constants[53]
        K_z = theta[:, 54]  # constants[54]
        n_Hill = theta[:, 55]  # constants[55]
        Ca_50ref = theta[:, 56]  # constants[56]
        z_p = theta[:, 57]  # constants[57]
        beta_1 = theta[:, 58]  # constants[58]
        beta_0 = theta[:, 59]  # constants[59]
        T_ref = theta[:, 60]  # constants[60]
        a = theta[:, 61]  # constants[61]
        A_1 = theta[:, 62]  # constants[62]
        A_2 = theta[:, 63]  # constants[63]
        A_3 = theta[:, 64]  # constants[64]
        alpha_1 = theta[:, 65]  # constants[65]
        alpha_2 = theta[:, 66]  # constants[66]
        alpha_3 = theta[:, 67]  # constants[67]
        V_L = theta[:, 68]  # constants[68]
        del_VL = theta[:, 69]  # constants[69]
        phi_L = theta[:, 70]  # constants[70]
        t_L = theta[:, 71]  # constants[71]
        tau_L = theta[:, 72]  # constants[72]
        tau_R = theta[:, 73]  # constants[73]
        phi_R = theta[:, 74]  # constants[74]
        theta_R = theta[:, 75]  # constants[75]
        K_RyR = theta[:, 76]  # constants[76]
        K_L = theta[:, 77]  # constants[77]
        a_2 = theta[:, 78]  # constants[78]
        b = theta[:, 79]  # constants[79]
        c = theta[:, 80]  # constants[80]
        d = theta[:, 81]  # constants[81]
        dExtensionRatiodt = theta[:, 82]  # constants[87]
        tau_s_ss = theta[:, 83]  # constants[88]

        # Derived constants from the original initConsts()
        sigma = (torch.exp(Na_o / 67.3) - 1.0) / 7.0  # constants[82]
        g_Na_endo = 1.33 * g_Na  # constants[83]
        g_t_endo = 0.4647 * g_t  # constants[84]
        f_K = 1.0 - f_Na  # constants[85]
        K_2 = alpha_r2 * z_p ** n_Rel / (z_p ** n_Rel + K_z ** n_Rel) * (1.0 - n_Rel * K_z ** n_Rel / (z_p ** n_Rel + K_z ** n_Rel))  # constants[86]
        t_R = 1.17 * t_L  # constants[89]
        K_1 = alpha_r2 * z_p ** (n_Rel - 1.0) * n_Rel * K_z ** n_Rel / (z_p ** n_Rel + K_z ** n_Rel) ** 2.0  # constants[90]
        alpha_m = phi_L / t_L  # constants[91]
        beta_m = phi_R / t_R  # constants[92]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_Ca_SR = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_TRPN = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_j = torch.zeros_like(y[:, 0])
        d_r = torch.zeros_like(y[:, 0])
        d_s = torch.zeros_like(y[:, 0])
        d_s_slow = torch.zeros_like(y[:, 0])
        d_r_ss = torch.zeros_like(y[:, 0])
        d_s_ss = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])
        d_z = torch.zeros_like(y[:, 0])
        d_Q_1 = torch.zeros_like(y[:, 0])
        d_Q_2 = torch.zeros_like(y[:, 0])
        d_Q_3 = torch.zeros_like(y[:, 0])
        d_z_1 = torch.zeros_like(y[:, 0])
        d_z_2 = torch.zeros_like(y[:, 0])
        d_z_3 = torch.zeros_like(y[:, 0])
        alg_0_I_Stim = torch.zeros_like(y[:, 0])
        alg_1_m_infinity = torch.zeros_like(y[:, 0])
        alg_2_h_infinity = torch.zeros_like(y[:, 0])
        alg_3_j_infinity = torch.zeros_like(y[:, 0])
        alg_4_r_infinity = torch.zeros_like(y[:, 0])
        alg_5_s_infinity = torch.zeros_like(y[:, 0])
        alg_6_s_slow_infinity = torch.zeros_like(y[:, 0])
        alg_7_r_ss_infinity = torch.zeros_like(y[:, 0])
        alg_8_s_ss_infinity = torch.zeros_like(y[:, 0])
        alg_9_y_infinity = torch.zeros_like(y[:, 0])
        alg_11_FVRT = torch.zeros_like(y[:, 0])
        alg_12_tau_m = torch.zeros_like(y[:, 0])
        alg_13_tau_h = torch.zeros_like(y[:, 0])
        alg_14_tau_j = torch.zeros_like(y[:, 0])
        alg_15_tau_r = torch.zeros_like(y[:, 0])
        alg_16_tau_s_endo = torch.zeros_like(y[:, 0])
        alg_17_tau_s_slow_endo = torch.zeros_like(y[:, 0])
        alg_18_tau_r_ss = torch.zeros_like(y[:, 0])
        alg_19_tau_y = torch.zeros_like(y[:, 0])
        alg_20_FVRT_Ca = torch.zeros_like(y[:, 0])
        alg_21_i_NaK = torch.zeros_like(y[:, 0])
        alg_23_Ca_b = torch.zeros_like(y[:, 0])
        alg_24_E_Na = torch.zeros_like(y[:, 0])
        alg_25_i_Na = torch.zeros_like(y[:, 0])
        alg_26_E_K = torch.zeros_like(y[:, 0])
        alg_27_i_t = torch.zeros_like(y[:, 0])
        alg_28_i_ss = torch.zeros_like(y[:, 0])
        alg_29_i_K1 = torch.zeros_like(y[:, 0])
        alg_30_i_f_Na = torch.zeros_like(y[:, 0])
        alg_31_i_f_K = torch.zeros_like(y[:, 0])
        alg_32_i_f = torch.zeros_like(y[:, 0])
        alg_33_i_B_Na = torch.zeros_like(y[:, 0])
        alg_35_i_B_K = torch.zeros_like(y[:, 0])
        alg_37_I_NaCa = torch.zeros_like(y[:, 0])
        alg_38_I_NaCa_2 = torch.zeros_like(y[:, 0])
        alg_39_I_SERCA = torch.zeros_like(y[:, 0])
        alg_40_I_pCa = torch.zeros_like(y[:, 0])
        alg_41_I_pCa_2 = torch.zeros_like(y[:, 0])
        alg_42_E_Ca = torch.zeros_like(y[:, 0])
        alg_43_I_CaB = torch.zeros_like(y[:, 0])
        alg_44_I_CaB_2 = torch.zeros_like(y[:, 0])
        alg_45_I_SR = torch.zeros_like(y[:, 0])
        alg_46_beta_CMDN = torch.zeros_like(y[:, 0])
        alg_47_beta_Tm = torch.zeros_like(y[:, 0])
        alg_48_ExtensionRatio = torch.zeros_like(y[:, 0])
        alg_49_lambda_var = torch.zeros_like(y[:, 0])
        alg_51_Ca_50 = torch.zeros_like(y[:, 0])
        alg_52_Ca_TRPN_50 = torch.zeros_like(y[:, 0])
        alg_53_alpha_Tm = torch.zeros_like(y[:, 0])
        alg_54_z_max = torch.zeros_like(y[:, 0])
        alg_55_overlap = torch.zeros_like(y[:, 0])
        alg_56_T_Base = torch.zeros_like(y[:, 0])
        alg_57_T_0 = torch.zeros_like(y[:, 0])
        alg_58_Q = torch.zeros_like(y[:, 0])
        alg_59_Tension = torch.zeros_like(y[:, 0])
        alg_60_k_off = torch.zeros_like(y[:, 0])
        alg_61_J_TRPN = torch.zeros_like(y[:, 0])
        alg_62_expVL = torch.zeros_like(y[:, 0])
        alg_63_alpha_p = torch.zeros_like(y[:, 0])
        alg_64_beta_pcc = torch.zeros_like(y[:, 0])
        alg_65_epsilon_pcc = torch.zeros_like(y[:, 0])
        alg_66_epsilon_m = torch.zeros_like(y[:, 0])
        alg_67_mu_pcc = torch.zeros_like(y[:, 0])
        alg_68_mu_mcc = torch.zeros_like(y[:, 0])
        alg_73_r_3 = torch.zeros_like(y[:, 0])
        alg_74_r_4 = torch.zeros_like(y[:, 0])
        alg_75_r_6 = torch.zeros_like(y[:, 0])
        alg_76_r_7 = torch.zeros_like(y[:, 0])
        alg_77_r_8 = torch.zeros_like(y[:, 0])
        alg_78_z_4 = torch.zeros_like(y[:, 0])
        alg_79_C_co = torch.zeros_like(y[:, 0])
        alg_80_epsilon_pco = torch.zeros_like(y[:, 0])
        alg_81_C_oc = torch.zeros_like(y[:, 0])
        alg_82_beta_poc = torch.zeros_like(y[:, 0])
        alg_83_mu_poc = torch.zeros_like(y[:, 0])
        alg_84_mu_moc = torch.zeros_like(y[:, 0])
        alg_85_denom = torch.zeros_like(y[:, 0])
        alg_86_y_oc = torch.zeros_like(y[:, 0])
        alg_87_y_co = torch.zeros_like(y[:, 0])
        alg_88_y_oo = torch.zeros_like(y[:, 0])
        alg_89_J_Rco = torch.zeros_like(y[:, 0])
        alg_90_y_cc = torch.zeros_like(y[:, 0])
        alg_91_J_Roo = torch.zeros_like(y[:, 0])
        alg_93_r_1 = torch.zeros_like(y[:, 0])
        alg_94_J_Loc = torch.zeros_like(y[:, 0])
        alg_95_r_2 = torch.zeros_like(y[:, 0])
        alg_96_J_Loo = torch.zeros_like(y[:, 0])
        alg_97_r_5 = torch.zeros_like(y[:, 0])
        alg_98_J_R1 = torch.zeros_like(y[:, 0])
        alg_99_J_R3 = torch.zeros_like(y[:, 0])
        alg_100_I_RyR = torch.zeros_like(y[:, 0])
        alg_101_I_RyR_2 = torch.zeros_like(y[:, 0])
        alg_102_J_L1 = torch.zeros_like(y[:, 0])
        alg_103_J_L2 = torch.zeros_like(y[:, 0])
        alg_104_I_LCC = torch.zeros_like(y[:, 0])
        alg_105_I_LCC_2 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Q_1 = torch.zeros_like(y[:, 0]) + (A_1 * dExtensionRatiodt - alpha_1 * Q_1)
        d_Q_2 = torch.zeros_like(y[:, 0]) + (A_2 * dExtensionRatiodt - alpha_2 * Q_2)
        d_Q_3 = torch.zeros_like(y[:, 0]) + (A_3 * dExtensionRatiodt - alpha_3 * Q_3)
        alg_8_s_ss_infinity = 1.0 / (1.0 + torch.exp((V + 87.5) / 10.3))
        d_s_ss = torch.zeros_like(y[:, 0]) + ((alg_8_s_ss_infinity - s_ss) / tau_s_ss * 0.001)
        alg_1_m_infinity = 1.0 / (1.0 + torch.exp((V + 45.0) / -6.5))
        alg_12_tau_m = 0.00136 / (0.32 * (V + 47.13) / (1.0 - torch.exp(-0.1 * (V + 47.13))) + 0.08 * torch.exp(-V / 11.0))
        d_m = torch.zeros_like(y[:, 0]) + ((alg_1_m_infinity - m) / alg_12_tau_m * 0.001)
        alg_2_h_infinity = 1.0 / (1.0 + torch.exp((V + 76.1) / 6.07))
        alg_13_tau_h = torch.where(V >= -40.0, 0.0004537 * (1.0 + torch.exp(-(V + 10.66) / 11.1)), 0.00349 / (0.135 * torch.exp(-(V + 80.0) / 6.8) + 3.56 * torch.exp(0.079 * V) + 310000.0 * torch.exp(0.35 * V)))
        d_h = torch.zeros_like(y[:, 0]) + ((alg_2_h_infinity - h) / alg_13_tau_h * 0.001)
        alg_3_j_infinity = 1.0 / (1.0 + torch.exp((V + 76.1) / 6.07))
        alg_14_tau_j = torch.where(V >= -40.0, 0.01163 * (1.0 + torch.exp(-0.1 * (V + 32.0))) / torch.exp(-2.535e-07 * V), 0.00349 / ((V + 37.78) / (1.0 + torch.exp(0.311 * (V + 79.23))) * (-127140.0 * torch.exp(0.2444 * V) - 3.474e-05 * torch.exp(-0.04391 * V)) + 0.1212 * torch.exp(-0.01052 * V) / (1.0 + torch.exp(-0.1378 * (V + 40.14)))))
        d_j = torch.zeros_like(y[:, 0]) + ((alg_3_j_infinity - j) / alg_14_tau_j * 0.001)
        alg_15_tau_r = 1.0 / (45.16 * torch.exp(0.03577 * (V + 50.0)) + 98.9 * torch.exp(-0.1 * (V + 38.0)))
        alg_4_r_infinity = 1.0 / (1.0 + torch.exp((V + 10.6) / -11.42))
        d_r = torch.zeros_like(y[:, 0]) + ((alg_4_r_infinity - r) / alg_15_tau_r * 0.001)
        alg_16_tau_s_endo = 0.55 * torch.exp(-((V + 70.0) / 25.0) ** 2.0) + 0.049
        alg_5_s_infinity = 1.0 / (1.0 + torch.exp((V + 45.3) / 6.8841))
        d_s = torch.zeros_like(y[:, 0]) + ((alg_5_s_infinity - s) / alg_16_tau_s_endo * 0.001)
        alg_17_tau_s_slow_endo = 3.3 * torch.exp(-(V + 70.0) / 30.0 * (V + 70.0) / 30.0) + 0.049
        alg_6_s_slow_infinity = 1.0 / (1.0 + torch.exp((V + 45.3) / 6.8841))
        d_s_slow = torch.zeros_like(y[:, 0]) + ((alg_6_s_slow_infinity - s_slow) / alg_17_tau_s_slow_endo * 0.001)
        alg_18_tau_r_ss = 10.0 / (45.16 * torch.exp(0.03577 * (V + 50.0)) + 98.9 * torch.exp(-0.1 * (V + 38.0)))
        alg_7_r_ss_infinity = 1.0 / (1.0 + torch.exp((V + 11.5) / -11.82))
        d_r_ss = torch.zeros_like(y[:, 0]) + ((alg_7_r_ss_infinity - r_ss) / alg_18_tau_r_ss * 0.001)
        alg_19_tau_y = 1.0 / (0.11885 * torch.exp((V + 80.0) / 28.37) + 0.5623 * torch.exp((V + 80.0) / -14.19))
        alg_9_y_infinity = 1.0 / (1.0 + torch.exp((V + 138.6) / 10.48))
        d_state_y = torch.zeros_like(y[:, 0]) + ((alg_9_y_infinity - state_y) / alg_19_tau_y * 0.001)
        alg_0_I_Stim = torch.where((t - torch.floor(t / stim_period) * stim_period >= 0.0) & (t - torch.floor(t / stim_period) * stim_period <= stim_duration), stim_amplitude, 0.0)
        alg_26_E_K = R * T / F * torch.log(K_o / K_i)
        alg_27_i_t = g_t_endo * r * (a_endo * s + b_endo * s_slow) * (V - alg_26_E_K)
        alg_28_i_ss = g_ss * r_ss * s_ss * (V - alg_26_E_K)
        alg_29_i_K1 = (48.0 / (torch.exp((V + 37.0) / 25.0) + torch.exp((V + 37.0) / -25.0)) + 10.0) * 0.001 / (1.0 + torch.exp((V - (alg_26_E_K + 76.77)) / -17.0)) + g_K1 * (V - (alg_26_E_K + 1.73)) / ((1.0 + torch.exp(1.613 * F * (V - (alg_26_E_K + 1.73)) / (R * T))) * (1.0 + torch.exp((K_o - 0.9988) / -0.124)))
        alg_35_i_B_K = g_B_K * (V - alg_26_E_K)
        alg_21_i_NaK = i_NaK_max * 1.0 / (1.0 + 0.1245 * torch.exp(-0.1 * V * F / (R * T)) + 0.0365 * sigma * torch.exp(-V * F / (R * T))) * K_o / (K_o + K_m_K) * 1.0 / (1.0 + (K_m_Na / Na_i) ** 4.0)
        alg_31_i_f_K = g_f * state_y * f_K * (V - alg_26_E_K)
        d_K_i = torch.zeros_like(y[:, 0]) + (-(alg_0_I_Stim + alg_28_i_ss * 0.001 + alg_35_i_B_K * 0.001 + alg_27_i_t * 0.001 + alg_29_i_K1 * 0.001 + alg_31_i_f_K * 0.001 + alg_21_i_NaK * -2.0) * 1.0 / (V_myo_uL * F))
        alg_24_E_Na = R * T / F * torch.log(Na_o / Na_i)
        alg_25_i_Na = g_Na_endo * m ** 3.0 * h * j * (V - alg_24_E_Na)
        alg_33_i_B_Na = g_B_Na * (V - alg_24_E_Na)
        alg_11_FVRT = F * V / (R * T)
        alg_37_I_NaCa = g_NCX * (torch.exp(eta * alg_11_FVRT) * Na_i ** 3.0 * Ca_o - torch.exp((eta - 1.0) * alg_11_FVRT) * Na_o ** 3.0 * Ca_i) / ((Na_o ** 3.0 + K_mNa ** 3.0) * (Ca_o + K_mCa) * (1.0 + k_sat * torch.exp((eta - 1.0) * alg_11_FVRT)))
        alg_38_I_NaCa_2 = alg_37_I_NaCa * V_myo_uL * F
        alg_30_i_f_Na = g_f * state_y * f_Na * (V - alg_24_E_Na)
        d_Na_i = torch.zeros_like(y[:, 0]) + (-(alg_25_i_Na * 0.001 + alg_33_i_B_Na * 0.001 + alg_38_I_NaCa_2 * 3.0 + alg_21_i_NaK * 3.0 + alg_30_i_f_Na * 0.001) * 1.0 / (V_myo_uL * F))
        alg_23_Ca_b = Ca_TRPN_Max - TRPN
        alg_48_ExtensionRatio = torch.where(t > 300000.0, 1.0, 1.0)
        alg_49_lambda_var = torch.where((alg_48_ExtensionRatio > 0.8) & (alg_48_ExtensionRatio <= 1.15), alg_48_ExtensionRatio, torch.where(alg_48_ExtensionRatio > 1.15, 1.15, 0.8))
        alg_51_Ca_50 = Ca_50ref * (1.0 + beta_1 * (alg_49_lambda_var - 1.0))
        alg_52_Ca_TRPN_50 = alg_51_Ca_50 * Ca_TRPN_Max / (alg_51_Ca_50 + k_Ref_off / k_on * (1.0 - (1.0 + beta_0 * (alg_49_lambda_var - 1.0)) * 0.5 / gamma_trpn))
        alg_53_alpha_Tm = alpha_0 * (alg_23_Ca_b / alg_52_Ca_TRPN_50) ** n_Hill
        alg_47_beta_Tm = alpha_r1 + alpha_r2 * z ** (n_Rel - 1.0) / (z ** n_Rel + K_z ** n_Rel)
        d_z = torch.zeros_like(y[:, 0]) + (alg_53_alpha_Tm * (1.0 - z) - alg_47_beta_Tm * z)
        alg_55_overlap = 1.0 + beta_0 * (alg_49_lambda_var - 1.0)
        alg_54_z_max = (alpha_0 / (alg_52_Ca_TRPN_50 / Ca_TRPN_Max) ** n_Hill - K_2) / (alpha_r1 + K_1 + alpha_0 / (alg_52_Ca_TRPN_50 / Ca_TRPN_Max) ** n_Hill)
        alg_56_T_Base = T_ref * z / alg_54_z_max
        alg_57_T_0 = alg_56_T_Base * alg_55_overlap
        alg_58_Q = Q_1 + Q_2 + Q_3
        alg_59_Tension = torch.where(alg_58_Q < 0.0, alg_57_T_0 * (a * alg_58_Q + 1.0) / (1.0 - alg_58_Q), alg_57_T_0 * (1.0 + (a + 2.0) * alg_58_Q) / (1.0 + alg_58_Q))
        alg_60_k_off = torch.where(1.0 - alg_59_Tension / (gamma_trpn * T_ref) > 0.1, k_Ref_off * (1.0 - alg_59_Tension / (gamma_trpn * T_ref)), k_Ref_off * 0.1)
        alg_61_J_TRPN = (Ca_TRPN_Max - TRPN) * alg_60_k_off - Ca_i * TRPN * k_on
        d_TRPN = torch.zeros_like(y[:, 0]) + (alg_61_J_TRPN)
        alg_20_FVRT_Ca = 2.0 * alg_11_FVRT
        alg_81_C_oc = torch.where(torch.abs(alg_20_FVRT_Ca) > 1e-09, (Ca_i + J_L / g_D * Ca_o * alg_20_FVRT_Ca * torch.exp(-alg_20_FVRT_Ca) / (1.0 - torch.exp(-alg_20_FVRT_Ca))) / (1.0 + J_L / g_D * alg_20_FVRT_Ca / (1.0 - torch.exp(-alg_20_FVRT_Ca))), (Ca_i + J_L / g_D * Ca_o) / (1.0 + J_L / g_D))
        alg_83_mu_poc = (alg_81_C_oc ** 2.0 + c * K_RyR ** 2.0) / (tau_R * (alg_81_C_oc ** 2.0 + K_RyR ** 2.0))
        alg_67_mu_pcc = (Ca_i ** 2.0 + c * K_RyR ** 2.0) / (tau_R * (Ca_i ** 2.0 + K_RyR ** 2.0))
        alg_62_expVL = torch.exp((V - V_L) / del_VL)
        alg_63_alpha_p = alg_62_expVL / (t_L * (alg_62_expVL + 1.0))
        alg_64_beta_pcc = Ca_i ** 2.0 / (t_R * (Ca_i ** 2.0 + K_RyR ** 2.0))
        alg_82_beta_poc = alg_81_C_oc ** 2.0 / (t_R * (alg_81_C_oc ** 2.0 + K_RyR ** 2.0))
        alg_85_denom = (alg_63_alpha_p + alpha_m) * ((alpha_m + beta_m + alg_82_beta_poc) * (beta_m + alg_64_beta_pcc) + alg_63_alpha_p * (beta_m + alg_82_beta_poc))
        alg_86_y_oc = alg_63_alpha_p * beta_m * (alg_63_alpha_p + alpha_m + beta_m + alg_64_beta_pcc) / alg_85_denom
        alg_90_y_cc = alpha_m * beta_m * (alpha_m + alg_63_alpha_p + beta_m + alg_82_beta_poc) / alg_85_denom
        alg_93_r_1 = alg_86_y_oc * alg_83_mu_poc + alg_90_y_cc * alg_67_mu_pcc
        alg_84_mu_moc = theta_R * d * (alg_81_C_oc ** 2.0 + c * K_RyR ** 2.0) / (tau_R * (d * alg_81_C_oc ** 2.0 + c * K_RyR ** 2.0))
        alg_68_mu_mcc = theta_R * d * (Ca_i ** 2.0 + c * K_RyR ** 2.0) / (tau_R * (d * Ca_i ** 2.0 + c * K_RyR ** 2.0))
        alg_95_r_2 = (alg_63_alpha_p * alg_84_mu_moc + alpha_m * alg_68_mu_mcc) / (alg_63_alpha_p + alpha_m)
        alg_65_epsilon_pcc = Ca_i * (alg_62_expVL + a_2) / (tau_L * K_L * (alg_62_expVL + 1.0))
        alg_76_r_7 = alpha_m * alg_65_epsilon_pcc / (alg_63_alpha_p + alpha_m)
        alg_66_epsilon_m = b * (alg_62_expVL + a_2) / (tau_L * (b * alg_62_expVL + a_2))
        alg_77_r_8 = alg_66_epsilon_m
        alg_78_z_4 = 1.0 - z_1 - z_2 - z_3
        d_z_2 = torch.zeros_like(y[:, 0]) + (alg_93_r_1 * z_1 - (alg_95_r_2 + alg_76_r_7) * z_2 + alg_77_r_8 * alg_78_z_4)
        alg_79_C_co = (Ca_i + J_R / g_D * Ca_SR) / (1.0 + J_R / g_D)
        alg_80_epsilon_pco = alg_79_C_co * (alg_62_expVL + a_2) / (tau_L * K_L * (alg_62_expVL + 1.0))
        alg_87_y_co = alpha_m * (alg_64_beta_pcc * (alpha_m + beta_m + alg_82_beta_poc) + alg_82_beta_poc * alg_63_alpha_p) / alg_85_denom
        alg_97_r_5 = alg_87_y_co * alg_80_epsilon_pco + alg_90_y_cc * alg_65_epsilon_pcc
        alg_75_r_6 = alg_66_epsilon_m
        d_z_1 = torch.zeros_like(y[:, 0]) + (-(alg_93_r_1 + alg_97_r_5) * z_1 + alg_95_r_2 * z_2 + alg_75_r_6 * z_3)
        alg_73_r_3 = beta_m * alg_67_mu_pcc / (beta_m + alg_64_beta_pcc)
        alg_74_r_4 = alg_68_mu_mcc
        d_z_3 = torch.zeros_like(y[:, 0]) + (alg_97_r_5 * z_1 - (alg_75_r_6 + alg_73_r_3) * z_3 + alg_74_r_4 * alg_78_z_4)
        alg_88_y_oo = alg_63_alpha_p * (alg_82_beta_poc * (alg_63_alpha_p + beta_m + alg_64_beta_pcc) + alg_64_beta_pcc * alpha_m) / alg_85_denom
        alg_89_J_Rco = J_R * (Ca_SR - Ca_i) / (1.0 + J_R / g_D)
        alg_91_J_Roo = torch.where(torch.abs(alg_20_FVRT_Ca) > 1e-05, J_R * (Ca_SR - Ca_i + J_L / g_D * alg_20_FVRT_Ca / (1.0 - torch.exp(-alg_20_FVRT_Ca)) * (Ca_SR - Ca_o * torch.exp(-alg_20_FVRT_Ca))) / (1.0 + J_R / g_D + J_L / g_D * alg_20_FVRT_Ca / (1.0 - torch.exp(-alg_20_FVRT_Ca))), J_R * (Ca_SR - Ca_i + J_L / g_D * 1e-05 / (1.0 - 0.9999900000499998) * (Ca_SR - Ca_o * 0.9999900000499998)) / (1.0 + J_R / g_D + J_L / g_D * 1e-05 / (1.0 - 0.9999900000499998)))
        alg_98_J_R1 = alg_88_y_oo * alg_91_J_Roo + alg_89_J_Rco * alg_87_y_co
        alg_99_J_R3 = alg_89_J_Rco * alg_64_beta_pcc / (beta_m + alg_64_beta_pcc)
        alg_100_I_RyR = (z_1 * alg_98_J_R1 + z_3 * alg_99_J_R3) * N / V_myo
        alg_101_I_RyR_2 = 1.5 * alg_100_I_RyR
        alg_39_I_SERCA = g_SERCA * Ca_i ** 2.0 / (K_SERCA ** 2.0 + Ca_i ** 2.0)
        alg_45_I_SR = g_SRl * (Ca_SR - Ca_i)
        d_Ca_SR = torch.zeros_like(y[:, 0]) + (V_myo_uL / V_SR_uL * (-alg_101_I_RyR_2 + alg_39_I_SERCA - alg_45_I_SR))
        alg_32_i_f = alg_30_i_f_Na + alg_31_i_f_K
        alg_96_J_Loo = torch.where(torch.abs(alg_20_FVRT_Ca) > 1e-05, J_L * alg_20_FVRT_Ca / (1.0 - torch.exp(-alg_20_FVRT_Ca)) * (Ca_o * torch.exp(-alg_20_FVRT_Ca) - Ca_i + J_R / g_D * (Ca_o * torch.exp(-alg_20_FVRT_Ca) - Ca_SR)) / (1.0 + J_R / g_D + J_L / g_D * alg_20_FVRT_Ca / (1.0 - torch.exp(alg_20_FVRT_Ca))), J_L * 1e-05 / (1.0 - 0.9999900000499998) * (Ca_o * 0.9999900000499998 - Ca_i + J_R / g_D * (Ca_o * 0.9999900000499998 - Ca_SR)) / (1.0 + J_R / g_D + J_L / g_D * 1e-05 / (1.0 - 0.9999900000499998)))
        alg_94_J_Loc = torch.where(torch.abs(alg_20_FVRT_Ca) > 1e-05, J_L * alg_20_FVRT_Ca / (1.0 - torch.exp(-alg_20_FVRT_Ca)) * (Ca_o * torch.exp(-alg_20_FVRT_Ca) - Ca_i) / (1.0 + J_L / g_D * alg_20_FVRT_Ca / (1.0 - torch.exp(-alg_20_FVRT_Ca))), J_L * 1e-05 / (1.0 - 0.9999900000499998) * (Ca_o * 0.9999900000499998 - Ca_i) / (1.0 + J_L / g_D * 1e-05 / (1.0 - 0.9999900000499998)))
        alg_102_J_L1 = alg_96_J_Loo * alg_88_y_oo + alg_94_J_Loc * alg_86_y_oc
        alg_103_J_L2 = alg_94_J_Loc * alg_63_alpha_p / (alg_63_alpha_p + alpha_m)
        alg_104_I_LCC = (z_1 * alg_102_J_L1 + z_2 * alg_103_J_L2) * N / V_myo
        alg_105_I_LCC_2 = -1.5 * alg_104_I_LCC * 2.0 * V_myo_uL * F
        alg_40_I_pCa = g_pCa * Ca_i / (K_mpCa + Ca_i)
        alg_41_I_pCa_2 = alg_40_I_pCa * 2.0 * V_myo_uL * F
        alg_42_E_Ca = R * T / (2.0 * F) * torch.log(Ca_o / Ca_i)
        alg_43_I_CaB = g_CaB * (alg_42_E_Ca - V)
        alg_44_I_CaB_2 = -alg_43_I_CaB * 2.0 * V_myo_uL * F
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_25_i_Na * 0.001 + alg_27_i_t * 0.001 + alg_28_i_ss * 0.001 + alg_32_i_f * 0.001 + alg_29_i_K1 * 0.001 + alg_33_i_B_Na * 0.001 + alg_35_i_B_K * 0.001 + alg_21_i_NaK + alg_0_I_Stim + alg_44_I_CaB_2 + alg_38_I_NaCa_2 + alg_41_I_pCa_2 + alg_105_I_LCC_2) / Cm)
        alg_46_beta_CMDN = (1.0 + k_CMDN * B_CMDN / (k_CMDN + Ca_i) ** 2.0) ** (-1.0)
        d_Ca_i = torch.zeros_like(y[:, 0]) + (alg_46_beta_CMDN * (alg_101_I_RyR_2 - alg_39_I_SERCA + alg_45_I_SR + alg_61_J_TRPN - (-2.0 * alg_38_I_NaCa_2 + alg_41_I_pCa_2 + alg_44_I_CaB_2 + alg_105_I_LCC_2) / (2.0 * V_myo_uL * F)))

        return torch.stack(
            [d_V, d_Na_i, d_Ca_i, d_Ca_SR, d_K_i, d_TRPN, d_m, d_h, d_j, d_r, d_s, d_s_slow, d_r_ss, d_s_ss, d_state_y, d_z, d_Q_1, d_Q_2, d_Q_3, d_z_1, d_z_2, d_z_3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
