import torch
import torch.nn as nn
import math


class ElectrophysiologyDemirClarkGiles1999ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/demir_clark_giles_1999/model.py.
    Dynamic states: 29; independent constants: 45; derived constants used by rates: 10.
    Original constant slots: 55; algebraic slots: 73.
    """
    source_file = 'electrophysiology/demir_clark_giles_1999/model.py'
    source_state_dim = 29
    source_constant_dim = 55
    algebraic_dim = 73
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 29
        self.parameter_dim = 45

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8314.472, 'T': 310, 'F': 96485.3415, 'Cm': 5.5e-05, 'P_Na': 0.00344, 'ACh': 0, 'g_Ca_L_cont': 0.02115, 'E_Ca_L': 46.4, 'g_Ca_T': 0.02521, 'E_Ca_T': 45, 'K_b': 5.4, 'g_B_Na': 0.00016, 'g_B_Ca': 3.64e-05, 'g_B_K': 6.94e-05, 'g_f_Na': 0.0067478, 'g_f_K': 0.0128821, 'K_m_Na': 5.46, 'K_m_K': 0.621, 'i_NaK_max': 0.2192, 'i_Ca_P_max': 0.02869, 'K_NaCa': 1.248e-05, 'd_NaCa': 0.0001, 'gamma': 0.5, 'g_K_ACh_base': 0.007833, 'f_Vagal': 200, 'Mg_i': 2.5, 'Vol': 3.497e-06, 'Na_b': 140, 'Ca_b': 2, 'tau_p': 0.01, 'alpha_up': 0.08, 'beta_up': 0.072, 'alpha_rel': 0.5, 'k_cyca': 5e-05, 'k_xcs': 0.9, 'k_SRCa': 22, 'k_rel': 0.004, 'cGMP': 0.002, 'Iso': 0, 'Km_Iso': 0.00014, 'Km_ACh': 0.00014, 'K_PDE': 6, 'K_ADC': 0.008, 'V_PDE': 20, 'P_M2_ADC': 0.02}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -49.54105, 'Na_c': 139.9988, 'm': 0.250113, 'h1': 0.001386897, 'h2': 0.002065463, 'cAMP': 0.003, 'd_L': 0.002572773, 'f_L': 0.98651, 'd_T': 0.02012114, 'f_T': 0.1945111, 'P_a': 0.02302278, 'P_i': 0.3777728, 'y': 0.09227776, 'Na_i': 9.701621, 'K_c': 5.389014, 'Ca_i': 0.0003787018, 'Ca_c': 2.00474, 'a': 0, 'K_i': 140.7347, 'Ca_Calmod': 0.1411678, 'Ca_Trop': 0.07331396, 'Ca_Mg_Trop': 0.7618549, 'Mg_Mg_Trop': 0.2097049, 'Ca_up': 16.95311, 'Ca_rel': 16.85024, 'Ca_Calse': 0.9528726, 'F1': 0.1133251, 'F2': 0.0007594214, 'F3': 0.8859153}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'Cm', 'P_Na', 'ACh', 'g_Ca_L_cont', 'E_Ca_L', 'g_Ca_T', 'E_Ca_T', 'K_b', 'g_B_Na', 'g_B_Ca', 'g_B_K', 'g_f_Na', 'g_f_K', 'K_m_Na', 'K_m_K', 'i_NaK_max', 'i_Ca_P_max', 'K_NaCa', 'd_NaCa', 'gamma', 'g_K_ACh_base', 'f_Vagal', 'Mg_i', 'Vol', 'Na_b', 'Ca_b', 'tau_p', 'alpha_up', 'beta_up', 'alpha_rel', 'k_cyca', 'k_xcs', 'k_SRCa', 'k_rel', 'cGMP', 'Iso', 'Km_Iso', 'Km_ACh', 'K_PDE', 'K_ADC', 'V_PDE', 'P_M2_ADC']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'Na_c', 'm', 'h1', 'h2', 'cAMP', 'd_L', 'f_L', 'd_T', 'f_T', 'P_a', 'P_i', 'y', 'Na_i', 'K_c', 'Ca_i', 'Ca_c', 'a', 'K_i', 'Ca_Calmod', 'Ca_Trop', 'Ca_Mg_Trop', 'Mg_Mg_Trop', 'Ca_up', 'Ca_rel', 'Ca_Calse', 'F1', 'F2', 'F3']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (29)
        V = y[:, 0]
        Na_c = y[:, 1]
        m = y[:, 2]
        h1 = y[:, 3]
        h2 = y[:, 4]
        cAMP = y[:, 5]
        d_L = y[:, 6]
        f_L = y[:, 7]
        d_T = y[:, 8]
        f_T = y[:, 9]
        P_a = y[:, 10]
        P_i = y[:, 11]
        state_y = y[:, 12]
        Na_i = y[:, 13]
        K_c = y[:, 14]
        Ca_i = y[:, 15]
        Ca_c = y[:, 16]
        a = y[:, 17]
        K_i = y[:, 18]
        Ca_Calmod = y[:, 19]
        Ca_Trop = y[:, 20]
        Ca_Mg_Trop = y[:, 21]
        Mg_Mg_Trop = y[:, 22]
        Ca_up = y[:, 23]
        Ca_rel = y[:, 24]
        Ca_Calse = y[:, 25]
        F1 = y[:, 26]
        F2 = y[:, 27]
        F3 = y[:, 28]

        # Independent parameters (source constant order)
        R = theta[:, 0]  # constants[0]
        T = theta[:, 1]  # constants[1]
        F = theta[:, 2]  # constants[2]
        Cm = theta[:, 3]  # constants[3]
        P_Na = theta[:, 4]  # constants[4]
        ACh = theta[:, 5]  # constants[5]
        g_Ca_L_cont = theta[:, 6]  # constants[6]
        E_Ca_L = theta[:, 7]  # constants[7]
        g_Ca_T = theta[:, 8]  # constants[8]
        E_Ca_T = theta[:, 9]  # constants[9]
        K_b = theta[:, 10]  # constants[10]
        g_B_Na = theta[:, 11]  # constants[11]
        g_B_Ca = theta[:, 12]  # constants[12]
        g_B_K = theta[:, 13]  # constants[13]
        g_f_Na = theta[:, 14]  # constants[14]
        g_f_K = theta[:, 15]  # constants[15]
        K_m_Na = theta[:, 16]  # constants[16]
        K_m_K = theta[:, 17]  # constants[17]
        i_NaK_max = theta[:, 18]  # constants[18]
        i_Ca_P_max = theta[:, 19]  # constants[19]
        K_NaCa = theta[:, 20]  # constants[20]
        d_NaCa = theta[:, 21]  # constants[21]
        gamma = theta[:, 22]  # constants[22]
        g_K_ACh_base = theta[:, 23]  # constants[23]
        f_Vagal = theta[:, 24]  # constants[24]
        Mg_i = theta[:, 25]  # constants[25]
        Vol = theta[:, 26]  # constants[26]
        Na_b = theta[:, 27]  # constants[27]
        Ca_b = theta[:, 28]  # constants[28]
        tau_p = theta[:, 29]  # constants[29]
        alpha_up = theta[:, 30]  # constants[30]
        beta_up = theta[:, 31]  # constants[31]
        alpha_rel = theta[:, 32]  # constants[32]
        k_cyca = theta[:, 33]  # constants[33]
        k_xcs = theta[:, 34]  # constants[34]
        k_SRCa = theta[:, 35]  # constants[35]
        k_rel = theta[:, 36]  # constants[36]
        cGMP = theta[:, 37]  # constants[37]
        Iso = theta[:, 38]  # constants[38]
        Km_Iso = theta[:, 39]  # constants[39]
        Km_ACh = theta[:, 40]  # constants[40]
        K_PDE = theta[:, 41]  # constants[41]
        K_ADC = theta[:, 42]  # constants[42]
        V_PDE = theta[:, 43]  # constants[43]
        P_M2_ADC = theta[:, 44]  # constants[44]

        # Derived constants from the original initConsts()
        F_ACh_Na = 1.0 - ACh / (ACh + 0.001)  # constants[45]
        F_ACh_bNa = 1.0 - ACh / (ACh + 0.5)  # constants[46]
        P_M2_KACh = torch.where((f_Vagal < 100.0) & (f_Vagal > 25.0), 1.026 / (1.0 + torch.exp((f_Vagal + 11.05) / -7.5095)) - 0.99, 0.0006)  # constants[47]
        beta_a = 12.32 / (1.0 + 0.0042 / ACh)  # constants[48]
        V_i = 0.465 * Vol  # constants[49]
        V_c = 0.136 * Vol  # constants[50]
        K1 = k_cyca * k_xcs / k_SRCa  # constants[51]
        g_K_ACh = P_M2_KACh * g_K_ACh_base  # constants[52]
        V_up = 0.01166 * V_i  # constants[53]
        V_rel = 0.001296 * V_i  # constants[54]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_Na_c = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h1 = torch.zeros_like(y[:, 0])
        d_h2 = torch.zeros_like(y[:, 0])
        d_cAMP = torch.zeros_like(y[:, 0])
        d_d_L = torch.zeros_like(y[:, 0])
        d_f_L = torch.zeros_like(y[:, 0])
        d_d_T = torch.zeros_like(y[:, 0])
        d_f_T = torch.zeros_like(y[:, 0])
        d_P_a = torch.zeros_like(y[:, 0])
        d_P_i = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_K_c = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_Ca_c = torch.zeros_like(y[:, 0])
        d_a = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_Ca_Calmod = torch.zeros_like(y[:, 0])
        d_Ca_Trop = torch.zeros_like(y[:, 0])
        d_Ca_Mg_Trop = torch.zeros_like(y[:, 0])
        d_Mg_Mg_Trop = torch.zeros_like(y[:, 0])
        d_Ca_up = torch.zeros_like(y[:, 0])
        d_Ca_rel = torch.zeros_like(y[:, 0])
        d_Ca_Calse = torch.zeros_like(y[:, 0])
        d_F1 = torch.zeros_like(y[:, 0])
        d_F2 = torch.zeros_like(y[:, 0])
        d_F3 = torch.zeros_like(y[:, 0])
        alg_0_alpha_m = torch.zeros_like(y[:, 0])
        alg_1_alpha_h1 = torch.zeros_like(y[:, 0])
        alg_2_alpha_f_L = torch.zeros_like(y[:, 0])
        alg_3_alpha_d_T = torch.zeros_like(y[:, 0])
        alg_4_alpha_f_T = torch.zeros_like(y[:, 0])
        alg_5_P_a_infinity = torch.zeros_like(y[:, 0])
        alg_6_alpha_P_i = torch.zeros_like(y[:, 0])
        alg_7_V_half = torch.zeros_like(y[:, 0])
        alg_8_alpha_a = torch.zeros_like(y[:, 0])
        alg_9_F_cAMP_CaL = torch.zeros_like(y[:, 0])
        alg_10_phi_TMgM = torch.zeros_like(y[:, 0])
        alg_11_r_act = torch.zeros_like(y[:, 0])
        alg_12_beta_m = torch.zeros_like(y[:, 0])
        alg_13_beta_h1 = torch.zeros_like(y[:, 0])
        alg_14_beta_f_L = torch.zeros_like(y[:, 0])
        alg_15_beta_d_T = torch.zeros_like(y[:, 0])
        alg_16_beta_f_T = torch.zeros_like(y[:, 0])
        alg_17_tau_P_a = torch.zeros_like(y[:, 0])
        alg_18_beta_P_i = torch.zeros_like(y[:, 0])
        alg_19_y_infinity = torch.zeros_like(y[:, 0])
        alg_20_g_Ca_L = torch.zeros_like(y[:, 0])
        alg_21_r_inact = torch.zeros_like(y[:, 0])
        alg_22_m_infinity = torch.zeros_like(y[:, 0])
        alg_23_h1_infinity = torch.zeros_like(y[:, 0])
        alg_24_tau_f_L = torch.zeros_like(y[:, 0])
        alg_25_tau_d_T = torch.zeros_like(y[:, 0])
        alg_26_tau_f_T = torch.zeros_like(y[:, 0])
        alg_27_tau_y = torch.zeros_like(y[:, 0])
        alg_28_alpha_d_L = torch.zeros_like(y[:, 0])
        alg_29_tau_m = torch.zeros_like(y[:, 0])
        alg_30_tau_h1 = torch.zeros_like(y[:, 0])
        alg_31_f_L_infinity = torch.zeros_like(y[:, 0])
        alg_32_d_T_infinity = torch.zeros_like(y[:, 0])
        alg_33_f_T_infinity = torch.zeros_like(y[:, 0])
        alg_34_beta_d_L = torch.zeros_like(y[:, 0])
        alg_35_h2_infinity = torch.zeros_like(y[:, 0])
        alg_36_tau_d_L = torch.zeros_like(y[:, 0])
        alg_37_tau_h2 = torch.zeros_like(y[:, 0])
        alg_38_d_L_infinity = torch.zeros_like(y[:, 0])
        alg_39_i_Ca_L = torch.zeros_like(y[:, 0])
        alg_40_i_Ca_T = torch.zeros_like(y[:, 0])
        alg_41_F_cAMP_K = torch.zeros_like(y[:, 0])
        alg_42_g_K = torch.zeros_like(y[:, 0])
        alg_43_i_f_Na = torch.zeros_like(y[:, 0])
        alg_44_i_f_K = torch.zeros_like(y[:, 0])
        alg_45_i_f = torch.zeros_like(y[:, 0])
        alg_46_F_cAMP_NaK = torch.zeros_like(y[:, 0])
        alg_47_i_NaK = torch.zeros_like(y[:, 0])
        alg_48_i_Ca_P = torch.zeros_like(y[:, 0])
        alg_49_i_NaCa = torch.zeros_like(y[:, 0])
        alg_50_phi_C = torch.zeros_like(y[:, 0])
        alg_51_phi_TC = torch.zeros_like(y[:, 0])
        alg_52_phi_TMgC = torch.zeros_like(y[:, 0])
        alg_53_F_C = torch.zeros_like(y[:, 0])
        alg_54_F_TC = torch.zeros_like(y[:, 0])
        alg_55_F_TMgC = torch.zeros_like(y[:, 0])
        alg_56_phi_B = torch.zeros_like(y[:, 0])
        alg_57_K2 = torch.zeros_like(y[:, 0])
        alg_58_phi_Calse = torch.zeros_like(y[:, 0])
        alg_59_i_rel = torch.zeros_like(y[:, 0])
        alg_60_i_tr = torch.zeros_like(y[:, 0])
        alg_61_i_up = torch.zeros_like(y[:, 0])
        alg_62_E_Na = torch.zeros_like(y[:, 0])
        alg_63_i_Na = torch.zeros_like(y[:, 0])
        alg_64_i_B_Na = torch.zeros_like(y[:, 0])
        alg_65_E_K = torch.zeros_like(y[:, 0])
        alg_66_i_K = torch.zeros_like(y[:, 0])
        alg_67_i_B_K = torch.zeros_like(y[:, 0])
        alg_68_I_K_ACh = torch.zeros_like(y[:, 0])
        alg_69_i_K_ACh = torch.zeros_like(y[:, 0])
        alg_70_E_Ca = torch.zeros_like(y[:, 0])
        alg_71_i_B_Ca = torch.zeros_like(y[:, 0])
        alg_72_i_B = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_cAMP = torch.zeros_like(y[:, 0]) + (1.0 * (K_ADC * (1.0 + Iso / (Iso + Km_Iso) - P_M2_ADC * ACh / (P_M2_ADC * ACh + Km_ACh)) - V_PDE * cGMP * cAMP / (cAMP + K_PDE * cGMP)))
        alg_8_alpha_a = 17.0 * torch.exp(0.0133 * (V + 40.0))
        d_a = torch.zeros_like(y[:, 0]) + (beta_a * (1.0 - a) - alg_8_alpha_a * a)
        alg_10_phi_TMgM = 1290.0 * Mg_i * (1.0 - (Ca_Mg_Trop + Mg_Mg_Trop)) - 429.0 * Mg_Mg_Trop
        d_Mg_Mg_Trop = torch.zeros_like(y[:, 0]) + (alg_10_phi_TMgM)
        alg_11_r_act = 240.0 * torch.exp((V - 40.0) * 0.08) + 240.0 * (Ca_i / (Ca_i + k_rel)) ** 4.0
        d_F1 = torch.zeros_like(y[:, 0]) + (0.96 * F3 - alg_11_r_act * F1)
        alg_17_tau_P_a = 1.0 / (17.0 * torch.exp(0.0398 * V) + 2.11 * torch.exp(-0.051 * V))
        alg_5_P_a_infinity = 1.0 / (1.0 + torch.exp((V + 5.1) / -7.4))
        d_P_a = torch.zeros_like(y[:, 0]) + ((alg_5_P_a_infinity - P_a) / alg_17_tau_P_a)
        alg_6_alpha_P_i = 100.0 * torch.exp(-0.0183 * V)
        alg_18_beta_P_i = 656.0 * torch.exp(0.00942 * V)
        d_P_i = torch.zeros_like(y[:, 0]) + (alg_6_alpha_P_i * (1.0 - P_i) - alg_18_beta_P_i * P_i)
        alg_21_r_inact = 40.0 + 240.0 * (Ca_i / (Ca_i + k_rel)) ** 4.0
        d_F2 = torch.zeros_like(y[:, 0]) + (alg_11_r_act * F1 - alg_21_r_inact * F2)
        d_F3 = torch.zeros_like(y[:, 0]) + (alg_21_r_inact * F2 - 0.96 * F3)
        alg_7_V_half = 20.5 / (1.0 + torch.exp((cAMP - 0.0034) / -0.0005)) - 78.56
        alg_19_y_infinity = 1.0 / (1.0 + torch.exp((V - alg_7_V_half) / 9.0))
        alg_27_tau_y = 1.0 / (1.6483 * torch.exp((V + 54.06) / -24.33) + 14.0106 / (0.7 + torch.exp((V + 60.0) / -5.5)))
        d_state_y = torch.zeros_like(y[:, 0]) + ((alg_19_y_infinity - state_y) / alg_27_tau_y)
        alg_0_alpha_m = -824.0 * (V + 51.9) / (torch.exp((V + 51.9) / -8.9) - 1.0)
        alg_12_beta_m = 32960.0 * torch.exp((V + 51.9) / -8.9)
        alg_22_m_infinity = alg_0_alpha_m / (alg_0_alpha_m + alg_12_beta_m)
        alg_29_tau_m = 1.0 / (alg_0_alpha_m + alg_12_beta_m) + 1.5e-05
        d_m = torch.zeros_like(y[:, 0]) + ((alg_22_m_infinity - m) / alg_29_tau_m)
        alg_1_alpha_h1 = 165.0 * torch.exp((V + 101.3) / -12.6)
        alg_13_beta_h1 = 12360.0 / (320.0 * torch.exp((V + 101.3) / -12.6) + 1.0)
        alg_23_h1_infinity = alg_1_alpha_h1 / (alg_1_alpha_h1 + alg_13_beta_h1)
        alg_30_tau_h1 = 1.0 / (alg_1_alpha_h1 + alg_13_beta_h1)
        d_h1 = torch.zeros_like(y[:, 0]) + ((alg_23_h1_infinity - h1) / alg_30_tau_h1)
        alg_31_f_L_infinity = 1.0 / (1.0 + torch.exp((V + 30.0) / 5.0))
        alg_2_alpha_f_L = 3.75 * (V + 28.0) / (torch.exp((V + 28.0) / 4.0) - 1.0)
        alg_14_beta_f_L = 30.0 / (1.0 + torch.exp((V + 28.0) / -4.0))
        alg_24_tau_f_L = 1.0 / (alg_2_alpha_f_L + alg_14_beta_f_L)
        d_f_L = torch.zeros_like(y[:, 0]) + ((alg_31_f_L_infinity - f_L) / alg_24_tau_f_L)
        alg_32_d_T_infinity = 1.0 / (1.0 + torch.exp((V + 26.3) / -6.0))
        alg_3_alpha_d_T = 1068.0 * torch.exp((V + 26.3) / 30.0)
        alg_15_beta_d_T = 1068.0 * torch.exp((V + 26.3) / -30.0)
        alg_25_tau_d_T = 1.0 / (alg_3_alpha_d_T + alg_15_beta_d_T)
        d_d_T = torch.zeros_like(y[:, 0]) + ((alg_32_d_T_infinity - d_T) / alg_25_tau_d_T)
        alg_33_f_T_infinity = 1.0 / (1.0 + torch.exp((V + 61.7) / 5.6))
        alg_4_alpha_f_T = 15.3 * torch.exp((V + 61.7) / -83.3)
        alg_16_beta_f_T = 15.0 * torch.exp((V + 61.7) / 15.38)
        alg_26_tau_f_T = 1.0 / (alg_4_alpha_f_T + alg_16_beta_f_T)
        d_f_T = torch.zeros_like(y[:, 0]) + ((alg_33_f_T_infinity - f_T) / alg_26_tau_f_T)
        alg_35_h2_infinity = alg_23_h1_infinity
        alg_37_tau_h2 = 20.0 * alg_30_tau_h1
        d_h2 = torch.zeros_like(y[:, 0]) + ((alg_35_h2_infinity - h2) / alg_37_tau_h2)
        alg_38_d_L_infinity = 1.0 / (1.0 + torch.exp((V + 14.1) / -6.0))
        alg_28_alpha_d_L = -28.39 * (V + 35.0) / (torch.exp((V + 35.0) / -2.5) - 1.0) + -84.9 * V / (torch.exp(-0.208 * V) - 1.0)
        alg_34_beta_d_L = 11.43 * (V - 5.0) / (torch.exp(0.4 * (V - 5.0)) - 1.0)
        alg_36_tau_d_L = 1.0 / (alg_28_alpha_d_L + alg_34_beta_d_L)
        d_d_L = torch.zeros_like(y[:, 0]) + ((alg_38_d_L_infinity - d_L) / alg_36_tau_d_L)
        alg_50_phi_C = 129000.0 * Ca_i * (1.0 - Ca_Calmod) - 307.0 * Ca_Calmod
        d_Ca_Calmod = torch.zeros_like(y[:, 0]) + (alg_50_phi_C)
        alg_51_phi_TC = 50500.0 * Ca_i * (1.0 - Ca_Trop) - 252.0 * Ca_Trop
        d_Ca_Trop = torch.zeros_like(y[:, 0]) + (alg_51_phi_TC)
        alg_52_phi_TMgC = 129000.0 * Ca_i * (1.0 - (Ca_Mg_Trop + Mg_Mg_Trop)) - 4.25 * Ca_Mg_Trop
        d_Ca_Mg_Trop = torch.zeros_like(y[:, 0]) + (alg_52_phi_TMgC)
        alg_58_phi_Calse = 770.0 * Ca_rel * (1.0 - Ca_Calse) - 641.0 * Ca_Calse
        d_Ca_Calse = torch.zeros_like(y[:, 0]) + (alg_58_phi_Calse)
        alg_59_i_rel = alpha_rel * (F2 / (F2 + 0.25)) ** 2.0 * Ca_rel
        alg_60_i_tr = (Ca_up - Ca_rel) * 2.0 * F * V_up / 0.06418
        d_Ca_rel = torch.zeros_like(y[:, 0]) + ((alg_60_i_tr - alg_59_i_rel) / (2.0 * V_rel * F) - 11.48 * alg_58_phi_Calse)
        alg_57_K2 = Ca_i + Ca_up * K1 + k_cyca * k_xcs + k_cyca
        alg_61_i_up = (alpha_up * Ca_i - beta_up * Ca_up * K1) / alg_57_K2
        d_Ca_up = torch.zeros_like(y[:, 0]) + ((alg_61_i_up - alg_60_i_tr) / (2.0 * V_up * F))
        alg_62_E_Na = R * T / F * torch.log(Na_c / Na_i)
        alg_63_i_Na = F_ACh_Na * P_Na * m ** 3.0 * h1 * h2 * Na_c * V * F ** 2.0 / (R * T) * (torch.exp((V - alg_62_E_Na) * F / (R * T)) - 1.0) / (torch.exp(V * F / (R * T)) - 1.0)
        alg_46_F_cAMP_NaK = 1.6 / (1.0 + torch.exp((cAMP - 0.00375) / -0.00015)) + 0.99
        alg_47_i_NaK = torch.where(Ca_i < 0.00015, (2.0 - alg_46_F_cAMP_NaK) * i_NaK_max * (Na_i / (K_m_Na + Na_i)) ** 3.0 * (K_c / (K_m_K + K_c)) ** 2.0 * 1.6 / (1.5 + torch.exp((V + 60.0) / -40.0)), alg_46_F_cAMP_NaK * i_NaK_max * (Na_i / (K_m_Na + Na_i)) ** 3.0 * (K_c / (K_m_K + K_c)) ** 2.0 * 1.6 / (1.5 + torch.exp((V + 60.0) / -40.0)))
        alg_49_i_NaCa = K_NaCa * (Na_i ** 3.0 * Ca_c * torch.exp(0.03743 * V * gamma) - Na_c ** 3.0 * Ca_i * torch.exp(0.03743 * V * (gamma - 1.0))) / (1.0 + d_NaCa * (Ca_i * Na_c ** 3.0 + Ca_c * Na_i ** 3.0))
        alg_64_i_B_Na = F_ACh_bNa * g_B_Na * (V - alg_62_E_Na)
        alg_43_i_f_Na = g_f_Na * state_y ** 2.0 * (V - 75.0)
        d_Na_i = torch.zeros_like(y[:, 0]) + (-(3.0 * alg_47_i_NaK + 3.0 * alg_49_i_NaCa + alg_64_i_B_Na + alg_43_i_f_Na + alg_63_i_Na) / (F * V_i))
        d_Na_c = torch.zeros_like(y[:, 0]) + ((Na_b - Na_c) / tau_p + (alg_63_i_Na + 3.0 * alg_49_i_NaCa + 3.0 * alg_47_i_NaK + alg_64_i_B_Na + alg_43_i_f_Na) / (F * V_c))
        alg_41_F_cAMP_K = 0.62 * (1.0 + 2.6129 * cAMP / (cAMP + 0.009)) - 0.025
        alg_42_g_K = alg_41_F_cAMP_K * 0.00693 * (K_b / 1.0) ** 0.59
        alg_65_E_K = R * T / F * torch.log(K_c / K_i)
        alg_66_i_K = alg_42_g_K * P_a * P_i * (V - alg_65_E_K)
        alg_67_i_B_K = g_B_K * (V - alg_65_E_K)
        alg_44_i_f_K = g_f_K * state_y ** 2.0 * (V + 85.0)
        d_K_i = torch.zeros_like(y[:, 0]) + ((2.0 * alg_47_i_NaK - (alg_66_i_K + alg_44_i_f_K + alg_67_i_B_K)) / (F * V_i))
        d_K_c = torch.zeros_like(y[:, 0]) + ((K_b - K_c) / tau_p + (-2.0 * alg_47_i_NaK + alg_66_i_K + alg_67_i_B_K + alg_44_i_f_K) / (F * V_c))
        alg_40_i_Ca_T = g_Ca_T * d_T * f_T * (V - E_Ca_T)
        alg_9_F_cAMP_CaL = 0.4 * (1.0 + 4.5 * cAMP / (cAMP + 0.0065)) + 0.03157
        alg_20_g_Ca_L = g_Ca_L_cont * alg_9_F_cAMP_CaL
        alg_39_i_Ca_L = alg_20_g_Ca_L * (f_L * d_L + 0.095 * alg_38_d_L_infinity) * (V - E_Ca_L)
        alg_48_i_Ca_P = i_Ca_P_max * Ca_i / (Ca_i + 0.0004)
        alg_70_E_Ca = 0.5 * R * T / F * torch.log(Ca_c / Ca_i)
        alg_71_i_B_Ca = g_B_Ca * (V - alg_70_E_Ca)
        alg_53_F_C = 0.09 * alg_50_phi_C
        alg_54_F_TC = 0.031 * alg_51_phi_TC
        alg_55_F_TMgC = 0.062 * alg_52_phi_TMgC
        alg_56_phi_B = alg_53_F_C + alg_54_F_TC + alg_55_F_TMgC
        d_Ca_i = torch.zeros_like(y[:, 0]) + ((2.0 * alg_49_i_NaCa + alg_59_i_rel - (alg_39_i_Ca_L + alg_40_i_Ca_T + alg_48_i_Ca_P + alg_71_i_B_Ca + alg_61_i_up)) / (2.0 * V_i * F) - alg_56_phi_B)
        d_Ca_c = torch.zeros_like(y[:, 0]) + ((Ca_b - Ca_c) / tau_p + (-2.0 * alg_49_i_NaCa + alg_39_i_Ca_L + alg_40_i_Ca_T + alg_48_i_Ca_P + alg_71_i_B_Ca) / (2.0 * F * V_c))
        alg_45_i_f = alg_43_i_f_Na + alg_44_i_f_K
        alg_72_i_B = alg_64_i_B_Na + alg_71_i_B_Ca + alg_67_i_B_K
        alg_68_I_K_ACh = g_K_ACh * (V - alg_65_E_K)
        alg_69_i_K_ACh = a * alg_68_I_K_ACh
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_63_i_Na + alg_40_i_Ca_T + alg_39_i_Ca_L + alg_66_i_K + alg_45_i_f + alg_72_i_B + alg_47_i_NaK + alg_49_i_NaCa + alg_48_i_Ca_P + alg_69_i_K_ACh) / Cm)

        return torch.stack(
            [d_V, d_Na_c, d_m, d_h1, d_h2, d_cAMP, d_d_L, d_f_L, d_d_T, d_f_T, d_P_a, d_P_i, d_state_y, d_Na_i, d_K_c, d_Ca_i, d_Ca_c, d_a, d_K_i, d_Ca_Calmod, d_Ca_Trop, d_Ca_Mg_Trop, d_Mg_Mg_Trop, d_Ca_up, d_Ca_rel, d_Ca_Calse, d_F1, d_F2, d_F3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
