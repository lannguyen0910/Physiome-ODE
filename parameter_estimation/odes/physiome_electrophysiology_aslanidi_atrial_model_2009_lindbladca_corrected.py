import torch
import torch.nn as nn
import math


class ElectrophysiologyAslanidiAtrialModel2009LindbladCaCorrectedODE(nn.Module):
    """
    PyTorch translation of electrophysiology/aslanidi_atrial_model_2009_LindbladCa_corrected/model.py.
    Dynamic states: 29; independent constants: 49; derived constants used by rates: 0.
    Original constant slots: 49; algebraic slots: 78.
    """
    source_file = 'electrophysiology/aslanidi_atrial_model_2009_LindbladCa_corrected/model.py'
    source_state_dim = 29
    source_constant_dim = 49
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

        self.state_dim = 29
        self.parameter_dim = 49

        self.parameters = (
            parameters
            if parameters is not None
            else {'CT': 1, 'PM': 0, 'R': 8314, 'T': 308, 'F': 96487, 'Cm': 5e-05, 'stim_start': 0.01, 'stim_end': 100, 'stim_period': 0.5, 'stim_duration': 0.0002, 'stim_amplitude': -20, 'P_Na': 1.4e-06, 'Na_c': 140, 'g_Ca_L': 0.004, 'E_Ca_app': 50, 'g_Ca_T': 0.006, 'E_Ca_T': 38, 'g_to': 0.050002, 'g_Ks': 0.0025, 'g_Kr': 0.0035, 'g_K1': 0.00508, 'KmK1': 0.59, 'steepK1': 1.393, 'shiftK1': -3.6, 'g_B_Na': 6.4e-05, 'g_B_Ca': 3.1e-05, 'Ca_c': 2.5, 'k_NaK_K': 1, 'k_NaK_Na': 11, 'i_NaK_max': 0.06441, 'i_CaP_max': 0.009509, 'k_CaP': 0.0002, 'k_NaCa': 2e-05, 'd_NaCa': 0.0003, 'gamma': 0.45, 'Vol_i': 1.26e-05, 'Vol_Ca': 5.884e-06, 'Mg_i': 2.5, 'Vol_c': 2.5e-06, 'I_up_max': 2.8, 'k_cyca': 0.0003, 'k_srca': 0.5, 'k_xcs': 0.4, 'alpha_rel': 200, 'Vol_up': 3.969e-07, 'Vol_rel': 4.4e-08, 'tau_tr': 0.01, 'k_rel': 0.0003, 'k_F3': 0.815}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -80, 'Na_i': 8.4, 'm': 0.01309, 'h1': 0.706, 'h2': 0.61493, 'd_L': 3e-05, 'f_L': 0.99981, 'd_T': 0.00046, 'f_T': 0.30752, 'K_c': 5, 'K_i': 100, 'r': 6e-05, 's1': 0.5753, 's2': 0.39871, 's3': 0.57363, 'z': 0.02032, 'p_a': 0.00016, 'p_i': 0.76898, 'Ca_i': 7.1e-05, 'O_C': 0.029108, 'O_TC': 0.014071, 'O_TMgC': 0.214036, 'O_TMgMg': 0.693565, 'Ca_rel': 0.726776, 'Ca_up': 0.730866, 'O_Calse': 0.465921, 'F1': 0.288039, 'F2': 0.002262, 'F3': 0.612697}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['CT', 'PM', 'R', 'T', 'F', 'Cm', 'stim_start', 'stim_end', 'stim_period', 'stim_duration', 'stim_amplitude', 'P_Na', 'Na_c', 'g_Ca_L', 'E_Ca_app', 'g_Ca_T', 'E_Ca_T', 'g_to', 'g_Ks', 'g_Kr', 'g_K1', 'KmK1', 'steepK1', 'shiftK1', 'g_B_Na', 'g_B_Ca', 'Ca_c', 'k_NaK_K', 'k_NaK_Na', 'i_NaK_max', 'i_CaP_max', 'k_CaP', 'k_NaCa', 'd_NaCa', 'gamma', 'Vol_i', 'Vol_Ca', 'Mg_i', 'Vol_c', 'I_up_max', 'k_cyca', 'k_srca', 'k_xcs', 'alpha_rel', 'Vol_up', 'Vol_rel', 'tau_tr', 'k_rel', 'k_F3']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'Na_i', 'm', 'h1', 'h2', 'd_L', 'f_L', 'd_T', 'f_T', 'K_c', 'K_i', 'r', 's1', 's2', 's3', 'z', 'p_a', 'p_i', 'Ca_i', 'O_C', 'O_TC', 'O_TMgC', 'O_TMgMg', 'Ca_rel', 'Ca_up', 'O_Calse', 'F1', 'F2', 'F3']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (29)
        V = y[:, 0]
        Na_i = y[:, 1]
        m = y[:, 2]
        h1 = y[:, 3]
        h2 = y[:, 4]
        d_L = y[:, 5]
        f_L = y[:, 6]
        d_T = y[:, 7]
        f_T = y[:, 8]
        K_c = y[:, 9]
        K_i = y[:, 10]
        r = y[:, 11]
        s1 = y[:, 12]
        s2 = y[:, 13]
        s3 = y[:, 14]
        z = y[:, 15]
        p_a = y[:, 16]
        p_i = y[:, 17]
        Ca_i = y[:, 18]
        O_C = y[:, 19]
        O_TC = y[:, 20]
        O_TMgC = y[:, 21]
        O_TMgMg = y[:, 22]
        Ca_rel = y[:, 23]
        Ca_up = y[:, 24]
        O_Calse = y[:, 25]
        F1 = y[:, 26]
        F2 = y[:, 27]
        F3 = y[:, 28]

        # Independent parameters (source constant order)
        CT = theta[:, 0]  # constants[0]
        PM = theta[:, 1]  # constants[1]
        R = theta[:, 2]  # constants[2]
        T = theta[:, 3]  # constants[3]
        F = theta[:, 4]  # constants[4]
        Cm = theta[:, 5]  # constants[5]
        stim_start = theta[:, 6]  # constants[6]
        stim_end = theta[:, 7]  # constants[7]
        stim_period = theta[:, 8]  # constants[8]
        stim_duration = theta[:, 9]  # constants[9]
        stim_amplitude = theta[:, 10]  # constants[10]
        P_Na = theta[:, 11]  # constants[11]
        Na_c = theta[:, 12]  # constants[12]
        g_Ca_L = theta[:, 13]  # constants[13]
        E_Ca_app = theta[:, 14]  # constants[14]
        g_Ca_T = theta[:, 15]  # constants[15]
        E_Ca_T = theta[:, 16]  # constants[16]
        g_to = theta[:, 17]  # constants[17]
        g_Ks = theta[:, 18]  # constants[18]
        g_Kr = theta[:, 19]  # constants[19]
        g_K1 = theta[:, 20]  # constants[20]
        KmK1 = theta[:, 21]  # constants[21]
        steepK1 = theta[:, 22]  # constants[22]
        shiftK1 = theta[:, 23]  # constants[23]
        g_B_Na = theta[:, 24]  # constants[24]
        g_B_Ca = theta[:, 25]  # constants[25]
        Ca_c = theta[:, 26]  # constants[26]
        k_NaK_K = theta[:, 27]  # constants[27]
        k_NaK_Na = theta[:, 28]  # constants[28]
        i_NaK_max = theta[:, 29]  # constants[29]
        i_CaP_max = theta[:, 30]  # constants[30]
        k_CaP = theta[:, 31]  # constants[31]
        k_NaCa = theta[:, 32]  # constants[32]
        d_NaCa = theta[:, 33]  # constants[33]
        gamma = theta[:, 34]  # constants[34]
        Vol_i = theta[:, 35]  # constants[35]
        Vol_Ca = theta[:, 36]  # constants[36]
        Mg_i = theta[:, 37]  # constants[37]
        Vol_c = theta[:, 38]  # constants[38]
        I_up_max = theta[:, 39]  # constants[39]
        k_cyca = theta[:, 40]  # constants[40]
        k_srca = theta[:, 41]  # constants[41]
        k_xcs = theta[:, 42]  # constants[42]
        alpha_rel = theta[:, 43]  # constants[43]
        Vol_up = theta[:, 44]  # constants[44]
        Vol_rel = theta[:, 45]  # constants[45]
        tau_tr = theta[:, 46]  # constants[46]
        k_rel = theta[:, 47]  # constants[47]
        k_F3 = theta[:, 48]  # constants[48]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h1 = torch.zeros_like(y[:, 0])
        d_h2 = torch.zeros_like(y[:, 0])
        d_d_L = torch.zeros_like(y[:, 0])
        d_f_L = torch.zeros_like(y[:, 0])
        d_d_T = torch.zeros_like(y[:, 0])
        d_f_T = torch.zeros_like(y[:, 0])
        d_K_c = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_r = torch.zeros_like(y[:, 0])
        d_s1 = torch.zeros_like(y[:, 0])
        d_s2 = torch.zeros_like(y[:, 0])
        d_s3 = torch.zeros_like(y[:, 0])
        d_z = torch.zeros_like(y[:, 0])
        d_p_a = torch.zeros_like(y[:, 0])
        d_p_i = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_O_C = torch.zeros_like(y[:, 0])
        d_O_TC = torch.zeros_like(y[:, 0])
        d_O_TMgC = torch.zeros_like(y[:, 0])
        d_O_TMgMg = torch.zeros_like(y[:, 0])
        d_Ca_rel = torch.zeros_like(y[:, 0])
        d_Ca_up = torch.zeros_like(y[:, 0])
        d_O_Calse = torch.zeros_like(y[:, 0])
        d_F1 = torch.zeros_like(y[:, 0])
        d_F2 = torch.zeros_like(y[:, 0])
        d_F3 = torch.zeros_like(y[:, 0])
        alg_0_i_Stim = torch.zeros_like(y[:, 0])
        alg_1_E0_m = torch.zeros_like(y[:, 0])
        alg_2_alpha_h = torch.zeros_like(y[:, 0])
        alg_3_E0_alpha_d_L = torch.zeros_like(y[:, 0])
        alg_4_E0_f_L = torch.zeros_like(y[:, 0])
        alg_5_E0_d_T = torch.zeros_like(y[:, 0])
        alg_6_E0_f_T = torch.zeros_like(y[:, 0])
        alg_7_alpha_r = torch.zeros_like(y[:, 0])
        alg_8_s1_infinity = torch.zeros_like(y[:, 0])
        alg_9_s2_infinity = torch.zeros_like(y[:, 0])
        alg_10_s3_infinity = torch.zeros_like(y[:, 0])
        alg_11_alpha_z = torch.zeros_like(y[:, 0])
        alg_12_alpha_p_a = torch.zeros_like(y[:, 0])
        alg_13_alpha_p_i = torch.zeros_like(y[:, 0])
        alg_14_r_act = torch.zeros_like(y[:, 0])
        alg_15_E_Na = torch.zeros_like(y[:, 0])
        alg_16_alpha_m = torch.zeros_like(y[:, 0])
        alg_17_beta_h = torch.zeros_like(y[:, 0])
        alg_18_E0_beta_d_L = torch.zeros_like(y[:, 0])
        alg_19_alpha_f_L = torch.zeros_like(y[:, 0])
        alg_20_alpha_d_T = torch.zeros_like(y[:, 0])
        alg_21_alpha_f_T = torch.zeros_like(y[:, 0])
        alg_22_beta_r = torch.zeros_like(y[:, 0])
        alg_23_tau_s1 = torch.zeros_like(y[:, 0])
        alg_24_tau_s2 = torch.zeros_like(y[:, 0])
        alg_25_tau_s3 = torch.zeros_like(y[:, 0])
        alg_26_beta_z = torch.zeros_like(y[:, 0])
        alg_27_beta_p_a = torch.zeros_like(y[:, 0])
        alg_28_beta_p_i = torch.zeros_like(y[:, 0])
        alg_29_r_inact = torch.zeros_like(y[:, 0])
        alg_30_i_Na = torch.zeros_like(y[:, 0])
        alg_31_beta_m = torch.zeros_like(y[:, 0])
        alg_32_h_infinity = torch.zeros_like(y[:, 0])
        alg_33_E10 = torch.zeros_like(y[:, 0])
        alg_34_beta_f_L = torch.zeros_like(y[:, 0])
        alg_35_beta_d_T = torch.zeros_like(y[:, 0])
        alg_36_beta_f_T = torch.zeros_like(y[:, 0])
        alg_37_r_infinity = torch.zeros_like(y[:, 0])
        alg_38_z_infinity = torch.zeros_like(y[:, 0])
        alg_39_p_a_infinity = torch.zeros_like(y[:, 0])
        alg_40_p_i_infinity = torch.zeros_like(y[:, 0])
        alg_41_d_prime = torch.zeros_like(y[:, 0])
        alg_42_tau_h1 = torch.zeros_like(y[:, 0])
        alg_43_tau_h2 = torch.zeros_like(y[:, 0])
        alg_44_alpha_d_L = torch.zeros_like(y[:, 0])
        alg_45_f_L_infinity = torch.zeros_like(y[:, 0])
        alg_46_d_T_infinity = torch.zeros_like(y[:, 0])
        alg_47_f_T_infinity = torch.zeros_like(y[:, 0])
        alg_48_tau_r = torch.zeros_like(y[:, 0])
        alg_49_tau_z = torch.zeros_like(y[:, 0])
        alg_50_tau_p_a = torch.zeros_like(y[:, 0])
        alg_51_tau_p_i = torch.zeros_like(y[:, 0])
        alg_52_i_Ca_L = torch.zeros_like(y[:, 0])
        alg_53_beta_d_L = torch.zeros_like(y[:, 0])
        alg_54_tau_f_L = torch.zeros_like(y[:, 0])
        alg_55_tau_d_T = torch.zeros_like(y[:, 0])
        alg_56_tau_f_T = torch.zeros_like(y[:, 0])
        alg_57_i_Ca_T = torch.zeros_like(y[:, 0])
        alg_58_d_L_infinity = torch.zeros_like(y[:, 0])
        alg_59_E_K = torch.zeros_like(y[:, 0])
        alg_60_tau_d_L = torch.zeros_like(y[:, 0])
        alg_61_i_to = torch.zeros_like(y[:, 0])
        alg_62_i_sus = torch.zeros_like(y[:, 0])
        alg_63_i_Ks = torch.zeros_like(y[:, 0])
        alg_64_i_Kr = torch.zeros_like(y[:, 0])
        alg_65_i_K1 = torch.zeros_like(y[:, 0])
        alg_66_E_Ca = torch.zeros_like(y[:, 0])
        alg_67_i_B_Na = torch.zeros_like(y[:, 0])
        alg_68_i_B_Ca = torch.zeros_like(y[:, 0])
        alg_69_i_p = torch.zeros_like(y[:, 0])
        alg_70_i_CaP = torch.zeros_like(y[:, 0])
        alg_71_i_NaCa = torch.zeros_like(y[:, 0])
        alg_72_dOCdt = torch.zeros_like(y[:, 0])
        alg_73_dOTCdt = torch.zeros_like(y[:, 0])
        alg_74_dOTMgCdt = torch.zeros_like(y[:, 0])
        alg_75_i_up = torch.zeros_like(y[:, 0])
        alg_76_i_tr = torch.zeros_like(y[:, 0])
        alg_77_i_rel = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_O_TMgMg = torch.zeros_like(y[:, 0]) + (2000.0 * Mg_i * (1.0 - O_TMgC - O_TMgMg) - 666.0 * O_TMgMg)
        alg_14_r_act = 240.0 * torch.exp(0.08 * (V - 20.0)) + 203.8 * (Ca_i / (Ca_i + k_rel)) ** 4.0
        d_F1 = torch.zeros_like(y[:, 0]) + (k_F3 * F3 - alg_14_r_act * F1)
        alg_23_tau_s1 = 0.5466 / (1.0 + torch.exp((V + 32.8) / 0.1)) + 0.0204
        alg_8_s1_infinity = 1.0 / (1.0 + torch.exp((V + 28.29) / 7.06))
        d_s1 = torch.zeros_like(y[:, 0]) + ((alg_8_s1_infinity - s1) / alg_23_tau_s1)
        alg_24_tau_s2 = 5.75 / (1.0 + torch.exp((V + 32.8) / 0.1)) + 0.45 / (1.0 + torch.exp((V - 13.54) / -13.97))
        alg_9_s2_infinity = 1.0 / (1.0 + torch.exp((V + 28.29) / 7.06))
        d_s2 = torch.zeros_like(y[:, 0]) + ((alg_9_s2_infinity - s2) / alg_24_tau_s2)
        alg_25_tau_s3 = 7.5 / (1.0 + torch.exp((V + 23.0) / 0.5)) + 0.5
        alg_10_s3_infinity = (1.0 / (1.0 + torch.exp((V + 50.67) / 27.38)) + 0.666) / 1.666
        d_s3 = torch.zeros_like(y[:, 0]) + ((alg_10_s3_infinity - s3) / alg_25_tau_s3)
        alg_29_r_inact = 33.96 + 339.6 * (Ca_i / (Ca_i + k_rel)) ** 4.0
        d_F2 = torch.zeros_like(y[:, 0]) + (alg_14_r_act * F1 - alg_29_r_inact * F2)
        d_F3 = torch.zeros_like(y[:, 0]) + (F2 * alg_29_r_inact - k_F3 * F3)
        alg_1_E0_m = V + 44.4
        alg_16_alpha_m = -460.0 * alg_1_E0_m / (torch.exp(alg_1_E0_m / -12.673) - 1.0)
        alg_31_beta_m = 18400.0 * torch.exp(alg_1_E0_m / -12.673)
        d_m = torch.zeros_like(y[:, 0]) + (alg_16_alpha_m * (1.0 - m) - alg_31_beta_m * m)
        alg_2_alpha_h = 44.9 * torch.exp((V + 66.9) / -5.57)
        alg_17_beta_h = 1491.0 / (1.0 + 323.3 * torch.exp((V + 94.6) / -12.9))
        alg_32_h_infinity = alg_2_alpha_h / (alg_2_alpha_h + alg_17_beta_h)
        alg_42_tau_h1 = 0.03 / (1.0 + torch.exp((V + 40.0) / 6.0)) + 0.00015
        d_h1 = torch.zeros_like(y[:, 0]) + ((alg_32_h_infinity - h1) / alg_42_tau_h1)
        alg_43_tau_h2 = 0.12 / (1.0 + torch.exp((V + 60.0) / 2.0)) + 0.00045
        d_h2 = torch.zeros_like(y[:, 0]) + ((alg_32_h_infinity - h2) / alg_43_tau_h2)
        alg_7_alpha_r = 386.6 * torch.exp(V / 12.0)
        alg_22_beta_r = 8.011 * torch.exp(V / -7.2)
        alg_48_tau_r = 1.0 / (alg_7_alpha_r + alg_22_beta_r) + 0.0004
        alg_37_r_infinity = 1.0 / (1.0 + torch.exp((V + 15.0) / -5.633))
        d_r = torch.zeros_like(y[:, 0]) + ((alg_37_r_infinity - r) / alg_48_tau_r)
        alg_11_alpha_z = 1.66 * torch.exp(V / 69.452)
        alg_26_beta_z = 0.3 * torch.exp(V / -21.826)
        alg_49_tau_z = 1.0 / (alg_11_alpha_z + alg_26_beta_z) + 0.06
        alg_38_z_infinity = 1.0 / (1.0 + torch.exp((V - 0.9) / -13.8))
        d_z = torch.zeros_like(y[:, 0]) + ((alg_38_z_infinity - z) / alg_49_tau_z)
        alg_12_alpha_p_a = 9.0 * torch.exp(V / 25.371)
        alg_27_beta_p_a = 1.3 * torch.exp(V / -13.026)
        alg_50_tau_p_a = 1.0 / (alg_12_alpha_p_a + alg_27_beta_p_a)
        alg_39_p_a_infinity = 1.0 / (1.0 + torch.exp((V + 5.1) / -7.4))
        d_p_a = torch.zeros_like(y[:, 0]) + ((alg_39_p_a_infinity - p_a) / alg_50_tau_p_a)
        alg_13_alpha_p_i = 100.0 * torch.exp(V / -54.645)
        alg_28_beta_p_i = 656.0 * torch.exp(V / 106.157)
        alg_51_tau_p_i = 1.0 / (alg_13_alpha_p_i + alg_28_beta_p_i)
        alg_40_p_i_infinity = 1.0 / (1.0 + torch.exp((V + 47.3921) / 18.6603))
        d_p_i = torch.zeros_like(y[:, 0]) + ((alg_40_p_i_infinity - p_i) / alg_51_tau_p_i)
        alg_4_E0_f_L = V + 18.0
        alg_19_alpha_f_L = 8.49 * alg_4_E0_f_L / (torch.exp(alg_4_E0_f_L / 4.0) - 1.0)
        alg_34_beta_f_L = 67.922 / (1.0 + torch.exp(alg_4_E0_f_L / -4.0))
        alg_45_f_L_infinity = alg_19_alpha_f_L / (alg_19_alpha_f_L + alg_34_beta_f_L)
        alg_54_tau_f_L = 1.0 / (alg_19_alpha_f_L + alg_34_beta_f_L)
        d_f_L = torch.zeros_like(y[:, 0]) + ((alg_45_f_L_infinity - f_L) / alg_54_tau_f_L)
        alg_5_E0_d_T = V + 23.3
        alg_46_d_T_infinity = 1.0 / (1.0 + torch.exp((alg_5_E0_d_T - 0.3) / -6.1))
        alg_20_alpha_d_T = 674.173 * torch.exp(alg_5_E0_d_T / 30.0)
        alg_35_beta_d_T = 674.173 * torch.exp(alg_5_E0_d_T / -30.0)
        alg_55_tau_d_T = 1.0 / (alg_20_alpha_d_T + alg_35_beta_d_T)
        d_d_T = torch.zeros_like(y[:, 0]) + ((alg_46_d_T_infinity - d_T) / alg_55_tau_d_T)
        alg_6_E0_f_T = V + 75.0
        alg_21_alpha_f_T = 9.637 * torch.exp(alg_6_E0_f_T / -83.3)
        alg_36_beta_f_T = 9.637 * torch.exp(alg_6_E0_f_T / 15.38)
        alg_47_f_T_infinity = alg_21_alpha_f_T / (alg_21_alpha_f_T + alg_36_beta_f_T)
        alg_56_tau_f_T = 1.0 / (alg_21_alpha_f_T + alg_36_beta_f_T)
        d_f_T = torch.zeros_like(y[:, 0]) + ((alg_47_f_T_infinity - f_T) / alg_56_tau_f_T)
        alg_33_E10 = V + 10.0
        alg_58_d_L_infinity = 1.0 / (1.0 + torch.exp((alg_33_E10 + 0.95) / -6.6))
        alg_3_E0_alpha_d_L = V + 45.0
        alg_44_alpha_d_L = -16.72 * alg_3_E0_alpha_d_L / (torch.exp(alg_3_E0_alpha_d_L / -2.5) - 1.0) + -50.0 * alg_33_E10 / (torch.exp(alg_33_E10 / -4.808) - 1.0)
        alg_18_E0_beta_d_L = V + 5.0
        alg_53_beta_d_L = 4.48 * alg_18_E0_beta_d_L / (torch.exp(alg_18_E0_beta_d_L / 2.5) - 1.0)
        alg_60_tau_d_L = 1.0 / (alg_44_alpha_d_L + alg_53_beta_d_L)
        d_d_L = torch.zeros_like(y[:, 0]) + ((alg_58_d_L_infinity - d_L) / alg_60_tau_d_L)
        alg_59_E_K = R * T / F * torch.log(K_c / K_i)
        alg_61_i_to = torch.where((CT == 1.0) & (PM == 0.0), 0.2 * g_to * r * (0.59 * s1 ** 3.0 + 0.41 * s2 ** 3.0) * (0.6 * s3 ** 6.0 + 0.4) * (V - alg_59_E_K), torch.where((CT == 0.0) & (PM == 1.0), 0.35 * g_to * r * (0.59 * s1 ** 3.0 + 0.41 * s2 ** 3.0) * (0.6 * s3 ** 6.0 + 0.4) * (V - alg_59_E_K), g_to * r * (0.59 * s1 ** 3.0 + 0.41 * s2 ** 3.0) * (0.6 * s3 ** 6.0 + 0.4) * (V - alg_59_E_K)))
        alg_62_i_sus = torch.where((CT == 1.0) & (PM == 0.0), 0.0014 * (V + 70.0), torch.where((CT == 0.0) & (PM == 1.0), 0.0024 * (V + 70.0), 0.001 * (V + 70.0)))
        alg_65_i_K1 = torch.where((CT == 1.0) & (PM == 0.0), 2.0 * g_K1 * (V - alg_59_E_K) * (K_c / (K_c + KmK1)) ** 3.0 * 1.0 / (1.0 + torch.exp(steepK1 * F * (V - alg_59_E_K - shiftK1) / (R * T))), torch.where((CT == 0.0) & (PM == 1.0), 2.5 * g_K1 * (V - alg_59_E_K) * (K_c / (K_c + KmK1)) ** 3.0 * 1.0 / (1.0 + torch.exp(steepK1 * F * (V - alg_59_E_K - shiftK1) / (R * T))), g_K1 * (V - alg_59_E_K) * (K_c / (K_c + KmK1)) ** 3.0 * 1.0 / (1.0 + torch.exp(steepK1 * F * (V - alg_59_E_K - shiftK1) / (R * T)))))
        alg_64_i_Kr = g_Kr * p_a * p_i * (V - alg_59_E_K)
        alg_63_i_Ks = g_Ks * z * (V - alg_59_E_K)
        alg_69_i_p = i_NaK_max * K_c / (K_c + k_NaK_K) * Na_i ** 1.5 / (Na_i ** 1.5 + k_NaK_Na ** 1.5) * 1.6 / (1.5 + torch.exp((V + 60.0) / -40.0))
        d_K_i = torch.zeros_like(y[:, 0]) + (-(alg_61_i_to + alg_62_i_sus + alg_65_i_K1 + alg_64_i_Kr + alg_63_i_Ks - 2.0 * alg_69_i_p) / (Vol_i * F))
        d_K_c = torch.zeros_like(y[:, 0]) + ((alg_61_i_to + alg_62_i_sus + alg_65_i_K1 + alg_64_i_Kr + alg_63_i_Ks - 2.0 * alg_69_i_p) / (Vol_c * F))
        alg_15_E_Na = R * T / F * torch.log(Na_c / Na_i)
        alg_30_i_Na = P_Na * m ** 3.0 * (0.635 * h1 + 0.365 * h2) * Na_c * V * F ** 2.0 / (R * T) * (torch.exp((V - alg_15_E_Na) * F / (R * T)) - 1.0) / (torch.exp(V * F / (R * T)) - 1.0)
        alg_41_d_prime = 1.0 / (1.0 + torch.exp((V - 23.0) / -12.0))
        alg_52_i_Ca_L = torch.where((CT == 1.0) & (PM == 0.0), 1.8 * g_Ca_L * (d_L * f_L + alg_41_d_prime) * (V - E_Ca_app), torch.where((CT == 0.0) & (PM == 1.0), 2.1 * g_Ca_L * (d_L * f_L + alg_41_d_prime) * (V - E_Ca_app), g_Ca_L * (d_L * f_L + alg_41_d_prime) * (V - E_Ca_app)))
        alg_57_i_Ca_T = g_Ca_T * d_T * f_T * (V - E_Ca_T)
        alg_67_i_B_Na = torch.where((CT == 1.0) & (PM == 0.0), 2e-05 * (V - alg_15_E_Na), torch.where((CT == 0.0) & (PM == 1.0), 3e-05 * (V - alg_15_E_Na), g_B_Na * (V - alg_15_E_Na)))
        alg_66_E_Ca = R * T / (2.0 * F) * torch.log(Ca_c / Ca_i)
        alg_68_i_B_Ca = torch.where((CT == 1.0) & (PM == 0.0), 2e-05 * (V - alg_66_E_Ca), torch.where((CT == 0.0) & (PM == 1.0), 3e-05 * (V - alg_66_E_Ca), g_B_Ca * (V - alg_66_E_Ca)))
        alg_70_i_CaP = i_CaP_max * Ca_i / (Ca_i + k_CaP)
        alg_71_i_NaCa = k_NaCa * (Na_i ** 3.0 * Ca_c * torch.exp(gamma * F * V / (R * T)) - Na_c ** 3.0 * Ca_i * torch.exp((gamma - 1.0) * V * F / (R * T))) / (1.0 + d_NaCa * (Na_c ** 3.0 * Ca_i + Na_i ** 3.0 * Ca_c))
        alg_0_i_Stim = torch.where((t >= stim_start) & (t <= stim_end) & (t - stim_start - torch.floor((t - stim_start) / stim_period) * stim_period <= stim_duration), stim_amplitude, 0.0)
        d_V = torch.zeros_like(y[:, 0]) + (-1.0 / Cm * (alg_64_i_Kr + alg_63_i_Ks + alg_30_i_Na + alg_52_i_Ca_L + alg_57_i_Ca_T + alg_61_i_to + alg_62_i_sus + alg_65_i_K1 + alg_67_i_B_Na + alg_68_i_B_Ca + alg_69_i_p + alg_70_i_CaP + alg_71_i_NaCa + alg_0_i_Stim))
        d_Na_i = torch.zeros_like(y[:, 0]) + (-(alg_30_i_Na + alg_67_i_B_Na + 3.0 * alg_69_i_p + 3.0 * alg_71_i_NaCa) / (Vol_i * F))
        alg_72_dOCdt = 200000.0 * Ca_i * (1.0 - O_C) - 476.0 * O_C
        d_O_C = torch.zeros_like(y[:, 0]) + (alg_72_dOCdt)
        alg_73_dOTCdt = 78400.0 * Ca_i * (1.0 - O_TC) - 392.0 * O_TC
        d_O_TC = torch.zeros_like(y[:, 0]) + (alg_73_dOTCdt)
        d_O_Calse = torch.zeros_like(y[:, 0]) + (480.0 * Ca_rel * (1.0 - O_Calse) - 400.0 * O_Calse)
        alg_74_dOTMgCdt = 200000.0 * Ca_i * (1.0 - O_TMgC - O_TMgMg) - 6.6 * O_TMgC
        d_O_TMgC = torch.zeros_like(y[:, 0]) + (alg_74_dOTMgCdt)
        alg_75_i_up = I_up_max * (Ca_i / k_cyca - k_xcs ** 2.0 * Ca_up / k_srca) / ((Ca_i + k_cyca) / k_cyca + k_xcs * (Ca_up + k_srca) / k_srca)
        alg_76_i_tr = (Ca_up - Ca_rel) * 2.0 * F * Vol_rel / tau_tr
        d_Ca_up = torch.zeros_like(y[:, 0]) + ((alg_75_i_up - alg_76_i_tr) / (2.0 * Vol_up * F))
        alg_77_i_rel = alpha_rel * (F2 / (F2 + 0.25)) ** 2.0 * (Ca_rel - Ca_i)
        d_Ca_i = torch.zeros_like(y[:, 0]) + (-(alg_52_i_Ca_L + alg_57_i_Ca_T + alg_68_i_B_Ca + alg_70_i_CaP - 2.0 * alg_71_i_NaCa + alg_75_i_up - alg_77_i_rel) / (2.0 * Vol_Ca * F) - (0.08 * alg_73_dOTCdt + 0.16 * alg_74_dOTMgCdt + 0.045 * alg_72_dOCdt))
        d_Ca_rel = torch.zeros_like(y[:, 0]) + ((alg_76_i_tr - alg_77_i_rel) / (2.0 * Vol_rel * F) - 31.0 * d_O_Calse)

        return torch.stack(
            [d_V, d_Na_i, d_m, d_h1, d_h2, d_d_L, d_f_L, d_d_T, d_f_T, d_K_c, d_K_i, d_r, d_s1, d_s2, d_s3, d_z, d_p_a, d_p_i, d_Ca_i, d_O_C, d_O_TC, d_O_TMgC, d_O_TMgMg, d_Ca_rel, d_Ca_up, d_O_Calse, d_F1, d_F2, d_F3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
