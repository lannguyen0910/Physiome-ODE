import torch
import torch.nn as nn
import math


class ElectrophysiologyNygrenFisetFirekClarkLindbladClarkGiles1998ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/nygren_fiset_firek_clark_lindblad_clark_giles_1998/model.py.
    Dynamic states: 29; independent constants: 49; derived constants used by rates: 2.
    Original constant slots: 51; algebraic slots: 49.
    """
    source_file = 'electrophysiology/nygren_fiset_firek_clark_lindblad_clark_giles_1998/model.py'
    source_state_dim = 29
    source_constant_dim = 51
    algebraic_dim = 49
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
            else {'R': 8314, 'T': 306.15, 'F': 96487, 'Cm': 0.05, 'stim_start': 0.1, 'stim_end': 100000000, 'stim_period': 1, 'stim_duration': 0.006, 'stim_amplitude': -280, 'P_Na': 0.0016, 'g_Ca_L': 6.75, 'E_Ca_app': 60, 'k_Ca': 0.025, 'g_t': 7.5, 'g_sus': 2.75, 'g_Ks': 1, 'g_Kr': 0.5, 'g_K1': 3, 'g_B_Na': 0.060599, 'g_B_Ca': 0.078681, 'k_NaK_K': 1, 'k_NaK_Na': 11, 'i_NaK_max': 70.8253, 'i_CaP_max': 4, 'k_CaP': 0.0002, 'k_NaCa': 0.0374842, 'd_NaCa': 0.0003, 'gamma': 0.45, 'phi_Na_en': -1.68, 'Vol_i': 0.005884, 'tau_di': 0.01, 'Mg_i': 2.5, 'tau_Na': 14.3, 'tau_K': 10, 'tau_Ca': 24.7, 'Na_b': 130, 'Ca_b': 1.8, 'K_b': 5.4, 'I_up_max': 2800, 'k_cyca': 0.0003, 'k_srca': 0.5, 'k_xcs': 0.4, 'alpha_rel': 200000, 'Vol_up': 0.0003969, 'Vol_rel': 4.41e-05, 'r_recov': 0.815, 'tau_tr': 0.01, 'k_rel_i': 0.0003, 'k_rel_d': 0.003}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -74.2525, 'Na_c': 130.011, 'Na_i': 8.5547, 'm': 0.0032017, 'h1': 0.8814, 'h2': 0.8742, 'Ca_d': 7.2495e-05, 'd_L': 1.3005e-05, 'f_L_1': 0.9986, 'f_L_2': 0.9986, 'K_c': 5.3581, 'K_i': 129.435, 'r': 0.0010678, 's': 0.949, 'r_sus': 0.00015949, 's_sus': 0.9912, 'n': 0.0048357, 'p_a': 0.0001, 'Ca_c': 1.8147, 'Ca_i': 6.729e-05, 'O_C': 0.0275, 'O_TC': 0.0133, 'O_TMgC': 0.1961, 'O_TMgMg': 0.7094, 'Ca_rel': 0.6465, 'Ca_up': 0.6646, 'O_Calse': 0.4369, 'F1': 0.4284, 'F2': 0.0028}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'Cm', 'stim_start', 'stim_end', 'stim_period', 'stim_duration', 'stim_amplitude', 'P_Na', 'g_Ca_L', 'E_Ca_app', 'k_Ca', 'g_t', 'g_sus', 'g_Ks', 'g_Kr', 'g_K1', 'g_B_Na', 'g_B_Ca', 'k_NaK_K', 'k_NaK_Na', 'i_NaK_max', 'i_CaP_max', 'k_CaP', 'k_NaCa', 'd_NaCa', 'gamma', 'phi_Na_en', 'Vol_i', 'tau_di', 'Mg_i', 'tau_Na', 'tau_K', 'tau_Ca', 'Na_b', 'Ca_b', 'K_b', 'I_up_max', 'k_cyca', 'k_srca', 'k_xcs', 'alpha_rel', 'Vol_up', 'Vol_rel', 'r_recov', 'tau_tr', 'k_rel_i', 'k_rel_d']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'Na_c', 'Na_i', 'm', 'h1', 'h2', 'Ca_d', 'd_L', 'f_L_1', 'f_L_2', 'K_c', 'K_i', 'r', 's', 'r_sus', 's_sus', 'n', 'p_a', 'Ca_c', 'Ca_i', 'O_C', 'O_TC', 'O_TMgC', 'O_TMgMg', 'Ca_rel', 'Ca_up', 'O_Calse', 'F1', 'F2']
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
        Na_i = y[:, 2]
        m = y[:, 3]
        h1 = y[:, 4]
        h2 = y[:, 5]
        Ca_d = y[:, 6]
        d_L = y[:, 7]
        f_L_1 = y[:, 8]
        f_L_2 = y[:, 9]
        K_c = y[:, 10]
        K_i = y[:, 11]
        r = y[:, 12]
        s = y[:, 13]
        r_sus = y[:, 14]
        s_sus = y[:, 15]
        n = y[:, 16]
        p_a = y[:, 17]
        Ca_c = y[:, 18]
        Ca_i = y[:, 19]
        O_C = y[:, 20]
        O_TC = y[:, 21]
        O_TMgC = y[:, 22]
        O_TMgMg = y[:, 23]
        Ca_rel = y[:, 24]
        Ca_up = y[:, 25]
        O_Calse = y[:, 26]
        F1 = y[:, 27]
        F2 = y[:, 28]

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
        P_Na = theta[:, 9]  # constants[9]
        g_Ca_L = theta[:, 10]  # constants[10]
        E_Ca_app = theta[:, 11]  # constants[11]
        k_Ca = theta[:, 12]  # constants[12]
        g_t = theta[:, 13]  # constants[13]
        g_sus = theta[:, 14]  # constants[14]
        g_Ks = theta[:, 15]  # constants[15]
        g_Kr = theta[:, 16]  # constants[16]
        g_K1 = theta[:, 17]  # constants[17]
        g_B_Na = theta[:, 18]  # constants[18]
        g_B_Ca = theta[:, 19]  # constants[19]
        k_NaK_K = theta[:, 20]  # constants[20]
        k_NaK_Na = theta[:, 21]  # constants[21]
        i_NaK_max = theta[:, 22]  # constants[22]
        i_CaP_max = theta[:, 23]  # constants[23]
        k_CaP = theta[:, 24]  # constants[24]
        k_NaCa = theta[:, 25]  # constants[25]
        d_NaCa = theta[:, 26]  # constants[26]
        gamma = theta[:, 27]  # constants[27]
        phi_Na_en = theta[:, 28]  # constants[28]
        Vol_i = theta[:, 29]  # constants[29]
        tau_di = theta[:, 30]  # constants[30]
        Mg_i = theta[:, 31]  # constants[31]
        tau_Na = theta[:, 32]  # constants[32]
        tau_K = theta[:, 33]  # constants[33]
        tau_Ca = theta[:, 34]  # constants[34]
        Na_b = theta[:, 35]  # constants[35]
        Ca_b = theta[:, 36]  # constants[36]
        K_b = theta[:, 37]  # constants[37]
        I_up_max = theta[:, 38]  # constants[38]
        k_cyca = theta[:, 39]  # constants[39]
        k_srca = theta[:, 40]  # constants[40]
        k_xcs = theta[:, 41]  # constants[41]
        alpha_rel = theta[:, 42]  # constants[42]
        Vol_up = theta[:, 43]  # constants[43]
        Vol_rel = theta[:, 44]  # constants[44]
        r_recov = theta[:, 45]  # constants[45]
        tau_tr = theta[:, 46]  # constants[46]
        k_rel_i = theta[:, 47]  # constants[47]
        k_rel_d = theta[:, 48]  # constants[48]

        # Derived constants from the original initConsts()
        Vol_d = 0.02 * Vol_i  # constants[49]
        Vol_c = 0.136 * Vol_i  # constants[50]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_Na_c = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h1 = torch.zeros_like(y[:, 0])
        d_h2 = torch.zeros_like(y[:, 0])
        d_Ca_d = torch.zeros_like(y[:, 0])
        d_d_L = torch.zeros_like(y[:, 0])
        d_f_L_1 = torch.zeros_like(y[:, 0])
        d_f_L_2 = torch.zeros_like(y[:, 0])
        d_K_c = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_r = torch.zeros_like(y[:, 0])
        d_s = torch.zeros_like(y[:, 0])
        d_r_sus = torch.zeros_like(y[:, 0])
        d_s_sus = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_p_a = torch.zeros_like(y[:, 0])
        d_Ca_c = torch.zeros_like(y[:, 0])
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
        alg_0_i_Stim = torch.zeros_like(y[:, 0])
        alg_1_m_infinity = torch.zeros_like(y[:, 0])
        alg_2_h_infinity = torch.zeros_like(y[:, 0])
        alg_3_d_L_infinity = torch.zeros_like(y[:, 0])
        alg_4_f_L_infinity = torch.zeros_like(y[:, 0])
        alg_5_r_infinity = torch.zeros_like(y[:, 0])
        alg_6_s_infinity = torch.zeros_like(y[:, 0])
        alg_7_r_sus_infinity = torch.zeros_like(y[:, 0])
        alg_8_s_sus_infinity = torch.zeros_like(y[:, 0])
        alg_9_n_infinity = torch.zeros_like(y[:, 0])
        alg_10_p_a_infinity = torch.zeros_like(y[:, 0])
        alg_11_r_act = torch.zeros_like(y[:, 0])
        alg_12_E_Na = torch.zeros_like(y[:, 0])
        alg_13_tau_m = torch.zeros_like(y[:, 0])
        alg_14_tau_h1 = torch.zeros_like(y[:, 0])
        alg_15_tau_h2 = torch.zeros_like(y[:, 0])
        alg_16_tau_d_L = torch.zeros_like(y[:, 0])
        alg_17_tau_f_L1 = torch.zeros_like(y[:, 0])
        alg_18_tau_f_L2 = torch.zeros_like(y[:, 0])
        alg_19_tau_r = torch.zeros_like(y[:, 0])
        alg_20_tau_s = torch.zeros_like(y[:, 0])
        alg_21_tau_r_sus = torch.zeros_like(y[:, 0])
        alg_22_tau_s_sus = torch.zeros_like(y[:, 0])
        alg_23_tau_n = torch.zeros_like(y[:, 0])
        alg_24_tau_p_a = torch.zeros_like(y[:, 0])
        alg_25_r_inact = torch.zeros_like(y[:, 0])
        alg_26_i_Na = torch.zeros_like(y[:, 0])
        alg_27_f_Ca = torch.zeros_like(y[:, 0])
        alg_28_i_Ca_L = torch.zeros_like(y[:, 0])
        alg_29_E_K = torch.zeros_like(y[:, 0])
        alg_30_i_t = torch.zeros_like(y[:, 0])
        alg_31_i_sus = torch.zeros_like(y[:, 0])
        alg_32_i_Ks = torch.zeros_like(y[:, 0])
        alg_33_p_i = torch.zeros_like(y[:, 0])
        alg_34_i_Kr = torch.zeros_like(y[:, 0])
        alg_35_i_K1 = torch.zeros_like(y[:, 0])
        alg_36_i_B_Na = torch.zeros_like(y[:, 0])
        alg_37_E_Ca = torch.zeros_like(y[:, 0])
        alg_38_i_B_Ca = torch.zeros_like(y[:, 0])
        alg_39_i_NaK = torch.zeros_like(y[:, 0])
        alg_40_i_CaP = torch.zeros_like(y[:, 0])
        alg_41_i_NaCa = torch.zeros_like(y[:, 0])
        alg_42_i_di = torch.zeros_like(y[:, 0])
        alg_43_dOCdt = torch.zeros_like(y[:, 0])
        alg_44_dOTCdt = torch.zeros_like(y[:, 0])
        alg_45_dOTMgCdt = torch.zeros_like(y[:, 0])
        alg_46_i_up = torch.zeros_like(y[:, 0])
        alg_47_i_tr = torch.zeros_like(y[:, 0])
        alg_48_i_rel = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_O_TMgMg = torch.zeros_like(y[:, 0]) + (2000.0 * Mg_i * (1.0 - O_TMgC - O_TMgMg) - 666.0 * O_TMgMg)
        alg_11_r_act = 203.8 * ((Ca_i / (Ca_i + k_rel_i)) ** 4.0 + (Ca_d / (Ca_d + k_rel_d)) ** 4.0)
        d_F1 = torch.zeros_like(y[:, 0]) + (r_recov * (1.0 - F1 - F2) - alg_11_r_act * F1)
        alg_1_m_infinity = 1.0 / (1.0 + torch.exp((V + 27.12) / -8.21))
        alg_13_tau_m = 4.2e-05 * torch.exp(-((V + 25.57) / 28.8) ** 2.0) + 2.4e-05
        d_m = torch.zeros_like(y[:, 0]) + ((alg_1_m_infinity - m) / alg_13_tau_m)
        alg_2_h_infinity = 1.0 / (1.0 + torch.exp((V + 63.6) / 5.3))
        alg_14_tau_h1 = 0.03 / (1.0 + torch.exp((V + 35.1) / 3.2)) + 0.0003
        d_h1 = torch.zeros_like(y[:, 0]) + ((alg_2_h_infinity - h1) / alg_14_tau_h1)
        alg_15_tau_h2 = 0.12 / (1.0 + torch.exp((V + 35.1) / 3.2)) + 0.003
        d_h2 = torch.zeros_like(y[:, 0]) + ((alg_2_h_infinity - h2) / alg_15_tau_h2)
        alg_3_d_L_infinity = 1.0 / (1.0 + torch.exp((V + 9.0) / -5.8))
        alg_16_tau_d_L = 0.0027 * torch.exp(-((V + 35.0) / 30.0) ** 2.0) + 0.002
        d_d_L = torch.zeros_like(y[:, 0]) + ((alg_3_d_L_infinity - d_L) / alg_16_tau_d_L)
        alg_4_f_L_infinity = 1.0 / (1.0 + torch.exp((V + 27.4) / 7.1))
        alg_17_tau_f_L1 = 0.161 * torch.exp(-((V + 40.0) / 14.4) ** 2.0) + 0.01
        d_f_L_1 = torch.zeros_like(y[:, 0]) + ((alg_4_f_L_infinity - f_L_1) / alg_17_tau_f_L1)
        alg_18_tau_f_L2 = 1.3323 * torch.exp(-((V + 40.0) / 14.2) ** 2.0) + 0.0626
        d_f_L_2 = torch.zeros_like(y[:, 0]) + ((alg_4_f_L_infinity - f_L_2) / alg_18_tau_f_L2)
        alg_19_tau_r = 0.0035 * torch.exp(-(V / 30.0) ** 2.0) + 0.0015
        alg_5_r_infinity = 1.0 / (1.0 + torch.exp((V - 1.0) / -11.0))
        d_r = torch.zeros_like(y[:, 0]) + ((alg_5_r_infinity - r) / alg_19_tau_r)
        alg_20_tau_s = 0.4812 * torch.exp(-((V + 52.45) / 14.97) ** 2.0) + 0.01414
        alg_6_s_infinity = 1.0 / (1.0 + torch.exp((V + 40.5) / 11.5))
        d_s = torch.zeros_like(y[:, 0]) + ((alg_6_s_infinity - s) / alg_20_tau_s)
        alg_21_tau_r_sus = 0.009 / (1.0 + torch.exp((V + 5.0) / 12.0)) + 0.0005
        alg_7_r_sus_infinity = 1.0 / (1.0 + torch.exp((V + 4.3) / -8.0))
        d_r_sus = torch.zeros_like(y[:, 0]) + ((alg_7_r_sus_infinity - r_sus) / alg_21_tau_r_sus)
        alg_22_tau_s_sus = 0.047 / (1.0 + torch.exp((V + 60.0) / 10.0)) + 0.3
        alg_8_s_sus_infinity = 0.4 / (1.0 + torch.exp((V + 20.0) / 10.0)) + 0.6
        d_s_sus = torch.zeros_like(y[:, 0]) + ((alg_8_s_sus_infinity - s_sus) / alg_22_tau_s_sus)
        alg_23_tau_n = 0.7 + 0.4 * torch.exp(-((V - 20.0) / 20.0) ** 2.0)
        alg_9_n_infinity = 1.0 / (1.0 + torch.exp((V - 19.9) / -12.7))
        d_n = torch.zeros_like(y[:, 0]) + ((alg_9_n_infinity - n) / alg_23_tau_n)
        alg_24_tau_p_a = 0.03118 + 0.21718 * torch.exp(-((V + 20.1376) / 22.1996) ** 2.0)
        alg_10_p_a_infinity = 1.0 / (1.0 + torch.exp((V + 15.0) / -6.0))
        d_p_a = torch.zeros_like(y[:, 0]) + ((alg_10_p_a_infinity - p_a) / alg_24_tau_p_a)
        alg_25_r_inact = 33.96 + 339.6 * (Ca_i / (Ca_i + k_rel_i)) ** 4.0
        d_F2 = torch.zeros_like(y[:, 0]) + (alg_11_r_act * F1 - alg_25_r_inact * F2)
        alg_29_E_K = R * T / F * torch.log(K_c / K_i)
        alg_30_i_t = g_t * r * s * (V - alg_29_E_K)
        alg_31_i_sus = g_sus * r_sus * s_sus * (V - alg_29_E_K)
        alg_35_i_K1 = g_K1 * (K_c / 1.0) ** 0.4457 * (V - alg_29_E_K) / (1.0 + torch.exp(1.5 * (V - alg_29_E_K + 3.6) * F / (R * T)))
        alg_33_p_i = 1.0 / (1.0 + torch.exp((V + 55.0) / 24.0))
        alg_34_i_Kr = g_Kr * p_a * alg_33_p_i * (V - alg_29_E_K)
        alg_32_i_Ks = g_Ks * n * (V - alg_29_E_K)
        alg_39_i_NaK = i_NaK_max * K_c / (K_c + k_NaK_K) * Na_i ** 1.5 / (Na_i ** 1.5 + k_NaK_Na ** 1.5) * (V + 150.0) / (V + 200.0)
        d_K_i = torch.zeros_like(y[:, 0]) + (-(alg_30_i_t + alg_31_i_sus + alg_35_i_K1 + alg_34_i_Kr + alg_32_i_Ks - 2.0 * alg_39_i_NaK) / (Vol_i * F))
        d_K_c = torch.zeros_like(y[:, 0]) + ((K_b - K_c) / tau_K + (alg_30_i_t + alg_31_i_sus + alg_35_i_K1 + alg_34_i_Kr + alg_32_i_Ks - 2.0 * alg_39_i_NaK) / (Vol_c * F))
        alg_12_E_Na = R * T / F * torch.log(Na_c / Na_i)
        alg_26_i_Na = P_Na * m ** 3.0 * (0.9 * h1 + 0.1 * h2) * Na_c * V * F ** 2.0 / (R * T) * (torch.exp((V - alg_12_E_Na) * F / (R * T)) - 1.0) / (torch.exp(V * F / (R * T)) - 1.0)
        alg_27_f_Ca = Ca_d / (Ca_d + k_Ca)
        alg_28_i_Ca_L = g_Ca_L * d_L * (alg_27_f_Ca * f_L_1 + (1.0 - alg_27_f_Ca) * f_L_2) * (V - E_Ca_app)
        alg_36_i_B_Na = g_B_Na * (V - alg_12_E_Na)
        alg_37_E_Ca = R * T / (2.0 * F) * torch.log(Ca_c / Ca_i)
        alg_38_i_B_Ca = g_B_Ca * (V - alg_37_E_Ca)
        alg_40_i_CaP = i_CaP_max * Ca_i / (Ca_i + k_CaP)
        alg_41_i_NaCa = k_NaCa * (Na_i ** 3.0 * Ca_c * torch.exp(gamma * F * V / (R * T)) - Na_c ** 3.0 * Ca_i * torch.exp((gamma - 1.0) * V * F / (R * T))) / (1.0 + d_NaCa * (Na_c ** 3.0 * Ca_i + Na_i ** 3.0 * Ca_c))
        alg_0_i_Stim = torch.where((t >= stim_start) & (t <= stim_end) & (t - stim_start - torch.floor((t - stim_start) / stim_period) * stim_period <= stim_duration), stim_amplitude, 0.0)
        d_V = torch.zeros_like(y[:, 0]) + (-1.0 / Cm * (alg_0_i_Stim + alg_26_i_Na + alg_28_i_Ca_L + alg_30_i_t + alg_31_i_sus + alg_35_i_K1 + alg_34_i_Kr + alg_32_i_Ks + alg_36_i_B_Na + alg_38_i_B_Ca + alg_39_i_NaK + alg_40_i_CaP + alg_41_i_NaCa))
        d_Na_i = torch.zeros_like(y[:, 0]) + (-(alg_26_i_Na + alg_36_i_B_Na + 3.0 * alg_39_i_NaK + 3.0 * alg_41_i_NaCa + phi_Na_en) / (Vol_i * F))
        d_Na_c = torch.zeros_like(y[:, 0]) + ((Na_b - Na_c) / tau_Na + (alg_26_i_Na + alg_36_i_B_Na + 3.0 * alg_39_i_NaK + 3.0 * alg_41_i_NaCa + phi_Na_en) / (Vol_c * F))
        d_Ca_c = torch.zeros_like(y[:, 0]) + ((Ca_b - Ca_c) / tau_Ca + (alg_28_i_Ca_L + alg_38_i_B_Ca + alg_40_i_CaP - 2.0 * alg_41_i_NaCa) / (2.0 * Vol_c * F))
        d_O_C = torch.zeros_like(y[:, 0]) + (200000.0 * Ca_i * (1.0 - O_C) - 476.0 * O_C)
        alg_42_i_di = (Ca_d - Ca_i) * 2.0 * F * Vol_d / tau_di
        d_Ca_d = torch.zeros_like(y[:, 0]) + (-(alg_28_i_Ca_L + alg_42_i_di) / (2.0 * Vol_d * F))
        d_O_TC = torch.zeros_like(y[:, 0]) + (78400.0 * Ca_i * (1.0 - O_TC) - 392.0 * O_TC)
        d_O_TMgC = torch.zeros_like(y[:, 0]) + (200000.0 * Ca_i * (1.0 - O_TMgC - O_TMgMg) - 6.6 * O_TMgC)
        alg_46_i_up = I_up_max * (Ca_i / k_cyca - k_xcs ** 2.0 * Ca_up / k_srca) / ((Ca_i + k_cyca) / k_cyca + k_xcs * (Ca_up + k_srca) / k_srca)
        alg_47_i_tr = (Ca_up - Ca_rel) * 2.0 * F * Vol_rel / tau_tr
        d_Ca_up = torch.zeros_like(y[:, 0]) + ((alg_46_i_up - alg_47_i_tr) / (2.0 * Vol_up * F))
        alg_48_i_rel = alpha_rel * (F2 / (F2 + 0.25)) ** 2.0 * (Ca_rel - Ca_i)
        alg_43_dOCdt = d_O_C
        alg_44_dOTCdt = d_O_TC
        alg_45_dOTMgCdt = d_O_TMgC
        d_Ca_i = torch.zeros_like(y[:, 0]) + (-(-alg_42_i_di + alg_38_i_B_Ca + alg_40_i_CaP - 2.0 * alg_41_i_NaCa + alg_46_i_up - alg_48_i_rel) / (2.0 * Vol_i * F) - (0.08 * alg_44_dOTCdt + 0.16 * alg_45_dOTMgCdt + 0.045 * alg_43_dOCdt))
        d_O_Calse = torch.zeros_like(y[:, 0]) + (480.0 * Ca_rel * (1.0 - O_Calse) - 400.0 * O_Calse)
        d_Ca_rel = torch.zeros_like(y[:, 0]) + ((alg_47_i_tr - alg_48_i_rel) / (2.0 * Vol_rel * F) - 31.0 * d_O_Calse)

        return torch.stack(
            [d_V, d_Na_c, d_Na_i, d_m, d_h1, d_h2, d_Ca_d, d_d_L, d_f_L_1, d_f_L_2, d_K_c, d_K_i, d_r, d_s, d_r_sus, d_s_sus, d_n, d_p_a, d_Ca_c, d_Ca_i, d_O_C, d_O_TC, d_O_TMgC, d_O_TMgMg, d_Ca_rel, d_Ca_up, d_O_Calse, d_F1, d_F2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
