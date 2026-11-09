import torch
import torch.nn as nn
import math


class ElectrophysiologyKatsnelson2004cODE(nn.Module):
    """
    PyTorch translation of electrophysiology/katsnelson_2004c/model.py.
    Dynamic states: 9; independent constants: 50; derived constants used by rates: 4.
    Original constant slots: 54; algebraic slots: 34.
    """
    source_file = 'electrophysiology/katsnelson_2004c/model.py'
    source_state_dim = 9
    source_constant_dim = 54
    algebraic_dim = 34
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 9
        self.parameter_dim = 50

        self.parameters = (
            parameters
            if parameters is not None
            else {'isotonic': 0, 'alpha_1': 19, 'beta_1': 0.29, 'alpha_2': 14.6, 'beta_2': 0.000924, 'alpha_3': 48, 'beta_3': 0.01, 'lambda_var': 96, 'A_half': 0.6, 'mu': 3, 'chi': 0.705, 'chi_0': 3, 'm_0': 0.9, 'v_max': 5.6, 'a': 0.25, 'd_h': 0.5, 'alpha_P': 4, 'F_afterload': 2, 'l_0': 0.527, 'S_0': 1.14, 'q_1': 17.3, 'q_2': 259, 'q_3': 17.3, 'q_4': 15, 'v_star': 5.3035675, 'alpha_G': 1, 'a_on': 2300, 'a_off': 290, 'k_A': 2.8, 'alpha_Q': 10, 'beta_Q': 5000, 'v_0': 560, 'q_star': 1000, 'alpha_P_lengthening': 16, 'beta_P_lengthening': 0.0015, 'alpha_P_shortening': 16, 'beta_P_shortening': 0.0015, 'alpha_S_lengthening': 39, 'beta_S_lengthening': 0.008, 'alpha_S_shortening': 46, 'beta_S_shortening': 0.006, 'A_tot': 1, 'B_tot': 0.4, 'b_on': 2600, 'b_off': 182, 'a_c': 5200, 'r_Ca': 650, 'q_Ca': 50, 't_d': 0.033, 'Ca_m': 0.03}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'v': 0, 'N': 0.0001, 'w': 0, 'l_1': 0.437, 'l_2': 0.439, 'l_3': 0.089, 'A': 0.01, 'B': 0, 'Ca_C': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['isotonic', 'alpha_1', 'beta_1', 'alpha_2', 'beta_2', 'alpha_3', 'beta_3', 'lambda_var', 'A_half', 'mu', 'chi', 'chi_0', 'm_0', 'v_max', 'a', 'd_h', 'alpha_P', 'F_afterload', 'l_0', 'S_0', 'q_1', 'q_2', 'q_3', 'q_4', 'v_star', 'alpha_G', 'a_on', 'a_off', 'k_A', 'alpha_Q', 'beta_Q', 'v_0', 'q_star', 'alpha_P_lengthening', 'beta_P_lengthening', 'alpha_P_shortening', 'beta_P_shortening', 'alpha_S_lengthening', 'beta_S_lengthening', 'alpha_S_shortening', 'beta_S_shortening', 'A_tot', 'B_tot', 'b_on', 'b_off', 'a_c', 'r_Ca', 'q_Ca', 't_d', 'Ca_m']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['v', 'N', 'w', 'l_1', 'l_2', 'l_3', 'A', 'B', 'Ca_C']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (9)
        v = y[:, 0]
        N = y[:, 1]
        w = y[:, 2]
        l_1 = y[:, 3]
        l_2 = y[:, 4]
        l_3 = y[:, 5]
        A = y[:, 6]
        B = y[:, 7]
        Ca_C = y[:, 8]

        # Independent parameters (source constant order)
        isotonic = theta[:, 0]  # constants[0]
        alpha_1 = theta[:, 1]  # constants[1]
        beta_1 = theta[:, 2]  # constants[2]
        alpha_2 = theta[:, 3]  # constants[3]
        beta_2 = theta[:, 4]  # constants[4]
        alpha_3 = theta[:, 5]  # constants[5]
        beta_3 = theta[:, 6]  # constants[6]
        lambda_var = theta[:, 7]  # constants[7]
        A_half = theta[:, 8]  # constants[8]
        mu = theta[:, 9]  # constants[9]
        chi = theta[:, 10]  # constants[10]
        chi_0 = theta[:, 11]  # constants[11]
        m_0 = theta[:, 12]  # constants[12]
        v_max = theta[:, 13]  # constants[13]
        a = theta[:, 14]  # constants[14]
        d_h = theta[:, 15]  # constants[15]
        alpha_P = theta[:, 16]  # constants[16]
        F_afterload = theta[:, 17]  # constants[17]
        l_0 = theta[:, 18]  # constants[18]
        S_0 = theta[:, 19]  # constants[19]
        q_1 = theta[:, 20]  # constants[20]
        q_2 = theta[:, 21]  # constants[21]
        q_3 = theta[:, 22]  # constants[22]
        q_4 = theta[:, 23]  # constants[23]
        v_star = theta[:, 24]  # constants[24]
        alpha_G = theta[:, 25]  # constants[25]
        a_on = theta[:, 26]  # constants[26]
        a_off = theta[:, 27]  # constants[27]
        k_A = theta[:, 28]  # constants[28]
        alpha_Q = theta[:, 29]  # constants[29]
        beta_Q = theta[:, 30]  # constants[30]
        v_0 = theta[:, 31]  # constants[31]
        q_star = theta[:, 32]  # constants[32]
        alpha_P_lengthening = theta[:, 33]  # constants[33]
        beta_P_lengthening = theta[:, 34]  # constants[34]
        alpha_P_shortening = theta[:, 35]  # constants[35]
        beta_P_shortening = theta[:, 36]  # constants[36]
        alpha_S_lengthening = theta[:, 37]  # constants[37]
        beta_S_lengthening = theta[:, 38]  # constants[38]
        alpha_S_shortening = theta[:, 39]  # constants[39]
        beta_S_shortening = theta[:, 40]  # constants[40]
        A_tot = theta[:, 41]  # constants[41]
        B_tot = theta[:, 42]  # constants[42]
        b_on = theta[:, 43]  # constants[43]
        b_off = theta[:, 44]  # constants[44]
        a_c = theta[:, 45]  # constants[45]
        r_Ca = theta[:, 46]  # constants[46]
        q_Ca = theta[:, 47]  # constants[47]
        t_d = theta[:, 48]  # constants[48]
        Ca_m = theta[:, 49]  # constants[49]

        # Derived constants from the original initConsts()
        v_1 = v_max / 10.0  # constants[50]
        case_1 = a * (0.4 + 0.4 * a) / (v_max * ((a + 1.0) * 0.4) ** 2.0)  # constants[51]
        case_3 = (0.4 * a + 1.0) / (a * v_max)  # constants[52]
        gamma = a * d_h * (v_1 / v_max) ** 2.0 / (3.0 * a * d_h - (a + 1.0) * v_1 / v_max)  # constants[53]

        # Initialize rates and algebraic variables, as in computeRates()
        d_v = torch.zeros_like(y[:, 0])
        d_N = torch.zeros_like(y[:, 0])
        d_w = torch.zeros_like(y[:, 0])
        d_l_1 = torch.zeros_like(y[:, 0])
        d_l_2 = torch.zeros_like(y[:, 0])
        d_l_3 = torch.zeros_like(y[:, 0])
        d_A = torch.zeros_like(y[:, 0])
        d_B = torch.zeros_like(y[:, 0])
        d_Ca_C = torch.zeros_like(y[:, 0])
        alg_0_q_v = torch.zeros_like(y[:, 0])
        alg_3_dl_1_dt = torch.zeros_like(y[:, 0])
        alg_4_F_XSE = torch.zeros_like(y[:, 0])
        alg_5_F_muscle = torch.zeros_like(y[:, 0])
        alg_6_M_A = torch.zeros_like(y[:, 0])
        alg_7_n_1 = torch.zeros_like(y[:, 0])
        alg_8_L_oz = torch.zeros_like(y[:, 0])
        alg_9_l = torch.zeros_like(y[:, 0])
        alg_10_N_A = torch.zeros_like(y[:, 0])
        alg_11_flag = torch.zeros_like(y[:, 0])
        alg_12_pi_N_A = torch.zeros_like(y[:, 0])
        alg_13_isotonic_mode = torch.zeros_like(y[:, 0])
        alg_14_dA_dt = torch.zeros_like(y[:, 0])
        alg_15_alp_p = torch.zeros_like(y[:, 0])
        alg_16_dB_dt = torch.zeros_like(y[:, 0])
        alg_17_k_P_vis = torch.zeros_like(y[:, 0])
        alg_18_phi_chi_2 = torch.zeros_like(y[:, 0])
        alg_19_alp_s = torch.zeros_like(y[:, 0])
        alg_20_k_S_vis = torch.zeros_like(y[:, 0])
        alg_21_P_star = torch.zeros_like(y[:, 0])
        alg_22_dl_2_dt = torch.zeros_like(y[:, 0])
        alg_23_dl_3_dt = torch.zeros_like(y[:, 0])
        alg_24_G_star = torch.zeros_like(y[:, 0])
        alg_25_k_p_v = torch.zeros_like(y[:, 0])
        alg_26_k_m_v = torch.zeros_like(y[:, 0])
        alg_27_K_chi = torch.zeros_like(y[:, 0])
        alg_28_case_2 = torch.zeros_like(y[:, 0])
        alg_29_case_4 = torch.zeros_like(y[:, 0])
        alg_30_p_v = torch.zeros_like(y[:, 0])
        alg_32_p_prime_v = torch.zeros_like(y[:, 0])
        alg_33_phi_chi = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_3_dl_1_dt = v
        d_l_1 = torch.zeros_like(y[:, 0]) + (alg_3_dl_1_dt)
        alg_8_L_oz = (l_1 + S_0) / (0.46 + S_0)
        alg_10_N_A = N / (alg_8_L_oz * A)
        alg_12_pi_N_A = torch.where(alg_10_N_A >= 1.0, 1.0, 0.02 ** alg_10_N_A)
        alg_14_dA_dt = a_on * (A_tot - A) * Ca_C - a_off * torch.exp(-k_A * A) * alg_12_pi_N_A * A
        d_A = torch.zeros_like(y[:, 0]) + (alg_14_dA_dt)
        alg_16_dB_dt = b_on * (B_tot - B) * Ca_C - b_off * B
        d_B = torch.zeros_like(y[:, 0]) + (alg_16_dB_dt)
        d_Ca_C = torch.zeros_like(y[:, 0]) + (torch.where(t < t_d, 4.0 * a_c * Ca_m * t * (1.0 - torch.exp(-a_c * t ** 2.0)) * torch.exp(-a_c * t ** 2.0), -alg_14_dA_dt - alg_16_dB_dt - r_Ca * torch.exp(-q_Ca * Ca_C) * Ca_C))
        alg_20_k_S_vis = torch.where(w <= v, beta_S_lengthening * torch.exp(alpha_S_lengthening * (l_2 - l_1)), beta_S_shortening * torch.exp(alpha_S_shortening * (l_2 - l_1)))
        alg_9_l = l_2 + l_3
        alg_4_F_XSE = beta_3 * (torch.exp(alpha_3 * l_3) - 1.0)
        alg_5_F_muscle = alg_4_F_XSE
        alg_11_flag = torch.where((alg_9_l >= l_0) & (t < 0.15), 0.0, 1.0)
        alg_13_isotonic_mode = torch.where(isotonic == 0.0, 0.0, torch.where((isotonic == 1.0) & (alg_5_F_muscle >= F_afterload), 1.0, torch.where((isotonic == 1.0) & (alg_9_l >= l_0) & (alg_11_flag == 1.0), 0.0, (1e309-1e309))))
        alg_18_phi_chi_2 = torch.where(alg_13_isotonic_mode == 1.0, alpha_1 * beta_1 * torch.exp(alpha_1 * (l_2 - l_1)) * v / (alpha_1 * beta_1 * torch.exp(alpha_1 * (l_2 - l_1)) + alpha_2 * beta_2 * torch.exp(alpha_2 * l_2)), alpha_1 * beta_1 * torch.exp(alpha_1 * (l_2 - l_1)) * v / (alpha_1 * beta_1 * torch.exp(alpha_1 * (l_2 - l_1)) + alpha_2 * beta_2 * torch.exp(alpha_2 * l_2) + alpha_3 * beta_3 * torch.exp(alpha_3 * l_3)))
        alg_22_dl_2_dt = torch.where(alg_20_k_S_vis == 0.0, alg_18_phi_chi_2, w)
        d_l_2 = torch.zeros_like(y[:, 0]) + (alg_22_dl_2_dt)
        alg_23_dl_3_dt = torch.where(alg_13_isotonic_mode == 1.0, 0.0, torch.where((alg_13_isotonic_mode == 0.0) & (alg_20_k_S_vis == 0.0), -alg_18_phi_chi_2, -w))
        d_l_3 = torch.zeros_like(y[:, 0]) + (alg_23_dl_3_dt)
        alg_6_M_A = A ** mu / (A ** mu + A_half ** mu)
        alg_7_n_1 = 0.6 * l_1 + 0.5
        alg_0_q_v = torch.where(v <= 0.0, q_1 - q_2 * v / v_max, torch.where((v <= v_star) & (0.0 < v), (q_4 - q_3) * v / v_star + q_3, q_4 / (1.0 + beta_Q * (v - v_star) / v_max) ** alpha_Q))
        alg_21_P_star = torch.where(v <= 0.0, a * (1.0 + v / v_max) / (a - v / v_max), 1.0 + d_h - d_h ** 2.0 * a / (a * d_h / gamma * (v / v_max) ** 2.0 + (a + 1.0) * v / v_max + a * d_h))
        alg_24_G_star = torch.where((-v_max <= v) & (v <= 0.0), 1.0 + 0.6 * v / v_max, torch.where((0.0 < v) & (v <= v_1), alg_21_P_star / ((0.4 * a + 1.0) * v / (a * v_max) + 1.0), alg_21_P_star * torch.exp(-alpha_G * ((v - v_1) / v_max) ** alpha_P) / ((0.4 * a + 1.0) * v / (a * v_max) + 1.0)))
        alg_25_k_p_v = chi * chi_0 * alg_0_q_v * m_0 * alg_24_G_star
        alg_26_k_m_v = torch.where(v <= v_star, chi_0 * alg_0_q_v * (1.0 - chi * m_0 * alg_24_G_star), chi_0 * (q_4 * (1.0 - chi * m_0 * alg_24_G_star) + q_star * (v - v_star) / (v_0 - v_star)))
        alg_27_K_chi = alg_25_k_p_v * alg_6_M_A * alg_7_n_1 * alg_8_L_oz * (1.0 - N) - alg_26_k_m_v * N
        d_N = torch.zeros_like(y[:, 0]) + (alg_27_K_chi)
        alg_17_k_P_vis = torch.where(v <= 0.0, beta_P_lengthening * torch.exp(alpha_P_lengthening * l_1), beta_P_shortening * torch.exp(alpha_P_shortening * l_1))
        alg_30_p_v = alg_21_P_star / alg_24_G_star
        alg_28_case_2 = a * 1.0 * (1.0 + 0.4 * a + 1.2 * v / v_max + 0.6 * (v / v_max) ** 2.0) / (v_max * ((a - v / v_max) * (1.0 + 0.6 * v / v_max)) ** 2.0)
        alg_29_case_4 = 1.0 / v_max * torch.exp(-alpha_G * (v / v_max - v_1 / v_max) ** alpha_P) * ((0.4 * a + 1.0) / a + alpha_G * alpha_P * (1.0 + (0.4 * a + 1.0) * v / (a * v_max)) * (v / v_max - v_1 / v_max) ** (alpha_P - 1.0))
        alg_32_p_prime_v = torch.where(v <= -v_max, case_1, torch.where((-v_max < v) & (v <= 0.0), alg_28_case_2, torch.where((0.0 < v) & (v <= v_1), case_3, alg_29_case_4)))
        alg_15_alp_p = torch.where(v <= 0.0, alpha_P_lengthening, alpha_P_shortening)
        alg_33_phi_chi = torch.where(alg_13_isotonic_mode == 1.0, -(lambda_var * alg_27_K_chi * alg_30_p_v + alg_15_alp_p * alg_17_k_P_vis * v ** 2.0 + alpha_2 * beta_2 * torch.exp(alpha_2 * l_2) * w) / (lambda_var * N * alg_32_p_prime_v + alg_17_k_P_vis), -(lambda_var * alg_27_K_chi * alg_30_p_v + alg_15_alp_p * alg_17_k_P_vis * v ** 2.0 + (alpha_2 * beta_2 * torch.exp(alpha_2 * l_2) + alpha_3 * beta_3 * torch.exp(alpha_3 * l_3)) * w) / (lambda_var * N * alg_32_p_prime_v + alg_17_k_P_vis))
        d_v = torch.zeros_like(y[:, 0]) + (torch.where(alg_20_k_S_vis == 0.0, (alpha_1 * beta_1 * torch.exp(alpha_1 * (l_2 - l_1)) * (alg_18_phi_chi_2 - v) - (lambda_var * alg_27_K_chi * alg_30_p_v + alg_15_alp_p * alg_17_k_P_vis * v ** 2.0)) / (lambda_var * N * alg_32_p_prime_v + alg_17_k_P_vis), alg_33_phi_chi))
        alg_19_alp_s = torch.where(w <= v, alpha_S_lengthening, alpha_S_shortening)
        d_w = torch.zeros_like(y[:, 0]) + (torch.where(alg_13_isotonic_mode == 1.0, (alg_20_k_S_vis * (alg_33_phi_chi - alg_19_alp_s * (w - v) ** 2.0) - alpha_1 * beta_1 * torch.exp(alpha_1 * (l_2 - l_1)) * (w - v) - alpha_2 * beta_2 * torch.exp(alpha_2 * l_2) * w) / alg_20_k_S_vis, alg_33_phi_chi - alg_19_alp_s * (w - v) ** 2.0 - (alpha_1 * beta_1 * torch.exp(alpha_1 * (l_2 - l_1)) * (w - v) + (alpha_2 * beta_2 * torch.exp(alpha_2 * l_2) + alpha_3 * beta_3 * torch.exp(alpha_3 * l_3)) * w) / alg_20_k_S_vis))

        return torch.stack(
            [d_v, d_N, d_w, d_l_1, d_l_2, d_l_3, d_A, d_B, d_Ca_C],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
