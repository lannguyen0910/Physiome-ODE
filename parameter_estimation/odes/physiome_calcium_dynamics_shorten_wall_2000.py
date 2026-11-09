import torch
import torch.nn as nn
import math


class CalciumDynamicsShortenWall2000ODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/shorten_wall_2000/model.py.
    Dynamic states: 7; independent constants: 38; derived constants used by rates: 2.
    Original constant slots: 40; algebraic slots: 17.
    """
    source_file = 'calcium_dynamics/shorten_wall_2000/model.py'
    source_state_dim = 7
    source_constant_dim = 40
    algebraic_dim = 17
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 7
        self.parameter_dim = 38

        self.parameters = (
            parameters
            if parameters is not None
            else {'V_cell': 1.77, 'Ca_e': 20, 'K_e': 5.6, 'K_i': 140, 'V_tau': -60, 'k_tau': 22, 'R': 8314, 'T': 310, 'F': 96845, 'Cm': 7, 'g_Ca_L': 9, 'tau_m_L_max': 27, 'V_m_L': -18, 'k_m_L': 12, 'g_Ca_T': 10, 'tau_m_T_max': 10, 'V_m_T': -30, 'k_m_T': 10.5, 'tau_h_T': 15, 'V_h_T': -57, 'k_h_T': 5, 'g_K_DR': 0.1, 'tau_n': 20, 'V_n': -20, 'k_n': 4.5, 'g_K_Ca': 0.09, 'Kc': 0.0004, 'g_L': 0.3, 'V_L': -67, 'K_er': 0.0002, 'f_er': 0.0025, 'P': 0.0012, 'v_er': 5e-05, 'K_p': 8e-05, 'f_cyt': 0.01, 'v_p': 4.5e-05, 'alpha': 7.4e-06, 'beta': 0.47}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -70, 'Ca_i': 0.00026, 'm_L': 0, 'm_T': 0, 'h_T': 0, 'n': 0, 'Ca_er': 0.0172}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['V_cell', 'Ca_e', 'K_e', 'K_i', 'V_tau', 'k_tau', 'R', 'T', 'F', 'Cm', 'g_Ca_L', 'tau_m_L_max', 'V_m_L', 'k_m_L', 'g_Ca_T', 'tau_m_T_max', 'V_m_T', 'k_m_T', 'tau_h_T', 'V_h_T', 'k_h_T', 'g_K_DR', 'tau_n', 'V_n', 'k_n', 'g_K_Ca', 'Kc', 'g_L', 'V_L', 'K_er', 'f_er', 'P', 'v_er', 'K_p', 'f_cyt', 'v_p', 'alpha', 'beta']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'Ca_i', 'm_L', 'm_T', 'h_T', 'n', 'Ca_er']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (7)
        V = y[:, 0]
        Ca_i = y[:, 1]
        m_L = y[:, 2]
        m_T = y[:, 3]
        h_T = y[:, 4]
        n = y[:, 5]
        Ca_er = y[:, 6]

        # Independent parameters (source constant order)
        V_cell = theta[:, 0]  # constants[0]
        Ca_e = theta[:, 1]  # constants[1]
        K_e = theta[:, 2]  # constants[2]
        K_i = theta[:, 3]  # constants[3]
        V_tau = theta[:, 4]  # constants[4]
        k_tau = theta[:, 5]  # constants[5]
        R = theta[:, 6]  # constants[6]
        T = theta[:, 7]  # constants[7]
        F = theta[:, 8]  # constants[8]
        Cm = theta[:, 9]  # constants[9]
        g_Ca_L = theta[:, 10]  # constants[10]
        tau_m_L_max = theta[:, 11]  # constants[11]
        V_m_L = theta[:, 12]  # constants[12]
        k_m_L = theta[:, 13]  # constants[13]
        g_Ca_T = theta[:, 14]  # constants[14]
        tau_m_T_max = theta[:, 15]  # constants[15]
        V_m_T = theta[:, 16]  # constants[16]
        k_m_T = theta[:, 17]  # constants[17]
        tau_h_T = theta[:, 18]  # constants[18]
        V_h_T = theta[:, 19]  # constants[19]
        k_h_T = theta[:, 20]  # constants[20]
        g_K_DR = theta[:, 21]  # constants[21]
        tau_n = theta[:, 22]  # constants[22]
        V_n = theta[:, 23]  # constants[23]
        k_n = theta[:, 24]  # constants[24]
        g_K_Ca = theta[:, 25]  # constants[25]
        Kc = theta[:, 26]  # constants[26]
        g_L = theta[:, 27]  # constants[27]
        V_L = theta[:, 28]  # constants[28]
        K_er = theta[:, 29]  # constants[29]
        f_er = theta[:, 30]  # constants[30]
        P = theta[:, 31]  # constants[31]
        v_er = theta[:, 32]  # constants[32]
        K_p = theta[:, 33]  # constants[33]
        f_cyt = theta[:, 34]  # constants[34]
        v_p = theta[:, 35]  # constants[35]
        alpha = theta[:, 36]  # constants[36]
        beta = theta[:, 37]  # constants[37]

        # Derived constants from the original initConsts()
        V_er = V_cell * 0.15  # constants[38]
        V_c = V_cell * 0.85  # constants[39]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_m_L = torch.zeros_like(y[:, 0])
        d_m_T = torch.zeros_like(y[:, 0])
        d_h_T = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_Ca_er = torch.zeros_like(y[:, 0])
        alg_0_phi_Ca = torch.zeros_like(y[:, 0])
        alg_1_m_L_infinity = torch.zeros_like(y[:, 0])
        alg_2_m_T_infinity = torch.zeros_like(y[:, 0])
        alg_3_h_T_infinity = torch.zeros_like(y[:, 0])
        alg_4_n_infinity = torch.zeros_like(y[:, 0])
        alg_5_i_Ca_L = torch.zeros_like(y[:, 0])
        alg_6_tau_m_L = torch.zeros_like(y[:, 0])
        alg_7_tau_m_T = torch.zeros_like(y[:, 0])
        alg_8_i_Ca_T = torch.zeros_like(y[:, 0])
        alg_9_phi_K = torch.zeros_like(y[:, 0])
        alg_10_J_rel = torch.zeros_like(y[:, 0])
        alg_11_i_K_DR = torch.zeros_like(y[:, 0])
        alg_12_J_up = torch.zeros_like(y[:, 0])
        alg_13_i_K_Ca = torch.zeros_like(y[:, 0])
        alg_14_J_in = torch.zeros_like(y[:, 0])
        alg_15_i_leak = torch.zeros_like(y[:, 0])
        alg_16_J_eff = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_3_h_T_infinity = 1.0 / (1.0 + torch.exp((V - V_h_T) / k_h_T))
        d_h_T = torch.zeros_like(y[:, 0]) + ((alg_3_h_T_infinity - h_T) / tau_h_T)
        alg_4_n_infinity = 1.0 / (1.0 + torch.exp((V_n - V) / k_n))
        d_n = torch.zeros_like(y[:, 0]) + ((alg_4_n_infinity - n) / tau_n)
        alg_1_m_L_infinity = 1.0 / (1.0 + torch.exp((V_m_L - V) / k_m_L))
        alg_6_tau_m_L = tau_m_L_max / (torch.exp((V - V_tau) / k_tau) + 2.0 * torch.exp(2.0 * (V_tau - V) / k_tau))
        d_m_L = torch.zeros_like(y[:, 0]) + ((alg_1_m_L_infinity - m_L) / alg_6_tau_m_L)
        alg_2_m_T_infinity = 1.0 / (1.0 + torch.exp((V_m_T - V) / k_m_T))
        alg_7_tau_m_T = tau_m_T_max / (torch.exp((V - V_tau) / k_tau) + 2.0 * torch.exp(2.0 * (V_tau - V) / k_tau))
        d_m_T = torch.zeros_like(y[:, 0]) + ((alg_2_m_T_infinity - m_T) / alg_7_tau_m_T)
        alg_10_J_rel = P * (Ca_er - Ca_i)
        alg_12_J_up = v_er * Ca_i ** 2.0 / (Ca_i ** 2.0 + K_er ** 2.0)
        d_Ca_er = torch.zeros_like(y[:, 0]) + (-f_er / V_er * (alg_10_J_rel - alg_12_J_up))
        alg_0_phi_Ca = V * (Ca_i - Ca_e) * torch.exp(-2.0 * F * V / (R * T)) / (1.0 - torch.exp(-2.0 * F * V / (R * T)))
        alg_5_i_Ca_L = g_Ca_L * m_L ** 2.0 * alg_0_phi_Ca
        alg_8_i_Ca_T = g_Ca_T * m_T ** 2.0 * h_T * alg_0_phi_Ca
        alg_9_phi_K = V * (K_i - K_e) * torch.exp(-1.0 * F * V / (R * T)) / (1.0 - torch.exp(-1.0 * F * V / (R * T)))
        alg_11_i_K_DR = g_K_DR * n * alg_9_phi_K
        alg_13_i_K_Ca = g_K_Ca * Ca_i ** 4.0 / (Ca_i ** 4.0 + Kc ** 4.0) * alg_9_phi_K
        alg_15_i_leak = g_L * (V - V_L)
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_5_i_Ca_L + alg_8_i_Ca_T + alg_11_i_K_DR + alg_13_i_K_Ca + alg_15_i_leak) / Cm)
        alg_14_J_in = -alpha * (alg_5_i_Ca_L + alg_8_i_Ca_T)
        alg_16_J_eff = v_p * Ca_i ** 2.0 / (Ca_i ** 2.0 + K_p ** 2.0)
        d_Ca_i = torch.zeros_like(y[:, 0]) + (f_cyt / V_c * (alg_10_J_rel - alg_12_J_up) + f_cyt * beta * (alg_14_J_in - alg_16_J_eff))

        return torch.stack(
            [d_V, d_Ca_i, d_m_L, d_m_T, d_h_T, d_n, d_Ca_er],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
