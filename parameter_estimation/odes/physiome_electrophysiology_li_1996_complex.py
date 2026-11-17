import torch
import torch.nn as nn
import math


class ElectrophysiologyLi1996ComplexODE(nn.Module):
    """
    PyTorch translation of electrophysiology/li_1996_complex/model.py.
    Dynamic states: 12; independent constants: 36; derived constants used by rates: 1.
    Original constant slots: 37; algebraic slots: 26.
    """
    source_file = 'electrophysiology/li_1996_complex/model.py'
    source_state_dim = 12
    source_constant_dim = 37
    algebraic_dim = 26
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 12
        self.parameter_dim = 36

        self.parameters = (
            parameters
            if parameters is not None
            else {'V_Na': 55, 'g_Na_s': 3.2, 'g_K_DR_s': 6.4, 'g_Ca_T': 1.5, 'g_K_Ca': 1.2, 'g_A': 2, 'g_h': 0.1, 'k_Ca': 1, 'K_Ca': 0.0004, 'V_h': -30, 'beta': 0.104, 'I_APP': -6.7, 'V_Ca': 120, 'V_K': -85, 'g_c': 0.1, 'p': 0.5, 'C_m': 1, 'K_p': 15, 'R_pump': 18, 'Na_eq': 8, 'alpha': 0.173, 'g_NMDA': 25, 'g_Na_NMDA': 5, 'g_L': 0.18, 'Mg_o': 1.4, 'K_Mg': 10, 'q': 12.5, 'V_NMDA': 0, 'V_L': -50, 'g_Ca_L': 0.19, 'g_K_DR_D': 0.14, 'tau_m_T': 1, 'tau_h_T': 10, 'tau_a': 0.5, 'tau_b': 10, 'tau_m_h': 190}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V_s': -64, 'Ca': 0, 'n': 0.002, 'h': 1, 'V_D': -77, 'm_T': 0.1, 'h_T': 0.1, 'a': 0.1, 'b': 0.1, 'm_h': 0.1, 'Na': 5.09, 'm_L': 0.1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['V_Na', 'g_Na_s', 'g_K_DR_s', 'g_Ca_T', 'g_K_Ca', 'g_A', 'g_h', 'k_Ca', 'K_Ca', 'V_h', 'beta', 'I_APP', 'V_Ca', 'V_K', 'g_c', 'p', 'C_m', 'K_p', 'R_pump', 'Na_eq', 'alpha', 'g_NMDA', 'g_Na_NMDA', 'g_L', 'Mg_o', 'K_Mg', 'q', 'V_NMDA', 'V_L', 'g_Ca_L', 'g_K_DR_D', 'tau_m_T', 'tau_h_T', 'tau_a', 'tau_b', 'tau_m_h']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V_s', 'Ca', 'n', 'h', 'V_D', 'm_T', 'h_T', 'a', 'b', 'm_h', 'Na', 'm_L']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (12)
        V_s = y[:, 0]
        Ca = y[:, 1]
        n = y[:, 2]
        h = y[:, 3]
        V_D = y[:, 4]
        m_T = y[:, 5]
        h_T = y[:, 6]
        a = y[:, 7]
        b = y[:, 8]
        m_h = y[:, 9]
        Na = y[:, 10]
        m_L = y[:, 11]

        # Independent parameters (source constant order)
        V_Na = theta[:, 0]  # constants[0]
        g_Na_s = theta[:, 1]  # constants[1]
        g_K_DR_s = theta[:, 2]  # constants[2]
        g_Ca_T = theta[:, 3]  # constants[3]
        g_K_Ca = theta[:, 4]  # constants[4]
        g_A = theta[:, 5]  # constants[5]
        g_h = theta[:, 6]  # constants[6]
        k_Ca = theta[:, 7]  # constants[7]
        K_Ca = theta[:, 8]  # constants[8]
        V_h = theta[:, 9]  # constants[9]
        beta = theta[:, 10]  # constants[10]
        I_APP = theta[:, 11]  # constants[11]
        V_Ca = theta[:, 12]  # constants[12]
        V_K = theta[:, 13]  # constants[13]
        g_c = theta[:, 14]  # constants[14]
        p = theta[:, 15]  # constants[15]
        C_m = theta[:, 16]  # constants[16]
        K_p = theta[:, 17]  # constants[17]
        R_pump = theta[:, 18]  # constants[18]
        Na_eq = theta[:, 19]  # constants[19]
        alpha = theta[:, 20]  # constants[20]
        g_NMDA = theta[:, 21]  # constants[21]
        g_Na_NMDA = theta[:, 22]  # constants[22]
        g_L = theta[:, 23]  # constants[23]
        Mg_o = theta[:, 24]  # constants[24]
        K_Mg = theta[:, 25]  # constants[25]
        q = theta[:, 26]  # constants[26]
        V_NMDA = theta[:, 27]  # constants[27]
        V_L = theta[:, 28]  # constants[28]
        g_Ca_L = theta[:, 29]  # constants[29]
        g_K_DR_D = theta[:, 30]  # constants[30]
        tau_m_T = theta[:, 31]  # constants[32]
        tau_h_T = theta[:, 32]  # constants[33]
        tau_a = theta[:, 33]  # constants[34]
        tau_b = theta[:, 34]  # constants[35]
        tau_m_h = theta[:, 35]  # constants[36]

        # Derived constants from the original initConsts()
        phi_Na_eq = Na_eq ** 3.0 / (Na_eq ** 3.0 + K_p ** 3.0)  # constants[31]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V_s = torch.zeros_like(y[:, 0])
        d_Ca = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_V_D = torch.zeros_like(y[:, 0])
        d_m_T = torch.zeros_like(y[:, 0])
        d_h_T = torch.zeros_like(y[:, 0])
        d_a = torch.zeros_like(y[:, 0])
        d_b = torch.zeros_like(y[:, 0])
        d_m_h = torch.zeros_like(y[:, 0])
        d_Na = torch.zeros_like(y[:, 0])
        d_m_L = torch.zeros_like(y[:, 0])
        alg_0_I_K_DR_s = torch.zeros_like(y[:, 0])
        alg_1_I_K_DR_D = torch.zeros_like(y[:, 0])
        alg_2_m_T_infinity = torch.zeros_like(y[:, 0])
        alg_3_h_T_infinity = torch.zeros_like(y[:, 0])
        alg_4_a_infinity = torch.zeros_like(y[:, 0])
        alg_5_b_infinity = torch.zeros_like(y[:, 0])
        alg_6_m_h_infinity = torch.zeros_like(y[:, 0])
        alg_7_m_L_infinity = torch.zeros_like(y[:, 0])
        alg_8_n_infinity = torch.zeros_like(y[:, 0])
        alg_9_h_infinity = torch.zeros_like(y[:, 0])
        alg_10_I_Ca_T = torch.zeros_like(y[:, 0])
        alg_11_phi_Na = torch.zeros_like(y[:, 0])
        alg_12_tau_h = torch.zeros_like(y[:, 0])
        alg_13_tau_n = torch.zeros_like(y[:, 0])
        alg_14_tau_m_L = torch.zeros_like(y[:, 0])
        alg_15_I_K_Ca = torch.zeros_like(y[:, 0])
        alg_16_I_pump = torch.zeros_like(y[:, 0])
        alg_17_I_A = torch.zeros_like(y[:, 0])
        alg_18_I_L = torch.zeros_like(y[:, 0])
        alg_19_I_Na_NMDA = torch.zeros_like(y[:, 0])
        alg_20_I_h = torch.zeros_like(y[:, 0])
        alg_21_I_NMDA = torch.zeros_like(y[:, 0])
        alg_22_m_infinity = torch.zeros_like(y[:, 0])
        alg_24_I_Ca_L = torch.zeros_like(y[:, 0])
        alg_25_I_Na_s = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_2_m_T_infinity = 1.0 / (1.0 + torch.exp(-(V_s + 55.0) / 7.0))
        d_m_T = torch.zeros_like(y[:, 0]) + ((alg_2_m_T_infinity - m_T) / tau_m_T)
        alg_6_m_h_infinity = 1.0 / (1.0 + torch.exp((V_s + 80.0) / 8.0))
        d_m_h = torch.zeros_like(y[:, 0]) + ((alg_6_m_h_infinity - m_h) / tau_m_h)
        alg_4_a_infinity = 1.0 / (1.0 + torch.exp(-(V_s + 60.0) / 10.0))
        d_a = torch.zeros_like(y[:, 0]) + ((alg_4_a_infinity - a) / tau_a)
        alg_5_b_infinity = 1.0 / (1.0 + torch.exp((V_s + 70.0) / 5.7))
        d_b = torch.zeros_like(y[:, 0]) + ((alg_5_b_infinity - b) / tau_b)
        alg_3_h_T_infinity = 1.0 / (1.0 + torch.exp((V_s + 81.0) / 11.0))
        d_h_T = torch.zeros_like(y[:, 0]) + ((alg_3_h_T_infinity - h_T) / tau_h_T)
        alg_10_I_Ca_T = g_Ca_T * h_T * (V_s - V_Ca) * m_T ** 2.0
        d_Ca = torch.zeros_like(y[:, 0]) + (-(beta * alg_10_I_Ca_T + k_Ca * Ca))
        alg_9_h_infinity = 1.0 / (1.0 + torch.exp((V_s + 30.0) / 8.3))
        alg_12_tau_h = 0.4 * (1.0 + 2.0 / (1.0 + torch.exp((V_s + 25.0) / 5.0)))
        d_h = torch.zeros_like(y[:, 0]) + ((alg_9_h_infinity - h) / alg_12_tau_h)
        alg_8_n_infinity = 1.0 / (1.0 + torch.exp(-(V_s + 31.0) / 5.3))
        alg_13_tau_n = 0.8 * (1.0 + 2.0 / (1.0 + torch.exp((V_s + 25.0) / 10.0))) / (1.0 + torch.exp(-(V_s + 70.0) / 10.0))
        d_n = torch.zeros_like(y[:, 0]) + ((alg_8_n_infinity - n) / alg_13_tau_n)
        alg_7_m_L_infinity = 1.0 / (1.0 + torch.exp(-(V_D + 20.0) / 5.3))
        alg_14_tau_m_L = 0.4 / (5.0 * torch.exp(-(V_D + 11.0) / 8.3) + -(V_D + 11.0) / 8.3 / (torch.exp(-(V_D + 11.0) / 8.3) - 1.0))
        d_m_L = torch.zeros_like(y[:, 0]) + ((alg_7_m_L_infinity - m_L) / alg_14_tau_m_L)
        alg_11_phi_Na = Na ** 3.0 / (Na ** 3.0 + K_p ** 3.0)
        alg_16_I_pump = R_pump * (alg_11_phi_Na - phi_Na_eq)
        alg_19_I_Na_NMDA = g_Na_NMDA / (1.0 + Mg_o / K_Mg * torch.exp(-V_D / q)) * (V_D - V_Na)
        d_Na = torch.zeros_like(y[:, 0]) + (alpha * (-alg_19_I_Na_NMDA - alg_16_I_pump * 3.0))
        alg_18_I_L = g_L * (V_D - V_L)
        alg_21_I_NMDA = g_NMDA / (1.0 + Mg_o / K_Mg * torch.exp(-V_D / q)) * (V_D - V_NMDA)
        alg_24_I_Ca_L = g_Ca_L * (V_D - V_Ca) * m_L ** 2.0
        alg_1_I_K_DR_D = g_K_DR_D * (V_D - V_K) * n ** 2.0
        d_V_D = torch.zeros_like(y[:, 0]) + (-(1000.0 * (alg_24_I_Ca_L + alg_1_I_K_DR_D + alg_21_I_NMDA + alg_16_I_pump + alg_18_I_L + g_c / (1.0 - p) * (V_D - V_s))) / C_m)
        alg_22_m_infinity = 1.0 / (1.0 + torch.exp(-(V_s + 35.0) / 6.2))
        alg_25_I_Na_s = g_Na_s * h * (V_s - V_Na) * alg_22_m_infinity ** 3.0
        alg_0_I_K_DR_s = g_K_DR_s * (V_s - V_K) * n ** 2.0
        alg_15_I_K_Ca = g_K_Ca * Ca ** 4.0 / (Ca ** 4.0 + K_Ca ** 4.0) * (V_s - V_K)
        alg_17_I_A = g_A * b * (V_s - V_K) * a ** 4.0
        alg_20_I_h = g_h * m_h * (V_s - V_h)
        d_V_s = torch.zeros_like(y[:, 0]) + (1000.0 * (I_APP - (alg_25_I_Na_s + alg_10_I_Ca_T + alg_0_I_K_DR_s + alg_15_I_K_Ca + alg_17_I_A + alg_20_I_h + g_c / p * (V_s - V_D))) / C_m)

        return torch.stack(
            [d_V_s, d_Ca, d_n, d_h, d_V_D, d_m_T, d_h_T, d_a, d_b, d_m_h, d_Na, d_m_L],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
