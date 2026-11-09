import torch
import torch.nn as nn
import math


class NeurobiologyLi1996SimpleFromPaperODE(nn.Module):
    """
    PyTorch translation of neurobiology/li_1996_simple_from_paper/model.py.
    Dynamic states: 5; independent constants: 19; derived constants used by rates: 1.
    Original constant slots: 20; algebraic slots: 13.
    """
    source_file = 'neurobiology/li_1996_simple_from_paper/model.py'
    source_state_dim = 5
    source_constant_dim = 20
    algebraic_dim = 13
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 5
        self.parameter_dim = 19

        self.parameters = (
            parameters
            if parameters is not None
            else {'V_Na': 55, 'V_K': -85, 'g_K_DR': 3.2, 'g_Na': 3.2, 'g_c': 0, 'C_m': 1, 'p': 0.5, 'V_L': -50, 'V_NMDA': 0, 'R_pump': 18, 'alpha': 0.5, 'g_NMDA': 1.25, 'g_Na_NMDA': 1, 'g_L': 0.18, 'Na_eq': 8, 'K_p': 15, 'Mg_o': 1.4, 'K_Mg': 10, 'q': 12.5}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V_s': -64, 'V_D': -25, 'n': 0.002, 'h': 1, 'Na': 5.09}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['V_Na', 'V_K', 'g_K_DR', 'g_Na', 'g_c', 'C_m', 'p', 'V_L', 'V_NMDA', 'R_pump', 'alpha', 'g_NMDA', 'g_Na_NMDA', 'g_L', 'Na_eq', 'K_p', 'Mg_o', 'K_Mg', 'q']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V_s', 'V_D', 'n', 'h', 'Na']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        V_s = y[:, 0]
        V_D = y[:, 1]
        n = y[:, 2]
        h = y[:, 3]
        Na = y[:, 4]

        # Independent parameters (source constant order)
        V_Na = theta[:, 0]  # constants[0]
        V_K = theta[:, 1]  # constants[1]
        g_K_DR = theta[:, 2]  # constants[2]
        g_Na = theta[:, 3]  # constants[3]
        g_c = theta[:, 4]  # constants[4]
        C_m = theta[:, 5]  # constants[5]
        p = theta[:, 6]  # constants[6]
        V_L = theta[:, 7]  # constants[7]
        V_NMDA = theta[:, 8]  # constants[8]
        R_pump = theta[:, 9]  # constants[9]
        alpha = theta[:, 10]  # constants[10]
        g_NMDA = theta[:, 11]  # constants[11]
        g_Na_NMDA = theta[:, 12]  # constants[12]
        g_L = theta[:, 13]  # constants[13]
        Na_eq = theta[:, 14]  # constants[14]
        K_p = theta[:, 15]  # constants[15]
        Mg_o = theta[:, 16]  # constants[16]
        K_Mg = theta[:, 17]  # constants[17]
        q = theta[:, 18]  # constants[18]

        # Derived constants from the original initConsts()
        phi_Na_eq = Na_eq ** 3.0 / (Na_eq ** 3.0 + K_p ** 3.0)  # constants[19]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V_s = torch.zeros_like(y[:, 0])
        d_V_D = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_Na = torch.zeros_like(y[:, 0])
        alg_0_I_K_DR = torch.zeros_like(y[:, 0])
        alg_1_I_L = torch.zeros_like(y[:, 0])
        alg_2_n_infinity = torch.zeros_like(y[:, 0])
        alg_3_h_infinity = torch.zeros_like(y[:, 0])
        alg_4_m_infinity = torch.zeros_like(y[:, 0])
        alg_5_phi_Na = torch.zeros_like(y[:, 0])
        alg_6_tau_h = torch.zeros_like(y[:, 0])
        alg_7_tau_n = torch.zeros_like(y[:, 0])
        alg_8_I_Na = torch.zeros_like(y[:, 0])
        alg_9_I_pump = torch.zeros_like(y[:, 0])
        alg_10_I_NMDA = torch.zeros_like(y[:, 0])
        alg_11_I_Na_NMDA = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_3_h_infinity = 1.0 / (1.0 + torch.exp((V_s + 30.0) / 8.3))
        alg_6_tau_h = 0.4 * (1.0 + 2.0 / (1.0 + torch.exp((V_s + 25.0) / 5.0)))
        d_h = torch.zeros_like(y[:, 0]) + ((alg_3_h_infinity - h) / alg_6_tau_h)
        alg_2_n_infinity = 1.0 / (1.0 + torch.exp(-(V_s + 31.0) / 5.3))
        alg_7_tau_n = (0.8 + 1.6 / (1.0 + torch.exp(0.1 * (V_s + 25.0)))) / (1.0 + torch.exp(-0.1 * (V_s + 70.0)))
        d_n = torch.zeros_like(y[:, 0]) + ((alg_2_n_infinity - n) / alg_7_tau_n)
        alg_4_m_infinity = 1.0 / (1.0 + torch.exp(-(V_s + 35.0) / 6.2))
        alg_8_I_Na = g_Na * h * (V_s - V_Na) * alg_4_m_infinity ** 3.0
        alg_0_I_K_DR = g_K_DR * (V_s - V_K) * n ** 2.0
        d_V_s = torch.zeros_like(y[:, 0]) + (-(1000.0 * (alg_0_I_K_DR + alg_8_I_Na + g_c / p * (V_s - V_D))) / C_m)
        alg_1_I_L = g_L * (V_D - V_L)
        alg_5_phi_Na = Na ** 3.0 / (Na ** 3.0 + K_p ** 3.0)
        alg_9_I_pump = R_pump * (alg_5_phi_Na - phi_Na_eq)
        alg_10_I_NMDA = g_NMDA / (1.0 + Mg_o / K_Mg * torch.exp(-(V_D / q))) * (V_D - V_NMDA)
        d_V_D = torch.zeros_like(y[:, 0]) + (-(1000.0 * (alg_10_I_NMDA + alg_9_I_pump + alg_1_I_L + g_c / (1.0 - p) * (V_D - V_s))) / C_m)
        alg_11_I_Na_NMDA = g_Na_NMDA / (1.0 + Mg_o / K_Mg * torch.exp(-(V_D / q))) * (V_D - V_Na)
        d_Na = torch.zeros_like(y[:, 0]) + (alpha * (-alg_11_I_Na_NMDA - 3.0 * alg_9_I_pump))

        return torch.stack(
            [d_V_s, d_V_D, d_n, d_h, d_Na],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
