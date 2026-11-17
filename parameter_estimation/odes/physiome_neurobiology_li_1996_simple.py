import torch
import torch.nn as nn
import math


class NeurobiologyLi1996SimpleODE(nn.Module):
    """
    PyTorch translation of neurobiology/li_1996_simple/model.py.
    Dynamic states: 5; independent constants: 15; derived constants used by rates: 5.
    Original constant slots: 20; algebraic slots: 17.
    """
    source_file = 'neurobiology/li_1996_simple/model.py'
    source_state_dim = 5
    source_constant_dim = 20
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

        self.state_dim = 5
        self.parameter_dim = 15

        self.parameters = (
            parameters
            if parameters is not None
            else {'C_m': 1, 'V_Na': 55, 'V_K': -85, 'g_K_DR': 3.2, 'g_Na': 3.2, 'g_c': 0.1, 'g_h': 0.1, 'p': 0.5, 'V_L': -50, 'alpha': 0.173, 'Na_eq': 8, 'K_p': 15, 'K_Na': 15, 'q': 12.5, 'tau_r': 190}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V_s': -64, 'V_D': -77, 'n': 0.002, 'h': 1, 'Na': 5.09}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['C_m', 'V_Na', 'V_K', 'g_K_DR', 'g_Na', 'g_c', 'g_h', 'p', 'V_L', 'alpha', 'Na_eq', 'K_p', 'K_Na', 'q', 'tau_r']
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
        C_m = theta[:, 0]  # constants[0]
        V_Na = theta[:, 1]  # constants[1]
        V_K = theta[:, 2]  # constants[2]
        g_K_DR = theta[:, 3]  # constants[3]
        g_Na = theta[:, 4]  # constants[4]
        g_c = theta[:, 5]  # constants[5]
        g_h = theta[:, 6]  # constants[6]
        p = theta[:, 7]  # constants[7]
        V_L = theta[:, 8]  # constants[8]
        alpha = theta[:, 9]  # constants[9]
        Na_eq = theta[:, 10]  # constants[10]
        K_p = theta[:, 11]  # constants[11]
        K_Na = theta[:, 12]  # constants[12]
        q = theta[:, 13]  # constants[13]
        tau_r = theta[:, 14]  # constants[18]

        # Derived constants from the original initConsts()
        R_pump = 18.0 * (p / (1.0 - p))  # constants[14]
        g_NMDA = 1.25 * (p / (1.0 - p))  # constants[15]
        g_Na_NMDA = 1.0 * (p / (1.0 - p))  # constants[16]
        g_L = 0.18 * (p / (1.0 - p))  # constants[17]
        I_pump_ss = R_pump * p / (1.0 - p) * Na_eq ** 3.0 / (K_Na ** 3.0 + Na_eq ** 3.0)  # constants[19]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V_s = torch.zeros_like(y[:, 0])
        d_V_D = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_Na = torch.zeros_like(y[:, 0])
        alg_0_I_h = torch.zeros_like(y[:, 0])
        alg_1_f_NMDA = torch.zeros_like(y[:, 0])
        alg_2_n_infinity = torch.zeros_like(y[:, 0])
        alg_3_h_infinity = torch.zeros_like(y[:, 0])
        alg_6_I_K_DR = torch.zeros_like(y[:, 0])
        alg_7_I_pump = torch.zeros_like(y[:, 0])
        alg_8_tau_h = torch.zeros_like(y[:, 0])
        alg_9_tau_n = torch.zeros_like(y[:, 0])
        alg_10_m_infinity = torch.zeros_like(y[:, 0])
        alg_11_I_L = torch.zeros_like(y[:, 0])
        alg_12_I_Na_NMDA = torch.zeros_like(y[:, 0])
        alg_13_I_Na = torch.zeros_like(y[:, 0])
        alg_14_I_NMDA = torch.zeros_like(y[:, 0])
        alg_15_I_soma = torch.zeros_like(y[:, 0])
        alg_16_I_D = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_3_h_infinity = 1.0 / (1.0 + torch.exp((V_s + 30.0) / 8.3))
        alg_8_tau_h = 0.43 + 0.86 / (1.0 + torch.exp((V_s + 25.0) / 5.0))
        d_h = torch.zeros_like(y[:, 0]) + ((alg_3_h_infinity - h) / alg_8_tau_h)
        alg_2_n_infinity = 1.0 / (1.0 + torch.exp(-(V_s + 31.0) / 5.3))
        alg_9_tau_n = (0.8 + 1.6 / (1.0 + torch.exp(0.1 * (V_s + 25.0)))) / (1.0 + torch.exp(-0.1 * (V_s + 70.0)))
        d_n = torch.zeros_like(y[:, 0]) + ((alg_2_n_infinity - n) / alg_9_tau_n)
        alg_7_I_pump = R_pump * p / (1.0 - p) * (V_D * 1.0 + Na) ** 3.0 / (K_p ** 3.0 + (V_D * 1.0 + Na) ** 3.0)
        alg_1_f_NMDA = 1.0 / (1.0 + 0.141 * torch.exp(-V_D / q))
        alg_12_I_Na_NMDA = g_Na_NMDA * alg_1_f_NMDA * (V_D - V_Na)
        d_Na = torch.zeros_like(y[:, 0]) + (alpha * (-alg_12_I_Na_NMDA - 3.0 * (alg_7_I_pump - I_pump_ss)))
        alg_10_m_infinity = 1.0 / (1.0 + torch.exp(-(V_s + 35.0) / 6.2))
        alg_13_I_Na = g_Na * h * (V_s - V_Na) * alg_10_m_infinity ** 3.0
        alg_0_I_h = g_h * h * (V_s + 30.0)
        alg_6_I_K_DR = g_K_DR * (V_s - V_K) * n ** 2.0
        alg_15_I_soma = alg_13_I_Na + alg_6_I_K_DR + alg_0_I_h
        d_V_s = torch.zeros_like(y[:, 0]) + ((-alg_15_I_soma - g_c / p * (V_D - V_s)) / C_m)
        alg_11_I_L = g_L * (V_D - V_L)
        alg_14_I_NMDA = g_NMDA * alg_1_f_NMDA * V_D
        alg_16_I_D = alg_14_I_NMDA + alg_7_I_pump - I_pump_ss + alg_11_I_L
        d_V_D = torch.zeros_like(y[:, 0]) + ((-alg_16_I_D + g_c / (1.0 - p) * (V_s - V_D)) / C_m)

        return torch.stack(
            [d_V_s, d_V_D, d_n, d_h, d_Na],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
