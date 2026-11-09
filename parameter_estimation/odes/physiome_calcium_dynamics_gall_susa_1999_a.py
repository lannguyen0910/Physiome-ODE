import torch
import torch.nn as nn
import math


class CalciumDynamicsGallSusa1999AODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/gall_susa_1999_a/model.py.
    Dynamic states: 3; independent constants: 28; derived constants used by rates: 0.
    Original constant slots: 28; algebraic slots: 9.
    """
    source_file = 'calcium_dynamics/gall_susa_1999_a/model.py'
    source_state_dim = 3
    source_constant_dim = 28
    algebraic_dim = 9
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 3
        self.parameter_dim = 28

        self.parameters = (
            parameters
            if parameters is not None
            else {'Cm': 5310, 'V_K': -75, 'g_K': 2500, 'lamda': 1.6, 'V_n': -15, 'S_n': 5.6, 'a': 65, 'b': 20, 'c': 6, 'V': -75, 'V_Ca': 110, 'g_Ca': 1400, 'V_m': 4, 'S_m': 14, 'V_h': -10, 'S_h': -10, 'g_K_Ca': 30000, 'K_d': 100, 'g_Na_Ca': 234, 'K_1_2': 1.5, 'RT_F': 26.54, 'nH': 5, 'Ca_o': 2600, 'Na_i': 10, 'Na_o': 140, 'f': 0.001, 'k_Ca': 0.03, 'alpha': 4.5055e-06}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -76, 'n': 0.1, 'Ca_i': 0.52}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Cm', 'V_K', 'g_K', 'lamda', 'V_n', 'S_n', 'a', 'b', 'c', 'V', 'V_Ca', 'g_Ca', 'V_m', 'S_m', 'V_h', 'S_h', 'g_K_Ca', 'K_d', 'g_Na_Ca', 'K_1_2', 'RT_F', 'nH', 'Ca_o', 'Na_i', 'Na_o', 'f', 'k_Ca', 'alpha']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'n', 'Ca_i']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        V = y[:, 0]
        n = y[:, 1]
        Ca_i = y[:, 2]

        # Independent parameters (source constant order)
        Cm = theta[:, 0]  # constants[0]
        V_K = theta[:, 1]  # constants[1]
        g_K = theta[:, 2]  # constants[2]
        lamda = theta[:, 3]  # constants[3]
        V_n = theta[:, 4]  # constants[4]
        S_n = theta[:, 5]  # constants[5]
        a = theta[:, 6]  # constants[6]
        b = theta[:, 7]  # constants[7]
        c = theta[:, 8]  # constants[8]
        const_V = theta[:, 9]  # constants[9]
        V_Ca = theta[:, 10]  # constants[10]
        g_Ca = theta[:, 11]  # constants[11]
        V_m = theta[:, 12]  # constants[12]
        S_m = theta[:, 13]  # constants[13]
        V_h = theta[:, 14]  # constants[14]
        S_h = theta[:, 15]  # constants[15]
        g_K_Ca = theta[:, 16]  # constants[16]
        K_d = theta[:, 17]  # constants[17]
        g_Na_Ca = theta[:, 18]  # constants[18]
        K_1_2 = theta[:, 19]  # constants[19]
        RT_F = theta[:, 20]  # constants[20]
        nH = theta[:, 21]  # constants[21]
        Ca_o = theta[:, 22]  # constants[22]
        Na_i = theta[:, 23]  # constants[23]
        Na_o = theta[:, 24]  # constants[24]
        f = theta[:, 25]  # constants[25]
        k_Ca = theta[:, 26]  # constants[26]
        alpha = theta[:, 27]  # constants[27]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        alg_0_i_K = torch.zeros_like(y[:, 0])
        alg_1_n_infinity = torch.zeros_like(y[:, 0])
        alg_2_m_infinity = torch.zeros_like(y[:, 0])
        alg_3_tau_n = torch.zeros_like(y[:, 0])
        alg_4_h = torch.zeros_like(y[:, 0])
        alg_5_i_Ca = torch.zeros_like(y[:, 0])
        alg_6_i_K_Ca = torch.zeros_like(y[:, 0])
        alg_7_V_Na_Ca = torch.zeros_like(y[:, 0])
        alg_8_i_Na_Ca = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_n_infinity = 1.0 / (1.0 + torch.exp((V_n - V) / S_n))
        alg_3_tau_n = c / (torch.exp((V - const_V) / a) + torch.exp((const_V - V) / b))
        d_n = torch.zeros_like(y[:, 0]) + (lamda * ((alg_1_n_infinity - n) / alg_3_tau_n))
        alg_2_m_infinity = 1.0 / (1.0 + torch.exp((V_m - V) / S_m))
        alg_4_h = 1.0 / (1.0 + torch.exp((V_h - V) / S_h))
        alg_5_i_Ca = g_Ca * alg_2_m_infinity * alg_4_h * (V - V_Ca)
        alg_0_i_K = g_K * n * (V - V_K)
        alg_6_i_K_Ca = g_K_Ca * (Ca_i / (K_d + Ca_i)) * (V - V_K)
        alg_7_V_Na_Ca = RT_F * (3.0 * torch.log(Na_o / Na_i - torch.log(Ca_o / Ca_i)))
        alg_8_i_Na_Ca = g_Na_Ca * (Ca_i ** nH / (K_1_2 ** nH + Ca_i ** nH)) * (V - alg_7_V_Na_Ca)
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_0_i_K + alg_5_i_Ca + alg_6_i_K_Ca + alg_8_i_Na_Ca) / Cm)
        d_Ca_i = torch.zeros_like(y[:, 0]) + (f * (-alpha * (alg_5_i_Ca - 2.0 * alg_8_i_Na_Ca) - k_Ca * Ca_i))

        return torch.stack(
            [d_V, d_n, d_Ca_i],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
