import torch
import torch.nn as nn
import math


class ElectrophysiologyHodgkinHuxley1952Variant01ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/hodgkin_huxley_1952_variant01/model.py.
    Dynamic states: 4; independent constants: 5; derived constants used by rates: 3.
    Original constant slots: 8; algebraic slots: 10.
    """
    source_file = 'electrophysiology/hodgkin_huxley_1952_variant01/model.py'
    source_state_dim = 4
    source_constant_dim = 8
    algebraic_dim = 10
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 4
        self.parameter_dim = 5

        self.parameters = (
            parameters
            if parameters is not None
            else {'E_R': 0, 'Cm': 1, 'g_Na': 120, 'g_K': 36, 'g_L': 0.3}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': 0, 'm': 0.05, 'h': 0.6, 'n': 0.325}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['E_R', 'Cm', 'g_Na', 'g_K', 'g_L']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'm', 'h', 'n']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        V = y[:, 0]
        m = y[:, 1]
        h = y[:, 2]
        n = y[:, 3]

        # Independent parameters (source constant order)
        E_R = theta[:, 0]  # constants[0]
        Cm = theta[:, 1]  # constants[1]
        g_Na = theta[:, 2]  # constants[2]
        g_K = theta[:, 3]  # constants[3]
        g_L = theta[:, 4]  # constants[4]

        # Derived constants from the original initConsts()
        E_Na = E_R - 115.0  # constants[5]
        E_K = E_R + 12.0  # constants[6]
        E_L = E_R - 10.613  # constants[7]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        alg_0_i_Stim = torch.zeros_like(y[:, 0])
        alg_1_alpha_m = torch.zeros_like(y[:, 0])
        alg_2_alpha_h = torch.zeros_like(y[:, 0])
        alg_3_alpha_n = torch.zeros_like(y[:, 0])
        alg_4_i_Na = torch.zeros_like(y[:, 0])
        alg_5_beta_m = torch.zeros_like(y[:, 0])
        alg_6_beta_h = torch.zeros_like(y[:, 0])
        alg_7_beta_n = torch.zeros_like(y[:, 0])
        alg_8_i_K = torch.zeros_like(y[:, 0])
        alg_9_i_L = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_alpha_m = 0.1 * (V + 25.0) / (torch.exp((V + 25.0) / 10.0) - 1.0)
        alg_5_beta_m = 4.0 * torch.exp(V / 18.0)
        d_m = torch.zeros_like(y[:, 0]) + (alg_1_alpha_m * (1.0 - m) - alg_5_beta_m * m)
        alg_2_alpha_h = 0.07 * torch.exp(V / 20.0)
        alg_6_beta_h = 1.0 / (torch.exp((V + 30.0) / 10.0) + 1.0)
        d_h = torch.zeros_like(y[:, 0]) + (alg_2_alpha_h * (1.0 - h) - alg_6_beta_h * h)
        alg_3_alpha_n = 0.01 * (V + 10.0) / (torch.exp((V + 10.0) / 10.0) - 1.0)
        alg_7_beta_n = 0.125 * torch.exp(V / 80.0)
        d_n = torch.zeros_like(y[:, 0]) + (alg_3_alpha_n * (1.0 - n) - alg_7_beta_n * n)
        alg_4_i_Na = g_Na * m ** 3.0 * h * (V - E_Na)
        alg_8_i_K = g_K * n ** 4.0 * (V - E_K)
        alg_9_i_L = g_L * (V - E_L)
        alg_0_i_Stim = torch.where((t >= 10.0) & (t <= 10.5), -20.0, 0.0)
        d_V = torch.zeros_like(y[:, 0]) + (-(-alg_0_i_Stim + alg_4_i_Na + alg_8_i_K + alg_9_i_L) / Cm)

        return torch.stack(
            [d_V, d_m, d_h, d_n],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
