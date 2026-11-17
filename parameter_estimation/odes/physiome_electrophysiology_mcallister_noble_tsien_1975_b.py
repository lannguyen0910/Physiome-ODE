import torch
import torch.nn as nn
import math


class ElectrophysiologyMcallisterNobleTsien1975BODE(nn.Module):
    """
    PyTorch translation of electrophysiology/mcallister_noble_tsien_1975_b/model.py.
    Dynamic states: 10; independent constants: 13; derived constants used by rates: 0.
    Original constant slots: 13; algebraic slots: 31.
    """
    source_file = 'electrophysiology/mcallister_noble_tsien_1975_b/model.py'
    source_state_dim = 10
    source_constant_dim = 13
    algebraic_dim = 31
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 10
        self.parameter_dim = 13

        self.parameters = (
            parameters
            if parameters is not None
            else {'C': 10, 'E_Na': 40, 'g_Na': 150, 'g_si': 0.8, 'g_si_2': 0.04, 'E_si': 70, 'E_K': -110, 'E_s': -52, 'E_Cl': -70, 'g_qr': 2.5, 'E_K1': -30, 'g_Nab': 0.105, 'g_Clb': 0.01}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -78.041367, 'm': 0.02566853, 'h': 0.78656359, 'd': 0.00293135, 'f': 0.80873917, 's': 0.75473994, 'x1': 0.02030289, 'x2': 0.0176854, 'q': 3.11285794, 'r': 0.13500116}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['C', 'E_Na', 'g_Na', 'g_si', 'g_si_2', 'E_si', 'E_K', 'E_s', 'E_Cl', 'g_qr', 'E_K1', 'g_Nab', 'g_Clb']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'm', 'h', 'd', 'f', 's', 'x1', 'x2', 'q', 'r']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (10)
        V = y[:, 0]
        m = y[:, 1]
        h = y[:, 2]
        d = y[:, 3]
        f = y[:, 4]
        s = y[:, 5]
        x1 = y[:, 6]
        x2 = y[:, 7]
        q = y[:, 8]
        r = y[:, 9]

        # Independent parameters (source constant order)
        C = theta[:, 0]  # constants[0]
        E_Na = theta[:, 1]  # constants[1]
        g_Na = theta[:, 2]  # constants[2]
        g_si = theta[:, 3]  # constants[3]
        g_si_2 = theta[:, 4]  # constants[4]
        E_si = theta[:, 5]  # constants[5]
        E_K = theta[:, 6]  # constants[6]
        E_s = theta[:, 7]  # constants[7]
        E_Cl = theta[:, 8]  # constants[8]
        g_qr = theta[:, 9]  # constants[9]
        E_K1 = theta[:, 10]  # constants[10]
        g_Nab = theta[:, 11]  # constants[11]
        g_Clb = theta[:, 12]  # constants[12]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_d = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_s = torch.zeros_like(y[:, 0])
        d_x1 = torch.zeros_like(y[:, 0])
        d_x2 = torch.zeros_like(y[:, 0])
        d_q = torch.zeros_like(y[:, 0])
        d_r = torch.zeros_like(y[:, 0])
        alg_0_i_Na = torch.zeros_like(y[:, 0])
        alg_1_alpha_m = torch.zeros_like(y[:, 0])
        alg_2_alpha_h = torch.zeros_like(y[:, 0])
        alg_3_alpha_d = torch.zeros_like(y[:, 0])
        alg_4_alpha_f = torch.zeros_like(y[:, 0])
        alg_5_alpha_s = torch.zeros_like(y[:, 0])
        alg_6_alpha_x1 = torch.zeros_like(y[:, 0])
        alg_7_alpha_x2 = torch.zeros_like(y[:, 0])
        alg_8_alpha_q = torch.zeros_like(y[:, 0])
        alg_9_alpha_r = torch.zeros_like(y[:, 0])
        alg_10_d1 = torch.zeros_like(y[:, 0])
        alg_11_beta_m = torch.zeros_like(y[:, 0])
        alg_12_beta_h = torch.zeros_like(y[:, 0])
        alg_13_beta_d = torch.zeros_like(y[:, 0])
        alg_14_beta_f = torch.zeros_like(y[:, 0])
        alg_15_beta_s = torch.zeros_like(y[:, 0])
        alg_16_beta_x1 = torch.zeros_like(y[:, 0])
        alg_17_beta_x2 = torch.zeros_like(y[:, 0])
        alg_18_beta_q = torch.zeros_like(y[:, 0])
        alg_19_beta_r = torch.zeros_like(y[:, 0])
        alg_20_i_si = torch.zeros_like(y[:, 0])
        alg_21_I_K2 = torch.zeros_like(y[:, 0])
        alg_22_i_K2 = torch.zeros_like(y[:, 0])
        alg_23_I_x1 = torch.zeros_like(y[:, 0])
        alg_24_i_x1 = torch.zeros_like(y[:, 0])
        alg_25_I_x2 = torch.zeros_like(y[:, 0])
        alg_26_i_x2 = torch.zeros_like(y[:, 0])
        alg_27_i_qr = torch.zeros_like(y[:, 0])
        alg_28_i_K1 = torch.zeros_like(y[:, 0])
        alg_29_i_Na_b = torch.zeros_like(y[:, 0])
        alg_30_i_Cl_b = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_alpha_m = 1.0 * (V + 47.0) / (1.0 - torch.exp(-(V + 47.0) / 10.0))
        alg_11_beta_m = 40.0 * torch.exp(-0.056 * (V + 72.0))
        d_m = torch.zeros_like(y[:, 0]) + (alg_1_alpha_m * (1.0 - m) - alg_11_beta_m * m)
        alg_2_alpha_h = 0.0085 * torch.exp(-0.184 * (V + 71.0))
        alg_12_beta_h = 2.5 / (torch.exp(-0.082 * (V + 10.0)) + 1.0)
        d_h = torch.zeros_like(y[:, 0]) + (alg_2_alpha_h * (1.0 - h) - alg_12_beta_h * h)
        alg_3_alpha_d = 0.002 * (V + 40.0) / (1.0 - torch.exp(-0.1 * (V + 40.0)))
        alg_13_beta_d = 0.02 * torch.exp(-0.0888 * (V + 40.0))
        d_d = torch.zeros_like(y[:, 0]) + (alg_3_alpha_d * (1.0 - d) - alg_13_beta_d * d)
        alg_4_alpha_f = 0.000987 * torch.exp(-0.04 * (V + 60.0))
        alg_14_beta_f = 0.02 / (torch.exp(-0.087 * (V + 26.0)) + 1.0)
        d_f = torch.zeros_like(y[:, 0]) + (alg_4_alpha_f * (1.0 - f) - alg_14_beta_f * f)
        alg_5_alpha_s = 0.001 * (V - E_s) / (1.0 - torch.exp(-0.2 * (V - E_s)))
        alg_15_beta_s = 5e-05 * torch.exp(-0.067 * (V - E_s))
        d_s = torch.zeros_like(y[:, 0]) + (alg_5_alpha_s * (1.0 - s) - alg_15_beta_s * s)
        alg_6_alpha_x1 = 0.0005 * torch.exp((V + 50.0) / 12.1) / (1.0 + torch.exp((V + 50.0) / 17.5))
        alg_16_beta_x1 = 0.0013 * torch.exp(-(V + 20.0) / 16.67) / (1.0 + torch.exp(-(V + 20.0) / 25.0))
        d_x1 = torch.zeros_like(y[:, 0]) + (alg_6_alpha_x1 * (1.0 - x1) - alg_16_beta_x1 * x1)
        alg_7_alpha_x2 = 0.000127 * 1.0 / (1.0 + torch.exp(-(V + 19.0) / 5.0))
        alg_17_beta_x2 = 0.0003 * torch.exp(-(V + 20.0) / 16.67) / (1.0 + torch.exp(-(V + 20.0) / 25.0))
        d_x2 = torch.zeros_like(y[:, 0]) + (alg_7_alpha_x2 * (1.0 - x2) - alg_17_beta_x2 * x2)
        alg_8_alpha_q = 0.008 * V / (1.0 - torch.exp(-0.1 * V))
        alg_18_beta_q = 0.08 * torch.exp(-0.0888 * V)
        d_q = torch.zeros_like(y[:, 0]) + (alg_8_alpha_q * (1.0 - q) - alg_18_beta_q * q)
        alg_9_alpha_r = 3.3e-05 * torch.exp(-V / 17.0)
        alg_19_beta_r = 0.033 / (torch.exp(-(V + 30.0) / 8.0) + 1.0)
        d_r = torch.zeros_like(y[:, 0]) + (alg_9_alpha_r * (1.0 - r) - alg_19_beta_r * r)
        alg_0_i_Na = g_Na * m ** 3.0 * h * (V - E_Na)
        alg_10_d1 = 1.0 / (1.0 + torch.exp(-0.15 * (V + 40.0)))
        alg_20_i_si = g_si * d * f * (V - E_si) + g_si_2 * alg_10_d1 * (V - E_si)
        alg_21_I_K2 = 2.8 * (torch.exp((V - E_K) / 25.0) - 1.0) / (torch.exp((V + 60.0) / 12.5) + torch.exp((V + 60.0) / 25.0))
        alg_22_i_K2 = alg_21_I_K2 * s
        alg_23_I_x1 = 1.2 * (torch.exp((V + 95.0) / 25.0) - 1.0) / torch.exp((V + 45.0) / 25.0)
        alg_24_i_x1 = x1 * alg_23_I_x1
        alg_25_I_x2 = 25.0 + 1.0 * 0.385 * V
        alg_26_i_x2 = x2 * alg_25_I_x2
        alg_27_i_qr = g_qr * q * r * (V - E_Cl)
        alg_28_i_K1 = alg_21_I_K2 / 2.8 + 0.2 * (V - E_K1) / (1.0 - torch.exp(-(V - E_K1) / 25.0))
        alg_29_i_Na_b = g_Nab * (V - E_Na)
        alg_30_i_Cl_b = g_Clb * (V - E_Cl)
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_0_i_Na + alg_20_i_si + alg_22_i_K2 + alg_24_i_x1 + alg_26_i_x2 + alg_27_i_qr + alg_28_i_K1 + alg_29_i_Na_b + alg_30_i_Cl_b) / C)

        return torch.stack(
            [d_V, d_m, d_h, d_d, d_f, d_s, d_x1, d_x2, d_q, d_r],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
