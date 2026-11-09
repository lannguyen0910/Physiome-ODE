import torch
import torch.nn as nn
import math


class ElectrophysiologyMiftakhovAbdushevaChristensen1999ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/miftakhov_abdusheva_christensen_1999/model.py.
    Dynamic states: 5; independent constants: 14; derived constants used by rates: 0.
    Original constant slots: 14; algebraic slots: 13.
    """
    source_file = 'electrophysiology/miftakhov_abdusheva_christensen_1999/model.py'
    source_state_dim = 5
    source_constant_dim = 14
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
        self.parameter_dim = 14

        self.parameters = (
            parameters
            if parameters is not None
            else {'alpha': 0.12, 'lamda': 12.5, 'Cm': 2.5, 'E_Ca': 80, 'g_Ca_T': 0.51, 'g_Ca_L': 0.004, 'tau_x_Ca': 500, 'E_K': -75, 'g_K': 0.3, 'g_Ca_K': 0.03, 'rho': 125, 'K_c': 0.00425, 'g_Cl': 0.003, 'E_Cl': -40}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -55, 'h': 0.01, 'x_Ca': 0.01, 'n': 0.01, 'Ca': 0.0001}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['alpha', 'lamda', 'Cm', 'E_Ca', 'g_Ca_T', 'g_Ca_L', 'tau_x_Ca', 'E_K', 'g_K', 'g_Ca_K', 'rho', 'K_c', 'g_Cl', 'E_Cl']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'h', 'x_Ca', 'n', 'Ca']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        V = y[:, 0]
        h = y[:, 1]
        x_Ca = y[:, 2]
        n = y[:, 3]
        Ca = y[:, 4]

        # Independent parameters (source constant order)
        alpha = theta[:, 0]  # constants[0]
        lamda = theta[:, 1]  # constants[1]
        Cm = theta[:, 2]  # constants[2]
        E_Ca = theta[:, 3]  # constants[3]
        g_Ca_T = theta[:, 4]  # constants[4]
        g_Ca_L = theta[:, 5]  # constants[5]
        tau_x_Ca = theta[:, 6]  # constants[6]
        E_K = theta[:, 7]  # constants[7]
        g_K = theta[:, 8]  # constants[8]
        g_Ca_K = theta[:, 9]  # constants[9]
        rho = theta[:, 10]  # constants[10]
        K_c = theta[:, 11]  # constants[11]
        g_Cl = theta[:, 12]  # constants[12]
        E_Cl = theta[:, 13]  # constants[13]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_x_Ca = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_Ca = torch.zeros_like(y[:, 0])
        alg_0_V_tilde = torch.zeros_like(y[:, 0])
        alg_1_alpha_m = torch.zeros_like(y[:, 0])
        alg_2_alpha_h = torch.zeros_like(y[:, 0])
        alg_3_alpha_n = torch.zeros_like(y[:, 0])
        alg_4_beta_m = torch.zeros_like(y[:, 0])
        alg_5_beta_h = torch.zeros_like(y[:, 0])
        alg_6_beta_n = torch.zeros_like(y[:, 0])
        alg_7_m = torch.zeros_like(y[:, 0])
        alg_8_i_Ca_T = torch.zeros_like(y[:, 0])
        alg_9_i_Ca_L = torch.zeros_like(y[:, 0])
        alg_10_i_K = torch.zeros_like(y[:, 0])
        alg_11_i_Ca_K = torch.zeros_like(y[:, 0])
        alg_12_i_Cl = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Ca = torch.zeros_like(y[:, 0]) + (rho / alpha * (K_c * x_Ca * (E_Ca - V) - Ca))
        alg_0_V_tilde = (127.0 * V + 8265.0) / 105.0
        d_x_Ca = torch.zeros_like(y[:, 0]) + ((1.0 / (1.0 + torch.exp(0.15 * (-alg_0_V_tilde - 50.0))) - x_Ca) / (alpha * tau_x_Ca))
        alg_2_alpha_h = 0.07 * torch.exp((25.0 - alg_0_V_tilde) / 20.0)
        alg_5_beta_h = 1.0 / (1.0 + torch.exp(5.5 - alg_0_V_tilde * 0.1))
        d_h = torch.zeros_like(y[:, 0]) + ((alg_2_alpha_h * (1.0 - h) - alg_5_beta_h * h) / (alpha * lamda))
        alg_3_alpha_n = 0.01 * (55.0 - alg_0_V_tilde) / (torch.exp((55.0 - alg_0_V_tilde) / 10.0) - 1.0)
        alg_6_beta_n = 0.125 * torch.exp((45.0 - alg_0_V_tilde) / 80.0)
        d_n = torch.zeros_like(y[:, 0]) + ((alg_3_alpha_n * (1.0 - n) - alg_6_beta_n * n) / (alpha * lamda))
        alg_1_alpha_m = 0.1 * (50.0 - alg_0_V_tilde) / (torch.exp(5.0 - alg_0_V_tilde * 0.1) - 1.0)
        alg_4_beta_m = 4.0 * torch.exp((25.0 - alg_0_V_tilde) / 18.0)
        alg_7_m = alg_1_alpha_m / (alg_1_alpha_m + alg_4_beta_m)
        alg_8_i_Ca_T = g_Ca_T * alg_7_m ** 3.0 * h * (V - E_Ca)
        alg_9_i_Ca_L = g_Ca_L * x_Ca * (V - E_Ca)
        alg_11_i_Ca_K = g_Ca_K * Ca * (V - E_K) / (0.5 + Ca)
        alg_10_i_K = g_K * n ** 4.0 * (V - E_K)
        alg_12_i_Cl = g_Cl * (V - E_Cl)
        d_V = torch.zeros_like(y[:, 0]) + (-(1.0 / (Cm * alpha)) * (alg_8_i_Ca_T + alg_9_i_Ca_L + alg_11_i_Ca_K + alg_10_i_K + alg_12_i_Cl))

        return torch.stack(
            [d_V, d_h, d_x_Ca, d_n, d_Ca],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
