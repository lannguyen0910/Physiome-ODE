import torch
import torch.nn as nn
import math


class NeurobiologyButeraRinzelSmith1999Model2ODE(nn.Module):
    """
    PyTorch translation of neurobiology/butera_rinzel_smith_1999_model2/model.py.
    Dynamic states: 4; independent constants: 25; derived constants used by rates: 0.
    Original constant slots: 25; algebraic slots: 14.
    """
    source_file = 'neurobiology/butera_rinzel_smith_1999_model2/model.py'
    source_state_dim = 4
    source_constant_dim = 25
    algebraic_dim = 14
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
        self.parameter_dim = 25

        self.parameters = (
            parameters
            if parameters is not None
            else {'C': 21, 'i_app': 0, 'E_Na': 50, 'g_Na': 28, 'theta_m': -34, 'sigma_m': -5, 'tau_n_max': 10, 'theta_n': -29, 'sigma_n': -4, 'E_K': -85, 'g_K': 11.2, 'tau_n_max_2': 10, 'theta_n_2': -29, 'sigma_n_2': -4, 'g_KS': 5.6, 'tau_k_max': 10000, 'theta_k': -38, 'sigma_k': -6, 'g_NaP': 2.8, 'theta_m_2': -40, 'sigma_m_2': -6, 'g_L': 2.8, 'E_L': -50, 'g_tonic_e': 0, 'E_syn_e': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -55, 'n': 0.01, 'n_2': 0.01, 'k': 0.22}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['C', 'i_app', 'E_Na', 'g_Na', 'theta_m', 'sigma_m', 'tau_n_max', 'theta_n', 'sigma_n', 'E_K', 'g_K', 'tau_n_max_2', 'theta_n_2', 'sigma_n_2', 'g_KS', 'tau_k_max', 'theta_k', 'sigma_k', 'g_NaP', 'theta_m_2', 'sigma_m_2', 'g_L', 'E_L', 'g_tonic_e', 'E_syn_e']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'n', 'n_2', 'k']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        V = y[:, 0]
        n = y[:, 1]
        n_2 = y[:, 2]
        k = y[:, 3]

        # Independent parameters (source constant order)
        C = theta[:, 0]  # constants[0]
        i_app = theta[:, 1]  # constants[1]
        E_Na = theta[:, 2]  # constants[2]
        g_Na = theta[:, 3]  # constants[3]
        theta_m = theta[:, 4]  # constants[4]
        sigma_m = theta[:, 5]  # constants[5]
        tau_n_max = theta[:, 6]  # constants[6]
        theta_n = theta[:, 7]  # constants[7]
        sigma_n = theta[:, 8]  # constants[8]
        E_K = theta[:, 9]  # constants[9]
        g_K = theta[:, 10]  # constants[10]
        tau_n_max_2 = theta[:, 11]  # constants[11]
        theta_n_2 = theta[:, 12]  # constants[12]
        sigma_n_2 = theta[:, 13]  # constants[13]
        g_KS = theta[:, 14]  # constants[14]
        tau_k_max = theta[:, 15]  # constants[15]
        theta_k = theta[:, 16]  # constants[16]
        sigma_k = theta[:, 17]  # constants[17]
        g_NaP = theta[:, 18]  # constants[18]
        theta_m_2 = theta[:, 19]  # constants[19]
        sigma_m_2 = theta[:, 20]  # constants[20]
        g_L = theta[:, 21]  # constants[21]
        E_L = theta[:, 22]  # constants[22]
        g_tonic_e = theta[:, 23]  # constants[23]
        E_syn_e = theta[:, 24]  # constants[24]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_n_2 = torch.zeros_like(y[:, 0])
        d_k = torch.zeros_like(y[:, 0])
        alg_0_m_infinity = torch.zeros_like(y[:, 0])
        alg_1_n_infinity = torch.zeros_like(y[:, 0])
        alg_2_n_infinity_2 = torch.zeros_like(y[:, 0])
        alg_3_k_infinity = torch.zeros_like(y[:, 0])
        alg_4_i_Na = torch.zeros_like(y[:, 0])
        alg_5_tau_n = torch.zeros_like(y[:, 0])
        alg_6_tau_n_2 = torch.zeros_like(y[:, 0])
        alg_7_tau_k = torch.zeros_like(y[:, 0])
        alg_8_i_K = torch.zeros_like(y[:, 0])
        alg_9_i_KS = torch.zeros_like(y[:, 0])
        alg_10_m_infinity_2 = torch.zeros_like(y[:, 0])
        alg_11_i_NaP = torch.zeros_like(y[:, 0])
        alg_12_i_L = torch.zeros_like(y[:, 0])
        alg_13_i_tonic_e = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_n_infinity = 1.0 / (1.0 + torch.exp((V - theta_n) / sigma_n))
        alg_5_tau_n = tau_n_max / torch.cosh((V - theta_n) / (2.0 * sigma_n))
        d_n = torch.zeros_like(y[:, 0]) + ((alg_1_n_infinity - n) / alg_5_tau_n)
        alg_2_n_infinity_2 = 1.0 / (1.0 + torch.exp((V - theta_n_2) / sigma_n_2))
        alg_6_tau_n_2 = tau_n_max_2 / torch.cosh((V - theta_n_2) / (2.0 * sigma_n_2))
        d_n_2 = torch.zeros_like(y[:, 0]) + ((alg_2_n_infinity_2 - n_2) / alg_6_tau_n_2)
        alg_3_k_infinity = 1.0 / (1.0 + torch.exp((V - theta_k) / sigma_k))
        alg_7_tau_k = tau_k_max / torch.cosh((V - theta_k) / (2.0 * sigma_k))
        d_k = torch.zeros_like(y[:, 0]) + ((alg_3_k_infinity - k) / alg_7_tau_k)
        alg_10_m_infinity_2 = 1.0 / (1.0 + torch.exp((V - theta_m_2) / sigma_m_2))
        alg_11_i_NaP = g_NaP * alg_10_m_infinity_2 * (V - E_Na)
        alg_0_m_infinity = 1.0 / (1.0 + torch.exp((V - theta_m) / sigma_m))
        alg_4_i_Na = g_Na * alg_0_m_infinity ** 3.0 * (1.0 - n) * (V - E_Na)
        alg_8_i_K = g_K * n_2 ** 4.0 * (V - E_K)
        alg_9_i_KS = g_KS * k * (V - E_K)
        alg_12_i_L = g_L * (V - E_L)
        alg_13_i_tonic_e = g_tonic_e * (V - E_syn_e)
        d_V = torch.zeros_like(y[:, 0]) + ((-(alg_11_i_NaP + alg_9_i_KS + alg_4_i_Na + alg_8_i_K + alg_12_i_L + alg_13_i_tonic_e) + i_app) / C)

        return torch.stack(
            [d_V, d_n, d_n_2, d_k],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
