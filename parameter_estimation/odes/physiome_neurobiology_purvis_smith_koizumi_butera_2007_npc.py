import torch
import torch.nn as nn
import math


class NeurobiologyPurvisSmithKoizumiButera2007NpcODE(nn.Module):
    """
    PyTorch translation of neurobiology/purvis_smith_koizumi_butera_2007_npc/model.py.
    Dynamic states: 3; independent constants: 21; derived constants used by rates: 0.
    Original constant slots: 21; algebraic slots: 11.
    """
    source_file = 'neurobiology/purvis_smith_koizumi_butera_2007_npc/model.py'
    source_state_dim = 3
    source_constant_dim = 21
    algebraic_dim = 11
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
        self.parameter_dim = 21

        self.parameters = (
            parameters
            if parameters is not None
            else {'C': 0.021, 'i_app': 0, 'E_Na': 50, 'g_Na': 28, 'theta_m': -34, 'omega_m': -5, 'g_K': 11.2, 'E_K': -85, 'tau_n_max': 10, 'theta_n': -29, 'omega_n': -4, 'g_NaP': 1.5, 'theta_m_2': -45.1, 'omega_m_2': -6, 'tau_h_max': 10000, 'theta_h': -53, 'omega_h': 6, 'g_leak': 2.2, 'E_leak': -57.5, 'g_tonic_e': 0.2, 'E_syn_e': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -50, 'n': 0.01, 'h': 0.92}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['C', 'i_app', 'E_Na', 'g_Na', 'theta_m', 'omega_m', 'g_K', 'E_K', 'tau_n_max', 'theta_n', 'omega_n', 'g_NaP', 'theta_m_2', 'omega_m_2', 'tau_h_max', 'theta_h', 'omega_h', 'g_leak', 'E_leak', 'g_tonic_e', 'E_syn_e']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'n', 'h']
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
        h = y[:, 2]

        # Independent parameters (source constant order)
        C = theta[:, 0]  # constants[0]
        i_app = theta[:, 1]  # constants[1]
        E_Na = theta[:, 2]  # constants[2]
        g_Na = theta[:, 3]  # constants[3]
        theta_m = theta[:, 4]  # constants[4]
        omega_m = theta[:, 5]  # constants[5]
        g_K = theta[:, 6]  # constants[6]
        E_K = theta[:, 7]  # constants[7]
        tau_n_max = theta[:, 8]  # constants[8]
        theta_n = theta[:, 9]  # constants[9]
        omega_n = theta[:, 10]  # constants[10]
        g_NaP = theta[:, 11]  # constants[11]
        theta_m_2 = theta[:, 12]  # constants[12]
        omega_m_2 = theta[:, 13]  # constants[13]
        tau_h_max = theta[:, 14]  # constants[14]
        theta_h = theta[:, 15]  # constants[15]
        omega_h = theta[:, 16]  # constants[16]
        g_leak = theta[:, 17]  # constants[17]
        E_leak = theta[:, 18]  # constants[18]
        g_tonic_e = theta[:, 19]  # constants[19]
        E_syn_e = theta[:, 20]  # constants[20]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        alg_0_m_infinity = torch.zeros_like(y[:, 0])
        alg_1_n_infinity = torch.zeros_like(y[:, 0])
        alg_2_h_infinity = torch.zeros_like(y[:, 0])
        alg_3_i_Na = torch.zeros_like(y[:, 0])
        alg_4_tau_n = torch.zeros_like(y[:, 0])
        alg_5_tau_h = torch.zeros_like(y[:, 0])
        alg_6_i_K = torch.zeros_like(y[:, 0])
        alg_7_m_infinity_2 = torch.zeros_like(y[:, 0])
        alg_8_i_NaP = torch.zeros_like(y[:, 0])
        alg_9_i_leak = torch.zeros_like(y[:, 0])
        alg_10_i_tonic_e = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_n_infinity = 1.0 / (1.0 + torch.exp((V - theta_n) / omega_n))
        alg_4_tau_n = tau_n_max / torch.cosh((V - theta_n) / (2.0 * omega_n))
        d_n = torch.zeros_like(y[:, 0]) + ((alg_1_n_infinity - n) / alg_4_tau_n)
        alg_2_h_infinity = 1.0 / (1.0 + torch.exp((V - theta_h) / omega_h))
        alg_5_tau_h = tau_h_max / torch.cosh((V - theta_h) / (2.0 * omega_h))
        d_h = torch.zeros_like(y[:, 0]) + ((alg_2_h_infinity - h) / alg_5_tau_h)
        alg_10_i_tonic_e = g_tonic_e * (1.0 / 1000.0) * (V - E_syn_e)
        alg_0_m_infinity = 1.0 / (1.0 + torch.exp((V - theta_m) / omega_m))
        alg_3_i_Na = g_Na * alg_0_m_infinity ** 3.0 * (1.0 - n) * (1.0 / 1000.0) * (V - E_Na)
        alg_7_m_infinity_2 = 1.0 / (1.0 + torch.exp((V - theta_m_2) / omega_m_2))
        alg_8_i_NaP = g_NaP * alg_7_m_infinity_2 * h * (1.0 / 1000.0) * (V - E_Na)
        alg_6_i_K = g_K * n ** 4.0 * (1.0 / 1000.0) * (V - E_K)
        alg_9_i_leak = g_leak * (1.0 / 1000.0) * (V - E_leak)
        d_V = torch.zeros_like(y[:, 0]) + ((-(alg_3_i_Na + alg_8_i_NaP + alg_6_i_K + alg_9_i_leak) + alg_10_i_tonic_e + i_app) / C)

        return torch.stack(
            [d_V, d_n, d_h],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
