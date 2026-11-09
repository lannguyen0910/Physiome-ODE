import torch
import torch.nn as nn
import math


class ElectrophysiologyChangFujitaB1999ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/chang_fujita_b_1999/model.py.
    Dynamic states: 9; independent constants: 27; derived constants used by rates: 0.
    Original constant slots: 27; algebraic slots: 30.
    """
    source_file = 'electrophysiology/chang_fujita_b_1999/model.py'
    source_state_dim = 9
    source_constant_dim = 27
    algebraic_dim = 30
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 9
        self.parameter_dim = 27

        self.parameters = (
            parameters
            if parameters is not None
            else {'C_m_Imp': 0.1033, 'C_c_Imp': 0.1124, 'psi_m': -28, 'psi_c': -86.4, 'RT': 2.579, 'F': 96.48, 'C_s_Imp': 0.04525, 'psi_s': 0, 'P_mc_Na': 3.27e-06, 'J_mc_NaCl_max': 3.21e-05, 'K_mc_Na_NaCl': 0.0511, 'K_mc_Cl_NaCl': 0.0192, 'J_mc_KCl_max': 6.31e-08, 'K_mc_K_KCl': 0.053, 'K_mc_Cl_KCl': 0.0213, 'P_mc_K': 4.9e-07, 'P_mc_Cl': 1.43e-06, 'J_a_max': 2.69e-06, 'K_Na_ATPase': 0.012, 'P_sc_K': 0.000474, 'P_sc_Cl': 9.16e-05, 'P_ms_Na': 4.8e-06, 'P_ms_K': 4.8e-06, 'P_ms_Cl': 2.4e-06, 'L_mc_v': 5.22e-09, 'L_ms_v': 0, 'L_sc_v': 5.22e-07}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'C_m_Na': 0.05, 'C_m_K': 0.002, 'C_m_Cl': 0.03, 'C_c_Na': 0.0164, 'C_c_K': 0.1637, 'C_c_Cl': 0.0203, 'C_s_Na': 0.1438, 'C_s_K': 0.00425, 'C_s_Cl': 0.112}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['C_m_Imp', 'C_c_Imp', 'psi_m', 'psi_c', 'RT', 'F', 'C_s_Imp', 'psi_s', 'P_mc_Na', 'J_mc_NaCl_max', 'K_mc_Na_NaCl', 'K_mc_Cl_NaCl', 'J_mc_KCl_max', 'K_mc_K_KCl', 'K_mc_Cl_KCl', 'P_mc_K', 'P_mc_Cl', 'J_a_max', 'K_Na_ATPase', 'P_sc_K', 'P_sc_Cl', 'P_ms_Na', 'P_ms_K', 'P_ms_Cl', 'L_mc_v', 'L_ms_v', 'L_sc_v']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['C_m_Na', 'C_m_K', 'C_m_Cl', 'C_c_Na', 'C_c_K', 'C_c_Cl', 'C_s_Na', 'C_s_K', 'C_s_Cl']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (9)
        C_m_Na = y[:, 0]
        C_m_K = y[:, 1]
        C_m_Cl = y[:, 2]
        C_c_Na = y[:, 3]
        C_c_K = y[:, 4]
        C_c_Cl = y[:, 5]
        C_s_Na = y[:, 6]
        C_s_K = y[:, 7]
        C_s_Cl = y[:, 8]

        # Independent parameters (source constant order)
        C_m_Imp = theta[:, 0]  # constants[0]
        C_c_Imp = theta[:, 1]  # constants[1]
        psi_m = theta[:, 2]  # constants[2]
        psi_c = theta[:, 3]  # constants[3]
        RT = theta[:, 4]  # constants[4]
        F = theta[:, 5]  # constants[5]
        C_s_Imp = theta[:, 6]  # constants[6]
        psi_s = theta[:, 7]  # constants[7]
        P_mc_Na = theta[:, 8]  # constants[8]
        J_mc_NaCl_max = theta[:, 9]  # constants[9]
        K_mc_Na_NaCl = theta[:, 10]  # constants[10]
        K_mc_Cl_NaCl = theta[:, 11]  # constants[11]
        J_mc_KCl_max = theta[:, 12]  # constants[12]
        K_mc_K_KCl = theta[:, 13]  # constants[13]
        K_mc_Cl_KCl = theta[:, 14]  # constants[14]
        P_mc_K = theta[:, 15]  # constants[15]
        P_mc_Cl = theta[:, 16]  # constants[16]
        J_a_max = theta[:, 17]  # constants[17]
        K_Na_ATPase = theta[:, 18]  # constants[18]
        P_sc_K = theta[:, 19]  # constants[19]
        P_sc_Cl = theta[:, 20]  # constants[20]
        P_ms_Na = theta[:, 21]  # constants[21]
        P_ms_K = theta[:, 22]  # constants[22]
        P_ms_Cl = theta[:, 23]  # constants[23]
        L_mc_v = theta[:, 24]  # constants[24]
        L_ms_v = theta[:, 25]  # constants[25]
        L_sc_v = theta[:, 26]  # constants[26]

        # Initialize rates and algebraic variables, as in computeRates()
        d_C_m_Na = torch.zeros_like(y[:, 0])
        d_C_m_K = torch.zeros_like(y[:, 0])
        d_C_m_Cl = torch.zeros_like(y[:, 0])
        d_C_c_Na = torch.zeros_like(y[:, 0])
        d_C_c_K = torch.zeros_like(y[:, 0])
        d_C_c_Cl = torch.zeros_like(y[:, 0])
        d_C_s_Na = torch.zeros_like(y[:, 0])
        d_C_s_K = torch.zeros_like(y[:, 0])
        d_C_s_Cl = torch.zeros_like(y[:, 0])
        alg_0_G_mc_Na = torch.zeros_like(y[:, 0])
        alg_2_J_mc_NaCl = torch.zeros_like(y[:, 0])
        alg_4_J_mc_Na = torch.zeros_like(y[:, 0])
        alg_6_G_mc_K = torch.zeros_like(y[:, 0])
        alg_9_J_mc_KCl = torch.zeros_like(y[:, 0])
        alg_11_J_mc_K = torch.zeros_like(y[:, 0])
        alg_12_G_mc_Cl = torch.zeros_like(y[:, 0])
        alg_14_J_a = torch.zeros_like(y[:, 0])
        alg_15_J_mc_Cl = torch.zeros_like(y[:, 0])
        alg_16_J_sc_Na = torch.zeros_like(y[:, 0])
        alg_17_G_sc_K = torch.zeros_like(y[:, 0])
        alg_18_G_sc_Cl = torch.zeros_like(y[:, 0])
        alg_19_G_ms_Na = torch.zeros_like(y[:, 0])
        alg_20_J_sc_K = torch.zeros_like(y[:, 0])
        alg_21_J_sc_Cl = torch.zeros_like(y[:, 0])
        alg_22_J_ms_Na = torch.zeros_like(y[:, 0])
        alg_23_G_ms_K = torch.zeros_like(y[:, 0])
        alg_24_G_ms_Cl = torch.zeros_like(y[:, 0])
        alg_25_J_ms_K = torch.zeros_like(y[:, 0])
        alg_26_J_ms_Cl = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_2_J_mc_NaCl = J_mc_NaCl_max * ((C_m_Na / K_mc_Na_NaCl * (C_m_Cl / K_mc_Cl_NaCl) - C_c_Na / K_mc_Na_NaCl * (C_c_Cl / K_mc_Cl_NaCl)) / ((1.0 + C_m_Na / K_mc_Na_NaCl * (C_m_Cl / K_mc_Cl_NaCl)) * (1.0 + C_c_Na / K_mc_Na_NaCl) * (1.0 + C_c_Cl / K_mc_Cl_NaCl) + (1.0 + C_c_Na / K_mc_Na_NaCl * (C_c_Cl / K_mc_Cl_NaCl)) * (1.0 + C_m_Na / K_mc_Na_NaCl) * (1.0 + C_m_Cl / K_mc_Cl_NaCl)))
        alg_0_G_mc_Na = P_mc_Na * (F * (psi_m - psi_c) / RT) * ((C_m_Na - C_c_Na * torch.exp(-(F / RT) * (psi_m - psi_c))) / (1.0 - torch.exp(-(F / RT) * (psi_m - psi_c))))
        alg_4_J_mc_Na = alg_2_J_mc_NaCl + alg_0_G_mc_Na
        alg_14_J_a = J_a_max * (1.0 / (1.0 + (K_Na_ATPase / C_c_Na) ** 3.0))
        alg_16_J_sc_Na = -3.0 * alg_14_J_a
        d_C_c_Na = torch.zeros_like(y[:, 0]) + (alg_4_J_mc_Na + alg_16_J_sc_Na)
        alg_9_J_mc_KCl = J_mc_KCl_max * ((C_m_K / K_mc_K_KCl * (C_m_Cl / K_mc_Cl_KCl) - C_c_K / K_mc_K_KCl * (C_c_Cl / K_mc_Cl_KCl)) / ((1.0 + C_m_K / K_mc_K_KCl * (C_m_Cl / K_mc_Cl_KCl)) * (1.0 + C_c_K / K_mc_K_KCl) * (1.0 + C_c_Cl / K_mc_Cl_KCl) + (1.0 + C_c_K / K_mc_K_KCl * (C_c_Cl / K_mc_Cl_KCl)) * (1.0 + C_m_K / K_mc_K_KCl) * (1.0 + C_m_Cl / K_mc_Cl_KCl)))
        alg_6_G_mc_K = P_mc_K * (F * (psi_m - psi_c) / RT) * ((C_m_K - C_c_K * torch.exp(-(F / RT) * (psi_m - psi_c))) / (1.0 - torch.exp(-(F / RT) * (psi_m - psi_c))))
        alg_11_J_mc_K = alg_9_J_mc_KCl + alg_6_G_mc_K
        alg_17_G_sc_K = P_sc_K * (F * (psi_s - psi_c) / RT) * ((C_s_K - C_c_K * torch.exp(-(F / RT) * (psi_s - psi_c))) / (1.0 - torch.exp(-(F / RT) * (psi_s - psi_c))))
        alg_20_J_sc_K = 2.0 * alg_14_J_a + alg_17_G_sc_K
        d_C_c_K = torch.zeros_like(y[:, 0]) + (alg_11_J_mc_K + alg_20_J_sc_K)
        alg_12_G_mc_Cl = P_mc_Cl * (-1.0 * F * (psi_m - psi_c) / RT) * ((C_m_Cl - C_c_Cl * torch.exp(-(-1.0 * F / RT) * (psi_m - psi_c))) / (1.0 - torch.exp(-(-1.0 * F / RT) * (psi_m - psi_c))))
        alg_15_J_mc_Cl = alg_2_J_mc_NaCl + alg_9_J_mc_KCl + alg_12_G_mc_Cl
        alg_18_G_sc_Cl = P_sc_Cl * (-1.0 * F * (psi_s - psi_c) / RT) * ((C_s_Cl - C_c_Cl * torch.exp(-(-1.0 * F / RT) * (psi_s - psi_c))) / (1.0 - torch.exp(-(-1.0 * F / RT) * (psi_s - psi_c))))
        alg_21_J_sc_Cl = alg_18_G_sc_Cl
        d_C_c_Cl = torch.zeros_like(y[:, 0]) + (alg_15_J_mc_Cl + alg_21_J_sc_Cl)
        alg_19_G_ms_Na = P_ms_Na * (F * (psi_m - psi_s) / RT) * ((C_m_Na - C_s_Na * torch.exp(-(F / RT) * (psi_m - psi_s))) / (1.0 - torch.exp(-(F / RT) * (psi_m - psi_s))))
        alg_22_J_ms_Na = alg_19_G_ms_Na
        d_C_m_Na = torch.zeros_like(y[:, 0]) + (-(alg_4_J_mc_Na + alg_22_J_ms_Na))
        d_C_s_Na = torch.zeros_like(y[:, 0]) + (alg_22_J_ms_Na - alg_16_J_sc_Na)
        alg_23_G_ms_K = P_ms_K * (F * (psi_m - psi_s) / RT) * ((C_m_K - C_s_K * torch.exp(-(F / RT) * (psi_m - psi_s))) / (1.0 - torch.exp(-(F / RT) * (psi_m - psi_s))))
        alg_25_J_ms_K = alg_23_G_ms_K
        d_C_m_K = torch.zeros_like(y[:, 0]) + (-(alg_11_J_mc_K + alg_25_J_ms_K))
        d_C_s_K = torch.zeros_like(y[:, 0]) + (alg_25_J_ms_K - alg_20_J_sc_K)
        alg_24_G_ms_Cl = P_ms_Cl * (-1.0 * F * (psi_m - psi_s) / RT) * ((C_m_Cl - C_s_Cl * torch.exp(-(-1.0 * F / RT) * (psi_m - psi_s))) / (1.0 - torch.exp(-(-1.0 * F / RT) * (psi_m - psi_s))))
        alg_26_J_ms_Cl = alg_24_G_ms_Cl
        d_C_m_Cl = torch.zeros_like(y[:, 0]) + (-(alg_15_J_mc_Cl + alg_26_J_ms_Cl))
        d_C_s_Cl = torch.zeros_like(y[:, 0]) + (alg_26_J_ms_Cl - alg_21_J_sc_Cl)

        return torch.stack(
            [d_C_m_Na, d_C_m_K, d_C_m_Cl, d_C_c_Na, d_C_c_K, d_C_c_Cl, d_C_s_Na, d_C_s_K, d_C_s_Cl],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
