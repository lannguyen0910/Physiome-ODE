import torch
import torch.nn as nn
import math


class CalciumDynamicsIzakovKatsnelsonBlyakhmanMarkhasinShkylar1991ODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/izakov_katsnelson_blyakhman_markhasin_shkylar_1991/model.py.
    Dynamic states: 2; independent constants: 24; derived constants used by rates: 3.
    Original constant slots: 30; algebraic slots: 13.
    """
    source_file = 'calcium_dynamics/izakov_katsnelson_blyakhman_markhasin_shkylar_1991/model.py'
    source_state_dim = 2
    source_constant_dim = 30
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

        self.state_dim = 2
        self.parameter_dim = 24

        self.parameters = (
            parameters
            if parameters is not None
            else {'alpha_1': 14.6, 'alpha_2': 14.6, 'beta_1': 1, 'beta_2': 0.0012, 'lambda_var': 30, 'V_max': 0.0043, 'Ca_m': 0.045, 't_d': 170, 'a_c': 0.00024, 'b_c': 0.0005, 'C_1': 0.029, 'C_20': 0.2, 'q_k': 4, 'a': 0.25, 'm_0': 0.87, 'g_1': 0.4, 'g_2': 0.6, 'pi_min': 0.05, 'S_0': 0.77, 'q_1': 0.017, 'q_2': 0.26, 'q_3': 0.03, 'TnC': 1, 'dl_1_dt': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'A_1': 0, 'n_2': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['alpha_1', 'alpha_2', 'beta_1', 'beta_2', 'lambda_var', 'V_max', 'Ca_m', 't_d', 'a_c', 'b_c', 'C_1', 'C_20', 'q_k', 'a', 'm_0', 'g_1', 'g_2', 'pi_min', 'S_0', 'q_1', 'q_2', 'q_3', 'TnC', 'dl_1_dt']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['A_1', 'n_2']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        A_1 = y[:, 0]
        n_2 = y[:, 1]

        # Independent parameters (source constant order)
        alpha_1 = theta[:, 0]  # constants[0]
        alpha_2 = theta[:, 1]  # constants[1]
        beta_1 = theta[:, 2]  # constants[2]
        beta_2 = theta[:, 3]  # constants[3]
        lambda_var = theta[:, 4]  # constants[4]
        V_max = theta[:, 5]  # constants[5]
        Ca_m = theta[:, 6]  # constants[6]
        t_d = theta[:, 7]  # constants[7]
        a_c = theta[:, 8]  # constants[8]
        b_c = theta[:, 9]  # constants[9]
        C_1 = theta[:, 10]  # constants[10]
        C_20 = theta[:, 11]  # constants[11]
        q_k = theta[:, 12]  # constants[12]
        a = theta[:, 13]  # constants[13]
        m_0 = theta[:, 14]  # constants[14]
        g_1 = theta[:, 15]  # constants[15]
        g_2 = theta[:, 16]  # constants[16]
        pi_min = theta[:, 17]  # constants[17]
        S_0 = theta[:, 18]  # constants[18]
        q_1 = theta[:, 19]  # constants[19]
        q_2 = theta[:, 20]  # constants[20]
        q_3 = theta[:, 21]  # constants[21]
        TnC = theta[:, 22]  # constants[22]
        dl_1_dt = theta[:, 23]  # constants[23]

        # Derived constants from the original initConsts()
        V = -dl_1_dt  # constants[25]
        q_V = torch.where(V <= 0.0, q_1 - q_2 * V / V_max, q_3)  # constants[26]
        G_V = 1.0 + 0.6 * V / V_max  # constants[27]

        # Initialize rates and algebraic variables, as in computeRates()
        d_A_1 = torch.zeros_like(y[:, 0])
        d_n_2 = torch.zeros_like(y[:, 0])
        alg_0_Ca = torch.zeros_like(y[:, 0])
        alg_1_phi_A_1 = torch.zeros_like(y[:, 0])
        alg_2_l_1 = torch.zeros_like(y[:, 0])
        alg_7_n_1 = torch.zeros_like(y[:, 0])
        alg_9_n = torch.zeros_like(y[:, 0])
        alg_11_pi_n = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_n_2 = torch.zeros_like(y[:, 0]) + (q_V * (m_0 * G_V - n_2))
        alg_0_Ca = torch.where(t <= t_d, Ca_m * (1.0 - torch.exp(-a_c * t ** 2.0)) ** 2.0, Ca_m * ((1.0 - torch.exp(-a_c * t ** 2.0)) * torch.exp(-b_c * (t - t_d) ** 2.0)) ** 2.0)
        alg_1_phi_A_1 = torch.exp(-q_k * A_1)
        alg_2_l_1 = torch.where((200.0 <= t) & (t <= 201.0), 0.0, 0.0)
        alg_7_n_1 = torch.where(g_1 * alg_2_l_1 + g_2 < 0.0, 0.0, torch.where((0.0 <= g_1 * alg_2_l_1 + g_2) & (g_1 * alg_2_l_1 + g_2 <= 1.0), g_1 * alg_2_l_1 + g_2, 1.0))
        alg_9_n = alg_7_n_1 * n_2
        alg_11_pi_n = torch.where((0.75 <= alg_9_n) & (alg_9_n <= 1.0), pi_min, torch.where((0.25 <= alg_9_n) & (alg_9_n < 0.75), pi_min ** (2.0 * alg_9_n - 0.5), 1.0))
        d_A_1 = torch.zeros_like(y[:, 0]) + (C_1 * alg_0_Ca * (TnC - A_1) - C_20 * alg_11_pi_n * alg_1_phi_A_1 * A_1)

        return torch.stack(
            [d_A_1, d_n_2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
