import torch
import torch.nn as nn
import math


class ElectrophysiologyBueno2007ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/bueno_2007/model.py.
    Dynamic states: 4; independent constants: 11; derived constants used by rates: 21.
    Original constant slots: 32; algebraic slots: 16.
    """
    source_file = 'electrophysiology/bueno_2007/model.py'
    source_state_dim = 4
    source_constant_dim = 32
    algebraic_dim = 16
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
        self.parameter_dim = 11

        self.parameters = (
            parameters
            if parameters is not None
            else {'epi': 1, 'endo': 0, 'mcell': 0, 'V_0': -83, 'V_fi': 2.7, 'u_m': 0.3, 'u_p': 0.13, 'tau_v_plus': 1.45, 'tau_s1': 2.7342, 'k_s': 2.0994, 'u_s': 0.9087}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'u': 0, 'v': 1, 'w': 1, 's': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['epi', 'endo', 'mcell', 'V_0', 'V_fi', 'u_m', 'u_p', 'tau_v_plus', 'tau_s1', 'k_s', 'u_s']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['u', 'v', 'w', 's']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        u = y[:, 0]
        v = y[:, 1]
        w = y[:, 2]
        s = y[:, 3]

        # Independent parameters (source constant order)
        epi = theta[:, 0]  # constants[0]
        endo = theta[:, 1]  # constants[1]
        mcell = theta[:, 2]  # constants[2]
        V_0 = theta[:, 3]  # constants[3]
        V_fi = theta[:, 4]  # constants[4]
        u_m = theta[:, 5]  # constants[5]
        u_p = theta[:, 6]  # constants[6]
        tau_v_plus = theta[:, 7]  # constants[7]
        tau_s1 = theta[:, 8]  # constants[8]
        k_s = theta[:, 9]  # constants[9]
        u_s = theta[:, 10]  # constants[10]

        # Derived constants from the original initConsts()
        tau_s2 = torch.where(epi == 1.0, 16.0, torch.where(endo == 1.0, 2.0, 4.0))  # constants[11]
        tau_v2_minus = torch.where(epi == 1.0, 1150.0, torch.where(endo == 1.0, 10.0, 1.45))  # constants[12]
        u_q = torch.where(epi == 1.0, 0.006, torch.where(endo == 1.0, 0.024, 0.1))  # constants[13]
        tau_o1 = torch.where(epi == 1.0, 400.0, torch.where(endo == 1.0, 470.0, 410.0))  # constants[14]
        u_r = torch.where(epi == 1.0, 0.006, torch.where(endo == 1.0, 0.006, 0.005))  # constants[15]
        tau_fi = torch.where(epi == 1.0, 0.11, torch.where(endo == 1.0, 0.104, 0.078))  # constants[16]
        u_u = torch.where(epi == 1.0, 1.55, torch.where(endo == 1.0, 1.56, 1.61))  # constants[17]
        tau_so1 = torch.where(epi == 1.0, 30.02, torch.where(endo == 1.0, 40.0, 91.0))  # constants[18]
        tau_so2 = torch.where(epi == 1.0, 0.996, torch.where(endo == 1.0, 1.2, 0.8))  # constants[19]
        k_so = torch.where(epi == 1.0, 2.046, torch.where(endo == 1.0, 2.0, 2.1))  # constants[20]
        u_so = torch.where(epi == 1.0, 0.65, torch.where(endo == 1.0, 0.65, 0.6))  # constants[21]
        tau_si = torch.where(epi == 1.0, 1.8875, torch.where(endo == 1.0, 2.9013, 3.3849))  # constants[22]
        tau_winf = torch.where(epi == 1.0, 0.07, torch.where(endo == 1.0, 0.0273, 0.01))  # constants[23]
        wstar_inf = torch.where(epi == 1.0, 0.94, torch.where(endo == 1.0, 0.78, 0.5))  # constants[24]
        tau_w1_minus = torch.where(epi == 1.0, 60.0, torch.where(endo == 1.0, 6.0, 70.0))  # constants[25]
        tau_w2_minus = torch.where(epi == 1.0, 15.0, torch.where(endo == 1.0, 140.0, 8.0))  # constants[26]
        k_w_minus = torch.where(epi == 1.0, 65.0, torch.where(endo == 1.0, 200.0, 200.0))  # constants[27]
        u_w_minus = torch.where(epi == 1.0, 0.03, torch.where(endo == 1.0, 0.016, 0.016))  # constants[28]
        tau_w_plus = torch.where(epi == 1.0, 200.0, torch.where(endo == 1.0, 280.0, 280.0))  # constants[29]
        tau_v1_minus = torch.where(epi == 1.0, 60.0, torch.where(endo == 1.0, 75.0, 80.0))  # constants[30]
        tau_o2 = torch.where(epi == 1.0, 6.0, torch.where(endo == 1.0, 6.0, 7.0))  # constants[31]

        # Initialize rates and algebraic variables, as in computeRates()
        d_u = torch.zeros_like(y[:, 0])
        d_v = torch.zeros_like(y[:, 0])
        d_w = torch.zeros_like(y[:, 0])
        d_s = torch.zeros_like(y[:, 0])
        alg_1_J_stim = torch.zeros_like(y[:, 0])
        alg_2_m = torch.zeros_like(y[:, 0])
        alg_3_q = torch.zeros_like(y[:, 0])
        alg_4_p = torch.zeros_like(y[:, 0])
        alg_5_r = torch.zeros_like(y[:, 0])
        alg_6_v_inf = torch.zeros_like(y[:, 0])
        alg_7_tau_s = torch.zeros_like(y[:, 0])
        alg_8_J_fi = torch.zeros_like(y[:, 0])
        alg_9_tau_v_minus = torch.zeros_like(y[:, 0])
        alg_10_w_inf = torch.zeros_like(y[:, 0])
        alg_11_tau_o = torch.zeros_like(y[:, 0])
        alg_12_tau_w_minus = torch.zeros_like(y[:, 0])
        alg_13_tau_so = torch.zeros_like(y[:, 0])
        alg_14_J_so = torch.zeros_like(y[:, 0])
        alg_15_J_si = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_4_p = torch.where(u < u_p, 0.0, 1.0)
        alg_7_tau_s = (1.0 - alg_4_p) * tau_s1 + alg_4_p * tau_s2
        d_s = torch.zeros_like(y[:, 0]) + (((1.0 + torch.tanh(k_s * (u - u_s))) / 2.0 - s) / alg_7_tau_s)
        alg_2_m = torch.where(u < u_m, 0.0, 1.0)
        alg_6_v_inf = torch.where(u < u_q, 1.0, 0.0)
        alg_3_q = torch.where(u < u_q, 0.0, 1.0)
        alg_9_tau_v_minus = alg_3_q * tau_v2_minus + (1.0 - alg_3_q) * tau_v1_minus
        d_v = torch.zeros_like(y[:, 0]) + ((1.0 - alg_2_m) * (alg_6_v_inf - v) / alg_9_tau_v_minus - alg_2_m * v / tau_v_plus)
        alg_5_r = torch.where(u < u_r, 0.0, 1.0)
        alg_10_w_inf = (1.0 - alg_5_r) * (1.0 - u * 1.0 / tau_winf) + alg_5_r * wstar_inf
        alg_12_tau_w_minus = tau_w1_minus + (tau_w2_minus - tau_w1_minus) * (1.0 + torch.tanh(k_w_minus * (u - u_w_minus))) / 2.0
        d_w = torch.zeros_like(y[:, 0]) + ((1.0 - alg_5_r) * (alg_10_w_inf - w) / alg_12_tau_w_minus - alg_5_r * w / tau_w_plus)
        alg_8_J_fi = -alg_2_m * v * (u - u_m) * (u_u - u) / tau_fi
        alg_11_tau_o = (1.0 - alg_5_r) * tau_o1 + alg_5_r * tau_o2
        alg_13_tau_so = tau_so1 + (tau_so2 - tau_so1) * (1.0 + torch.tanh(k_so * (u - u_so))) / 2.0
        alg_14_J_so = u * (1.0 - alg_4_p) / alg_11_tau_o + alg_4_p / alg_13_tau_so
        alg_15_J_si = -alg_4_p * w * s / tau_si
        alg_1_J_stim = torch.where((t >= 100.0) & (t <= 101.0), -1.0, 0.0)
        d_u = torch.zeros_like(y[:, 0]) + (-(alg_8_J_fi + alg_14_J_so + alg_15_J_si + alg_1_J_stim))

        return torch.stack(
            [d_u, d_v, d_w, d_s],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
