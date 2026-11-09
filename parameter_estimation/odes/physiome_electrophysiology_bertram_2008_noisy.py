import torch
import torch.nn as nn
import math


class ElectrophysiologyBertram2008NoisyODE(nn.Module):
    """
    PyTorch translation of electrophysiology/bertram_2008_noisy/model.py.
    Dynamic states: 5; independent constants: 30; derived constants used by rates: 2.
    Original constant slots: 32; algebraic slots: 12.
    """
    source_file = 'electrophysiology/bertram_2008_noisy/model.py'
    source_state_dim = 5
    source_constant_dim = 32
    algebraic_dim = 12
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
        self.parameter_dim = 30

        self.parameters = (
            parameters
            if parameters is not None
            else {'Cm': 4525, 'gCa': 280, 'VCa': 100, 'vm': -22, 'sm': 7.5, 'gs1': 22, 'VK': -80, 'autos1': 1, 's1knot': 1, 'tau_s1': 1000, 'vs1': -50, 'ss1': 5, 'gK': 1300, 'tau_n_bar': 8.25, 'vn': -9, 'sn': 10, 'gl': 41, 'Vl': -40, 'alpha_p': 1, 'tau_p': 100, 'p0': 0.2, 'noise': 1, 'w': 1, 'delNoise': 3, 'gs2': 16, 'autos2': 1, 's2knot': 0.47, 'tau_s2': 30000, 'vs2': -40, 'ss2': 15}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -40, 's1': 0.9, 'n': 0, 'p': 0.14, 's2': 0.5}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Cm', 'gCa', 'VCa', 'vm', 'sm', 'gs1', 'VK', 'autos1', 's1knot', 'tau_s1', 'vs1', 'ss1', 'gK', 'tau_n_bar', 'vn', 'sn', 'gl', 'Vl', 'alpha_p', 'tau_p', 'p0', 'noise', 'w', 'delNoise', 'gs2', 'autos2', 's2knot', 'tau_s2', 'vs2', 'ss2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 's1', 'n', 'p', 's2']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        V = y[:, 0]
        s1 = y[:, 1]
        n = y[:, 2]
        p = y[:, 3]
        s2 = y[:, 4]

        # Independent parameters (source constant order)
        Cm = theta[:, 0]  # constants[0]
        gCa = theta[:, 1]  # constants[1]
        VCa = theta[:, 2]  # constants[2]
        vm = theta[:, 3]  # constants[3]
        sm = theta[:, 4]  # constants[4]
        gs1 = theta[:, 5]  # constants[5]
        VK = theta[:, 6]  # constants[6]
        autos1 = theta[:, 7]  # constants[7]
        s1knot = theta[:, 8]  # constants[8]
        tau_s1 = theta[:, 9]  # constants[9]
        vs1 = theta[:, 10]  # constants[10]
        ss1 = theta[:, 11]  # constants[11]
        gK = theta[:, 12]  # constants[12]
        tau_n_bar = theta[:, 13]  # constants[13]
        vn = theta[:, 14]  # constants[14]
        sn = theta[:, 15]  # constants[15]
        gl = theta[:, 16]  # constants[16]
        Vl = theta[:, 17]  # constants[17]
        alpha_p = theta[:, 18]  # constants[18]
        tau_p = theta[:, 19]  # constants[19]
        p0 = theta[:, 20]  # constants[20]
        noise = theta[:, 21]  # constants[21]
        w = theta[:, 22]  # constants[22]
        delNoise = theta[:, 23]  # constants[23]
        gs2 = theta[:, 24]  # constants[24]
        autos2 = theta[:, 25]  # constants[25]
        s2knot = theta[:, 26]  # constants[26]
        tau_s2 = theta[:, 27]  # constants[27]
        vs2 = theta[:, 28]  # constants[28]
        ss2 = theta[:, 29]  # constants[29]

        # Derived constants from the original initConsts()
        beta_p = alpha_p * (1.0 / p0 - 1.0)  # constants[30]
        nstoc = 1000.0 / delNoise ** 2.0  # constants[31]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_s1 = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_p = torch.zeros_like(y[:, 0])
        d_s2 = torch.zeros_like(y[:, 0])
        alg_0_m_infinity = torch.zeros_like(y[:, 0])
        alg_1_s1_infinity = torch.zeros_like(y[:, 0])
        alg_2_n_infinity = torch.zeros_like(y[:, 0])
        alg_3_sigma = torch.zeros_like(y[:, 0])
        alg_4_s2_infinity = torch.zeros_like(y[:, 0])
        alg_5_Ica = torch.zeros_like(y[:, 0])
        alg_6_tau_n = torch.zeros_like(y[:, 0])
        alg_7_Is1 = torch.zeros_like(y[:, 0])
        alg_8_Ik = torch.zeros_like(y[:, 0])
        alg_9_q = torch.zeros_like(y[:, 0])
        alg_10_Il = torch.zeros_like(y[:, 0])
        alg_11_Is2 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_s1_infinity = 1.0 / (1.0 + torch.exp((vs1 - V) / ss1))
        d_s1 = torch.zeros_like(y[:, 0]) + (autos1 * ((alg_1_s1_infinity - s1) / tau_s1) + (1.0 - autos1) * (s1knot - s1))
        alg_3_sigma = ((alpha_p * (1.0 - p) + beta_p * p) / (tau_p * nstoc)) ** (1.0 / 2)
        d_p = torch.zeros_like(y[:, 0]) + ((alpha_p * (1.0 - p) - beta_p * p) / tau_p + noise * w * alg_3_sigma)
        alg_4_s2_infinity = 1.0 / (1.0 + torch.exp((vs2 - V) / ss2))
        d_s2 = torch.zeros_like(y[:, 0]) + (autos2 * ((alg_4_s2_infinity - s2) / tau_s2) + (1.0 - autos2) * (s2knot - s2))
        alg_2_n_infinity = 1.0 / (1.0 + torch.exp((vn - V) / sn))
        alg_6_tau_n = tau_n_bar / (1.0 + torch.exp((V - vn) / sn))
        d_n = torch.zeros_like(y[:, 0]) + ((alg_2_n_infinity - n) / alg_6_tau_n)
        alg_0_m_infinity = 1.0 / (1.0 + torch.exp((vm - V) / sm))
        alg_5_Ica = gCa * alg_0_m_infinity * (V - VCa)
        alg_7_Is1 = gs1 * s1 * (V - VK)
        alg_11_Is2 = gs2 * s2 * (V - VK)
        alg_9_q = (1.0 + p) / 2.0
        alg_10_Il = gl * alg_9_q * (V - Vl)
        alg_8_Ik = gK * n * (V - VK)
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_5_Ica + alg_7_Is1 + alg_11_Is2 + alg_10_Il + alg_8_Ik) / Cm)

        return torch.stack(
            [d_V, d_s1, d_n, d_p, d_s2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
