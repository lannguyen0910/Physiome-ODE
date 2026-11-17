import torch
import torch.nn as nn
import math


class MetabolismNazaret2009ScaledODE(nn.Module):
    """
    PyTorch translation of metabolism/nazaret_2009_scaled/model.py.
    Dynamic states: 8; independent constants: 31; derived constants used by rates: 24.
    Original constant slots: 55; algebraic slots: 13.
    """
    source_file = 'metabolism/nazaret_2009_scaled/model.py'
    source_state_dim = 8
    source_constant_dim = 55
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

        self.state_dim = 8
        self.parameter_dim = 31

        self.parameters = (
            parameters
            if parameters is not None
            else {'At': 4.16, 'Nt': 1.07, 'Pyr_bar': 0.161, 'Cit_bar': 0.46, 'AcCoA_bar': 0.105, 'KG_bar': 0.146, 'OAA_bar': 0.004, 'k1': 38, 'k2': 152, 'k3': 57142, 'k4': 53, 'k5': 40, 'k6': 82361, 'k7': 0.0032, 'k8': 3.6, 'kresp': 2.5, 'kATP': 131.9, 'kANT': 0.1, 'kleak': 0.426, 'Keq': 0.3975, 'K': 2, 'alpha': 0.1, 'b': 0.004, 'delta_psi_m': 150, 'R': 8.314, 'T': 298, 'F': 96485, 'C': 6.75e-06, 'Kapp': 4.4e-06, 'Pi': 2.44, 'v1': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'p': 0.154, 'a': 0.063, 'c': 0.44, 'k': 0.225, 'o': 0.005, 'n': 0.856, 'en': 3.536, 's': 150}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['At', 'Nt', 'Pyr_bar', 'Cit_bar', 'AcCoA_bar', 'KG_bar', 'OAA_bar', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'kresp', 'kATP', 'kANT', 'kleak', 'Keq', 'K', 'alpha', 'b', 'delta_psi_m', 'R', 'T', 'F', 'C', 'Kapp', 'Pi', 'v1']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['p', 'a', 'c', 'k', 'o', 'n', 'en', 's']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (8)
        p = y[:, 0]
        a = y[:, 1]
        c = y[:, 2]
        k = y[:, 3]
        o = y[:, 4]
        n = y[:, 5]
        en = y[:, 6]
        s = y[:, 7]

        # Independent parameters (source constant order)
        At = theta[:, 0]  # constants[0]
        Nt = theta[:, 1]  # constants[1]
        Pyr_bar = theta[:, 2]  # constants[2]
        Cit_bar = theta[:, 3]  # constants[3]
        AcCoA_bar = theta[:, 4]  # constants[4]
        KG_bar = theta[:, 5]  # constants[5]
        OAA_bar = theta[:, 6]  # constants[6]
        k1 = theta[:, 7]  # constants[7]
        k2 = theta[:, 8]  # constants[8]
        k3 = theta[:, 9]  # constants[9]
        k4 = theta[:, 10]  # constants[10]
        k5 = theta[:, 11]  # constants[11]
        k6 = theta[:, 12]  # constants[12]
        k7 = theta[:, 13]  # constants[13]
        k8 = theta[:, 14]  # constants[14]
        kresp = theta[:, 15]  # constants[15]
        kATP = theta[:, 16]  # constants[16]
        kANT = theta[:, 17]  # constants[17]
        kleak = theta[:, 18]  # constants[18]
        Keq = theta[:, 19]  # constants[19]
        K = theta[:, 20]  # constants[20]
        alpha = theta[:, 21]  # constants[21]
        b = theta[:, 22]  # constants[22]
        delta_psi_m = theta[:, 23]  # constants[23]
        R = theta[:, 24]  # constants[24]
        T = theta[:, 25]  # constants[25]
        F = theta[:, 26]  # constants[26]
        C = theta[:, 27]  # constants[27]
        Kapp = theta[:, 28]  # constants[28]
        Pi = theta[:, 29]  # constants[29]
        v1 = theta[:, 30]  # constants[30]

        # Derived constants from the original initConsts()
        beta2 = k2 / k1 * Nt * Pyr_bar  # constants[31]
        beta3 = k3 / k1 * OAA_bar * AcCoA_bar  # constants[32]
        beta4 = k4 / k1 * Nt * Cit_bar  # constants[33]
        beta5 = k5 / k1 * Nt * At * KG_bar  # constants[34]
        delta_6 = KG_bar / (OAA_bar * Keq)  # constants[35]
        beta6 = k6 / k1 * OAA_bar  # constants[36]
        beta7 = k7 / k1 * At * Pyr_bar  # constants[37]
        beta8 = k8 / k1 * OAA_bar  # constants[38]
        beta_ANT = kANT / k1 * At  # constants[39]
        beta_leak = kleak / k1 * delta_psi_m  # constants[40]
        beta_resp = kresp / k1  # constants[41]
        delta_r1 = K / Nt  # constants[42]
        delta_r2 = alpha * delta_psi_m  # constants[43]
        beta_ATP = kATP / k1  # constants[44]
        delta_atp = b * At  # constants[45]
        Kapp_dash = Kapp * Pi  # constants[46]
        delta_crit = 3.0 * (1.2 * F * delta_psi_m / (R * T))  # constants[47]
        epsilon1 = AcCoA_bar / Pyr_bar  # constants[48]
        epsilon2 = Cit_bar / Pyr_bar  # constants[49]
        epsilon3 = KG_bar / Pyr_bar  # constants[50]
        epsilon4 = OAA_bar / Pyr_bar  # constants[51]
        epsilon5 = Nt / Pyr_bar  # constants[52]
        epsilon6 = At / Pyr_bar  # constants[53]
        epsilon7 = delta_psi_m / Pyr_bar * C  # constants[54]

        # Initialize rates and algebraic variables, as in computeRates()
        d_p = torch.zeros_like(y[:, 0])
        d_a = torch.zeros_like(y[:, 0])
        d_c = torch.zeros_like(y[:, 0])
        d_k = torch.zeros_like(y[:, 0])
        d_o = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_en = torch.zeros_like(y[:, 0])
        d_s = torch.zeros_like(y[:, 0])
        alg_1_v2 = torch.zeros_like(y[:, 0])
        alg_2_v3 = torch.zeros_like(y[:, 0])
        alg_3_v4 = torch.zeros_like(y[:, 0])
        alg_4_v5 = torch.zeros_like(y[:, 0])
        alg_5_v6 = torch.zeros_like(y[:, 0])
        alg_6_v7 = torch.zeros_like(y[:, 0])
        alg_7_v8 = torch.zeros_like(y[:, 0])
        alg_8_vANT = torch.zeros_like(y[:, 0])
        alg_9_vleak = torch.zeros_like(y[:, 0])
        alg_10_vresp = torch.zeros_like(y[:, 0])
        alg_11_en_crit = torch.zeros_like(y[:, 0])
        alg_12_vATP = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_v2 = beta2 * p * n
        alg_2_v3 = beta3 * o * a
        d_a = torch.zeros_like(y[:, 0]) + ((alg_1_v2 - alg_2_v3) / epsilon1)
        alg_3_v4 = beta4 * c * n
        d_c = torch.zeros_like(y[:, 0]) + ((alg_2_v3 - alg_3_v4) / epsilon2)
        alg_4_v5 = beta5 * k * n * (1.0 - en)
        alg_5_v6 = beta6 * (o - delta_6 * k)
        d_k = torch.zeros_like(y[:, 0]) + ((alg_3_v4 + alg_5_v6 - alg_4_v5) / epsilon3)
        alg_6_v7 = beta7 * p * en
        d_p = torch.zeros_like(y[:, 0]) + (v1 - (alg_1_v2 + alg_6_v7))
        alg_7_v8 = beta8 * o
        d_o = torch.zeros_like(y[:, 0]) + ((alg_4_v5 + alg_6_v7 - (alg_2_v3 + alg_7_v8 + alg_5_v6)) / epsilon4)
        alg_10_vresp = beta_resp * ((1.0 - n) / (delta_r1 + 1.0 - n)) * (1.0 / (1.0 + torch.exp(delta_r2 * (s - 1.0))))
        d_n = torch.zeros_like(y[:, 0]) + ((alg_10_vresp - (alg_1_v2 + alg_3_v4 + 2.0 * alg_4_v5)) / epsilon5)
        alg_11_en_crit = Kapp_dash / (Kapp_dash + torch.exp(-1.0 * delta_crit * s))
        alg_12_vATP = beta_ATP * (2.0 / (1.0 + torch.exp(delta_atp * (en - alg_11_en_crit * s))) - 1.0)
        alg_8_vANT = beta_ANT * en
        d_en = torch.zeros_like(y[:, 0]) + ((alg_12_vATP + alg_4_v5 - (alg_8_vANT + alg_6_v7)) / epsilon6)
        alg_9_vleak = beta_leak * s
        d_s = torch.zeros_like(y[:, 0]) + ((10.0 * alg_10_vresp - (3.0 * alg_12_vATP + alg_9_vleak + alg_8_vANT)) / epsilon7)

        return torch.stack(
            [d_p, d_a, d_c, d_k, d_o, d_n, d_en, d_s],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
