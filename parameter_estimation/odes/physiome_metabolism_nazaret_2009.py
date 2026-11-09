import torch
import torch.nn as nn
import math


class MetabolismNazaret2009ODE(nn.Module):
    """
    PyTorch translation of metabolism/nazaret_2009/model.py.
    Dynamic states: 8; independent constants: 24; derived constants used by rates: 1.
    Original constant slots: 25; algebraic slots: 13.
    """
    source_file = 'metabolism/nazaret_2009/model.py'
    source_state_dim = 8
    source_constant_dim = 25
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
        self.parameter_dim = 24

        self.parameters = (
            parameters
            if parameters is not None
            else {'C': 6.75e-06, 'k1': 38, 'k2': 152, 'k3': 57142, 'k4': 53, 'k5': 82361, 'At': 4.16, 'k6': 0.0032, 'k7': 40, 'k8': 3.6, 'kANT': 0.1, 'kleak': 0.426, 'kresp': 2.5, 'K': 2, 'a': 0.1, 'delta_psi_m': 150, 'Nt': 1.07, 'kATP': 131.9, 'b': 4, 'R': 8.314, 'T': 298, 'F': 96485, 'Kapp': 4.4e-06, 'Pi': 2.44}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Pyr': 0.154, 'AcCoA': 0.063, 'Cit': 0.44, 'KG': 0.225, 'OAA': 0.005, 'NAD': 0.856, 'ATP': 3.536, 'delta_psi': 150}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['C', 'k1', 'k2', 'k3', 'k4', 'k5', 'At', 'k6', 'k7', 'k8', 'kANT', 'kleak', 'kresp', 'K', 'a', 'delta_psi_m', 'Nt', 'kATP', 'b', 'R', 'T', 'F', 'Kapp', 'Pi']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Pyr', 'AcCoA', 'Cit', 'KG', 'OAA', 'NAD', 'ATP', 'delta_psi']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (8)
        Pyr = y[:, 0]
        AcCoA = y[:, 1]
        Cit = y[:, 2]
        KG = y[:, 3]
        OAA = y[:, 4]
        NAD = y[:, 5]
        ATP = y[:, 6]
        delta_psi = y[:, 7]

        # Independent parameters (source constant order)
        C = theta[:, 0]  # constants[0]
        k1 = theta[:, 1]  # constants[1]
        k2 = theta[:, 2]  # constants[2]
        k3 = theta[:, 3]  # constants[3]
        k4 = theta[:, 4]  # constants[4]
        k5 = theta[:, 5]  # constants[5]
        At = theta[:, 6]  # constants[6]
        k6 = theta[:, 7]  # constants[7]
        k7 = theta[:, 8]  # constants[8]
        k8 = theta[:, 9]  # constants[9]
        kANT = theta[:, 10]  # constants[10]
        kleak = theta[:, 11]  # constants[11]
        kresp = theta[:, 12]  # constants[12]
        K = theta[:, 13]  # constants[13]
        a = theta[:, 14]  # constants[14]
        delta_psi_m = theta[:, 15]  # constants[15]
        Nt = theta[:, 16]  # constants[16]
        kATP = theta[:, 17]  # constants[17]
        b = theta[:, 18]  # constants[18]
        R = theta[:, 19]  # constants[19]
        T = theta[:, 20]  # constants[20]
        F = theta[:, 21]  # constants[21]
        Kapp = theta[:, 22]  # constants[22]
        Pi = theta[:, 23]  # constants[23]

        # Derived constants from the original initConsts()
        v1 = k1  # constants[24]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Pyr = torch.zeros_like(y[:, 0])
        d_AcCoA = torch.zeros_like(y[:, 0])
        d_Cit = torch.zeros_like(y[:, 0])
        d_KG = torch.zeros_like(y[:, 0])
        d_OAA = torch.zeros_like(y[:, 0])
        d_NAD = torch.zeros_like(y[:, 0])
        d_ATP = torch.zeros_like(y[:, 0])
        d_delta_psi = torch.zeros_like(y[:, 0])
        alg_0_v2 = torch.zeros_like(y[:, 0])
        alg_1_v3 = torch.zeros_like(y[:, 0])
        alg_2_v4 = torch.zeros_like(y[:, 0])
        alg_3_v5 = torch.zeros_like(y[:, 0])
        alg_4_v6 = torch.zeros_like(y[:, 0])
        alg_5_v7 = torch.zeros_like(y[:, 0])
        alg_6_v8 = torch.zeros_like(y[:, 0])
        alg_7_vANT = torch.zeros_like(y[:, 0])
        alg_8_vleak = torch.zeros_like(y[:, 0])
        alg_9_vresp = torch.zeros_like(y[:, 0])
        alg_10_delta_G_transport = torch.zeros_like(y[:, 0])
        alg_11_ATP_crit_delta_psi = torch.zeros_like(y[:, 0])
        alg_12_vATP = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_v2 = k2 * Pyr * NAD
        alg_1_v3 = k3 * OAA * AcCoA
        d_AcCoA = torch.zeros_like(y[:, 0]) + (alg_0_v2 - alg_1_v3)
        alg_2_v4 = k4 * Cit * NAD
        d_Cit = torch.zeros_like(y[:, 0]) + (alg_1_v3 - alg_2_v4)
        alg_3_v5 = k5 * KG * NAD * (At - ATP)
        alg_4_v6 = k6 * (OAA - KG)
        d_KG = torch.zeros_like(y[:, 0]) + (alg_2_v4 + alg_4_v6 - alg_3_v5)
        alg_5_v7 = k7 * Pyr * ATP
        d_Pyr = torch.zeros_like(y[:, 0]) + (v1 - (alg_0_v2 + alg_5_v7))
        alg_6_v8 = k8 * OAA
        d_OAA = torch.zeros_like(y[:, 0]) + (alg_3_v5 + alg_5_v7 - (alg_1_v3 + alg_6_v8 + alg_4_v6))
        alg_9_vresp = kresp * ((Nt - NAD) / (K + Nt - NAD)) * (1.0 / (1.0 + torch.exp(a * (delta_psi - delta_psi_m))))
        d_NAD = torch.zeros_like(y[:, 0]) + (alg_9_vresp - (alg_0_v2 + alg_2_v4 + 2.0 * alg_3_v5))
        alg_10_delta_G_transport = 0.0012 * F * delta_psi
        alg_11_ATP_crit_delta_psi = At / (1.0 + torch.exp(-3.0 * alg_10_delta_G_transport / (R * T)) / (Kapp * Pi))
        alg_12_vATP = kATP * (2.0 / (1.0 + torch.exp(b * (ATP - alg_11_ATP_crit_delta_psi))) - 1.0)
        alg_7_vANT = kANT * ATP
        d_ATP = torch.zeros_like(y[:, 0]) + (alg_12_vATP + alg_3_v5 - (alg_7_vANT + alg_5_v7))
        alg_8_vleak = kleak * delta_psi
        d_delta_psi = torch.zeros_like(y[:, 0]) + (1.0 / C * (10.0 * alg_9_vresp - (3.0 * alg_12_vATP + alg_8_vleak + alg_7_vANT)))

        return torch.stack(
            [d_Pyr, d_AcCoA, d_Cit, d_KG, d_OAA, d_NAD, d_ATP, d_delta_psi],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
