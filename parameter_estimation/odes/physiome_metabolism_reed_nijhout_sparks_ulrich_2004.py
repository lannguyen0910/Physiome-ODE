import torch
import torch.nn as nn
import math


class MetabolismReedNijhoutSparksUlrich2004ODE(nn.Module):
    """
    PyTorch translation of metabolism/reed_nijhout_sparks_ulrich_2004/model.py.
    Dynamic states: 4; independent constants: 21; derived constants used by rates: 0.
    Original constant slots: 21; algebraic slots: 11.
    """
    source_file = 'metabolism/reed_nijhout_sparks_ulrich_2004/model.py'
    source_state_dim = 4
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

        self.state_dim = 4
        self.parameter_dim = 21

        self.parameters = (
            parameters
            if parameters is not None
            else {'V_MATImax': 561, 'Km_MATI': 41, 'Ki_MATI': 50, 'V_MATIIImax': 22870, 'Km2_MATIII': 21.1, 'V_GNMTmax': 10600, 'Km_GNMT': 4500, 'Ki_GNMT': 20, 'V_METHmax': 4521, 'Km2_METH_A': 10, 'five_mTHF': 5.2, 'V_MSmax': 500, 'Kd_MS': 1, 'Km_Hcy_MS': 0.1, 'Km_five_mTHF_MS': 25, 'alpha1': 100, 'alpha2': 10, 'beta1': 1.7, 'beta2': 30, 'V_BHMTmax': 2500, 'Km_BHMT': 12}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Met': 53.5, 'AdoMet': 137.6, 'AdoHcy': 13.2, 'Hcy': 0.88}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['V_MATImax', 'Km_MATI', 'Ki_MATI', 'V_MATIIImax', 'Km2_MATIII', 'V_GNMTmax', 'Km_GNMT', 'Ki_GNMT', 'V_METHmax', 'Km2_METH_A', 'five_mTHF', 'V_MSmax', 'Kd_MS', 'Km_Hcy_MS', 'Km_five_mTHF_MS', 'alpha1', 'alpha2', 'beta1', 'beta2', 'V_BHMTmax', 'Km_BHMT']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Met', 'AdoMet', 'AdoHcy', 'Hcy']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        Met = y[:, 0]
        AdoMet = y[:, 1]
        AdoHcy = y[:, 2]
        Hcy = y[:, 3]

        # Independent parameters (source constant order)
        V_MATImax = theta[:, 0]  # constants[0]
        Km_MATI = theta[:, 1]  # constants[1]
        Ki_MATI = theta[:, 2]  # constants[2]
        V_MATIIImax = theta[:, 3]  # constants[3]
        Km2_MATIII = theta[:, 4]  # constants[4]
        V_GNMTmax = theta[:, 5]  # constants[5]
        Km_GNMT = theta[:, 6]  # constants[6]
        Ki_GNMT = theta[:, 7]  # constants[7]
        V_METHmax = theta[:, 8]  # constants[8]
        Km2_METH_A = theta[:, 9]  # constants[9]
        five_mTHF = theta[:, 10]  # constants[10]
        V_MSmax = theta[:, 11]  # constants[11]
        Kd_MS = theta[:, 12]  # constants[12]
        Km_Hcy_MS = theta[:, 13]  # constants[13]
        Km_five_mTHF_MS = theta[:, 14]  # constants[14]
        alpha1 = theta[:, 15]  # constants[15]
        alpha2 = theta[:, 16]  # constants[16]
        beta1 = theta[:, 17]  # constants[17]
        beta2 = theta[:, 18]  # constants[18]
        V_BHMTmax = theta[:, 19]  # constants[19]
        Km_BHMT = theta[:, 20]  # constants[20]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Met = torch.zeros_like(y[:, 0])
        d_AdoMet = torch.zeros_like(y[:, 0])
        d_AdoHcy = torch.zeros_like(y[:, 0])
        d_Hcy = torch.zeros_like(y[:, 0])
        alg_0_Metin = torch.zeros_like(y[:, 0])
        alg_1_V_MATI = torch.zeros_like(y[:, 0])
        alg_2_Km1_MATIII = torch.zeros_like(y[:, 0])
        alg_3_V_MATIII = torch.zeros_like(y[:, 0])
        alg_4_V_GNMT = torch.zeros_like(y[:, 0])
        alg_5_Km1_METH = torch.zeros_like(y[:, 0])
        alg_6_V_METH = torch.zeros_like(y[:, 0])
        alg_7_V_MS = torch.zeros_like(y[:, 0])
        alg_8_V_AH = torch.zeros_like(y[:, 0])
        alg_9_V_CBS = torch.zeros_like(y[:, 0])
        alg_10_V_BHMT = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_V_MATI = V_MATImax / (1.0 + Km_MATI / Met * (1.0 + AdoMet / Ki_MATI))
        alg_2_Km1_MATIII = 20000.0 / (1.0 + 5.7 * (AdoMet / (AdoMet + 600.0)) ** 2.0)
        alg_3_V_MATIII = V_MATIIImax / (1.0 + alg_2_Km1_MATIII * Km2_MATIII / (Met ** 2.0 + Met * Km2_MATIII))
        alg_5_Km1_METH = 1.0 * (1.0 + AdoHcy / 4.0)
        alg_6_V_METH = V_METHmax / (1.0 + alg_5_Km1_METH / AdoMet + Km2_METH_A + Km2_METH_A * alg_5_Km1_METH / AdoMet)
        alg_4_V_GNMT = V_GNMTmax / (1.0 + (Km_GNMT / AdoMet) ** 2.3) * 1.0 / (1.0 + AdoHcy / Ki_GNMT)
        d_AdoMet = torch.zeros_like(y[:, 0]) + (alg_1_V_MATI + alg_3_V_MATIII - (alg_6_V_METH + alg_4_V_GNMT))
        alg_8_V_AH = alpha1 * (AdoHcy - alpha2 * Hcy)
        d_AdoHcy = torch.zeros_like(y[:, 0]) + (alg_6_V_METH + alg_4_V_GNMT - alg_8_V_AH)
        alg_0_Metin = torch.where((t < 2.0) | (t >= 8.0), 200.0, torch.where((t >= 2.0) & (t < 5.0), 300.0, torch.where((t >= 5.0) & (t < 8.0), 100.0, 200.0)))
        alg_7_V_MS = V_MSmax * five_mTHF * Hcy / (Kd_MS * Km_Hcy_MS + Km_Hcy_MS * five_mTHF + Km_five_mTHF_MS * Hcy + five_mTHF * Hcy)
        alg_10_V_BHMT = (0.7 - 0.025 * (AdoMet + AdoHcy - 150.0)) * V_BHMTmax * Hcy / (Km_BHMT + Hcy)
        d_Met = torch.zeros_like(y[:, 0]) + (alg_7_V_MS + alg_10_V_BHMT + alg_0_Metin - (alg_1_V_MATI + alg_3_V_MATIII))
        alg_9_V_CBS = (beta1 * (AdoMet + AdoHcy) - beta2) * Hcy
        d_Hcy = torch.zeros_like(y[:, 0]) + (alg_8_V_AH - (alg_9_V_CBS + alg_7_V_MS + alg_10_V_BHMT))

        return torch.stack(
            [d_Met, d_AdoMet, d_AdoHcy, d_Hcy],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
