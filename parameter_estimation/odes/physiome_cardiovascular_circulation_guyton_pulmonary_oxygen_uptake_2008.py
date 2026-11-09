import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonPulmonaryOxygenUptake2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_pulmonary_oxygen_uptake_2008/model.py.
    Dynamic states: 2; independent constants: 9; derived constants used by rates: 2.
    Original constant slots: 11; algebraic slots: 11.
    """
    source_file = 'cardiovascular_circulation/guyton_pulmonary_oxygen_uptake_2008/model.py'
    source_state_dim = 2
    source_constant_dim = 11
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

        self.state_dim = 2
        self.parameter_dim = 9

        self.parameters = (
            parameters
            if parameters is not None
            else {'VPF': 0.0123238, 'DOB': 163.508, 'QRO': 4.97838, 'RMO': 56.8057, 'HM': 40.0381, 'VNTSTM': 1, 'PO2AMB': 150, 'PL2': 1.8, 'VPTISS': 0.0175}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'OVA': 204.497, 'O2VAD1': 2.368e-07}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['VPF', 'DOB', 'QRO', 'RMO', 'HM', 'VNTSTM', 'PO2AMB', 'PL2', 'VPTISS']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['OVA', 'O2VAD1']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        OVA = y[:, 0]
        O2VAD1 = y[:, 1]

        # Independent parameters (source constant order)
        VPF = theta[:, 0]  # constants[0]
        DOB = theta[:, 1]  # constants[1]
        QRO = theta[:, 2]  # constants[2]
        RMO = theta[:, 3]  # constants[3]
        HM = theta[:, 4]  # constants[4]
        VNTSTM = theta[:, 5]  # constants[5]
        PO2AMB = theta[:, 6]  # constants[6]
        PL2 = theta[:, 7]  # constants[7]
        VPTISS = theta[:, 8]  # constants[8]

        # Derived constants from the original initConsts()
        O2UTIL = DOB + RMO  # constants[9]
        RSPDFC = PL2 / (VPTISS + VPF)  # constants[10]

        # Initialize rates and algebraic variables, as in computeRates()
        d_OVA = torch.zeros_like(y[:, 0])
        d_O2VAD1 = torch.zeros_like(y[:, 0])
        alg_0_OSA = torch.zeros_like(y[:, 0])
        alg_1_PO2ART = torch.zeros_like(y[:, 0])
        alg_2_O2VTST1 = torch.zeros_like(y[:, 0])
        alg_3_O2VTST = torch.zeros_like(y[:, 0])
        alg_4_O2VTS2 = torch.zeros_like(y[:, 0])
        alg_5_O2VAD2 = torch.zeros_like(y[:, 0])
        alg_6_DO2VAD = torch.zeros_like(y[:, 0])
        alg_7_ALVENT = torch.zeros_like(y[:, 0])
        alg_8_PO2ALV = torch.zeros_like(y[:, 0])
        alg_9_O2DFS = torch.zeros_like(y[:, 0])
        alg_10_DOVA = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_OSA = OVA / HM / 5.25
        alg_1_PO2ART = torch.where(alg_0_OSA > 1.0, 114.0 + (alg_0_OSA - 1.0) * 6667.0, torch.where((alg_0_OSA > 0.936) & (alg_0_OSA <= 1.0), 74.0 + (alg_0_OSA - 0.936) * 625.0, torch.where((alg_0_OSA > 0.8) & (alg_0_OSA <= 0.936), 46.0 + (alg_0_OSA - 0.8) * 205.882, alg_0_OSA * 57.5)))
        alg_2_O2VTST1 = (alg_1_PO2ART - 67.0) / 30.0
        alg_3_O2VTST = torch.where(alg_2_O2VTST1 > 1.0, 1.0, torch.where(alg_2_O2VTST1 < 0.6, 0.6, alg_2_O2VTST1))
        alg_4_O2VTS2 = 1.0 / alg_3_O2VTST
        alg_6_DO2VAD = ((alg_4_O2VTS2 - 1.0) * 3.0 - O2VAD1) * 0.0005
        d_O2VAD1 = torch.zeros_like(y[:, 0]) + (alg_6_DO2VAD)
        alg_5_O2VAD2 = O2VAD1 + 1.0
        alg_7_ALVENT = O2UTIL * VNTSTM * 0.026667 * alg_4_O2VTS2 * alg_5_O2VAD2
        alg_8_PO2ALV = PO2AMB - O2UTIL / alg_7_ALVENT / 0.761
        alg_9_O2DFS = (alg_8_PO2ALV - alg_1_PO2ART) * RSPDFC
        alg_10_DOVA = (alg_9_O2DFS - O2UTIL) / (QRO * 1.0)
        d_OVA = torch.zeros_like(y[:, 0]) + (alg_10_DOVA)

        return torch.stack(
            [d_OVA, d_O2VAD1],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
