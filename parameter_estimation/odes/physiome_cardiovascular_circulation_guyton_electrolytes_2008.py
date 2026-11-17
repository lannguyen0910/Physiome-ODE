import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonElectrolytes2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_electrolytes_2008/model.py.
    Dynamic states: 4; independent constants: 11; derived constants used by rates: 6.
    Original constant slots: 17; algebraic slots: 8.
    """
    source_file = 'cardiovascular_circulation/guyton_electrolytes_2008/model.py'
    source_state_dim = 4
    source_constant_dim = 17
    algebraic_dim = 8
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
            else {'AMK': 1.037, 'TVD': 0.000980838, 'NOD': 0.0959449, 'STH': 0.977181, 'KOD': 0.0804374, 'VUD': 0.000989, 'NID': 0.1, 'TRPL': 0, 'ALCLK': 0.3, 'KID': 0.08, 'VIDML': 0.01}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'NAE': 2109.91, 'KTOT': 3622.54, 'VIC': 25.0404, 'VTW': 39.8952}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['AMK', 'TVD', 'NOD', 'STH', 'KOD', 'VUD', 'NID', 'TRPL', 'ALCLK', 'KID', 'VIDML']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['NAE', 'KTOT', 'VIC', 'VTW']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        NAE = y[:, 0]
        KTOT = y[:, 1]
        VIC = y[:, 2]
        VTW = y[:, 3]

        # Independent parameters (source constant order)
        AMK = theta[:, 0]  # constants[0]
        TVD = theta[:, 1]  # constants[1]
        NOD = theta[:, 2]  # constants[2]
        STH = theta[:, 3]  # constants[3]
        KOD = theta[:, 4]  # constants[4]
        VUD = theta[:, 5]  # constants[5]
        NID = theta[:, 6]  # constants[6]
        TRPL = theta[:, 7]  # constants[7]
        ALCLK = theta[:, 8]  # constants[8]
        KID = theta[:, 9]  # constants[9]
        VIDML = theta[:, 10]  # constants[10]

        # Derived constants from the original initConsts()
        NED = NID * STH - NOD + TRPL * 142.0  # constants[11]
        AMK1 = (AMK - 1.0) * ALCLK + 1.0  # constants[12]
        KTOTD = KID - KOD  # constants[13]
        constant_14 = TVD - VUD  # constants[14]
        constant_15 = NED  # constants[15]
        constant_16 = KTOTD  # constants[16]

        # Initialize rates and algebraic variables, as in computeRates()
        d_NAE = torch.zeros_like(y[:, 0])
        d_KTOT = torch.zeros_like(y[:, 0])
        d_VIC = torch.zeros_like(y[:, 0])
        d_VTW = torch.zeros_like(y[:, 0])
        alg_0_KE = torch.zeros_like(y[:, 0])
        alg_1_KI = torch.zeros_like(y[:, 0])
        alg_2_CKI = torch.zeros_like(y[:, 0])
        alg_3_VEC = torch.zeros_like(y[:, 0])
        alg_4_CNA = torch.zeros_like(y[:, 0])
        alg_6_CCD = torch.zeros_like(y[:, 0])
        alg_7_VID = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_VTW = torch.zeros_like(y[:, 0]) + (constant_14)
        d_NAE = torch.zeros_like(y[:, 0]) + (constant_15)
        d_KTOT = torch.zeros_like(y[:, 0]) + (constant_16)
        alg_3_VEC = VTW - VIC
        alg_4_CNA = NAE / alg_3_VEC
        alg_0_KE = (KTOT - 3000.0) / (AMK1 * 9.3333)
        alg_1_KI = KTOT - alg_0_KE
        alg_2_CKI = alg_1_KI / VIC
        alg_6_CCD = alg_2_CKI - alg_4_CNA
        alg_7_VID = alg_6_CCD * VIDML
        d_VIC = torch.zeros_like(y[:, 0]) + (alg_7_VID)

        return torch.stack(
            [d_NAE, d_KTOT, d_VIC, d_VTW],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
