import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonRedCellsAndViscosity2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_red_cells_and_viscosity_2008/model.py.
    Dynamic states: 1; independent constants: 9; derived constants used by rates: 2.
    Original constant slots: 11; algebraic slots: 12.
    """
    source_file = 'cardiovascular_circulation/guyton_red_cells_and_viscosity_2008/model.py'
    source_state_dim = 1
    source_constant_dim = 11
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

        self.state_dim = 1
        self.parameter_dim = 9

        self.parameters = (
            parameters
            if parameters is not None
            else {'VP': 3.00449, 'HMK': 90, 'HKM': 0.53333, 'PO2AMB': 150, 'HM6': 1850, 'HM8': 4.714e-08, 'REK': 1, 'RKC': 5.8e-06, 'TRRBC': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'VRC': 2.00439}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['VP', 'HMK', 'HKM', 'PO2AMB', 'HM6', 'HM8', 'REK', 'RKC', 'TRRBC']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['VRC']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (1)
        VRC = y[:, 0]

        # Independent parameters (source constant order)
        VP = theta[:, 0]  # constants[0]
        HMK = theta[:, 1]  # constants[1]
        HKM = theta[:, 2]  # constants[2]
        PO2AMB = theta[:, 3]  # constants[3]
        HM6 = theta[:, 4]  # constants[4]
        HM8 = theta[:, 5]  # constants[5]
        REK = theta[:, 6]  # constants[6]
        RKC = theta[:, 7]  # constants[7]
        TRRBC = theta[:, 8]  # constants[8]

        # Derived constants from the original initConsts()
        PO2AM1 = torch.where(PO2AMB > 80.0, 80.0, PO2AMB)  # constants[9]
        HM4 = PO2AMB - 40.0  # constants[10]

        # Initialize rates and algebraic variables, as in computeRates()
        d_VRC = torch.zeros_like(y[:, 0])
        alg_0_VB = torch.zeros_like(y[:, 0])
        alg_1_HM1 = torch.zeros_like(y[:, 0])
        alg_2_HM = torch.zeros_like(y[:, 0])
        alg_3_VIE = torch.zeros_like(y[:, 0])
        alg_4_VIB = torch.zeros_like(y[:, 0])
        alg_5_VIM = torch.zeros_like(y[:, 0])
        alg_6_HM3 = torch.zeros_like(y[:, 0])
        alg_7_HM5 = torch.zeros_like(y[:, 0])
        alg_8_HM7 = torch.zeros_like(y[:, 0])
        alg_9_RC1 = torch.zeros_like(y[:, 0])
        alg_10_RC2 = torch.zeros_like(y[:, 0])
        alg_11_RCD = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_VB = VP + VRC
        alg_1_HM1 = VRC / alg_0_VB
        alg_2_HM = 100.0 * alg_1_HM1
        alg_6_HM3 = (PO2AM1 - 40.0) * alg_2_HM
        alg_7_HM5 = torch.where(alg_6_HM3 + HM4 < 0.0, 0.0, alg_6_HM3 + HM4)
        alg_8_HM7 = HM6 - alg_7_HM5
        alg_9_RC1 = torch.where(alg_8_HM7 * HM8 * REK + 5e-06 < 0.0, 0.0, alg_8_HM7 * HM8 * REK + 5e-06)
        alg_3_VIE = alg_2_HM / ((HMK - alg_2_HM) * HKM)
        alg_4_VIB = alg_3_VIE + 1.5
        alg_5_VIM = 0.3333 * alg_4_VIB
        alg_10_RC2 = VRC * RKC * alg_5_VIM
        alg_11_RCD = alg_9_RC1 - alg_10_RC2 + TRRBC
        d_VRC = torch.zeros_like(y[:, 0]) + (alg_11_RCD)

        return torch.stack(
            [d_VRC],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
