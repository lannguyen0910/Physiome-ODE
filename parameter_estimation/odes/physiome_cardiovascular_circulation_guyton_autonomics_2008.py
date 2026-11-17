import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonAutonomics2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_autonomics_2008/model.py.
    Dynamic states: 2; independent constants: 31; derived constants used by rates: 10.
    Original constant slots: 41; algebraic slots: 13.
    """
    source_file = 'cardiovascular_circulation/guyton_autonomics_2008/model.py'
    source_state_dim = 2
    source_constant_dim = 41
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
        self.parameter_dim = 31

        self.parameters = (
            parameters
            if parameters is not None
            else {'PA': 103.525, 'PO2ART': 97.0439, 'PLA': 2, 'PRA': 0.00852183, 'PPA': 15.6376, 'CRRFLX': 0, 'EXE': 0, 'AUC1': 0.3, 'O2CHMO': 0.01, 'AUX': 1, 'AUK': 0.004, 'BAROTC': 0.16, 'AU4': -0.060024, 'AUN1': 0.5, 'AULPM': 0, 'EXC': 1, 'EXCXP': 1, 'EXCML': 0.01, 'AUDMP': 0.3, 'AUMAX': 5, 'AUMIN': 0.4, 'AUSLP': 1.5, 'VV9': 2.51, 'AUL': 1.5, 'AUV': 0.55, 'AUS': 1, 'O2A': 0.1, 'AUM1': 3, 'AUM2': 1, 'AUY': 0, 'MDMP': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'AU6': 1.00132, 'AU1': 1.00007}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['PA', 'PO2ART', 'PLA', 'PRA', 'PPA', 'CRRFLX', 'EXE', 'AUC1', 'O2CHMO', 'AUX', 'AUK', 'BAROTC', 'AU4', 'AUN1', 'AULPM', 'EXC', 'EXCXP', 'EXCML', 'AUDMP', 'AUMAX', 'AUMIN', 'AUSLP', 'VV9', 'AUL', 'AUV', 'AUS', 'O2A', 'AUM1', 'AUM2', 'AUY', 'MDMP']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['AU6', 'AU1']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        AU6 = y[:, 0]
        AU1 = y[:, 1]

        # Independent parameters (source constant order)
        PA = theta[:, 0]  # constants[0]
        PO2ART = theta[:, 1]  # constants[1]
        PLA = theta[:, 2]  # constants[2]
        PRA = theta[:, 3]  # constants[3]
        PPA = theta[:, 4]  # constants[4]
        CRRFLX = theta[:, 5]  # constants[5]
        EXE = theta[:, 6]  # constants[6]
        AUC1 = theta[:, 7]  # constants[7]
        O2CHMO = theta[:, 8]  # constants[8]
        AUX = theta[:, 9]  # constants[9]
        AUK = theta[:, 10]  # constants[10]
        BAROTC = theta[:, 11]  # constants[11]
        AU4 = theta[:, 12]  # constants[12]
        AUN1 = theta[:, 13]  # constants[13]
        AULPM = theta[:, 14]  # constants[14]
        EXC = theta[:, 15]  # constants[15]
        EXCXP = theta[:, 16]  # constants[16]
        EXCML = theta[:, 17]  # constants[17]
        AUDMP = theta[:, 18]  # constants[18]
        AUMAX = theta[:, 19]  # constants[19]
        AUMIN = theta[:, 20]  # constants[20]
        AUSLP = theta[:, 21]  # constants[21]
        VV9 = theta[:, 22]  # constants[22]
        AUL = theta[:, 23]  # constants[23]
        AUV = theta[:, 24]  # constants[24]
        AUS = theta[:, 25]  # constants[25]
        O2A = theta[:, 26]  # constants[26]
        AUM1 = theta[:, 27]  # constants[27]
        AUM2 = theta[:, 28]  # constants[28]
        AUY = theta[:, 29]  # constants[29]
        MDMP = theta[:, 30]  # constants[30]

        # Derived constants from the original initConsts()
        PA1 = torch.where(CRRFLX > 1e-07, CRRFLX, PA - EXE)  # constants[31]
        AUC = torch.where((PA1 < 80.0) & (PA1 >= 40.0), 0.005 * (80.0 - PA1) * AUC1, torch.where(PA1 < 40.0, 0.2 * AUC1, 0.0))  # constants[32]
        AUB = torch.where((PA1 < 160.0) & (PA1 >= 80.0), 0.016667 * (160.0 - PA1), torch.where(PA1 < 80.0, 1.3336, 0.0))  # constants[33]
        AUC2 = torch.where((PO2ART < 80.0) & (PO2ART >= 40.0), O2CHMO * (80.0 - PO2ART), torch.where(PO2ART < 40.0, O2CHMO * 40.0, 0.0))  # constants[34]
        A1B = (AUB - 1.0) * AUX + 1.0  # constants[35]
        AU6A = A1B - AU4  # constants[36]
        AUC3 = AUC + AUC2  # constants[37]
        AUN = torch.where(PA1 < 40.0, 0.04 * (40.0 - PA1) * AUN1, 0.0)  # constants[38]
        AULP = (15.0 / (PLA + PRA + PPA) - 1.0) * AULPM + 1.0  # constants[39]
        AUEX = EXC ** EXCXP  # constants[40]

        # Initialize rates and algebraic variables, as in computeRates()
        d_AU6 = torch.zeros_like(y[:, 0])
        d_AU1 = torch.zeros_like(y[:, 0])
        alg_0_AU6C = torch.zeros_like(y[:, 0])
        alg_2_AUTTL1 = torch.zeros_like(y[:, 0])
        alg_4_AUTTL = torch.zeros_like(y[:, 0])
        alg_7_DAU = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_AU6 = torch.zeros_like(y[:, 0]) + ((AU6A - AU6) / BAROTC)
        alg_0_AU6C = AU6
        alg_2_AUTTL1 = (AUEX * AULP * (AUC3 + alg_0_AU6C + AUN) - 1.0) * EXCML + 1.0
        alg_4_AUTTL = torch.where(alg_2_AUTTL1 < 0.0, 0.0, alg_2_AUTTL1)
        alg_7_DAU = (alg_4_AUTTL - AU1) / AUDMP
        d_AU1 = torch.zeros_like(y[:, 0]) + (alg_7_DAU)

        return torch.stack(
            [d_AU6, d_AU1],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
