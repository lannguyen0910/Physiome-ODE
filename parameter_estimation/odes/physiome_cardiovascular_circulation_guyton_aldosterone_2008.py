import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonAldosterone2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_aldosterone_2008/model.py.
    Dynamic states: 1; independent constants: 15; derived constants used by rates: 6.
    Original constant slots: 21; algebraic slots: 6.
    """
    source_file = 'cardiovascular_circulation/guyton_aldosterone_2008/model.py'
    source_state_dim = 1
    source_constant_dim = 21
    algebraic_dim = 6
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
        self.parameter_dim = 15

        self.parameters = (
            parameters
            if parameters is not None
            else {'ANM': 0.987545, 'CKE': 4.44092, 'ANMALD': 2.5, 'AMKMUL': 12, 'ALDINF': 0, 'ALDKNS': 0, 'AMT': 60, 'AM1UL': 5, 'AM1LL': 0, 'AMCSNS': 0.65, 'ALDMM': 2.5, 'AMKM': 0.5, 'AMNAM': 0.8, 'AMNAUL': 15, 'AMNALL': 0.04}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'AMC': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['ANM', 'CKE', 'ANMALD', 'AMKMUL', 'ALDINF', 'ALDKNS', 'AMT', 'AM1UL', 'AM1LL', 'AMCSNS', 'ALDMM', 'AMKM', 'AMNAM', 'AMNAUL', 'AMNALL']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['AMC']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (1)
        AMC = y[:, 0]

        # Independent parameters (source constant order)
        ANM = theta[:, 0]  # constants[0]
        CKE = theta[:, 1]  # constants[1]
        ANMALD = theta[:, 2]  # constants[2]
        AMKMUL = theta[:, 3]  # constants[3]
        ALDINF = theta[:, 4]  # constants[4]
        ALDKNS = theta[:, 5]  # constants[5]
        AMT = theta[:, 6]  # constants[6]
        AM1UL = theta[:, 7]  # constants[7]
        AM1LL = theta[:, 8]  # constants[8]
        AMCSNS = theta[:, 9]  # constants[9]
        ALDMM = theta[:, 10]  # constants[10]
        AMKM = theta[:, 11]  # constants[11]
        AMNAM = theta[:, 12]  # constants[12]
        AMNAUL = theta[:, 13]  # constants[13]
        AMNALL = theta[:, 14]  # constants[14]

        # Derived constants from the original initConsts()
        ANMAL = (ANM - 1.0) * ANMALD + 1.0  # constants[15]
        OSMAL = (CKE - 3.3) / 1.0  # constants[16]
        AMRBSC = ANMAL * 0.909 * OSMAL  # constants[17]
        AMRT = (AMRBSC - 1.0) * AMKMUL + 1.0  # constants[18]
        AMR = torch.where(AMRT < 0.0, 0.0, AMRT)  # constants[19]
        AMR1 = torch.where(ALDKNS > 0.0, ALDKNS, AMR + ALDINF)  # constants[20]

        # Initialize rates and algebraic variables, as in computeRates()
        d_AMC = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_AMC = torch.zeros_like(y[:, 0]) + ((AMR1 - AMC) / AMT)

        return torch.stack(
            [d_AMC],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
