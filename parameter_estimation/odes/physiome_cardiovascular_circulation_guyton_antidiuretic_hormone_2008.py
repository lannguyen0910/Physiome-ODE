import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonAntidiureticHormone2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_antidiuretic_hormone_2008/model.py.
    Dynamic states: 1; independent constants: 11; derived constants used by rates: 6.
    Original constant slots: 17; algebraic slots: 4.
    """
    source_file = 'cardiovascular_circulation/guyton_antidiuretic_hormone_2008/model.py'
    source_state_dim = 1
    source_constant_dim = 17
    algebraic_dim = 4
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
        self.parameter_dim = 11

        self.parameters = (
            parameters
            if parameters is not None
            else {'CNA': 142.035, 'PA1': 103.525, 'CNR': 139, 'ADHPUL': 85, 'ADHPAM': 0.3, 'ADHINF': 0, 'ADHTC': 15, 'ADHVUL': 2.5, 'ADHVLL': 0.93617, 'ADHKLL': 0.2, 'ADHKUL': 5}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'ADHC': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['CNA', 'PA1', 'CNR', 'ADHPUL', 'ADHPAM', 'ADHINF', 'ADHTC', 'ADHVUL', 'ADHVLL', 'ADHKLL', 'ADHKUL']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['ADHC']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (1)
        ADHC = y[:, 0]

        # Independent parameters (source constant order)
        CNA = theta[:, 0]  # constants[0]
        PA1 = theta[:, 1]  # constants[1]
        CNR = theta[:, 2]  # constants[2]
        ADHPUL = theta[:, 3]  # constants[3]
        ADHPAM = theta[:, 4]  # constants[4]
        ADHINF = theta[:, 5]  # constants[5]
        ADHTC = theta[:, 6]  # constants[6]
        ADHVUL = theta[:, 7]  # constants[7]
        ADHVLL = theta[:, 8]  # constants[8]
        ADHKLL = theta[:, 9]  # constants[9]
        ADHKUL = theta[:, 10]  # constants[10]

        # Derived constants from the original initConsts()
        ADHNA1 = (CNA - CNR) / (142.0 - CNR)  # constants[11]
        ADHNA = torch.where(ADHNA1 < 0.0, 0.0, ADHNA1)  # constants[12]
        ADHPA = torch.where(PA1 > ADHPUL, ADHPUL, PA1)  # constants[13]
        ADHPR = (ADHPUL - ADHPA) ** 2.0 * ADHPAM  # constants[14]
        ADH1 = ADHNA + ADHPR + ADHINF  # constants[15]
        ADH = torch.where(ADH1 < 0.0, 0.0, ADH1)  # constants[16]

        # Initialize rates and algebraic variables, as in computeRates()
        d_ADHC = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_ADHC = torch.zeros_like(y[:, 0]) + ((ADH - ADHC) / ADHTC)

        return torch.stack(
            [d_ADHC],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
