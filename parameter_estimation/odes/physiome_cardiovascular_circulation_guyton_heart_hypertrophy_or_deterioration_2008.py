import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonHeartHypertrophyOrDeterioration2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_heart_hypertrophy_or_deterioration_2008/model.py.
    Dynamic states: 3; independent constants: 8; derived constants used by rates: 2.
    Original constant slots: 10; algebraic slots: 1.
    """
    source_file = 'cardiovascular_circulation/guyton_heart_hypertrophy_or_deterioration_2008/model.py'
    source_state_dim = 3
    source_constant_dim = 10
    algebraic_dim = 1
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 3
        self.parameter_dim = 8

        self.parameters = (
            parameters
            if parameters is not None
            else {'QAO': 5.00707, 'PA': 103.525, 'POT': 35.1148, 'PPA': 15.6376, 'HSL': 1, 'Z13': 0.625, 'HSR': 1, 'DHDTR': 0.05}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'HPL': 1.00163, 'HPR': 1.00237, 'HMD1': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['QAO', 'PA', 'POT', 'PPA', 'HSL', 'Z13', 'HSR', 'DHDTR']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['HPL', 'HPR', 'HMD1']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        HPL = y[:, 0]
        HPR = y[:, 1]
        HMD1 = y[:, 2]

        # Independent parameters (source constant order)
        QAO = theta[:, 0]  # constants[0]
        PA = theta[:, 1]  # constants[1]
        POT = theta[:, 2]  # constants[2]
        PPA = theta[:, 3]  # constants[3]
        HSL = theta[:, 4]  # constants[4]
        Z13 = theta[:, 5]  # constants[5]
        HSR = theta[:, 6]  # constants[6]
        DHDTR = theta[:, 7]  # constants[7]

        # Derived constants from the original initConsts()
        DHM = (POT - 10.0) * DHDTR  # constants[8]
        constant_9 = DHM  # constants[9]

        # Initialize rates and algebraic variables, as in computeRates()
        d_HPL = torch.zeros_like(y[:, 0])
        d_HPR = torch.zeros_like(y[:, 0])
        d_HMD1 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_HMD1 = torch.zeros_like(y[:, 0]) + (constant_9)
        d_HPL = torch.zeros_like(y[:, 0]) + (((PA * QAO / (500.0 * HSL)) ** Z13 - HPL) / 57600.0)
        d_HPR = torch.zeros_like(y[:, 0]) + (((PPA * QAO / (75.0 * HSR)) ** Z13 - HPR) / 57600.0)

        return torch.stack(
            [d_HPL, d_HPR, d_HMD1],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
