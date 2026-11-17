import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonStressRelaxation2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_stress_relaxation_2008/model.py.
    Dynamic states: 2; independent constants: 5; derived constants used by rates: 0.
    Original constant slots: 5; algebraic slots: 0.
    """
    source_file = 'cardiovascular_circulation/guyton_stress_relaxation_2008/model.py'
    source_state_dim = 2
    source_constant_dim = 5
    algebraic_dim = 0
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
        self.parameter_dim = 5

        self.parameters = (
            parameters
            if parameters is not None
            else {'VVE': 0.743224, 'SR': 1, 'SRK': 5, 'SR2': 1, 'SRK2': 10000}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'VV7': 0.00366525, 'VV6': 0.0101913}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['VVE', 'SR', 'SRK', 'SR2', 'SRK2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['VV7', 'VV6']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        VV7 = y[:, 0]
        VV6 = y[:, 1]

        # Independent parameters (source constant order)
        VVE = theta[:, 0]  # constants[0]
        SR = theta[:, 1]  # constants[1]
        SRK = theta[:, 2]  # constants[2]
        SR2 = theta[:, 3]  # constants[3]
        SRK2 = theta[:, 4]  # constants[4]

        # Initialize rates and algebraic variables, as in computeRates()
        d_VV7 = torch.zeros_like(y[:, 0])
        d_VV6 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_VV7 = torch.zeros_like(y[:, 0]) + (((VVE - 0.74) * SR - VV7) / SRK)
        d_VV6 = torch.zeros_like(y[:, 0]) + (((VVE - 0.74) * SR2 - VV6) / SRK2)

        return torch.stack(
            [d_VV7, d_VV6],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
