import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonNonMuscleBloodFlowControl2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_non_muscle_blood_flow_control_2008/model.py.
    Dynamic states: 3; independent constants: 9; derived constants used by rates: 4.
    Original constant slots: 13; algebraic slots: 5.
    """
    source_file = 'cardiovascular_circulation/guyton_non_muscle_blood_flow_control_2008/model.py'
    source_state_dim = 3
    source_constant_dim = 13
    algebraic_dim = 5
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
        self.parameter_dim = 9

        self.parameters = (
            parameters
            if parameters is not None
            else {'POT': 35.1148, 'POR': 35, 'POK': 0.1, 'A1K': 0.5, 'PON': 0.1, 'A2K': 60, 'POZ': 2, 'A3K': 40000, 'AUTOSN': 0.9}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'AR1T': 1.02127, 'AR2T': 1.01179, 'AR3T': 1.1448}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['POT', 'POR', 'POK', 'A1K', 'PON', 'A2K', 'POZ', 'A3K', 'AUTOSN']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['AR1T', 'AR2T', 'AR3T']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        AR1T = y[:, 0]
        AR2T = y[:, 1]
        AR3T = y[:, 2]

        # Independent parameters (source constant order)
        POT = theta[:, 0]  # constants[0]
        POR = theta[:, 1]  # constants[1]
        POK = theta[:, 2]  # constants[2]
        A1K = theta[:, 3]  # constants[3]
        PON = theta[:, 4]  # constants[4]
        A2K = theta[:, 5]  # constants[5]
        POZ = theta[:, 6]  # constants[6]
        A3K = theta[:, 7]  # constants[7]
        AUTOSN = theta[:, 8]  # constants[8]

        # Derived constants from the original initConsts()
        POD = POT - POR  # constants[9]
        POB = POD * POK + 1.0  # constants[10]
        POA = PON * POD + 1.0  # constants[11]
        POC = POZ * POD + 1.0  # constants[12]

        # Initialize rates and algebraic variables, as in computeRates()
        d_AR1T = torch.zeros_like(y[:, 0])
        d_AR2T = torch.zeros_like(y[:, 0])
        d_AR3T = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_AR1T = torch.zeros_like(y[:, 0]) + ((POB * 1.0 - AR1T) / A1K)
        d_AR2T = torch.zeros_like(y[:, 0]) + ((POA * 1.0 - AR2T) / A2K)
        d_AR3T = torch.zeros_like(y[:, 0]) + ((POC * 1.0 - AR3T) / A3K)

        return torch.stack(
            [d_AR1T, d_AR2T, d_AR3T],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
