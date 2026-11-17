import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonMuscleBloodFlowControl2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_muscle_blood_flow_control_2008/model.py.
    Dynamic states: 2; independent constants: 6; derived constants used by rates: 3.
    Original constant slots: 9; algebraic slots: 2.
    """
    source_file = 'cardiovascular_circulation/guyton_muscle_blood_flow_control_2008/model.py'
    source_state_dim = 2
    source_constant_dim = 9
    algebraic_dim = 2
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
        self.parameter_dim = 6

        self.parameters = (
            parameters
            if parameters is not None
            else {'PMO': 38.0666, 'POM': 0.04, 'A4K': 0.1, 'AMM4': 0.005, 'POM2': 2, 'A4K2': 40000}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'AMM1T': 1.00269, 'AMM2': 1.09071}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['PMO', 'POM', 'A4K', 'AMM4', 'POM2', 'A4K2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['AMM1T', 'AMM2']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        AMM1T = y[:, 0]
        AMM2 = y[:, 1]

        # Independent parameters (source constant order)
        PMO = theta[:, 0]  # constants[0]
        POM = theta[:, 1]  # constants[1]
        A4K = theta[:, 2]  # constants[2]
        AMM4 = theta[:, 3]  # constants[3]
        POM2 = theta[:, 4]  # constants[4]
        A4K2 = theta[:, 5]  # constants[5]

        # Derived constants from the original initConsts()
        PDO = PMO - 38.0  # constants[6]
        POE = PDO * POM + 1.0  # constants[7]
        POF = POM2 * PDO + 1.0  # constants[8]

        # Initialize rates and algebraic variables, as in computeRates()
        d_AMM1T = torch.zeros_like(y[:, 0])
        d_AMM2 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_AMM1T = torch.zeros_like(y[:, 0]) + ((POE * 1.0 - AMM1T) / A4K)
        d_AMM2 = torch.zeros_like(y[:, 0]) + ((POF * 1.0 - AMM2) / A4K2)

        return torch.stack(
            [d_AMM1T, d_AMM2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
