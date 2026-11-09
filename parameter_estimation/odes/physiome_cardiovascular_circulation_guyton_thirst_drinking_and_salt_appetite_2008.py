import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonThirstDrinkingAndSaltAppetite2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_thirst_drinking_and_salt_appetite_2008/model.py.
    Dynamic states: 1; independent constants: 10; derived constants used by rates: 9.
    Original constant slots: 19; algebraic slots: 0.
    """
    source_file = 'cardiovascular_circulation/guyton_thirst_drinking_and_salt_appetite_2008/model.py'
    source_state_dim = 1
    source_constant_dim = 19
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

        self.state_dim = 1
        self.parameter_dim = 10

        self.parameters = (
            parameters
            if parameters is not None
            else {'ADHC': 1, 'ANM': 0.987545, 'POT': 35.1148, 'ANMSLT': 2, 'Z10': 45, 'Z11': 0.01, 'AHTHM': 2, 'ANMTM': 1.5, 'DR': 0, 'TVDDL': 30}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'TVD': 0.000980838}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['ADHC', 'ANM', 'POT', 'ANMSLT', 'Z10', 'Z11', 'AHTHM', 'ANMTM', 'DR', 'TVDDL']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['TVD']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (1)
        TVD = y[:, 0]

        # Independent parameters (source constant order)
        ADHC = theta[:, 0]  # constants[0]
        ANM = theta[:, 1]  # constants[1]
        POT = theta[:, 2]  # constants[2]
        ANMSLT = theta[:, 3]  # constants[3]
        Z10 = theta[:, 4]  # constants[4]
        Z11 = theta[:, 5]  # constants[5]
        AHTHM = theta[:, 6]  # constants[6]
        ANMTM = theta[:, 7]  # constants[7]
        DR = theta[:, 8]  # constants[8]
        TVDDL = theta[:, 9]  # constants[9]

        # Derived constants from the original initConsts()
        ANMSML = (ANM - 1.0) * ANMSLT + 1.0  # constants[10]
        STH1 = (Z10 - POT) ** 2.0 * Z11 * ANMSML  # constants[11]
        STH = torch.where(STH1 < 0.8, 0.8, torch.where(STH1 > 8.0, 8.0, STH1))  # constants[12]
        AHCM = (ADHC - 1.0) * AHTHM + 1.0  # constants[13]
        ANMTH = (ANM - 1.0) * ANMTM * 0.001  # constants[14]
        AHTH1 = AHCM * STH * 0.001  # constants[15]
        AHTH = torch.where(AHTH1 < 0.0, 0.0, AHTH1)  # constants[16]
        TVZ1 = (ANMTH + AHTH) * 1.0  # constants[17]
        TVZ = torch.where(TVZ1 < 0.0, 0.0, TVZ1)  # constants[18]

        # Initialize rates and algebraic variables, as in computeRates()
        d_TVD = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_TVD = torch.zeros_like(y[:, 0]) + ((TVZ + DR - TVD) / TVDDL)

        return torch.stack(
            [d_TVD],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
