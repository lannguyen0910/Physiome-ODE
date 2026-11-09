import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonAngiotensin2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_angiotensin_2008/model.py.
    Dynamic states: 2; independent constants: 14; derived constants used by rates: 3.
    Original constant slots: 17; algebraic slots: 7.
    """
    source_file = 'cardiovascular_circulation/guyton_angiotensin_2008/model.py'
    source_state_dim = 2
    source_constant_dim = 17
    algebraic_dim = 7
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
        self.parameter_dim = 14

        self.parameters = (
            parameters
            if parameters is not None
            else {'MDFLW': 1.00051, 'ANXM': 0, 'ANV': 5000, 'REK': 1, 'ANGKNS': 0, 'ANGINF': 0, 'ANT': 12, 'ANMUL': 1.8, 'ANMLL': 0.7, 'ANCSNS': 0.4, 'ANUM': 6, 'ANULL': 0.8, 'ANUVM': 0, 'Z12': 5}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'ANX1': 0, 'ANC': 0.859476}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['MDFLW', 'ANXM', 'ANV', 'REK', 'ANGKNS', 'ANGINF', 'ANT', 'ANMUL', 'ANMLL', 'ANCSNS', 'ANUM', 'ANULL', 'ANUVM', 'Z12']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['ANX1', 'ANC']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        ANX1 = y[:, 0]
        ANC = y[:, 1]

        # Independent parameters (source constant order)
        MDFLW = theta[:, 0]  # constants[0]
        ANXM = theta[:, 1]  # constants[1]
        ANV = theta[:, 2]  # constants[2]
        REK = theta[:, 3]  # constants[3]
        ANGKNS = theta[:, 4]  # constants[4]
        ANGINF = theta[:, 5]  # constants[5]
        ANT = theta[:, 6]  # constants[6]
        ANMUL = theta[:, 7]  # constants[7]
        ANMLL = theta[:, 8]  # constants[8]
        ANCSNS = theta[:, 9]  # constants[9]
        ANUM = theta[:, 10]  # constants[10]
        ANULL = theta[:, 11]  # constants[11]
        ANUVM = theta[:, 12]  # constants[12]
        Z12 = theta[:, 13]  # constants[13]

        # Derived constants from the original initConsts()
        MDFLW3 = MDFLW  # constants[14]
        ANGSCR = torch.where(MDFLW3 > 1.0, 1.0 / (1.0 + (MDFLW3 - 1.0) * 72.0), 10.0 - 9.0 / (1.0 + (1.0 - MDFLW3) * 8.0))  # constants[15]
        ANX = (ANGSCR - 1.0) * ANXM  # constants[16]

        # Initialize rates and algebraic variables, as in computeRates()
        d_ANX1 = torch.zeros_like(y[:, 0])
        d_ANC = torch.zeros_like(y[:, 0])
        alg_0_ANPRT = torch.zeros_like(y[:, 0])
        alg_2_ANPR = torch.zeros_like(y[:, 0])
        alg_4_ANPR1 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_ANX1 = torch.zeros_like(y[:, 0]) + ((ANX - ANX1) / ANV)
        alg_0_ANPRT = (ANGSCR + ANX1) * REK
        alg_2_ANPR = torch.where(alg_0_ANPRT < 1e-05, 1e-05, alg_0_ANPRT)
        alg_4_ANPR1 = torch.where(ANGKNS > 0.0, ANGKNS, alg_2_ANPR + ANGINF)
        d_ANC = torch.zeros_like(y[:, 0]) + ((alg_4_ANPR1 - ANC) / ANT)

        return torch.stack(
            [d_ANX1, d_ANC],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
