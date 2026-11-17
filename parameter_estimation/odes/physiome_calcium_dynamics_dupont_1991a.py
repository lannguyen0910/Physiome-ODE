import torch
import torch.nn as nn
import math


class CalciumDynamicsDupont1991aODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/dupont_1991a/model.py.
    Dynamic states: 2; independent constants: 13; derived constants used by rates: 0.
    Original constant slots: 13; algebraic slots: 2.
    """
    source_file = 'calcium_dynamics/dupont_1991a/model.py'
    source_state_dim = 2
    source_constant_dim = 13
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
        self.parameter_dim = 13

        self.parameters = (
            parameters
            if parameters is not None
            else {'v0': 1, 'v1': 7.3, 'beta': 0.6, 'VM2': 65, 'VM3': 500, 'KR': 2, 'KA': 0.9, 'kf': 1, 'k': 10, 'K2': 1, 'n': 2, 'm': 2, 'p': 4}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Z': 0.52, 'Y': 0.93}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['v0', 'v1', 'beta', 'VM2', 'VM3', 'KR', 'KA', 'kf', 'k', 'K2', 'n', 'm', 'p']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Z', 'Y']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        Z = y[:, 0]
        Y = y[:, 1]

        # Independent parameters (source constant order)
        v0 = theta[:, 0]  # constants[0]
        v1 = theta[:, 1]  # constants[1]
        beta = theta[:, 2]  # constants[2]
        VM2 = theta[:, 3]  # constants[3]
        VM3 = theta[:, 4]  # constants[4]
        KR = theta[:, 5]  # constants[5]
        KA = theta[:, 6]  # constants[6]
        kf = theta[:, 7]  # constants[7]
        k = theta[:, 8]  # constants[8]
        K2 = theta[:, 9]  # constants[9]
        n = theta[:, 10]  # constants[10]
        m = theta[:, 11]  # constants[11]
        p = theta[:, 12]  # constants[12]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Z = torch.zeros_like(y[:, 0])
        d_Y = torch.zeros_like(y[:, 0])
        alg_0_v2 = torch.zeros_like(y[:, 0])
        alg_1_v3 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_v2 = VM2 * Z ** n / (K2 ** n + Z ** n)
        alg_1_v3 = VM3 * (Y ** m / (KR ** m + Y ** m)) * (Z ** p / (KA ** p + Z ** p))
        d_Z = torch.zeros_like(y[:, 0]) + (v0 + v1 * beta - alg_0_v2 + alg_1_v3 + kf * Y - k * Z)
        d_Y = torch.zeros_like(y[:, 0]) + (alg_0_v2 - alg_1_v3 - kf * Y)

        return torch.stack(
            [d_Z, d_Y],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
