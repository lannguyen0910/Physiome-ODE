import torch
import torch.nn as nn
import math


class CalciumDynamicsDupont1992bODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/dupont_1992b/model.py.
    Dynamic states: 3; independent constants: 18; derived constants used by rates: 1.
    Original constant slots: 19; algebraic slots: 3.
    """
    source_file = 'calcium_dynamics/dupont_1992b/model.py'
    source_state_dim = 3
    source_constant_dim = 19
    algebraic_dim = 3
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
        self.parameter_dim = 18

        self.parameters = (
            parameters
            if parameters is not None
            else {'VM2': 65, 'VM3': 500, 'KR': 2, 'KA': 0.9, 'KP': 1, 'n': 2, 'm': 2, 'p': 4, 'kf': 1, 'k': 10, 'v0': 1, 'v1beta': 2.4, 'K1': 0.01, 'K2': 0.01, 'WT': 10, 'vMK': 100, 'Ka': 1, 'q': 4}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Y': 1.454, 'Z': 0.2281, 'Wstar': 0.8916}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['VM2', 'VM3', 'KR', 'KA', 'KP', 'n', 'm', 'p', 'kf', 'k', 'v0', 'v1beta', 'K1', 'K2', 'WT', 'vMK', 'Ka', 'q']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Y', 'Z', 'Wstar']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        Y = y[:, 0]
        Z = y[:, 1]
        Wstar = y[:, 2]

        # Independent parameters (source constant order)
        VM2 = theta[:, 0]  # constants[0]
        VM3 = theta[:, 1]  # constants[1]
        KR = theta[:, 2]  # constants[2]
        KA = theta[:, 3]  # constants[3]
        KP = theta[:, 4]  # constants[4]
        n = theta[:, 5]  # constants[5]
        m = theta[:, 6]  # constants[6]
        p = theta[:, 7]  # constants[7]
        kf = theta[:, 8]  # constants[8]
        k = theta[:, 9]  # constants[9]
        v0 = theta[:, 10]  # constants[10]
        v1beta = theta[:, 11]  # constants[11]
        K1 = theta[:, 12]  # constants[12]
        K2 = theta[:, 13]  # constants[13]
        WT = theta[:, 14]  # constants[14]
        vMK = theta[:, 15]  # constants[15]
        Ka = theta[:, 16]  # constants[16]
        q = theta[:, 17]  # constants[17]

        # Derived constants from the original initConsts()
        vP = vMK * (0.89 / 20.0)  # constants[18]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Y = torch.zeros_like(y[:, 0])
        d_Z = torch.zeros_like(y[:, 0])
        d_Wstar = torch.zeros_like(y[:, 0])
        alg_0_v2 = torch.zeros_like(y[:, 0])
        alg_1_vK = torch.zeros_like(y[:, 0])
        alg_2_v3 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_vK = vMK * (Z ** q / (Ka ** q + Z ** q))
        d_Wstar = torch.zeros_like(y[:, 0]) + (vP / WT * (alg_1_vK / vP * (1.0 - Wstar) / (K1 + 1.0 - Wstar) - Wstar / (K2 + Wstar)))
        alg_0_v2 = VM2 * Z ** n / (KP ** n + Z ** n)
        alg_2_v3 = VM3 * (Y ** m / (KR ** m + Y ** m)) * (Z ** p / (KA ** p + Z ** p))
        d_Z = torch.zeros_like(y[:, 0]) + (v0 + v1beta - alg_0_v2 + alg_2_v3 + kf * Y - k * Z)
        d_Y = torch.zeros_like(y[:, 0]) + (alg_0_v2 - alg_2_v3 - kf * Y)

        return torch.stack(
            [d_Y, d_Z, d_Wstar],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
