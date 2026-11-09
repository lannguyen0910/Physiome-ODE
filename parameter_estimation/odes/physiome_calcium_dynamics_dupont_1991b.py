import torch
import torch.nn as nn
import math


class CalciumDynamicsDupont1991bODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/dupont_1991b/model.py.
    Dynamic states: 2; independent constants: 14; derived constants used by rates: 0.
    Original constant slots: 14; algebraic slots: 3.
    """
    source_file = 'calcium_dynamics/dupont_1991b/model.py'
    source_state_dim = 2
    source_constant_dim = 14
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

        self.state_dim = 2
        self.parameter_dim = 14

        self.parameters = (
            parameters
            if parameters is not None
            else {'v0': 1, 'v1': 7.3, 'VM2': 65, 'VM3': 500, 'KR': 2, 'KA': 0.9, 'kf': 1, 'k': 10, 'K2': 1, 'n': 2, 'm': 2, 'p': 4, 'betaf': 0.96, 'tp': 4}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Z': 0.1, 'Y': 0.64}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['v0', 'v1', 'VM2', 'VM3', 'KR', 'KA', 'kf', 'k', 'K2', 'n', 'm', 'p', 'betaf', 'tp']
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
        VM2 = theta[:, 2]  # constants[2]
        VM3 = theta[:, 3]  # constants[3]
        KR = theta[:, 4]  # constants[4]
        KA = theta[:, 5]  # constants[5]
        kf = theta[:, 6]  # constants[6]
        k = theta[:, 7]  # constants[7]
        K2 = theta[:, 8]  # constants[8]
        n = theta[:, 9]  # constants[9]
        m = theta[:, 10]  # constants[10]
        p = theta[:, 11]  # constants[11]
        betaf = theta[:, 12]  # constants[12]
        tp = theta[:, 13]  # constants[13]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Z = torch.zeros_like(y[:, 0])
        d_Y = torch.zeros_like(y[:, 0])
        alg_0_v2 = torch.zeros_like(y[:, 0])
        alg_1_v3 = torch.zeros_like(y[:, 0])
        alg_2_beta = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_v2 = VM2 * Z ** n / (K2 ** n + Z ** n)
        alg_1_v3 = VM3 * (Y ** m / (KR ** m + Y ** m)) * (Z ** p / (KA ** p + Z ** p))
        d_Y = torch.zeros_like(y[:, 0]) + (alg_0_v2 - alg_1_v3 - kf * Y)
        alg_2_beta = torch.where(t < tp, 0.0, torch.where(t >= tp, betaf * torch.exp(-0.2 * (t - tp)), (1e309-1e309)))
        d_Z = torch.zeros_like(y[:, 0]) + (v0 + v1 * alg_2_beta - alg_0_v2 + alg_1_v3 + kf * Y - k * Z)

        return torch.stack(
            [d_Z, d_Y],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
