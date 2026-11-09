import torch
import torch.nn as nn
import math


class CellCycleGardnerDolnikCollins1998ODE(nn.Module):
    """
    PyTorch translation of cell_cycle/gardner_dolnik_collins_1998/model.py.
    Dynamic states: 5; independent constants: 20; derived constants used by rates: 0.
    Original constant slots: 20; algebraic slots: 6.
    """
    source_file = 'cell_cycle/gardner_dolnik_collins_1998/model.py'
    source_state_dim = 5
    source_constant_dim = 20
    algebraic_dim = 6
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 5
        self.parameter_dim = 20

        self.parameters = (
            parameters
            if parameters is not None
            else {'vi': 0.1, 'k1': 0.5, 'K5': 0.02, 'kd': 0.02, 'K1': 0.02, 'V2': 0.25, 'K2': 0.02, 'K3': 0.1, 'V4': 0.1, 'K4': 0.1, 'vs': 0.1, 'd1': 0.05, 'a1': 1.5, 'a2': 1.5, 'alpha': 0.1, 'Kd': 1, 'K6': 0.3, 'V1_dash': 0.75, 'V3_dash': 0.3, 'M': -1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'C': 0.01, 'X': 0.01, 'Z': 0.28, 'M': 0.01, 'Y': 0.01}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['vi', 'k1', 'K5', 'kd', 'K1', 'V2', 'K2', 'K3', 'V4', 'K4', 'vs', 'd1', 'a1', 'a2', 'alpha', 'Kd', 'K6', 'V1_dash', 'V3_dash', 'M']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['C', 'X', 'Z', 'M', 'Y']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        C = y[:, 0]
        X = y[:, 1]
        Z = y[:, 2]
        M = y[:, 3]
        Y = y[:, 4]

        # Independent parameters (source constant order)
        vi = theta[:, 0]  # constants[0]
        k1 = theta[:, 1]  # constants[1]
        K5 = theta[:, 2]  # constants[2]
        kd = theta[:, 3]  # constants[3]
        K1 = theta[:, 4]  # constants[4]
        V2 = theta[:, 5]  # constants[5]
        K2 = theta[:, 6]  # constants[6]
        K3 = theta[:, 7]  # constants[7]
        V4 = theta[:, 8]  # constants[8]
        K4 = theta[:, 9]  # constants[9]
        vs = theta[:, 10]  # constants[10]
        d1 = theta[:, 11]  # constants[11]
        a1 = theta[:, 12]  # constants[12]
        a2 = theta[:, 13]  # constants[13]
        alpha = theta[:, 14]  # constants[14]
        Kd = theta[:, 15]  # constants[15]
        K6 = theta[:, 16]  # constants[16]
        V1_dash = theta[:, 17]  # constants[17]
        V3_dash = theta[:, 18]  # constants[18]
        const_M = theta[:, 19]  # constants[19]

        # Initialize rates and algebraic variables, as in computeRates()
        d_C = torch.zeros_like(y[:, 0])
        d_X = torch.zeros_like(y[:, 0])
        d_Z = torch.zeros_like(y[:, 0])
        d_M = torch.zeros_like(y[:, 0])
        d_Y = torch.zeros_like(y[:, 0])
        alg_1_V1 = torch.zeros_like(y[:, 0])
        alg_2_X = torch.zeros_like(y[:, 0])
        alg_4_V3 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_C = torch.zeros_like(y[:, 0]) + (vi - k1 * X * C / (C + K5) - kd * C)
        d_Y = torch.zeros_like(y[:, 0]) + (vs - d1 * Y - a1 * C * Y + (a2 + alpha * kd) * Z)
        d_Z = torch.zeros_like(y[:, 0]) + (a1 * C * Y - (a2 + alpha * kd + alpha * d1) * Z)
        alg_1_V1 = C * V1_dash / (C + K6)
        d_M = torch.zeros_like(y[:, 0]) + (alg_1_V1 * const_M / (const_M + K1) - V2 * M / (M + K2))
        alg_2_X = 1.0 - X
        alg_4_V3 = M * V3_dash
        d_X = torch.zeros_like(y[:, 0]) + (alg_4_V3 * alg_2_X / (alg_2_X + K3) - V4 * X / (X + K4))

        return torch.stack(
            [d_C, d_X, d_Z, d_M, d_Y],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
