import torch
import torch.nn as nn
import math


class EndocrineHaugh2004ODE(nn.Module):
    """
    PyTorch translation of endocrine/haugh_2004/model.py.
    Dynamic states: 4; independent constants: 12; derived constants used by rates: 2.
    Original constant slots: 14; algebraic slots: 1.
    """
    source_file = 'endocrine/haugh_2004/model.py'
    source_state_dim = 4
    source_constant_dim = 14
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

        self.state_dim = 4
        self.parameter_dim = 12

        self.parameters = (
            parameters
            if parameters is not None
            else {'kf1': 0.1, 'kx2': 4.83, 'k_x2': 0.016, 'kt': 0.005, 'L': 0.01, 'ke': 0.1, 'R_initial': 2000, 'krec': 0, 'Vs': 10, 'kdeg': 0.05, 'kappaE': 0.2, 'KD': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'C': 0, 'D': 0, 'R': 2000, 'Ri': 200}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['kf1', 'kx2', 'k_x2', 'kt', 'L', 'ke', 'R_initial', 'krec', 'Vs', 'kdeg', 'kappaE', 'KD']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['C', 'D', 'R', 'Ri']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        C = y[:, 0]
        D = y[:, 1]
        R = y[:, 2]
        Ri = y[:, 3]

        # Independent parameters (source constant order)
        kf1 = theta[:, 0]  # constants[0]
        kx2 = theta[:, 1]  # constants[1]
        k_x2 = theta[:, 2]  # constants[2]
        kt = theta[:, 3]  # constants[3]
        L = theta[:, 4]  # constants[4]
        ke = theta[:, 5]  # constants[5]
        R_initial = theta[:, 6]  # constants[6]
        krec = theta[:, 7]  # constants[7]
        Vs = theta[:, 8]  # constants[8]
        kdeg = theta[:, 9]  # constants[9]
        kappaE = theta[:, 10]  # constants[10]
        KD = theta[:, 11]  # constants[11]

        # Derived constants from the original initConsts()
        kr1 = KD * kf1  # constants[12]
        k_x1 = 0.01 * kr1  # constants[13]

        # Initialize rates and algebraic variables, as in computeRates()
        d_C = torch.zeros_like(y[:, 0])
        d_D = torch.zeros_like(y[:, 0])
        d_R = torch.zeros_like(y[:, 0])
        d_Ri = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_C = torch.zeros_like(y[:, 0]) + (kf1 * L * R + k_x2 * D - (kr1 + kx2 * R + kt) * C)
        d_D = torch.zeros_like(y[:, 0]) + (kx2 * R * C - (k_x2 + k_x1 + ke) * D)
        d_R = torch.zeros_like(y[:, 0]) + (Vs + kr1 * C + (k_x2 + 2.0 * k_x1) * D + krec * Ri - (kf1 * L + kx2 * C + kt) * R)
        d_Ri = torch.zeros_like(y[:, 0]) + (kt * (R + C) - (krec + kdeg) * Ri)

        return torch.stack(
            [d_C, d_D, d_R, d_Ri],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
