import torch
import torch.nn as nn
import math


class SignalTransductionRiccobeneOmannLinderman1999ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/riccobene_omann_linderman_1999/model.py.
    Dynamic states: 8; independent constants: 11; derived constants used by rates: 0.
    Original constant slots: 11; algebraic slots: 0.
    """
    source_file = 'signal_transduction/riccobene_omann_linderman_1999/model.py'
    source_state_dim = 8
    source_constant_dim = 11
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

        self.state_dim = 8
        self.parameter_dim = 11

        self.parameters = (
            parameters
            if parameters is not None
            else {'L': 1e-12, 'kf': 84000000, 'kr': 0.37, 'kfR': 10, 'Kact': 0.0001, 'alpha': 10, 'kds': 0.0001, 'kf2': 84000000, 'kr2': 0.0046, 'ka': 1e-07, 'ki': 0.2}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'R': 0.01, 'LR': 0.01, 'R_star': 0.01, 'LR_star': 0.01, 'LR_ds': 0.01, 'R_ds': 0.01, 'G_star': 0.01, 'G': 0.01}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['L', 'kf', 'kr', 'kfR', 'Kact', 'alpha', 'kds', 'kf2', 'kr2', 'ka', 'ki']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['R', 'LR', 'R_star', 'LR_star', 'LR_ds', 'R_ds', 'G_star', 'G']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (8)
        R = y[:, 0]
        LR = y[:, 1]
        R_star = y[:, 2]
        LR_star = y[:, 3]
        LR_ds = y[:, 4]
        R_ds = y[:, 5]
        G_star = y[:, 6]
        G = y[:, 7]

        # Independent parameters (source constant order)
        L = theta[:, 0]  # constants[0]
        kf = theta[:, 1]  # constants[1]
        kr = theta[:, 2]  # constants[2]
        kfR = theta[:, 3]  # constants[3]
        Kact = theta[:, 4]  # constants[4]
        alpha = theta[:, 5]  # constants[5]
        kds = theta[:, 6]  # constants[6]
        kf2 = theta[:, 7]  # constants[7]
        kr2 = theta[:, 8]  # constants[8]
        ka = theta[:, 9]  # constants[9]
        ki = theta[:, 10]  # constants[10]

        # Initialize rates and algebraic variables, as in computeRates()
        d_R = torch.zeros_like(y[:, 0])
        d_LR = torch.zeros_like(y[:, 0])
        d_R_star = torch.zeros_like(y[:, 0])
        d_LR_star = torch.zeros_like(y[:, 0])
        d_LR_ds = torch.zeros_like(y[:, 0])
        d_R_ds = torch.zeros_like(y[:, 0])
        d_G_star = torch.zeros_like(y[:, 0])
        d_G = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_R = torch.zeros_like(y[:, 0]) + (kr * LR + kfR / Kact * R_star - (kf * L * R + kfR * R))
        d_R_star = torch.zeros_like(y[:, 0]) + (kr * LR_star + kfR * R - (alpha * kf * L * R_star + kfR / Kact * R_star))
        d_LR = torch.zeros_like(y[:, 0]) + (kf * L * R + kfR / (alpha * Kact) * LR_star - (kr * LR + kfR * LR))
        d_LR_star = torch.zeros_like(y[:, 0]) + (kfR * LR + alpha * kf * L * R_star - (kfR / (alpha * Kact) * LR_star + kds * LR_star + kr * LR_star))
        d_LR_ds = torch.zeros_like(y[:, 0]) + (kds * LR_star + kf2 * L * R_ds - kr2 * LR_ds)
        d_R_ds = torch.zeros_like(y[:, 0]) + (kr2 * LR_ds - kf2 * L * R_ds)
        d_G_star = torch.zeros_like(y[:, 0]) + (ka * G * (LR_star + R_star) - ki * G_star)
        d_G = torch.zeros_like(y[:, 0]) + (ki * G_star - ka * G * (LR_star + R_star))

        return torch.stack(
            [d_R, d_LR, d_R_star, d_LR_star, d_LR_ds, d_R_ds, d_G_star, d_G],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
