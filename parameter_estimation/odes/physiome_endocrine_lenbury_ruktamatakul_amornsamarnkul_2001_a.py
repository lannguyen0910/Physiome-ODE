import torch
import torch.nn as nn
import math


class EndocrineLenburyRuktamatakulAmornsamarnkul2001AODE(nn.Module):
    """
    PyTorch translation of endocrine/lenbury_ruktamatakul_amornsamarnkul_2001_a/model.py.
    Dynamic states: 3; independent constants: 12; derived constants used by rates: 0.
    Original constant slots: 12; algebraic slots: 0.
    """
    source_file = 'endocrine/lenbury_ruktamatakul_amornsamarnkul_2001_a/model.py'
    source_state_dim = 3
    source_constant_dim = 12
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

        self.state_dim = 3
        self.parameter_dim = 12

        self.parameters = (
            parameters
            if parameters is not None
            else {'r_1': 0.2, 'r_2': 0.1, 'c_1': 0.1, 'r_3': 0.1, 'r_4': 0.1, 'c_2': 0.1, 'epsilon': 0.1, 'r_5': 0.1, 'r_6': 0.1, 'r_7': 0.05, 'z_hat': 2, 'y_hat': 1.24}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'x': 4, 'z': 1, 'y': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['r_1', 'r_2', 'c_1', 'r_3', 'r_4', 'c_2', 'epsilon', 'r_5', 'r_6', 'r_7', 'z_hat', 'y_hat']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['x', 'z', 'y']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        x = y[:, 0]
        z = y[:, 1]
        state_y = y[:, 2]

        # Independent parameters (source constant order)
        r_1 = theta[:, 0]  # constants[0]
        r_2 = theta[:, 1]  # constants[1]
        c_1 = theta[:, 2]  # constants[2]
        r_3 = theta[:, 3]  # constants[3]
        r_4 = theta[:, 4]  # constants[4]
        c_2 = theta[:, 5]  # constants[5]
        epsilon = theta[:, 6]  # constants[6]
        r_5 = theta[:, 7]  # constants[7]
        r_6 = theta[:, 8]  # constants[8]
        r_7 = theta[:, 9]  # constants[9]
        z_hat = theta[:, 10]  # constants[10]
        y_hat = theta[:, 11]  # constants[11]

        # Initialize rates and algebraic variables, as in computeRates()
        d_x = torch.zeros_like(y[:, 0])
        d_z = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_x = torch.zeros_like(y[:, 0]) + (z * (r_1 * state_y + -r_2 * x + c_1))
        d_state_y = torch.zeros_like(y[:, 0]) + (epsilon * (r_3 / z - r_4 * x + c_2))
        d_z = torch.zeros_like(y[:, 0]) + (r_5 * (state_y - y_hat) * (z_hat - z) + r_6 * z * (z_hat - z) - r_7 * z)

        return torch.stack(
            [d_x, d_z, d_state_y],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
