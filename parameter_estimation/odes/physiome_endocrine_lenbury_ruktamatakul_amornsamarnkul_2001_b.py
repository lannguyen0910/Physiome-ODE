import torch
import torch.nn as nn
import math


class EndocrineLenburyRuktamatakulAmornsamarnkul2001BODE(nn.Module):
    """
    PyTorch translation of endocrine/lenbury_ruktamatakul_amornsamarnkul_2001_b/model.py.
    Dynamic states: 5; independent constants: 15; derived constants used by rates: 0.
    Original constant slots: 15; algebraic slots: 0.
    """
    source_file = 'endocrine/lenbury_ruktamatakul_amornsamarnkul_2001_b/model.py'
    source_state_dim = 5
    source_constant_dim = 15
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

        self.state_dim = 5
        self.parameter_dim = 15

        self.parameters = (
            parameters
            if parameters is not None
            else {'r1': 0.15, 'r2': 0.12, 'c1': 0.1, 'r3': 0.05, 'r4': 0.03, 'c2': 0.1, 'c3': 0.005, 'epsilon': 0.1, 'r5': 0.09, 'r6': 0.1, 'r7': 0.05, 'z': 1.01, 'y': 1.08, 'delta': 0.01, 'omega': 0.05}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'x': 5, 'z': 1, 'y': 0, 'u': 1, 'v': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['r1', 'r2', 'c1', 'r3', 'r4', 'c2', 'c3', 'epsilon', 'r5', 'r6', 'r7', 'z', 'y', 'delta', 'omega']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['x', 'z', 'y', 'u', 'v']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        x = y[:, 0]
        z = y[:, 1]
        state_y = y[:, 2]
        u = y[:, 3]
        v = y[:, 4]

        # Independent parameters (source constant order)
        r1 = theta[:, 0]  # constants[0]
        r2 = theta[:, 1]  # constants[1]
        c1 = theta[:, 2]  # constants[2]
        r3 = theta[:, 3]  # constants[3]
        r4 = theta[:, 4]  # constants[4]
        c2 = theta[:, 5]  # constants[5]
        c3 = theta[:, 6]  # constants[6]
        epsilon = theta[:, 7]  # constants[7]
        r5 = theta[:, 8]  # constants[8]
        r6 = theta[:, 9]  # constants[9]
        r7 = theta[:, 10]  # constants[10]
        const_z = theta[:, 11]  # constants[11]
        const_y = theta[:, 12]  # constants[12]
        delta = theta[:, 13]  # constants[13]
        omega = theta[:, 14]  # constants[14]

        # Initialize rates and algebraic variables, as in computeRates()
        d_x = torch.zeros_like(y[:, 0])
        d_z = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])
        d_u = torch.zeros_like(y[:, 0])
        d_v = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_x = torch.zeros_like(y[:, 0]) + (z * (r1 * state_y + -r2 * x + c1))
        d_state_y = torch.zeros_like(y[:, 0]) + (epsilon * (r3 / z + -r4 * x + c2 + c3 * u))
        d_z = torch.zeros_like(y[:, 0]) + (epsilon * delta * (r5 * (state_y - const_y) * (const_z - z) + r6 * z * (const_z - z) - r7 * z))
        d_u = torch.zeros_like(y[:, 0]) + (-omega * v)
        d_v = torch.zeros_like(y[:, 0]) + (omega * u)

        return torch.stack(
            [d_x, d_z, d_state_y, d_u, d_v],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
