import torch
import torch.nn as nn
import math


class EndocrineRattananakulLenburyKrishnamaraWollkind2003AODE(nn.Module):
    """
    PyTorch translation of endocrine/rattananakul_lenbury_krishnamara_wollkind_2003_a/model.py.
    Dynamic states: 3; independent constants: 13; derived constants used by rates: 0.
    Original constant slots: 13; algebraic slots: 0.
    """
    source_file = 'endocrine/rattananakul_lenbury_krishnamara_wollkind_2003_a/model.py'
    source_state_dim = 3
    source_constant_dim = 13
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
        self.parameter_dim = 13

        self.parameters = (
            parameters
            if parameters is not None
            else {'a1': 0.05, 'k1': 0.1, 'b1': 0.1, 'epsilon': 0.1, 'a2': 0.009, 'a3': 0.675, 'k2': 0.5, 'b2': 0.3, 'delta': 0.9, 'a4': 0.01, 'a5': 0.005, 'k3': 0.025, 'b3': 0.01}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'x': 2, 'y': 1, 'z': 0.15}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['a1', 'k1', 'b1', 'epsilon', 'a2', 'a3', 'k2', 'b2', 'delta', 'a4', 'a5', 'k3', 'b3']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['x', 'y', 'z']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        x = y[:, 0]
        state_y = y[:, 1]
        z = y[:, 2]

        # Independent parameters (source constant order)
        a1 = theta[:, 0]  # constants[0]
        k1 = theta[:, 1]  # constants[1]
        b1 = theta[:, 2]  # constants[2]
        epsilon = theta[:, 3]  # constants[3]
        a2 = theta[:, 4]  # constants[4]
        a3 = theta[:, 5]  # constants[5]
        k2 = theta[:, 6]  # constants[6]
        b2 = theta[:, 7]  # constants[7]
        delta = theta[:, 8]  # constants[8]
        a4 = theta[:, 9]  # constants[9]
        a5 = theta[:, 10]  # constants[10]
        k3 = theta[:, 11]  # constants[11]
        b3 = theta[:, 12]  # constants[12]

        # Initialize rates and algebraic variables, as in computeRates()
        d_x = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])
        d_z = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_x = torch.zeros_like(y[:, 0]) + (a1 / (k1 + state_y) - b1 * x)
        d_state_y = torch.zeros_like(y[:, 0]) + (epsilon * ((a2 + a3 * x) * state_y * z / (k2 + x ** 2.0) - b2 * state_y))
        d_z = torch.zeros_like(y[:, 0]) + (epsilon * delta * (a4 * x - (a5 * x * z / (k3 + x) + b3 * z)))

        return torch.stack(
            [d_x, d_state_y, d_z],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
