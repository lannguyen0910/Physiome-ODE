import torch
import torch.nn as nn
import math


class SignalTransductionWanantQuon2000AODE(nn.Module):
    """
    PyTorch translation of signal_transduction/wanant_quon_2000_a/model.py.
    Dynamic states: 4; independent constants: 4; derived constants used by rates: 0.
    Original constant slots: 4; algebraic slots: 1.
    """
    source_file = 'signal_transduction/wanant_quon_2000_a/model.py'
    source_state_dim = 4
    source_constant_dim = 4
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
        self.parameter_dim = 4

        self.parameters = (
            parameters
            if parameters is not None
            else {'k1': 1000000, 'k1_2': 0.0004, 'k2': 0.04, 'k2_2': 1000000}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'x1': 1000, 'x2': 0.1, 'x3': 0, 'x4': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k1', 'k1_2', 'k2', 'k2_2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['x1', 'x2', 'x3', 'x4']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        x1 = y[:, 0]
        x2 = y[:, 1]
        x3 = y[:, 2]
        x4 = y[:, 3]

        # Independent parameters (source constant order)
        k1 = theta[:, 0]  # constants[0]
        k1_2 = theta[:, 1]  # constants[1]
        k2 = theta[:, 2]  # constants[2]
        k2_2 = theta[:, 3]  # constants[3]

        # Initialize rates and algebraic variables, as in computeRates()
        d_x1 = torch.zeros_like(y[:, 0])
        d_x2 = torch.zeros_like(y[:, 0])
        d_x3 = torch.zeros_like(y[:, 0])
        d_x4 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_x1 = torch.zeros_like(y[:, 0]) + (k1_2 * x3 - k1 * x1 * x2 + k2 * x4 - k2_2 * x1 * x3)
        d_x2 = torch.zeros_like(y[:, 0]) + (k1_2 * x3 - k1 * x1 * x2)
        d_x3 = torch.zeros_like(y[:, 0]) + (k1 * x1 * x2 - k1_2 * x3 + k2 * x4 - k2_2 * x1 * x3)
        d_x4 = torch.zeros_like(y[:, 0]) + (k2_2 * x1 * x3 - k2 * x4)

        return torch.stack(
            [d_x1, d_x2, d_x3, d_x4],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
