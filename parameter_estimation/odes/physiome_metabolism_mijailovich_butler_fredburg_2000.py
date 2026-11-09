import torch
import torch.nn as nn
import math


class MetabolismMijailovichButlerFredburg2000ODE(nn.Module):
    """
    PyTorch translation of metabolism/mijailovich_butler_fredburg_2000/model.py.
    Dynamic states: 4; independent constants: 8; derived constants used by rates: 0.
    Original constant slots: 12; algebraic slots: 2.
    """
    source_file = 'metabolism/mijailovich_butler_fredburg_2000/model.py'
    source_state_dim = 4
    source_constant_dim = 12
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

        self.state_dim = 4
        self.parameter_dim = 8

        self.parameters = (
            parameters
            if parameters is not None
            else {'k2': 0.1, 'gx': 0.11, 'fp': 0.44, 'gp': 0.11, 'k5': 0.1, 'fp1': 0.88, 'gp1': 0.22, 'g1': 0.01}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'M': 1, 'AM': 0, 'Mp': 0, 'AMp': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k2', 'gx', 'fp', 'gp', 'k5', 'fp1', 'gp1', 'g1']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['M', 'AM', 'Mp', 'AMp']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        M = y[:, 0]
        AM = y[:, 1]
        Mp = y[:, 2]
        AMp = y[:, 3]

        # Independent parameters (source constant order)
        k2 = theta[:, 0]  # constants[0]
        gx = theta[:, 1]  # constants[1]
        fp = theta[:, 2]  # constants[2]
        gp = theta[:, 3]  # constants[3]
        k5 = theta[:, 4]  # constants[4]
        fp1 = theta[:, 5]  # constants[5]
        gp1 = theta[:, 6]  # constants[6]
        g1 = theta[:, 7]  # constants[7]

        # Initialize rates and algebraic variables, as in computeRates()
        d_M = torch.zeros_like(y[:, 0])
        d_AM = torch.zeros_like(y[:, 0])
        d_Mp = torch.zeros_like(y[:, 0])
        d_AMp = torch.zeros_like(y[:, 0])
        alg_0_k1 = torch.zeros_like(y[:, 0])
        alg_1_k6 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_k1 = torch.where((t > 0.0) & (t < 5.0), 0.35, 0.06)
        d_M = torch.zeros_like(y[:, 0]) + (-(alg_0_k1 * M) + k2 * Mp + gx * AM)
        d_Mp = torch.zeros_like(y[:, 0]) + (gp * AMp + alg_0_k1 * M - (k2 + fp) * Mp)
        alg_1_k6 = alg_0_k1
        d_AMp = torch.zeros_like(y[:, 0]) + (fp * Mp + alg_1_k6 * AM - (k5 + gp) * AMp)
        d_AM = torch.zeros_like(y[:, 0]) + (k5 * AMp - (alg_1_k6 + gx) * AM)

        return torch.stack(
            [d_M, d_AM, d_Mp, d_AMp],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
