import torch
import torch.nn as nn
import math


class CellCycleHalloyBernardLoussouarnGoldbeter2002ODE(nn.Module):
    """
    PyTorch translation of cell_cycle/halloy_bernard_loussouarn_goldbeter_2002/model.py.
    Dynamic states: 4; independent constants: 4; derived constants used by rates: 0.
    Original constant slots: 7; algebraic slots: 0.
    """
    source_file = 'cell_cycle/halloy_bernard_loussouarn_goldbeter_2002/model.py'
    source_state_dim = 4
    source_constant_dim = 7
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

        self.state_dim = 4
        self.parameter_dim = 4

        self.parameters = (
            parameters
            if parameters is not None
            else {'mu_T': 2.4, 'mu_L': 2, 'mu_A': 3.6, 'epsilon': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'L': 0.33, 'M': 0, 'T': 0.33, 'A': 0.34}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['mu_T', 'mu_L', 'mu_A', 'epsilon']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['L', 'M', 'T', 'A']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        L = y[:, 0]
        M = y[:, 1]
        T = y[:, 2]
        A = y[:, 3]

        # Independent parameters (source constant order)
        mu_T = theta[:, 0]  # constants[0]
        mu_L = theta[:, 1]  # constants[1]
        mu_A = theta[:, 2]  # constants[2]
        epsilon = theta[:, 3]  # constants[3]

        # Initialize rates and algebraic variables, as in computeRates()
        d_L = torch.zeros_like(y[:, 0])
        d_M = torch.zeros_like(y[:, 0])
        d_T = torch.zeros_like(y[:, 0])
        d_A = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_L = torch.zeros_like(y[:, 0]) + (1.0 / mu_T * T - 1.0 / mu_L * L + epsilon * M)
        d_A = torch.zeros_like(y[:, 0]) + (1.0 / mu_L * L - 1.0 / mu_A * A)
        d_T = torch.zeros_like(y[:, 0]) + (1.0 / mu_A * A - 1.0 / mu_T * T)
        d_M = torch.zeros_like(y[:, 0]) + (epsilon * T)

        return torch.stack(
            [d_L, d_M, d_T, d_A],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
