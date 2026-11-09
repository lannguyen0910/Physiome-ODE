import torch
import torch.nn as nn
import math


class CellCycleYatesStarkKleinAntiaCallard2007AODE(nn.Module):
    """
    PyTorch translation of cell_cycle/yates_stark_klein_antia_callard_2007_a/model.py.
    Dynamic states: 2; independent constants: 4; derived constants used by rates: 1.
    Original constant slots: 5; algebraic slots: 0.
    """
    source_file = 'cell_cycle/yates_stark_klein_antia_callard_2007_a/model.py'
    source_state_dim = 2
    source_constant_dim = 5
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

        self.state_dim = 2
        self.parameter_dim = 4

        self.parameters = (
            parameters
            if parameters is not None
            else {'d1': 0.005, 'a': 0.03333, 'r': 1, 'kappa': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'x': 1, 'y': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['d1', 'a', 'r', 'kappa']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['x', 'y']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        x = y[:, 0]
        state_y = y[:, 1]

        # Independent parameters (source constant order)
        d1 = theta[:, 0]  # constants[0]
        a = theta[:, 1]  # constants[1]
        r = theta[:, 2]  # constants[2]
        kappa = theta[:, 3]  # constants[3]

        # Derived constants from the original initConsts()
        d2 = (-(99.0 * a * d1) + a * r + d1 * r) / (a - d1)  # constants[4]

        # Initialize rates and algebraic variables, as in computeRates()
        d_x = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_x = torch.zeros_like(y[:, 0]) + (2.0 * r * state_y - (a * x * (1.0 - x / kappa) + d1 * x * (x / kappa)))
        d_state_y = torch.zeros_like(y[:, 0]) + (a * x * (1.0 - x / kappa) - (r + d2) * state_y)

        return torch.stack(
            [d_x, d_state_y],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
