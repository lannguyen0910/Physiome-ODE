import torch
import torch.nn as nn
import math


class CalciumDynamicsBaylorHollingworthChandler2002EODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/baylor_hollingworth_chandler_2002_e/model.py.
    Dynamic states: 3; independent constants: 2; derived constants used by rates: 0.
    Original constant slots: 2; algebraic slots: 0.
    """
    source_file = 'calcium_dynamics/baylor_hollingworth_chandler_2002_e/model.py'
    source_state_dim = 3
    source_constant_dim = 2
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
        self.parameter_dim = 2

        self.parameters = (
            parameters
            if parameters is not None
            else {'k1': 101700000000000, 'k1_2': 132}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Ca': 0.05, 'Trop': 360, 'CaTrop': 0.01}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k1', 'k1_2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Ca', 'Trop', 'CaTrop']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        Ca = y[:, 0]
        Trop = y[:, 1]
        CaTrop = y[:, 2]

        # Independent parameters (source constant order)
        k1 = theta[:, 0]  # constants[0]
        k1_2 = theta[:, 1]  # constants[1]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Ca = torch.zeros_like(y[:, 0])
        d_Trop = torch.zeros_like(y[:, 0])
        d_CaTrop = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Ca = torch.zeros_like(y[:, 0]) + (k1_2 * CaTrop - k1 * Ca * Trop)
        d_Trop = torch.zeros_like(y[:, 0]) + (k1_2 * CaTrop - k1 * Ca * Trop)
        d_CaTrop = torch.zeros_like(y[:, 0]) + (k1 * Ca * Trop - k1_2 * CaTrop)

        return torch.stack(
            [d_Ca, d_Trop, d_CaTrop],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
