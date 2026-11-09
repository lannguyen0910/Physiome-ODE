import torch
import torch.nn as nn
import math


class MyofilamentMechanicsCampbellChandra2006ODE(nn.Module):
    """
    PyTorch translation of myofilament_mechanics/campbell_chandra_2006/model.py.
    Dynamic states: 4; independent constants: 10; derived constants used by rates: 0.
    Original constant slots: 10; algebraic slots: 4.
    """
    source_file = 'myofilament_mechanics/campbell_chandra_2006/model.py'
    source_state_dim = 4
    source_constant_dim = 10
    algebraic_dim = 4
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
        self.parameter_dim = 10

        self.parameters = (
            parameters
            if parameters is not None
            else {'alpha': 0.1, 'k_a': 0, 'x_0': 1, 'epsilon': 1, 'beta': 2, 'g': 1, 'f': 1, 'k_off': 1, 'k_on': 1, 'L_0': 2.12}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'R_on': 1, 'A': 0, 'x': 1, 'L': 2.12}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['alpha', 'k_a', 'x_0', 'epsilon', 'beta', 'g', 'f', 'k_off', 'k_on', 'L_0']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['R_on', 'A', 'x', 'L']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        R_on = y[:, 0]
        A = y[:, 1]
        x = y[:, 2]
        L = y[:, 3]

        # Independent parameters (source constant order)
        alpha = theta[:, 0]  # constants[0]
        k_a = theta[:, 1]  # constants[1]
        x_0 = theta[:, 2]  # constants[2]
        epsilon = theta[:, 3]  # constants[3]
        beta = theta[:, 4]  # constants[4]
        g = theta[:, 5]  # constants[5]
        f = theta[:, 6]  # constants[6]
        k_off = theta[:, 7]  # constants[7]
        k_on = theta[:, 8]  # constants[8]
        L_0 = theta[:, 9]  # constants[9]

        # Initialize rates and algebraic variables, as in computeRates()
        d_R_on = torch.zeros_like(y[:, 0])
        d_A = torch.zeros_like(y[:, 0])
        d_x = torch.zeros_like(y[:, 0])
        d_L = torch.zeros_like(y[:, 0])
        alg_1_D = torch.zeros_like(y[:, 0])
        alg_2_dL_dt = torch.zeros_like(y[:, 0])
        alg_3_k_XB = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_D = 1.0 - A
        d_A = torch.zeros_like(y[:, 0]) + (f * alg_1_D * R_on - g * A)
        alg_2_dL_dt = torch.where((0.001 < t) & (t <= 0.003), 10.6, 0.0)
        d_x = torch.zeros_like(y[:, 0]) + (-g * (x - x_0) + alg_2_dL_dt)
        d_L = torch.zeros_like(y[:, 0]) + (alg_2_dL_dt)
        alg_3_k_XB = k_a * A
        d_R_on = torch.zeros_like(y[:, 0]) + (-((k_off + alg_3_k_XB + alpha * k_on) / (1.0 + alpha) + f * alg_1_D) * R_on + (g - (alg_3_k_XB + alpha * k_on) / (1.0 + alpha)) * A + (alg_3_k_XB + alpha * k_on / (1.0 + alpha)) * beta * (L - L_0))

        return torch.stack(
            [d_R_on, d_A, d_x, d_L],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
