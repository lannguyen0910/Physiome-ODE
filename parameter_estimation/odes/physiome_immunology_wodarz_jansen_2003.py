import torch
import torch.nn as nn
import math


class ImmunologyWodarzJansen2003ODE(nn.Module):
    """
    PyTorch translation of immunology/wodarz_jansen_2003/model.py.
    Dynamic states: 4; independent constants: 12; derived constants used by rates: 0.
    Original constant slots: 12; algebraic slots: 1.
    """
    source_file = 'immunology/wodarz_jansen_2003/model.py'
    source_state_dim = 4
    source_constant_dim = 12
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
        self.parameter_dim = 12

        self.parameters = (
            parameters
            if parameters is not None
            else {'k': 10, 'r': 0.5, 'd': 0.1, 'gamma': 1, 'lambda_var': 1, 'delta_1': 0.1, 'alpha': 0.05, 'delta_2': 1.5, 'eta': 2, 'epsilon': 1, 'q': 0.5, 'mu': 0.1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'T': 0.1, 'C': 0.015, 'A': 1, 'A_star': 2}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k', 'r', 'd', 'gamma', 'lambda_var', 'delta_1', 'alpha', 'delta_2', 'eta', 'epsilon', 'q', 'mu']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['T', 'C', 'A', 'A_star']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        T = y[:, 0]
        C = y[:, 1]
        A = y[:, 2]
        A_star = y[:, 3]

        # Independent parameters (source constant order)
        k = theta[:, 0]  # constants[0]
        r = theta[:, 1]  # constants[1]
        d = theta[:, 2]  # constants[2]
        gamma = theta[:, 3]  # constants[3]
        lambda_var = theta[:, 4]  # constants[4]
        delta_1 = theta[:, 5]  # constants[5]
        alpha = theta[:, 6]  # constants[6]
        delta_2 = theta[:, 7]  # constants[7]
        eta = theta[:, 8]  # constants[8]
        epsilon = theta[:, 9]  # constants[9]
        q = theta[:, 10]  # constants[10]
        mu = theta[:, 11]  # constants[11]

        # Initialize rates and algebraic variables, as in computeRates()
        d_T = torch.zeros_like(y[:, 0])
        d_C = torch.zeros_like(y[:, 0])
        d_A = torch.zeros_like(y[:, 0])
        d_A_star = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_T = torch.zeros_like(y[:, 0]) + (r * T * (1.0 - T * 1.0 / k) - d * T - gamma * T * C)
        d_A = torch.zeros_like(y[:, 0]) + (lambda_var - delta_1 * A - alpha * A * T)
        d_A_star = torch.zeros_like(y[:, 0]) + (alpha * A * T - delta_2 * A_star)
        d_C = torch.zeros_like(y[:, 0]) + (eta * A_star * C / (epsilon * C + 1.0) - q * T * C - mu * C)

        return torch.stack(
            [d_T, d_C, d_A, d_A_star],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
