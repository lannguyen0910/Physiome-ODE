import torch
import torch.nn as nn
import math


class ImmunologySchenzle1994ODE(nn.Module):
    """
    PyTorch translation of immunology/schenzle_1994/model.py.
    Dynamic states: 5; independent constants: 14; derived constants used by rates: 1.
    Original constant slots: 15; algebraic slots: 2.
    """
    source_file = 'immunology/schenzle_1994/model.py'
    source_state_dim = 5
    source_constant_dim = 15
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

        self.state_dim = 5
        self.parameter_dim = 14

        self.parameters = (
            parameters
            if parameters is not None
            else {'mu0': 0.004, 'mu1': 0.3, 'delta1': 0.3, 'mu2': 1, 'phi': 1000, 'delta2': 1, 'alpha': 1e-06, 'beta': 0.5, 'gamma': 0.5, 'a': 1000, 'b': 0.2, 'X0': 100000000000, 'omega': 1.6e-16, 'Kmax': 2.6999999999999996e-13}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'X': 100000000000, 'K': 1.35e-14, 'V': 1000, 'Y': 100, 'Z': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['mu0', 'mu1', 'delta1', 'mu2', 'phi', 'delta2', 'alpha', 'beta', 'gamma', 'a', 'b', 'X0', 'omega', 'Kmax']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['X', 'K', 'V', 'Y', 'Z']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        X = y[:, 0]
        K = y[:, 1]
        V = y[:, 2]
        Y = y[:, 3]
        Z = y[:, 4]

        # Independent parameters (source constant order)
        mu0 = theta[:, 0]  # constants[0]
        mu1 = theta[:, 1]  # constants[1]
        delta1 = theta[:, 2]  # constants[2]
        mu2 = theta[:, 3]  # constants[3]
        phi = theta[:, 4]  # constants[4]
        delta2 = theta[:, 5]  # constants[5]
        alpha = theta[:, 6]  # constants[6]
        beta = theta[:, 7]  # constants[7]
        gamma = theta[:, 8]  # constants[8]
        a = theta[:, 9]  # constants[9]
        b = theta[:, 10]  # constants[10]
        X0 = theta[:, 11]  # constants[11]
        omega = theta[:, 12]  # constants[12]
        Kmax = theta[:, 13]  # constants[14]

        # Derived constants from the original initConsts()
        s = mu0 * 100000000000.0  # constants[13]

        # Initialize rates and algebraic variables, as in computeRates()
        d_X = torch.zeros_like(y[:, 0])
        d_K = torch.zeros_like(y[:, 0])
        d_V = torch.zeros_like(y[:, 0])
        d_Y = torch.zeros_like(y[:, 0])
        d_Z = torch.zeros_like(y[:, 0])
        alg_1_f_X = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_X = torch.zeros_like(y[:, 0]) + (s - (mu0 * X + K * V * X))
        d_Y = torch.zeros_like(y[:, 0]) + (K * V * Y - (mu1 + delta1 * Z) * Y)
        d_V = torch.zeros_like(y[:, 0]) + (phi * Y - (mu2 + delta2 * Z) * V)
        d_K = torch.zeros_like(y[:, 0]) + (omega * V * (Kmax - K))
        alg_1_f_X = (X / X0) ** 2.0 / (b + (X / X0) ** 2.0)
        d_Z = torch.zeros_like(y[:, 0]) + (alpha * (V / (a + V)) + beta * alg_1_f_X * Z - gamma * Z ** 2.0)

        return torch.stack(
            [d_X, d_K, d_V, d_Y, d_Z],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
