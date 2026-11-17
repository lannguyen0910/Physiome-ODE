import torch
import torch.nn as nn
import math


class CalciumDynamicsSneyd1995ODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/sneyd_1995/model.py.
    Dynamic states: 3; independent constants: 12; derived constants used by rates: 1.
    Original constant slots: 13; algebraic slots: 3.
    """
    source_file = 'calcium_dynamics/sneyd_1995/model.py'
    source_state_dim = 3
    source_constant_dim = 13
    algebraic_dim = 3
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
        self.parameter_dim = 12

        self.parameters = (
            parameters
            if parameters is not None
            else {'V_p': 0.08, 'k_p': 1, 'IPR_3_flux': 0.72, 'k_flux': 3, 'b': 0.11, 'k_1': 0.7, 'gamma': 1, 'k_gamma': 0.27, 'beta': 0.15, 'k_mu': 0.01, 'k_2': 0.7, 'tau_h': 0.2}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'P': 0, 'c': 0.3, 'h': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['V_p', 'k_p', 'IPR_3_flux', 'k_flux', 'b', 'k_1', 'gamma', 'k_gamma', 'beta', 'k_mu', 'k_2', 'tau_h']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['P', 'c', 'h']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        P = y[:, 0]
        c = y[:, 1]
        h = y[:, 2]

        # Independent parameters (source constant order)
        V_p = theta[:, 0]  # constants[0]
        k_p = theta[:, 1]  # constants[1]
        IPR_3_flux = theta[:, 2]  # constants[2]
        k_flux = theta[:, 3]  # constants[3]
        b = theta[:, 4]  # constants[4]
        k_1 = theta[:, 5]  # constants[5]
        gamma = theta[:, 6]  # constants[6]
        k_gamma = theta[:, 7]  # constants[7]
        beta = theta[:, 8]  # constants[8]
        k_mu = theta[:, 9]  # constants[9]
        k_2 = theta[:, 10]  # constants[10]
        tau_h = theta[:, 11]  # constants[11]

        # Derived constants from the original initConsts()
        J_leak = beta  # constants[12]

        # Initialize rates and algebraic variables, as in computeRates()
        d_P = torch.zeros_like(y[:, 0])
        d_c = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        alg_0_J_pump = torch.zeros_like(y[:, 0])
        alg_1_mu = torch.zeros_like(y[:, 0])
        alg_2_J_flux = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_P = torch.zeros_like(y[:, 0]) + (torch.where(t <= 15.0, IPR_3_flux - V_p * P * k_p / (k_p + P), -V_p * P * k_p / (k_p + P)))
        d_h = torch.zeros_like(y[:, 0]) + ((k_2 ** 2.0 / (k_2 ** 2.0 + c ** 2.0) - h) / tau_h)
        alg_1_mu = P ** 3.0 / (k_mu ** 3.0 + P ** 3.0)
        alg_2_J_flux = k_flux * alg_1_mu * h * (b + (1.0 - b) * c / (k_1 + c))
        alg_0_J_pump = gamma * c ** 2.0 / (k_gamma ** 2.0 + c ** 2.0)
        d_c = torch.zeros_like(y[:, 0]) + (alg_2_J_flux - alg_0_J_pump + J_leak)

        return torch.stack(
            [d_P, d_c, d_h],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
