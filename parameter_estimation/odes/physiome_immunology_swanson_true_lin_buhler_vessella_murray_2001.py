import torch
import torch.nn as nn
import math


class ImmunologySwansonTrueLinBuhlerVessellaMurray2001ODE(nn.Module):
    """
    PyTorch translation of immunology/swanson_true_lin_buhler_vessella_murray_2001/model.py.
    Dynamic states: 1; independent constants: 6; derived constants used by rates: 0.
    Original constant slots: 6; algebraic slots: 1.
    """
    source_file = 'immunology/swanson_true_lin_buhler_vessella_murray_2001/model.py'
    source_state_dim = 1
    source_constant_dim = 6
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

        self.state_dim = 1
        self.parameter_dim = 6

        self.parameters = (
            parameters
            if parameters is not None
            else {'Vo': 20, 'Vh': 0, 'beta_h': 0, 'beta_c': 1.721, 'gamma': 1.2896, 'rho': 0.0655}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'p': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Vo', 'Vh', 'beta_h', 'beta_c', 'gamma', 'rho']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['p']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (1)
        p = y[:, 0]

        # Independent parameters (source constant order)
        Vo = theta[:, 0]  # constants[0]
        Vh = theta[:, 1]  # constants[1]
        beta_h = theta[:, 2]  # constants[2]
        beta_c = theta[:, 3]  # constants[3]
        gamma = theta[:, 4]  # constants[4]
        rho = theta[:, 5]  # constants[5]

        # Initialize rates and algebraic variables, as in computeRates()
        d_p = torch.zeros_like(y[:, 0])
        alg_0_Vc = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_Vc = Vo * torch.exp(rho * t)
        d_p = torch.zeros_like(y[:, 0]) + (1.0 * beta_h * Vh + 1.0 * beta_c * alg_0_Vc - gamma * p)

        return torch.stack(
            [d_p],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
