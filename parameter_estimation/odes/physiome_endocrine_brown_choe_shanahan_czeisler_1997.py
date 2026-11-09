import torch
import torch.nn as nn
import math


class EndocrineBrownChoeShanahanCzeisler1997ODE(nn.Module):
    """
    PyTorch translation of endocrine/brown_choe_shanahan_czeisler_1997/model.py.
    Dynamic states: 2; independent constants: 7; derived constants used by rates: 4.
    Original constant slots: 11; algebraic slots: 1.
    """
    source_file = 'endocrine/brown_choe_shanahan_czeisler_1997/model.py'
    source_state_dim = 2
    source_constant_dim = 11
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

        self.state_dim = 2
        self.parameter_dim = 7

        self.parameters = (
            parameters
            if parameters is not None
            else {'t_on': 1316, 't_off': 1792, 'A_max': 6.51, 'tau_I': 2.82, 'tau_C': 23.67, 'tau_alpha': 25.92, 'tau_lamda': 24.04}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'H1': 0.05, 'H2': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['t_on', 't_off', 'A_max', 'tau_I', 'tau_C', 'tau_alpha', 'tau_lamda']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['H1', 'H2']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        H1 = y[:, 0]
        H2 = y[:, 1]

        # Independent parameters (source constant order)
        t_on = theta[:, 0]  # constants[0]
        t_off = theta[:, 1]  # constants[1]
        A_max = theta[:, 2]  # constants[2]
        tau_I = theta[:, 3]  # constants[3]
        tau_C = theta[:, 4]  # constants[4]
        tau_alpha = theta[:, 5]  # constants[5]
        tau_lamda = theta[:, 6]  # constants[6]

        # Derived constants from the original initConsts()
        beta_I = 0.6931471805599453 / tau_I  # constants[7]
        beta_C = 0.6931471805599453 / tau_C  # constants[8]
        alpha = 0.6931471805599453 / tau_alpha  # constants[9]
        lamda = 0.6931471805599453 / tau_lamda  # constants[10]

        # Initialize rates and algebraic variables, as in computeRates()
        d_H1 = torch.zeros_like(y[:, 0])
        d_H2 = torch.zeros_like(y[:, 0])
        alg_0_A = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_H2 = torch.zeros_like(y[:, 0]) + (beta_I * H1 - beta_C * H2)
        alg_0_A = torch.where((t < t_off) & (t >= t_on), A_max * ((1.0 - torch.exp(-lamda * (t - t_on))) / (1.0 - torch.exp(-lamda * (t_off - t_on)))), torch.where(t >= t_off, A_max * torch.exp(-alpha * (t - t_off)), 0.0))
        d_H1 = torch.zeros_like(y[:, 0]) + (-(beta_I * H1) + alg_0_A)

        return torch.stack(
            [d_H1, d_H2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
