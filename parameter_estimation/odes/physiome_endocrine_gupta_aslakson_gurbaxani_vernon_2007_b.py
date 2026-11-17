import torch
import torch.nn as nn
import math


class EndocrineGuptaAslaksonGurbaxaniVernon2007BODE(nn.Module):
    """
    PyTorch translation of endocrine/gupta_aslakson_gurbaxani_vernon_2007_b/model.py.
    Dynamic states: 4; independent constants: 7; derived constants used by rates: 0.
    Original constant slots: 7; algebraic slots: 1.
    """
    source_file = 'endocrine/gupta_aslakson_gurbaxani_vernon_2007_b/model.py'
    source_state_dim = 4
    source_constant_dim = 7
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
        self.parameter_dim = 7

        self.parameters = (
            parameters
            if parameters is not None
            else {'kcd': 1, 'ki1': 0.1, 'kad': 10, 'ki2': 0.1, 'kcr': 0.05, 'krd': 0.9, 'k': 0.001}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'c': 0.6, 'o': 0.055, 'a': 0.055, 'r': 0.01}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['kcd', 'ki1', 'kad', 'ki2', 'kcr', 'krd', 'k']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['c', 'o', 'a', 'r']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        c = y[:, 0]
        o = y[:, 1]
        a = y[:, 2]
        r = y[:, 3]

        # Independent parameters (source constant order)
        kcd = theta[:, 0]  # constants[0]
        ki1 = theta[:, 1]  # constants[1]
        kad = theta[:, 2]  # constants[2]
        ki2 = theta[:, 3]  # constants[3]
        kcr = theta[:, 4]  # constants[4]
        krd = theta[:, 5]  # constants[5]
        k = theta[:, 6]  # constants[6]

        # Initialize rates and algebraic variables, as in computeRates()
        d_c = torch.zeros_like(y[:, 0])
        d_o = torch.zeros_like(y[:, 0])
        d_a = torch.zeros_like(y[:, 0])
        d_r = torch.zeros_like(y[:, 0])
        alg_0_f = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_a = torch.zeros_like(y[:, 0]) + (c / (1.0 + o * r / ki2) - kad * a)
        d_r = torch.zeros_like(y[:, 0]) + ((o * r) ** 2.0 / (1.0 * (k + (o * r) ** 2.0)) + kcr - krd * r)
        d_o = torch.zeros_like(y[:, 0]) + (1.0 * (a - o))
        alg_0_f = torch.where((t > 0.0) & (t < 1.0) | (t > 4.0) & (t < 5.0) | (t > 8.0) & (t < 9.0) | (t > 12.0) & (t < 13.0) | (t > 16.0) & (t < 17.0), 1.0, 0.0)
        d_c = torch.zeros_like(y[:, 0]) + (1.0 * ((1.0 + alg_0_f) / (1.0 + o / ki1)) - kcd * c)

        return torch.stack(
            [d_c, d_o, d_a, d_r],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
