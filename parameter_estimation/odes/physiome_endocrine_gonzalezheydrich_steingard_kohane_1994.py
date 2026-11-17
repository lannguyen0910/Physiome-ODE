import torch
import torch.nn as nn
import math


class EndocrineGonzalezheydrichSteingardKohane1994ODE(nn.Module):
    """
    PyTorch translation of endocrine/gonzalezheydrich_steingard_kohane_1994/model.py.
    Dynamic states: 3; independent constants: 11; derived constants used by rates: 0.
    Original constant slots: 11; algebraic slots: 0.
    """
    source_file = 'endocrine/gonzalezheydrich_steingard_kohane_1994/model.py'
    source_state_dim = 3
    source_constant_dim = 11
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
        self.parameter_dim = 11

        self.parameters = (
            parameters
            if parameters is not None
            else {'k1': 500000, 'k2': 0.01, 'k3': 0.01, 'k4': 10, 'k5': 0.004, 'k6': 0.035, 'Kd': 0.004, 'Imax': 0.99, 'k7': 1e-06, 'k8': 0.01, 'pulse': 50}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'cortisol': 0, 'ACTH': 0, 'CRH': 50}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'Kd', 'Imax', 'k7', 'k8', 'pulse']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['cortisol', 'ACTH', 'CRH']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        cortisol = y[:, 0]
        ACTH = y[:, 1]
        CRH = y[:, 2]

        # Independent parameters (source constant order)
        k1 = theta[:, 0]  # constants[0]
        k2 = theta[:, 1]  # constants[1]
        k3 = theta[:, 2]  # constants[2]
        k4 = theta[:, 3]  # constants[3]
        k5 = theta[:, 4]  # constants[4]
        k6 = theta[:, 5]  # constants[5]
        Kd = theta[:, 6]  # constants[6]
        Imax = theta[:, 7]  # constants[7]
        k7 = theta[:, 8]  # constants[8]
        k8 = theta[:, 9]  # constants[9]
        pulse = theta[:, 10]  # constants[10]

        # Initialize rates and algebraic variables, as in computeRates()
        d_cortisol = torch.zeros_like(y[:, 0])
        d_ACTH = torch.zeros_like(y[:, 0])
        d_CRH = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_cortisol = torch.zeros_like(y[:, 0]) + (k1 * ACTH + k2 - k3 * cortisol)
        d_ACTH = torch.zeros_like(y[:, 0]) + (k4 * CRH + k5 - (k6 * ACTH + (k4 * CRH + k5) * (Imax * cortisol / (Kd + cortisol))))
        d_CRH = torch.zeros_like(y[:, 0]) + (pulse + k7 - k8 * CRH)

        return torch.stack(
            [d_cortisol, d_ACTH, d_CRH],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
