import torch
import torch.nn as nn
import math


class ImmunologyMittlerSulzerNeumannPerelson1998ODE(nn.Module):
    """
    PyTorch translation of immunology/mittler_sulzer_neumann_perelson_1998/model.py.
    Dynamic states: 7; independent constants: 9; derived constants used by rates: 3.
    Original constant slots: 13; algebraic slots: 2.
    """
    source_file = 'immunology/mittler_sulzer_neumann_perelson_1998/model.py'
    source_state_dim = 7
    source_constant_dim = 13
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

        self.state_dim = 7
        self.parameter_dim = 9

        self.parameters = (
            parameters
            if parameters is not None
            else {'k': 2.4e-05, 'p': 774, 'c': 3, 'delta': 0.5, 'VI_0': 200000, 'tau_p': 0, 'b': 0.25, 'm': 0.01, 'n': 4}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'I': 0.1, 'E4': 0, 'VI': 200000, 'VNI': 0, 'E1': 0, 'E2': 0, 'E3': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k', 'p', 'c', 'delta', 'VI_0', 'tau_p', 'b', 'm', 'n']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['I', 'E4', 'VI', 'VNI', 'E1', 'E2', 'E3']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (7)
        I = y[:, 0]
        E4 = y[:, 1]
        VI = y[:, 2]
        VNI = y[:, 3]
        E1 = y[:, 4]
        E2 = y[:, 5]
        E3 = y[:, 6]

        # Independent parameters (source constant order)
        k = theta[:, 0]  # constants[0]
        p = theta[:, 1]  # constants[1]
        c = theta[:, 2]  # constants[2]
        delta = theta[:, 3]  # constants[3]
        VI_0 = theta[:, 4]  # constants[4]
        tau_p = theta[:, 5]  # constants[5]
        b = theta[:, 6]  # constants[6]
        m = theta[:, 7]  # constants[7]
        n = theta[:, 8]  # constants[8]

        # Derived constants from the original initConsts()
        T = c * delta / (k * p)  # constants[9]
        k_2 = k / (1.0 + m * b) ** n  # constants[11]
        b_2 = b / (1.0 + m * b)  # constants[12]

        # Initialize rates and algebraic variables, as in computeRates()
        d_I = torch.zeros_like(y[:, 0])
        d_E4 = torch.zeros_like(y[:, 0])
        d_VI = torch.zeros_like(y[:, 0])
        d_VNI = torch.zeros_like(y[:, 0])
        d_E1 = torch.zeros_like(y[:, 0])
        d_E2 = torch.zeros_like(y[:, 0])
        d_E3 = torch.zeros_like(y[:, 0])
        alg_0_h = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_I = torch.zeros_like(y[:, 0]) + (k_2 * T * E4 - delta * I)
        d_E1 = torch.zeros_like(y[:, 0]) + ((VI - E1) / b_2)
        d_E2 = torch.zeros_like(y[:, 0]) + ((E1 - E2) / b_2)
        d_E3 = torch.zeros_like(y[:, 0]) + ((E2 - E3) / b_2)
        d_E4 = torch.zeros_like(y[:, 0]) + ((E3 - E4) / b_2)
        alg_0_h = torch.where(t < tau_p, 0.0, 1.0)
        d_VI = torch.zeros_like(y[:, 0]) + ((1.0 - alg_0_h) * p * I - c * VI)
        d_VNI = torch.zeros_like(y[:, 0]) + (alg_0_h * p * I - c * VNI)

        return torch.stack(
            [d_I, d_E4, d_VI, d_VNI, d_E1, d_E2, d_E3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
