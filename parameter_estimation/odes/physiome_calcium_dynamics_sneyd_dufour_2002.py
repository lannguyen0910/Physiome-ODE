import torch
import torch.nn as nn
import math


class CalciumDynamicsSneydDufour2002ODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/sneyd_dufour_2002/model.py.
    Dynamic states: 5; independent constants: 19; derived constants used by rates: 7.
    Original constant slots: 26; algebraic slots: 2.
    """
    source_file = 'calcium_dynamics/sneyd_dufour_2002/model.py'
    source_state_dim = 5
    source_constant_dim = 26
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
        self.parameter_dim = 19

        self.parameters = (
            parameters
            if parameters is not None
            else {'k_1b': 0.04, 'l_2b': 0.8, 'p': 10, 'k_3b': 29.8, 'k_1a': 0.64, 'k_2a': 37.4, 'k_2b': 1.4, 'k_3a': 0.11, 'k_4a': 4, 'k_4b': 0.54, 'l_2a': 1.7, 'l_4a': 1.7, 'l_4b': 2.5, 'l_6a': 4707, 'l_6b': 11.4, 'L_1': 0.12, 'L_3': 0.025, 'L_5': 54.7, 'c': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'R': 1, 'I_1': 0, 'O': 0, 'A': 0, 'I_2': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k_1b', 'l_2b', 'p', 'k_3b', 'k_1a', 'k_2a', 'k_2b', 'k_3a', 'k_4a', 'k_4b', 'l_2a', 'l_4a', 'l_4b', 'l_6a', 'l_6b', 'L_1', 'L_3', 'L_5', 'c']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['R', 'I_1', 'O', 'A', 'I_2']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        R = y[:, 0]
        I_1 = y[:, 1]
        O = y[:, 2]
        A = y[:, 3]
        I_2 = y[:, 4]

        # Independent parameters (source constant order)
        k_1b = theta[:, 0]  # constants[0]
        l_2b = theta[:, 1]  # constants[1]
        p = theta[:, 2]  # constants[2]
        k_3b = theta[:, 3]  # constants[3]
        k_1a = theta[:, 4]  # constants[4]
        k_2a = theta[:, 5]  # constants[5]
        k_2b = theta[:, 6]  # constants[6]
        k_3a = theta[:, 7]  # constants[7]
        k_4a = theta[:, 8]  # constants[8]
        k_4b = theta[:, 9]  # constants[9]
        l_2a = theta[:, 10]  # constants[10]
        l_4a = theta[:, 11]  # constants[11]
        l_4b = theta[:, 12]  # constants[12]
        l_6a = theta[:, 13]  # constants[13]
        l_6b = theta[:, 14]  # constants[14]
        L_1 = theta[:, 15]  # constants[15]
        L_3 = theta[:, 16]  # constants[16]
        L_5 = theta[:, 17]  # constants[17]
        c = theta[:, 18]  # constants[18]

        # Derived constants from the original initConsts()
        phi_1 = (k_1a * L_1 + l_2a) * c / (L_1 + c * (1.0 + L_1 / L_3))  # constants[19]
        phi_2 = (k_2a * L_3 + l_4a * c) / (L_3 + c * (1.0 + L_3 / L_1))  # constants[20]
        phi_2b = (k_2b + l_4b * c) / (1.0 + c / L_5)  # constants[21]
        phi_3 = k_3a * L_5 / (c + L_5)  # constants[22]
        phi_4 = (k_4a * L_5 + l_6a) * c / (c + L_5)  # constants[23]
        phi_4b = L_1 * (k_4b + l_6b) / (c + L_1)  # constants[24]
        phi_5 = (k_1a * L_1 + l_2a) * c / (c + L_1)  # constants[25]

        # Initialize rates and algebraic variables, as in computeRates()
        d_R = torch.zeros_like(y[:, 0])
        d_I_1 = torch.zeros_like(y[:, 0])
        d_O = torch.zeros_like(y[:, 0])
        d_A = torch.zeros_like(y[:, 0])
        d_I_2 = torch.zeros_like(y[:, 0])
        alg_0_S = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_R = torch.zeros_like(y[:, 0]) + (phi_2b * O + (k_1b + l_2b) * I_1 - (phi_2 * p * R + phi_1 * R))
        d_I_1 = torch.zeros_like(y[:, 0]) + (phi_1 * R - (k_1b + l_2b) * I_1)
        d_I_2 = torch.zeros_like(y[:, 0]) + (phi_5 * A - (k_1b + l_2b) * I_2)
        d_A = torch.zeros_like(y[:, 0]) + (phi_4 * O + (k_1b + l_2b) * I_2 - (phi_4b * A + phi_5 * A))
        alg_0_S = 1.0 - (R + O + A + I_1 + I_2)
        d_O = torch.zeros_like(y[:, 0]) + (phi_2 * p * R + phi_4b * A + k_3b * alg_0_S - (phi_2b + phi_4 + 1.0 * phi_3) * O)

        return torch.stack(
            [d_R, d_I_1, d_O, d_A, d_I_2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
