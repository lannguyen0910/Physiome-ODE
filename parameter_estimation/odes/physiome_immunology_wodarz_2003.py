import torch
import torch.nn as nn
import math


class ImmunologyWodarz2003ODE(nn.Module):
    """
    PyTorch translation of immunology/wodarz_2003/model.py.
    Dynamic states: 10; independent constants: 13; derived constants used by rates: 0.
    Original constant slots: 14; algebraic slots: 4.
    """
    source_file = 'immunology/wodarz_2003/model.py'
    source_state_dim = 10
    source_constant_dim = 14
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

        self.state_dim = 10
        self.parameter_dim = 13

        self.parameters = (
            parameters
            if parameters is not None
            else {'r': 0.5, 'epsilon': 0.1, 'd': 0.01, 'g': 0.01, 'beta_1': 1, 'beta_2': 1, 'a_1': 0.03, 'alpha_1': 0.1, 'a_2': 1, 'alpha_2': 0.1, 'k_1': 1, 'u': 0.5, 'k_2': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'S': 100, 'R_1': 0, 'R_2': 0, 'R_12': 0, 'P_1': 1, 'P_2': 1, 'I_1': 0, 'I_2': 0, 'I_12': 0, 'I_21': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['r', 'epsilon', 'd', 'g', 'beta_1', 'beta_2', 'a_1', 'alpha_1', 'a_2', 'alpha_2', 'k_1', 'u', 'k_2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['S', 'R_1', 'R_2', 'R_12', 'P_1', 'P_2', 'I_1', 'I_2', 'I_12', 'I_21']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (10)
        S = y[:, 0]
        R_1 = y[:, 1]
        R_2 = y[:, 2]
        R_12 = y[:, 3]
        P_1 = y[:, 4]
        P_2 = y[:, 5]
        I_1 = y[:, 6]
        I_2 = y[:, 7]
        I_12 = y[:, 8]
        I_21 = y[:, 9]

        # Independent parameters (source constant order)
        r = theta[:, 0]  # constants[0]
        epsilon = theta[:, 1]  # constants[1]
        d = theta[:, 2]  # constants[2]
        g = theta[:, 3]  # constants[3]
        beta_1 = theta[:, 4]  # constants[4]
        beta_2 = theta[:, 5]  # constants[5]
        a_1 = theta[:, 6]  # constants[6]
        alpha_1 = theta[:, 7]  # constants[7]
        a_2 = theta[:, 8]  # constants[8]
        alpha_2 = theta[:, 9]  # constants[9]
        k_1 = theta[:, 10]  # constants[10]
        u = theta[:, 11]  # constants[11]
        k_2 = theta[:, 12]  # constants[12]

        # Initialize rates and algebraic variables, as in computeRates()
        d_S = torch.zeros_like(y[:, 0])
        d_R_1 = torch.zeros_like(y[:, 0])
        d_R_2 = torch.zeros_like(y[:, 0])
        d_R_12 = torch.zeros_like(y[:, 0])
        d_P_1 = torch.zeros_like(y[:, 0])
        d_P_2 = torch.zeros_like(y[:, 0])
        d_I_1 = torch.zeros_like(y[:, 0])
        d_I_2 = torch.zeros_like(y[:, 0])
        d_I_12 = torch.zeros_like(y[:, 0])
        d_I_21 = torch.zeros_like(y[:, 0])
        alg_0_H = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_I_1 = torch.zeros_like(y[:, 0]) + (beta_1 * S * P_1 - a_1 * I_1 - alpha_1 * I_1)
        d_I_2 = torch.zeros_like(y[:, 0]) + (beta_2 * S * P_2 - a_2 * I_2 - alpha_2 * I_2)
        d_I_12 = torch.zeros_like(y[:, 0]) + (beta_2 * R_1 * P_2 - a_2 * I_12 - alpha_2 * I_12)
        d_I_21 = torch.zeros_like(y[:, 0]) + (beta_1 * R_2 * P_1 - a_1 * I_21 - alpha_1 * I_21)
        d_R_1 = torch.zeros_like(y[:, 0]) + (alpha_1 * I_1 - d * R_1 - g * R_1 - beta_2 * R_1 * P_2)
        d_R_2 = torch.zeros_like(y[:, 0]) + (alpha_2 * I_2 - d * R_2 - g * R_2 - beta_1 * R_2 * P_1)
        d_R_12 = torch.zeros_like(y[:, 0]) + (alpha_2 * I_12 + alpha_1 * I_21 - d * R_12 - g * R_12)
        d_P_1 = torch.zeros_like(y[:, 0]) + (k_1 * (I_1 + I_21) - u * P_1)
        d_P_2 = torch.zeros_like(y[:, 0]) + (k_2 * (I_2 + I_12) - u * P_2)
        alg_0_H = S + I_1 + R_1 + I_2 + R_2 + I_12 + I_21 + R_12
        d_S = torch.zeros_like(y[:, 0]) + (r * alg_0_H / (epsilon * alg_0_H + 1.0) - d * S - beta_1 * S * P_1 - beta_2 * S * P_2 + g * (R_1 + R_2 + R_12))

        return torch.stack(
            [d_S, d_R_1, d_R_2, d_R_12, d_P_1, d_P_2, d_I_1, d_I_2, d_I_12, d_I_21],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
