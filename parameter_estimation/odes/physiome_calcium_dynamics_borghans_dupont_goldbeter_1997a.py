import torch
import torch.nn as nn
import math


class CalciumDynamicsBorghansDupontGoldbeter1997aODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/borghans_dupont_goldbeter_1997a/model.py.
    Dynamic states: 3; independent constants: 13; derived constants used by rates: 1.
    Original constant slots: 14; algebraic slots: 4.
    """
    source_file = 'calcium_dynamics/borghans_dupont_goldbeter_1997a/model.py'
    source_state_dim = 3
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

        self.state_dim = 3
        self.parameter_dim = 13

        self.parameters = (
            parameters
            if parameters is not None
            else {'K_f': 1, 'K': 10, 'beta': 1, 'v_0': 1, 'v_1': 1, 'V_M2': 6.5, 'K_2': 0.1, 'K_y': 0.2, 'V_M3': 50, 'k_d': 5000, 'k_r': 5, 'a': 10000, 'd': 100}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Z': 0.3, 'Y': 2.7, 'rho': 0.2}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['K_f', 'K', 'beta', 'v_0', 'v_1', 'V_M2', 'K_2', 'K_y', 'V_M3', 'k_d', 'k_r', 'a', 'd']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Z', 'Y', 'rho']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        Z = y[:, 0]
        Y = y[:, 1]
        rho = y[:, 2]

        # Independent parameters (source constant order)
        K_f = theta[:, 0]  # constants[0]
        K = theta[:, 1]  # constants[1]
        beta = theta[:, 2]  # constants[2]
        v_0 = theta[:, 3]  # constants[3]
        v_1 = theta[:, 4]  # constants[4]
        V_M2 = theta[:, 5]  # constants[5]
        K_2 = theta[:, 6]  # constants[6]
        K_y = theta[:, 7]  # constants[7]
        V_M3 = theta[:, 8]  # constants[8]
        k_d = theta[:, 9]  # constants[9]
        k_r = theta[:, 10]  # constants[10]
        a = theta[:, 11]  # constants[11]
        d = theta[:, 12]  # constants[12]

        # Derived constants from the original initConsts()
        V_in = v_0 + v_1 * beta  # constants[13]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Z = torch.zeros_like(y[:, 0])
        d_Y = torch.zeros_like(y[:, 0])
        d_rho = torch.zeros_like(y[:, 0])
        alg_0_V_2 = torch.zeros_like(y[:, 0])
        alg_1_gamma = torch.zeros_like(y[:, 0])
        alg_2_R_plus = torch.zeros_like(y[:, 0])
        alg_3_V_3 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_rho = torch.zeros_like(y[:, 0]) + (-(k_d * Z ** 4.0 * rho * 1.0) + k_r * (1.0 - rho))
        alg_0_V_2 = V_M2 * (Z ** 2.0 / (K_2 ** 2.0 + Z ** 2.0))
        alg_1_gamma = a / d * Z ** 4.0 * 1.0
        alg_2_R_plus = alg_1_gamma * (rho / (1.0 + alg_1_gamma))
        alg_3_V_3 = beta * alg_2_R_plus * V_M3 * (Y ** 2.0 / (K_y ** 2.0 + Y ** 2.0))
        d_Z = torch.zeros_like(y[:, 0]) + (V_in - alg_0_V_2 + alg_3_V_3 + (K_f * Y - K * Z))
        d_Y = torch.zeros_like(y[:, 0]) + (alg_0_V_2 - alg_3_V_3 - K_f * Y)

        return torch.stack(
            [d_Z, d_Y, d_rho],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
