import torch
import torch.nn as nn
import math


class CalciumDynamicsBorghansDupontGoldbeter1997cODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/borghans_dupont_goldbeter_1997c/model.py.
    Dynamic states: 3; independent constants: 14; derived constants used by rates: 1.
    Original constant slots: 15; algebraic slots: 4.
    """
    source_file = 'calcium_dynamics/borghans_dupont_goldbeter_1997c/model.py'
    source_state_dim = 3
    source_constant_dim = 15
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
        self.parameter_dim = 14

        self.parameters = (
            parameters
            if parameters is not None
            else {'K_f': 0.5, 'K': 1, 'beta': 1, 'v_0': 0.015, 'v_1': 0.012, 'V_M2i': 3.1, 'K_2i': 0.005, 'V_M3i': 25, 'K_3z': 0.022, 'K_3y': 0.065, 'V_M2s': 1.5, 'K_2s': 0.0265, 'V_M3s': 0.169, 'K_3s': 0.1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Z': 0, 'Y': 0, 'X': 0.5}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['K_f', 'K', 'beta', 'v_0', 'v_1', 'V_M2i', 'K_2i', 'V_M3i', 'K_3z', 'K_3y', 'V_M2s', 'K_2s', 'V_M3s', 'K_3s']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Z', 'Y', 'X']
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
        X = y[:, 2]

        # Independent parameters (source constant order)
        K_f = theta[:, 0]  # constants[0]
        K = theta[:, 1]  # constants[1]
        beta = theta[:, 2]  # constants[2]
        v_0 = theta[:, 3]  # constants[3]
        v_1 = theta[:, 4]  # constants[4]
        V_M2i = theta[:, 5]  # constants[5]
        K_2i = theta[:, 6]  # constants[6]
        V_M3i = theta[:, 7]  # constants[7]
        K_3z = theta[:, 8]  # constants[8]
        K_3y = theta[:, 9]  # constants[9]
        V_M2s = theta[:, 10]  # constants[10]
        K_2s = theta[:, 11]  # constants[11]
        V_M3s = theta[:, 12]  # constants[12]
        K_3s = theta[:, 13]  # constants[13]

        # Derived constants from the original initConsts()
        V_in = v_0 + v_1 * beta  # constants[14]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Z = torch.zeros_like(y[:, 0])
        d_Y = torch.zeros_like(y[:, 0])
        d_X = torch.zeros_like(y[:, 0])
        alg_0_V_2i = torch.zeros_like(y[:, 0])
        alg_1_V_3i = torch.zeros_like(y[:, 0])
        alg_2_V_2s = torch.zeros_like(y[:, 0])
        alg_3_V_3s = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_V_2i = V_M2i * (Z ** 2.0 / (K_2i ** 2.0 + Z ** 2.0))
        alg_1_V_3i = V_M3i * (Y ** 2.0 / (K_3y ** 2.0 + Y ** 2.0)) * (Z ** 2.0 / (K_3z ** 2.0 + Z ** 2.0))
        d_Y = torch.zeros_like(y[:, 0]) + (alg_0_V_2i + -alg_1_V_3i + -(K_f * Y))
        alg_2_V_2s = V_M2s * (Z ** 2.0 / (K_2s ** 2.0 + Z ** 2.0))
        alg_3_V_3s = beta * V_M3s * (X ** 2.0 / (K_3s ** 2.0 + X ** 2.0))
        d_Z = torch.zeros_like(y[:, 0]) + (V_in + -alg_0_V_2i + alg_1_V_3i + K_f * Y + -alg_2_V_2s + alg_3_V_3s + K_f * X + -(K * Z))
        d_X = torch.zeros_like(y[:, 0]) + (alg_2_V_2s + -alg_3_V_3s + -(K_f * X))

        return torch.stack(
            [d_Z, d_Y, d_X],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
