import torch
import torch.nn as nn
import math


class CellCycleRomondRusticiGonzeGoldbeter1999ODE(nn.Module):
    """
    PyTorch translation of cell_cycle/romond_rustici_gonze_goldbeter_1999/model.py.
    Dynamic states: 6; independent constants: 28; derived constants used by rates: 0.
    Original constant slots: 28; algebraic slots: 4.
    """
    source_file = 'cell_cycle/romond_rustici_gonze_goldbeter_1999/model.py'
    source_state_dim = 6
    source_constant_dim = 28
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

        self.state_dim = 6
        self.parameter_dim = 28

        self.parameters = (
            parameters
            if parameters is not None
            else {'H_1': 0.01, 'H_2': 0.01, 'H_3': 0.01, 'H_4': 0.01, 'K_1': 0.01, 'K_2': 0.01, 'K_3': 0.01, 'K_4': 0.01, 'V_M1': 0.3, 'V_M3': 0.1, 'U_M1': 0.3, 'U_M3': 0.1, 'V_2': 0.15, 'V_4': 0.05, 'U_2': 0.15, 'U_4': 0.05, 'K_c1': 0.5, 'K_c2': 0.5, 'K_d1': 0.02, 'K_d2': 0.02, 'v_d1': 0.025, 'v_d2': 0.025, 'v_i1': 0.05, 'v_i2': 0.05, 'k_d1': 0.001, 'k_d2': 0.001, 'K_im1': 0.03, 'K_im2': 0.03}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'C_1': 2, 'M_2': 0, 'X_1': 0, 'M_1': 1, 'C_2': 0, 'X_2': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['H_1', 'H_2', 'H_3', 'H_4', 'K_1', 'K_2', 'K_3', 'K_4', 'V_M1', 'V_M3', 'U_M1', 'U_M3', 'V_2', 'V_4', 'U_2', 'U_4', 'K_c1', 'K_c2', 'K_d1', 'K_d2', 'v_d1', 'v_d2', 'v_i1', 'v_i2', 'k_d1', 'k_d2', 'K_im1', 'K_im2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['C_1', 'M_2', 'X_1', 'M_1', 'C_2', 'X_2']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (6)
        C_1 = y[:, 0]
        M_2 = y[:, 1]
        X_1 = y[:, 2]
        M_1 = y[:, 3]
        C_2 = y[:, 4]
        X_2 = y[:, 5]

        # Independent parameters (source constant order)
        H_1 = theta[:, 0]  # constants[0]
        H_2 = theta[:, 1]  # constants[1]
        H_3 = theta[:, 2]  # constants[2]
        H_4 = theta[:, 3]  # constants[3]
        K_1 = theta[:, 4]  # constants[4]
        K_2 = theta[:, 5]  # constants[5]
        K_3 = theta[:, 6]  # constants[6]
        K_4 = theta[:, 7]  # constants[7]
        V_M1 = theta[:, 8]  # constants[8]
        V_M3 = theta[:, 9]  # constants[9]
        U_M1 = theta[:, 10]  # constants[10]
        U_M3 = theta[:, 11]  # constants[11]
        V_2 = theta[:, 12]  # constants[12]
        V_4 = theta[:, 13]  # constants[13]
        U_2 = theta[:, 14]  # constants[14]
        U_4 = theta[:, 15]  # constants[15]
        K_c1 = theta[:, 16]  # constants[16]
        K_c2 = theta[:, 17]  # constants[17]
        K_d1 = theta[:, 18]  # constants[18]
        K_d2 = theta[:, 19]  # constants[19]
        v_d1 = theta[:, 20]  # constants[20]
        v_d2 = theta[:, 21]  # constants[21]
        v_i1 = theta[:, 22]  # constants[22]
        v_i2 = theta[:, 23]  # constants[23]
        k_d1 = theta[:, 24]  # constants[24]
        k_d2 = theta[:, 25]  # constants[25]
        K_im1 = theta[:, 26]  # constants[26]
        K_im2 = theta[:, 27]  # constants[27]

        # Initialize rates and algebraic variables, as in computeRates()
        d_C_1 = torch.zeros_like(y[:, 0])
        d_M_2 = torch.zeros_like(y[:, 0])
        d_X_1 = torch.zeros_like(y[:, 0])
        d_M_1 = torch.zeros_like(y[:, 0])
        d_C_2 = torch.zeros_like(y[:, 0])
        d_X_2 = torch.zeros_like(y[:, 0])
        alg_0_V_1 = torch.zeros_like(y[:, 0])
        alg_1_V_3 = torch.zeros_like(y[:, 0])
        alg_2_U_1 = torch.zeros_like(y[:, 0])
        alg_3_U_3 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_C_1 = torch.zeros_like(y[:, 0]) + (v_i1 * K_im1 / (K_im1 + M_2) - v_d1 * X_1 * C_1 / (K_d1 + C_1) - k_d1 * C_1)
        d_C_2 = torch.zeros_like(y[:, 0]) + (v_i2 * K_im2 / (K_im2 + M_1) - v_d2 * X_2 * C_2 / (K_d2 + C_2) - k_d2 * C_2)
        alg_0_V_1 = C_1 / (K_c1 + C_1) * V_M1
        d_M_1 = torch.zeros_like(y[:, 0]) + (alg_0_V_1 * (1.0 - M_1) / (K_1 + (1.0 - M_1)) - V_2 * M_1 / (K_2 + M_1))
        alg_1_V_3 = M_1 * V_M3
        d_X_1 = torch.zeros_like(y[:, 0]) + (alg_1_V_3 * (1.0 - X_1) / (K_3 + (1.0 - X_1)) - V_4 * X_1 / (K_4 + X_1))
        alg_2_U_1 = C_2 / (K_c2 + C_2) * U_M1
        d_M_2 = torch.zeros_like(y[:, 0]) + (alg_2_U_1 * (1.0 - M_2) / (H_1 + (1.0 - M_2)) - U_2 * M_2 / (H_2 + M_2))
        alg_3_U_3 = M_2 * U_M3
        d_X_2 = torch.zeros_like(y[:, 0]) + (alg_3_U_3 * (1.0 - X_2) / (H_3 + (1.0 - X_2)) - U_4 * X_2 / (H_4 + X_2))

        return torch.stack(
            [d_C_1, d_M_2, d_X_1, d_M_1, d_C_2, d_X_2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
