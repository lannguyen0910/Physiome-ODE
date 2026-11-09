import torch
import torch.nn as nn
import math


class CellCycleSriramBernotKepes2007ODE(nn.Module):
    """
    PyTorch translation of cell_cycle/sriram_bernot_kepes_2007/model.py.
    Dynamic states: 6; independent constants: 25; derived constants used by rates: 0.
    Original constant slots: 25; algebraic slots: 0.
    """
    source_file = 'cell_cycle/sriram_bernot_kepes_2007/model.py'
    source_state_dim = 6
    source_constant_dim = 25
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

        self.state_dim = 6
        self.parameter_dim = 25

        self.parameters = (
            parameters
            if parameters is not None
            else {'j_1': 0.9, 'j_2': 0.5, 'j_3': 0.6, 'v_d1': 6, 'v_d2': 1.052, 'v_d3': 3, 'k_d1': 0.8, 'k_d2': 0.9, 'k_d3': 0.8, 'k_c1': 0.2, 'k_c2': 0.22, 'k_c3': 0.6, 'k_m1': 5, 'k_m2': 5, 'k_m3': 5, 'v_12': 15, 'v_11': 15, 'v_10': 15, 'k_120': 10, 'k_110': 10, 'k_100': 10, 'k_d4': 0.16, 'k_d5': 0.16, 'k_d6': 0.16, 'n': 2}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'C_1': 0, 'C_2': 0, 'T_1': 6, 'C_3': 0, 'T_2': 5, 'T_3': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['j_1', 'j_2', 'j_3', 'v_d1', 'v_d2', 'v_d3', 'k_d1', 'k_d2', 'k_d3', 'k_c1', 'k_c2', 'k_c3', 'k_m1', 'k_m2', 'k_m3', 'v_12', 'v_11', 'v_10', 'k_120', 'k_110', 'k_100', 'k_d4', 'k_d5', 'k_d6', 'n']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['C_1', 'C_2', 'T_1', 'C_3', 'T_2', 'T_3']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (6)
        C_1 = y[:, 0]
        C_2 = y[:, 1]
        T_1 = y[:, 2]
        C_3 = y[:, 3]
        T_2 = y[:, 4]
        T_3 = y[:, 5]

        # Independent parameters (source constant order)
        j_1 = theta[:, 0]  # constants[0]
        j_2 = theta[:, 1]  # constants[1]
        j_3 = theta[:, 2]  # constants[2]
        v_d1 = theta[:, 3]  # constants[3]
        v_d2 = theta[:, 4]  # constants[4]
        v_d3 = theta[:, 5]  # constants[5]
        k_d1 = theta[:, 6]  # constants[6]
        k_d2 = theta[:, 7]  # constants[7]
        k_d3 = theta[:, 8]  # constants[8]
        k_c1 = theta[:, 9]  # constants[9]
        k_c2 = theta[:, 10]  # constants[10]
        k_c3 = theta[:, 11]  # constants[11]
        k_m1 = theta[:, 12]  # constants[12]
        k_m2 = theta[:, 13]  # constants[13]
        k_m3 = theta[:, 14]  # constants[14]
        v_12 = theta[:, 15]  # constants[15]
        v_11 = theta[:, 16]  # constants[16]
        v_10 = theta[:, 17]  # constants[17]
        k_120 = theta[:, 18]  # constants[18]
        k_110 = theta[:, 19]  # constants[19]
        k_100 = theta[:, 20]  # constants[20]
        k_d4 = theta[:, 21]  # constants[21]
        k_d5 = theta[:, 22]  # constants[22]
        k_d6 = theta[:, 23]  # constants[23]
        n = theta[:, 24]  # constants[24]

        # Initialize rates and algebraic variables, as in computeRates()
        d_C_1 = torch.zeros_like(y[:, 0])
        d_C_2 = torch.zeros_like(y[:, 0])
        d_T_1 = torch.zeros_like(y[:, 0])
        d_C_3 = torch.zeros_like(y[:, 0])
        d_T_2 = torch.zeros_like(y[:, 0])
        d_T_3 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_C_1 = torch.zeros_like(y[:, 0]) + (v_12 * T_1 ** n / (k_120 ** n + T_1 ** n + C_2 ** n) - k_d4 * C_1)
        d_C_2 = torch.zeros_like(y[:, 0]) + (v_11 * T_2 ** n / (k_110 ** n + T_2 ** n + C_3 ** n) - k_d5 * C_2)
        d_C_3 = torch.zeros_like(y[:, 0]) + (v_10 * T_3 ** n / (k_100 ** n + T_3 ** n + C_1 ** n) - k_d6 * C_3)
        d_T_1 = torch.zeros_like(y[:, 0]) + (j_1 + v_d1 * T_3 ** n / (k_m1 ** n + T_3 ** n) + k_c1 * C_1 - k_d1 * T_1)
        d_T_2 = torch.zeros_like(y[:, 0]) + (j_2 + v_d2 * T_1 ** n / (k_m2 ** n + T_1 ** n) + k_c2 * C_2 - k_d2 * T_2)
        d_T_3 = torch.zeros_like(y[:, 0]) + (j_3 + v_d3 * T_2 ** n / (k_m3 ** n + T_2 ** n) + k_c3 * C_3 - k_d3 * T_3)

        return torch.stack(
            [d_C_1, d_C_2, d_T_1, d_C_3, d_T_2, d_T_3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
