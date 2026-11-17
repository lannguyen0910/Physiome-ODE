import torch
import torch.nn as nn
import math


class SignalTransductionChenPopel2007CODE(nn.Module):
    """
    PyTorch translation of signal_transduction/chen_popel_2007_c/model.py.
    Dynamic states: 14; independent constants: 20; derived constants used by rates: 0.
    Original constant slots: 20; algebraic slots: 1.
    """
    source_file = 'signal_transduction/chen_popel_2007_c/model.py'
    source_state_dim = 14
    source_constant_dim = 20
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

        self.state_dim = 14
        self.parameter_dim = 20

        self.parameters = (
            parameters
            if parameters is not None
            else {'Arg': 100, 'O2': 100, 'k1': 6.6, 'k_1': 6.6, 'k2': 20.8, 'k13': 39.9, 'k12': 0.01, 'k3': 20.8, 'k_4': 6.6, 'k4': 6.6, 'k5': 8.5, 'k_5': 215.6, 'k6': 175.6, 'k7': 20.8, 'k9': 8.6, 'k_9': 399.2, 'k_8': 13.2, 'k8': 13.2, 'k10': 39.1, 'k11': 20.8}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Fe3': 0.9, 'Fe3_Arg': 0, 'Fe3_NO': 0, 'Fe2_NO': 0, 'Fe2': 0, 'Fe2_Arg': 0, 'Fe3_O2_Arg': 0, 'Fe3_NOHA': 0, 'Fe2_NOHA': 0, 'NOHA': 0, 'Fe3_O2_NOHA': 0, 'NO': 0, 'citrulline': 0, 'NO3': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Arg', 'O2', 'k1', 'k_1', 'k2', 'k13', 'k12', 'k3', 'k_4', 'k4', 'k5', 'k_5', 'k6', 'k7', 'k9', 'k_9', 'k_8', 'k8', 'k10', 'k11']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Fe3', 'Fe3_Arg', 'Fe3_NO', 'Fe2_NO', 'Fe2', 'Fe2_Arg', 'Fe3_O2_Arg', 'Fe3_NOHA', 'Fe2_NOHA', 'NOHA', 'Fe3_O2_NOHA', 'NO', 'citrulline', 'NO3']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (14)
        Fe3 = y[:, 0]
        Fe3_Arg = y[:, 1]
        Fe3_NO = y[:, 2]
        Fe2_NO = y[:, 3]
        Fe2 = y[:, 4]
        Fe2_Arg = y[:, 5]
        Fe3_O2_Arg = y[:, 6]
        Fe3_NOHA = y[:, 7]
        Fe2_NOHA = y[:, 8]
        NOHA = y[:, 9]
        Fe3_O2_NOHA = y[:, 10]
        NO = y[:, 11]
        citrulline = y[:, 12]
        NO3 = y[:, 13]

        # Independent parameters (source constant order)
        Arg = theta[:, 0]  # constants[0]
        O2 = theta[:, 1]  # constants[1]
        k1 = theta[:, 2]  # constants[2]
        k_1 = theta[:, 3]  # constants[3]
        k2 = theta[:, 4]  # constants[4]
        k13 = theta[:, 5]  # constants[5]
        k12 = theta[:, 6]  # constants[6]
        k3 = theta[:, 7]  # constants[7]
        k_4 = theta[:, 8]  # constants[8]
        k4 = theta[:, 9]  # constants[9]
        k5 = theta[:, 10]  # constants[10]
        k_5 = theta[:, 11]  # constants[11]
        k6 = theta[:, 12]  # constants[12]
        k7 = theta[:, 13]  # constants[13]
        k9 = theta[:, 14]  # constants[14]
        k_9 = theta[:, 15]  # constants[15]
        k_8 = theta[:, 16]  # constants[16]
        k8 = theta[:, 17]  # constants[17]
        k10 = theta[:, 18]  # constants[18]
        k11 = theta[:, 19]  # constants[19]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Fe3 = torch.zeros_like(y[:, 0])
        d_Fe3_Arg = torch.zeros_like(y[:, 0])
        d_Fe3_NO = torch.zeros_like(y[:, 0])
        d_Fe2_NO = torch.zeros_like(y[:, 0])
        d_Fe2 = torch.zeros_like(y[:, 0])
        d_Fe2_Arg = torch.zeros_like(y[:, 0])
        d_Fe3_O2_Arg = torch.zeros_like(y[:, 0])
        d_Fe3_NOHA = torch.zeros_like(y[:, 0])
        d_Fe2_NOHA = torch.zeros_like(y[:, 0])
        d_NOHA = torch.zeros_like(y[:, 0])
        d_Fe3_O2_NOHA = torch.zeros_like(y[:, 0])
        d_NO = torch.zeros_like(y[:, 0])
        d_citrulline = torch.zeros_like(y[:, 0])
        d_NO3 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_NO = torch.zeros_like(y[:, 0]) + (k13 * Fe3_NO)
        d_Fe3 = torch.zeros_like(y[:, 0]) + (k_1 * Fe3_Arg + k13 * Fe3_NO + k12 * Fe2_NO * O2 - (k1 * Arg * Fe3 + k2 * Fe3))
        d_Fe3_Arg = torch.zeros_like(y[:, 0]) + (k1 * Fe3 * Arg - (k_1 * Fe3_Arg + k3 * Fe3_Arg))
        d_Fe2 = torch.zeros_like(y[:, 0]) + (k2 * Fe3 + k_4 * Fe2_Arg - k4 * Fe2 * Arg)
        d_Fe2_Arg = torch.zeros_like(y[:, 0]) + (k3 * Fe3_Arg + k_5 * Fe3_O2_Arg + k4 * Fe2 * Arg - (k5 * Fe2_Arg * O2 + k_4 * Fe2_Arg))
        d_Fe3_O2_Arg = torch.zeros_like(y[:, 0]) + (k5 * Fe2_Arg * O2 - (k6 * Fe3_O2_Arg + k_5 * Fe3_O2_Arg))
        d_Fe3_NOHA = torch.zeros_like(y[:, 0]) + (k6 * Fe3_O2_Arg - k7 * Fe3_NOHA)
        d_Fe2_NOHA = torch.zeros_like(y[:, 0]) + (k7 * Fe3_NOHA + k_9 * Fe3_O2_NOHA + k8 * Fe2 * NOHA - (k_8 * Fe2_NOHA + k9 * Fe2_NOHA * O2))
        d_Fe3_O2_NOHA = torch.zeros_like(y[:, 0]) + (k9 * Fe2_NOHA * O2 - (k10 * Fe3_O2_NOHA + k_9 * Fe3_O2_NOHA))
        d_Fe3_NO = torch.zeros_like(y[:, 0]) + (k10 * Fe3_O2_NOHA - (k13 * Fe3_NO + k11 * Fe3_NO))
        d_Fe2_NO = torch.zeros_like(y[:, 0]) + (k11 * Fe3_NO - k12 * Fe2_NO * O2)
        d_citrulline = torch.zeros_like(y[:, 0]) + (k10 * Fe3_O2_NOHA)
        d_NO3 = torch.zeros_like(y[:, 0]) + (k12 * Fe2_NO * O2)
        d_NOHA = torch.zeros_like(y[:, 0]) + (k_8 * Fe2_NOHA - k8 * Fe2 * NOHA)

        return torch.stack(
            [d_Fe3, d_Fe3_Arg, d_Fe3_NO, d_Fe2_NO, d_Fe2, d_Fe2_Arg, d_Fe3_O2_Arg, d_Fe3_NOHA, d_Fe2_NOHA, d_NOHA, d_Fe3_O2_NOHA, d_NO, d_citrulline, d_NO3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
