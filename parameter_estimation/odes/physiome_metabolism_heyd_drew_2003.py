import torch
import torch.nn as nn
import math


class MetabolismHeydDrew2003ODE(nn.Module):
    """
    PyTorch translation of metabolism/heyd_drew_2003/model.py.
    Dynamic states: 11; independent constants: 18; derived constants used by rates: 0.
    Original constant slots: 18; algebraic slots: 0.
    """
    source_file = 'metabolism/heyd_drew_2003/model.py'
    source_state_dim = 11
    source_constant_dim = 18
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

        self.state_dim = 11
        self.parameter_dim = 18

        self.parameters = (
            parameters
            if parameters is not None
            else {'k11': 110, 'k11_2': 25, 'k71': 0.1, 'k72': 60, 'k12': 110, 'k12_2': 0.2, 'k51': 7, 'k52': 0, 'k41': 60, 'k31': 500, 'k32': 50, 'k42': 70, 'k22': 100, 'k22_2': 17, 'k21': 100, 'k21_2': 0.2, 'kreset1': 35, 'PE2': 0.1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'PB': 0.1, 'A1': 0.1, 'A2': 0.1, 'PC1': 0.1, 'PF1': 0.1, 'PG1': 0.1, 'PC2': 0.1, 'PF2': 0.1, 'PD1': 0.1, 'PE1': 0.1, 'PD2': 0.1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k11', 'k11_2', 'k71', 'k72', 'k12', 'k12_2', 'k51', 'k52', 'k41', 'k31', 'k32', 'k42', 'k22', 'k22_2', 'k21', 'k21_2', 'kreset1', 'PE2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['PB', 'A1', 'A2', 'PC1', 'PF1', 'PG1', 'PC2', 'PF2', 'PD1', 'PE1', 'PD2']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (11)
        PB = y[:, 0]
        A1 = y[:, 1]
        A2 = y[:, 2]
        PC1 = y[:, 3]
        PF1 = y[:, 4]
        PG1 = y[:, 5]
        PC2 = y[:, 6]
        PF2 = y[:, 7]
        PD1 = y[:, 8]
        PE1 = y[:, 9]
        PD2 = y[:, 10]

        # Independent parameters (source constant order)
        k11 = theta[:, 0]  # constants[0]
        k11_2 = theta[:, 1]  # constants[1]
        k71 = theta[:, 2]  # constants[2]
        k72 = theta[:, 3]  # constants[3]
        k12 = theta[:, 4]  # constants[4]
        k12_2 = theta[:, 5]  # constants[5]
        k51 = theta[:, 6]  # constants[6]
        k52 = theta[:, 7]  # constants[7]
        k41 = theta[:, 8]  # constants[8]
        k31 = theta[:, 9]  # constants[9]
        k32 = theta[:, 10]  # constants[10]
        k42 = theta[:, 11]  # constants[11]
        k22 = theta[:, 12]  # constants[12]
        k22_2 = theta[:, 13]  # constants[13]
        k21 = theta[:, 14]  # constants[14]
        k21_2 = theta[:, 15]  # constants[15]
        kreset1 = theta[:, 16]  # constants[16]
        PE2 = theta[:, 17]  # constants[17]

        # Initialize rates and algebraic variables, as in computeRates()
        d_PB = torch.zeros_like(y[:, 0])
        d_A1 = torch.zeros_like(y[:, 0])
        d_A2 = torch.zeros_like(y[:, 0])
        d_PC1 = torch.zeros_like(y[:, 0])
        d_PF1 = torch.zeros_like(y[:, 0])
        d_PG1 = torch.zeros_like(y[:, 0])
        d_PC2 = torch.zeros_like(y[:, 0])
        d_PF2 = torch.zeros_like(y[:, 0])
        d_PD1 = torch.zeros_like(y[:, 0])
        d_PE1 = torch.zeros_like(y[:, 0])
        d_PD2 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_PB = torch.zeros_like(y[:, 0]) + (k11_2 * PC1 + k71 * PF1 + kreset1 * PG1 + k12_2 * PC2 + k72 * PF2 - (k11 * A1 * PB + k12 * A2 * PB))
        d_A1 = torch.zeros_like(y[:, 0]) + (k11_2 * PC1 - k11 * A1 * PB)
        d_A2 = torch.zeros_like(y[:, 0]) + (k12_2 * PC2 - k12 * A2 * PB)
        d_PC1 = torch.zeros_like(y[:, 0]) + (k11 * A1 * PB - (k11_2 + k21) * PC1 + k21_2 * PD1)
        d_PD1 = torch.zeros_like(y[:, 0]) + (k21 * PC1 - (k21_2 + k31) * PD1)
        d_PE1 = torch.zeros_like(y[:, 0]) + (k31 * PD1 - k41 * PE1)
        d_PF1 = torch.zeros_like(y[:, 0]) + (k41 * PE1 - (k51 + k71) * PF1)
        d_PG1 = torch.zeros_like(y[:, 0]) + (k51 * PF1 - kreset1 * PG1)
        d_PC2 = torch.zeros_like(y[:, 0]) + (k12 * A2 * PB - (k12_2 + k22) * PC2 + k22_2 * PD2)
        d_PD2 = torch.zeros_like(y[:, 0]) + (k22 * PC2 - (k22_2 + k32) * PD2)
        d_PF2 = torch.zeros_like(y[:, 0]) + (k42 * PE2 - (k52 + k72) * PF2)

        return torch.stack(
            [d_PB, d_A1, d_A2, d_PC1, d_PF1, d_PG1, d_PC2, d_PF2, d_PD1, d_PE1, d_PD2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
