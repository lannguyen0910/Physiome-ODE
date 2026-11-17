import torch
import torch.nn as nn
import math


class GeneRegulationVilarKuehBarkaiLeibler2002ODE(nn.Module):
    """
    PyTorch translation of gene_regulation/vilar_kueh_barkai_leibler_2002/model.py.
    Dynamic states: 9; independent constants: 16; derived constants used by rates: 0.
    Original constant slots: 16; algebraic slots: 16.
    """
    source_file = 'gene_regulation/vilar_kueh_barkai_leibler_2002/model.py'
    source_state_dim = 9
    source_constant_dim = 16
    algebraic_dim = 16
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 9
        self.parameter_dim = 16

        self.parameters = (
            parameters
            if parameters is not None
            else {'Gamma_1': 2, 'Delta_1': 1, 'Delta_2': 1, 'Delta_3': 0.2, 'Gamma_2': 1, 'Thetha_1': 50, 'Alpha_1': 50, 'Alpha_2': 500, 'Delta_4': 10, 'BetaA_1': 50, 'Gamma_3': 1, 'Theta_2': 100, 'Alpha_3': 0.01, 'Alpha_4': 50, 'Delta_5': 0.5, 'BetaR_1': 5}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'A': 0, 'C': 0, 'DA': 1, 'DAp': 0, 'DR': 1, 'DRP': 1, 'MA': 0, 'MR': 0, 'R': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Gamma_1', 'Delta_1', 'Delta_2', 'Delta_3', 'Gamma_2', 'Thetha_1', 'Alpha_1', 'Alpha_2', 'Delta_4', 'BetaA_1', 'Gamma_3', 'Theta_2', 'Alpha_3', 'Alpha_4', 'Delta_5', 'BetaR_1']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['A', 'C', 'DA', 'DAp', 'DR', 'DRP', 'MA', 'MR', 'R']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (9)
        A = y[:, 0]
        C = y[:, 1]
        DA = y[:, 2]
        DAp = y[:, 3]
        DR = y[:, 4]
        DRP = y[:, 5]
        MA = y[:, 6]
        MR = y[:, 7]
        R = y[:, 8]

        # Independent parameters (source constant order)
        Gamma_1 = theta[:, 0]  # constants[0]
        Delta_1 = theta[:, 1]  # constants[1]
        Delta_2 = theta[:, 2]  # constants[2]
        Delta_3 = theta[:, 3]  # constants[3]
        Gamma_2 = theta[:, 4]  # constants[4]
        Thetha_1 = theta[:, 5]  # constants[5]
        Alpha_1 = theta[:, 6]  # constants[6]
        Alpha_2 = theta[:, 7]  # constants[7]
        Delta_4 = theta[:, 8]  # constants[8]
        BetaA_1 = theta[:, 9]  # constants[9]
        Gamma_3 = theta[:, 10]  # constants[10]
        Theta_2 = theta[:, 11]  # constants[11]
        Alpha_3 = theta[:, 12]  # constants[12]
        Alpha_4 = theta[:, 13]  # constants[13]
        Delta_5 = theta[:, 14]  # constants[14]
        BetaR_1 = theta[:, 15]  # constants[15]

        # Initialize rates and algebraic variables, as in computeRates()
        d_A = torch.zeros_like(y[:, 0])
        d_C = torch.zeros_like(y[:, 0])
        d_DA = torch.zeros_like(y[:, 0])
        d_DAp = torch.zeros_like(y[:, 0])
        d_DR = torch.zeros_like(y[:, 0])
        d_DRP = torch.zeros_like(y[:, 0])
        d_MA = torch.zeros_like(y[:, 0])
        d_MR = torch.zeros_like(y[:, 0])
        d_R = torch.zeros_like(y[:, 0])
        alg_0_RXN1 = torch.zeros_like(y[:, 0])
        alg_1_RXN2 = torch.zeros_like(y[:, 0])
        alg_2_RXN3 = torch.zeros_like(y[:, 0])
        alg_3_RXN4 = torch.zeros_like(y[:, 0])
        alg_4_RXN5 = torch.zeros_like(y[:, 0])
        alg_5_RXN6 = torch.zeros_like(y[:, 0])
        alg_6_RXN7 = torch.zeros_like(y[:, 0])
        alg_7_RXN8 = torch.zeros_like(y[:, 0])
        alg_8_RXN9 = torch.zeros_like(y[:, 0])
        alg_9_RXN10 = torch.zeros_like(y[:, 0])
        alg_10_RXN11 = torch.zeros_like(y[:, 0])
        alg_11_RXN12 = torch.zeros_like(y[:, 0])
        alg_12_RXN13 = torch.zeros_like(y[:, 0])
        alg_13_RXN14 = torch.zeros_like(y[:, 0])
        alg_14_RXN15 = torch.zeros_like(y[:, 0])
        alg_15_RXN16 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_RXN1 = A * R * Gamma_1
        alg_2_RXN3 = C * Delta_2
        d_C = torch.zeros_like(y[:, 0]) + (1.0 / 1.0 * (-1.0 * alg_2_RXN3 + alg_0_RXN1))
        alg_4_RXN5 = A * DA * Gamma_2
        alg_5_RXN6 = DAp * Thetha_1
        alg_6_RXN7 = DA * Alpha_1
        d_DA = torch.zeros_like(y[:, 0]) + (1.0 / 1.0 * (-1.0 * alg_4_RXN5 + -1.0 * alg_6_RXN7 + alg_5_RXN6 + alg_6_RXN7))
        alg_7_RXN8 = DAp * Alpha_2
        d_DAp = torch.zeros_like(y[:, 0]) + (1.0 / 1.0 * (-1.0 * alg_5_RXN6 + -1.0 * alg_7_RXN8 + alg_4_RXN5 + alg_7_RXN8))
        alg_9_RXN10 = MA * BetaA_1
        alg_8_RXN9 = MA * Delta_4
        d_MA = torch.zeros_like(y[:, 0]) + (1.0 / 1.0 * (-1.0 * alg_8_RXN9 + -1.0 * alg_9_RXN10 + alg_6_RXN7 + alg_7_RXN8 + alg_9_RXN10))
        alg_1_RXN2 = A * Delta_1
        alg_10_RXN11 = A * DR * Gamma_3
        alg_11_RXN12 = DRP * Theta_2
        d_A = torch.zeros_like(y[:, 0]) + (1.0 / 1.0 * (-1.0 * alg_0_RXN1 + -1.0 * alg_1_RXN2 + -1.0 * alg_4_RXN5 + -1.0 * alg_10_RXN11 + alg_5_RXN6 + alg_9_RXN10 + alg_11_RXN12))
        alg_12_RXN13 = DR * Alpha_3
        d_DR = torch.zeros_like(y[:, 0]) + (1.0 / 1.0 * (-1.0 * alg_10_RXN11 + -1.0 * alg_12_RXN13 + alg_11_RXN12 + alg_12_RXN13))
        alg_13_RXN14 = DRP * Alpha_4
        d_DRP = torch.zeros_like(y[:, 0]) + (1.0 / 1.0 * (-1.0 * alg_11_RXN12 + -1.0 * alg_13_RXN14 + alg_10_RXN11 + alg_13_RXN14))
        alg_14_RXN15 = MR * Delta_5
        alg_15_RXN16 = MR * BetaR_1
        d_MR = torch.zeros_like(y[:, 0]) + (1.0 / 1.0 * (-1.0 * alg_14_RXN15 + -1.0 * alg_15_RXN16 + alg_12_RXN13 + alg_13_RXN14 + alg_15_RXN16))
        alg_3_RXN4 = R * Delta_3
        d_R = torch.zeros_like(y[:, 0]) + (1.0 / 1.0 * (-1.0 * alg_0_RXN1 + -1.0 * alg_3_RXN4 + alg_2_RXN3 + alg_15_RXN16))

        return torch.stack(
            [d_A, d_C, d_DA, d_DAp, d_DR, d_DRP, d_MA, d_MR, d_R],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
