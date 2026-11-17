import torch
import torch.nn as nn
import math


class GeneRegulationLeloupGonzeGoldbeter1999BODE(nn.Module):
    """
    PyTorch translation of gene_regulation/leloup_gonze_goldbeter_1999_b/model.py.
    Dynamic states: 3; independent constants: 10; derived constants used by rates: 0.
    Original constant slots: 10; algebraic slots: 1.
    """
    source_file = 'gene_regulation/leloup_gonze_goldbeter_1999_b/model.py'
    source_state_dim = 3
    source_constant_dim = 10
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

        self.state_dim = 3
        self.parameter_dim = 10

        self.parameters = (
            parameters
            if parameters is not None
            else {'vs': 1.6, 'vm': 0.505, 'Km': 0.5, 'KI': 1, 'n': 4, 'ks': 0.5, 'vd': 1.4, 'Kd': 0.13, 'k1': 0.5, 'k2': 0.6}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'M': 0.1, 'FN': 0.1, 'FC': 0.1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['vs', 'vm', 'Km', 'KI', 'n', 'ks', 'vd', 'Kd', 'k1', 'k2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['M', 'FN', 'FC']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        M = y[:, 0]
        FN = y[:, 1]
        FC = y[:, 2]

        # Independent parameters (source constant order)
        vs = theta[:, 0]  # constants[0]
        vm = theta[:, 1]  # constants[1]
        Km = theta[:, 2]  # constants[2]
        KI = theta[:, 3]  # constants[3]
        n = theta[:, 4]  # constants[4]
        ks = theta[:, 5]  # constants[5]
        vd = theta[:, 6]  # constants[6]
        Kd = theta[:, 7]  # constants[7]
        k1 = theta[:, 8]  # constants[8]
        k2 = theta[:, 9]  # constants[9]

        # Initialize rates and algebraic variables, as in computeRates()
        d_M = torch.zeros_like(y[:, 0])
        d_FN = torch.zeros_like(y[:, 0])
        d_FC = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_M = torch.zeros_like(y[:, 0]) + (vs * (KI ** n / (KI ** n + FN ** n)) - vm * (M / (Km + M)))
        d_FC = torch.zeros_like(y[:, 0]) + (ks * M + k2 * FN - (vd * (FC / (Kd + FC)) + k1 * FC))
        d_FN = torch.zeros_like(y[:, 0]) + (k1 * FC - k2 * FN)

        return torch.stack(
            [d_M, d_FN, d_FC],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
