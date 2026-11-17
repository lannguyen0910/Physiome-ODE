import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonAtrialNatriureticPeptide2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_atrial_natriuretic_peptide_2008/model.py.
    Dynamic states: 1; independent constants: 6; derived constants used by rates: 4.
    Original constant slots: 10; algebraic slots: 2.
    """
    source_file = 'cardiovascular_circulation/guyton_atrial_natriuretic_peptide_2008/model.py'
    source_state_dim = 1
    source_constant_dim = 10
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

        self.state_dim = 1
        self.parameter_dim = 6

        self.parameters = (
            parameters
            if parameters is not None
            else {'PLA': 2, 'PRA': 0.00852183, 'ANPKNS': 0, 'ANPINF': 0, 'ANPTC': 4, 'ANPXUL': 10}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'ANPC': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['PLA', 'PRA', 'ANPKNS', 'ANPINF', 'ANPTC', 'ANPXUL']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['ANPC']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (1)
        ANPC = y[:, 0]

        # Independent parameters (source constant order)
        PLA = theta[:, 0]  # constants[0]
        PRA = theta[:, 1]  # constants[1]
        ANPKNS = theta[:, 2]  # constants[2]
        ANPINF = theta[:, 3]  # constants[3]
        ANPTC = theta[:, 4]  # constants[4]
        ANPXUL = theta[:, 5]  # constants[5]

        # Derived constants from the original initConsts()
        ANPL = torch.where((PLA - 1.0) * 1.0 < 0.0, 0.0, (PLA - 1.0) * 1.0)  # constants[6]
        ANPR2 = torch.where((PRA + 1.0) * 2.0 < 0.0, 0.0, (PRA + 1.0) * 2.0)  # constants[7]
        ANP = (ANPL + ANPR2) / 3.0  # constants[8]
        ANP1 = torch.where(ANPKNS > 0.0, ANPKNS, ANP + ANPINF)  # constants[9]

        # Initialize rates and algebraic variables, as in computeRates()
        d_ANPC = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_ANPC = torch.zeros_like(y[:, 0]) + ((ANP1 - ANPC) / ANPTC)

        return torch.stack(
            [d_ANPC],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
