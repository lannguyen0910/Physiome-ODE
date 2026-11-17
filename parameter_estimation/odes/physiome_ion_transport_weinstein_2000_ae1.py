import torch
import torch.nn as nn
import math


class IonTransportWeinstein2000AE1ODE(nn.Module):
    """
    PyTorch translation of ion_transport/Weinstein_2000_AE1/model.py.
    Dynamic states: 1; independent constants: 14; derived constants used by rates: 0.
    Original constant slots: 17; algebraic slots: 15.
    """
    source_file = 'ion_transport/Weinstein_2000_AE1/model.py'
    source_state_dim = 1
    source_constant_dim = 17
    algebraic_dim = 15
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
        self.parameter_dim = 14

        self.parameters = (
            parameters
            if parameters is not None
            else {'HCO3_e': 26, 'Cl_i': 29, 'Cl_e': 114, 'x_Tmax': 1, 'K_I': 172, 'Kc_p': 50, 'Kc_pp': 50, 'Kb_p': 198, 'Kb_pp': 198, 'Pc_p': 562, 'Pc_pp': 61, 'Pb_p': 1247, 'Pb_pp': 135, 'constant_16': 60}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'HCO3_i': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['HCO3_e', 'Cl_i', 'Cl_e', 'x_Tmax', 'K_I', 'Kc_p', 'Kc_pp', 'Kb_p', 'Kb_pp', 'Pc_p', 'Pc_pp', 'Pb_p', 'Pb_pp', 'constant_16']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['HCO3_i']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (1)
        HCO3_i = y[:, 0]

        # Independent parameters (source constant order)
        HCO3_e = theta[:, 0]  # constants[0]
        Cl_i = theta[:, 1]  # constants[1]
        Cl_e = theta[:, 2]  # constants[2]
        x_Tmax = theta[:, 3]  # constants[3]
        K_I = theta[:, 4]  # constants[4]
        Kc_p = theta[:, 5]  # constants[5]
        Kc_pp = theta[:, 6]  # constants[6]
        Kb_p = theta[:, 7]  # constants[7]
        Kb_pp = theta[:, 8]  # constants[8]
        Pc_p = theta[:, 9]  # constants[9]
        Pc_pp = theta[:, 10]  # constants[10]
        Pb_p = theta[:, 11]  # constants[11]
        Pb_pp = theta[:, 12]  # constants[12]
        constant_16 = theta[:, 13]  # constants[16]

        # Initialize rates and algebraic variables, as in computeRates()
        d_HCO3_i = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_HCO3_i = torch.zeros_like(y[:, 0]) + (constant_16)

        return torch.stack(
            [d_HCO3_i],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
