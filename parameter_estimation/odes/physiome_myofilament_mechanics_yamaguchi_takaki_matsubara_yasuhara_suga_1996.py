import torch
import torch.nn as nn
import math


class MyofilamentMechanicsYamaguchiTakakiMatsubaraYasuharaSuga1996ODE(nn.Module):
    """
    PyTorch translation of myofilament_mechanics/yamaguchi_takaki_matsubara_yasuhara_suga_1996/model.py.
    Dynamic states: 5; independent constants: 15; derived constants used by rates: 0.
    Original constant slots: 20; algebraic slots: 2.
    """
    source_file = 'myofilament_mechanics/yamaguchi_takaki_matsubara_yasuhara_suga_1996/model.py'
    source_state_dim = 5
    source_constant_dim = 20
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

        self.state_dim = 5
        self.parameter_dim = 15

        self.parameters = (
            parameters
            if parameters is not None
            else {'D_Ca': 0.1, 'k_1': 40, 'k_2': 20, 'f': 10, 'g': 10, 'Ca_max': 1, 'Total_Tn': 1, 'Total_CB': 1, 'phi': 1, 's': 1, 'L': 1, 'L_0': 0, 'F_max': 0.228, 'epsilon': 1, 'CumCB_on_end': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'TnCa': 0, 'CB_on': 0, 'CumCB_on': 0, 'CumCB_off': 0, 'FTI': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['D_Ca', 'k_1', 'k_2', 'f', 'g', 'Ca_max', 'Total_Tn', 'Total_CB', 'phi', 's', 'L', 'L_0', 'F_max', 'epsilon', 'CumCB_on_end']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['TnCa', 'CB_on', 'CumCB_on', 'CumCB_off', 'FTI']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        TnCa = y[:, 0]
        CB_on = y[:, 1]
        CumCB_on = y[:, 2]
        CumCB_off = y[:, 3]
        FTI = y[:, 4]

        # Independent parameters (source constant order)
        D_Ca = theta[:, 0]  # constants[0]
        k_1 = theta[:, 1]  # constants[1]
        k_2 = theta[:, 2]  # constants[2]
        f = theta[:, 3]  # constants[3]
        g = theta[:, 4]  # constants[4]
        Ca_max = theta[:, 5]  # constants[5]
        Total_Tn = theta[:, 6]  # constants[6]
        Total_CB = theta[:, 7]  # constants[7]
        phi = theta[:, 8]  # constants[8]
        s = theta[:, 9]  # constants[9]
        L = theta[:, 10]  # constants[10]
        L_0 = theta[:, 11]  # constants[11]
        F_max = theta[:, 12]  # constants[12]
        epsilon = theta[:, 13]  # constants[13]
        CumCB_on_end = theta[:, 14]  # constants[14]

        # Initialize rates and algebraic variables, as in computeRates()
        d_TnCa = torch.zeros_like(y[:, 0])
        d_CB_on = torch.zeros_like(y[:, 0])
        d_CumCB_on = torch.zeros_like(y[:, 0])
        d_CumCB_off = torch.zeros_like(y[:, 0])
        d_FTI = torch.zeros_like(y[:, 0])
        alg_0_Ca_t = torch.zeros_like(y[:, 0])
        alg_1_F = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_CB_on = torch.zeros_like(y[:, 0]) + (f * TnCa * (Total_CB - CB_on) - g * CB_on)
        d_CumCB_on = torch.zeros_like(y[:, 0]) + (f * TnCa * (Total_CB - CB_on))
        d_CumCB_off = torch.zeros_like(y[:, 0]) + (g * CB_on)
        alg_0_Ca_t = torch.where((t >= 0.0) & (t < 0.3 * D_Ca), Ca_max * (1.0 + torch.sin(math.pi * (t / D_Ca - 0.15) / 0.3)) / 2.0, torch.where((t >= 0.3 * D_Ca) & (t < D_Ca), Ca_max * (1.0 - torch.sin(math.pi * (t / D_Ca - 0.65) / 0.7)) / 2.0, 0.0))
        d_TnCa = torch.zeros_like(y[:, 0]) + (k_1 * alg_0_Ca_t * (Total_Tn - TnCa) - k_2 * TnCa)
        alg_1_F = CB_on * phi
        d_FTI = torch.zeros_like(y[:, 0]) + (alg_1_F)

        return torch.stack(
            [d_TnCa, d_CB_on, d_CumCB_on, d_CumCB_off, d_FTI],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
