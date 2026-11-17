import torch
import torch.nn as nn
import math


class ElectrophysiologyChangFujita1999ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/chang_fujita_1999/model.py.
    Dynamic states: 12; independent constants: 38; derived constants used by rates: 0.
    Original constant slots: 38; algebraic slots: 0.
    """
    source_file = 'electrophysiology/chang_fujita_1999/model.py'
    source_state_dim = 12
    source_constant_dim = 38
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

        self.state_dim = 12
        self.parameter_dim = 38

        self.parameters = (
            parameters
            if parameters is not None
            else {'k23': 100000, 'k24': 31.92, 'k17': 458700, 'k18': 100000, 'k1': 100000, 'k2': 418300, 'k3': 100000, 'k4': 4928000, 'D': 1e-06, 'Na': 50, 'Cl': 96, 'k29': 100000, 'k30': 0.3514, 'k11': 100000, 'k12': 4982000, 'k9': 100000, 'k10': 418300, 'D_2': 1e-06, 'Na_2': 10, 'Cl_2': 40, 'k21': 100000, 'k22': 418300, 'k27': 100000, 'k28': 138900, 'k25': 100000, 'k26': 31.92, 'k31': 100000, 'k32': 0.1166, 'k5': 100000, 'k6': 1065000, 'k13': 100000, 'k14': 1065000, 'k7': 100000, 'k8': 89400, 'k15': 100000, 'k16': 89400, 'k19': 1000, 'k20': 218}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'E': 0.08333, 'ED': 0.08333, 'E_2': 0.08333, 'ENa': 0.08333, 'ECl': 0.08333, 'ED_2': 0.08333, 'ENa_2': 0.08333, 'ECl_2': 0.08333, 'ENaD': 0.08333, 'ENaD_2': 0.08333, 'ENaCl': 0.08333, 'ENaCl_2': 0.08333}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k23', 'k24', 'k17', 'k18', 'k1', 'k2', 'k3', 'k4', 'D', 'Na', 'Cl', 'k29', 'k30', 'k11', 'k12', 'k9', 'k10', 'D_2', 'Na_2', 'Cl_2', 'k21', 'k22', 'k27', 'k28', 'k25', 'k26', 'k31', 'k32', 'k5', 'k6', 'k13', 'k14', 'k7', 'k8', 'k15', 'k16', 'k19', 'k20']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['E', 'ED', 'E_2', 'ENa', 'ECl', 'ED_2', 'ENa_2', 'ECl_2', 'ENaD', 'ENaD_2', 'ENaCl', 'ENaCl_2']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (12)
        E = y[:, 0]
        ED = y[:, 1]
        E_2 = y[:, 2]
        ENa = y[:, 3]
        ECl = y[:, 4]
        ED_2 = y[:, 5]
        ENa_2 = y[:, 6]
        ECl_2 = y[:, 7]
        ENaD = y[:, 8]
        ENaD_2 = y[:, 9]
        ENaCl = y[:, 10]
        ENaCl_2 = y[:, 11]

        # Independent parameters (source constant order)
        k23 = theta[:, 0]  # constants[0]
        k24 = theta[:, 1]  # constants[1]
        k17 = theta[:, 2]  # constants[2]
        k18 = theta[:, 3]  # constants[3]
        k1 = theta[:, 4]  # constants[4]
        k2 = theta[:, 5]  # constants[5]
        k3 = theta[:, 6]  # constants[6]
        k4 = theta[:, 7]  # constants[7]
        D = theta[:, 8]  # constants[8]
        Na = theta[:, 9]  # constants[9]
        Cl = theta[:, 10]  # constants[10]
        k29 = theta[:, 11]  # constants[11]
        k30 = theta[:, 12]  # constants[12]
        k11 = theta[:, 13]  # constants[13]
        k12 = theta[:, 14]  # constants[14]
        k9 = theta[:, 15]  # constants[15]
        k10 = theta[:, 16]  # constants[16]
        D_2 = theta[:, 17]  # constants[17]
        Na_2 = theta[:, 18]  # constants[18]
        Cl_2 = theta[:, 19]  # constants[19]
        k21 = theta[:, 20]  # constants[20]
        k22 = theta[:, 21]  # constants[21]
        k27 = theta[:, 22]  # constants[22]
        k28 = theta[:, 23]  # constants[23]
        k25 = theta[:, 24]  # constants[24]
        k26 = theta[:, 25]  # constants[25]
        k31 = theta[:, 26]  # constants[26]
        k32 = theta[:, 27]  # constants[27]
        k5 = theta[:, 28]  # constants[28]
        k6 = theta[:, 29]  # constants[29]
        k13 = theta[:, 30]  # constants[30]
        k14 = theta[:, 31]  # constants[31]
        k7 = theta[:, 32]  # constants[32]
        k8 = theta[:, 33]  # constants[33]
        k15 = theta[:, 34]  # constants[34]
        k16 = theta[:, 35]  # constants[35]
        k19 = theta[:, 36]  # constants[36]
        k20 = theta[:, 37]  # constants[37]

        # Initialize rates and algebraic variables, as in computeRates()
        d_E = torch.zeros_like(y[:, 0])
        d_ED = torch.zeros_like(y[:, 0])
        d_E_2 = torch.zeros_like(y[:, 0])
        d_ENa = torch.zeros_like(y[:, 0])
        d_ECl = torch.zeros_like(y[:, 0])
        d_ED_2 = torch.zeros_like(y[:, 0])
        d_ENa_2 = torch.zeros_like(y[:, 0])
        d_ECl_2 = torch.zeros_like(y[:, 0])
        d_ENaD = torch.zeros_like(y[:, 0])
        d_ENaD_2 = torch.zeros_like(y[:, 0])
        d_ENaCl = torch.zeros_like(y[:, 0])
        d_ENaCl_2 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_E = torch.zeros_like(y[:, 0]) + (k24 * ED + k18 * E_2 + k2 * ENa + k4 * ECl - (k23 * D * E + k17 * E + k1 * Na * E + k3 * Cl * E))
        d_E_2 = torch.zeros_like(y[:, 0]) + (k30 * ED_2 + k17 * E + k10 * ENa_2 + k12 * ECl_2 - (k29 * D_2 * E_2 + k18 * E_2 + k9 * Na_2 * E_2 + k11 * Cl_2 * E_2))
        d_ED = torch.zeros_like(y[:, 0]) + (k23 * E * D + k22 * ENaD - (k24 * ED + k21 * Na * ED))
        d_ED_2 = torch.zeros_like(y[:, 0]) + (k29 * E_2 * D_2 + k28 * ENaD_2 - (k30 * ED_2 + k27 * Na_2 * ED_2))
        d_ENaD = torch.zeros_like(y[:, 0]) + (k21 * Na * ED + k25 * D * ENa - (k22 * ENaD + k26 * ENaD))
        d_ENaD_2 = torch.zeros_like(y[:, 0]) + (k27 * Na_2 * ED_2 + k31 * D_2 * ENa_2 - (k28 * ENaD_2 + k32 * ENaD_2))
        d_ENa = torch.zeros_like(y[:, 0]) + (k1 * Na * E + k26 * ENaD + k6 * ENaCl - (k2 * ENa + k5 * Cl * ENa + k25 * D * ENa))
        d_ENa_2 = torch.zeros_like(y[:, 0]) + (k9 * Na_2 * E_2 + k32 * ENaD_2 + k14 * ENaCl_2 - (k10 * ENa_2 + k13 * Cl_2 * ENa_2 + k31 * D_2 * ENa_2))
        d_ECl = torch.zeros_like(y[:, 0]) + (k3 * Cl * E + k8 * ENaCl - (k7 * Na * ECl + k4 * ECl))
        d_ECl_2 = torch.zeros_like(y[:, 0]) + (k11 * Cl_2 * E_2 + k16 * ENaCl_2 - (k15 * Na_2 * ECl_2 + k12 * ECl_2))
        d_ENaCl = torch.zeros_like(y[:, 0]) + (k5 * Cl * ENa + k7 * Na * ECl + k20 * ENaCl_2 - (k6 * ENaCl + k8 * ENaCl + k19 * ENaCl))
        d_ENaCl_2 = torch.zeros_like(y[:, 0]) + (k13 * Cl_2 * ENa_2 + k15 * Na_2 * ECl_2 + k19 * ENaCl - (k14 * ENaCl_2 + k16 * ENaCl_2 + k20 * ENaCl_2))

        return torch.stack(
            [d_E, d_ED, d_E_2, d_ENa, d_ECl, d_ED_2, d_ENa_2, d_ECl_2, d_ENaD, d_ENaD_2, d_ENaCl, d_ENaCl_2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
