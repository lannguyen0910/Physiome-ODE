import torch
import torch.nn as nn
import math


class SignalTransductionSedaghatShermanQuon2002ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/sedaghat_sherman_quon_2002/model.py.
    Dynamic states: 20; independent constants: 22; derived constants used by rates: 11.
    Original constant slots: 33; algebraic slots: 7.
    """
    source_file = 'signal_transduction/sedaghat_sherman_quon_2002/model.py'
    source_state_dim = 20
    source_constant_dim = 33
    algebraic_dim = 7
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 20
        self.parameter_dim = 22

        self.parameters = (
            parameters
            if parameters is not None
            else {'k1': 60000000, 'k_minus1': 0.2, 'k_minus4': 0.003, 'PTP': 1, 'k3': 2500, 'k4b': 0.0021, 'k_minus4b': 0.00021, 'k_minus5': 1.67e-18, 'k6': 0.461, 'k7': 4.16, 'IRp': 8.97e-13, 'k_minus8': 10, 'k_minus10': 2.77, 'PTEN': 1, 'SHIP': 1, 'k_minus13': 0.167, 'k_minus14': 0.001155, 'k9_stimulated': 1.39, 'APequil': 9.09, 'PI3K': 5e-15, 'k_minus11': 6.931471805599453, 'k_minus12': 6.931471805599453}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'x2': 9e-13, 'x3': 0, 'x5': 0, 'x6': 1e-13, 'x4': 0, 'x7': 0, 'x8': 0, 'x9': 1e-13, 'x10': 1e-13, 'x11': 1e-13, 'x12': 0, 'x13': 0.31, 'x14': 99.4, 'x15': 0.29, 'x16': 100, 'x17': 0, 'x18': 100, 'x19': 0, 'x20': 96, 'x21': 4}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k1', 'k_minus1', 'k_minus4', 'PTP', 'k3', 'k4b', 'k_minus4b', 'k_minus5', 'k6', 'k7', 'IRp', 'k_minus8', 'k_minus10', 'PTEN', 'SHIP', 'k_minus13', 'k_minus14', 'k9_stimulated', 'APequil', 'PI3K', 'k_minus11', 'k_minus12']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['x2', 'x3', 'x5', 'x6', 'x4', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (20)
        x2 = y[:, 0]
        x3 = y[:, 1]
        x5 = y[:, 2]
        x6 = y[:, 3]
        x4 = y[:, 4]
        x7 = y[:, 5]
        x8 = y[:, 6]
        x9 = y[:, 7]
        x10 = y[:, 8]
        x11 = y[:, 9]
        x12 = y[:, 10]
        x13 = y[:, 11]
        x14 = y[:, 12]
        x15 = y[:, 13]
        x16 = y[:, 14]
        x17 = y[:, 15]
        x18 = y[:, 16]
        x19 = y[:, 17]
        x20 = y[:, 18]
        x21 = y[:, 19]

        # Independent parameters (source constant order)
        k1 = theta[:, 0]  # constants[0]
        k_minus1 = theta[:, 1]  # constants[1]
        k_minus4 = theta[:, 2]  # constants[2]
        PTP = theta[:, 3]  # constants[3]
        k3 = theta[:, 4]  # constants[4]
        k4b = theta[:, 5]  # constants[5]
        k_minus4b = theta[:, 6]  # constants[6]
        k_minus5 = theta[:, 7]  # constants[7]
        k6 = theta[:, 8]  # constants[8]
        k7 = theta[:, 9]  # constants[9]
        IRp = theta[:, 10]  # constants[10]
        k_minus8 = theta[:, 11]  # constants[11]
        k_minus10 = theta[:, 12]  # constants[12]
        PTEN = theta[:, 13]  # constants[13]
        SHIP = theta[:, 14]  # constants[14]
        k_minus13 = theta[:, 15]  # constants[15]
        k_minus14 = theta[:, 16]  # constants[16]
        k9_stimulated = theta[:, 17]  # constants[17]
        APequil = theta[:, 18]  # constants[18]
        PI3K = theta[:, 19]  # constants[19]
        k_minus11 = theta[:, 20]  # constants[28]
        k_minus12 = theta[:, 21]  # constants[29]

        # Derived constants from the original initConsts()
        k4 = k_minus4 / 9.0  # constants[20]
        k_minus3 = k_minus1 / 1.0  # constants[21]
        k2 = k1  # constants[22]
        k_minus2 = 100.0 * k_minus1  # constants[23]
        k_minus7 = 2.5 / 7.45 * k7  # constants[24]
        k8 = k_minus8 * 5.0 / 70.775 * 1000000000000.0  # constants[25]
        k_minus9 = 94.0 / 3.1 * k9_stimulated  # constants[26]
        k10 = 3.1 / 2.9 * k_minus10  # constants[27]
        k13 = 4.0 / 96.0 * k_minus13  # constants[30]
        k14 = k_minus14 / 96.0  # constants[31]
        k9_basal = 0.31 / 99.4 * k_minus9  # constants[32]

        # Initialize rates and algebraic variables, as in computeRates()
        d_x2 = torch.zeros_like(y[:, 0])
        d_x3 = torch.zeros_like(y[:, 0])
        d_x5 = torch.zeros_like(y[:, 0])
        d_x6 = torch.zeros_like(y[:, 0])
        d_x4 = torch.zeros_like(y[:, 0])
        d_x7 = torch.zeros_like(y[:, 0])
        d_x8 = torch.zeros_like(y[:, 0])
        d_x9 = torch.zeros_like(y[:, 0])
        d_x10 = torch.zeros_like(y[:, 0])
        d_x11 = torch.zeros_like(y[:, 0])
        d_x12 = torch.zeros_like(y[:, 0])
        d_x13 = torch.zeros_like(y[:, 0])
        d_x14 = torch.zeros_like(y[:, 0])
        d_x15 = torch.zeros_like(y[:, 0])
        d_x16 = torch.zeros_like(y[:, 0])
        d_x17 = torch.zeros_like(y[:, 0])
        d_x18 = torch.zeros_like(y[:, 0])
        d_x19 = torch.zeros_like(y[:, 0])
        d_x20 = torch.zeros_like(y[:, 0])
        d_x21 = torch.zeros_like(y[:, 0])
        alg_0_x1 = torch.zeros_like(y[:, 0])
        alg_1_k5 = torch.zeros_like(y[:, 0])
        alg_2_k9 = torch.zeros_like(y[:, 0])
        alg_3_k11 = torch.zeros_like(y[:, 0])
        alg_4_k12 = torch.zeros_like(y[:, 0])
        alg_5_effect = torch.zeros_like(y[:, 0])
        alg_6_k13b = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_x7 = torch.zeros_like(y[:, 0]) + (k4b * x4 - (k_minus4b * x7 + k6 * PTP * x7))
        d_x8 = torch.zeros_like(y[:, 0]) + (k4b * x5 - (k_minus4b * x8 + k6 * PTP * x8))
        d_x9 = torch.zeros_like(y[:, 0]) + (k_minus7 * PTP * x10 - k7 * x9 * (x4 + x5) / IRp)
        d_x10 = torch.zeros_like(y[:, 0]) + (k7 * x9 * (x4 + x5) / IRp + k_minus8 * x12 - (k_minus7 * PTP + k8 * x11) * x10)
        d_x11 = torch.zeros_like(y[:, 0]) + (k_minus8 * x12 - k8 * x10 * x11)
        d_x12 = torch.zeros_like(y[:, 0]) + (k8 * x10 * x11 - k_minus8 * x12)
        d_x15 = torch.zeros_like(y[:, 0]) + (k_minus10 * SHIP * x13 - k10 * x15)
        alg_0_x1 = torch.where(t < 15.0, 1e-07, 0.0)
        d_x2 = torch.zeros_like(y[:, 0]) + (k_minus1 * x3 + k_minus3 * PTP * x5 + k_minus4 * x6 - (k1 * alg_0_x1 * x2 + k4 * x2))
        d_x3 = torch.zeros_like(y[:, 0]) + (k1 * alg_0_x1 * x2 - (k_minus1 * x3 + k3 * x3))
        d_x4 = torch.zeros_like(y[:, 0]) + (k2 * alg_0_x1 * x5 + k_minus4b * x7 - (k_minus2 * x4 + k4b * x4))
        d_x5 = torch.zeros_like(y[:, 0]) + (k3 * x3 + k_minus2 * x4 + k_minus4b * x8 - (k2 * alg_0_x1 * x5 + k_minus3 * PTP * x5 + k4b * x5))
        alg_1_k5 = torch.where(x6 + x7 + x8 > 1e-13, 10.0 * k_minus5, 60.0 * k_minus5)
        d_x6 = torch.zeros_like(y[:, 0]) + (alg_1_k5 + k6 * PTP * (x7 + x8) + k4 * x2 - (k_minus5 * x6 + k_minus4 * x6))
        alg_2_k9 = (k9_stimulated - k9_basal) * x12 / PI3K + k9_basal
        d_x13 = torch.zeros_like(y[:, 0]) + (alg_2_k9 * x14 + k10 * x15 - (k_minus9 * PTEN + k_minus10 * SHIP) * x13)
        d_x14 = torch.zeros_like(y[:, 0]) + (k_minus9 * PTEN * x13 - alg_2_k9 * x14)
        alg_3_k11 = 0.1 * k_minus11 * (x13 - 0.31) / (3.1 - 0.31)
        d_x16 = torch.zeros_like(y[:, 0]) + (k_minus11 * x17 - alg_3_k11 * x16)
        d_x17 = torch.zeros_like(y[:, 0]) + (alg_3_k11 * x16 - k_minus11 * x17)
        alg_4_k12 = 0.1 * k_minus12 * (x13 - 0.31) / (3.1 - 0.31)
        d_x18 = torch.zeros_like(y[:, 0]) + (k_minus12 * x19 - alg_4_k12 * x18)
        d_x19 = torch.zeros_like(y[:, 0]) + (alg_4_k12 * x18 - k_minus12 * x19)
        alg_5_effect = (0.2 * x17 + 0.8 * x19) / APequil
        alg_6_k13b = (40.0 / 60.0 - 4.0 / 96.0) * k_minus13 * alg_5_effect
        d_x20 = torch.zeros_like(y[:, 0]) + (k_minus13 * x21 + k14 - ((k13 + alg_6_k13b) * x20 + k_minus14 * x20))
        d_x21 = torch.zeros_like(y[:, 0]) + (-k_minus13 * x21 + (k13 + alg_6_k13b) * x20)

        return torch.stack(
            [d_x2, d_x3, d_x5, d_x6, d_x4, d_x7, d_x8, d_x9, d_x10, d_x11, d_x12, d_x13, d_x14, d_x15, d_x16, d_x17, d_x18, d_x19, d_x20, d_x21],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
