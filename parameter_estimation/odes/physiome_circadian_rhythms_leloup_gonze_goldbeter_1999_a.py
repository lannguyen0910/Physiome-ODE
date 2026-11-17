import torch
import torch.nn as nn
import math


class CircadianRhythmsLeloupGonzeGoldbeter1999AODE(nn.Module):
    """
    PyTorch translation of circadian_rhythms/leloup_gonze_goldbeter_1999_a/model.py.
    Dynamic states: 10; independent constants: 38; derived constants used by rates: 0.
    Original constant slots: 38; algebraic slots: 1.
    """
    source_file = 'circadian_rhythms/leloup_gonze_goldbeter_1999_a/model.py'
    source_state_dim = 10
    source_constant_dim = 38
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

        self.state_dim = 10
        self.parameter_dim = 38

        self.parameters = (
            parameters
            if parameters is not None
            else {'vsP': 1, 'vmP': 0.7, 'KmP': 0.2, 'KIP': 1, 'kd': 0.01, 'n': 4, 'ksP': 0.9, 'V1P': 8, 'V2P': 1, 'K1P': 2, 'K2P': 2, 'V3P': 8, 'V4P': 1, 'K3P': 2, 'K4P': 1, 'vdP': 2, 'KdP': 0.2, 'k3': 1.2, 'k4': 0.6, 'vsT': 1, 'vmT': 0.7, 'KmT': 0.2, 'KIT': 1, 'ksT': 0.9, 'V1T': 8, 'V2T': 1, 'K1T': 2, 'K2T': 2, 'V3T': 8, 'V4T': 1, 'K3T': 2, 'K4T': 1, 'vdT': 2, 'KdT': 0.2, 'kdC': 0.01, 'k1': 0.6, 'k2': 0.2, 'kdN': 0.01}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'MP': 0.1, 'CN': 1.25, 'P0': 0.1, 'P1': 0.1, 'P2': 0.1, 'T2': 0.1, 'C': 0.1, 'MT': 1.6, 'T0': 0.1, 'T1': 0.1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['vsP', 'vmP', 'KmP', 'KIP', 'kd', 'n', 'ksP', 'V1P', 'V2P', 'K1P', 'K2P', 'V3P', 'V4P', 'K3P', 'K4P', 'vdP', 'KdP', 'k3', 'k4', 'vsT', 'vmT', 'KmT', 'KIT', 'ksT', 'V1T', 'V2T', 'K1T', 'K2T', 'V3T', 'V4T', 'K3T', 'K4T', 'vdT', 'KdT', 'kdC', 'k1', 'k2', 'kdN']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['MP', 'CN', 'P0', 'P1', 'P2', 'T2', 'C', 'MT', 'T0', 'T1']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (10)
        MP = y[:, 0]
        CN = y[:, 1]
        P0 = y[:, 2]
        P1 = y[:, 3]
        P2 = y[:, 4]
        T2 = y[:, 5]
        C = y[:, 6]
        MT = y[:, 7]
        T0 = y[:, 8]
        T1 = y[:, 9]

        # Independent parameters (source constant order)
        vsP = theta[:, 0]  # constants[0]
        vmP = theta[:, 1]  # constants[1]
        KmP = theta[:, 2]  # constants[2]
        KIP = theta[:, 3]  # constants[3]
        kd = theta[:, 4]  # constants[4]
        n = theta[:, 5]  # constants[5]
        ksP = theta[:, 6]  # constants[6]
        V1P = theta[:, 7]  # constants[7]
        V2P = theta[:, 8]  # constants[8]
        K1P = theta[:, 9]  # constants[9]
        K2P = theta[:, 10]  # constants[10]
        V3P = theta[:, 11]  # constants[11]
        V4P = theta[:, 12]  # constants[12]
        K3P = theta[:, 13]  # constants[13]
        K4P = theta[:, 14]  # constants[14]
        vdP = theta[:, 15]  # constants[15]
        KdP = theta[:, 16]  # constants[16]
        k3 = theta[:, 17]  # constants[17]
        k4 = theta[:, 18]  # constants[18]
        vsT = theta[:, 19]  # constants[19]
        vmT = theta[:, 20]  # constants[20]
        KmT = theta[:, 21]  # constants[21]
        KIT = theta[:, 22]  # constants[22]
        ksT = theta[:, 23]  # constants[23]
        V1T = theta[:, 24]  # constants[24]
        V2T = theta[:, 25]  # constants[25]
        K1T = theta[:, 26]  # constants[26]
        K2T = theta[:, 27]  # constants[27]
        V3T = theta[:, 28]  # constants[28]
        V4T = theta[:, 29]  # constants[29]
        K3T = theta[:, 30]  # constants[30]
        K4T = theta[:, 31]  # constants[31]
        vdT = theta[:, 32]  # constants[32]
        KdT = theta[:, 33]  # constants[33]
        kdC = theta[:, 34]  # constants[34]
        k1 = theta[:, 35]  # constants[35]
        k2 = theta[:, 36]  # constants[36]
        kdN = theta[:, 37]  # constants[37]

        # Initialize rates and algebraic variables, as in computeRates()
        d_MP = torch.zeros_like(y[:, 0])
        d_CN = torch.zeros_like(y[:, 0])
        d_P0 = torch.zeros_like(y[:, 0])
        d_P1 = torch.zeros_like(y[:, 0])
        d_P2 = torch.zeros_like(y[:, 0])
        d_T2 = torch.zeros_like(y[:, 0])
        d_C = torch.zeros_like(y[:, 0])
        d_MT = torch.zeros_like(y[:, 0])
        d_T0 = torch.zeros_like(y[:, 0])
        d_T1 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_MP = torch.zeros_like(y[:, 0]) + (vsP * (KIP ** n / (KIP ** n + CN ** n)) - (vmP * (MP / (KmP + MP)) + kd * MP))
        d_P0 = torch.zeros_like(y[:, 0]) + (ksP * MP + V2P * (P1 / (K2P + P1)) - (V1P * (P0 / (K1P + P0)) + kd * P0))
        d_P1 = torch.zeros_like(y[:, 0]) + (V1P * (P0 / (K1P + P0)) + V4P * (P2 / (K4P + P2)) - (V2P * (P1 / (K2P + P1)) + V3P * (P1 / (K3P + P1)) + kd * P1))
        d_P2 = torch.zeros_like(y[:, 0]) + (V3P * (P1 / (K3P + P1)) + k4 * C - (V4P * (P2 / (K4P + P2)) + k3 * P2 * T2 + vdP * (P2 / (KdP + P2)) + kd * P2))
        d_MT = torch.zeros_like(y[:, 0]) + (vsT * (KIT ** n / (KIT ** n + CN ** n)) - (vmT * (MT / (KmT + MT)) + kd * MT))
        d_T0 = torch.zeros_like(y[:, 0]) + (ksT * MT + V2T * (T1 / (K2T + T1)) - (V1T * (T0 / (K1T + T0)) + kd * T0))
        d_T1 = torch.zeros_like(y[:, 0]) + (V1T * (T0 / (K1T + T0)) + V4T * (T2 / (K4T + T2)) - (V2T * (T1 / (K2T + T1)) + V3T * (T1 / (K3T + T1)) + kd * T1))
        d_T2 = torch.zeros_like(y[:, 0]) + (V3T * (T1 / (K3T + T1)) + k4 * C - (V4T * (T2 / (K4T + T2)) + k3 * P2 * T2 + vdT * (T2 / (KdT + T2)) + kd * T2))
        d_C = torch.zeros_like(y[:, 0]) + (k3 * P2 * T2 + k2 * CN - (k4 * C + k1 * C + kdC * C))
        d_CN = torch.zeros_like(y[:, 0]) + (k1 * C - (k2 * CN + kdN * CN))

        return torch.stack(
            [d_MP, d_CN, d_P0, d_P1, d_P2, d_T2, d_C, d_MT, d_T0, d_T1],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
