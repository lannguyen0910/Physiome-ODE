import torch
import torch.nn as nn
import math


class EndocrineLemaireTobinGrellerChoSuva2004AODE(nn.Module):
    """
    PyTorch translation of endocrine/lemaire_tobin_greller_cho_suva_2004_a/model.py.
    Dynamic states: 3; independent constants: 23; derived constants used by rates: 5.
    Original constant slots: 28; algebraic slots: 3.
    """
    source_file = 'endocrine/lemaire_tobin_greller_cho_suva_2004_a/model.py'
    source_state_dim = 3
    source_constant_dim = 28
    algebraic_dim = 3
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
        self.parameter_dim = 23

        self.parameters = (
            parameters
            if parameters is not None
            else {'DR': 0.0007, 'kB': 0.189, 'DC': 0.0021, 'DA': 0.7, 'k1': 0.01, 'k2': 10, 'k3': 0.00058, 'k4': 0.017, 'K': 10, 'ko': 0.35, 'Io': 0, 'IL': 0, 'rL': 1000, 'KOP': 200000, 'KLP': 3000000, 'f0': 0.05, 'dB': 0.7, 'IP': 0, 'kP': 86, 'C_s': 0.005, 'SP': 250, 'k5': 0.02, 'k6': 3}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'R': 0.00077, 'B': 0.00073, 'C': 0.00091}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['DR', 'kB', 'DC', 'DA', 'k1', 'k2', 'k3', 'k4', 'K', 'ko', 'Io', 'IL', 'rL', 'KOP', 'KLP', 'f0', 'dB', 'IP', 'kP', 'C_s', 'SP', 'k5', 'k6']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['R', 'B', 'C']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        R = y[:, 0]
        B = y[:, 1]
        C = y[:, 2]

        # Independent parameters (source constant order)
        DR = theta[:, 0]  # constants[0]
        kB = theta[:, 1]  # constants[1]
        DC = theta[:, 2]  # constants[2]
        DA = theta[:, 3]  # constants[3]
        k1 = theta[:, 4]  # constants[4]
        k2 = theta[:, 5]  # constants[5]
        k3 = theta[:, 6]  # constants[6]
        k4 = theta[:, 7]  # constants[7]
        K = theta[:, 8]  # constants[8]
        ko = theta[:, 9]  # constants[9]
        Io = theta[:, 10]  # constants[10]
        IL = theta[:, 11]  # constants[11]
        rL = theta[:, 12]  # constants[12]
        KOP = theta[:, 13]  # constants[13]
        KLP = theta[:, 14]  # constants[14]
        f0 = theta[:, 15]  # constants[15]
        dB = theta[:, 16]  # constants[16]
        IP = theta[:, 17]  # constants[17]
        kP = theta[:, 18]  # constants[18]
        C_s = theta[:, 19]  # constants[19]
        SP = theta[:, 20]  # constants[20]
        k5 = theta[:, 21]  # constants[21]
        k6 = theta[:, 22]  # constants[22]

        # Derived constants from the original initConsts()
        DB = f0 * dB  # constants[23]
        P = IP / kP  # constants[24]
        P_0 = SP / kP  # constants[25]
        P_s = k6 / k5  # constants[26]
        pi_P = (P + P_0) / (P + P_s)  # constants[27]

        # Initialize rates and algebraic variables, as in computeRates()
        d_R = torch.zeros_like(y[:, 0])
        d_B = torch.zeros_like(y[:, 0])
        d_C = torch.zeros_like(y[:, 0])
        alg_0_f = torch.zeros_like(y[:, 0])
        alg_1_pi_L = torch.zeros_like(y[:, 0])
        alg_2_pi_C = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_f = torch.where((t > 20.0) & (t <= 80.0), 0.0001, 0.0)
        alg_2_pi_C = (C + f0 * C_s) / (C + C_s)
        d_R = torch.zeros_like(y[:, 0]) + (DR * alg_2_pi_C - DB / alg_2_pi_C * R + alg_0_f)
        d_B = torch.zeros_like(y[:, 0]) + (DB / alg_2_pi_C * R - kB * B)
        alg_1_pi_L = k3 / k4 * (KLP / 1.0) * pi_P * B / (1.0 + k3 * K / k4 + k1 / (k2 * ko) * (KOP / (1.0 * pi_P) * R + Io)) * (1.0 + IL / rL)
        d_C = torch.zeros_like(y[:, 0]) + (DC * alg_1_pi_L - DA * alg_2_pi_C * C)

        return torch.stack(
            [d_R, d_B, d_C],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
