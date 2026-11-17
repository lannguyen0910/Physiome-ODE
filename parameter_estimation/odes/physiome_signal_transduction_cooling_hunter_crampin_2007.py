import torch
import torch.nn as nn
import math


class SignalTransductionCoolingHunterCrampin2007ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/cooling_hunter_crampin_2007/model.py.
    Dynamic states: 13; independent constants: 31; derived constants used by rates: 7.
    Original constant slots: 38; algebraic slots: 17.
    """
    source_file = 'signal_transduction/cooling_hunter_crampin_2007/model.py'
    source_state_dim = 13
    source_constant_dim = 38
    algebraic_dim = 17
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 13
        self.parameter_dim = 31

        self.parameters = (
            parameters
            if parameters is not None
            else {'Ls': 0.1, 'ts': 10, 'PIP2': 4000, 'kf1': 0.0003, 'Kd1': 3e-05, 'kf2': 0.000275, 'Kd2': 27500, 'kf3': 1, 'kr3': 0.001, 'kf4': 0.3, 'Kd4': 3e-05, 'kf5': 0.0004, 'kf6': 1, 'kf7': 0.15, 'kf8': 0.0167, 'kr8': 0.0167, 'kf9': 0.0042, 'kr9': 1, 'kf10': 0.042, 'kr10': 1, 'kf11': 0.0334, 'Kd11': 0.1, 'kf12': 6, 'kf13': 6, 'kf14': 0.444, 'Km14': 19.8, 'kf15': 3.8, 'Km15': 5, 'kf16': 1.25, 'Vc': 2550, 'Rpc': 4.61}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Ca': 0.1, 'Gd': 10000, 'Gt': 0, 'R': 13.9, 'Rl': 0, 'Rg': 5.06, 'Rlg': 0, 'Rlgp': 0, 'IP3': 0.015, 'Pc': 9.09, 'Pcg': 0, 'P': 90.9, 'Pg': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Ls', 'ts', 'PIP2', 'kf1', 'Kd1', 'kf2', 'Kd2', 'kf3', 'kr3', 'kf4', 'Kd4', 'kf5', 'kf6', 'kf7', 'kf8', 'kr8', 'kf9', 'kr9', 'kf10', 'kr10', 'kf11', 'Kd11', 'kf12', 'kf13', 'kf14', 'Km14', 'kf15', 'Km15', 'kf16', 'Vc', 'Rpc']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Ca', 'Gd', 'Gt', 'R', 'Rl', 'Rg', 'Rlg', 'Rlgp', 'IP3', 'Pc', 'Pcg', 'P', 'Pg']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (13)
        Ca = y[:, 0]
        Gd = y[:, 1]
        Gt = y[:, 2]
        R = y[:, 3]
        Rl = y[:, 4]
        Rg = y[:, 5]
        Rlg = y[:, 6]
        Rlgp = y[:, 7]
        IP3 = y[:, 8]
        Pc = y[:, 9]
        Pcg = y[:, 10]
        P = y[:, 11]
        Pg = y[:, 12]

        # Independent parameters (source constant order)
        Ls = theta[:, 0]  # constants[0]
        ts = theta[:, 1]  # constants[1]
        PIP2 = theta[:, 2]  # constants[2]
        kf1 = theta[:, 3]  # constants[3]
        Kd1 = theta[:, 4]  # constants[4]
        kf2 = theta[:, 5]  # constants[5]
        Kd2 = theta[:, 6]  # constants[6]
        kf3 = theta[:, 7]  # constants[7]
        kr3 = theta[:, 8]  # constants[8]
        kf4 = theta[:, 9]  # constants[9]
        Kd4 = theta[:, 10]  # constants[10]
        kf5 = theta[:, 11]  # constants[11]
        kf6 = theta[:, 12]  # constants[12]
        kf7 = theta[:, 13]  # constants[13]
        kf8 = theta[:, 14]  # constants[14]
        kr8 = theta[:, 15]  # constants[15]
        kf9 = theta[:, 16]  # constants[16]
        kr9 = theta[:, 17]  # constants[17]
        kf10 = theta[:, 18]  # constants[18]
        kr10 = theta[:, 19]  # constants[19]
        kf11 = theta[:, 20]  # constants[20]
        Kd11 = theta[:, 21]  # constants[21]
        kf12 = theta[:, 22]  # constants[22]
        kf13 = theta[:, 23]  # constants[23]
        kf14 = theta[:, 24]  # constants[24]
        Km14 = theta[:, 25]  # constants[25]
        kf15 = theta[:, 26]  # constants[26]
        Km15 = theta[:, 27]  # constants[27]
        kf16 = theta[:, 28]  # constants[28]
        Vc = theta[:, 29]  # constants[29]
        Rpc = theta[:, 30]  # constants[30]

        # Derived constants from the original initConsts()
        kr1 = kf1 * Kd1  # constants[31]
        kr2 = kf2 * Kd2  # constants[32]
        kr4 = kf4 * Kd4  # constants[33]
        kr11 = kf11 * Kd11  # constants[34]
        Cc = 1.0 / (Vc * 602.2)  # constants[35]
        Cp = 1.0 / (Vc * Rpc)  # constants[36]
        Cpc = Cc / Cp  # constants[37]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Ca = torch.zeros_like(y[:, 0])
        d_Gd = torch.zeros_like(y[:, 0])
        d_Gt = torch.zeros_like(y[:, 0])
        d_R = torch.zeros_like(y[:, 0])
        d_Rl = torch.zeros_like(y[:, 0])
        d_Rg = torch.zeros_like(y[:, 0])
        d_Rlg = torch.zeros_like(y[:, 0])
        d_Rlgp = torch.zeros_like(y[:, 0])
        d_IP3 = torch.zeros_like(y[:, 0])
        d_Pc = torch.zeros_like(y[:, 0])
        d_Pcg = torch.zeros_like(y[:, 0])
        d_P = torch.zeros_like(y[:, 0])
        d_Pg = torch.zeros_like(y[:, 0])
        alg_0_J13 = torch.zeros_like(y[:, 0])
        alg_1_J16 = torch.zeros_like(y[:, 0])
        alg_2_J12 = torch.zeros_like(y[:, 0])
        alg_3_J14 = torch.zeros_like(y[:, 0])
        alg_4_J11 = torch.zeros_like(y[:, 0])
        alg_5_J15 = torch.zeros_like(y[:, 0])
        alg_6_J10 = torch.zeros_like(y[:, 0])
        alg_7_J8 = torch.zeros_like(y[:, 0])
        alg_8_J9 = torch.zeros_like(y[:, 0])
        alg_9_J7 = torch.zeros_like(y[:, 0])
        alg_10_L = torch.zeros_like(y[:, 0])
        alg_11_J1 = torch.zeros_like(y[:, 0])
        alg_12_J2 = torch.zeros_like(y[:, 0])
        alg_13_J3 = torch.zeros_like(y[:, 0])
        alg_14_J4 = torch.zeros_like(y[:, 0])
        alg_15_J5 = torch.zeros_like(y[:, 0])
        alg_16_J6 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_3_J14 = kf14 * Pc * PIP2 / (Km14 / Cpc + PIP2)
        alg_5_J15 = kf15 * Pcg * PIP2 / (Km15 / Cpc + PIP2)
        alg_1_J16 = kf16 * IP3
        d_IP3 = torch.zeros_like(y[:, 0]) + (Cpc * (alg_3_J14 + alg_5_J15) - alg_1_J16)
        alg_6_J10 = kf10 * Pc * Gt - kr10 * Pcg
        alg_4_J11 = kf11 * Pg * Ca - kr11 * Pcg
        alg_2_J12 = kf12 * Pcg
        d_Pcg = torch.zeros_like(y[:, 0]) + (alg_6_J10 + alg_4_J11 - alg_2_J12)
        alg_7_J8 = kf8 * P * Ca - kr8 * Pc
        d_Pc = torch.zeros_like(y[:, 0]) + (alg_7_J8 + alg_2_J12 - alg_6_J10)
        d_Ca = torch.zeros_like(y[:, 0]) + (Cpc * -1.0 * (alg_7_J8 + alg_4_J11))
        alg_8_J9 = kf9 * P * Gt - kr9 * Pg
        alg_0_J13 = kf13 * Pg
        d_P = torch.zeros_like(y[:, 0]) + (alg_0_J13 - (alg_8_J9 + alg_7_J8))
        d_Pg = torch.zeros_like(y[:, 0]) + (alg_8_J9 - (alg_4_J11 + alg_0_J13))
        alg_10_L = torch.where((t < ts + 0.15) & (t >= ts), Ls / (1.0 + torch.exp(-80.0 * (t - ts - 0.05))), torch.where(t >= ts + 0.15, Ls, 0.0))
        alg_11_J1 = kf1 * R * alg_10_L - kr1 * Rl
        alg_12_J2 = kf2 * R * Gd - kr2 * Rg
        d_R = torch.zeros_like(y[:, 0]) + (-1.0 * (alg_11_J1 + alg_12_J2))
        alg_13_J3 = kf3 * Rl * Gd - kr3 * Rlg
        alg_9_J7 = kf7 * Gt
        d_Gd = torch.zeros_like(y[:, 0]) + (alg_9_J7 + alg_0_J13 + alg_2_J12 - (alg_12_J2 + alg_13_J3))
        alg_14_J4 = kf4 * alg_10_L * Rg - kr4 * Rlg
        d_Rg = torch.zeros_like(y[:, 0]) + (alg_12_J2 - alg_14_J4)
        alg_15_J5 = kf5 * Rlg
        d_Rlgp = torch.zeros_like(y[:, 0]) + (alg_15_J5)
        alg_16_J6 = kf6 * Rlg
        d_Gt = torch.zeros_like(y[:, 0]) + (alg_16_J6 - (alg_9_J7 + alg_8_J9 + alg_6_J10))
        d_Rl = torch.zeros_like(y[:, 0]) + (alg_11_J1 + alg_16_J6 - alg_13_J3)
        d_Rlg = torch.zeros_like(y[:, 0]) + (alg_13_J3 - alg_15_J5 + alg_14_J4 - alg_16_J6)

        return torch.stack(
            [d_Ca, d_Gd, d_Gt, d_R, d_Rl, d_Rg, d_Rlg, d_Rlgp, d_IP3, d_Pc, d_Pcg, d_P, d_Pg],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
