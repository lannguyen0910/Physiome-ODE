import torch
import torch.nn as nn
import math


class CellCycleNovakTyson1997ODE(nn.Module):
    """
    PyTorch translation of cell_cycle/novak_tyson_1997/model.py.
    Dynamic states: 13; independent constants: 43; derived constants used by rates: 0.
    Original constant slots: 43; algebraic slots: 9.
    """
    source_file = 'cell_cycle/novak_tyson_1997/model.py'
    source_state_dim = 13
    source_constant_dim = 43
    algebraic_dim = 9
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
        self.parameter_dim = 43

        self.parameters = (
            parameters
            if parameters is not None
            else {'k1': 0.015, 'k2': 0.05, 'k3': 0.09375, 'k4': 0.1875, 'k5': 0.00175, 'k6': 0, 'k7': 100, 'k7r': 0.1, 'k8': 10, 'k8r': 0.1, 'ku': 0.2, 'kp': 3.25, 'kur': 0.1, 'ku2': 1, 'kur2': 0.3, 'kc': 1, 'kcr': 0.25, 'kw': 1, 'kwr': 0.25, 'ki': 0.4, 'kir': 0.1, 'mu': 0.00495, 'Kmu2': 0.05, 'Kmur2': 0.05, 'Kmi': 0.01, 'Kmir': 0.01, 'Kmw': 0.1, 'Kmwr': 0.1, 'Kmu': 0.01, 'Kmur': 0.01, 'Kmc': 0.1, 'Kmcr': 0.1, 'Kmp': 0.001, 'V2': 0.25, 'V2_2': 0.0075, 'V6': 7.5, 'V6_2': 0.0375, 'V25': 0.5, 'V25_2': 0.025, 'Vw': 0.35, 'Vw_2': 0.035, 'alpha': 0.25, 'beta': 0.05}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Cdc25': 0, 'UbE': 1, 'UbE2': 0, 'Wee1': 0, 'G1K': 0, 'G2K': 0, 'PG2': 0, 'G2R': 0, 'R': 0.4, 'G1R': 0, 'PG2R': 0, 'mass': 0.5, 'IE': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k7r', 'k8', 'k8r', 'ku', 'kp', 'kur', 'ku2', 'kur2', 'kc', 'kcr', 'kw', 'kwr', 'ki', 'kir', 'mu', 'Kmu2', 'Kmur2', 'Kmi', 'Kmir', 'Kmw', 'Kmwr', 'Kmu', 'Kmur', 'Kmc', 'Kmcr', 'Kmp', 'V2', 'V2_2', 'V6', 'V6_2', 'V25', 'V25_2', 'Vw', 'Vw_2', 'alpha', 'beta']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Cdc25', 'UbE', 'UbE2', 'Wee1', 'G1K', 'G2K', 'PG2', 'G2R', 'R', 'G1R', 'PG2R', 'mass', 'IE']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (13)
        Cdc25 = y[:, 0]
        UbE = y[:, 1]
        UbE2 = y[:, 2]
        Wee1 = y[:, 3]
        G1K = y[:, 4]
        G2K = y[:, 5]
        PG2 = y[:, 6]
        G2R = y[:, 7]
        R = y[:, 8]
        G1R = y[:, 9]
        PG2R = y[:, 10]
        mass = y[:, 11]
        IE = y[:, 12]

        # Independent parameters (source constant order)
        k1 = theta[:, 0]  # constants[0]
        k2 = theta[:, 1]  # constants[1]
        k3 = theta[:, 2]  # constants[2]
        k4 = theta[:, 3]  # constants[3]
        k5 = theta[:, 4]  # constants[4]
        k6 = theta[:, 5]  # constants[5]
        k7 = theta[:, 6]  # constants[6]
        k7r = theta[:, 7]  # constants[7]
        k8 = theta[:, 8]  # constants[8]
        k8r = theta[:, 9]  # constants[9]
        ku = theta[:, 10]  # constants[10]
        kp = theta[:, 11]  # constants[11]
        kur = theta[:, 12]  # constants[12]
        ku2 = theta[:, 13]  # constants[13]
        kur2 = theta[:, 14]  # constants[14]
        kc = theta[:, 15]  # constants[15]
        kcr = theta[:, 16]  # constants[16]
        kw = theta[:, 17]  # constants[17]
        kwr = theta[:, 18]  # constants[18]
        ki = theta[:, 19]  # constants[19]
        kir = theta[:, 20]  # constants[20]
        mu = theta[:, 21]  # constants[21]
        Kmu2 = theta[:, 22]  # constants[22]
        Kmur2 = theta[:, 23]  # constants[23]
        Kmi = theta[:, 24]  # constants[24]
        Kmir = theta[:, 25]  # constants[25]
        Kmw = theta[:, 26]  # constants[26]
        Kmwr = theta[:, 27]  # constants[27]
        Kmu = theta[:, 28]  # constants[28]
        Kmur = theta[:, 29]  # constants[29]
        Kmc = theta[:, 30]  # constants[30]
        Kmcr = theta[:, 31]  # constants[31]
        Kmp = theta[:, 32]  # constants[32]
        V2 = theta[:, 33]  # constants[33]
        V2_2 = theta[:, 34]  # constants[34]
        V6 = theta[:, 35]  # constants[35]
        V6_2 = theta[:, 36]  # constants[36]
        V25 = theta[:, 37]  # constants[37]
        V25_2 = theta[:, 38]  # constants[38]
        Vw = theta[:, 39]  # constants[39]
        Vw_2 = theta[:, 40]  # constants[40]
        alpha = theta[:, 41]  # constants[41]
        beta = theta[:, 42]  # constants[42]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Cdc25 = torch.zeros_like(y[:, 0])
        d_UbE = torch.zeros_like(y[:, 0])
        d_UbE2 = torch.zeros_like(y[:, 0])
        d_Wee1 = torch.zeros_like(y[:, 0])
        d_G1K = torch.zeros_like(y[:, 0])
        d_G2K = torch.zeros_like(y[:, 0])
        d_PG2 = torch.zeros_like(y[:, 0])
        d_G2R = torch.zeros_like(y[:, 0])
        d_R = torch.zeros_like(y[:, 0])
        d_G1R = torch.zeros_like(y[:, 0])
        d_PG2R = torch.zeros_like(y[:, 0])
        d_mass = torch.zeros_like(y[:, 0])
        d_IE = torch.zeros_like(y[:, 0])
        alg_0_k6 = torch.zeros_like(y[:, 0])
        alg_1_k2 = torch.zeros_like(y[:, 0])
        alg_5_kwee = torch.zeros_like(y[:, 0])
        alg_6_MPF = torch.zeros_like(y[:, 0])
        alg_7_k25 = torch.zeros_like(y[:, 0])
        alg_8_SPF = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_G1R = torch.zeros_like(y[:, 0]) + (k8 * R * G1K - (k8r + k4 + k6) * G1R)
        d_UbE = torch.zeros_like(y[:, 0]) + (ku * IE * (1.0 - UbE) / (Kmu + 1.0 - UbE) - kur * UbE / (Kmur + UbE))
        d_mass = torch.zeros_like(y[:, 0]) + (mu * mass)
        alg_0_k6 = V6_2 * (1.0 - UbE2) + V6 * UbE2
        d_G1K = torch.zeros_like(y[:, 0]) + (k5 + (k8r + k4) * G1R - (alg_0_k6 + k8 * R) * G1K)
        alg_1_k2 = V2_2 * (1.0 - UbE) + V2 * UbE
        d_G2R = torch.zeros_like(y[:, 0]) + (k7 * R * G2K - (k7r + k4 + alg_1_k2 + k2) * G2R)
        d_PG2R = torch.zeros_like(y[:, 0]) + (k7 * R * PG2 - (k7r + k4 + alg_1_k2 + k2) * PG2R)
        alg_6_MPF = G2K + beta * PG2
        d_IE = torch.zeros_like(y[:, 0]) + (ki * alg_6_MPF * (1.0 - IE) / (Kmi + 1.0 - IE) - kir * IE / (Kmir + IE))
        d_UbE2 = torch.zeros_like(y[:, 0]) + (ku2 * alg_6_MPF * (1.0 - UbE2) / (Kmu2 + 1.0 - UbE2) - kur2 * UbE2 / (Kmur2 + UbE2))
        d_Wee1 = torch.zeros_like(y[:, 0]) + (kwr * (1.0 - Wee1) / (Kmwr + 1.0 - Wee1) - kw * alg_6_MPF * Wee1 / (Kmw + Wee1))
        d_Cdc25 = torch.zeros_like(y[:, 0]) + (kc * alg_6_MPF * (1.0 - Cdc25) / (Kmc + 1.0 - Cdc25) - kcr * Cdc25 / (Kmcr + Cdc25))
        alg_5_kwee = Vw_2 * (1.0 - Wee1) + Vw * Wee1
        alg_7_k25 = V25_2 * (1.0 - Cdc25) + V25 * Cdc25
        d_G2K = torch.zeros_like(y[:, 0]) + (k1 + alg_7_k25 * PG2 + (k7r + k4) * G2R - (alg_1_k2 + alg_5_kwee + k7 * R) * G2K)
        alg_8_SPF = alg_6_MPF + alpha * G1K
        d_R = torch.zeros_like(y[:, 0]) + (k3 + (k7r + alg_1_k2 + k2) * (G2R + PG2R) + (k8r + k6) * G1R - (k4 * R + k7 * R * (G2K + PG2) + k8 * R * G1K + kp * R * alg_8_SPF * mass / (Kmp + R)))
        d_PG2 = torch.zeros_like(y[:, 0]) + (alg_5_kwee * G2K + (k7r + k4) * PG2R - (alg_7_k25 + alg_1_k2 + k7 * R) * PG2)

        return torch.stack(
            [d_Cdc25, d_UbE, d_UbE2, d_Wee1, d_G1K, d_G2K, d_PG2, d_G2R, d_R, d_G1R, d_PG2R, d_mass, d_IE],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
