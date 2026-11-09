import torch
import torch.nn as nn
import math


class CellCycleNovakCsikasznagyGyorffyChenTyson1998ODE(nn.Module):
    """
    PyTorch translation of cell_cycle/novak_csikasznagy_gyorffy_chen_tyson_1998/model.py.
    Dynamic states: 11; independent constants: 44; derived constants used by rates: 1.
    Original constant slots: 45; algebraic slots: 10.
    """
    source_file = 'cell_cycle/novak_csikasznagy_gyorffy_chen_tyson_1998/model.py'
    source_state_dim = 11
    source_constant_dim = 45
    algebraic_dim = 10
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 11
        self.parameter_dim = 44

        self.parameters = (
            parameters
            if parameters is not None
            else {'k1': 0.03, 'ki': 200, 'kir': 1, 'k3': 0.15, 'k25': 0.5, 'k25r': 0.2, 'J25': 0.2, 'J25r': 0.2, 'kw': 0.5, 'kwr': 0.2, 'Jw': 0.2, 'Jwr': 0.2, 'km': 0.1, 'kmr': 0, 'Jm': 0.2, 'Jmr': 0.2, 'kas': 0.25, 'kad': 0.1, 'kaa': 0.001, 'kaa_2': 1, 'kai': 0.25, 'kx': 0, 'Jaa': 0.1, 'Jai': 0.1, 'kapr': 0.04, 'kapr_2': 3, 'kap': 4, 'Japr': 0.01, 'Jap': 0.01, 'Puc1': 0.013, 'mu': 0.005776, 'V2': 0.03, 'V2_2': 1, 'V2c': 0.03, 'V2c_2': 0.16, 'k4': 0.15, 'k4_2': 20, 'V25': 0.01, 'V25_2': 0.4, 'Vwee': 0.01, 'Vwee_2': 0.93, 'Vmik': 0.002, 'Vmik_2': 0.2, 'alpha': 0.1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Cdc13_Cdc2': 1, 'mass': 1, 'Cdc13_P_Cdc2': 0, 'Rum1_Cdc13_Cdc2': 0, 'Rum1': 0, 'Cdc25P': 0, 'Wee1': 0, 'Mik1': 0, 'AAE_total': 2, 'AAE': 2, 'APC': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k1', 'ki', 'kir', 'k3', 'k25', 'k25r', 'J25', 'J25r', 'kw', 'kwr', 'Jw', 'Jwr', 'km', 'kmr', 'Jm', 'Jmr', 'kas', 'kad', 'kaa', 'kaa_2', 'kai', 'kx', 'Jaa', 'Jai', 'kapr', 'kapr_2', 'kap', 'Japr', 'Jap', 'Puc1', 'mu', 'V2', 'V2_2', 'V2c', 'V2c_2', 'k4', 'k4_2', 'V25', 'V25_2', 'Vwee', 'Vwee_2', 'Vmik', 'Vmik_2', 'alpha']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Cdc13_Cdc2', 'mass', 'Cdc13_P_Cdc2', 'Rum1_Cdc13_Cdc2', 'Rum1', 'Cdc25P', 'Wee1', 'Mik1', 'AAE_total', 'AAE', 'APC']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (11)
        Cdc13_Cdc2 = y[:, 0]
        mass = y[:, 1]
        Cdc13_P_Cdc2 = y[:, 2]
        Rum1_Cdc13_Cdc2 = y[:, 3]
        Rum1 = y[:, 4]
        Cdc25P = y[:, 5]
        Wee1 = y[:, 6]
        Mik1 = y[:, 7]
        AAE_total = y[:, 8]
        AAE = y[:, 9]
        APC = y[:, 10]

        # Independent parameters (source constant order)
        k1 = theta[:, 0]  # constants[0]
        ki = theta[:, 1]  # constants[1]
        kir = theta[:, 2]  # constants[2]
        k3 = theta[:, 3]  # constants[3]
        k25 = theta[:, 4]  # constants[4]
        k25r = theta[:, 5]  # constants[5]
        J25 = theta[:, 6]  # constants[6]
        J25r = theta[:, 7]  # constants[7]
        kw = theta[:, 8]  # constants[8]
        kwr = theta[:, 9]  # constants[9]
        Jw = theta[:, 10]  # constants[10]
        Jwr = theta[:, 11]  # constants[11]
        km = theta[:, 12]  # constants[12]
        kmr = theta[:, 13]  # constants[13]
        Jm = theta[:, 14]  # constants[14]
        Jmr = theta[:, 15]  # constants[15]
        kas = theta[:, 16]  # constants[16]
        kad = theta[:, 17]  # constants[17]
        kaa = theta[:, 18]  # constants[18]
        kaa_2 = theta[:, 19]  # constants[19]
        kai = theta[:, 20]  # constants[20]
        kx = theta[:, 21]  # constants[21]
        Jaa = theta[:, 22]  # constants[22]
        Jai = theta[:, 23]  # constants[23]
        kapr = theta[:, 24]  # constants[24]
        kapr_2 = theta[:, 25]  # constants[25]
        kap = theta[:, 26]  # constants[26]
        Japr = theta[:, 27]  # constants[27]
        Jap = theta[:, 28]  # constants[28]
        Puc1 = theta[:, 29]  # constants[29]
        mu = theta[:, 30]  # constants[30]
        V2 = theta[:, 31]  # constants[31]
        V2_2 = theta[:, 32]  # constants[32]
        V2c = theta[:, 33]  # constants[33]
        V2c_2 = theta[:, 34]  # constants[34]
        k4 = theta[:, 35]  # constants[35]
        k4_2 = theta[:, 36]  # constants[36]
        V25 = theta[:, 37]  # constants[37]
        V25_2 = theta[:, 38]  # constants[38]
        Vwee = theta[:, 39]  # constants[39]
        Vwee_2 = theta[:, 40]  # constants[40]
        Vmik = theta[:, 41]  # constants[41]
        Vmik_2 = theta[:, 42]  # constants[42]
        alpha = theta[:, 43]  # constants[43]

        # Derived constants from the original initConsts()
        kai_2 = kai + kx  # constants[44]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Cdc13_Cdc2 = torch.zeros_like(y[:, 0])
        d_mass = torch.zeros_like(y[:, 0])
        d_Cdc13_P_Cdc2 = torch.zeros_like(y[:, 0])
        d_Rum1_Cdc13_Cdc2 = torch.zeros_like(y[:, 0])
        d_Rum1 = torch.zeros_like(y[:, 0])
        d_Cdc25P = torch.zeros_like(y[:, 0])
        d_Wee1 = torch.zeros_like(y[:, 0])
        d_Mik1 = torch.zeros_like(y[:, 0])
        d_AAE_total = torch.zeros_like(y[:, 0])
        d_AAE = torch.zeros_like(y[:, 0])
        d_APC = torch.zeros_like(y[:, 0])
        alg_0_k2 = torch.zeros_like(y[:, 0])
        alg_1_k2c = torch.zeros_like(y[:, 0])
        alg_2_kcdc25 = torch.zeros_like(y[:, 0])
        alg_3_kwee = torch.zeros_like(y[:, 0])
        alg_4_MPF = torch.zeros_like(y[:, 0])
        alg_5_k4 = torch.zeros_like(y[:, 0])
        alg_6_ks = torch.zeros_like(y[:, 0])
        alg_7_k25r = torch.zeros_like(y[:, 0])
        alg_8_kwr = torch.zeros_like(y[:, 0])
        alg_9_kmr = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_mass = torch.zeros_like(y[:, 0]) + (mu * mass)
        alg_3_kwee = Vwee * ((1.0 - Wee1) * 1.0 + Vwee_2 * Wee1 + Vmik * (1.0 - Mik1)) * 1.0 + Vmik_2 * Mik1
        alg_2_kcdc25 = V25 * (1.0 - Cdc25P) + V25_2 * Cdc25P
        alg_0_k2 = V2 * (1.0 - APC) + V2_2 * APC
        d_Cdc13_P_Cdc2 = torch.zeros_like(y[:, 0]) + (alg_3_kwee * Cdc13_Cdc2 - (alg_2_kcdc25 * Cdc13_P_Cdc2 + alg_0_k2 * Cdc13_P_Cdc2))
        alg_4_MPF = Cdc13_Cdc2 + alpha * Cdc13_P_Cdc2
        d_AAE_total = torch.zeros_like(y[:, 0]) + (kas * alg_4_MPF - kad * AAE_total)
        d_AAE = torch.zeros_like(y[:, 0]) + ((kaa + kaa_2 * alg_4_MPF) * (AAE_total - AAE) / (Jaa + AAE_total - AAE) - (kai_2 * AAE / (Jai + AAE) + kad * AAE))
        d_APC = torch.zeros_like(y[:, 0]) + ((kapr + kapr_2 * AAE) * (1.0 - APC) / (Japr + 1.0 - APC) - kap * (Puc1 * mass + alg_4_MPF) * APC / (Jap + APC))
        alg_5_k4 = k4 + k4_2 * (Puc1 * mass + alg_4_MPF)
        d_Cdc13_Cdc2 = torch.zeros_like(y[:, 0]) + (k1 * mass + alg_2_kcdc25 * Cdc13_P_Cdc2 + Rum1_Cdc13_Cdc2 * (kir + alg_5_k4) - (alg_3_kwee * Cdc13_Cdc2 + ki * Cdc13_Cdc2 * Rum1 + alg_0_k2 * Cdc13_Cdc2))
        alg_1_k2c = V2c * (1.0 - APC) + V2c_2 * APC
        d_Rum1_Cdc13_Cdc2 = torch.zeros_like(y[:, 0]) + (ki * Rum1 * Cdc13_Cdc2 - Rum1_Cdc13_Cdc2 * (alg_5_k4 + alg_1_k2c + kir))
        d_Rum1 = torch.zeros_like(y[:, 0]) + (k3 + Rum1_Cdc13_Cdc2 * (kir + alg_1_k2c) - (alg_5_k4 * Rum1 + ki * Cdc13_Cdc2 * Rum1))
        alg_6_ks = torch.where(APC <= 0.2, 0.5, 0.0)
        alg_7_k25r = k25r + alg_6_ks
        d_Cdc25P = torch.zeros_like(y[:, 0]) + (k25 * alg_4_MPF * (1.0 - Cdc25P) / (J25 + 1.0 - Cdc25P) - alg_7_k25r * Cdc25P / (J25r + Cdc25P))
        alg_8_kwr = kwr + alg_6_ks
        d_Wee1 = torch.zeros_like(y[:, 0]) + (alg_8_kwr * (1.0 - Wee1) / (Jwr + 1.0 - Wee1) - kw * alg_4_MPF * Wee1 / (Jw + Wee1))
        alg_9_kmr = kmr + alg_6_ks
        d_Mik1 = torch.zeros_like(y[:, 0]) + (alg_9_kmr * (1.0 - Mik1) / (Jmr + 1.0 - Mik1) - km * Mik1 / (Jm + Mik1))

        return torch.stack(
            [d_Cdc13_Cdc2, d_mass, d_Cdc13_P_Cdc2, d_Rum1_Cdc13_Cdc2, d_Rum1, d_Cdc25P, d_Wee1, d_Mik1, d_AAE_total, d_AAE, d_APC],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
