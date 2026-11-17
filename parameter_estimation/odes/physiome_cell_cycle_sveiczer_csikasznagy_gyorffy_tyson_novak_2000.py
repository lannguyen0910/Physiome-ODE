import torch
import torch.nn as nn
import math


class CellCycleSveiczerCsikasznagyGyorffyTysonNovak2000ODE(nn.Module):
    """
    PyTorch translation of cell_cycle/sveiczer_csikasznagy_gyorffy_tyson_novak_2000/model.py.
    Dynamic states: 16; independent constants: 58; derived constants used by rates: 0.
    Original constant slots: 58; algebraic slots: 8.
    """
    source_file = 'cell_cycle/sveiczer_csikasznagy_gyorffy_tyson_novak_2000/model.py'
    source_state_dim = 16
    source_constant_dim = 58
    algebraic_dim = 8
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 16
        self.parameter_dim = 58

        self.parameters = (
            parameters
            if parameters is not None
            else {'k1': 0.02, 'kj': 400, 'kjr': 1, 'k6': 0.1, 'k6_2': 5, 'kste9r': 0.03, 'kste9r_2': 8, 'kste9': 5, 'Jste9r': 0.01, 'Jste9': 0.01, 'SK': 0.018, 'ks': 0.1, 'kmr': 0.01, 'kmr_2': 5, 'km': 1, 'Jmikr': 0.15, 'Jmik': 0.15, 'kwr': 0.4, 'kwr_2': 1, 'kw': 2, 'Jweer': 0.2, 'Jwee': 0.2, 'k25': 1, 'k5': 0.1, 'k25r': 0.4, 'k25r_2': 2, 'J25': 0.05, 'J25r': 0.05, 'kas': 0.1, 'kad': 0.1, 'kaa': 0.01, 'kaa_2': 0.1, 'kai': 0.1, 'k3': 0.1, 'ki': 50, 'kir': 0.5, 'kp': 100, 'kpp': 1, 'kpp_2': 100, 'epsilon_p': 0.025, 'mu': 0.00462, 'K': 0.06, 'Y': 0, 'epsilon': 0.05, 'V2': 0.02, 'V2_2': 1, 'V2c': 0.5, 'V2c_2': 0.02, 'Vwee': 10, 'Vwee_2': 0.08, 'Vmik': 2, 'Vmik_2': 0.04, 'V25': 10, 'V25_2': 0.05, 'Vpyp': 0.07, 'V4': 1, 'V4_2': 0.01, 'Pyp3': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'MPF': 0, 'mass': 1, 'preMPF': 1, 'Rum1': 0, 'Rum1P': 0, 'CR': 0, 'CRP': 0, 'Ste9': 0, 'Mik1': 0, 'Wee1': 0, 'Cdc25': 0, 'Slp1': 0, 'Slp1_a': 0, 'Inh': 0, 'PI': 0.2, 'R_dna': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k1', 'kj', 'kjr', 'k6', 'k6_2', 'kste9r', 'kste9r_2', 'kste9', 'Jste9r', 'Jste9', 'SK', 'ks', 'kmr', 'kmr_2', 'km', 'Jmikr', 'Jmik', 'kwr', 'kwr_2', 'kw', 'Jweer', 'Jwee', 'k25', 'k5', 'k25r', 'k25r_2', 'J25', 'J25r', 'kas', 'kad', 'kaa', 'kaa_2', 'kai', 'k3', 'ki', 'kir', 'kp', 'kpp', 'kpp_2', 'epsilon_p', 'mu', 'K', 'Y', 'epsilon', 'V2', 'V2_2', 'V2c', 'V2c_2', 'Vwee', 'Vwee_2', 'Vmik', 'Vmik_2', 'V25', 'V25_2', 'Vpyp', 'V4', 'V4_2', 'Pyp3']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['MPF', 'mass', 'preMPF', 'Rum1', 'Rum1P', 'CR', 'CRP', 'Ste9', 'Mik1', 'Wee1', 'Cdc25', 'Slp1', 'Slp1_a', 'Inh', 'PI', 'R_dna']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (16)
        MPF = y[:, 0]
        mass = y[:, 1]
        preMPF = y[:, 2]
        Rum1 = y[:, 3]
        Rum1P = y[:, 4]
        CR = y[:, 5]
        CRP = y[:, 6]
        Ste9 = y[:, 7]
        Mik1 = y[:, 8]
        Wee1 = y[:, 9]
        Cdc25 = y[:, 10]
        Slp1 = y[:, 11]
        Slp1_a = y[:, 12]
        Inh = y[:, 13]
        PI = y[:, 14]
        R_dna = y[:, 15]

        # Independent parameters (source constant order)
        k1 = theta[:, 0]  # constants[0]
        kj = theta[:, 1]  # constants[1]
        kjr = theta[:, 2]  # constants[2]
        k6 = theta[:, 3]  # constants[3]
        k6_2 = theta[:, 4]  # constants[4]
        kste9r = theta[:, 5]  # constants[5]
        kste9r_2 = theta[:, 6]  # constants[6]
        kste9 = theta[:, 7]  # constants[7]
        Jste9r = theta[:, 8]  # constants[8]
        Jste9 = theta[:, 9]  # constants[9]
        SK = theta[:, 10]  # constants[10]
        ks = theta[:, 11]  # constants[11]
        kmr = theta[:, 12]  # constants[12]
        kmr_2 = theta[:, 13]  # constants[13]
        km = theta[:, 14]  # constants[14]
        Jmikr = theta[:, 15]  # constants[15]
        Jmik = theta[:, 16]  # constants[16]
        kwr = theta[:, 17]  # constants[17]
        kwr_2 = theta[:, 18]  # constants[18]
        kw = theta[:, 19]  # constants[19]
        Jweer = theta[:, 20]  # constants[20]
        Jwee = theta[:, 21]  # constants[21]
        k25 = theta[:, 22]  # constants[22]
        k5 = theta[:, 23]  # constants[23]
        k25r = theta[:, 24]  # constants[24]
        k25r_2 = theta[:, 25]  # constants[25]
        J25 = theta[:, 26]  # constants[26]
        J25r = theta[:, 27]  # constants[27]
        kas = theta[:, 28]  # constants[28]
        kad = theta[:, 29]  # constants[29]
        kaa = theta[:, 30]  # constants[30]
        kaa_2 = theta[:, 31]  # constants[31]
        kai = theta[:, 32]  # constants[32]
        k3 = theta[:, 33]  # constants[33]
        ki = theta[:, 34]  # constants[34]
        kir = theta[:, 35]  # constants[35]
        kp = theta[:, 36]  # constants[36]
        kpp = theta[:, 37]  # constants[37]
        kpp_2 = theta[:, 38]  # constants[38]
        epsilon_p = theta[:, 39]  # constants[39]
        mu = theta[:, 40]  # constants[40]
        K = theta[:, 41]  # constants[41]
        Y = theta[:, 42]  # constants[42]
        epsilon = theta[:, 43]  # constants[43]
        V2 = theta[:, 44]  # constants[44]
        V2_2 = theta[:, 45]  # constants[45]
        V2c = theta[:, 46]  # constants[46]
        V2c_2 = theta[:, 47]  # constants[47]
        Vwee = theta[:, 48]  # constants[48]
        Vwee_2 = theta[:, 49]  # constants[49]
        Vmik = theta[:, 50]  # constants[50]
        Vmik_2 = theta[:, 51]  # constants[51]
        V25 = theta[:, 52]  # constants[52]
        V25_2 = theta[:, 53]  # constants[53]
        Vpyp = theta[:, 54]  # constants[54]
        V4 = theta[:, 55]  # constants[55]
        V4_2 = theta[:, 56]  # constants[56]
        Pyp3 = theta[:, 57]  # constants[57]

        # Initialize rates and algebraic variables, as in computeRates()
        d_MPF = torch.zeros_like(y[:, 0])
        d_mass = torch.zeros_like(y[:, 0])
        d_preMPF = torch.zeros_like(y[:, 0])
        d_Rum1 = torch.zeros_like(y[:, 0])
        d_Rum1P = torch.zeros_like(y[:, 0])
        d_CR = torch.zeros_like(y[:, 0])
        d_CRP = torch.zeros_like(y[:, 0])
        d_Ste9 = torch.zeros_like(y[:, 0])
        d_Mik1 = torch.zeros_like(y[:, 0])
        d_Wee1 = torch.zeros_like(y[:, 0])
        d_Cdc25 = torch.zeros_like(y[:, 0])
        d_Slp1 = torch.zeros_like(y[:, 0])
        d_Slp1_a = torch.zeros_like(y[:, 0])
        d_Inh = torch.zeros_like(y[:, 0])
        d_PI = torch.zeros_like(y[:, 0])
        d_R_dna = torch.zeros_like(y[:, 0])
        alg_0_k2 = torch.zeros_like(y[:, 0])
        alg_1_MPF_a = torch.zeros_like(y[:, 0])
        alg_2_kwee = torch.zeros_like(y[:, 0])
        alg_3_PP = torch.zeros_like(y[:, 0])
        alg_4_kc25 = torch.zeros_like(y[:, 0])
        alg_5_k4 = torch.zeros_like(y[:, 0])
        alg_6_k2c = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_mass = torch.zeros_like(y[:, 0]) + (mu * mass)
        alg_1_MPF_a = MPF + epsilon * preMPF
        d_Slp1 = torch.zeros_like(y[:, 0]) + (kas * alg_1_MPF_a - kad * Slp1)
        d_Slp1_a = torch.zeros_like(y[:, 0]) + ((kaa + kaa_2 * alg_1_MPF_a) * (Slp1 - Slp1_a) - (kai + kad) * Slp1_a)
        d_R_dna = torch.zeros_like(y[:, 0]) + (K * 1.0 / (1.0 + Y * alg_1_MPF_a))
        alg_3_PP = 1.0 - PI
        d_Ste9 = torch.zeros_like(y[:, 0]) + ((kste9r + kste9r_2 * alg_3_PP) * (1.0 - Ste9) / (Jste9r + 1.0 - Ste9) - kste9 * (alg_1_MPF_a + SK * mass) * Ste9 / (Jste9 + Ste9))
        d_Mik1 = torch.zeros_like(y[:, 0]) + ((ks + kmr + kmr_2 * alg_3_PP) * (1.0 - Mik1) / (Jmikr + 1.0 - Mik1) - km * alg_1_MPF_a * Mik1 / (Jmik + Mik1))
        d_Wee1 = torch.zeros_like(y[:, 0]) + ((kwr + kwr_2 * alg_3_PP) * (1.0 - Wee1) / (Jweer + 1.0 - Wee1) - kw * alg_1_MPF_a * Wee1 / (Jwee + Wee1))
        d_Cdc25 = torch.zeros_like(y[:, 0]) + (k25 * alg_1_MPF_a * (1.0 - Cdc25) / (J25 + 1.0 - Cdc25) - (k5 + k25r + k25r_2 * alg_3_PP) * Cdc25 / (J25r + Cdc25))
        alg_0_k2 = V2 + V2_2 * Ste9
        alg_2_kwee = Vwee * Wee1 + Vwee_2 * (1.0 - Wee1) + Vmik * Mik1 + Vmik_2 * (1.0 - Mik1)
        alg_4_kc25 = V25 * Cdc25 + V25_2 * (1.0 - Cdc25) + Vpyp * Pyp3
        d_MPF = torch.zeros_like(y[:, 0]) + (k1 * mass - alg_0_k2 * MPF - alg_2_kwee * MPF + alg_4_kc25 * preMPF - kj * MPF * (Rum1 + Rum1P) + kjr * (CR + CRP) + k6 * CR + (k6 + k6_2) * CRP)
        d_preMPF = torch.zeros_like(y[:, 0]) + (alg_2_kwee * MPF - alg_4_kc25 * preMPF - alg_0_k2 * preMPF)
        alg_5_k4 = V4_2 + V4 * Slp1_a
        d_Inh = torch.zeros_like(y[:, 0]) + (k3 - ki * Inh * alg_3_PP + kir * PI - alg_5_k4 * Inh)
        d_PI = torch.zeros_like(y[:, 0]) + (ki * Inh * alg_3_PP - kir * PI - alg_5_k4 * PI)
        alg_6_k2c = V2c_2 + V2c * Ste9
        d_CRP = torch.zeros_like(y[:, 0]) + (kp * (alg_1_MPF_a + epsilon_p * SK * mass) * CR - (kpp + kpp_2 * alg_3_PP) * CRP + kj * MPF * Rum1P - kjr * CRP - alg_6_k2c * CRP - (k6 + k6_2) * CRP)
        d_CR = torch.zeros_like(y[:, 0]) + (kj * MPF * Rum1 - kjr * CR - alg_6_k2c * CR - k6 * CR - kp * (alg_1_MPF_a + epsilon_p * SK * mass) * CR + (kpp + kpp_2 * alg_3_PP) * CRP)
        d_Rum1 = torch.zeros_like(y[:, 0]) + (k5 - k6 * Rum1 - kp * (alg_1_MPF_a + epsilon_p * SK * mass) * Rum1 + (kpp + kpp_2 * alg_3_PP) * Rum1P - kj * MPF * Rum1 + kjr * CR + alg_6_k2c * CR)
        d_Rum1P = torch.zeros_like(y[:, 0]) + (kp * (alg_1_MPF_a + epsilon_p * SK * mass) * Rum1 - (kpp + kpp_2 * alg_3_PP) * Rum1P - (k6 + k6_2) * Rum1P - kj * MPF * Rum1P + kjr * CRP + alg_6_k2c * CRP)

        return torch.stack(
            [d_MPF, d_mass, d_preMPF, d_Rum1, d_Rum1P, d_CR, d_CRP, d_Ste9, d_Mik1, d_Wee1, d_Cdc25, d_Slp1, d_Slp1_a, d_Inh, d_PI, d_R_dna],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
