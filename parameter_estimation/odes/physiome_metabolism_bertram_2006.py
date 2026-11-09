import torch
import torch.nn as nn
import math


class MetabolismBertram2006ODE(nn.Module):
    """
    PyTorch translation of metabolism/bertram_2006/model.py.
    Dynamic states: 7; independent constants: 47; derived constants used by rates: 0.
    Original constant slots: 47; algebraic slots: 33.
    """
    source_file = 'metabolism/bertram_2006/model.py'
    source_state_dim = 7
    source_constant_dim = 47
    algebraic_dim = 33
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 7
        self.parameter_dim = 47

        self.parameters = (
            parameters
            if parameters is not None
            else {'gamma': 0.001, 'NADtot': 10, 'fm': 0.01, 'Cmito': 1.8, 'Amtot': 15, 'Ac_tot': 2500, 'khyd': 5e-05, 'Jhydbas': 5e-05, 'Fhold': 1, 'Ftest': 5, 'Fton': 90000, 'Ftoff': 330000, 'Chold': 0.1, 'Ctest': 0.1, 'Cton1': 120000, 'Cton2': 180000, 'Cton3': 240000, 'Ctoff1': 150000, 'Ctoff2': 210000, 'Ctoff3': 270000, 'p1': 400, 'p2': 1, 'p3': 0.01, 'p21': 0.01, 'p22': 1.1, 'p23': 0.001, 'p24': 0.016, 'p4': 0.6, 'p5': 0.1, 'p6': 177, 'p7': 5, 'p8': 7, 'p9': 0.1, 'p10': 177, 'p11': 5, 'p12': 120, 'p13': 10, 'p14': 190, 'p15': 8.5, 'p16': 35, 'p17': 0.002, 'p18': -0.03, 'p19': 0.35, 'p20': 2, 'FRT': 0.037, 'kgpdh': 0.0005, 'delta': 0.07330827067669173}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'NADHm': 0.6, 'ADPm': 7.4, 'PSIm': 93, 'Cam': 0.1, 'c': 0.1, 'ADPc': 1850, 'FBP': 0.5}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['gamma', 'NADtot', 'fm', 'Cmito', 'Amtot', 'Ac_tot', 'khyd', 'Jhydbas', 'Fhold', 'Ftest', 'Fton', 'Ftoff', 'Chold', 'Ctest', 'Cton1', 'Cton2', 'Cton3', 'Ctoff1', 'Ctoff2', 'Ctoff3', 'p1', 'p2', 'p3', 'p21', 'p22', 'p23', 'p24', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'FRT', 'kgpdh', 'delta']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['NADHm', 'ADPm', 'PSIm', 'Cam', 'c', 'ADPc', 'FBP']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (7)
        NADHm = y[:, 0]
        ADPm = y[:, 1]
        PSIm = y[:, 2]
        Cam = y[:, 3]
        c = y[:, 4]
        ADPc = y[:, 5]
        FBP = y[:, 6]

        # Independent parameters (source constant order)
        gamma = theta[:, 0]  # constants[0]
        NADtot = theta[:, 1]  # constants[1]
        fm = theta[:, 2]  # constants[2]
        Cmito = theta[:, 3]  # constants[3]
        Amtot = theta[:, 4]  # constants[4]
        Ac_tot = theta[:, 5]  # constants[5]
        khyd = theta[:, 6]  # constants[6]
        Jhydbas = theta[:, 7]  # constants[7]
        Fhold = theta[:, 8]  # constants[8]
        Ftest = theta[:, 9]  # constants[9]
        Fton = theta[:, 10]  # constants[10]
        Ftoff = theta[:, 11]  # constants[11]
        Chold = theta[:, 12]  # constants[12]
        Ctest = theta[:, 13]  # constants[13]
        Cton1 = theta[:, 14]  # constants[14]
        Cton2 = theta[:, 15]  # constants[15]
        Cton3 = theta[:, 16]  # constants[16]
        Ctoff1 = theta[:, 17]  # constants[17]
        Ctoff2 = theta[:, 18]  # constants[18]
        Ctoff3 = theta[:, 19]  # constants[19]
        p1 = theta[:, 20]  # constants[20]
        p2 = theta[:, 21]  # constants[21]
        p3 = theta[:, 22]  # constants[22]
        p21 = theta[:, 23]  # constants[23]
        p22 = theta[:, 24]  # constants[24]
        p23 = theta[:, 25]  # constants[25]
        p24 = theta[:, 26]  # constants[26]
        p4 = theta[:, 27]  # constants[27]
        p5 = theta[:, 28]  # constants[28]
        p6 = theta[:, 29]  # constants[29]
        p7 = theta[:, 30]  # constants[30]
        p8 = theta[:, 31]  # constants[31]
        p9 = theta[:, 32]  # constants[32]
        p10 = theta[:, 33]  # constants[33]
        p11 = theta[:, 34]  # constants[34]
        p12 = theta[:, 35]  # constants[35]
        p13 = theta[:, 36]  # constants[36]
        p14 = theta[:, 37]  # constants[37]
        p15 = theta[:, 38]  # constants[38]
        p16 = theta[:, 39]  # constants[39]
        p17 = theta[:, 40]  # constants[40]
        p18 = theta[:, 41]  # constants[41]
        p19 = theta[:, 42]  # constants[42]
        p20 = theta[:, 43]  # constants[43]
        FRT = theta[:, 44]  # constants[44]
        kgpdh = theta[:, 45]  # constants[45]
        delta = theta[:, 46]  # constants[46]

        # Initialize rates and algebraic variables, as in computeRates()
        d_NADHm = torch.zeros_like(y[:, 0])
        d_ADPm = torch.zeros_like(y[:, 0])
        d_PSIm = torch.zeros_like(y[:, 0])
        d_Cam = torch.zeros_like(y[:, 0])
        d_c = torch.zeros_like(y[:, 0])
        d_ADPc = torch.zeros_like(y[:, 0])
        d_FBP = torch.zeros_like(y[:, 0])
        alg_0_NADm = torch.zeros_like(y[:, 0])
        alg_1_ATPm = torch.zeros_like(y[:, 0])
        alg_2_heav_on = torch.zeros_like(y[:, 0])
        alg_3_heav_Cton1 = torch.zeros_like(y[:, 0])
        alg_4_MM1 = torch.zeros_like(y[:, 0])
        alg_5_RATm = torch.zeros_like(y[:, 0])
        alg_6_heav_off = torch.zeros_like(y[:, 0])
        alg_7_heav_Cton2 = torch.zeros_like(y[:, 0])
        alg_8_Jo = torch.zeros_like(y[:, 0])
        alg_9_ATPc = torch.zeros_like(y[:, 0])
        alg_10_Fproto = torch.zeros_like(y[:, 0])
        alg_11_heav_Cton3 = torch.zeros_like(y[:, 0])
        alg_12_Jgpdh = torch.zeros_like(y[:, 0])
        alg_13_Jh_yd = torch.zeros_like(y[:, 0])
        alg_14_heav_Ctoff1 = torch.zeros_like(y[:, 0])
        alg_15_Jpdh = torch.zeros_like(y[:, 0])
        alg_16_Juni = torch.zeros_like(y[:, 0])
        alg_17_pulse1 = torch.zeros_like(y[:, 0])
        alg_18_Jnaca = torch.zeros_like(y[:, 0])
        alg_19_heav_Ctoff2 = torch.zeros_like(y[:, 0])
        alg_20_Jh_leak = torch.zeros_like(y[:, 0])
        alg_21_Jmito = torch.zeros_like(y[:, 0])
        alg_22_pulse2 = torch.zeros_like(y[:, 0])
        alg_23_MM2 = torch.zeros_like(y[:, 0])
        alg_24_heav_Ctoff3 = torch.zeros_like(y[:, 0])
        alg_25_Jh_res = torch.zeros_like(y[:, 0])
        alg_26_pulse3 = torch.zeros_like(y[:, 0])
        alg_27_b13 = torch.zeros_like(y[:, 0])
        alg_28_Cproto = torch.zeros_like(y[:, 0])
        alg_29_Jh_atp = torch.zeros_like(y[:, 0])
        alg_30_b2 = torch.zeros_like(y[:, 0])
        alg_31_Jf1f0 = torch.zeros_like(y[:, 0])
        alg_32_Jant = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_2_heav_on = torch.where(t - Fton >= 0.0, 1.0, 0.0)
        alg_6_heav_off = torch.where(t - Ftoff >= 0.0, 1.0, 0.0)
        alg_10_Fproto = Fhold + (Ftest - Fhold) * (alg_2_heav_on - alg_6_heav_off)
        d_FBP = torch.zeros_like(y[:, 0]) + ((alg_10_Fproto - FBP) / 0.1)
        alg_0_NADm = NADtot - NADHm
        alg_12_Jgpdh = kgpdh * (FBP / 1.0) ** (1.0 / 2)
        alg_15_Jpdh = p1 / (p2 + NADHm / alg_0_NADm) * (Cam / (p3 + Cam)) * alg_12_Jgpdh
        alg_4_MM1 = p4 * NADHm / (p5 + NADHm)
        alg_8_Jo = alg_4_MM1 / (1.0 + torch.exp((PSIm - p6) / p7))
        d_NADHm = torch.zeros_like(y[:, 0]) + (gamma * (alg_15_Jpdh - alg_8_Jo))
        alg_18_Jnaca = p23 * Cam / c * torch.exp(p24 * PSIm)
        alg_16_Juni = (p21 * PSIm - p22) * c ** 2.0
        alg_21_Jmito = alg_18_Jnaca - alg_16_Juni
        d_Cam = torch.zeros_like(y[:, 0]) + (-fm * alg_21_Jmito)
        alg_3_heav_Cton1 = torch.where(t - Cton1 >= 0.0, 1.0, 0.0)
        alg_14_heav_Ctoff1 = torch.where(t - Ctoff1 >= 0.0, 1.0, 0.0)
        alg_17_pulse1 = (Ctest - Chold) * (alg_3_heav_Cton1 - alg_14_heav_Ctoff1)
        alg_7_heav_Cton2 = torch.where(t - Cton2 >= 0.0, 1.0, 0.0)
        alg_19_heav_Ctoff2 = torch.where(t - Ctoff2 >= 0.0, 1.0, 0.0)
        alg_22_pulse2 = (Ctest - Chold) * (alg_7_heav_Cton2 - alg_19_heav_Ctoff2)
        alg_11_heav_Cton3 = torch.where(t - Cton3 >= 0.0, 1.0, 0.0)
        alg_24_heav_Ctoff3 = torch.where(t - Ctoff3 >= 0.0, 1.0, 0.0)
        alg_26_pulse3 = (Ctest - Chold) * (alg_11_heav_Cton3 - alg_24_heav_Ctoff3)
        alg_28_Cproto = Chold + alg_17_pulse1 + alg_22_pulse2 + alg_26_pulse3
        d_c = torch.zeros_like(y[:, 0]) + ((alg_28_Cproto - c) / 0.1)
        alg_1_ATPm = Amtot - ADPm
        alg_5_RATm = alg_1_ATPm / ADPm
        alg_32_Jant = p19 * alg_5_RATm / (alg_5_RATm + p20) / torch.exp(-0.5 * FRT * PSIm)
        alg_30_b2 = p16 * p13 / (p13 + alg_1_ATPm)
        alg_31_Jf1f0 = alg_30_b2 / (1.0 + torch.exp((p14 - PSIm) / p15))
        d_ADPm = torch.zeros_like(y[:, 0]) + (gamma * (alg_32_Jant - alg_31_Jf1f0))
        alg_23_MM2 = p8 * NADHm / (p9 + NADHm)
        alg_25_Jh_res = alg_23_MM2 / (1.0 + torch.exp((PSIm - p10) / p11))
        alg_27_b13 = p12 * p13 / (p13 + alg_1_ATPm)
        alg_29_Jh_atp = alg_27_b13 / (1.0 + torch.exp((p14 - PSIm) / p15))
        alg_20_Jh_leak = p17 * PSIm + p18
        d_PSIm = torch.zeros_like(y[:, 0]) + ((alg_25_Jh_res - alg_29_Jh_atp - alg_32_Jant - alg_20_Jh_leak - alg_18_Jnaca - alg_16_Juni * 2.0) / Cmito)
        alg_9_ATPc = Ac_tot - ADPc
        alg_13_Jh_yd = (khyd * c + Jhydbas) * alg_9_ATPc / 1.0
        d_ADPc = torch.zeros_like(y[:, 0]) + (-delta * alg_32_Jant + alg_13_Jh_yd * 1.0)

        return torch.stack(
            [d_NADHm, d_ADPm, d_PSIm, d_Cam, d_c, d_ADPc, d_FBP],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
