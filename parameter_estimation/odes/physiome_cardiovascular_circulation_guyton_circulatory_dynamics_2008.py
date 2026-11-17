import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonCirculatoryDynamics2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_circulatory_dynamics_2008/model.py.
    Dynamic states: 5; independent constants: 40; derived constants used by rates: 5.
    Original constant slots: 45; algebraic slots: 62.
    """
    source_file = 'cardiovascular_circulation/guyton_circulatory_dynamics_2008/model.py'
    source_state_dim = 5
    source_constant_dim = 45
    algebraic_dim = 62
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
        self.parameter_dim = 40

        self.parameters = (
            parameters
            if parameters is not None
            else {'ADHMV': 1, 'AMM': 1, 'ANU': 0.925271, 'ANUVN': 1, 'ARM': 1.16463, 'ATRRFB': 1, 'ATRVFB': 0, 'AU': 1.00022, 'AUH': 1.00012, 'AUM': 1.00066, 'AVE': 1, 'HMD': 1, 'HPL': 1.00163, 'HPR': 1.00237, 'MYOGRS': 1, 'OSA': 0.97287, 'PAMK': 1, 'PC': 16.9144, 'RBF': 1.22057, 'VIM': 1.00076, 'VP': 3.00449, 'VRC': 2.00439, 'VV6': 0.0101913, 'VV7': 0.00366525, 'VVR': 2.50967, 'HTAUML': 0.4, 'QRF': 0.15, 'HSR': 1, 'HSL': 1, 'ANY': -0.2, 'CV': 0.1, 'PR1LL': 0, 'CN7': 0.2, 'CN2': 0.0212, 'RVSM': 1, 'PAEX': 2, 'RAR': 30.52, 'RMULT1': 1, 'RAM': 96.3, 'FIS': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'VVS1': 3.28246, 'VAS1': 0.862514, 'VLA1': 0.379883, 'VPA1': 0.38131, 'VRA1': 0.100043}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['ADHMV', 'AMM', 'ANU', 'ANUVN', 'ARM', 'ATRRFB', 'ATRVFB', 'AU', 'AUH', 'AUM', 'AVE', 'HMD', 'HPL', 'HPR', 'MYOGRS', 'OSA', 'PAMK', 'PC', 'RBF', 'VIM', 'VP', 'VRC', 'VV6', 'VV7', 'VVR', 'HTAUML', 'QRF', 'HSR', 'HSL', 'ANY', 'CV', 'PR1LL', 'CN7', 'CN2', 'RVSM', 'PAEX', 'RAR', 'RMULT1', 'RAM', 'FIS']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['VVS1', 'VAS1', 'VLA1', 'VPA1', 'VRA1']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        VVS1 = y[:, 0]
        VAS1 = y[:, 1]
        VLA1 = y[:, 2]
        VPA1 = y[:, 3]
        VRA1 = y[:, 4]

        # Independent parameters (source constant order)
        ADHMV = theta[:, 0]  # constants[0]
        AMM = theta[:, 1]  # constants[1]
        ANU = theta[:, 2]  # constants[2]
        ANUVN = theta[:, 3]  # constants[3]
        ARM = theta[:, 4]  # constants[4]
        ATRRFB = theta[:, 5]  # constants[5]
        ATRVFB = theta[:, 6]  # constants[6]
        AU = theta[:, 7]  # constants[7]
        AUH = theta[:, 8]  # constants[8]
        AUM = theta[:, 9]  # constants[9]
        AVE = theta[:, 10]  # constants[10]
        HMD = theta[:, 11]  # constants[11]
        HPL = theta[:, 12]  # constants[12]
        HPR = theta[:, 13]  # constants[13]
        MYOGRS = theta[:, 14]  # constants[14]
        OSA = theta[:, 15]  # constants[15]
        PAMK = theta[:, 16]  # constants[16]
        PC = theta[:, 17]  # constants[17]
        RBF = theta[:, 18]  # constants[18]
        VIM = theta[:, 19]  # constants[19]
        VP = theta[:, 20]  # constants[20]
        VRC = theta[:, 21]  # constants[21]
        VV6 = theta[:, 22]  # constants[22]
        VV7 = theta[:, 23]  # constants[23]
        VVR = theta[:, 24]  # constants[24]
        HTAUML = theta[:, 25]  # constants[25]
        QRF = theta[:, 26]  # constants[26]
        HSR = theta[:, 27]  # constants[27]
        HSL = theta[:, 28]  # constants[28]
        ANY = theta[:, 29]  # constants[29]
        CV = theta[:, 30]  # constants[30]
        PR1LL = theta[:, 31]  # constants[31]
        CN7 = theta[:, 32]  # constants[32]
        CN2 = theta[:, 33]  # constants[33]
        RVSM = theta[:, 34]  # constants[34]
        PAEX = theta[:, 35]  # constants[35]
        RAR = theta[:, 36]  # constants[36]
        RMULT1 = theta[:, 37]  # constants[37]
        RAM = theta[:, 38]  # constants[38]
        FIS = theta[:, 39]  # constants[39]

        # Derived constants from the original initConsts()
        VVA = (ANU - 1.0) * ANY  # constants[40]
        CN3 = ((PC - 17.0) * CN7 + 17.0) * CN2  # constants[41]
        RV1 = RVSM / CN3  # constants[42]
        RVS = AVE * RV1 * VIM * ANUVN  # constants[43]
        NNRVR = RVS * 1.79  # constants[44]

        # Initialize rates and algebraic variables, as in computeRates()
        d_VVS1 = torch.zeros_like(y[:, 0])
        d_VAS1 = torch.zeros_like(y[:, 0])
        d_VLA1 = torch.zeros_like(y[:, 0])
        d_VPA1 = torch.zeros_like(y[:, 0])
        d_VRA1 = torch.zeros_like(y[:, 0])
        alg_0_VBD = torch.zeros_like(y[:, 0])
        alg_1_VRA = torch.zeros_like(y[:, 0])
        alg_2_VRE = torch.zeros_like(y[:, 0])
        alg_3_PRA = torch.zeros_like(y[:, 0])
        alg_4_PRA1 = torch.zeros_like(y[:, 0])
        alg_5_QRN = torch.zeros_like(y[:, 0])
        alg_6_VPA = torch.zeros_like(y[:, 0])
        alg_7_VPE = torch.zeros_like(y[:, 0])
        alg_8_PPA = torch.zeros_like(y[:, 0])
        alg_9_PP2 = torch.zeros_like(y[:, 0])
        alg_10_RVM = torch.zeros_like(y[:, 0])
        alg_11_PP1T = torch.zeros_like(y[:, 0])
        alg_12_PP1 = torch.zeros_like(y[:, 0])
        alg_13_CPA = torch.zeros_like(y[:, 0])
        alg_14_RPA = torch.zeros_like(y[:, 0])
        alg_15_VLA = torch.zeros_like(y[:, 0])
        alg_16_VLE = torch.zeros_like(y[:, 0])
        alg_17_PLA = torch.zeros_like(y[:, 0])
        alg_18_PL1 = torch.zeros_like(y[:, 0])
        alg_19_RPV = torch.zeros_like(y[:, 0])
        alg_20_RPT = torch.zeros_like(y[:, 0])
        alg_21_PGL = torch.zeros_like(y[:, 0])
        alg_22_QPO = torch.zeros_like(y[:, 0])
        alg_23_PLA1 = torch.zeros_like(y[:, 0])
        alg_24_QLN = torch.zeros_like(y[:, 0])
        alg_25_VVS = torch.zeros_like(y[:, 0])
        alg_26_VVE1 = torch.zeros_like(y[:, 0])
        alg_27_VVE = torch.zeros_like(y[:, 0])
        alg_28_PVS1 = torch.zeros_like(y[:, 0])
        alg_29_PVS = torch.zeros_like(y[:, 0])
        alg_30_PR1 = torch.zeros_like(y[:, 0])
        alg_31_RVG = torch.zeros_like(y[:, 0])
        alg_32_PGV = torch.zeros_like(y[:, 0])
        alg_33_QVO = torch.zeros_like(y[:, 0])
        alg_34_VAS = torch.zeros_like(y[:, 0])
        alg_35_VAE = torch.zeros_like(y[:, 0])
        alg_36_PA = torch.zeros_like(y[:, 0])
        alg_37_PA2 = torch.zeros_like(y[:, 0])
        alg_38_LVM = torch.zeros_like(y[:, 0])
        alg_39_QLOT = torch.zeros_like(y[:, 0])
        alg_40_QLO1 = torch.zeros_like(y[:, 0])
        alg_41_QLO = torch.zeros_like(y[:, 0])
        alg_42_HPEF = torch.zeros_like(y[:, 0])
        alg_43_DLA = torch.zeros_like(y[:, 0])
        alg_44_PAG = torch.zeros_like(y[:, 0])
        alg_45_QRO = torch.zeros_like(y[:, 0])
        alg_46_PAM = torch.zeros_like(y[:, 0])
        alg_47_DRA = torch.zeros_like(y[:, 0])
        alg_48_DPA = torch.zeros_like(y[:, 0])
        alg_49_R1 = torch.zeros_like(y[:, 0])
        alg_50_NNRAR = torch.zeros_like(y[:, 0])
        alg_51_PGS = torch.zeros_like(y[:, 0])
        alg_52_RSM = torch.zeros_like(y[:, 0])
        alg_53_RSN = torch.zeros_like(y[:, 0])
        alg_54_BFM = torch.zeros_like(y[:, 0])
        alg_55_BFN = torch.zeros_like(y[:, 0])
        alg_56_FISFLO = torch.zeros_like(y[:, 0])
        alg_57_SYSFLO = torch.zeros_like(y[:, 0])
        alg_58_QAO = torch.zeros_like(y[:, 0])
        alg_59_DVS = torch.zeros_like(y[:, 0])
        alg_60_DAS = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_VBD = (VP + VRC - VVS1 - VAS1 - VLA1 - VPA1 - VRA1) / 2.0
        alg_15_VLA = VLA1 + alg_0_VBD * 0.128
        alg_16_VLE = alg_15_VLA - 0.38
        alg_17_PLA = alg_16_VLE / 0.01
        alg_23_PLA1 = (alg_17_PLA + 4.0) * (HTAUML * (AU - 1.0) + 1.0) - 4.0
        alg_24_QLN = torch.where(alg_23_PLA1 <= -2.0, 0.01, torch.where((alg_23_PLA1 > -2.0) & (alg_23_PLA1 <= 1.0), 0.01 + (3.6 - 0.01) * (alg_23_PLA1 - -2.0) / (1.0 - -2.0), torch.where((alg_23_PLA1 > 1.0) & (alg_23_PLA1 <= 5.0), 3.6 + (9.4 - 3.6) * (alg_23_PLA1 - 1.0) / (5.0 - 1.0), torch.where((alg_23_PLA1 > 5.0) & (alg_23_PLA1 <= 8.0), 9.4 + (11.6 - 9.4) * (alg_23_PLA1 - 5.0) / (8.0 - 5.0), torch.where((alg_23_PLA1 > 8.0) & (alg_23_PLA1 <= 12.0), 11.6 + (13.5 - 11.6) * (alg_23_PLA1 - 8.0) / (12.0 - 8.0), 13.5)))))
        alg_34_VAS = VAS1 + alg_0_VBD * 0.261
        alg_35_VAE = alg_34_VAS - 0.495
        alg_36_PA = alg_35_VAE / 0.00355
        alg_37_PA2 = alg_36_PA / (AUH * OSA)
        alg_38_LVM = torch.where(alg_37_PA2 <= 0.0, 1.04, torch.where((alg_37_PA2 > 0.0) & (alg_37_PA2 <= 60.0), 1.04 + (1.025 - 1.04) * (alg_37_PA2 - 0.0) / (60.0 - 0.0), torch.where((alg_37_PA2 > 60.0) & (alg_37_PA2 <= 125.0), 1.025 + (0.97 - 1.025) * (alg_37_PA2 - 60.0) / (125.0 - 60.0), torch.where((alg_37_PA2 > 125.0) & (alg_37_PA2 <= 160.0), 0.97 + (0.88 - 0.97) * (alg_37_PA2 - 125.0) / (160.0 - 125.0), torch.where((alg_37_PA2 > 160.0) & (alg_37_PA2 <= 200.0), 0.88 + (0.59 - 0.88) * (alg_37_PA2 - 160.0) / (200.0 - 160.0), torch.where((alg_37_PA2 > 200.0) & (alg_37_PA2 <= 240.0), 0.59 + (0.0 - 0.59) * (alg_37_PA2 - 200.0) / (240.0 - 200.0), 0.0))))))
        alg_39_QLOT = alg_38_LVM * alg_24_QLN * AUH * HSL * HMD * HPL
        alg_40_QLO1 = (alg_17_PLA - alg_36_PA) / 3.0
        alg_41_QLO = torch.where(alg_40_QLO1 > 0.0, alg_39_QLOT + alg_40_QLO1, alg_39_QLOT)
        alg_6_VPA = VPA1 + alg_0_VBD * 0.155
        alg_7_VPE = alg_6_VPA - 0.30625
        alg_8_PPA = alg_7_VPE / 0.0048
        alg_11_PP1T = 0.026 * alg_8_PPA
        alg_12_PP1 = torch.where(alg_11_PP1T < 1e-05, 1e-05, alg_11_PP1T)
        alg_13_CPA = alg_12_PP1 ** 0.5
        alg_14_RPA = 1.0 / alg_13_CPA
        alg_18_PL1 = alg_17_PLA + 18.0
        alg_19_RPV = 1.0 / (alg_18_PL1 * 0.0357)
        alg_20_RPT = alg_19_RPV + alg_14_RPA
        alg_21_PGL = alg_8_PPA - alg_17_PLA
        alg_22_QPO = alg_21_PGL / alg_20_RPT
        alg_43_DLA = alg_22_QPO - alg_41_QLO
        d_VLA1 = torch.zeros_like(y[:, 0]) + (alg_43_DLA)
        alg_25_VVS = VVS1 + alg_0_VBD * 0.3986
        alg_26_VVE1 = alg_25_VVS - VVR - VVA - VV7 - VV6 - ATRVFB
        alg_27_VVE = torch.where(alg_26_VVE1 < 0.0001, 0.0001, alg_26_VVE1)
        alg_28_PVS1 = 3.7 + (alg_27_VVE - 0.74) / CV
        alg_29_PVS = torch.where(alg_28_PVS1 < 0.0001, 0.0001, alg_28_PVS1)
        alg_31_RVG = 0.74 / (alg_29_PVS / (VIM * 3.7)) ** 0.5
        alg_1_VRA = VRA1 + alg_0_VBD * 0.0574
        alg_2_VRE = alg_1_VRA - 0.1
        alg_3_PRA = alg_2_VRE / 0.005
        alg_30_PR1 = torch.where(alg_3_PRA < PR1LL, PR1LL, alg_3_PRA)
        alg_32_PGV = alg_29_PVS - alg_30_PR1
        alg_33_QVO = alg_32_PGV / alg_31_RVG
        alg_9_PP2 = alg_8_PPA / AUH / OSA
        alg_10_RVM = torch.where(alg_9_PP2 <= 0.0, 1.06, torch.where((alg_9_PP2 > 0.0) & (alg_9_PP2 <= 32.0), 1.06 + (0.97 - 1.06) * (alg_9_PP2 - 0.0) / (32.0 - 0.0), torch.where((alg_9_PP2 > 32.0) & (alg_9_PP2 <= 38.4), 0.97 + (0.93 - 0.97) * (alg_9_PP2 - 32.0) / (38.4 - 32.0), torch.where((alg_9_PP2 > 38.4) & (alg_9_PP2 <= 48.0), 0.93 + (0.8 - 0.93) * (alg_9_PP2 - 38.4) / (48.0 - 38.4), torch.where((alg_9_PP2 > 48.0) & (alg_9_PP2 <= 60.8), 0.8 + (0.46 - 0.8) * (alg_9_PP2 - 48.0) / (60.8 - 48.0), torch.where((alg_9_PP2 > 60.8) & (alg_9_PP2 <= 72.0), 0.46 + (0.0 - 0.46) * (alg_9_PP2 - 60.8) / (72.0 - 60.8), 0.0))))))
        alg_42_HPEF = (1.0 - QRF) * AUH * alg_10_RVM * HSR * HMD * HPR + QRF * alg_41_QLO / alg_24_QLN
        alg_4_PRA1 = (alg_3_PRA + 8.0) * (HTAUML * (AU - 1.0) + 1.0) - 8.0
        alg_5_QRN = torch.where(alg_4_PRA1 <= -8.0, 0.0, torch.where((alg_4_PRA1 > -8.0) & (alg_4_PRA1 <= -6.0), 0.0 + (0.75 - 0.0) * (alg_4_PRA1 - -8.0) / (-6.0 - -8.0), torch.where((alg_4_PRA1 > -6.0) & (alg_4_PRA1 <= -2.0), 0.75 + (2.6 - 0.75) * (alg_4_PRA1 - -6.0) / (-2.0 - -6.0), torch.where((alg_4_PRA1 > -2.0) & (alg_4_PRA1 <= 4.0), 2.6 + (9.8 - 2.6) * (alg_4_PRA1 - -2.0) / (4.0 - -2.0), torch.where((alg_4_PRA1 > 4.0) & (alg_4_PRA1 <= 12.0), 9.8 + (13.5 - 9.8) * (alg_4_PRA1 - 4.0) / (12.0 - 4.0), 13.5)))))
        alg_45_QRO = alg_5_QRN * alg_42_HPEF
        alg_47_DRA = alg_33_QVO - alg_45_QRO
        d_VRA1 = torch.zeros_like(y[:, 0]) + (alg_47_DRA)
        alg_48_DPA = alg_45_QRO - alg_22_QPO
        d_VPA1 = torch.zeros_like(y[:, 0]) + (alg_48_DPA)
        alg_44_PAG = alg_36_PA - alg_3_PRA
        alg_56_FISFLO = alg_44_PAG * FIS
        alg_51_PGS = alg_36_PA - alg_29_PVS
        alg_46_PAM = (alg_36_PA / 100.0) ** PAEX
        alg_49_R1 = ANU * ADHMV * AUM * VIM * PAMK / alg_46_PAM / ATRRFB
        alg_52_RSM = RAM * AMM * alg_49_R1 * MYOGRS * RMULT1
        alg_54_BFM = alg_51_PGS / alg_52_RSM
        alg_50_NNRAR = RAR * ARM * alg_49_R1 * MYOGRS * RMULT1
        alg_53_RSN = alg_50_NNRAR + NNRVR
        alg_55_BFN = alg_51_PGS / alg_53_RSN
        alg_57_SYSFLO = alg_54_BFM + alg_55_BFN + RBF
        alg_58_QAO = alg_57_SYSFLO + alg_56_FISFLO
        alg_59_DVS = alg_58_QAO - alg_33_QVO
        d_VVS1 = torch.zeros_like(y[:, 0]) + (alg_59_DVS)
        alg_60_DAS = alg_41_QLO - alg_58_QAO
        d_VAS1 = torch.zeros_like(y[:, 0]) + (alg_60_DAS)

        return torch.stack(
            [d_VVS1, d_VAS1, d_VLA1, d_VPA1, d_VRA1],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
