import torch
import torch.nn as nn
import math


class MetabolismHynneDanoSorensen2001ODE(nn.Module):
    """
    PyTorch translation of metabolism/hynne_dano_sorensen_2001/model.py.
    Dynamic states: 22; independent constants: 60; derived constants used by rates: 0.
    Original constant slots: 60; algebraic slots: 24.
    """
    source_file = 'metabolism/hynne_dano_sorensen_2001/model.py'
    source_state_dim = 22
    source_constant_dim = 60
    algebraic_dim = 24
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 22
        self.parameter_dim = 60

        self.parameters = (
            parameters
            if parameters is not None
            else {'Glc_x_0': 24, 'CN_x_0': 5.6, 'k0': 0.048, 'K2_Glc': 1.7, 'K2_IG6P': 1.2, 'K2_IIG6P': 7.2, 'V_2m': 1014.96, 'P2': 1, 'y_vol': 59, 'K3_DGlc': 0.37, 'K3_Glc': 0, 'K3_ATP': 0.1, 'V_3m': 51.7547, 'V_4m': 496.042, 'K4_G6P': 0.8, 'K4_F6P': 0.15, 'K4_eq': 0.13, 'K5': 0.021, 'kappa5': 0.15, 'V_5m': 45.4327, 'K6_eq': 0.081, 'K6_FBP': 0.3, 'K6_DHAP': 2, 'K6_GAP': 4, 'K6_IGAP': 10, 'V_6r': 11039.1, 'V_6f': 2207.82, 'K7_eq': 0.055, 'K7_DHAP': 1.23, 'K7_GAP': 1.27, 'V_7m': 116.365, 'K8_NAD': 0.1, 'K8_NADH': 0.06, 'K8_GAP': 0.6, 'K8_BPG': 0.01, 'K8_eq': 0.0055, 'V_8m': 833.858, 'k9f': 443866, 'k9r': 1528.62, 'K10_PEP': 0.2, 'K10_ADP': 0.17, 'V_10m': 343.096, 'K11': 0.3, 'V_11m': 53.1328, 'K12_NADH': 0.1, 'K12_ACA': 0.71, 'V_12m': 89.8023, 'k13': 16.72, 'K15_NADH': 0.13, 'K15_INADH': 0.034, 'K15_INAD': 0.13, 'K15_DHAP': 25, 'V_15m': 81.4797, 'k16': 1.9, 'k18': 24.7, 'k20': 0.00283828, 'k22': 2.25932, 'k23': 3.2076, 'k24f': 432.9, 'k24r': 133.333}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Glc_x': 6.7, 'Glc': 0.573074, 'G6P': 4.2, 'F6P': 0.49, 'FBP': 4.64, 'GAP': 0.115, 'DHAP': 2.95, 'BPG': 0.00027, 'PEP': 0.04, 'Pyr': 8.7, 'ATP': 2.1, 'ADP': 1.5, 'AMP': 0.33, 'CN_x': 5.20358, 'ACA': 1.48153, 'ACA_x': 1.28836, 'EtOH': 19.2379, 'EtOH_x': 16.4514, 'Glyc': 4.196, 'Glyc_x': 1.68478, 'NADH': 0.33, 'NAD': 0.65}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Glc_x_0', 'CN_x_0', 'k0', 'K2_Glc', 'K2_IG6P', 'K2_IIG6P', 'V_2m', 'P2', 'y_vol', 'K3_DGlc', 'K3_Glc', 'K3_ATP', 'V_3m', 'V_4m', 'K4_G6P', 'K4_F6P', 'K4_eq', 'K5', 'kappa5', 'V_5m', 'K6_eq', 'K6_FBP', 'K6_DHAP', 'K6_GAP', 'K6_IGAP', 'V_6r', 'V_6f', 'K7_eq', 'K7_DHAP', 'K7_GAP', 'V_7m', 'K8_NAD', 'K8_NADH', 'K8_GAP', 'K8_BPG', 'K8_eq', 'V_8m', 'k9f', 'k9r', 'K10_PEP', 'K10_ADP', 'V_10m', 'K11', 'V_11m', 'K12_NADH', 'K12_ACA', 'V_12m', 'k13', 'K15_NADH', 'K15_INADH', 'K15_INAD', 'K15_DHAP', 'V_15m', 'k16', 'k18', 'k20', 'k22', 'k23', 'k24f', 'k24r']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Glc_x', 'Glc', 'G6P', 'F6P', 'FBP', 'GAP', 'DHAP', 'BPG', 'PEP', 'Pyr', 'ATP', 'ADP', 'AMP', 'CN_x', 'ACA', 'ACA_x', 'EtOH', 'EtOH_x', 'Glyc', 'Glyc_x', 'NADH', 'NAD']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (22)
        Glc_x = y[:, 0]
        Glc = y[:, 1]
        G6P = y[:, 2]
        F6P = y[:, 3]
        FBP = y[:, 4]
        GAP = y[:, 5]
        DHAP = y[:, 6]
        BPG = y[:, 7]
        PEP = y[:, 8]
        Pyr = y[:, 9]
        ATP = y[:, 10]
        ADP = y[:, 11]
        AMP = y[:, 12]
        CN_x = y[:, 13]
        ACA = y[:, 14]
        ACA_x = y[:, 15]
        EtOH = y[:, 16]
        EtOH_x = y[:, 17]
        Glyc = y[:, 18]
        Glyc_x = y[:, 19]
        NADH = y[:, 20]
        NAD = y[:, 21]

        # Independent parameters (source constant order)
        Glc_x_0 = theta[:, 0]  # constants[0]
        CN_x_0 = theta[:, 1]  # constants[1]
        k0 = theta[:, 2]  # constants[2]
        K2_Glc = theta[:, 3]  # constants[3]
        K2_IG6P = theta[:, 4]  # constants[4]
        K2_IIG6P = theta[:, 5]  # constants[5]
        V_2m = theta[:, 6]  # constants[6]
        P2 = theta[:, 7]  # constants[7]
        y_vol = theta[:, 8]  # constants[8]
        K3_DGlc = theta[:, 9]  # constants[9]
        K3_Glc = theta[:, 10]  # constants[10]
        K3_ATP = theta[:, 11]  # constants[11]
        V_3m = theta[:, 12]  # constants[12]
        V_4m = theta[:, 13]  # constants[13]
        K4_G6P = theta[:, 14]  # constants[14]
        K4_F6P = theta[:, 15]  # constants[15]
        K4_eq = theta[:, 16]  # constants[16]
        K5 = theta[:, 17]  # constants[17]
        kappa5 = theta[:, 18]  # constants[18]
        V_5m = theta[:, 19]  # constants[19]
        K6_eq = theta[:, 20]  # constants[20]
        K6_FBP = theta[:, 21]  # constants[21]
        K6_DHAP = theta[:, 22]  # constants[22]
        K6_GAP = theta[:, 23]  # constants[23]
        K6_IGAP = theta[:, 24]  # constants[24]
        V_6r = theta[:, 25]  # constants[25]
        V_6f = theta[:, 26]  # constants[26]
        K7_eq = theta[:, 27]  # constants[27]
        K7_DHAP = theta[:, 28]  # constants[28]
        K7_GAP = theta[:, 29]  # constants[29]
        V_7m = theta[:, 30]  # constants[30]
        K8_NAD = theta[:, 31]  # constants[31]
        K8_NADH = theta[:, 32]  # constants[32]
        K8_GAP = theta[:, 33]  # constants[33]
        K8_BPG = theta[:, 34]  # constants[34]
        K8_eq = theta[:, 35]  # constants[35]
        V_8m = theta[:, 36]  # constants[36]
        k9f = theta[:, 37]  # constants[37]
        k9r = theta[:, 38]  # constants[38]
        K10_PEP = theta[:, 39]  # constants[39]
        K10_ADP = theta[:, 40]  # constants[40]
        V_10m = theta[:, 41]  # constants[41]
        K11 = theta[:, 42]  # constants[42]
        V_11m = theta[:, 43]  # constants[43]
        K12_NADH = theta[:, 44]  # constants[44]
        K12_ACA = theta[:, 45]  # constants[45]
        V_12m = theta[:, 46]  # constants[46]
        k13 = theta[:, 47]  # constants[47]
        K15_NADH = theta[:, 48]  # constants[48]
        K15_INADH = theta[:, 49]  # constants[49]
        K15_INAD = theta[:, 50]  # constants[50]
        K15_DHAP = theta[:, 51]  # constants[51]
        V_15m = theta[:, 52]  # constants[52]
        k16 = theta[:, 53]  # constants[53]
        k18 = theta[:, 54]  # constants[54]
        k20 = theta[:, 55]  # constants[55]
        k22 = theta[:, 56]  # constants[56]
        k23 = theta[:, 57]  # constants[57]
        k24f = theta[:, 58]  # constants[58]
        k24r = theta[:, 59]  # constants[59]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Glc_x = torch.zeros_like(y[:, 0])
        d_Glc = torch.zeros_like(y[:, 0])
        d_G6P = torch.zeros_like(y[:, 0])
        d_F6P = torch.zeros_like(y[:, 0])
        d_FBP = torch.zeros_like(y[:, 0])
        d_GAP = torch.zeros_like(y[:, 0])
        d_DHAP = torch.zeros_like(y[:, 0])
        d_BPG = torch.zeros_like(y[:, 0])
        d_PEP = torch.zeros_like(y[:, 0])
        d_Pyr = torch.zeros_like(y[:, 0])
        d_ATP = torch.zeros_like(y[:, 0])
        d_ADP = torch.zeros_like(y[:, 0])
        d_AMP = torch.zeros_like(y[:, 0])
        d_CN_x = torch.zeros_like(y[:, 0])
        d_ACA = torch.zeros_like(y[:, 0])
        d_ACA_x = torch.zeros_like(y[:, 0])
        d_EtOH = torch.zeros_like(y[:, 0])
        d_EtOH_x = torch.zeros_like(y[:, 0])
        d_Glyc = torch.zeros_like(y[:, 0])
        d_Glyc_x = torch.zeros_like(y[:, 0])
        d_NADH = torch.zeros_like(y[:, 0])
        d_NAD = torch.zeros_like(y[:, 0])
        alg_0_inGlc = torch.zeros_like(y[:, 0])
        alg_1_GlcTrans = torch.zeros_like(y[:, 0])
        alg_2_HK = torch.zeros_like(y[:, 0])
        alg_3_PGI = torch.zeros_like(y[:, 0])
        alg_4_PFK = torch.zeros_like(y[:, 0])
        alg_5_ALD = torch.zeros_like(y[:, 0])
        alg_6_TIM = torch.zeros_like(y[:, 0])
        alg_7_GAPDH = torch.zeros_like(y[:, 0])
        alg_8_lpPEP = torch.zeros_like(y[:, 0])
        alg_9_PK = torch.zeros_like(y[:, 0])
        alg_10_storage = torch.zeros_like(y[:, 0])
        alg_11_PDC = torch.zeros_like(y[:, 0])
        alg_12_consum = torch.zeros_like(y[:, 0])
        alg_13_ADH = torch.zeros_like(y[:, 0])
        alg_14_lpGlyc = torch.zeros_like(y[:, 0])
        alg_15_AK = torch.zeros_like(y[:, 0])
        alg_16_outACA = torch.zeros_like(y[:, 0])
        alg_17_outEtOH = torch.zeros_like(y[:, 0])
        alg_18_difACA = torch.zeros_like(y[:, 0])
        alg_19_difEtOH = torch.zeros_like(y[:, 0])
        alg_20_outGlyc = torch.zeros_like(y[:, 0])
        alg_21_lacto = torch.zeros_like(y[:, 0])
        alg_22_difGlyc = torch.zeros_like(y[:, 0])
        alg_23_inCN = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_GlcTrans = V_2m / y_vol * (Glc_x / K2_Glc / (1.0 + Glc_x / K2_Glc + (P2 * (Glc_x / K2_Glc) + 1.0) / (P2 * (Glc / K2_Glc) + 1.0) * (1.0 + Glc / K2_Glc + G6P / K2_IG6P + Glc * G6P / (K2_Glc * K2_IIG6P)))) - V_2m / y_vol * (Glc / K2_Glc / (1.0 + Glc / K2_Glc + (P2 * (Glc / K2_Glc) + 1.0) / (P2 * (Glc_x / K2_Glc) + 1.0) * (1.0 + Glc_x / K2_Glc) + G6P / K2_IG6P + Glc * G6P / (K2_Glc * K2_IIG6P)))
        alg_0_inGlc = k0 * (Glc_x_0 - Glc_x)
        d_Glc_x = torch.zeros_like(y[:, 0]) + (alg_0_inGlc - alg_1_GlcTrans)
        alg_2_HK = V_3m * ATP * Glc / (K3_DGlc * K3_ATP + K3_Glc * ATP + K3_ATP * Glc + Glc * ATP)
        d_Glc = torch.zeros_like(y[:, 0]) + (59.0 * alg_1_GlcTrans - alg_2_HK)
        alg_3_PGI = V_4m * G6P / (K4_G6P + G6P + K4_G6P / K4_F6P * F6P) - V_4m * (F6P / K4_eq) / (K4_G6P + G6P + K4_G6P / K4_F6P * F6P)
        alg_4_PFK = V_5m * F6P ** 2.0 / (K5 * (1.0 + kappa5 * (ATP / AMP) * (ATP / AMP)) + F6P ** 2.0)
        d_F6P = torch.zeros_like(y[:, 0]) + (alg_3_PGI - alg_4_PFK)
        alg_5_ALD = V_6f * FBP / (K6_FBP + FBP + GAP * K6_DHAP * V_6f / (K6_eq * V_6r) + DHAP * K6_GAP * V_6f / (K6_eq * V_6r) + FBP * GAP / K6_IGAP + GAP * DHAP * V_6f / (K6_eq * V_6r)) - V_6f * (GAP * DHAP / K6_eq) / (K6_FBP + FBP + GAP * K6_DHAP * V_6f / (K6_eq * V_6r) + DHAP * K6_GAP * V_6f / (K6_eq * V_6r) + FBP * GAP / K6_IGAP + GAP * DHAP * V_6f / (K6_eq * V_6r))
        d_FBP = torch.zeros_like(y[:, 0]) + (alg_4_PFK - alg_5_ALD)
        alg_7_GAPDH = V_8m * GAP * NAD / (K8_GAP * K8_NAD * (1.0 + GAP / K8_GAP + BPG / K8_BPG) * (1.0 + NAD / K8_NAD + NADH / K8_NADH)) - V_8m * (BPG * NADH / K8_eq) / (K8_GAP * K8_NAD * (1.0 + GAP / K8_GAP + BPG / K8_BPG) * (1.0 + NAD / K8_NAD + NADH / K8_NADH))
        alg_6_TIM = V_7m * DHAP / (K7_DHAP + DHAP + K7_DHAP / K7_GAP * GAP) - V_7m * (GAP / K7_eq) / (K7_DHAP + DHAP + K7_DHAP / K7_GAP * GAP)
        d_GAP = torch.zeros_like(y[:, 0]) + (alg_5_ALD + alg_6_TIM - alg_7_GAPDH)
        alg_8_lpPEP = k9f * BPG * ADP - k9r * PEP * ATP
        d_BPG = torch.zeros_like(y[:, 0]) + (alg_7_GAPDH - alg_8_lpPEP)
        alg_9_PK = V_10m * ADP * PEP / ((K10_PEP + PEP) * (K10_ADP + ADP))
        d_PEP = torch.zeros_like(y[:, 0]) + (alg_8_lpPEP - alg_9_PK)
        alg_10_storage = k22 * ATP * G6P
        d_G6P = torch.zeros_like(y[:, 0]) + (alg_2_HK - (alg_3_PGI + alg_10_storage))
        alg_11_PDC = V_11m * Pyr / (K11 + Pyr)
        d_Pyr = torch.zeros_like(y[:, 0]) + (alg_9_PK - alg_11_PDC)
        alg_14_lpGlyc = V_15m * DHAP / (K15_DHAP * (1.0 + K15_INADH / NADH * (1.0 + NAD / K15_INAD)) + DHAP * (1.0 + K15_NADH / NADH * (1.0 + NAD / K15_INAD)))
        d_DHAP = torch.zeros_like(y[:, 0]) + (alg_5_ALD - (alg_6_TIM + alg_14_lpGlyc))
        alg_12_consum = k23 * ATP
        alg_15_AK = k24f * AMP * ATP - k24r * ADP ** 2.0
        d_ATP = torch.zeros_like(y[:, 0]) + (alg_9_PK + alg_8_lpPEP - (alg_4_PFK + alg_10_storage + alg_2_HK + alg_12_consum + alg_15_AK))
        d_ADP = torch.zeros_like(y[:, 0]) + (alg_4_PFK + alg_10_storage + alg_2_HK + alg_12_consum + 2.0 * alg_15_AK - (alg_9_PK + alg_8_lpPEP))
        d_AMP = torch.zeros_like(y[:, 0]) + (-alg_15_AK)
        alg_13_ADH = V_12m * ACA * NADH / ((K12_NADH + NADH) * (K12_ACA + ACA))
        d_NADH = torch.zeros_like(y[:, 0]) + (alg_7_GAPDH - (alg_13_ADH + alg_14_lpGlyc))
        d_NAD = torch.zeros_like(y[:, 0]) + (alg_13_ADH + alg_14_lpGlyc - alg_7_GAPDH)
        alg_18_difACA = k18 / y_vol * (ACA - ACA_x)
        d_ACA = torch.zeros_like(y[:, 0]) + (alg_11_PDC - (59.0 * alg_18_difACA + alg_13_ADH))
        alg_19_difEtOH = k13 / y_vol * (EtOH - EtOH_x)
        d_EtOH = torch.zeros_like(y[:, 0]) + (alg_13_ADH - 59.0 * alg_19_difEtOH)
        alg_17_outEtOH = k0 * EtOH_x
        d_EtOH_x = torch.zeros_like(y[:, 0]) + (alg_19_difEtOH - alg_17_outEtOH)
        alg_21_lacto = k20 * ACA_x * CN_x
        alg_16_outACA = k0 * ACA_x
        d_ACA_x = torch.zeros_like(y[:, 0]) + (alg_18_difACA - (alg_21_lacto + alg_16_outACA))
        alg_22_difGlyc = k16 / y_vol * (Glyc - Glyc_x)
        d_Glyc = torch.zeros_like(y[:, 0]) + (alg_14_lpGlyc - 59.0 * alg_22_difGlyc)
        alg_20_outGlyc = k0 * Glyc_x
        d_Glyc_x = torch.zeros_like(y[:, 0]) + (alg_22_difGlyc - alg_20_outGlyc)
        alg_23_inCN = k0 * (CN_x_0 - CN_x)
        d_CN_x = torch.zeros_like(y[:, 0]) + (alg_23_inCN - alg_21_lacto)

        return torch.stack(
            [d_Glc_x, d_Glc, d_G6P, d_F6P, d_FBP, d_GAP, d_DHAP, d_BPG, d_PEP, d_Pyr, d_ATP, d_ADP, d_AMP, d_CN_x, d_ACA, d_ACA_x, d_EtOH, d_EtOH_x, d_Glyc, d_Glyc_x, d_NADH, d_NAD],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
