import torch
import torch.nn as nn
import math


class SignalTransductionHatakeyamaKimuraNakaKawasakiYumotoIchikawaKimSaitoSaekiShirouzuYokoyamaKonagaya2003ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/hatakeyama_kimura_naka_kawasaki_yumoto_ichikawa_kim_saito_saeki_shirouzu_yokoyama_konagaya_2003/model.py.
    Dynamic states: 33; independent constants: 80; derived constants used by rates: 0.
    Original constant slots: 80; algebraic slots: 0.
    """
    source_file = 'signal_transduction/hatakeyama_kimura_naka_kawasaki_yumoto_ichikawa_kim_saito_saeki_shirouzu_yokoyama_konagaya_2003/model.py'
    source_state_dim = 33
    source_constant_dim = 80
    algebraic_dim = 0
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 33
        self.parameter_dim = 80

        self.parameters = (
            parameters
            if parameters is not None
            else {'kf1': 0.0012, 'kb1': 0.00076, 'kf2': 0.01, 'kb2': 0.1, 'kf3': 1, 'kb3': 0.01, 'kf34': 0.001, 'kb34': 0, 'V4': 62.5, 'k4': 50, 'kf5': 0.1, 'kb5': 1, 'kf6': 20, 'kb6': 5, 'kf7': 60, 'kb7': 546, 'kf8': 2040, 'kb8': 15700, 'kf9': 40.8, 'kb9': 0, 'V10': 0.0154, 'k10': 340, 'kf23': 0.1, 'kb23': 2, 'kf24': 9.85, 'kb24': 0.0985, 'kf25': 45.8, 'kb25': 0.047, 'V26': 2620, 'k26': 3680, 'two': 2, 'kf11': 0.222, 'k11': 0.181, 'V12': 0.289, 'k12': 0.0571, 'kf13': 1.53, 'k13': 11.7, 'kf14': 0.00673, 'k14': 8.07, 'E': 7, 'kf27': 16.9, 'k27': 39.1, 'V28': 17000, 'k28': 9.02, 'kf29': 507, 'kb29': 234, 'V30': 20000, 'k30': 80000, 'kf31': 0.107, 'k31': 4.35, 'V32': 20000, 'k32': 80000, 'kf33': 0.211, 'k33': 12, 'k16': 2200, 'k18': 60, 'PP2A': 11.4, 'one': 1, 'PP2A_2': 11.4, 'kf15': 3.5, 'k15': 317, 'kf16': 0.058, 'k16_2': 2200, 'kf17': 2.9, 'k17': 317, 'kf18': 0.058, 'k18_2': 60, 'k31_2': 4.35, 'k33_2': 12, 'one_2': 1, 'MKP3': 2.4, 'kf19': 9.5, 'k19': 146000, 'kf20': 0.3, 'k20': 160, 'kf21': 16, 'k21': 146000, 'kf22': 0.27, 'k22': 60, 'one_3': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'R': 80, 'Shc': 1000, 'PI3K': 10, 'HRG': 10, 'R_HRG': 0, 'R_HRG2': 0, 'Internalisation': 0, 'RP': 0, 'R_Shc': 0, 'R_ShP': 0, 'ShP': 0, 'R_ShGS': 0, 'ShGS': 0, 'GS': 10, 'R_PI3K': 0, 'R_PI3Kstar': 0, 'PI3Kstar': 0, 'RasGTP': 0, 'RasGDP': 120, 'Akt_PIPP': 0, 'RAF_star': 100, 'RAF': 0, 'MEKP': 0, 'MEKPP': 0, 'P': 800, 'PIP3': 0, 'Akt': 10, 'Akt_PIP3': 0, 'Akt_PIP': 0, 'MEK': 120, 'ERK': 1000, 'ERKP': 0, 'ERKPP': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['kf1', 'kb1', 'kf2', 'kb2', 'kf3', 'kb3', 'kf34', 'kb34', 'V4', 'k4', 'kf5', 'kb5', 'kf6', 'kb6', 'kf7', 'kb7', 'kf8', 'kb8', 'kf9', 'kb9', 'V10', 'k10', 'kf23', 'kb23', 'kf24', 'kb24', 'kf25', 'kb25', 'V26', 'k26', 'two', 'kf11', 'k11', 'V12', 'k12', 'kf13', 'k13', 'kf14', 'k14', 'E', 'kf27', 'k27', 'V28', 'k28', 'kf29', 'kb29', 'V30', 'k30', 'kf31', 'k31', 'V32', 'k32', 'kf33', 'k33', 'k16', 'k18', 'PP2A', 'one', 'PP2A_2', 'kf15', 'k15', 'kf16', 'k16_2', 'kf17', 'k17', 'kf18', 'k18_2', 'k31_2', 'k33_2', 'one_2', 'MKP3', 'kf19', 'k19', 'kf20', 'k20', 'kf21', 'k21', 'kf22', 'k22', 'one_3']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['R', 'Shc', 'PI3K', 'HRG', 'R_HRG', 'R_HRG2', 'Internalisation', 'RP', 'R_Shc', 'R_ShP', 'ShP', 'R_ShGS', 'ShGS', 'GS', 'R_PI3K', 'R_PI3Kstar', 'PI3Kstar', 'RasGTP', 'RasGDP', 'Akt_PIPP', 'RAF_star', 'RAF', 'MEKP', 'MEKPP', 'P', 'PIP3', 'Akt', 'Akt_PIP3', 'Akt_PIP', 'MEK', 'ERK', 'ERKP', 'ERKPP']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (33)
        R = y[:, 0]
        Shc = y[:, 1]
        PI3K = y[:, 2]
        HRG = y[:, 3]
        R_HRG = y[:, 4]
        R_HRG2 = y[:, 5]
        Internalisation = y[:, 6]
        RP = y[:, 7]
        R_Shc = y[:, 8]
        R_ShP = y[:, 9]
        ShP = y[:, 10]
        R_ShGS = y[:, 11]
        ShGS = y[:, 12]
        GS = y[:, 13]
        R_PI3K = y[:, 14]
        R_PI3Kstar = y[:, 15]
        PI3Kstar = y[:, 16]
        RasGTP = y[:, 17]
        RasGDP = y[:, 18]
        Akt_PIPP = y[:, 19]
        RAF_star = y[:, 20]
        RAF = y[:, 21]
        MEKP = y[:, 22]
        MEKPP = y[:, 23]
        P = y[:, 24]
        PIP3 = y[:, 25]
        Akt = y[:, 26]
        Akt_PIP3 = y[:, 27]
        Akt_PIP = y[:, 28]
        MEK = y[:, 29]
        ERK = y[:, 30]
        ERKP = y[:, 31]
        ERKPP = y[:, 32]

        # Independent parameters (source constant order)
        kf1 = theta[:, 0]  # constants[0]
        kb1 = theta[:, 1]  # constants[1]
        kf2 = theta[:, 2]  # constants[2]
        kb2 = theta[:, 3]  # constants[3]
        kf3 = theta[:, 4]  # constants[4]
        kb3 = theta[:, 5]  # constants[5]
        kf34 = theta[:, 6]  # constants[6]
        kb34 = theta[:, 7]  # constants[7]
        V4 = theta[:, 8]  # constants[8]
        k4 = theta[:, 9]  # constants[9]
        kf5 = theta[:, 10]  # constants[10]
        kb5 = theta[:, 11]  # constants[11]
        kf6 = theta[:, 12]  # constants[12]
        kb6 = theta[:, 13]  # constants[13]
        kf7 = theta[:, 14]  # constants[14]
        kb7 = theta[:, 15]  # constants[15]
        kf8 = theta[:, 16]  # constants[16]
        kb8 = theta[:, 17]  # constants[17]
        kf9 = theta[:, 18]  # constants[18]
        kb9 = theta[:, 19]  # constants[19]
        V10 = theta[:, 20]  # constants[20]
        k10 = theta[:, 21]  # constants[21]
        kf23 = theta[:, 22]  # constants[22]
        kb23 = theta[:, 23]  # constants[23]
        kf24 = theta[:, 24]  # constants[24]
        kb24 = theta[:, 25]  # constants[25]
        kf25 = theta[:, 26]  # constants[26]
        kb25 = theta[:, 27]  # constants[27]
        V26 = theta[:, 28]  # constants[28]
        k26 = theta[:, 29]  # constants[29]
        two = theta[:, 30]  # constants[30]
        kf11 = theta[:, 31]  # constants[31]
        k11 = theta[:, 32]  # constants[32]
        V12 = theta[:, 33]  # constants[33]
        k12 = theta[:, 34]  # constants[34]
        kf13 = theta[:, 35]  # constants[35]
        k13 = theta[:, 36]  # constants[36]
        kf14 = theta[:, 37]  # constants[37]
        k14 = theta[:, 38]  # constants[38]
        E = theta[:, 39]  # constants[39]
        kf27 = theta[:, 40]  # constants[40]
        k27 = theta[:, 41]  # constants[41]
        V28 = theta[:, 42]  # constants[42]
        k28 = theta[:, 43]  # constants[43]
        kf29 = theta[:, 44]  # constants[44]
        kb29 = theta[:, 45]  # constants[45]
        V30 = theta[:, 46]  # constants[46]
        k30 = theta[:, 47]  # constants[47]
        kf31 = theta[:, 48]  # constants[48]
        k31 = theta[:, 49]  # constants[49]
        V32 = theta[:, 50]  # constants[50]
        k32 = theta[:, 51]  # constants[51]
        kf33 = theta[:, 52]  # constants[52]
        k33 = theta[:, 53]  # constants[53]
        k16 = theta[:, 54]  # constants[54]
        k18 = theta[:, 55]  # constants[55]
        PP2A = theta[:, 56]  # constants[56]
        one = theta[:, 57]  # constants[57]
        PP2A_2 = theta[:, 58]  # constants[58]
        kf15 = theta[:, 59]  # constants[59]
        k15 = theta[:, 60]  # constants[60]
        kf16 = theta[:, 61]  # constants[61]
        k16_2 = theta[:, 62]  # constants[62]
        kf17 = theta[:, 63]  # constants[63]
        k17 = theta[:, 64]  # constants[64]
        kf18 = theta[:, 65]  # constants[65]
        k18_2 = theta[:, 66]  # constants[66]
        k31_2 = theta[:, 67]  # constants[67]
        k33_2 = theta[:, 68]  # constants[68]
        one_2 = theta[:, 69]  # constants[69]
        MKP3 = theta[:, 70]  # constants[70]
        kf19 = theta[:, 71]  # constants[71]
        k19 = theta[:, 72]  # constants[72]
        kf20 = theta[:, 73]  # constants[73]
        k20 = theta[:, 74]  # constants[74]
        kf21 = theta[:, 75]  # constants[75]
        k21 = theta[:, 76]  # constants[76]
        kf22 = theta[:, 77]  # constants[77]
        k22 = theta[:, 78]  # constants[78]
        one_3 = theta[:, 79]  # constants[79]

        # Initialize rates and algebraic variables, as in computeRates()
        d_R = torch.zeros_like(y[:, 0])
        d_Shc = torch.zeros_like(y[:, 0])
        d_PI3K = torch.zeros_like(y[:, 0])
        d_HRG = torch.zeros_like(y[:, 0])
        d_R_HRG = torch.zeros_like(y[:, 0])
        d_R_HRG2 = torch.zeros_like(y[:, 0])
        d_Internalisation = torch.zeros_like(y[:, 0])
        d_RP = torch.zeros_like(y[:, 0])
        d_R_Shc = torch.zeros_like(y[:, 0])
        d_R_ShP = torch.zeros_like(y[:, 0])
        d_ShP = torch.zeros_like(y[:, 0])
        d_R_ShGS = torch.zeros_like(y[:, 0])
        d_ShGS = torch.zeros_like(y[:, 0])
        d_GS = torch.zeros_like(y[:, 0])
        d_R_PI3K = torch.zeros_like(y[:, 0])
        d_R_PI3Kstar = torch.zeros_like(y[:, 0])
        d_PI3Kstar = torch.zeros_like(y[:, 0])
        d_RasGTP = torch.zeros_like(y[:, 0])
        d_RasGDP = torch.zeros_like(y[:, 0])
        d_Akt_PIPP = torch.zeros_like(y[:, 0])
        d_RAF_star = torch.zeros_like(y[:, 0])
        d_RAF = torch.zeros_like(y[:, 0])
        d_MEKP = torch.zeros_like(y[:, 0])
        d_MEKPP = torch.zeros_like(y[:, 0])
        d_P = torch.zeros_like(y[:, 0])
        d_PIP3 = torch.zeros_like(y[:, 0])
        d_Akt = torch.zeros_like(y[:, 0])
        d_Akt_PIP3 = torch.zeros_like(y[:, 0])
        d_Akt_PIP = torch.zeros_like(y[:, 0])
        d_MEK = torch.zeros_like(y[:, 0])
        d_ERK = torch.zeros_like(y[:, 0])
        d_ERKP = torch.zeros_like(y[:, 0])
        d_ERKPP = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_R = torch.zeros_like(y[:, 0]) + (-kf1 * R * HRG + kb1 * R_HRG)
        d_HRG = torch.zeros_like(y[:, 0]) + (-kf1 * R * HRG + kb1 * R_HRG)
        d_R_HRG = torch.zeros_like(y[:, 0]) + (kf1 * R * HRG - kb1 * R_HRG - two * (kf2 * R_HRG * R_HRG - kb2 * R_HRG2))
        d_R_HRG2 = torch.zeros_like(y[:, 0]) + (kf2 * R_HRG * R_HRG - kb2 * R_HRG2 - (kf3 * R_HRG2 - kb3 * RP) + V4 * RP / (k4 + RP))
        d_RP = torch.zeros_like(y[:, 0]) + (kf3 * R_HRG2 - kb3 * RP - V4 * RP / (k4 + RP) - (kf5 * RP * Shc - kb5 * R_Shc) + (kf8 * R_ShP - kb8 * ShGS * RP) - (kf23 * RP * PI3K - kb23 * R_PI3K) + (kf25 * R_PI3Kstar - kb25 * RP * PI3Kstar) - (kf34 * RP - kb34 * Internalisation))
        d_Internalisation = torch.zeros_like(y[:, 0]) + (kf34 * RP - kb34 * Internalisation)
        d_R_Shc = torch.zeros_like(y[:, 0]) + (kf5 * RP * Shc - kb5 * R_Shc - (kf6 * R_Shc - kb6 * R_ShP))
        d_Shc = torch.zeros_like(y[:, 0]) + (-(kf5 * RP * Shc - kb5 * R_Shc) + V10 * ShP / (k10 + ShP))
        d_R_ShP = torch.zeros_like(y[:, 0]) + (kf6 * R_Shc - kb6 * R_ShP - (kf7 * R_ShP - kb7 * R_ShGS))
        d_GS = torch.zeros_like(y[:, 0]) + (-(kf7 * R_ShP - kb7 * R_ShGS) + (kf9 * ShGS - kb9 * GS * ShP))
        d_ShP = torch.zeros_like(y[:, 0]) + (kf9 * ShGS - kb9 * GS * ShP - V10 * ShP / (k10 + ShP))
        d_R_ShGS = torch.zeros_like(y[:, 0]) + (kf7 * R_ShP - kb7 * R_ShGS - (kf8 * R_ShP - kb8 * ShGS * RP))
        d_ShGS = torch.zeros_like(y[:, 0]) + (kf8 * R_ShP - kb8 * ShGS * RP - (kf9 * ShGS - kb9 * GS * ShP))
        d_R_PI3K = torch.zeros_like(y[:, 0]) + (kf23 * RP * PI3K - kb23 * R_PI3K - (kf24 * R_PI3K - kb24 * R_PI3Kstar))
        d_PI3K = torch.zeros_like(y[:, 0]) + (-(kf23 * RP * PI3K - kb23 * R_PI3K) + V26 * PI3Kstar / (k26 + PI3Kstar))
        d_R_PI3Kstar = torch.zeros_like(y[:, 0]) + (kf24 * R_PI3K - kb24 * R_PI3Kstar - (kf25 * R_PI3Kstar - kb25 * RP * PI3Kstar))
        d_PI3Kstar = torch.zeros_like(y[:, 0]) + (kf25 * R_PI3Kstar - kb25 * RP * PI3Kstar - V26 * PI3Kstar / (k26 + PI3Kstar))
        d_RasGDP = torch.zeros_like(y[:, 0]) + (-(kf11 * ShGS * RasGDP / (k11 + RasGDP)) + V12 * RasGTP / (k12 + RasGTP))
        d_RasGTP = torch.zeros_like(y[:, 0]) + (kf11 * ShGS * RasGDP / (k11 + RasGDP) - V12 * RasGTP / (k12 + RasGTP))
        d_RAF = torch.zeros_like(y[:, 0]) + (kf14 * (Akt_PIPP + E) * RAF_star / (k14 + RAF_star) - kf13 * RasGTP * RAF / (k13 + RAF))
        d_RAF_star = torch.zeros_like(y[:, 0]) + (-(kf14 * (Akt_PIPP + E) * RAF_star / (k14 + RAF_star)) + kf13 * RasGTP * RAF / (k13 + RAF))
        d_P = torch.zeros_like(y[:, 0]) + (V28 * PIP3 / (k28 + PIP3) - kf27 * PI3Kstar * P / (k27 + P))
        d_PIP3 = torch.zeros_like(y[:, 0]) + (-(V28 * PIP3 / (k28 + PIP3)) + kf27 * PI3Kstar * P / (k27 + P) - (kf29 * PIP3 * Akt - kb29 * Akt_PIP3))
        d_Akt = torch.zeros_like(y[:, 0]) + (-(kf29 * PIP3 * Akt - kb29 * Akt_PIP3))
        d_Akt_PIP3 = torch.zeros_like(y[:, 0]) + (kf29 * PIP3 * Akt - kb29 * Akt_PIP3 - V30 * Akt_PIP3 / (k30 * (one + Akt_PIP / k32) + Akt_PIP3) + kf31 * PP2A * Akt_PIP / (k31 * (one + MEKP / k16 + MEKPP / k18 + Akt_PIPP / k33) + Akt_PIP))
        d_Akt_PIP = torch.zeros_like(y[:, 0]) + (V30 * Akt_PIP3 / (k30 * (one + Akt_PIP / k32) + Akt_PIP3) - kf31 * PP2A * Akt_PIP / (k31 * (one + MEKP / k16 + MEKPP / k18 + Akt_PIPP / k33) + Akt_PIP) - V32 * Akt_PIP / (k32 * (one + Akt_PIP3 / k30) + Akt_PIP) + kf33 * PP2A * Akt_PIPP / (k33 * (one + MEKP / k16 + MEKPP / k18 + Akt_PIP / k31) + Akt_PIPP))
        d_Akt_PIPP = torch.zeros_like(y[:, 0]) + (V32 * Akt_PIP / (k32 * (one + Akt_PIP3 / k30) + Akt_PIP) - kf33 * PP2A * Akt_PIPP / (k33 * (one + MEKP / k16 + MEKPP / k18 + Akt_PIP / k31) + Akt_PIPP))
        d_MEK = torch.zeros_like(y[:, 0]) + (-(kf15 * RAF_star * MEK / (k15 * (one_2 + MEKP / k17) + MEK)) + kf16 * PP2A_2 * MEKP / (k16_2 * (one_2 + MEKPP / k18_2 + Akt_PIP / k31_2 + Akt_PIPP / k33_2) + MEKP))
        d_MEKP = torch.zeros_like(y[:, 0]) + (kf15 * RAF_star * MEK / (k15 * (one_2 + MEKP / k17) + MEK) - kf16 * PP2A_2 * MEKP / (k16_2 * (one_2 + MEKPP / k18_2 + Akt_PIP / k31_2 + Akt_PIPP / k33_2) + MEKP) - kf17 * RAF_star * MEKP / (k17 * (one_2 + MEK / k15) + MEKP) + kf18 * PP2A_2 * MEKPP / (k18_2 * (one_2 + MEKP / k16_2 + Akt_PIP / k31_2 + Akt_PIPP / k33_2) + MEKPP))
        d_MEKPP = torch.zeros_like(y[:, 0]) + (kf17 * RAF_star * MEKP / (k17 * (one_2 + MEK / k15) + MEKP) - kf18 * PP2A_2 * MEKPP / (k18_2 * (one_2 + MEKP / k16_2 + Akt_PIP / k31_2 + Akt_PIPP / k33_2) + MEKPP))
        d_ERK = torch.zeros_like(y[:, 0]) + (-(kf19 * MEKPP * ERK / (k19 * (one_3 + ERKP / k21) + ERK)) + kf20 * MKP3 * ERKP / (k20 * (one_3 + ERKPP / k22) + ERKP))
        d_ERKPP = torch.zeros_like(y[:, 0]) + (kf21 * MEKPP * ERKP / (k21 * (one_3 + ERK / k19) + ERKP) - kf22 * MKP3 * ERKPP / (k22 * (one_3 + ERKP / k20) + ERKPP))
        d_ERKP = torch.zeros_like(y[:, 0]) + (kf19 * MEKPP * ERK / (k19 * (one_3 + ERKP / k21) + ERK) - kf20 * MKP3 * ERKP / (k20 * (one_3 + ERKPP / k22) + ERKP) - kf21 * MEKPP * ERKP / (k21 * (one_3 + ERK / k19) + ERKP) + kf22 * MKP3 * ERKPP / (k22 * (one_3 + ERKP / k20) + ERKPP))

        return torch.stack(
            [d_R, d_Shc, d_PI3K, d_HRG, d_R_HRG, d_R_HRG2, d_Internalisation, d_RP, d_R_Shc, d_R_ShP, d_ShP, d_R_ShGS, d_ShGS, d_GS, d_R_PI3K, d_R_PI3Kstar, d_PI3Kstar, d_RasGTP, d_RasGDP, d_Akt_PIPP, d_RAF_star, d_RAF, d_MEKP, d_MEKPP, d_P, d_PIP3, d_Akt, d_Akt_PIP3, d_Akt_PIP, d_MEK, d_ERK, d_ERKP, d_ERKPP],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
