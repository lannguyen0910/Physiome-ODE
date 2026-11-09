import torch
import torch.nn as nn
import math


class EndocrinePedersenBertramSherman2005ODE(nn.Module):
    """
    PyTorch translation of endocrine/pedersen_bertram_sherman_2005/model.py.
    Dynamic states: 9; independent constants: 53; derived constants used by rates: 1.
    Original constant slots: 54; algebraic slots: 75.
    """
    source_file = 'endocrine/pedersen_bertram_sherman_2005/model.py'
    source_state_dim = 9
    source_constant_dim = 54
    algebraic_dim = 75
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 9
        self.parameter_dim = 53

        self.parameters = (
            parameters
            if parameters is not None
            else {'gluper': 7, 'gluamp': 0, 'glubasa1': 7, 'glustep': 0, 'steptime': 480000, 'kappa': 0.005, 'Kglut': 7, 'Vglut': 8, 'Kgk': 7, 'Vgk': 0.8, 'ngk': 4, 'pfkbas': 0.06, 'cat': 2, 'famp': 0.02, 'ffbp': 0.2, 'fmt': 20, 'fbt': 20, 'fatp': 20, 'K1': 30, 'K2': 1, 'K3': 50000, 'K4': 1000, 'Atot': 3000, 'tau_a': 300000, 'r': 0.5, 'r1': 0.35, 'autoadp': 1, 'adpknot': 800, 'ky': 2.2, 'kg': 0.1, 'cm': 5300, 'gK': 2700, 'vK': -75, 'gCa': 1000, 'vCa': 25, 'gkCa': 400, 'KD': 0.5, 'nh': 2, 'gkATP_bar': 2000, 'fcyt': 0.01, 'fer': 0.01, 'sigmav': 31, 'kPMCA': 0.18, 'alpha': 4.5e-06, 'kSERCA': 0.4, 'pleak': 0.0002, 'tau_I': 10000, 'I_max': 20, 'delta': 8, 'ki': 0.1, 'topa1': 0, 'weight1': 1, 'bottom1': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Gi': 6.0637, 'G6P': 525.97, 'FBP': 0.2088, 'ADP': 537.6, 'Ca': 0.05626, 'v': -66.7, 'n': 0.00012, 'Caer': 121.8, 'I': 0.316}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['gluper', 'gluamp', 'glubasa1', 'glustep', 'steptime', 'kappa', 'Kglut', 'Vglut', 'Kgk', 'Vgk', 'ngk', 'pfkbas', 'cat', 'famp', 'ffbp', 'fmt', 'fbt', 'fatp', 'K1', 'K2', 'K3', 'K4', 'Atot', 'tau_a', 'r', 'r1', 'autoadp', 'adpknot', 'ky', 'kg', 'cm', 'gK', 'vK', 'gCa', 'vCa', 'gkCa', 'KD', 'nh', 'gkATP_bar', 'fcyt', 'fer', 'sigmav', 'kPMCA', 'alpha', 'kSERCA', 'pleak', 'tau_I', 'I_max', 'delta', 'ki', 'topa1', 'weight1', 'bottom1']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Gi', 'G6P', 'FBP', 'ADP', 'Ca', 'v', 'n', 'Caer', 'I']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (9)
        Gi = y[:, 0]
        G6P = y[:, 1]
        FBP = y[:, 2]
        ADP = y[:, 3]
        Ca = y[:, 4]
        v = y[:, 5]
        n = y[:, 6]
        Caer = y[:, 7]
        I = y[:, 8]

        # Independent parameters (source constant order)
        gluper = theta[:, 0]  # constants[0]
        gluamp = theta[:, 1]  # constants[1]
        glubasa1 = theta[:, 2]  # constants[2]
        glustep = theta[:, 3]  # constants[3]
        steptime = theta[:, 4]  # constants[4]
        kappa = theta[:, 5]  # constants[5]
        Kglut = theta[:, 6]  # constants[6]
        Vglut = theta[:, 7]  # constants[7]
        Kgk = theta[:, 8]  # constants[8]
        Vgk = theta[:, 9]  # constants[9]
        ngk = theta[:, 10]  # constants[10]
        pfkbas = theta[:, 11]  # constants[11]
        cat = theta[:, 12]  # constants[12]
        famp = theta[:, 13]  # constants[13]
        ffbp = theta[:, 14]  # constants[14]
        fmt = theta[:, 15]  # constants[15]
        fbt = theta[:, 16]  # constants[16]
        fatp = theta[:, 17]  # constants[17]
        K1 = theta[:, 18]  # constants[18]
        K2 = theta[:, 19]  # constants[19]
        K3 = theta[:, 20]  # constants[20]
        K4 = theta[:, 21]  # constants[21]
        Atot = theta[:, 22]  # constants[22]
        tau_a = theta[:, 23]  # constants[23]
        r = theta[:, 24]  # constants[24]
        r1 = theta[:, 25]  # constants[25]
        autoadp = theta[:, 26]  # constants[26]
        adpknot = theta[:, 27]  # constants[27]
        ky = theta[:, 28]  # constants[28]
        kg = theta[:, 29]  # constants[29]
        cm = theta[:, 30]  # constants[30]
        gK = theta[:, 31]  # constants[31]
        vK = theta[:, 32]  # constants[32]
        gCa = theta[:, 33]  # constants[33]
        vCa = theta[:, 34]  # constants[34]
        gkCa = theta[:, 35]  # constants[35]
        KD = theta[:, 36]  # constants[36]
        nh = theta[:, 37]  # constants[37]
        gkATP_bar = theta[:, 38]  # constants[38]
        fcyt = theta[:, 39]  # constants[39]
        fer = theta[:, 40]  # constants[40]
        sigmav = theta[:, 41]  # constants[41]
        kPMCA = theta[:, 42]  # constants[42]
        alpha = theta[:, 43]  # constants[43]
        kSERCA = theta[:, 44]  # constants[44]
        pleak = theta[:, 45]  # constants[45]
        tau_I = theta[:, 46]  # constants[46]
        I_max = theta[:, 47]  # constants[47]
        delta = theta[:, 48]  # constants[48]
        ki = theta[:, 49]  # constants[49]
        topa1 = theta[:, 50]  # constants[50]
        weight1 = theta[:, 51]  # constants[51]
        bottom1 = theta[:, 52]  # constants[52]

        # Derived constants from the original initConsts()
        topa2 = topa1  # constants[53]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Gi = torch.zeros_like(y[:, 0])
        d_G6P = torch.zeros_like(y[:, 0])
        d_FBP = torch.zeros_like(y[:, 0])
        d_ADP = torch.zeros_like(y[:, 0])
        d_Ca = torch.zeros_like(y[:, 0])
        d_v = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_Caer = torch.zeros_like(y[:, 0])
        d_I = torch.zeros_like(y[:, 0])
        alg_0_GeStep = torch.zeros_like(y[:, 0])
        alg_1_tau_n = torch.zeros_like(y[:, 0])
        alg_2_I_infinity = torch.zeros_like(y[:, 0])
        alg_3_Ge = torch.zeros_like(y[:, 0])
        alg_4_n_infinity = torch.zeros_like(y[:, 0])
        alg_5_F6P = torch.zeros_like(y[:, 0])
        alg_6_Jglut = torch.zeros_like(y[:, 0])
        alg_7_Jgk = torch.zeros_like(y[:, 0])
        alg_8_JGPDH = torch.zeros_like(y[:, 0])
        alg_9_weight5 = torch.zeros_like(y[:, 0])
        alg_10_weight3 = torch.zeros_like(y[:, 0])
        alg_11_topa3 = torch.zeros_like(y[:, 0])
        alg_12_weight7 = torch.zeros_like(y[:, 0])
        alg_13_rad = torch.zeros_like(y[:, 0])
        alg_14_ATP = torch.zeros_like(y[:, 0])
        alg_15_weight2 = torch.zeros_like(y[:, 0])
        alg_16_y = torch.zeros_like(y[:, 0])
        alg_17_I_K = torch.zeros_like(y[:, 0])
        alg_18_bottom2 = torch.zeros_like(y[:, 0])
        alg_19_fback = torch.zeros_like(y[:, 0])
        alg_20_m_infinity = torch.zeros_like(y[:, 0])
        alg_21_bottom3 = torch.zeros_like(y[:, 0])
        alg_22_I_Ca = torch.zeros_like(y[:, 0])
        alg_23_weight6 = torch.zeros_like(y[:, 0])
        alg_24_I_K_Ca = torch.zeros_like(y[:, 0])
        alg_25_Jmem = torch.zeros_like(y[:, 0])
        alg_26_weight4 = torch.zeros_like(y[:, 0])
        alg_27_MgADP = torch.zeros_like(y[:, 0])
        alg_28_JSERCA = torch.zeros_like(y[:, 0])
        alg_29_topa4 = torch.zeros_like(y[:, 0])
        alg_30_topo = torch.zeros_like(y[:, 0])
        alg_31_Jleak = torch.zeros_like(y[:, 0])
        alg_32_topa5 = torch.zeros_like(y[:, 0])
        alg_33_ADP3 = torch.zeros_like(y[:, 0])
        alg_34_Jer = torch.zeros_like(y[:, 0])
        alg_35_topa6 = torch.zeros_like(y[:, 0])
        alg_36_ATP4 = torch.zeros_like(y[:, 0])
        alg_37_bottom4 = torch.zeros_like(y[:, 0])
        alg_38_bottomo = torch.zeros_like(y[:, 0])
        alg_39_bottom5 = torch.zeros_like(y[:, 0])
        alg_40_katpo = torch.zeros_like(y[:, 0])
        alg_41_bottom6 = torch.zeros_like(y[:, 0])
        alg_42_I_K_ATP = torch.zeros_like(y[:, 0])
        alg_43_topa7 = torch.zeros_like(y[:, 0])
        alg_44_bottom7 = torch.zeros_like(y[:, 0])
        alg_45_weight8 = torch.zeros_like(y[:, 0])
        alg_46_topa8 = torch.zeros_like(y[:, 0])
        alg_47_topa9 = torch.zeros_like(y[:, 0])
        alg_48_topa10 = torch.zeros_like(y[:, 0])
        alg_49_bottom8 = torch.zeros_like(y[:, 0])
        alg_50_AMP = torch.zeros_like(y[:, 0])
        alg_51_weight9 = torch.zeros_like(y[:, 0])
        alg_52_bottom9 = torch.zeros_like(y[:, 0])
        alg_53_weight13 = torch.zeros_like(y[:, 0])
        alg_54_weight11 = torch.zeros_like(y[:, 0])
        alg_55_topa11 = torch.zeros_like(y[:, 0])
        alg_56_weight10 = torch.zeros_like(y[:, 0])
        alg_57_bottom10 = torch.zeros_like(y[:, 0])
        alg_58_bottom11 = torch.zeros_like(y[:, 0])
        alg_59_weight15 = torch.zeros_like(y[:, 0])
        alg_60_topb = torch.zeros_like(y[:, 0])
        alg_61_weight12 = torch.zeros_like(y[:, 0])
        alg_62_topa12 = torch.zeros_like(y[:, 0])
        alg_63_topa13 = torch.zeros_like(y[:, 0])
        alg_64_bottom12 = torch.zeros_like(y[:, 0])
        alg_65_bottom13 = torch.zeros_like(y[:, 0])
        alg_66_weight14 = torch.zeros_like(y[:, 0])
        alg_67_topa14 = torch.zeros_like(y[:, 0])
        alg_68_topa15 = torch.zeros_like(y[:, 0])
        alg_69_bottom14 = torch.zeros_like(y[:, 0])
        alg_70_bottom15 = torch.zeros_like(y[:, 0])
        alg_71_weight16 = torch.zeros_like(y[:, 0])
        alg_72_topa16 = torch.zeros_like(y[:, 0])
        alg_73_bottom16 = torch.zeros_like(y[:, 0])
        alg_74_JPFK = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_2_I_infinity = I_max * (Ca ** delta / (ki ** delta + Ca ** delta))
        d_I = torch.zeros_like(y[:, 0]) + ((alg_2_I_infinity - I) / tau_I)
        alg_4_n_infinity = 0.5 * (1.0 + torch.tanh((v - -16.0) / 11.2))
        alg_1_tau_n = 1.0 / (0.035 * torch.cosh((v - -16.0) / 22.4))
        d_n = torch.zeros_like(y[:, 0]) + ((alg_4_n_infinity - n) / alg_1_tau_n)
        alg_0_GeStep = torch.where((t - steptime) / 1000.0 > 1.0, glustep * gluamp, torch.where(((t - steptime) / 1000.0 > 0.0) & ((t - steptime) / 1000.0 < 1.0), (t - steptime) / 1000.0 * glustep * gluamp, torch.where((t - steptime) / 1000.0 < 0.0, 0.0, (1e309-1e309))))
        alg_3_Ge = glubasa1 + gluamp * (1.0 - glustep) * torch.cos(t * 2.0 * math.pi / (60000.0 * gluper)) + alg_0_GeStep
        alg_6_Jglut = Vglut * ((alg_3_Ge - Gi) * Kglut / ((Kglut + alg_3_Ge) * (Kglut + Gi)))
        alg_7_Jgk = Vgk * Gi ** ngk / (Kgk ** ngk + Gi ** ngk)
        d_Gi = torch.zeros_like(y[:, 0]) + (alg_6_Jglut - alg_7_Jgk)
        alg_13_rad = ((ADP - Atot) ** 2.0 - 4.0 * ADP ** 2.0) ** (1.0 / 2)
        alg_14_ATP = 0.5 * (Atot + alg_13_rad - ADP)
        alg_8_JGPDH = 0.2 * (FBP / 1.0) ** (1.0 / 2)
        alg_16_y = ky * alg_8_JGPDH / (kg + alg_8_JGPDH)
        alg_19_fback = r + alg_16_y
        d_ADP = torch.zeros_like(y[:, 0]) + (autoadp * ((alg_14_ATP - ADP * torch.exp(alg_19_fback * ((1.0 - Ca) / r1))) / tau_a) + 1.0 * (1.0 - autoadp) * (adpknot - ADP))
        alg_20_m_infinity = 0.5 * (1.0 + torch.tanh((v - -20.0) / 24.0))
        alg_22_I_Ca = gCa * alg_20_m_infinity * (v - vCa)
        alg_25_Jmem = -(alpha * 1.0 * alg_22_I_Ca + kPMCA * Ca)
        alg_31_Jleak = pleak * (Caer - Ca)
        alg_28_JSERCA = kSERCA * Ca
        alg_34_Jer = alg_31_Jleak - alg_28_JSERCA
        d_Ca = torch.zeros_like(y[:, 0]) + (fcyt * (alg_25_Jmem + alg_34_Jer))
        d_Caer = torch.zeros_like(y[:, 0]) + (-fer * sigmav * alg_34_Jer)
        alg_17_I_K = gK * n * (v - vK)
        alg_24_I_K_Ca = gkCa / (1.0 + (KD / Ca) ** nh) * (v - vK)
        alg_27_MgADP = 0.165 * ADP
        alg_30_topo = 0.08 * (1.0 + 2.0 * alg_27_MgADP / 17.0) + 0.89 * (alg_27_MgADP / 17.0) ** 2.0
        alg_33_ADP3 = 0.135 * ADP
        alg_36_ATP4 = 0.05 * alg_14_ATP
        alg_38_bottomo = (1.0 + alg_27_MgADP / 17.0) ** 2.0 * (1.0 + alg_33_ADP3 / 26.0 + alg_36_ATP4 / 1.0)
        alg_40_katpo = 20.0 * (alg_30_topo / alg_38_bottomo)
        alg_42_I_K_ATP = gkATP_bar * alg_40_katpo * (v - vK)
        d_v = torch.zeros_like(y[:, 0]) + (-(alg_17_I_K + alg_22_I_Ca + alg_24_I_K_Ca + alg_42_I_K_ATP) / cm)
        alg_5_F6P = 0.3 * G6P
        alg_50_AMP = ADP ** 2.0 / alg_14_ATP
        alg_59_weight15 = alg_50_AMP * FBP * alg_5_F6P ** 2.0 / (1.0 * K1 * K2 * K3 * ffbp * famp)
        alg_60_topb = alg_59_weight15
        alg_71_weight16 = alg_50_AMP * FBP * alg_5_F6P ** 2.0 * alg_14_ATP ** 2.0 / (1.0 * K1 * K2 * K3 * K4 * ffbp * famp * fbt * fmt * fatp)
        alg_61_weight12 = alg_50_AMP * alg_5_F6P ** 2.0 * alg_14_ATP ** 2.0 / (1.0 * K1 * K3 * K4 * famp * fmt * fatp)
        alg_54_weight11 = alg_50_AMP * alg_5_F6P ** 2.0 / (1.0 * K1 * K3 * famp)
        alg_45_weight8 = FBP * alg_5_F6P ** 2.0 * alg_14_ATP ** 2.0 / (1.0 * K2 * K3 * K4 * ffbp * fbt * fatp)
        alg_12_weight7 = FBP * alg_5_F6P ** 2.0 / (1.0 * K2 * K3 * ffbp)
        alg_26_weight4 = (alg_5_F6P * alg_14_ATP) ** 2.0 / (1.0 * fatp * K3 * K4)
        alg_10_weight3 = alg_5_F6P ** 2.0 / (1.0 * K3)
        alg_11_topa3 = topa2 + alg_10_weight3
        alg_29_topa4 = alg_11_topa3 + alg_26_weight4
        alg_32_topa5 = alg_29_topa4
        alg_35_topa6 = alg_32_topa5
        alg_43_topa7 = alg_35_topa6 + alg_12_weight7
        alg_46_topa8 = alg_43_topa7 + alg_45_weight8
        alg_47_topa9 = alg_46_topa8
        alg_48_topa10 = alg_47_topa9
        alg_55_topa11 = alg_48_topa10 + alg_54_weight11
        alg_62_topa12 = alg_55_topa11 + alg_61_weight12
        alg_63_topa13 = alg_62_topa12
        alg_67_topa14 = alg_63_topa13
        alg_68_topa15 = alg_67_topa14
        alg_72_topa16 = alg_68_topa15 + alg_71_weight16
        alg_66_weight14 = alg_50_AMP * FBP * alg_14_ATP ** 2.0 / (1.0 * K1 * K2 * K4 * fbt * fmt)
        alg_53_weight13 = alg_50_AMP * FBP / (K1 * K2)
        alg_56_weight10 = alg_50_AMP * alg_14_ATP ** 2.0 / (1.0 * K1 * K4 * fmt)
        alg_51_weight9 = alg_50_AMP / K1
        alg_23_weight6 = FBP * alg_14_ATP ** 2.0 / (1.0 * K2 * K4 * fbt)
        alg_9_weight5 = FBP / K2
        alg_15_weight2 = alg_14_ATP ** 2.0 / (1.0 * K4)
        alg_18_bottom2 = bottom1 + alg_15_weight2
        alg_21_bottom3 = alg_18_bottom2 + alg_10_weight3
        alg_37_bottom4 = alg_21_bottom3 + alg_26_weight4
        alg_39_bottom5 = alg_37_bottom4 + alg_9_weight5
        alg_41_bottom6 = alg_39_bottom5 + alg_23_weight6
        alg_44_bottom7 = alg_41_bottom6 + alg_12_weight7
        alg_49_bottom8 = alg_44_bottom7 + alg_45_weight8
        alg_52_bottom9 = alg_49_bottom8 + alg_51_weight9
        alg_57_bottom10 = alg_52_bottom9 + alg_56_weight10
        alg_58_bottom11 = alg_57_bottom10 + alg_54_weight11
        alg_64_bottom12 = alg_58_bottom11 + alg_61_weight12
        alg_65_bottom13 = alg_64_bottom12 + alg_53_weight13
        alg_69_bottom14 = alg_65_bottom13 + alg_66_weight14
        alg_70_bottom15 = alg_69_bottom14 + alg_59_weight15
        alg_73_bottom16 = alg_70_bottom15 + alg_71_weight16
        alg_74_JPFK = (pfkbas * cat * alg_72_topa16 + cat * alg_60_topb) / alg_73_bottom16
        d_G6P = torch.zeros_like(y[:, 0]) + (kappa * (alg_7_Jgk - alg_74_JPFK))
        d_FBP = torch.zeros_like(y[:, 0]) + (kappa * (alg_74_JPFK - 0.5 * alg_8_JGPDH))

        return torch.stack(
            [d_Gi, d_G6P, d_FBP, d_ADP, d_Ca, d_v, d_n, d_Caer, d_I],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
