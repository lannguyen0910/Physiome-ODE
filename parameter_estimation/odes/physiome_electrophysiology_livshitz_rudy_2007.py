import torch
import torch.nn as nn
import math


class ElectrophysiologyLivshitzRudy2007ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/livshitz_rudy_2007/model.py.
    Dynamic states: 18; independent constants: 54; derived constants used by rates: 11.
    Original constant slots: 68; algebraic slots: 67.
    """
    source_file = 'electrophysiology/livshitz_rudy_2007/model.py'
    source_state_dim = 18
    source_constant_dim = 68
    algebraic_dim = 67
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 18
        self.parameter_dim = 54

        self.parameters = (
            parameters
            if parameters is not None
            else {'F': 96485, 'R': 8314, 'Temp': 310, 'K_o': 4.5, 'Na_o': 140, 'Ca_o': 1.8, 'l': 0.01, 'ra': 0.0011, 'stim_offset': 0, 'stim_period': 400, 'stim_duration': 3, 'stim_amplitude': -15, 'GNa': 16, 'gacai': 1, 'gacao': 0.341, 'kmca': 0.0006, 'pca': 0.00054, 'pna': 6.75e-07, 'ganai': 0.75, 'ganao': 0.75, 'pk': 1.93e-07, 'gaki': 0.75, 'gako': 0.75, 'GK1max': 0.75, 'gkrmax': 0.02614, 'GKsmax': 0.433, 'kmnai': 10, 'kmko': 1.5, 'ibarnak': 2.25, 'c1': 0.00025, 'c2': 0.0001, 'gammas': 0.15, 'GKpmax': 0.00552, 'ibarpca': 1.15, 'kmpca': 0.0005, 'gcab': 0.003016, 'GNab': 0.004, 'gcat': 0.05, 'prnak': 0.01833, 'K_Relss': 1, 'tau': 4.75, 'kappa': 0.125, 'qn': 9, 'kmcsqn': 0.8, 'csqnbar': 10, 'kmup': 0.00092, 'iupbar': 0.00875, 'nsrbar': 15, 'tautr': 120, 'cmdnbar': 0.05, 'trpnbar': 0.07, 'kmcmdn': 0.00238, 'kmtrpn': 0.0005, 'constant_67': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -89.4356034692784, 'H': 0.994401369032678, 'm': 0.000734780346592185, 'J': 0.996100688673679, 'Na_i': 16.612739313555, 'K_i': 139.730914103161, 'd': 3.2514786721066e-27, 'f': 0.997404948824816, 'xr': 0.000162194715543637, 'xs1': 0.0285147332973946, 'xs2': 0.0764114040188678, 'b': 0.000927461915392873, 'g': 0.952834331760863, 'Ca_JSR_T': 7.87371650296567, 'Rel': 1.06874246141923e-23, 'Ca_NSR': 2.71518235696672, 'Ca_T': 0.0257059808595638, 'Over': 1e-12}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['F', 'R', 'Temp', 'K_o', 'Na_o', 'Ca_o', 'l', 'ra', 'stim_offset', 'stim_period', 'stim_duration', 'stim_amplitude', 'GNa', 'gacai', 'gacao', 'kmca', 'pca', 'pna', 'ganai', 'ganao', 'pk', 'gaki', 'gako', 'GK1max', 'gkrmax', 'GKsmax', 'kmnai', 'kmko', 'ibarnak', 'c1', 'c2', 'gammas', 'GKpmax', 'ibarpca', 'kmpca', 'gcab', 'GNab', 'gcat', 'prnak', 'K_Relss', 'tau', 'kappa', 'qn', 'kmcsqn', 'csqnbar', 'kmup', 'iupbar', 'nsrbar', 'tautr', 'cmdnbar', 'trpnbar', 'kmcmdn', 'kmtrpn', 'constant_67']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'H', 'm', 'J', 'Na_i', 'K_i', 'd', 'f', 'xr', 'xs1', 'xs2', 'b', 'g', 'Ca_JSR_T', 'Rel', 'Ca_NSR', 'Ca_T', 'Over']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (18)
        V = y[:, 0]
        H = y[:, 1]
        m = y[:, 2]
        J = y[:, 3]
        Na_i = y[:, 4]
        K_i = y[:, 5]
        d = y[:, 6]
        f = y[:, 7]
        xr = y[:, 8]
        xs1 = y[:, 9]
        xs2 = y[:, 10]
        b = y[:, 11]
        g = y[:, 12]
        Ca_JSR_T = y[:, 13]
        Rel = y[:, 14]
        Ca_NSR = y[:, 15]
        Ca_T = y[:, 16]
        Over = y[:, 17]

        # Independent parameters (source constant order)
        F = theta[:, 0]  # constants[0]
        R = theta[:, 1]  # constants[1]
        Temp = theta[:, 2]  # constants[2]
        K_o = theta[:, 3]  # constants[3]
        Na_o = theta[:, 4]  # constants[4]
        Ca_o = theta[:, 5]  # constants[5]
        l = theta[:, 6]  # constants[6]
        ra = theta[:, 7]  # constants[7]
        stim_offset = theta[:, 8]  # constants[8]
        stim_period = theta[:, 9]  # constants[9]
        stim_duration = theta[:, 10]  # constants[10]
        stim_amplitude = theta[:, 11]  # constants[11]
        GNa = theta[:, 12]  # constants[12]
        gacai = theta[:, 13]  # constants[13]
        gacao = theta[:, 14]  # constants[14]
        kmca = theta[:, 15]  # constants[15]
        pca = theta[:, 16]  # constants[16]
        pna = theta[:, 17]  # constants[17]
        ganai = theta[:, 18]  # constants[18]
        ganao = theta[:, 19]  # constants[19]
        pk = theta[:, 20]  # constants[20]
        gaki = theta[:, 21]  # constants[21]
        gako = theta[:, 22]  # constants[22]
        GK1max = theta[:, 23]  # constants[23]
        gkrmax = theta[:, 24]  # constants[24]
        GKsmax = theta[:, 25]  # constants[25]
        kmnai = theta[:, 26]  # constants[26]
        kmko = theta[:, 27]  # constants[27]
        ibarnak = theta[:, 28]  # constants[28]
        c1 = theta[:, 29]  # constants[29]
        c2 = theta[:, 30]  # constants[30]
        gammas = theta[:, 31]  # constants[31]
        GKpmax = theta[:, 32]  # constants[32]
        ibarpca = theta[:, 33]  # constants[33]
        kmpca = theta[:, 34]  # constants[34]
        gcab = theta[:, 35]  # constants[35]
        GNab = theta[:, 36]  # constants[36]
        gcat = theta[:, 37]  # constants[37]
        prnak = theta[:, 38]  # constants[38]
        K_Relss = theta[:, 39]  # constants[39]
        tau = theta[:, 40]  # constants[40]
        kappa = theta[:, 41]  # constants[41]
        qn = theta[:, 42]  # constants[42]
        kmcsqn = theta[:, 43]  # constants[43]
        csqnbar = theta[:, 44]  # constants[44]
        kmup = theta[:, 45]  # constants[45]
        iupbar = theta[:, 46]  # constants[46]
        nsrbar = theta[:, 47]  # constants[47]
        tautr = theta[:, 48]  # constants[48]
        cmdnbar = theta[:, 49]  # constants[49]
        trpnbar = theta[:, 50]  # constants[50]
        kmcmdn = theta[:, 51]  # constants[51]
        kmtrpn = theta[:, 52]  # constants[52]
        constant_67 = theta[:, 53]  # constants[67]

        # Derived constants from the original initConsts()
        FonRT = F / Temp / R  # constants[53]
        GK1 = GK1max * (K_o / 5.4) ** (1.0 / 2)  # constants[54]
        sigma = (torch.exp(Na_o / 67.3) - 1.0) / 7.0  # constants[55]
        vcell = 1000.0 * math.pi * ra * ra * l  # constants[56]
        ageo = 2.0 * math.pi * ra * ra + 2.0 * math.pi * ra * l  # constants[57]
        Acap = ageo * 2.0  # constants[61]
        vmyo = vcell * 0.68  # constants[62]
        vnsr = vcell * 0.0552  # constants[63]
        vjsr = vcell * 0.0048  # constants[64]
        AF = Acap / F  # constants[65]
        alpha_Rel = tau * kappa  # constants[66]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_H = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_J = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_d = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_xr = torch.zeros_like(y[:, 0])
        d_xs1 = torch.zeros_like(y[:, 0])
        d_xs2 = torch.zeros_like(y[:, 0])
        d_b = torch.zeros_like(y[:, 0])
        d_g = torch.zeros_like(y[:, 0])
        d_Ca_JSR_T = torch.zeros_like(y[:, 0])
        d_Rel = torch.zeros_like(y[:, 0])
        d_Ca_NSR = torch.zeros_like(y[:, 0])
        d_Ca_T = torch.zeros_like(y[:, 0])
        d_Over = torch.zeros_like(y[:, 0])
        alg_0_past = torch.zeros_like(y[:, 0])
        alg_1_a = torch.zeros_like(y[:, 0])
        alg_2_am = torch.zeros_like(y[:, 0])
        alg_3_dss0 = torch.zeros_like(y[:, 0])
        alg_4_fss = torch.zeros_like(y[:, 0])
        alg_5_xrss = torch.zeros_like(y[:, 0])
        alg_6_xss = torch.zeros_like(y[:, 0])
        alg_7_bss = torch.zeros_like(y[:, 0])
        alg_8_gss = torch.zeros_like(y[:, 0])
        alg_9_i_Stim = torch.zeros_like(y[:, 0])
        alg_10_aH = torch.zeros_like(y[:, 0])
        alg_11_aj = torch.zeros_like(y[:, 0])
        alg_12_bm = torch.zeros_like(y[:, 0])
        alg_13_taud = torch.zeros_like(y[:, 0])
        alg_14_tauf = torch.zeros_like(y[:, 0])
        alg_15_tauxr = torch.zeros_like(y[:, 0])
        alg_16_tauxs = torch.zeros_like(y[:, 0])
        alg_17_taub = torch.zeros_like(y[:, 0])
        alg_18_aa = torch.zeros_like(y[:, 0])
        alg_19_ibarna = torch.zeros_like(y[:, 0])
        alg_20_bH = torch.zeros_like(y[:, 0])
        alg_21_bj = torch.zeros_like(y[:, 0])
        alg_22_dss1 = torch.zeros_like(y[:, 0])
        alg_23_taug = torch.zeros_like(y[:, 0])
        alg_24_ibark = torch.zeros_like(y[:, 0])
        alg_25_dss = torch.zeros_like(y[:, 0])
        alg_26_r = torch.zeros_like(y[:, 0])
        alg_27_fnak = torch.zeros_like(y[:, 0])
        alg_28_inak = torch.zeros_like(y[:, 0])
        alg_29_ENa = torch.zeros_like(y[:, 0])
        alg_30_ina = torch.zeros_like(y[:, 0])
        alg_31_inab = torch.zeros_like(y[:, 0])
        alg_32_EK = torch.zeros_like(y[:, 0])
        alg_33_ak1 = torch.zeros_like(y[:, 0])
        alg_34_bk1 = torch.zeros_like(y[:, 0])
        alg_35_gK1 = torch.zeros_like(y[:, 0])
        alg_36_IK1 = torch.zeros_like(y[:, 0])
        alg_37_ikr = torch.zeros_like(y[:, 0])
        alg_38_ikp = torch.zeros_like(y[:, 0])
        alg_39_EKs = torch.zeros_like(y[:, 0])
        alg_40_bbb = torch.zeros_like(y[:, 0])
        alg_41_c = torch.zeros_like(y[:, 0])
        alg_42_Ca_JSR_free = torch.zeros_like(y[:, 0])
        alg_43_tau_Rel = torch.zeros_like(y[:, 0])
        alg_44_ileak = torch.zeros_like(y[:, 0])
        alg_45_itr = torch.zeros_like(y[:, 0])
        alg_46_bmyo = torch.zeros_like(y[:, 0])
        alg_47_cmyo = torch.zeros_like(y[:, 0])
        alg_48_dmyo = torch.zeros_like(y[:, 0])
        alg_49_Ca_i = torch.zeros_like(y[:, 0])
        alg_50_ibarca = torch.zeros_like(y[:, 0])
        alg_51_fca = torch.zeros_like(y[:, 0])
        alg_52_ilca = torch.zeros_like(y[:, 0])
        alg_53_ilcana = torch.zeros_like(y[:, 0])
        alg_54_Rel_ss = torch.zeros_like(y[:, 0])
        alg_55_ilcak = torch.zeros_like(y[:, 0])
        alg_56_gks = torch.zeros_like(y[:, 0])
        alg_57_iks = torch.zeros_like(y[:, 0])
        alg_58_kiont = torch.zeros_like(y[:, 0])
        alg_59_inaca = torch.zeros_like(y[:, 0])
        alg_60_naiont = torch.zeros_like(y[:, 0])
        alg_61_ipca = torch.zeros_like(y[:, 0])
        alg_62_icab = torch.zeros_like(y[:, 0])
        alg_63_ECa = torch.zeros_like(y[:, 0])
        alg_64_icat = torch.zeros_like(y[:, 0])
        alg_65_caiont = torch.zeros_like(y[:, 0])
        alg_66_iup = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Over = torch.zeros_like(y[:, 0]) + (constant_67)
        alg_2_am = 0.32 * 1.0 * (V + 47.13) / (1.0 - torch.exp(-0.1 * (V + 47.13)))
        alg_12_bm = 0.08 * torch.exp(-V / 11.0)
        d_m = torch.zeros_like(y[:, 0]) + (alg_2_am * (1.0 - m) - alg_12_bm * m)
        alg_4_fss = 1.0 / (1.0 + torch.exp((V + 32.0) / 8.0)) + 0.6 / (1.0 + torch.exp((50.0 - V) / 20.0))
        alg_14_tauf = 1.0 / (0.0197 * torch.exp(-(0.0337 * (V + 10.0)) ** 2.0) + 0.02)
        d_f = torch.zeros_like(y[:, 0]) + ((alg_4_fss - f) / alg_14_tauf)
        alg_5_xrss = 1.0 / (1.0 + torch.exp(-(V + 10.085) / 4.25))
        alg_15_tauxr = 1.0 / (0.00138 * (V + 14.2) / (1.0 - torch.exp(-0.123 * (V + 14.2))) + 0.00061 * (V + 38.9) / (torch.exp(0.145 * (V + 38.9)) - 1.0))
        d_xr = torch.zeros_like(y[:, 0]) + ((alg_5_xrss - xr) / alg_15_tauxr)
        alg_6_xss = 1.0 / (1.0 + torch.exp(-(V - 1.5) / 16.7))
        alg_16_tauxs = 1.0 / (7.19e-05 * (V + 30.0) / (1.0 - torch.exp(-0.148 * (V + 30.0))) + 0.000131 * (V + 30.0) / (torch.exp(0.0687 * (V + 30.0)) - 1.0))
        d_xs1 = torch.zeros_like(y[:, 0]) + ((alg_6_xss - xs1) / alg_16_tauxs)
        d_xs2 = torch.zeros_like(y[:, 0]) + ((alg_6_xss - xs2) / alg_16_tauxs / 4.0)
        alg_7_bss = 1.0 / (1.0 + torch.exp(-(V + 14.0) / 10.8))
        alg_17_taub = 3.7 + 6.1 / (1.0 + torch.exp((V + 25.0) / 4.5))
        d_b = torch.zeros_like(y[:, 0]) + ((alg_7_bss - b) / alg_17_taub)
        alg_1_a = 1.0 - 1.0 / (1.0 + torch.exp(-(V + 40.0) / 0.024))
        alg_10_aH = alg_1_a * 0.135 * torch.exp((80.0 + V) / -6.8)
        alg_20_bH = (1.0 - alg_1_a) / (0.13 * (1.0 + torch.exp((V + 10.66) / -11.1))) + alg_1_a * (3.56 * torch.exp(0.079 * V) + 3.1 * 100000.0 * torch.exp(0.35 * V))
        d_H = torch.zeros_like(y[:, 0]) + (alg_10_aH * (1.0 - H) - alg_20_bH * H)
        alg_11_aj = alg_1_a * (-127140.0 * torch.exp(0.2444 * V) - 3.474e-05 * torch.exp(-0.04391 * V)) * 1.0 * (V + 37.78) / (1.0 + torch.exp(0.311 * (V + 79.23)))
        alg_21_bj = (1.0 - alg_1_a) * 0.3 * torch.exp(-2.535e-07 * V) / (1.0 + torch.exp(-0.1 * (V + 32.0))) + alg_1_a * 0.1212 * torch.exp(-0.01052 * V) / (1.0 + torch.exp(-0.1378 * (V + 40.14)))
        d_J = torch.zeros_like(y[:, 0]) + (alg_11_aj * (1.0 - J) - alg_21_bj * J)
        alg_8_gss = 1.0 / (1.0 + torch.exp((V + 60.0) / 5.6))
        alg_18_aa = 1.0 - 1.0 / (1.0 + torch.exp(-V / 0.0024))
        alg_23_taug = alg_18_aa * 1.0 * (-0.875 * V + 12.0) + 12.0 * (1.0 - alg_18_aa)
        d_g = torch.zeros_like(y[:, 0]) + ((alg_8_gss - g) / alg_23_taug)
        alg_3_dss0 = 1.0 / (1.0 + torch.exp(-(V + 10.0) / 6.24))
        alg_13_taud = alg_3_dss0 * 1.0 * (1.0 - torch.exp(-(V + 10.0) / 6.24)) / (0.035 * (V + 10.0))
        alg_22_dss1 = 1.0 / (1.0 + torch.exp(-(V + 60.0) / 0.024))
        alg_25_dss = alg_3_dss0 * alg_22_dss1
        d_d = torch.zeros_like(y[:, 0]) + ((alg_25_dss - d) / alg_13_taud)
        alg_40_bbb = csqnbar + kmcsqn - Ca_JSR_T
        alg_41_c = Ca_JSR_T * kmcsqn
        alg_42_Ca_JSR_free = -alg_40_bbb / 2.0 + (alg_40_bbb ** 2.0 + 4.0 * alg_41_c) ** (1.0 / 2) / 2.0
        alg_45_itr = (Ca_NSR - alg_42_Ca_JSR_free) / tautr
        d_Ca_JSR_T = torch.zeros_like(y[:, 0]) + (alg_45_itr - Rel)
        alg_46_bmyo = cmdnbar + trpnbar - Ca_T + kmtrpn + kmcmdn
        alg_47_cmyo = kmcmdn * kmtrpn - Ca_T * (kmtrpn + kmcmdn) + trpnbar * kmcmdn + cmdnbar * kmtrpn
        alg_48_dmyo = -kmtrpn * kmcmdn * Ca_T
        alg_49_Ca_i = 2.0 * (alg_46_bmyo * alg_46_bmyo - 3.0 * alg_47_cmyo) ** (1.0 / 2) / 3.0 * torch.cos(torch.acos((9.0 * alg_46_bmyo * alg_47_cmyo - 2.0 * alg_46_bmyo * alg_46_bmyo * alg_46_bmyo - 27.0 * alg_48_dmyo) / (2.0 * (alg_46_bmyo * alg_46_bmyo - 3.0 * alg_47_cmyo) ** 1.5)) / 3.0) - alg_46_bmyo / 3.0
        alg_50_ibarca = pca * 4.0 * V * F * FonRT * (gacai * alg_49_Ca_i * torch.exp(2.0 * V * FonRT) - gacao * Ca_o) / (torch.exp(2.0 * V * FonRT) - 1.0)
        alg_51_fca = 1.0 / (1.0 + alg_49_Ca_i / kmca)
        alg_52_ilca = d * f * alg_51_fca * alg_50_ibarca
        alg_54_Rel_ss = alg_52_ilca * alpha_Rel / (1.0 + (K_Relss / alg_42_Ca_JSR_free) ** qn)
        alg_43_tau_Rel = tau / (1.0 + 0.0123 / alg_42_Ca_JSR_free)
        d_Rel = torch.zeros_like(y[:, 0]) + (-(alg_54_Rel_ss + Rel) / alg_43_tau_Rel)
        alg_27_fnak = 1.0 / (1.0 + 0.1245 * torch.exp(-0.1 * V * FonRT) + 0.0365 * sigma * torch.exp(-V * FonRT))
        alg_28_inak = ibarnak * alg_27_fnak / (1.0 + (kmnai / Na_i) ** 2.0) / (1.0 + kmko / K_o)
        alg_32_EK = torch.log(K_o / K_i) / FonRT
        alg_26_r = 1.0 / (1.0 + torch.exp((V + 9.0) / 22.4))
        alg_37_ikr = gkrmax * (K_o / 5.4) ** (1.0 / 2) * xr * alg_26_r * (V - alg_32_EK)
        alg_39_EKs = torch.log((K_o + prnak * Na_o) / (K_i + prnak * Na_i)) / FonRT
        alg_56_gks = GKsmax * (1.0 + 0.6 / (1.0 + (3.8e-05 / alg_49_Ca_i) ** 1.4))
        alg_57_iks = alg_56_gks * xs1 * xs2 * (V - alg_39_EKs)
        alg_33_ak1 = 1.02 / (1.0 + torch.exp(0.2385 * (V - alg_32_EK - 59.215)))
        alg_34_bk1 = (0.49124 * torch.exp(0.08032 * (V - alg_32_EK + 5.476)) + 1.0 * torch.exp(0.06175 * (V - alg_32_EK - 594.31))) / (1.0 + torch.exp(-0.5143 * (V - alg_32_EK + 4.753)))
        alg_35_gK1 = GK1 * alg_33_ak1 / (alg_33_ak1 + alg_34_bk1)
        alg_36_IK1 = alg_35_gK1 * (V - alg_32_EK)
        alg_38_ikp = GKpmax * (V - alg_32_EK) / (1.0 + torch.exp((7.488 - V) / 5.98))
        alg_24_ibark = pk * V * F * FonRT * (gaki * K_i * torch.exp(V * FonRT) - gako * K_o) / (torch.exp(V * FonRT) - 1.0)
        alg_55_ilcak = d * f * alg_51_fca * alg_24_ibark
        alg_0_past = torch.floor(t / stim_period) * stim_period
        alg_9_i_Stim = torch.where((t - alg_0_past >= stim_offset) & (t - alg_0_past <= stim_offset + stim_duration), stim_amplitude, 0.0)
        alg_58_kiont = alg_37_ikr + alg_57_iks + alg_36_IK1 + alg_38_ikp + alg_55_ilcak - 2.0 * alg_28_inak + alg_9_i_Stim
        d_K_i = torch.zeros_like(y[:, 0]) + (-alg_58_kiont * AF / vmyo)
        alg_59_inaca = c1 * torch.exp((gammas - 1.0) * V * FonRT) * (torch.exp(V * FonRT) * Na_i ** 3.0 * Ca_o - Na_o ** 3.0 * alg_49_Ca_i) / (1.0 + c2 * torch.exp((gammas - 1.0) * V * FonRT) * (torch.exp(V * FonRT) * Na_i ** 3.0 * Ca_o + Na_o ** 3.0 * alg_49_Ca_i))
        alg_29_ENa = torch.log(Na_o / Na_i) / FonRT
        alg_30_ina = GNa * m * m * m * H * J * (V - alg_29_ENa)
        alg_31_inab = GNab * (V - alg_29_ENa)
        alg_19_ibarna = pna * V * F * FonRT * (ganai * Na_i * torch.exp(V * FonRT) - ganao * Na_o) / (torch.exp(V * FonRT) - 1.0)
        alg_53_ilcana = d * f * alg_51_fca * alg_19_ibarna
        alg_60_naiont = alg_30_ina + alg_31_inab + 3.0 * alg_59_inaca + alg_53_ilcana + 3.0 * alg_28_inak
        d_Na_i = torch.zeros_like(y[:, 0]) + (-alg_60_naiont * AF / vmyo)
        alg_62_icab = gcab * (V - torch.log(Ca_o / alg_49_Ca_i) / (2.0 * FonRT))
        alg_61_ipca = ibarpca * alg_49_Ca_i / (kmpca + alg_49_Ca_i)
        alg_63_ECa = torch.log(Ca_o / alg_49_Ca_i) / 2.0 / FonRT
        alg_64_icat = gcat * b * b * g * (V - alg_63_ECa)
        alg_65_caiont = alg_52_ilca + alg_62_icab + alg_61_ipca - 2.0 * alg_59_inaca + alg_64_icat
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_60_naiont + alg_58_kiont + alg_65_caiont))
        alg_66_iup = iupbar * alg_49_Ca_i / (alg_49_Ca_i + kmup)
        alg_44_ileak = iupbar * Ca_NSR / nsrbar
        d_Ca_T = torch.zeros_like(y[:, 0]) + (-alg_65_caiont * AF / (vmyo * 2.0) + (alg_44_ileak - alg_66_iup) * vnsr / vmyo + Rel * vjsr / vmyo)
        d_Ca_NSR = torch.zeros_like(y[:, 0]) + (alg_66_iup - alg_45_itr * vjsr / vnsr - alg_44_ileak)

        return torch.stack(
            [d_V, d_H, d_m, d_J, d_Na_i, d_K_i, d_d, d_f, d_xr, d_xs1, d_xs2, d_b, d_g, d_Ca_JSR_T, d_Rel, d_Ca_NSR, d_Ca_T, d_Over],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
