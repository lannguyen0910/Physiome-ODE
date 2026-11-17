import torch
import torch.nn as nn
import math


class ElectrophysiologyMahajanShiferawSatoBaherOlceseXieYangChenRestrepoKarmaGarfinkelQuWeiss2008ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/mahajan_shiferaw_sato_baher_olcese_xie_yang_chen_restrepo_karma_garfinkel_qu_weiss_2008/model.py.
    Dynamic states: 26; independent constants: 75; derived constants used by rates: 5.
    Original constant slots: 80; algebraic slots: 108.
    """
    source_file = 'electrophysiology/mahajan_shiferaw_sato_baher_olcese_xie_yang_chen_restrepo_karma_garfinkel_qu_weiss_2008/model.py'
    source_state_dim = 26
    source_constant_dim = 80
    algebraic_dim = 108
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 26
        self.parameter_dim = 75

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8.314472, 'T': 308, 'F': 96.4853415, 'K_o': 5.4, 'Ca_o': 1.8, 'Na_o': 136, 'wca': 8, 'stim_offset': 0, 'stim_period': 400, 'stim_duration': 3, 'stim_amplitude': -15, 'gna': 12, 'gca': 182, 'pca': 0.00054, 'vth': 0, 's6': 8, 'vx': -40, 'sx': 3, 'vy': -40, 'sy': 4, 'vyr': -40, 'syr': 11.32, 'cat': 3, 'cpt': 6.09365, 'k2': 0.000103615, 'k1t': 0.00413, 'k2t': 0.00224, 'r1': 0.3, 'r2': 3, 's1t': 0.00195, 'tca': 78.0329, 'taupo': 1, 'tau3': 3, 'gkix': 0.3, 'gkr': 0.0125, 'gks': 0.1386, 'gtos': 0.04, 'gtof': 0.11, 'gNaK': 1.5, 'xkmko': 1.5, 'xkmnai': 12, 'gNaCa': 0.84, 'xkdna': 0.3, 'xmcao': 1.3, 'xmnao': 87.5, 'xmnai': 12.3, 'xmcai': 0.0036, 'cstar': 90, 'gryr': 2.58079, 'gbarsr': 26841.8, 'gdyad': 9000, 'ax': 0.3576, 'ay': 0.05, 'av': 11.3, 'taua': 100, 'taur': 30, 'cup': 0.5, 'kj': 50, 'vup': 0.4, 'gleak': 2.069e-05, 'bcal': 24, 'xkcal': 7, 'srmax': 47, 'srkd': 0.6, 'bmem': 15, 'kmem': 0.3, 'bsar': 42, 'ksar': 13, 'xkon': 0.0327, 'xkoff': 0.0196, 'btrop': 70, 'taud': 4, 'taups': 0.5, 'K_i': 140, 'prNaK': 0.01833}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -87.169816169406, 'xm': 0.001075453357, 'xh': 0.990691306716, 'xj': 0.993888937283, 'Ca_dyad': 1.716573130685, 'c1': 1.8211252e-05, 'c2': 0.979322592773, 'xi1ca': 0.001208153482, 'xi1ba': 3.3616596e-05, 'xi2ca': 0.004173008466, 'xi2ba': 0.015242594688, 'xr': 0.007074239331, 'Ca_i': 0.256752008084, 'xs1': 0.048267587131, 'xs2': 0.105468807033, 'xtos': 0.00364776906, 'ytos': 0.174403618112, 'xtof': 0.003643592594, 'ytof': 0.993331326442, 'Na_i': 11.441712311614, 'Ca_submem': 0.226941113355, 'Ca_NSR': 104.450004990523, 'Ca_JSR': 97.505463697266, 'xir': 0.006679257264, 'tropi': 22.171689894953, 'trops': 19.864701949854}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'K_o', 'Ca_o', 'Na_o', 'wca', 'stim_offset', 'stim_period', 'stim_duration', 'stim_amplitude', 'gna', 'gca', 'pca', 'vth', 's6', 'vx', 'sx', 'vy', 'sy', 'vyr', 'syr', 'cat', 'cpt', 'k2', 'k1t', 'k2t', 'r1', 'r2', 's1t', 'tca', 'taupo', 'tau3', 'gkix', 'gkr', 'gks', 'gtos', 'gtof', 'gNaK', 'xkmko', 'xkmnai', 'gNaCa', 'xkdna', 'xmcao', 'xmnao', 'xmnai', 'xmcai', 'cstar', 'gryr', 'gbarsr', 'gdyad', 'ax', 'ay', 'av', 'taua', 'taur', 'cup', 'kj', 'vup', 'gleak', 'bcal', 'xkcal', 'srmax', 'srkd', 'bmem', 'kmem', 'bsar', 'ksar', 'xkon', 'xkoff', 'btrop', 'taud', 'taups', 'K_i', 'prNaK']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'xm', 'xh', 'xj', 'Ca_dyad', 'c1', 'c2', 'xi1ca', 'xi1ba', 'xi2ca', 'xi2ba', 'xr', 'Ca_i', 'xs1', 'xs2', 'xtos', 'ytos', 'xtof', 'ytof', 'Na_i', 'Ca_submem', 'Ca_NSR', 'Ca_JSR', 'xir', 'tropi', 'trops']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (26)
        V = y[:, 0]
        xm = y[:, 1]
        xh = y[:, 2]
        xj = y[:, 3]
        Ca_dyad = y[:, 4]
        c1 = y[:, 5]
        c2 = y[:, 6]
        xi1ca = y[:, 7]
        xi1ba = y[:, 8]
        xi2ca = y[:, 9]
        xi2ba = y[:, 10]
        xr = y[:, 11]
        Ca_i = y[:, 12]
        xs1 = y[:, 13]
        xs2 = y[:, 14]
        xtos = y[:, 15]
        ytos = y[:, 16]
        xtof = y[:, 17]
        ytof = y[:, 18]
        Na_i = y[:, 19]
        Ca_submem = y[:, 20]
        Ca_NSR = y[:, 21]
        Ca_JSR = y[:, 22]
        xir = y[:, 23]
        tropi = y[:, 24]
        trops = y[:, 25]

        # Independent parameters (source constant order)
        R = theta[:, 0]  # constants[0]
        T = theta[:, 1]  # constants[1]
        F = theta[:, 2]  # constants[2]
        K_o = theta[:, 3]  # constants[3]
        Ca_o = theta[:, 4]  # constants[4]
        Na_o = theta[:, 5]  # constants[5]
        wca = theta[:, 6]  # constants[6]
        stim_offset = theta[:, 7]  # constants[7]
        stim_period = theta[:, 8]  # constants[8]
        stim_duration = theta[:, 9]  # constants[9]
        stim_amplitude = theta[:, 10]  # constants[10]
        gna = theta[:, 11]  # constants[11]
        gca = theta[:, 12]  # constants[12]
        pca = theta[:, 13]  # constants[13]
        vth = theta[:, 14]  # constants[14]
        s6 = theta[:, 15]  # constants[15]
        vx = theta[:, 16]  # constants[16]
        sx = theta[:, 17]  # constants[17]
        vy = theta[:, 18]  # constants[18]
        sy = theta[:, 19]  # constants[19]
        vyr = theta[:, 20]  # constants[20]
        syr = theta[:, 21]  # constants[21]
        cat = theta[:, 22]  # constants[22]
        cpt = theta[:, 23]  # constants[23]
        k2 = theta[:, 24]  # constants[24]
        k1t = theta[:, 25]  # constants[25]
        k2t = theta[:, 26]  # constants[26]
        r1 = theta[:, 27]  # constants[27]
        r2 = theta[:, 28]  # constants[28]
        s1t = theta[:, 29]  # constants[29]
        tca = theta[:, 30]  # constants[30]
        taupo = theta[:, 31]  # constants[31]
        tau3 = theta[:, 32]  # constants[32]
        gkix = theta[:, 33]  # constants[33]
        gkr = theta[:, 34]  # constants[34]
        gks = theta[:, 35]  # constants[35]
        gtos = theta[:, 36]  # constants[36]
        gtof = theta[:, 37]  # constants[37]
        gNaK = theta[:, 38]  # constants[38]
        xkmko = theta[:, 39]  # constants[39]
        xkmnai = theta[:, 40]  # constants[40]
        gNaCa = theta[:, 41]  # constants[41]
        xkdna = theta[:, 42]  # constants[42]
        xmcao = theta[:, 43]  # constants[43]
        xmnao = theta[:, 44]  # constants[44]
        xmnai = theta[:, 45]  # constants[45]
        xmcai = theta[:, 46]  # constants[46]
        cstar = theta[:, 47]  # constants[47]
        gryr = theta[:, 48]  # constants[48]
        gbarsr = theta[:, 49]  # constants[49]
        gdyad = theta[:, 50]  # constants[50]
        ax = theta[:, 51]  # constants[51]
        ay = theta[:, 52]  # constants[52]
        av = theta[:, 53]  # constants[53]
        taua = theta[:, 54]  # constants[54]
        taur = theta[:, 55]  # constants[55]
        cup = theta[:, 56]  # constants[56]
        kj = theta[:, 57]  # constants[57]
        vup = theta[:, 58]  # constants[58]
        gleak = theta[:, 59]  # constants[59]
        bcal = theta[:, 60]  # constants[60]
        xkcal = theta[:, 61]  # constants[61]
        srmax = theta[:, 62]  # constants[62]
        srkd = theta[:, 63]  # constants[63]
        bmem = theta[:, 64]  # constants[64]
        kmem = theta[:, 65]  # constants[65]
        bsar = theta[:, 66]  # constants[66]
        ksar = theta[:, 67]  # constants[67]
        xkon = theta[:, 68]  # constants[68]
        xkoff = theta[:, 69]  # constants[69]
        btrop = theta[:, 70]  # constants[70]
        taud = theta[:, 71]  # constants[71]
        taups = theta[:, 72]  # constants[72]
        K_i = theta[:, 73]  # constants[73]
        prNaK = theta[:, 74]  # constants[74]

        # Derived constants from the original initConsts()
        FonRT = F / (R * T)  # constants[75]
        s2t = s1t * r1 / r2 * k2t / k1t  # constants[76]
        sigma = (torch.exp(Na_o / 67.3) - 1.0) / 7.0  # constants[77]
        bv = (1.0 - av) * cstar - 50.0  # constants[78]
        ek = 1.0 / FonRT * torch.log(K_o / K_i)  # constants[79]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_xm = torch.zeros_like(y[:, 0])
        d_xh = torch.zeros_like(y[:, 0])
        d_xj = torch.zeros_like(y[:, 0])
        d_Ca_dyad = torch.zeros_like(y[:, 0])
        d_c1 = torch.zeros_like(y[:, 0])
        d_c2 = torch.zeros_like(y[:, 0])
        d_xi1ca = torch.zeros_like(y[:, 0])
        d_xi1ba = torch.zeros_like(y[:, 0])
        d_xi2ca = torch.zeros_like(y[:, 0])
        d_xi2ba = torch.zeros_like(y[:, 0])
        d_xr = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_xs1 = torch.zeros_like(y[:, 0])
        d_xs2 = torch.zeros_like(y[:, 0])
        d_xtos = torch.zeros_like(y[:, 0])
        d_ytos = torch.zeros_like(y[:, 0])
        d_xtof = torch.zeros_like(y[:, 0])
        d_ytof = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_Ca_submem = torch.zeros_like(y[:, 0])
        d_Ca_NSR = torch.zeros_like(y[:, 0])
        d_Ca_JSR = torch.zeros_like(y[:, 0])
        d_xir = torch.zeros_like(y[:, 0])
        d_tropi = torch.zeros_like(y[:, 0])
        d_trops = torch.zeros_like(y[:, 0])
        alg_0_am = torch.zeros_like(y[:, 0])
        alg_1_ah = torch.zeros_like(y[:, 0])
        alg_2_aj = torch.zeros_like(y[:, 0])
        alg_3_past = torch.zeros_like(y[:, 0])
        alg_4_xkrv1 = torch.zeros_like(y[:, 0])
        alg_5_xs1ss = torch.zeros_like(y[:, 0])
        alg_6_rt1 = torch.zeros_like(y[:, 0])
        alg_7_bm = torch.zeros_like(y[:, 0])
        alg_8_bh = torch.zeros_like(y[:, 0])
        alg_9_bj = torch.zeros_like(y[:, 0])
        alg_10_i_Stim = torch.zeros_like(y[:, 0])
        alg_11_xkrv2 = torch.zeros_like(y[:, 0])
        alg_12_xs2ss = torch.zeros_like(y[:, 0])
        alg_13_rt4 = torch.zeros_like(y[:, 0])
        alg_14_za = torch.zeros_like(y[:, 0])
        alg_15_taukr = torch.zeros_like(y[:, 0])
        alg_16_tauxs1 = torch.zeros_like(y[:, 0])
        alg_17_xtos_inf = torch.zeros_like(y[:, 0])
        alg_18_poinf = torch.zeros_like(y[:, 0])
        alg_19_xkrinf = torch.zeros_like(y[:, 0])
        alg_20_tauxs2 = torch.zeros_like(y[:, 0])
        alg_21_xtof_inf = torch.zeros_like(y[:, 0])
        alg_22_txs = torch.zeros_like(y[:, 0])
        alg_23_alpha = torch.zeros_like(y[:, 0])
        alg_24_txf = torch.zeros_like(y[:, 0])
        alg_25_beta = torch.zeros_like(y[:, 0])
        alg_26_fca = torch.zeros_like(y[:, 0])
        alg_27_s1 = torch.zeros_like(y[:, 0])
        alg_28_k1 = torch.zeros_like(y[:, 0])
        alg_29_s2 = torch.zeros_like(y[:, 0])
        alg_30_poi = torch.zeros_like(y[:, 0])
        alg_31_k3 = torch.zeros_like(y[:, 0])
        alg_32_k3t = torch.zeros_like(y[:, 0])
        alg_33_Pr = torch.zeros_like(y[:, 0])
        alg_34_recov = torch.zeros_like(y[:, 0])
        alg_35_tau_ca = torch.zeros_like(y[:, 0])
        alg_36_tauca = torch.zeros_like(y[:, 0])
        alg_37_tauba = torch.zeros_like(y[:, 0])
        alg_38_Ps = torch.zeros_like(y[:, 0])
        alg_39_k6 = torch.zeros_like(y[:, 0])
        alg_40_k5 = torch.zeros_like(y[:, 0])
        alg_41_k6t = torch.zeros_like(y[:, 0])
        alg_42_k5t = torch.zeros_like(y[:, 0])
        alg_43_k4 = torch.zeros_like(y[:, 0])
        alg_44_k4t = torch.zeros_like(y[:, 0])
        alg_45_po = torch.zeros_like(y[:, 0])
        alg_46_aki = torch.zeros_like(y[:, 0])
        alg_47_bki = torch.zeros_like(y[:, 0])
        alg_48_xkin = torch.zeros_like(y[:, 0])
        alg_49_xik1 = torch.zeros_like(y[:, 0])
        alg_50_rg = torch.zeros_like(y[:, 0])
        alg_51_xikr = torch.zeros_like(y[:, 0])
        alg_52_gksx = torch.zeros_like(y[:, 0])
        alg_53_rt2 = torch.zeros_like(y[:, 0])
        alg_54_rs_inf = torch.zeros_like(y[:, 0])
        alg_55_rt3 = torch.zeros_like(y[:, 0])
        alg_56_xitos = torch.zeros_like(y[:, 0])
        alg_57_rt5 = torch.zeros_like(y[:, 0])
        alg_58_xitof = torch.zeros_like(y[:, 0])
        alg_59_ytos_inf = torch.zeros_like(y[:, 0])
        alg_60_xito = torch.zeros_like(y[:, 0])
        alg_61_ytof_inf = torch.zeros_like(y[:, 0])
        alg_62_tys = torch.zeros_like(y[:, 0])
        alg_63_fNaK = torch.zeros_like(y[:, 0])
        alg_64_tyf = torch.zeros_like(y[:, 0])
        alg_65_xiNaK = torch.zeros_like(y[:, 0])
        alg_66_zw4 = torch.zeros_like(y[:, 0])
        alg_67_aloss = torch.zeros_like(y[:, 0])
        alg_68_yz3 = torch.zeros_like(y[:, 0])
        alg_69_Qr0 = torch.zeros_like(y[:, 0])
        alg_70_Qr = torch.zeros_like(y[:, 0])
        alg_71_sparkV = torch.zeros_like(y[:, 0])
        alg_72_jup = torch.zeros_like(y[:, 0])
        alg_73_jleak = torch.zeros_like(y[:, 0])
        alg_74_bpxs = torch.zeros_like(y[:, 0])
        alg_75_spxs = torch.zeros_like(y[:, 0])
        alg_76_mempxs = torch.zeros_like(y[:, 0])
        alg_77_sarpxs = torch.zeros_like(y[:, 0])
        alg_78_dcsib = torch.zeros_like(y[:, 0])
        alg_79_bpxi = torch.zeros_like(y[:, 0])
        alg_80_spxi = torch.zeros_like(y[:, 0])
        alg_81_mempxi = torch.zeros_like(y[:, 0])
        alg_82_sarpxi = torch.zeros_like(y[:, 0])
        alg_83_dciib = torch.zeros_like(y[:, 0])
        alg_84_jd = torch.zeros_like(y[:, 0])
        alg_85_xbs = torch.zeros_like(y[:, 0])
        alg_86_xbi = torch.zeros_like(y[:, 0])
        alg_87_dCa_JSR = torch.zeros_like(y[:, 0])
        alg_88_csm = torch.zeros_like(y[:, 0])
        alg_89_rxa = torch.zeros_like(y[:, 0])
        alg_90_jca = torch.zeros_like(y[:, 0])
        alg_91_spark_rate = torch.zeros_like(y[:, 0])
        alg_92_xirp = torch.zeros_like(y[:, 0])
        alg_93_xica = torch.zeros_like(y[:, 0])
        alg_94_xicap = torch.zeros_like(y[:, 0])
        alg_95_zw3 = torch.zeros_like(y[:, 0])
        alg_96_xiryr = torch.zeros_like(y[:, 0])
        alg_97_yz1 = torch.zeros_like(y[:, 0])
        alg_98_yz2 = torch.zeros_like(y[:, 0])
        alg_99_yz4 = torch.zeros_like(y[:, 0])
        alg_100_zw8 = torch.zeros_like(y[:, 0])
        alg_101_jNaCa = torch.zeros_like(y[:, 0])
        alg_102_xiNaCa = torch.zeros_like(y[:, 0])
        alg_103_eks = torch.zeros_like(y[:, 0])
        alg_104_xiks = torch.zeros_like(y[:, 0])
        alg_105_ena = torch.zeros_like(y[:, 0])
        alg_106_xina = torch.zeros_like(y[:, 0])
        alg_107_Itotal = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Ca_JSR = torch.zeros_like(y[:, 0]) + ((Ca_NSR - Ca_JSR) / taua)
        alg_1_ah = torch.where(V < -40.0, 0.135 * torch.exp((80.0 + V) / -6.8), 0.0)
        alg_8_bh = torch.where(V < -40.0, 3.56 * torch.exp(0.079 * V) + 310000.0 * torch.exp(0.35 * V), 1.0 / (0.13 * (1.0 + torch.exp((V + 10.66) / -11.1))))
        d_xh = torch.zeros_like(y[:, 0]) + (alg_1_ah * (1.0 - xh) - alg_8_bh * xh)
        alg_2_aj = torch.where(V < -40.0, (-127140.0 * torch.exp(0.2444 * V) - 3.474e-05 * torch.exp(-0.04391 * V)) * 1.0 * (V + 37.78) / (1.0 + torch.exp(0.311 * (V + 79.23))), 0.0)
        alg_9_bj = torch.where(V < -40.0, 0.1212 * torch.exp(-0.01052 * V) / (1.0 + torch.exp(-0.1378 * (V + 40.14))), 0.3 * torch.exp(-2.535e-07 * V) / (1.0 + torch.exp(-0.1 * (V + 32.0))))
        d_xj = torch.zeros_like(y[:, 0]) + (alg_2_aj * (1.0 - xj) - alg_9_bj * xj)
        alg_0_am = torch.where(torch.abs(V + 47.13) > 0.001, 0.32 * 1.0 * (V + 47.13) / (1.0 - torch.exp(-0.1 * (V + 47.13))), 3.2)
        alg_7_bm = 0.08 * torch.exp(-V / 11.0)
        d_xm = torch.zeros_like(y[:, 0]) + (alg_0_am * (1.0 - xm) - alg_7_bm * xm)
        alg_5_xs1ss = 1.0 / (1.0 + torch.exp(-(V - 1.5) / 16.7))
        alg_16_tauxs1 = torch.where(torch.abs(V + 30.0) < 0.001 / 0.0687, 1.0 / (7.19e-05 / 0.148 + 0.000131 / 0.0687), 1.0 / (7.19e-05 * (V + 30.0) / (1.0 - torch.exp(-0.148 * (V + 30.0))) + 0.000131 * (V + 30.0) / (torch.exp(0.0687 * (V + 30.0)) - 1.0)))
        d_xs1 = torch.zeros_like(y[:, 0]) + ((alg_5_xs1ss - xs1) / alg_16_tauxs1)
        alg_4_xkrv1 = torch.where(torch.abs(V + 7.0) > 0.001, 0.00138 * 1.0 * (V + 7.0) / (1.0 - torch.exp(-0.123 * (V + 7.0))), 0.00138 / 0.123)
        alg_11_xkrv2 = torch.where(torch.abs(V + 10.0) > 0.001, 0.00061 * 1.0 * (V + 10.0) / (torch.exp(0.145 * (V + 10.0)) - 1.0), 0.00061 / 0.145)
        alg_15_taukr = 1.0 / (alg_4_xkrv1 + alg_11_xkrv2)
        alg_19_xkrinf = 1.0 / (1.0 + torch.exp(-(V + 50.0) / 7.5))
        d_xr = torch.zeros_like(y[:, 0]) + ((alg_19_xkrinf - xr) / alg_15_taukr)
        alg_12_xs2ss = alg_5_xs1ss
        alg_20_tauxs2 = 4.0 * alg_16_tauxs1
        d_xs2 = torch.zeros_like(y[:, 0]) + ((alg_12_xs2ss - xs2) / alg_20_tauxs2)
        alg_6_rt1 = -(V + 3.0) / 15.0
        alg_17_xtos_inf = 1.0 / (1.0 + torch.exp(alg_6_rt1))
        alg_22_txs = 9.0 / (1.0 + torch.exp(-alg_6_rt1)) + 0.5
        d_xtos = torch.zeros_like(y[:, 0]) + ((alg_17_xtos_inf - xtos) / alg_22_txs)
        alg_21_xtof_inf = alg_17_xtos_inf
        alg_13_rt4 = -V / 30.0 * V / 30.0
        alg_24_txf = 3.5 * torch.exp(alg_13_rt4) + 1.5
        d_xtof = torch.zeros_like(y[:, 0]) + ((alg_21_xtof_inf - xtof) / alg_24_txf)
        alg_18_poinf = 1.0 / (1.0 + torch.exp(-(V - vth) / s6))
        alg_23_alpha = alg_18_poinf / taupo
        alg_25_beta = (1.0 - alg_18_poinf) / taupo
        alg_26_fca = 1.0 / (1.0 + (cat / Ca_dyad) ** 3.0)
        alg_34_recov = 10.0 + 4954.0 * torch.exp(V / 15.6)
        alg_35_tau_ca = tca / (1.0 + (Ca_dyad / cpt) ** 4.0) + 0.1
        alg_33_Pr = 1.0 - 1.0 / (1.0 + torch.exp(-(V - vy) / sy))
        alg_36_tauca = (alg_34_recov - alg_35_tau_ca) * alg_33_Pr + alg_35_tau_ca
        alg_38_Ps = 1.0 / (1.0 + torch.exp(-(V - vyr) / syr))
        alg_39_k6 = alg_26_fca * alg_38_Ps / alg_36_tauca
        alg_40_k5 = (1.0 - alg_38_Ps) / alg_36_tauca
        alg_37_tauba = (alg_34_recov - 450.0) * alg_33_Pr + 450.0
        alg_41_k6t = alg_38_Ps / alg_37_tauba
        alg_42_k5t = (1.0 - alg_38_Ps) / alg_37_tauba
        d_c2 = torch.zeros_like(y[:, 0]) + (alg_25_beta * c1 + alg_40_k5 * xi2ca + alg_42_k5t * xi2ba - (alg_39_k6 + alg_41_k6t + alg_23_alpha) * c2)
        alg_30_poi = 1.0 / (1.0 + torch.exp(-(V - vx) / sx))
        alg_31_k3 = (1.0 - alg_30_poi) / tau3
        alg_28_k1 = 0.024168 * alg_26_fca
        alg_43_k4 = alg_31_k3 * alg_23_alpha / alg_25_beta * alg_28_k1 / k2 * alg_40_k5 / alg_39_k6
        d_xi2ca = torch.zeros_like(y[:, 0]) + (alg_31_k3 * xi1ca + alg_39_k6 * c2 - (alg_40_k5 + alg_43_k4) * xi2ca)
        alg_32_k3t = alg_31_k3
        alg_44_k4t = alg_32_k3t * alg_23_alpha / alg_25_beta * k1t / k2t * alg_42_k5t / alg_41_k6t
        d_xi2ba = torch.zeros_like(y[:, 0]) + (alg_32_k3t * xi1ba + alg_41_k6t * c2 - (alg_42_k5t + alg_44_k4t) * xi2ba)
        alg_45_po = 1.0 - xi1ca - xi2ca - xi1ba - xi2ba - c1 - c2
        d_c1 = torch.zeros_like(y[:, 0]) + (alg_23_alpha * c2 + k2 * xi1ca + k2t * xi1ba + r2 * alg_45_po - (alg_25_beta + r1 + k1t + alg_28_k1) * c1)
        alg_27_s1 = 0.0182688 * alg_26_fca
        alg_29_s2 = alg_27_s1 * r1 / r2 * k2 / alg_28_k1
        d_xi1ca = torch.zeros_like(y[:, 0]) + (alg_28_k1 * c1 + alg_43_k4 * xi2ca + alg_27_s1 * alg_45_po - (alg_31_k3 + k2 + alg_29_s2) * xi1ca)
        d_xi1ba = torch.zeros_like(y[:, 0]) + (k1t * c1 + alg_44_k4t * xi2ba + s1t * alg_45_po - (alg_32_k3t + k2t + s2t) * xi1ba)
        alg_53_rt2 = (V + 33.5) / 10.0
        alg_59_ytos_inf = 1.0 / (1.0 + torch.exp(alg_53_rt2))
        alg_55_rt3 = (V + 60.0) / 10.0
        alg_62_tys = 3000.0 / (1.0 + torch.exp(alg_55_rt3)) + 30.0
        d_ytos = torch.zeros_like(y[:, 0]) + ((alg_59_ytos_inf - ytos) / alg_62_tys)
        alg_61_ytof_inf = alg_59_ytos_inf
        alg_57_rt5 = (V + 33.5) / 10.0
        alg_64_tyf = 20.0 / (1.0 + torch.exp(alg_57_rt5)) + 20.0
        d_ytof = torch.zeros_like(y[:, 0]) + ((alg_61_ytof_inf - ytof) / alg_64_tyf)
        alg_72_jup = vup * Ca_i * Ca_i / (Ca_i * Ca_i + cup * cup)
        alg_73_jleak = gleak * Ca_NSR * Ca_NSR / (Ca_NSR * Ca_NSR + kj * kj) * (Ca_NSR * 16.667 - Ca_i)
        alg_79_bpxi = bcal * xkcal / ((xkcal + Ca_i) * (xkcal + Ca_i))
        alg_80_spxi = srmax * srkd / ((srkd + Ca_i) * (srkd + Ca_i))
        alg_81_mempxi = bmem * kmem / ((kmem + Ca_i) * (kmem + Ca_i))
        alg_82_sarpxi = bsar * ksar / ((ksar + Ca_i) * (ksar + Ca_i))
        alg_83_dciib = 1.0 / (1.0 + alg_79_bpxi + alg_80_spxi + alg_81_mempxi + alg_82_sarpxi)
        alg_86_xbi = xkon * Ca_i * (btrop - tropi) - xkoff * tropi
        alg_84_jd = (Ca_submem - Ca_i) / taud
        d_Ca_i = torch.zeros_like(y[:, 0]) + (alg_83_dciib * (alg_84_jd - alg_72_jup + alg_73_jleak - alg_86_xbi))
        d_tropi = torch.zeros_like(y[:, 0]) + (alg_86_xbi)
        alg_85_xbs = xkon * Ca_submem * (btrop - trops) - xkoff * trops
        d_trops = torch.zeros_like(y[:, 0]) + (alg_85_xbs)
        alg_87_dCa_JSR = -xir + alg_72_jup - alg_73_jleak
        d_Ca_NSR = torch.zeros_like(y[:, 0]) + (alg_87_dCa_JSR)
        alg_69_Qr0 = torch.where((Ca_JSR > 50.0) & (Ca_JSR < cstar), (Ca_JSR - 50.0) / 1.0, torch.where(Ca_JSR >= cstar, av * Ca_JSR + bv, 0.0))
        alg_70_Qr = Ca_NSR * alg_69_Qr0 / cstar
        alg_88_csm = Ca_submem / 1000.0
        alg_14_za = V * 2.0 * FonRT
        alg_89_rxa = torch.where(torch.abs(alg_14_za) < 0.001, 4.0 * pca * F * FonRT * (alg_88_csm * torch.exp(alg_14_za) - 0.341 * Ca_o) / (2.0 * FonRT), 4.0 * pca * V * F * FonRT * (alg_88_csm * torch.exp(alg_14_za) - 0.341 * Ca_o) / (torch.exp(alg_14_za) - 1.0))
        alg_71_sparkV = torch.exp(-ay * (V + 30.0)) / (1.0 + torch.exp(-ay * (V + 30.0)))
        alg_91_spark_rate = gryr / 1.0 * alg_45_po * torch.abs(alg_89_rxa) * alg_71_sparkV
        d_xir = torch.zeros_like(y[:, 0]) + (alg_91_spark_rate * alg_70_Qr - xir * (1.0 - taur * alg_87_dCa_JSR / Ca_NSR) / taur)
        alg_92_xirp = alg_45_po * alg_70_Qr * torch.abs(alg_89_rxa) * gbarsr / 1.0 * torch.exp(-ax * (V + 30.0)) / (1.0 + torch.exp(-ax * (V + 30.0)))
        alg_94_xicap = alg_45_po * gdyad * torch.abs(alg_89_rxa)
        alg_96_xiryr = alg_92_xirp + alg_94_xicap
        d_Ca_dyad = torch.zeros_like(y[:, 0]) + (alg_96_xiryr - (Ca_dyad - Ca_submem) / taups)
        alg_90_jca = gca * alg_45_po * alg_89_rxa
        alg_67_aloss = 1.0 / (1.0 + (xkdna / Ca_submem) ** 3.0)
        alg_95_zw3 = Na_i ** 3.0 * Ca_o * torch.exp(V * 0.35 * FonRT) - Na_o ** 3.0 * alg_88_csm * torch.exp(V * (0.35 - 1.0) * FonRT)
        alg_66_zw4 = 1.0 + 0.2 * torch.exp(V * (0.35 - 1.0) * FonRT)
        alg_97_yz1 = xmcao * Na_i ** 3.0 + xmnao ** 3.0 * alg_88_csm
        alg_98_yz2 = xmnai ** 3.0 * Ca_o * (1.0 + alg_88_csm / xmcai)
        alg_68_yz3 = xmcai * Na_o ** 3.0 * (1.0 + (Na_i / xmnai) ** 3.0)
        alg_99_yz4 = Na_i ** 3.0 * Ca_o + Na_o ** 3.0 * alg_88_csm
        alg_100_zw8 = alg_97_yz1 + alg_98_yz2 + alg_68_yz3 + alg_99_yz4
        alg_101_jNaCa = gNaCa * alg_67_aloss * alg_95_zw3 / (alg_66_zw4 * alg_100_zw8)
        alg_74_bpxs = bcal * xkcal / ((xkcal + Ca_submem) * (xkcal + Ca_submem))
        alg_75_spxs = srmax * srkd / ((srkd + Ca_submem) * (srkd + Ca_submem))
        alg_76_mempxs = bmem * kmem / ((kmem + Ca_submem) * (kmem + Ca_submem))
        alg_77_sarpxs = bsar * ksar / ((ksar + Ca_submem) * (ksar + Ca_submem))
        alg_78_dcsib = 1.0 / (1.0 + alg_74_bpxs + alg_75_spxs + alg_76_mempxs + alg_77_sarpxs)
        d_Ca_submem = torch.zeros_like(y[:, 0]) + (alg_78_dcsib * (50.0 * (xir - alg_84_jd - alg_90_jca + alg_101_jNaCa) - alg_85_xbs))
        alg_63_fNaK = 1.0 / (1.0 + 0.1245 * torch.exp(-0.1 * V * FonRT) + 0.0365 * sigma * torch.exp(-V * FonRT))
        alg_65_xiNaK = gNaK * alg_63_fNaK * Na_i / (Na_i + xkmnai) * K_o / (K_o + xkmko)
        alg_102_xiNaCa = wca * alg_101_jNaCa
        alg_105_ena = 1.0 / FonRT * torch.log(Na_o / Na_i)
        alg_106_xina = gna * xh * xj * xm * xm * xm * (V - alg_105_ena)
        d_Na_i = torch.zeros_like(y[:, 0]) + (-(alg_106_xina + 3.0 * alg_65_xiNaK + 3.0 * alg_102_xiNaCa) / (wca * 1000.0))
        alg_46_aki = 1.02 / (1.0 + torch.exp(0.2385 * (V - ek - 59.215)))
        alg_47_bki = (0.49124 * torch.exp(0.08032 * (V - ek + 5.476)) + 1.0 * torch.exp(0.06175 * (V - ek - 594.31))) / (1.0 + torch.exp(-0.5143 * (V - ek + 4.753)))
        alg_48_xkin = alg_46_aki / (alg_46_aki + alg_47_bki)
        alg_49_xik1 = gkix * (K_o / 5.4) ** (1.0 / 2) * alg_48_xkin * (V - ek)
        alg_54_rs_inf = 1.0 / (1.0 + torch.exp(alg_53_rt2))
        alg_56_xitos = gtos * xtos * (ytos + 0.5 * alg_54_rs_inf) * (V - ek)
        alg_58_xitof = gtof * xtof * ytof * (V - ek)
        alg_60_xito = alg_56_xitos + alg_58_xitof
        alg_93_xica = 2.0 * wca * alg_90_jca
        alg_50_rg = 1.0 / (1.0 + torch.exp((V + 33.0) / 22.4))
        alg_51_xikr = gkr * (K_o / 5.4) ** (1.0 / 2) * xr * alg_50_rg * (V - ek)
        alg_103_eks = 1.0 / FonRT * torch.log((K_o + prNaK * Na_o) / (K_i + prNaK * Na_i))
        alg_52_gksx = 1.0 + 0.8 / (1.0 + (0.5 / Ca_i) ** 3.0)
        alg_104_xiks = gks * alg_52_gksx * xs1 * xs2 * (V - alg_103_eks)
        alg_3_past = torch.floor(t / stim_period) * stim_period
        alg_10_i_Stim = torch.where((t - alg_3_past >= stim_offset) & (t - alg_3_past <= stim_offset + stim_duration), stim_amplitude, 0.0)
        alg_107_Itotal = -(alg_106_xina + alg_49_xik1 + alg_51_xikr + alg_104_xiks + alg_60_xito + alg_102_xiNaCa + alg_93_xica + alg_65_xiNaK + alg_10_i_Stim)
        d_V = torch.zeros_like(y[:, 0]) + (alg_107_Itotal)

        return torch.stack(
            [d_V, d_xm, d_xh, d_xj, d_Ca_dyad, d_c1, d_c2, d_xi1ca, d_xi1ba, d_xi2ca, d_xi2ba, d_xr, d_Ca_i, d_xs1, d_xs2, d_xtos, d_ytos, d_xtof, d_ytof, d_Na_i, d_Ca_submem, d_Ca_NSR, d_Ca_JSR, d_xir, d_tropi, d_trops],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
