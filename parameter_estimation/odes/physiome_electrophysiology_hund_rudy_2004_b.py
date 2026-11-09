import torch
import torch.nn as nn
import math


class ElectrophysiologyHundRudy2004BODE(nn.Module):
    """
    PyTorch translation of electrophysiology/hund_rudy_2004_b/model.py.
    Dynamic states: 29; independent constants: 56; derived constants used by rates: 11.
    Original constant slots: 69; algebraic slots: 92.
    """
    source_file = 'electrophysiology/hund_rudy_2004_b/model.py'
    source_state_dim = 29
    source_constant_dim = 69
    algebraic_dim = 92
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 29
        self.parameter_dim = 56

        self.parameters = (
            parameters
            if parameters is not None
            else {'F': 96485, 'K_o': 5.4, 'Ca_o': 1.8, 'Na_o': 140, 'Cl_o': 100, 'FonRT': 0.0374358835078, 'tissue': 0, 'l': 0.01, 'a': 0.0011, 'stim_offset': 0, 'stim_period': 1000, 'stim_duration': 3, 'stim_amplitude': -15, 'pca': 0.000243, 'gacai': 1, 'gacao': 0.341, 'gitodv': 0.19, 'kmnai': 10, 'kmko': 1.5, 'ibarnak': 0.61875, 'KmCa': 0.000125, 'NCXmax': 4.5, 'ksat': 0.27, 'eta': 0.35, 'KmNai': 12.3, 'KmNao': 87.5, 'KmCai': 0.0036, 'KmCao': 1.3, 'ibarpca': 0.0575, 'kmpca': 0.0005, 'PCl': 4e-07, 'Kmto2': 0.1502, 'GClb': 0.000225, 'GNaL': 0.0065, 'prnak': 0.01833, 'dtau_rel_max': 10, 'CaMK0': 0.05, 'Km': 0.0015, 'KmCaMK': 0.15, 'dKmPLBmax': 0.00017, 'dJupmax': 0.75, 'iupmax': 0.004375, 'Kmup': 0.00092, 'nsrmax': 15, 'CTNaClmax': 9.8443e-06, 'CTKClmax': 7.0756e-06, 'kmt': 0.0005, 'kmc': 0.00238, 'tbar': 0.07, 'cbar': 0.05, 'kmcsqn': 0.8, 'csqnbar': 10, 'BSRmax': 0.047, 'KmBSR': 0.00087, 'BSLmax': 1.124, 'KmBSL': 0.0087}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -85.781844107117, 'H': 0.987317750543, 'm': 0.001356538159, 'J': 0.991924983076, 'Ca_ss': 0.00012271265, 'd': 1.64013e-06, 'dp': 8.98230672628, 'f': 0.999961508634, 'fca': 0.97836624923, 'fca2': 0.893052931249, 'f2': 0.992234519148, 'xr': 7.24074e-06, 'Ca_i': 0.00012131666, 'xs1': 0.019883138161, 'xs2': 0.019890650554, 'ydv': 0.013970786703, 'ydv2': 0.99996472752, 'zdv': 0.829206149767, 'Na_i': 12.972433387269, 'Cl_i': 15.59207157178, 'AA': 0.000816605172, 'mL': 0.001356538159, 'hL': 0.26130711759, 'K_i': 135.469546216758, 'Ca_jsr': 1.737580994071, 'CaMKtrap': 0.021123704774, 'ro': 0, 'ri': 0.862666650318, 'Ca_nsr': 1.832822335168}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['F', 'K_o', 'Ca_o', 'Na_o', 'Cl_o', 'FonRT', 'tissue', 'l', 'a', 'stim_offset', 'stim_period', 'stim_duration', 'stim_amplitude', 'pca', 'gacai', 'gacao', 'gitodv', 'kmnai', 'kmko', 'ibarnak', 'KmCa', 'NCXmax', 'ksat', 'eta', 'KmNai', 'KmNao', 'KmCai', 'KmCao', 'ibarpca', 'kmpca', 'PCl', 'Kmto2', 'GClb', 'GNaL', 'prnak', 'dtau_rel_max', 'CaMK0', 'Km', 'KmCaMK', 'dKmPLBmax', 'dJupmax', 'iupmax', 'Kmup', 'nsrmax', 'CTNaClmax', 'CTKClmax', 'kmt', 'kmc', 'tbar', 'cbar', 'kmcsqn', 'csqnbar', 'BSRmax', 'KmBSR', 'BSLmax', 'KmBSL']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'H', 'm', 'J', 'Ca_ss', 'd', 'dp', 'f', 'fca', 'fca2', 'f2', 'xr', 'Ca_i', 'xs1', 'xs2', 'ydv', 'ydv2', 'zdv', 'Na_i', 'Cl_i', 'AA', 'mL', 'hL', 'K_i', 'Ca_jsr', 'CaMKtrap', 'ro', 'ri', 'Ca_nsr']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (29)
        V = y[:, 0]
        H = y[:, 1]
        m = y[:, 2]
        J = y[:, 3]
        Ca_ss = y[:, 4]
        d = y[:, 5]
        dp = y[:, 6]
        f = y[:, 7]
        fca = y[:, 8]
        fca2 = y[:, 9]
        f2 = y[:, 10]
        xr = y[:, 11]
        Ca_i = y[:, 12]
        xs1 = y[:, 13]
        xs2 = y[:, 14]
        ydv = y[:, 15]
        ydv2 = y[:, 16]
        zdv = y[:, 17]
        Na_i = y[:, 18]
        Cl_i = y[:, 19]
        AA = y[:, 20]
        mL = y[:, 21]
        hL = y[:, 22]
        K_i = y[:, 23]
        Ca_jsr = y[:, 24]
        CaMKtrap = y[:, 25]
        ro = y[:, 26]
        ri = y[:, 27]
        Ca_nsr = y[:, 28]

        # Independent parameters (source constant order)
        F = theta[:, 0]  # constants[0]
        K_o = theta[:, 1]  # constants[1]
        Ca_o = theta[:, 2]  # constants[2]
        Na_o = theta[:, 3]  # constants[3]
        Cl_o = theta[:, 4]  # constants[4]
        FonRT = theta[:, 5]  # constants[5]
        tissue = theta[:, 6]  # constants[6]
        l = theta[:, 7]  # constants[7]
        a = theta[:, 8]  # constants[8]
        stim_offset = theta[:, 9]  # constants[9]
        stim_period = theta[:, 10]  # constants[10]
        stim_duration = theta[:, 11]  # constants[11]
        stim_amplitude = theta[:, 12]  # constants[12]
        pca = theta[:, 13]  # constants[13]
        gacai = theta[:, 14]  # constants[14]
        gacao = theta[:, 15]  # constants[15]
        gitodv = theta[:, 16]  # constants[16]
        kmnai = theta[:, 17]  # constants[17]
        kmko = theta[:, 18]  # constants[18]
        ibarnak = theta[:, 19]  # constants[19]
        KmCa = theta[:, 20]  # constants[20]
        NCXmax = theta[:, 21]  # constants[21]
        ksat = theta[:, 22]  # constants[22]
        eta = theta[:, 23]  # constants[23]
        KmNai = theta[:, 24]  # constants[24]
        KmNao = theta[:, 25]  # constants[25]
        KmCai = theta[:, 26]  # constants[26]
        KmCao = theta[:, 27]  # constants[27]
        ibarpca = theta[:, 28]  # constants[28]
        kmpca = theta[:, 29]  # constants[29]
        PCl = theta[:, 30]  # constants[30]
        Kmto2 = theta[:, 31]  # constants[31]
        GClb = theta[:, 32]  # constants[32]
        GNaL = theta[:, 33]  # constants[33]
        prnak = theta[:, 34]  # constants[34]
        dtau_rel_max = theta[:, 35]  # constants[35]
        CaMK0 = theta[:, 36]  # constants[36]
        Km = theta[:, 37]  # constants[37]
        KmCaMK = theta[:, 38]  # constants[38]
        dKmPLBmax = theta[:, 39]  # constants[39]
        dJupmax = theta[:, 40]  # constants[40]
        iupmax = theta[:, 41]  # constants[41]
        Kmup = theta[:, 42]  # constants[42]
        nsrmax = theta[:, 43]  # constants[43]
        CTNaClmax = theta[:, 44]  # constants[44]
        CTKClmax = theta[:, 45]  # constants[45]
        kmt = theta[:, 46]  # constants[46]
        kmc = theta[:, 47]  # constants[47]
        tbar = theta[:, 48]  # constants[48]
        cbar = theta[:, 49]  # constants[49]
        kmcsqn = theta[:, 50]  # constants[50]
        csqnbar = theta[:, 51]  # constants[51]
        BSRmax = theta[:, 52]  # constants[52]
        KmBSR = theta[:, 53]  # constants[53]
        BSLmax = theta[:, 54]  # constants[54]
        KmBSL = theta[:, 55]  # constants[55]

        # Derived constants from the original initConsts()
        vcell = 1000.0 * math.pi * a * a * l  # constants[56]
        GNa = torch.where(tissue == 0.0, 8.25, 4.0 * 8.25)  # constants[57]
        gkr = 0.0138542 * (K_o / 5.4) ** (1.0 / 2)  # constants[58]
        sigma = (torch.exp(Na_o / 67.3) - 1.0) / 7.0  # constants[59]
        ageo = 2.0 * math.pi * a * a + 2.0 * math.pi * a * l  # constants[60]
        Acap = ageo * 2.0  # constants[63]
        vmyo = vcell * 0.68  # constants[64]
        vnsr = vcell * 0.0552  # constants[65]
        vjsr = vcell * 0.0048  # constants[66]
        vss = vcell * 0.02  # constants[67]
        AF = Acap / F  # constants[68]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_H = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_J = torch.zeros_like(y[:, 0])
        d_Ca_ss = torch.zeros_like(y[:, 0])
        d_d = torch.zeros_like(y[:, 0])
        d_dp = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_fca = torch.zeros_like(y[:, 0])
        d_fca2 = torch.zeros_like(y[:, 0])
        d_f2 = torch.zeros_like(y[:, 0])
        d_xr = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_xs1 = torch.zeros_like(y[:, 0])
        d_xs2 = torch.zeros_like(y[:, 0])
        d_ydv = torch.zeros_like(y[:, 0])
        d_ydv2 = torch.zeros_like(y[:, 0])
        d_zdv = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_Cl_i = torch.zeros_like(y[:, 0])
        d_AA = torch.zeros_like(y[:, 0])
        d_mL = torch.zeros_like(y[:, 0])
        d_hL = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_Ca_jsr = torch.zeros_like(y[:, 0])
        d_CaMKtrap = torch.zeros_like(y[:, 0])
        d_ro = torch.zeros_like(y[:, 0])
        d_ri = torch.zeros_like(y[:, 0])
        d_Ca_nsr = torch.zeros_like(y[:, 0])
        alg_0_past = torch.zeros_like(y[:, 0])
        alg_1_am = torch.zeros_like(y[:, 0])
        alg_2_ah = torch.zeros_like(y[:, 0])
        alg_3_aj = torch.zeros_like(y[:, 0])
        alg_4_dss = torch.zeros_like(y[:, 0])
        alg_5_fss = torch.zeros_like(y[:, 0])
        alg_6_f2ss = torch.zeros_like(y[:, 0])
        alg_7_dpss = torch.zeros_like(y[:, 0])
        alg_8_xrss = torch.zeros_like(y[:, 0])
        alg_9_xss = torch.zeros_like(y[:, 0])
        alg_10_ay = torch.zeros_like(y[:, 0])
        alg_11_ay2 = torch.zeros_like(y[:, 0])
        alg_12_ay3 = torch.zeros_like(y[:, 0])
        alg_13_AAss = torch.zeros_like(y[:, 0])
        alg_14_amL = torch.zeros_like(y[:, 0])
        alg_15_hLss = torch.zeros_like(y[:, 0])
        alg_16_i_Stim = torch.zeros_like(y[:, 0])
        alg_17_bm = torch.zeros_like(y[:, 0])
        alg_18_bh = torch.zeros_like(y[:, 0])
        alg_19_bj = torch.zeros_like(y[:, 0])
        alg_20_taud = torch.zeros_like(y[:, 0])
        alg_21_tauf = torch.zeros_like(y[:, 0])
        alg_22_tauf2 = torch.zeros_like(y[:, 0])
        alg_23_tauxr = torch.zeros_like(y[:, 0])
        alg_24_tauxs = torch.zeros_like(y[:, 0])
        alg_25_by = torch.zeros_like(y[:, 0])
        alg_26_by2 = torch.zeros_like(y[:, 0])
        alg_27_by3 = torch.zeros_like(y[:, 0])
        alg_28_bmL = torch.zeros_like(y[:, 0])
        alg_29_gNa = torch.zeros_like(y[:, 0])
        alg_30_ibarca = torch.zeros_like(y[:, 0])
        alg_31_ICaL = torch.zeros_like(y[:, 0])
        alg_32_fcass = torch.zeros_like(y[:, 0])
        alg_33_fca2ss = torch.zeros_like(y[:, 0])
        alg_34_dro_inf = torch.zeros_like(y[:, 0])
        alg_35_taufca2 = torch.zeros_like(y[:, 0])
        alg_36_ross = torch.zeros_like(y[:, 0])
        alg_37_r = torch.zeros_like(y[:, 0])
        alg_38_gks = torch.zeros_like(y[:, 0])
        alg_39_rv = torch.zeros_like(y[:, 0])
        alg_40_fnak = torch.zeros_like(y[:, 0])
        alg_41_INaK = torch.zeros_like(y[:, 0])
        alg_42_ca_i_NaCa = torch.zeros_like(y[:, 0])
        alg_43_allo = torch.zeros_like(y[:, 0])
        alg_44_num = torch.zeros_like(y[:, 0])
        alg_45_denom1 = torch.zeros_like(y[:, 0])
        alg_46_denom2 = torch.zeros_like(y[:, 0])
        alg_47_denom3 = torch.zeros_like(y[:, 0])
        alg_48_INaCa = torch.zeros_like(y[:, 0])
        alg_49_IpCa = torch.zeros_like(y[:, 0])
        alg_50_ICab = torch.zeros_like(y[:, 0])
        alg_51_caiont = torch.zeros_like(y[:, 0])
        alg_52_CaMKbound = torch.zeros_like(y[:, 0])
        alg_53_Ito2_max = torch.zeros_like(y[:, 0])
        alg_54_CaMKactive = torch.zeros_like(y[:, 0])
        alg_55_Ito2 = torch.zeros_like(y[:, 0])
        alg_56_taufca = torch.zeros_like(y[:, 0])
        alg_57_cafac = torch.zeros_like(y[:, 0])
        alg_58_vg = torch.zeros_like(y[:, 0])
        alg_59_ENa = torch.zeros_like(y[:, 0])
        alg_60_dtau_rel = torch.zeros_like(y[:, 0])
        alg_61_Grel = torch.zeros_like(y[:, 0])
        alg_62_INa = torch.zeros_like(y[:, 0])
        alg_63_riss = torch.zeros_like(y[:, 0])
        alg_64_irelcicr = torch.zeros_like(y[:, 0])
        alg_65_INal = torch.zeros_like(y[:, 0])
        alg_66_tauri = torch.zeros_like(y[:, 0])
        alg_67_dKmPLB = torch.zeros_like(y[:, 0])
        alg_68_naiont = torch.zeros_like(y[:, 0])
        alg_69_dJup = torch.zeros_like(y[:, 0])
        alg_70_EK = torch.zeros_like(y[:, 0])
        alg_71_iup = torch.zeros_like(y[:, 0])
        alg_72_ak1 = torch.zeros_like(y[:, 0])
        alg_73_ileak = torch.zeros_like(y[:, 0])
        alg_74_bk1 = torch.zeros_like(y[:, 0])
        alg_75_itr = torch.zeros_like(y[:, 0])
        alg_76_idiff = torch.zeros_like(y[:, 0])
        alg_77_IK1 = torch.zeros_like(y[:, 0])
        alg_78_bcsqn = torch.zeros_like(y[:, 0])
        alg_79_bmyo = torch.zeros_like(y[:, 0])
        alg_80_bss = torch.zeros_like(y[:, 0])
        alg_81_IKr = torch.zeros_like(y[:, 0])
        alg_82_Ito = torch.zeros_like(y[:, 0])
        alg_83_IKp = torch.zeros_like(y[:, 0])
        alg_84_EKs = torch.zeros_like(y[:, 0])
        alg_85_IKs = torch.zeros_like(y[:, 0])
        alg_86_kiont = torch.zeros_like(y[:, 0])
        alg_87_ECl = torch.zeros_like(y[:, 0])
        alg_88_IClb = torch.zeros_like(y[:, 0])
        alg_89_clont = torch.zeros_like(y[:, 0])
        alg_90_CTNaCl = torch.zeros_like(y[:, 0])
        alg_91_CTKCl = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_7_dpss = 9.0 - 8.0 / (1.0 + torch.exp(-(V + 65.0) / 3.4))
        d_dp = torch.zeros_like(y[:, 0]) + ((alg_7_dpss - dp) / 10.0)
        alg_13_AAss = 1.0 / (1.0 + Kmto2 / Ca_ss)
        d_AA = torch.zeros_like(y[:, 0]) + ((alg_13_AAss - AA) / 1.0)
        alg_15_hLss = 1.0 / (1.0 + torch.exp((V + 91.0) / 6.1))
        d_hL = torch.zeros_like(y[:, 0]) + ((alg_15_hLss - hL) / 600.0)
        alg_2_ah = torch.where(V >= -40.0, 0.0, 0.135 * torch.exp((80.0 + V) / -6.8))
        alg_18_bh = torch.where(V >= -40.0, 1.0 / (0.13 * (1.0 + torch.exp((V + 10.66) / -11.1))), 3.56 * torch.exp(0.079 * V) + 310000.0 * torch.exp(0.35 * V))
        d_H = torch.zeros_like(y[:, 0]) + (alg_2_ah * (1.0 - H) - alg_18_bh * H)
        alg_1_am = 0.32 * 1.0 * (V + 47.13) / (1.0 - torch.exp(-0.1 * (V + 47.13)))
        alg_17_bm = 0.08 * torch.exp(-V / 11.0)
        d_m = torch.zeros_like(y[:, 0]) + (alg_1_am * (1.0 - m) - alg_17_bm * m)
        alg_3_aj = torch.where(V >= -40.0, 0.0, (-127140.0 * torch.exp(0.2444 * V) - 3.474e-05 * torch.exp(-0.04391 * V)) * 1.0 * (V + 37.78) / (1.0 + torch.exp(0.311 * (V + 79.23))))
        alg_19_bj = torch.where(V >= -40.0, 0.3 * torch.exp(-2.535e-07 * V) / (1.0 + torch.exp(-0.1 * (V + 32.0))), 0.1212 * torch.exp(-0.01052 * V) / (1.0 + torch.exp(-0.1378 * (V + 40.14))))
        d_J = torch.zeros_like(y[:, 0]) + (alg_3_aj * (1.0 - J) - alg_19_bj * J)
        alg_4_dss = 1.0 / (1.0 + torch.exp(-(V - 4.0) / 6.74))
        alg_20_taud = 0.59 + 0.8 * torch.exp(0.052 * (V + 13.0)) / (1.0 + torch.exp(0.132 * (V + 13.0)))
        d_d = torch.zeros_like(y[:, 0]) + ((alg_4_dss - d) / alg_20_taud)
        alg_5_fss = 0.7 / (1.0 + torch.exp((V + 17.12) / 7.0)) + 0.3
        alg_21_tauf = 1.0 / (0.2411 * torch.exp(-(0.045 * (V - 9.6914)) ** 2.0) + 0.0529)
        d_f = torch.zeros_like(y[:, 0]) + ((alg_5_fss - f) / alg_21_tauf)
        alg_6_f2ss = 0.77 / (1.0 + torch.exp((V + 17.12) / 7.0)) + 0.23
        alg_22_tauf2 = 1.0 / (0.0423 * torch.exp(-(0.059 * (V - 18.5726)) ** 2.0) + 0.0054)
        d_f2 = torch.zeros_like(y[:, 0]) + ((alg_6_f2ss - f2) / alg_22_tauf2)
        alg_8_xrss = 1.0 / (1.0 + torch.exp(-(V + 10.085) / 4.25))
        alg_23_tauxr = 1.0 / (0.0006 * (V - 1.7384) / (1.0 - torch.exp(-0.136 * (V - 1.7384))) + 0.0003 * (V + 38.3608) / (torch.exp(0.1522 * (V + 38.3608)) - 1.0))
        d_xr = torch.zeros_like(y[:, 0]) + ((alg_8_xrss - xr) / alg_23_tauxr)
        alg_9_xss = 1.0 / (1.0 + torch.exp(-(V - 10.5) / 24.7))
        alg_24_tauxs = 1.0 / (7.61e-05 * (V + 44.6) / (1.0 - torch.exp(-9.97 * (V + 44.6))) + 0.00036 * (V - 0.55) / (torch.exp(0.128 * (V - 0.55)) - 1.0))
        d_xs1 = torch.zeros_like(y[:, 0]) + ((alg_9_xss - xs1) / alg_24_tauxs)
        d_xs2 = torch.zeros_like(y[:, 0]) + ((alg_9_xss - xs2) / alg_24_tauxs / 2.0)
        alg_10_ay = 25.0 * torch.exp((V - 40.0) / 25.0) / (1.0 + torch.exp((V - 40.0) / 25.0))
        alg_25_by = 25.0 * torch.exp(-(V + 90.0) / 25.0) / (1.0 + torch.exp(-(V + 90.0) / 25.0))
        d_ydv = torch.zeros_like(y[:, 0]) + (alg_10_ay * (1.0 - ydv) - alg_25_by * ydv)
        alg_11_ay2 = 0.03 / (1.0 + torch.exp((V + 60.0) / 5.0))
        alg_26_by2 = 0.2 * torch.exp((V + 25.0) / 5.0) / (1.0 + torch.exp((V + 25.0) / 5.0))
        d_ydv2 = torch.zeros_like(y[:, 0]) + (alg_11_ay2 * (1.0 - ydv2) - alg_26_by2 * ydv2)
        alg_12_ay3 = 0.00225 / (1.0 + torch.exp((V + 60.0) / 5.0))
        alg_27_by3 = 0.1 * torch.exp((V + 25.0) / 5.0) / (1.0 + torch.exp((V + 25.0) / 5.0))
        d_zdv = torch.zeros_like(y[:, 0]) + (alg_12_ay3 * (1.0 - zdv) - alg_27_by3 * zdv)
        alg_14_amL = 0.32 * 1.0 * (V + 47.13) / (1.0 - torch.exp(-0.1 * (V + 47.13)))
        alg_28_bmL = 0.08 * torch.exp(-V / 11.0)
        d_mL = torch.zeros_like(y[:, 0]) + (alg_14_amL * (1.0 - mL) - alg_28_bmL * mL)
        alg_30_ibarca = pca * 4.0 * (V - 15.0) * F * FonRT * (gacai * Ca_ss * torch.exp(2.0 * (V - 15.0) * FonRT) - gacao * Ca_o) / (torch.exp(2.0 * (V - 15.0) * FonRT) - 1.0)
        alg_31_ICaL = torch.where(tissue == 0.0, d ** dp * f * f2 * fca * fca2 * alg_30_ibarca, d * f * f2 * fca * fca2 * alg_30_ibarca)
        alg_33_fca2ss = 1.0 / (1.0 - alg_31_ICaL / 0.01)
        alg_35_taufca2 = 300.0 / (1.0 + torch.exp((-alg_31_ICaL - 0.175) / 0.04)) + 125.0
        d_fca2 = torch.zeros_like(y[:, 0]) + ((alg_33_fca2ss - fca2) / alg_35_taufca2)
        alg_34_dro_inf = Ca_jsr ** 1.9 / (Ca_jsr ** 1.9 + (49.28 * Ca_ss / (Ca_ss + 0.0028)) ** 1.9)
        alg_36_ross = alg_34_dro_inf / ((1.0 / alg_31_ICaL) ** 2.0 + 1.0)
        d_ro = torch.zeros_like(y[:, 0]) + ((alg_36_ross - ro) / 3.0)
        alg_52_CaMKbound = CaMK0 * (1.0 - CaMKtrap) / (1.0 + Km / Ca_ss)
        alg_54_CaMKactive = alg_52_CaMKbound + CaMKtrap
        d_CaMKtrap = torch.zeros_like(y[:, 0]) + (0.05 * alg_54_CaMKactive * (alg_54_CaMKactive - CaMKtrap) - 0.00068 * CaMKtrap)
        alg_32_fcass = 0.3 / (1.0 - alg_31_ICaL / 0.05) + 0.55 / (1.0 + Ca_ss / 0.003) + 0.15
        alg_56_taufca = 10.0 * alg_54_CaMKactive / (0.15 + alg_54_CaMKactive) + 1.0 / (1.0 + Ca_ss / 0.003) + 0.5
        d_fca = torch.zeros_like(y[:, 0]) + ((alg_32_fcass - fca) / alg_56_taufca)
        alg_57_cafac = 1.0 / (1.0 + torch.exp((alg_31_ICaL + 0.05) / 0.015))
        alg_63_riss = 1.0 / (1.0 + torch.exp((Ca_ss - 0.0004 + 0.002 * alg_57_cafac) / 2.5e-05))
        alg_60_dtau_rel = dtau_rel_max * alg_54_CaMKactive / (KmCaMK + alg_54_CaMKactive)
        alg_66_tauri = 3.0 + alg_60_dtau_rel + (350.0 - alg_60_dtau_rel) / (1.0 + torch.exp((Ca_ss - 0.003 + 0.003 * alg_57_cafac) / 0.0002))
        d_ri = torch.zeros_like(y[:, 0]) + ((alg_63_riss - ri) / alg_66_tauri)
        alg_67_dKmPLB = dKmPLBmax * alg_54_CaMKactive / (KmCaMK + alg_54_CaMKactive)
        alg_69_dJup = dJupmax * alg_54_CaMKactive / (KmCaMK + alg_54_CaMKactive)
        alg_71_iup = (alg_69_dJup + 1.0) * iupmax * Ca_i / (Ca_i + Kmup - alg_67_dKmPLB)
        alg_73_ileak = iupmax * Ca_nsr / nsrmax
        alg_75_itr = (Ca_nsr - Ca_jsr) / 120.0
        d_Ca_nsr = torch.zeros_like(y[:, 0]) + (alg_71_iup - alg_75_itr * vjsr / vnsr - alg_73_ileak)
        alg_49_IpCa = ibarpca * Ca_i / (kmpca + Ca_i)
        alg_50_ICab = 1.99508e-07 * 4.0 * V * F * FonRT * (Ca_i * torch.exp(2.0 * V * FonRT) - 0.341 * Ca_o) / (torch.exp(2.0 * V * FonRT) - 1.0)
        alg_42_ca_i_NaCa = 1.5 * Ca_i
        alg_43_allo = 1.0 / (1.0 + (KmCa / alg_42_ca_i_NaCa) ** 2.0)
        alg_44_num = Na_i ** 3.0 * Ca_o * torch.exp(eta * V * FonRT) - Na_o ** 3.0 * alg_42_ca_i_NaCa * torch.exp((eta - 1.0) * V * FonRT)
        alg_45_denom1 = 1.0 + ksat * torch.exp((eta - 1.0) * V * FonRT)
        alg_46_denom2 = KmCao * Na_i ** 3.0 + KmNao ** 3.0 * alg_42_ca_i_NaCa + KmNai ** 3.0 * Ca_o * (1.0 + alg_42_ca_i_NaCa / KmCai)
        alg_47_denom3 = KmCai * Na_o ** 3.0 * (1.0 + (Na_i / KmNai) ** 3.0) + Na_i ** 3.0 * Ca_o + Na_o ** 3.0 * alg_42_ca_i_NaCa
        alg_48_INaCa = NCXmax * alg_43_allo * alg_44_num / (alg_45_denom1 * (alg_46_denom2 + alg_47_denom3))
        alg_76_idiff = (Ca_ss - Ca_i) / 0.2
        alg_79_bmyo = 1.0 / (1.0 + cbar * kmc / (Ca_i + kmc) ** 2.0 + kmt * tbar / (Ca_i + kmt) ** 2.0)
        d_Ca_i = torch.zeros_like(y[:, 0]) + (alg_79_bmyo * (-(alg_50_ICab + alg_49_IpCa - 2.0 * alg_48_INaCa) * AF / (vmyo * 2.0) + (alg_73_ileak - alg_71_iup) * vnsr / vmyo + alg_76_idiff * vss / vmyo))
        alg_58_vg = torch.where(tissue == 0.0, 1.0 / (1.0 + torch.exp((alg_30_ibarca + 13.0) / 5.0)), 1.0)
        alg_61_Grel = 3000.0 * alg_58_vg
        alg_64_irelcicr = alg_61_Grel * ro * ri * (Ca_jsr - Ca_ss)
        alg_80_bss = 1.0 / (1.0 + BSRmax * KmBSR / (KmBSR + Ca_ss) ** 2.0 + BSLmax * KmBSL / (KmBSL + Ca_ss) ** 2.0)
        d_Ca_ss = torch.zeros_like(y[:, 0]) + (alg_80_bss * (-alg_31_ICaL * AF / (vss * 2.0) + alg_64_irelcicr * vjsr / vss - alg_76_idiff))
        alg_78_bcsqn = 1.0 / (1.0 + kmcsqn * csqnbar / (Ca_jsr + kmcsqn) ** 2.0)
        d_Ca_jsr = torch.zeros_like(y[:, 0]) + (alg_78_bcsqn * (alg_75_itr - alg_64_irelcicr))
        alg_51_caiont = alg_31_ICaL + alg_50_ICab + alg_49_IpCa - 2.0 * alg_48_INaCa
        alg_59_ENa = torch.log(Na_o / Na_i) / FonRT
        alg_29_gNa = GNa * m * m * m * H * J
        alg_62_INa = alg_29_gNa * (V - alg_59_ENa)
        alg_40_fnak = 1.0 / (1.0 + 0.1245 * torch.exp(-0.1 * V * FonRT) + 0.0365 * sigma * torch.exp(-V * FonRT))
        alg_41_INaK = ibarnak * alg_40_fnak * 1.0 / (1.0 + (kmnai / Na_i) ** 2.0) * K_o / (K_o + kmko)
        alg_65_INal = GNaL * mL ** 3.0 * hL * (V - alg_59_ENa)
        alg_68_naiont = alg_62_INa + 3.0 * alg_48_INaCa + 3.0 * alg_41_INaK + alg_65_INal
        alg_70_EK = torch.log(K_o / K_i) / FonRT
        alg_72_ak1 = 1.02 / (1.0 + torch.exp(0.2385 * (V - alg_70_EK - 59.215)))
        alg_74_bk1 = (0.49124 * torch.exp(0.08032 * (V - alg_70_EK + 5.476)) + 1.0 * torch.exp(0.06175 * (V - alg_70_EK - 594.31))) / (1.0 + torch.exp(-0.5143 * (V - alg_70_EK + 4.753)))
        alg_77_IK1 = 0.5 * (K_o / 5.4) ** (1.0 / 2) * alg_72_ak1 / (alg_72_ak1 + alg_74_bk1) * (V - alg_70_EK)
        alg_83_IKp = 0.00276 * (V - alg_70_EK) / (1.0 + torch.exp((7.488 - V) / 5.98))
        alg_38_gks = 0.0248975 * (1.0 + 0.6 / (1.0 + (3.8e-05 / Ca_i) ** 1.4))
        alg_84_EKs = torch.log((K_o + prnak * Na_o) / (K_i + prnak * Na_i)) / FonRT
        alg_85_IKs = alg_38_gks * xs1 * xs2 * (V - alg_84_EKs)
        alg_37_r = 1.0 / (1.0 + torch.exp((V + 10.0) / 15.4))
        alg_81_IKr = gkr * xr * alg_37_r * (V - alg_70_EK)
        alg_39_rv = torch.exp(V / 300.0)
        alg_82_Ito = gitodv * ydv ** 3.0 * ydv2 * zdv * alg_39_rv * (V - alg_70_EK)
        alg_0_past = torch.floor(t / stim_period) * stim_period
        alg_16_i_Stim = torch.where((t - alg_0_past >= stim_offset) & (t - alg_0_past <= stim_offset + stim_duration), stim_amplitude, 0.0)
        alg_86_kiont = alg_81_IKr + alg_85_IKs + alg_77_IK1 + alg_83_IKp - 2.0 * alg_41_INaK + alg_82_Ito + 0.5 * alg_16_i_Stim
        alg_53_Ito2_max = PCl * V * F * FonRT * (Cl_i - Cl_o * torch.exp(V * FonRT)) / (1.0 - torch.exp(V * FonRT))
        alg_55_Ito2 = alg_53_Ito2_max * AA
        alg_87_ECl = -torch.log(Cl_o / Cl_i) / FonRT
        alg_88_IClb = GClb * (V - alg_87_ECl)
        alg_89_clont = alg_88_IClb + alg_55_Ito2 + 0.5 * alg_16_i_Stim
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_68_naiont + alg_86_kiont + alg_51_caiont + alg_89_clont))
        alg_90_CTNaCl = CTNaClmax * (alg_59_ENa - alg_87_ECl) ** 4.0 / ((alg_59_ENa - alg_87_ECl) ** 4.0 + 59494197.640692554)
        d_Na_i = torch.zeros_like(y[:, 0]) + (-alg_68_naiont * AF / vmyo + alg_90_CTNaCl)
        alg_91_CTKCl = CTKClmax * (alg_70_EK - alg_87_ECl) / (alg_70_EK - alg_87_ECl + 87.8251)
        d_K_i = torch.zeros_like(y[:, 0]) + (-alg_86_kiont * AF / vmyo + alg_91_CTKCl)
        d_Cl_i = torch.zeros_like(y[:, 0]) + (alg_89_clont * AF / vmyo + alg_90_CTNaCl + alg_91_CTKCl)

        return torch.stack(
            [d_V, d_H, d_m, d_J, d_Ca_ss, d_d, d_dp, d_f, d_fca, d_fca2, d_f2, d_xr, d_Ca_i, d_xs1, d_xs2, d_ydv, d_ydv2, d_zdv, d_Na_i, d_Cl_i, d_AA, d_mL, d_hL, d_K_i, d_Ca_jsr, d_CaMKtrap, d_ro, d_ri, d_Ca_nsr],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
