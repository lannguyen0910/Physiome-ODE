import torch
import torch.nn as nn
import math


class MetabolismCloutier2009ODE(nn.Module):
    """
    PyTorch translation of metabolism/cloutier_2009/model.py.
    Dynamic states: 36; independent constants: 125; derived constants used by rates: 0.
    Original constant slots: 125; algebraic slots: 64.
    """
    source_file = 'metabolism/cloutier_2009/model.py'
    source_state_dim = 36
    source_constant_dim = 125
    algebraic_dim = 64
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 36
        self.parameter_dim = 125

        self.parameters = (
            parameters
            if parameters is not None
            else {'nOP': 15, 'NAero': 3, 'Rng': 1.8, 'Reg': 0.8, 'Ren': 0.4444444444444444, 'Rcn': 0.01222, 'Rcg': 0.022, 'Rce': 0.0275, 'O2a': 8.34, 'gn_NA': 0.0039, 'Sm_n': 40500, 'Vm': -70, 'Vn': 0.45, 'RT': 2577340, 'F': 96500, 'NAe': 150, 'kpump': 3.17e-07, 'Km_pump': 0.4243, 'Km_en_GLC': 5.32, 'Vm_en_GLC': 0.50417, 'Vmax_n_hk': 0.0513, 'Km_GLC': 0.105, 'G6P_inh_hk': 0.6, 'aG6P_inh_hk': 20, 'Vmaxf_n_pgi': 0.5, 'Vmaxr_n_pgi': 0.45, 'Km_G6P': 0.5, 'Km_F6P_pgi': 0.06, 'kn_pfk': 0.55783, 'Km_F6P_pfk': 0.18, 'Ki_ATP': 0.7595, 'nH': 4, 'kn_pgk': 0.4287, 'kn_pk': 28.6, 'kfn_ldh': 5.3, 'krn_ldh': 0.1046, 'Vmax_n_mito': 0.05557, 'Km_O2': 0.0029658, 'Km_ADP': 0.00107, 'Km_PYR': 0.0632, 'rATP_mito': 20, 'aATP_mito': 5, 'Vmax_ne_LAC': 0.1978, 'Km_ne_LAC': 0.09314, 'Vmax_n_ATPase': 0.04889, 'krn_ck': 0.015, 'kfn_ck': 0.0524681, 'nh_O2': 2.7, 'PScapn': 0.2202, 'Ko2': 0.089733, 'HbOP': 8.6, 'gg_NA': 0.00325, 'Sm_g': 10500, 'Vg': 0.25, 'Km_eg_GLC': 3.53, 'Vm_eg_GLC': 0.038089, 'KO1': 1, 'Km_cg_GLC': 9.92, 'Vm_cg_GLC': 0.0098394, 'Vmax_g_hk': 0.050461, 'Vmaxf_g_pgi': 0.5, 'Vmaxr_g_pgi': 0.45, 'kg_pfk': 0.403, 'kg_pgk': 0.2514, 'kg_pk': 2.73, 'kfg_ldh': 6.2613, 'krg_ldh': 0.54682, 'Vmax_g_mito': 0.008454, 'Vmax_ge_LAC': 0.086124, 'Km_ge_LAC': 0.22163, 'Vmax_gc_LAC': 0.00021856, 'Km_gc_LAC': 0.12862, 'Vmax_g_ATPase': 0.035657, 'krg_ck': 0.02073, 'kfg_ck': 0.0243, 'PScapg': 0.2457, 'nh_O2_2': 2.7, 'Vc': 0.0055, 'GLCa': 4.8, 'Km_ce_GLC': 8.4568, 'Vm_ce_GLC': 0.0489, 'LACa': 0.313, 'Km_ec_LAC': 0.764818, 'Vm_ec_LAC': 0.0325, 'R_GLU_NA': 0.075, 'Km_GLU': 0.05, 'KO2': 1, 'Vmax_g_gs': 0.3, 'Km_ATP': 0.01532, 'Vmax_eg_GLU': 0.0208, 'CO2a': 1.2, 'Vmax_glys': 0.0001528, 'Km_G6P_glys': 0.5, 'GLY_inh': 4.2, 'aGLY_inh': 20, 'Vmax_glyp': 4.922e-05, 'Km_GLY': 1, 'stim': 1, 'to': 200, 'to_GLY': 83, 'tend_GLY': 440, 'sr_GLY': 4, 't1': 2, 'delta_GLY': 62, 'KO3': 1, 'CBF0': 0.012, 'tend': 300, 'sr': 4.59186, 'deltaf': 0.42, 'CBF0_2': 0.012, 'Vv0': 0.0236, 'tv': 35, 'NADH_n_tot': 0.22, 'NADH_g_tot': 0.22, 'PCrn_tot': 5, 'PCrg_tot': 5, 'ATPtot': 2.379, 'qak': 0.92, 'k1': 2.22, 'k2': 0.46, 'k3': 0.43, 'dHb0': 0.064, 't_n_stim': 2, 'v1_n': 0.041, 'v2_n': 2.55}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'NAn': 15.533, 'GLCn': 0.2633, 'G6Pn': 0.7275, 'F6Pn': 0.1091, 'GAPn': 0.0418, 'PEPn': 0.0037, 'PYRn': 0.0388, 'LACn': 0.3856, 'NADHn': 0.0319, 'ATPn': 2.2592, 'PCrn': 4.2529, 'O2n': 0.0975, 'GLUn': 3, 'NAg': 13.36, 'GLCg': 0.1656, 'G6Pg': 0.7326, 'F6Pg': 0.1116, 'GAPg': 0.0698, 'PEPg': 0.0254, 'PYRg': 0.1711, 'LACg': 0.4651, 'NADHg': 0.0445, 'ATPg': 2.24, 'PCrg': 4.6817, 'O2g': 0.1589, 'GLYg': 2.5, 'GLUg': 0, 'GLCe': 0.3339, 'LACe': 0.3986, 'GLUe': 0, 'O2c': 7.4201, 'GLCc': 4.6401, 'LACc': 0.3251, 'CO2c': 2.12, 'Vv': 0.0237, 'dHb': 0.0218}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['nOP', 'NAero', 'Rng', 'Reg', 'Ren', 'Rcn', 'Rcg', 'Rce', 'O2a', 'gn_NA', 'Sm_n', 'Vm', 'Vn', 'RT', 'F', 'NAe', 'kpump', 'Km_pump', 'Km_en_GLC', 'Vm_en_GLC', 'Vmax_n_hk', 'Km_GLC', 'G6P_inh_hk', 'aG6P_inh_hk', 'Vmaxf_n_pgi', 'Vmaxr_n_pgi', 'Km_G6P', 'Km_F6P_pgi', 'kn_pfk', 'Km_F6P_pfk', 'Ki_ATP', 'nH', 'kn_pgk', 'kn_pk', 'kfn_ldh', 'krn_ldh', 'Vmax_n_mito', 'Km_O2', 'Km_ADP', 'Km_PYR', 'rATP_mito', 'aATP_mito', 'Vmax_ne_LAC', 'Km_ne_LAC', 'Vmax_n_ATPase', 'krn_ck', 'kfn_ck', 'nh_O2', 'PScapn', 'Ko2', 'HbOP', 'gg_NA', 'Sm_g', 'Vg', 'Km_eg_GLC', 'Vm_eg_GLC', 'KO1', 'Km_cg_GLC', 'Vm_cg_GLC', 'Vmax_g_hk', 'Vmaxf_g_pgi', 'Vmaxr_g_pgi', 'kg_pfk', 'kg_pgk', 'kg_pk', 'kfg_ldh', 'krg_ldh', 'Vmax_g_mito', 'Vmax_ge_LAC', 'Km_ge_LAC', 'Vmax_gc_LAC', 'Km_gc_LAC', 'Vmax_g_ATPase', 'krg_ck', 'kfg_ck', 'PScapg', 'nh_O2_2', 'Vc', 'GLCa', 'Km_ce_GLC', 'Vm_ce_GLC', 'LACa', 'Km_ec_LAC', 'Vm_ec_LAC', 'R_GLU_NA', 'Km_GLU', 'KO2', 'Vmax_g_gs', 'Km_ATP', 'Vmax_eg_GLU', 'CO2a', 'Vmax_glys', 'Km_G6P_glys', 'GLY_inh', 'aGLY_inh', 'Vmax_glyp', 'Km_GLY', 'stim', 'to', 'to_GLY', 'tend_GLY', 'sr_GLY', 't1', 'delta_GLY', 'KO3', 'CBF0', 'tend', 'sr', 'deltaf', 'CBF0_2', 'Vv0', 'tv', 'NADH_n_tot', 'NADH_g_tot', 'PCrn_tot', 'PCrg_tot', 'ATPtot', 'qak', 'k1', 'k2', 'k3', 'dHb0', 't_n_stim', 'v1_n', 'v2_n']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['NAn', 'GLCn', 'G6Pn', 'F6Pn', 'GAPn', 'PEPn', 'PYRn', 'LACn', 'NADHn', 'ATPn', 'PCrn', 'O2n', 'GLUn', 'NAg', 'GLCg', 'G6Pg', 'F6Pg', 'GAPg', 'PEPg', 'PYRg', 'LACg', 'NADHg', 'ATPg', 'PCrg', 'O2g', 'GLYg', 'GLUg', 'GLCe', 'LACe', 'GLUe', 'O2c', 'GLCc', 'LACc', 'CO2c', 'Vv', 'dHb']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (36)
        NAn = y[:, 0]
        GLCn = y[:, 1]
        G6Pn = y[:, 2]
        F6Pn = y[:, 3]
        GAPn = y[:, 4]
        PEPn = y[:, 5]
        PYRn = y[:, 6]
        LACn = y[:, 7]
        NADHn = y[:, 8]
        ATPn = y[:, 9]
        PCrn = y[:, 10]
        O2n = y[:, 11]
        GLUn = y[:, 12]
        NAg = y[:, 13]
        GLCg = y[:, 14]
        G6Pg = y[:, 15]
        F6Pg = y[:, 16]
        GAPg = y[:, 17]
        PEPg = y[:, 18]
        PYRg = y[:, 19]
        LACg = y[:, 20]
        NADHg = y[:, 21]
        ATPg = y[:, 22]
        PCrg = y[:, 23]
        O2g = y[:, 24]
        GLYg = y[:, 25]
        GLUg = y[:, 26]
        GLCe = y[:, 27]
        LACe = y[:, 28]
        GLUe = y[:, 29]
        O2c = y[:, 30]
        GLCc = y[:, 31]
        LACc = y[:, 32]
        CO2c = y[:, 33]
        Vv = y[:, 34]
        dHb = y[:, 35]

        # Independent parameters (source constant order)
        nOP = theta[:, 0]  # constants[0]
        NAero = theta[:, 1]  # constants[1]
        Rng = theta[:, 2]  # constants[2]
        Reg = theta[:, 3]  # constants[3]
        Ren = theta[:, 4]  # constants[4]
        Rcn = theta[:, 5]  # constants[5]
        Rcg = theta[:, 6]  # constants[6]
        Rce = theta[:, 7]  # constants[7]
        O2a = theta[:, 8]  # constants[8]
        gn_NA = theta[:, 9]  # constants[9]
        Sm_n = theta[:, 10]  # constants[10]
        Vm = theta[:, 11]  # constants[11]
        Vn = theta[:, 12]  # constants[12]
        RT = theta[:, 13]  # constants[13]
        F = theta[:, 14]  # constants[14]
        NAe = theta[:, 15]  # constants[15]
        kpump = theta[:, 16]  # constants[16]
        Km_pump = theta[:, 17]  # constants[17]
        Km_en_GLC = theta[:, 18]  # constants[18]
        Vm_en_GLC = theta[:, 19]  # constants[19]
        Vmax_n_hk = theta[:, 20]  # constants[20]
        Km_GLC = theta[:, 21]  # constants[21]
        G6P_inh_hk = theta[:, 22]  # constants[22]
        aG6P_inh_hk = theta[:, 23]  # constants[23]
        Vmaxf_n_pgi = theta[:, 24]  # constants[24]
        Vmaxr_n_pgi = theta[:, 25]  # constants[25]
        Km_G6P = theta[:, 26]  # constants[26]
        Km_F6P_pgi = theta[:, 27]  # constants[27]
        kn_pfk = theta[:, 28]  # constants[28]
        Km_F6P_pfk = theta[:, 29]  # constants[29]
        Ki_ATP = theta[:, 30]  # constants[30]
        nH = theta[:, 31]  # constants[31]
        kn_pgk = theta[:, 32]  # constants[32]
        kn_pk = theta[:, 33]  # constants[33]
        kfn_ldh = theta[:, 34]  # constants[34]
        krn_ldh = theta[:, 35]  # constants[35]
        Vmax_n_mito = theta[:, 36]  # constants[36]
        Km_O2 = theta[:, 37]  # constants[37]
        Km_ADP = theta[:, 38]  # constants[38]
        Km_PYR = theta[:, 39]  # constants[39]
        rATP_mito = theta[:, 40]  # constants[40]
        aATP_mito = theta[:, 41]  # constants[41]
        Vmax_ne_LAC = theta[:, 42]  # constants[42]
        Km_ne_LAC = theta[:, 43]  # constants[43]
        Vmax_n_ATPase = theta[:, 44]  # constants[44]
        krn_ck = theta[:, 45]  # constants[45]
        kfn_ck = theta[:, 46]  # constants[46]
        nh_O2 = theta[:, 47]  # constants[47]
        PScapn = theta[:, 48]  # constants[48]
        Ko2 = theta[:, 49]  # constants[49]
        HbOP = theta[:, 50]  # constants[50]
        gg_NA = theta[:, 51]  # constants[51]
        Sm_g = theta[:, 52]  # constants[52]
        Vg = theta[:, 53]  # constants[53]
        Km_eg_GLC = theta[:, 54]  # constants[54]
        Vm_eg_GLC = theta[:, 55]  # constants[55]
        KO1 = theta[:, 56]  # constants[56]
        Km_cg_GLC = theta[:, 57]  # constants[57]
        Vm_cg_GLC = theta[:, 58]  # constants[58]
        Vmax_g_hk = theta[:, 59]  # constants[59]
        Vmaxf_g_pgi = theta[:, 60]  # constants[60]
        Vmaxr_g_pgi = theta[:, 61]  # constants[61]
        kg_pfk = theta[:, 62]  # constants[62]
        kg_pgk = theta[:, 63]  # constants[63]
        kg_pk = theta[:, 64]  # constants[64]
        kfg_ldh = theta[:, 65]  # constants[65]
        krg_ldh = theta[:, 66]  # constants[66]
        Vmax_g_mito = theta[:, 67]  # constants[67]
        Vmax_ge_LAC = theta[:, 68]  # constants[68]
        Km_ge_LAC = theta[:, 69]  # constants[69]
        Vmax_gc_LAC = theta[:, 70]  # constants[70]
        Km_gc_LAC = theta[:, 71]  # constants[71]
        Vmax_g_ATPase = theta[:, 72]  # constants[72]
        krg_ck = theta[:, 73]  # constants[73]
        kfg_ck = theta[:, 74]  # constants[74]
        PScapg = theta[:, 75]  # constants[75]
        nh_O2_2 = theta[:, 76]  # constants[76]
        Vc = theta[:, 77]  # constants[77]
        GLCa = theta[:, 78]  # constants[78]
        Km_ce_GLC = theta[:, 79]  # constants[79]
        Vm_ce_GLC = theta[:, 80]  # constants[80]
        LACa = theta[:, 81]  # constants[81]
        Km_ec_LAC = theta[:, 82]  # constants[82]
        Vm_ec_LAC = theta[:, 83]  # constants[83]
        R_GLU_NA = theta[:, 84]  # constants[84]
        Km_GLU = theta[:, 85]  # constants[85]
        KO2 = theta[:, 86]  # constants[86]
        Vmax_g_gs = theta[:, 87]  # constants[87]
        Km_ATP = theta[:, 88]  # constants[88]
        Vmax_eg_GLU = theta[:, 89]  # constants[89]
        CO2a = theta[:, 90]  # constants[90]
        Vmax_glys = theta[:, 91]  # constants[91]
        Km_G6P_glys = theta[:, 92]  # constants[92]
        GLY_inh = theta[:, 93]  # constants[93]
        aGLY_inh = theta[:, 94]  # constants[94]
        Vmax_glyp = theta[:, 95]  # constants[95]
        Km_GLY = theta[:, 96]  # constants[96]
        stim = theta[:, 97]  # constants[97]
        to = theta[:, 98]  # constants[98]
        to_GLY = theta[:, 99]  # constants[99]
        tend_GLY = theta[:, 100]  # constants[100]
        sr_GLY = theta[:, 101]  # constants[101]
        t1 = theta[:, 102]  # constants[102]
        delta_GLY = theta[:, 103]  # constants[103]
        KO3 = theta[:, 104]  # constants[104]
        CBF0 = theta[:, 105]  # constants[105]
        tend = theta[:, 106]  # constants[106]
        sr = theta[:, 107]  # constants[107]
        deltaf = theta[:, 108]  # constants[108]
        CBF0_2 = theta[:, 109]  # constants[109]
        Vv0 = theta[:, 110]  # constants[110]
        tv = theta[:, 111]  # constants[111]
        NADH_n_tot = theta[:, 112]  # constants[112]
        NADH_g_tot = theta[:, 113]  # constants[113]
        PCrn_tot = theta[:, 114]  # constants[114]
        PCrg_tot = theta[:, 115]  # constants[115]
        ATPtot = theta[:, 116]  # constants[116]
        qak = theta[:, 117]  # constants[117]
        k1 = theta[:, 118]  # constants[118]
        k2 = theta[:, 119]  # constants[119]
        k3 = theta[:, 120]  # constants[120]
        dHb0 = theta[:, 121]  # constants[121]
        t_n_stim = theta[:, 122]  # constants[122]
        v1_n = theta[:, 123]  # constants[123]
        v2_n = theta[:, 124]  # constants[124]

        # Initialize rates and algebraic variables, as in computeRates()
        d_NAn = torch.zeros_like(y[:, 0])
        d_GLCn = torch.zeros_like(y[:, 0])
        d_G6Pn = torch.zeros_like(y[:, 0])
        d_F6Pn = torch.zeros_like(y[:, 0])
        d_GAPn = torch.zeros_like(y[:, 0])
        d_PEPn = torch.zeros_like(y[:, 0])
        d_PYRn = torch.zeros_like(y[:, 0])
        d_LACn = torch.zeros_like(y[:, 0])
        d_NADHn = torch.zeros_like(y[:, 0])
        d_ATPn = torch.zeros_like(y[:, 0])
        d_PCrn = torch.zeros_like(y[:, 0])
        d_O2n = torch.zeros_like(y[:, 0])
        d_GLUn = torch.zeros_like(y[:, 0])
        d_NAg = torch.zeros_like(y[:, 0])
        d_GLCg = torch.zeros_like(y[:, 0])
        d_G6Pg = torch.zeros_like(y[:, 0])
        d_F6Pg = torch.zeros_like(y[:, 0])
        d_GAPg = torch.zeros_like(y[:, 0])
        d_PEPg = torch.zeros_like(y[:, 0])
        d_PYRg = torch.zeros_like(y[:, 0])
        d_LACg = torch.zeros_like(y[:, 0])
        d_NADHg = torch.zeros_like(y[:, 0])
        d_ATPg = torch.zeros_like(y[:, 0])
        d_PCrg = torch.zeros_like(y[:, 0])
        d_O2g = torch.zeros_like(y[:, 0])
        d_GLYg = torch.zeros_like(y[:, 0])
        d_GLUg = torch.zeros_like(y[:, 0])
        d_GLCe = torch.zeros_like(y[:, 0])
        d_LACe = torch.zeros_like(y[:, 0])
        d_GLUe = torch.zeros_like(y[:, 0])
        d_O2c = torch.zeros_like(y[:, 0])
        d_GLCc = torch.zeros_like(y[:, 0])
        d_LACc = torch.zeros_like(y[:, 0])
        d_CO2c = torch.zeros_like(y[:, 0])
        d_Vv = torch.zeros_like(y[:, 0])
        d_dHb = torch.zeros_like(y[:, 0])
        alg_0_Vn_leak_Na = torch.zeros_like(y[:, 0])
        alg_3_Vn_pump = torch.zeros_like(y[:, 0])
        alg_4_V_en_GLC = torch.zeros_like(y[:, 0])
        alg_5_Vn_hk = torch.zeros_like(y[:, 0])
        alg_6_Vn_pgi = torch.zeros_like(y[:, 0])
        alg_7_Vn_pfk = torch.zeros_like(y[:, 0])
        alg_8_Vne_LAC = torch.zeros_like(y[:, 0])
        alg_9_Vn_ATPase = torch.zeros_like(y[:, 0])
        alg_10_Vcn_O2 = torch.zeros_like(y[:, 0])
        alg_11_Vg_leak_Na = torch.zeros_like(y[:, 0])
        alg_12_Vg_pump = torch.zeros_like(y[:, 0])
        alg_13_Veg_GLC = torch.zeros_like(y[:, 0])
        alg_14_Vcg_GLC = torch.zeros_like(y[:, 0])
        alg_15_Vg_hk = torch.zeros_like(y[:, 0])
        alg_16_Vg_pgi = torch.zeros_like(y[:, 0])
        alg_17_Vg_pfk = torch.zeros_like(y[:, 0])
        alg_18_Vg_glys = torch.zeros_like(y[:, 0])
        alg_19_Vge_LAC = torch.zeros_like(y[:, 0])
        alg_20_unitstepSB2 = torch.zeros_like(y[:, 0])
        alg_21_Vgc_LAC = torch.zeros_like(y[:, 0])
        alg_22_deltaVt_GLY = torch.zeros_like(y[:, 0])
        alg_23_Vg_ATPase = torch.zeros_like(y[:, 0])
        alg_24_Vg_glyp = torch.zeros_like(y[:, 0])
        alg_25_Vcg_O2 = torch.zeros_like(y[:, 0])
        alg_26_Vce_GLC = torch.zeros_like(y[:, 0])
        alg_27_Vec_LAC = torch.zeros_like(y[:, 0])
        alg_28_Vg_gs = torch.zeros_like(y[:, 0])
        alg_29_Veg_GLU = torch.zeros_like(y[:, 0])
        alg_30_Fin_t = torch.zeros_like(y[:, 0])
        alg_31_unitpulseSB = torch.zeros_like(y[:, 0])
        alg_32_Vc_CO2 = torch.zeros_like(y[:, 0])
        alg_33_Vc_O2 = torch.zeros_like(y[:, 0])
        alg_34_Vc_GLC = torch.zeros_like(y[:, 0])
        alg_35_Vc_LAC = torch.zeros_like(y[:, 0])
        alg_36_Fout_t = torch.zeros_like(y[:, 0])
        alg_37_v_stim = torch.zeros_like(y[:, 0])
        alg_38_NADn = torch.zeros_like(y[:, 0])
        alg_39_Vn_stim = torch.zeros_like(y[:, 0])
        alg_40_Vn_ldh = torch.zeros_like(y[:, 0])
        alg_41_NADg = torch.zeros_like(y[:, 0])
        alg_42_Vn_stim_GLU = torch.zeros_like(y[:, 0])
        alg_43_Vg_ldh = torch.zeros_like(y[:, 0])
        alg_44_CRn = torch.zeros_like(y[:, 0])
        alg_45_CRg = torch.zeros_like(y[:, 0])
        alg_46_ADPn = torch.zeros_like(y[:, 0])
        alg_47_Vn_pgk = torch.zeros_like(y[:, 0])
        alg_49_Vn_pk = torch.zeros_like(y[:, 0])
        alg_50_Vn_mito = torch.zeros_like(y[:, 0])
        alg_51_Vn_ck = torch.zeros_like(y[:, 0])
        alg_52_Vnc_CO2 = torch.zeros_like(y[:, 0])
        alg_53_u_n = torch.zeros_like(y[:, 0])
        alg_54_ADPg = torch.zeros_like(y[:, 0])
        alg_55_dAMP_dATPn = torch.zeros_like(y[:, 0])
        alg_56_Vg_pgk = torch.zeros_like(y[:, 0])
        alg_58_Vg_pk = torch.zeros_like(y[:, 0])
        alg_59_Vg_mito = torch.zeros_like(y[:, 0])
        alg_60_Vg_ck = torch.zeros_like(y[:, 0])
        alg_61_Vgc_CO2 = torch.zeros_like(y[:, 0])
        alg_62_u_g = torch.zeros_like(y[:, 0])
        alg_63_dAMP_dATPg = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_4_V_en_GLC = Vm_en_GLC * (GLCe / (GLCe + Km_en_GLC) - GLCn / (GLCn + Km_en_GLC))
        alg_5_Vn_hk = Vmax_n_hk * ATPn * (GLCn / (GLCn + Km_GLC)) * (1.0 - 1.0 / (1.0 + torch.exp(-aG6P_inh_hk * (1.0 * (G6Pn - G6P_inh_hk)))))
        d_GLCn = torch.zeros_like(y[:, 0]) + (alg_4_V_en_GLC - alg_5_Vn_hk)
        alg_6_Vn_pgi = Vmaxf_n_pgi * (G6Pn / (G6Pn + Km_G6P)) - Vmaxr_n_pgi * (F6Pn / (F6Pn + Km_F6P_pgi))
        d_G6Pn = torch.zeros_like(y[:, 0]) + (alg_5_Vn_hk - alg_6_Vn_pgi)
        alg_7_Vn_pfk = kn_pfk * ATPn * (F6Pn / (F6Pn + Km_F6P_pfk)) * (1.0 + (ATPn / Ki_ATP) ** nH) ** (-1.0)
        d_F6Pn = torch.zeros_like(y[:, 0]) + (alg_6_Vn_pgi - alg_7_Vn_pfk)
        alg_14_Vcg_GLC = Vm_cg_GLC * (GLCc / (GLCc + Km_cg_GLC) - GLCg / (GLCg + Km_cg_GLC))
        alg_13_Veg_GLC = KO1 * Vm_eg_GLC * (GLCe / (GLCe + Km_eg_GLC) - GLCg / (GLCg + Km_eg_GLC))
        alg_15_Vg_hk = Vmax_g_hk * ATPg * (GLCg / (GLCg + Km_GLC)) * (1.0 - 1.0 / (1.0 + torch.exp(-aG6P_inh_hk * (1.0 * (G6Pg - G6P_inh_hk)))))
        d_GLCg = torch.zeros_like(y[:, 0]) + (alg_14_Vcg_GLC + alg_13_Veg_GLC - alg_15_Vg_hk)
        alg_16_Vg_pgi = Vmaxf_g_pgi * (G6Pg / (G6Pg + Km_G6P)) - Vmaxr_g_pgi * (F6Pg / (F6Pg + Km_F6P_pgi))
        alg_17_Vg_pfk = kg_pfk * ATPg * (F6Pg / (F6Pg + Km_F6P_pfk)) * (1.0 + (ATPg / Ki_ATP) ** nH) ** (-1.0)
        d_F6Pg = torch.zeros_like(y[:, 0]) + (alg_16_Vg_pgi - alg_17_Vg_pfk)
        alg_18_Vg_glys = Vmax_glys * (G6Pg / (G6Pg + Km_G6P_glys)) * (1.0 - 1.0 / (1.0 + torch.exp(-aGLY_inh * (1.0 * (GLYg - GLY_inh)))))
        alg_20_unitstepSB2 = torch.where(t - (tend_GLY + to + to_GLY) >= 0.0, 1.0, 0.0)
        alg_22_deltaVt_GLY = 1.0 + stim * (delta_GLY * KO3 * (1.0 / (1.0 + torch.exp(1.0 * -sr_GLY * (t - (to + to_GLY))))) * (1.0 - alg_20_unitstepSB2))
        alg_24_Vg_glyp = Vmax_glyp * (GLYg / (GLYg + Km_GLY)) * alg_22_deltaVt_GLY
        d_G6Pg = torch.zeros_like(y[:, 0]) + (alg_15_Vg_hk + alg_24_Vg_glyp - (alg_16_Vg_pgi + alg_18_Vg_glys))
        d_GLYg = torch.zeros_like(y[:, 0]) + (alg_18_Vg_glys - alg_24_Vg_glyp)
        alg_26_Vce_GLC = Vm_ce_GLC * (GLCc / (GLCc + Km_ce_GLC) - GLCe / (GLCe + Km_ce_GLC))
        d_GLCe = torch.zeros_like(y[:, 0]) + (alg_26_Vce_GLC - (alg_13_Veg_GLC * (1.0 / Reg) + alg_4_V_en_GLC * (1.0 / Ren)))
        alg_8_Vne_LAC = Vmax_ne_LAC * (LACn / (LACn + Km_ne_LAC) - LACe / (LACe + Km_ne_LAC))
        alg_19_Vge_LAC = Vmax_ge_LAC * (LACg / (LACg + Km_ge_LAC) - LACe / (LACe + Km_ge_LAC))
        alg_27_Vec_LAC = Vm_ec_LAC * (LACe / (LACe + Km_ec_LAC) - LACc / (LACc + Km_ec_LAC))
        d_LACe = torch.zeros_like(y[:, 0]) + (alg_8_Vne_LAC * (1.0 / Ren) + alg_19_Vge_LAC * (1.0 / Reg) - alg_27_Vec_LAC)
        alg_11_Vg_leak_Na = Sm_g / Vg * (gg_NA / F) * (RT / F * torch.log(NAe / NAg) - Vm)
        alg_12_Vg_pump = Sm_g / Vg * kpump * ATPg * NAg * (1.0 + ATPg / Km_pump) ** (-1.0)
        alg_29_Veg_GLU = Vmax_eg_GLU * (GLUe / (GLUe + Km_GLU))
        d_NAg = torch.zeros_like(y[:, 0]) + (alg_11_Vg_leak_Na + 3.0 * alg_29_Veg_GLU - 3.0 * alg_12_Vg_pump)
        alg_28_Vg_gs = Vmax_g_gs * (GLUg / (GLUg + Km_GLU) * (ATPg / (ATPg + Km_ATP)))
        d_GLUg = torch.zeros_like(y[:, 0]) + (alg_29_Veg_GLU - alg_28_Vg_gs)
        alg_10_Vcn_O2 = PScapn / Vn * (Ko2 * (HbOP / O2c - 1.0) ** (-1.0 / nh_O2) - O2n)
        alg_25_Vcg_O2 = PScapg / Vg * (Ko2 * (HbOP / O2c - 1.0) ** (-1.0 / nh_O2_2) - O2g)
        alg_30_Fin_t = CBF0 + (stim * CBF0 * deltaf * (1.0 / (1.0 + torch.exp(1.0 * -sr * (t - (to + t1 - 3.0))))) - stim * CBF0 * deltaf * (1.0 / (1.0 + torch.exp(1.0 * -sr * (t - (to + tend + t1 + 3.0))))))
        alg_33_Vc_O2 = 2.0 * (alg_30_Fin_t / Vc) * (O2a - O2c)
        d_O2c = torch.zeros_like(y[:, 0]) + (alg_33_Vc_O2 - (alg_10_Vcn_O2 * (1.0 / Rcn) + alg_25_Vcg_O2 * (1.0 / Rcg)))
        alg_34_Vc_GLC = 2.0 * (alg_30_Fin_t / Vc) * (GLCa - GLCc)
        d_GLCc = torch.zeros_like(y[:, 0]) + (alg_34_Vc_GLC - (alg_26_Vce_GLC * (1.0 / Rce) + alg_14_Vcg_GLC * (1.0 / Rcg)))
        alg_21_Vgc_LAC = Vmax_gc_LAC * (LACg / (LACg + Km_gc_LAC) - LACc / (LACc + Km_gc_LAC))
        alg_35_Vc_LAC = 2.0 * (alg_30_Fin_t / Vc) * (LACa - LACc)
        d_LACc = torch.zeros_like(y[:, 0]) + (alg_35_Vc_LAC + (alg_27_Vec_LAC * (1.0 / Rce) + alg_21_Vgc_LAC * (1.0 / Rcg)))
        alg_36_Fout_t = CBF0_2 * (((Vv / Vv0) ** 2.0 + tv * (Vv / Vv0) ** (-0.5) * (alg_30_Fin_t / Vv0)) / (1.0 + CBF0_2 * tv * (Vv / Vv0) ** (-0.5) * (1.0 / Vv0)))
        d_Vv = torch.zeros_like(y[:, 0]) + (alg_30_Fin_t - alg_36_Fout_t)
        d_dHb = torch.zeros_like(y[:, 0]) + (alg_30_Fin_t * (O2a - O2c) - alg_36_Fout_t * (dHb / Vv))
        alg_0_Vn_leak_Na = Sm_n / Vn * (gn_NA / F) * (RT / F * torch.log(NAe / NAn) - Vm)
        alg_3_Vn_pump = Sm_n / Vn * kpump * ATPn * NAn * (1.0 + ATPn / Km_pump) ** (-1.0)
        alg_31_unitpulseSB = torch.where((t >= to) & (t <= to + tend), 1.0, 0.0)
        alg_37_v_stim = stim * (v1_n + v2_n * ((t - to) / t_n_stim) * torch.exp(-((t - to) * (alg_31_unitpulseSB / t_n_stim)))) * alg_31_unitpulseSB
        alg_39_Vn_stim = alg_37_v_stim
        d_NAn = torch.zeros_like(y[:, 0]) + (alg_0_Vn_leak_Na + alg_39_Vn_stim - 3.0 * alg_3_Vn_pump)
        alg_38_NADn = NADH_n_tot - NADHn
        alg_40_Vn_ldh = kfn_ldh * PYRn * NADHn - krn_ldh * LACn * alg_38_NADn
        d_LACn = torch.zeros_like(y[:, 0]) + (alg_40_Vn_ldh - alg_8_Vne_LAC)
        alg_42_Vn_stim_GLU = alg_39_Vn_stim * R_GLU_NA * KO2 * (GLUn / (GLUn + Km_GLU))
        d_GLUn = torch.zeros_like(y[:, 0]) + (alg_28_Vg_gs * (1.0 / Rng) - alg_42_Vn_stim_GLU)
        d_GLUe = torch.zeros_like(y[:, 0]) + (alg_42_Vn_stim_GLU * (1.0 / Ren) - alg_29_Veg_GLU * (1.0 / Reg))
        alg_41_NADg = NADH_g_tot - NADHg
        alg_43_Vg_ldh = kfg_ldh * PYRg * NADHg - krg_ldh * LACg * alg_41_NADg
        d_LACg = torch.zeros_like(y[:, 0]) + (alg_43_Vg_ldh - (alg_19_Vge_LAC + alg_21_Vgc_LAC))
        alg_46_ADPn = ATPn / 2.0 * (-qak + (qak ** 2.0 + 4.0 * qak * (ATPtot / ATPn - 1.0)) ** (1.0 / 2))
        alg_47_Vn_pgk = kn_pgk * GAPn * alg_46_ADPn * (alg_38_NADn / NADHn)
        d_GAPn = torch.zeros_like(y[:, 0]) + (2.0 * alg_7_Vn_pfk - alg_47_Vn_pgk)
        alg_49_Vn_pk = kn_pk * PEPn * alg_46_ADPn
        d_PEPn = torch.zeros_like(y[:, 0]) + (alg_47_Vn_pgk - alg_49_Vn_pk)
        alg_50_Vn_mito = Vmax_n_mito * (O2n / (O2n + Km_O2)) * (alg_46_ADPn / (alg_46_ADPn + Km_ADP)) * (PYRn / (PYRn + Km_PYR)) * (1.0 - 1.0 / (1.0 + torch.exp(-aATP_mito * (1.0 * (ATPn / alg_46_ADPn - 1.0 * rATP_mito)))))
        d_PYRn = torch.zeros_like(y[:, 0]) + (alg_49_Vn_pk - (alg_40_Vn_ldh + alg_50_Vn_mito))
        d_NADHn = torch.zeros_like(y[:, 0]) + (alg_47_Vn_pgk - (alg_40_Vn_ldh + alg_50_Vn_mito))
        d_O2n = torch.zeros_like(y[:, 0]) + (alg_10_Vcn_O2 - NAero * alg_50_Vn_mito)
        alg_44_CRn = PCrn_tot - PCrn
        alg_51_Vn_ck = kfn_ck * PCrn * alg_46_ADPn - krn_ck * alg_44_CRn * ATPn
        d_PCrn = torch.zeros_like(y[:, 0]) + (-alg_51_Vn_ck)
        alg_9_Vn_ATPase = Vmax_n_ATPase * (ATPn / (ATPn + 0.001))
        alg_53_u_n = qak ** 2.0 + 4.0 * qak * (ATPtot / ATPn - 1.0)
        alg_55_dAMP_dATPn = qak / 2.0 + qak * (ATPtot / (ATPn * alg_53_u_n ** (1.0 / 2))) - (1.0 + 0.5 * alg_53_u_n ** (1.0 / 2))
        d_ATPn = torch.zeros_like(y[:, 0]) + ((alg_47_Vn_pgk + alg_49_Vn_pk + nOP * alg_50_Vn_mito + alg_51_Vn_ck - (alg_5_Vn_hk + alg_7_Vn_pfk + alg_9_Vn_ATPase + alg_3_Vn_pump)) * (1.0 - alg_55_dAMP_dATPn) ** (-1.0))
        alg_54_ADPg = ATPg / 2.0 * (-qak + (qak ** 2.0 + 4.0 * qak * (ATPtot / ATPg - 1.0)) ** (1.0 / 2))
        alg_56_Vg_pgk = kg_pgk * GAPg * alg_54_ADPg * (alg_41_NADg / NADHg)
        d_GAPg = torch.zeros_like(y[:, 0]) + (2.0 * alg_17_Vg_pfk - alg_56_Vg_pgk)
        alg_58_Vg_pk = kg_pk * PEPg * alg_54_ADPg
        d_PEPg = torch.zeros_like(y[:, 0]) + (alg_56_Vg_pgk - alg_58_Vg_pk)
        alg_59_Vg_mito = Vmax_g_mito * (O2g / (O2g + Km_O2)) * (alg_54_ADPg / (alg_54_ADPg + Km_ADP)) * (PYRg / (PYRg + Km_PYR)) * (1.0 - 1.0 / (1.0 + torch.exp(1.0 * -aATP_mito * (ATPg / alg_54_ADPg - 1.0 * rATP_mito))))
        d_PYRg = torch.zeros_like(y[:, 0]) + (alg_58_Vg_pk - (alg_43_Vg_ldh + alg_59_Vg_mito))
        d_NADHg = torch.zeros_like(y[:, 0]) + (alg_56_Vg_pgk - (alg_43_Vg_ldh + alg_59_Vg_mito))
        d_O2g = torch.zeros_like(y[:, 0]) + (alg_25_Vcg_O2 - NAero * alg_59_Vg_mito)
        alg_45_CRg = PCrg_tot - PCrg
        alg_60_Vg_ck = kfg_ck * PCrg * alg_54_ADPg - krg_ck * alg_45_CRg * ATPg
        d_PCrg = torch.zeros_like(y[:, 0]) + (-alg_60_Vg_ck)
        alg_52_Vnc_CO2 = 3.0 * alg_50_Vn_mito
        alg_32_Vc_CO2 = 2.0 * (alg_30_Fin_t / Vc) * (CO2c - CO2a)
        alg_61_Vgc_CO2 = 3.0 * alg_59_Vg_mito
        d_CO2c = torch.zeros_like(y[:, 0]) + (alg_52_Vnc_CO2 * (1.0 / Rcn) + alg_61_Vgc_CO2 * (1.0 / Rcg) - alg_32_Vc_CO2)
        alg_23_Vg_ATPase = Vmax_g_ATPase * (ATPg / (ATPg + 0.001))
        alg_62_u_g = qak ** 2.0 + 4.0 * qak * (ATPtot / ATPg - 1.0)
        alg_63_dAMP_dATPg = qak / 2.0 + qak * (ATPtot / (ATPg * alg_62_u_g ** (1.0 / 2))) - (1.0 + 0.5 * alg_62_u_g ** (1.0 / 2))
        d_ATPg = torch.zeros_like(y[:, 0]) + ((alg_56_Vg_pgk + alg_58_Vg_pk + nOP * alg_59_Vg_mito + alg_60_Vg_ck - (alg_15_Vg_hk + alg_17_Vg_pfk + alg_23_Vg_ATPase + alg_12_Vg_pump + alg_28_Vg_gs)) * (1.0 - alg_63_dAMP_dATPg) ** (-1.0))

        return torch.stack(
            [d_NAn, d_GLCn, d_G6Pn, d_F6Pn, d_GAPn, d_PEPn, d_PYRn, d_LACn, d_NADHn, d_ATPn, d_PCrn, d_O2n, d_GLUn, d_NAg, d_GLCg, d_G6Pg, d_F6Pg, d_GAPg, d_PEPg, d_PYRg, d_LACg, d_NADHg, d_ATPg, d_PCrg, d_O2g, d_GLYg, d_GLUg, d_GLCe, d_LACe, d_GLUe, d_O2c, d_GLCc, d_LACc, d_CO2c, d_Vv, d_dHb],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
