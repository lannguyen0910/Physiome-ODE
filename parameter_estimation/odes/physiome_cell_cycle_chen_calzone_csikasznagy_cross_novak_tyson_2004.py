import torch
import torch.nn as nn
import math


class CellCycleChenCalzoneCsikasznagyCrossNovakTyson2004ODE(nn.Module):
    """
    PyTorch translation of cell_cycle/chen_calzone_csikasznagy_cross_novak_tyson_2004/model.py.
    Dynamic states: 36; independent constants: 136; derived constants used by rates: 0.
    Original constant slots: 136; algebraic slots: 30.
    """
    source_file = 'cell_cycle/chen_calzone_csikasznagy_cross_novak_tyson_2004/model.py'
    source_state_dim = 36
    source_constant_dim = 136
    algebraic_dim = 30
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
        self.parameter_dim = 136

        self.parameters = (
            parameters
            if parameters is not None
            else {'kg': 0.007702, 'ks_n2': 0, 'ks_n2_2': 0.15, 'kd_n2': 0.12, 'ks_b5': 0.0008, 'ks_b5_2': 0.005, 'kdi_f5': 0.01, 'kdi_b5': 0.06, 'kas_b5': 50, 'kas_f5': 0.01, 'kd3_f6': 1, 'kd3_c1': 1, 'ks_b2': 0.001, 'ks_b2_2': 0.04, 'kdi_b2': 0.05, 'kdi_f2': 0.5, 'kas_b2': 50, 'kas_f2': 15, 'ks_c1': 0.012, 'ks_c1_2': 0.12, 'kpp_c1': 4, 'ks_f6': 0.024, 'ks_f6_2': 0.12, 'ks_f6_3': 0.004, 'kpp_f6': 4, 'ks_pds': 0, 'ks1_pds': 0.03, 'ks2_pds': 0.055, 'kdi_esp': 0.5, 'kas_esp': 50, 'ks_ori': 2, 'kd_ori': 0.06, 'epsilon_ori_b5': 0.9, 'epsilon_ori_b2': 0.45, 'ks_bud': 0.2, 'kd_bud': 0.06, 'epsilon_bud_n2': 0.25, 'epsilon_bud_n3': 0.05, 'epsilon_bud_b5': 1, 'ks_spn': 0.1, 'kd_spn': 0.06, 'Jspn': 0.14, 'Ji_sbf': 0.01, 'Ja_sbf': 0.01, 'Ji_mcm': 0.1, 'Ja_mcm': 0.1, 'ki_mcm': 0.15, 'ka_mcm': 1, 'C0': 0.4, 'Dn3': 1, 'Jn3': 6, 'B0': 0.054, 'ks_swi': 0.005, 'ks_swi_2': 0.08, 'kd_swi': 0.08, 'ka_swi': 2, 'ki_swi': 0.05, 'ka_apc': 0.1, 'ki_apc': 0.15, 'Ja_apc': 0.1, 'Ji_apc': 0.1, 'ks_20': 0.006, 'ks_20_2': 0.6, 'kd_20': 0.3, 'ka_20': 0.05, 'ka_20_2': 0.2, 'ks_cdh': 0.01, 'kd_cdh': 0.01, 'Ja_cdh': 0.03, 'Ji_cdh': 0.03, 'Ja_tem': 0.1, 'Ji_tem': 0.1, 'Tem1_T': 1, 'ka_15': 0.002, 'ka_15_2': 1, 'ka_15_3': 0.001, 'ki_15': 0.5, 'Cdc15_T': 1, 'ks_14': 0.2, 'kd_14': 0.1, 'kd_net': 0.03, 'kdi_rent': 1, 'kdi_rentp': 2, 'kas_rent': 200, 'kas_rentp': 1, 'ks_net': 0.084, 'Esp1_T': 1, 'kd_b5': 0.01, 'kd_b5_2': 0.16, 'kd_b2': 0.003, 'kd_b2_2': 0.4, 'kd_b2p': 0.15, 'ka_sbf': 0.38, 'epsilon_sbf_n2': 2, 'epsilon_sbf_n3': 10, 'epsilon_sbf_b5': 2, 'ki_sbf': 0.6, 'ki_sbf_2': 8, 'kd1_c1': 0.01, 'kd2_c1': 1, 'Jd2_c1': 0.05, 'epsilon_c1_n2': 0.06, 'epsilon_c1_n3': 0.3, 'epsilon_c1_k2': 0.03, 'epsilon_c1_b5': 0.1, 'epsilon_c1_b2': 0.45, 'Jd2_f6': 0.05, 'kd1_f6': 0.01, 'kd2_f6': 1, 'epsilon_f6_n2': 0.06, 'epsilon_f6_n3': 0.3, 'epsilon_f6_k2': 0.03, 'epsilon_f6_b5': 0.1, 'epsilon_f6_b2': 0.55, 'ka_cdh': 0.01, 'ka_cdh_2': 0.8, 'ki_cdh': 0.001, 'ki_cdh_2': 0.08, 'epsilon_cdh_n2': 0.4, 'epsilon_cdh_n3': 0.25, 'epsilon_cdh_b5': 8, 'epsilon_cdh_b2': 1.2, 'kpp_net': 0.05, 'kpp_net_2': 3, 'kkp_net': 0.01, 'kkp_net_2': 0.6, 'ks_ppx': 0.1, 'kd_ppx': 0.17, 'kd_ppx_2': 2, 'Jpds': 0.04, 'J20_ppx': 0.15, 'kd1_pds': 0.01, 'kd2_pds': 0.2, 'kd3_pds': 0.04, 'Kez': 0.3, 'Kez2': 0.2}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'mass': 1.206, 'Cln2': 0.0652, 'Clb5': 0.0518, 'C5P': 0.0069, 'C5': 0.0701, 'F5P': 7.9e-06, 'F5': 7.2e-05, 'Sic1': 0.0229, 'Cdc6': 0.1076, 'Clb2': 0.1469, 'C2P': 0.024, 'C2': 0.2384, 'F2P': 0.0274, 'F2': 0.2361, 'Swi5': 0.9562, 'Cdc14': 0.4683, 'Sic1P': 0.0064, 'Cdc6P': 0.0155, 'Pds1': 0.0256, 'Esp1': 0.3013, 'ORI': 0.0009, 'BUD': 0.0085, 'SPN': 0.0305, 'Swi5_T': 0.9765, 'APC_P': 0.1015, 'Cdc20_T': 1.9163, 'Cdc20_A': 0.4443, 'Cdh1_T': 1, 'Cdh1': 0.9305, 'Tem1': 0.9039, 'Cdc15': 0.6565, 'Cdc14_T': 2, 'RENT': 1.0495, 'Net1': 0.0186, 'Net1_T': 2.8, 'PPX': 0.1232}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['kg', 'ks_n2', 'ks_n2_2', 'kd_n2', 'ks_b5', 'ks_b5_2', 'kdi_f5', 'kdi_b5', 'kas_b5', 'kas_f5', 'kd3_f6', 'kd3_c1', 'ks_b2', 'ks_b2_2', 'kdi_b2', 'kdi_f2', 'kas_b2', 'kas_f2', 'ks_c1', 'ks_c1_2', 'kpp_c1', 'ks_f6', 'ks_f6_2', 'ks_f6_3', 'kpp_f6', 'ks_pds', 'ks1_pds', 'ks2_pds', 'kdi_esp', 'kas_esp', 'ks_ori', 'kd_ori', 'epsilon_ori_b5', 'epsilon_ori_b2', 'ks_bud', 'kd_bud', 'epsilon_bud_n2', 'epsilon_bud_n3', 'epsilon_bud_b5', 'ks_spn', 'kd_spn', 'Jspn', 'Ji_sbf', 'Ja_sbf', 'Ji_mcm', 'Ja_mcm', 'ki_mcm', 'ka_mcm', 'C0', 'Dn3', 'Jn3', 'B0', 'ks_swi', 'ks_swi_2', 'kd_swi', 'ka_swi', 'ki_swi', 'ka_apc', 'ki_apc', 'Ja_apc', 'Ji_apc', 'ks_20', 'ks_20_2', 'kd_20', 'ka_20', 'ka_20_2', 'ks_cdh', 'kd_cdh', 'Ja_cdh', 'Ji_cdh', 'Ja_tem', 'Ji_tem', 'Tem1_T', 'ka_15', 'ka_15_2', 'ka_15_3', 'ki_15', 'Cdc15_T', 'ks_14', 'kd_14', 'kd_net', 'kdi_rent', 'kdi_rentp', 'kas_rent', 'kas_rentp', 'ks_net', 'Esp1_T', 'kd_b5', 'kd_b5_2', 'kd_b2', 'kd_b2_2', 'kd_b2p', 'ka_sbf', 'epsilon_sbf_n2', 'epsilon_sbf_n3', 'epsilon_sbf_b5', 'ki_sbf', 'ki_sbf_2', 'kd1_c1', 'kd2_c1', 'Jd2_c1', 'epsilon_c1_n2', 'epsilon_c1_n3', 'epsilon_c1_k2', 'epsilon_c1_b5', 'epsilon_c1_b2', 'Jd2_f6', 'kd1_f6', 'kd2_f6', 'epsilon_f6_n2', 'epsilon_f6_n3', 'epsilon_f6_k2', 'epsilon_f6_b5', 'epsilon_f6_b2', 'ka_cdh', 'ka_cdh_2', 'ki_cdh', 'ki_cdh_2', 'epsilon_cdh_n2', 'epsilon_cdh_n3', 'epsilon_cdh_b5', 'epsilon_cdh_b2', 'kpp_net', 'kpp_net_2', 'kkp_net', 'kkp_net_2', 'ks_ppx', 'kd_ppx', 'kd_ppx_2', 'Jpds', 'J20_ppx', 'kd1_pds', 'kd2_pds', 'kd3_pds', 'Kez', 'Kez2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['mass', 'Cln2', 'Clb5', 'C5P', 'C5', 'F5P', 'F5', 'Sic1', 'Cdc6', 'Clb2', 'C2P', 'C2', 'F2P', 'F2', 'Swi5', 'Cdc14', 'Sic1P', 'Cdc6P', 'Pds1', 'Esp1', 'ORI', 'BUD', 'SPN', 'Swi5_T', 'APC_P', 'Cdc20_T', 'Cdc20_A', 'Cdh1_T', 'Cdh1', 'Tem1', 'Cdc15', 'Cdc14_T', 'RENT', 'Net1', 'Net1_T', 'PPX']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (36)
        mass = y[:, 0]
        Cln2 = y[:, 1]
        Clb5 = y[:, 2]
        C5P = y[:, 3]
        C5 = y[:, 4]
        F5P = y[:, 5]
        F5 = y[:, 6]
        Sic1 = y[:, 7]
        Cdc6 = y[:, 8]
        Clb2 = y[:, 9]
        C2P = y[:, 10]
        C2 = y[:, 11]
        F2P = y[:, 12]
        F2 = y[:, 13]
        Swi5 = y[:, 14]
        Cdc14 = y[:, 15]
        Sic1P = y[:, 16]
        Cdc6P = y[:, 17]
        Pds1 = y[:, 18]
        Esp1 = y[:, 19]
        ORI = y[:, 20]
        BUD = y[:, 21]
        SPN = y[:, 22]
        Swi5_T = y[:, 23]
        APC_P = y[:, 24]
        Cdc20_T = y[:, 25]
        Cdc20_A = y[:, 26]
        Cdh1_T = y[:, 27]
        Cdh1 = y[:, 28]
        Tem1 = y[:, 29]
        Cdc15 = y[:, 30]
        Cdc14_T = y[:, 31]
        RENT = y[:, 32]
        Net1 = y[:, 33]
        Net1_T = y[:, 34]
        PPX = y[:, 35]

        # Independent parameters (source constant order)
        kg = theta[:, 0]  # constants[0]
        ks_n2 = theta[:, 1]  # constants[1]
        ks_n2_2 = theta[:, 2]  # constants[2]
        kd_n2 = theta[:, 3]  # constants[3]
        ks_b5 = theta[:, 4]  # constants[4]
        ks_b5_2 = theta[:, 5]  # constants[5]
        kdi_f5 = theta[:, 6]  # constants[6]
        kdi_b5 = theta[:, 7]  # constants[7]
        kas_b5 = theta[:, 8]  # constants[8]
        kas_f5 = theta[:, 9]  # constants[9]
        kd3_f6 = theta[:, 10]  # constants[10]
        kd3_c1 = theta[:, 11]  # constants[11]
        ks_b2 = theta[:, 12]  # constants[12]
        ks_b2_2 = theta[:, 13]  # constants[13]
        kdi_b2 = theta[:, 14]  # constants[14]
        kdi_f2 = theta[:, 15]  # constants[15]
        kas_b2 = theta[:, 16]  # constants[16]
        kas_f2 = theta[:, 17]  # constants[17]
        ks_c1 = theta[:, 18]  # constants[18]
        ks_c1_2 = theta[:, 19]  # constants[19]
        kpp_c1 = theta[:, 20]  # constants[20]
        ks_f6 = theta[:, 21]  # constants[21]
        ks_f6_2 = theta[:, 22]  # constants[22]
        ks_f6_3 = theta[:, 23]  # constants[23]
        kpp_f6 = theta[:, 24]  # constants[24]
        ks_pds = theta[:, 25]  # constants[25]
        ks1_pds = theta[:, 26]  # constants[26]
        ks2_pds = theta[:, 27]  # constants[27]
        kdi_esp = theta[:, 28]  # constants[28]
        kas_esp = theta[:, 29]  # constants[29]
        ks_ori = theta[:, 30]  # constants[30]
        kd_ori = theta[:, 31]  # constants[31]
        epsilon_ori_b5 = theta[:, 32]  # constants[32]
        epsilon_ori_b2 = theta[:, 33]  # constants[33]
        ks_bud = theta[:, 34]  # constants[34]
        kd_bud = theta[:, 35]  # constants[35]
        epsilon_bud_n2 = theta[:, 36]  # constants[36]
        epsilon_bud_n3 = theta[:, 37]  # constants[37]
        epsilon_bud_b5 = theta[:, 38]  # constants[38]
        ks_spn = theta[:, 39]  # constants[39]
        kd_spn = theta[:, 40]  # constants[40]
        Jspn = theta[:, 41]  # constants[41]
        Ji_sbf = theta[:, 42]  # constants[42]
        Ja_sbf = theta[:, 43]  # constants[43]
        Ji_mcm = theta[:, 44]  # constants[44]
        Ja_mcm = theta[:, 45]  # constants[45]
        ki_mcm = theta[:, 46]  # constants[46]
        ka_mcm = theta[:, 47]  # constants[47]
        C0 = theta[:, 48]  # constants[48]
        Dn3 = theta[:, 49]  # constants[49]
        Jn3 = theta[:, 50]  # constants[50]
        B0 = theta[:, 51]  # constants[51]
        ks_swi = theta[:, 52]  # constants[52]
        ks_swi_2 = theta[:, 53]  # constants[53]
        kd_swi = theta[:, 54]  # constants[54]
        ka_swi = theta[:, 55]  # constants[55]
        ki_swi = theta[:, 56]  # constants[56]
        ka_apc = theta[:, 57]  # constants[57]
        ki_apc = theta[:, 58]  # constants[58]
        Ja_apc = theta[:, 59]  # constants[59]
        Ji_apc = theta[:, 60]  # constants[60]
        ks_20 = theta[:, 61]  # constants[61]
        ks_20_2 = theta[:, 62]  # constants[62]
        kd_20 = theta[:, 63]  # constants[63]
        ka_20 = theta[:, 64]  # constants[64]
        ka_20_2 = theta[:, 65]  # constants[65]
        ks_cdh = theta[:, 66]  # constants[66]
        kd_cdh = theta[:, 67]  # constants[67]
        Ja_cdh = theta[:, 68]  # constants[68]
        Ji_cdh = theta[:, 69]  # constants[69]
        Ja_tem = theta[:, 70]  # constants[70]
        Ji_tem = theta[:, 71]  # constants[71]
        Tem1_T = theta[:, 72]  # constants[72]
        ka_15 = theta[:, 73]  # constants[73]
        ka_15_2 = theta[:, 74]  # constants[74]
        ka_15_3 = theta[:, 75]  # constants[75]
        ki_15 = theta[:, 76]  # constants[76]
        Cdc15_T = theta[:, 77]  # constants[77]
        ks_14 = theta[:, 78]  # constants[78]
        kd_14 = theta[:, 79]  # constants[79]
        kd_net = theta[:, 80]  # constants[80]
        kdi_rent = theta[:, 81]  # constants[81]
        kdi_rentp = theta[:, 82]  # constants[82]
        kas_rent = theta[:, 83]  # constants[83]
        kas_rentp = theta[:, 84]  # constants[84]
        ks_net = theta[:, 85]  # constants[85]
        Esp1_T = theta[:, 86]  # constants[86]
        kd_b5 = theta[:, 87]  # constants[87]
        kd_b5_2 = theta[:, 88]  # constants[88]
        kd_b2 = theta[:, 89]  # constants[89]
        kd_b2_2 = theta[:, 90]  # constants[90]
        kd_b2p = theta[:, 91]  # constants[91]
        ka_sbf = theta[:, 92]  # constants[92]
        epsilon_sbf_n2 = theta[:, 93]  # constants[93]
        epsilon_sbf_n3 = theta[:, 94]  # constants[94]
        epsilon_sbf_b5 = theta[:, 95]  # constants[95]
        ki_sbf = theta[:, 96]  # constants[96]
        ki_sbf_2 = theta[:, 97]  # constants[97]
        kd1_c1 = theta[:, 98]  # constants[98]
        kd2_c1 = theta[:, 99]  # constants[99]
        Jd2_c1 = theta[:, 100]  # constants[100]
        epsilon_c1_n2 = theta[:, 101]  # constants[101]
        epsilon_c1_n3 = theta[:, 102]  # constants[102]
        epsilon_c1_k2 = theta[:, 103]  # constants[103]
        epsilon_c1_b5 = theta[:, 104]  # constants[104]
        epsilon_c1_b2 = theta[:, 105]  # constants[105]
        Jd2_f6 = theta[:, 106]  # constants[106]
        kd1_f6 = theta[:, 107]  # constants[107]
        kd2_f6 = theta[:, 108]  # constants[108]
        epsilon_f6_n2 = theta[:, 109]  # constants[109]
        epsilon_f6_n3 = theta[:, 110]  # constants[110]
        epsilon_f6_k2 = theta[:, 111]  # constants[111]
        epsilon_f6_b5 = theta[:, 112]  # constants[112]
        epsilon_f6_b2 = theta[:, 113]  # constants[113]
        ka_cdh = theta[:, 114]  # constants[114]
        ka_cdh_2 = theta[:, 115]  # constants[115]
        ki_cdh = theta[:, 116]  # constants[116]
        ki_cdh_2 = theta[:, 117]  # constants[117]
        epsilon_cdh_n2 = theta[:, 118]  # constants[118]
        epsilon_cdh_n3 = theta[:, 119]  # constants[119]
        epsilon_cdh_b5 = theta[:, 120]  # constants[120]
        epsilon_cdh_b2 = theta[:, 121]  # constants[121]
        kpp_net = theta[:, 122]  # constants[122]
        kpp_net_2 = theta[:, 123]  # constants[123]
        kkp_net = theta[:, 124]  # constants[124]
        kkp_net_2 = theta[:, 125]  # constants[125]
        ks_ppx = theta[:, 126]  # constants[126]
        kd_ppx = theta[:, 127]  # constants[127]
        kd_ppx_2 = theta[:, 128]  # constants[128]
        Jpds = theta[:, 129]  # constants[129]
        J20_ppx = theta[:, 130]  # constants[130]
        kd1_pds = theta[:, 131]  # constants[131]
        kd2_pds = theta[:, 132]  # constants[132]
        kd3_pds = theta[:, 133]  # constants[133]
        Kez = theta[:, 134]  # constants[134]
        Kez2 = theta[:, 135]  # constants[135]

        # Initialize rates and algebraic variables, as in computeRates()
        d_mass = torch.zeros_like(y[:, 0])
        d_Cln2 = torch.zeros_like(y[:, 0])
        d_Clb5 = torch.zeros_like(y[:, 0])
        d_C5P = torch.zeros_like(y[:, 0])
        d_C5 = torch.zeros_like(y[:, 0])
        d_F5P = torch.zeros_like(y[:, 0])
        d_F5 = torch.zeros_like(y[:, 0])
        d_Sic1 = torch.zeros_like(y[:, 0])
        d_Cdc6 = torch.zeros_like(y[:, 0])
        d_Clb2 = torch.zeros_like(y[:, 0])
        d_C2P = torch.zeros_like(y[:, 0])
        d_C2 = torch.zeros_like(y[:, 0])
        d_F2P = torch.zeros_like(y[:, 0])
        d_F2 = torch.zeros_like(y[:, 0])
        d_Swi5 = torch.zeros_like(y[:, 0])
        d_Cdc14 = torch.zeros_like(y[:, 0])
        d_Sic1P = torch.zeros_like(y[:, 0])
        d_Cdc6P = torch.zeros_like(y[:, 0])
        d_Pds1 = torch.zeros_like(y[:, 0])
        d_Esp1 = torch.zeros_like(y[:, 0])
        d_ORI = torch.zeros_like(y[:, 0])
        d_BUD = torch.zeros_like(y[:, 0])
        d_SPN = torch.zeros_like(y[:, 0])
        d_Swi5_T = torch.zeros_like(y[:, 0])
        d_APC_P = torch.zeros_like(y[:, 0])
        d_Cdc20_T = torch.zeros_like(y[:, 0])
        d_Cdc20_A = torch.zeros_like(y[:, 0])
        d_Cdh1_T = torch.zeros_like(y[:, 0])
        d_Cdh1 = torch.zeros_like(y[:, 0])
        d_Tem1 = torch.zeros_like(y[:, 0])
        d_Cdc15 = torch.zeros_like(y[:, 0])
        d_Cdc14_T = torch.zeros_like(y[:, 0])
        d_RENT = torch.zeros_like(y[:, 0])
        d_Net1 = torch.zeros_like(y[:, 0])
        d_Net1_T = torch.zeros_like(y[:, 0])
        d_PPX = torch.zeros_like(y[:, 0])
        alg_0_G_mcm = torch.zeros_like(y[:, 0])
        alg_3_kmad2 = torch.zeros_like(y[:, 0])
        alg_4_kbub2 = torch.zeros_like(y[:, 0])
        alg_5_RENTP = torch.zeros_like(y[:, 0])
        alg_6_Vd_ppx = torch.zeros_like(y[:, 0])
        alg_7_Mcm1 = torch.zeros_like(y[:, 0])
        alg_8_klte1 = torch.zeros_like(y[:, 0])
        alg_9_Net1P = torch.zeros_like(y[:, 0])
        alg_10_Cln3 = torch.zeros_like(y[:, 0])
        alg_11_Vpp_net = torch.zeros_like(y[:, 0])
        alg_12_Bck2 = torch.zeros_like(y[:, 0])
        alg_13_Va_cdh = torch.zeros_like(y[:, 0])
        alg_14_Vkp_net = torch.zeros_like(y[:, 0])
        alg_15_Sic1_T = torch.zeros_like(y[:, 0])
        alg_16_Vi_cdh = torch.zeros_like(y[:, 0])
        alg_17_Cdc6_T = torch.zeros_like(y[:, 0])
        alg_18_PE = torch.zeros_like(y[:, 0])
        alg_20_Vd_b5 = torch.zeros_like(y[:, 0])
        alg_21_Vd_b2 = torch.zeros_like(y[:, 0])
        alg_22_Va_sbf = torch.zeros_like(y[:, 0])
        alg_23_Vkp_c1 = torch.zeros_like(y[:, 0])
        alg_24_Vi_sbf = torch.zeros_like(y[:, 0])
        alg_25_G_sbf = torch.zeros_like(y[:, 0])
        alg_26_MBF = torch.zeros_like(y[:, 0])
        alg_27_SBF = torch.zeros_like(y[:, 0])
        alg_28_Vkp_f6 = torch.zeros_like(y[:, 0])
        alg_29_Vd_pds = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_mass = torch.zeros_like(y[:, 0]) + (mass * kg)
        d_ORI = torch.zeros_like(y[:, 0]) + (ks_ori * (epsilon_ori_b5 * Clb5 + epsilon_ori_b2 * Clb2) - kd_ori * ORI)
        d_SPN = torch.zeros_like(y[:, 0]) + (ks_spn * Clb2 / (Jspn + Clb2) - kd_spn * SPN)
        d_APC_P = torch.zeros_like(y[:, 0]) + (ka_apc * Clb2 * (1.0 - APC_P) / (Ja_apc + 1.0 - APC_P) - ki_apc * APC_P / (Ji_apc + APC_P))
        d_Cdh1_T = torch.zeros_like(y[:, 0]) + (ks_cdh - kd_cdh * Cdh1_T)
        d_Cdc15 = torch.zeros_like(y[:, 0]) + ((ka_15 * (Tem1_T - Tem1) + ka_15_2 * Tem1 + ka_15_3 * Cdc14) * (Cdc15_T - Cdc15) - ki_15 * Cdc15)
        d_Cdc14_T = torch.zeros_like(y[:, 0]) + (ks_14 - kd_14 * Cdc14_T)
        d_Net1_T = torch.zeros_like(y[:, 0]) + (ks_net - kd_net * Net1_T)
        alg_3_kmad2 = torch.where((ORI > 1.0) & (SPN < 1.0), 8.0, 0.01)
        d_Cdc20_A = torch.zeros_like(y[:, 0]) + ((ka_20 + ka_20_2 * APC_P) * (Cdc20_T - Cdc20_A) - (alg_3_kmad2 + kd_20) * Cdc20_A)
        alg_6_Vd_ppx = kd_ppx + kd_ppx_2 * (J20_ppx + Cdc20_A) * Jpds / (Jpds + Pds1)
        d_PPX = torch.zeros_like(y[:, 0]) + (ks_ppx - alg_6_Vd_ppx * PPX)
        alg_0_G_mcm = 2.0 * Ji_mcm * ka_mcm * Clb2 / (ki_mcm + Ja_mcm * ki_mcm + Ji_mcm * ka_mcm * Clb2 + ((ki_mcm + Ja_mcm * ki_mcm + Ji_mcm * ka_mcm * Clb2 - ka_mcm * Clb2) ** 2.0 - 4.0 * (ki_mcm - ka_mcm * Clb2) * Ji_mcm * ka_mcm * Clb2) ** (1.0 / 2) - ka_mcm * Clb2)
        alg_7_Mcm1 = alg_0_G_mcm
        d_Swi5_T = torch.zeros_like(y[:, 0]) + (ks_swi + ks_swi_2 * alg_7_Mcm1 - kd_swi * Swi5_T)
        d_Swi5 = torch.zeros_like(y[:, 0]) + (ks_swi + ks_swi_2 * alg_7_Mcm1 + ka_swi * Cdc14 * (Swi5_T - Swi5) - (kd_swi + ki_swi * Clb2) * Swi5)
        d_Cdc20_T = torch.zeros_like(y[:, 0]) + (ks_20 + ks_20_2 * alg_7_Mcm1 - kd_20 * Cdc20_T)
        alg_4_kbub2 = torch.where((ORI > 1.0) & (SPN < 1.0), 1.0, 0.2)
        alg_8_klte1 = torch.where((SPN > 1.0) & (Clb2 > Kez), 1.0, 0.1)
        d_Tem1 = torch.zeros_like(y[:, 0]) + (alg_8_klte1 * (Tem1_T - Tem1) / (Ja_tem + Tem1_T - Tem1) - alg_4_kbub2 * Tem1 / (Ji_tem + Tem1))
        alg_5_RENTP = Cdc14_T - (RENT + Cdc14)
        alg_9_Net1P = Net1_T + Cdc14 - (Net1 + Cdc14_T)
        d_Cdc14 = torch.zeros_like(y[:, 0]) + (ks_14 + kd_net * (RENT + alg_5_RENTP) + kdi_rent * RENT + kdi_rentp * alg_5_RENTP - (kd_14 * Cdc14 + (kas_rent * Net1 + kas_rentp * alg_9_Net1P) * Cdc14))
        alg_10_Cln3 = C0 * Dn3 * mass / (Jn3 + Dn3 * mass)
        d_BUD = torch.zeros_like(y[:, 0]) + (ks_bud * (epsilon_bud_n2 * Cln2 + epsilon_bud_n3 * alg_10_Cln3 + epsilon_bud_b5 * Clb5) - kd_bud * BUD)
        alg_11_Vpp_net = kpp_net + kpp_net_2 * PPX
        alg_14_Vkp_net = (kkp_net + kkp_net_2 * Cdc15) * mass
        d_Net1 = torch.zeros_like(y[:, 0]) + (ks_net + kd_14 * RENT + kdi_rent * RENT + alg_11_Vpp_net * alg_9_Net1P - (kd_net * Net1 + kas_rent * Cdc14 * Net1 + alg_14_Vkp_net * Net1))
        d_RENT = torch.zeros_like(y[:, 0]) + (kas_rent * Cdc14 * Net1 + alg_11_Vpp_net * alg_5_RENTP - ((kd_14 + kd_net) * RENT + kdi_rent * RENT + alg_14_Vkp_net * RENT))
        alg_13_Va_cdh = ka_cdh + ka_cdh_2 * Cdc14
        alg_16_Vi_cdh = ki_cdh + ki_cdh_2 * (epsilon_cdh_n3 * alg_10_Cln3 + epsilon_cdh_n2 * Cln2 + epsilon_cdh_b5 * Clb5 + epsilon_cdh_b2 * Clb2)
        d_Cdh1 = torch.zeros_like(y[:, 0]) + (ks_cdh + alg_13_Va_cdh * (Cdh1_T - Cdh1) / (Ja_cdh + Cdh1_T - Cdh1) - (kd_cdh * Cdh1 + alg_16_Vi_cdh * Cdh1 / (Ji_cdh + Cdh1)))
        alg_21_Vd_b2 = kd_b2 + kd_b2_2 * Cdh1 + kd_b2p * Cdc20_A
        d_Clb2 = torch.zeros_like(y[:, 0]) + ((ks_b2 + ks_b2_2 * alg_7_Mcm1) * mass + kd3_c1 * C2P + kdi_b2 * C2 + kd3_f6 * F2P + kdi_f2 * F2 - (alg_21_Vd_b2 + kas_b2 * Sic1 + kas_f2 * Cdc6) * Clb2)
        alg_20_Vd_b5 = kd_b5 + kd_b5_2 * Cdc20_A
        alg_12_Bck2 = B0 * mass
        alg_15_Sic1_T = Sic1 + Sic1P + C2 + C2P + C5 + C5P
        alg_23_Vkp_c1 = kd1_c1 + kd2_c1 * (epsilon_c1_n3 * alg_10_Cln3 + epsilon_c1_k2 * alg_12_Bck2 + epsilon_c1_n2 * Cln2 + epsilon_c1_b5 * Clb5 + epsilon_c1_b2 * Clb2) / (Jd2_c1 + alg_15_Sic1_T)
        d_Sic1 = torch.zeros_like(y[:, 0]) + (ks_c1 + ks_c1_2 * Swi5 + (alg_21_Vd_b2 + kdi_b2) * C2 + (alg_20_Vd_b5 + kdi_b5) * C5 + kpp_c1 * Cdc14 * Sic1P - (kas_b2 * Clb2 + kas_b5 * Clb5 + alg_23_Vkp_c1) * Sic1)
        d_Sic1P = torch.zeros_like(y[:, 0]) + (alg_23_Vkp_c1 * Sic1 - (kpp_c1 * Cdc14 + kd3_c1) * Sic1P + alg_21_Vd_b2 * C2P + alg_20_Vd_b5 * C5P)
        d_C2 = torch.zeros_like(y[:, 0]) + (kas_b2 * Clb2 * Sic1 + kpp_c1 * Cdc14 * C2P - (kdi_b2 + alg_21_Vd_b2 + alg_23_Vkp_c1) * C2)
        d_C5 = torch.zeros_like(y[:, 0]) + (kas_b5 * Clb5 * Sic1 + kpp_c1 * Cdc14 * C5P - (kdi_b5 + alg_20_Vd_b5 + alg_23_Vkp_c1) * C5)
        d_C2P = torch.zeros_like(y[:, 0]) + (alg_23_Vkp_c1 * C2 - (kpp_c1 * Cdc14 + kd3_c1 + alg_21_Vd_b2) * C2P)
        d_C5P = torch.zeros_like(y[:, 0]) + (alg_23_Vkp_c1 * C5 - (kpp_c1 * Cdc14 + kd3_c1 + alg_20_Vd_b5) * C5P)
        alg_24_Vi_sbf = ki_sbf + ki_sbf_2 * Clb2
        alg_22_Va_sbf = ka_sbf * (epsilon_sbf_n2 * Cln2 + epsilon_sbf_n3 * (alg_10_Cln3 + alg_12_Bck2) + epsilon_sbf_b5 * Clb5)
        alg_25_G_sbf = 2.0 * Ji_sbf * alg_22_Va_sbf / (alg_24_Vi_sbf + Ja_sbf * alg_24_Vi_sbf + Ji_sbf * alg_22_Va_sbf + ((alg_24_Vi_sbf + Ja_sbf * alg_24_Vi_sbf + Ji_sbf * alg_22_Va_sbf - alg_22_Va_sbf) ** 2.0 - 4.0 * (alg_24_Vi_sbf - alg_22_Va_sbf) * Ji_sbf * alg_22_Va_sbf) ** (1.0 / 2) - alg_22_Va_sbf)
        alg_26_MBF = alg_25_G_sbf
        d_Clb5 = torch.zeros_like(y[:, 0]) + ((ks_b5 + ks_b5_2 * alg_26_MBF) * mass + kd3_c1 * C5P + kdi_b5 * C5 + kd3_f6 * F5P + kdi_f5 * F5 - (alg_20_Vd_b5 + kas_b5 * Sic1 + kas_f5 * Cdc6) * Clb5)
        alg_27_SBF = alg_26_MBF
        d_Cln2 = torch.zeros_like(y[:, 0]) + ((ks_n2 + ks_n2_2 * alg_27_SBF) * mass - kd_n2 * Cln2)
        alg_17_Cdc6_T = Cdc6 + Cdc6P + F2 + F2P + F5 + F5P
        alg_28_Vkp_f6 = kd1_f6 + kd2_f6 * (epsilon_f6_n3 * alg_10_Cln3 + epsilon_f6_k2 * alg_12_Bck2 + epsilon_f6_n2 * Cln2 + epsilon_f6_b5 * Clb5 + epsilon_f6_b2 * Clb2) / (Jd2_f6 + alg_17_Cdc6_T)
        d_Cdc6 = torch.zeros_like(y[:, 0]) + (ks_f6 + ks_f6_2 * Swi5 + ks_f6_3 * alg_27_SBF + (alg_21_Vd_b2 + kdi_f2) * F2 + (alg_20_Vd_b5 + kdi_f5) * F5 + kpp_f6 * Cdc14 * Cdc6P - (kas_f2 * Clb2 + kas_f5 * Clb5 + alg_28_Vkp_f6) * Cdc6)
        d_Cdc6P = torch.zeros_like(y[:, 0]) + (alg_28_Vkp_f6 * Cdc6 - (kpp_f6 * Cdc14 + kd3_f6) * Cdc6P + alg_21_Vd_b2 * F2P + alg_20_Vd_b5 * F5P)
        d_F2 = torch.zeros_like(y[:, 0]) + (kas_f2 * Clb2 * Cdc6 + kpp_f6 * Cdc14 * F2P - (kdi_f2 + alg_21_Vd_b2 + alg_28_Vkp_f6) * F2)
        alg_29_Vd_pds = kd1_pds + kd2_pds * Cdc20_A + kd3_pds * Cdh1
        alg_18_PE = Esp1_T - Esp1
        d_Pds1 = torch.zeros_like(y[:, 0]) + (ks_pds + ks1_pds * alg_27_SBF + ks2_pds * alg_7_Mcm1 + kdi_esp * alg_18_PE - (alg_29_Vd_pds + kas_esp * Esp1) * Pds1)
        d_Esp1 = torch.zeros_like(y[:, 0]) + (-kas_esp * Pds1 * Esp1 + (kdi_esp + alg_29_Vd_pds) * alg_18_PE)
        d_F5 = torch.zeros_like(y[:, 0]) + (kas_f5 * Clb5 * Cdc6 + kpp_f6 * Cdc14 * F5P - (kdi_f5 + alg_20_Vd_b5 + alg_28_Vkp_f6) * F5)
        d_F2P = torch.zeros_like(y[:, 0]) + (alg_28_Vkp_f6 * F2 - (kpp_f6 * Cdc14 + kd3_f6 + alg_21_Vd_b2) * F2P)
        d_F5P = torch.zeros_like(y[:, 0]) + (alg_28_Vkp_f6 * F5 - (kpp_f6 * Cdc14 + kd3_f6 + alg_20_Vd_b5) * F5P)

        return torch.stack(
            [d_mass, d_Cln2, d_Clb5, d_C5P, d_C5, d_F5P, d_F5, d_Sic1, d_Cdc6, d_Clb2, d_C2P, d_C2, d_F2P, d_F2, d_Swi5, d_Cdc14, d_Sic1P, d_Cdc6P, d_Pds1, d_Esp1, d_ORI, d_BUD, d_SPN, d_Swi5_T, d_APC_P, d_Cdc20_T, d_Cdc20_A, d_Cdh1_T, d_Cdh1, d_Tem1, d_Cdc15, d_Cdc14_T, d_RENT, d_Net1, d_Net1_T, d_PPX],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
