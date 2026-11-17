import torch
import torch.nn as nn
import math


class CellCycleCsikaszNagy2006MammalianFromCodeODE(nn.Module):
    """
    PyTorch translation of cell_cycle/csikasz_nagy_2006_mammalian_from_code/model.py.
    Dynamic states: 14; independent constants: 93; derived constants used by rates: 0.
    Original constant slots: 93; algebraic slots: 30.
    """
    source_file = 'cell_cycle/csikasz_nagy_2006_mammalian_from_code/model.py'
    source_state_dim = 14
    source_constant_dim = 93
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

        self.state_dim = 14
        self.parameter_dim = 93

        self.parameters = (
            parameters
            if parameters is not None
            else {'k_assa': 25, 'k_dissa': 1, 'k_assb': 0, 'k_dissb': 0, 'k_disse': 1, 'k_asse': 50, 'CycD_0': 0.05, 'k_a20': 0.5, 'k_i20': 0.25, 'k_d20': 0.15, 'J_a20': 0.005, 'J_i20': 0.005, 'k_s20p': 0, 'k_s20pp': 0.15, 'n20': 1, 'J_20': 1, 'J_a20_2': 0.005, 'J_i20_2': 0.005, 'APC_T': 1, 'k_aie': 0.07, 'k_iie': 0.18, 'J_aie': 0.01, 'J_iie': 0.01, 'Cdh1_T': 1, 'J_ah1': 0.01, 'J_ih1': 0.01, 'mu': 0.004951, 'maxmass': 10000, 'k_atfp': 0, 'k_atfapp': 0.2, 'k_atfepp': 0.5, 'k_atfdpp': 3, 'k_itfp': 0.25, 'k_itfapp': 0.1, 'k_itfbpp': 0.1, 'J_itf': 0.01, 'J_atf': 0.01, 'k_dep': 0.01, 'k_deepp': 0.1, 'k_deapp': 0.5, 'k_debpp': 0.5, 'k_dap': 0.02, 'k_dapp': 2, 'k_dappp': 0, 'k_afi': 88, 'k_ifip': 88, 'k_ifibpp': 88, 'J_ifi': 88, 'J_afi': 88, 'k_ah1p': 0.18, 'k_ah1pp': 3.5, 'k_ih1p': 0, 'k_ih1app': 0.2, 'k_ih1bpp': 1, 'k_ih1epp': 0.1, 'k_ih1dpp': 0, 'k_sip': 1.8, 'k_sipp': 0, 'k_dip': 0.8, 'k_diapp': 5, 'k_diepp': 5, 'k_didpp': 0, 'k_dibpp': 5, 'k_14di': 0, 'k_afb': 1, 'k_ifb': 0.1, 'J_ifb': 0.1, 'J_afb': 0.1, 'k_sbp': 0.01, 'k_sbpp': 0.03, 'k_sap': 0, 'k_sapp': 0.025, 'k_sep': 0.01, 'k_sepp': 0.18, 'k_dbp': 0.005, 'k_dbhpp': 2, 'k_dbcpp': 0.1, 'k_aweep': 0.3, 'k_aweepp': 0, 'k_iweep': 0, 'k_iweepp': 1, 'J_iwee': 0.05, 'J_awee': 0.05, 'k_weep': 0.02, 'k_weepp': 0.2, 'k_a25p': 0, 'k_a25pp': 1, 'k_i25p': 0.3, 'k_i25pp': 0, 'J_i25': 0.1, 'J_a25': 0.1, 'k_25p': 0.01, 'k_25pp': 5}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'CycA': 0.00994044542312622, 'Tri_A': 0.01715379953384399, 'CKI': 0.2954076826572418, 'CycB': 0.1668413728475571, 'pB': 0.009814870543777943, 'BCKI': 0, 'CycE': 0.07760512828826904, 'Tri_E': 0.3117263317108154, 'mass': 1.174216866493225, 'Cdc20_A': 0.6605867147445679, 'APCP': 0.6716265678405762, 'Cdc20_i': 0.01855352707207203, 'pBCKI': 0, 'Cdh1': 0.9992357492446899}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k_assa', 'k_dissa', 'k_assb', 'k_dissb', 'k_disse', 'k_asse', 'CycD_0', 'k_a20', 'k_i20', 'k_d20', 'J_a20', 'J_i20', 'k_s20p', 'k_s20pp', 'n20', 'J_20', 'J_a20_2', 'J_i20_2', 'APC_T', 'k_aie', 'k_iie', 'J_aie', 'J_iie', 'Cdh1_T', 'J_ah1', 'J_ih1', 'mu', 'maxmass', 'k_atfp', 'k_atfapp', 'k_atfepp', 'k_atfdpp', 'k_itfp', 'k_itfapp', 'k_itfbpp', 'J_itf', 'J_atf', 'k_dep', 'k_deepp', 'k_deapp', 'k_debpp', 'k_dap', 'k_dapp', 'k_dappp', 'k_afi', 'k_ifip', 'k_ifibpp', 'J_ifi', 'J_afi', 'k_ah1p', 'k_ah1pp', 'k_ih1p', 'k_ih1app', 'k_ih1bpp', 'k_ih1epp', 'k_ih1dpp', 'k_sip', 'k_sipp', 'k_dip', 'k_diapp', 'k_diepp', 'k_didpp', 'k_dibpp', 'k_14di', 'k_afb', 'k_ifb', 'J_ifb', 'J_afb', 'k_sbp', 'k_sbpp', 'k_sap', 'k_sapp', 'k_sep', 'k_sepp', 'k_dbp', 'k_dbhpp', 'k_dbcpp', 'k_aweep', 'k_aweepp', 'k_iweep', 'k_iweepp', 'J_iwee', 'J_awee', 'k_weep', 'k_weepp', 'k_a25p', 'k_a25pp', 'k_i25p', 'k_i25pp', 'J_i25', 'J_a25', 'k_25p', 'k_25pp']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['CycA', 'Tri_A', 'CKI', 'CycB', 'pB', 'BCKI', 'CycE', 'Tri_E', 'mass', 'Cdc20_A', 'APCP', 'Cdc20_i', 'pBCKI', 'Cdh1']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (14)
        CycA = y[:, 0]
        Tri_A = y[:, 1]
        CKI = y[:, 2]
        CycB = y[:, 3]
        pB = y[:, 4]
        BCKI = y[:, 5]
        CycE = y[:, 6]
        Tri_E = y[:, 7]
        mass = y[:, 8]
        Cdc20_A = y[:, 9]
        APCP = y[:, 10]
        Cdc20_i = y[:, 11]
        pBCKI = y[:, 12]
        Cdh1 = y[:, 13]

        # Independent parameters (source constant order)
        k_assa = theta[:, 0]  # constants[0]
        k_dissa = theta[:, 1]  # constants[1]
        k_assb = theta[:, 2]  # constants[2]
        k_dissb = theta[:, 3]  # constants[3]
        k_disse = theta[:, 4]  # constants[4]
        k_asse = theta[:, 5]  # constants[5]
        CycD_0 = theta[:, 6]  # constants[6]
        k_a20 = theta[:, 7]  # constants[7]
        k_i20 = theta[:, 8]  # constants[8]
        k_d20 = theta[:, 9]  # constants[9]
        J_a20 = theta[:, 10]  # constants[10]
        J_i20 = theta[:, 11]  # constants[11]
        k_s20p = theta[:, 12]  # constants[12]
        k_s20pp = theta[:, 13]  # constants[13]
        n20 = theta[:, 14]  # constants[14]
        J_20 = theta[:, 15]  # constants[15]
        J_a20_2 = theta[:, 16]  # constants[16]
        J_i20_2 = theta[:, 17]  # constants[17]
        APC_T = theta[:, 18]  # constants[18]
        k_aie = theta[:, 19]  # constants[19]
        k_iie = theta[:, 20]  # constants[20]
        J_aie = theta[:, 21]  # constants[21]
        J_iie = theta[:, 22]  # constants[22]
        Cdh1_T = theta[:, 23]  # constants[23]
        J_ah1 = theta[:, 24]  # constants[24]
        J_ih1 = theta[:, 25]  # constants[25]
        mu = theta[:, 26]  # constants[26]
        maxmass = theta[:, 27]  # constants[27]
        k_atfp = theta[:, 28]  # constants[28]
        k_atfapp = theta[:, 29]  # constants[29]
        k_atfepp = theta[:, 30]  # constants[30]
        k_atfdpp = theta[:, 31]  # constants[31]
        k_itfp = theta[:, 32]  # constants[32]
        k_itfapp = theta[:, 33]  # constants[33]
        k_itfbpp = theta[:, 34]  # constants[34]
        J_itf = theta[:, 35]  # constants[35]
        J_atf = theta[:, 36]  # constants[36]
        k_dep = theta[:, 37]  # constants[37]
        k_deepp = theta[:, 38]  # constants[38]
        k_deapp = theta[:, 39]  # constants[39]
        k_debpp = theta[:, 40]  # constants[40]
        k_dap = theta[:, 41]  # constants[41]
        k_dapp = theta[:, 42]  # constants[42]
        k_dappp = theta[:, 43]  # constants[43]
        k_afi = theta[:, 44]  # constants[44]
        k_ifip = theta[:, 45]  # constants[45]
        k_ifibpp = theta[:, 46]  # constants[46]
        J_ifi = theta[:, 47]  # constants[47]
        J_afi = theta[:, 48]  # constants[48]
        k_ah1p = theta[:, 49]  # constants[49]
        k_ah1pp = theta[:, 50]  # constants[50]
        k_ih1p = theta[:, 51]  # constants[51]
        k_ih1app = theta[:, 52]  # constants[52]
        k_ih1bpp = theta[:, 53]  # constants[53]
        k_ih1epp = theta[:, 54]  # constants[54]
        k_ih1dpp = theta[:, 55]  # constants[55]
        k_sip = theta[:, 56]  # constants[56]
        k_sipp = theta[:, 57]  # constants[57]
        k_dip = theta[:, 58]  # constants[58]
        k_diapp = theta[:, 59]  # constants[59]
        k_diepp = theta[:, 60]  # constants[60]
        k_didpp = theta[:, 61]  # constants[61]
        k_dibpp = theta[:, 62]  # constants[62]
        k_14di = theta[:, 63]  # constants[63]
        k_afb = theta[:, 64]  # constants[64]
        k_ifb = theta[:, 65]  # constants[65]
        J_ifb = theta[:, 66]  # constants[66]
        J_afb = theta[:, 67]  # constants[67]
        k_sbp = theta[:, 68]  # constants[68]
        k_sbpp = theta[:, 69]  # constants[69]
        k_sap = theta[:, 70]  # constants[70]
        k_sapp = theta[:, 71]  # constants[71]
        k_sep = theta[:, 72]  # constants[72]
        k_sepp = theta[:, 73]  # constants[73]
        k_dbp = theta[:, 74]  # constants[74]
        k_dbhpp = theta[:, 75]  # constants[75]
        k_dbcpp = theta[:, 76]  # constants[76]
        k_aweep = theta[:, 77]  # constants[77]
        k_aweepp = theta[:, 78]  # constants[78]
        k_iweep = theta[:, 79]  # constants[79]
        k_iweepp = theta[:, 80]  # constants[80]
        J_iwee = theta[:, 81]  # constants[81]
        J_awee = theta[:, 82]  # constants[82]
        k_weep = theta[:, 83]  # constants[83]
        k_weepp = theta[:, 84]  # constants[84]
        k_a25p = theta[:, 85]  # constants[85]
        k_a25pp = theta[:, 86]  # constants[86]
        k_i25p = theta[:, 87]  # constants[87]
        k_i25pp = theta[:, 88]  # constants[88]
        J_i25 = theta[:, 89]  # constants[89]
        J_a25 = theta[:, 90]  # constants[90]
        k_25p = theta[:, 91]  # constants[91]
        k_25pp = theta[:, 92]  # constants[92]

        # Initialize rates and algebraic variables, as in computeRates()
        d_CycA = torch.zeros_like(y[:, 0])
        d_Tri_A = torch.zeros_like(y[:, 0])
        d_CKI = torch.zeros_like(y[:, 0])
        d_CycB = torch.zeros_like(y[:, 0])
        d_pB = torch.zeros_like(y[:, 0])
        d_BCKI = torch.zeros_like(y[:, 0])
        d_CycE = torch.zeros_like(y[:, 0])
        d_Tri_E = torch.zeros_like(y[:, 0])
        d_mass = torch.zeros_like(y[:, 0])
        d_Cdc20_A = torch.zeros_like(y[:, 0])
        d_APCP = torch.zeros_like(y[:, 0])
        d_Cdc20_i = torch.zeros_like(y[:, 0])
        d_pBCKI = torch.zeros_like(y[:, 0])
        d_Cdh1 = torch.zeros_like(y[:, 0])
        alg_0_APC = torch.zeros_like(y[:, 0])
        alg_1_CycD = torch.zeros_like(y[:, 0])
        alg_9_Cdh1_i = torch.zeros_like(y[:, 0])
        alg_10_V_atf = torch.zeros_like(y[:, 0])
        alg_11_V_itf = torch.zeros_like(y[:, 0])
        alg_12_TF_E = torch.zeros_like(y[:, 0])
        alg_13_V_de = torch.zeros_like(y[:, 0])
        alg_14_V_da = torch.zeros_like(y[:, 0])
        alg_15_Cdc14 = torch.zeros_like(y[:, 0])
        alg_16_TF_I = torch.zeros_like(y[:, 0])
        alg_17_V_ah1 = torch.zeros_like(y[:, 0])
        alg_18_V_si = torch.zeros_like(y[:, 0])
        alg_19_V_ih1 = torch.zeros_like(y[:, 0])
        alg_20_V_di = torch.zeros_like(y[:, 0])
        alg_21_V_sa = torch.zeros_like(y[:, 0])
        alg_22_TF_B = torch.zeros_like(y[:, 0])
        alg_23_V_se = torch.zeros_like(y[:, 0])
        alg_24_V_sb = torch.zeros_like(y[:, 0])
        alg_25_V_db = torch.zeros_like(y[:, 0])
        alg_26_Wee1 = torch.zeros_like(y[:, 0])
        alg_27_V_wee = torch.zeros_like(y[:, 0])
        alg_28_Cdc25 = torch.zeros_like(y[:, 0])
        alg_29_V_25 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Cdc20_A = torch.zeros_like(y[:, 0]) + (k_a20 * APCP * Cdc20_i / (J_a20 + Cdc20_i) - Cdc20_A * k_i20 / (J_i20 + Cdc20_A) - Cdc20_A * k_d20)
        d_Cdc20_i = torch.zeros_like(y[:, 0]) + ((k_s20p + k_s20pp * CycB ** n20) / (J_20 ** n20 + CycB ** n20) + Cdc20_A * k_i20 / (J_i20_2 + Cdc20_A) - k_d20 * Cdc20_i - k_a20 * APCP * Cdc20_i / (J_a20_2 + Cdc20_i))
        d_mass = torch.zeros_like(y[:, 0]) + (mu * mass * (1.0 - mass / maxmass))
        alg_0_APC = (APC_T - APCP) / 1.0
        d_APCP = torch.zeros_like(y[:, 0]) + (k_aie * CycB * alg_0_APC / (J_aie + alg_0_APC) - k_iie * APCP / (J_iie + APCP))
        alg_9_Cdh1_i = (Cdh1_T - Cdh1) / 1.0
        alg_15_Cdc14 = Cdc20_A
        alg_17_V_ah1 = k_ah1p + k_ah1pp * alg_15_Cdc14
        alg_1_CycD = CycD_0 * mass
        alg_19_V_ih1 = k_ih1p + k_ih1app * CycA + k_ih1bpp * CycB + k_ih1epp * CycE + k_ih1dpp * alg_1_CycD
        d_Cdh1 = torch.zeros_like(y[:, 0]) + (alg_9_Cdh1_i * alg_17_V_ah1 / (J_ah1 + alg_9_Cdh1_i) - Cdh1 * alg_19_V_ih1 / (J_ih1 + Cdh1))
        alg_20_V_di = (k_dip + k_diapp * CycA + k_dibpp * CycB + k_diepp * CycE + k_didpp * alg_1_CycD) / (1.0 + k_14di * alg_15_Cdc14)
        alg_14_V_da = k_dap + (k_dapp + k_dappp) * Cdc20_A + k_dappp * Cdc20_i
        d_Tri_A = torch.zeros_like(y[:, 0]) + (k_assa * CKI * CycA - k_dissa * Tri_A - alg_20_V_di * Tri_A - alg_14_V_da * Tri_A)
        alg_13_V_de = k_dep + k_deepp * CycE + k_deapp * CycA + k_debpp * CycB
        d_Tri_E = torch.zeros_like(y[:, 0]) + (k_asse * CKI * CycE - k_disse * Tri_E - alg_20_V_di * Tri_E - alg_13_V_de * Tri_E)
        alg_10_V_atf = k_atfp + k_atfapp * CycA + k_atfepp * CycE + k_atfdpp * alg_1_CycD
        alg_11_V_itf = k_itfp + k_itfapp * CycA + k_itfbpp * CycB
        alg_12_TF_E = 2.0 * alg_10_V_atf * J_itf / (alg_11_V_itf - alg_10_V_atf + J_atf * alg_11_V_itf + J_itf * alg_10_V_atf + ((alg_11_V_itf - alg_10_V_atf + J_atf * alg_11_V_itf + J_itf * alg_10_V_atf) ** 2.0 - 4.0 * (alg_11_V_itf - alg_10_V_atf) * J_itf * alg_10_V_atf) ** (1.0 / 2))
        alg_21_V_sa = mass * (k_sap + k_sapp * alg_12_TF_E)
        d_CycA = torch.zeros_like(y[:, 0]) + (k_dissa * Tri_A + alg_20_V_di * Tri_A + alg_21_V_sa - alg_14_V_da * CycA - k_assa * CKI * CycA)
        alg_23_V_se = mass * (k_sep + k_sepp * alg_12_TF_E)
        d_CycE = torch.zeros_like(y[:, 0]) + (k_disse * Tri_E + alg_20_V_di * Tri_E + alg_23_V_se - alg_13_V_de * CycE - k_asse * CKI * CycE)
        alg_25_V_db = k_dbp + k_dbhpp * Cdh1 + k_dbcpp * Cdc20_A
        alg_16_TF_I = 2.0 * k_afi * alg_15_Cdc14 * J_ifi / (k_ifip + k_ifibpp * CycB - k_afi * alg_15_Cdc14 + J_afi * (k_ifip + k_ifibpp * CycB) + J_ifi * k_afi * alg_15_Cdc14 + ((k_ifip + k_ifibpp * CycB - k_afi * alg_15_Cdc14 + J_afi * (k_ifip + k_ifibpp * CycB) + J_ifi * k_afi * alg_15_Cdc14) ** 2.0 - 4.0 * (k_ifip + k_ifibpp * CycB - k_afi * alg_15_Cdc14) * J_ifi * k_afi * alg_15_Cdc14) ** (1.0 / 2))
        alg_18_V_si = k_sip + k_sipp * alg_16_TF_I
        d_CKI = torch.zeros_like(y[:, 0]) + (-k_assb * CycB * CKI + k_dissb * BCKI - k_assb * pB * CKI + k_dissb * pBCKI + alg_25_V_db * BCKI + alg_25_V_db * pBCKI + alg_18_V_si - alg_20_V_di * CKI - k_assa * CKI * CycA + k_dissa * Tri_A + alg_14_V_da * Tri_A - k_asse * CKI * CycE + k_disse * Tri_E + alg_13_V_de * Tri_E)
        alg_22_TF_B = 2.0 * k_afb * CycB * J_ifb / (k_ifb - k_afb * CycB + J_afb * k_ifb + J_ifb * k_afb * CycB + ((k_ifb - k_afb * CycB + J_afb * k_ifb + J_ifb * k_afb * CycB) ** 2.0 - 4.0 * (k_ifb - k_afb * CycB) * k_afb * CycB * J_ifb) ** (1.0 / 2))
        alg_24_V_sb = mass * (k_sbp + k_sbpp * alg_22_TF_B)
        alg_28_Cdc25 = 2.0 * (k_a25p + k_a25pp * CycB) * J_i25 / (k_i25p + k_i25pp * alg_15_Cdc14 - (k_a25p + k_a25pp * CycB) + J_a25 * (k_i25p + k_i25pp * alg_15_Cdc14) + (k_a25p + k_a25pp * CycB) * J_i25 + ((k_i25p + k_i25pp * alg_15_Cdc14 - (k_a25p + k_a25pp * CycB) + J_a25 * (k_i25p + k_i25pp * alg_15_Cdc14) + (k_a25p + k_a25pp * CycB) * J_i25) ** 2.0 - 4.0 * (k_i25p + k_i25pp * alg_15_Cdc14 - (k_a25p + k_a25pp * CycB)) * (k_a25p + k_a25pp * CycB) * J_i25) ** (1.0 / 2))
        alg_29_V_25 = k_25p + k_25pp * alg_28_Cdc25
        alg_26_Wee1 = 2.0 * (k_aweep + k_aweepp * alg_15_Cdc14) * J_iwee / (k_iweep + k_iweepp * CycB - (k_aweep + k_aweepp * alg_15_Cdc14) + J_awee * (k_iweep + k_iweepp * CycB) + J_iwee * (k_aweep + k_aweepp * alg_15_Cdc14) + ((k_iweep + k_iweepp * CycB - (k_aweep + k_aweepp * alg_15_Cdc14) + J_awee * (k_iweep + k_iweepp * CycB) + J_iwee * (k_aweep + k_aweepp * alg_15_Cdc14)) ** 2.0 - 4.0 * (k_iweep + k_iweepp * CycB - (k_aweep + k_aweepp * alg_15_Cdc14)) * (k_aweep + k_aweepp * alg_15_Cdc14) * J_iwee) ** (1.0 / 2))
        alg_27_V_wee = k_weep + k_weepp * alg_26_Wee1
        d_CycB = torch.zeros_like(y[:, 0]) + (alg_24_V_sb - alg_25_V_db * CycB + alg_29_V_25 * pB - alg_27_V_wee * CycB - k_assb * CycB * CKI + k_dissb * BCKI + alg_20_V_di * BCKI)
        d_pB = torch.zeros_like(y[:, 0]) + (alg_27_V_wee * CycB - alg_25_V_db * pB - k_assb * pB * CKI + k_dissb * pBCKI + alg_20_V_di * pBCKI - alg_29_V_25 * pB)
        d_BCKI = torch.zeros_like(y[:, 0]) + (k_assb * CycB * CKI - k_dissb * BCKI + alg_29_V_25 * pBCKI - alg_27_V_wee * BCKI - alg_25_V_db * BCKI - alg_20_V_di * BCKI)
        d_pBCKI = torch.zeros_like(y[:, 0]) + (k_assb * pB * CKI - k_dissb * pBCKI - alg_29_V_25 * pBCKI + alg_27_V_wee * BCKI - alg_25_V_db * pBCKI - alg_20_V_di * pBCKI)

        return torch.stack(
            [d_CycA, d_Tri_A, d_CKI, d_CycB, d_pB, d_BCKI, d_CycE, d_Tri_E, d_mass, d_Cdc20_A, d_APCP, d_Cdc20_i, d_pBCKI, d_Cdh1],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
