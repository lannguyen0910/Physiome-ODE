import torch
import torch.nn as nn
import math


class CardiovascularCirculationCircPbpkODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/circ_pbpk/model.py.
    Dynamic states: 44; independent constants: 91; derived constants used by rates: 5.
    Original constant slots: 96; algebraic slots: 86.
    """
    source_file = 'cardiovascular_circulation/circ_pbpk/model.py'
    source_state_dim = 44
    source_constant_dim = 96
    algebraic_dim = 86
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 44
        self.parameter_dim = 91

        self.parameters = (
            parameters
            if parameters is not None
            else {'HR': 77, 'Esys': 0.145, 'Edia': 0.131, 'TsK': 0.2, 'P_LA_ext': 0, 'V_LA_0': 15, 'R_LA': 0.01, 'PRint': 0.12, 'Esys_2': 5.6, 'Edia_2': 0.19, 'TsK_2': 0.35, 'pLVPext': 0, 'V_LV_0': 30, 'R_LV': 0.001, 'C_AO': 1.2, 'L_AO': 5e-05, 'R_AO': 0.01, 'V_AO_0': 200, 'R_CO': 1, 'R_COV': 0.2, 'V_CO_0': 10, 'C_CO': 0.1, 'R_CR': 5, 'R_CRV': 0.02, 'V_CR_0': 10, 'C_CR': 1, 'D_CRtissue': 1, 'V_CRtissue': 1, 'R_AR': 1.5, 'C_AR': 0.5, 'V_AR_0': 370, 'C_AD': 1.81, 'R_AD': 1, 'R_ADV': 0.33, 'V_AD_0': 75, 'C_MU': 1.81, 'R_MU': 1, 'R_MUV': 0.33, 'V_MU_0': 75, 'C_GI': 1.81, 'R_GI': 1, 'R_GIV': 0.33, 'V_GI_0': 75, 'C_LI': 1.81, 'R_LI': 1, 'R_LIV': 0.33, 'V_LI_0': 75, 'metabolism': 0, 'C_KI': 1.81, 'R_KI': 1, 'V_KI_0': 75, 'R_KIV': 0.33, 'C_SK': 1.81, 'R_SK': 1, 'R_SKV': 0.33, 'V_SK_0': 75, 'C_OT': 1.81, 'R_OT': 1, 'R_OTV': 0.33, 'V_OT_0': 75, 'R_VEV': 0.0223, 'C_VE': 13.24, 'V_VE_0': 596, 'R_VCV': 0.0267, 'V_VC_0': 600, 'C_VC': 73.88, 'Esys_3': 0.11, 'Edia_3': 0.099, 'TsK_3': 0.2, 'P_RA_ext': 0, 'V_RA_0': 15, 'PRinterval': 0.12, 'Esys_4': 0.67, 'Edia_4': 0.104, 'TsK_4': 0.35, 'P_RV_ext': 0, 'V_RV_0': 25, 'R_RV': 0.002, 'R_PA': 0.0227, 'C_PA': 2.222, 'V_PA_0': 50, 'C_Pa': 1.481, 'R_Pa': 0.053, 'V_Pa_0': 30, 'R_SH': 2, 'C_PC': 1.778, 'R_PC': 0.0379, 'R_PCV': 0.0379, 'V_PC_0': 53, 'C_PV': 5, 'V_PV_0': 150}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V_LA': 144.038898121062, 'sub_X_LA': 0, 'V_LV': 118.75315693072, 'sub_X_LV': 0, 'V_AO': 302.714548324312, 'F_AOalways': -6898.97886842467, 'sub_X_AO': 0, 'V_CO': 11.4012164759732, 'sub_X_CO': 0, 'V_CR': 27.2147903053333, 'sub_X_CR': 0, 'sub_X_CRtissue': 0, 'V_AR': 383.557125288364, 'sub_X_AR': 0, 'V_AD': 111.639088723426, 'sub_X_AD': 0, 'V_MU': 111.639088723426, 'sub_X_MU': 0, 'V_GI': 111.639088723426, 'sub_X_GI': 0, 'V_LI': 111.639088723426, 'sub_X_LI': 0, 'V_KI': 111.639088723426, 'sub_X_KI': 0, 'V_SK': 111.639088723426, 'sub_X_SK': 0, 'V_OT': 111.639088723426, 'sub_X_OT': 0, 'V_VE': 833.291684587657, 'sub_X_VE': 5, 'V_VC': 1849.93788284599, 'sub_X_VC': 0, 'V_RA': 152.150765432928, 'sub_X_RA': 0, 'V_RV': 154.337916417774, 'sub_X_RV': 0, 'V_PA': 112.059036699839, 'sub_X_PA': 0, 'V_Pa': 66.9910537203609, 'sub_X_Pa': 0, 'V_PC': 91.7784213531534, 'sub_X_PC': 0, 'V_PV': 239.25897115602, 'sub_X_PV': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['HR', 'Esys', 'Edia', 'TsK', 'P_LA_ext', 'V_LA_0', 'R_LA', 'PRint', 'Esys_2', 'Edia_2', 'TsK_2', 'pLVPext', 'V_LV_0', 'R_LV', 'C_AO', 'L_AO', 'R_AO', 'V_AO_0', 'R_CO', 'R_COV', 'V_CO_0', 'C_CO', 'R_CR', 'R_CRV', 'V_CR_0', 'C_CR', 'D_CRtissue', 'V_CRtissue', 'R_AR', 'C_AR', 'V_AR_0', 'C_AD', 'R_AD', 'R_ADV', 'V_AD_0', 'C_MU', 'R_MU', 'R_MUV', 'V_MU_0', 'C_GI', 'R_GI', 'R_GIV', 'V_GI_0', 'C_LI', 'R_LI', 'R_LIV', 'V_LI_0', 'metabolism', 'C_KI', 'R_KI', 'V_KI_0', 'R_KIV', 'C_SK', 'R_SK', 'R_SKV', 'V_SK_0', 'C_OT', 'R_OT', 'R_OTV', 'V_OT_0', 'R_VEV', 'C_VE', 'V_VE_0', 'R_VCV', 'V_VC_0', 'C_VC', 'Esys_3', 'Edia_3', 'TsK_3', 'P_RA_ext', 'V_RA_0', 'PRinterval', 'Esys_4', 'Edia_4', 'TsK_4', 'P_RV_ext', 'V_RV_0', 'R_RV', 'R_PA', 'C_PA', 'V_PA_0', 'C_Pa', 'R_Pa', 'V_Pa_0', 'R_SH', 'C_PC', 'R_PC', 'R_PCV', 'V_PC_0', 'C_PV', 'V_PV_0']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V_LA', 'sub_X_LA', 'V_LV', 'sub_X_LV', 'V_AO', 'F_AOalways', 'sub_X_AO', 'V_CO', 'sub_X_CO', 'V_CR', 'sub_X_CR', 'sub_X_CRtissue', 'V_AR', 'sub_X_AR', 'V_AD', 'sub_X_AD', 'V_MU', 'sub_X_MU', 'V_GI', 'sub_X_GI', 'V_LI', 'sub_X_LI', 'V_KI', 'sub_X_KI', 'V_SK', 'sub_X_SK', 'V_OT', 'sub_X_OT', 'V_VE', 'sub_X_VE', 'V_VC', 'sub_X_VC', 'V_RA', 'sub_X_RA', 'V_RV', 'sub_X_RV', 'V_PA', 'sub_X_PA', 'V_Pa', 'sub_X_Pa', 'V_PC', 'sub_X_PC', 'V_PV', 'sub_X_PV']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (44)
        V_LA = y[:, 0]
        sub_X_LA = y[:, 1]
        V_LV = y[:, 2]
        sub_X_LV = y[:, 3]
        V_AO = y[:, 4]
        F_AOalways = y[:, 5]
        sub_X_AO = y[:, 6]
        V_CO = y[:, 7]
        sub_X_CO = y[:, 8]
        V_CR = y[:, 9]
        sub_X_CR = y[:, 10]
        sub_X_CRtissue = y[:, 11]
        V_AR = y[:, 12]
        sub_X_AR = y[:, 13]
        V_AD = y[:, 14]
        sub_X_AD = y[:, 15]
        V_MU = y[:, 16]
        sub_X_MU = y[:, 17]
        V_GI = y[:, 18]
        sub_X_GI = y[:, 19]
        V_LI = y[:, 20]
        sub_X_LI = y[:, 21]
        V_KI = y[:, 22]
        sub_X_KI = y[:, 23]
        V_SK = y[:, 24]
        sub_X_SK = y[:, 25]
        V_OT = y[:, 26]
        sub_X_OT = y[:, 27]
        V_VE = y[:, 28]
        sub_X_VE = y[:, 29]
        V_VC = y[:, 30]
        sub_X_VC = y[:, 31]
        V_RA = y[:, 32]
        sub_X_RA = y[:, 33]
        V_RV = y[:, 34]
        sub_X_RV = y[:, 35]
        V_PA = y[:, 36]
        sub_X_PA = y[:, 37]
        V_Pa = y[:, 38]
        sub_X_Pa = y[:, 39]
        V_PC = y[:, 40]
        sub_X_PC = y[:, 41]
        V_PV = y[:, 42]
        sub_X_PV = y[:, 43]

        # Independent parameters (source constant order)
        HR = theta[:, 0]  # constants[0]
        Esys = theta[:, 1]  # constants[1]
        Edia = theta[:, 2]  # constants[2]
        TsK = theta[:, 3]  # constants[3]
        P_LA_ext = theta[:, 4]  # constants[4]
        V_LA_0 = theta[:, 5]  # constants[5]
        R_LA = theta[:, 6]  # constants[6]
        PRint = theta[:, 7]  # constants[7]
        Esys_2 = theta[:, 8]  # constants[8]
        Edia_2 = theta[:, 9]  # constants[9]
        TsK_2 = theta[:, 10]  # constants[10]
        pLVPext = theta[:, 11]  # constants[11]
        V_LV_0 = theta[:, 12]  # constants[12]
        R_LV = theta[:, 13]  # constants[13]
        C_AO = theta[:, 14]  # constants[14]
        L_AO = theta[:, 15]  # constants[15]
        R_AO = theta[:, 16]  # constants[16]
        V_AO_0 = theta[:, 17]  # constants[17]
        R_CO = theta[:, 18]  # constants[18]
        R_COV = theta[:, 19]  # constants[19]
        V_CO_0 = theta[:, 20]  # constants[20]
        C_CO = theta[:, 21]  # constants[21]
        R_CR = theta[:, 22]  # constants[22]
        R_CRV = theta[:, 23]  # constants[23]
        V_CR_0 = theta[:, 24]  # constants[24]
        C_CR = theta[:, 25]  # constants[25]
        D_CRtissue = theta[:, 26]  # constants[26]
        V_CRtissue = theta[:, 27]  # constants[27]
        R_AR = theta[:, 28]  # constants[28]
        C_AR = theta[:, 29]  # constants[29]
        V_AR_0 = theta[:, 30]  # constants[30]
        C_AD = theta[:, 31]  # constants[31]
        R_AD = theta[:, 32]  # constants[32]
        R_ADV = theta[:, 33]  # constants[33]
        V_AD_0 = theta[:, 34]  # constants[34]
        C_MU = theta[:, 35]  # constants[35]
        R_MU = theta[:, 36]  # constants[36]
        R_MUV = theta[:, 37]  # constants[37]
        V_MU_0 = theta[:, 38]  # constants[38]
        C_GI = theta[:, 39]  # constants[39]
        R_GI = theta[:, 40]  # constants[40]
        R_GIV = theta[:, 41]  # constants[41]
        V_GI_0 = theta[:, 42]  # constants[42]
        C_LI = theta[:, 43]  # constants[43]
        R_LI = theta[:, 44]  # constants[44]
        R_LIV = theta[:, 45]  # constants[45]
        V_LI_0 = theta[:, 46]  # constants[46]
        metabolism = theta[:, 47]  # constants[47]
        C_KI = theta[:, 48]  # constants[48]
        R_KI = theta[:, 49]  # constants[49]
        V_KI_0 = theta[:, 50]  # constants[50]
        R_KIV = theta[:, 51]  # constants[51]
        C_SK = theta[:, 52]  # constants[52]
        R_SK = theta[:, 53]  # constants[53]
        R_SKV = theta[:, 54]  # constants[54]
        V_SK_0 = theta[:, 55]  # constants[55]
        C_OT = theta[:, 56]  # constants[56]
        R_OT = theta[:, 57]  # constants[57]
        R_OTV = theta[:, 58]  # constants[58]
        V_OT_0 = theta[:, 59]  # constants[59]
        R_VEV = theta[:, 60]  # constants[60]
        C_VE = theta[:, 61]  # constants[61]
        V_VE_0 = theta[:, 62]  # constants[62]
        R_VCV = theta[:, 63]  # constants[63]
        V_VC_0 = theta[:, 64]  # constants[64]
        C_VC = theta[:, 65]  # constants[65]
        Esys_3 = theta[:, 66]  # constants[66]
        Edia_3 = theta[:, 67]  # constants[67]
        TsK_3 = theta[:, 68]  # constants[68]
        P_RA_ext = theta[:, 69]  # constants[69]
        V_RA_0 = theta[:, 70]  # constants[70]
        PRinterval = theta[:, 71]  # constants[71]
        Esys_4 = theta[:, 72]  # constants[72]
        Edia_4 = theta[:, 73]  # constants[73]
        TsK_4 = theta[:, 74]  # constants[74]
        P_RV_ext = theta[:, 75]  # constants[75]
        V_RV_0 = theta[:, 76]  # constants[76]
        R_RV = theta[:, 77]  # constants[77]
        R_PA = theta[:, 78]  # constants[78]
        C_PA = theta[:, 79]  # constants[79]
        V_PA_0 = theta[:, 80]  # constants[80]
        C_Pa = theta[:, 81]  # constants[81]
        R_Pa = theta[:, 82]  # constants[82]
        V_Pa_0 = theta[:, 83]  # constants[83]
        R_SH = theta[:, 84]  # constants[84]
        C_PC = theta[:, 85]  # constants[85]
        R_PC = theta[:, 86]  # constants[86]
        R_PCV = theta[:, 87]  # constants[87]
        V_PC_0 = theta[:, 88]  # constants[88]
        C_PV = theta[:, 89]  # constants[89]
        V_PV_0 = theta[:, 90]  # constants[90]

        # Derived constants from the original initConsts()
        hrf = HR / 0.06  # constants[91]
        Ts = TsK * (hrf / 1000.0) ** (1.0 / 2)  # constants[92]
        Ts_2 = TsK_2 * (hrf / 1000.0) ** (1.0 / 2)  # constants[93]
        Ts_3 = TsK_3 * (hrf / 1000.0) ** (1.0 / 2)  # constants[94]
        Ts_4 = TsK_4 * (hrf / 1000.0) ** (1.0 / 2)  # constants[95]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V_LA = torch.zeros_like(y[:, 0])
        d_sub_X_LA = torch.zeros_like(y[:, 0])
        d_V_LV = torch.zeros_like(y[:, 0])
        d_sub_X_LV = torch.zeros_like(y[:, 0])
        d_V_AO = torch.zeros_like(y[:, 0])
        d_F_AOalways = torch.zeros_like(y[:, 0])
        d_sub_X_AO = torch.zeros_like(y[:, 0])
        d_V_CO = torch.zeros_like(y[:, 0])
        d_sub_X_CO = torch.zeros_like(y[:, 0])
        d_V_CR = torch.zeros_like(y[:, 0])
        d_sub_X_CR = torch.zeros_like(y[:, 0])
        d_sub_X_CRtissue = torch.zeros_like(y[:, 0])
        d_V_AR = torch.zeros_like(y[:, 0])
        d_sub_X_AR = torch.zeros_like(y[:, 0])
        d_V_AD = torch.zeros_like(y[:, 0])
        d_sub_X_AD = torch.zeros_like(y[:, 0])
        d_V_MU = torch.zeros_like(y[:, 0])
        d_sub_X_MU = torch.zeros_like(y[:, 0])
        d_V_GI = torch.zeros_like(y[:, 0])
        d_sub_X_GI = torch.zeros_like(y[:, 0])
        d_V_LI = torch.zeros_like(y[:, 0])
        d_sub_X_LI = torch.zeros_like(y[:, 0])
        d_V_KI = torch.zeros_like(y[:, 0])
        d_sub_X_KI = torch.zeros_like(y[:, 0])
        d_V_SK = torch.zeros_like(y[:, 0])
        d_sub_X_SK = torch.zeros_like(y[:, 0])
        d_V_OT = torch.zeros_like(y[:, 0])
        d_sub_X_OT = torch.zeros_like(y[:, 0])
        d_V_VE = torch.zeros_like(y[:, 0])
        d_sub_X_VE = torch.zeros_like(y[:, 0])
        d_V_VC = torch.zeros_like(y[:, 0])
        d_sub_X_VC = torch.zeros_like(y[:, 0])
        d_V_RA = torch.zeros_like(y[:, 0])
        d_sub_X_RA = torch.zeros_like(y[:, 0])
        d_V_RV = torch.zeros_like(y[:, 0])
        d_sub_X_RV = torch.zeros_like(y[:, 0])
        d_V_PA = torch.zeros_like(y[:, 0])
        d_sub_X_PA = torch.zeros_like(y[:, 0])
        d_V_Pa = torch.zeros_like(y[:, 0])
        d_sub_X_Pa = torch.zeros_like(y[:, 0])
        d_V_PC = torch.zeros_like(y[:, 0])
        d_sub_X_PC = torch.zeros_like(y[:, 0])
        d_V_PV = torch.zeros_like(y[:, 0])
        d_sub_X_PV = torch.zeros_like(y[:, 0])
        alg_0_beattime = torch.zeros_like(y[:, 0])
        alg_1_E_LA = torch.zeros_like(y[:, 0])
        alg_2_conc_X_LA = torch.zeros_like(y[:, 0])
        alg_3_P_LA = torch.zeros_like(y[:, 0])
        alg_4_beattime_2 = torch.zeros_like(y[:, 0])
        alg_5_E_LV = torch.zeros_like(y[:, 0])
        alg_6_conc_X_LV = torch.zeros_like(y[:, 0])
        alg_7_P_LV = torch.zeros_like(y[:, 0])
        alg_9_F_LV = torch.zeros_like(y[:, 0])
        alg_10_conc_X_AO = torch.zeros_like(y[:, 0])
        alg_11_P_AO = torch.zeros_like(y[:, 0])
        alg_12_F_AO = torch.zeros_like(y[:, 0])
        alg_13_conc_X_CO = torch.zeros_like(y[:, 0])
        alg_14_P_CO = torch.zeros_like(y[:, 0])
        alg_15_F_CO = torch.zeros_like(y[:, 0])
        alg_16_conc_X_CRtissue = torch.zeros_like(y[:, 0])
        alg_17_conc_X_CR = torch.zeros_like(y[:, 0])
        alg_18_P_CR = torch.zeros_like(y[:, 0])
        alg_19_F_CR = torch.zeros_like(y[:, 0])
        alg_20_conc_X_AR = torch.zeros_like(y[:, 0])
        alg_21_P_AR = torch.zeros_like(y[:, 0])
        alg_22_F_AR = torch.zeros_like(y[:, 0])
        alg_23_conc_X_AD = torch.zeros_like(y[:, 0])
        alg_24_P_AD = torch.zeros_like(y[:, 0])
        alg_25_F_AD = torch.zeros_like(y[:, 0])
        alg_26_conc_X_MU = torch.zeros_like(y[:, 0])
        alg_27_P_MU = torch.zeros_like(y[:, 0])
        alg_28_F_MU = torch.zeros_like(y[:, 0])
        alg_29_conc_X_GI = torch.zeros_like(y[:, 0])
        alg_30_P_GI = torch.zeros_like(y[:, 0])
        alg_31_F_GI = torch.zeros_like(y[:, 0])
        alg_32_conc_X_LI = torch.zeros_like(y[:, 0])
        alg_33_P_LI = torch.zeros_like(y[:, 0])
        alg_34_F_GIV = torch.zeros_like(y[:, 0])
        alg_35_F_LI = torch.zeros_like(y[:, 0])
        alg_36_conc_X_KI = torch.zeros_like(y[:, 0])
        alg_37_P_KI = torch.zeros_like(y[:, 0])
        alg_38_F_KI = torch.zeros_like(y[:, 0])
        alg_39_conc_X_SK = torch.zeros_like(y[:, 0])
        alg_40_P_SK = torch.zeros_like(y[:, 0])
        alg_41_F_SK = torch.zeros_like(y[:, 0])
        alg_42_conc_X_OT = torch.zeros_like(y[:, 0])
        alg_43_P_OT = torch.zeros_like(y[:, 0])
        alg_44_F_OT = torch.zeros_like(y[:, 0])
        alg_45_conc_X_VE = torch.zeros_like(y[:, 0])
        alg_46_P_VE = torch.zeros_like(y[:, 0])
        alg_47_F_ADV = torch.zeros_like(y[:, 0])
        alg_48_F_MUV = torch.zeros_like(y[:, 0])
        alg_49_F_LIV = torch.zeros_like(y[:, 0])
        alg_50_F_KIV = torch.zeros_like(y[:, 0])
        alg_51_F_SKV = torch.zeros_like(y[:, 0])
        alg_52_F_OTV = torch.zeros_like(y[:, 0])
        alg_53_F_VE = torch.zeros_like(y[:, 0])
        alg_54_conc_X_VC = torch.zeros_like(y[:, 0])
        alg_55_P_VC = torch.zeros_like(y[:, 0])
        alg_56_F_CRV = torch.zeros_like(y[:, 0])
        alg_57_F_VEV = torch.zeros_like(y[:, 0])
        alg_58_F_VC = torch.zeros_like(y[:, 0])
        alg_59_beattime_3 = torch.zeros_like(y[:, 0])
        alg_60_E_RA = torch.zeros_like(y[:, 0])
        alg_61_conc_X_RA = torch.zeros_like(y[:, 0])
        alg_62_P_RA = torch.zeros_like(y[:, 0])
        alg_63_F_COV = torch.zeros_like(y[:, 0])
        alg_64_F_VCV = torch.zeros_like(y[:, 0])
        alg_65_F_RA = torch.zeros_like(y[:, 0])
        alg_66_beattime_4 = torch.zeros_like(y[:, 0])
        alg_67_E_RV = torch.zeros_like(y[:, 0])
        alg_68_conc_X_RV = torch.zeros_like(y[:, 0])
        alg_69_P_RV = torch.zeros_like(y[:, 0])
        alg_70_F_RV = torch.zeros_like(y[:, 0])
        alg_71_conc_X_PA = torch.zeros_like(y[:, 0])
        alg_72_P_PA = torch.zeros_like(y[:, 0])
        alg_73_F_PA = torch.zeros_like(y[:, 0])
        alg_74_conc_X_Pa = torch.zeros_like(y[:, 0])
        alg_75_P_Pa = torch.zeros_like(y[:, 0])
        alg_76_F_Pa = torch.zeros_like(y[:, 0])
        alg_77_conc_X_PC = torch.zeros_like(y[:, 0])
        alg_78_P_PC = torch.zeros_like(y[:, 0])
        alg_79_F_SH = torch.zeros_like(y[:, 0])
        alg_80_F_PC = torch.zeros_like(y[:, 0])
        alg_81_conc_X_PV = torch.zeros_like(y[:, 0])
        alg_82_P_PV = torch.zeros_like(y[:, 0])
        alg_83_F_LA = torch.zeros_like(y[:, 0])
        alg_84_F_PCV = torch.zeros_like(y[:, 0])
        alg_85_F_PV = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_4_beattime_2 = (t - torch.floor(t / hrf) * hrf) / 1000.0 - PRint
        alg_5_E_LV = torch.where((alg_4_beattime_2 * 0.001 >= 0.0) & (alg_4_beattime_2 * 0.001 <= Ts_2), Edia_2 + (Esys_2 - Edia_2) * (1.0 - torch.cos(math.pi * (alg_4_beattime_2 * 0.001) / Ts_2)) / 2.0, torch.where((alg_4_beattime_2 * 0.001 < 1.5 * Ts_2) & (alg_4_beattime_2 * 0.001 >= Ts_2), Edia_2 + (Esys_2 - Edia_2) * (1.0 + torch.cos(2.0 * math.pi * (alg_4_beattime_2 * 0.001 - Ts_2) / Ts_2)) / 2.0, Edia_2))
        alg_7_P_LV = alg_5_E_LV * (V_LV - V_LV_0) + pLVPext
        alg_11_P_AO = 1.0 / C_AO * (V_AO - V_AO_0)
        d_F_AOalways = torch.zeros_like(y[:, 0]) + ((1.0 / L_AO * (alg_7_P_LV - alg_11_P_AO) - R_AO / L_AO * F_AOalways) / 1000.0)
        alg_0_beattime = (t - torch.floor(t / hrf) * hrf) / 1000.0
        alg_1_E_LA = torch.where((alg_0_beattime >= 0.0) & (alg_0_beattime <= Ts), Edia + (Esys - Edia) * (1.0 - torch.cos(math.pi * alg_0_beattime / Ts)) / 2.0, torch.where((alg_0_beattime < 1.5 * Ts) & (alg_0_beattime >= Ts), Edia + (Esys - Edia) * (1.0 + torch.cos(2.0 * math.pi * (alg_0_beattime - Ts) / Ts)) / 2.0, Edia))
        alg_3_P_LA = alg_1_E_LA * (V_LA - V_LA_0) + P_LA_ext
        alg_9_F_LV = torch.where(alg_3_P_LA > alg_7_P_LV, (alg_3_P_LA - alg_7_P_LV) / R_LV, 0.0)
        alg_2_conc_X_LA = sub_X_LA / V_LA
        alg_12_F_AO = torch.where(alg_7_P_LV > alg_11_P_AO, F_AOalways, 0.0)
        alg_14_P_CO = 1.0 / C_CO * (V_CO - V_CO_0)
        alg_15_F_CO = torch.where(alg_7_P_LV > alg_14_P_CO, (alg_7_P_LV - alg_14_P_CO) / R_CO, 0.0)
        alg_6_conc_X_LV = sub_X_LV / V_LV
        d_sub_X_LV = torch.zeros_like(y[:, 0]) + ((alg_2_conc_X_LA * alg_9_F_LV - alg_6_conc_X_LV * (alg_12_F_AO + alg_15_F_CO)) / 1000.0)
        d_V_LV = torch.zeros_like(y[:, 0]) + ((alg_9_F_LV - alg_12_F_AO - alg_15_F_CO) / 1000.0)
        alg_17_conc_X_CR = sub_X_CR / V_CR
        alg_16_conc_X_CRtissue = sub_X_CRtissue / V_CRtissue
        d_sub_X_CRtissue = torch.zeros_like(y[:, 0]) + ((alg_17_conc_X_CR - alg_16_conc_X_CRtissue) * D_CRtissue / 1000.0)
        alg_18_P_CR = 1.0 / C_CR * (V_CR - V_CR_0)
        alg_19_F_CR = (alg_11_P_AO - alg_18_P_CR) / R_CR
        alg_21_P_AR = 1.0 / C_AR * (V_AR - V_AR_0)
        alg_22_F_AR = (alg_11_P_AO - alg_21_P_AR) / R_AR
        alg_10_conc_X_AO = sub_X_AO / V_AO
        d_sub_X_AO = torch.zeros_like(y[:, 0]) + ((alg_6_conc_X_LV * alg_12_F_AO - alg_10_conc_X_AO * (alg_19_F_CR + alg_22_F_AR)) / 1000.0)
        d_V_AO = torch.zeros_like(y[:, 0]) + ((alg_12_F_AO - alg_22_F_AR - alg_19_F_CR) / 1000.0)
        alg_30_P_GI = 1.0 / C_GI * (V_GI - V_GI_0)
        alg_31_F_GI = (alg_21_P_AR - alg_30_P_GI) / R_GI
        alg_20_conc_X_AR = sub_X_AR / V_AR
        alg_33_P_LI = 1.0 / C_LI * (V_LI - V_LI_0)
        alg_34_F_GIV = (alg_30_P_GI - alg_33_P_LI) / R_GIV
        alg_29_conc_X_GI = sub_X_GI / V_GI
        d_sub_X_GI = torch.zeros_like(y[:, 0]) + ((alg_20_conc_X_AR * alg_31_F_GI - alg_29_conc_X_GI * alg_34_F_GIV) / 1000.0 * 0.001)
        d_V_GI = torch.zeros_like(y[:, 0]) + ((alg_31_F_GI - alg_34_F_GIV) / 1000.0 * 0.001)
        alg_24_P_AD = 1.0 / C_AD * (V_AD - V_AD_0)
        alg_25_F_AD = (alg_21_P_AR - alg_24_P_AD) / R_AD
        alg_27_P_MU = 1.0 / C_MU * (V_MU - V_MU_0)
        alg_28_F_MU = (alg_21_P_AR - alg_27_P_MU) / R_MU
        alg_35_F_LI = (alg_21_P_AR - alg_33_P_LI) / R_LI
        alg_37_P_KI = 1.0 / C_KI * (V_KI - V_KI_0)
        alg_38_F_KI = (alg_21_P_AR - alg_37_P_KI) / R_KI
        alg_43_P_OT = 1.0 / C_OT * (V_OT - V_OT_0)
        alg_44_F_OT = (alg_21_P_AR - alg_43_P_OT) / R_OT
        alg_40_P_SK = 1.0 / C_SK * (V_SK - V_SK_0)
        alg_41_F_SK = (alg_21_P_AR - alg_40_P_SK) / R_SK
        d_sub_X_AR = torch.zeros_like(y[:, 0]) + ((alg_10_conc_X_AO * alg_22_F_AR - alg_20_conc_X_AR * (alg_41_F_SK + alg_25_F_AD + alg_28_F_MU + alg_31_F_GI + alg_35_F_LI + alg_38_F_KI + alg_44_F_OT)) / 1000.0)
        d_V_AR = torch.zeros_like(y[:, 0]) + ((alg_22_F_AR - alg_25_F_AD - alg_28_F_MU - alg_31_F_GI - alg_35_F_LI - alg_38_F_KI - alg_44_F_OT - alg_41_F_SK) / 1000.0)
        alg_46_P_VE = 1.0 / C_VE * (V_VE - V_VE_0)
        alg_47_F_ADV = (alg_24_P_AD - alg_46_P_VE) / R_ADV
        alg_23_conc_X_AD = sub_X_AD / V_AD
        d_sub_X_AD = torch.zeros_like(y[:, 0]) + ((alg_20_conc_X_AR * alg_25_F_AD - alg_23_conc_X_AD * alg_47_F_ADV) / 1000.0 * 0.001)
        d_V_AD = torch.zeros_like(y[:, 0]) + ((alg_25_F_AD - alg_47_F_ADV) / 1000.0 * 0.001)
        alg_48_F_MUV = (alg_27_P_MU - alg_46_P_VE) / R_MUV
        alg_26_conc_X_MU = sub_X_MU / V_MU
        d_sub_X_MU = torch.zeros_like(y[:, 0]) + ((alg_20_conc_X_AR * alg_28_F_MU - alg_26_conc_X_MU * alg_48_F_MUV) / 1000.0 * 0.001)
        d_V_MU = torch.zeros_like(y[:, 0]) + ((alg_28_F_MU - alg_48_F_MUV) / 1000.0 * 0.001)
        alg_49_F_LIV = (alg_33_P_LI - alg_46_P_VE) / R_LIV
        alg_32_conc_X_LI = sub_X_LI / V_LI
        d_sub_X_LI = torch.zeros_like(y[:, 0]) + ((alg_29_conc_X_GI * alg_34_F_GIV + alg_20_conc_X_AR * alg_35_F_LI - alg_32_conc_X_LI * alg_49_F_LIV + alg_32_conc_X_LI * metabolism) / 1000.0 * 0.001)
        d_V_LI = torch.zeros_like(y[:, 0]) + ((alg_34_F_GIV + alg_35_F_LI - alg_49_F_LIV) / 1000.0 * 0.001)
        alg_50_F_KIV = (alg_37_P_KI - alg_46_P_VE) / R_KIV
        alg_36_conc_X_KI = sub_X_KI / V_KI
        d_sub_X_KI = torch.zeros_like(y[:, 0]) + ((alg_20_conc_X_AR * alg_38_F_KI - alg_36_conc_X_KI * alg_50_F_KIV) / 1000.0 * 0.001)
        d_V_KI = torch.zeros_like(y[:, 0]) + ((alg_38_F_KI - alg_50_F_KIV) / 1000.0 * 0.001)
        alg_51_F_SKV = (alg_40_P_SK - alg_46_P_VE) / R_SKV
        alg_39_conc_X_SK = sub_X_SK / V_SK
        d_sub_X_SK = torch.zeros_like(y[:, 0]) + ((alg_20_conc_X_AR * alg_41_F_SK - alg_39_conc_X_SK * alg_51_F_SKV) / 1000.0 * 0.001)
        d_V_SK = torch.zeros_like(y[:, 0]) + ((alg_41_F_SK - alg_51_F_SKV) / 1000.0 * 0.001)
        alg_52_F_OTV = (alg_43_P_OT - alg_46_P_VE) / R_OTV
        alg_42_conc_X_OT = sub_X_OT / V_OT
        d_sub_X_OT = torch.zeros_like(y[:, 0]) + ((alg_20_conc_X_AR * alg_44_F_OT - alg_42_conc_X_OT * alg_52_F_OTV) / 1000.0 * 0.001)
        d_V_OT = torch.zeros_like(y[:, 0]) + ((alg_44_F_OT - alg_52_F_OTV) / 1000.0 * 0.001)
        alg_55_P_VC = 1.0 / C_VC * (V_VC - V_VC_0)
        alg_56_F_CRV = (alg_18_P_CR - alg_55_P_VC) / R_CRV
        d_sub_X_CR = torch.zeros_like(y[:, 0]) + ((alg_10_conc_X_AO * alg_19_F_CR - alg_17_conc_X_CR * alg_56_F_CRV - (alg_17_conc_X_CR - alg_16_conc_X_CRtissue) * D_CRtissue) / 1000.0)
        d_V_CR = torch.zeros_like(y[:, 0]) + ((alg_19_F_CR - alg_56_F_CRV) / 1000.0)
        alg_57_F_VEV = (alg_46_P_VE - alg_55_P_VC) / R_VEV
        alg_45_conc_X_VE = sub_X_VE / V_VE
        d_sub_X_VE = torch.zeros_like(y[:, 0]) + ((alg_39_conc_X_SK * alg_51_F_SKV + alg_23_conc_X_AD * alg_47_F_ADV + alg_26_conc_X_MU * alg_48_F_MUV + alg_32_conc_X_LI * alg_49_F_LIV + alg_36_conc_X_KI * alg_50_F_KIV + alg_42_conc_X_OT * alg_52_F_OTV - alg_45_conc_X_VE * alg_57_F_VEV) / 1000.0)
        alg_53_F_VE = alg_51_F_SKV + alg_47_F_ADV + alg_48_F_MUV + alg_49_F_LIV + alg_50_F_KIV + alg_52_F_OTV
        d_V_VE = torch.zeros_like(y[:, 0]) + ((alg_53_F_VE - alg_57_F_VEV) / 1000.0)
        alg_59_beattime_3 = (t - torch.floor(t / hrf) * hrf) / 1000.0
        alg_60_E_RA = torch.where((alg_59_beattime_3 * 0.001 >= 0.0) & (alg_59_beattime_3 * 0.001 <= Ts_3), Edia_3 + (Esys_3 - Edia_3) * (1.0 - torch.cos(math.pi * (alg_59_beattime_3 * 0.001) / Ts_3)) / 2.0, torch.where((alg_59_beattime_3 * 0.001 < 1.5 * Ts_3) & (alg_59_beattime_3 * 0.001 >= Ts_3), Edia_3 + (Esys_3 - Edia_3) * (1.0 + torch.cos(2.0 * math.pi * (alg_59_beattime_3 * 0.001 - Ts_3) / Ts_3)) / 2.0, Edia_3))
        alg_62_P_RA = alg_60_E_RA * (V_RA - V_RA_0) + P_RA_ext
        alg_63_F_COV = (alg_14_P_CO - alg_62_P_RA) / R_COV
        alg_13_conc_X_CO = sub_X_CO / V_CO
        d_sub_X_CO = torch.zeros_like(y[:, 0]) + ((alg_6_conc_X_LV * alg_15_F_CO - alg_13_conc_X_CO * alg_63_F_COV) / 1000.0)
        d_V_CO = torch.zeros_like(y[:, 0]) + ((alg_15_F_CO - alg_63_F_COV) / 1000.0)
        alg_64_F_VCV = (alg_55_P_VC - alg_62_P_RA) / R_VCV
        alg_54_conc_X_VC = sub_X_VC / V_VC
        d_sub_X_VC = torch.zeros_like(y[:, 0]) + ((alg_45_conc_X_VE * alg_57_F_VEV + alg_17_conc_X_CR * alg_56_F_CRV - alg_54_conc_X_VC * alg_64_F_VCV) / 1000.0)
        alg_58_F_VC = alg_56_F_CRV + alg_57_F_VEV
        d_V_VC = torch.zeros_like(y[:, 0]) + ((alg_58_F_VC - alg_64_F_VCV) / 1000.0)
        alg_66_beattime_4 = (t - torch.floor(t / hrf) * hrf) / 1000.0 - PRinterval
        alg_67_E_RV = torch.where((alg_66_beattime_4 * 0.001 >= 0.0) & (alg_66_beattime_4 * 0.001 <= Ts_4), Edia_4 + (Esys_4 - Edia_4) * (1.0 - torch.cos(math.pi * (alg_66_beattime_4 * 0.001) / Ts_4)) / 2.0, torch.where((alg_66_beattime_4 * 0.001 < 1.5 * Ts_4) & (alg_66_beattime_4 * 0.001 >= Ts_4), Edia_4 + (Esys_4 - Edia_4) * (1.0 + torch.cos(2.0 * math.pi * (alg_66_beattime_4 * 0.001 - Ts_4) / Ts_4)) / 2.0, Edia_4))
        alg_69_P_RV = alg_67_E_RV * (V_RV - V_RV_0) + P_RV_ext
        alg_70_F_RV = torch.where(alg_62_P_RA > alg_69_P_RV, (alg_62_P_RA - alg_69_P_RV) / R_RV, 0.0)
        alg_61_conc_X_RA = sub_X_RA / V_RA
        d_sub_X_RA = torch.zeros_like(y[:, 0]) + ((alg_54_conc_X_VC * alg_64_F_VCV + alg_13_conc_X_CO * alg_63_F_COV - alg_61_conc_X_RA * alg_70_F_RV) / 1000.0)
        alg_65_F_RA = alg_64_F_VCV + alg_63_F_COV
        d_V_RA = torch.zeros_like(y[:, 0]) + ((alg_65_F_RA - alg_70_F_RV) / 1000.0)
        alg_72_P_PA = 1.0 / C_PA * (V_PA - V_PA_0)
        alg_73_F_PA = torch.where(alg_69_P_RV > alg_72_P_PA, (alg_69_P_RV - alg_72_P_PA) / R_PA, 0.0)
        alg_68_conc_X_RV = sub_X_RV / V_RV
        d_sub_X_RV = torch.zeros_like(y[:, 0]) + ((alg_61_conc_X_RA * alg_70_F_RV - alg_68_conc_X_RV * alg_73_F_PA) / 1000.0)
        d_V_RV = torch.zeros_like(y[:, 0]) + ((alg_70_F_RV - alg_73_F_PA) / 1000.0)
        alg_75_P_Pa = 1.0 / C_Pa * (V_Pa - V_Pa_0)
        alg_76_F_Pa = (alg_72_P_PA - alg_75_P_Pa) / R_Pa
        alg_71_conc_X_PA = sub_X_PA / V_PA
        d_sub_X_PA = torch.zeros_like(y[:, 0]) + ((alg_68_conc_X_RV * alg_73_F_PA - alg_71_conc_X_PA * alg_76_F_Pa) / 1000.0)
        d_V_PA = torch.zeros_like(y[:, 0]) + ((alg_73_F_PA - alg_76_F_Pa) / 1000.0)
        alg_78_P_PC = 1.0 / C_PC * (V_PC - V_PC_0)
        alg_80_F_PC = (alg_75_P_Pa - alg_78_P_PC) / R_PC
        alg_79_F_SH = (alg_75_P_Pa - alg_78_P_PC) / R_SH
        alg_74_conc_X_Pa = sub_X_Pa / V_Pa
        d_sub_X_Pa = torch.zeros_like(y[:, 0]) + ((alg_71_conc_X_PA * alg_76_F_Pa - alg_74_conc_X_Pa * (alg_80_F_PC + alg_79_F_SH)) / 1000.0)
        d_V_Pa = torch.zeros_like(y[:, 0]) + ((alg_76_F_Pa - alg_80_F_PC - alg_79_F_SH) / 1000.0)
        alg_82_P_PV = 1.0 / C_PV * (V_PV - V_PV_0)
        alg_83_F_LA = (alg_82_P_PV - alg_3_P_LA) / R_LA
        alg_81_conc_X_PV = sub_X_PV / V_PV
        d_sub_X_LA = torch.zeros_like(y[:, 0]) + ((alg_81_conc_X_PV * alg_83_F_LA - alg_2_conc_X_LA * alg_9_F_LV) / 1000.0)
        d_V_LA = torch.zeros_like(y[:, 0]) + ((alg_83_F_LA - alg_9_F_LV) / 1000.0)
        alg_84_F_PCV = (alg_78_P_PC - alg_82_P_PV) / R_PCV
        alg_77_conc_X_PC = sub_X_PC / V_PC
        d_sub_X_PC = torch.zeros_like(y[:, 0]) + ((alg_74_conc_X_Pa * alg_80_F_PC - alg_77_conc_X_PC * alg_84_F_PCV) / 1000.0)
        d_V_PC = torch.zeros_like(y[:, 0]) + ((alg_80_F_PC - alg_84_F_PCV) / 1000.0)
        d_sub_X_PV = torch.zeros_like(y[:, 0]) + ((alg_77_conc_X_PC * alg_84_F_PCV + alg_74_conc_X_Pa * alg_79_F_SH - alg_81_conc_X_PV * alg_83_F_LA) / 1000.0)
        alg_85_F_PV = alg_84_F_PCV + alg_79_F_SH
        d_V_PV = torch.zeros_like(y[:, 0]) + ((alg_85_F_PV - alg_83_F_LA) / 1000.0)

        return torch.stack(
            [d_V_LA, d_sub_X_LA, d_V_LV, d_sub_X_LV, d_V_AO, d_F_AOalways, d_sub_X_AO, d_V_CO, d_sub_X_CO, d_V_CR, d_sub_X_CR, d_sub_X_CRtissue, d_V_AR, d_sub_X_AR, d_V_AD, d_sub_X_AD, d_V_MU, d_sub_X_MU, d_V_GI, d_sub_X_GI, d_V_LI, d_sub_X_LI, d_V_KI, d_sub_X_KI, d_V_SK, d_sub_X_SK, d_V_OT, d_sub_X_OT, d_V_VE, d_sub_X_VE, d_V_VC, d_sub_X_VC, d_V_RA, d_sub_X_RA, d_V_RV, d_sub_X_RV, d_V_PA, d_sub_X_PA, d_V_Pa, d_sub_X_Pa, d_V_PC, d_sub_X_PC, d_V_PV, d_sub_X_PV],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
