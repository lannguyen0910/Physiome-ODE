import torch
import torch.nn as nn
import math


class ElectrophysiologyMaltsev2009PaperODE(nn.Module):
    """
    PyTorch translation of electrophysiology/maltsev_2009_paper/model.py.
    Dynamic states: 29; independent constants: 75; derived constants used by rates: 11.
    Original constant slots: 86; algebraic slots: 76.
    """
    source_file = 'electrophysiology/maltsev_2009_paper/model.py'
    source_state_dim = 29
    source_constant_dim = 86
    algebraic_dim = 76
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
        self.parameter_dim = 75

        self.parameters = (
            parameters
            if parameters is not None
            else {'Cm': 32, 'R': 8314.4, 'T': 310.15, 'F': 96485, 'Nai': 10, 'Nao': 140, 'Ki': 140, 'Ko': 5.4, 'E_CaL': 45, 'g_CaL': 0.464, 'alpha_fCa': 0.021, 'Km_fCa': 0.00035, 'g_CaT': 0.1832, 'E_CaT': 45, 'g_Kr': 0.08113973, 'g_Ks': 0.0259, 'g_to': 0.252, 'g_sus': 0.02, 'g_if': 0.15, 'VIf_half': -64, 'g_st': 0.003, 'E_st': 37.4, 'g_b_Na': 0.00486, 'Km_Kp': 1.4, 'Km_Nap': 14, 'i_NaK_max': 2.88, 'g_b_Ca': 0.0006, 'kNaCa': 187.5, 'Qci': 0.1369, 'Qn': 0.4315, 'Qco': 0, 'K3ni': 26.44, 'Kci': 0.0207, 'K1ni': 395.3, 'K2ni': 2.289, 'Kcni': 26.44, 'K3no': 4.663, 'K1no': 1628, 'K2no': 561.4, 'Kco': 3.663, 'Cao': 2, 'ks': 250000, 'MaxSR': 15, 'MinSR': 1, 'EC50_SR': 0.45, 'HSR': 2.5, 'koCa': 10, 'kiCa': 0.5, 'kim': 0.005, 'kom': 0.06, 'tau_dif_Ca': 0.04, 'tau_tr': 40, 'P_up': 0.012, 'K_up': 0.0006, 'TC_tot': 0.031, 'TMC_tot': 0.062, 'CM_tot': 0.045, 'CQ_tot': 10, 'kf_TC': 88.8, 'kf_TMM': 2.277, 'kf_TMC': 227.7, 'kf_CM': 227.7, 'kf_CQ': 0.534, 'kb_TC': 0.446, 'kb_TMC': 0.00751, 'kb_TMM': 0.751, 'kb_CM': 0.542, 'kb_CQ': 0.445, 'Mgi': 2.5, 'V_jsr_part': 0.0012, 'V_i_part': 0.46, 'V_nsr_part': 0.0116, 'R_cell': 4, 'L_cell': 70, 'L_sub': 0.02}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Vm': -65, 'Ca_sub': 0.000223, 'dL': 0, 'fL': 1, 'fCa': 1, 'dT': 0, 'fT': 1, 'paS': 0, 'paF': 0, 'pi': 1, 'n': 0, 'q': 1, 'r': 0, 'y': 1, 'qa': 0, 'qi': 1, 'R': 0.7499955, 'O': 3.4e-06, 'I': 1.1e-06, 'RI': 0.25, 'Ca_jsr': 0.029, 'Ca_nsr': 1.35, 'Cai': 0.0001, 'fTMM': 0.69, 'fCMi': 0.042, 'fCMs': 0.089, 'fTC': 0.02, 'fTMC': 0.22, 'fCQ': 0.032}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Cm', 'R', 'T', 'F', 'Nai', 'Nao', 'Ki', 'Ko', 'E_CaL', 'g_CaL', 'alpha_fCa', 'Km_fCa', 'g_CaT', 'E_CaT', 'g_Kr', 'g_Ks', 'g_to', 'g_sus', 'g_if', 'VIf_half', 'g_st', 'E_st', 'g_b_Na', 'Km_Kp', 'Km_Nap', 'i_NaK_max', 'g_b_Ca', 'kNaCa', 'Qci', 'Qn', 'Qco', 'K3ni', 'Kci', 'K1ni', 'K2ni', 'Kcni', 'K3no', 'K1no', 'K2no', 'Kco', 'Cao', 'ks', 'MaxSR', 'MinSR', 'EC50_SR', 'HSR', 'koCa', 'kiCa', 'kim', 'kom', 'tau_dif_Ca', 'tau_tr', 'P_up', 'K_up', 'TC_tot', 'TMC_tot', 'CM_tot', 'CQ_tot', 'kf_TC', 'kf_TMM', 'kf_TMC', 'kf_CM', 'kf_CQ', 'kb_TC', 'kb_TMC', 'kb_TMM', 'kb_CM', 'kb_CQ', 'Mgi', 'V_jsr_part', 'V_i_part', 'V_nsr_part', 'R_cell', 'L_cell', 'L_sub']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Vm', 'Ca_sub', 'dL', 'fL', 'fCa', 'dT', 'fT', 'paS', 'paF', 'pi', 'n', 'q', 'r', 'y', 'qa', 'qi', 'R', 'O', 'I', 'RI', 'Ca_jsr', 'Ca_nsr', 'Cai', 'fTMM', 'fCMi', 'fCMs', 'fTC', 'fTMC', 'fCQ']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (29)
        Vm = y[:, 0]
        Ca_sub = y[:, 1]
        dL = y[:, 2]
        fL = y[:, 3]
        fCa = y[:, 4]
        dT = y[:, 5]
        fT = y[:, 6]
        paS = y[:, 7]
        paF = y[:, 8]
        pi = y[:, 9]
        n = y[:, 10]
        q = y[:, 11]
        r = y[:, 12]
        state_y = y[:, 13]
        qa = y[:, 14]
        qi = y[:, 15]
        R = y[:, 16]
        O = y[:, 17]
        I = y[:, 18]
        RI = y[:, 19]
        Ca_jsr = y[:, 20]
        Ca_nsr = y[:, 21]
        Cai = y[:, 22]
        fTMM = y[:, 23]
        fCMi = y[:, 24]
        fCMs = y[:, 25]
        fTC = y[:, 26]
        fTMC = y[:, 27]
        fCQ = y[:, 28]

        # Independent parameters (source constant order)
        Cm = theta[:, 0]  # constants[0]
        const_R = theta[:, 1]  # constants[1]
        T = theta[:, 2]  # constants[2]
        F = theta[:, 3]  # constants[3]
        Nai = theta[:, 4]  # constants[4]
        Nao = theta[:, 5]  # constants[5]
        Ki = theta[:, 6]  # constants[6]
        Ko = theta[:, 7]  # constants[7]
        E_CaL = theta[:, 8]  # constants[8]
        g_CaL = theta[:, 9]  # constants[9]
        alpha_fCa = theta[:, 10]  # constants[10]
        Km_fCa = theta[:, 11]  # constants[11]
        g_CaT = theta[:, 12]  # constants[12]
        E_CaT = theta[:, 13]  # constants[13]
        g_Kr = theta[:, 14]  # constants[14]
        g_Ks = theta[:, 15]  # constants[15]
        g_to = theta[:, 16]  # constants[16]
        g_sus = theta[:, 17]  # constants[17]
        g_if = theta[:, 18]  # constants[18]
        VIf_half = theta[:, 19]  # constants[19]
        g_st = theta[:, 20]  # constants[20]
        E_st = theta[:, 21]  # constants[21]
        g_b_Na = theta[:, 22]  # constants[22]
        Km_Kp = theta[:, 23]  # constants[23]
        Km_Nap = theta[:, 24]  # constants[24]
        i_NaK_max = theta[:, 25]  # constants[25]
        g_b_Ca = theta[:, 26]  # constants[26]
        kNaCa = theta[:, 27]  # constants[27]
        Qci = theta[:, 28]  # constants[28]
        Qn = theta[:, 29]  # constants[29]
        Qco = theta[:, 30]  # constants[30]
        K3ni = theta[:, 31]  # constants[31]
        Kci = theta[:, 32]  # constants[32]
        K1ni = theta[:, 33]  # constants[33]
        K2ni = theta[:, 34]  # constants[34]
        Kcni = theta[:, 35]  # constants[35]
        K3no = theta[:, 36]  # constants[36]
        K1no = theta[:, 37]  # constants[37]
        K2no = theta[:, 38]  # constants[38]
        Kco = theta[:, 39]  # constants[39]
        Cao = theta[:, 40]  # constants[40]
        ks = theta[:, 41]  # constants[41]
        MaxSR = theta[:, 42]  # constants[42]
        MinSR = theta[:, 43]  # constants[43]
        EC50_SR = theta[:, 44]  # constants[44]
        HSR = theta[:, 45]  # constants[45]
        koCa = theta[:, 46]  # constants[46]
        kiCa = theta[:, 47]  # constants[47]
        kim = theta[:, 48]  # constants[48]
        kom = theta[:, 49]  # constants[49]
        tau_dif_Ca = theta[:, 50]  # constants[50]
        tau_tr = theta[:, 51]  # constants[51]
        P_up = theta[:, 52]  # constants[52]
        K_up = theta[:, 53]  # constants[53]
        TC_tot = theta[:, 54]  # constants[54]
        TMC_tot = theta[:, 55]  # constants[55]
        CM_tot = theta[:, 56]  # constants[56]
        CQ_tot = theta[:, 57]  # constants[57]
        kf_TC = theta[:, 58]  # constants[58]
        kf_TMM = theta[:, 59]  # constants[59]
        kf_TMC = theta[:, 60]  # constants[60]
        kf_CM = theta[:, 61]  # constants[61]
        kf_CQ = theta[:, 62]  # constants[62]
        kb_TC = theta[:, 63]  # constants[63]
        kb_TMC = theta[:, 64]  # constants[64]
        kb_TMM = theta[:, 65]  # constants[65]
        kb_CM = theta[:, 66]  # constants[66]
        kb_CQ = theta[:, 67]  # constants[67]
        Mgi = theta[:, 68]  # constants[68]
        V_jsr_part = theta[:, 69]  # constants[69]
        V_i_part = theta[:, 70]  # constants[70]
        V_nsr_part = theta[:, 71]  # constants[71]
        R_cell = theta[:, 72]  # constants[72]
        L_cell = theta[:, 73]  # constants[73]
        L_sub = theta[:, 74]  # constants[74]

        # Derived constants from the original initConsts()
        E_Na = const_R * T / F * torch.log(Nao / Nai)  # constants[75]
        E_K = const_R * T / F * torch.log(Ko / Ki)  # constants[76]
        E_Ks = const_R * T / F * torch.log((Ko + 0.12 * Nao) / (Ki + 0.12 * Nai))  # constants[77]
        RTOnF = const_R * T / F  # constants[78]
        k43 = Nai / (K3ni + Nai)  # constants[79]
        k34 = Nao / (K3no + Nao)  # constants[80]
        V_cell = 0.001 * math.pi * R_cell ** 2.0 * L_cell  # constants[81]
        V_sub = 0.001 * 2.0 * math.pi * L_sub * (R_cell - L_sub / 2.0) * L_cell  # constants[82]
        V_jsr = V_jsr_part * V_cell  # constants[83]
        V_i = V_i_part * V_cell - V_sub  # constants[84]
        V_nsr = V_nsr_part * V_cell  # constants[85]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Vm = torch.zeros_like(y[:, 0])
        d_Ca_sub = torch.zeros_like(y[:, 0])
        d_dL = torch.zeros_like(y[:, 0])
        d_fL = torch.zeros_like(y[:, 0])
        d_fCa = torch.zeros_like(y[:, 0])
        d_dT = torch.zeros_like(y[:, 0])
        d_fT = torch.zeros_like(y[:, 0])
        d_paS = torch.zeros_like(y[:, 0])
        d_paF = torch.zeros_like(y[:, 0])
        d_pi = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_q = torch.zeros_like(y[:, 0])
        d_r = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])
        d_qa = torch.zeros_like(y[:, 0])
        d_qi = torch.zeros_like(y[:, 0])
        d_R = torch.zeros_like(y[:, 0])
        d_O = torch.zeros_like(y[:, 0])
        d_I = torch.zeros_like(y[:, 0])
        d_RI = torch.zeros_like(y[:, 0])
        d_Ca_jsr = torch.zeros_like(y[:, 0])
        d_Ca_nsr = torch.zeros_like(y[:, 0])
        d_Cai = torch.zeros_like(y[:, 0])
        d_fTMM = torch.zeros_like(y[:, 0])
        d_fCMi = torch.zeros_like(y[:, 0])
        d_fCMs = torch.zeros_like(y[:, 0])
        d_fTC = torch.zeros_like(y[:, 0])
        d_fTMC = torch.zeros_like(y[:, 0])
        d_fCQ = torch.zeros_like(y[:, 0])
        alg_0_dL_infinity = torch.zeros_like(y[:, 0])
        alg_1_fL_infinity = torch.zeros_like(y[:, 0])
        alg_2_fCa_infinity = torch.zeros_like(y[:, 0])
        alg_3_dT_infinity = torch.zeros_like(y[:, 0])
        alg_4_fT_infinity = torch.zeros_like(y[:, 0])
        alg_5_pa_infinity = torch.zeros_like(y[:, 0])
        alg_6_pi_infinity = torch.zeros_like(y[:, 0])
        alg_7_alpha_n = torch.zeros_like(y[:, 0])
        alg_8_q_infinity = torch.zeros_like(y[:, 0])
        alg_9_r_infinity = torch.zeros_like(y[:, 0])
        alg_10_y_infinity = torch.zeros_like(y[:, 0])
        alg_11_qa_infinity = torch.zeros_like(y[:, 0])
        alg_12_alpha_qi = torch.zeros_like(y[:, 0])
        alg_13_kCaSR = torch.zeros_like(y[:, 0])
        alg_14_i_CaL = torch.zeros_like(y[:, 0])
        alg_15_delta_fTMM = torch.zeros_like(y[:, 0])
        alg_16_adVm = torch.zeros_like(y[:, 0])
        alg_17_tau_fL = torch.zeros_like(y[:, 0])
        alg_18_tau_fCa = torch.zeros_like(y[:, 0])
        alg_19_tau_dT = torch.zeros_like(y[:, 0])
        alg_20_tau_fT = torch.zeros_like(y[:, 0])
        alg_21_tau_paS = torch.zeros_like(y[:, 0])
        alg_22_tau_paF = torch.zeros_like(y[:, 0])
        alg_23_tau_pi = torch.zeros_like(y[:, 0])
        alg_24_beta_n = torch.zeros_like(y[:, 0])
        alg_25_tau_q = torch.zeros_like(y[:, 0])
        alg_26_tau_r = torch.zeros_like(y[:, 0])
        alg_27_tau_y = torch.zeros_like(y[:, 0])
        alg_28_alpha_qa = torch.zeros_like(y[:, 0])
        alg_29_beta_qi = torch.zeros_like(y[:, 0])
        alg_30_koSRCa = torch.zeros_like(y[:, 0])
        alg_31_i_CaT = torch.zeros_like(y[:, 0])
        alg_32_alpha_dL = torch.zeros_like(y[:, 0])
        alg_33_n_infinity = torch.zeros_like(y[:, 0])
        alg_34_beta_qa = torch.zeros_like(y[:, 0])
        alg_35_qi_infinity = torch.zeros_like(y[:, 0])
        alg_36_kiSRCa = torch.zeros_like(y[:, 0])
        alg_37_i_Kr = torch.zeros_like(y[:, 0])
        alg_38_bdVm = torch.zeros_like(y[:, 0])
        alg_39_tau_n = torch.zeros_like(y[:, 0])
        alg_40_tau_qa = torch.zeros_like(y[:, 0])
        alg_41_tau_qi = torch.zeros_like(y[:, 0])
        alg_42_i_Ks = torch.zeros_like(y[:, 0])
        alg_43_beta_dL = torch.zeros_like(y[:, 0])
        alg_44_i_to = torch.zeros_like(y[:, 0])
        alg_45_tau_dL = torch.zeros_like(y[:, 0])
        alg_46_i_sus = torch.zeros_like(y[:, 0])
        alg_47_i_f_Na = torch.zeros_like(y[:, 0])
        alg_48_i_f_K = torch.zeros_like(y[:, 0])
        alg_49_i_f = torch.zeros_like(y[:, 0])
        alg_50_i_st = torch.zeros_like(y[:, 0])
        alg_51_i_b_Na = torch.zeros_like(y[:, 0])
        alg_52_i_NaK = torch.zeros_like(y[:, 0])
        alg_53_i_b_Ca = torch.zeros_like(y[:, 0])
        alg_54_k41 = torch.zeros_like(y[:, 0])
        alg_55_di = torch.zeros_like(y[:, 0])
        alg_56_k12 = torch.zeros_like(y[:, 0])
        alg_57_k14 = torch.zeros_like(y[:, 0])
        alg_58_k32 = torch.zeros_like(y[:, 0])
        alg_59_x2 = torch.zeros_like(y[:, 0])
        alg_60_do = torch.zeros_like(y[:, 0])
        alg_61_k21 = torch.zeros_like(y[:, 0])
        alg_62_k23 = torch.zeros_like(y[:, 0])
        alg_63_x1 = torch.zeros_like(y[:, 0])
        alg_64_x3 = torch.zeros_like(y[:, 0])
        alg_65_x4 = torch.zeros_like(y[:, 0])
        alg_66_i_NaCa = torch.zeros_like(y[:, 0])
        alg_67_j_SRCarel = torch.zeros_like(y[:, 0])
        alg_68_j_Ca_dif = torch.zeros_like(y[:, 0])
        alg_69_delta_fCMs = torch.zeros_like(y[:, 0])
        alg_70_j_up = torch.zeros_like(y[:, 0])
        alg_71_delta_fTC = torch.zeros_like(y[:, 0])
        alg_72_j_tr = torch.zeros_like(y[:, 0])
        alg_73_delta_fTMC = torch.zeros_like(y[:, 0])
        alg_74_delta_fCQ = torch.zeros_like(y[:, 0])
        alg_75_delta_fCMi = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_15_delta_fTMM = kf_TMM * Mgi * (1.0 - (fTMC + fTMM)) - kb_TMM * fTMM
        d_fTMM = torch.zeros_like(y[:, 0]) + (alg_15_delta_fTMM)
        alg_1_fL_infinity = 1.0 / (1.0 + torch.exp((Vm + 35.0) / 7.3))
        alg_17_tau_fL = 44.3 + 257.1 * torch.exp(-((Vm + 32.5) / 13.9) ** 2.0)
        d_fL = torch.zeros_like(y[:, 0]) + ((alg_1_fL_infinity - fL) / alg_17_tau_fL)
        alg_2_fCa_infinity = Km_fCa / (Km_fCa + Ca_sub)
        alg_18_tau_fCa = alg_2_fCa_infinity / alpha_fCa
        d_fCa = torch.zeros_like(y[:, 0]) + ((alg_2_fCa_infinity - fCa) / alg_18_tau_fCa)
        alg_3_dT_infinity = 1.0 / (1.0 + torch.exp(-(Vm + 26.3) / 6.0))
        alg_19_tau_dT = 1.0 / (1.068 * torch.exp((Vm + 26.3) / 30.0) + 1.068 * torch.exp(-(Vm + 26.3) / 30.0))
        d_dT = torch.zeros_like(y[:, 0]) + ((alg_3_dT_infinity - dT) / alg_19_tau_dT)
        alg_4_fT_infinity = 1.0 / (1.0 + torch.exp((Vm + 61.7) / 5.6))
        alg_20_tau_fT = 1.0 / (0.0153 * torch.exp(-(Vm + 61.7) / 83.3) + 0.015 * torch.exp((Vm + 61.7) / 15.38))
        d_fT = torch.zeros_like(y[:, 0]) + ((alg_4_fT_infinity - fT) / alg_20_tau_fT)
        alg_5_pa_infinity = 1.0 / (1.0 + torch.exp(-(Vm + 23.2) / 10.6))
        alg_21_tau_paS = 0.846554 / (0.0042 * torch.exp(Vm / 17.0) + 0.00015 * torch.exp(-Vm / 21.6))
        d_paS = torch.zeros_like(y[:, 0]) + ((alg_5_pa_infinity - paS) / alg_21_tau_paS)
        alg_22_tau_paF = 0.846554 / (0.0372 * torch.exp(Vm / 15.9) + 0.00096 * torch.exp(-Vm / 22.5))
        d_paF = torch.zeros_like(y[:, 0]) + ((alg_5_pa_infinity - paF) / alg_22_tau_paF)
        alg_6_pi_infinity = 1.0 / (1.0 + torch.exp((Vm + 28.6) / 17.1))
        alg_23_tau_pi = 1.0 / (0.1 * torch.exp(-Vm / 54.645) + 0.656 * torch.exp(Vm / 106.157))
        d_pi = torch.zeros_like(y[:, 0]) + ((alg_6_pi_infinity - pi) / alg_23_tau_pi)
        alg_8_q_infinity = 1.0 / (1.0 + torch.exp((Vm + 49.0) / 13.0))
        alg_25_tau_q = 6.06 + 39.102 / (0.57 * torch.exp(-0.08 * (Vm + 44.0)) + 0.065 * torch.exp(0.1 * (Vm + 45.93)))
        d_q = torch.zeros_like(y[:, 0]) + ((alg_8_q_infinity - q) / alg_25_tau_q)
        alg_9_r_infinity = 1.0 / (1.0 + torch.exp(-(Vm - 19.3) / 15.0))
        alg_26_tau_r = 2.75352 + 14.4052 / (1.037 * torch.exp(0.09 * (Vm + 30.61)) + 0.369 * torch.exp(-0.12 * (Vm + 23.84)))
        d_r = torch.zeros_like(y[:, 0]) + ((alg_9_r_infinity - r) / alg_26_tau_r)
        alg_10_y_infinity = 1.0 / (1.0 + torch.exp((Vm - VIf_half) / 13.5))
        alg_27_tau_y = 0.716653 / (torch.exp(-(Vm + 386.9) / 45.302) + torch.exp((Vm - 73.08) / 19.231))
        d_state_y = torch.zeros_like(y[:, 0]) + ((alg_10_y_infinity - state_y) / alg_27_tau_y)
        alg_13_kCaSR = MaxSR - (MaxSR - MinSR) / (1.0 + (EC50_SR / Ca_jsr) ** HSR)
        alg_30_koSRCa = koCa / alg_13_kCaSR
        alg_36_kiSRCa = kiCa * alg_13_kCaSR
        d_R = torch.zeros_like(y[:, 0]) + (kim * RI - alg_36_kiSRCa * Ca_sub * R - (alg_30_koSRCa * Ca_sub ** 2.0 * R - kom * O))
        d_O = torch.zeros_like(y[:, 0]) + (alg_30_koSRCa * Ca_sub ** 2.0 * R - kom * O - (alg_36_kiSRCa * Ca_sub * O - kim * I))
        d_I = torch.zeros_like(y[:, 0]) + (alg_36_kiSRCa * Ca_sub * O - kim * I - (kom * I - alg_30_koSRCa * Ca_sub ** 2.0 * RI))
        d_RI = torch.zeros_like(y[:, 0]) + (kom * I - alg_30_koSRCa * Ca_sub ** 2.0 * RI - (kim * RI - alg_36_kiSRCa * Ca_sub * R))
        alg_7_alpha_n = 0.014 / (1.0 + torch.exp(-(Vm - 40.0) / 9.0))
        alg_24_beta_n = 0.001 * torch.exp(-Vm / 45.0)
        alg_33_n_infinity = alg_7_alpha_n / (alg_7_alpha_n + alg_24_beta_n)
        alg_39_tau_n = 1.0 / (alg_7_alpha_n + alg_24_beta_n)
        d_n = torch.zeros_like(y[:, 0]) + ((alg_33_n_infinity - n) / alg_39_tau_n)
        alg_11_qa_infinity = 1.0 / (1.0 + torch.exp(-(Vm + 57.0) / 5.0))
        alg_28_alpha_qa = 1.0 / (0.15 * torch.exp(-Vm / 11.0) + 0.2 * torch.exp(-Vm / 700.0))
        alg_34_beta_qa = 1.0 / (16.0 * torch.exp(Vm / 8.0) + 15.0 * torch.exp(Vm / 50.0))
        alg_40_tau_qa = 1.0 / (alg_28_alpha_qa + alg_34_beta_qa)
        d_qa = torch.zeros_like(y[:, 0]) + ((alg_11_qa_infinity - qa) / alg_40_tau_qa)
        alg_12_alpha_qi = 1.0 / (3100.0 * torch.exp(Vm / 13.0) + 700.0 * torch.exp(Vm / 70.0))
        alg_29_beta_qi = 1.0 / (95.0 * torch.exp(-Vm / 10.0) + 50.0 * torch.exp(-Vm / 700.0)) + 0.000229 / (1.0 + torch.exp(-Vm / 5.0))
        alg_35_qi_infinity = alg_12_alpha_qi / (alg_12_alpha_qi + alg_29_beta_qi)
        alg_41_tau_qi = 6.65 / (alg_12_alpha_qi + alg_29_beta_qi)
        d_qi = torch.zeros_like(y[:, 0]) + ((alg_35_qi_infinity - qi) / alg_41_tau_qi)
        alg_0_dL_infinity = 1.0 / (1.0 + torch.exp(-(Vm + 13.5) / 6.0))
        alg_16_adVm = torch.where(Vm == -35.0, -35.0, torch.where(Vm == 0.0, 1e-05, Vm))
        alg_32_alpha_dL = -0.02839 * (alg_16_adVm + 35.0) / (torch.exp(-(alg_16_adVm + 35.0) / 2.5) - 1.0) - 0.0849 * alg_16_adVm / (torch.exp(-alg_16_adVm / 4.8) - 1.0)
        alg_38_bdVm = torch.where(Vm == 5.0, 5.00001, Vm)
        alg_43_beta_dL = 0.01143 * (alg_38_bdVm - 5.0) / (torch.exp((alg_38_bdVm - 5.0) / 2.5) - 1.0)
        alg_45_tau_dL = 1.0 / (alg_32_alpha_dL + alg_43_beta_dL)
        d_dL = torch.zeros_like(y[:, 0]) + ((alg_0_dL_infinity - dL) / alg_45_tau_dL)
        alg_31_i_CaT = Cm * g_CaT * (Vm - E_CaT) * dT * fT
        alg_14_i_CaL = Cm * g_CaL * (Vm - E_CaL) * dL * fL * fCa
        alg_47_i_f_Na = Cm * 0.3833 * g_if * (Vm - E_Na) * state_y ** 2.0
        alg_48_i_f_K = Cm * 0.6167 * g_if * (Vm - E_K) * state_y ** 2.0
        alg_49_i_f = alg_47_i_f_Na + alg_48_i_f_K
        alg_50_i_st = Cm * g_st * (Vm - E_st) * qa * qi
        alg_37_i_Kr = Cm * g_Kr * (Vm - E_K) * (0.6 * paF + 0.4 * paS) * pi
        alg_42_i_Ks = Cm * g_Ks * (Vm - E_Ks) * n ** 2.0
        alg_44_i_to = Cm * g_to * (Vm - E_K) * q * r
        alg_46_i_sus = Cm * g_sus * (Vm - E_K) * r
        alg_52_i_NaK = Cm * i_NaK_max / ((1.0 + (Km_Kp / Ko) ** 1.2) * (1.0 + (Km_Nap / Nai) ** 1.3) * (1.0 + torch.exp(-(Vm - E_Na + 120.0) / 30.0)))
        alg_54_k41 = torch.exp(-Qn * Vm / (2.0 * RTOnF))
        alg_60_do = 1.0 + Cao / Kco * (1.0 + torch.exp(Qco * Vm / RTOnF)) + Nao / K1no * (1.0 + Nao / K2no * (1.0 + Nao / K3no))
        alg_62_k23 = Nao / K1no * Nao / K2no * (1.0 + Nao / K3no) * torch.exp(-Qn * Vm / (2.0 * RTOnF)) / alg_60_do
        alg_61_k21 = Cao / Kco * torch.exp(Qco * Vm / RTOnF) / alg_60_do
        alg_58_k32 = torch.exp(Qn * Vm / (2.0 * RTOnF))
        alg_63_x1 = alg_54_k41 * k34 * (alg_62_k23 + alg_61_k21) + alg_61_k21 * alg_58_k32 * (k43 + alg_54_k41)
        alg_55_di = 1.0 + Ca_sub / Kci * (1.0 + torch.exp(-Qci * Vm / RTOnF) + Nai / Kcni) + Nai / K1ni * (1.0 + Nai / K2ni * (1.0 + Nai / K3ni))
        alg_56_k12 = Ca_sub / Kci * torch.exp(-Qci * Vm / RTOnF) / alg_55_di
        alg_57_k14 = Nai / K1ni * Nai / K2ni * (1.0 + Nai / K3ni) * torch.exp(Qn * Vm / (2.0 * RTOnF)) / alg_55_di
        alg_59_x2 = alg_58_k32 * k43 * (alg_57_k14 + alg_56_k12) + alg_54_k41 * alg_56_k12 * (k34 + alg_58_k32)
        alg_64_x3 = alg_57_k14 * k43 * (alg_62_k23 + alg_61_k21) + alg_56_k12 * alg_62_k23 * (k43 + alg_54_k41)
        alg_65_x4 = alg_62_k23 * k34 * (alg_57_k14 + alg_56_k12) + alg_57_k14 * alg_61_k21 * (k34 + alg_58_k32)
        alg_66_i_NaCa = Cm * kNaCa * (alg_59_x2 * alg_61_k21 - alg_63_x1 * alg_56_k12) / (alg_63_x1 + alg_59_x2 + alg_64_x3 + alg_65_x4)
        alg_53_i_b_Ca = Cm * g_b_Ca * (Vm - E_CaL)
        alg_51_i_b_Na = Cm * g_b_Na * (Vm - E_Na)
        d_Vm = torch.zeros_like(y[:, 0]) + (-(alg_14_i_CaL + alg_31_i_CaT + alg_49_i_f + alg_50_i_st + alg_37_i_Kr + alg_42_i_Ks + alg_44_i_to + alg_46_i_sus + alg_52_i_NaK + alg_66_i_NaCa + alg_53_i_b_Ca + alg_51_i_b_Na) / Cm)
        alg_69_delta_fCMs = kf_CM * Ca_sub * (1.0 - fCMs) - kb_CM * fCMs
        d_fCMs = torch.zeros_like(y[:, 0]) + (alg_69_delta_fCMs)
        alg_67_j_SRCarel = ks * O * (Ca_jsr - Ca_sub)
        alg_68_j_Ca_dif = (Ca_sub - Cai) / tau_dif_Ca
        d_Ca_sub = torch.zeros_like(y[:, 0]) + (alg_67_j_SRCarel * V_jsr / V_sub - ((alg_14_i_CaL + alg_31_i_CaT + alg_53_i_b_Ca - 2.0 * alg_66_i_NaCa) / (2.0 * F * V_sub) + alg_68_j_Ca_dif + CM_tot * alg_69_delta_fCMs))
        alg_71_delta_fTC = kf_TC * Cai * (1.0 - fTC) - kb_TC * fTC
        d_fTC = torch.zeros_like(y[:, 0]) + (alg_71_delta_fTC)
        alg_70_j_up = P_up / (1.0 + K_up / Cai)
        alg_72_j_tr = (Ca_nsr - Ca_jsr) / tau_tr
        d_Ca_nsr = torch.zeros_like(y[:, 0]) + (alg_70_j_up - alg_72_j_tr * V_jsr / V_nsr)
        alg_73_delta_fTMC = kf_TMC * Cai * (1.0 - (fTMC + fTMM)) - kb_TMC * fTMC
        d_fTMC = torch.zeros_like(y[:, 0]) + (alg_73_delta_fTMC)
        alg_74_delta_fCQ = kf_CQ * Ca_jsr * (1.0 - fCQ) - kb_CQ * fCQ
        d_fCQ = torch.zeros_like(y[:, 0]) + (alg_74_delta_fCQ)
        d_Ca_jsr = torch.zeros_like(y[:, 0]) + (alg_72_j_tr - (alg_67_j_SRCarel + CQ_tot * alg_74_delta_fCQ))
        alg_75_delta_fCMi = kf_CM * Cai * (1.0 - fCMi) - kb_CM * fCMi
        d_fCMi = torch.zeros_like(y[:, 0]) + (alg_75_delta_fCMi)
        d_Cai = torch.zeros_like(y[:, 0]) + ((alg_68_j_Ca_dif * V_sub - alg_70_j_up * V_nsr) / V_i - (CM_tot * alg_75_delta_fCMi + TC_tot * alg_71_delta_fTC + TMC_tot * alg_73_delta_fTMC))

        return torch.stack(
            [d_Vm, d_Ca_sub, d_dL, d_fL, d_fCa, d_dT, d_fT, d_paS, d_paF, d_pi, d_n, d_q, d_r, d_state_y, d_qa, d_qi, d_R, d_O, d_I, d_RI, d_Ca_jsr, d_Ca_nsr, d_Cai, d_fTMM, d_fCMi, d_fCMs, d_fTC, d_fTMC, d_fCQ],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
