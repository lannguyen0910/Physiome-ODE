import torch
import torch.nn as nn
import math


class ElectrophysiologyNiedererSmith2007ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/niederer_smith_2007/model.py.
    Dynamic states: 26; independent constants: 142; derived constants used by rates: 45.
    Original constant slots: 188; algebraic slots: 140.
    """
    source_file = 'electrophysiology/niederer_smith_2007/model.py'
    source_state_dim = 26
    source_constant_dim = 188
    algebraic_dim = 140
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
        self.parameter_dim = 142

        self.parameters = (
            parameters
            if parameters is not None
            else {'NO_on': 0, 'pH_on': 0, 'SAC_on': 0, 'V_myo': 2.584e-05, 'V_SR': 2.098e-06, 'N': 75000, 'A_cap': 0.01534, 'R': 8314.5, 'T': 295, 'F': 96487, 'CmF': 0.0001, 'stim_period': 1000, 'stim_duration': 10, 'stim_amplitude': -0.0006, 'ExtensionRatio': 1, 'g_SAC': 4.1333e-07, 'E_R': -10, 'g_Ko': 1.2e-06, 'g_Na': 0.0008, 'Na_o': 140, 'g_t': 3.5e-05, 'a_endo': 0.583, 'b_endo': 0.417, 'K_o': 5.4, 'g_ss': 7e-06, 'g_K1': 2.4e-05, 'g_f': 1.45e-06, 'f_Na': 0.2, 'g_B_Na': 8.015e-08, 'g_B_K': 1.38e-07, 'scale_Na': 0, 'scale_K': 0, 'i_NaK_max': 9.5e-05, 'K_m_K': 1.5, 'K_m_Na': 10, 'Pco2': 5.8e-05, 'pH_e': 7.4, 'KA': 21.4935205515214, 'KB': 1.77948608115949e-07, 'kp1': 0.51156085723193, 'km1': 0.14069285629105, 'kp2': 0.001698, 'K_Hi': 0.000416869383470335, 'n_Hi': 2, 'Q_10Scaler': 0.35, 'Cle': 126, 'K_Cl': 17970.36, 'K_OH': 0.0008907, 'kp1_2': 14975.6, 'km1_2': 257.4, 'kp2_2': 4084.9, 'Q_10Scaler_2': 0.35, 'K_Na': 4866.11, 'K_HCO3': 0.008017, 'kp1_3': 4677, 'km1_3': 777.92, 'kp2_3': 6.656, 'K_Hi_2': 0.000183, 'n_Hi_2': 2.91, 'K_He': 6.53e-05, 'n_He': 2.18, 'Q_10Scaler_3': 0.35, 'K_Cl_2': 983.5, 'K_HCO3_2': 110.64, 'kp1_4': 20789.9, 'km1_4': 21118.2, 'kp2_4': 21256.54, 'K_Hi_3': 2.67e-05, 'n_Hi_3': 5.11, 'K_He_2': 0.000312, 'n_He_2': 1.44, 'Q_10Scaler_4': 0.35, 'PP_co2e': 0.05, 'CO_2sol': 0.03253, 'P_atm': 760, 'kf_co2hyd': 0.000365, 'kr_co2hyd': 0.481, 'pKa_ib1': 6.40013149605198, 'ib1': 31.2044223705215, 'pKa_ib2': 7.48048792987277, 'ib2': 6.84513535697271, 'pH_scale': 1, 'g_Cl': 2e-05, 'g_H': 4e-06, 'J_CO2': 0, 'Ca_o': 2, 'g_pCa': 3.5e-06, 'K_mpCa': 0.0005, 'delta': 0.075, 'g_Cab': 2.4216e-08, 'K_mNa': 87.5, 'K_mCa': 1.38, 'eta': 0.35, 'k_sat': 0.1, 'g_NCX': 0.0385, 'g_SERCA': 0.00045, 'K_SERCA': 0.0005, 'B_CMDN': 0.05, 'K_CMDN': 0.00238, 'g_SRl': 5.2e-06, 'B_TRPN': 0.07, 'g_D': 6.5e-11, 'J_R': 2e-11, 'J_L': 9.13e-13, 'V_L0': -2, 'delta_VL': 7, 't_L': 1, 'a': 0.0625, 'b': 14, 'tau_L': 650, 'K_L': 0.00022, 'phi_L': 2.35, 'phi_R_base': 0.05, 'tau_R': 2.43, 'theta_R': 0.012, 'K_RyR': 0.041, 'c': 0.01, 'd': 100, 'k_on': 100, 'k_off': 0.2, 'gamma_trpn': 2, 'TRPN_tot': 0.07, 'T_ref': 56.2, 'alpha_0': 0.008, 'alpha_r1': 0.002, 'alpha_r2': 0.00175, 'n_Rel': 3, 'K_z': 0.15, 'n_Hill': 3, 'Ca_50ref': 0.00105, 'z_p': 0.85, 'beta_1': -4, 'beta_0': 4.9, 'a_2': 0.35, 'A_1': -29, 'A_2': 138, 'A_3': 129, 'alpha_1': 0.03, 'alpha_2': 0.13, 'alpha_3': 0.625, 'tau_s_ss': 2100, 'dExtensionRatiodt': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -80.5561112771341, 'Na_i': 10.9735342589175, 'm': 0.00419283833975832, 'h': 0.675647510010602, 'j': 0.675294627571238, 'K_i': 142.021261491871, 'r': 0.00218107055088884, 's': 0.922573773425153, 's_slow': 0.530207467628341, 'r_ss': 0.00289374682707489, 's_ss': 0.308572839431787, 'y': 0.00331726547133295, 'CO2i': 1.23610845162584, 'pH_i': 7.20905044774657, 'Cli': 16.2285651746901, 'HCO3i': 15.1428688811927, 'Ca_i': 8.60587059887361e-05, 'Ca_SR': 0.843733859307907, 'TRPN': 0.0670845085340179, 'z_1': 0.988955429362334, 'z_2': 0.00860686643357905, 'z_3': 0.00241692704700704, 'z': 0.0175189833295748, 'Q_1': 0, 'Q_2': 0, 'Q_3': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['NO_on', 'pH_on', 'SAC_on', 'V_myo', 'V_SR', 'N', 'A_cap', 'R', 'T', 'F', 'CmF', 'stim_period', 'stim_duration', 'stim_amplitude', 'ExtensionRatio', 'g_SAC', 'E_R', 'g_Ko', 'g_Na', 'Na_o', 'g_t', 'a_endo', 'b_endo', 'K_o', 'g_ss', 'g_K1', 'g_f', 'f_Na', 'g_B_Na', 'g_B_K', 'scale_Na', 'scale_K', 'i_NaK_max', 'K_m_K', 'K_m_Na', 'Pco2', 'pH_e', 'KA', 'KB', 'kp1', 'km1', 'kp2', 'K_Hi', 'n_Hi', 'Q_10Scaler', 'Cle', 'K_Cl', 'K_OH', 'kp1_2', 'km1_2', 'kp2_2', 'Q_10Scaler_2', 'K_Na', 'K_HCO3', 'kp1_3', 'km1_3', 'kp2_3', 'K_Hi_2', 'n_Hi_2', 'K_He', 'n_He', 'Q_10Scaler_3', 'K_Cl_2', 'K_HCO3_2', 'kp1_4', 'km1_4', 'kp2_4', 'K_Hi_3', 'n_Hi_3', 'K_He_2', 'n_He_2', 'Q_10Scaler_4', 'PP_co2e', 'CO_2sol', 'P_atm', 'kf_co2hyd', 'kr_co2hyd', 'pKa_ib1', 'ib1', 'pKa_ib2', 'ib2', 'pH_scale', 'g_Cl', 'g_H', 'J_CO2', 'Ca_o', 'g_pCa', 'K_mpCa', 'delta', 'g_Cab', 'K_mNa', 'K_mCa', 'eta', 'k_sat', 'g_NCX', 'g_SERCA', 'K_SERCA', 'B_CMDN', 'K_CMDN', 'g_SRl', 'B_TRPN', 'g_D', 'J_R', 'J_L', 'V_L0', 'delta_VL', 't_L', 'a', 'b', 'tau_L', 'K_L', 'phi_L', 'phi_R_base', 'tau_R', 'theta_R', 'K_RyR', 'c', 'd', 'k_on', 'k_off', 'gamma_trpn', 'TRPN_tot', 'T_ref', 'alpha_0', 'alpha_r1', 'alpha_r2', 'n_Rel', 'K_z', 'n_Hill', 'Ca_50ref', 'z_p', 'beta_1', 'beta_0', 'a_2', 'A_1', 'A_2', 'A_3', 'alpha_1', 'alpha_2', 'alpha_3', 'tau_s_ss', 'dExtensionRatiodt']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'Na_i', 'm', 'h', 'j', 'K_i', 'r', 's', 's_slow', 'r_ss', 's_ss', 'y', 'CO2i', 'pH_i', 'Cli', 'HCO3i', 'Ca_i', 'Ca_SR', 'TRPN', 'z_1', 'z_2', 'z_3', 'z', 'Q_1', 'Q_2', 'Q_3']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (26)
        V = y[:, 0]
        Na_i = y[:, 1]
        m = y[:, 2]
        h = y[:, 3]
        j = y[:, 4]
        K_i = y[:, 5]
        r = y[:, 6]
        s = y[:, 7]
        s_slow = y[:, 8]
        r_ss = y[:, 9]
        s_ss = y[:, 10]
        state_y = y[:, 11]
        CO2i = y[:, 12]
        pH_i = y[:, 13]
        Cli = y[:, 14]
        HCO3i = y[:, 15]
        Ca_i = y[:, 16]
        Ca_SR = y[:, 17]
        TRPN = y[:, 18]
        z_1 = y[:, 19]
        z_2 = y[:, 20]
        z_3 = y[:, 21]
        z = y[:, 22]
        Q_1 = y[:, 23]
        Q_2 = y[:, 24]
        Q_3 = y[:, 25]

        # Independent parameters (source constant order)
        NO_on = theta[:, 0]  # constants[0]
        pH_on = theta[:, 1]  # constants[1]
        SAC_on = theta[:, 2]  # constants[2]
        V_myo = theta[:, 3]  # constants[3]
        V_SR = theta[:, 4]  # constants[4]
        N = theta[:, 5]  # constants[5]
        A_cap = theta[:, 6]  # constants[6]
        R = theta[:, 7]  # constants[7]
        T = theta[:, 8]  # constants[8]
        F = theta[:, 9]  # constants[9]
        CmF = theta[:, 10]  # constants[10]
        stim_period = theta[:, 11]  # constants[11]
        stim_duration = theta[:, 12]  # constants[12]
        stim_amplitude = theta[:, 13]  # constants[13]
        ExtensionRatio = theta[:, 14]  # constants[14]
        g_SAC = theta[:, 15]  # constants[15]
        E_R = theta[:, 16]  # constants[16]
        g_Ko = theta[:, 17]  # constants[17]
        g_Na = theta[:, 18]  # constants[18]
        Na_o = theta[:, 19]  # constants[19]
        g_t = theta[:, 20]  # constants[20]
        a_endo = theta[:, 21]  # constants[21]
        b_endo = theta[:, 22]  # constants[22]
        K_o = theta[:, 23]  # constants[23]
        g_ss = theta[:, 24]  # constants[24]
        g_K1 = theta[:, 25]  # constants[25]
        g_f = theta[:, 26]  # constants[26]
        f_Na = theta[:, 27]  # constants[27]
        g_B_Na = theta[:, 28]  # constants[28]
        g_B_K = theta[:, 29]  # constants[29]
        scale_Na = theta[:, 30]  # constants[30]
        scale_K = theta[:, 31]  # constants[31]
        i_NaK_max = theta[:, 32]  # constants[32]
        K_m_K = theta[:, 33]  # constants[33]
        K_m_Na = theta[:, 34]  # constants[34]
        Pco2 = theta[:, 35]  # constants[35]
        pH_e = theta[:, 36]  # constants[36]
        KA = theta[:, 37]  # constants[37]
        KB = theta[:, 38]  # constants[38]
        kp1 = theta[:, 39]  # constants[39]
        km1 = theta[:, 40]  # constants[40]
        kp2 = theta[:, 41]  # constants[41]
        K_Hi = theta[:, 42]  # constants[42]
        n_Hi = theta[:, 43]  # constants[43]
        Q_10Scaler = theta[:, 44]  # constants[44]
        Cle = theta[:, 45]  # constants[45]
        K_Cl = theta[:, 46]  # constants[46]
        K_OH = theta[:, 47]  # constants[47]
        kp1_2 = theta[:, 48]  # constants[48]
        km1_2 = theta[:, 49]  # constants[49]
        kp2_2 = theta[:, 50]  # constants[50]
        Q_10Scaler_2 = theta[:, 51]  # constants[51]
        K_Na = theta[:, 52]  # constants[52]
        K_HCO3 = theta[:, 53]  # constants[53]
        kp1_3 = theta[:, 54]  # constants[54]
        km1_3 = theta[:, 55]  # constants[55]
        kp2_3 = theta[:, 56]  # constants[56]
        K_Hi_2 = theta[:, 57]  # constants[57]
        n_Hi_2 = theta[:, 58]  # constants[58]
        K_He = theta[:, 59]  # constants[59]
        n_He = theta[:, 60]  # constants[60]
        Q_10Scaler_3 = theta[:, 61]  # constants[61]
        K_Cl_2 = theta[:, 62]  # constants[62]
        K_HCO3_2 = theta[:, 63]  # constants[63]
        kp1_4 = theta[:, 64]  # constants[64]
        km1_4 = theta[:, 65]  # constants[65]
        kp2_4 = theta[:, 66]  # constants[66]
        K_Hi_3 = theta[:, 67]  # constants[67]
        n_Hi_3 = theta[:, 68]  # constants[68]
        K_He_2 = theta[:, 69]  # constants[69]
        n_He_2 = theta[:, 70]  # constants[70]
        Q_10Scaler_4 = theta[:, 71]  # constants[71]
        PP_co2e = theta[:, 72]  # constants[72]
        CO_2sol = theta[:, 73]  # constants[73]
        P_atm = theta[:, 74]  # constants[74]
        kf_co2hyd = theta[:, 75]  # constants[75]
        kr_co2hyd = theta[:, 76]  # constants[76]
        pKa_ib1 = theta[:, 77]  # constants[77]
        ib1 = theta[:, 78]  # constants[78]
        pKa_ib2 = theta[:, 79]  # constants[79]
        ib2 = theta[:, 80]  # constants[80]
        pH_scale = theta[:, 81]  # constants[81]
        g_Cl = theta[:, 82]  # constants[82]
        g_H = theta[:, 83]  # constants[83]
        J_CO2 = theta[:, 84]  # constants[84]
        Ca_o = theta[:, 85]  # constants[85]
        g_pCa = theta[:, 86]  # constants[86]
        K_mpCa = theta[:, 87]  # constants[87]
        delta = theta[:, 88]  # constants[88]
        g_Cab = theta[:, 89]  # constants[89]
        K_mNa = theta[:, 90]  # constants[90]
        K_mCa = theta[:, 91]  # constants[91]
        eta = theta[:, 92]  # constants[92]
        k_sat = theta[:, 93]  # constants[93]
        g_NCX = theta[:, 94]  # constants[94]
        g_SERCA = theta[:, 95]  # constants[95]
        K_SERCA = theta[:, 96]  # constants[96]
        B_CMDN = theta[:, 97]  # constants[97]
        K_CMDN = theta[:, 98]  # constants[98]
        g_SRl = theta[:, 99]  # constants[99]
        B_TRPN = theta[:, 100]  # constants[100]
        g_D = theta[:, 101]  # constants[101]
        J_R = theta[:, 102]  # constants[102]
        J_L = theta[:, 103]  # constants[103]
        V_L0 = theta[:, 104]  # constants[104]
        delta_VL = theta[:, 105]  # constants[105]
        t_L = theta[:, 106]  # constants[106]
        a = theta[:, 107]  # constants[107]
        b = theta[:, 108]  # constants[108]
        tau_L = theta[:, 109]  # constants[109]
        K_L = theta[:, 110]  # constants[110]
        phi_L = theta[:, 111]  # constants[111]
        phi_R_base = theta[:, 112]  # constants[112]
        tau_R = theta[:, 113]  # constants[113]
        theta_R = theta[:, 114]  # constants[114]
        K_RyR = theta[:, 115]  # constants[115]
        c = theta[:, 116]  # constants[116]
        d = theta[:, 117]  # constants[117]
        k_on = theta[:, 118]  # constants[118]
        k_off = theta[:, 119]  # constants[119]
        gamma_trpn = theta[:, 120]  # constants[120]
        TRPN_tot = theta[:, 121]  # constants[121]
        T_ref = theta[:, 122]  # constants[122]
        alpha_0 = theta[:, 123]  # constants[123]
        alpha_r1 = theta[:, 124]  # constants[124]
        alpha_r2 = theta[:, 125]  # constants[125]
        n_Rel = theta[:, 126]  # constants[126]
        K_z = theta[:, 127]  # constants[127]
        n_Hill = theta[:, 128]  # constants[128]
        Ca_50ref = theta[:, 129]  # constants[129]
        z_p = theta[:, 130]  # constants[130]
        beta_1 = theta[:, 131]  # constants[131]
        beta_0 = theta[:, 132]  # constants[132]
        a_2 = theta[:, 133]  # constants[133]
        A_1 = theta[:, 134]  # constants[134]
        A_2 = theta[:, 135]  # constants[135]
        A_3 = theta[:, 136]  # constants[136]
        alpha_1 = theta[:, 137]  # constants[137]
        alpha_2 = theta[:, 138]  # constants[138]
        alpha_3 = theta[:, 139]  # constants[139]
        tau_s_ss = theta[:, 140]  # constants[146]
        dExtensionRatiodt = theta[:, 141]  # constants[154]

        # Derived constants from the original initConsts()
        rho = A_cap / V_myo  # constants[140]
        gamma_SLSAC = torch.where(SAC_on == 1.0, (ExtensionRatio - 1.0) * 10.0, 0.0)  # constants[141]
        const_r = -(E_R + 85.0) / (E_R - 65.0)  # constants[142]
        gamma_SLKO = torch.where(SAC_on == 1.0, 0.7 + (ExtensionRatio - 1.0) * 3.0, 0.7)  # constants[143]
        g_Na_endo = 1.33 * g_Na  # constants[144]
        g_t_endo = 0.4647 * g_t  # constants[145]
        f_K = 1.0 - f_Na  # constants[147]
        sigma = (torch.exp(Na_o / 67.3) - 1.0) / 7.0  # constants[148]
        Cm = CmF / A_cap  # constants[149]
        H_o = 1000.0 * 10.0 ** (-pH_e)  # constants[150]
        Nae3 = Na_o ** 3.0  # constants[151]
        t_R = 1.17 * t_L  # constants[152]
        lamda = torch.where((ExtensionRatio > 0.8) & (ExtensionRatio <= 1.15), ExtensionRatio, torch.where(ExtensionRatio > 1.15, 1.15, 0.8))  # constants[153]
        gamma_NHE = torch.where(pH_on == 1.0, 1.0 - 2.804 * (ExtensionRatio - 1.0), 1.0)  # constants[156]
        alpha_m = phi_L / t_L  # constants[157]
        K_2 = alpha_r2 * z_p ** n_Rel / (z_p ** n_Rel + K_z ** n_Rel) * (1.0 - n_Rel * K_z ** n_Rel / (z_p ** n_Rel + K_z ** n_Rel))  # constants[158]
        overlap = 1.0 + beta_0 * (lamda - 1.0)  # constants[159]
        K_Hs = K_Hi * gamma_NHE  # constants[160]
        gamma_NO = torch.where(NO_on == 1.0, 1.0 + 22.41 * (ExtensionRatio - 1.0), 1.0)  # constants[161]
        K_1 = alpha_r2 * z_p ** (n_Rel - 1.0) * n_Rel * K_z ** n_Rel / (z_p ** n_Rel + K_z ** n_Rel) ** 2.0  # constants[162]
        km2 = kp1 * kp2 / km1  # constants[163]
        phi_R = phi_R_base * gamma_NO  # constants[164]
        Ca_50 = Ca_50ref * (1.0 + beta_1 * (lamda - 1.0))  # constants[165]
        Be = 10.0 ** (-pH_e) * 1000.0  # constants[166]
        beta_m = phi_R / t_R  # constants[167]
        Ca_TRPN_50 = Ca_50 * TRPN_tot / (Ca_50 + k_off / k_on * (1.0 - (1.0 + beta_0 * (lamda - 1.0)) * 0.5 / gamma_trpn))  # constants[168]
        ap1 = KB * Na_o * kp1 / (KA * KB + KB * Na_o + Na_o * Be + KA * Be)  # constants[169]
        z_max = (alpha_0 / (Ca_TRPN_50 / TRPN_tot) ** n_Hill - K_2) / (alpha_r1 + K_1 + alpha_0 / (Ca_TRPN_50 / TRPN_tot) ** n_Hill)  # constants[170]
        am2 = KA * Be * km2 / (KA * KB + KB * Na_o + Na_o * Be + KA * Be)  # constants[171]
        km2_2 = kp2_2 * km1_2 / kp1_2  # constants[172]
        OHe = 1000.0 * 10.0 ** (-14.0 + pH_e)  # constants[173]
        a_3 = 1.0 + K_OH / OHe + K_OH * Cle / (OHe * K_Cl)  # constants[174]
        b_2 = 1.0 + K_Cl / Cle + K_Cl * OHe / (Cle * K_OH)  # constants[175]
        km2_3 = kp2_3 * km1_3 / kp1_3  # constants[176]
        He = 1000.0 * 10.0 ** (-pH_e)  # constants[177]
        gamma_AE = torch.where(pH_on == 1.0, 1.0 + 2.5 * (ExtensionRatio - 1.0), 1.0)  # constants[178]
        K_Hs_2 = K_Hi_3 * gamma_AE  # constants[179]
        km2_4 = kp2_4 * km1_4 / kp1_4  # constants[180]
        He_2 = 1000.0 * 10.0 ** (-pH_e)  # constants[181]
        CO2e = PP_co2e * CO_2sol * P_atm  # constants[182]
        HCO3e = kf_co2hyd / kr_co2hyd * CO2e / (10.0 ** (-pH_e) * 1000.0)  # constants[183]
        a_4 = 1.0 + Na_o / K_Na + Na_o * HCO3e / (K_Na * K_HCO3)  # constants[184]
        b_3 = 1.0 + K_HCO3 / HCO3e + K_Na * K_HCO3 / (HCO3e * Na_o)  # constants[185]
        a_5 = 1.0 + K_HCO3_2 / HCO3e + K_HCO3_2 * Cle / (HCO3e * K_Cl_2)  # constants[186]
        b_4 = 1.0 + K_Cl_2 / Cle + K_Cl_2 * HCO3e / (Cle * K_HCO3_2)  # constants[187]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_j = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_r = torch.zeros_like(y[:, 0])
        d_s = torch.zeros_like(y[:, 0])
        d_s_slow = torch.zeros_like(y[:, 0])
        d_r_ss = torch.zeros_like(y[:, 0])
        d_s_ss = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])
        d_CO2i = torch.zeros_like(y[:, 0])
        d_pH_i = torch.zeros_like(y[:, 0])
        d_Cli = torch.zeros_like(y[:, 0])
        d_HCO3i = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_Ca_SR = torch.zeros_like(y[:, 0])
        d_TRPN = torch.zeros_like(y[:, 0])
        d_z_1 = torch.zeros_like(y[:, 0])
        d_z_2 = torch.zeros_like(y[:, 0])
        d_z_3 = torch.zeros_like(y[:, 0])
        d_z = torch.zeros_like(y[:, 0])
        d_Q_1 = torch.zeros_like(y[:, 0])
        d_Q_2 = torch.zeros_like(y[:, 0])
        d_Q_3 = torch.zeros_like(y[:, 0])
        alg_0_m_infinity = torch.zeros_like(y[:, 0])
        alg_1_h_infinity = torch.zeros_like(y[:, 0])
        alg_2_j_infinity = torch.zeros_like(y[:, 0])
        alg_3_r_infinity = torch.zeros_like(y[:, 0])
        alg_4_s_infinity = torch.zeros_like(y[:, 0])
        alg_5_s_slow_infinity = torch.zeros_like(y[:, 0])
        alg_6_r_ss_infinity = torch.zeros_like(y[:, 0])
        alg_7_s_ss_infinity = torch.zeros_like(y[:, 0])
        alg_8_y_infinity = torch.zeros_like(y[:, 0])
        alg_9_i_Stim = torch.zeros_like(y[:, 0])
        alg_10_Ca_b = torch.zeros_like(y[:, 0])
        alg_12_tau_m = torch.zeros_like(y[:, 0])
        alg_13_tau_h = torch.zeros_like(y[:, 0])
        alg_14_tau_j = torch.zeros_like(y[:, 0])
        alg_15_tau_r = torch.zeros_like(y[:, 0])
        alg_16_tau_s_endo = torch.zeros_like(y[:, 0])
        alg_17_tau_s_slow_endo = torch.zeros_like(y[:, 0])
        alg_18_tau_r_ss = torch.zeros_like(y[:, 0])
        alg_19_tau_y = torch.zeros_like(y[:, 0])
        alg_20_E_Na = torch.zeros_like(y[:, 0])
        alg_21_alpha_Tm = torch.zeros_like(y[:, 0])
        alg_22_I_SAC_Na = torch.zeros_like(y[:, 0])
        alg_23_beta_Tm = torch.zeros_like(y[:, 0])
        alg_24_i_Na = torch.zeros_like(y[:, 0])
        alg_25_E_K = torch.zeros_like(y[:, 0])
        alg_27_I_SAC_K = torch.zeros_like(y[:, 0])
        alg_29_I_Ko = torch.zeros_like(y[:, 0])
        alg_30_i_t = torch.zeros_like(y[:, 0])
        alg_31_i_ss = torch.zeros_like(y[:, 0])
        alg_32_i_K1 = torch.zeros_like(y[:, 0])
        alg_33_i_f_Na = torch.zeros_like(y[:, 0])
        alg_34_i_f_K = torch.zeros_like(y[:, 0])
        alg_36_i_f = torch.zeros_like(y[:, 0])
        alg_37_i_B_Na = torch.zeros_like(y[:, 0])
        alg_38_i_B_K = torch.zeros_like(y[:, 0])
        alg_39_p_nai = torch.zeros_like(y[:, 0])
        alg_40_p_v = torch.zeros_like(y[:, 0])
        alg_41_i_NaK = torch.zeros_like(y[:, 0])
        alg_42_jco2 = torch.zeros_like(y[:, 0])
        alg_46_Bi = torch.zeros_like(y[:, 0])
        alg_47_am1 = torch.zeros_like(y[:, 0])
        alg_48_ap2 = torch.zeros_like(y[:, 0])
        alg_49_reg = torch.zeros_like(y[:, 0])
        alg_50_flux_nhe = torch.zeros_like(y[:, 0])
        alg_51_v_nhe = torch.zeros_like(y[:, 0])
        alg_52_OHi = torch.zeros_like(y[:, 0])
        alg_53_c = torch.zeros_like(y[:, 0])
        alg_54_d = torch.zeros_like(y[:, 0])
        alg_55_s1 = torch.zeros_like(y[:, 0])
        alg_56_s6 = torch.zeros_like(y[:, 0])
        alg_57_v_che = torch.zeros_like(y[:, 0])
        alg_58_c_2 = torch.zeros_like(y[:, 0])
        alg_59_d_2 = torch.zeros_like(y[:, 0])
        alg_60_Hi = torch.zeros_like(y[:, 0])
        alg_61_s1_2 = torch.zeros_like(y[:, 0])
        alg_62_s6_2 = torch.zeros_like(y[:, 0])
        alg_63_reg_2 = torch.zeros_like(y[:, 0])
        alg_64_v_nbc = torch.zeros_like(y[:, 0])
        alg_65_Hi_2 = torch.zeros_like(y[:, 0])
        alg_66_c_3 = torch.zeros_like(y[:, 0])
        alg_67_d_3 = torch.zeros_like(y[:, 0])
        alg_68_s1_3 = torch.zeros_like(y[:, 0])
        alg_69_s6_3 = torch.zeros_like(y[:, 0])
        alg_70_reg_3 = torch.zeros_like(y[:, 0])
        alg_71_v_ae = torch.zeros_like(y[:, 0])
        alg_72_E_Cl = torch.zeros_like(y[:, 0])
        alg_73_I_Cl = torch.zeros_like(y[:, 0])
        alg_74_H_i = torch.zeros_like(y[:, 0])
        alg_75_E_H = torch.zeros_like(y[:, 0])
        alg_76_I_H = torch.zeros_like(y[:, 0])
        alg_77_beta_intr = torch.zeros_like(y[:, 0])
        alg_79_v_co2hyd = torch.zeros_like(y[:, 0])
        alg_81_I_pCa = torch.zeros_like(y[:, 0])
        alg_82_E_Ca = torch.zeros_like(y[:, 0])
        alg_83_I_Cab = torch.zeros_like(y[:, 0])
        alg_84_edv = torch.zeros_like(y[:, 0])
        alg_85_edv2 = torch.zeros_like(y[:, 0])
        alg_86_Nai3 = torch.zeros_like(y[:, 0])
        alg_87_I_NaCa = torch.zeros_like(y[:, 0])
        alg_90_I_SERCA = torch.zeros_like(y[:, 0])
        alg_92_dV = torch.zeros_like(y[:, 0])
        alg_93_expmdV = torch.zeros_like(y[:, 0])
        alg_94_expVL = torch.zeros_like(y[:, 0])
        alg_96_C_co = torch.zeros_like(y[:, 0])
        alg_97_C_oc = torch.zeros_like(y[:, 0])
        alg_98_J_Rco = torch.zeros_like(y[:, 0])
        alg_99_J_Roo = torch.zeros_like(y[:, 0])
        alg_100_J_Loc = torch.zeros_like(y[:, 0])
        alg_101_J_Loo = torch.zeros_like(y[:, 0])
        alg_102_epsilon_pco = torch.zeros_like(y[:, 0])
        alg_103_epsilon_pcc = torch.zeros_like(y[:, 0])
        alg_104_epsilon_m = torch.zeros_like(y[:, 0])
        alg_105_alpha_p = torch.zeros_like(y[:, 0])
        alg_106_beta_poc = torch.zeros_like(y[:, 0])
        alg_107_beta_pcc = torch.zeros_like(y[:, 0])
        alg_108_mu_poc = torch.zeros_like(y[:, 0])
        alg_109_mu_pcc = torch.zeros_like(y[:, 0])
        alg_110_mu_moc = torch.zeros_like(y[:, 0])
        alg_111_mu_mcc = torch.zeros_like(y[:, 0])
        alg_112_denom = torch.zeros_like(y[:, 0])
        alg_113_y_oc = torch.zeros_like(y[:, 0])
        alg_114_y_co = torch.zeros_like(y[:, 0])
        alg_115_y_oo = torch.zeros_like(y[:, 0])
        alg_116_y_cc = torch.zeros_like(y[:, 0])
        alg_117_J_R1 = torch.zeros_like(y[:, 0])
        alg_118_r_1 = torch.zeros_like(y[:, 0])
        alg_119_J_R3 = torch.zeros_like(y[:, 0])
        alg_120_r_2 = torch.zeros_like(y[:, 0])
        alg_121_J_L1 = torch.zeros_like(y[:, 0])
        alg_122_r_3 = torch.zeros_like(y[:, 0])
        alg_123_J_L2 = torch.zeros_like(y[:, 0])
        alg_124_r_4 = torch.zeros_like(y[:, 0])
        alg_125_J_LC = torch.zeros_like(y[:, 0])
        alg_126_r_5 = torch.zeros_like(y[:, 0])
        alg_127_I_CaL = torch.zeros_like(y[:, 0])
        alg_128_r_6 = torch.zeros_like(y[:, 0])
        alg_129_J_RY = torch.zeros_like(y[:, 0])
        alg_130_r_7 = torch.zeros_like(y[:, 0])
        alg_131_J_SR = torch.zeros_like(y[:, 0])
        alg_132_r_8 = torch.zeros_like(y[:, 0])
        alg_133_z_4 = torch.zeros_like(y[:, 0])
        alg_134_T_Base = torch.zeros_like(y[:, 0])
        alg_135_T_0 = torch.zeros_like(y[:, 0])
        alg_136_Q = torch.zeros_like(y[:, 0])
        alg_137_Tension = torch.zeros_like(y[:, 0])
        alg_138_betaCab = torch.zeros_like(y[:, 0])
        alg_139_J_TPRN = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Q_1 = torch.zeros_like(y[:, 0]) + (A_1 * dExtensionRatiodt - alpha_1 * Q_1)
        d_Q_2 = torch.zeros_like(y[:, 0]) + (A_2 * dExtensionRatiodt - alpha_2 * Q_2)
        d_Q_3 = torch.zeros_like(y[:, 0]) + (A_3 * dExtensionRatiodt - alpha_3 * Q_3)
        alg_7_s_ss_infinity = 1.0 / (1.0 + torch.exp((V + 87.5) / 10.3))
        d_s_ss = torch.zeros_like(y[:, 0]) + ((alg_7_s_ss_infinity - s_ss) / tau_s_ss)
        alg_0_m_infinity = 1.0 / (1.0 + torch.exp((V + 45.0) / -6.5))
        alg_12_tau_m = 1.36 / (0.32 * (V + 47.13) / (1.0 - torch.exp(-0.1 * (V + 47.13))) + 0.08 * torch.exp(-V / 11.0))
        d_m = torch.zeros_like(y[:, 0]) + ((alg_0_m_infinity - m) / alg_12_tau_m)
        alg_1_h_infinity = 1.0 / (1.0 + torch.exp((V + 76.1) / 6.07))
        alg_13_tau_h = torch.where(V >= -40.0, 0.4537 * (1.0 + torch.exp(-(V + 10.66) / 11.1)), 3.49 / (0.135 * torch.exp(-(V + 80.0) / 6.8) + 3.56 * torch.exp(0.079 * V) + 310000.0 * torch.exp(0.35 * V)))
        d_h = torch.zeros_like(y[:, 0]) + ((alg_1_h_infinity - h) / alg_13_tau_h)
        alg_2_j_infinity = 1.0 / (1.0 + torch.exp((V + 76.1) / 6.07))
        alg_14_tau_j = torch.where(V >= -40.0, 11.63 * (1.0 + torch.exp(-0.1 * (V + 32.0))) / torch.exp(-2.535e-07 * V), 3.49 / ((V + 37.78) / (1.0 + torch.exp(0.311 * (V + 79.23))) * (-127140.0 * torch.exp(0.2444 * V) - 3.474e-05 * torch.exp(-0.04391 * V)) + 0.1212 * torch.exp(-0.01052 * V) / (1.0 + torch.exp(-0.1378 * (V + 40.14)))))
        d_j = torch.zeros_like(y[:, 0]) + ((alg_2_j_infinity - j) / alg_14_tau_j)
        alg_15_tau_r = 1000.0 / (45.16 * torch.exp(0.03577 * (V + 50.0)) + 98.9 * torch.exp(-0.1 * (V + 38.0)))
        alg_3_r_infinity = 1.0 / (1.0 + torch.exp((V + 10.6) / -11.42))
        d_r = torch.zeros_like(y[:, 0]) + ((alg_3_r_infinity - r) / alg_15_tau_r)
        alg_16_tau_s_endo = 550.0 * torch.exp(-((V + 70.0) / 25.0) ** 2.0) + 49.0
        alg_4_s_infinity = 1.0 / (1.0 + torch.exp((V + 45.3) / 6.8841))
        d_s = torch.zeros_like(y[:, 0]) + ((alg_4_s_infinity - s) / alg_16_tau_s_endo)
        alg_17_tau_s_slow_endo = 3300.0 * torch.exp(-((V + 70.0) / 30.0) ** 2.0) + 49.0
        alg_5_s_slow_infinity = 1.0 / (1.0 + torch.exp((V + 45.3) / 6.8841))
        d_s_slow = torch.zeros_like(y[:, 0]) + ((alg_5_s_slow_infinity - s_slow) / alg_17_tau_s_slow_endo)
        alg_18_tau_r_ss = 10000.0 / (45.16 * torch.exp(0.03577 * (V + 50.0)) + 98.9 * torch.exp(-0.1 * (V + 38.0)))
        alg_6_r_ss_infinity = 1.0 / (1.0 + torch.exp((V + 11.5) / -11.82))
        d_r_ss = torch.zeros_like(y[:, 0]) + ((alg_6_r_ss_infinity - r_ss) / alg_18_tau_r_ss)
        alg_19_tau_y = 1000.0 / (0.11885 * torch.exp((V + 80.0) / 28.37) + 0.5623 * torch.exp((V + 80.0) / -14.19))
        alg_8_y_infinity = 1.0 / (1.0 + torch.exp((V + 138.6) / 10.48))
        d_state_y = torch.zeros_like(y[:, 0]) + ((alg_8_y_infinity - state_y) / alg_19_tau_y)
        alg_10_Ca_b = B_TRPN - TRPN
        alg_21_alpha_Tm = alpha_0 * (alg_10_Ca_b / Ca_TRPN_50) ** n_Hill
        alg_23_beta_Tm = alpha_r1 + alpha_r2 * z ** (n_Rel - 1.0) / (z ** n_Rel + K_z ** n_Rel)
        d_z = torch.zeros_like(y[:, 0]) + (alg_21_alpha_Tm * (1.0 - z) - alg_23_beta_Tm * z)
        alg_25_E_K = R * T / F * torch.log(K_o / K_i)
        alg_30_i_t = g_t_endo * r * (a_endo * s + b_endo * s_slow) * (V - alg_25_E_K) / A_cap
        alg_31_i_ss = g_ss * r_ss * s_ss * (V - alg_25_E_K) / A_cap
        alg_32_i_K1 = ((0.048 / (torch.exp((V + 37.0) / 25.0) + torch.exp((V + 37.0) / -25.0)) + 0.01) * 0.001 / (1.0 + torch.exp((V - (alg_25_E_K + 76.77)) / -17.0)) + g_K1 * (V - (alg_25_E_K + 1.73)) / ((1.0 + torch.exp(1.613 * F * (V - (alg_25_E_K + 1.73)) / (R * T))) * (1.0 + torch.exp((K_o - 0.9988) / -0.124)))) / A_cap
        alg_39_p_nai = 1.0 / (1.0 + (K_m_Na / Na_i) ** 4.0)
        alg_40_p_v = 1.0 / (1.0 + 0.1245 * torch.exp(-0.1 * V * F / (R * T)) + 0.0365 * sigma * torch.exp(-V * F / (R * T)))
        alg_41_i_NaK = i_NaK_max / A_cap * alg_40_p_v * K_o / (K_o + K_m_K) * alg_39_p_nai
        alg_9_i_Stim = torch.where((t - torch.floor(t / stim_period) * stim_period >= 0.0) & (t - torch.floor(t / stim_period) * stim_period <= stim_duration), stim_amplitude / A_cap, 0.0)
        alg_38_i_B_K = scale_K * g_B_K * (V - alg_25_E_K) / A_cap
        alg_27_I_SAC_K = g_SAC * gamma_SLSAC * (V - alg_25_E_K) / A_cap
        alg_29_I_Ko = g_Ko / (1.0 + torch.exp(-(10.0 + V) / 45.0)) * (V - alg_25_E_K) * gamma_SLKO / A_cap
        alg_34_i_f_K = g_f * state_y * f_K * (V - alg_25_E_K) / A_cap
        d_K_i = torch.zeros_like(y[:, 0]) + (-(alg_9_i_Stim + alg_27_I_SAC_K + alg_31_i_ss + alg_29_I_Ko + alg_30_i_t + alg_32_i_K1 + alg_34_i_f_K - 2.0 * alg_41_i_NaK + alg_38_i_B_K) * A_cap / (V_myo * F))
        alg_72_E_Cl = R * T / F * torch.log(Cli / Cle)
        alg_73_I_Cl = g_Cl * (V - alg_72_E_Cl)
        alg_52_OHi = 1000.0 * 10.0 ** (-14.0 + pH_i)
        alg_53_c = 1.0 + K_Cl / Cli + K_Cl * alg_52_OHi / (Cli * K_OH)
        alg_54_d = 1.0 + K_OH / alg_52_OHi + K_OH * Cli / (alg_52_OHi * K_Cl)
        alg_55_s1 = 1.0 / (a_3 + alg_54_d * (kp1_2 + kp2_2 * a_3 / b_2) / (km1_2 + km2_2 * alg_54_d / alg_53_c))
        alg_56_s6 = 1.0 / (alg_54_d + a_3 * (km1_2 + km2_2 * alg_54_d / alg_53_c) / (kp1_2 + kp2_2 * a_3 / b_2))
        alg_57_v_che = V_myo * (km1_2 * alg_56_s6 - kp1_2 * alg_55_s1) / (60.0 * 1000.0) * Q_10Scaler_2
        alg_66_c_3 = 1.0 + K_Cl_2 / Cli + K_Cl_2 * HCO3i / (Cli * K_HCO3_2)
        alg_67_d_3 = 1.0 + K_HCO3_2 / HCO3i + K_HCO3_2 * Cli / (HCO3i * K_Cl_2)
        alg_68_s1_3 = 1.0 / (a_5 + alg_67_d_3 * (kp1_4 + kp2_4 * a_5 / b_4) / (km1_4 + km2_4 * alg_67_d_3 / alg_66_c_3))
        alg_69_s6_3 = 1.0 / (alg_67_d_3 + a_5 * (km1_4 + km2_4 * alg_67_d_3 / alg_66_c_3) / (kp1_4 + kp2_4 * a_5 / b_4))
        alg_65_Hi_2 = 1000.0 * 10.0 ** (-pH_i)
        alg_70_reg_3 = K_Hs_2 ** n_Hi_3 / (alg_65_Hi_2 ** n_Hi_3 + K_Hs_2 ** n_Hi_3) * He_2 ** n_He_2 / (He_2 ** n_He_2 + K_He_2 ** n_He_2)
        alg_71_v_ae = V_myo * alg_70_reg_3 * (km1_4 * alg_69_s6_3 - kp1_4 * alg_68_s1_3) / (60.0 * 1000.0) * Q_10Scaler_4
        d_Cli = torch.zeros_like(y[:, 0]) + (alg_73_I_Cl * A_cap / (V_myo * F) + pH_scale * (alg_57_v_che + alg_71_v_ae) / V_myo)
        alg_58_c_2 = 1.0 + K_HCO3 / HCO3i + K_Na * K_HCO3 / (HCO3i * Na_i)
        alg_59_d_2 = 1.0 + Na_i / K_Na + Na_i * HCO3i / (K_Na * K_HCO3)
        alg_61_s1_2 = 1.0 / (a_4 + alg_59_d_2 * (kp1_3 + kp2_3 * a_4 / b_3) / (km1_3 + km2_3 * alg_59_d_2 / alg_58_c_2))
        alg_62_s6_2 = 1.0 / (alg_59_d_2 + a_4 * (km1_3 + km2_3 * alg_59_d_2 / alg_58_c_2) / (kp1_3 + kp2_3 * a_4 / b_3))
        alg_60_Hi = 1000.0 * 10.0 ** (-pH_i)
        alg_63_reg_2 = alg_60_Hi ** n_Hi_2 / (alg_60_Hi ** n_Hi_2 + K_Hi_2 ** n_Hi_2) * (1.0 - He ** n_He / (He ** n_He + K_He ** n_He))
        alg_64_v_nbc = V_myo * alg_63_reg_2 * (km1_3 * alg_62_s6_2 - kp1_3 * alg_61_s1_2) / (60.0 * 1000.0) * Q_10Scaler_3
        alg_79_v_co2hyd = (V_myo + V_SR) * (kf_co2hyd * CO2i - kr_co2hyd * HCO3i * 10.0 ** (-pH_i) * 1000.0)
        d_HCO3i = torch.zeros_like(y[:, 0]) + (alg_79_v_co2hyd / (V_myo + V_SR) + (alg_64_v_nbc - alg_71_v_ae) / V_myo)
        alg_42_jco2 = V_myo * rho * Pco2 * (CO2e - CO2i)
        d_CO2i = torch.zeros_like(y[:, 0]) + (alg_42_jco2 / V_myo - alg_79_v_co2hyd / (V_myo + V_SR) + J_CO2)
        alg_74_H_i = 1000.0 * 10.0 ** (-pH_i)
        alg_75_E_H = R * T / F * torch.log(H_o / alg_74_H_i)
        alg_76_I_H = g_H * (V - alg_75_E_H)
        alg_46_Bi = 10.0 ** (-pH_i) * 1000.0
        alg_47_am1 = KB * Na_i * km1 / (KA * KB + KB * Na_i + Na_i * alg_46_Bi + KA * alg_46_Bi)
        alg_48_ap2 = KA * alg_46_Bi * kp2 / (KA * KB + KB * Na_i + Na_i * alg_46_Bi + KA * alg_46_Bi)
        alg_49_reg = alg_46_Bi ** n_Hi / (alg_46_Bi ** n_Hi + K_Hs ** n_Hi)
        alg_50_flux_nhe = alg_49_reg * (ap1 * alg_48_ap2 - alg_47_am1 * am2) / (ap1 + alg_48_ap2 + alg_47_am1 + am2)
        alg_51_v_nhe = alg_50_flux_nhe * V_myo * Q_10Scaler
        alg_77_beta_intr = 2.302585092994046 * (10.0 ** (-pH_i) + 10.0 ** (pH_i + pKa_ib1) * ib1 / (10.0 ** pH_i + 10.0 ** pKa_ib1) ** 2.0 + 10.0 ** (pH_i + pKa_ib2) * ib2 / (10.0 ** pH_i + 10.0 ** pKa_ib2) ** 2.0)
        d_pH_i = torch.zeros_like(y[:, 0]) + (1.0 / -alg_77_beta_intr * ((-alg_51_v_nhe + alg_57_v_che) / V_myo + alg_79_v_co2hyd / (V_myo + V_SR) - alg_76_I_H * A_cap / (V_myo * F)))
        alg_20_E_Na = R * T / F * torch.log(Na_o / Na_i)
        alg_24_i_Na = g_Na_endo * m ** 3.0 * h * j * (V - alg_20_E_Na) / A_cap
        alg_84_edv = torch.exp(delta * 0.5 * V * eta)
        alg_85_edv2 = torch.exp(delta * 0.5 * V * (eta - 1.0))
        alg_86_Nai3 = Na_i ** 3.0
        alg_87_I_NaCa = g_NCX / ((Nae3 + K_mNa ** 3.0) * (Ca_o + K_mCa)) * (alg_84_edv * alg_86_Nai3 * Ca_o - alg_85_edv2 * Nae3 * Ca_i) / (1.0 + k_sat * alg_85_edv2) * V_myo * F / A_cap
        alg_37_i_B_Na = scale_Na * g_B_Na * (V - alg_20_E_Na) / A_cap
        alg_22_I_SAC_Na = g_SAC * gamma_SLSAC * (V - alg_20_E_Na) / A_cap * const_r
        alg_33_i_f_Na = g_f * state_y * f_Na * (V - alg_20_E_Na) / A_cap
        d_Na_i = torch.zeros_like(y[:, 0]) + (-(alg_37_i_B_Na + alg_22_I_SAC_Na + alg_24_i_Na + alg_87_I_NaCa * 3.0 + alg_41_i_NaK * 3.0 + alg_33_i_f_Na) * A_cap / (V_myo * F) + (alg_51_v_nhe + alg_64_v_nbc) / V_myo)
        alg_36_i_f = alg_33_i_f_Na + alg_34_i_f_K
        alg_92_dV = delta * V
        alg_93_expmdV = torch.exp(-alg_92_dV)
        alg_101_J_Loo = torch.where(torch.abs(alg_92_dV) > 1e-05, J_L * alg_92_dV / (1.0 - alg_93_expmdV) * (Ca_o * alg_93_expmdV - Ca_i + J_R / g_D * (Ca_o * alg_93_expmdV - Ca_SR)) / (1.0 + J_R / g_D + J_L / g_D * alg_92_dV / (1.0 - alg_93_expmdV)), J_L * 1e-05 / (1.0 - 0.9999900000499998) * (Ca_o * 0.9999900000499998 - Ca_i + J_R / g_D * (Ca_o * 0.9999900000499998 - Ca_SR)) / (1.0 + J_R / g_D + J_L / g_D * 1e-05 / (1.0 - 0.9999900000499998)))
        alg_100_J_Loc = torch.where(torch.abs(alg_92_dV) > 1e-05, J_L * alg_92_dV / (1.0 - alg_93_expmdV) * (Ca_o * alg_93_expmdV - Ca_i) / (1.0 + J_L / g_D * alg_92_dV / (1.0 - alg_93_expmdV)), J_L * 1e-05 / (1.0 - 0.9999900000499998) * (Ca_o * 0.9999900000499998 - Ca_i) / (1.0 + J_L / g_D * 1e-05 / (1.0 - 0.9999900000499998)))
        alg_94_expVL = torch.exp((V - V_L0) / delta_VL)
        alg_105_alpha_p = alg_94_expVL / (t_L * (alg_94_expVL + 1.0))
        alg_107_beta_pcc = 1.0 / t_R * Ca_i ** 2.0 / (Ca_i ** 2.0 + K_RyR ** 2.0)
        alg_97_C_oc = torch.where(torch.abs(alg_92_dV) > 1e-09, (Ca_i + J_L / g_D * Ca_o * alg_92_dV * alg_93_expmdV / (1.0 - alg_93_expmdV)) / (1.0 + J_L / g_D * alg_92_dV / (1.0 - alg_93_expmdV)), (Ca_i + J_L / g_D * Ca_o) / (1.0 + J_L / g_D))
        alg_106_beta_poc = 1.0 / t_R * alg_97_C_oc ** 2.0 / (alg_97_C_oc ** 2.0 + K_RyR ** 2.0)
        alg_112_denom = (alg_105_alpha_p + alpha_m) * ((beta_m + alg_106_beta_poc + alpha_m) * (beta_m + alg_107_beta_pcc) + alg_105_alpha_p * (beta_m + alg_106_beta_poc))
        alg_113_y_oc = alg_105_alpha_p * beta_m * (alg_105_alpha_p + alpha_m + beta_m + alg_107_beta_pcc) / alg_112_denom
        alg_115_y_oo = alg_105_alpha_p * (alg_106_beta_poc * (alg_105_alpha_p + beta_m + alg_107_beta_pcc) + alg_107_beta_pcc * alpha_m) / alg_112_denom
        alg_121_J_L1 = alg_101_J_Loo * alg_115_y_oo + alg_100_J_Loc * alg_113_y_oc
        alg_123_J_L2 = alg_100_J_Loc * alg_105_alpha_p / (alg_105_alpha_p + alpha_m)
        alg_125_J_LC = (z_1 * alg_121_J_L1 + z_2 * alg_123_J_L2) * N / V_myo
        alg_127_I_CaL = -alg_125_J_LC * 2.0 * V_myo * F / A_cap
        alg_82_E_Ca = torch.log(Ca_o / Ca_i) / 2.302585092994046 / delta
        alg_83_I_Cab = g_Cab * (V - alg_82_E_Ca) * 2.0 * V_myo * F / A_cap
        alg_81_I_pCa = g_pCa * Ca_i / (K_mpCa + Ca_i) * 2.0 * V_myo * F / A_cap
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_76_I_H + alg_73_I_Cl + alg_24_i_Na + alg_127_I_CaL + alg_30_i_t + alg_31_i_ss + alg_36_i_f + alg_32_i_K1 + alg_83_I_Cab + alg_22_I_SAC_Na + alg_38_i_B_K + alg_37_i_B_Na + alg_29_I_Ko + alg_27_I_SAC_K + alg_41_i_NaK + alg_87_I_NaCa + alg_81_I_pCa + alg_9_i_Stim) / Cm)
        alg_108_mu_poc = 1.0 / tau_R * (alg_97_C_oc ** 2.0 + c * K_RyR ** 2.0) / (alg_97_C_oc ** 2.0 + K_RyR ** 2.0)
        alg_109_mu_pcc = 1.0 / tau_R * (Ca_i ** 2.0 + c * K_RyR ** 2.0) / (Ca_i ** 2.0 + K_RyR ** 2.0)
        alg_116_y_cc = alpha_m * beta_m * (alpha_m + alg_105_alpha_p + beta_m + alg_106_beta_poc) / alg_112_denom
        alg_118_r_1 = alg_113_y_oc * alg_108_mu_poc + alg_116_y_cc * alg_109_mu_pcc
        alg_110_mu_moc = theta_R / tau_R * d * (alg_97_C_oc ** 2.0 + c * K_RyR ** 2.0) / (d * alg_97_C_oc ** 2.0 + c * K_RyR ** 2.0)
        alg_111_mu_mcc = theta_R / tau_R * d * (Ca_i ** 2.0 + c * K_RyR ** 2.0) / (d * Ca_i ** 2.0 + c * K_RyR ** 2.0)
        alg_120_r_2 = (alg_105_alpha_p * alg_110_mu_moc + alpha_m * alg_111_mu_mcc) / (alg_105_alpha_p + alpha_m)
        alg_96_C_co = (Ca_i + J_R / g_D * Ca_SR) / (1.0 + J_R / g_D)
        alg_102_epsilon_pco = 1.0 / tau_L * alg_96_C_co / K_L * (alg_94_expVL + a) / (alg_94_expVL + 1.0)
        alg_103_epsilon_pcc = 1.0 / tau_L * Ca_i / K_L * (alg_94_expVL + a) / (alg_94_expVL + 1.0)
        alg_114_y_co = alpha_m * (alg_107_beta_pcc * (alpha_m + beta_m + alg_106_beta_poc) + alg_106_beta_poc * alg_105_alpha_p) / alg_112_denom
        alg_126_r_5 = alg_114_y_co * alg_102_epsilon_pco + alg_116_y_cc * alg_103_epsilon_pcc
        alg_104_epsilon_m = 1.0 / tau_L * b * (alg_94_expVL + a) / (b * alg_94_expVL + a)
        alg_128_r_6 = alg_104_epsilon_m
        d_z_1 = torch.zeros_like(y[:, 0]) + (-(alg_118_r_1 + alg_126_r_5) * z_1 + alg_120_r_2 * z_2 + alg_128_r_6 * z_3)
        alg_90_I_SERCA = g_SERCA * Ca_i ** 2.0 / (K_SERCA ** 2.0 + Ca_i ** 2.0)
        alg_98_J_Rco = J_R * (Ca_SR - Ca_i) / (1.0 + J_R / g_D)
        alg_99_J_Roo = torch.where(torch.abs(alg_92_dV) > 1e-05, J_R * (Ca_SR - Ca_i + J_L / g_D * alg_92_dV / (1.0 - alg_93_expmdV) * (Ca_SR - Ca_o * alg_93_expmdV)) / (1.0 + J_R / g_D + J_L / g_D * alg_92_dV / (1.0 - alg_93_expmdV)), J_R * (Ca_SR - Ca_i + J_L / g_D * 1e-05 / (1.0 - 0.9999900000499998) * (Ca_SR - Ca_o * 0.9999900000499998)) / (1.0 + J_R / g_D + J_L / g_D * 1e-05 / (1.0 - 0.9999900000499998)))
        alg_117_J_R1 = alg_115_y_oo * alg_99_J_Roo + alg_98_J_Rco * alg_114_y_co
        alg_119_J_R3 = alg_98_J_Rco * alg_107_beta_pcc / (beta_m + alg_107_beta_pcc)
        alg_129_J_RY = (z_1 * alg_117_J_R1 + z_3 * alg_119_J_R3) * N / V_myo
        alg_131_J_SR = -alg_129_J_RY + alg_90_I_SERCA - g_SRl * (Ca_SR - Ca_i)
        d_Ca_SR = torch.zeros_like(y[:, 0]) + (V_myo / V_SR * alg_131_J_SR)
        alg_130_r_7 = alpha_m * alg_103_epsilon_pcc / (alg_105_alpha_p + alpha_m)
        alg_132_r_8 = alg_104_epsilon_m
        alg_133_z_4 = 1.0 - z_1 - z_2 - z_3
        d_z_2 = torch.zeros_like(y[:, 0]) + (alg_118_r_1 * z_1 - (alg_120_r_2 + alg_130_r_7) * z_2 + alg_132_r_8 * alg_133_z_4)
        alg_122_r_3 = beta_m * alg_109_mu_pcc / (beta_m + alg_107_beta_pcc)
        alg_124_r_4 = alg_111_mu_mcc
        d_z_3 = torch.zeros_like(y[:, 0]) + (alg_126_r_5 * z_1 - (alg_128_r_6 + alg_122_r_3) * z_3 + alg_124_r_4 * alg_133_z_4)
        alg_134_T_Base = T_ref * z / z_max
        alg_135_T_0 = alg_134_T_Base * overlap
        alg_136_Q = Q_1 + Q_2 + Q_3
        alg_137_Tension = torch.where(alg_136_Q < 0.0, alg_135_T_0 * (a_2 * alg_136_Q + 1.0) / (1.0 - alg_136_Q), alg_135_T_0 * (1.0 + (a_2 + 2.0) * alg_136_Q) / (1.0 + alg_136_Q))
        alg_138_betaCab = torch.where(1.0 - alg_137_Tension / (gamma_trpn * T_ref) > 0.1, k_off * (1.0 - alg_137_Tension / (gamma_trpn * T_ref)), k_off * 0.1)
        alg_139_J_TPRN = (B_TRPN - TRPN) * alg_138_betaCab - Ca_i * TRPN * k_on
        d_Ca_i = torch.zeros_like(y[:, 0]) + (1.0 / (1.0 + B_CMDN * K_CMDN / ((Ca_i + K_CMDN) * (Ca_i + K_CMDN))) * (alg_139_J_TPRN - alg_131_J_SR + (2.0 * alg_87_I_NaCa - alg_81_I_pCa - alg_83_I_Cab - alg_127_I_CaL) * A_cap / (2.0 * V_myo * F)))
        d_TRPN = torch.zeros_like(y[:, 0]) + (alg_139_J_TPRN)

        return torch.stack(
            [d_V, d_Na_i, d_m, d_h, d_j, d_K_i, d_r, d_s, d_s_slow, d_r_ss, d_s_ss, d_state_y, d_CO2i, d_pH_i, d_Cli, d_HCO3i, d_Ca_i, d_Ca_SR, d_TRPN, d_z_1, d_z_2, d_z_3, d_z, d_Q_1, d_Q_2, d_Q_3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
