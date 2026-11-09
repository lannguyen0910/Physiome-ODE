import torch
import torch.nn as nn
import math


class ExcitationContractionCouplingShortenOcallaghanDavidsonSoboleva2007Variant01ODE(nn.Module):
    """
    PyTorch translation of excitation-contraction_coupling/shorten_ocallaghan_davidson_soboleva_2007_variant01/model.py.
    Dynamic states: 56; independent constants: 99; derived constants used by rates: 6.
    Original constant slots: 105; algebraic slots: 71.
    """
    source_file = 'excitation-contraction_coupling/shorten_ocallaghan_davidson_soboleva_2007_variant01/model.py'
    source_state_dim = 56
    source_constant_dim = 105
    algebraic_dim = 71
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 56
        self.parameter_dim = 99

        self.parameters = (
            parameters
            if parameters is not None
            else {'C_m': 0.58, 'gam': 2.79, 'R_a': 150, 'tsi': 1e-06, 'tsi2': 0.0025, 'tsi3': 0.0005, 'FF': 96485, 'tau_K': 559, 'tau_Na': 559, 'f_T': 0.00174, 'tau_K2': 40229.885, 'tau_Na2': 40229.885, 'I_K_rest': 0.34, 'I_Na_rest': -0.43, 'alpha_h_bar': 0.0081, 'alpha_m_bar': 0.288, 'alpha_n_bar': 0.0131, 'beta_h_bar': 4.38, 'beta_m_bar': 1.38, 'beta_n_bar': 0.067, 'V_m': -46, 'V_n': -40, 'V_h': -45, 'V_a': 70, 'V_S_inf': -68, 'V_h_K_inf': -40, 'A_a': 150, 'A_S_inf': 7.1, 'A_h_K_inf': 7.5, 'K_alpha_h': 14.7, 'K_beta_h': 9, 'K_alpha_m': 10, 'K_alpha_n': 7, 'K_beta_m': 18, 'K_beta_n': 40, 'RR': 8314.41, 'TT': 293, 'g_Cl_bar': 3.275, 'g_K_bar': 10.8, 'g_Na_bar': 134, 'G_K': 1.85, 'del_var': 0.4, 'K_K': 950, 'K_S': 1, 'K_m_K': 1, 'K_m_Na': 13, 'S_i': 10, 'J_NaK_bar': 0.0001656, 'V_tau': 70, 'eta_Cl': 0.1, 'eta_IR': 1, 'eta_DR': 0.45, 'eta_Na': 0.1, 'eta_NaK': 0.1, 'k_L': 0.002, 'k_Lm': 1000, 'f': 0.2, 'alpha1': 0.2, 'K': 4.5, 'Vbar': -20, 'nu_SR': 2.4375, 'K_SR': 1, 'L_e': 4e-05, 'tau_R': 0.75, 'tau_SR_R': 0.75, 'L_x': 1.1, 'R_R': 0.5, 'k_T_on': 0.0885, 'k_T_off': 0.115, 'T_tot': 140, 'k_P_on': 0, 'k_P_off': 0, 'P_tot': 1500, 'k_Mg_on': 0, 'k_Mg_off': 0, 'k_Cs_on': 4e-06, 'k_Cs_off': 0.005, 'Cs_tot': 31000, 'k_CATP_on': 0.15, 'k_CATP_off': 30, 'k_MATP_on': 0.0015, 'k_MATP_off': 0.15, 'tau_ATP': 0.375, 'tau_Mg': 1.5, 'k_0_on': 0, 'k_0_off': 0.15, 'k_Ca_on': 0.15, 'k_Ca_off': 0.05, 'f_o': 0.5, 'f_p': 5, 'h_o': 0.08, 'h_p': 0.06, 'g_o': 0.04, 'b_p': 3.94e-06, 'k_p': 3.62e-06, 'A_p': 1, 'B_p': 0.0001, 'PP': 6, 'i2': 60}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'vS': -79.974, 'vT': -80.2, 'K_t': 5.9, 'K_i': 150.9, 'K_e': 5.9, 'Na_i': 12.7, 'Na_t': 133, 'Na_e': 133, 'n': 0.009466, 'h_K': 0.9952, 'm': 0.0358, 'h': 0.4981, 'S': 0.581, 'n_t': 0.009466, 'h_K_t': 0.9952, 'm_t': 0.0358, 'h_t': 0.4981, 'S_t': 0.581, 'O_0': 0, 'O_1': 0, 'O_2': 0, 'O_3': 0, 'O_4': 0, 'C_0': 1, 'C_1': 0, 'C_2': 0, 'C_3': 0, 'C_4': 0, 'Ca_1': 0.1, 'Ca_SR1': 1500, 'Ca_2': 0.1, 'Ca_SR2': 1500, 'Ca_T_2': 25, 'Ca_P1': 615, 'Ca_P2': 615, 'Mg_P1': 811, 'Mg_P2': 811, 'Ca_Cs1': 16900, 'Ca_Cs2': 16900, 'Ca_ATP1': 0.4, 'Ca_ATP2': 0.4, 'Mg_ATP1': 7200, 'Mg_ATP2': 7200, 'ATP1': 799.6, 'ATP2': 799.6, 'Mg1': 1000, 'Mg2': 1000, 'Ca_CaT2': 3, 'D_0': 0.8, 'D_1': 1.2, 'D_2': 3, 'A_1': 0.3, 'A_2': 0.23, 'P': 0.23, 'P_SR': 0.23, 'P_C_SR': 0.23}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['C_m', 'gam', 'R_a', 'tsi', 'tsi2', 'tsi3', 'FF', 'tau_K', 'tau_Na', 'f_T', 'tau_K2', 'tau_Na2', 'I_K_rest', 'I_Na_rest', 'alpha_h_bar', 'alpha_m_bar', 'alpha_n_bar', 'beta_h_bar', 'beta_m_bar', 'beta_n_bar', 'V_m', 'V_n', 'V_h', 'V_a', 'V_S_inf', 'V_h_K_inf', 'A_a', 'A_S_inf', 'A_h_K_inf', 'K_alpha_h', 'K_beta_h', 'K_alpha_m', 'K_alpha_n', 'K_beta_m', 'K_beta_n', 'RR', 'TT', 'g_Cl_bar', 'g_K_bar', 'g_Na_bar', 'G_K', 'del_var', 'K_K', 'K_S', 'K_m_K', 'K_m_Na', 'S_i', 'J_NaK_bar', 'V_tau', 'eta_Cl', 'eta_IR', 'eta_DR', 'eta_Na', 'eta_NaK', 'k_L', 'k_Lm', 'f', 'alpha1', 'K', 'Vbar', 'nu_SR', 'K_SR', 'L_e', 'tau_R', 'tau_SR_R', 'L_x', 'R_R', 'k_T_on', 'k_T_off', 'T_tot', 'k_P_on', 'k_P_off', 'P_tot', 'k_Mg_on', 'k_Mg_off', 'k_Cs_on', 'k_Cs_off', 'Cs_tot', 'k_CATP_on', 'k_CATP_off', 'k_MATP_on', 'k_MATP_off', 'tau_ATP', 'tau_Mg', 'k_0_on', 'k_0_off', 'k_Ca_on', 'k_Ca_off', 'f_o', 'f_p', 'h_o', 'h_p', 'g_o', 'b_p', 'k_p', 'A_p', 'B_p', 'PP', 'i2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['vS', 'vT', 'K_t', 'K_i', 'K_e', 'Na_i', 'Na_t', 'Na_e', 'n', 'h_K', 'm', 'h', 'S', 'n_t', 'h_K_t', 'm_t', 'h_t', 'S_t', 'O_0', 'O_1', 'O_2', 'O_3', 'O_4', 'C_0', 'C_1', 'C_2', 'C_3', 'C_4', 'Ca_1', 'Ca_SR1', 'Ca_2', 'Ca_SR2', 'Ca_T_2', 'Ca_P1', 'Ca_P2', 'Mg_P1', 'Mg_P2', 'Ca_Cs1', 'Ca_Cs2', 'Ca_ATP1', 'Ca_ATP2', 'Mg_ATP1', 'Mg_ATP2', 'ATP1', 'ATP2', 'Mg1', 'Mg2', 'Ca_CaT2', 'D_0', 'D_1', 'D_2', 'A_1', 'A_2', 'P', 'P_SR', 'P_C_SR']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (56)
        vS = y[:, 0]
        vT = y[:, 1]
        K_t = y[:, 2]
        K_i = y[:, 3]
        K_e = y[:, 4]
        Na_i = y[:, 5]
        Na_t = y[:, 6]
        Na_e = y[:, 7]
        n = y[:, 8]
        h_K = y[:, 9]
        m = y[:, 10]
        h = y[:, 11]
        S = y[:, 12]
        n_t = y[:, 13]
        h_K_t = y[:, 14]
        m_t = y[:, 15]
        h_t = y[:, 16]
        S_t = y[:, 17]
        O_0 = y[:, 18]
        O_1 = y[:, 19]
        O_2 = y[:, 20]
        O_3 = y[:, 21]
        O_4 = y[:, 22]
        C_0 = y[:, 23]
        C_1 = y[:, 24]
        C_2 = y[:, 25]
        C_3 = y[:, 26]
        C_4 = y[:, 27]
        Ca_1 = y[:, 28]
        Ca_SR1 = y[:, 29]
        Ca_2 = y[:, 30]
        Ca_SR2 = y[:, 31]
        Ca_T_2 = y[:, 32]
        Ca_P1 = y[:, 33]
        Ca_P2 = y[:, 34]
        Mg_P1 = y[:, 35]
        Mg_P2 = y[:, 36]
        Ca_Cs1 = y[:, 37]
        Ca_Cs2 = y[:, 38]
        Ca_ATP1 = y[:, 39]
        Ca_ATP2 = y[:, 40]
        Mg_ATP1 = y[:, 41]
        Mg_ATP2 = y[:, 42]
        ATP1 = y[:, 43]
        ATP2 = y[:, 44]
        Mg1 = y[:, 45]
        Mg2 = y[:, 46]
        Ca_CaT2 = y[:, 47]
        D_0 = y[:, 48]
        D_1 = y[:, 49]
        D_2 = y[:, 50]
        A_1 = y[:, 51]
        A_2 = y[:, 52]
        P = y[:, 53]
        P_SR = y[:, 54]
        P_C_SR = y[:, 55]

        # Independent parameters (source constant order)
        C_m = theta[:, 0]  # constants[0]
        gam = theta[:, 1]  # constants[1]
        R_a = theta[:, 2]  # constants[2]
        tsi = theta[:, 3]  # constants[3]
        tsi2 = theta[:, 4]  # constants[4]
        tsi3 = theta[:, 5]  # constants[5]
        FF = theta[:, 6]  # constants[6]
        tau_K = theta[:, 7]  # constants[7]
        tau_Na = theta[:, 8]  # constants[8]
        f_T = theta[:, 9]  # constants[9]
        tau_K2 = theta[:, 10]  # constants[10]
        tau_Na2 = theta[:, 11]  # constants[11]
        I_K_rest = theta[:, 12]  # constants[12]
        I_Na_rest = theta[:, 13]  # constants[13]
        alpha_h_bar = theta[:, 14]  # constants[14]
        alpha_m_bar = theta[:, 15]  # constants[15]
        alpha_n_bar = theta[:, 16]  # constants[16]
        beta_h_bar = theta[:, 17]  # constants[17]
        beta_m_bar = theta[:, 18]  # constants[18]
        beta_n_bar = theta[:, 19]  # constants[19]
        V_m = theta[:, 20]  # constants[20]
        V_n = theta[:, 21]  # constants[21]
        V_h = theta[:, 22]  # constants[22]
        V_a = theta[:, 23]  # constants[23]
        V_S_inf = theta[:, 24]  # constants[24]
        V_h_K_inf = theta[:, 25]  # constants[25]
        A_a = theta[:, 26]  # constants[26]
        A_S_inf = theta[:, 27]  # constants[27]
        A_h_K_inf = theta[:, 28]  # constants[28]
        K_alpha_h = theta[:, 29]  # constants[29]
        K_beta_h = theta[:, 30]  # constants[30]
        K_alpha_m = theta[:, 31]  # constants[31]
        K_alpha_n = theta[:, 32]  # constants[32]
        K_beta_m = theta[:, 33]  # constants[33]
        K_beta_n = theta[:, 34]  # constants[34]
        RR = theta[:, 35]  # constants[35]
        TT = theta[:, 36]  # constants[36]
        g_Cl_bar = theta[:, 37]  # constants[37]
        g_K_bar = theta[:, 38]  # constants[38]
        g_Na_bar = theta[:, 39]  # constants[39]
        G_K = theta[:, 40]  # constants[40]
        del_var = theta[:, 41]  # constants[41]
        K_K = theta[:, 42]  # constants[42]
        K_S = theta[:, 43]  # constants[43]
        K_m_K = theta[:, 44]  # constants[44]
        K_m_Na = theta[:, 45]  # constants[45]
        S_i = theta[:, 46]  # constants[46]
        J_NaK_bar = theta[:, 47]  # constants[47]
        V_tau = theta[:, 48]  # constants[48]
        eta_Cl = theta[:, 49]  # constants[49]
        eta_IR = theta[:, 50]  # constants[50]
        eta_DR = theta[:, 51]  # constants[51]
        eta_Na = theta[:, 52]  # constants[52]
        eta_NaK = theta[:, 53]  # constants[53]
        k_L = theta[:, 54]  # constants[54]
        k_Lm = theta[:, 55]  # constants[55]
        f = theta[:, 56]  # constants[56]
        alpha1 = theta[:, 57]  # constants[57]
        K = theta[:, 58]  # constants[58]
        Vbar = theta[:, 59]  # constants[59]
        nu_SR = theta[:, 60]  # constants[60]
        K_SR = theta[:, 61]  # constants[61]
        L_e = theta[:, 62]  # constants[62]
        tau_R = theta[:, 63]  # constants[63]
        tau_SR_R = theta[:, 64]  # constants[64]
        L_x = theta[:, 65]  # constants[65]
        R_R = theta[:, 66]  # constants[66]
        k_T_on = theta[:, 67]  # constants[67]
        k_T_off = theta[:, 68]  # constants[68]
        T_tot = theta[:, 69]  # constants[69]
        k_P_on = theta[:, 70]  # constants[70]
        k_P_off = theta[:, 71]  # constants[71]
        P_tot = theta[:, 72]  # constants[72]
        k_Mg_on = theta[:, 73]  # constants[73]
        k_Mg_off = theta[:, 74]  # constants[74]
        k_Cs_on = theta[:, 75]  # constants[75]
        k_Cs_off = theta[:, 76]  # constants[76]
        Cs_tot = theta[:, 77]  # constants[77]
        k_CATP_on = theta[:, 78]  # constants[78]
        k_CATP_off = theta[:, 79]  # constants[79]
        k_MATP_on = theta[:, 80]  # constants[80]
        k_MATP_off = theta[:, 81]  # constants[81]
        tau_ATP = theta[:, 82]  # constants[82]
        tau_Mg = theta[:, 83]  # constants[83]
        k_0_on = theta[:, 84]  # constants[84]
        k_0_off = theta[:, 85]  # constants[85]
        k_Ca_on = theta[:, 86]  # constants[86]
        k_Ca_off = theta[:, 87]  # constants[87]
        f_o = theta[:, 88]  # constants[88]
        f_p = theta[:, 89]  # constants[89]
        h_o = theta[:, 90]  # constants[90]
        h_p = theta[:, 91]  # constants[91]
        g_o = theta[:, 92]  # constants[92]
        b_p = theta[:, 93]  # constants[93]
        k_p = theta[:, 94]  # constants[94]
        A_p = theta[:, 95]  # constants[95]
        B_p = theta[:, 96]  # constants[96]
        PP = theta[:, 97]  # constants[97]
        i2 = theta[:, 98]  # constants[98]

        # Derived constants from the original initConsts()
        V_o = 0.95 * L_x * math.pi * R_R ** 2.0  # constants[99]
        V_SR = 0.05 * L_x * math.pi * R_R ** 2.0  # constants[100]
        V_1 = 0.01 * V_o  # constants[101]
        V_2 = 0.99 * V_o  # constants[102]
        V_SR1 = 0.01 * V_SR  # constants[103]
        V_SR2 = 0.99 * V_SR  # constants[104]

        # Initialize rates and algebraic variables, as in computeRates()
        d_vS = torch.zeros_like(y[:, 0])
        d_vT = torch.zeros_like(y[:, 0])
        d_K_t = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_K_e = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_Na_t = torch.zeros_like(y[:, 0])
        d_Na_e = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_h_K = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_S = torch.zeros_like(y[:, 0])
        d_n_t = torch.zeros_like(y[:, 0])
        d_h_K_t = torch.zeros_like(y[:, 0])
        d_m_t = torch.zeros_like(y[:, 0])
        d_h_t = torch.zeros_like(y[:, 0])
        d_S_t = torch.zeros_like(y[:, 0])
        d_O_0 = torch.zeros_like(y[:, 0])
        d_O_1 = torch.zeros_like(y[:, 0])
        d_O_2 = torch.zeros_like(y[:, 0])
        d_O_3 = torch.zeros_like(y[:, 0])
        d_O_4 = torch.zeros_like(y[:, 0])
        d_C_0 = torch.zeros_like(y[:, 0])
        d_C_1 = torch.zeros_like(y[:, 0])
        d_C_2 = torch.zeros_like(y[:, 0])
        d_C_3 = torch.zeros_like(y[:, 0])
        d_C_4 = torch.zeros_like(y[:, 0])
        d_Ca_1 = torch.zeros_like(y[:, 0])
        d_Ca_SR1 = torch.zeros_like(y[:, 0])
        d_Ca_2 = torch.zeros_like(y[:, 0])
        d_Ca_SR2 = torch.zeros_like(y[:, 0])
        d_Ca_T_2 = torch.zeros_like(y[:, 0])
        d_Ca_P1 = torch.zeros_like(y[:, 0])
        d_Ca_P2 = torch.zeros_like(y[:, 0])
        d_Mg_P1 = torch.zeros_like(y[:, 0])
        d_Mg_P2 = torch.zeros_like(y[:, 0])
        d_Ca_Cs1 = torch.zeros_like(y[:, 0])
        d_Ca_Cs2 = torch.zeros_like(y[:, 0])
        d_Ca_ATP1 = torch.zeros_like(y[:, 0])
        d_Ca_ATP2 = torch.zeros_like(y[:, 0])
        d_Mg_ATP1 = torch.zeros_like(y[:, 0])
        d_Mg_ATP2 = torch.zeros_like(y[:, 0])
        d_ATP1 = torch.zeros_like(y[:, 0])
        d_ATP2 = torch.zeros_like(y[:, 0])
        d_Mg1 = torch.zeros_like(y[:, 0])
        d_Mg2 = torch.zeros_like(y[:, 0])
        d_Ca_CaT2 = torch.zeros_like(y[:, 0])
        d_D_0 = torch.zeros_like(y[:, 0])
        d_D_1 = torch.zeros_like(y[:, 0])
        d_D_2 = torch.zeros_like(y[:, 0])
        d_A_1 = torch.zeros_like(y[:, 0])
        d_A_2 = torch.zeros_like(y[:, 0])
        d_P = torch.zeros_like(y[:, 0])
        d_P_SR = torch.zeros_like(y[:, 0])
        d_P_C_SR = torch.zeros_like(y[:, 0])
        alg_0_I_T = torch.zeros_like(y[:, 0])
        alg_1_alpha_n = torch.zeros_like(y[:, 0])
        alg_2_h_K_inf = torch.zeros_like(y[:, 0])
        alg_3_alpha_h = torch.zeros_like(y[:, 0])
        alg_4_alpha_m = torch.zeros_like(y[:, 0])
        alg_5_S_inf = torch.zeros_like(y[:, 0])
        alg_6_alpha_n_t = torch.zeros_like(y[:, 0])
        alg_7_h_K_inf_t = torch.zeros_like(y[:, 0])
        alg_8_alpha_h_t = torch.zeros_like(y[:, 0])
        alg_9_alpha_m_t = torch.zeros_like(y[:, 0])
        alg_10_S_inf_t = torch.zeros_like(y[:, 0])
        alg_11_k_C = torch.zeros_like(y[:, 0])
        alg_12_T_0 = torch.zeros_like(y[:, 0])
        alg_13_E_K = torch.zeros_like(y[:, 0])
        alg_14_beta_n = torch.zeros_like(y[:, 0])
        alg_15_tau_h_K = torch.zeros_like(y[:, 0])
        alg_16_beta_h = torch.zeros_like(y[:, 0])
        alg_17_beta_m = torch.zeros_like(y[:, 0])
        alg_18_tau_S = torch.zeros_like(y[:, 0])
        alg_19_beta_n_t = torch.zeros_like(y[:, 0])
        alg_20_tau_h_K_t = torch.zeros_like(y[:, 0])
        alg_21_beta_h_t = torch.zeros_like(y[:, 0])
        alg_22_beta_m_t = torch.zeros_like(y[:, 0])
        alg_23_tau_S_t = torch.zeros_like(y[:, 0])
        alg_24_k_Cm = torch.zeros_like(y[:, 0])
        alg_25_E_K_t = torch.zeros_like(y[:, 0])
        alg_26_Cl_i = torch.zeros_like(y[:, 0])
        alg_27_Cl_o = torch.zeros_like(y[:, 0])
        alg_28_Cl_i_t = torch.zeros_like(y[:, 0])
        alg_29_Cl_o_t = torch.zeros_like(y[:, 0])
        alg_30_J_K = torch.zeros_like(y[:, 0])
        alg_31_J_K_t = torch.zeros_like(y[:, 0])
        alg_32_I_HH = torch.zeros_like(y[:, 0])
        alg_33_a = torch.zeros_like(y[:, 0])
        alg_34_J_Cl = torch.zeros_like(y[:, 0])
        alg_35_g_Cl = torch.zeros_like(y[:, 0])
        alg_36_I_Cl = torch.zeros_like(y[:, 0])
        alg_37_K_R = torch.zeros_like(y[:, 0])
        alg_38_g_IR_bar = torch.zeros_like(y[:, 0])
        alg_39_y = torch.zeros_like(y[:, 0])
        alg_40_g_IR = torch.zeros_like(y[:, 0])
        alg_41_I_IR = torch.zeros_like(y[:, 0])
        alg_42_g_DR = torch.zeros_like(y[:, 0])
        alg_43_I_DR = torch.zeros_like(y[:, 0])
        alg_44_J_Na = torch.zeros_like(y[:, 0])
        alg_45_g_Na = torch.zeros_like(y[:, 0])
        alg_46_I_Na = torch.zeros_like(y[:, 0])
        alg_47_sig = torch.zeros_like(y[:, 0])
        alg_48_f1 = torch.zeros_like(y[:, 0])
        alg_49_I_NaK_bar = torch.zeros_like(y[:, 0])
        alg_50_I_NaK = torch.zeros_like(y[:, 0])
        alg_51_I_ionic_s = torch.zeros_like(y[:, 0])
        alg_52_a_t = torch.zeros_like(y[:, 0])
        alg_53_J_Cl_t = torch.zeros_like(y[:, 0])
        alg_54_g_Cl_t = torch.zeros_like(y[:, 0])
        alg_55_I_Cl_t = torch.zeros_like(y[:, 0])
        alg_56_K_R_t = torch.zeros_like(y[:, 0])
        alg_57_g_IR_bar_t = torch.zeros_like(y[:, 0])
        alg_58_y_t = torch.zeros_like(y[:, 0])
        alg_59_g_IR_t = torch.zeros_like(y[:, 0])
        alg_60_I_IR_t = torch.zeros_like(y[:, 0])
        alg_61_g_DR_t = torch.zeros_like(y[:, 0])
        alg_62_I_DR_t = torch.zeros_like(y[:, 0])
        alg_63_J_Na_t = torch.zeros_like(y[:, 0])
        alg_64_g_Na_t = torch.zeros_like(y[:, 0])
        alg_65_I_Na_t = torch.zeros_like(y[:, 0])
        alg_66_sig_t = torch.zeros_like(y[:, 0])
        alg_67_f1_t = torch.zeros_like(y[:, 0])
        alg_68_I_NaK_bar_t = torch.zeros_like(y[:, 0])
        alg_69_I_NaK_t = torch.zeros_like(y[:, 0])
        alg_70_I_ionic_t = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Ca_1 = torch.zeros_like(y[:, 0]) + (i2 * (O_0 + O_1 + O_2 + O_3 + O_4) * ((Ca_SR1 - Ca_1) / V_1) - nu_SR * (Ca_1 / (Ca_1 + K_SR) / V_1) + L_e * ((Ca_SR1 - Ca_1) / V_1) + -tau_R * ((Ca_1 - Ca_2) / V_1) + -(k_P_on * Ca_1 * (P_tot + -Ca_P1 + -Mg_P1) + -k_P_off * Ca_P1) + -(k_CATP_on * Ca_1 * ATP1 + -k_CATP_off * Ca_ATP1))
        d_Ca_SR1 = torch.zeros_like(y[:, 0]) + (-(i2 * (O_0 + O_1 + O_2 + O_3 + O_4)) * ((Ca_SR1 - Ca_1) / V_SR1) + nu_SR * (Ca_1 / (Ca_1 + K_SR) / V_SR1) + -L_e * ((Ca_SR1 - Ca_1) / V_SR1) + -tau_SR_R * ((Ca_SR1 - Ca_SR2) / V_SR1) + -(k_Cs_on * Ca_SR1 * (Cs_tot - Ca_Cs1) + -k_Cs_off * Ca_Cs1))
        d_Ca_SR2 = torch.zeros_like(y[:, 0]) + (nu_SR * (Ca_2 / (Ca_2 + K_SR) / V_SR2) + -L_e * ((Ca_SR2 + -Ca_2) / V_SR2) + tau_SR_R * ((Ca_SR1 + -Ca_SR2) / V_SR2) + -(k_Cs_on * Ca_SR2 * (Cs_tot + -Ca_Cs2) + -k_Cs_off * Ca_Cs2) - 1000.0 / 1.0 * (A_p * (P_SR * (0.001 / 1.0) * Ca_SR2 - PP) * torch.where(P_SR * (0.001 / 1.0) * Ca_SR2 - PP > 0.0, 1.0, 0.0) * (0.001 / 1.0) * P_SR * Ca_SR2 - B_p * P_C_SR * (PP - P_SR * (0.001 / 1.0) * Ca_SR2) * torch.where(PP - P_SR * (0.001 / 1.0) * Ca_SR2 > 0.0, 1.0, 0.0)))
        d_Ca_P1 = torch.zeros_like(y[:, 0]) + (k_P_on * Ca_1 * (P_tot + -Ca_P1 + -Mg_P1) + -k_P_off * Ca_P1)
        d_Ca_P2 = torch.zeros_like(y[:, 0]) + (k_P_on * Ca_2 * (P_tot + -Ca_P2 + -Mg_P2) + -k_P_off * Ca_P2)
        d_Mg_P1 = torch.zeros_like(y[:, 0]) + (k_Mg_on * (P_tot + -Ca_P1 + -Mg_P1) * Mg1 + -k_Mg_off * Mg_P1)
        d_Mg_P2 = torch.zeros_like(y[:, 0]) + (k_Mg_on * (P_tot + -Ca_P2 + -Mg_P2) * Mg2 + -k_Mg_off * Mg_P2)
        d_Ca_Cs1 = torch.zeros_like(y[:, 0]) + (k_Cs_on * Ca_SR1 * (Cs_tot + -Ca_Cs1) + -k_Cs_off * Ca_Cs1)
        d_Ca_Cs2 = torch.zeros_like(y[:, 0]) + (k_Cs_on * Ca_SR2 * (Cs_tot + -Ca_Cs2) + -k_Cs_off * Ca_Cs2)
        d_Ca_ATP1 = torch.zeros_like(y[:, 0]) + (k_CATP_on * Ca_1 * ATP1 + -k_CATP_off * Ca_ATP1 + -tau_ATP * ((Ca_ATP1 + -Ca_ATP2) / V_1))
        d_Ca_ATP2 = torch.zeros_like(y[:, 0]) + (k_CATP_on * Ca_2 * ATP2 + -k_CATP_off * Ca_ATP2 + tau_ATP * ((Ca_ATP1 + -Ca_ATP2) / V_2))
        d_Mg_ATP1 = torch.zeros_like(y[:, 0]) + (k_MATP_on * Mg1 * ATP1 + -k_MATP_off * Mg_ATP1 + -tau_ATP * ((Mg_ATP1 + -Mg_ATP2) / V_1))
        d_Mg_ATP2 = torch.zeros_like(y[:, 0]) + (k_MATP_on * Mg2 * ATP2 + -k_MATP_off * Mg_ATP2 + tau_ATP * ((Mg_ATP1 + -Mg_ATP2) / V_2))
        d_ATP1 = torch.zeros_like(y[:, 0]) + (-(k_CATP_on * Ca_1 * ATP1 + -k_CATP_off * Ca_ATP1) + -(k_MATP_on * Mg1 * ATP1 + -k_MATP_off * Mg_ATP1) + -tau_ATP * ((ATP1 + -ATP2) / V_1))
        d_ATP2 = torch.zeros_like(y[:, 0]) + (-(k_CATP_on * Ca_2 * ATP2 + -k_CATP_off * Ca_ATP2) + -(k_MATP_on * Mg2 * ATP2 + -k_MATP_off * Mg_ATP2) + tau_ATP * ((ATP1 + -ATP2) / V_2))
        d_Mg1 = torch.zeros_like(y[:, 0]) + (-(k_Mg_on * (P_tot + -Ca_P1 + -Mg_P1) * Mg1 + -k_Mg_off * Mg_P1) + -(k_MATP_on * Mg1 * ATP1 + -k_MATP_off * Mg_ATP1) + -tau_Mg * ((Mg1 + -Mg2) / V_1))
        d_Mg2 = torch.zeros_like(y[:, 0]) + (-(k_Mg_on * (P_tot + -Ca_P2 + -Mg_P2) * Mg2 + -k_Mg_off * Mg_P2) + -(k_MATP_on * Mg2 * ATP2 + -k_MATP_off * Mg_ATP2) + tau_Mg * ((Mg1 + -Mg2) / V_2))
        d_Ca_CaT2 = torch.zeros_like(y[:, 0]) + (k_T_on * Ca_2 * Ca_T_2 + -k_T_off * Ca_CaT2 + -k_Ca_on * Ca_CaT2 + k_Ca_off * D_2)
        d_D_1 = torch.zeros_like(y[:, 0]) + (k_T_on * Ca_2 * D_0 + -k_T_off * D_1 + k_0_on * Ca_T_2 + -k_0_off * D_1 + -k_T_on * Ca_2 * D_1 + k_T_off * D_2)
        d_D_2 = torch.zeros_like(y[:, 0]) + (k_T_on * Ca_2 * D_1 + -k_T_off * D_2 + k_Ca_on * Ca_CaT2 + -k_Ca_off * D_2 + -f_o * D_2 + f_p * A_1 + g_o * A_2)
        d_A_1 = torch.zeros_like(y[:, 0]) + (f_o * D_2 + -f_p * A_1 + h_p * A_2 + -h_o * A_1)
        d_A_2 = torch.zeros_like(y[:, 0]) + (-h_p * A_2 + h_o * A_1 + -g_o * A_2)
        d_P = torch.zeros_like(y[:, 0]) + (0.001 / 1.0 * (h_o * A_1 - h_p * A_2) + -1.0 * b_p * P + -1.0 * k_p * ((P - P_SR) / V_2))
        d_P_SR = torch.zeros_like(y[:, 0]) + (k_p * ((P - P_SR) / V_SR2) - 1.0 * (A_p * (P_SR * (0.001 / 1.0) * Ca_SR2 - PP) * torch.where(P_SR * (0.001 / 1.0) * Ca_SR2 - PP > 0.0, 1.0, 0.0) * (0.001 / 1.0) * P_SR * Ca_SR2 - B_p * P_C_SR * (PP - P_SR * (0.001 / 1.0) * Ca_SR2) * torch.where(PP - P_SR * (0.001 / 1.0) * Ca_SR2 > 0.0, 1.0, 0.0)))
        d_P_C_SR = torch.zeros_like(y[:, 0]) + (1.0 * (A_p * (P_SR * (0.001 / 1.0) * Ca_SR2 - PP) * torch.where(P_SR * (0.001 / 1.0) * Ca_SR2 - PP > 0.0, 1.0, 0.0) * (0.001 / 1.0) * P_SR * Ca_SR2 - B_p * P_C_SR * (PP - P_SR * (0.001 / 1.0) * Ca_SR2) * torch.where(PP - P_SR * (0.001 / 1.0) * Ca_SR2 > 0.0, 1.0, 0.0)))
        alg_12_T_0 = T_tot + -Ca_T_2 + -Ca_CaT2 + -D_0 + -D_1 + -D_2 + -A_1 + -A_2
        d_Ca_2 = torch.zeros_like(y[:, 0]) + (-nu_SR * (Ca_2 / (Ca_2 + K_SR) / V_2) + L_e * ((Ca_SR2 + -Ca_2) / V_2) + tau_R * ((Ca_1 - Ca_2) / V_2) + -(k_T_on * Ca_2 * alg_12_T_0 + -k_T_off * Ca_T_2 + k_T_on * Ca_2 * Ca_T_2 + -k_T_off * Ca_CaT2 + k_T_on * Ca_2 * D_0 + -k_T_off * D_1 + k_T_on * Ca_2 * D_1 + -k_T_off * D_2) + -(k_P_on * Ca_2 * (P_tot + -Ca_P2 + -Mg_P2) + -k_P_off * Ca_P2) + -(k_CATP_on * Ca_2 * ATP2 + -k_CATP_off * Ca_ATP2))
        d_Ca_T_2 = torch.zeros_like(y[:, 0]) + (k_T_on * Ca_2 * alg_12_T_0 + -k_T_off * Ca_T_2 + -k_T_on * Ca_2 * Ca_T_2 + k_T_off * Ca_CaT2 + -k_0_on * Ca_T_2 + k_0_off * D_1)
        d_D_0 = torch.zeros_like(y[:, 0]) + (-k_T_on * Ca_2 * D_0 + k_T_off * D_1 + k_0_on * alg_12_T_0 + -k_0_off * D_0)
        alg_1_alpha_n = alpha_n_bar * ((vS - V_n) / (1.0 - torch.exp(-((vS - V_n) / K_alpha_n))))
        alg_14_beta_n = beta_n_bar * torch.exp(-((vS - V_n) / K_beta_n))
        d_n = torch.zeros_like(y[:, 0]) + (alg_1_alpha_n * (1.0 - n) - alg_14_beta_n * n)
        alg_2_h_K_inf = 1.0 / (1.0 + torch.exp((vS - V_h_K_inf) / A_h_K_inf))
        alg_15_tau_h_K = 1000.0 * torch.exp(-((vS + 40.0) / 25.75))
        d_h_K = torch.zeros_like(y[:, 0]) + ((alg_2_h_K_inf - h_K) / alg_15_tau_h_K)
        alg_4_alpha_m = alpha_m_bar * ((vS - V_m) / (1.0 - torch.exp(-((vS - V_m) / K_alpha_m))))
        alg_17_beta_m = beta_m_bar * torch.exp(-((vS - V_m) / K_beta_m))
        d_m = torch.zeros_like(y[:, 0]) + (alg_4_alpha_m * (1.0 - m) - alg_17_beta_m * m)
        alg_3_alpha_h = alpha_h_bar * torch.exp(-((vS - V_h) / K_alpha_h))
        alg_16_beta_h = beta_h_bar / (1.0 + torch.exp(-((vS - V_h) / K_beta_h)))
        d_h = torch.zeros_like(y[:, 0]) + (alg_3_alpha_h * (1.0 - h) - alg_16_beta_h * h)
        alg_5_S_inf = 1.0 / (1.0 + torch.exp((vS - V_S_inf) / A_S_inf))
        alg_18_tau_S = 8571.0 / (0.2 + 5.65 * ((vS + V_tau) / 100.0) ** 2.0)
        d_S = torch.zeros_like(y[:, 0]) + ((alg_5_S_inf - S) / alg_18_tau_S)
        alg_6_alpha_n_t = alpha_n_bar * ((vT - V_n) / (1.0 - torch.exp(-((vT - V_n) / K_alpha_n))))
        alg_19_beta_n_t = beta_n_bar * torch.exp(-((vT - V_n) / K_beta_n))
        d_n_t = torch.zeros_like(y[:, 0]) + (alg_6_alpha_n_t * (1.0 - n_t) - alg_19_beta_n_t * n_t)
        alg_7_h_K_inf_t = 1.0 / (1.0 + torch.exp((vT - V_h_K_inf) / A_h_K_inf))
        alg_20_tau_h_K_t = 1.0 * torch.exp(-((vT + 40.0) / 25.75))
        d_h_K_t = torch.zeros_like(y[:, 0]) + ((alg_7_h_K_inf_t - h_K_t) / alg_20_tau_h_K_t)
        alg_9_alpha_m_t = alpha_m_bar * ((vT - V_m) / (1.0 - torch.exp(-((vT - V_m) / K_alpha_m))))
        alg_22_beta_m_t = beta_m_bar * torch.exp(-((vT - V_m) / K_beta_m))
        d_m_t = torch.zeros_like(y[:, 0]) + (alg_9_alpha_m_t * (1.0 - m_t) - alg_22_beta_m_t * m_t)
        alg_8_alpha_h_t = alpha_h_bar * torch.exp(-((vT - V_h) / K_alpha_h))
        alg_21_beta_h_t = beta_h_bar / (1.0 + torch.exp(-((vT - V_h) / K_beta_h)))
        d_h_t = torch.zeros_like(y[:, 0]) + (alg_8_alpha_h_t * (1.0 - h_t) - alg_21_beta_h_t * h_t)
        alg_10_S_inf_t = 1.0 / (1.0 + torch.exp((vT - V_S_inf) / A_S_inf))
        alg_23_tau_S_t = 8571.0 / (0.2 + 5.65 * ((vT + V_tau) / 100.0) ** 2.0)
        d_S_t = torch.zeros_like(y[:, 0]) + ((alg_10_S_inf_t - S_t) / alg_23_tau_S_t)
        alg_11_k_C = 0.5 * alpha1 * torch.exp((vT - Vbar) / (8.0 * K))
        alg_24_k_Cm = 0.5 * alpha1 * torch.exp((Vbar - vT) / (8.0 * K))
        d_C_0 = torch.zeros_like(y[:, 0]) + (-k_L * C_0 + k_Lm * O_0 + -4.0 * alg_11_k_C * C_0 + alg_24_k_Cm * C_1)
        d_O_0 = torch.zeros_like(y[:, 0]) + (k_L * C_0 + -k_Lm * O_0 + -4.0 * alg_11_k_C * O_0 / f + f * alg_24_k_Cm * O_1)
        d_C_1 = torch.zeros_like(y[:, 0]) + (4.0 * alg_11_k_C * C_0 + -alg_24_k_Cm * C_1 + -k_L * C_1 / f + f * k_Lm * O_1 + -3.0 * alg_11_k_C * C_1 + 2.0 * alg_24_k_Cm * C_2)
        d_O_1 = torch.zeros_like(y[:, 0]) + (k_L * C_1 / f + -k_Lm * f * O_1 + 4.0 * alg_11_k_C * O_0 / f + -f * alg_24_k_Cm * O_1 + -3.0 * alg_11_k_C * O_1 / f + 2.0 * f * alg_24_k_Cm * O_2)
        d_C_2 = torch.zeros_like(y[:, 0]) + (3.0 * alg_11_k_C * C_1 + -2.0 * alg_24_k_Cm * C_2 + -k_L * C_2 / f ** 2.0 + f ** 2.0 * k_Lm * O_2 + -2.0 * alg_11_k_C * C_2 + 3.0 * alg_24_k_Cm * C_3)
        d_O_2 = torch.zeros_like(y[:, 0]) + (3.0 * alg_11_k_C * O_1 / f + -2.0 * f * alg_24_k_Cm * O_2 + k_L * C_2 / f ** 2.0 + -k_Lm * f ** 2.0 * O_2 + -2.0 * alg_11_k_C * O_2 / f + 3.0 * f * alg_24_k_Cm * O_3)
        d_C_3 = torch.zeros_like(y[:, 0]) + (2.0 * alg_11_k_C * C_2 + -3.0 * alg_24_k_Cm * C_3 + -k_L * C_3 / f ** 3.0 + k_Lm * f ** 3.0 * O_3 + -alg_11_k_C * C_3 + 4.0 * alg_24_k_Cm * C_4)
        d_O_3 = torch.zeros_like(y[:, 0]) + (k_L * C_3 / f ** 3.0 + -k_Lm * f ** 3.0 * O_3 + 2.0 * alg_11_k_C * O_2 / f + -3.0 * alg_24_k_Cm * f * O_3 + -alg_11_k_C * O_3 / f + 4.0 * f * alg_24_k_Cm * O_4)
        d_C_4 = torch.zeros_like(y[:, 0]) + (alg_11_k_C * C_3 + -4.0 * alg_24_k_Cm * C_4 + -k_L * C_4 / f ** 4.0 + k_Lm * f ** 4.0 * O_4)
        d_O_4 = torch.zeros_like(y[:, 0]) + (alg_11_k_C * O_3 / f + -4.0 * f * alg_24_k_Cm * O_4 + k_L * C_4 / f ** 4.0 + -k_Lm * f ** 4.0 * O_4)
        alg_30_J_K = vS * ((K_i - K_e * torch.exp(-1.0 * FF * vS / (RR * TT))) / (1.0 - torch.exp(-1.0 * FF * vS / (RR * TT))))
        alg_13_E_K = RR * TT / FF * torch.log(K_e / K_i)
        alg_37_K_R = K_e * torch.exp(-del_var * alg_13_E_K * (FF / (RR * TT)))
        alg_38_g_IR_bar = G_K * (alg_37_K_R ** 2.0 / (K_K + alg_37_K_R ** 2.0))
        alg_39_y = 1.0 - (1.0 + K_S * (1.0 + alg_37_K_R ** 2.0 / K_K) / (S_i ** 2.0 * torch.exp(2.0 * (1.0 - del_var) * vS * FF / (RR * TT)))) ** (-1.0)
        alg_40_g_IR = alg_38_g_IR_bar * alg_39_y
        alg_41_I_IR = alg_40_g_IR * torch.where(alg_30_J_K > 0.0, 1.0, 0.0) * (alg_30_J_K / 50.0)
        alg_42_g_DR = g_K_bar * n ** 4.0 * h_K
        alg_43_I_DR = alg_42_g_DR * (alg_30_J_K / 50.0)
        alg_47_sig = 1.0 / 7.0 * (torch.exp(Na_e / 67.3) - 1.0)
        alg_48_f1 = (1.0 + 0.12 * torch.exp(-0.1 * vS * (FF / (RR * TT))) + 0.04 * alg_47_sig * torch.exp(-(vS * (FF / (RR * TT))))) ** (-1.0)
        alg_49_I_NaK_bar = FF * (J_NaK_bar / ((1.0 + K_m_K / K_e) ** 2.0 * (1.0 + K_m_Na / Na_i) ** 3.0))
        alg_50_I_NaK = alg_49_I_NaK_bar * alg_48_f1
        d_K_e = torch.zeros_like(y[:, 0]) + ((alg_41_I_IR + alg_43_I_DR + I_K_rest + -2.0 * alg_50_I_NaK) / (1000.0 / 1.0 * FF * tsi3) + (K_t - K_e) / tau_K2)
        alg_45_g_Na = g_Na_bar * m ** 3.0 * h * S
        alg_44_J_Na = vS * ((Na_i - Na_e * torch.exp(-1.0 * FF * vS / (RR * TT))) / (1.0 - torch.exp(-1.0 * FF * vS / (RR * TT))))
        alg_46_I_Na = alg_45_g_Na * (alg_44_J_Na / 75.0)
        d_Na_e = torch.zeros_like(y[:, 0]) + ((alg_46_I_Na + I_Na_rest + 3.0 * alg_50_I_NaK) / (1000.0 / 1.0 * FF * tsi3) + (Na_t - Na_e) / tau_Na2)
        alg_0_I_T = 1000.0 / 1.0 * ((vS - vT) / R_a)
        alg_26_Cl_i = 156.5 / (5.0 + torch.exp(-FF * alg_13_E_K / (RR * TT)))
        alg_27_Cl_o = 156.5 - 5.0 * alg_26_Cl_i
        alg_34_J_Cl = vS * ((alg_26_Cl_i - alg_27_Cl_o * torch.exp(FF * vS / (RR * TT))) / (1.0 - torch.exp(FF * vS / (RR * TT))))
        alg_33_a = 1.0 / (1.0 + torch.exp((vS - V_a) / A_a))
        alg_35_g_Cl = g_Cl_bar * alg_33_a ** 4.0
        alg_36_I_Cl = alg_35_g_Cl * (alg_34_J_Cl / 45.0)
        alg_32_I_HH = torch.where((t >= 0.0) & (t < 0.5), 150.0, torch.where((t >= 50.0) & (t < 50.5), 150.0, torch.where((t >= 100.0) & (t < 100.5), 150.0, torch.where((t >= 150.0) & (t < 150.5), 150.0, torch.where((t >= 200.0) & (t < 200.5), 150.0, torch.where((t >= 250.0) & (t < 250.5), 150.0, torch.where((t >= 300.0) & (t < 300.5), 150.0, torch.where((t >= 350.0) & (t < 350.5), 150.0, torch.where((t >= 400.0) & (t < 400.5), 150.0, 0.0)))))))))
        alg_51_I_ionic_s = alg_36_I_Cl + alg_41_I_IR + alg_43_I_DR + alg_46_I_Na + alg_50_I_NaK + -alg_32_I_HH
        d_vS = torch.zeros_like(y[:, 0]) + (-((alg_51_I_ionic_s + alg_0_I_T) / C_m))
        alg_31_J_K_t = vT * ((K_i - K_t * torch.exp(-1.0 * FF * vT / (RR * TT))) / (1.0 - torch.exp(-1.0 * FF * vT / (RR * TT))))
        alg_25_E_K_t = RR * TT / FF * torch.log(K_t / K_i)
        alg_56_K_R_t = K_t * torch.exp(-del_var * alg_25_E_K_t * (FF / (RR * TT)))
        alg_57_g_IR_bar_t = G_K * (alg_56_K_R_t ** 2.0 / (K_K + alg_56_K_R_t ** 2.0))
        alg_58_y_t = 1.0 - (1.0 + K_S * (1.0 + alg_56_K_R_t ** 2.0 / K_K) / (S_i ** 2.0 * torch.exp(2.0 * (1.0 - del_var) * vT * FF / (RR * TT)))) ** (-1.0)
        alg_59_g_IR_t = alg_57_g_IR_bar_t * alg_58_y_t
        alg_60_I_IR_t = eta_IR * alg_59_g_IR_t * (alg_31_J_K_t / 50.0)
        alg_61_g_DR_t = g_K_bar * n_t ** 4.0 * h_K_t
        alg_62_I_DR_t = eta_DR * alg_61_g_DR_t * (alg_31_J_K_t / 50.0)
        alg_66_sig_t = 1.0 / 7.0 * (torch.exp(Na_t / 67.3) - 1.0)
        alg_67_f1_t = (1.0 + 0.12 * torch.exp(-0.1 * vT * (FF / (RR * TT))) + 0.04 * alg_66_sig_t * torch.exp(-(vT * (FF / (RR * TT))))) ** (-1.0)
        alg_68_I_NaK_bar_t = FF * (J_NaK_bar / ((1.0 + K_m_K / K_t) ** 2.0 * (1.0 + K_m_Na / Na_i) ** 3.0))
        alg_69_I_NaK_t = eta_NaK * alg_68_I_NaK_bar_t * alg_67_f1_t
        d_K_i = torch.zeros_like(y[:, 0]) + (-f_T * ((alg_60_I_IR_t + alg_62_I_DR_t + I_K_rest + -2.0 * alg_69_I_NaK_t) / (1000.0 / 1.0 * FF * tsi)) - (alg_41_I_IR + alg_43_I_DR + I_K_rest + -2.0 * alg_50_I_NaK) / (1000.0 / 1.0 * FF * tsi2))
        d_K_t = torch.zeros_like(y[:, 0]) + ((alg_60_I_IR_t + alg_62_I_DR_t + I_K_rest + -2.0 * alg_69_I_NaK_t) / (1000.0 / 1.0 * FF * tsi) - (K_t - K_e) / tau_K)
        alg_64_g_Na_t = g_Na_bar * m_t ** 3.0 * h_t * S_t
        alg_63_J_Na_t = vT * ((Na_i - Na_t * torch.exp(-1.0 * FF * vT / (RR * TT))) / (1.0 - torch.exp(-1.0 * FF * vT / (RR * TT))))
        alg_65_I_Na_t = eta_Na * alg_64_g_Na_t * (alg_63_J_Na_t / 75.0)
        d_Na_i = torch.zeros_like(y[:, 0]) + (-f_T * ((alg_65_I_Na_t + I_Na_rest + 3.0 * alg_69_I_NaK_t) / (1000.0 / 1.0 * FF * tsi)) - (alg_46_I_Na + I_Na_rest + 3.0 * alg_50_I_NaK) / (1000.0 / 1.0 * FF * tsi2))
        d_Na_t = torch.zeros_like(y[:, 0]) + ((alg_65_I_Na_t + I_Na_rest + 3.0 * alg_69_I_NaK_t) / (1000.0 / 1.0 * FF * tsi) - (Na_t - Na_e) / tau_Na)
        alg_28_Cl_i_t = 156.5 / (5.0 + torch.exp(-FF * alg_25_E_K_t / (RR * TT)))
        alg_29_Cl_o_t = 156.5 - 5.0 * alg_28_Cl_i_t
        alg_53_J_Cl_t = vT * ((alg_28_Cl_i_t - alg_29_Cl_o_t * torch.exp(FF * vT / (RR * TT))) / (1.0 - torch.exp(FF * vT / (RR * TT))))
        alg_52_a_t = 1.0 / (1.0 + torch.exp((vT - V_a) / A_a))
        alg_54_g_Cl_t = g_Cl_bar * alg_52_a_t ** 4.0
        alg_55_I_Cl_t = eta_Cl * alg_54_g_Cl_t * (alg_53_J_Cl_t / 45.0)
        alg_70_I_ionic_t = alg_55_I_Cl_t + alg_60_I_IR_t + alg_62_I_DR_t + alg_65_I_Na_t + alg_69_I_NaK_t
        d_vT = torch.zeros_like(y[:, 0]) + (-((alg_70_I_ionic_t - alg_0_I_T / gam) / C_m))

        return torch.stack(
            [d_vS, d_vT, d_K_t, d_K_i, d_K_e, d_Na_i, d_Na_t, d_Na_e, d_n, d_h_K, d_m, d_h, d_S, d_n_t, d_h_K_t, d_m_t, d_h_t, d_S_t, d_O_0, d_O_1, d_O_2, d_O_3, d_O_4, d_C_0, d_C_1, d_C_2, d_C_3, d_C_4, d_Ca_1, d_Ca_SR1, d_Ca_2, d_Ca_SR2, d_Ca_T_2, d_Ca_P1, d_Ca_P2, d_Mg_P1, d_Mg_P2, d_Ca_Cs1, d_Ca_Cs2, d_Ca_ATP1, d_Ca_ATP2, d_Mg_ATP1, d_Mg_ATP2, d_ATP1, d_ATP2, d_Mg1, d_Mg2, d_Ca_CaT2, d_D_0, d_D_1, d_D_2, d_A_1, d_A_2, d_P, d_P_SR, d_P_C_SR],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
