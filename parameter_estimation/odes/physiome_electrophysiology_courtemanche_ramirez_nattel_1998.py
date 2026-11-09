import torch
import torch.nn as nn
import math


class ElectrophysiologyCourtemancheRamirezNattel1998ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/courtemanche_ramirez_nattel_1998/model.py.
    Dynamic states: 21; independent constants: 45; derived constants used by rates: 4.
    Original constant slots: 49; algebraic slots: 75.
    """
    source_file = 'electrophysiology/courtemanche_ramirez_nattel_1998/model.py'
    source_state_dim = 21
    source_constant_dim = 49
    algebraic_dim = 75
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 21
        self.parameter_dim = 45

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8.3143, 'T': 310, 'F': 96.4867, 'Cm': 100, 'stim_start': 50, 'stim_end': 50000, 'stim_period': 1000, 'stim_duration': 2, 'stim_amplitude': -2000, 'g_Na': 7.8, 'Na_o': 140, 'g_K1': 0.09, 'K_o': 5.4, 'K_Q10': 3, 'g_to': 0.1652, 'g_Kr': 0.029411765, 'g_Ks': 0.12941176, 'g_Ca_L': 0.12375, 'Km_Na_i': 10, 'Km_K_o': 1.5, 'i_NaK_max': 0.59933874, 'g_B_Na': 0.0006744375, 'g_B_Ca': 0.001131, 'g_B_K': 0, 'Ca_o': 1.8, 'I_NaCa_max': 1600, 'K_mNa': 87.5, 'K_mCa': 1.38, 'K_sat': 0.1, 'gamma': 0.35, 'i_CaP_max': 0.275, 'K_rel': 30, 'tau_tr': 180, 'I_up_max': 0.005, 'K_up': 0.00092, 'Ca_up_max': 15, 'CMDN_max': 0.05, 'TRPN_max': 0.07, 'CSQN_max': 10, 'Km_CMDN': 0.00238, 'Km_TRPN': 0.0005, 'Km_CSQN': 0.8, 'V_cell': 20100, 'tau_f_Ca': 2, 'tau_u': 8}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -81.18, 'Na_i': 11.17, 'm': 0.002908, 'h': 0.9649, 'j': 0.9775, 'K_i': 139, 'oa': 0.03043, 'oi': 0.9992, 'ua': 0.004966, 'ui': 0.9986, 'xr': 3.296e-05, 'xs': 0.01869, 'Ca_i': 0.0001013, 'd': 0.0001367, 'f': 0.9996, 'f_Ca': 0.7755, 'Ca_rel': 1.488, 'u': 2.35e-112, 'v': 1, 'w': 0.9992, 'Ca_up': 1.488}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'Cm', 'stim_start', 'stim_end', 'stim_period', 'stim_duration', 'stim_amplitude', 'g_Na', 'Na_o', 'g_K1', 'K_o', 'K_Q10', 'g_to', 'g_Kr', 'g_Ks', 'g_Ca_L', 'Km_Na_i', 'Km_K_o', 'i_NaK_max', 'g_B_Na', 'g_B_Ca', 'g_B_K', 'Ca_o', 'I_NaCa_max', 'K_mNa', 'K_mCa', 'K_sat', 'gamma', 'i_CaP_max', 'K_rel', 'tau_tr', 'I_up_max', 'K_up', 'Ca_up_max', 'CMDN_max', 'TRPN_max', 'CSQN_max', 'Km_CMDN', 'Km_TRPN', 'Km_CSQN', 'V_cell', 'tau_f_Ca', 'tau_u']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'Na_i', 'm', 'h', 'j', 'K_i', 'oa', 'oi', 'ua', 'ui', 'xr', 'xs', 'Ca_i', 'd', 'f', 'f_Ca', 'Ca_rel', 'u', 'v', 'w', 'Ca_up']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (21)
        V = y[:, 0]
        Na_i = y[:, 1]
        m = y[:, 2]
        h = y[:, 3]
        j = y[:, 4]
        K_i = y[:, 5]
        oa = y[:, 6]
        oi = y[:, 7]
        ua = y[:, 8]
        ui = y[:, 9]
        xr = y[:, 10]
        xs = y[:, 11]
        Ca_i = y[:, 12]
        d = y[:, 13]
        f = y[:, 14]
        f_Ca = y[:, 15]
        Ca_rel = y[:, 16]
        u = y[:, 17]
        v = y[:, 18]
        w = y[:, 19]
        Ca_up = y[:, 20]

        # Independent parameters (source constant order)
        R = theta[:, 0]  # constants[0]
        T = theta[:, 1]  # constants[1]
        F = theta[:, 2]  # constants[2]
        Cm = theta[:, 3]  # constants[3]
        stim_start = theta[:, 4]  # constants[4]
        stim_end = theta[:, 5]  # constants[5]
        stim_period = theta[:, 6]  # constants[6]
        stim_duration = theta[:, 7]  # constants[7]
        stim_amplitude = theta[:, 8]  # constants[8]
        g_Na = theta[:, 9]  # constants[9]
        Na_o = theta[:, 10]  # constants[10]
        g_K1 = theta[:, 11]  # constants[11]
        K_o = theta[:, 12]  # constants[12]
        K_Q10 = theta[:, 13]  # constants[13]
        g_to = theta[:, 14]  # constants[14]
        g_Kr = theta[:, 15]  # constants[15]
        g_Ks = theta[:, 16]  # constants[16]
        g_Ca_L = theta[:, 17]  # constants[17]
        Km_Na_i = theta[:, 18]  # constants[18]
        Km_K_o = theta[:, 19]  # constants[19]
        i_NaK_max = theta[:, 20]  # constants[20]
        g_B_Na = theta[:, 21]  # constants[21]
        g_B_Ca = theta[:, 22]  # constants[22]
        g_B_K = theta[:, 23]  # constants[23]
        Ca_o = theta[:, 24]  # constants[24]
        I_NaCa_max = theta[:, 25]  # constants[25]
        K_mNa = theta[:, 26]  # constants[26]
        K_mCa = theta[:, 27]  # constants[27]
        K_sat = theta[:, 28]  # constants[28]
        gamma = theta[:, 29]  # constants[29]
        i_CaP_max = theta[:, 30]  # constants[30]
        K_rel = theta[:, 31]  # constants[31]
        tau_tr = theta[:, 32]  # constants[32]
        I_up_max = theta[:, 33]  # constants[33]
        K_up = theta[:, 34]  # constants[34]
        Ca_up_max = theta[:, 35]  # constants[35]
        CMDN_max = theta[:, 36]  # constants[36]
        TRPN_max = theta[:, 37]  # constants[37]
        CSQN_max = theta[:, 38]  # constants[38]
        Km_CMDN = theta[:, 39]  # constants[39]
        Km_TRPN = theta[:, 40]  # constants[40]
        Km_CSQN = theta[:, 41]  # constants[41]
        V_cell = theta[:, 42]  # constants[42]
        tau_f_Ca = theta[:, 43]  # constants[44]
        tau_u = theta[:, 44]  # constants[46]

        # Derived constants from the original initConsts()
        V_i = V_cell * 0.68  # constants[43]
        sigma = 1.0 / 7.0 * (torch.exp(Na_o / 67.3) - 1.0)  # constants[45]
        V_rel = 0.0048 * V_cell  # constants[47]
        V_up = 0.0552 * V_cell  # constants[48]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_j = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_oa = torch.zeros_like(y[:, 0])
        d_oi = torch.zeros_like(y[:, 0])
        d_ua = torch.zeros_like(y[:, 0])
        d_ui = torch.zeros_like(y[:, 0])
        d_xr = torch.zeros_like(y[:, 0])
        d_xs = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_d = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_f_Ca = torch.zeros_like(y[:, 0])
        d_Ca_rel = torch.zeros_like(y[:, 0])
        d_u = torch.zeros_like(y[:, 0])
        d_v = torch.zeros_like(y[:, 0])
        d_w = torch.zeros_like(y[:, 0])
        d_Ca_up = torch.zeros_like(y[:, 0])
        alg_0_i_st = torch.zeros_like(y[:, 0])
        alg_1_alpha_m = torch.zeros_like(y[:, 0])
        alg_2_alpha_h = torch.zeros_like(y[:, 0])
        alg_3_alpha_j = torch.zeros_like(y[:, 0])
        alg_4_alpha_oa = torch.zeros_like(y[:, 0])
        alg_5_alpha_oi = torch.zeros_like(y[:, 0])
        alg_6_alpha_ua = torch.zeros_like(y[:, 0])
        alg_7_alpha_ui = torch.zeros_like(y[:, 0])
        alg_8_alpha_xr = torch.zeros_like(y[:, 0])
        alg_9_alpha_xs = torch.zeros_like(y[:, 0])
        alg_10_d_infinity = torch.zeros_like(y[:, 0])
        alg_11_f_infinity = torch.zeros_like(y[:, 0])
        alg_12_f_Ca_infinity = torch.zeros_like(y[:, 0])
        alg_13_tau_w = torch.zeros_like(y[:, 0])
        alg_17_E_Na = torch.zeros_like(y[:, 0])
        alg_18_beta_m = torch.zeros_like(y[:, 0])
        alg_19_beta_h = torch.zeros_like(y[:, 0])
        alg_20_beta_j = torch.zeros_like(y[:, 0])
        alg_21_beta_oa = torch.zeros_like(y[:, 0])
        alg_22_beta_oi = torch.zeros_like(y[:, 0])
        alg_23_beta_ua = torch.zeros_like(y[:, 0])
        alg_24_beta_ui = torch.zeros_like(y[:, 0])
        alg_25_beta_xr = torch.zeros_like(y[:, 0])
        alg_26_beta_xs = torch.zeros_like(y[:, 0])
        alg_27_tau_d = torch.zeros_like(y[:, 0])
        alg_28_tau_f = torch.zeros_like(y[:, 0])
        alg_29_w_infinity = torch.zeros_like(y[:, 0])
        alg_30_i_Na = torch.zeros_like(y[:, 0])
        alg_31_m_inf = torch.zeros_like(y[:, 0])
        alg_32_h_inf = torch.zeros_like(y[:, 0])
        alg_33_j_inf = torch.zeros_like(y[:, 0])
        alg_34_tau_oa = torch.zeros_like(y[:, 0])
        alg_35_tau_oi = torch.zeros_like(y[:, 0])
        alg_36_tau_ua = torch.zeros_like(y[:, 0])
        alg_37_tau_ui = torch.zeros_like(y[:, 0])
        alg_38_tau_xr = torch.zeros_like(y[:, 0])
        alg_39_tau_xs = torch.zeros_like(y[:, 0])
        alg_40_E_K = torch.zeros_like(y[:, 0])
        alg_41_tau_m = torch.zeros_like(y[:, 0])
        alg_42_tau_h = torch.zeros_like(y[:, 0])
        alg_43_tau_j = torch.zeros_like(y[:, 0])
        alg_44_oa_infinity = torch.zeros_like(y[:, 0])
        alg_45_oi_infinity = torch.zeros_like(y[:, 0])
        alg_46_ua_infinity = torch.zeros_like(y[:, 0])
        alg_47_ui_infinity = torch.zeros_like(y[:, 0])
        alg_48_xr_infinity = torch.zeros_like(y[:, 0])
        alg_49_xs_infinity = torch.zeros_like(y[:, 0])
        alg_50_i_K1 = torch.zeros_like(y[:, 0])
        alg_51_i_to = torch.zeros_like(y[:, 0])
        alg_52_g_Kur = torch.zeros_like(y[:, 0])
        alg_53_i_Kur = torch.zeros_like(y[:, 0])
        alg_54_i_Kr = torch.zeros_like(y[:, 0])
        alg_55_i_Ks = torch.zeros_like(y[:, 0])
        alg_56_i_Ca_L = torch.zeros_like(y[:, 0])
        alg_57_f_NaK = torch.zeros_like(y[:, 0])
        alg_58_i_NaK = torch.zeros_like(y[:, 0])
        alg_59_E_Ca = torch.zeros_like(y[:, 0])
        alg_60_i_B_K = torch.zeros_like(y[:, 0])
        alg_61_i_B_Na = torch.zeros_like(y[:, 0])
        alg_62_i_B_Ca = torch.zeros_like(y[:, 0])
        alg_63_i_NaCa = torch.zeros_like(y[:, 0])
        alg_64_i_CaP = torch.zeros_like(y[:, 0])
        alg_65_i_rel = torch.zeros_like(y[:, 0])
        alg_66_Fn = torch.zeros_like(y[:, 0])
        alg_67_i_tr = torch.zeros_like(y[:, 0])
        alg_68_u_infinity = torch.zeros_like(y[:, 0])
        alg_69_tau_v = torch.zeros_like(y[:, 0])
        alg_70_i_up = torch.zeros_like(y[:, 0])
        alg_71_v_infinity = torch.zeros_like(y[:, 0])
        alg_72_i_up_leak = torch.zeros_like(y[:, 0])
        alg_73_B1 = torch.zeros_like(y[:, 0])
        alg_74_B2 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_12_f_Ca_infinity = (1.0 + Ca_i / 0.00035) ** (-1.0)
        d_f_Ca = torch.zeros_like(y[:, 0]) + ((alg_12_f_Ca_infinity - f_Ca) / tau_f_Ca)
        alg_10_d_infinity = (1.0 + torch.exp((V + 10.0) / -8.0)) ** (-1.0)
        alg_27_tau_d = torch.where(torch.abs(V + 10.0) < 1e-10, 4.579 / (1.0 + torch.exp((V + 10.0) / -6.24)), (1.0 - torch.exp((V + 10.0) / -6.24)) / (0.035 * (V + 10.0) * (1.0 + torch.exp((V + 10.0) / -6.24))))
        d_d = torch.zeros_like(y[:, 0]) + ((alg_10_d_infinity - d) / alg_27_tau_d)
        alg_11_f_infinity = torch.exp(-(V + 28.0) / 6.9) / (1.0 + torch.exp(-(V + 28.0) / 6.9))
        alg_28_tau_f = 9.0 * (0.0197 * torch.exp(-0.00113569 * (V + 10.0) ** 2.0) + 0.02) ** (-1.0)
        d_f = torch.zeros_like(y[:, 0]) + ((alg_11_f_infinity - f) / alg_28_tau_f)
        alg_13_tau_w = torch.where(torch.abs(V - 7.9) < 1e-10, 6.0 * 0.2 / 1.3, 6.0 * (1.0 - torch.exp(-(V - 7.9) / 5.0)) / ((1.0 + 0.3 * torch.exp(-(V - 7.9) / 5.0)) * 1.0 * (V - 7.9)))
        alg_29_w_infinity = 1.0 - (1.0 + torch.exp(-(V - 40.0) / 17.0)) ** (-1.0)
        d_w = torch.zeros_like(y[:, 0]) + ((alg_29_w_infinity - w) / alg_13_tau_w)
        alg_1_alpha_m = torch.where(V == -47.13, 3.2, 0.32 * (V + 47.13) / (1.0 - torch.exp(-0.1 * (V + 47.13))))
        alg_18_beta_m = 0.08 * torch.exp(-V / 11.0)
        alg_31_m_inf = alg_1_alpha_m / (alg_1_alpha_m + alg_18_beta_m)
        alg_41_tau_m = 1.0 / (alg_1_alpha_m + alg_18_beta_m)
        d_m = torch.zeros_like(y[:, 0]) + ((alg_31_m_inf - m) / alg_41_tau_m)
        alg_2_alpha_h = torch.where(V < -40.0, 0.135 * torch.exp((V + 80.0) / -6.8), 0.0)
        alg_19_beta_h = torch.where(V < -40.0, 3.56 * torch.exp(0.079 * V) + 310000.0 * torch.exp(0.35 * V), 1.0 / (0.13 * (1.0 + torch.exp((V + 10.66) / -11.1))))
        alg_32_h_inf = alg_2_alpha_h / (alg_2_alpha_h + alg_19_beta_h)
        alg_42_tau_h = 1.0 / (alg_2_alpha_h + alg_19_beta_h)
        d_h = torch.zeros_like(y[:, 0]) + ((alg_32_h_inf - h) / alg_42_tau_h)
        alg_3_alpha_j = torch.where(V < -40.0, (-127140.0 * torch.exp(0.2444 * V) - 3.474e-05 * torch.exp(-0.04391 * V)) * (V + 37.78) / (1.0 + torch.exp(0.311 * (V + 79.23))), 0.0)
        alg_20_beta_j = torch.where(V < -40.0, 0.1212 * torch.exp(-0.01052 * V) / (1.0 + torch.exp(-0.1378 * (V + 40.14))), 0.3 * torch.exp(-2.535e-07 * V) / (1.0 + torch.exp(-0.1 * (V + 32.0))))
        alg_33_j_inf = alg_3_alpha_j / (alg_3_alpha_j + alg_20_beta_j)
        alg_43_tau_j = 1.0 / (alg_3_alpha_j + alg_20_beta_j)
        d_j = torch.zeros_like(y[:, 0]) + ((alg_33_j_inf - j) / alg_43_tau_j)
        alg_4_alpha_oa = 0.65 * (torch.exp((V - -10.0) / -8.5) + torch.exp((V - -10.0 - 40.0) / -59.0)) ** (-1.0)
        alg_21_beta_oa = 0.65 * (2.5 + torch.exp((V - -10.0 + 72.0) / 17.0)) ** (-1.0)
        alg_34_tau_oa = (alg_4_alpha_oa + alg_21_beta_oa) ** (-1.0) / K_Q10
        alg_44_oa_infinity = (1.0 + torch.exp((V - -10.0 + 10.47) / -17.54)) ** (-1.0)
        d_oa = torch.zeros_like(y[:, 0]) + ((alg_44_oa_infinity - oa) / alg_34_tau_oa)
        alg_5_alpha_oi = (18.53 + 1.0 * torch.exp((V - -10.0 + 103.7) / 10.95)) ** (-1.0)
        alg_22_beta_oi = (35.56 + 1.0 * torch.exp((V - -10.0 - 8.74) / -7.44)) ** (-1.0)
        alg_35_tau_oi = (alg_5_alpha_oi + alg_22_beta_oi) ** (-1.0) / K_Q10
        alg_45_oi_infinity = (1.0 + torch.exp((V - -10.0 + 33.1) / 5.3)) ** (-1.0)
        d_oi = torch.zeros_like(y[:, 0]) + ((alg_45_oi_infinity - oi) / alg_35_tau_oi)
        alg_6_alpha_ua = 0.65 * (torch.exp((V - -10.0) / -8.5) + torch.exp((V - -10.0 - 40.0) / -59.0)) ** (-1.0)
        alg_23_beta_ua = 0.65 * (2.5 + torch.exp((V - -10.0 + 72.0) / 17.0)) ** (-1.0)
        alg_36_tau_ua = (alg_6_alpha_ua + alg_23_beta_ua) ** (-1.0) / K_Q10
        alg_46_ua_infinity = (1.0 + torch.exp((V - -10.0 + 20.3) / -9.6)) ** (-1.0)
        d_ua = torch.zeros_like(y[:, 0]) + ((alg_46_ua_infinity - ua) / alg_36_tau_ua)
        alg_7_alpha_ui = (21.0 + 1.0 * torch.exp((V - -10.0 - 195.0) / -28.0)) ** (-1.0)
        alg_24_beta_ui = 1.0 / torch.exp((V - -10.0 - 168.0) / -16.0)
        alg_37_tau_ui = (alg_7_alpha_ui + alg_24_beta_ui) ** (-1.0) / K_Q10
        alg_47_ui_infinity = (1.0 + torch.exp((V - -10.0 - 109.45) / 27.48)) ** (-1.0)
        d_ui = torch.zeros_like(y[:, 0]) + ((alg_47_ui_infinity - ui) / alg_37_tau_ui)
        alg_8_alpha_xr = torch.where(torch.abs(V + 14.1) < 1e-10, 0.0015, 0.0003 * (V + 14.1) / (1.0 - torch.exp((V + 14.1) / -5.0)))
        alg_25_beta_xr = torch.where(torch.abs(V - 3.3328) < 1e-10, 0.000378361, 7.3898e-05 * (V - 3.3328) / (torch.exp((V - 3.3328) / 5.1237) - 1.0))
        alg_38_tau_xr = (alg_8_alpha_xr + alg_25_beta_xr) ** (-1.0)
        alg_48_xr_infinity = (1.0 + torch.exp((V + 14.1) / -6.5)) ** (-1.0)
        d_xr = torch.zeros_like(y[:, 0]) + ((alg_48_xr_infinity - xr) / alg_38_tau_xr)
        alg_9_alpha_xs = torch.where(torch.abs(V - 19.9) < 1e-10, 0.00068, 4e-05 * (V - 19.9) / (1.0 - torch.exp((V - 19.9) / -17.0)))
        alg_26_beta_xs = torch.where(torch.abs(V - 19.9) < 1e-10, 0.000315, 3.5e-05 * (V - 19.9) / (torch.exp((V - 19.9) / 9.0) - 1.0))
        alg_39_tau_xs = 0.5 * (alg_9_alpha_xs + alg_26_beta_xs) ** (-1.0)
        alg_49_xs_infinity = (1.0 + torch.exp((V - 19.9) / -12.7)) ** (-0.5)
        d_xs = torch.zeros_like(y[:, 0]) + ((alg_49_xs_infinity - xs) / alg_39_tau_xs)
        alg_40_E_K = R * T / F * torch.log(K_o / K_i)
        alg_50_i_K1 = Cm * g_K1 * (V - alg_40_E_K) / (1.0 + torch.exp(0.07 * (V + 80.0)))
        alg_51_i_to = Cm * g_to * oa ** 3.0 * oi * (V - alg_40_E_K)
        alg_52_g_Kur = 0.005 + 0.05 / (1.0 + torch.exp((V - 15.0) / -13.0))
        alg_53_i_Kur = Cm * alg_52_g_Kur * ua ** 3.0 * ui * (V - alg_40_E_K)
        alg_54_i_Kr = Cm * g_Kr * xr * (V - alg_40_E_K) / (1.0 + torch.exp((V + 15.0) / 22.4))
        alg_55_i_Ks = Cm * g_Ks * xs ** 2.0 * (V - alg_40_E_K)
        alg_57_f_NaK = (1.0 + 0.1245 * torch.exp(-0.1 * F * V / (R * T)) + 0.0365 * sigma * torch.exp(-F * V / (R * T))) ** (-1.0)
        alg_58_i_NaK = Cm * i_NaK_max * alg_57_f_NaK * 1.0 / (1.0 + (Km_Na_i / Na_i) ** 1.5) * K_o / (K_o + Km_K_o)
        alg_60_i_B_K = Cm * g_B_K * (V - alg_40_E_K)
        d_K_i = torch.zeros_like(y[:, 0]) + ((2.0 * alg_58_i_NaK - (alg_50_i_K1 + alg_51_i_to + alg_53_i_Kur + alg_54_i_Kr + alg_55_i_Ks + alg_60_i_B_K)) / (V_i * F))
        alg_17_E_Na = R * T / F * torch.log(Na_o / Na_i)
        alg_30_i_Na = Cm * g_Na * m ** 3.0 * h * j * (V - alg_17_E_Na)
        alg_63_i_NaCa = Cm * I_NaCa_max * (torch.exp(gamma * F * V / (R * T)) * Na_i ** 3.0 * Ca_o - torch.exp((gamma - 1.0) * F * V / (R * T)) * Na_o ** 3.0 * Ca_i) / ((K_mNa ** 3.0 + Na_o ** 3.0) * (K_mCa + Ca_o) * (1.0 + K_sat * torch.exp((gamma - 1.0) * V * F / (R * T))))
        alg_61_i_B_Na = Cm * g_B_Na * (V - alg_17_E_Na)
        d_Na_i = torch.zeros_like(y[:, 0]) + ((-3.0 * alg_58_i_NaK - (3.0 * alg_63_i_NaCa + alg_61_i_B_Na + alg_30_i_Na)) / (V_i * F))
        alg_0_i_st = torch.where((t >= stim_start) & (t <= stim_end) & (t - stim_start - torch.floor((t - stim_start) / stim_period) * stim_period <= stim_duration), stim_amplitude, 0.0)
        alg_56_i_Ca_L = Cm * g_Ca_L * d * f * f_Ca * (V - 65.0)
        alg_64_i_CaP = Cm * i_CaP_max * Ca_i / (0.0005 + Ca_i)
        alg_59_E_Ca = R * T / (2.0 * F) * torch.log(Ca_o / Ca_i)
        alg_62_i_B_Ca = Cm * g_B_Ca * (V - alg_59_E_Ca)
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_30_i_Na + alg_50_i_K1 + alg_51_i_to + alg_53_i_Kur + alg_54_i_Kr + alg_55_i_Ks + alg_61_i_B_Na + alg_62_i_B_Ca + alg_58_i_NaK + alg_64_i_CaP + alg_63_i_NaCa + alg_56_i_Ca_L + alg_0_i_st) / Cm)
        alg_65_i_rel = K_rel * u ** 2.0 * v * w * (Ca_rel - Ca_i)
        alg_67_i_tr = (Ca_up - Ca_rel) / tau_tr
        d_Ca_rel = torch.zeros_like(y[:, 0]) + ((alg_67_i_tr - alg_65_i_rel) * (1.0 + CSQN_max * Km_CSQN / (Ca_rel + Km_CSQN) ** 2.0) ** (-1.0))
        alg_66_Fn = 1000.0 * (1e-15 * V_rel * alg_65_i_rel - 1e-15 / (2.0 * F) * (0.5 * alg_56_i_Ca_L - 0.2 * alg_63_i_NaCa))
        alg_68_u_infinity = (1.0 + torch.exp(-(alg_66_Fn - 3.4175e-13) / 1.367e-15)) ** (-1.0)
        d_u = torch.zeros_like(y[:, 0]) + ((alg_68_u_infinity - u) / tau_u)
        alg_69_tau_v = 1.91 + 2.09 * (1.0 + torch.exp(-(alg_66_Fn - 3.4175e-13) / 1.367e-15)) ** (-1.0)
        alg_71_v_infinity = 1.0 - (1.0 + torch.exp(-(alg_66_Fn - 6.835e-14) / 1.367e-15)) ** (-1.0)
        d_v = torch.zeros_like(y[:, 0]) + ((alg_71_v_infinity - v) / alg_69_tau_v)
        alg_70_i_up = I_up_max / (1.0 + K_up / Ca_i)
        alg_72_i_up_leak = I_up_max * Ca_up / Ca_up_max
        d_Ca_up = torch.zeros_like(y[:, 0]) + (alg_70_i_up - (alg_72_i_up_leak + alg_67_i_tr * V_rel / V_up))
        alg_73_B1 = (2.0 * alg_63_i_NaCa - (alg_64_i_CaP + alg_56_i_Ca_L + alg_62_i_B_Ca)) / (2.0 * V_i * F) + (V_up * (alg_72_i_up_leak - alg_70_i_up) + alg_65_i_rel * V_rel) / V_i
        alg_74_B2 = 1.0 + TRPN_max * Km_TRPN / (Ca_i + Km_TRPN) ** 2.0 + CMDN_max * Km_CMDN / (Ca_i + Km_CMDN) ** 2.0
        d_Ca_i = torch.zeros_like(y[:, 0]) + (alg_73_B1 / alg_74_B2)

        return torch.stack(
            [d_V, d_Na_i, d_m, d_h, d_j, d_K_i, d_oa, d_oi, d_ua, d_ui, d_xr, d_xs, d_Ca_i, d_d, d_f, d_f_Ca, d_Ca_rel, d_u, d_v, d_w, d_Ca_up],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
