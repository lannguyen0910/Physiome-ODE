import torch
import torch.nn as nn
import math


class ElectrophysiologyInadaAN2009ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/inada_AN_2009/model.py.
    Dynamic states: 29; independent constants: 53; derived constants used by rates: 10.
    Original constant slots: 63; algebraic slots: 82.
    """
    source_file = 'electrophysiology/inada_AN_2009/model.py'
    source_state_dim = 29
    source_constant_dim = 63
    algebraic_dim = 82
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
        self.parameter_dim = 53

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8314.472, 'T': 310, 'F': 96485.3415, 'C': 4e-05, 'stim_start': 0.1, 'stim_end': 99999, 'stim_period': 1, 'stim_duration': 0.001, 'stim_amplitude': -2, 'g_f': 0, 'ACh': 0, 'g_Kr': 0.0015, 'Ki': 140, 'Kc': 5.4, 'g_K1': 0.0125, 'g_b': 0.0018, 'E_b': -52.5, 'I_p': 0.0246, 'Nai': 8, 'kNaCa': 5.916, 'Qci': 0.1369, 'Qn': 0.4315, 'Qco': 0, 'Kci': 0.0207, 'K1ni': 395.3, 'K2ni': 2.289, 'K3ni': 26.44, 'Kcni': 26.44, 'K3no': 4.663, 'K1no': 1628, 'K2no': 561.4, 'Kco': 3.663, 'Cao': 2, 'Nao': 140, 'g_Na': 5e-07, 'delta_m': 1e-05, 'g_CaL': 0.0185, 'E_CaL': 62.5, 'act_shift': 0, 'slope_factor_act': -6.61, 'inact_shift': -5, 'inact_shift_2': -5, 'g_to': 0.02, 'E_st': -37.4, 'g_st': 0, 'g_ACh_max': 0.0198, 'K_ACh': 0.00035, 'alpha_achf': 73.1, 'alpha_achs': 3.7, 'V_cell': 4.39823e-06, 'P_rel': 1805.6, 'K_up': 0.0006, 'tau_tr': 0.06}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -71.5535452525735, 'y': 0.231892950445813, 'paf': 0.000907827363439979, 'pas': 0.00289901267127429, 'pik': 0.987889953123897, 'Casub': 2.86962804165375e-05, 'm': 0.0104794040295793, 'h1': 0.792210965943567, 'h2': 0.78834840378483, 'd': 3.2286733432613e-05, 'f': 0.998822085003546, 'f2': 0.998815467202695, 'r': 0.00802880907824072, 'q_fast': 0.995494395556732, 'q_slow': 0.547966933708077, 'qa': 0.0758461021803425, 'qi': 0.943345608766177, 'achf': 0.760265641624297, 'achs': 0.764664867332735, 'Cai': 3.10430405261017e-05, 'Ca_up': 0.667220893111124, 'Ca_rel': 0.557458998353581, 'f_TC': 0.0061557815047835, 'f_TMC': 0.112200926320798, 'f_TMM': 0.784309464255762, 'f_CMi': 0.012895686161953, 'f_CMs': 0.0119201823967662, 'f_CQ': 0.400684024879925, 'f_CSL': 8.91124266400812e-06}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'C', 'stim_start', 'stim_end', 'stim_period', 'stim_duration', 'stim_amplitude', 'g_f', 'ACh', 'g_Kr', 'Ki', 'Kc', 'g_K1', 'g_b', 'E_b', 'I_p', 'Nai', 'kNaCa', 'Qci', 'Qn', 'Qco', 'Kci', 'K1ni', 'K2ni', 'K3ni', 'Kcni', 'K3no', 'K1no', 'K2no', 'Kco', 'Cao', 'Nao', 'g_Na', 'delta_m', 'g_CaL', 'E_CaL', 'act_shift', 'slope_factor_act', 'inact_shift', 'inact_shift_2', 'g_to', 'E_st', 'g_st', 'g_ACh_max', 'K_ACh', 'alpha_achf', 'alpha_achs', 'V_cell', 'P_rel', 'K_up', 'tau_tr']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'y', 'paf', 'pas', 'pik', 'Casub', 'm', 'h1', 'h2', 'd', 'f', 'f2', 'r', 'q_fast', 'q_slow', 'qa', 'qi', 'achf', 'achs', 'Cai', 'Ca_up', 'Ca_rel', 'f_TC', 'f_TMC', 'f_TMM', 'f_CMi', 'f_CMs', 'f_CQ', 'f_CSL']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (29)
        V = y[:, 0]
        state_y = y[:, 1]
        paf = y[:, 2]
        pas = y[:, 3]
        pik = y[:, 4]
        Casub = y[:, 5]
        m = y[:, 6]
        h1 = y[:, 7]
        h2 = y[:, 8]
        d = y[:, 9]
        f = y[:, 10]
        f2 = y[:, 11]
        r = y[:, 12]
        q_fast = y[:, 13]
        q_slow = y[:, 14]
        qa = y[:, 15]
        qi = y[:, 16]
        achf = y[:, 17]
        achs = y[:, 18]
        Cai = y[:, 19]
        Ca_up = y[:, 20]
        Ca_rel = y[:, 21]
        f_TC = y[:, 22]
        f_TMC = y[:, 23]
        f_TMM = y[:, 24]
        f_CMi = y[:, 25]
        f_CMs = y[:, 26]
        f_CQ = y[:, 27]
        f_CSL = y[:, 28]

        # Independent parameters (source constant order)
        R = theta[:, 0]  # constants[0]
        T = theta[:, 1]  # constants[1]
        F = theta[:, 2]  # constants[2]
        C = theta[:, 3]  # constants[3]
        stim_start = theta[:, 4]  # constants[4]
        stim_end = theta[:, 5]  # constants[5]
        stim_period = theta[:, 6]  # constants[6]
        stim_duration = theta[:, 7]  # constants[7]
        stim_amplitude = theta[:, 8]  # constants[8]
        g_f = theta[:, 9]  # constants[9]
        ACh = theta[:, 10]  # constants[10]
        g_Kr = theta[:, 11]  # constants[11]
        Ki = theta[:, 12]  # constants[12]
        Kc = theta[:, 13]  # constants[13]
        g_K1 = theta[:, 14]  # constants[14]
        g_b = theta[:, 15]  # constants[15]
        E_b = theta[:, 16]  # constants[16]
        I_p = theta[:, 17]  # constants[17]
        Nai = theta[:, 18]  # constants[18]
        kNaCa = theta[:, 19]  # constants[19]
        Qci = theta[:, 20]  # constants[20]
        Qn = theta[:, 21]  # constants[21]
        Qco = theta[:, 22]  # constants[22]
        Kci = theta[:, 23]  # constants[23]
        K1ni = theta[:, 24]  # constants[24]
        K2ni = theta[:, 25]  # constants[25]
        K3ni = theta[:, 26]  # constants[26]
        Kcni = theta[:, 27]  # constants[27]
        K3no = theta[:, 28]  # constants[28]
        K1no = theta[:, 29]  # constants[29]
        K2no = theta[:, 30]  # constants[30]
        Kco = theta[:, 31]  # constants[31]
        Cao = theta[:, 32]  # constants[32]
        Nao = theta[:, 33]  # constants[33]
        g_Na = theta[:, 34]  # constants[34]
        delta_m = theta[:, 35]  # constants[35]
        g_CaL = theta[:, 36]  # constants[36]
        E_CaL = theta[:, 37]  # constants[37]
        act_shift = theta[:, 38]  # constants[38]
        slope_factor_act = theta[:, 39]  # constants[39]
        inact_shift = theta[:, 40]  # constants[40]
        inact_shift_2 = theta[:, 41]  # constants[41]
        g_to = theta[:, 42]  # constants[42]
        E_st = theta[:, 43]  # constants[43]
        g_st = theta[:, 44]  # constants[44]
        g_ACh_max = theta[:, 45]  # constants[45]
        K_ACh = theta[:, 46]  # constants[46]
        alpha_achf = theta[:, 47]  # constants[47]
        alpha_achs = theta[:, 48]  # constants[48]
        V_cell = theta[:, 49]  # constants[49]
        P_rel = theta[:, 50]  # constants[50]
        K_up = theta[:, 51]  # constants[51]
        tau_tr = theta[:, 52]  # constants[52]

        # Derived constants from the original initConsts()
        RTONF = R * T / F  # constants[53]
        k34 = Nao / (K3no + Nao)  # constants[54]
        E_K = RTONF * torch.log(Kc / Ki)  # constants[55]
        E_Na = RTONF * torch.log(Nao / Nai)  # constants[56]
        E_K_2 = RTONF * torch.log(Kc / Ki)  # constants[57]
        k43 = Nai / (K3ni + Nai)  # constants[58]
        V_up = 0.0116 * V_cell  # constants[59]
        V_rel = 0.0012 * V_cell  # constants[60]
        V_sub = 0.01 * V_cell  # constants[61]
        Vi = 0.46 * V_cell - V_sub  # constants[62]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])
        d_paf = torch.zeros_like(y[:, 0])
        d_pas = torch.zeros_like(y[:, 0])
        d_pik = torch.zeros_like(y[:, 0])
        d_Casub = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h1 = torch.zeros_like(y[:, 0])
        d_h2 = torch.zeros_like(y[:, 0])
        d_d = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_f2 = torch.zeros_like(y[:, 0])
        d_r = torch.zeros_like(y[:, 0])
        d_q_fast = torch.zeros_like(y[:, 0])
        d_q_slow = torch.zeros_like(y[:, 0])
        d_qa = torch.zeros_like(y[:, 0])
        d_qi = torch.zeros_like(y[:, 0])
        d_achf = torch.zeros_like(y[:, 0])
        d_achs = torch.zeros_like(y[:, 0])
        d_Cai = torch.zeros_like(y[:, 0])
        d_Ca_up = torch.zeros_like(y[:, 0])
        d_Ca_rel = torch.zeros_like(y[:, 0])
        d_f_TC = torch.zeros_like(y[:, 0])
        d_f_TMC = torch.zeros_like(y[:, 0])
        d_f_TMM = torch.zeros_like(y[:, 0])
        d_f_CMi = torch.zeros_like(y[:, 0])
        d_f_CMs = torch.zeros_like(y[:, 0])
        d_f_CQ = torch.zeros_like(y[:, 0])
        d_f_CSL = torch.zeros_like(y[:, 0])
        alg_0_y_inf = torch.zeros_like(y[:, 0])
        alg_1_paf_infinity = torch.zeros_like(y[:, 0])
        alg_2_pas_infinity = torch.zeros_like(y[:, 0])
        alg_3_pik_infinity = torch.zeros_like(y[:, 0])
        alg_4_E0_m = torch.zeros_like(y[:, 0])
        alg_5_alpha_h1 = torch.zeros_like(y[:, 0])
        alg_6_alpha_h2 = torch.zeros_like(y[:, 0])
        alg_7_alpha_d = torch.zeros_like(y[:, 0])
        alg_8_f_inf = torch.zeros_like(y[:, 0])
        alg_9_f2_inf = torch.zeros_like(y[:, 0])
        alg_10_r_infinity = torch.zeros_like(y[:, 0])
        alg_11_qfast_infinity = torch.zeros_like(y[:, 0])
        alg_12_qslow_infinity = torch.zeros_like(y[:, 0])
        alg_13_qa_infinity = torch.zeros_like(y[:, 0])
        alg_14_alpha_qi = torch.zeros_like(y[:, 0])
        alg_15_beta_achf = torch.zeros_like(y[:, 0])
        alg_16_beta_achs = torch.zeros_like(y[:, 0])
        alg_17_i_Stim = torch.zeros_like(y[:, 0])
        alg_18_diff_f_TMM = torch.zeros_like(y[:, 0])
        alg_19_tau_y = torch.zeros_like(y[:, 0])
        alg_20_tau_paf = torch.zeros_like(y[:, 0])
        alg_21_tau_pas = torch.zeros_like(y[:, 0])
        alg_22_alpha_pik = torch.zeros_like(y[:, 0])
        alg_23_alpha_m = torch.zeros_like(y[:, 0])
        alg_24_beta_h1 = torch.zeros_like(y[:, 0])
        alg_25_beta_h2 = torch.zeros_like(y[:, 0])
        alg_26_beta_d = torch.zeros_like(y[:, 0])
        alg_27_tau_f = torch.zeros_like(y[:, 0])
        alg_28_tau_f2 = torch.zeros_like(y[:, 0])
        alg_29_tau_r = torch.zeros_like(y[:, 0])
        alg_30_tau_qfast = torch.zeros_like(y[:, 0])
        alg_31_tau_qslow = torch.zeros_like(y[:, 0])
        alg_32_alpha_qa = torch.zeros_like(y[:, 0])
        alg_33_beta_qi = torch.zeros_like(y[:, 0])
        alg_34_i_f = torch.zeros_like(y[:, 0])
        alg_35_beta_pik = torch.zeros_like(y[:, 0])
        alg_36_beta_m = torch.zeros_like(y[:, 0])
        alg_37_h1_inf = torch.zeros_like(y[:, 0])
        alg_38_h2_inf = torch.zeros_like(y[:, 0])
        alg_39_d_inf = torch.zeros_like(y[:, 0])
        alg_40_beta_qa = torch.zeros_like(y[:, 0])
        alg_41_qi_infinity = torch.zeros_like(y[:, 0])
        alg_42_i_Kr = torch.zeros_like(y[:, 0])
        alg_43_tau_pik = torch.zeros_like(y[:, 0])
        alg_44_tau_h1 = torch.zeros_like(y[:, 0])
        alg_45_tau_h2 = torch.zeros_like(y[:, 0])
        alg_46_tau_d = torch.zeros_like(y[:, 0])
        alg_47_tau_qa = torch.zeros_like(y[:, 0])
        alg_48_tau_qi = torch.zeros_like(y[:, 0])
        alg_49_g_K1_prime = torch.zeros_like(y[:, 0])
        alg_50_i_K1 = torch.zeros_like(y[:, 0])
        alg_51_i_b = torch.zeros_like(y[:, 0])
        alg_52_i_p = torch.zeros_like(y[:, 0])
        alg_53_do = torch.zeros_like(y[:, 0])
        alg_54_k32 = torch.zeros_like(y[:, 0])
        alg_55_k23 = torch.zeros_like(y[:, 0])
        alg_56_k21 = torch.zeros_like(y[:, 0])
        alg_57_di = torch.zeros_like(y[:, 0])
        alg_58_k41 = torch.zeros_like(y[:, 0])
        alg_59_k14 = torch.zeros_like(y[:, 0])
        alg_60_k12 = torch.zeros_like(y[:, 0])
        alg_61_x1 = torch.zeros_like(y[:, 0])
        alg_62_x2 = torch.zeros_like(y[:, 0])
        alg_63_x3 = torch.zeros_like(y[:, 0])
        alg_64_x4 = torch.zeros_like(y[:, 0])
        alg_65_i_NaCa = torch.zeros_like(y[:, 0])
        alg_66_i_Na = torch.zeros_like(y[:, 0])
        alg_67_i_to = torch.zeros_like(y[:, 0])
        alg_68_i_st = torch.zeros_like(y[:, 0])
        alg_69_g_ACh = torch.zeros_like(y[:, 0])
        alg_70_i_ACh = torch.zeros_like(y[:, 0])
        alg_71_i_CaL = torch.zeros_like(y[:, 0])
        alg_72_i_diff = torch.zeros_like(y[:, 0])
        alg_73_i_up = torch.zeros_like(y[:, 0])
        alg_74_i_tr = torch.zeros_like(y[:, 0])
        alg_75_diff_f_TC = torch.zeros_like(y[:, 0])
        alg_76_i_rel = torch.zeros_like(y[:, 0])
        alg_77_diff_f_TMC = torch.zeros_like(y[:, 0])
        alg_78_diff_f_CMi = torch.zeros_like(y[:, 0])
        alg_79_diff_f_CMs = torch.zeros_like(y[:, 0])
        alg_80_diff_f_CQ = torch.zeros_like(y[:, 0])
        alg_81_diff_f_CSL = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_15_beta_achf = 120.0 / (1.0 + torch.exp(-(V + 50.0) / 15.0))
        d_achf = torch.zeros_like(y[:, 0]) + (alpha_achf * (1.0 - achf) - alg_15_beta_achf * achf)
        alg_16_beta_achs = 5.82 / (1.0 + torch.exp(-(V + 50.0) / 15.0))
        d_achs = torch.zeros_like(y[:, 0]) + (alpha_achs * (1.0 - achs) - alg_16_beta_achs * achs)
        alg_18_diff_f_TMM = 2277.0 * 2.5 * (1.0 - f_TMC - f_TMM) - 751.0 * f_TMM
        d_f_TMM = torch.zeros_like(y[:, 0]) + (alg_18_diff_f_TMM)
        alg_0_y_inf = 1.0 / (1.0 + torch.exp((V + 83.19 - -7.2 * ACh ** 0.69 / (0.0004161555358336801 + ACh ** 0.69)) / 13.56))
        alg_19_tau_y = 0.25 + 2.0 * torch.exp(-(V + 70.0) ** 2.0 / 500.0)
        d_state_y = torch.zeros_like(y[:, 0]) + ((alg_0_y_inf - state_y) / alg_19_tau_y)
        alg_1_paf_infinity = 1.0 / (1.0 + torch.exp((V + 10.22) / -8.5))
        alg_20_tau_paf = 1.0 / (17.0 * torch.exp(0.0398 * V) + 0.211 * torch.exp(-0.051 * V))
        d_paf = torch.zeros_like(y[:, 0]) + ((alg_1_paf_infinity - paf) / alg_20_tau_paf)
        alg_2_pas_infinity = 1.0 / (1.0 + torch.exp((V + 10.22) / -8.5))
        alg_21_tau_pas = 0.33581 + 0.90673 * torch.exp(-(V + 10.0) ** 2.0 / 988.05)
        d_pas = torch.zeros_like(y[:, 0]) + ((alg_2_pas_infinity - pas) / alg_21_tau_pas)
        alg_8_f_inf = 1.0 / (1.0 + torch.exp((V - (-24.0 + inact_shift)) / 6.31))
        alg_27_tau_f = 0.01 + 0.1539 * torch.exp(-(V + 40.0) ** 2.0 / 185.67)
        d_f = torch.zeros_like(y[:, 0]) + ((alg_8_f_inf - f) / alg_27_tau_f)
        alg_9_f2_inf = 1.0 / (1.0 + torch.exp((V - (-24.0 + inact_shift_2)) / 6.31))
        alg_28_tau_f2 = 0.06 + 0.48076 * 2.25 * torch.exp(-(V - -40.0) ** 2.0 / 138.04)
        d_f2 = torch.zeros_like(y[:, 0]) + ((alg_9_f2_inf - f2) / alg_28_tau_f2)
        alg_29_tau_r = 0.000596 + 0.003118 / (1.037 * torch.exp(0.09 * (V + 30.61)) + 0.396 * torch.exp(-0.12 * (V + 23.84)))
        alg_10_r_infinity = 1.0 / (1.0 + torch.exp((V - 7.44) / -16.4))
        d_r = torch.zeros_like(y[:, 0]) + ((alg_10_r_infinity - r) / alg_29_tau_r)
        alg_30_tau_qfast = 0.01266 + 4.72716 / (1.0 + torch.exp((V + 154.5) / 23.96))
        alg_11_qfast_infinity = 1.0 / (1.0 + torch.exp((V + 33.8) / 6.12))
        d_q_fast = torch.zeros_like(y[:, 0]) + ((alg_11_qfast_infinity - q_fast) / alg_30_tau_qfast)
        alg_31_tau_qslow = 0.1 + 4.0 * torch.exp(-(V + 65.0) ** 2.0 / 500.0)
        alg_12_qslow_infinity = 1.0 / (1.0 + torch.exp((V + 33.8) / 6.12))
        d_q_slow = torch.zeros_like(y[:, 0]) + ((alg_12_qslow_infinity - q_slow) / alg_31_tau_qslow)
        alg_4_E0_m = V + 44.4
        alg_23_alpha_m = torch.where(torch.abs(alg_4_E0_m) < delta_m, -460.0 * -12.673 / torch.exp(alg_4_E0_m / -12.673), -460.0 * alg_4_E0_m / (torch.exp(alg_4_E0_m / -12.673) - 1.0))
        alg_36_beta_m = 18400.0 * torch.exp(alg_4_E0_m / -12.673)
        d_m = torch.zeros_like(y[:, 0]) + (alg_23_alpha_m * (1.0 - m) - alg_36_beta_m * m)
        alg_3_pik_infinity = 1.0 / (1.0 + torch.exp((V + 4.9) / 15.14)) * (1.0 - 0.3 * torch.exp(-V ** 2.0 / 500.0))
        alg_22_alpha_pik = 92.01 * torch.exp(-0.0183 * V)
        alg_35_beta_pik = 603.6 * torch.exp(0.00942 * V)
        alg_43_tau_pik = 1.0 / (alg_22_alpha_pik + alg_35_beta_pik)
        d_pik = torch.zeros_like(y[:, 0]) + ((alg_3_pik_infinity - pik) / alg_43_tau_pik)
        alg_5_alpha_h1 = 44.9 * torch.exp((V + 66.9) / -5.57)
        alg_24_beta_h1 = 1491.0 / (1.0 + 323.3 * torch.exp((V + 94.6) / -12.9))
        alg_37_h1_inf = alg_5_alpha_h1 / (alg_5_alpha_h1 + alg_24_beta_h1)
        alg_44_tau_h1 = 0.03 / (1.0 + torch.exp((V + 40.0) / 6.0)) + 0.00035
        d_h1 = torch.zeros_like(y[:, 0]) + ((alg_37_h1_inf - h1) / alg_44_tau_h1)
        alg_6_alpha_h2 = 44.9 * torch.exp((V + 66.9) / -5.57)
        alg_25_beta_h2 = 1491.0 / (1.0 + 323.3 * torch.exp((V + 94.6) / -12.9))
        alg_38_h2_inf = alg_6_alpha_h2 / (alg_6_alpha_h2 + alg_25_beta_h2)
        alg_45_tau_h2 = 0.12 / (1.0 + torch.exp((V + 60.0) / 2.0)) + 0.00295
        d_h2 = torch.zeros_like(y[:, 0]) + ((alg_38_h2_inf - h2) / alg_45_tau_h2)
        alg_39_d_inf = 1.0 / (1.0 + torch.exp((V - (-3.2 + act_shift)) / slope_factor_act))
        alg_7_alpha_d = -26.12 * (V + 35.0) / (torch.exp((V + 35.0) / -2.5) - 1.0) + -78.11 * V / (torch.exp(-0.208 * V) - 1.0)
        alg_26_beta_d = 10.52 * (V - 5.0) / (torch.exp(0.4 * (V - 5.0)) - 1.0)
        alg_46_tau_d = 1.0 / (alg_7_alpha_d + alg_26_beta_d)
        d_d = torch.zeros_like(y[:, 0]) + ((alg_39_d_inf - d) / alg_46_tau_d)
        alg_32_alpha_qa = 1.0 / (0.15 * torch.exp(-V / 11.0) + 0.2 * torch.exp(-V / 700.0))
        alg_40_beta_qa = 1.0 / (16.0 * torch.exp(V / 8.0) + 15.0 * torch.exp(V / 50.0))
        alg_47_tau_qa = 0.001 / (alg_32_alpha_qa + alg_40_beta_qa)
        alg_13_qa_infinity = 1.0 / (1.0 + torch.exp((V - -49.1) / -8.98))
        d_qa = torch.zeros_like(y[:, 0]) + ((alg_13_qa_infinity - qa) / alg_47_tau_qa)
        alg_14_alpha_qi = 0.1504 / (3100.0 * torch.exp(V / 13.0) + 700.0 * torch.exp(V / 70.0))
        alg_33_beta_qi = 0.1504 / (95.0 * torch.exp(-V / 10.0) + 50.0 * torch.exp(-V / 700.0)) + 0.000229 / (1.0 + torch.exp(-V / 5.0))
        alg_48_tau_qi = 0.001 / (alg_14_alpha_qi + alg_33_beta_qi)
        alg_41_qi_infinity = alg_14_alpha_qi / (alg_14_alpha_qi + alg_33_beta_qi)
        d_qi = torch.zeros_like(y[:, 0]) + ((alg_41_qi_infinity - qi) / alg_48_tau_qi)
        alg_66_i_Na = g_Na * m ** 3.0 * (0.635 * h1 + 0.365 * h2) * Nao * V * F / RTONF * (torch.exp((V - E_Na) / RTONF) - 1.0) / (torch.exp(V / RTONF) - 1.0)
        alg_69_g_ACh = g_ACh_max * achf * achs * ACh ** 1.5 / (K_ACh ** 1.5 + ACh ** 1.5)
        alg_70_i_ACh = alg_69_g_ACh * Kc / (10.0 + Kc) * (V - E_K) / (1.0 + torch.exp((V - E_K - 140.0) / (2.5 * RTONF)))
        alg_71_i_CaL = g_CaL * d * (0.675 * f + 0.325 * f2) * (V - E_CaL) * (1.0 - alg_70_i_ACh * ACh / (9e-05 + ACh) / 1.0)
        alg_67_i_to = g_to * r * (0.45 * q_fast + 0.55 * q_slow) * (V - E_K_2)
        alg_42_i_Kr = g_Kr * (0.9 * paf + 0.1 * pas) * pik * (V - E_K)
        alg_34_i_f = state_y * g_f * (V - -30.0)
        alg_68_i_st = g_st * qa * qi * (V - E_st)
        alg_49_g_K1_prime = g_K1 * (0.5 + 0.5 / (1.0 + torch.exp((V + 30.0) / 5.0)))
        alg_50_i_K1 = alg_49_g_K1_prime * (Kc / (Kc + 0.59)) ** 3.0 * (V + 81.9) / (1.0 + torch.exp(1.393 * (V + 81.9 + 3.6) / RTONF))
        alg_58_k41 = torch.exp(-Qn * V / (2.0 * RTONF))
        alg_53_do = 1.0 + Cao / Kco * (1.0 + torch.exp(Qco * V / RTONF)) + Nao / K1no + Nao ** 2.0 / (K1no * K2no) + Nao ** 3.0 / (K1no * K2no * K3no)
        alg_55_k23 = (Nao ** 2.0 / (K1no * K2no) + Nao ** 3.0 / (K1no * K2no * K3no)) * torch.exp(-Qn * V / (2.0 * RTONF)) / alg_53_do
        alg_56_k21 = Cao / Kco * torch.exp(-Qco * V / RTONF) / alg_53_do
        alg_54_k32 = torch.exp(Qn * V / (2.0 * RTONF))
        alg_61_x1 = alg_58_k41 * k34 * (alg_55_k23 + alg_56_k21) + alg_56_k21 * alg_54_k32 * (k43 + alg_58_k41)
        alg_57_di = 1.0 + Casub / Kci * (1.0 + torch.exp(-Qci * V / RTONF) + Nai / Kcni) + Nai / K1ni + Nai ** 2.0 / (K1ni * K2ni) + Nai ** 3.0 / (K1ni * K2ni * K3ni)
        alg_60_k12 = Casub / Kci * torch.exp(-Qci * V / RTONF) / alg_57_di
        alg_59_k14 = (Nai ** 2.0 / (K1ni * K2ni) + Nai ** 3.0 / (K1ni * K2ni * K3ni)) * torch.exp(Qn * V / (2.0 * RTONF)) / alg_57_di
        alg_62_x2 = alg_54_k32 * k43 * (alg_59_k14 + alg_60_k12) + alg_58_k41 * alg_60_k12 * (k34 + alg_54_k32)
        alg_63_x3 = alg_59_k14 * k43 * (alg_55_k23 + alg_56_k21) + alg_60_k12 * alg_55_k23 * (k43 + alg_58_k41)
        alg_64_x4 = alg_55_k23 * k34 * (alg_59_k14 + alg_60_k12) + alg_59_k14 * alg_56_k21 * (k34 + alg_54_k32)
        alg_65_i_NaCa = kNaCa * (alg_62_x2 * alg_56_k21 - alg_61_x1 * alg_60_k12) / (alg_61_x1 + alg_62_x2 + alg_63_x3 + alg_64_x4)
        alg_52_i_p = I_p * (Nai / (5.64 + Nai)) ** 3.0 * (Kc / (0.621 + Kc)) ** 2.0 * 1.6 / (1.5 + torch.exp(-(V + 60.0) / 40.0))
        alg_51_i_b = g_b * (V - E_b)
        alg_17_i_Stim = torch.where((t >= stim_start) & (t <= stim_end) & (t - stim_start - torch.floor((t - stim_start) / stim_period) * stim_period <= stim_duration), stim_amplitude, 0.0)
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_66_i_Na + alg_71_i_CaL + alg_67_i_to + alg_42_i_Kr + alg_34_i_f + alg_68_i_st + alg_50_i_K1 + alg_65_i_NaCa + alg_52_i_p + alg_51_i_b + alg_70_i_ACh + alg_17_i_Stim) / C)
        alg_73_i_up = 5.0 / (1.0 + K_up / Cai)
        alg_74_i_tr = (Ca_up - Ca_rel) / tau_tr
        d_Ca_up = torch.zeros_like(y[:, 0]) + (alg_73_i_up - alg_74_i_tr * V_rel / V_up)
        alg_75_diff_f_TC = 88800.0 * Cai * (1.0 - f_TC) - 446.0 * f_TC
        d_f_TC = torch.zeros_like(y[:, 0]) + (alg_75_diff_f_TC)
        alg_77_diff_f_TMC = 227700.0 * Cai * (1.0 - f_TMC - f_TMM) - 7.51 * f_TMC
        d_f_TMC = torch.zeros_like(y[:, 0]) + (alg_77_diff_f_TMC)
        alg_76_i_rel = P_rel * (Ca_rel - Casub) / (1.0 + (0.0012 / Casub) ** 2.0)
        alg_80_diff_f_CQ = 534.0 * Ca_rel * (1.0 - f_CQ) - 445.0 * f_CQ
        d_Ca_rel = torch.zeros_like(y[:, 0]) + (alg_74_i_tr - alg_76_i_rel - 10.0 * alg_80_diff_f_CQ)
        alg_72_i_diff = (Casub - Cai) / 4e-05
        alg_78_diff_f_CMi = 227700.0 * Cai * (1.0 - f_CMi) - 542.0 * f_CMi
        d_Cai = torch.zeros_like(y[:, 0]) + ((alg_72_i_diff * V_sub - alg_73_i_up * V_up) / Vi - (0.045 * alg_78_diff_f_CMi + 0.031 * alg_75_diff_f_TC + 0.062 * alg_77_diff_f_TMC))
        d_f_CMi = torch.zeros_like(y[:, 0]) + (alg_78_diff_f_CMi)
        alg_79_diff_f_CMs = 227700.0 * Casub * (1.0 - f_CMs) - 542.0 * f_CMs
        d_f_CMs = torch.zeros_like(y[:, 0]) + (alg_79_diff_f_CMs)
        d_f_CQ = torch.zeros_like(y[:, 0]) + (alg_80_diff_f_CQ)
        alg_81_diff_f_CSL = 0.001 * (115.0 * Casub * (1.0 - f_CSL) - 1000.0 * f_CSL)
        d_Casub = torch.zeros_like(y[:, 0]) + ((-(alg_71_i_CaL - 2.0 * alg_65_i_NaCa) / (2.0 * F) + alg_76_i_rel * V_rel) / V_sub - alg_72_i_diff - 0.045 * alg_79_diff_f_CMs - 0.031 / 1.2 * alg_81_diff_f_CSL)
        d_f_CSL = torch.zeros_like(y[:, 0]) + (alg_81_diff_f_CSL)

        return torch.stack(
            [d_V, d_state_y, d_paf, d_pas, d_pik, d_Casub, d_m, d_h1, d_h2, d_d, d_f, d_f2, d_r, d_q_fast, d_q_slow, d_qa, d_qi, d_achf, d_achs, d_Cai, d_Ca_up, d_Ca_rel, d_f_TC, d_f_TMC, d_f_TMM, d_f_CMi, d_f_CMs, d_f_CQ, d_f_CSL],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
