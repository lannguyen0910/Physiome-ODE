import torch
import torch.nn as nn
import math


class CalciumDynamicsNobleNoble2001ODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/noble_noble_2001/model.py.
    Dynamic states: 20; independent constants: 65; derived constants used by rates: 4.
    Original constant slots: 69; algebraic slots: 60.
    """
    source_file = 'calcium_dynamics/noble_noble_2001/model.py'
    source_state_dim = 20
    source_constant_dim = 69
    algebraic_dim = 60
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 20
        self.parameter_dim = 65

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8314.472, 'T': 310, 'F': 96485.3415, 'Cm': 9.5e-05, 'stim_start': 0.1, 'stim_end': 600, 'stim_period': 1, 'stim_duration': 0.002, 'stim_amplitude': -6, 'P_kna': 0.03, 'K_o': 4, 'Na_o': 140, 'Ca_o': 2, 'K_mk1': 10, 'g_K1': 0.5, 'g_Kr1': 0.0028, 'g_Kr2': 0.0017, 'g_Ks': 0.0016, 'g_Na': 0.5, 'delta_m': 1e-05, 'shift_h': 0, 'g_pna': 0.0027, 'g_bna': 0.0006, 'P_Ca_L': 1, 'P_CaK': 0.002, 'P_CaNa': 0.01, 'KCaChoff': 0.01, 'Kmdsinact': 0.001, 'FrICa': 0.7, 'speed_d': 3, 'speed_f': 0.3, 'delta_f': 0.0001, 'g_bca': 0.00025, 'g_to': 0.005, 'g_tos': 0, 'i_NaK_max': 0.7, 'K_mK': 1, 'K_mNa': 40, 'k_NaCa': 0.0005, 'n_NaCa': 3, 'd_NaCa': 0, 'gamma': 0.2, 'FRiNaCa': 0.001, 'K_cyca': 0.0003, 'K_xcs': 0.4, 'K_srca': 0.5, 'alpha_up': 0.4, 'beta_up': 0.03, 'K_leak_rate': 0.05, 'K_m_rel': 250, 'K_m_Ca_cyt': 0.0005, 'K_m_Ca_ds': 0.01, 'Calmod': 0.02, 'Trop': 0.05, 'alpha_Calmod': 100000, 'beta_Calmod': 50, 'alpha_Trop': 100000, 'beta_Trop': 200, 'radius': 12, 'length': 74, 'V_ds_ratio': 0.1, 'V_rel_ratio': 0.1, 'V_e_ratio': 0.4, 'V_up_ratio': 0.01, 'Kdecay': 10}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -93.0550060340878, 'K_i': 138.49601832864, 'Na_i': 5.68929064089788, 'Ca_i': 5.60867201979414e-06, 'xr1': 9.88715446781661e-06, 'xr2': 2.15138346929839e-07, 'xs': 6.72192033116639e-07, 'm': 0.00157541794600382, 'h': 0.99465397024286, 'Ca_ds': 2.15623822848619e-07, 'd': 2.27643784381792e-08, 'f': 4.03082807230434e-09, 's': 0.999870016703514, 'r': 1.8398417043009e-08, 'Ca_up': 0.187470810108547, 'Ca_rel': 0.185609773872992, 'ActFrac': 0.000183030061767109, 'ProdFrac': 0.0109931117325847, 'Ca_Calmod': 0.000221858224298955, 'Ca_Trop': 0.00013982468565579}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'Cm', 'stim_start', 'stim_end', 'stim_period', 'stim_duration', 'stim_amplitude', 'P_kna', 'K_o', 'Na_o', 'Ca_o', 'K_mk1', 'g_K1', 'g_Kr1', 'g_Kr2', 'g_Ks', 'g_Na', 'delta_m', 'shift_h', 'g_pna', 'g_bna', 'P_Ca_L', 'P_CaK', 'P_CaNa', 'KCaChoff', 'Kmdsinact', 'FrICa', 'speed_d', 'speed_f', 'delta_f', 'g_bca', 'g_to', 'g_tos', 'i_NaK_max', 'K_mK', 'K_mNa', 'k_NaCa', 'n_NaCa', 'd_NaCa', 'gamma', 'FRiNaCa', 'K_cyca', 'K_xcs', 'K_srca', 'alpha_up', 'beta_up', 'K_leak_rate', 'K_m_rel', 'K_m_Ca_cyt', 'K_m_Ca_ds', 'Calmod', 'Trop', 'alpha_Calmod', 'beta_Calmod', 'alpha_Trop', 'beta_Trop', 'radius', 'length', 'V_ds_ratio', 'V_rel_ratio', 'V_e_ratio', 'V_up_ratio', 'Kdecay']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'K_i', 'Na_i', 'Ca_i', 'xr1', 'xr2', 'xs', 'm', 'h', 'Ca_ds', 'd', 'f', 's', 'r', 'Ca_up', 'Ca_rel', 'ActFrac', 'ProdFrac', 'Ca_Calmod', 'Ca_Trop']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (20)
        V = y[:, 0]
        K_i = y[:, 1]
        Na_i = y[:, 2]
        Ca_i = y[:, 3]
        xr1 = y[:, 4]
        xr2 = y[:, 5]
        xs = y[:, 6]
        m = y[:, 7]
        h = y[:, 8]
        Ca_ds = y[:, 9]
        d = y[:, 10]
        f = y[:, 11]
        s = y[:, 12]
        r = y[:, 13]
        Ca_up = y[:, 14]
        Ca_rel = y[:, 15]
        ActFrac = y[:, 16]
        ProdFrac = y[:, 17]
        Ca_Calmod = y[:, 18]
        Ca_Trop = y[:, 19]

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
        P_kna = theta[:, 9]  # constants[9]
        K_o = theta[:, 10]  # constants[10]
        Na_o = theta[:, 11]  # constants[11]
        Ca_o = theta[:, 12]  # constants[12]
        K_mk1 = theta[:, 13]  # constants[13]
        g_K1 = theta[:, 14]  # constants[14]
        g_Kr1 = theta[:, 15]  # constants[15]
        g_Kr2 = theta[:, 16]  # constants[16]
        g_Ks = theta[:, 17]  # constants[17]
        g_Na = theta[:, 18]  # constants[18]
        delta_m = theta[:, 19]  # constants[19]
        shift_h = theta[:, 20]  # constants[20]
        g_pna = theta[:, 21]  # constants[21]
        g_bna = theta[:, 22]  # constants[22]
        P_Ca_L = theta[:, 23]  # constants[23]
        P_CaK = theta[:, 24]  # constants[24]
        P_CaNa = theta[:, 25]  # constants[25]
        KCaChoff = theta[:, 26]  # constants[26]
        Kmdsinact = theta[:, 27]  # constants[27]
        FrICa = theta[:, 28]  # constants[28]
        speed_d = theta[:, 29]  # constants[29]
        speed_f = theta[:, 30]  # constants[30]
        delta_f = theta[:, 31]  # constants[31]
        g_bca = theta[:, 32]  # constants[32]
        g_to = theta[:, 33]  # constants[33]
        g_tos = theta[:, 34]  # constants[34]
        i_NaK_max = theta[:, 35]  # constants[35]
        K_mK = theta[:, 36]  # constants[36]
        K_mNa = theta[:, 37]  # constants[37]
        k_NaCa = theta[:, 38]  # constants[38]
        n_NaCa = theta[:, 39]  # constants[39]
        d_NaCa = theta[:, 40]  # constants[40]
        gamma = theta[:, 41]  # constants[41]
        FRiNaCa = theta[:, 42]  # constants[42]
        K_cyca = theta[:, 43]  # constants[43]
        K_xcs = theta[:, 44]  # constants[44]
        K_srca = theta[:, 45]  # constants[45]
        alpha_up = theta[:, 46]  # constants[46]
        beta_up = theta[:, 47]  # constants[47]
        K_leak_rate = theta[:, 48]  # constants[48]
        K_m_rel = theta[:, 49]  # constants[49]
        K_m_Ca_cyt = theta[:, 50]  # constants[50]
        K_m_Ca_ds = theta[:, 51]  # constants[51]
        Calmod = theta[:, 52]  # constants[52]
        Trop = theta[:, 53]  # constants[53]
        alpha_Calmod = theta[:, 54]  # constants[54]
        beta_Calmod = theta[:, 55]  # constants[55]
        alpha_Trop = theta[:, 56]  # constants[56]
        beta_Trop = theta[:, 57]  # constants[57]
        radius = theta[:, 58]  # constants[58]
        length = theta[:, 59]  # constants[59]
        V_ds_ratio = theta[:, 60]  # constants[60]
        V_rel_ratio = theta[:, 61]  # constants[61]
        V_e_ratio = theta[:, 62]  # constants[62]
        V_up_ratio = theta[:, 63]  # constants[63]
        Kdecay = theta[:, 64]  # constants[64]

        # Derived constants from the original initConsts()
        V_Cell = 3.14159 * (radius / 1000.0) ** 2.0 * length / 1000.0  # constants[65]
        K_1 = K_cyca * K_xcs / K_srca  # constants[66]
        V_i_ratio = 1.0 - V_e_ratio - V_up_ratio - V_rel_ratio  # constants[67]
        V_i = V_Cell * V_i_ratio  # constants[68]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_K_i = torch.zeros_like(y[:, 0])
        d_Na_i = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_xr1 = torch.zeros_like(y[:, 0])
        d_xr2 = torch.zeros_like(y[:, 0])
        d_xs = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_Ca_ds = torch.zeros_like(y[:, 0])
        d_d = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_s = torch.zeros_like(y[:, 0])
        d_r = torch.zeros_like(y[:, 0])
        d_Ca_up = torch.zeros_like(y[:, 0])
        d_Ca_rel = torch.zeros_like(y[:, 0])
        d_ActFrac = torch.zeros_like(y[:, 0])
        d_ProdFrac = torch.zeros_like(y[:, 0])
        d_Ca_Calmod = torch.zeros_like(y[:, 0])
        d_Ca_Trop = torch.zeros_like(y[:, 0])
        alg_0_alpha_xr1 = torch.zeros_like(y[:, 0])
        alg_1_alpha_xr2 = torch.zeros_like(y[:, 0])
        alg_2_alpha_xs = torch.zeros_like(y[:, 0])
        alg_3_E0_m = torch.zeros_like(y[:, 0])
        alg_4_alpha_h = torch.zeros_like(y[:, 0])
        alg_5_E0_d = torch.zeros_like(y[:, 0])
        alg_6_i_Stim = torch.zeros_like(y[:, 0])
        alg_7_alpha_s = torch.zeros_like(y[:, 0])
        alg_8_VoltDep = torch.zeros_like(y[:, 0])
        alg_9_beta_xr1 = torch.zeros_like(y[:, 0])
        alg_10_beta_xr2 = torch.zeros_like(y[:, 0])
        alg_11_beta_xs = torch.zeros_like(y[:, 0])
        alg_12_alpha_m = torch.zeros_like(y[:, 0])
        alg_13_beta_h = torch.zeros_like(y[:, 0])
        alg_14_alpha_d = torch.zeros_like(y[:, 0])
        alg_15_E_Na = torch.zeros_like(y[:, 0])
        alg_16_beta_s = torch.zeros_like(y[:, 0])
        alg_17_CaiReg = torch.zeros_like(y[:, 0])
        alg_18_beta_m = torch.zeros_like(y[:, 0])
        alg_19_beta_d = torch.zeros_like(y[:, 0])
        alg_20_E_K = torch.zeros_like(y[:, 0])
        alg_21_CadsReg = torch.zeros_like(y[:, 0])
        alg_22_E_Ks = torch.zeros_like(y[:, 0])
        alg_23_RegBindSite = torch.zeros_like(y[:, 0])
        alg_24_E_Ca = torch.zeros_like(y[:, 0])
        alg_25_ActRate = torch.zeros_like(y[:, 0])
        alg_26_E_mh = torch.zeros_like(y[:, 0])
        alg_27_InactRate = torch.zeros_like(y[:, 0])
        alg_28_i_K1 = torch.zeros_like(y[:, 0])
        alg_29_SpeedRel = torch.zeros_like(y[:, 0])
        alg_30_i_Kr = torch.zeros_like(y[:, 0])
        alg_31_PrecFrac = torch.zeros_like(y[:, 0])
        alg_32_i_Ks = torch.zeros_like(y[:, 0])
        alg_33_i_Na = torch.zeros_like(y[:, 0])
        alg_34_i_p_Na = torch.zeros_like(y[:, 0])
        alg_35_i_b_Na = torch.zeros_like(y[:, 0])
        alg_36_E0_f = torch.zeros_like(y[:, 0])
        alg_37_alpha_f = torch.zeros_like(y[:, 0])
        alg_38_beta_f = torch.zeros_like(y[:, 0])
        alg_39_CaChoffcyt = torch.zeros_like(y[:, 0])
        alg_40_CaChoncyt = torch.zeros_like(y[:, 0])
        alg_41_i_Ca_L_Ca_cyt = torch.zeros_like(y[:, 0])
        alg_42_i_Ca_L_K_cyt = torch.zeros_like(y[:, 0])
        alg_43_i_Ca_L_Na_cyt = torch.zeros_like(y[:, 0])
        alg_44_CaChoffds = torch.zeros_like(y[:, 0])
        alg_45_CaChonds = torch.zeros_like(y[:, 0])
        alg_46_i_Ca_L_Ca_ds = torch.zeros_like(y[:, 0])
        alg_47_i_Ca_L_K_ds = torch.zeros_like(y[:, 0])
        alg_48_i_Ca_L_Na_ds = torch.zeros_like(y[:, 0])
        alg_50_i_b_Ca = torch.zeros_like(y[:, 0])
        alg_51_i_to = torch.zeros_like(y[:, 0])
        alg_52_i_NaK = torch.zeros_like(y[:, 0])
        alg_53_i_NaCa_cyt = torch.zeros_like(y[:, 0])
        alg_54_i_NaCa_ds = torch.zeros_like(y[:, 0])
        alg_56_K_2 = torch.zeros_like(y[:, 0])
        alg_57_i_up = torch.zeros_like(y[:, 0])
        alg_58_i_trans = torch.zeros_like(y[:, 0])
        alg_59_i_rel = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_r = torch.zeros_like(y[:, 0]) + (333.0 * (1.0 / (1.0 + torch.exp(-(V + 4.0) / 5.0)) - r))
        alg_0_alpha_xr1 = 50.0 / (1.0 + torch.exp(-(V - 5.0) / 9.0))
        alg_9_beta_xr1 = 0.05 * torch.exp(-(V - 20.0) / 15.0)
        d_xr1 = torch.zeros_like(y[:, 0]) + (alg_0_alpha_xr1 * (1.0 - xr1) - alg_9_beta_xr1 * xr1)
        alg_1_alpha_xr2 = 50.0 / (1.0 + torch.exp(-(V - 5.0) / 9.0))
        alg_10_beta_xr2 = 0.4 * torch.exp(-((V + 30.0) / 30.0) ** 3.0)
        d_xr2 = torch.zeros_like(y[:, 0]) + (alg_1_alpha_xr2 * (1.0 - xr2) - alg_10_beta_xr2 * xr2)
        alg_2_alpha_xs = 14.0 / (1.0 + torch.exp(-(V - 40.0) / 9.0))
        alg_11_beta_xs = 1.0 * torch.exp(-V / 45.0)
        d_xs = torch.zeros_like(y[:, 0]) + (alg_2_alpha_xs * (1.0 - xs) - alg_11_beta_xs * xs)
        alg_4_alpha_h = 20.0 * torch.exp(-0.125 * (V + 75.0 - shift_h))
        alg_13_beta_h = 2000.0 / (1.0 + 320.0 * torch.exp(-0.1 * (V + 75.0 - shift_h)))
        d_h = torch.zeros_like(y[:, 0]) + (alg_4_alpha_h * (1.0 - h) - alg_13_beta_h * h)
        alg_7_alpha_s = 0.033 * torch.exp(-V / 17.0)
        alg_16_beta_s = 33.0 / (1.0 + torch.exp(-0.125 * (V + 10.0)))
        d_s = torch.zeros_like(y[:, 0]) + (alg_7_alpha_s * (1.0 - s) - alg_16_beta_s * s)
        alg_3_E0_m = V + 41.0
        alg_12_alpha_m = torch.where(torch.abs(alg_3_E0_m) < delta_m, 2000.0, 200.0 * alg_3_E0_m / (1.0 - torch.exp(-0.1 * alg_3_E0_m)))
        alg_18_beta_m = 8000.0 * torch.exp(-0.056 * (V + 66.0))
        d_m = torch.zeros_like(y[:, 0]) + (alg_12_alpha_m * (1.0 - m) - alg_18_beta_m * m)
        alg_5_E0_d = V + 24.0 - 5.0
        alg_14_alpha_d = torch.where(torch.abs(alg_5_E0_d) < 0.0001, 120.0, 30.0 * alg_5_E0_d / (1.0 - torch.exp(-alg_5_E0_d / 4.0)))
        alg_19_beta_d = torch.where(torch.abs(alg_5_E0_d) < 0.0001, 120.0, 12.0 * alg_5_E0_d / (torch.exp(alg_5_E0_d / 10.0) - 1.0))
        d_d = torch.zeros_like(y[:, 0]) + (speed_d * (alg_14_alpha_d * (1.0 - d) - alg_19_beta_d * d))
        alg_17_CaiReg = Ca_i / (Ca_i + K_m_Ca_cyt)
        alg_21_CadsReg = Ca_ds / (Ca_ds + K_m_Ca_ds)
        alg_23_RegBindSite = alg_17_CaiReg + (1.0 - alg_17_CaiReg) * alg_21_CadsReg
        alg_27_InactRate = 60.0 + 500.0 * alg_23_RegBindSite ** 2.0
        alg_29_SpeedRel = torch.where(V < -50.0, 5.0, 1.0)
        d_ProdFrac = torch.zeros_like(y[:, 0]) + (ActFrac * alg_29_SpeedRel * alg_27_InactRate - alg_29_SpeedRel * 1.0 * ProdFrac)
        alg_8_VoltDep = torch.exp(0.08 * (V - 40.0))
        alg_25_ActRate = 0.0 * alg_8_VoltDep + 90.0 * alg_23_RegBindSite ** 2.0
        alg_31_PrecFrac = 1.0 - ActFrac - ProdFrac
        d_ActFrac = torch.zeros_like(y[:, 0]) + (alg_31_PrecFrac * alg_29_SpeedRel * alg_25_ActRate - ActFrac * alg_29_SpeedRel * alg_27_InactRate)
        alg_39_CaChoffcyt = Ca_i / (KCaChoff + Ca_i)
        alg_36_E0_f = V + 34.0
        alg_37_alpha_f = torch.where(torch.abs(alg_36_E0_f) < delta_f, 25.0, 6.25 * alg_36_E0_f / (torch.exp(alg_36_E0_f / 4.0) - 1.0))
        alg_38_beta_f = 12.0 / (1.0 + torch.exp(-1.0 * (V + 34.0) / 4.0))
        d_f = torch.zeros_like(y[:, 0]) + ((120.0 * (1.0 - f) * alg_39_CaChoffcyt + (1.0 - f) * (1.0 - alg_39_CaChoffcyt)) * speed_f * alg_38_beta_f - alg_37_alpha_f * f)
        alg_44_CaChoffds = Ca_ds / (Kmdsinact + Ca_ds)
        alg_45_CaChonds = (1.0 - f) * (1.0 - alg_44_CaChoffds)
        alg_46_i_Ca_L_Ca_ds = FrICa * 4.0 * P_Ca_L * d * alg_45_CaChonds * (V - 50.0) * F / (R * T) / (1.0 - torch.exp(-(V - 50.0) * F * 2.0 / (R * T))) * (Ca_i * torch.exp(100.0 * F / (R * T)) - Ca_o * torch.exp(-(V - 50.0) * F * 2.0 / (R * T)))
        d_Ca_ds = torch.zeros_like(y[:, 0]) + (-1.0 * alg_46_i_Ca_L_Ca_ds / (2.0 * 1.0 * V_ds_ratio * V_i * F) - Ca_ds * Kdecay)
        alg_20_E_K = R * T / F * torch.log(K_o / K_i)
        alg_28_i_K1 = g_K1 * K_o / (K_o + K_mk1) * (V - alg_20_E_K) / (1.0 + torch.exp((V - alg_20_E_K - 10.0) * F * 1.25 / (R * T)))
        alg_51_i_to = g_to * (g_tos + s * (1.0 - g_tos)) * r * (V - alg_20_E_K)
        alg_30_i_Kr = (g_Kr1 * xr1 + g_Kr2 * xr2) * 1.0 / (1.0 + torch.exp((V + 9.0) / 22.4)) * (V - alg_20_E_K)
        alg_22_E_Ks = R * T / F * torch.log((K_o + P_kna * Na_o) / (K_i + P_kna * Na_i))
        alg_32_i_Ks = g_Ks * xs ** 2.0 * (V - alg_22_E_Ks)
        alg_40_CaChoncyt = (1.0 - f) * (1.0 - alg_39_CaChoffcyt)
        alg_42_i_Ca_L_K_cyt = (1.0 - FrICa) * P_CaK * P_Ca_L * d * alg_40_CaChoncyt * (V - 50.0) * F / (R * T) / (1.0 - torch.exp(-(V - 50.0) * F / (R * T))) * (K_i * torch.exp(50.0 * F / (R * T)) - K_o * torch.exp(-(V - 50.0) * F / (R * T)))
        alg_47_i_Ca_L_K_ds = FrICa * P_CaK * P_Ca_L * d * alg_45_CaChonds * (V - 50.0) * F / (R * T) / (1.0 - torch.exp(-(V - 50.0) * F / (R * T))) * (K_i * torch.exp(50.0 * F / (R * T)) - K_o * torch.exp(-(V - 50.0) * F / (R * T)))
        alg_52_i_NaK = i_NaK_max * K_o / (K_mK + K_o) * Na_i / (K_mNa + Na_i)
        d_K_i = torch.zeros_like(y[:, 0]) + (-1.0 / (1.0 * V_i * F) * (alg_28_i_K1 + alg_30_i_Kr + alg_32_i_Ks + alg_42_i_Ca_L_K_cyt + alg_47_i_Ca_L_K_ds + alg_51_i_to - 2.0 * alg_52_i_NaK))
        alg_26_E_mh = R * T / F * torch.log((Na_o + 0.12 * K_o) / (Na_i + 0.12 * K_i))
        alg_33_i_Na = g_Na * m ** 3.0 * h * (V - alg_26_E_mh)
        alg_15_E_Na = R * T / F * torch.log(Na_o / Na_i)
        alg_35_i_b_Na = g_bna * (V - alg_15_E_Na)
        alg_34_i_p_Na = g_pna * 1.0 / (1.0 + torch.exp(-(V + 52.0) / 8.0)) * (V - alg_15_E_Na)
        alg_43_i_Ca_L_Na_cyt = (1.0 - FrICa) * P_CaNa * P_Ca_L * d * alg_40_CaChoncyt * (V - 50.0) * F / (R * T) / (1.0 - torch.exp(-(V - 50.0) * F / (R * T))) * (Na_i * torch.exp(50.0 * F / (R * T)) - Na_o * torch.exp(-(V - 50.0) * F / (R * T)))
        alg_48_i_Ca_L_Na_ds = FrICa * P_CaNa * P_Ca_L * d * alg_45_CaChonds * (V - 50.0) * F / (R * T) / (1.0 - torch.exp(-(V - 50.0) * F / (R * T))) * (Na_i * torch.exp(50.0 * F / (R * T)) - Na_o * torch.exp(-(V - 50.0) * F / (R * T)))
        alg_53_i_NaCa_cyt = (1.0 - FRiNaCa) * k_NaCa * (torch.exp(gamma * (n_NaCa - 2.0) * V * F / (R * T)) * Na_i ** n_NaCa * Ca_o - torch.exp((gamma - 1.0) * (n_NaCa - 2.0) * V * F / (R * T)) * Na_o ** n_NaCa * Ca_i) / ((1.0 + d_NaCa * (Ca_i * Na_o ** n_NaCa + Ca_o * Na_i ** n_NaCa)) * (1.0 + Ca_i / 0.0069))
        d_Na_i = torch.zeros_like(y[:, 0]) + (-1.0 / (1.0 * V_i * F) * (alg_33_i_Na + alg_34_i_p_Na + alg_35_i_b_Na + 3.0 * alg_52_i_NaK + 3.0 * alg_53_i_NaCa_cyt + alg_43_i_Ca_L_Na_cyt + alg_48_i_Ca_L_Na_ds))
        alg_54_i_NaCa_ds = FRiNaCa * k_NaCa * (torch.exp(gamma * (n_NaCa - 2.0) * V * F / (R * T)) * Na_i ** n_NaCa * Ca_o - torch.exp((gamma - 1.0) * (n_NaCa - 2.0) * V * F / (R * T)) * Na_o ** n_NaCa * Ca_ds) / ((1.0 + d_NaCa * (Ca_ds * Na_o ** n_NaCa + Ca_o * Na_i ** n_NaCa)) * (1.0 + Ca_ds / 0.0069))
        alg_41_i_Ca_L_Ca_cyt = (1.0 - FrICa) * 4.0 * P_Ca_L * d * alg_40_CaChoncyt * (V - 50.0) * F / (R * T) / (1.0 - torch.exp(-(V - 50.0) * F * 2.0 / (R * T))) * (Ca_i * torch.exp(100.0 * F / (R * T)) - Ca_o * torch.exp(-(V - 50.0) * F * 2.0 / (R * T)))
        alg_24_E_Ca = 0.5 * R * T / F * torch.log(Ca_o / Ca_i)
        alg_50_i_b_Ca = g_bca * (V - alg_24_E_Ca)
        alg_6_i_Stim = torch.where((t >= stim_start) & (t <= stim_end) & (t - stim_start - torch.floor((t - stim_start) / stim_period) * stim_period <= stim_duration), stim_amplitude, 0.0)
        d_V = torch.zeros_like(y[:, 0]) + (-1.0 / Cm * (alg_6_i_Stim + alg_28_i_K1 + alg_51_i_to + alg_30_i_Kr + alg_32_i_Ks + alg_52_i_NaK + alg_33_i_Na + alg_35_i_b_Na + alg_34_i_p_Na + alg_43_i_Ca_L_Na_cyt + alg_48_i_Ca_L_Na_ds + alg_53_i_NaCa_cyt + alg_54_i_NaCa_ds + alg_41_i_Ca_L_Ca_cyt + alg_46_i_Ca_L_Ca_ds + alg_42_i_Ca_L_K_cyt + alg_47_i_Ca_L_K_ds + alg_50_i_b_Ca))
        alg_56_K_2 = Ca_i + Ca_up * K_1 + K_cyca * K_xcs + K_cyca
        alg_57_i_up = Ca_i / alg_56_K_2 * alpha_up - Ca_up * K_1 / alg_56_K_2 * beta_up
        alg_58_i_trans = 50.0 * (Ca_up - Ca_rel)
        d_Ca_up = torch.zeros_like(y[:, 0]) + (V_i_ratio / V_up_ratio * alg_57_i_up - alg_58_i_trans)
        d_Ca_Calmod = torch.zeros_like(y[:, 0]) + (alpha_Calmod * Ca_i * (Calmod - Ca_Calmod) - beta_Calmod * Ca_Calmod)
        alg_59_i_rel = ((ActFrac / (ActFrac + 0.25)) ** 2.0 * K_m_rel + K_leak_rate) * Ca_rel
        d_Ca_rel = torch.zeros_like(y[:, 0]) + (V_up_ratio / V_rel_ratio * alg_58_i_trans - alg_59_i_rel)
        d_Ca_Trop = torch.zeros_like(y[:, 0]) + (alpha_Trop * Ca_i * (Trop - Ca_Trop) - beta_Trop * Ca_Trop)
        d_Ca_i = torch.zeros_like(y[:, 0]) + (-1.0 / (2.0 * 1.0 * V_i * F) * (alg_41_i_Ca_L_Ca_cyt + alg_50_i_b_Ca - 2.0 * alg_53_i_NaCa_cyt - 2.0 * alg_54_i_NaCa_ds) + Ca_ds * V_ds_ratio * Kdecay + alg_59_i_rel * V_rel_ratio / V_i_ratio - d_Ca_Calmod - d_Ca_Trop - alg_57_i_up)

        return torch.stack(
            [d_V, d_K_i, d_Na_i, d_Ca_i, d_xr1, d_xr2, d_xs, d_m, d_h, d_Ca_ds, d_d, d_f, d_s, d_r, d_Ca_up, d_Ca_rel, d_ActFrac, d_ProdFrac, d_Ca_Calmod, d_Ca_Trop],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
