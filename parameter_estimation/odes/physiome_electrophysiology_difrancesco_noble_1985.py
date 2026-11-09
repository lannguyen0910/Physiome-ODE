import torch
import torch.nn as nn
import math


class ElectrophysiologyDifrancescoNoble1985ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/difrancesco_noble_1985/model.py.
    Dynamic states: 16; independent constants: 45; derived constants used by rates: 5.
    Original constant slots: 50; algebraic slots: 45.
    """
    source_file = 'electrophysiology/difrancesco_noble_1985/model.py'
    source_state_dim = 16
    source_constant_dim = 50
    algebraic_dim = 45
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 16
        self.parameter_dim = 45

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8314.472, 'T': 310, 'F': 96485.3415, 'C': 0.075, 'i_pulse': 0, 'g_f_Na': 3, 'g_f_K': 3, 'Km_f': 45, 'Nao': 140, 'delta_y': 1e-05, 'i_K_max': 180, 'g_K1': 920, 'Km_K1': 210, 'Km_to': 10, 'Km_Ca': 0.0005, 'g_to': 0.28, 'g_Nab': 0.18, 'g_Cab': 0.02, 'Cao': 2, 'I_p': 125, 'K_mK': 1, 'K_mNa': 40, 'n_NaCa': 3, 'K_NaCa': 0.02, 'd_NaCa': 0.001, 'gamma': 0.5, 'g_Na': 750, 'delta_m': 1e-05, 'P_si': 15, 'delta_d': 0.0001, 'delta_f': 0.0001, 'alpha_f2': 5, 'K_mf2': 0.001, 'radius': 0.05, 'length': 2, 'V_e_ratio': 0.1, 'Ca_up_max': 5, 'K_mCa': 0.001, 'tau_up': 0.025, 'tau_rep': 2, 'tau_rel': 0.05, 'rCa': 2, 'Ve': 0.00157, 'Kb': 4, 'pf': 0.7}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -87, 'Kc': 4, 'Ki': 140, 'Nai': 8, 'y': 0.2, 'x': 0.01, 'Cai': 5e-05, 's': 1, 'm': 0.01, 'h': 0.8, 'd': 0.005, 'f': 1, 'f2': 1, 'Ca_up': 2, 'Ca_rel': 1, 'p': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'C', 'i_pulse', 'g_f_Na', 'g_f_K', 'Km_f', 'Nao', 'delta_y', 'i_K_max', 'g_K1', 'Km_K1', 'Km_to', 'Km_Ca', 'g_to', 'g_Nab', 'g_Cab', 'Cao', 'I_p', 'K_mK', 'K_mNa', 'n_NaCa', 'K_NaCa', 'd_NaCa', 'gamma', 'g_Na', 'delta_m', 'P_si', 'delta_d', 'delta_f', 'alpha_f2', 'K_mf2', 'radius', 'length', 'V_e_ratio', 'Ca_up_max', 'K_mCa', 'tau_up', 'tau_rep', 'tau_rel', 'rCa', 'Ve', 'Kb', 'pf']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'Kc', 'Ki', 'Nai', 'y', 'x', 'Cai', 's', 'm', 'h', 'd', 'f', 'f2', 'Ca_up', 'Ca_rel', 'p']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (16)
        V = y[:, 0]
        Kc = y[:, 1]
        Ki = y[:, 2]
        Nai = y[:, 3]
        state_y = y[:, 4]
        x = y[:, 5]
        Cai = y[:, 6]
        s = y[:, 7]
        m = y[:, 8]
        h = y[:, 9]
        d = y[:, 10]
        f = y[:, 11]
        f2 = y[:, 12]
        Ca_up = y[:, 13]
        Ca_rel = y[:, 14]
        p = y[:, 15]

        # Independent parameters (source constant order)
        R = theta[:, 0]  # constants[0]
        T = theta[:, 1]  # constants[1]
        F = theta[:, 2]  # constants[2]
        C = theta[:, 3]  # constants[3]
        i_pulse = theta[:, 4]  # constants[4]
        g_f_Na = theta[:, 5]  # constants[5]
        g_f_K = theta[:, 6]  # constants[6]
        Km_f = theta[:, 7]  # constants[7]
        Nao = theta[:, 8]  # constants[8]
        delta_y = theta[:, 9]  # constants[9]
        i_K_max = theta[:, 10]  # constants[10]
        g_K1 = theta[:, 11]  # constants[11]
        Km_K1 = theta[:, 12]  # constants[12]
        Km_to = theta[:, 13]  # constants[13]
        Km_Ca = theta[:, 14]  # constants[14]
        g_to = theta[:, 15]  # constants[15]
        g_Nab = theta[:, 16]  # constants[16]
        g_Cab = theta[:, 17]  # constants[17]
        Cao = theta[:, 18]  # constants[18]
        I_p = theta[:, 19]  # constants[19]
        K_mK = theta[:, 20]  # constants[20]
        K_mNa = theta[:, 21]  # constants[21]
        n_NaCa = theta[:, 22]  # constants[22]
        K_NaCa = theta[:, 23]  # constants[23]
        d_NaCa = theta[:, 24]  # constants[24]
        gamma = theta[:, 25]  # constants[25]
        g_Na = theta[:, 26]  # constants[26]
        delta_m = theta[:, 27]  # constants[27]
        P_si = theta[:, 28]  # constants[28]
        delta_d = theta[:, 29]  # constants[29]
        delta_f = theta[:, 30]  # constants[30]
        alpha_f2 = theta[:, 31]  # constants[31]
        K_mf2 = theta[:, 32]  # constants[32]
        radius = theta[:, 33]  # constants[33]
        length = theta[:, 34]  # constants[34]
        V_e_ratio = theta[:, 35]  # constants[35]
        Ca_up_max = theta[:, 36]  # constants[36]
        K_mCa = theta[:, 37]  # constants[37]
        tau_up = theta[:, 38]  # constants[38]
        tau_rep = theta[:, 39]  # constants[39]
        tau_rel = theta[:, 40]  # constants[40]
        rCa = theta[:, 41]  # constants[41]
        Ve = theta[:, 42]  # constants[42]
        Kb = theta[:, 43]  # constants[43]
        pf = theta[:, 44]  # constants[44]

        # Derived constants from the original initConsts()
        RTONF = R * T / F  # constants[45]
        V_Cell = 3.14159 * radius ** 2.0 * length  # constants[46]
        Vi = V_Cell * (1.0 - V_e_ratio)  # constants[47]
        V_up = Vi * 0.05  # constants[48]
        V_rel = Vi * 0.02  # constants[49]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_Kc = torch.zeros_like(y[:, 0])
        d_Ki = torch.zeros_like(y[:, 0])
        d_Nai = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])
        d_x = torch.zeros_like(y[:, 0])
        d_Cai = torch.zeros_like(y[:, 0])
        d_s = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_d = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_f2 = torch.zeros_like(y[:, 0])
        d_Ca_up = torch.zeros_like(y[:, 0])
        d_Ca_rel = torch.zeros_like(y[:, 0])
        d_p = torch.zeros_like(y[:, 0])
        alg_0_E_Na = torch.zeros_like(y[:, 0])
        alg_1_alpha_y = torch.zeros_like(y[:, 0])
        alg_2_alpha_x = torch.zeros_like(y[:, 0])
        alg_3_alpha_s = torch.zeros_like(y[:, 0])
        alg_4_E0_m = torch.zeros_like(y[:, 0])
        alg_5_alpha_h = torch.zeros_like(y[:, 0])
        alg_6_E0_d = torch.zeros_like(y[:, 0])
        alg_7_E0_f = torch.zeros_like(y[:, 0])
        alg_8_beta_f2 = torch.zeros_like(y[:, 0])
        alg_9_alpha_p = torch.zeros_like(y[:, 0])
        alg_10_E_K = torch.zeros_like(y[:, 0])
        alg_11_E0_y = torch.zeros_like(y[:, 0])
        alg_12_beta_x = torch.zeros_like(y[:, 0])
        alg_13_beta_s = torch.zeros_like(y[:, 0])
        alg_14_alpha_m = torch.zeros_like(y[:, 0])
        alg_15_beta_h = torch.zeros_like(y[:, 0])
        alg_16_alpha_d = torch.zeros_like(y[:, 0])
        alg_17_alpha_f = torch.zeros_like(y[:, 0])
        alg_18_beta_p = torch.zeros_like(y[:, 0])
        alg_19_beta_y = torch.zeros_like(y[:, 0])
        alg_20_beta_m = torch.zeros_like(y[:, 0])
        alg_21_beta_d = torch.zeros_like(y[:, 0])
        alg_22_beta_f = torch.zeros_like(y[:, 0])
        alg_23_i_fNa = torch.zeros_like(y[:, 0])
        alg_24_i_fK = torch.zeros_like(y[:, 0])
        alg_25_i_f = torch.zeros_like(y[:, 0])
        alg_26_I_K = torch.zeros_like(y[:, 0])
        alg_27_i_K = torch.zeros_like(y[:, 0])
        alg_28_i_K1 = torch.zeros_like(y[:, 0])
        alg_29_i_to = torch.zeros_like(y[:, 0])
        alg_30_i_Na_b = torch.zeros_like(y[:, 0])
        alg_31_E_Ca = torch.zeros_like(y[:, 0])
        alg_32_i_Ca_b = torch.zeros_like(y[:, 0])
        alg_33_i_p = torch.zeros_like(y[:, 0])
        alg_34_i_NaCa = torch.zeros_like(y[:, 0])
        alg_35_E_mh = torch.zeros_like(y[:, 0])
        alg_36_i_Na = torch.zeros_like(y[:, 0])
        alg_37_i_siCa = torch.zeros_like(y[:, 0])
        alg_38_i_siK = torch.zeros_like(y[:, 0])
        alg_39_i_up = torch.zeros_like(y[:, 0])
        alg_40_i_siNa = torch.zeros_like(y[:, 0])
        alg_41_i_tr = torch.zeros_like(y[:, 0])
        alg_42_i_mK = torch.zeros_like(y[:, 0])
        alg_43_i_si = torch.zeros_like(y[:, 0])
        alg_44_i_rel = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_8_beta_f2 = Cai * alpha_f2 / K_mf2
        d_f2 = torch.zeros_like(y[:, 0]) + (alpha_f2 - f2 * (alpha_f2 + alg_8_beta_f2))
        alg_2_alpha_x = 0.5 * torch.exp(0.0826 * (V + 50.0)) / (1.0 + torch.exp(0.057 * (V + 50.0)))
        alg_12_beta_x = 1.3 * torch.exp(-0.06 * (V + 20.0)) / (1.0 + torch.exp(-0.04 * (V + 20.0)))
        d_x = torch.zeros_like(y[:, 0]) + (alg_2_alpha_x * (1.0 - x) - alg_12_beta_x * x)
        alg_3_alpha_s = 0.033 * torch.exp(-V / 17.0)
        alg_13_beta_s = 33.0 / (1.0 + torch.exp(-(V + 10.0) / 8.0))
        d_s = torch.zeros_like(y[:, 0]) + (alg_3_alpha_s * (1.0 - s) - alg_13_beta_s * s)
        alg_5_alpha_h = 20.0 * torch.exp(-0.125 * (V + 75.0))
        alg_15_beta_h = 2000.0 / (320.0 * torch.exp(-0.1 * (V + 75.0)) + 1.0)
        d_h = torch.zeros_like(y[:, 0]) + (alg_5_alpha_h * (1.0 - h) - alg_15_beta_h * h)
        alg_9_alpha_p = 0.625 * (V + 34.0) / (torch.exp((V + 34.0) / 4.0) - 1.0)
        alg_18_beta_p = 5.0 / (1.0 + torch.exp(-1.0 * (V + 34.0) / 4.0))
        d_p = torch.zeros_like(y[:, 0]) + (alg_9_alpha_p * (1.0 - p) - alg_18_beta_p * p)
        alg_1_alpha_y = 0.05 * torch.exp(-0.067 * (V + 52.0 - 10.0))
        alg_11_E0_y = V + 52.0 - 10.0
        alg_19_beta_y = torch.where(torch.abs(alg_11_E0_y) < delta_y, 2.5, 1.0 * alg_11_E0_y / (1.0 - torch.exp(-0.2 * alg_11_E0_y)))
        d_state_y = torch.zeros_like(y[:, 0]) + (alg_1_alpha_y * (1.0 - state_y) - alg_19_beta_y * state_y)
        alg_4_E0_m = V + 41.0
        alg_14_alpha_m = torch.where(torch.abs(alg_4_E0_m) < delta_m, 2000.0, 200.0 * alg_4_E0_m / (1.0 - torch.exp(-0.1 * alg_4_E0_m)))
        alg_20_beta_m = 8000.0 * torch.exp(-0.056 * (V + 66.0))
        d_m = torch.zeros_like(y[:, 0]) + (alg_14_alpha_m * (1.0 - m) - alg_20_beta_m * m)
        alg_6_E0_d = V + 24.0 - 5.0
        alg_16_alpha_d = torch.where(torch.abs(alg_6_E0_d) < delta_d, 120.0, 30.0 * alg_6_E0_d / (1.0 - torch.exp(-1.0 * alg_6_E0_d / 4.0)))
        alg_21_beta_d = torch.where(torch.abs(alg_6_E0_d) < delta_d, 120.0, 12.0 * alg_6_E0_d / (torch.exp(alg_6_E0_d / 10.0) - 1.0))
        d_d = torch.zeros_like(y[:, 0]) + (alg_16_alpha_d * (1.0 - d) - alg_21_beta_d * d)
        alg_7_E0_f = V + 34.0
        alg_17_alpha_f = torch.where(torch.abs(alg_7_E0_f) < delta_f, 25.0, 6.25 * alg_7_E0_f / (torch.exp(alg_7_E0_f / 4.0) - 1.0))
        alg_22_beta_f = 50.0 / (1.0 + torch.exp(-1.0 * (V + 34.0) / 4.0))
        d_f = torch.zeros_like(y[:, 0]) + (alg_17_alpha_f * (1.0 - f) - alg_22_beta_f * f)
        alg_0_E_Na = RTONF * torch.log(Nao / Nai)
        alg_30_i_Na_b = g_Nab * (V - alg_0_E_Na)
        alg_33_i_p = I_p * Kc / (K_mK + Kc) * Nai / (K_mNa + Nai)
        alg_34_i_NaCa = K_NaCa * (torch.exp(gamma * (n_NaCa - 2.0) * V / RTONF) * Nai ** n_NaCa * Cao - torch.exp((gamma - 1.0) * (n_NaCa - 2.0) * V / RTONF) * Nao ** n_NaCa * Cai) / ((1.0 + d_NaCa * (Cai * Nao ** n_NaCa + Cao * Nai ** n_NaCa)) * (1.0 + Cai / 0.0069))
        alg_35_E_mh = RTONF * torch.log((Nao + 0.12 * Kc) / (Nai + 0.12 * Ki))
        alg_36_i_Na = g_Na * m ** 3.0 * h * (V - alg_35_E_mh)
        alg_23_i_fNa = state_y * Kc / (Kc + Km_f) * g_f_Na * (V - alg_0_E_Na)
        alg_40_i_siNa = 0.01 * P_si * (V - 50.0) / (RTONF * (1.0 - torch.exp(-1.0 * (V - 50.0) / RTONF))) * (Nai * torch.exp(50.0 / RTONF) - Nao * torch.exp(-1.0 * (V - 50.0) / RTONF)) * d * f * f2
        d_Nai = torch.zeros_like(y[:, 0]) + (-1.0 * (alg_36_i_Na + alg_30_i_Na_b + alg_23_i_fNa + alg_40_i_siNa + alg_33_i_p * 3.0 + alg_34_i_NaCa * n_NaCa / (n_NaCa - 2.0)) / (1.0 * Vi * F))
        alg_39_i_up = 2.0 * 1.0 * Vi * F / (1.0 * tau_up * Ca_up_max) * Cai * (Ca_up_max - Ca_up)
        alg_41_i_tr = 2.0 * 1.0 * V_rel * F / (1.0 * tau_rep) * p * (Ca_up - Ca_rel)
        d_Ca_up = torch.zeros_like(y[:, 0]) + (1.0 * (alg_39_i_up - alg_41_i_tr) / (2.0 * 1.0 * V_up * F))
        alg_26_I_K = i_K_max * (Ki - Kc * torch.exp(-V / RTONF)) / 140.0
        alg_27_i_K = x * alg_26_I_K
        alg_10_E_K = RTONF * torch.log(Kc / Ki)
        alg_28_i_K1 = g_K1 * Kc / (Kc + Km_K1) * (V - alg_10_E_K) / (1.0 + torch.exp((V + 10.0 - alg_10_E_K) * 2.0 / RTONF))
        alg_29_i_to = s * g_to * (0.2 + Kc / (Km_to + Kc)) * Cai / (Km_Ca + Cai) * (V + 10.0) / (1.0 - torch.exp(-0.2 * (V + 10.0))) * (Ki * torch.exp(0.5 * V / RTONF) - Kc * torch.exp(-0.5 * V / RTONF))
        alg_24_i_fK = state_y * Kc / (Kc + Km_f) * g_f_K * (V - alg_10_E_K)
        alg_38_i_siK = 0.01 * P_si * (V - 50.0) / (RTONF * (1.0 - torch.exp(-1.0 * (V - 50.0) / RTONF))) * (Ki * torch.exp(50.0 / RTONF) - Kc * torch.exp(-1.0 * (V - 50.0) / RTONF)) * d * f * f2
        alg_42_i_mK = alg_28_i_K1 + alg_27_i_K + alg_24_i_fK + alg_38_i_siK + alg_29_i_to - 2.0 * alg_33_i_p
        d_Kc = torch.zeros_like(y[:, 0]) + (-pf * (Kc - Kb) + 1.0 * alg_42_i_mK / (1.0 * Ve * F))
        d_Ki = torch.zeros_like(y[:, 0]) + (-1.0 * alg_42_i_mK / (1.0 * Vi * F))
        alg_25_i_f = alg_23_i_fNa + alg_24_i_fK
        alg_31_E_Ca = 0.5 * RTONF * torch.log(Cao / Cai)
        alg_32_i_Ca_b = g_Cab * (V - alg_31_E_Ca)
        alg_37_i_siCa = 4.0 * P_si * (V - 50.0) / (RTONF * (1.0 - torch.exp(-1.0 * (V - 50.0) * 2.0 / RTONF))) * (Cai * torch.exp(100.0 / RTONF) - Cao * torch.exp(-2.0 * (V - 50.0) / RTONF)) * d * f * f2
        alg_43_i_si = alg_37_i_siCa + alg_38_i_siK + alg_40_i_siNa
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_25_i_f + alg_27_i_K + alg_28_i_K1 + alg_29_i_to + alg_30_i_Na_b + alg_32_i_Ca_b + alg_33_i_p + alg_34_i_NaCa + alg_36_i_Na + alg_43_i_si + i_pulse) / C)
        alg_44_i_rel = 2.0 * 1.0 * V_rel * F / (1.0 * tau_rel) * Ca_rel * Cai ** rCa / (Cai ** rCa + K_mCa ** rCa)
        d_Ca_rel = torch.zeros_like(y[:, 0]) + (1.0 * (alg_41_i_tr - alg_44_i_rel) / (2.0 * 1.0 * V_rel * F))
        d_Cai = torch.zeros_like(y[:, 0]) + (-1.0 * (alg_37_i_siCa + alg_32_i_Ca_b - 2.0 * alg_34_i_NaCa / (n_NaCa - 2.0) - alg_44_i_rel + alg_39_i_up) / (2.0 * 1.0 * Vi * F))

        return torch.stack(
            [d_V, d_Kc, d_Ki, d_Nai, d_state_y, d_x, d_Cai, d_s, d_m, d_h, d_d, d_f, d_f2, d_Ca_up, d_Ca_rel, d_p],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
