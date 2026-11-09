import torch
import torch.nn as nn
import math


class ElectrophysiologyDokosCellerLovell1996ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/dokos_celler_lovell_1996/model.py.
    Dynamic states: 18; independent constants: 43; derived constants used by rates: 3.
    Original constant slots: 46; algebraic slots: 49.
    """
    source_file = 'electrophysiology/dokos_celler_lovell_1996/model.py'
    source_state_dim = 18
    source_constant_dim = 46
    algebraic_dim = 49
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 18
        self.parameter_dim = 43

        self.parameters = (
            parameters
            if parameters is not None
            else {'R': 8314.472, 'T': 310, 'F': 96485.3415, 'C': 3.2e-05, 'g_CaL': 0.4, 'tau_dL': 0.002, 'alpha_fL2': 3, 'beta_fL2': 40000, 'g_CaT': 0.085, 'g_Na': 0.25, 'Kk': 0.00026, 'P_KNa': 0.035, 'Kmf': 10.3, 'g_fNa': 0.0081, 'g_fK': 0.0135, 'KmNa': 40, 'KmK': 1, 'i_pmax': 0.226, 'kNaCa': 4, 'Qci': 0.1369, 'Qn': 0.4315, 'Qco': 0, 'K3ni': 26.44, 'Kci': 0.0207, 'K1ni': 395.3, 'K2ni': 2.289, 'Kcni': 26.44, 'K3no': 4.663, 'K1no': 1628, 'K2no': 561.4, 'Kco': 3.663, 'g_Nab': 0.00024, 'KbK': 7e-05, 'V_i': 2.5e-06, 'i_up_max': 0.0212, 'KmCaup': 0.0005, 'KmCarel': 0.001, 'tau_rel': 0.005, 'tau_tr': 0.4, 'tau_b': 0.1, 'Nab': 140, 'Cab': 2, 'Kb': 5.4}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'E': -64.9, 'Cai': 3.4e-05, 'Cao': 2.0004, 'Nai': 7.4994, 'Nao': 139.9929, 'Ki': 140.0073, 'Ko': 5.4243, 'dL': 0.0001, 'fL': 0.1505, 'fL2': 0.219, 'dT': 0.001, 'fT': 0.1328, 'm': 0.0139, 'h': 0.0087, 'x': 0.5682, 'y': 0.0287, 'Caup': 0.5832, 'Carel': 0.1101}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R', 'T', 'F', 'C', 'g_CaL', 'tau_dL', 'alpha_fL2', 'beta_fL2', 'g_CaT', 'g_Na', 'Kk', 'P_KNa', 'Kmf', 'g_fNa', 'g_fK', 'KmNa', 'KmK', 'i_pmax', 'kNaCa', 'Qci', 'Qn', 'Qco', 'K3ni', 'Kci', 'K1ni', 'K2ni', 'Kcni', 'K3no', 'K1no', 'K2no', 'Kco', 'g_Nab', 'KbK', 'V_i', 'i_up_max', 'KmCaup', 'KmCarel', 'tau_rel', 'tau_tr', 'tau_b', 'Nab', 'Cab', 'Kb']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['E', 'Cai', 'Cao', 'Nai', 'Nao', 'Ki', 'Ko', 'dL', 'fL', 'fL2', 'dT', 'fT', 'm', 'h', 'x', 'y', 'Caup', 'Carel']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (18)
        E = y[:, 0]
        Cai = y[:, 1]
        Cao = y[:, 2]
        Nai = y[:, 3]
        Nao = y[:, 4]
        Ki = y[:, 5]
        Ko = y[:, 6]
        dL = y[:, 7]
        fL = y[:, 8]
        fL2 = y[:, 9]
        dT = y[:, 10]
        fT = y[:, 11]
        m = y[:, 12]
        h = y[:, 13]
        x = y[:, 14]
        state_y = y[:, 15]
        Caup = y[:, 16]
        Carel = y[:, 17]

        # Independent parameters (source constant order)
        R = theta[:, 0]  # constants[0]
        T = theta[:, 1]  # constants[1]
        F = theta[:, 2]  # constants[2]
        C = theta[:, 3]  # constants[3]
        g_CaL = theta[:, 4]  # constants[4]
        tau_dL = theta[:, 5]  # constants[5]
        alpha_fL2 = theta[:, 6]  # constants[6]
        beta_fL2 = theta[:, 7]  # constants[7]
        g_CaT = theta[:, 8]  # constants[8]
        g_Na = theta[:, 9]  # constants[9]
        Kk = theta[:, 10]  # constants[10]
        P_KNa = theta[:, 11]  # constants[11]
        Kmf = theta[:, 12]  # constants[12]
        g_fNa = theta[:, 13]  # constants[13]
        g_fK = theta[:, 14]  # constants[14]
        KmNa = theta[:, 15]  # constants[15]
        KmK = theta[:, 16]  # constants[16]
        i_pmax = theta[:, 17]  # constants[17]
        kNaCa = theta[:, 18]  # constants[18]
        Qci = theta[:, 19]  # constants[19]
        Qn = theta[:, 20]  # constants[20]
        Qco = theta[:, 21]  # constants[21]
        K3ni = theta[:, 22]  # constants[22]
        Kci = theta[:, 23]  # constants[23]
        K1ni = theta[:, 24]  # constants[24]
        K2ni = theta[:, 25]  # constants[25]
        Kcni = theta[:, 26]  # constants[26]
        K3no = theta[:, 27]  # constants[27]
        K1no = theta[:, 28]  # constants[28]
        K2no = theta[:, 29]  # constants[29]
        Kco = theta[:, 30]  # constants[30]
        g_Nab = theta[:, 31]  # constants[31]
        KbK = theta[:, 32]  # constants[32]
        V_i = theta[:, 33]  # constants[33]
        i_up_max = theta[:, 34]  # constants[34]
        KmCaup = theta[:, 35]  # constants[35]
        KmCarel = theta[:, 36]  # constants[36]
        tau_rel = theta[:, 37]  # constants[37]
        tau_tr = theta[:, 38]  # constants[38]
        tau_b = theta[:, 39]  # constants[39]
        Nab = theta[:, 40]  # constants[40]
        Cab = theta[:, 41]  # constants[41]
        Kb = theta[:, 42]  # constants[42]

        # Derived constants from the original initConsts()
        V_rel = 0.006 * V_i  # constants[43]
        V_e = 0.2 * V_i  # constants[44]
        V_up = 0.014 * V_i  # constants[45]

        # Initialize rates and algebraic variables, as in computeRates()
        d_E = torch.zeros_like(y[:, 0])
        d_Cai = torch.zeros_like(y[:, 0])
        d_Cao = torch.zeros_like(y[:, 0])
        d_Nai = torch.zeros_like(y[:, 0])
        d_Nao = torch.zeros_like(y[:, 0])
        d_Ki = torch.zeros_like(y[:, 0])
        d_Ko = torch.zeros_like(y[:, 0])
        d_dL = torch.zeros_like(y[:, 0])
        d_fL = torch.zeros_like(y[:, 0])
        d_fL2 = torch.zeros_like(y[:, 0])
        d_dT = torch.zeros_like(y[:, 0])
        d_fT = torch.zeros_like(y[:, 0])
        d_m = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_x = torch.zeros_like(y[:, 0])
        d_state_y = torch.zeros_like(y[:, 0])
        d_Caup = torch.zeros_like(y[:, 0])
        d_Carel = torch.zeros_like(y[:, 0])
        alg_0_dL_infinity = torch.zeros_like(y[:, 0])
        alg_1_fL_infinity = torch.zeros_like(y[:, 0])
        alg_2_dT_infinity = torch.zeros_like(y[:, 0])
        alg_3_fT_infinity = torch.zeros_like(y[:, 0])
        alg_4_alpha_m = torch.zeros_like(y[:, 0])
        alg_5_alpha_h = torch.zeros_like(y[:, 0])
        alg_6_x_infinity = torch.zeros_like(y[:, 0])
        alg_7_alpha_y = torch.zeros_like(y[:, 0])
        alg_8_E_Ca = torch.zeros_like(y[:, 0])
        alg_9_tau_fL = torch.zeros_like(y[:, 0])
        alg_10_tau_dT = torch.zeros_like(y[:, 0])
        alg_11_tau_fT = torch.zeros_like(y[:, 0])
        alg_12_beta_m = torch.zeros_like(y[:, 0])
        alg_13_beta_h = torch.zeros_like(y[:, 0])
        alg_14_tau_x = torch.zeros_like(y[:, 0])
        alg_15_beta_y = torch.zeros_like(y[:, 0])
        alg_16_E_Na = torch.zeros_like(y[:, 0])
        alg_17_E_K = torch.zeros_like(y[:, 0])
        alg_18_i_CaL = torch.zeros_like(y[:, 0])
        alg_19_i_CaT = torch.zeros_like(y[:, 0])
        alg_20_i_Na = torch.zeros_like(y[:, 0])
        alg_21_i_KK = torch.zeros_like(y[:, 0])
        alg_22_i_KNa = torch.zeros_like(y[:, 0])
        alg_23_i_K = torch.zeros_like(y[:, 0])
        alg_24_i_fNa = torch.zeros_like(y[:, 0])
        alg_25_i_fK = torch.zeros_like(y[:, 0])
        alg_26_i_f = torch.zeros_like(y[:, 0])
        alg_27_i_p = torch.zeros_like(y[:, 0])
        alg_28_do = torch.zeros_like(y[:, 0])
        alg_29_k32 = torch.zeros_like(y[:, 0])
        alg_30_k23 = torch.zeros_like(y[:, 0])
        alg_31_k21 = torch.zeros_like(y[:, 0])
        alg_32_k34 = torch.zeros_like(y[:, 0])
        alg_33_di = torch.zeros_like(y[:, 0])
        alg_34_k41 = torch.zeros_like(y[:, 0])
        alg_35_k14 = torch.zeros_like(y[:, 0])
        alg_36_k12 = torch.zeros_like(y[:, 0])
        alg_37_k43 = torch.zeros_like(y[:, 0])
        alg_38_x1 = torch.zeros_like(y[:, 0])
        alg_39_x2 = torch.zeros_like(y[:, 0])
        alg_40_x3 = torch.zeros_like(y[:, 0])
        alg_41_x4 = torch.zeros_like(y[:, 0])
        alg_42_i_NaCa = torch.zeros_like(y[:, 0])
        alg_43_i_bNa = torch.zeros_like(y[:, 0])
        alg_44_i_up = torch.zeros_like(y[:, 0])
        alg_45_i_bK = torch.zeros_like(y[:, 0])
        alg_46_i_tr = torch.zeros_like(y[:, 0])
        alg_47_i_tot = torch.zeros_like(y[:, 0])
        alg_48_i_rel = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_fL2 = torch.zeros_like(y[:, 0]) + (alpha_fL2 * (1.0 - fL2) - beta_fL2 * Cai * fL2)
        alg_0_dL_infinity = 1.0 / (1.0 + torch.exp((E + 6.6) / -6.6))
        d_dL = torch.zeros_like(y[:, 0]) + ((alg_0_dL_infinity - dL) / tau_dL)
        alg_1_fL_infinity = 1.0 / (1.0 + torch.exp((E + 25.0) / 6.0))
        alg_9_tau_fL = 0.031 + 1.0 / (1.0 + torch.exp((E + 37.6) / 8.1))
        d_fL = torch.zeros_like(y[:, 0]) + ((alg_1_fL_infinity - fL) / alg_9_tau_fL)
        alg_2_dT_infinity = 1.0 / (1.0 + torch.exp((E + 23.0) / -6.1))
        alg_10_tau_dT = 0.0006 + 0.0054 / (1.0 + torch.exp(0.03 * (E + 100.0)))
        d_dT = torch.zeros_like(y[:, 0]) + ((alg_2_dT_infinity - dT) / alg_10_tau_dT)
        alg_3_fT_infinity = 1.0 / (1.0 + torch.exp((E + 75.0) / 6.6))
        alg_11_tau_fT = 0.001 + 0.04 / (1.0 + torch.exp(0.08 * (E + 65.0)))
        d_fT = torch.zeros_like(y[:, 0]) + ((alg_3_fT_infinity - fT) / alg_11_tau_fT)
        alg_4_alpha_m = 200.0 * (E + 34.3) / (1.0 - torch.exp(-0.09 * (E + 34.3)))
        alg_12_beta_m = 8000.0 * torch.exp(-0.15 * (E + 56.2))
        d_m = torch.zeros_like(y[:, 0]) + (alg_4_alpha_m * (1.0 - m) - alg_12_beta_m * m)
        alg_5_alpha_h = 32.4 * torch.exp(-0.14 * (E + 93.4))
        alg_13_beta_h = 709.0 / (1.0 + 4.2 * torch.exp(-0.06 * (E + 45.4)))
        d_h = torch.zeros_like(y[:, 0]) + (alg_5_alpha_h * (1.0 - h) - alg_13_beta_h * h)
        alg_6_x_infinity = 1.0 / (1.0 + torch.exp((E + 25.1) / -7.4))
        alg_14_tau_x = 1.0 / (17.0 * torch.exp(0.0398 * E) + 0.211 * torch.exp(-0.051 * E))
        d_x = torch.zeros_like(y[:, 0]) + ((alg_6_x_infinity - x) / alg_14_tau_x)
        alg_7_alpha_y = 0.36 * (E + 137.8) / (torch.exp(0.066 * (E + 137.8)) - 1.0)
        alg_15_beta_y = 0.1 * (E + 76.3) / (1.0 - torch.exp(-0.21 * (E + 76.3)))
        d_state_y = torch.zeros_like(y[:, 0]) + (alg_7_alpha_y * (1.0 - state_y) - alg_15_beta_y * state_y)
        alg_8_E_Ca = R * T / (2.0 * F) * torch.log(Cao / Cai)
        alg_18_i_CaL = g_CaL * dL * fL * fL2 * (E - alg_8_E_Ca + 75.0)
        alg_19_i_CaT = g_CaT * dT * fT * (E - alg_8_E_Ca + 75.0)
        alg_34_k41 = torch.exp(-Qn * E * F / (2.0 * R * T))
        alg_32_k34 = Nao / (K3no + Nao)
        alg_28_do = 1.0 + Cao / Kco + Cao / Kco * torch.exp(Qco * E * F / (R * T)) + Nao / K1no + Nao ** 2.0 / (K1no * K2no) + Nao ** 3.0 / (K1no * K2no * K3no)
        alg_30_k23 = (Nao ** 2.0 / (K1no * K2no) + Nao ** 3.0 / (K1no * K2no * K3no)) * torch.exp(-Qn * E * F / (2.0 * R * T)) / alg_28_do
        alg_31_k21 = Cao / Kco * torch.exp(-Qco * E * F / (R * T)) / alg_28_do
        alg_29_k32 = torch.exp(Qn * E * F / (2.0 * R * T))
        alg_37_k43 = Nai / (K3ni + Nai)
        alg_38_x1 = alg_34_k41 * alg_32_k34 * (alg_30_k23 + alg_31_k21) + alg_31_k21 * alg_29_k32 * (alg_37_k43 + alg_34_k41)
        alg_33_di = 1.0 + Cai / Kci + Cai / Kci * torch.exp(-Qci * E * F / (R * T)) + Cai * Nai / (Kci * Kcni) + Nai / K1ni + Nai ** 2.0 / (K1ni * K2ni) + Nai ** 3.0 / (K1ni * K2ni * K3ni)
        alg_36_k12 = Cai / Kci * torch.exp(-Qci * E * F / (R * T)) / alg_33_di
        alg_35_k14 = (Nai ** 2.0 / (K1ni * K2ni) + Nai ** 3.0 / (K1ni * K2ni * K3ni)) * torch.exp(Qn * E * F / (2.0 * R * T)) / alg_33_di
        alg_39_x2 = alg_29_k32 * alg_37_k43 * (alg_35_k14 + alg_36_k12) + alg_34_k41 * alg_36_k12 * (alg_32_k34 + alg_29_k32)
        alg_40_x3 = alg_35_k14 * alg_37_k43 * (alg_30_k23 + alg_31_k21) + alg_36_k12 * alg_30_k23 * (alg_37_k43 + alg_34_k41)
        alg_41_x4 = alg_30_k23 * alg_32_k34 * (alg_35_k14 + alg_36_k12) + alg_35_k14 * alg_31_k21 * (alg_32_k34 + alg_29_k32)
        alg_42_i_NaCa = kNaCa * (alg_39_x2 * alg_31_k21 - alg_38_x1 * alg_36_k12) / (alg_38_x1 + alg_39_x2 + alg_40_x3 + alg_41_x4)
        d_Cao = torch.zeros_like(y[:, 0]) + (1.0 * (alg_18_i_CaL + alg_19_i_CaT - 2.0 * alg_42_i_NaCa) / (2.0 * F * 1.0 * V_e) + (Cab - Cao) / tau_b)
        alg_16_E_Na = R * T / F * torch.log(Nao / Nai)
        alg_20_i_Na = g_Na * m ** 3.0 * h * (E - alg_16_E_Na)
        alg_27_i_p = i_pmax * Nai / (Nai + KmNa) * Ko / (Ko + KmK) * (1.0 - ((E - 40.0) / 211.0) ** 2.0)
        alg_43_i_bNa = g_Nab * (E - alg_16_E_Na)
        alg_22_i_KNa = x * Kk * P_KNa * (Ko / 1.0) ** 0.59 * (Nai - Nao * torch.exp(-E * F / (R * T)))
        alg_24_i_fNa = state_y * Ko ** 1.83 / (Ko ** 1.83 + Kmf ** 1.83) * g_fNa * (E - alg_16_E_Na)
        d_Nai = torch.zeros_like(y[:, 0]) + (-1.0 * (alg_43_i_bNa + alg_24_i_fNa + alg_20_i_Na + 3.0 * alg_27_i_p + 3.0 * alg_42_i_NaCa + alg_22_i_KNa) / (F * 1.0 * V_i))
        d_Nao = torch.zeros_like(y[:, 0]) + (1.0 * (alg_43_i_bNa + alg_24_i_fNa + alg_20_i_Na + 3.0 * alg_27_i_p + 3.0 * alg_42_i_NaCa + alg_22_i_KNa) / (F * 1.0 * V_e) + (Nab - Nao) / tau_b)
        alg_45_i_bK = KbK * (Ko / 1.0) ** 0.41 * (Ki - Ko * torch.exp(-E * F / (R * T)))
        alg_21_i_KK = x * Kk * (Ko / 1.0) ** 0.59 * (Ki - Ko * torch.exp(-E * F / (R * T)))
        alg_17_E_K = R * T / F * torch.log(Ko / Ki)
        alg_25_i_fK = state_y * Ko ** 1.83 / (Ko ** 1.83 + Kmf ** 1.83) * g_fK * (E - alg_17_E_K)
        d_Ki = torch.zeros_like(y[:, 0]) + (-1.0 * (alg_21_i_KK + alg_25_i_fK - 2.0 * alg_27_i_p + alg_45_i_bK) / (F * 1.0 * V_i))
        d_Ko = torch.zeros_like(y[:, 0]) + (1.0 * (alg_21_i_KK + alg_25_i_fK - 2.0 * alg_27_i_p + alg_45_i_bK) / (F * 1.0 * V_e) + (Kb - Ko) / tau_b)
        alg_44_i_up = i_up_max * Cai ** 2.0 / (Cai ** 2.0 + KmCaup ** 2.0)
        alg_46_i_tr = 2.0 * 1.0 * V_rel * F / (1.0 * tau_tr) * Caup
        d_Caup = torch.zeros_like(y[:, 0]) + (1.0 * (alg_44_i_up - alg_46_i_tr) / (2.0 * 1.0 * V_up * F))
        alg_23_i_K = alg_21_i_KK + alg_22_i_KNa
        alg_26_i_f = alg_25_i_fK + alg_24_i_fNa
        alg_47_i_tot = alg_18_i_CaL + alg_19_i_CaT + alg_20_i_Na + alg_23_i_K + alg_26_i_f + alg_27_i_p + alg_42_i_NaCa + alg_43_i_bNa + alg_45_i_bK
        d_E = torch.zeros_like(y[:, 0]) + (-alg_47_i_tot / C)
        alg_48_i_rel = 2.0 * 1.0 * V_rel * F / (1.0 * tau_rel) * Carel * Cai ** 2.0 / (Cai ** 2.0 + KmCarel ** 2.0)
        d_Cai = torch.zeros_like(y[:, 0]) + (-1.0 * (alg_18_i_CaL + alg_19_i_CaT - 2.0 * alg_42_i_NaCa + alg_44_i_up + -alg_48_i_rel) / (2.0 * F * 1.0 * V_i))
        d_Carel = torch.zeros_like(y[:, 0]) + (1.0 * (alg_46_i_tr - alg_48_i_rel) / (2.0 * 1.0 * V_rel * F))

        return torch.stack(
            [d_E, d_Cai, d_Cao, d_Nai, d_Nao, d_Ki, d_Ko, d_dL, d_fL, d_fL2, d_dT, d_fT, d_m, d_h, d_x, d_state_y, d_Caup, d_Carel],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
