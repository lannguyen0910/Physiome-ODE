import torch
import torch.nn as nn
import math


class MetabolismCloutier2009AODE(nn.Module):
    """
    PyTorch translation of metabolism/cloutier_2009_a/model.py.
    Dynamic states: 7; independent constants: 42; derived constants used by rates: 0.
    Original constant slots: 42; algebraic slots: 19.
    """
    source_file = 'metabolism/cloutier_2009_a/model.py'
    source_state_dim = 7
    source_constant_dim = 42
    algebraic_dim = 19
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 7
        self.parameter_dim = 42

        self.parameters = (
            parameters
            if parameters is not None
            else {'Vmax_hk': 2.5, 'Km_ATP_hk': 0.5, 'KI_F6P': 0.068, 'Vmax_pfk': 3.85, 'Km_ATP_pfk': 0.05, 'Km_F6P_pfk': 0.18, 'Km_F26P_pfk': 0.01, 'Vmaxf_pfk2': 0, 'Vmaxr_pfk2': 0, 'Km_ATP_pfk2': 0.05, 'Km_F6P_pfk2': 0.01, 'Km_F26P_pfk2': 0.0001, 'Vmax_pk': 5, 'Km_ADP_pk': 0.005, 'Km_GAP_pk': 0.4, 'Vmax_op': 1, 'Km_ADP_op': 0.005, 'Km_PYR_op': 0.5, 'kf_ldh': 0, 'kr_ldh': 0, 'kf_ck': 0, 'kr_ck': 0, 'PCrtot': 20, 'Vmax_ATPase': 0.9355, 'Km_ATP': 0.5, 'Vlac_0': 0.355, 'K_LAC_eff': 0.71, 'K_LAC': 0.641, 'ANP': 2.51, 'Q_adk': 0.92, 'nATP': 0.4, 'KI_ATP': 1, 'nAMP': 0.5, 'Ka_AMP': 0.05, 'stim': 1, 'to': 50, 'tend': 200, 'v1_n': 0.5, 'v2_n': 0, 't_n_stim': 2, 'Kamp_pfk2': 0.005, 'nh_amp': 2}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'F6P': 0.2, 'F26P': 0.001, 'GAP': 0.0405, 'PYR': 0.1, 'LAC': 0.5, 'ATP': 2.402, 'PCr': 18.14}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Vmax_hk', 'Km_ATP_hk', 'KI_F6P', 'Vmax_pfk', 'Km_ATP_pfk', 'Km_F6P_pfk', 'Km_F26P_pfk', 'Vmaxf_pfk2', 'Vmaxr_pfk2', 'Km_ATP_pfk2', 'Km_F6P_pfk2', 'Km_F26P_pfk2', 'Vmax_pk', 'Km_ADP_pk', 'Km_GAP_pk', 'Vmax_op', 'Km_ADP_op', 'Km_PYR_op', 'kf_ldh', 'kr_ldh', 'kf_ck', 'kr_ck', 'PCrtot', 'Vmax_ATPase', 'Km_ATP', 'Vlac_0', 'K_LAC_eff', 'K_LAC', 'ANP', 'Q_adk', 'nATP', 'KI_ATP', 'nAMP', 'Ka_AMP', 'stim', 'to', 'tend', 'v1_n', 'v2_n', 't_n_stim', 'Kamp_pfk2', 'nh_amp']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['F6P', 'F26P', 'GAP', 'PYR', 'LAC', 'ATP', 'PCr']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (7)
        F6P = y[:, 0]
        F26P = y[:, 1]
        GAP = y[:, 2]
        PYR = y[:, 3]
        LAC = y[:, 4]
        ATP = y[:, 5]
        PCr = y[:, 6]

        # Independent parameters (source constant order)
        Vmax_hk = theta[:, 0]  # constants[0]
        Km_ATP_hk = theta[:, 1]  # constants[1]
        KI_F6P = theta[:, 2]  # constants[2]
        Vmax_pfk = theta[:, 3]  # constants[3]
        Km_ATP_pfk = theta[:, 4]  # constants[4]
        Km_F6P_pfk = theta[:, 5]  # constants[5]
        Km_F26P_pfk = theta[:, 6]  # constants[6]
        Vmaxf_pfk2 = theta[:, 7]  # constants[7]
        Vmaxr_pfk2 = theta[:, 8]  # constants[8]
        Km_ATP_pfk2 = theta[:, 9]  # constants[9]
        Km_F6P_pfk2 = theta[:, 10]  # constants[10]
        Km_F26P_pfk2 = theta[:, 11]  # constants[11]
        Vmax_pk = theta[:, 12]  # constants[12]
        Km_ADP_pk = theta[:, 13]  # constants[13]
        Km_GAP_pk = theta[:, 14]  # constants[14]
        Vmax_op = theta[:, 15]  # constants[15]
        Km_ADP_op = theta[:, 16]  # constants[16]
        Km_PYR_op = theta[:, 17]  # constants[17]
        kf_ldh = theta[:, 18]  # constants[18]
        kr_ldh = theta[:, 19]  # constants[19]
        kf_ck = theta[:, 20]  # constants[20]
        kr_ck = theta[:, 21]  # constants[21]
        PCrtot = theta[:, 22]  # constants[22]
        Vmax_ATPase = theta[:, 23]  # constants[23]
        Km_ATP = theta[:, 24]  # constants[24]
        Vlac_0 = theta[:, 25]  # constants[25]
        K_LAC_eff = theta[:, 26]  # constants[26]
        K_LAC = theta[:, 27]  # constants[27]
        ANP = theta[:, 28]  # constants[28]
        Q_adk = theta[:, 29]  # constants[29]
        nATP = theta[:, 30]  # constants[30]
        KI_ATP = theta[:, 31]  # constants[31]
        nAMP = theta[:, 32]  # constants[32]
        Ka_AMP = theta[:, 33]  # constants[33]
        stim = theta[:, 34]  # constants[34]
        to = theta[:, 35]  # constants[35]
        tend = theta[:, 36]  # constants[36]
        v1_n = theta[:, 37]  # constants[37]
        v2_n = theta[:, 38]  # constants[38]
        t_n_stim = theta[:, 39]  # constants[39]
        Kamp_pfk2 = theta[:, 40]  # constants[40]
        nh_amp = theta[:, 41]  # constants[41]

        # Initialize rates and algebraic variables, as in computeRates()
        d_F6P = torch.zeros_like(y[:, 0])
        d_F26P = torch.zeros_like(y[:, 0])
        d_GAP = torch.zeros_like(y[:, 0])
        d_PYR = torch.zeros_like(y[:, 0])
        d_LAC = torch.zeros_like(y[:, 0])
        d_ATP = torch.zeros_like(y[:, 0])
        d_PCr = torch.zeros_like(y[:, 0])
        alg_0_V_hk = torch.zeros_like(y[:, 0])
        alg_1_V_ldh = torch.zeros_like(y[:, 0])
        alg_2_Cr = torch.zeros_like(y[:, 0])
        alg_3_u = torch.zeros_like(y[:, 0])
        alg_4_ADP = torch.zeros_like(y[:, 0])
        alg_5_V_pk = torch.zeros_like(y[:, 0])
        alg_6_V_op = torch.zeros_like(y[:, 0])
        alg_7_V_ck = torch.zeros_like(y[:, 0])
        alg_8_dAMP_dATP = torch.zeros_like(y[:, 0])
        alg_9_AMP = torch.zeros_like(y[:, 0])
        alg_10_ATP_inh = torch.zeros_like(y[:, 0])
        alg_11_AMP_act = torch.zeros_like(y[:, 0])
        alg_12_V_pfk = torch.zeros_like(y[:, 0])
        alg_13_unitpulseSB = torch.zeros_like(y[:, 0])
        alg_14_v_stim = torch.zeros_like(y[:, 0])
        alg_15_V_ATPase = torch.zeros_like(y[:, 0])
        alg_16_V_lac = torch.zeros_like(y[:, 0])
        alg_17_AMP_pfk2 = torch.zeros_like(y[:, 0])
        alg_18_V_pfk2 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_3_u = Q_adk ** 2.0 + 4.0 * Q_adk * (ANP / ATP - 1.0)
        alg_4_ADP = ATP / 2.0 * (-Q_adk + alg_3_u ** (1.0 / 2))
        alg_5_V_pk = Vmax_pk * (GAP / (GAP + Km_GAP_pk)) * (alg_4_ADP / (alg_4_ADP + Km_ADP_pk)) * 0.1104
        alg_1_V_ldh = kf_ldh * PYR - kr_ldh * LAC
        alg_6_V_op = Vmax_op * (PYR / (PYR + Km_PYR_op)) * (alg_4_ADP / (alg_4_ADP + Km_ADP_op)) * (1.0 / (1.0 + 0.1 * 24.0))
        d_PYR = torch.zeros_like(y[:, 0]) + (alg_5_V_pk - (alg_6_V_op + alg_1_V_ldh))
        alg_2_Cr = PCrtot - PCr
        alg_7_V_ck = kf_ck * PCr * alg_4_ADP - kr_ck * alg_2_Cr * ATP
        d_PCr = torch.zeros_like(y[:, 0]) + (-alg_7_V_ck)
        alg_9_AMP = ANP - (ATP + alg_4_ADP)
        alg_11_AMP_act = ((1.0 + alg_9_AMP / Ka_AMP) / (1.0 + nAMP * (alg_9_AMP / Ka_AMP))) ** 4.0
        alg_10_ATP_inh = ((1.0 + nATP * (ATP / KI_ATP)) / (1.0 + ATP / KI_ATP)) ** 4.0
        alg_12_V_pfk = Vmax_pfk * (F6P / (F6P + Km_F6P_pfk)) * (ATP / (ATP + Km_ATP_pfk)) * (F26P / (F26P + Km_F26P_pfk)) * alg_10_ATP_inh * alg_11_AMP_act
        d_GAP = torch.zeros_like(y[:, 0]) + (2.0 * alg_12_V_pfk - alg_5_V_pk)
        alg_13_unitpulseSB = torch.where((t >= to) & (t <= to + tend), 1.0, 0.0)
        alg_14_v_stim = stim * (v1_n + v2_n * ((t - to) / t_n_stim) * torch.exp(-((t - to) * (alg_13_unitpulseSB / t_n_stim)))) * alg_13_unitpulseSB
        alg_16_V_lac = Vlac_0 * (1.0 + alg_14_v_stim * K_LAC) - K_LAC_eff * LAC
        d_LAC = torch.zeros_like(y[:, 0]) + (2.25 * alg_1_V_ldh + alg_16_V_lac)
        alg_0_V_hk = Vmax_hk * (ATP / (ATP + Km_ATP_hk)) * (1.0 + (F6P / KI_F6P) ** 4.0) ** (-1.0)
        alg_17_AMP_pfk2 = (alg_9_AMP / Kamp_pfk2) ** nh_amp / (1.0 + (alg_9_AMP / Kamp_pfk2) ** nh_amp)
        alg_18_V_pfk2 = Vmaxf_pfk2 * (ATP / (ATP + Km_ATP_pfk2)) * (F6P / (F6P + Km_F6P_pfk2)) * alg_17_AMP_pfk2 - Vmaxr_pfk2 * (F26P / (F26P + Km_F26P_pfk2))
        d_F6P = torch.zeros_like(y[:, 0]) + (alg_0_V_hk - (alg_12_V_pfk - alg_18_V_pfk2))
        d_F26P = torch.zeros_like(y[:, 0]) + (alg_18_V_pfk2)
        alg_15_V_ATPase = Vmax_ATPase * (ATP / (ATP + Km_ATP)) * (1.0 + alg_14_v_stim)
        alg_8_dAMP_dATP = -1.0 + Q_adk / 2.0 + -(0.5 * alg_3_u ** (1.0 / 2)) + Q_adk * (ANP / (ATP * alg_3_u ** (1.0 / 2)))
        d_ATP = torch.zeros_like(y[:, 0]) + ((2.0 * alg_5_V_pk + 15.0 * alg_6_V_op + alg_7_V_ck - (alg_0_V_hk + alg_12_V_pfk + alg_18_V_pfk2 + alg_15_V_ATPase)) * (1.0 - alg_8_dAMP_dATP) ** (-1.0))

        return torch.stack(
            [d_F6P, d_F26P, d_GAP, d_PYR, d_LAC, d_ATP, d_PCr],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
