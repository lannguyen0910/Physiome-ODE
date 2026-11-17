import torch
import torch.nn as nn
import math


class CardiovascularCirculationMainIVadODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/MainIVad/model.py.
    Dynamic states: 7; independent constants: 57; derived constants used by rates: 1.
    Original constant slots: 58; algebraic slots: 24.
    """
    source_file = 'cardiovascular_circulation/MainIVad/model.py'
    source_state_dim = 7
    source_constant_dim = 58
    algebraic_dim = 24
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
        self.parameter_dim = 57

        self.parameters = (
            parameters
            if parameters is not None
            else {'CVao': 350, 'PlvIni': 1, 'VlvIni': 5, 'Vlv0': 500, 'CVmi': 400, 'PlaIni': 1, 'VlaIni': 4, 'Vla0': 20, 'ElvMax': 0.5, 'ElvMin': 0.1, 'T': 1, 'Ts1': 0.3, 'Ts2': 0.45, 'ElaMax': 0.25, 'ElaMin': 0.15, 'Tpwb': 0.92, 'Tpww': 0.09, 'Rsas': 0.003, 'Csas': 0.08, 'Lsas': 6.2e-05, 'P0sas': 100, 'Q0sas': 0, 'Rsat': 0.05, 'Csat': 1.6, 'Lsat': 0.0017, 'P0sat': 100, 'Q0sat': 0, 'Rsar': 0.5, 'Rscp': 0.52, 'Rsvn': 0.075, 'Csvn': 20.5, 'P0svn': 0, 'Ts2_2': 0.45, 'T_2': 1, 'Kp0': 19.5184, 'Kp1': -0.0030361, 'Kp2': -1.23045, 'Kp3': 0.000578974, 'Kp4': -5.8777e-08, 'Kp5': -1.27539e-06, 'Kp6': 2.04834e-10, 'W': 3600, 'ElaMax_2': 0.25, 'ElaMin_2': 0.15, 'PlaIni_2': 1, 'VlaIni_2': 4, 'ElvMax_2': 0.5, 'ElvMin_2': 0.1, 'PlvIni_2': 1, 'VlvIni_2': 5, 'Tpwb_2': 0.92, 'Tpww_2': 0.09, 'Ts1_2': 0.3, 'CVao_2': 350, 'CVmi_2': 400, 'Vlv0_2': 500, 'Vla0_2': 20}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Pi': 100, 'V': 500, 'V_2': 20, 'Qo': 0, 'Pi_2': 100, 'Qo_2': 0, 'Pi_3': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['CVao', 'PlvIni', 'VlvIni', 'Vlv0', 'CVmi', 'PlaIni', 'VlaIni', 'Vla0', 'ElvMax', 'ElvMin', 'T', 'Ts1', 'Ts2', 'ElaMax', 'ElaMin', 'Tpwb', 'Tpww', 'Rsas', 'Csas', 'Lsas', 'P0sas', 'Q0sas', 'Rsat', 'Csat', 'Lsat', 'P0sat', 'Q0sat', 'Rsar', 'Rscp', 'Rsvn', 'Csvn', 'P0svn', 'Ts2_2', 'T_2', 'Kp0', 'Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'W', 'ElaMax_2', 'ElaMin_2', 'PlaIni_2', 'VlaIni_2', 'ElvMax_2', 'ElvMin_2', 'PlvIni_2', 'VlvIni_2', 'Tpwb_2', 'Tpww_2', 'Ts1_2', 'CVao_2', 'CVmi_2', 'Vlv0_2', 'Vla0_2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Pi', 'V', 'V_2', 'Qo', 'Pi_2', 'Qo_2', 'Pi_3']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (7)
        Pi = y[:, 0]
        V = y[:, 1]
        V_2 = y[:, 2]
        Qo = y[:, 3]
        Pi_2 = y[:, 4]
        Qo_2 = y[:, 5]
        Pi_3 = y[:, 6]

        # Independent parameters (source constant order)
        CVao = theta[:, 0]  # constants[0]
        PlvIni = theta[:, 1]  # constants[1]
        VlvIni = theta[:, 2]  # constants[2]
        Vlv0 = theta[:, 3]  # constants[3]
        CVmi = theta[:, 4]  # constants[4]
        PlaIni = theta[:, 5]  # constants[5]
        VlaIni = theta[:, 6]  # constants[6]
        Vla0 = theta[:, 7]  # constants[7]
        ElvMax = theta[:, 8]  # constants[8]
        ElvMin = theta[:, 9]  # constants[9]
        T = theta[:, 10]  # constants[10]
        Ts1 = theta[:, 11]  # constants[11]
        Ts2 = theta[:, 12]  # constants[12]
        ElaMax = theta[:, 13]  # constants[13]
        ElaMin = theta[:, 14]  # constants[14]
        Tpwb = theta[:, 15]  # constants[15]
        Tpww = theta[:, 16]  # constants[16]
        Rsas = theta[:, 17]  # constants[17]
        Csas = theta[:, 18]  # constants[18]
        Lsas = theta[:, 19]  # constants[19]
        P0sas = theta[:, 20]  # constants[20]
        Q0sas = theta[:, 21]  # constants[21]
        Rsat = theta[:, 22]  # constants[22]
        Csat = theta[:, 23]  # constants[23]
        Lsat = theta[:, 24]  # constants[24]
        P0sat = theta[:, 25]  # constants[25]
        Q0sat = theta[:, 26]  # constants[26]
        Rsar = theta[:, 27]  # constants[27]
        Rscp = theta[:, 28]  # constants[28]
        Rsvn = theta[:, 29]  # constants[29]
        Csvn = theta[:, 30]  # constants[30]
        P0svn = theta[:, 31]  # constants[31]
        Ts2_2 = theta[:, 32]  # constants[32]
        T_2 = theta[:, 33]  # constants[33]
        Kp0 = theta[:, 34]  # constants[34]
        Kp1 = theta[:, 35]  # constants[35]
        Kp2 = theta[:, 36]  # constants[36]
        Kp3 = theta[:, 37]  # constants[37]
        Kp4 = theta[:, 38]  # constants[38]
        Kp5 = theta[:, 39]  # constants[39]
        Kp6 = theta[:, 40]  # constants[40]
        W = theta[:, 41]  # constants[41]
        ElaMax_2 = theta[:, 42]  # constants[42]
        ElaMin_2 = theta[:, 43]  # constants[43]
        PlaIni_2 = theta[:, 44]  # constants[44]
        VlaIni_2 = theta[:, 45]  # constants[45]
        ElvMax_2 = theta[:, 46]  # constants[46]
        ElvMin_2 = theta[:, 47]  # constants[47]
        PlvIni_2 = theta[:, 48]  # constants[48]
        VlvIni_2 = theta[:, 49]  # constants[49]
        Tpwb_2 = theta[:, 50]  # constants[50]
        Tpww_2 = theta[:, 51]  # constants[51]
        Ts1_2 = theta[:, 52]  # constants[52]
        CVao_2 = theta[:, 53]  # constants[53]
        CVmi_2 = theta[:, 54]  # constants[54]
        Vlv0_2 = theta[:, 55]  # constants[55]
        Vla0_2 = theta[:, 56]  # constants[56]

        # Derived constants from the original initConsts()
        Wn = 1.0 * W  # constants[57]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Pi = torch.zeros_like(y[:, 0])
        d_V = torch.zeros_like(y[:, 0])
        d_V_2 = torch.zeros_like(y[:, 0])
        d_Qo = torch.zeros_like(y[:, 0])
        d_Pi_2 = torch.zeros_like(y[:, 0])
        d_Qo_2 = torch.zeros_like(y[:, 0])
        d_Pi_3 = torch.zeros_like(y[:, 0])
        alg_0_mt = torch.zeros_like(y[:, 0])
        alg_1_et = torch.zeros_like(y[:, 0])
        alg_2_E = torch.zeros_like(y[:, 0])
        alg_3_Pi = torch.zeros_like(y[:, 0])
        alg_4_Tao = torch.zeros_like(y[:, 0])
        alg_5_Qo = torch.zeros_like(y[:, 0])
        alg_6_Pi_2 = torch.zeros_like(y[:, 0])
        alg_7_mt_2 = torch.zeros_like(y[:, 0])
        alg_8_et_2 = torch.zeros_like(y[:, 0])
        alg_9_E_2 = torch.zeros_like(y[:, 0])
        alg_10_Pi_3 = torch.zeros_like(y[:, 0])
        alg_11_Tao_2 = torch.zeros_like(y[:, 0])
        alg_12_Qo_2 = torch.zeros_like(y[:, 0])
        alg_13_Qo_3 = torch.zeros_like(y[:, 0])
        alg_14_Pi_4 = torch.zeros_like(y[:, 0])
        alg_15_Qo_4 = torch.zeros_like(y[:, 0])
        alg_16_dPv = torch.zeros_like(y[:, 0])
        alg_17_Pi_5 = torch.zeros_like(y[:, 0])
        alg_18_Qo_5 = torch.zeros_like(y[:, 0])
        alg_19_Qvo = torch.zeros_like(y[:, 0])
        alg_20_Pi_6 = torch.zeros_like(y[:, 0])
        alg_21_Qvi = torch.zeros_like(y[:, 0])
        alg_22_Qo_6 = torch.zeros_like(y[:, 0])
        alg_23_Qo_7 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_mt = t - T * torch.floor(t / T)
        alg_1_et = torch.where((alg_0_mt >= 0.0) & (alg_0_mt <= Ts1 * T), 1.0 - torch.cos(3.14159 * alg_0_mt / (Ts1 * T)), torch.where((alg_0_mt > Ts1 * T) & (alg_0_mt <= Ts2 * T), 1.0 + torch.cos(3.14159 * (alg_0_mt - Ts1 * T) / ((Ts2 - Ts1) * T)), torch.where((alg_0_mt > Ts2 * T) & (alg_0_mt < T), 0.0, (1e309-1e309))))
        alg_2_E = ElvMin + alg_1_et * (ElvMax - ElvMin) / 2.0
        alg_3_Pi = PlvIni + alg_2_E * (V - VlvIni)
        alg_4_Tao = torch.where(alg_3_Pi >= Pi, 1.0, torch.where(alg_3_Pi < Pi, 0.0, (1e309-1e309)))
        alg_5_Qo = torch.where(alg_3_Pi >= Pi, CVao * alg_4_Tao * torch.abs(alg_3_Pi - Pi) ** 0.5, torch.where(alg_3_Pi < Pi, -1.0 * CVao * alg_4_Tao * torch.abs(Pi - alg_3_Pi) ** 0.5, (1e309-1e309)))
        d_Pi = torch.zeros_like(y[:, 0]) + ((alg_5_Qo - Qo) / Csas)
        alg_14_Pi_4 = Pi_2
        d_Qo = torch.zeros_like(y[:, 0]) + ((Pi - alg_14_Pi_4 - Rsas * Qo) / Lsas)
        alg_7_mt_2 = t - T * torch.floor(t / T)
        alg_8_et_2 = torch.where((alg_7_mt_2 >= 0.0) & (alg_7_mt_2 <= (Tpwb + Tpww - 1.0) * T), 1.0 - torch.cos(2.0 * 3.14159 * (alg_7_mt_2 - (Tpwb - 1.0) * T) / (Tpww * T)), torch.where((alg_7_mt_2 > (Tpwb + Tpww - 1.0) * T) & (alg_7_mt_2 <= Tpwb * T), 0.0, torch.where((alg_7_mt_2 > Tpwb * T) & (alg_7_mt_2 <= T), 1.0 - torch.cos(2.0 * 3.14159 * (alg_7_mt_2 - Tpwb * T) / (Tpww * T)), (1e309-1e309))))
        alg_9_E_2 = ElaMin + alg_8_et_2 * (ElaMax - ElaMin) / 2.0
        alg_10_Pi_3 = PlaIni + alg_9_E_2 * (V_2 - VlaIni)
        alg_18_Qo_5 = (Pi_3 - alg_10_Pi_3) / Rsvn
        alg_6_Pi_2 = alg_3_Pi
        alg_11_Tao_2 = torch.where(alg_10_Pi_3 >= alg_6_Pi_2, 1.0, torch.where(alg_10_Pi_3 < alg_6_Pi_2, 0.0, (1e309-1e309)))
        alg_12_Qo_2 = torch.where(alg_10_Pi_3 >= alg_6_Pi_2, CVmi * alg_11_Tao_2 * torch.abs(alg_10_Pi_3 - alg_6_Pi_2) ** 0.5, torch.where(alg_10_Pi_3 < alg_6_Pi_2, -1.0 * CVmi * alg_11_Tao_2 * torch.abs(alg_6_Pi_2 - alg_10_Pi_3) ** 0.5, (1e309-1e309)))
        d_V_2 = torch.zeros_like(y[:, 0]) + (alg_18_Qo_5 - alg_12_Qo_2)
        alg_13_Qo_3 = Qo_2
        alg_17_Pi_5 = Pi_3 + Rscp * alg_13_Qo_3
        alg_20_Pi_6 = alg_17_Pi_5 + Rsar * Qo_2
        d_Qo_2 = torch.zeros_like(y[:, 0]) + ((Pi_2 - alg_20_Pi_6 - Rsat * Qo_2) / Lsat)
        alg_15_Qo_4 = alg_13_Qo_3
        d_Pi_3 = torch.zeros_like(y[:, 0]) + ((alg_15_Qo_4 - alg_18_Qo_5) / Csvn)
        alg_16_dPv = alg_14_Pi_4 - alg_6_Pi_2
        alg_19_Qvo = (Kp0 + Kp1 * Wn + Kp2 * alg_16_dPv + Kp3 * Wn * alg_16_dPv + Kp4 * Wn ** 2.0 * alg_16_dPv + Kp5 * Wn * alg_16_dPv ** 2.0 + Kp6 * Wn ** 2.0 * alg_16_dPv ** 2.0) * 16.6667
        alg_22_Qo_6 = Qo + alg_19_Qvo
        d_Pi_2 = torch.zeros_like(y[:, 0]) + ((alg_22_Qo_6 - Qo_2) / Csat)
        alg_21_Qvi = -1.0 * alg_19_Qvo
        alg_23_Qo_7 = alg_12_Qo_2 + alg_21_Qvi
        d_V = torch.zeros_like(y[:, 0]) + (alg_23_Qo_7 - alg_5_Qo)

        return torch.stack(
            [d_Pi, d_V, d_V_2, d_Qo, d_Pi_2, d_Qo_2, d_Pi_3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
