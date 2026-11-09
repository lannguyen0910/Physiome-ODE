import torch
import torch.nn as nn
import math


class CardiovascularCirculationModelMainODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/ModelMain/model.py.
    Dynamic states: 14; independent constants: 88; derived constants used by rates: 0.
    Original constant slots: 88; algebraic slots: 34.
    """
    source_file = 'cardiovascular_circulation/ModelMain/model.py'
    source_state_dim = 14
    source_constant_dim = 88
    algebraic_dim = 34
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 14
        self.parameter_dim = 88

        self.parameters = (
            parameters
            if parameters is not None
            else {'CVao': 350, 'PlvIni': 1, 'VlvIni': 5, 'Vlv0': 500, 'CVmi': 400, 'PlaIni': 1, 'VlaIni': 4, 'Vla0': 20, 'ElvMax': 2.5, 'ElvMin': 0.1, 'T': 1, 'Ts1': 0.3, 'Ts2': 0.45, 'ElaMax': 0.25, 'ElaMin': 0.15, 'Tpwb': 0.92, 'Tpww': 0.09, 'EraMax': 0.25, 'EraMin': 0.15, 'PraIni': 1, 'VraIni': 4, 'ErvMax': 1.15, 'ErvMin': 0.1, 'PrvIni': 1, 'VrvIni': 10, 'CVpa': 350, 'CVti': 400, 'Vra0': 20, 'Vrv0': 500, 'CVpa_2': 350, 'PrvIni_2': 1, 'VrvIni_2': 10, 'Vrv0_2': 500, 'CVti_2': 400, 'PraIni_2': 1, 'VraIni_2': 4, 'Vra0_2': 20, 'ErvMax_2': 1.15, 'ErvMin_2': 0.1, 'T_2': 1, 'Ts1_2': 0.3, 'Ts2_2': 0.45, 'EraMax_2': 0.25, 'EraMin_2': 0.15, 'Tpwb_2': 0.92, 'Tpww_2': 0.09, 'ElaMax_2': 0.25, 'ElaMin_2': 0.15, 'PlaIni_2': 1, 'VlaIni_2': 4, 'ElvMax_2': 2.5, 'ElvMin_2': 0.1, 'PlvIni_2': 1, 'VlvIni_2': 5, 'CVao_2': 350, 'CVmi_2': 400, 'Vlv0_2': 500, 'Vla0_2': 20, 'Rsas': 0.003, 'Csas': 0.08, 'Lsas': 6.2e-05, 'P0sas': 100, 'Q0sas': 0, 'Rsat': 0.05, 'Csat': 1.6, 'Lsat': 0.0017, 'P0sat': 100, 'Q0sat': 0, 'Rsar': 0.5, 'Rscp': 0.52, 'Rsvn': 0.075, 'Csvn': 20.5, 'P0svn': 0, 'Rpas': 0.002, 'Cpas': 0.18, 'Lpas': 5.2e-05, 'P0pas': 30, 'Q0pas': 0, 'Rpat': 0.01, 'Cpat': 3.8, 'Lpat': 0.0017, 'P0pat': 30, 'Q0pat': 0, 'Rpar': 0.05, 'Rpcp': 0.25, 'Rpvn': 0.0006, 'Cpvn': 20.5, 'P0pvn': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Pi': 100, 'Pi_2': 30, 'V': 500, 'V_2': 20, 'V_3': 500, 'V_4': 20, 'Pi_3': 100, 'Qo': 0, 'Qo_2': 0, 'Pi_4': 0, 'Pi_5': 30, 'Qo_3': 0, 'Qo_4': 0, 'Pi_6': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['CVao', 'PlvIni', 'VlvIni', 'Vlv0', 'CVmi', 'PlaIni', 'VlaIni', 'Vla0', 'ElvMax', 'ElvMin', 'T', 'Ts1', 'Ts2', 'ElaMax', 'ElaMin', 'Tpwb', 'Tpww', 'EraMax', 'EraMin', 'PraIni', 'VraIni', 'ErvMax', 'ErvMin', 'PrvIni', 'VrvIni', 'CVpa', 'CVti', 'Vra0', 'Vrv0', 'CVpa_2', 'PrvIni_2', 'VrvIni_2', 'Vrv0_2', 'CVti_2', 'PraIni_2', 'VraIni_2', 'Vra0_2', 'ErvMax_2', 'ErvMin_2', 'T_2', 'Ts1_2', 'Ts2_2', 'EraMax_2', 'EraMin_2', 'Tpwb_2', 'Tpww_2', 'ElaMax_2', 'ElaMin_2', 'PlaIni_2', 'VlaIni_2', 'ElvMax_2', 'ElvMin_2', 'PlvIni_2', 'VlvIni_2', 'CVao_2', 'CVmi_2', 'Vlv0_2', 'Vla0_2', 'Rsas', 'Csas', 'Lsas', 'P0sas', 'Q0sas', 'Rsat', 'Csat', 'Lsat', 'P0sat', 'Q0sat', 'Rsar', 'Rscp', 'Rsvn', 'Csvn', 'P0svn', 'Rpas', 'Cpas', 'Lpas', 'P0pas', 'Q0pas', 'Rpat', 'Cpat', 'Lpat', 'P0pat', 'Q0pat', 'Rpar', 'Rpcp', 'Rpvn', 'Cpvn', 'P0pvn']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Pi', 'Pi_2', 'V', 'V_2', 'V_3', 'V_4', 'Pi_3', 'Qo', 'Qo_2', 'Pi_4', 'Pi_5', 'Qo_3', 'Qo_4', 'Pi_6']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (14)
        Pi = y[:, 0]
        Pi_2 = y[:, 1]
        V = y[:, 2]
        V_2 = y[:, 3]
        V_3 = y[:, 4]
        V_4 = y[:, 5]
        Pi_3 = y[:, 6]
        Qo = y[:, 7]
        Qo_2 = y[:, 8]
        Pi_4 = y[:, 9]
        Pi_5 = y[:, 10]
        Qo_3 = y[:, 11]
        Qo_4 = y[:, 12]
        Pi_6 = y[:, 13]

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
        EraMax = theta[:, 17]  # constants[17]
        EraMin = theta[:, 18]  # constants[18]
        PraIni = theta[:, 19]  # constants[19]
        VraIni = theta[:, 20]  # constants[20]
        ErvMax = theta[:, 21]  # constants[21]
        ErvMin = theta[:, 22]  # constants[22]
        PrvIni = theta[:, 23]  # constants[23]
        VrvIni = theta[:, 24]  # constants[24]
        CVpa = theta[:, 25]  # constants[25]
        CVti = theta[:, 26]  # constants[26]
        Vra0 = theta[:, 27]  # constants[27]
        Vrv0 = theta[:, 28]  # constants[28]
        CVpa_2 = theta[:, 29]  # constants[29]
        PrvIni_2 = theta[:, 30]  # constants[30]
        VrvIni_2 = theta[:, 31]  # constants[31]
        Vrv0_2 = theta[:, 32]  # constants[32]
        CVti_2 = theta[:, 33]  # constants[33]
        PraIni_2 = theta[:, 34]  # constants[34]
        VraIni_2 = theta[:, 35]  # constants[35]
        Vra0_2 = theta[:, 36]  # constants[36]
        ErvMax_2 = theta[:, 37]  # constants[37]
        ErvMin_2 = theta[:, 38]  # constants[38]
        T_2 = theta[:, 39]  # constants[39]
        Ts1_2 = theta[:, 40]  # constants[40]
        Ts2_2 = theta[:, 41]  # constants[41]
        EraMax_2 = theta[:, 42]  # constants[42]
        EraMin_2 = theta[:, 43]  # constants[43]
        Tpwb_2 = theta[:, 44]  # constants[44]
        Tpww_2 = theta[:, 45]  # constants[45]
        ElaMax_2 = theta[:, 46]  # constants[46]
        ElaMin_2 = theta[:, 47]  # constants[47]
        PlaIni_2 = theta[:, 48]  # constants[48]
        VlaIni_2 = theta[:, 49]  # constants[49]
        ElvMax_2 = theta[:, 50]  # constants[50]
        ElvMin_2 = theta[:, 51]  # constants[51]
        PlvIni_2 = theta[:, 52]  # constants[52]
        VlvIni_2 = theta[:, 53]  # constants[53]
        CVao_2 = theta[:, 54]  # constants[54]
        CVmi_2 = theta[:, 55]  # constants[55]
        Vlv0_2 = theta[:, 56]  # constants[56]
        Vla0_2 = theta[:, 57]  # constants[57]
        Rsas = theta[:, 58]  # constants[58]
        Csas = theta[:, 59]  # constants[59]
        Lsas = theta[:, 60]  # constants[60]
        P0sas = theta[:, 61]  # constants[61]
        Q0sas = theta[:, 62]  # constants[62]
        Rsat = theta[:, 63]  # constants[63]
        Csat = theta[:, 64]  # constants[64]
        Lsat = theta[:, 65]  # constants[65]
        P0sat = theta[:, 66]  # constants[66]
        Q0sat = theta[:, 67]  # constants[67]
        Rsar = theta[:, 68]  # constants[68]
        Rscp = theta[:, 69]  # constants[69]
        Rsvn = theta[:, 70]  # constants[70]
        Csvn = theta[:, 71]  # constants[71]
        P0svn = theta[:, 72]  # constants[72]
        Rpas = theta[:, 73]  # constants[73]
        Cpas = theta[:, 74]  # constants[74]
        Lpas = theta[:, 75]  # constants[75]
        P0pas = theta[:, 76]  # constants[76]
        Q0pas = theta[:, 77]  # constants[77]
        Rpat = theta[:, 78]  # constants[78]
        Cpat = theta[:, 79]  # constants[79]
        Lpat = theta[:, 80]  # constants[80]
        P0pat = theta[:, 81]  # constants[81]
        Q0pat = theta[:, 82]  # constants[82]
        Rpar = theta[:, 83]  # constants[83]
        Rpcp = theta[:, 84]  # constants[84]
        Rpvn = theta[:, 85]  # constants[85]
        Cpvn = theta[:, 86]  # constants[86]
        P0pvn = theta[:, 87]  # constants[87]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Pi = torch.zeros_like(y[:, 0])
        d_Pi_2 = torch.zeros_like(y[:, 0])
        d_V = torch.zeros_like(y[:, 0])
        d_V_2 = torch.zeros_like(y[:, 0])
        d_V_3 = torch.zeros_like(y[:, 0])
        d_V_4 = torch.zeros_like(y[:, 0])
        d_Pi_3 = torch.zeros_like(y[:, 0])
        d_Qo = torch.zeros_like(y[:, 0])
        d_Qo_2 = torch.zeros_like(y[:, 0])
        d_Pi_4 = torch.zeros_like(y[:, 0])
        d_Pi_5 = torch.zeros_like(y[:, 0])
        d_Qo_3 = torch.zeros_like(y[:, 0])
        d_Qo_4 = torch.zeros_like(y[:, 0])
        d_Pi_6 = torch.zeros_like(y[:, 0])
        alg_0_mt = torch.zeros_like(y[:, 0])
        alg_1_mt_2 = torch.zeros_like(y[:, 0])
        alg_2_et = torch.zeros_like(y[:, 0])
        alg_3_et_2 = torch.zeros_like(y[:, 0])
        alg_4_E = torch.zeros_like(y[:, 0])
        alg_5_E_2 = torch.zeros_like(y[:, 0])
        alg_6_Pi = torch.zeros_like(y[:, 0])
        alg_7_Pi_2 = torch.zeros_like(y[:, 0])
        alg_8_Tao = torch.zeros_like(y[:, 0])
        alg_9_Tao_2 = torch.zeros_like(y[:, 0])
        alg_10_Qo = torch.zeros_like(y[:, 0])
        alg_11_Qo_2 = torch.zeros_like(y[:, 0])
        alg_12_mt_3 = torch.zeros_like(y[:, 0])
        alg_13_mt_4 = torch.zeros_like(y[:, 0])
        alg_14_et_3 = torch.zeros_like(y[:, 0])
        alg_15_et_4 = torch.zeros_like(y[:, 0])
        alg_16_E_3 = torch.zeros_like(y[:, 0])
        alg_17_E_4 = torch.zeros_like(y[:, 0])
        alg_18_Pi_3 = torch.zeros_like(y[:, 0])
        alg_19_Pi_4 = torch.zeros_like(y[:, 0])
        alg_20_Tao_3 = torch.zeros_like(y[:, 0])
        alg_21_Tao_4 = torch.zeros_like(y[:, 0])
        alg_22_Qo_3 = torch.zeros_like(y[:, 0])
        alg_23_Qo_4 = torch.zeros_like(y[:, 0])
        alg_24_Qo_5 = torch.zeros_like(y[:, 0])
        alg_25_Qo_6 = torch.zeros_like(y[:, 0])
        alg_26_Qo_7 = torch.zeros_like(y[:, 0])
        alg_27_Qo_8 = torch.zeros_like(y[:, 0])
        alg_28_Pi_5 = torch.zeros_like(y[:, 0])
        alg_29_Pi_6 = torch.zeros_like(y[:, 0])
        alg_30_Qo_9 = torch.zeros_like(y[:, 0])
        alg_31_Qo_10 = torch.zeros_like(y[:, 0])
        alg_32_Pi_7 = torch.zeros_like(y[:, 0])
        alg_33_Pi_8 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Qo = torch.zeros_like(y[:, 0]) + ((Pi - Pi_3 - Rsas * Qo) / Lsas)
        d_Pi_3 = torch.zeros_like(y[:, 0]) + ((Qo - Qo_2) / Csat)
        d_Qo_3 = torch.zeros_like(y[:, 0]) + ((Pi_2 - Pi_5 - Rpas * Qo_3) / Lpas)
        d_Pi_5 = torch.zeros_like(y[:, 0]) + ((Qo_3 - Qo_4) / Cpat)
        alg_0_mt = t - T * torch.floor(t / T)
        alg_2_et = torch.where((alg_0_mt >= 0.0) & (alg_0_mt <= Ts1 * T), 1.0 - torch.cos(3.14159 * alg_0_mt / (Ts1 * T)), torch.where((alg_0_mt > Ts1 * T) & (alg_0_mt <= Ts2 * T), 1.0 + torch.cos(3.14159 * (alg_0_mt - Ts1 * T) / ((Ts2 - Ts1) * T)), torch.where((alg_0_mt > Ts2 * T) & (alg_0_mt < T), 0.0, (1e309-1e309))))
        alg_4_E = ElvMin + alg_2_et * (ElvMax - ElvMin) / 2.0
        alg_6_Pi = PlvIni + alg_4_E * (V - VlvIni)
        alg_8_Tao = torch.where(alg_6_Pi >= Pi, 1.0, torch.where(alg_6_Pi < Pi, 0.0, (1e309-1e309)))
        alg_10_Qo = torch.where(alg_6_Pi >= Pi, CVao * alg_8_Tao * torch.abs(alg_6_Pi - Pi) ** 0.5, torch.where(alg_6_Pi < Pi, -1.0 * CVao * alg_8_Tao * torch.abs(Pi - alg_6_Pi) ** 0.5, (1e309-1e309)))
        d_Pi = torch.zeros_like(y[:, 0]) + ((alg_10_Qo - Qo) / Csas)
        alg_1_mt_2 = t - T_2 * torch.floor(t / T_2)
        alg_3_et_2 = torch.where((alg_1_mt_2 >= 0.0) & (alg_1_mt_2 <= Ts1_2 * T_2), 1.0 - torch.cos(3.14159 * alg_1_mt_2 / (Ts1_2 * T_2)), torch.where((alg_1_mt_2 > Ts1_2 * T_2) & (alg_1_mt_2 <= Ts2_2 * T_2), 1.0 + torch.cos(3.14159 * (alg_1_mt_2 - Ts1_2 * T_2) / ((Ts2_2 - Ts1_2) * T_2)), torch.where((alg_1_mt_2 > Ts2_2 * T_2) & (alg_1_mt_2 < T_2), 0.0, (1e309-1e309))))
        alg_5_E_2 = ErvMin_2 + alg_3_et_2 * (ErvMax_2 - ErvMin_2) / 2.0
        alg_7_Pi_2 = PrvIni_2 + alg_5_E_2 * (V_3 - VrvIni_2)
        alg_9_Tao_2 = torch.where(alg_7_Pi_2 >= Pi_2, 1.0, torch.where(alg_7_Pi_2 < Pi_2, 0.0, (1e309-1e309)))
        alg_11_Qo_2 = torch.where(alg_7_Pi_2 >= Pi_2, CVpa_2 * alg_9_Tao_2 * torch.abs(alg_7_Pi_2 - Pi_2) ** 0.5, torch.where(alg_7_Pi_2 < Pi_2, -1.0 * CVpa_2 * alg_9_Tao_2 * torch.abs(Pi_2 - alg_7_Pi_2) ** 0.5, (1e309-1e309)))
        d_Pi_2 = torch.zeros_like(y[:, 0]) + ((alg_11_Qo_2 - Qo_3) / Cpas)
        alg_12_mt_3 = t - T * torch.floor(t / T)
        alg_14_et_3 = torch.where((alg_12_mt_3 >= 0.0) & (alg_12_mt_3 <= (Tpwb + Tpww) * T - T), 1.0 - torch.cos(2.0 * 3.14159 * (alg_12_mt_3 - Tpwb * T + T) / (Tpww * T)), torch.where((alg_12_mt_3 > (Tpwb + Tpww) * T - T) & (alg_12_mt_3 <= Tpwb * T), 0.0, torch.where((alg_12_mt_3 > Tpwb * T) & (alg_12_mt_3 <= T), 1.0 - torch.cos(2.0 * 3.14159 * (alg_12_mt_3 - Tpwb * T) / (Tpww * T)), (1e309-1e309))))
        alg_16_E_3 = ElaMin + alg_14_et_3 * (ElaMax - ElaMin) / 2.0
        alg_18_Pi_3 = PlaIni + alg_16_E_3 * (V_2 - VlaIni)
        alg_20_Tao_3 = torch.where(alg_18_Pi_3 >= alg_6_Pi, 1.0, torch.where(alg_18_Pi_3 < alg_6_Pi, 0.0, (1e309-1e309)))
        alg_22_Qo_3 = torch.where(alg_18_Pi_3 >= alg_6_Pi, CVmi * alg_20_Tao_3 * torch.abs(alg_18_Pi_3 - alg_6_Pi) ** 0.5, torch.where(alg_18_Pi_3 < alg_6_Pi, -1.0 * CVmi * alg_20_Tao_3 * torch.abs(alg_6_Pi - alg_18_Pi_3) ** 0.5, (1e309-1e309)))
        d_V = torch.zeros_like(y[:, 0]) + (alg_22_Qo_3 - alg_10_Qo)
        alg_13_mt_4 = t - T_2 * torch.floor(t / T_2)
        alg_15_et_4 = torch.where((alg_13_mt_4 >= 0.0) & (alg_13_mt_4 <= (Tpwb_2 + Tpww_2) * T_2 - T_2), 1.0 - torch.cos(2.0 * 3.14159 * (alg_13_mt_4 - Tpwb_2 * T_2 + T_2) / (Tpww_2 * T_2)), torch.where((alg_13_mt_4 > (Tpwb_2 + Tpww_2) * T_2 - T_2) & (alg_13_mt_4 <= Tpwb_2 * T_2), 0.0, torch.where((alg_13_mt_4 > Tpwb_2 * T_2) & (alg_13_mt_4 <= T_2), 1.0 - torch.cos(2.0 * 3.14159 * (alg_13_mt_4 - Tpwb_2 * T_2) / (Tpww_2 * T_2)), (1e309-1e309))))
        alg_17_E_4 = EraMin_2 + alg_15_et_4 * (EraMax_2 - EraMin_2) / 2.0
        alg_19_Pi_4 = PraIni_2 + alg_17_E_4 * (V_4 - VraIni_2)
        alg_21_Tao_4 = torch.where(alg_19_Pi_4 >= alg_7_Pi_2, 1.0, torch.where(alg_19_Pi_4 < alg_7_Pi_2, 0.0, (1e309-1e309)))
        alg_23_Qo_4 = torch.where(alg_19_Pi_4 >= alg_7_Pi_2, CVti_2 * alg_21_Tao_4 * torch.abs(alg_19_Pi_4 - alg_7_Pi_2) ** 0.5, torch.where(alg_19_Pi_4 < alg_7_Pi_2, -1.0 * CVti_2 * alg_21_Tao_4 * torch.abs(alg_7_Pi_2 - alg_19_Pi_4) ** 0.5, (1e309-1e309)))
        d_V_3 = torch.zeros_like(y[:, 0]) + (alg_23_Qo_4 - alg_11_Qo_2)
        alg_30_Qo_9 = (Pi_6 - alg_18_Pi_3) / Rpvn
        d_V_2 = torch.zeros_like(y[:, 0]) + (alg_30_Qo_9 - alg_22_Qo_3)
        alg_31_Qo_10 = (Pi_4 - alg_19_Pi_4) / Rsvn
        d_V_4 = torch.zeros_like(y[:, 0]) + (alg_31_Qo_10 - alg_23_Qo_4)
        alg_25_Qo_6 = Qo_2
        alg_28_Pi_5 = Pi_4 + Rscp * alg_25_Qo_6
        alg_32_Pi_7 = alg_28_Pi_5 + Rsar * Qo_2
        d_Qo_2 = torch.zeros_like(y[:, 0]) + ((Pi_3 - alg_32_Pi_7 - Rsat * Qo_2) / Lsat)
        alg_27_Qo_8 = alg_25_Qo_6
        d_Pi_4 = torch.zeros_like(y[:, 0]) + ((alg_27_Qo_8 - alg_31_Qo_10) / Csvn)
        alg_24_Qo_5 = Qo_4
        alg_29_Pi_6 = Pi_6 + Rpcp * alg_24_Qo_5
        alg_33_Pi_8 = alg_29_Pi_6 + Rpar * Qo_4
        d_Qo_4 = torch.zeros_like(y[:, 0]) + ((Pi_5 - alg_33_Pi_8 - Rpat * Qo_4) / Lpat)
        alg_26_Qo_7 = alg_24_Qo_5
        d_Pi_6 = torch.zeros_like(y[:, 0]) + ((alg_26_Qo_7 - alg_30_Qo_9) / Cpvn)

        return torch.stack(
            [d_Pi, d_Pi_2, d_V, d_V_2, d_V_3, d_V_4, d_Pi_3, d_Qo, d_Qo_2, d_Pi_4, d_Pi_5, d_Qo_3, d_Qo_4, d_Pi_6],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
