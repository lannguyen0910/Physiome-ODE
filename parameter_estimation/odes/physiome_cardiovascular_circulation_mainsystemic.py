import torch
import torch.nn as nn
import math


class CardiovascularCirculationMainSystemicODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/MainSystemic/model.py.
    Dynamic states: 7; independent constants: 32; derived constants used by rates: 0.
    Original constant slots: 32; algebraic slots: 17.
    """
    source_file = 'cardiovascular_circulation/MainSystemic/model.py'
    source_state_dim = 7
    source_constant_dim = 32
    algebraic_dim = 17
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
        self.parameter_dim = 32

        self.parameters = (
            parameters
            if parameters is not None
            else {'CVao': 350, 'PlvIni': 1, 'VlvIni': 5, 'Vlv0': 500, 'CVmi': 400, 'PlaIni': 1, 'VlaIni': 4, 'Vla0': 20, 'ElvMax': 2.5, 'ElvMin': 0.1, 'T': 1, 'Ts1': 0.3, 'Ts2': 0.45, 'ElaMax': 0.25, 'ElaMin': 0.15, 'Tpwb': 0.92, 'Tpww': 0.09, 'Rsas': 0.003, 'Csas': 0.08, 'Lsas': 6.2e-05, 'P0sas': 100, 'Q0sas': 0, 'Rsat': 0.05, 'Csat': 1.6, 'Lsat': 0.0017, 'P0sat': 100, 'Q0sat': 0, 'Rsar': 0.5, 'Rscp': 0.52, 'Rsvn': 0.075, 'Csvn': 20.5, 'P0svn': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Pi': 100, 'V': 500, 'V_2': 20, 'Pi_2': 100, 'Qo': 0, 'Qo_2': 0, 'Pi_3': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['CVao', 'PlvIni', 'VlvIni', 'Vlv0', 'CVmi', 'PlaIni', 'VlaIni', 'Vla0', 'ElvMax', 'ElvMin', 'T', 'Ts1', 'Ts2', 'ElaMax', 'ElaMin', 'Tpwb', 'Tpww', 'Rsas', 'Csas', 'Lsas', 'P0sas', 'Q0sas', 'Rsat', 'Csat', 'Lsat', 'P0sat', 'Q0sat', 'Rsar', 'Rscp', 'Rsvn', 'Csvn', 'P0svn']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Pi', 'V', 'V_2', 'Pi_2', 'Qo', 'Qo_2', 'Pi_3']
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
        Pi_2 = y[:, 3]
        Qo = y[:, 4]
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

        # Initialize rates and algebraic variables, as in computeRates()
        d_Pi = torch.zeros_like(y[:, 0])
        d_V = torch.zeros_like(y[:, 0])
        d_V_2 = torch.zeros_like(y[:, 0])
        d_Pi_2 = torch.zeros_like(y[:, 0])
        d_Qo = torch.zeros_like(y[:, 0])
        d_Qo_2 = torch.zeros_like(y[:, 0])
        d_Pi_3 = torch.zeros_like(y[:, 0])
        alg_0_mt = torch.zeros_like(y[:, 0])
        alg_1_et = torch.zeros_like(y[:, 0])
        alg_2_E = torch.zeros_like(y[:, 0])
        alg_3_Pi = torch.zeros_like(y[:, 0])
        alg_4_Tao = torch.zeros_like(y[:, 0])
        alg_5_Qo = torch.zeros_like(y[:, 0])
        alg_6_mt_2 = torch.zeros_like(y[:, 0])
        alg_7_et_2 = torch.zeros_like(y[:, 0])
        alg_8_E_2 = torch.zeros_like(y[:, 0])
        alg_9_Pi_2 = torch.zeros_like(y[:, 0])
        alg_10_Tao_2 = torch.zeros_like(y[:, 0])
        alg_11_Qo_2 = torch.zeros_like(y[:, 0])
        alg_12_Qo_3 = torch.zeros_like(y[:, 0])
        alg_13_Qo_4 = torch.zeros_like(y[:, 0])
        alg_14_Pi_3 = torch.zeros_like(y[:, 0])
        alg_15_Qo_5 = torch.zeros_like(y[:, 0])
        alg_16_Pi_4 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Qo = torch.zeros_like(y[:, 0]) + ((Pi - Pi_2 - Rsas * Qo) / Lsas)
        d_Pi_2 = torch.zeros_like(y[:, 0]) + ((Qo - Qo_2) / Csat)
        alg_0_mt = t - T * torch.floor(t / T)
        alg_1_et = torch.where((alg_0_mt >= 0.0) & (alg_0_mt <= Ts1 * T), 1.0 - torch.cos(3.14159 * alg_0_mt / (Ts1 * T)), torch.where((alg_0_mt > Ts1 * T) & (alg_0_mt <= Ts2 * T), 1.0 + torch.cos(3.14159 * (alg_0_mt - Ts1 * T) / ((Ts2 - Ts1) * T)), torch.where((alg_0_mt > Ts2 * T) & (alg_0_mt < T), 0.0, (1e309-1e309))))
        alg_2_E = ElvMin + alg_1_et * (ElvMax - ElvMin) / 2.0
        alg_3_Pi = PlvIni + alg_2_E * (V - VlvIni)
        alg_4_Tao = torch.where(alg_3_Pi >= Pi, 1.0, torch.where(alg_3_Pi < Pi, 0.0, (1e309-1e309)))
        alg_5_Qo = torch.where(alg_3_Pi >= Pi, CVao * alg_4_Tao * torch.abs(alg_3_Pi - Pi) ** 0.5, torch.where(alg_3_Pi < Pi, -1.0 * CVao * alg_4_Tao * torch.abs(Pi - alg_3_Pi) ** 0.5, (1e309-1e309)))
        d_Pi = torch.zeros_like(y[:, 0]) + ((alg_5_Qo - Qo) / Csas)
        alg_6_mt_2 = t - T * torch.floor(t / T)
        alg_7_et_2 = torch.where((alg_6_mt_2 >= 0.0) & (alg_6_mt_2 <= (Tpwb + Tpww - 1.0) * T), 1.0 - torch.cos(2.0 * 3.14159 * (alg_6_mt_2 - (Tpwb - 1.0) * T) / (Tpww * T)), torch.where((alg_6_mt_2 > (Tpwb + Tpww - 1.0) * T) & (alg_6_mt_2 <= Tpwb * T), 0.0, torch.where((alg_6_mt_2 > Tpwb * T) & (alg_6_mt_2 <= T), 1.0 - torch.cos(2.0 * 3.14159 * (alg_6_mt_2 - Tpwb * T) / (Tpww * T)), (1e309-1e309))))
        alg_8_E_2 = ElaMin + alg_7_et_2 * (ElaMax - ElaMin) / 2.0
        alg_9_Pi_2 = PlaIni + alg_8_E_2 * (V_2 - VlaIni)
        alg_10_Tao_2 = torch.where(alg_9_Pi_2 >= alg_3_Pi, 1.0, torch.where(alg_9_Pi_2 < alg_3_Pi, 0.0, (1e309-1e309)))
        alg_11_Qo_2 = torch.where(alg_9_Pi_2 >= alg_3_Pi, CVmi * alg_10_Tao_2 * torch.abs(alg_9_Pi_2 - alg_3_Pi) ** 0.5, torch.where(alg_9_Pi_2 < alg_3_Pi, -1.0 * CVmi * alg_10_Tao_2 * torch.abs(alg_3_Pi - alg_9_Pi_2) ** 0.5, (1e309-1e309)))
        d_V = torch.zeros_like(y[:, 0]) + (alg_11_Qo_2 - alg_5_Qo)
        alg_15_Qo_5 = (Pi_3 - alg_9_Pi_2) / Rsvn
        d_V_2 = torch.zeros_like(y[:, 0]) + (alg_15_Qo_5 - alg_11_Qo_2)
        alg_12_Qo_3 = Qo_2
        alg_14_Pi_3 = Pi_3 + Rscp * alg_12_Qo_3
        alg_16_Pi_4 = alg_14_Pi_3 + Rsar * Qo_2
        d_Qo_2 = torch.zeros_like(y[:, 0]) + ((Pi_2 - alg_16_Pi_4 - Rsat * Qo_2) / Lsat)
        alg_13_Qo_4 = alg_12_Qo_3
        d_Pi_3 = torch.zeros_like(y[:, 0]) + ((alg_13_Qo_4 - alg_15_Qo_5) / Csvn)

        return torch.stack(
            [d_Pi, d_V, d_V_2, d_Pi_2, d_Qo, d_Qo_2, d_Pi_3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
