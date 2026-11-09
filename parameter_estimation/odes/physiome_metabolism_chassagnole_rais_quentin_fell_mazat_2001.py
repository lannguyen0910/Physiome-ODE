import torch
import torch.nn as nn
import math


class MetabolismChassagnoleRaisQuentinFellMazat2001ODE(nn.Module):
    """
    PyTorch translation of metabolism/chassagnole_rais_quentin_fell_mazat_2001/model.py.
    Dynamic states: 12; independent constants: 43; derived constants used by rates: 0.
    Original constant slots: 43; algebraic slots: 6.
    """
    source_file = 'metabolism/chassagnole_rais_quentin_fell_mazat_2001/model.py'
    source_state_dim = 12
    source_constant_dim = 43
    algebraic_dim = 6
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 12
        self.parameter_dim = 43

        self.parameters = (
            parameters
            if parameters is not None
            else {'K_asp': 0.97, 'K_ATP': 0.98, 'K_aspp': 0.017, 'K_ADP': 0.25, 'K_iThr': 0.167, 'alpha': 2.47, 'h_Thr': 4.09, 'K_eq': 0.00064, 'V_max_AK_I': 463, 'K_aspp_2': 0.022, 'K_NADPH': 0.029, 'K_ASA': 0.11, 'K_NADP': 0.144, 'K_Pi': 10.2, 'K_eq_2': 284000, 'V_max_ASD': 598, 'K_ASA_2': 0.24, 'K_NADPH_2': 0.037, 'K_hs': 3.39, 'K_NADP_2': 0.067, 'K_iThr_2': 0.097, 'alpha_2': 3.93, 'h': 1.41, 'K_eq_3': 100000000000, 'V_max_HDH': 2585, 'K_hs_2': 0.11, 'K_ATP_2': 0.072, 'K_iThr_3': 1.09, 'K_iLys': 9.45, 'K_ihs': 4.7, 'K_iATP': 4.35, 'V_max_HK': 483, 'K_hsp': 0.31, 'V_max_TS': 208, 'K_asp_2': 0.32, 'K_ATP_3': 0.22, 'K_aspp_3': 0.017, 'K_ADP_2': 0.25, 'K_iLys_2': 0.391, 'h_Lys': 2.8, 'K_eq_4': 0.00064, 'V_max_AK_III': 299, 'constant_42': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Asp': 10, 'aspp': 0, 'ASA': 0, 'hs': 0, 'hsp': 0, 'Thr': 0, 'ATP': 30, 'ADP': 0, 'NADPH': 0, 'NADP': 0, 'Pi': 0, 'Lys': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['K_asp', 'K_ATP', 'K_aspp', 'K_ADP', 'K_iThr', 'alpha', 'h_Thr', 'K_eq', 'V_max_AK_I', 'K_aspp_2', 'K_NADPH', 'K_ASA', 'K_NADP', 'K_Pi', 'K_eq_2', 'V_max_ASD', 'K_ASA_2', 'K_NADPH_2', 'K_hs', 'K_NADP_2', 'K_iThr_2', 'alpha_2', 'h', 'K_eq_3', 'V_max_HDH', 'K_hs_2', 'K_ATP_2', 'K_iThr_3', 'K_iLys', 'K_ihs', 'K_iATP', 'V_max_HK', 'K_hsp', 'V_max_TS', 'K_asp_2', 'K_ATP_3', 'K_aspp_3', 'K_ADP_2', 'K_iLys_2', 'h_Lys', 'K_eq_4', 'V_max_AK_III', 'constant_42']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Asp', 'aspp', 'ASA', 'hs', 'hsp', 'Thr', 'ATP', 'ADP', 'NADPH', 'NADP', 'Pi', 'Lys']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (12)
        Asp = y[:, 0]
        aspp = y[:, 1]
        ASA = y[:, 2]
        hs = y[:, 3]
        hsp = y[:, 4]
        Thr = y[:, 5]
        ATP = y[:, 6]
        ADP = y[:, 7]
        NADPH = y[:, 8]
        NADP = y[:, 9]
        Pi = y[:, 10]
        Lys = y[:, 11]

        # Independent parameters (source constant order)
        K_asp = theta[:, 0]  # constants[0]
        K_ATP = theta[:, 1]  # constants[1]
        K_aspp = theta[:, 2]  # constants[2]
        K_ADP = theta[:, 3]  # constants[3]
        K_iThr = theta[:, 4]  # constants[4]
        alpha = theta[:, 5]  # constants[5]
        h_Thr = theta[:, 6]  # constants[6]
        K_eq = theta[:, 7]  # constants[7]
        V_max_AK_I = theta[:, 8]  # constants[8]
        K_aspp_2 = theta[:, 9]  # constants[9]
        K_NADPH = theta[:, 10]  # constants[10]
        K_ASA = theta[:, 11]  # constants[11]
        K_NADP = theta[:, 12]  # constants[12]
        K_Pi = theta[:, 13]  # constants[13]
        K_eq_2 = theta[:, 14]  # constants[14]
        V_max_ASD = theta[:, 15]  # constants[15]
        K_ASA_2 = theta[:, 16]  # constants[16]
        K_NADPH_2 = theta[:, 17]  # constants[17]
        K_hs = theta[:, 18]  # constants[18]
        K_NADP_2 = theta[:, 19]  # constants[19]
        K_iThr_2 = theta[:, 20]  # constants[20]
        alpha_2 = theta[:, 21]  # constants[21]
        h = theta[:, 22]  # constants[22]
        K_eq_3 = theta[:, 23]  # constants[23]
        V_max_HDH = theta[:, 24]  # constants[24]
        K_hs_2 = theta[:, 25]  # constants[25]
        K_ATP_2 = theta[:, 26]  # constants[26]
        K_iThr_3 = theta[:, 27]  # constants[27]
        K_iLys = theta[:, 28]  # constants[28]
        K_ihs = theta[:, 29]  # constants[29]
        K_iATP = theta[:, 30]  # constants[30]
        V_max_HK = theta[:, 31]  # constants[31]
        K_hsp = theta[:, 32]  # constants[32]
        V_max_TS = theta[:, 33]  # constants[33]
        K_asp_2 = theta[:, 34]  # constants[34]
        K_ATP_3 = theta[:, 35]  # constants[35]
        K_aspp_3 = theta[:, 36]  # constants[36]
        K_ADP_2 = theta[:, 37]  # constants[37]
        K_iLys_2 = theta[:, 38]  # constants[38]
        h_Lys = theta[:, 39]  # constants[39]
        K_eq_4 = theta[:, 40]  # constants[40]
        V_max_AK_III = theta[:, 41]  # constants[41]
        constant_42 = theta[:, 42]  # constants[42]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Asp = torch.zeros_like(y[:, 0])
        d_aspp = torch.zeros_like(y[:, 0])
        d_ASA = torch.zeros_like(y[:, 0])
        d_hs = torch.zeros_like(y[:, 0])
        d_hsp = torch.zeros_like(y[:, 0])
        d_Thr = torch.zeros_like(y[:, 0])
        d_ATP = torch.zeros_like(y[:, 0])
        d_ADP = torch.zeros_like(y[:, 0])
        d_NADPH = torch.zeros_like(y[:, 0])
        d_NADP = torch.zeros_like(y[:, 0])
        d_Pi = torch.zeros_like(y[:, 0])
        d_Lys = torch.zeros_like(y[:, 0])
        alg_0_vAKI = torch.zeros_like(y[:, 0])
        alg_1_vASD = torch.zeros_like(y[:, 0])
        alg_2_vHDH = torch.zeros_like(y[:, 0])
        alg_3_vHK = torch.zeros_like(y[:, 0])
        alg_4_vAKIII = torch.zeros_like(y[:, 0])
        alg_5_vTS = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Lys = torch.zeros_like(y[:, 0]) + (constant_42)
        alg_1_vASD = V_max_ASD * (aspp * NADPH - ASA * NADP * Pi / K_eq_2) / ((K_aspp_2 * (1.0 + ASA / K_ASA) * (1.0 + Pi / K_Pi) + aspp) * (K_NADPH * (1.0 + NADP / K_NADP) + NADPH))
        alg_2_vHDH = V_max_HDH * (ASA * NADPH - hs * NADP / K_eq_3) / ((1.0 + (Thr / K_iThr_2) ** h) / (1.0 + (Thr / (alpha_2 * K_iThr_2)) ** h) * (K_ASA_2 + hs * K_ASA_2 / K_hs + ASA) * (K_NADPH_2 * (1.0 + NADP / K_NADP_2) + NADPH))
        d_ASA = torch.zeros_like(y[:, 0]) + (alg_1_vASD - alg_2_vHDH)
        d_NADPH = torch.zeros_like(y[:, 0]) + (-alg_1_vASD - alg_2_vHDH)
        d_NADP = torch.zeros_like(y[:, 0]) + (alg_1_vASD + alg_2_vHDH)
        alg_3_vHK = V_max_HK * hs * ATP / ((K_hs_2 * (1.0 + ATP / K_iATP) / (1.0 + Thr / K_iThr_3) + hs) * (K_ATP_2 * (1.0 + hs / K_ihs) + ATP) * (1.0 + Lys / K_iLys))
        d_hs = torch.zeros_like(y[:, 0]) + (alg_2_vHDH - alg_3_vHK)
        alg_0_vAKI = V_max_AK_I * (Asp * ATP - aspp * ADP / K_eq) / ((K_asp * (1.0 + (Thr / K_iThr) ** h_Thr) / (1.0 + (Thr / (K_iThr * alpha)) ** h_Thr) + aspp * K_asp / K_aspp + Asp) * (K_ATP * (1.0 + ADP / K_ADP) + ATP))
        alg_4_vAKIII = V_max_AK_III * (Asp * ATP - aspp * ADP / K_eq_4) / ((1.0 + (Lys / K_iLys_2) ** h_Lys) * (K_asp_2 * (1.0 + aspp / K_aspp_3) + Asp) * (K_ATP_3 * (1.0 + ADP / K_ADP_2) + ATP))
        d_Asp = torch.zeros_like(y[:, 0]) + (-alg_0_vAKI - alg_4_vAKIII)
        d_aspp = torch.zeros_like(y[:, 0]) + (alg_0_vAKI + alg_4_vAKIII - alg_1_vASD)
        alg_5_vTS = V_max_TS * hsp / (K_hsp + hsp)
        d_hsp = torch.zeros_like(y[:, 0]) + (alg_3_vHK + alg_5_vTS)
        d_Thr = torch.zeros_like(y[:, 0]) + (alg_5_vTS)
        d_ATP = torch.zeros_like(y[:, 0]) + (-alg_0_vAKI - alg_4_vAKIII - alg_3_vHK)
        d_ADP = torch.zeros_like(y[:, 0]) + (alg_0_vAKI + alg_4_vAKIII + alg_3_vHK)
        d_Pi = torch.zeros_like(y[:, 0]) + (alg_1_vASD + alg_5_vTS)

        return torch.stack(
            [d_Asp, d_aspp, d_ASA, d_hs, d_hsp, d_Thr, d_ATP, d_ADP, d_NADPH, d_NADP, d_Pi, d_Lys],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
