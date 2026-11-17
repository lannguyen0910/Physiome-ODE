import torch
import torch.nn as nn
import math


class CellCycleCalzoneThieffryTysonNovak2007ODE(nn.Module):
    """
    PyTorch translation of cell_cycle/calzone_thieffry_tyson_novak_2007/model.py.
    Dynamic states: 17; independent constants: 43; derived constants used by rates: 0.
    Original constant slots: 43; algebraic slots: 4.
    """
    source_file = 'cell_cycle/calzone_thieffry_tyson_novak_2007/model.py'
    source_state_dim = 17
    source_constant_dim = 43
    algebraic_dim = 4
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 17
        self.parameter_dim = 43

        self.parameters = (
            parameters
            if parameters is not None
            else {'k_in': 0.15, 'k_out': 0, 'k_dn': 0.01, 'k_dn_2': 1.5, 'k_wee': 0.005, 'k_wee_2': 1, 'k_stg': 0.2, 'k_stg_2': 2, 'k_sc': 0.01, 'k_dc': 0.01, 'epsilon': 7e-05, 'j_aie': 0.01, 'j_iie': 0.01, 'k_aie': 1, 'k_iie': 0.4, 'j_afz': 0.01, 'j_ifz': 0.01, 'k_afz': 1, 'k_ifz': 0.2, 'k_dm': 0.002, 'k_dm_2': 0.2, 'j_m': 0.05, 'k_sxm': 0.0005, 'k_sxp': 0.001, 'k_ins': 0.08, 'k_outs': 0.02, 'k_astg': 0, 'k_astg_2': 1, 'k_istg': 0.3, 'k_dstg': 0.015, 'j_astg': 0.05, 'j_istg': 0.05, 'k_sstg': 0.02, 'k_inw': 0.04, 'k_outw': 0.01, 'k_awee': 0.3, 'k_iwee': 0.01, 'k_iwee_2': 1, 'j_awee': 0.05, 'j_iwee': 0.05, 'Wee1_T': 0.8, 'k_ez': 0.5, 'constant_42': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'MPF_n': 0, 'MPF_c': 1, 'FZY': 0, 'Wee1_n': 0, 'StgP_n': 0, 'preMPF_n': 0, 'preMPF_c': 0, 'N': 1, 'StgP_c': 0, 'Wee1_c': 0, 'IE': 0, 'Stg_m': 1, 'Xp': 0, 'Xm': 0, 'Stg_n': 0, 'Stg_c': 1, 'Wee1P_n': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k_in', 'k_out', 'k_dn', 'k_dn_2', 'k_wee', 'k_wee_2', 'k_stg', 'k_stg_2', 'k_sc', 'k_dc', 'epsilon', 'j_aie', 'j_iie', 'k_aie', 'k_iie', 'j_afz', 'j_ifz', 'k_afz', 'k_ifz', 'k_dm', 'k_dm_2', 'j_m', 'k_sxm', 'k_sxp', 'k_ins', 'k_outs', 'k_astg', 'k_astg_2', 'k_istg', 'k_dstg', 'j_astg', 'j_istg', 'k_sstg', 'k_inw', 'k_outw', 'k_awee', 'k_iwee', 'k_iwee_2', 'j_awee', 'j_iwee', 'Wee1_T', 'k_ez', 'constant_42']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['MPF_n', 'MPF_c', 'FZY', 'Wee1_n', 'StgP_n', 'preMPF_n', 'preMPF_c', 'N', 'StgP_c', 'Wee1_c', 'IE', 'Stg_m', 'Xp', 'Xm', 'Stg_n', 'Stg_c', 'Wee1P_n']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (17)
        MPF_n = y[:, 0]
        MPF_c = y[:, 1]
        FZY = y[:, 2]
        Wee1_n = y[:, 3]
        StgP_n = y[:, 4]
        preMPF_n = y[:, 5]
        preMPF_c = y[:, 6]
        N = y[:, 7]
        StgP_c = y[:, 8]
        Wee1_c = y[:, 9]
        IE = y[:, 10]
        Stg_m = y[:, 11]
        Xp = y[:, 12]
        Xm = y[:, 13]
        Stg_n = y[:, 14]
        Stg_c = y[:, 15]
        Wee1P_n = y[:, 16]

        # Independent parameters (source constant order)
        k_in = theta[:, 0]  # constants[0]
        k_out = theta[:, 1]  # constants[1]
        k_dn = theta[:, 2]  # constants[2]
        k_dn_2 = theta[:, 3]  # constants[3]
        k_wee = theta[:, 4]  # constants[4]
        k_wee_2 = theta[:, 5]  # constants[5]
        k_stg = theta[:, 6]  # constants[6]
        k_stg_2 = theta[:, 7]  # constants[7]
        k_sc = theta[:, 8]  # constants[8]
        k_dc = theta[:, 9]  # constants[9]
        epsilon = theta[:, 10]  # constants[10]
        j_aie = theta[:, 11]  # constants[11]
        j_iie = theta[:, 12]  # constants[12]
        k_aie = theta[:, 13]  # constants[13]
        k_iie = theta[:, 14]  # constants[14]
        j_afz = theta[:, 15]  # constants[15]
        j_ifz = theta[:, 16]  # constants[16]
        k_afz = theta[:, 17]  # constants[17]
        k_ifz = theta[:, 18]  # constants[18]
        k_dm = theta[:, 19]  # constants[19]
        k_dm_2 = theta[:, 20]  # constants[20]
        j_m = theta[:, 21]  # constants[21]
        k_sxm = theta[:, 22]  # constants[22]
        k_sxp = theta[:, 23]  # constants[23]
        k_ins = theta[:, 24]  # constants[24]
        k_outs = theta[:, 25]  # constants[25]
        k_astg = theta[:, 26]  # constants[26]
        k_astg_2 = theta[:, 27]  # constants[27]
        k_istg = theta[:, 28]  # constants[28]
        k_dstg = theta[:, 29]  # constants[29]
        j_astg = theta[:, 30]  # constants[30]
        j_istg = theta[:, 31]  # constants[31]
        k_sstg = theta[:, 32]  # constants[32]
        k_inw = theta[:, 33]  # constants[33]
        k_outw = theta[:, 34]  # constants[34]
        k_awee = theta[:, 35]  # constants[35]
        k_iwee = theta[:, 36]  # constants[36]
        k_iwee_2 = theta[:, 37]  # constants[37]
        j_awee = theta[:, 38]  # constants[38]
        j_iwee = theta[:, 39]  # constants[39]
        Wee1_T = theta[:, 40]  # constants[40]
        k_ez = theta[:, 41]  # constants[41]
        constant_42 = theta[:, 42]  # constants[42]

        # Initialize rates and algebraic variables, as in computeRates()
        d_MPF_n = torch.zeros_like(y[:, 0])
        d_MPF_c = torch.zeros_like(y[:, 0])
        d_FZY = torch.zeros_like(y[:, 0])
        d_Wee1_n = torch.zeros_like(y[:, 0])
        d_StgP_n = torch.zeros_like(y[:, 0])
        d_preMPF_n = torch.zeros_like(y[:, 0])
        d_preMPF_c = torch.zeros_like(y[:, 0])
        d_N = torch.zeros_like(y[:, 0])
        d_StgP_c = torch.zeros_like(y[:, 0])
        d_Wee1_c = torch.zeros_like(y[:, 0])
        d_IE = torch.zeros_like(y[:, 0])
        d_Stg_m = torch.zeros_like(y[:, 0])
        d_Xp = torch.zeros_like(y[:, 0])
        d_Xm = torch.zeros_like(y[:, 0])
        d_Stg_n = torch.zeros_like(y[:, 0])
        d_Stg_c = torch.zeros_like(y[:, 0])
        d_Wee1P_n = torch.zeros_like(y[:, 0])
        alg_0_Wee1P_c = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_N = torch.zeros_like(y[:, 0]) + (constant_42)
        d_MPF_n = torch.zeros_like(y[:, 0]) + (k_in * MPF_c - k_out * MPF_n - (k_dn + k_dn_2 * FZY) * MPF_n - (k_wee + k_wee_2 * Wee1_n) * MPF_n + (k_stg + k_stg_2 * StgP_n) * preMPF_n)
        d_preMPF_n = torch.zeros_like(y[:, 0]) + (k_in * preMPF_c - k_out * preMPF_n - (k_dn + k_dn_2 * FZY) * preMPF_n + (k_wee + k_wee_2 * Wee1_n) * MPF_n - (k_wee + k_stg_2 * StgP_n) * preMPF_n)
        d_MPF_c = torch.zeros_like(y[:, 0]) + (k_sc - epsilon * N * (k_in * MPF_c - k_out * MPF_n) / (1.0 - N * epsilon) - k_dc * MPF_c + (k_stg + k_stg_2 * StgP_c) * preMPF_c - (k_wee + k_wee_2 * Wee1_c) * MPF_c)
        d_preMPF_c = torch.zeros_like(y[:, 0]) + (-epsilon * N * (k_in * preMPF_c - k_out * preMPF_n) / (1.0 - N * epsilon) - k_dc * preMPF_c - (k_stg + k_stg_2 * StgP_c) * preMPF_c + (k_wee + k_wee_2 * Wee1_c) * MPF_c)
        d_IE = torch.zeros_like(y[:, 0]) + (k_aie * (1.0 - IE) * MPF_n / (j_aie + 1.0 - IE) - k_iie * IE / (j_iie + IE))
        d_FZY = torch.zeros_like(y[:, 0]) + (k_afz * IE * (1.0 - FZY) / (j_afz + 1.0 - FZY) - k_ifz * FZY / (j_ifz + FZY))
        d_Stg_m = torch.zeros_like(y[:, 0]) + (-(k_dm / (j_m + Stg_m) + k_dm_2 * Xp) * Stg_m)
        d_Xm = torch.zeros_like(y[:, 0]) + (k_sxm * N)
        d_Xp = torch.zeros_like(y[:, 0]) + (k_sxp * Xm)
        d_StgP_n = torch.zeros_like(y[:, 0]) + (k_ins * StgP_c - k_outs * StgP_n + (k_astg + k_astg_2 * MPF_n) * Stg_n / (j_astg + Stg_n) - k_istg * StgP_n / (j_istg + StgP_n) - k_dstg * StgP_n)
        d_Stg_n = torch.zeros_like(y[:, 0]) + (k_ins * Stg_c - k_outs * Stg_n - (k_astg + k_astg_2 * MPF_n) * Stg_n / (j_astg + Stg_n) + k_istg * StgP_n / (j_istg + StgP_n) - k_dstg * Stg_n)
        d_StgP_c = torch.zeros_like(y[:, 0]) + (-k_dstg * StgP_c - epsilon * N * (k_ins * StgP_c - k_outs * StgP_n) / (1.0 - N * epsilon) + (k_astg + k_astg_2 * MPF_c) * Stg_c / (j_astg + Stg_c) - k_istg * StgP_c / (j_istg + StgP_c))
        d_Stg_c = torch.zeros_like(y[:, 0]) + (k_sstg * Stg_m - k_dstg * Stg_c - epsilon * N * (k_ins * Stg_c - k_outs * Stg_n) / (1.0 - N * epsilon) - (k_astg + k_astg_2 * MPF_c) * Stg_c / (j_astg + Stg_c) + k_istg * StgP_c / (j_istg + StgP_c))
        d_Wee1_n = torch.zeros_like(y[:, 0]) + (k_inw * Wee1_c - k_outw * Wee1_n + k_awee * Wee1P_n / (j_awee + Wee1P_n) - (k_iwee + k_iwee_2 * MPF_n) * Wee1_n / (j_iwee + Wee1_n))
        alg_0_Wee1P_c = (Wee1_T - N * epsilon * (Wee1_n + Wee1P_n)) / (1.0 - N * epsilon) - Wee1_c
        d_Wee1_c = torch.zeros_like(y[:, 0]) + (-(k_inw * Wee1_c - k_outw * Wee1_n) * N * epsilon / (1.0 - N * epsilon) + k_awee * alg_0_Wee1P_c / (j_awee + alg_0_Wee1P_c) - (k_iwee + k_iwee_2 * MPF_c) * Wee1_c / (j_iwee + Wee1_c))
        d_Wee1P_n = torch.zeros_like(y[:, 0]) + (k_inw * alg_0_Wee1P_c - k_outw * Wee1P_n - k_awee * Wee1P_n / (j_awee + Wee1P_n) + (k_iwee + k_iwee_2 * MPF_n) * Wee1_n / (j_iwee + Wee1_n))

        return torch.stack(
            [d_MPF_n, d_MPF_c, d_FZY, d_Wee1_n, d_StgP_n, d_preMPF_n, d_preMPF_c, d_N, d_StgP_c, d_Wee1_c, d_IE, d_Stg_m, d_Xp, d_Xm, d_Stg_n, d_Stg_c, d_Wee1P_n],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
