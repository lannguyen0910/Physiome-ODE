import torch
import torch.nn as nn
import math


class CellCycleNovakTyson1993BODE(nn.Module):
    """
    PyTorch translation of cell_cycle/novak_tyson_1993_b/model.py.
    Dynamic states: 9; independent constants: 31; derived constants used by rates: 0.
    Original constant slots: 31; algebraic slots: 5.
    """
    source_file = 'cell_cycle/novak_tyson_1993_b/model.py'
    source_state_dim = 9
    source_constant_dim = 31
    algebraic_dim = 5
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 9
        self.parameter_dim = 31

        self.parameters = (
            parameters
            if parameters is not None
            else {'k1AA': 1, 'k3': 0.01, 'kinh': 0.025, 'kcak': 0.25, 'kbPPase': 0.125, 'kfPPase': 0.1, 'khPPase': 0.087, 'kd_antiIE': 0.095, 'ka': 0.01, 'kc': 0.1, 'ke': 0.0133, 'kg': 0.0065, 'V2': 0.015, 'V2_2': 1, 'V25': 0.1, 'V25_2': 2, 'Vwee': 0.1, 'Vwee_2': 1, 'K_a': 0.1, 'K_b': 0.1, 'K_c': 0.01, 'K_d': 0.01, 'K_e': 0.3, 'K_f': 0.3, 'K_g': 0.01, 'K_h': 0.01, 'Cdc25_T': 1, 'Wee1_T': 1, 'UbE_T': 1, 'Cdc2_T': 100, 'IE_T': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Cdc25_P': 0, 'Wee1_P': 0, 'UbE': 0, 'cyclin': 1, 'cyclin_Cdc2': 0, 'MPF_active': 0, 'Tyr15P_dimer': 0, 'MPF_pre': 0, 'IE_P': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k1AA', 'k3', 'kinh', 'kcak', 'kbPPase', 'kfPPase', 'khPPase', 'kd_antiIE', 'ka', 'kc', 'ke', 'kg', 'V2', 'V2_2', 'V25', 'V25_2', 'Vwee', 'Vwee_2', 'K_a', 'K_b', 'K_c', 'K_d', 'K_e', 'K_f', 'K_g', 'K_h', 'Cdc25_T', 'Wee1_T', 'UbE_T', 'Cdc2_T', 'IE_T']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Cdc25_P', 'Wee1_P', 'UbE', 'cyclin', 'cyclin_Cdc2', 'MPF_active', 'Tyr15P_dimer', 'MPF_pre', 'IE_P']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (9)
        Cdc25_P = y[:, 0]
        Wee1_P = y[:, 1]
        UbE = y[:, 2]
        cyclin = y[:, 3]
        cyclin_Cdc2 = y[:, 4]
        MPF_active = y[:, 5]
        Tyr15P_dimer = y[:, 6]
        MPF_pre = y[:, 7]
        IE_P = y[:, 8]

        # Independent parameters (source constant order)
        k1AA = theta[:, 0]  # constants[0]
        k3 = theta[:, 1]  # constants[1]
        kinh = theta[:, 2]  # constants[2]
        kcak = theta[:, 3]  # constants[3]
        kbPPase = theta[:, 4]  # constants[4]
        kfPPase = theta[:, 5]  # constants[5]
        khPPase = theta[:, 6]  # constants[6]
        kd_antiIE = theta[:, 7]  # constants[7]
        ka = theta[:, 8]  # constants[8]
        kc = theta[:, 9]  # constants[9]
        ke = theta[:, 10]  # constants[10]
        kg = theta[:, 11]  # constants[11]
        V2 = theta[:, 12]  # constants[12]
        V2_2 = theta[:, 13]  # constants[13]
        V25 = theta[:, 14]  # constants[14]
        V25_2 = theta[:, 15]  # constants[15]
        Vwee = theta[:, 16]  # constants[16]
        Vwee_2 = theta[:, 17]  # constants[17]
        K_a = theta[:, 18]  # constants[18]
        K_b = theta[:, 19]  # constants[19]
        K_c = theta[:, 20]  # constants[20]
        K_d = theta[:, 21]  # constants[21]
        K_e = theta[:, 22]  # constants[22]
        K_f = theta[:, 23]  # constants[23]
        K_g = theta[:, 24]  # constants[24]
        K_h = theta[:, 25]  # constants[25]
        Cdc25_T = theta[:, 26]  # constants[26]
        Wee1_T = theta[:, 27]  # constants[27]
        UbE_T = theta[:, 28]  # constants[28]
        Cdc2_T = theta[:, 29]  # constants[29]
        IE_T = theta[:, 30]  # constants[30]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Cdc25_P = torch.zeros_like(y[:, 0])
        d_Wee1_P = torch.zeros_like(y[:, 0])
        d_UbE = torch.zeros_like(y[:, 0])
        d_cyclin = torch.zeros_like(y[:, 0])
        d_cyclin_Cdc2 = torch.zeros_like(y[:, 0])
        d_MPF_active = torch.zeros_like(y[:, 0])
        d_Tyr15P_dimer = torch.zeros_like(y[:, 0])
        d_MPF_pre = torch.zeros_like(y[:, 0])
        d_IE_P = torch.zeros_like(y[:, 0])
        alg_0_k25 = torch.zeros_like(y[:, 0])
        alg_2_kwee = torch.zeros_like(y[:, 0])
        alg_3_k2 = torch.zeros_like(y[:, 0])
        alg_4_Cdc2 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Cdc25_P = torch.zeros_like(y[:, 0]) + (ka * MPF_active * (Cdc25_T - Cdc25_P) / (K_a + Cdc25_T - Cdc25_P) - kbPPase * Cdc25_P / (K_b + Cdc25_P))
        d_Wee1_P = torch.zeros_like(y[:, 0]) + (ke * MPF_active * (Wee1_T - Wee1_P) / (K_e + Wee1_T - Wee1_P) - kfPPase * Wee1_P / (K_f + Wee1_P))
        d_IE_P = torch.zeros_like(y[:, 0]) + (kg * MPF_active * (IE_T - IE_P) / (K_g + IE_T - IE_P) - khPPase * IE_P / (K_h + IE_P))
        d_UbE = torch.zeros_like(y[:, 0]) + (kc * IE_P * (UbE_T - UbE) / (K_c + UbE_T - UbE) - kd_antiIE * UbE / (K_d + UbE))
        alg_0_k25 = V25 * (Cdc25_T - Cdc25_P) + V25_2 * Cdc25_P
        alg_2_kwee = Vwee * Wee1_P + Vwee_2 * (Wee1_T - Wee1_P)
        alg_3_k2 = V2 * (UbE_T - UbE) + V2_2 * UbE
        d_Tyr15P_dimer = torch.zeros_like(y[:, 0]) + (alg_2_kwee * cyclin_Cdc2 - (alg_0_k25 + kcak + alg_3_k2) * Tyr15P_dimer + kinh * MPF_pre)
        d_MPF_pre = torch.zeros_like(y[:, 0]) + (alg_2_kwee * MPF_active - (kinh + alg_0_k25 + alg_3_k2) * MPF_pre + kcak * Tyr15P_dimer)
        d_MPF_active = torch.zeros_like(y[:, 0]) + (kcak * cyclin_Cdc2 - (kinh + alg_2_kwee + alg_3_k2) * MPF_active + alg_0_k25 * MPF_pre)
        alg_4_Cdc2 = Cdc2_T - (cyclin_Cdc2 + MPF_active + MPF_pre + Tyr15P_dimer)
        d_cyclin = torch.zeros_like(y[:, 0]) + (k1AA - alg_3_k2 * cyclin - k3 * cyclin * alg_4_Cdc2)
        d_cyclin_Cdc2 = torch.zeros_like(y[:, 0]) + (kinh * MPF_active - (alg_2_kwee + kcak + alg_3_k2) * cyclin_Cdc2 + alg_0_k25 * Tyr15P_dimer + k3 * cyclin * alg_4_Cdc2)

        return torch.stack(
            [d_Cdc25_P, d_Wee1_P, d_UbE, d_cyclin, d_cyclin_Cdc2, d_MPF_active, d_Tyr15P_dimer, d_MPF_pre, d_IE_P],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
