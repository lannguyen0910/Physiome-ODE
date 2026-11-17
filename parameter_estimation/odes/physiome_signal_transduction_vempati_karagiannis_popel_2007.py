import torch
import torch.nn as nn
import math


class SignalTransductionVempatiKaragiannisPopel2007ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/vempati_karagiannis_popel_2007/model.py.
    Dynamic states: 17; independent constants: 33; derived constants used by rates: 0.
    Original constant slots: 33; algebraic slots: 20.
    """
    source_file = 'signal_transduction/vempati_karagiannis_popel_2007/model.py'
    source_state_dim = 17
    source_constant_dim = 33
    algebraic_dim = 20
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
        self.parameter_dim = 33

        self.parameters = (
            parameters
            if parameters is not None
            else {'kon_M3pM9': 10000, 'koff_M3pM9': 0.001, 'kact_M3pM9': 0.0019, 'kon_M3T1': 1900000, 'koff_M3T1': 0.000126, 'kon_pM9T1': 34000, 'koff_pM9T1': 7.48e-05, 'kon_M3pM9T1': 10000, 'koff_M3pM9T1': 0.001, 'kact_M3pM9T1': 0.00057, 'kon_M9T1_star': 5600000, 'koff_M9T1_star': 1.6e-05, 'kon_M9T1': 0.00063, 'koff_M9T1': 0.00025, 'kon_M9_M3T1': 7.48e-05, 'koff_M9_M3T1': 34000, 'kon_M3_pM9T1': 1900000, 'koff_M3_pM9T1': 0.000126, 'kon_pM9T1M3': 34000, 'koff_pM9T1M3': 7.48e-05, 'kon_M3pM9T1M3': 10000, 'koff_M3pM9T1M3': 0.001, 'kact_M3pM9T1M3': 0.00057, 'kact_M9T1M3': 0.000126, 'kon_TrpM9': 10000, 'koff_TrpM9': 0.007, 'kact_TrpM9': 0.002, 'kon_TrpM9T1': 10000, 'koff_TrpM9T1': 0.007, 'kact_TrpM9T1': 0.0006, 'kon_TrpM9T1M3': 10000, 'koff_TrpM9T1M3': 0.007, 'kact_TrpM9T1M3': 0.0006}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'pM9': 3e-07, 'M3': 5e-08, 'M3_pM9': 0, 'M3_T1': 0, 'T1': 3e-07, 'pM9_T1': 0, 'M3_pM9_T1': 0, 'M9_T1_star': 0, 'M9': 0, 'M9_T1': 0, 'M9_T1_M3': 0, 'pM9_T1_M3': 0, 'M3_pM9_T1_M3': 0, 'Tr': 1e-07, 'Tr_pM9': 0, 'Tr_pM9_T1': 0, 'Tr_pM9_T1_M3': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['kon_M3pM9', 'koff_M3pM9', 'kact_M3pM9', 'kon_M3T1', 'koff_M3T1', 'kon_pM9T1', 'koff_pM9T1', 'kon_M3pM9T1', 'koff_M3pM9T1', 'kact_M3pM9T1', 'kon_M9T1_star', 'koff_M9T1_star', 'kon_M9T1', 'koff_M9T1', 'kon_M9_M3T1', 'koff_M9_M3T1', 'kon_M3_pM9T1', 'koff_M3_pM9T1', 'kon_pM9T1M3', 'koff_pM9T1M3', 'kon_M3pM9T1M3', 'koff_M3pM9T1M3', 'kact_M3pM9T1M3', 'kact_M9T1M3', 'kon_TrpM9', 'koff_TrpM9', 'kact_TrpM9', 'kon_TrpM9T1', 'koff_TrpM9T1', 'kact_TrpM9T1', 'kon_TrpM9T1M3', 'koff_TrpM9T1M3', 'kact_TrpM9T1M3']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['pM9', 'M3', 'M3_pM9', 'M3_T1', 'T1', 'pM9_T1', 'M3_pM9_T1', 'M9_T1_star', 'M9', 'M9_T1', 'M9_T1_M3', 'pM9_T1_M3', 'M3_pM9_T1_M3', 'Tr', 'Tr_pM9', 'Tr_pM9_T1', 'Tr_pM9_T1_M3']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (17)
        pM9 = y[:, 0]
        M3 = y[:, 1]
        M3_pM9 = y[:, 2]
        M3_T1 = y[:, 3]
        T1 = y[:, 4]
        pM9_T1 = y[:, 5]
        M3_pM9_T1 = y[:, 6]
        M9_T1_star = y[:, 7]
        M9 = y[:, 8]
        M9_T1 = y[:, 9]
        M9_T1_M3 = y[:, 10]
        pM9_T1_M3 = y[:, 11]
        M3_pM9_T1_M3 = y[:, 12]
        Tr = y[:, 13]
        Tr_pM9 = y[:, 14]
        Tr_pM9_T1 = y[:, 15]
        Tr_pM9_T1_M3 = y[:, 16]

        # Independent parameters (source constant order)
        kon_M3pM9 = theta[:, 0]  # constants[0]
        koff_M3pM9 = theta[:, 1]  # constants[1]
        kact_M3pM9 = theta[:, 2]  # constants[2]
        kon_M3T1 = theta[:, 3]  # constants[3]
        koff_M3T1 = theta[:, 4]  # constants[4]
        kon_pM9T1 = theta[:, 5]  # constants[5]
        koff_pM9T1 = theta[:, 6]  # constants[6]
        kon_M3pM9T1 = theta[:, 7]  # constants[7]
        koff_M3pM9T1 = theta[:, 8]  # constants[8]
        kact_M3pM9T1 = theta[:, 9]  # constants[9]
        kon_M9T1_star = theta[:, 10]  # constants[10]
        koff_M9T1_star = theta[:, 11]  # constants[11]
        kon_M9T1 = theta[:, 12]  # constants[12]
        koff_M9T1 = theta[:, 13]  # constants[13]
        kon_M9_M3T1 = theta[:, 14]  # constants[14]
        koff_M9_M3T1 = theta[:, 15]  # constants[15]
        kon_M3_pM9T1 = theta[:, 16]  # constants[16]
        koff_M3_pM9T1 = theta[:, 17]  # constants[17]
        kon_pM9T1M3 = theta[:, 18]  # constants[18]
        koff_pM9T1M3 = theta[:, 19]  # constants[19]
        kon_M3pM9T1M3 = theta[:, 20]  # constants[20]
        koff_M3pM9T1M3 = theta[:, 21]  # constants[21]
        kact_M3pM9T1M3 = theta[:, 22]  # constants[22]
        kact_M9T1M3 = theta[:, 23]  # constants[23]
        kon_TrpM9 = theta[:, 24]  # constants[24]
        koff_TrpM9 = theta[:, 25]  # constants[25]
        kact_TrpM9 = theta[:, 26]  # constants[26]
        kon_TrpM9T1 = theta[:, 27]  # constants[27]
        koff_TrpM9T1 = theta[:, 28]  # constants[28]
        kact_TrpM9T1 = theta[:, 29]  # constants[29]
        kon_TrpM9T1M3 = theta[:, 30]  # constants[30]
        koff_TrpM9T1M3 = theta[:, 31]  # constants[31]
        kact_TrpM9T1M3 = theta[:, 32]  # constants[32]

        # Initialize rates and algebraic variables, as in computeRates()
        d_pM9 = torch.zeros_like(y[:, 0])
        d_M3 = torch.zeros_like(y[:, 0])
        d_M3_pM9 = torch.zeros_like(y[:, 0])
        d_M3_T1 = torch.zeros_like(y[:, 0])
        d_T1 = torch.zeros_like(y[:, 0])
        d_pM9_T1 = torch.zeros_like(y[:, 0])
        d_M3_pM9_T1 = torch.zeros_like(y[:, 0])
        d_M9_T1_star = torch.zeros_like(y[:, 0])
        d_M9 = torch.zeros_like(y[:, 0])
        d_M9_T1 = torch.zeros_like(y[:, 0])
        d_M9_T1_M3 = torch.zeros_like(y[:, 0])
        d_pM9_T1_M3 = torch.zeros_like(y[:, 0])
        d_M3_pM9_T1_M3 = torch.zeros_like(y[:, 0])
        d_Tr = torch.zeros_like(y[:, 0])
        d_Tr_pM9 = torch.zeros_like(y[:, 0])
        d_Tr_pM9_T1 = torch.zeros_like(y[:, 0])
        d_Tr_pM9_T1_M3 = torch.zeros_like(y[:, 0])
        alg_0_v1 = torch.zeros_like(y[:, 0])
        alg_1_v2 = torch.zeros_like(y[:, 0])
        alg_2_v3 = torch.zeros_like(y[:, 0])
        alg_3_v4 = torch.zeros_like(y[:, 0])
        alg_4_v5 = torch.zeros_like(y[:, 0])
        alg_5_v6 = torch.zeros_like(y[:, 0])
        alg_6_v7 = torch.zeros_like(y[:, 0])
        alg_7_v8 = torch.zeros_like(y[:, 0])
        alg_8_v9 = torch.zeros_like(y[:, 0])
        alg_9_v10 = torch.zeros_like(y[:, 0])
        alg_10_v11 = torch.zeros_like(y[:, 0])
        alg_11_v12 = torch.zeros_like(y[:, 0])
        alg_12_v13 = torch.zeros_like(y[:, 0])
        alg_13_v14 = torch.zeros_like(y[:, 0])
        alg_14_v15 = torch.zeros_like(y[:, 0])
        alg_15_v16 = torch.zeros_like(y[:, 0])
        alg_16_v17 = torch.zeros_like(y[:, 0])
        alg_17_v18 = torch.zeros_like(y[:, 0])
        alg_18_v19 = torch.zeros_like(y[:, 0])
        alg_19_v20 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_v1 = kon_M3pM9 * M3 * pM9 - koff_M3pM9 * M3_pM9
        alg_1_v2 = kact_M3pM9 * M3_pM9
        d_M3_pM9 = torch.zeros_like(y[:, 0]) + (alg_0_v1 - alg_1_v2)
        alg_4_v5 = kon_M3pM9T1 * pM9_T1 * M3 - koff_M3pM9T1 * M3_pM9_T1
        alg_5_v6 = kact_M3pM9T1 * M3_pM9_T1
        d_M3_pM9_T1 = torch.zeros_like(y[:, 0]) + (alg_4_v5 - alg_5_v6)
        alg_2_v3 = kon_M3T1 * M3 * T1 - koff_M3T1 * M3_T1
        alg_3_v4 = kon_pM9T1 * pM9 * T1 - koff_pM9T1 * pM9_T1
        alg_6_v7 = kon_M9T1_star * T1 * M9 - koff_M9T1_star * M9_T1_star
        d_T1 = torch.zeros_like(y[:, 0]) + (-(alg_2_v3 + alg_3_v4 + alg_6_v7))
        alg_8_v9 = kon_M9_M3T1 * M9_T1_M3 - koff_M9_M3T1 * M9 * M3_T1
        alg_10_v11 = kon_pM9T1M3 * pM9 * M3_T1 - koff_pM9T1M3 * pM9_T1_M3
        d_M3_T1 = torch.zeros_like(y[:, 0]) + (alg_2_v3 + alg_8_v9 - alg_10_v11)
        alg_11_v12 = kon_M3pM9T1M3 * M3 * pM9_T1_M3 - koff_M3pM9T1M3 * M3_pM9_T1_M3
        alg_12_v13 = kact_M3pM9T1M3 * M3_pM9_T1_M3
        d_M3_pM9_T1_M3 = torch.zeros_like(y[:, 0]) + (alg_11_v12 - alg_12_v13)
        alg_9_v10 = kon_M3_pM9T1 * pM9_T1 * M3 - koff_M3_pM9T1 * pM9_T1_M3
        alg_13_v14 = kact_M9T1M3 * M9_T1_M3
        d_M3 = torch.zeros_like(y[:, 0]) + (alg_1_v2 + alg_5_v6 + alg_12_v13 + alg_13_v14 - (alg_0_v1 + alg_2_v3 + alg_4_v5 + alg_9_v10 + alg_11_v12))
        alg_7_v8 = kon_M9T1 * M9_T1_star - koff_M9T1 * M9_T1
        d_M9_T1_star = torch.zeros_like(y[:, 0]) + (alg_6_v7 + alg_13_v14 - alg_7_v8)
        alg_14_v15 = kon_TrpM9 * Tr * pM9 - koff_TrpM9 * Tr_pM9
        d_pM9 = torch.zeros_like(y[:, 0]) + (-(alg_0_v1 + alg_3_v4 + alg_14_v15 + alg_10_v11))
        alg_15_v16 = kact_TrpM9 * Tr_pM9
        d_M9 = torch.zeros_like(y[:, 0]) + (alg_1_v2 + alg_8_v9 + alg_15_v16 - alg_6_v7)
        d_Tr_pM9 = torch.zeros_like(y[:, 0]) + (alg_14_v15 - alg_15_v16)
        alg_16_v17 = kon_TrpM9T1 * pM9_T1 * Tr - koff_TrpM9T1 * Tr_pM9_T1
        d_pM9_T1 = torch.zeros_like(y[:, 0]) + (alg_3_v4 - (alg_4_v5 + alg_9_v10 + alg_16_v17))
        alg_17_v18 = kact_TrpM9T1 * Tr_pM9_T1
        d_M9_T1 = torch.zeros_like(y[:, 0]) + (alg_7_v8 + alg_17_v18)
        d_Tr_pM9_T1 = torch.zeros_like(y[:, 0]) + (alg_16_v17 - alg_17_v18)
        alg_18_v19 = kon_TrpM9T1M3 * Tr * pM9_T1_M3 - koff_TrpM9T1M3 * Tr_pM9_T1_M3
        d_pM9_T1_M3 = torch.zeros_like(y[:, 0]) + (alg_9_v10 + alg_10_v11 - (alg_11_v12 + alg_18_v19))
        alg_19_v20 = kact_TrpM9T1M3 * Tr_pM9_T1_M3
        d_M9_T1_M3 = torch.zeros_like(y[:, 0]) + (alg_12_v13 + alg_19_v20 - (alg_13_v14 + alg_8_v9))
        d_Tr = torch.zeros_like(y[:, 0]) + (alg_15_v16 + alg_17_v18 + alg_19_v20 - (alg_14_v15 + alg_16_v17 + alg_18_v19))
        d_Tr_pM9_T1_M3 = torch.zeros_like(y[:, 0]) + (alg_18_v19 - alg_19_v20)

        return torch.stack(
            [d_pM9, d_M3, d_M3_pM9, d_M3_T1, d_T1, d_pM9_T1, d_M3_pM9_T1, d_M9_T1_star, d_M9, d_M9_T1, d_M9_T1_M3, d_pM9_T1_M3, d_M3_pM9_T1_M3, d_Tr, d_Tr_pM9, d_Tr_pM9_T1, d_Tr_pM9_T1_M3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
