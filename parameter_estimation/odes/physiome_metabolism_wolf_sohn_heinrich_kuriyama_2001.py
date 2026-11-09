import torch
import torch.nn as nn
import math


class MetabolismWolfSohnHeinrichKuriyama2001ODE(nn.Module):
    """
    PyTorch translation of metabolism/wolf_sohn_heinrich_kuriyama_2001/model.py.
    Dynamic states: 13; independent constants: 28; derived constants used by rates: 0.
    Original constant slots: 28; algebraic slots: 23.
    """
    source_file = 'metabolism/wolf_sohn_heinrich_kuriyama_2001/model.py'
    source_state_dim = 13
    source_constant_dim = 28
    algebraic_dim = 23
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 13
        self.parameter_dim = 28

        self.parameters = (
            parameters
            if parameters is not None
            else {'S': 2, 'v13': 4, 'v10': 80, 'A_c': 2, 'A_m': 2, 'N': 2, 'v0': 1.6, 'n': 4, 'Kc': 0.1, 'k2': 0.2, 'k3': 0.2, 'k4': 0.2, 'k5': 0.1, 'k6': 0.12, 'k7': 10, 'k8': 10, 'k9': 10, 'alpha': 0.1, 'm': 4, 'KH': 0.5, 'k11': 10, 'KA': 1, 'k12': 5, 'k14': 10, 'k15': 5, 'k16': 10, 'k17': 0.02, 'k18': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'S1': 1.892468811, 'sul': 0.012871557, 'aps': 0.053366179, 'pap': 0.082791522, 'hyd': 0.159317029, 'cys': 0.805117739, 'eth': 10.25246999, 'aco': 1.139360412, 'oxy': 6.146549084, 'A3_c': 1.570847601, 'A3_m': 1.835158493, 'N2': 1.933616149, 'oah': 5.571685096}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['S', 'v13', 'v10', 'A_c', 'A_m', 'N', 'v0', 'n', 'Kc', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'alpha', 'm', 'KH', 'k11', 'KA', 'k12', 'k14', 'k15', 'k16', 'k17', 'k18']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['S1', 'sul', 'aps', 'pap', 'hyd', 'cys', 'eth', 'aco', 'oxy', 'A3_c', 'A3_m', 'N2', 'oah']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (13)
        S1 = y[:, 0]
        sul = y[:, 1]
        aps = y[:, 2]
        pap = y[:, 3]
        hyd = y[:, 4]
        cys = y[:, 5]
        eth = y[:, 6]
        aco = y[:, 7]
        oxy = y[:, 8]
        A3_c = y[:, 9]
        A3_m = y[:, 10]
        N2 = y[:, 11]
        oah = y[:, 12]

        # Independent parameters (source constant order)
        S = theta[:, 0]  # constants[0]
        v13 = theta[:, 1]  # constants[1]
        v10 = theta[:, 2]  # constants[2]
        A_c = theta[:, 3]  # constants[3]
        A_m = theta[:, 4]  # constants[4]
        N = theta[:, 5]  # constants[5]
        v0 = theta[:, 6]  # constants[6]
        n = theta[:, 7]  # constants[7]
        Kc = theta[:, 8]  # constants[8]
        k2 = theta[:, 9]  # constants[9]
        k3 = theta[:, 10]  # constants[10]
        k4 = theta[:, 11]  # constants[11]
        k5 = theta[:, 12]  # constants[12]
        k6 = theta[:, 13]  # constants[13]
        k7 = theta[:, 14]  # constants[14]
        k8 = theta[:, 15]  # constants[15]
        k9 = theta[:, 16]  # constants[16]
        alpha = theta[:, 17]  # constants[17]
        m = theta[:, 18]  # constants[18]
        KH = theta[:, 19]  # constants[19]
        k11 = theta[:, 20]  # constants[20]
        KA = theta[:, 21]  # constants[21]
        k12 = theta[:, 22]  # constants[22]
        k14 = theta[:, 23]  # constants[23]
        k15 = theta[:, 24]  # constants[24]
        k16 = theta[:, 25]  # constants[25]
        k17 = theta[:, 26]  # constants[26]
        k18 = theta[:, 27]  # constants[27]

        # Initialize rates and algebraic variables, as in computeRates()
        d_S1 = torch.zeros_like(y[:, 0])
        d_sul = torch.zeros_like(y[:, 0])
        d_aps = torch.zeros_like(y[:, 0])
        d_pap = torch.zeros_like(y[:, 0])
        d_hyd = torch.zeros_like(y[:, 0])
        d_cys = torch.zeros_like(y[:, 0])
        d_eth = torch.zeros_like(y[:, 0])
        d_aco = torch.zeros_like(y[:, 0])
        d_oxy = torch.zeros_like(y[:, 0])
        d_A3_c = torch.zeros_like(y[:, 0])
        d_A3_m = torch.zeros_like(y[:, 0])
        d_N2 = torch.zeros_like(y[:, 0])
        d_oah = torch.zeros_like(y[:, 0])
        alg_0_S2 = torch.zeros_like(y[:, 0])
        alg_1_A2_c = torch.zeros_like(y[:, 0])
        alg_2_A2_m = torch.zeros_like(y[:, 0])
        alg_3_N1 = torch.zeros_like(y[:, 0])
        alg_4_f1 = torch.zeros_like(y[:, 0])
        alg_5_v1 = torch.zeros_like(y[:, 0])
        alg_6_v2 = torch.zeros_like(y[:, 0])
        alg_7_v3 = torch.zeros_like(y[:, 0])
        alg_8_v4 = torch.zeros_like(y[:, 0])
        alg_9_v5 = torch.zeros_like(y[:, 0])
        alg_10_v7 = torch.zeros_like(y[:, 0])
        alg_11_v17 = torch.zeros_like(y[:, 0])
        alg_12_v6 = torch.zeros_like(y[:, 0])
        alg_13_v8 = torch.zeros_like(y[:, 0])
        alg_14_v15 = torch.zeros_like(y[:, 0])
        alg_15_v9 = torch.zeros_like(y[:, 0])
        alg_16_f2 = torch.zeros_like(y[:, 0])
        alg_17_v18 = torch.zeros_like(y[:, 0])
        alg_18_v11A = torch.zeros_like(y[:, 0])
        alg_19_v14 = torch.zeros_like(y[:, 0])
        alg_20_v11B = torch.zeros_like(y[:, 0])
        alg_21_v12 = torch.zeros_like(y[:, 0])
        alg_22_v16 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_4_f1 = (1.0 + (cys / Kc) ** n) ** (-1.0)
        alg_5_v1 = v0 * alg_4_f1
        alg_6_v2 = k2 * sul * A3_c
        d_sul = torch.zeros_like(y[:, 0]) + (alg_5_v1 - alg_6_v2)
        alg_7_v3 = k3 * aps * A3_c
        d_aps = torch.zeros_like(y[:, 0]) + (alg_6_v2 - alg_7_v3)
        alg_8_v4 = k4 * pap * N2
        d_pap = torch.zeros_like(y[:, 0]) + (alg_7_v3 - alg_8_v4)
        alg_9_v5 = k5 * hyd * oah
        alg_11_v17 = k17 * hyd
        d_hyd = torch.zeros_like(y[:, 0]) + (alg_8_v4 - (alg_9_v5 + alg_11_v17))
        alg_12_v6 = k6 * cys
        d_cys = torch.zeros_like(y[:, 0]) + (alg_9_v5 - alg_12_v6)
        alg_3_N1 = N - N2
        alg_10_v7 = k7 * eth * alg_3_N1
        d_eth = torch.zeros_like(y[:, 0]) + (v13 - alg_10_v7)
        alg_0_S2 = S - S1
        alg_13_v8 = k8 * aco * alg_0_S2
        alg_15_v9 = k9 * S1 * alg_3_N1
        d_S1 = torch.zeros_like(y[:, 0]) + (alg_13_v8 - alg_15_v9)
        alg_14_v15 = k15 * aco
        d_aco = torch.zeros_like(y[:, 0]) + (alg_10_v7 - (alg_13_v8 + alg_14_v15))
        alg_17_v18 = k18 * oah
        d_oah = torch.zeros_like(y[:, 0]) + (alg_14_v15 - (alg_9_v5 + alg_17_v18))
        alg_16_f2 = (1.0 + (hyd / KH) ** m) ** (-1.0)
        alg_18_v11A = k11 * N2 * oxy * alg_16_f2 / (alpha * N2 + oxy)
        d_N2 = torch.zeros_like(y[:, 0]) + (2.0 * alg_10_v7 + 4.0 * alg_15_v9 - (3.0 * alg_8_v4 + alg_18_v11A))
        alg_19_v14 = k14 * oxy
        d_oxy = torch.zeros_like(y[:, 0]) + (v10 - (alg_18_v11A + alg_19_v14))
        alg_21_v12 = k12 * A3_c
        alg_1_A2_c = A_c - A3_c
        alg_22_v16 = k16 * A3_m * alg_1_A2_c
        d_A3_c = torch.zeros_like(y[:, 0]) + (alg_22_v16 - (alg_6_v2 + alg_7_v3 + alg_21_v12))
        alg_2_A2_m = A_m - A3_m
        alg_20_v11B = 3.0 * alg_18_v11A * alg_2_A2_m / (KA + alg_2_A2_m)
        d_A3_m = torch.zeros_like(y[:, 0]) + (alg_20_v11B - alg_22_v16)

        return torch.stack(
            [d_S1, d_sul, d_aps, d_pap, d_hyd, d_cys, d_eth, d_aco, d_oxy, d_A3_c, d_A3_m, d_N2, d_oah],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
