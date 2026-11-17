import torch
import torch.nn as nn
import math


class SignalTransductionProctorSotiBoysGillespieShanleyWilkinsonKirkwood2005ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/proctor_soti_boys_gillespie_shanley_wilkinson_kirkwood_2005/model.py.
    Dynamic states: 14; independent constants: 22; derived constants used by rates: 2.
    Original constant slots: 24; algebraic slots: 20.
    """
    source_file = 'signal_transduction/proctor_soti_boys_gillespie_shanley_wilkinson_kirkwood_2005/model.py'
    source_state_dim = 14
    source_constant_dim = 24
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

        self.state_dim = 14
        self.parameter_dim = 22

        self.parameters = (
            parameters
            if parameters is not None
            else {'k1': 10, 'k2': 2e-05, 'k3': 50, 'k4': 1e-05, 'k5': 4e-06, 'k6': 6e-07, 'k7': 1e-07, 'k7_2': 1e-07, 'k8': 500, 'k9': 1, 'k10': 0.01, 'k11': 100, 'k12': 0.5, 'k13': 0.5, 'k14': 0.05, 'k15': 0.08, 'k16': 1000, 'k17': 8.02e-09, 'k18': 12, 'k19': 0.02, 'k20': 0.1, 'k21': 1e-05}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'NatP': 6000000, 'MisP': 0, 'MCom': 0, 'Hsp90': 300000, 'HCom': 5900, 'HSF1': 100, 'DiH': 0, 'TriH': 0, 'HSE_TriH': 0, 'HSE': 1, 'AggP': 0, 'ROS': 100, 'ATP': 10000, 'ADP': 1000}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k7_2', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15', 'k16', 'k17', 'k18', 'k19', 'k20', 'k21']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['NatP', 'MisP', 'MCom', 'Hsp90', 'HCom', 'HSF1', 'DiH', 'TriH', 'HSE_TriH', 'HSE', 'AggP', 'ROS', 'ATP', 'ADP']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (14)
        NatP = y[:, 0]
        MisP = y[:, 1]
        MCom = y[:, 2]
        Hsp90 = y[:, 3]
        HCom = y[:, 4]
        HSF1 = y[:, 5]
        DiH = y[:, 6]
        TriH = y[:, 7]
        HSE_TriH = y[:, 8]
        HSE = y[:, 9]
        AggP = y[:, 10]
        ROS = y[:, 11]
        ATP = y[:, 12]
        ADP = y[:, 13]

        # Independent parameters (source constant order)
        k1 = theta[:, 0]  # constants[0]
        k2 = theta[:, 1]  # constants[1]
        k3 = theta[:, 2]  # constants[2]
        k4 = theta[:, 3]  # constants[3]
        k5 = theta[:, 4]  # constants[4]
        k6 = theta[:, 5]  # constants[5]
        k7 = theta[:, 6]  # constants[6]
        k7_2 = theta[:, 7]  # constants[7]
        k8 = theta[:, 8]  # constants[8]
        k9 = theta[:, 9]  # constants[9]
        k10 = theta[:, 10]  # constants[10]
        k11 = theta[:, 11]  # constants[11]
        k12 = theta[:, 12]  # constants[12]
        k13 = theta[:, 13]  # constants[13]
        k14 = theta[:, 14]  # constants[14]
        k15 = theta[:, 15]  # constants[15]
        k16 = theta[:, 16]  # constants[16]
        k17 = theta[:, 17]  # constants[17]
        k18 = theta[:, 18]  # constants[18]
        k19 = theta[:, 19]  # constants[19]
        k20 = theta[:, 20]  # constants[20]
        k21 = theta[:, 21]  # constants[21]

        # Derived constants from the original initConsts()
        v1 = k1  # constants[22]
        v20 = k20  # constants[23]

        # Initialize rates and algebraic variables, as in computeRates()
        d_NatP = torch.zeros_like(y[:, 0])
        d_MisP = torch.zeros_like(y[:, 0])
        d_MCom = torch.zeros_like(y[:, 0])
        d_Hsp90 = torch.zeros_like(y[:, 0])
        d_HCom = torch.zeros_like(y[:, 0])
        d_HSF1 = torch.zeros_like(y[:, 0])
        d_DiH = torch.zeros_like(y[:, 0])
        d_TriH = torch.zeros_like(y[:, 0])
        d_HSE_TriH = torch.zeros_like(y[:, 0])
        d_HSE = torch.zeros_like(y[:, 0])
        d_AggP = torch.zeros_like(y[:, 0])
        d_ROS = torch.zeros_like(y[:, 0])
        d_ATP = torch.zeros_like(y[:, 0])
        d_ADP = torch.zeros_like(y[:, 0])
        alg_0_v2 = torch.zeros_like(y[:, 0])
        alg_1_v3 = torch.zeros_like(y[:, 0])
        alg_2_v21 = torch.zeros_like(y[:, 0])
        alg_3_v4 = torch.zeros_like(y[:, 0])
        alg_4_v5 = torch.zeros_like(y[:, 0])
        alg_5_v6 = torch.zeros_like(y[:, 0])
        alg_6_v7 = torch.zeros_like(y[:, 0])
        alg_7_v8 = torch.zeros_like(y[:, 0])
        alg_8_v9 = torch.zeros_like(y[:, 0])
        alg_9_v7b = torch.zeros_like(y[:, 0])
        alg_10_v10 = torch.zeros_like(y[:, 0])
        alg_11_v11 = torch.zeros_like(y[:, 0])
        alg_12_v12 = torch.zeros_like(y[:, 0])
        alg_13_v14 = torch.zeros_like(y[:, 0])
        alg_14_v13 = torch.zeros_like(y[:, 0])
        alg_15_v15 = torch.zeros_like(y[:, 0])
        alg_16_v16 = torch.zeros_like(y[:, 0])
        alg_17_v17 = torch.zeros_like(y[:, 0])
        alg_18_v18 = torch.zeros_like(y[:, 0])
        alg_19_v19 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_v2 = k2 * NatP * ROS
        alg_2_v21 = k21 * ROS
        d_ROS = torch.zeros_like(y[:, 0]) + (alg_0_v2 + v20 - (alg_0_v2 + alg_2_v21))
        alg_4_v5 = k5 * MCom * ATP
        d_NatP = torch.zeros_like(y[:, 0]) + (v1 + alg_4_v5 - alg_0_v2)
        alg_1_v3 = k3 * MisP * Hsp90
        alg_3_v4 = k4 * MCom
        d_MCom = torch.zeros_like(y[:, 0]) + (alg_1_v3 - (alg_3_v4 + alg_4_v5))
        alg_5_v6 = k6 * MisP * ATP
        alg_6_v7 = k7 * MisP * MisP
        d_MisP = torch.zeros_like(y[:, 0]) + (alg_0_v2 + alg_3_v4 - (alg_1_v3 + alg_5_v6 + 2.0 * alg_6_v7))
        alg_8_v9 = k9 * HCom
        alg_7_v8 = k8 * HSF1 * Hsp90
        d_HCom = torch.zeros_like(y[:, 0]) + (alg_7_v8 - alg_8_v9)
        alg_9_v7b = k7_2 * MisP * AggP
        d_AggP = torch.zeros_like(y[:, 0]) + (alg_6_v7 + 2.0 * alg_9_v7b - alg_9_v7b)
        alg_14_v13 = k13 * DiH
        alg_12_v12 = k12 * TriH
        alg_10_v10 = k10 * HSF1 * HSF1
        alg_11_v11 = k11 * DiH * HSF1
        d_HSF1 = torch.zeros_like(y[:, 0]) + (alg_8_v9 + 2.0 * alg_14_v13 + alg_12_v12 - (alg_7_v8 + 2.0 * alg_10_v10 + alg_11_v11))
        d_DiH = torch.zeros_like(y[:, 0]) + (alg_10_v10 + alg_12_v12 - (alg_11_v11 + alg_14_v13))
        alg_13_v14 = k14 * HSE * TriH
        alg_15_v15 = k15 * HSE_TriH
        d_TriH = torch.zeros_like(y[:, 0]) + (alg_11_v11 + alg_15_v15 - (alg_12_v12 + alg_13_v14))
        d_HSE = torch.zeros_like(y[:, 0]) + (alg_15_v15 - alg_13_v14)
        alg_16_v16 = k16 * HSE_TriH
        d_HSE_TriH = torch.zeros_like(y[:, 0]) + (alg_13_v14 + alg_16_v16 - (alg_15_v15 + alg_16_v16))
        alg_17_v17 = k17 * Hsp90 * ATP
        d_Hsp90 = torch.zeros_like(y[:, 0]) + (alg_4_v5 + alg_3_v4 + alg_8_v9 + alg_16_v16 - (alg_1_v3 + alg_7_v8 + alg_17_v17))
        alg_18_v18 = k18 * ADP
        alg_19_v19 = k19 * ATP
        d_ATP = torch.zeros_like(y[:, 0]) + (alg_18_v18 - (alg_4_v5 + alg_5_v6 + alg_17_v17 + alg_19_v19))
        d_ADP = torch.zeros_like(y[:, 0]) + (alg_4_v5 + alg_5_v6 + alg_17_v17 + alg_19_v19 - alg_18_v18)

        return torch.stack(
            [d_NatP, d_MisP, d_MCom, d_Hsp90, d_HCom, d_HSF1, d_DiH, d_TriH, d_HSE_TriH, d_HSE, d_AggP, d_ROS, d_ATP, d_ADP],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
