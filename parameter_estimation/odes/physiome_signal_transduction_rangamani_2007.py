import torch
import torch.nn as nn
import math


class SignalTransductionRangamani2007ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/rangamani_2007/model.py.
    Dynamic states: 32; independent constants: 32; derived constants used by rates: 0.
    Original constant slots: 32; algebraic slots: 21.
    """
    source_file = 'signal_transduction/rangamani_2007/model.py'
    source_state_dim = 32
    source_constant_dim = 32
    algebraic_dim = 21
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 32
        self.parameter_dim = 32

        self.parameters = (
            parameters
            if parameters is not None
            else {'k1': 0.000185, 'k2': 1.25e-06, 'k3': 0.000185, 'k4': 1.25e-06, 'k5': 0.000185, 'k6': 1.25e-06, 'k7': 0.000185, 'k8': 1.25e-06, 'k9': 0.000185, 'k10': 1.25e-06, 'k110': 0.00037, 'k120': 1.4e-05, 'k130': 1.25e-06, 'k140': 0.00037, 'p': 0.00175, 'k15': 0.000185, 'k16': 1.25e-06, 'k17': 0.00037, 'k18': 0.0005, 'k19': 0.0002, 'k20': 0.0001, 'k21': 0.0001, 'k22': 6e-05, 'k23': 0.1, 'k24': 0.000185, 'k25': 1.25e-06, 'k26': 0.00037, 'k28': 0.0005, 'k29': 0.75, 'd_6': 0.000283, 'k_14': 41.6, 'k_m1': 1000000}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'R_NOS2': 0.616, 'c1': 0, 'c2': 100, 'c3': 0, 'c4': 150, 'c5': 0, 'c6': 100, 'c7': 0, 'c8': 100, 'c9': 0, 'c10': 100, 'c11': 0, 'c12': 0, 'c13': 250, 'c14': 0, 'c15': 0, 'c16': 0, 'c17': 100, 'c18': 0, 'c19': 0, 'c20': 80, 'c21': 0, 'c22': 0, 'c23': 200, 'c24': 0, 'c25': 0, 'c26': 0, 'c27': 0, 'c28': 0, 'c29': 800, 'c30': 0, 'c31': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k110', 'k120', 'k130', 'k140', 'p', 'k15', 'k16', 'k17', 'k18', 'k19', 'k20', 'k21', 'k22', 'k23', 'k24', 'k25', 'k26', 'k28', 'k29', 'd_6', 'k_14', 'k_m1']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['R_NOS2', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26', 'c27', 'c28', 'c29', 'c30', 'c31']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (32)
        R_NOS2 = y[:, 0]
        c1 = y[:, 1]
        c2 = y[:, 2]
        c3 = y[:, 3]
        c4 = y[:, 4]
        c5 = y[:, 5]
        c6 = y[:, 6]
        c7 = y[:, 7]
        c8 = y[:, 8]
        c9 = y[:, 9]
        c10 = y[:, 10]
        c11 = y[:, 11]
        c12 = y[:, 12]
        c13 = y[:, 13]
        c14 = y[:, 14]
        c15 = y[:, 15]
        c16 = y[:, 16]
        c17 = y[:, 17]
        c18 = y[:, 18]
        c19 = y[:, 19]
        c20 = y[:, 20]
        c21 = y[:, 21]
        c22 = y[:, 22]
        c23 = y[:, 23]
        c24 = y[:, 24]
        c25 = y[:, 25]
        c26 = y[:, 26]
        c27 = y[:, 27]
        c28 = y[:, 28]
        c29 = y[:, 29]
        c30 = y[:, 30]
        c31 = y[:, 31]

        # Independent parameters (source constant order)
        k1 = theta[:, 0]  # constants[0]
        k2 = theta[:, 1]  # constants[1]
        k3 = theta[:, 2]  # constants[2]
        k4 = theta[:, 3]  # constants[3]
        k5 = theta[:, 4]  # constants[4]
        k6 = theta[:, 5]  # constants[5]
        k7 = theta[:, 6]  # constants[6]
        k8 = theta[:, 7]  # constants[7]
        k9 = theta[:, 8]  # constants[8]
        k10 = theta[:, 9]  # constants[9]
        k110 = theta[:, 10]  # constants[10]
        k120 = theta[:, 11]  # constants[11]
        k130 = theta[:, 12]  # constants[12]
        k140 = theta[:, 13]  # constants[13]
        p = theta[:, 14]  # constants[14]
        k15 = theta[:, 15]  # constants[15]
        k16 = theta[:, 16]  # constants[16]
        k17 = theta[:, 17]  # constants[17]
        k18 = theta[:, 18]  # constants[18]
        k19 = theta[:, 19]  # constants[19]
        k20 = theta[:, 20]  # constants[20]
        k21 = theta[:, 21]  # constants[21]
        k22 = theta[:, 22]  # constants[22]
        k23 = theta[:, 23]  # constants[23]
        k24 = theta[:, 24]  # constants[24]
        k25 = theta[:, 25]  # constants[25]
        k26 = theta[:, 26]  # constants[26]
        k28 = theta[:, 27]  # constants[27]
        k29 = theta[:, 28]  # constants[28]
        d_6 = theta[:, 29]  # constants[29]
        k_14 = theta[:, 30]  # constants[30]
        k_m1 = theta[:, 31]  # constants[31]

        # Initialize rates and algebraic variables, as in computeRates()
        d_R_NOS2 = torch.zeros_like(y[:, 0])
        d_c1 = torch.zeros_like(y[:, 0])
        d_c2 = torch.zeros_like(y[:, 0])
        d_c3 = torch.zeros_like(y[:, 0])
        d_c4 = torch.zeros_like(y[:, 0])
        d_c5 = torch.zeros_like(y[:, 0])
        d_c6 = torch.zeros_like(y[:, 0])
        d_c7 = torch.zeros_like(y[:, 0])
        d_c8 = torch.zeros_like(y[:, 0])
        d_c9 = torch.zeros_like(y[:, 0])
        d_c10 = torch.zeros_like(y[:, 0])
        d_c11 = torch.zeros_like(y[:, 0])
        d_c12 = torch.zeros_like(y[:, 0])
        d_c13 = torch.zeros_like(y[:, 0])
        d_c14 = torch.zeros_like(y[:, 0])
        d_c15 = torch.zeros_like(y[:, 0])
        d_c16 = torch.zeros_like(y[:, 0])
        d_c17 = torch.zeros_like(y[:, 0])
        d_c18 = torch.zeros_like(y[:, 0])
        d_c19 = torch.zeros_like(y[:, 0])
        d_c20 = torch.zeros_like(y[:, 0])
        d_c21 = torch.zeros_like(y[:, 0])
        d_c22 = torch.zeros_like(y[:, 0])
        d_c23 = torch.zeros_like(y[:, 0])
        d_c24 = torch.zeros_like(y[:, 0])
        d_c25 = torch.zeros_like(y[:, 0])
        d_c26 = torch.zeros_like(y[:, 0])
        d_c27 = torch.zeros_like(y[:, 0])
        d_c28 = torch.zeros_like(y[:, 0])
        d_c29 = torch.zeros_like(y[:, 0])
        d_c30 = torch.zeros_like(y[:, 0])
        d_c31 = torch.zeros_like(y[:, 0])
        alg_0_J1 = torch.zeros_like(y[:, 0])
        alg_1_v14 = torch.zeros_like(y[:, 0])
        alg_2_J2 = torch.zeros_like(y[:, 0])
        alg_3_deg_no = torch.zeros_like(y[:, 0])
        alg_4_J3 = torch.zeros_like(y[:, 0])
        alg_5_J4 = torch.zeros_like(y[:, 0])
        alg_6_J5 = torch.zeros_like(y[:, 0])
        alg_7_J6 = torch.zeros_like(y[:, 0])
        alg_8_J7 = torch.zeros_like(y[:, 0])
        alg_9_J8 = torch.zeros_like(y[:, 0])
        alg_10_J9 = torch.zeros_like(y[:, 0])
        alg_11_J10 = torch.zeros_like(y[:, 0])
        alg_12_J11 = torch.zeros_like(y[:, 0])
        alg_13_J12 = torch.zeros_like(y[:, 0])
        alg_14_J13 = torch.zeros_like(y[:, 0])
        alg_15_J14 = torch.zeros_like(y[:, 0])
        alg_16_J15 = torch.zeros_like(y[:, 0])
        alg_17_J16 = torch.zeros_like(y[:, 0])
        alg_18_J17 = torch.zeros_like(y[:, 0])
        alg_19_J18 = torch.zeros_like(y[:, 0])
        alg_20_J19 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_J1 = k1 * c1 * c2 - k2 * c3
        d_c1 = torch.zeros_like(y[:, 0]) + (-alg_0_J1)
        alg_1_v14 = k_14 * c16 ** 2.0 / (k_m1 + c16 ** 2.0)
        alg_3_deg_no = d_6 * R_NOS2
        d_R_NOS2 = torch.zeros_like(y[:, 0]) + (alg_1_v14 - alg_3_deg_no)
        alg_2_J2 = k3 * c3 * c4 - k4 * c5
        d_c3 = torch.zeros_like(y[:, 0]) + (alg_0_J1 - alg_2_J2)
        alg_4_J3 = k5 * c5 * c6 - k6 * c7
        d_c5 = torch.zeros_like(y[:, 0]) + (alg_2_J2 - alg_4_J3)
        alg_5_J4 = k7 * c7 * c8 - k8 * c9
        d_c7 = torch.zeros_like(y[:, 0]) + (alg_4_J3 - alg_5_J4)
        alg_6_J5 = k9 * c10 * c9 - k10 * c11
        alg_7_J6 = k110 * c11
        d_c11 = torch.zeros_like(y[:, 0]) + (alg_6_J5 - alg_7_J6)
        alg_8_J7 = k120 * c12 * c13 - k130 * c14
        d_c12 = torch.zeros_like(y[:, 0]) + (-alg_8_J7 + alg_7_J6)
        alg_9_J8 = k140 * c14
        d_c10 = torch.zeros_like(y[:, 0]) + (-alg_6_J5 + alg_9_J8)
        d_c14 = torch.zeros_like(y[:, 0]) + (alg_8_J7 - alg_9_J8)
        d_c15 = torch.zeros_like(y[:, 0]) + (alg_9_J8)
        alg_10_J9 = k15 * c9 * c17 - k16 * c18
        d_c9 = torch.zeros_like(y[:, 0]) + (alg_5_J4 - alg_6_J5 - alg_10_J9)
        alg_11_J10 = k17 * c18
        d_c2 = torch.zeros_like(y[:, 0]) + (-alg_0_J1 + alg_11_J10 + alg_7_J6)
        d_c18 = torch.zeros_like(y[:, 0]) + (alg_10_J9 - alg_11_J10)
        alg_12_J11 = k18 * c19 * c20 - k19 * c21
        d_c19 = torch.zeros_like(y[:, 0]) + (alg_11_J10 - alg_12_J11)
        d_c20 = torch.zeros_like(y[:, 0]) + (-alg_12_J11)
        alg_13_J12 = k20 * c21
        d_c4 = torch.zeros_like(y[:, 0]) + (-alg_2_J2 + alg_7_J6 + alg_13_J12)
        d_c6 = torch.zeros_like(y[:, 0]) + (-alg_4_J3 + alg_7_J6 + alg_13_J12)
        d_c8 = torch.zeros_like(y[:, 0]) + (-alg_5_J4 + alg_7_J6 + alg_13_J12)
        d_c17 = torch.zeros_like(y[:, 0]) + (-alg_10_J9 + alg_13_J12)
        d_c21 = torch.zeros_like(y[:, 0]) + (alg_12_J11 - alg_13_J12)
        alg_14_J13 = k21 * c22 * c23 - k22 * c24
        d_c23 = torch.zeros_like(y[:, 0]) + (-alg_14_J13)
        alg_15_J14 = k23 * c24
        d_c22 = torch.zeros_like(y[:, 0]) + (alg_13_J12 - alg_14_J13 + alg_15_J14)
        d_c24 = torch.zeros_like(y[:, 0]) + (alg_14_J13 - alg_15_J14)
        alg_16_J15 = k24 * c25 * c29 - k25 * c30
        d_c29 = torch.zeros_like(y[:, 0]) + (-alg_16_J15)
        alg_17_J16 = k26 * c30
        d_c26 = torch.zeros_like(y[:, 0]) + (alg_17_J16)
        d_c30 = torch.zeros_like(y[:, 0]) + (alg_16_J15 - alg_17_J16)
        alg_20_J19 = k29 * c16 * c31
        d_c13 = torch.zeros_like(y[:, 0]) + (-alg_8_J7 + alg_20_J19)
        d_c16 = torch.zeros_like(y[:, 0]) + (alg_9_J8 - alg_20_J19)
        alg_19_J18 = k28 * c27 * c25
        d_c25 = torch.zeros_like(y[:, 0]) + (alg_15_J14 - alg_19_J18 - alg_16_J15 + alg_17_J16)
        alg_18_J17 = p * c16
        d_c27 = torch.zeros_like(y[:, 0]) + (alg_18_J17 - alg_19_J18)
        d_c28 = torch.zeros_like(y[:, 0]) + (alg_19_J18)
        d_c31 = torch.zeros_like(y[:, 0]) + (alg_18_J17 - alg_20_J19)

        return torch.stack(
            [d_R_NOS2, d_c1, d_c2, d_c3, d_c4, d_c5, d_c6, d_c7, d_c8, d_c9, d_c10, d_c11, d_c12, d_c13, d_c14, d_c15, d_c16, d_c17, d_c18, d_c19, d_c20, d_c21, d_c22, d_c23, d_c24, d_c25, d_c26, d_c27, d_c28, d_c29, d_c30, d_c31],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
