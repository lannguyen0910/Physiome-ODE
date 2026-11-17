import torch
import torch.nn as nn
import math


class IonTransportParentEtAl1992ODE(nn.Module):
    """
    PyTorch translation of ion_transport/Parent_et_al_1992/model.py.
    Dynamic states: 7; independent constants: 30; derived constants used by rates: 5.
    Original constant slots: 35; algebraic slots: 43.
    """
    source_file = 'ion_transport/Parent_et_al_1992/model.py'
    source_state_dim = 7
    source_constant_dim = 35
    algebraic_dim = 43
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
        self.parameter_dim = 30

        self.parameters = (
            parameters
            if parameters is not None
            else {'k0_12': 80000, 'k0_21': 500, 'k0_23': 100000, 'k0_32': 20, 'k0_34': 50, 'k0_43': 50, 'k0_45': 800, 'k0_54': 18300000, 'k0_25': 0.3, 'k0_52': 1.37, 'k0_56': 10, 'k0_65': 50, 'k0_61': 5, 'k0_16': 35, 'delta': 0.7, 'alpha_pp': 0, 'N_C': 60000000000, 'N_Avo': 6.022e+23, 'area': 1000000, 'n': 2, 'z_c': -2, 'z_Na': 1, 'F': 96485.34, 'R': 8.314, 'T': 310, 'Na_i': 0.02, 'Na_o': 0.1, 'glucose_i': 1e-05, 'glucose_o': 0, 'constant_34': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -0.05, 'C_1': 1.505e-08, 'C_2': 7.3976e-08, 'C_3': 2.4603e-10, 'C_4': 3.4444e-10, 'C_5': 1.5338e-09, 'C_6_temp': 8.484e-09}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k0_12', 'k0_21', 'k0_23', 'k0_32', 'k0_34', 'k0_43', 'k0_45', 'k0_54', 'k0_25', 'k0_52', 'k0_56', 'k0_65', 'k0_61', 'k0_16', 'delta', 'alpha_pp', 'N_C', 'N_Avo', 'area', 'n', 'z_c', 'z_Na', 'F', 'R', 'T', 'Na_i', 'Na_o', 'glucose_i', 'glucose_o', 'constant_34']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6_temp']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (7)
        V = y[:, 0]
        C_1 = y[:, 1]
        C_2 = y[:, 2]
        C_3 = y[:, 3]
        C_4 = y[:, 4]
        C_5 = y[:, 5]
        C_6_temp = y[:, 6]

        # Independent parameters (source constant order)
        k0_12 = theta[:, 0]  # constants[0]
        k0_21 = theta[:, 1]  # constants[1]
        k0_23 = theta[:, 2]  # constants[2]
        k0_32 = theta[:, 3]  # constants[3]
        k0_34 = theta[:, 4]  # constants[4]
        k0_43 = theta[:, 5]  # constants[5]
        k0_45 = theta[:, 6]  # constants[6]
        k0_54 = theta[:, 7]  # constants[7]
        k0_25 = theta[:, 8]  # constants[8]
        k0_52 = theta[:, 9]  # constants[9]
        k0_56 = theta[:, 10]  # constants[10]
        k0_65 = theta[:, 11]  # constants[11]
        k0_61 = theta[:, 12]  # constants[12]
        k0_16 = theta[:, 13]  # constants[13]
        delta = theta[:, 14]  # constants[14]
        alpha_pp = theta[:, 15]  # constants[15]
        N_C = theta[:, 16]  # constants[16]
        N_Avo = theta[:, 17]  # constants[17]
        area = theta[:, 18]  # constants[18]
        n = theta[:, 19]  # constants[19]
        z_c = theta[:, 20]  # constants[20]
        z_Na = theta[:, 21]  # constants[21]
        F = theta[:, 22]  # constants[22]
        R = theta[:, 23]  # constants[23]
        T = theta[:, 24]  # constants[24]
        Na_i = theta[:, 25]  # constants[25]
        Na_o = theta[:, 26]  # constants[26]
        glucose_i = theta[:, 27]  # constants[27]
        glucose_o = theta[:, 28]  # constants[28]
        constant_34 = theta[:, 29]  # constants[34]

        # Derived constants from the original initConsts()
        alpha_p = 1.0 - delta - alpha_pp  # constants[29]
        C_T = 1000000.0 * N_C / N_Avo  # constants[30]
        k_23 = k0_23 * glucose_o  # constants[31]
        k_32 = k0_32  # constants[32]
        k_45 = k0_45  # constants[33]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_C_1 = torch.zeros_like(y[:, 0])
        d_C_2 = torch.zeros_like(y[:, 0])
        d_C_3 = torch.zeros_like(y[:, 0])
        d_C_4 = torch.zeros_like(y[:, 0])
        d_C_5 = torch.zeros_like(y[:, 0])
        d_C_6_temp = torch.zeros_like(y[:, 0])
        alg_0_mu = torch.zeros_like(y[:, 0])
        alg_1_ks_12 = torch.zeros_like(y[:, 0])
        alg_3_k_12 = torch.zeros_like(y[:, 0])
        alg_4_k_21 = torch.zeros_like(y[:, 0])
        alg_5_k_34 = torch.zeros_like(y[:, 0])
        alg_6_k_43 = torch.zeros_like(y[:, 0])
        alg_7_k_25 = torch.zeros_like(y[:, 0])
        alg_8_k_56 = torch.zeros_like(y[:, 0])
        alg_9_k_65 = torch.zeros_like(y[:, 0])
        alg_10_k_61 = torch.zeros_like(y[:, 0])
        alg_11_k_16 = torch.zeros_like(y[:, 0])
        alg_12_k_52 = torch.zeros_like(y[:, 0])
        alg_13_k0_54_temp = torch.zeros_like(y[:, 0])
        alg_14_k_54 = torch.zeros_like(y[:, 0])
        alg_15_C_6 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_V = torch.zeros_like(y[:, 0]) + (constant_34)
        alg_0_mu = F * V / (R * T)
        alg_5_k_34 = k0_34 * torch.exp(-(z_c + n) * delta * alg_0_mu / 2.0)
        alg_6_k_43 = k0_43 * torch.exp((z_c + n) * delta * alg_0_mu / 2.0)
        d_C_3 = torch.zeros_like(y[:, 0]) + (k_23 * C_2 + alg_6_k_43 * C_4 - (k_32 + alg_5_k_34) * C_3)
        alg_8_k_56 = k0_56 * torch.exp(-n * z_Na * alpha_pp * alg_0_mu / 2.0)
        alg_9_k_65 = k0_65 * Na_i ** n * torch.exp(n * z_Na * alpha_pp * alg_0_mu / 2.0)
        alg_10_k_61 = k0_61 * torch.exp(z_c * delta * alg_0_mu / 2.0)
        alg_11_k_16 = k0_16 * torch.exp(-z_c * delta * alg_0_mu / 2.0)
        d_C_6_temp = torch.zeros_like(y[:, 0]) + (alg_11_k_16 * C_1 + alg_8_k_56 * C_5 - (alg_10_k_61 + alg_9_k_65) * C_6_temp)
        alg_1_ks_12 = k0_12 * torch.exp(-n * alpha_p * alg_0_mu / 2.0)
        alg_3_k_12 = alg_1_ks_12 * Na_o ** n
        alg_4_k_21 = k0_21 * torch.exp(n * z_Na * alpha_p * alg_0_mu / 2.0)
        alg_7_k_25 = k0_25 * torch.exp(-(z_c + n) * delta * alg_0_mu / 2.0)
        alg_12_k_52 = k0_12 * alg_7_k_25 * k0_56 * k0_61 / (k0_21 * k0_16 * k0_65)
        d_C_2 = torch.zeros_like(y[:, 0]) + (alg_3_k_12 * C_1 + k_32 * C_3 + alg_12_k_52 * C_5 - (alg_4_k_21 + k_23 + alg_7_k_25) * C_2)
        alg_13_k0_54_temp = k0_23 * alg_5_k_34 * k_45 * alg_12_k_52 / (alg_6_k_43 * k_32 * alg_7_k_25)
        alg_14_k_54 = alg_13_k0_54_temp * glucose_i
        d_C_4 = torch.zeros_like(y[:, 0]) + (alg_5_k_34 * C_3 + alg_14_k_54 * C_5 - (k_45 + alg_6_k_43) * C_4)
        alg_15_C_6 = C_T - (C_1 + C_2 + C_3 + C_4 + C_5)
        d_C_1 = torch.zeros_like(y[:, 0]) + (alg_4_k_21 * C_2 + alg_10_k_61 * alg_15_C_6 - (alg_3_k_12 + alg_11_k_16) * C_1)
        d_C_5 = torch.zeros_like(y[:, 0]) + (k_45 * C_4 + alg_9_k_65 * alg_15_C_6 + alg_7_k_25 * C_2 - (alg_14_k_54 + alg_12_k_52 + alg_8_k_56) * C_5)

        return torch.stack(
            [d_V, d_C_1, d_C_2, d_C_3, d_C_4, d_C_5, d_C_6_temp],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
