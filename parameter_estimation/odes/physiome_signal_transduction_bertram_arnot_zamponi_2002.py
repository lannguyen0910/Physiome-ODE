import torch
import torch.nn as nn
import math


class SignalTransductionBertramArnotZamponi2002ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/bertram_arnot_zamponi_2002/model.py.
    Dynamic states: 11; independent constants: 22; derived constants used by rates: 0.
    Original constant slots: 22; algebraic slots: 21.
    """
    source_file = 'signal_transduction/bertram_arnot_zamponi_2002/model.py'
    source_state_dim = 11
    source_constant_dim = 22
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

        self.state_dim = 11
        self.parameter_dim = 22

        self.parameters = (
            parameters
            if parameters is not None
            else {'R_gas_const': 8314.41, 'Temp': 310, 'F': 96485, 'Cm': 1, 'IstimStart': 10, 'IstimEnd': 50000, 'IstimAmplitude': 40, 'IstimPeriod': 100, 'IstimPulseDuration': 1, 'T_bar': 4000, 'kr_plus': 0.15, 'kr_minus': 2.5, 'Ca_ex': 2000, 'Dc': 220, 'r': 10, 'g_Ca': 1.2, 'P': 0.006, 'kG_minus': 0.00025, 'kG2_minus': 0.016, 'kG3_minus': 1.024, 'ka_plus': 200, 'ka_minus': 0.0015}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -65, 'n': 0, 'R': 0, 'a': 0, 'C1': 1, 'C2': 0, 'C_G1': 0, 'C3': 0, 'C_G2': 0, 'C4': 0, 'C_G3': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R_gas_const', 'Temp', 'F', 'Cm', 'IstimStart', 'IstimEnd', 'IstimAmplitude', 'IstimPeriod', 'IstimPulseDuration', 'T_bar', 'kr_plus', 'kr_minus', 'Ca_ex', 'Dc', 'r', 'g_Ca', 'P', 'kG_minus', 'kG2_minus', 'kG3_minus', 'ka_plus', 'ka_minus']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'n', 'R', 'a', 'C1', 'C2', 'C_G1', 'C3', 'C_G2', 'C4', 'C_G3']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (11)
        V = y[:, 0]
        n = y[:, 1]
        R = y[:, 2]
        a = y[:, 3]
        C1 = y[:, 4]
        C2 = y[:, 5]
        C_G1 = y[:, 6]
        C3 = y[:, 7]
        C_G2 = y[:, 8]
        C4 = y[:, 9]
        C_G3 = y[:, 10]

        # Independent parameters (source constant order)
        R_gas_const = theta[:, 0]  # constants[0]
        Temp = theta[:, 1]  # constants[1]
        F = theta[:, 2]  # constants[2]
        Cm = theta[:, 3]  # constants[3]
        IstimStart = theta[:, 4]  # constants[4]
        IstimEnd = theta[:, 5]  # constants[5]
        IstimAmplitude = theta[:, 6]  # constants[6]
        IstimPeriod = theta[:, 7]  # constants[7]
        IstimPulseDuration = theta[:, 8]  # constants[8]
        T_bar = theta[:, 9]  # constants[9]
        kr_plus = theta[:, 10]  # constants[10]
        kr_minus = theta[:, 11]  # constants[11]
        Ca_ex = theta[:, 12]  # constants[12]
        Dc = theta[:, 13]  # constants[13]
        r = theta[:, 14]  # constants[14]
        g_Ca = theta[:, 15]  # constants[15]
        P = theta[:, 16]  # constants[16]
        kG_minus = theta[:, 17]  # constants[17]
        kG2_minus = theta[:, 18]  # constants[18]
        kG3_minus = theta[:, 19]  # constants[19]
        ka_plus = theta[:, 20]  # constants[20]
        ka_minus = theta[:, 21]  # constants[21]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_R = torch.zeros_like(y[:, 0])
        d_a = torch.zeros_like(y[:, 0])
        d_C1 = torch.zeros_like(y[:, 0])
        d_C2 = torch.zeros_like(y[:, 0])
        d_C_G1 = torch.zeros_like(y[:, 0])
        d_C3 = torch.zeros_like(y[:, 0])
        d_C_G2 = torch.zeros_like(y[:, 0])
        d_C4 = torch.zeros_like(y[:, 0])
        d_C_G3 = torch.zeros_like(y[:, 0])
        alg_0_i_app = torch.zeros_like(y[:, 0])
        alg_1_alpha_n = torch.zeros_like(y[:, 0])
        alg_2_T = torch.zeros_like(y[:, 0])
        alg_3_i_V = torch.zeros_like(y[:, 0])
        alg_5_alpha_x = torch.zeros_like(y[:, 0])
        alg_6_beta_n = torch.zeros_like(y[:, 0])
        alg_7_sigma = torch.zeros_like(y[:, 0])
        alg_8_beta_x = torch.zeros_like(y[:, 0])
        alg_9_Ca_open = torch.zeros_like(y[:, 0])
        alg_10_x_infinity = torch.zeros_like(y[:, 0])
        alg_11_alpha = torch.zeros_like(y[:, 0])
        alg_12_i_Na = torch.zeros_like(y[:, 0])
        alg_13_alpha_2 = torch.zeros_like(y[:, 0])
        alg_14_i_K = torch.zeros_like(y[:, 0])
        alg_15_beta = torch.zeros_like(y[:, 0])
        alg_16_i_leak = torch.zeros_like(y[:, 0])
        alg_17_O = torch.zeros_like(y[:, 0])
        alg_18_beta_2 = torch.zeros_like(y[:, 0])
        alg_19_Ca = torch.zeros_like(y[:, 0])
        alg_20_kG_plus = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_2_T = T_bar * R
        d_a = torch.zeros_like(y[:, 0]) + (ka_plus * alg_2_T * (1.0 - a) - ka_minus * a)
        alg_1_alpha_n = 0.02 * (V + 55.0) / (1.0 - torch.exp(-(V + 55.0) / 10.0))
        alg_6_beta_n = 0.25 * torch.exp(-(V + 65.0) / 80.0)
        d_n = torch.zeros_like(y[:, 0]) + (alg_1_alpha_n * (1.0 - n) - alg_6_beta_n * n)
        alg_0_i_app = torch.where((t >= IstimStart) & (t <= IstimEnd) & (t - IstimStart - torch.floor((t - IstimStart) / IstimPeriod) * IstimPeriod <= IstimPulseDuration), IstimAmplitude, 0.0)
        alg_5_alpha_x = 0.2 * (V + 40.0) / (1.0 - 1.0 * torch.exp(-(V + 40.0) / 10.0))
        alg_8_beta_x = 8.0 * torch.exp(1.0 / -(V + 65.0 / 18.0))
        alg_10_x_infinity = alg_5_alpha_x / (alg_5_alpha_x + alg_8_beta_x)
        alg_12_i_Na = 120.0 * alg_10_x_infinity ** 3.0 * (1.0 - n) * (V - 120.0)
        alg_14_i_K = 36.0 * n ** 4.0 * (V + 77.0) / 1.0
        alg_16_i_leak = 0.3 * (V + 54.0)
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_12_i_Na + alg_14_i_K + alg_16_i_leak - alg_0_i_app) / Cm)
        alg_17_O = 1.0 - C1 - C2 - C3 - C4 - C_G1 - C_G2 - C_G3
        alg_11_alpha = 0.45 * torch.exp(V / 22.0)
        alg_15_beta = 0.015 * torch.exp(-V / 14.0)
        d_C4 = torch.zeros_like(y[:, 0]) + (2.0 * alg_11_alpha * C3 + 4.0 * alg_15_beta * alg_17_O - C4 * (3.0 * alg_15_beta + alg_11_alpha))
        alg_3_i_V = g_Ca * P * 2.0 * F * V / (R_gas_const * Temp) * Ca_ex / (1.0 - torch.exp(2.0 * F * V / (R_gas_const * Temp)))
        alg_7_sigma = -5.182 * alg_3_i_V
        alg_9_Ca_open = alg_7_sigma / (2.0 * Dc * r * math.pi)
        alg_19_Ca = alg_17_O * alg_9_Ca_open + 0.1
        d_R = torch.zeros_like(y[:, 0]) + (kr_plus * alg_19_Ca * (1.0 - R) - kr_minus * R)
        alg_20_kG_plus = 3.0 * a / (680.0 + 320.0 * a)
        d_C1 = torch.zeros_like(y[:, 0]) + (alg_15_beta * C2 + kG_minus * C_G1 - C1 * (4.0 * alg_11_alpha + alg_20_kG_plus))
        d_C2 = torch.zeros_like(y[:, 0]) + (4.0 * alg_11_alpha * C1 + 2.0 * alg_15_beta * C3 + kG2_minus * C_G2 - C2 * (alg_15_beta + 3.0 * alg_11_alpha + alg_20_kG_plus))
        d_C3 = torch.zeros_like(y[:, 0]) + (3.0 * alg_11_alpha * C2 + 3.0 * alg_15_beta * C4 + kG3_minus * C_G3 - C3 * (2.0 * alg_15_beta + 2.0 * alg_11_alpha + alg_20_kG_plus))
        alg_13_alpha_2 = alg_11_alpha / 8.0
        alg_18_beta_2 = alg_15_beta * 8.0
        d_C_G1 = torch.zeros_like(y[:, 0]) + (alg_18_beta_2 * C_G2 + alg_20_kG_plus * C1 - C_G1 * (4.0 * alg_13_alpha_2 + kG_minus))
        d_C_G2 = torch.zeros_like(y[:, 0]) + (4.0 * alg_13_alpha_2 * C_G1 + 2.0 * alg_18_beta_2 * C_G3 + alg_20_kG_plus * C2 - C_G2 * (alg_18_beta_2 + 3.0 * alg_13_alpha_2 + kG2_minus))
        d_C_G3 = torch.zeros_like(y[:, 0]) + (3.0 * alg_13_alpha_2 * C_G2 + alg_20_kG_plus * C3 - C_G3 * (2.0 * alg_18_beta_2 + kG3_minus))

        return torch.stack(
            [d_V, d_n, d_R, d_a, d_C1, d_C2, d_C_G1, d_C3, d_C_G2, d_C4, d_C_G3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
