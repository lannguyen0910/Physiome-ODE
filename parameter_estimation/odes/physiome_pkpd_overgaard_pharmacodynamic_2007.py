import torch
import torch.nn as nn
import math


class PKPDOvergaardPharmacodynamic2007ODE(nn.Module):
    """
    PyTorch translation of PKPD/overgaard_pharmacodynamic_2007/model.py.
    Dynamic states: 3; independent constants: 30; derived constants used by rates: 7.
    Original constant slots: 37; algebraic slots: 16.
    """
    source_file = 'PKPD/overgaard_pharmacodynamic_2007/model.py'
    source_state_dim = 3
    source_constant_dim = 37
    algebraic_dim = 16
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 3
        self.parameter_dim = 30

        self.parameters = (
            parameters
            if parameters is not None
            else {'T_a': 21, 'T_b': 38, 'delta_T': 1.57, 'kinc': 0.0258, 'tdose1': 24, 'tdose2': 72, 'tdose3': 120, 't_day': 17.5, 't_night': 6.73, 'day_length': 86400, 'km': 1.1375, 'c': 3.47, 'pEtot': 0.144, 'kR': 5.35, 'AMT_dose': 3, 'pEf1': 1, 'pEs1': 0.2, 'pEf2': 3.57, 'pEs2': 2.43, 'pEf3': 8, 'pEs3': 50, 'M_b': 3, 't_prime': 45.12, 'alpha': 0.2229166, 'delta_high_dose': 1, 'Ns': 4, 'Nf': 4, 'Ts': 2.45, 'Tf': 0.368, 'f2_drug': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'M': 3.5, 'T': 38.785, 'BR': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['T_a', 'T_b', 'delta_T', 'kinc', 'tdose1', 'tdose2', 'tdose3', 't_day', 't_night', 'day_length', 'km', 'c', 'pEtot', 'kR', 'AMT_dose', 'pEf1', 'pEs1', 'pEf2', 'pEs2', 'pEf3', 'pEs3', 'M_b', 't_prime', 'alpha', 'delta_high_dose', 'Ns', 'Nf', 'Ts', 'Tf', 'f2_drug']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['M', 'T', 'BR']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        M = y[:, 0]
        T = y[:, 1]
        BR = y[:, 2]

        # Independent parameters (source constant order)
        T_a = theta[:, 0]  # constants[0]
        T_b = theta[:, 1]  # constants[1]
        delta_T = theta[:, 2]  # constants[2]
        kinc = theta[:, 3]  # constants[3]
        tdose1 = theta[:, 4]  # constants[4]
        tdose2 = theta[:, 5]  # constants[5]
        tdose3 = theta[:, 6]  # constants[6]
        t_day = theta[:, 7]  # constants[7]
        t_night = theta[:, 8]  # constants[8]
        day_length = theta[:, 9]  # constants[9]
        km = theta[:, 10]  # constants[10]
        c = theta[:, 11]  # constants[11]
        pEtot = theta[:, 12]  # constants[12]
        kR = theta[:, 13]  # constants[13]
        AMT_dose = theta[:, 14]  # constants[14]
        pEf1 = theta[:, 15]  # constants[15]
        pEs1 = theta[:, 16]  # constants[16]
        pEf2 = theta[:, 17]  # constants[17]
        pEs2 = theta[:, 18]  # constants[18]
        pEf3 = theta[:, 19]  # constants[19]
        pEs3 = theta[:, 20]  # constants[20]
        M_b = theta[:, 21]  # constants[21]
        t_prime = theta[:, 22]  # constants[22]
        alpha = theta[:, 23]  # constants[23]
        delta_high_dose = theta[:, 24]  # constants[24]
        Ns = theta[:, 25]  # constants[25]
        Nf = theta[:, 26]  # constants[26]
        Ts = theta[:, 27]  # constants[27]
        Tf = theta[:, 28]  # constants[28]
        f2_drug = theta[:, 29]  # constants[30]

        # Derived constants from the original initConsts()
        T_day = T_b + delta_T / 2.0  # constants[29]
        Kf = Nf / Tf  # constants[31]
        T_night = T_b - delta_T / 2.0  # constants[32]
        Ks = Ns / Ts  # constants[33]
        kb = M_b / (T_b - T_a)  # constants[34]
        M_day = (kb + kinc * (T_day - T_b)) * (T_day - T_a)  # constants[35]
        M_night_baseline = (kb + kinc * (T_night - T_b)) * (T_night - T_a)  # constants[36]

        # Initialize rates and algebraic variables, as in computeRates()
        d_M = torch.zeros_like(y[:, 0])
        d_T = torch.zeros_like(y[:, 0])
        d_BR = torch.zeros_like(y[:, 0])
        alg_0_k = torch.zeros_like(y[:, 0])
        alg_1_tprime = torch.zeros_like(y[:, 0])
        alg_2_f_prime = torch.zeros_like(y[:, 0])
        alg_3_M_night = torch.zeros_like(y[:, 0])
        alg_4_X1 = torch.zeros_like(y[:, 0])
        alg_5_M_c = torch.zeros_like(y[:, 0])
        alg_6_gNsTs1 = torch.zeros_like(y[:, 0])
        alg_7_gNfTf1 = torch.zeros_like(y[:, 0])
        alg_8_X2 = torch.zeros_like(y[:, 0])
        alg_9_gNsTs2 = torch.zeros_like(y[:, 0])
        alg_10_gNfTf2 = torch.zeros_like(y[:, 0])
        alg_11_X3 = torch.zeros_like(y[:, 0])
        alg_12_gNsTs3 = torch.zeros_like(y[:, 0])
        alg_13_E_slow = torch.zeros_like(y[:, 0])
        alg_14_gNfTf3 = torch.zeros_like(y[:, 0])
        alg_15_E_fast = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_k = kb + kinc * (T - T_b * (1.0 + pEtot * BR)) + f2_drug
        d_T = torch.zeros_like(y[:, 0]) + (c ** (-1.0) * (M - alg_0_k * (T - T_a)))
        alg_1_tprime = t * 3600.0 * 1.0 % day_length
        alg_2_f_prime = delta_high_dose * (1.0 + torch.exp(-alpha * (t - (tdose1 + t_prime)))) ** (-1.0)
        alg_3_M_night = (1.0 - alg_2_f_prime) * M_night_baseline + alg_2_f_prime * M_day
        alg_5_M_c = torch.where((alg_1_tprime / 3600.0 >= t_night) & (alg_1_tprime / 3600.0 < t_day), alg_3_M_night, M_day)
        d_M = torch.zeros_like(y[:, 0]) + (-km * (M - alg_5_M_c))
        alg_4_X1 = (t - tdose1) / 24.0
        alg_6_gNsTs1 = torch.where(alg_4_X1 > 0.0, Ks ** Ns / 6.0 * torch.exp(-Ks * alg_4_X1) * alg_4_X1 ** (Ns - 1.0), 0.0)
        alg_8_X2 = (t - tdose2) / 24.0
        alg_9_gNsTs2 = torch.where(alg_8_X2 > 0.0, Ks ** Ns / 6.0 * torch.exp(-Ks * alg_8_X2) * alg_8_X2 ** (Ns - 1.0), 0.0)
        alg_11_X3 = (t - tdose3) / 24.0
        alg_12_gNsTs3 = torch.where(alg_11_X3 > 0.0, Ks ** Ns / 6.0 * torch.exp(-Ks * alg_11_X3) * alg_11_X3 ** (Ns - 1.0), 0.0)
        alg_13_E_slow = AMT_dose * pEs2 * (alg_6_gNsTs1 + alg_9_gNsTs2 + alg_12_gNsTs3)
        alg_7_gNfTf1 = torch.where(alg_4_X1 > 0.0, Kf ** Nf / 6.0 * torch.exp(-Kf * alg_4_X1) * alg_4_X1 ** (Nf - 1.0), 0.0)
        alg_10_gNfTf2 = torch.where(alg_8_X2 > 0.0, Kf ** Nf / 6.0 * torch.exp(-Kf * alg_8_X2) * alg_8_X2 ** (Nf - 1.0), 0.0)
        alg_14_gNfTf3 = torch.where(alg_11_X3 > 0.0, Kf ** Nf / 6.0 * torch.exp(-Kf * alg_11_X3) * alg_11_X3 ** (Nf - 1.0), 0.0)
        alg_15_E_fast = pEf2 * (alg_7_gNfTf1 + alg_10_gNfTf2 + alg_14_gNfTf3)
        d_BR = torch.zeros_like(y[:, 0]) + (alg_2_f_prime * (alg_13_E_slow + alg_15_E_fast) * (1.0 - BR) - kR * BR)

        return torch.stack(
            [d_M, d_T, d_BR],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
