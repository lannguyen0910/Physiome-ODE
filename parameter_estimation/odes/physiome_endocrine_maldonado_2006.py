import torch
import torch.nn as nn
import math


class EndocrineMaldonado2006ODE(nn.Module):
    """
    PyTorch translation of endocrine/maldonado_2006/model.py.
    Dynamic states: 9; independent constants: 46; derived constants used by rates: 5.
    Original constant slots: 51; algebraic slots: 6.
    """
    source_file = 'endocrine/maldonado_2006/model.py'
    source_state_dim = 9
    source_constant_dim = 51
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

        self.state_dim = 9
        self.parameter_dim = 46

        self.parameters = (
            parameters
            if parameters is not None
            else {'X_c': 0.0009127, 'X_b': 0.0007282, 'X_y': 0.0073, 'k_rB': 1, 'k_for': 0.001, 'k_res': 0.01, 'X_bss': 0.0007282, 'X_yss': 0.0073, 'k_byp': 0.1, 'k_yd': 1, 'k_Fs': 1, 'k_y': 1, 'X_noe': 0, 'k_yno': 20000, 'k_nod': 1000, 'X_pgex': 0, 'k_pged': 100, 'k_nopge': 1, 'k_ypge': 100, 'k_opgd': 0.35, 'k_nopg': 10, 'Io': 0, 'K_o_p': 200000, 'K_l_p': 3000000, 'K': 10, 'k_nokl': 100, 'Il': 0, 'rl': 1000, 'k1': 0.01, 'k2': 10, 'k3': 0.00058, 'k4': 0.017, 'ko': 0.35, 'D_R': 0.0007, 'k_pger': 0.0001, 'k_B': 0.189, 'D_C': 0.189, 'D_A': 0.7, 'f0': 0.05, 'dB': 0.7, 'C_s': 0.005, 'SP': 250, 'k5': 0.02, 'k6': 3, 'IP': 0, 'kP': 86}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'r_B': 0.9912, 'xc': 0.0009127, 'xb': 0.0007282, 'xy': 0.0073, 'x_no': 0.02, 'x_pge': 0.01, 'x_opg': 0.01, 'xr': 0.0007734, 'x_kl': 0.01}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['X_c', 'X_b', 'X_y', 'k_rB', 'k_for', 'k_res', 'X_bss', 'X_yss', 'k_byp', 'k_yd', 'k_Fs', 'k_y', 'X_noe', 'k_yno', 'k_nod', 'X_pgex', 'k_pged', 'k_nopge', 'k_ypge', 'k_opgd', 'k_nopg', 'Io', 'K_o_p', 'K_l_p', 'K', 'k_nokl', 'Il', 'rl', 'k1', 'k2', 'k3', 'k4', 'ko', 'D_R', 'k_pger', 'k_B', 'D_C', 'D_A', 'f0', 'dB', 'C_s', 'SP', 'k5', 'k6', 'IP', 'kP']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['r_B', 'xc', 'xb', 'xy', 'x_no', 'x_pge', 'x_opg', 'xr', 'x_kl']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (9)
        r_B = y[:, 0]
        xc = y[:, 1]
        xb = y[:, 2]
        xy = y[:, 3]
        x_no = y[:, 4]
        x_pge = y[:, 5]
        x_opg = y[:, 6]
        xr = y[:, 7]
        x_kl = y[:, 8]

        # Independent parameters (source constant order)
        X_c = theta[:, 0]  # constants[0]
        X_b = theta[:, 1]  # constants[1]
        X_y = theta[:, 2]  # constants[2]
        k_rB = theta[:, 3]  # constants[3]
        k_for = theta[:, 4]  # constants[4]
        k_res = theta[:, 5]  # constants[5]
        X_bss = theta[:, 6]  # constants[6]
        X_yss = theta[:, 7]  # constants[7]
        k_byp = theta[:, 8]  # constants[8]
        k_yd = theta[:, 9]  # constants[9]
        k_Fs = theta[:, 10]  # constants[10]
        k_y = theta[:, 11]  # constants[11]
        X_noe = theta[:, 12]  # constants[12]
        k_yno = theta[:, 13]  # constants[13]
        k_nod = theta[:, 14]  # constants[14]
        X_pgex = theta[:, 15]  # constants[15]
        k_pged = theta[:, 16]  # constants[16]
        k_nopge = theta[:, 17]  # constants[17]
        k_ypge = theta[:, 18]  # constants[18]
        k_opgd = theta[:, 19]  # constants[19]
        k_nopg = theta[:, 20]  # constants[20]
        Io = theta[:, 21]  # constants[21]
        K_o_p = theta[:, 22]  # constants[22]
        K_l_p = theta[:, 23]  # constants[23]
        K = theta[:, 24]  # constants[24]
        k_nokl = theta[:, 25]  # constants[25]
        Il = theta[:, 26]  # constants[26]
        rl = theta[:, 27]  # constants[27]
        k1 = theta[:, 28]  # constants[28]
        k2 = theta[:, 29]  # constants[29]
        k3 = theta[:, 30]  # constants[30]
        k4 = theta[:, 31]  # constants[31]
        ko = theta[:, 32]  # constants[32]
        D_R = theta[:, 33]  # constants[33]
        k_pger = theta[:, 34]  # constants[34]
        k_B = theta[:, 35]  # constants[35]
        D_C = theta[:, 36]  # constants[36]
        D_A = theta[:, 37]  # constants[37]
        f0 = theta[:, 38]  # constants[38]
        dB = theta[:, 39]  # constants[39]
        C_s = theta[:, 40]  # constants[40]
        SP = theta[:, 41]  # constants[41]
        k5 = theta[:, 42]  # constants[42]
        k6 = theta[:, 43]  # constants[43]
        IP = theta[:, 44]  # constants[44]
        kP = theta[:, 45]  # constants[45]

        # Derived constants from the original initConsts()
        D_B = f0 * dB  # constants[46]
        P = IP / kP  # constants[47]
        P_0 = SP / kP  # constants[48]
        P_s = k6 / k5  # constants[49]
        pi_p = (P + P_0) / (P + P_s)  # constants[50]

        # Initialize rates and algebraic variables, as in computeRates()
        d_r_B = torch.zeros_like(y[:, 0])
        d_xc = torch.zeros_like(y[:, 0])
        d_xb = torch.zeros_like(y[:, 0])
        d_xy = torch.zeros_like(y[:, 0])
        d_x_no = torch.zeros_like(y[:, 0])
        d_x_pge = torch.zeros_like(y[:, 0])
        d_x_opg = torch.zeros_like(y[:, 0])
        d_xr = torch.zeros_like(y[:, 0])
        d_x_kl = torch.zeros_like(y[:, 0])
        alg_0_A_B = torch.zeros_like(y[:, 0])
        alg_1_pi_L = torch.zeros_like(y[:, 0])
        alg_2_F_a = torch.zeros_like(y[:, 0])
        alg_3_pi_c = torch.zeros_like(y[:, 0])
        alg_4_F_s = torch.zeros_like(y[:, 0])
        alg_5_F_sti = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_r_B = torch.zeros_like(y[:, 0]) + (k_for / X_b * xb + 1.0 / X_y * xy - (k_res / X_c * xc + k_rB * r_B))
        d_xy = torch.zeros_like(y[:, 0]) + (k_byp * (xb - X_bss) - k_yd * (xy - X_yss))
        d_x_kl = torch.zeros_like(y[:, 0]) + (rl + Il - (k_nokl * x_no + rl * ((1.0 + k1 / k2 * x_opg + k3 / k4 * K) / (K_l_p * pi_p * xb)) * x_kl))
        alg_3_pi_c = (xc + f0 * C_s) / (xc + C_s)
        d_x_opg = torch.zeros_like(y[:, 0]) + (K_o_p / 1.0 * alg_3_pi_c * xr + Io + k_nopg * x_no - k_opgd * x_opg)
        d_xr = torch.zeros_like(y[:, 0]) + (D_R * alg_3_pi_c + k_pger * x_pge - D_B / alg_3_pi_c * xr)
        d_xb = torch.zeros_like(y[:, 0]) + (D_B / alg_3_pi_c * xr - k_B * xb)
        alg_1_pi_L = k3 / k4 * K_l_p * pi_p * xb / (1.0 + k3 * K / k4 + k1 / (k2 * ko) * (K_o_p / 1.0 / pi_p * xr + Io)) * (1.0 + Il / rl)
        d_xc = torch.zeros_like(y[:, 0]) + (D_C * alg_1_pi_L - D_A * alg_3_pi_c * xc)
        alg_0_A_B = r_B ** 2.0 * math.pi
        alg_2_F_a = torch.where((t >= 100.0) & (t < 130.0), 10000.0, 100.0)
        alg_4_F_s = alg_2_F_a / alg_0_A_B
        alg_5_F_sti = alg_4_F_s * xy / (1.0 + torch.exp(-(k_Fs * alg_4_F_s + k_y * xy)))
        d_x_no = torch.zeros_like(y[:, 0]) + (k_yno * alg_5_F_sti + X_noe - k_nod * x_no)
        d_x_pge = torch.zeros_like(y[:, 0]) + (k_ypge * alg_5_F_sti + k_nopge * x_no + X_pgex - k_pged * x_pge)

        return torch.stack(
            [d_r_B, d_xc, d_xb, d_xy, d_x_no, d_x_pge, d_x_opg, d_xr, d_x_kl],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
