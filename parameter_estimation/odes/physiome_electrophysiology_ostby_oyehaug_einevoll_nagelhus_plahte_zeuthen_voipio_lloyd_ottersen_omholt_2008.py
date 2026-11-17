import torch
import torch.nn as nn
import math


class ElectrophysiologyOstbyOyehaugEinevollNagelhusPlahteZeuthenVoipioLloydOttersenOmholt2008ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/ostby_oyehaug_einevoll_nagelhus_plahte_zeuthen_voipio_lloyd_ottersen_omholt_2008/model.py.
    Dynamic states: 8; independent constants: 27; derived constants used by rates: 4.
    Original constant slots: 31; algebraic slots: 28.
    """
    source_file = 'electrophysiology/ostby_oyehaug_einevoll_nagelhus_plahte_zeuthen_voipio_lloyd_ottersen_omholt_2008/model.py'
    source_state_dim = 8
    source_constant_dim = 31
    algebraic_dim = 28
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 8
        self.parameter_dim = 27

        self.parameters = (
            parameters
            if parameters is not None
            else {'Lp': 0.00021, 'Xi': 12.41, 'w_tot': 0.0879, 'gNa': 0.0001314, 'F': 96490, 'theta_NKCC1': 1, 'theta_NBC': 1, 'gK': 0.004, 'theta_KCC1': 1, 'kc': 0.0735, 'T': 300, 'R': 8.315, 'gKCC1': 1e-06, 'gNBC': 9.03e-05, 'gNKCC1': 5.54e-06, 'zNBC': -1, 'gCl': 8.797e-05, 'Pmax': 1.4207e-10, 'K_Nai': 10, 'K_Ko': 1.5, 'nA': 6.0221415e+23, 'elementary_charge': 1.6e-19, 't0': 10, 't1': 20, 't2': 30, 'alpha': 2, 'beta': 14}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'wi': 0.061, 'N_Nai': 0.99796, 'N_Ki': 5.52782, 'N_HCO3i': 0.58804, 'N_Cli': 0.32879, 'N_Nao': 4.301041, 'N_Ko': 0.0807, 'N_HCO3o': 0.432552}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Lp', 'Xi', 'w_tot', 'gNa', 'F', 'theta_NKCC1', 'theta_NBC', 'gK', 'theta_KCC1', 'kc', 'T', 'R', 'gKCC1', 'gNBC', 'gNKCC1', 'zNBC', 'gCl', 'Pmax', 'K_Nai', 'K_Ko', 'nA', 'elementary_charge', 't0', 't1', 't2', 'alpha', 'beta']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['wi', 'N_Nai', 'N_Ki', 'N_HCO3i', 'N_Cli', 'N_Nao', 'N_Ko', 'N_HCO3o']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (8)
        wi = y[:, 0]
        N_Nai = y[:, 1]
        N_Ki = y[:, 2]
        N_HCO3i = y[:, 3]
        N_Cli = y[:, 4]
        N_Nao = y[:, 5]
        N_Ko = y[:, 6]
        N_HCO3o = y[:, 7]

        # Independent parameters (source constant order)
        Lp = theta[:, 0]  # constants[0]
        Xi = theta[:, 1]  # constants[1]
        w_tot = theta[:, 2]  # constants[2]
        gNa = theta[:, 3]  # constants[3]
        F = theta[:, 4]  # constants[4]
        theta_NKCC1 = theta[:, 5]  # constants[5]
        theta_NBC = theta[:, 6]  # constants[6]
        gK = theta[:, 7]  # constants[7]
        theta_KCC1 = theta[:, 8]  # constants[8]
        kc = theta[:, 9]  # constants[9]
        T = theta[:, 10]  # constants[10]
        R = theta[:, 11]  # constants[11]
        gKCC1 = theta[:, 12]  # constants[12]
        gNBC = theta[:, 13]  # constants[13]
        gNKCC1 = theta[:, 14]  # constants[14]
        zNBC = theta[:, 15]  # constants[15]
        gCl = theta[:, 16]  # constants[16]
        Pmax = theta[:, 17]  # constants[17]
        K_Nai = theta[:, 18]  # constants[18]
        K_Ko = theta[:, 19]  # constants[19]
        nA = theta[:, 20]  # constants[20]
        elementary_charge = theta[:, 21]  # constants[21]
        t0 = theta[:, 22]  # constants[22]
        t1 = theta[:, 23]  # constants[23]
        t2 = theta[:, 24]  # constants[24]
        alpha = theta[:, 25]  # constants[25]
        beta = theta[:, 26]  # constants[26]

        # Derived constants from the original initConsts()
        deltaT = t1 - t0  # constants[27]
        gamma_alpha_beta = torch.exp(torch.lgamma(alpha + beta - 1.0 + 1.0))  # constants[28]
        gamma_alpha = torch.exp(torch.lgamma(alpha - 1.0 + 1.0))  # constants[29]
        gamma_beta = torch.exp(torch.lgamma(beta - 1.0 + 1.0))  # constants[30]

        # Initialize rates and algebraic variables, as in computeRates()
        d_wi = torch.zeros_like(y[:, 0])
        d_N_Nai = torch.zeros_like(y[:, 0])
        d_N_Ki = torch.zeros_like(y[:, 0])
        d_N_HCO3i = torch.zeros_like(y[:, 0])
        d_N_Cli = torch.zeros_like(y[:, 0])
        d_N_Nao = torch.zeros_like(y[:, 0])
        d_N_Ko = torch.zeros_like(y[:, 0])
        d_N_HCO3o = torch.zeros_like(y[:, 0])
        alg_0_wo = torch.zeros_like(y[:, 0])
        alg_1_N_Clo = torch.zeros_like(y[:, 0])
        alg_2_ft = torch.zeros_like(y[:, 0])
        alg_3_ft_2 = torch.zeros_like(y[:, 0])
        alg_4_y = torch.zeros_like(y[:, 0])
        alg_5_Ki = torch.zeros_like(y[:, 0])
        alg_6_Ko = torch.zeros_like(y[:, 0])
        alg_7_EK = torch.zeros_like(y[:, 0])
        alg_8_Nai = torch.zeros_like(y[:, 0])
        alg_9_J_NaKATPase = torch.zeros_like(y[:, 0])
        alg_10_Nao = torch.zeros_like(y[:, 0])
        alg_11_ENa = torch.zeros_like(y[:, 0])
        alg_12_Cli = torch.zeros_like(y[:, 0])
        alg_13_Clo = torch.zeros_like(y[:, 0])
        alg_14_ECl = torch.zeros_like(y[:, 0])
        alg_15_J_KCC1 = torch.zeros_like(y[:, 0])
        alg_16_J_NKCC1 = torch.zeros_like(y[:, 0])
        alg_17_HCO3i = torch.zeros_like(y[:, 0])
        alg_18_HCO3o = torch.zeros_like(y[:, 0])
        alg_19_ENBC = torch.zeros_like(y[:, 0])
        alg_20_Vm = torch.zeros_like(y[:, 0])
        alg_21_dN_Ki_dt = torch.zeros_like(y[:, 0])
        alg_24_J_NBC = torch.zeros_like(y[:, 0])
        alg_25_dN_Nai_dt = torch.zeros_like(y[:, 0])
        alg_27_dN_HCO3i_dt = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_wo = w_tot - wi
        alg_10_Nao = torch.where(N_Nao / alg_0_wo > 0.0, N_Nao / alg_0_wo, 1e-180)
        alg_6_Ko = torch.where(N_Ko / alg_0_wo > 0.0, N_Ko / alg_0_wo, 1e-180)
        alg_1_N_Clo = N_Nao + N_Ko - N_HCO3o
        alg_13_Clo = torch.where(alg_1_N_Clo / alg_0_wo > 0.0, alg_1_N_Clo / alg_0_wo, 1e-180)
        alg_18_HCO3o = torch.where(N_HCO3o / alg_0_wo > 0.0, N_HCO3o / alg_0_wo, 1e-180)
        alg_8_Nai = torch.where(N_Nai / wi > 0.0, N_Nai / wi, 1e-180)
        alg_5_Ki = torch.where(N_Ki / wi > 0.0, N_Ki / wi, 1e-180)
        alg_12_Cli = torch.where(N_Cli / wi > 0.0, N_Cli / wi, 1e-180)
        alg_17_HCO3i = torch.where(N_HCO3i / wi > 0.0, N_HCO3i / wi, 1e-180)
        d_wi = torch.zeros_like(y[:, 0]) + (1.0 * Lp * (alg_8_Nai + alg_5_Ki + alg_12_Cli + alg_17_HCO3i + Xi / wi - (alg_10_Nao + alg_6_Ko + alg_13_Clo + alg_18_HCO3o)))
        alg_11_ENa = R * T / (1.0 * F) * torch.log(alg_10_Nao / alg_8_Nai)
        alg_9_J_NaKATPase = Pmax * alg_8_Nai ** 1.5 / (alg_8_Nai ** 1.5 + K_Nai ** 1.5) * alg_6_Ko / (alg_6_Ko + K_Ko)
        alg_7_EK = R * T / (1.0 * F) * torch.log(alg_6_Ko / alg_5_Ki)
        alg_14_ECl = R * T / (-1.0 * F) * torch.log(alg_13_Clo / alg_12_Cli)
        alg_19_ENBC = R * T / (zNBC * F) * torch.log(alg_10_Nao / alg_8_Nai * alg_18_HCO3o ** 2.0 / alg_17_HCO3i ** 2.0)
        alg_20_Vm = (gNa * alg_11_ENa + gK * alg_7_EK + gCl * alg_14_ECl + theta_NBC * gNBC * alg_19_ENBC - alg_9_J_NaKATPase * F) / (gNa + gK + gCl + theta_NBC * gNBC)
        alg_3_ft_2 = torch.where((t >= 10.0) & (t < 20.0), 1.0, 0.0)
        alg_16_J_NKCC1 = 10000000000.0 * gNKCC1 * alg_3_ft_2 / F * R * T / F * torch.log(alg_10_Nao / alg_8_Nai * alg_6_Ko / alg_5_Ki * (alg_13_Clo / alg_12_Cli) ** 2.0)
        alg_2_ft = torch.where((t >= 10.0) & (t < 20.0), 1.0, 0.0)
        alg_15_J_KCC1 = 10000000000.0 * gKCC1 * alg_2_ft / F * R * T / F * torch.log(alg_6_Ko * alg_13_Clo / (alg_5_Ki * alg_12_Cli))
        alg_4_y = torch.where((t >= t0) & (t <= t1), gamma_alpha_beta / (gamma_alpha * gamma_beta) * (1.0 - (t - t0) / deltaT) ** (beta - 1.0) * ((t - t0) / deltaT) ** (alpha - 1.0), torch.where((t > t1) & (t < t2), -1.0, 0.0))
        d_N_Ko = torch.zeros_like(y[:, 0]) + (10000000000.0 * gK / F * (alg_20_Vm - alg_7_EK) + kc * alg_4_y - (theta_NKCC1 * alg_16_J_NKCC1 + theta_KCC1 * alg_15_J_KCC1 + 10000000000.0 * 2.0 * alg_9_J_NaKATPase))
        alg_21_dN_Ki_dt = theta_NKCC1 * alg_16_J_NKCC1 + theta_KCC1 * alg_15_J_KCC1 + 10000000000.0 * 2.0 * alg_9_J_NaKATPase - 10000000000.0 * gK / F * (alg_20_Vm - alg_7_EK)
        d_N_Ki = torch.zeros_like(y[:, 0]) + (alg_21_dN_Ki_dt)
        alg_24_J_NBC = 10000000000.0 * gNBC / F * (alg_20_Vm - alg_19_ENBC)
        d_N_Nao = torch.zeros_like(y[:, 0]) + (10000000000.0 * (gNa / F * (alg_20_Vm - alg_11_ENa) + 3.0 * alg_9_J_NaKATPase) - (theta_NKCC1 * alg_16_J_NKCC1 + theta_NBC * alg_24_J_NBC + kc * alg_4_y))
        d_N_HCO3o = torch.zeros_like(y[:, 0]) + (-2.0 * theta_NBC * alg_24_J_NBC)
        alg_25_dN_Nai_dt = theta_NKCC1 * alg_16_J_NKCC1 + theta_NBC * alg_24_J_NBC - 10000000000.0 * (gNa / F * (alg_20_Vm - alg_11_ENa) + 3.0 * alg_9_J_NaKATPase)
        d_N_Nai = torch.zeros_like(y[:, 0]) + (alg_25_dN_Nai_dt)
        alg_27_dN_HCO3i_dt = 2.0 * theta_NBC * alg_24_J_NBC
        d_N_HCO3i = torch.zeros_like(y[:, 0]) + (alg_27_dN_HCO3i_dt)
        d_N_Cli = torch.zeros_like(y[:, 0]) + (alg_21_dN_Ki_dt + alg_25_dN_Nai_dt + alg_27_dN_HCO3i_dt)

        return torch.stack(
            [d_wi, d_N_Nai, d_N_Ki, d_N_HCO3i, d_N_Cli, d_N_Nao, d_N_Ko, d_N_HCO3o],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
