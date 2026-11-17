import torch
import torch.nn as nn
import math


class ElectrophysiologyNiedererHunterSmith2006ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/niederer_hunter_smith_2006/model.py.
    Dynamic states: 5; independent constants: 23; derived constants used by rates: 2.
    Original constant slots: 25; algebraic slots: 17.
    """
    source_file = 'electrophysiology/niederer_hunter_smith_2006/model.py'
    source_state_dim = 5
    source_constant_dim = 25
    algebraic_dim = 17
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 5
        self.parameter_dim = 23

        self.parameters = (
            parameters
            if parameters is not None
            else {'Ca_TRPN_Max': 0.07, 'k_on': 100, 'k_Ref_off': 0.2, 'gamma_trpn': 2, 'alpha_0': 0.008, 'alpha_r1': 0.002, 'alpha_r2': 0.00175, 'n_Rel': 3, 'K_z': 0.15, 'n_Hill': 3, 'Ca_50ref': 0.00105, 'z_p': 0.85, 'beta_1': -4, 'beta_0': 4.9, 'T_ref': 56.2, 'a': 0.35, 'A_1': -29, 'A_2': 138, 'A_3': 129, 'alpha_1': 0.03, 'alpha_2': 0.13, 'alpha_3': 0.625, 'dExtensionRatiodt': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'TRPN': 0.067593139865, 'z': 0.014417937837, 'Q_1': 0, 'Q_2': 0, 'Q_3': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Ca_TRPN_Max', 'k_on', 'k_Ref_off', 'gamma_trpn', 'alpha_0', 'alpha_r1', 'alpha_r2', 'n_Rel', 'K_z', 'n_Hill', 'Ca_50ref', 'z_p', 'beta_1', 'beta_0', 'T_ref', 'a', 'A_1', 'A_2', 'A_3', 'alpha_1', 'alpha_2', 'alpha_3', 'dExtensionRatiodt']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['TRPN', 'z', 'Q_1', 'Q_2', 'Q_3']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        TRPN = y[:, 0]
        z = y[:, 1]
        Q_1 = y[:, 2]
        Q_2 = y[:, 3]
        Q_3 = y[:, 4]

        # Independent parameters (source constant order)
        Ca_TRPN_Max = theta[:, 0]  # constants[0]
        k_on = theta[:, 1]  # constants[1]
        k_Ref_off = theta[:, 2]  # constants[2]
        gamma_trpn = theta[:, 3]  # constants[3]
        alpha_0 = theta[:, 4]  # constants[4]
        alpha_r1 = theta[:, 5]  # constants[5]
        alpha_r2 = theta[:, 6]  # constants[6]
        n_Rel = theta[:, 7]  # constants[7]
        K_z = theta[:, 8]  # constants[8]
        n_Hill = theta[:, 9]  # constants[9]
        Ca_50ref = theta[:, 10]  # constants[10]
        z_p = theta[:, 11]  # constants[11]
        beta_1 = theta[:, 12]  # constants[12]
        beta_0 = theta[:, 13]  # constants[13]
        T_ref = theta[:, 14]  # constants[14]
        a = theta[:, 15]  # constants[15]
        A_1 = theta[:, 16]  # constants[16]
        A_2 = theta[:, 17]  # constants[17]
        A_3 = theta[:, 18]  # constants[18]
        alpha_1 = theta[:, 19]  # constants[19]
        alpha_2 = theta[:, 20]  # constants[20]
        alpha_3 = theta[:, 21]  # constants[21]
        dExtensionRatiodt = theta[:, 22]  # constants[23]

        # Derived constants from the original initConsts()
        K_2 = alpha_r2 * z_p ** n_Rel / (z_p ** n_Rel + K_z ** n_Rel) * (1.0 - n_Rel * K_z ** n_Rel / (z_p ** n_Rel + K_z ** n_Rel))  # constants[22]
        K_1 = alpha_r2 * z_p ** (n_Rel - 1.0) * n_Rel * K_z ** n_Rel / (z_p ** n_Rel + K_z ** n_Rel) ** 2.0  # constants[24]

        # Initialize rates and algebraic variables, as in computeRates()
        d_TRPN = torch.zeros_like(y[:, 0])
        d_z = torch.zeros_like(y[:, 0])
        d_Q_1 = torch.zeros_like(y[:, 0])
        d_Q_2 = torch.zeros_like(y[:, 0])
        d_Q_3 = torch.zeros_like(y[:, 0])
        alg_0_Ca_b = torch.zeros_like(y[:, 0])
        alg_1_Ca_i = torch.zeros_like(y[:, 0])
        alg_2_beta_Tm = torch.zeros_like(y[:, 0])
        alg_3_ExtensionRatio = torch.zeros_like(y[:, 0])
        alg_4_lambda_var = torch.zeros_like(y[:, 0])
        alg_6_Ca_50 = torch.zeros_like(y[:, 0])
        alg_7_Ca_TRPN_50 = torch.zeros_like(y[:, 0])
        alg_8_alpha_Tm = torch.zeros_like(y[:, 0])
        alg_9_z_max = torch.zeros_like(y[:, 0])
        alg_10_overlap = torch.zeros_like(y[:, 0])
        alg_11_T_Base = torch.zeros_like(y[:, 0])
        alg_12_T_0 = torch.zeros_like(y[:, 0])
        alg_13_Q = torch.zeros_like(y[:, 0])
        alg_14_Tension = torch.zeros_like(y[:, 0])
        alg_15_k_off = torch.zeros_like(y[:, 0])
        alg_16_J_TRPN = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Q_1 = torch.zeros_like(y[:, 0]) + (A_1 * dExtensionRatiodt - alpha_1 * Q_1)
        d_Q_2 = torch.zeros_like(y[:, 0]) + (A_2 * dExtensionRatiodt - alpha_2 * Q_2)
        d_Q_3 = torch.zeros_like(y[:, 0]) + (A_3 * dExtensionRatiodt - alpha_3 * Q_3)
        alg_0_Ca_b = Ca_TRPN_Max - TRPN
        alg_3_ExtensionRatio = torch.where(t > 1000.0, 1.0, 1.0)
        alg_4_lambda_var = torch.where((alg_3_ExtensionRatio > 0.8) & (alg_3_ExtensionRatio <= 1.15), alg_3_ExtensionRatio, torch.where(alg_3_ExtensionRatio > 1.15, 1.15, 0.8))
        alg_6_Ca_50 = Ca_50ref * (1.0 + beta_1 * (alg_4_lambda_var - 1.0))
        alg_7_Ca_TRPN_50 = alg_6_Ca_50 * Ca_TRPN_Max / (alg_6_Ca_50 + k_Ref_off / k_on * (1.0 - (1.0 + beta_0 * (alg_4_lambda_var - 1.0)) * 0.5 / gamma_trpn))
        alg_8_alpha_Tm = alpha_0 * (alg_0_Ca_b / alg_7_Ca_TRPN_50) ** n_Hill
        alg_2_beta_Tm = alpha_r1 + alpha_r2 * z ** (n_Rel - 1.0) / (z ** n_Rel + K_z ** n_Rel)
        d_z = torch.zeros_like(y[:, 0]) + (alg_8_alpha_Tm * (1.0 - z) - alg_2_beta_Tm * z)
        alg_1_Ca_i = torch.where(t < 1.0, 1000.0 * 1.8433e-07, torch.where((t >= 10.0) & (t < 15.0), 1000.0 * (1.055 * (t / 1000.0) ** 3.0 - 0.03507 * (t / 1000.0) ** 2.0 + 0.0003992 * t / 1000.0 - 1.356e-06), torch.where((t >= 15.0) & (t < 55.0), 1000.0 * (0.014 * (t / 1000.0) ** 3.0 - 0.002555 * (t / 1000.0) ** 2.0 + 0.0001494 * t / 1000.0 - 1.428e-06), torch.where((t >= 55.0) & (t < 250.0), 1000.0 * (1.739e-05 * (t / 1000.0) ** 3.0 - 3.209e-06 * (t / 1000.0) ** 2.0 - 5.689e-06 * t / 1000.0 + 1.719e-06), torch.where((t >= 250.0) & (t < 490.0), 1000.0 * (0.0001321 * (t / 1000.0) ** 4.0 - 0.0002197 * (t / 1000.0) ** 3.0 + 0.0001374 * (t / 1000.0) ** 2.0 - 3.895e-05 * t / 1000.0 + 4.441e-06), 1000.0 * 1.2148e-07)))))
        alg_10_overlap = 1.0 + beta_0 * (alg_4_lambda_var - 1.0)
        alg_9_z_max = (alpha_0 / (alg_7_Ca_TRPN_50 / Ca_TRPN_Max) ** n_Hill - K_2) / (alpha_r1 + K_1 + alpha_0 / (alg_7_Ca_TRPN_50 / Ca_TRPN_Max) ** n_Hill)
        alg_11_T_Base = T_ref * z / alg_9_z_max
        alg_12_T_0 = alg_11_T_Base * alg_10_overlap
        alg_13_Q = Q_1 + Q_2 + Q_3
        alg_14_Tension = torch.where(alg_13_Q < 0.0, alg_12_T_0 * (a * alg_13_Q + 1.0) / (1.0 - alg_13_Q), alg_12_T_0 * (1.0 + (a + 2.0) * alg_13_Q) / (1.0 + alg_13_Q))
        alg_15_k_off = torch.where(1.0 - alg_14_Tension / (gamma_trpn * T_ref) > 0.1, k_Ref_off * (1.0 - alg_14_Tension / (gamma_trpn * T_ref)), k_Ref_off * 0.1)
        alg_16_J_TRPN = (Ca_TRPN_Max - TRPN) * alg_15_k_off - alg_1_Ca_i * TRPN * k_on
        d_TRPN = torch.zeros_like(y[:, 0]) + (alg_16_J_TRPN)

        return torch.stack(
            [d_TRPN, d_z, d_Q_1, d_Q_2, d_Q_3],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
