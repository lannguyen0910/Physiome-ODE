import torch
import torch.nn as nn
import math


class CalciumDynamicsSnyderPalmerMoore2000ODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/snyder_palmer_moore_2000/model.py.
    Dynamic states: 10; independent constants: 27; derived constants used by rates: 0.
    Original constant slots: 27; algebraic slots: 11.
    """
    source_file = 'calcium_dynamics/snyder_palmer_moore_2000/model.py'
    source_state_dim = 10
    source_constant_dim = 27
    algebraic_dim = 11
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 10
        self.parameter_dim = 27

        self.parameters = (
            parameters
            if parameters is not None
            else {'Ca_e': 0.002, 'k1': 0.2, 'k_on1': 2000000000, 'k_off1': 1400, 'k_on2': 13000000, 'k_off2': 3.9, 'k_s': 9, 'Km_NaCaX': 3.6e-05, 'Vmax_NaCaX': 0.0012, 'kf': 2500, 'Km_s': 2.5e-07, 'Vmax_s': 0.000525, 'K_ons': 8772, 'K_offs': 5.596536, 'Bmax_s': 0.008, 'Rt': 1.5e-07, 'Bmax_f1': 0.0002, 'Bmax_f2': 0.0011, 'Kb_f1': 1.7e-05, 'Kb_f2': 1.3e-05, 'V_f': 0.0013, 'Bmax_c': 0.00012, 'dye_c': 0, 'Kb_c': 9.6e-07, 'Kb_dye': 2e-07, 'V_c': 0.9287, 'V_s': 0.07}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Ca_f': 1.2e-07, 'Ca_2_S1': 0, 'Ca_S2': 0, 'S1': 0, 'S2': 0, 'Ca_s': 0.000201, 'cas1': 0, 'cas2': 0, 'Ca_c': 1e-07, 'Ca_CSQ': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Ca_e', 'k1', 'k_on1', 'k_off1', 'k_on2', 'k_off2', 'k_s', 'Km_NaCaX', 'Vmax_NaCaX', 'kf', 'Km_s', 'Vmax_s', 'K_ons', 'K_offs', 'Bmax_s', 'Rt', 'Bmax_f1', 'Bmax_f2', 'Kb_f1', 'Kb_f2', 'V_f', 'Bmax_c', 'dye_c', 'Kb_c', 'Kb_dye', 'V_c', 'V_s']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Ca_f', 'Ca_2_S1', 'Ca_S2', 'S1', 'S2', 'Ca_s', 'cas1', 'cas2', 'Ca_c', 'Ca_CSQ']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (10)
        Ca_f = y[:, 0]
        Ca_2_S1 = y[:, 1]
        Ca_S2 = y[:, 2]
        S1 = y[:, 3]
        S2 = y[:, 4]
        Ca_s = y[:, 5]
        cas1 = y[:, 6]
        cas2 = y[:, 7]
        Ca_c = y[:, 8]
        Ca_CSQ = y[:, 9]

        # Independent parameters (source constant order)
        Ca_e = theta[:, 0]  # constants[0]
        k1 = theta[:, 1]  # constants[1]
        k_on1 = theta[:, 2]  # constants[2]
        k_off1 = theta[:, 3]  # constants[3]
        k_on2 = theta[:, 4]  # constants[4]
        k_off2 = theta[:, 5]  # constants[5]
        k_s = theta[:, 6]  # constants[6]
        Km_NaCaX = theta[:, 7]  # constants[7]
        Vmax_NaCaX = theta[:, 8]  # constants[8]
        kf = theta[:, 9]  # constants[9]
        Km_s = theta[:, 10]  # constants[10]
        Vmax_s = theta[:, 11]  # constants[11]
        K_ons = theta[:, 12]  # constants[12]
        K_offs = theta[:, 13]  # constants[13]
        Bmax_s = theta[:, 14]  # constants[14]
        Rt = theta[:, 15]  # constants[15]
        Bmax_f1 = theta[:, 16]  # constants[16]
        Bmax_f2 = theta[:, 17]  # constants[17]
        Kb_f1 = theta[:, 18]  # constants[18]
        Kb_f2 = theta[:, 19]  # constants[19]
        V_f = theta[:, 20]  # constants[20]
        Bmax_c = theta[:, 21]  # constants[21]
        dye_c = theta[:, 22]  # constants[22]
        Kb_c = theta[:, 23]  # constants[23]
        Kb_dye = theta[:, 24]  # constants[24]
        V_c = theta[:, 25]  # constants[25]
        V_s = theta[:, 26]  # constants[26]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Ca_f = torch.zeros_like(y[:, 0])
        d_Ca_2_S1 = torch.zeros_like(y[:, 0])
        d_Ca_S2 = torch.zeros_like(y[:, 0])
        d_S1 = torch.zeros_like(y[:, 0])
        d_S2 = torch.zeros_like(y[:, 0])
        d_Ca_s = torch.zeros_like(y[:, 0])
        d_cas1 = torch.zeros_like(y[:, 0])
        d_cas2 = torch.zeros_like(y[:, 0])
        d_Ca_c = torch.zeros_like(y[:, 0])
        d_Ca_CSQ = torch.zeros_like(y[:, 0])
        alg_0_dCa2_S1_dt = torch.zeros_like(y[:, 0])
        alg_1_dCa_S2_dt = torch.zeros_like(y[:, 0])
        alg_2_Rate_Ca_influx_across_the_SR = torch.zeros_like(y[:, 0])
        alg_3_r_o = torch.zeros_like(y[:, 0])
        alg_4_dcas1_dt = torch.zeros_like(y[:, 0])
        alg_5_dcas2_dt = torch.zeros_like(y[:, 0])
        alg_6_Rate_Ca_movement_through_the_SRRC = torch.zeros_like(y[:, 0])
        alg_7_Rate_Ca_efflux_across_the_SR_by_NaCa_exchange = torch.zeros_like(y[:, 0])
        alg_8_Rate_Ca_movement_between_the_fuzzy_space_and_cytosol = torch.zeros_like(y[:, 0])
        alg_9_Rate_Ca_uptake_by_SR_Ca_ATPase = torch.zeros_like(y[:, 0])
        alg_10_Rate_Ca_buffering_in_the_SR = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_dCa2_S1_dt = k_on1 * Ca_f * S1 - k_off1 ** 2.0 / (k_on1 * Ca_f) * Ca_2_S1
        d_S1 = torch.zeros_like(y[:, 0]) + (-alg_0_dCa2_S1_dt)
        alg_1_dCa_S2_dt = k_on2 * Ca_f * S2 - k_off2 * Ca_S2
        d_S2 = torch.zeros_like(y[:, 0]) + (-alg_1_dCa_S2_dt)
        d_Ca_2_S1 = torch.zeros_like(y[:, 0]) + (alg_0_dCa2_S1_dt)
        d_Ca_S2 = torch.zeros_like(y[:, 0]) + (alg_1_dCa_S2_dt)
        alg_4_dcas1_dt = k_on1 * Ca_f * (1.0 - cas1) - k_off1 ** 2.0 / (k_on1 * Ca_f) * cas1
        d_cas1 = torch.zeros_like(y[:, 0]) + (alg_4_dcas1_dt)
        alg_5_dcas2_dt = k_on2 * Ca_f * (1.0 - cas2) - k_off2 * cas2
        d_cas2 = torch.zeros_like(y[:, 0]) + (alg_5_dcas2_dt)
        alg_2_Rate_Ca_influx_across_the_SR = k1 * (Ca_e - Ca_f)
        alg_3_r_o = cas1 * (1.0 - cas2)
        alg_6_Rate_Ca_movement_through_the_SRRC = k_s * alg_3_r_o * (Ca_s - Ca_f)
        alg_7_Rate_Ca_efflux_across_the_SR_by_NaCa_exchange = Vmax_NaCaX * Ca_f / (Km_NaCaX + Ca_f)
        alg_8_Rate_Ca_movement_between_the_fuzzy_space_and_cytosol = kf * (Ca_f - Ca_c)
        d_Ca_f = torch.zeros_like(y[:, 0]) + ((alg_6_Rate_Ca_movement_through_the_SRRC - (Rt * (alg_4_dcas1_dt + alg_5_dcas2_dt) + alg_8_Rate_Ca_movement_between_the_fuzzy_space_and_cytosol + alg_7_Rate_Ca_efflux_across_the_SR_by_NaCa_exchange) + alg_2_Rate_Ca_influx_across_the_SR) / (Bmax_f1 * Kb_f1 / (Ca_f + Kb_f1) ** 2.0 + Bmax_f2 * Kb_f2 / (Ca_f + Kb_f2) ** 2.0 + V_f))
        alg_9_Rate_Ca_uptake_by_SR_Ca_ATPase = Vmax_s * (Ca_c ** 2.0 - Ca_s ** 2.0 / 49000000.0) / (Km_s ** 2.0 + Ca_c ** 2.0 + Ca_s ** 2.0 / 49000000.0)
        d_Ca_c = torch.zeros_like(y[:, 0]) + ((alg_8_Rate_Ca_movement_between_the_fuzzy_space_and_cytosol - alg_9_Rate_Ca_uptake_by_SR_Ca_ATPase) / (Bmax_c * Kb_c / (Ca_c + Kb_c) ** 2.0 + dye_c * Kb_dye / (Ca_c + Kb_dye) ** 2.0 + V_c))
        alg_10_Rate_Ca_buffering_in_the_SR = K_ons * Ca_s * (Bmax_s - Ca_CSQ) - K_offs * Ca_CSQ
        d_Ca_s = torch.zeros_like(y[:, 0]) + ((alg_9_Rate_Ca_uptake_by_SR_Ca_ATPase - alg_6_Rate_Ca_movement_through_the_SRRC) / V_s - alg_10_Rate_Ca_buffering_in_the_SR)
        d_Ca_CSQ = torch.zeros_like(y[:, 0]) + (alg_10_Rate_Ca_buffering_in_the_SR)

        return torch.stack(
            [d_Ca_f, d_Ca_2_S1, d_Ca_S2, d_S1, d_S2, d_Ca_s, d_cas1, d_cas2, d_Ca_c, d_Ca_CSQ],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
