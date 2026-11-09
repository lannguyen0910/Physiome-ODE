import torch
import torch.nn as nn
import math


class CellCycleChenCsikaszNagyGyorffyValNovakTyson2000ODE(nn.Module):
    """
    PyTorch translation of cell_cycle/chen_csikasz-nagy_gyorffy_val_novak_tyson_2000/model.py.
    Dynamic states: 13; independent constants: 71; derived constants used by rates: 0.
    Original constant slots: 71; algebraic slots: 15.
    """
    source_file = 'cell_cycle/chen_csikasz-nagy_gyorffy_val_novak_tyson_2000/model.py'
    source_state_dim = 13
    source_constant_dim = 71
    algebraic_dim = 15
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
        self.parameter_dim = 71

        self.parameters = (
            parameters
            if parameters is not None
            else {'ks_n2': 0, 'ks_n2_2': 0.05, 'kd_n2': 0.1, 'Hct1_T': 1, 'ks_b2': 0.002, 'ks_b2_2': 0.05, 'kd_b2': 0.01, 'kd_b2_2': 2, 'kd_b2_3': 0.05, 'ks_b5': 0.006, 'ks_b5_2': 0.02, 'kd_b5': 0.1, 'kd_b5_2': 0.25, 'Bck2_0': 0.0027, 'Jn3': 6, 'Dn3': 1, 'Cln3_max': 0.02, 'ks_c1': 0.02, 'ks_c1_2': 0.1, 'kd1_c1': 0.01, 'Jd2_c1': 0.05, 'kas_b2': 50, 'kdi_b2': 0.05, 'kas_b5': 50, 'kdi_b5': 0.05, 'kd2_c1': 0.3, 'epsilonc1_n3': 20, 'epsilonc1_k2': 2, 'epsilonc1_b5': 1, 'epsilonc1_b2': 0.067, 'ks_20': 0.005, 'ks_20_2': 0.06, 'kd_20': 0.08, 'ka_20': 1, 'ki_20': 0.1, 'ki_20_2': 10, 'ka_t1': 0.04, 'ka_t1_2': 2, 'ki_t1': 0, 'ki_t1_2': 0.64, 'Ji_t1': 0.05, 'Ja_t1': 0.05, 'epsiloni_t1_n2': 1, 'epsiloni_t1_b5': 0.5, 'epsiloni_t1_b2': 1, 'mu': 0.005776, 'ks_ori': 2, 'kd_ori': 0.06, 'epsilonori_b2': 0.4, 'ks_bud': 0.3, 'kd_bud': 0.06, 'epsilonbud_b5': 1, 'ks_spn': 0.08, 'kd_spn': 0.06, 'J_spn': 0.2, 'ka_sbf': 1, 'ki_sbf': 0.5, 'ki_sbf_2': 6, 'Ji_sbf': 0.01, 'Ja_sbf': 0.01, 'epsilonsbf_n3': 75, 'epsilonsbf_b5': 0.5, 'ka_mcm': 1, 'ki_mcm': 0.15, 'Ji_mcm': 1, 'Ja_mcm': 1, 'ka_swi': 1, 'ki_swi': 0.3, 'ki_swi_2': 0.2, 'Ji_swi': 0.1, 'Ja_swi': 0.1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Cln2': 0.0078, 'mass': 0.6608, 'Clb2_T': 0.2342, 'Hct1': 0.9946, 'Cdc20': 0.6848, 'Clb2_Sic1': 0.079, 'Clb5_Sic1': 0.0207, 'Clb5_T': 0.0614, 'Sic1_T': 0.1231, 'Cdc20_T': 0.8332, 'ORI': 0, 'SPN': 0, 'BUD': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['ks_n2', 'ks_n2_2', 'kd_n2', 'Hct1_T', 'ks_b2', 'ks_b2_2', 'kd_b2', 'kd_b2_2', 'kd_b2_3', 'ks_b5', 'ks_b5_2', 'kd_b5', 'kd_b5_2', 'Bck2_0', 'Jn3', 'Dn3', 'Cln3_max', 'ks_c1', 'ks_c1_2', 'kd1_c1', 'Jd2_c1', 'kas_b2', 'kdi_b2', 'kas_b5', 'kdi_b5', 'kd2_c1', 'epsilonc1_n3', 'epsilonc1_k2', 'epsilonc1_b5', 'epsilonc1_b2', 'ks_20', 'ks_20_2', 'kd_20', 'ka_20', 'ki_20', 'ki_20_2', 'ka_t1', 'ka_t1_2', 'ki_t1', 'ki_t1_2', 'Ji_t1', 'Ja_t1', 'epsiloni_t1_n2', 'epsiloni_t1_b5', 'epsiloni_t1_b2', 'mu', 'ks_ori', 'kd_ori', 'epsilonori_b2', 'ks_bud', 'kd_bud', 'epsilonbud_b5', 'ks_spn', 'kd_spn', 'J_spn', 'ka_sbf', 'ki_sbf', 'ki_sbf_2', 'Ji_sbf', 'Ja_sbf', 'epsilonsbf_n3', 'epsilonsbf_b5', 'ka_mcm', 'ki_mcm', 'Ji_mcm', 'Ja_mcm', 'ka_swi', 'ki_swi', 'ki_swi_2', 'Ji_swi', 'Ja_swi']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Cln2', 'mass', 'Clb2_T', 'Hct1', 'Cdc20', 'Clb2_Sic1', 'Clb5_Sic1', 'Clb5_T', 'Sic1_T', 'Cdc20_T', 'ORI', 'SPN', 'BUD']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (13)
        Cln2 = y[:, 0]
        mass = y[:, 1]
        Clb2_T = y[:, 2]
        Hct1 = y[:, 3]
        Cdc20 = y[:, 4]
        Clb2_Sic1 = y[:, 5]
        Clb5_Sic1 = y[:, 6]
        Clb5_T = y[:, 7]
        Sic1_T = y[:, 8]
        Cdc20_T = y[:, 9]
        ORI = y[:, 10]
        SPN = y[:, 11]
        BUD = y[:, 12]

        # Independent parameters (source constant order)
        ks_n2 = theta[:, 0]  # constants[0]
        ks_n2_2 = theta[:, 1]  # constants[1]
        kd_n2 = theta[:, 2]  # constants[2]
        Hct1_T = theta[:, 3]  # constants[3]
        ks_b2 = theta[:, 4]  # constants[4]
        ks_b2_2 = theta[:, 5]  # constants[5]
        kd_b2 = theta[:, 6]  # constants[6]
        kd_b2_2 = theta[:, 7]  # constants[7]
        kd_b2_3 = theta[:, 8]  # constants[8]
        ks_b5 = theta[:, 9]  # constants[9]
        ks_b5_2 = theta[:, 10]  # constants[10]
        kd_b5 = theta[:, 11]  # constants[11]
        kd_b5_2 = theta[:, 12]  # constants[12]
        Bck2_0 = theta[:, 13]  # constants[13]
        Jn3 = theta[:, 14]  # constants[14]
        Dn3 = theta[:, 15]  # constants[15]
        Cln3_max = theta[:, 16]  # constants[16]
        ks_c1 = theta[:, 17]  # constants[17]
        ks_c1_2 = theta[:, 18]  # constants[18]
        kd1_c1 = theta[:, 19]  # constants[19]
        Jd2_c1 = theta[:, 20]  # constants[20]
        kas_b2 = theta[:, 21]  # constants[21]
        kdi_b2 = theta[:, 22]  # constants[22]
        kas_b5 = theta[:, 23]  # constants[23]
        kdi_b5 = theta[:, 24]  # constants[24]
        kd2_c1 = theta[:, 25]  # constants[25]
        epsilonc1_n3 = theta[:, 26]  # constants[26]
        epsilonc1_k2 = theta[:, 27]  # constants[27]
        epsilonc1_b5 = theta[:, 28]  # constants[28]
        epsilonc1_b2 = theta[:, 29]  # constants[29]
        ks_20 = theta[:, 30]  # constants[30]
        ks_20_2 = theta[:, 31]  # constants[31]
        kd_20 = theta[:, 32]  # constants[32]
        ka_20 = theta[:, 33]  # constants[33]
        ki_20 = theta[:, 34]  # constants[34]
        ki_20_2 = theta[:, 35]  # constants[35]
        ka_t1 = theta[:, 36]  # constants[36]
        ka_t1_2 = theta[:, 37]  # constants[37]
        ki_t1 = theta[:, 38]  # constants[38]
        ki_t1_2 = theta[:, 39]  # constants[39]
        Ji_t1 = theta[:, 40]  # constants[40]
        Ja_t1 = theta[:, 41]  # constants[41]
        epsiloni_t1_n2 = theta[:, 42]  # constants[42]
        epsiloni_t1_b5 = theta[:, 43]  # constants[43]
        epsiloni_t1_b2 = theta[:, 44]  # constants[44]
        mu = theta[:, 45]  # constants[45]
        ks_ori = theta[:, 46]  # constants[46]
        kd_ori = theta[:, 47]  # constants[47]
        epsilonori_b2 = theta[:, 48]  # constants[48]
        ks_bud = theta[:, 49]  # constants[49]
        kd_bud = theta[:, 50]  # constants[50]
        epsilonbud_b5 = theta[:, 51]  # constants[51]
        ks_spn = theta[:, 52]  # constants[52]
        kd_spn = theta[:, 53]  # constants[53]
        J_spn = theta[:, 54]  # constants[54]
        ka_sbf = theta[:, 55]  # constants[55]
        ki_sbf = theta[:, 56]  # constants[56]
        ki_sbf_2 = theta[:, 57]  # constants[57]
        Ji_sbf = theta[:, 58]  # constants[58]
        Ja_sbf = theta[:, 59]  # constants[59]
        epsilonsbf_n3 = theta[:, 60]  # constants[60]
        epsilonsbf_b5 = theta[:, 61]  # constants[61]
        ka_mcm = theta[:, 62]  # constants[62]
        ki_mcm = theta[:, 63]  # constants[63]
        Ji_mcm = theta[:, 64]  # constants[64]
        Ja_mcm = theta[:, 65]  # constants[65]
        ka_swi = theta[:, 66]  # constants[66]
        ki_swi = theta[:, 67]  # constants[67]
        ki_swi_2 = theta[:, 68]  # constants[68]
        Ji_swi = theta[:, 69]  # constants[69]
        Ja_swi = theta[:, 70]  # constants[70]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Cln2 = torch.zeros_like(y[:, 0])
        d_mass = torch.zeros_like(y[:, 0])
        d_Clb2_T = torch.zeros_like(y[:, 0])
        d_Hct1 = torch.zeros_like(y[:, 0])
        d_Cdc20 = torch.zeros_like(y[:, 0])
        d_Clb2_Sic1 = torch.zeros_like(y[:, 0])
        d_Clb5_Sic1 = torch.zeros_like(y[:, 0])
        d_Clb5_T = torch.zeros_like(y[:, 0])
        d_Sic1_T = torch.zeros_like(y[:, 0])
        d_Cdc20_T = torch.zeros_like(y[:, 0])
        d_ORI = torch.zeros_like(y[:, 0])
        d_SPN = torch.zeros_like(y[:, 0])
        d_BUD = torch.zeros_like(y[:, 0])
        alg_0_Vd_b2 = torch.zeros_like(y[:, 0])
        alg_1_Vi_20 = torch.zeros_like(y[:, 0])
        alg_2_Clb2 = torch.zeros_like(y[:, 0])
        alg_3_Mcm1 = torch.zeros_like(y[:, 0])
        alg_4_Clb5 = torch.zeros_like(y[:, 0])
        alg_5_Sic1 = torch.zeros_like(y[:, 0])
        alg_6_Vd_b5 = torch.zeros_like(y[:, 0])
        alg_7_Bck2 = torch.zeros_like(y[:, 0])
        alg_8_Cln3 = torch.zeros_like(y[:, 0])
        alg_9_Va_sbf = torch.zeros_like(y[:, 0])
        alg_10_Vd2_c1 = torch.zeros_like(y[:, 0])
        alg_11_Vi_t1 = torch.zeros_like(y[:, 0])
        alg_12_SBF = torch.zeros_like(y[:, 0])
        alg_13_Swi5 = torch.zeros_like(y[:, 0])
        alg_14_MBF = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_mass = torch.zeros_like(y[:, 0]) + (mu * mass)
        alg_1_Vi_20 = torch.where(ORI >= 1.0, ki_20_2, torch.where(SPN >= 1.0, ki_20, 0.1))
        d_Cdc20 = torch.zeros_like(y[:, 0]) + (ka_20 * (Cdc20_T - Cdc20) - Cdc20 * (alg_1_Vi_20 + kd_20))
        alg_2_Clb2 = Clb2_T - Clb2_Sic1
        d_Cdc20_T = torch.zeros_like(y[:, 0]) + (ks_20 + ks_20_2 * alg_2_Clb2 - kd_20 * Cdc20_T)
        d_SPN = torch.zeros_like(y[:, 0]) + (ks_spn * alg_2_Clb2 / (J_spn + alg_2_Clb2) - kd_spn * SPN)
        alg_0_Vd_b2 = kd_b2 * (Hct1_T - Hct1) + kd_b2_2 * Hct1 + kd_b2_3 * Cdc20
        alg_3_Mcm1 = 2.0 * ka_mcm * alg_2_Clb2 * Ji_mcm / (ki_mcm + ka_mcm * alg_2_Clb2 * Ji_mcm + ki_mcm * Ja_mcm - ka_mcm * alg_2_Clb2 + ((ki_mcm + ka_mcm * alg_2_Clb2 * Ji_mcm + ki_mcm * Ja_mcm - ka_mcm * alg_2_Clb2) ** 2.0 - 4.0 * (ki_mcm - ka_mcm * alg_2_Clb2) * ka_mcm * alg_2_Clb2 * Ji_mcm) ** (1.0 / 2))
        d_Clb2_T = torch.zeros_like(y[:, 0]) + (mass * (ks_b2 + ks_b2_2 * alg_3_Mcm1) - alg_0_Vd_b2 * Clb2_T)
        alg_4_Clb5 = Clb5_T - Clb5_Sic1
        d_ORI = torch.zeros_like(y[:, 0]) + (ks_ori * (alg_4_Clb5 + epsilonori_b2 * alg_2_Clb2) - kd_ori * ORI)
        alg_8_Cln3 = Cln3_max * Dn3 * mass / (Jn3 + Dn3 * mass)
        d_BUD = torch.zeros_like(y[:, 0]) + (ks_bud * (Cln2 + alg_8_Cln3 + epsilonbud_b5 * alg_4_Clb5) - kd_bud * BUD)
        alg_5_Sic1 = Sic1_T - (Clb2_Sic1 + Clb5_Sic1)
        alg_7_Bck2 = Bck2_0 * mass
        alg_10_Vd2_c1 = kd2_c1 * (epsilonc1_n3 * alg_8_Cln3 + epsilonc1_k2 * alg_7_Bck2 + Cln2 + epsilonc1_b5 * alg_4_Clb5 + epsilonc1_b2 * alg_2_Clb2)
        d_Clb2_Sic1 = torch.zeros_like(y[:, 0]) + (kas_b2 * alg_2_Clb2 * alg_5_Sic1 - Clb2_Sic1 * (kdi_b2 + alg_0_Vd_b2 + kd1_c1 + alg_10_Vd2_c1 / (Jd2_c1 + Sic1_T)))
        alg_6_Vd_b5 = kd_b5 + kd_b5_2 * Cdc20
        d_Clb5_Sic1 = torch.zeros_like(y[:, 0]) + (kas_b5 * alg_4_Clb5 * alg_5_Sic1 - Clb5_Sic1 * (kdi_b5 + alg_6_Vd_b5 + kd1_c1 + alg_10_Vd2_c1 / (Jd2_c1 + Sic1_T)))
        alg_11_Vi_t1 = ki_t1 + ki_t1_2 * (alg_8_Cln3 + epsiloni_t1_n2 * Cln2 + epsiloni_t1_b5 * alg_4_Clb5 + epsiloni_t1_b2 * alg_2_Clb2)
        d_Hct1 = torch.zeros_like(y[:, 0]) + ((ka_t1 + ka_t1_2 * Cdc20) * (Hct1_T - Hct1) / (Ja_t1 + Hct1_T - Hct1) - alg_11_Vi_t1 * Hct1 / (Ji_t1 + Hct1))
        alg_9_Va_sbf = ka_sbf * (Cln2 + epsilonsbf_n3 * (alg_8_Cln3 + alg_7_Bck2) + epsilonsbf_b5 * alg_4_Clb5)
        alg_12_SBF = 2.0 * alg_9_Va_sbf * Ji_sbf / (ki_sbf + ki_sbf_2 * alg_2_Clb2 + alg_9_Va_sbf * Ji_sbf + (ki_sbf + ki_sbf_2 * alg_2_Clb2) * Ja_sbf - alg_9_Va_sbf + ((ki_sbf + ki_sbf_2 * alg_2_Clb2 + alg_9_Va_sbf * Ji_sbf + (ki_sbf + ki_sbf_2 * alg_2_Clb2) * Ja_sbf - alg_9_Va_sbf) ** 2.0 - 4.0 * alg_9_Va_sbf * Ji_sbf * (ki_sbf + ki_sbf_2 * alg_2_Clb2 - alg_9_Va_sbf)) ** (1.0 / 2))
        d_Cln2 = torch.zeros_like(y[:, 0]) + (mass * (ks_n2 + ks_n2_2 * alg_12_SBF) - kd_n2 * Cln2)
        alg_13_Swi5 = 2.0 * ka_swi * Cdc20 * Ji_swi / (ki_swi + ki_swi_2 * alg_2_Clb2 + ka_swi * Cdc20 * Ji_swi + (ki_swi + ki_swi_2 * alg_2_Clb2) * Ja_swi - ka_swi * Cdc20 + ((ki_swi + ki_swi_2 * alg_2_Clb2 + ka_swi * Cdc20 * Ji_swi + (ki_swi + ki_swi_2 * alg_2_Clb2) * Ja_swi - ka_swi * Cdc20) ** 2.0 - 4.0 * (ki_swi + ki_swi_2 * alg_2_Clb2 - ka_swi * Cdc20) * ka_swi * Cdc20 * Ji_swi) ** (1.0 / 2))
        d_Sic1_T = torch.zeros_like(y[:, 0]) + (ks_c1 + ks_c1_2 * alg_13_Swi5 - Sic1_T * (kd1_c1 + alg_10_Vd2_c1 / (Jd2_c1 + Sic1_T)))
        alg_14_MBF = alg_12_SBF
        d_Clb5_T = torch.zeros_like(y[:, 0]) + (mass * (ks_b5 + ks_b5_2 * alg_14_MBF) - alg_6_Vd_b5 * Clb5_T)

        return torch.stack(
            [d_Cln2, d_mass, d_Clb2_T, d_Hct1, d_Cdc20, d_Clb2_Sic1, d_Clb5_Sic1, d_Clb5_T, d_Sic1_T, d_Cdc20_T, d_ORI, d_SPN, d_BUD],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
