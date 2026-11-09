import torch
import torch.nn as nn
import math


class CellCycleCilibertoNovakTyson2003ODE(nn.Module):
    """
    PyTorch translation of cell_cycle/ciliberto_novak_tyson_2003/model.py.
    Dynamic states: 19; independent constants: 69; derived constants used by rates: 0.
    Original constant slots: 69; algebraic slots: 9.
    """
    source_file = 'cell_cycle/ciliberto_novak_tyson_2003/model.py'
    source_state_dim = 19
    source_constant_dim = 69
    algebraic_dim = 9
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 19
        self.parameter_dim = 69

        self.parameters = (
            parameters
            if parameters is not None
            else {'epsilon': 0.5, 'Jm': 10, 'ks_clb': 0.015, 'kd_clb_1': 0.015, 'kd_clb_2': 1, 'kd_clb_3': 0.1, 'kass': 300, 'kdiss': 0.1, 'kd_sic': 0.01, 'kd_sic_1': 1, 'kd_sic_2': 3, 'Ji_mcm': 0.1, 'Ja_mcm': 0.1, 'ki_mcm': 0.15, 'ka_mcm': 1, 'ks_sic': 0.1, 'Ji_mih': 0.1, 'Ja_mih': 0.1, 'Vi_mih': 0.3, 'Va_mih': 1, 'Ji_ie': 0.01, 'Ja_ie': 0.01, 'ki_ie': 0.04, 'ka_ie': 0.1, 'Ji_cdc20': 0.001, 'Ja_cdc20': 0.001, 'ki_cdc20': 0.25, 'kd_cdc20': 0.1, 'ka_cdc20': 1, 'ks_cdc20_1': 0.005, 'ks_cdc20_2': 0.3, 'Js_cdc20': 0.3, 'Ji_cdh': 0.01, 'Ja_cdh': 0.01, 'ki_cdh': 35, 'ka_cdh_1': 1, 'ki_cdh_1': 2, 'ka_cdh_2': 10, 'ks_cln': 0.1, 'kd_cln': 0.1, 'Ji_sbf': 0.01, 'Ja_sbf': 0.01, 'ki_sbf_1': 1, 'ka_sbf_1': 1, 'ki_sbf_2': 2, 'ka_sbf_2': 0, 'ks_swe': 0.0025, 'ks_sweC': 0, 'khsl1': 1, 'khsl1r': 0.01, 'kd_swe_1': 0.007, 'Ji_wee': 0.05, 'Ja_wee': 0.05, 'Vi_wee': 1, 'Va_wee': 0.3, 'kd_swe_2': 0.05, 'ks_bud': 0.1, 'kd_bud': 0.1, 'mu': 0.005, 'IE_total': 1, 'Cdh1_total': 1, 'Mih1_total': 1, 'Mcm_total': 1, 'SBF_total': 1, 'kswe_1': 2, 'kswe_2': 0.01, 'kswe_3': 0.2, 'kmih_1': 5, 'kmih_2': 0.5}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Clb2': 0.184, 'M': 0.802, 'Mcm_a': 0.933, 'Cdh1_a': 0.993, 'Cdc20_a': 1.438, 'Sic1': 0.003, 'PClb2': 0, 'Trim': 0.084, 'Cln': 0.054, 'PTrim': 0, 'Mih1_a': 0.808, 'IE_a': 0.522, 'Cdc20': 1.172, 'SBF_a': 0.124, 'Swe1': 0, 'PSwe1': 0, 'Swe1M': 0.018, 'PSwe1M': 0.013, 'BE': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['epsilon', 'Jm', 'ks_clb', 'kd_clb_1', 'kd_clb_2', 'kd_clb_3', 'kass', 'kdiss', 'kd_sic', 'kd_sic_1', 'kd_sic_2', 'Ji_mcm', 'Ja_mcm', 'ki_mcm', 'ka_mcm', 'ks_sic', 'Ji_mih', 'Ja_mih', 'Vi_mih', 'Va_mih', 'Ji_ie', 'Ja_ie', 'ki_ie', 'ka_ie', 'Ji_cdc20', 'Ja_cdc20', 'ki_cdc20', 'kd_cdc20', 'ka_cdc20', 'ks_cdc20_1', 'ks_cdc20_2', 'Js_cdc20', 'Ji_cdh', 'Ja_cdh', 'ki_cdh', 'ka_cdh_1', 'ki_cdh_1', 'ka_cdh_2', 'ks_cln', 'kd_cln', 'Ji_sbf', 'Ja_sbf', 'ki_sbf_1', 'ka_sbf_1', 'ki_sbf_2', 'ka_sbf_2', 'ks_swe', 'ks_sweC', 'khsl1', 'khsl1r', 'kd_swe_1', 'Ji_wee', 'Ja_wee', 'Vi_wee', 'Va_wee', 'kd_swe_2', 'ks_bud', 'kd_bud', 'mu', 'IE_total', 'Cdh1_total', 'Mih1_total', 'Mcm_total', 'SBF_total', 'kswe_1', 'kswe_2', 'kswe_3', 'kmih_1', 'kmih_2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Clb2', 'M', 'Mcm_a', 'Cdh1_a', 'Cdc20_a', 'Sic1', 'PClb2', 'Trim', 'Cln', 'PTrim', 'Mih1_a', 'IE_a', 'Cdc20', 'SBF_a', 'Swe1', 'PSwe1', 'Swe1M', 'PSwe1M', 'BE']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (19)
        Clb2 = y[:, 0]
        M = y[:, 1]
        Mcm_a = y[:, 2]
        Cdh1_a = y[:, 3]
        Cdc20_a = y[:, 4]
        Sic1 = y[:, 5]
        PClb2 = y[:, 6]
        Trim = y[:, 7]
        Cln = y[:, 8]
        PTrim = y[:, 9]
        Mih1_a = y[:, 10]
        IE_a = y[:, 11]
        Cdc20 = y[:, 12]
        SBF_a = y[:, 13]
        Swe1 = y[:, 14]
        PSwe1 = y[:, 15]
        Swe1M = y[:, 16]
        PSwe1M = y[:, 17]
        BE = y[:, 18]

        # Independent parameters (source constant order)
        epsilon = theta[:, 0]  # constants[0]
        Jm = theta[:, 1]  # constants[1]
        ks_clb = theta[:, 2]  # constants[2]
        kd_clb_1 = theta[:, 3]  # constants[3]
        kd_clb_2 = theta[:, 4]  # constants[4]
        kd_clb_3 = theta[:, 5]  # constants[5]
        kass = theta[:, 6]  # constants[6]
        kdiss = theta[:, 7]  # constants[7]
        kd_sic = theta[:, 8]  # constants[8]
        kd_sic_1 = theta[:, 9]  # constants[9]
        kd_sic_2 = theta[:, 10]  # constants[10]
        Ji_mcm = theta[:, 11]  # constants[11]
        Ja_mcm = theta[:, 12]  # constants[12]
        ki_mcm = theta[:, 13]  # constants[13]
        ka_mcm = theta[:, 14]  # constants[14]
        ks_sic = theta[:, 15]  # constants[15]
        Ji_mih = theta[:, 16]  # constants[16]
        Ja_mih = theta[:, 17]  # constants[17]
        Vi_mih = theta[:, 18]  # constants[18]
        Va_mih = theta[:, 19]  # constants[19]
        Ji_ie = theta[:, 20]  # constants[20]
        Ja_ie = theta[:, 21]  # constants[21]
        ki_ie = theta[:, 22]  # constants[22]
        ka_ie = theta[:, 23]  # constants[23]
        Ji_cdc20 = theta[:, 24]  # constants[24]
        Ja_cdc20 = theta[:, 25]  # constants[25]
        ki_cdc20 = theta[:, 26]  # constants[26]
        kd_cdc20 = theta[:, 27]  # constants[27]
        ka_cdc20 = theta[:, 28]  # constants[28]
        ks_cdc20_1 = theta[:, 29]  # constants[29]
        ks_cdc20_2 = theta[:, 30]  # constants[30]
        Js_cdc20 = theta[:, 31]  # constants[31]
        Ji_cdh = theta[:, 32]  # constants[32]
        Ja_cdh = theta[:, 33]  # constants[33]
        ki_cdh = theta[:, 34]  # constants[34]
        ka_cdh_1 = theta[:, 35]  # constants[35]
        ki_cdh_1 = theta[:, 36]  # constants[36]
        ka_cdh_2 = theta[:, 37]  # constants[37]
        ks_cln = theta[:, 38]  # constants[38]
        kd_cln = theta[:, 39]  # constants[39]
        Ji_sbf = theta[:, 40]  # constants[40]
        Ja_sbf = theta[:, 41]  # constants[41]
        ki_sbf_1 = theta[:, 42]  # constants[42]
        ka_sbf_1 = theta[:, 43]  # constants[43]
        ki_sbf_2 = theta[:, 44]  # constants[44]
        ka_sbf_2 = theta[:, 45]  # constants[45]
        ks_swe = theta[:, 46]  # constants[46]
        ks_sweC = theta[:, 47]  # constants[47]
        khsl1 = theta[:, 48]  # constants[48]
        khsl1r = theta[:, 49]  # constants[49]
        kd_swe_1 = theta[:, 50]  # constants[50]
        Ji_wee = theta[:, 51]  # constants[51]
        Ja_wee = theta[:, 52]  # constants[52]
        Vi_wee = theta[:, 53]  # constants[53]
        Va_wee = theta[:, 54]  # constants[54]
        kd_swe_2 = theta[:, 55]  # constants[55]
        ks_bud = theta[:, 56]  # constants[56]
        kd_bud = theta[:, 57]  # constants[57]
        mu = theta[:, 58]  # constants[58]
        IE_total = theta[:, 59]  # constants[59]
        Cdh1_total = theta[:, 60]  # constants[60]
        Mih1_total = theta[:, 61]  # constants[61]
        Mcm_total = theta[:, 62]  # constants[62]
        SBF_total = theta[:, 63]  # constants[63]
        kswe_1 = theta[:, 64]  # constants[64]
        kswe_2 = theta[:, 65]  # constants[65]
        kswe_3 = theta[:, 66]  # constants[66]
        kmih_1 = theta[:, 67]  # constants[67]
        kmih_2 = theta[:, 68]  # constants[68]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Clb2 = torch.zeros_like(y[:, 0])
        d_M = torch.zeros_like(y[:, 0])
        d_Mcm_a = torch.zeros_like(y[:, 0])
        d_Cdh1_a = torch.zeros_like(y[:, 0])
        d_Cdc20_a = torch.zeros_like(y[:, 0])
        d_Sic1 = torch.zeros_like(y[:, 0])
        d_PClb2 = torch.zeros_like(y[:, 0])
        d_Trim = torch.zeros_like(y[:, 0])
        d_Cln = torch.zeros_like(y[:, 0])
        d_PTrim = torch.zeros_like(y[:, 0])
        d_Mih1_a = torch.zeros_like(y[:, 0])
        d_IE_a = torch.zeros_like(y[:, 0])
        d_Cdc20 = torch.zeros_like(y[:, 0])
        d_SBF_a = torch.zeros_like(y[:, 0])
        d_Swe1 = torch.zeros_like(y[:, 0])
        d_PSwe1 = torch.zeros_like(y[:, 0])
        d_Swe1M = torch.zeros_like(y[:, 0])
        d_PSwe1M = torch.zeros_like(y[:, 0])
        d_BE = torch.zeros_like(y[:, 0])
        alg_0_Mcm = torch.zeros_like(y[:, 0])
        alg_1_Mih1 = torch.zeros_like(y[:, 0])
        alg_2_IE = torch.zeros_like(y[:, 0])
        alg_3_Cdh1 = torch.zeros_like(y[:, 0])
        alg_4_SBF = torch.zeros_like(y[:, 0])
        alg_5_BUD = torch.zeros_like(y[:, 0])
        alg_7_kmih = torch.zeros_like(y[:, 0])
        alg_8_kswe = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Sic1 = torch.zeros_like(y[:, 0]) + (ks_sic + kdiss * (PTrim + Trim) + (kd_clb_1 + kd_clb_2 * Cdh1_a + kd_clb_3 * Cdc20_a) * (PTrim + Trim) - ((kd_sic + kd_sic_1 * Cln + kd_sic_2 * Clb2) * Sic1 + kass * Sic1 * (Clb2 + PClb2)))
        d_Cdc20_a = torch.zeros_like(y[:, 0]) + (ka_cdc20 * IE_a * Cdc20 / (Ja_cdc20 + Cdc20) - (ki_cdc20 * Cdc20_a / (Ji_cdc20 + Cdc20_a) + kd_cdc20 * Cdc20_a))
        d_Cdc20 = torch.zeros_like(y[:, 0]) + (ks_cdc20_1 + ks_cdc20_2 * Clb2 ** 4.0 / (Js_cdc20 ** 4.0 + Clb2 ** 4.0) + ki_cdc20 * Cdc20_a / (Ji_cdc20 + Cdc20_a) - (ka_cdc20 * IE_a * Cdc20 / (Ja_cdc20 + Cdc20) + kd_cdc20 * Cdc20))
        d_Cln = torch.zeros_like(y[:, 0]) + (ks_cln * SBF_a - kd_cln * Cln)
        d_BE = torch.zeros_like(y[:, 0]) + (ks_bud * Cln - kd_bud * BE)
        d_M = torch.zeros_like(y[:, 0]) + (mu * M)
        alg_0_Mcm = Mcm_total - Mcm_a
        d_Mcm_a = torch.zeros_like(y[:, 0]) + (-ki_mcm * Mcm_a / (Ji_mcm + Mcm_a) + ka_mcm * alg_0_Mcm * Clb2 / (Ja_mcm + alg_0_Mcm))
        alg_1_Mih1 = Mih1_total - Mih1_a
        d_Mih1_a = torch.zeros_like(y[:, 0]) + (-Vi_mih * Mih1_a / (Ji_mih + Mih1_a) + Va_mih * Clb2 * alg_1_Mih1 / (Ja_mih + alg_1_Mih1))
        alg_2_IE = IE_total - IE_a
        d_IE_a = torch.zeros_like(y[:, 0]) + (-ki_ie * IE_a / (Ji_ie + IE_a) + ka_ie * Clb2 * alg_2_IE / (Ja_ie + alg_2_IE))
        alg_3_Cdh1 = Cdh1_total - Cdh1_a
        d_Cdh1_a = torch.zeros_like(y[:, 0]) + (-(ki_cdh * Clb2 + ki_cdh_1 * Cln) * Cdh1_a / (Ji_cdh + Cdh1_a) + (ka_cdh_1 + ka_cdh_2 * Cdc20_a) * alg_3_Cdh1 / (Ja_cdh + alg_3_Cdh1))
        alg_4_SBF = SBF_total - SBF_a
        d_SBF_a = torch.zeros_like(y[:, 0]) + (-(ki_sbf_1 + ki_sbf_2 * Clb2) * SBF_a / (Ji_sbf + SBF_a) + (ka_sbf_1 * M + ka_sbf_2 * Cln) * alg_4_SBF / (Ja_sbf + alg_4_SBF))
        alg_5_BUD = torch.where((BE <= 0.6) & (Clb2 < 0.2), 0.0, 1.0)
        d_Swe1 = torch.zeros_like(y[:, 0]) + (ks_swe * SBF_a + ks_sweC + khsl1r * Swe1M + Va_wee * PSwe1 / (Ja_wee + PSwe1) - (khsl1 * alg_5_BUD * Swe1 + Vi_wee * Clb2 * Swe1 / (Ji_wee + Swe1) + kd_swe_1 * Swe1))
        d_PSwe1 = torch.zeros_like(y[:, 0]) + (khsl1r * PSwe1M + Vi_wee * Swe1 * Clb2 / (Ji_wee + Swe1) - (khsl1 * alg_5_BUD * PSwe1 + Va_wee * PSwe1 / (Ja_wee + PSwe1) + kd_swe_1 * PSwe1))
        d_Swe1M = torch.zeros_like(y[:, 0]) + (khsl1 * alg_5_BUD * Swe1 + Va_wee * PSwe1M / (Ja_wee + PSwe1M) - (khsl1r * Swe1M + Vi_wee * Clb2 * Swe1M / (Ji_wee + Swe1M) + kd_swe_1 * Swe1M))
        d_PSwe1M = torch.zeros_like(y[:, 0]) + (khsl1 * alg_5_BUD * PSwe1 + Vi_wee * Clb2 * Swe1M / (Ji_wee + Swe1M) - (khsl1r * PSwe1M + Va_wee * PSwe1M / (Ja_wee + PSwe1M) + kd_swe_2 * PSwe1M))
        alg_8_kswe = kswe_1 * Swe1 + kswe_2 * Swe1M + kswe_3 * PSwe1
        alg_7_kmih = kmih_1 * Mih1_a + kmih_2 * alg_1_Mih1
        d_Clb2 = torch.zeros_like(y[:, 0]) + (ks_clb * (epsilon + Mcm_a) * M / (1.0 + M / Jm) + alg_7_kmih * PClb2 + kdiss * Trim + (kd_sic + kd_sic_1 * Cln + kd_sic_2 * Clb2) * Trim - ((kd_clb_1 + kd_clb_2 * Cdh1_a + kd_clb_3 * Cdc20_a) * Clb2 + alg_8_kswe * Clb2 + kass * Sic1 * Clb2))
        d_PClb2 = torch.zeros_like(y[:, 0]) + (alg_8_kswe * Clb2 + kdiss * PTrim + (kd_sic + kd_sic_1 * Cln + kd_sic_2 * Clb2) * PTrim - ((kd_clb_1 + kd_clb_2 * Cdh1_a + kd_clb_3 * Cdc20_a) * PClb2 + alg_7_kmih * PClb2 + kass * Sic1 * PClb2))
        d_Trim = torch.zeros_like(y[:, 0]) + (kass * Sic1 * Clb2 + alg_7_kmih * PTrim - (kdiss * Trim + (kd_sic + kd_sic_1 * Cln + kd_sic_2 * Clb2) * Trim + (kd_clb_1 + kd_clb_2 * Cdh1_a + kd_clb_3 * Cdc20_a) * Trim + alg_8_kswe * Trim))
        d_PTrim = torch.zeros_like(y[:, 0]) + (alg_8_kswe * Trim + kass * Sic1 * PClb2 - (kdiss * PTrim + (kd_sic + kd_sic_1 * Cln + kd_sic_2 * Clb2) * PTrim + (kd_clb_1 + kd_clb_2 * Cdh1_a + kd_clb_3 * Cdc20_a) * PTrim + alg_7_kmih * PTrim))

        return torch.stack(
            [d_Clb2, d_M, d_Mcm_a, d_Cdh1_a, d_Cdc20_a, d_Sic1, d_PClb2, d_Trim, d_Cln, d_PTrim, d_Mih1_a, d_IE_a, d_Cdc20, d_SBF_a, d_Swe1, d_PSwe1, d_Swe1M, d_PSwe1M, d_BE],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
