import torch
import torch.nn as nn
import math


class MetabolismVanbeek2007ODE(nn.Module):
    """
    PyTorch translation of metabolism/vanbeek_2007/model.py.
    Dynamic states: 10; independent constants: 31; derived constants used by rates: 7.
    Original constant slots: 38; algebraic slots: 15.
    """
    source_file = 'metabolism/vanbeek_2007/model.py'
    source_state_dim = 10
    source_constant_dim = 38
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

        self.state_dim = 10
        self.parameter_dim = 31

        self.parameters = (
            parameters
            if parameters is not None
            else {'V_cyt': 0.75, 'V_ims': 0.0625, 'Kia': 900, 'Kb': 15500, 'Kib': 34900, 'Kic': 222.4, 'Kd': 1670, 'Kid': 4730, 'Vmax_MM_f': 11440, 'Vmax_MM_b': 48040, 'Kia_2': 750, 'Kb_2': 5200, 'Kib_2': 28800, 'Kic_2': 204.8, 'Kd_2': 500, 'Kid_2': 1600, 'Vmax_Mi_f': 882, 'Vmax_Mi_b': 3704, 'J_hyd_basis_1': 486.5, 'J_hyd_basis_2': 627.6, 'freq_1': 135, 'freq_2': 220, 'nb_of_cycles_1': 5, 'KPi': 800, 'KADP': 25, 'V_max_syn': 15040, 'PS_tot_ATP': 13.3, 'PS_tot_ADP': 13.3, 'PS_tot_PCr': 155, 'PS_tot_Cr': 155, 'PS_tot_Pi': 194}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'ATP_cyt': 5912.77, 'ADP_cyt': 64, 'PCr_cyt': 5000, 'Cr_cyt': 10500, 'Pi_cyt': 913, 'ATP_ims': 5912.77, 'ADP_ims': 39, 'PCr_ims': 5000, 'Cr_ims': 10500, 'Pi_ims': 910}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['V_cyt', 'V_ims', 'Kia', 'Kb', 'Kib', 'Kic', 'Kd', 'Kid', 'Vmax_MM_f', 'Vmax_MM_b', 'Kia_2', 'Kb_2', 'Kib_2', 'Kic_2', 'Kd_2', 'Kid_2', 'Vmax_Mi_f', 'Vmax_Mi_b', 'J_hyd_basis_1', 'J_hyd_basis_2', 'freq_1', 'freq_2', 'nb_of_cycles_1', 'KPi', 'KADP', 'V_max_syn', 'PS_tot_ATP', 'PS_tot_ADP', 'PS_tot_PCr', 'PS_tot_Cr', 'PS_tot_Pi']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['ATP_cyt', 'ADP_cyt', 'PCr_cyt', 'Cr_cyt', 'Pi_cyt', 'ATP_ims', 'ADP_ims', 'PCr_ims', 'Cr_ims', 'Pi_ims']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (10)
        ATP_cyt = y[:, 0]
        ADP_cyt = y[:, 1]
        PCr_cyt = y[:, 2]
        Cr_cyt = y[:, 3]
        Pi_cyt = y[:, 4]
        ATP_ims = y[:, 5]
        ADP_ims = y[:, 6]
        PCr_ims = y[:, 7]
        Cr_ims = y[:, 8]
        Pi_ims = y[:, 9]

        # Independent parameters (source constant order)
        V_cyt = theta[:, 0]  # constants[0]
        V_ims = theta[:, 1]  # constants[1]
        Kia = theta[:, 2]  # constants[2]
        Kb = theta[:, 3]  # constants[3]
        Kib = theta[:, 4]  # constants[4]
        Kic = theta[:, 5]  # constants[5]
        Kd = theta[:, 6]  # constants[6]
        Kid = theta[:, 7]  # constants[7]
        Vmax_MM_f = theta[:, 8]  # constants[8]
        Vmax_MM_b = theta[:, 9]  # constants[9]
        Kia_2 = theta[:, 10]  # constants[10]
        Kb_2 = theta[:, 11]  # constants[11]
        Kib_2 = theta[:, 12]  # constants[12]
        Kic_2 = theta[:, 13]  # constants[13]
        Kd_2 = theta[:, 14]  # constants[14]
        Kid_2 = theta[:, 15]  # constants[15]
        Vmax_Mi_f = theta[:, 16]  # constants[16]
        Vmax_Mi_b = theta[:, 17]  # constants[17]
        J_hyd_basis_1 = theta[:, 18]  # constants[18]
        J_hyd_basis_2 = theta[:, 19]  # constants[19]
        freq_1 = theta[:, 20]  # constants[20]
        freq_2 = theta[:, 21]  # constants[21]
        nb_of_cycles_1 = theta[:, 22]  # constants[22]
        KPi = theta[:, 23]  # constants[23]
        KADP = theta[:, 24]  # constants[24]
        V_max_syn = theta[:, 25]  # constants[25]
        PS_tot_ATP = theta[:, 26]  # constants[26]
        PS_tot_ADP = theta[:, 27]  # constants[27]
        PS_tot_PCr = theta[:, 28]  # constants[28]
        PS_tot_Cr = theta[:, 29]  # constants[29]
        PS_tot_Pi = theta[:, 30]  # constants[30]

        # Derived constants from the original initConsts()
        KIb = Kib  # constants[31]
        Kc = Kic * Kd / Kid  # constants[32]
        KIb_2 = Kib_2  # constants[33]
        Kc_2 = Kic_2 * Kd_2 / Kid_2  # constants[34]
        t_cycle_1 = 60.0 / freq_1  # constants[35]
        t_cycle_2 = 60.0 / freq_2  # constants[36]
        duration_1 = nb_of_cycles_1 * t_cycle_1  # constants[37]

        # Initialize rates and algebraic variables, as in computeRates()
        d_ATP_cyt = torch.zeros_like(y[:, 0])
        d_ADP_cyt = torch.zeros_like(y[:, 0])
        d_PCr_cyt = torch.zeros_like(y[:, 0])
        d_Cr_cyt = torch.zeros_like(y[:, 0])
        d_Pi_cyt = torch.zeros_like(y[:, 0])
        d_ATP_ims = torch.zeros_like(y[:, 0])
        d_ADP_ims = torch.zeros_like(y[:, 0])
        d_PCr_ims = torch.zeros_like(y[:, 0])
        d_Cr_ims = torch.zeros_like(y[:, 0])
        d_Pi_ims = torch.zeros_like(y[:, 0])
        alg_0_Den_MMCK = torch.zeros_like(y[:, 0])
        alg_1_J_CKMM = torch.zeros_like(y[:, 0])
        alg_2_Den_MiCK = torch.zeros_like(y[:, 0])
        alg_3_J_CKMi = torch.zeros_like(y[:, 0])
        alg_4_ltime = torch.zeros_like(y[:, 0])
        alg_5_J_diff_PCr = torch.zeros_like(y[:, 0])
        alg_6_J_diff_Cr = torch.zeros_like(y[:, 0])
        alg_7_t_cycle = torch.zeros_like(y[:, 0])
        alg_8_H_ATPmax = torch.zeros_like(y[:, 0])
        alg_9_J_hyd = torch.zeros_like(y[:, 0])
        alg_10_Den_syn = torch.zeros_like(y[:, 0])
        alg_11_J_syn = torch.zeros_like(y[:, 0])
        alg_12_J_diff_ATP = torch.zeros_like(y[:, 0])
        alg_13_J_diff_ADP = torch.zeros_like(y[:, 0])
        alg_14_J_diff_Pi = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_Den_MMCK = 1.0 + Cr_cyt / Kib + PCr_cyt / Kid + ATP_cyt * (1.0 / Kia + Cr_cyt / (Kia * Kb)) + ADP_cyt * (1.0 / Kic + PCr_cyt / (Kid * Kc) + Cr_cyt / (Kic * KIb))
        alg_1_J_CKMM = (Vmax_MM_f * (ATP_cyt * Cr_cyt / (Kia * Kb)) - Vmax_MM_b * (ADP_cyt * PCr_cyt / (Kic * Kd))) / alg_0_Den_MMCK
        alg_5_J_diff_PCr = PS_tot_PCr * (PCr_ims - PCr_cyt)
        d_PCr_cyt = torch.zeros_like(y[:, 0]) + ((alg_5_J_diff_PCr + alg_1_J_CKMM) / V_cyt)
        alg_6_J_diff_Cr = PS_tot_Cr * (Cr_ims - Cr_cyt)
        d_Cr_cyt = torch.zeros_like(y[:, 0]) + ((alg_6_J_diff_Cr - alg_1_J_CKMM) / V_cyt)
        alg_2_Den_MiCK = 1.0 + Cr_ims / Kib_2 + PCr_ims / Kid_2 + ATP_ims * (1.0 / Kia_2 + Cr_ims / (Kia_2 * Kb_2)) + ADP_ims * (1.0 / Kic_2 + PCr_ims / (Kid_2 * Kc_2) + Cr_ims / (Kic_2 * KIb_2))
        alg_3_J_CKMi = (Vmax_Mi_f * (ATP_ims * Cr_ims / (Kia_2 * Kb_2)) - Vmax_Mi_b * (ADP_ims * PCr_ims / (Kic_2 * Kd_2))) / alg_2_Den_MiCK
        d_PCr_ims = torch.zeros_like(y[:, 0]) + ((alg_3_J_CKMi - alg_5_J_diff_PCr) / V_ims)
        d_Cr_ims = torch.zeros_like(y[:, 0]) + (-(alg_3_J_CKMi + alg_6_J_diff_Cr) / V_ims)
        alg_8_H_ATPmax = torch.where(t <= duration_1, 6.0 * J_hyd_basis_1, 6.0 * J_hyd_basis_2)
        alg_7_t_cycle = torch.where(t <= duration_1, t_cycle_1, t_cycle_2)
        alg_4_ltime = torch.where(t <= duration_1, t - t_cycle_1 * torch.floor(t / t_cycle_1), t - duration_1 - t_cycle_2 * torch.floor((t - duration_1) / t_cycle_2))
        alg_9_J_hyd = torch.where((alg_4_ltime >= 0.0) & (alg_4_ltime < 1.0 / 6.0 * alg_7_t_cycle), alg_8_H_ATPmax * alg_4_ltime / alg_7_t_cycle * 6.0, torch.where((alg_4_ltime >= 1.0 / 6.0 * alg_7_t_cycle) & (alg_4_ltime < 1.0 / 3.0 * alg_7_t_cycle), alg_8_H_ATPmax * (1.0 - 6.0 * (alg_4_ltime / alg_7_t_cycle - 1.0 / 6.0)), torch.where((alg_4_ltime >= 1.0 / 3.0 * alg_7_t_cycle) & (alg_4_ltime < alg_7_t_cycle), 0.0, (1e309-1e309))))
        alg_12_J_diff_ATP = PS_tot_ATP * (ATP_ims - ATP_cyt)
        d_ATP_cyt = torch.zeros_like(y[:, 0]) + ((alg_12_J_diff_ATP - (alg_9_J_hyd + alg_1_J_CKMM)) / V_cyt)
        alg_13_J_diff_ADP = PS_tot_ADP * (ADP_ims - ADP_cyt)
        d_ADP_cyt = torch.zeros_like(y[:, 0]) + ((alg_13_J_diff_ADP + alg_9_J_hyd + alg_1_J_CKMM) / V_cyt)
        alg_14_J_diff_Pi = PS_tot_Pi * (Pi_ims - Pi_cyt)
        d_Pi_cyt = torch.zeros_like(y[:, 0]) + ((alg_14_J_diff_Pi + alg_9_J_hyd) / V_cyt)
        alg_10_Den_syn = 1.0 + ADP_ims / KADP + Pi_ims / KPi + ADP_ims * Pi_ims / (KADP * KPi)
        alg_11_J_syn = V_max_syn * (ADP_ims * Pi_ims / (KPi * KADP * alg_10_Den_syn))
        d_ATP_ims = torch.zeros_like(y[:, 0]) + ((alg_11_J_syn - (alg_12_J_diff_ATP + alg_3_J_CKMi)) / V_ims)
        d_ADP_ims = torch.zeros_like(y[:, 0]) + ((alg_3_J_CKMi - (alg_11_J_syn + alg_13_J_diff_ADP)) / V_ims)
        d_Pi_ims = torch.zeros_like(y[:, 0]) + (-(alg_11_J_syn + alg_14_J_diff_Pi) / V_ims)

        return torch.stack(
            [d_ATP_cyt, d_ADP_cyt, d_PCr_cyt, d_Cr_cyt, d_Pi_cyt, d_ATP_ims, d_ADP_ims, d_PCr_ims, d_Cr_ims, d_Pi_ims],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
