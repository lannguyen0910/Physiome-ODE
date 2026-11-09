import torch
import torch.nn as nn
import math


class ElectrophysiologyCorriasBuist2007ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/corrias_buist_2007/model.py.
    Dynamic states: 14; independent constants: 33; derived constants used by rates: 17.
    Original constant slots: 51; algebraic slots: 34.
    """
    source_file = 'electrophysiology/corrias_buist_2007/model.py'
    source_state_dim = 14
    source_constant_dim = 51
    algebraic_dim = 34
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 14
        self.parameter_dim = 33

        self.parameters = (
            parameters
            if parameters is not None
            else {'Ach': 1e-05, 'Gcouple': 1.3, 'T': 310, 'T_exp': 297, 'F': 96486, 'R': 8314.4, 'Q10Ca': 2.1, 'Q10K': 1.365, 'Q10Na': 2.45, 'Ca_o': 2.5, 'Na_o': 137, 'K_o': 5.9, 'Cl_o': 134, 'Cm_SM': 77, 'Vol_SM': 3500, 'Na_i': 10, 'K_i': 164, 'period': 20000, 'delta_VICC': 59, 't_ICCpeak': 98, 't_ICCplateau': 7582, 't_ICC_stimulus': 10000, 'V_decay': 37.25, 'G_max_Ltype': 65, 'J_max_CaSR': 0.31705, 'G_max_LVA': 0.18, 'G_max_BK': 45.7, 'G_max_bk': 0.0144, 'G_max_kr_SM': 35, 'G_max_Na_SM': 3, 'G_max_ka_SM': 9, 'E_NSCC': -28, 'G_max_NSCC_SM': 50}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Vm_SM': -69.75, 'Ca_i': 8e-05, 'd_Ltype_SM': 0, 'f_Ltype_SM': 0.95, 'f_ca_Ltype_SM': 1, 'd_LVA_SM': 0.02, 'f_LVA_SM': 0.99, 'xr1_SM': 0, 'xr2_SM': 0.82, 'm_Na_SM': 0.005, 'h_Na_SM': 0.05787, 'xa1_SM': 0.00414, 'xa2_SM': 0.72, 'm_NSCC_SM': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Ach', 'Gcouple', 'T', 'T_exp', 'F', 'R', 'Q10Ca', 'Q10K', 'Q10Na', 'Ca_o', 'Na_o', 'K_o', 'Cl_o', 'Cm_SM', 'Vol_SM', 'Na_i', 'K_i', 'period', 'delta_VICC', 't_ICCpeak', 't_ICCplateau', 't_ICC_stimulus', 'V_decay', 'G_max_Ltype', 'J_max_CaSR', 'G_max_LVA', 'G_max_BK', 'G_max_bk', 'G_max_kr_SM', 'G_max_Na_SM', 'G_max_ka_SM', 'E_NSCC', 'G_max_NSCC_SM']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Vm_SM', 'Ca_i', 'd_Ltype_SM', 'f_Ltype_SM', 'f_ca_Ltype_SM', 'd_LVA_SM', 'f_LVA_SM', 'xr1_SM', 'xr2_SM', 'm_Na_SM', 'h_Na_SM', 'xa1_SM', 'xa2_SM', 'm_NSCC_SM']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (14)
        Vm_SM = y[:, 0]
        Ca_i = y[:, 1]
        d_Ltype_SM = y[:, 2]
        f_Ltype_SM = y[:, 3]
        f_ca_Ltype_SM = y[:, 4]
        d_LVA_SM = y[:, 5]
        f_LVA_SM = y[:, 6]
        xr1_SM = y[:, 7]
        xr2_SM = y[:, 8]
        m_Na_SM = y[:, 9]
        h_Na_SM = y[:, 10]
        xa1_SM = y[:, 11]
        xa2_SM = y[:, 12]
        m_NSCC_SM = y[:, 13]

        # Independent parameters (source constant order)
        Ach = theta[:, 0]  # constants[0]
        Gcouple = theta[:, 1]  # constants[1]
        T = theta[:, 2]  # constants[2]
        T_exp = theta[:, 3]  # constants[3]
        F = theta[:, 4]  # constants[4]
        R = theta[:, 5]  # constants[5]
        Q10Ca = theta[:, 6]  # constants[6]
        Q10K = theta[:, 7]  # constants[7]
        Q10Na = theta[:, 8]  # constants[8]
        Ca_o = theta[:, 9]  # constants[9]
        Na_o = theta[:, 10]  # constants[10]
        K_o = theta[:, 11]  # constants[11]
        Cl_o = theta[:, 12]  # constants[12]
        Cm_SM = theta[:, 13]  # constants[13]
        Vol_SM = theta[:, 14]  # constants[14]
        Na_i = theta[:, 15]  # constants[15]
        K_i = theta[:, 16]  # constants[16]
        period = theta[:, 17]  # constants[17]
        delta_VICC = theta[:, 18]  # constants[18]
        t_ICCpeak = theta[:, 19]  # constants[19]
        t_ICCplateau = theta[:, 20]  # constants[20]
        t_ICC_stimulus = theta[:, 21]  # constants[21]
        V_decay = theta[:, 22]  # constants[22]
        G_max_Ltype = theta[:, 23]  # constants[23]
        J_max_CaSR = theta[:, 24]  # constants[24]
        G_max_LVA = theta[:, 25]  # constants[25]
        G_max_BK = theta[:, 26]  # constants[26]
        G_max_bk = theta[:, 27]  # constants[27]
        G_max_kr_SM = theta[:, 28]  # constants[28]
        G_max_Na_SM = theta[:, 29]  # constants[29]
        G_max_ka_SM = theta[:, 30]  # constants[30]
        E_NSCC = theta[:, 31]  # constants[31]
        G_max_NSCC_SM = theta[:, 32]  # constants[32]

        # Derived constants from the original initConsts()
        T_correction_Na = Q10Na ** ((T - T_exp) / 10.0)  # constants[33]
        T_correction_K = Q10K ** ((T - T_exp) / 10.0)  # constants[34]
        T_correction_Ca = Q10Ca ** ((T - T_exp) / 10.0)  # constants[35]
        T_correction_BK = 1.1 * (T - T_exp)  # constants[36]
        RToF = R * T / F  # constants[38]
        rach_NSCC_SM = 1.0 / (1.0 + 0.01 / Ach)  # constants[39]
        tau_d_Ltype_SM = T_correction_Ca * 0.47  # constants[40]
        tau_f_Ltype_SM = T_correction_Ca * 86.0  # constants[41]
        tau_f_ca_Ltype_SM = T_correction_Ca * 2.0  # constants[42]
        tau_d_LVA_SM = T_correction_Ca * 3.0  # constants[43]
        E_K = RToF * torch.log(K_o / K_i)  # constants[44]
        E_K_2 = RToF * torch.log(K_o / K_i)  # constants[45]
        tau_xr1_SM = T_correction_K * 80.0  # constants[46]
        E_K_3 = RToF * torch.log(K_o / K_i)  # constants[47]
        E_Na = RToF * torch.log(Na_o / Na_i)  # constants[48]
        tau_xa2_SM = T_correction_K * 90.0  # constants[49]
        E_K_4 = RToF * torch.log(K_o / K_i)  # constants[50]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Vm_SM = torch.zeros_like(y[:, 0])
        d_Ca_i = torch.zeros_like(y[:, 0])
        d_d_Ltype_SM = torch.zeros_like(y[:, 0])
        d_f_Ltype_SM = torch.zeros_like(y[:, 0])
        d_f_ca_Ltype_SM = torch.zeros_like(y[:, 0])
        d_d_LVA_SM = torch.zeros_like(y[:, 0])
        d_f_LVA_SM = torch.zeros_like(y[:, 0])
        d_xr1_SM = torch.zeros_like(y[:, 0])
        d_xr2_SM = torch.zeros_like(y[:, 0])
        d_m_Na_SM = torch.zeros_like(y[:, 0])
        d_h_Na_SM = torch.zeros_like(y[:, 0])
        d_xa1_SM = torch.zeros_like(y[:, 0])
        d_xa2_SM = torch.zeros_like(y[:, 0])
        d_m_NSCC_SM = torch.zeros_like(y[:, 0])
        alg_0_stim_start = torch.zeros_like(y[:, 0])
        alg_1_d_inf_Ltype_SM = torch.zeros_like(y[:, 0])
        alg_2_f_inf_Ltype_SM = torch.zeros_like(y[:, 0])
        alg_3_f_ca_inf_Ltype_SM = torch.zeros_like(y[:, 0])
        alg_4_d_inf_LVA_SM = torch.zeros_like(y[:, 0])
        alg_5_f_inf_LVA_SM = torch.zeros_like(y[:, 0])
        alg_6_xr1_inf_SM = torch.zeros_like(y[:, 0])
        alg_7_xr2_inf_SM = torch.zeros_like(y[:, 0])
        alg_8_m_inf_Na = torch.zeros_like(y[:, 0])
        alg_9_h_inf_Na = torch.zeros_like(y[:, 0])
        alg_10_xa1_inf_SM = torch.zeros_like(y[:, 0])
        alg_11_xa2_inf_SM = torch.zeros_like(y[:, 0])
        alg_12_m_inf_NSCC_SM = torch.zeros_like(y[:, 0])
        alg_13_local_time = torch.zeros_like(y[:, 0])
        alg_14_tau_f_LVA_SM = torch.zeros_like(y[:, 0])
        alg_15_tau_xr2_SM = torch.zeros_like(y[:, 0])
        alg_16_tau_m_Na = torch.zeros_like(y[:, 0])
        alg_17_tau_h_Na = torch.zeros_like(y[:, 0])
        alg_18_tau_xa1_SM = torch.zeros_like(y[:, 0])
        alg_19_tau_m_NSCC_SM = torch.zeros_like(y[:, 0])
        alg_20_I_stim = torch.zeros_like(y[:, 0])
        alg_21_E_Ca = torch.zeros_like(y[:, 0])
        alg_22_I_Ltype_SM = torch.zeros_like(y[:, 0])
        alg_23_J_CaSR_SM = torch.zeros_like(y[:, 0])
        alg_24_E_Ca_2 = torch.zeros_like(y[:, 0])
        alg_25_I_LVA_SM = torch.zeros_like(y[:, 0])
        alg_26_d_BK_SM = torch.zeros_like(y[:, 0])
        alg_27_I_BK_SM = torch.zeros_like(y[:, 0])
        alg_28_I_bk_SM = torch.zeros_like(y[:, 0])
        alg_29_I_kr_SM = torch.zeros_like(y[:, 0])
        alg_30_I_Na_SM = torch.zeros_like(y[:, 0])
        alg_31_I_ka_SM = torch.zeros_like(y[:, 0])
        alg_32_f_ca_NSCC_SM = torch.zeros_like(y[:, 0])
        alg_33_I_NSCC_SM = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_d_inf_Ltype_SM = 1.0 / (1.0 + torch.exp((Vm_SM + 17.0) / -4.3))
        d_d_Ltype_SM = torch.zeros_like(y[:, 0]) + ((alg_1_d_inf_Ltype_SM - d_Ltype_SM) / tau_d_Ltype_SM)
        alg_2_f_inf_Ltype_SM = 1.0 / (1.0 + torch.exp((Vm_SM + 43.0) / 8.9))
        d_f_Ltype_SM = torch.zeros_like(y[:, 0]) + ((alg_2_f_inf_Ltype_SM - f_Ltype_SM) / tau_f_Ltype_SM)
        alg_3_f_ca_inf_Ltype_SM = 1.0 - 1.0 / (1.0 + torch.exp((Ca_i - 8.999e-05 - 0.000214) / -1.31e-05))
        d_f_ca_Ltype_SM = torch.zeros_like(y[:, 0]) + ((alg_3_f_ca_inf_Ltype_SM - f_ca_Ltype_SM) / tau_f_ca_Ltype_SM)
        alg_4_d_inf_LVA_SM = 1.0 / (1.0 + torch.exp((Vm_SM + 27.5) / -10.9))
        d_d_LVA_SM = torch.zeros_like(y[:, 0]) + ((alg_4_d_inf_LVA_SM - d_LVA_SM) / tau_d_LVA_SM)
        alg_6_xr1_inf_SM = 1.0 / (1.0 + torch.exp((Vm_SM + 27.0) / -5.0))
        d_xr1_SM = torch.zeros_like(y[:, 0]) + ((alg_6_xr1_inf_SM - xr1_SM) / tau_xr1_SM)
        alg_11_xa2_inf_SM = 0.1 + 0.9 / (1.0 + torch.exp((Vm_SM + 65.0) / 6.2))
        d_xa2_SM = torch.zeros_like(y[:, 0]) + ((alg_11_xa2_inf_SM - xa2_SM) / tau_xa2_SM)
        alg_5_f_inf_LVA_SM = 1.0 / (1.0 + torch.exp((Vm_SM + 15.8) / 7.0))
        alg_14_tau_f_LVA_SM = T_correction_Ca * (7.58 * torch.exp(Vm_SM * 0.00817))
        d_f_LVA_SM = torch.zeros_like(y[:, 0]) + ((alg_5_f_inf_LVA_SM - f_LVA_SM) / alg_14_tau_f_LVA_SM)
        alg_7_xr2_inf_SM = 0.2 + 0.8 / (1.0 + torch.exp((Vm_SM + 58.0) / 10.0))
        alg_15_tau_xr2_SM = T_correction_K * (-707.0 + 1481.0 * torch.exp((Vm_SM + 36.0) / 95.0))
        d_xr2_SM = torch.zeros_like(y[:, 0]) + ((alg_7_xr2_inf_SM - xr2_SM) / alg_15_tau_xr2_SM)
        alg_8_m_inf_Na = 1.0 / (1.0 + torch.exp((Vm_SM + 47.0) / -4.8))
        alg_16_tau_m_Na = T_correction_Na * (Vm_SM * -0.017 * 1.0 + 0.44)
        d_m_Na_SM = torch.zeros_like(y[:, 0]) + ((alg_8_m_inf_Na - m_Na_SM) / alg_16_tau_m_Na)
        alg_9_h_inf_Na = 1.0 / (1.0 + torch.exp((Vm_SM + 78.0) / 3.0))
        alg_17_tau_h_Na = T_correction_Na * (Vm_SM * -0.25 * 1.0 + 5.5)
        d_h_Na_SM = torch.zeros_like(y[:, 0]) + ((alg_9_h_inf_Na - h_Na_SM) / alg_17_tau_h_Na)
        alg_10_xa1_inf_SM = 1.0 / (1.0 + torch.exp((Vm_SM + 26.5) / -7.9))
        alg_18_tau_xa1_SM = T_correction_K * (31.8 + 175.0 * torch.exp(-0.5 * ((Vm_SM + 44.4) / 22.3) ** 2.0))
        d_xa1_SM = torch.zeros_like(y[:, 0]) + ((alg_10_xa1_inf_SM - xa1_SM) / alg_18_tau_xa1_SM)
        alg_12_m_inf_NSCC_SM = 1.0 / (1.0 + torch.exp((Vm_SM + 25.0) / -20.0))
        alg_19_tau_m_NSCC_SM = 1.0 / (1.0 + torch.exp((Vm_SM + 66.0) / -26.0)) * 150.0
        d_m_NSCC_SM = torch.zeros_like(y[:, 0]) + ((alg_12_m_inf_NSCC_SM - m_NSCC_SM) / alg_19_tau_m_NSCC_SM)
        alg_21_E_Ca = 0.5 * RToF * torch.log(Ca_o / Ca_i)
        alg_22_I_Ltype_SM = G_max_Ltype * (f_Ltype_SM * d_Ltype_SM * f_ca_Ltype_SM) * (Vm_SM - alg_21_E_Ca)
        alg_24_E_Ca_2 = 0.5 * RToF * torch.log(Ca_o / Ca_i)
        alg_25_I_LVA_SM = G_max_LVA * (f_LVA_SM * d_LVA_SM) * (Vm_SM - alg_24_E_Ca_2)
        alg_23_J_CaSR_SM = J_max_CaSR * (Ca_i * 1.0) ** 1.34
        d_Ca_i = torch.zeros_like(y[:, 0]) + ((-1.0 * alg_22_I_Ltype_SM + -1.0 * alg_25_I_LVA_SM) / (2.0 * 0.001 * F * Vol_SM) + -1.0 * alg_23_J_CaSR_SM)
        alg_30_I_Na_SM = G_max_Na_SM * (h_Na_SM * m_Na_SM) * (Vm_SM - E_Na)
        alg_29_I_kr_SM = G_max_kr_SM * (xr1_SM * xr2_SM) * (Vm_SM - E_K_3)
        alg_31_I_ka_SM = G_max_ka_SM * (xa1_SM * xa2_SM) * (Vm_SM - E_K_4)
        alg_26_d_BK_SM = 1.0 / (1.0 + torch.exp(Vm_SM / -17.0 - 2.0 * torch.log(Ca_i / 0.001)))
        alg_27_I_BK_SM = (G_max_BK + T_correction_BK) * alg_26_d_BK_SM * (Vm_SM - E_K)
        alg_32_f_ca_NSCC_SM = 1.0 / (1.0 + (Ca_i / 0.0002) ** (-4.0))
        alg_33_I_NSCC_SM = G_max_NSCC_SM * m_NSCC_SM * alg_32_f_ca_NSCC_SM * rach_NSCC_SM * (Vm_SM - E_NSCC)
        alg_28_I_bk_SM = G_max_bk * (Vm_SM - E_K_2)
        alg_0_stim_start = torch.where((t > period * 1.0) & (t <= period * 2.0), period * 1.0, torch.where((t > period * 2.0) & (t <= period * 3.0), period * 2.0, torch.where((t > period * 3.0) & (t <= period * 4.0), period * 3.0, torch.where((t > period * 4.0) & (t <= period * 5.0), period * 4.0, 0.0))))
        alg_13_local_time = t - (alg_0_stim_start + t_ICCpeak)
        alg_20_I_stim = torch.where(alg_13_local_time < t_ICCpeak, Gcouple * delta_VICC, torch.where((alg_13_local_time >= t_ICCpeak) & (alg_13_local_time <= t_ICCplateau), Gcouple * delta_VICC * (1.0 / (1.0 + torch.exp((alg_13_local_time - 8000.0) / 1000.0))), torch.where((alg_13_local_time > t_ICCplateau) & (alg_13_local_time < t_ICC_stimulus), Gcouple * V_decay * (1.0 / (1.0 + torch.exp((alg_13_local_time - 8000.0) / 150.0))), 0.0)))
        d_Vm_SM = torch.zeros_like(y[:, 0]) + (-1.0 * (1.0 / Cm_SM) * (alg_30_I_Na_SM + alg_22_I_Ltype_SM + alg_25_I_LVA_SM + alg_29_I_kr_SM + alg_31_I_ka_SM + alg_27_I_BK_SM + alg_33_I_NSCC_SM + alg_28_I_bk_SM + -1.0 * alg_20_I_stim))

        return torch.stack(
            [d_Vm_SM, d_Ca_i, d_d_Ltype_SM, d_f_Ltype_SM, d_f_ca_Ltype_SM, d_d_LVA_SM, d_f_LVA_SM, d_xr1_SM, d_xr2_SM, d_m_Na_SM, d_h_Na_SM, d_xa1_SM, d_xa2_SM, d_m_NSCC_SM],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
