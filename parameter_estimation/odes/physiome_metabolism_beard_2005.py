import torch
import torch.nn as nn
import math


class MetabolismBeard2005ODE(nn.Module):
    """
    PyTorch translation of metabolism/beard_2005/model.py.
    Dynamic states: 19; independent constants: 51; derived constants used by rates: 8.
    Original constant slots: 60; algebraic slots: 34.
    """
    source_file = 'metabolism/beard_2005/model.py'
    source_state_dim = 19
    source_constant_dim = 60
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

        self.state_dim = 19
        self.parameter_dim = 51

        self.parameters = (
            parameters
            if parameters is not None
            else {'Mg_tot': 0.005, 'Pi_e': 0.000125, 'ADP_e': 0, 'RT': 2.4734, 'F': 0.096484, 'n_A': 3, 'dG_C1o': -69.37, 'dG_C3o': -32.53, 'dG_C4o': -122.94, 'dG_F1o': 36.03, 'pH_e': 7.1, 'K_e': 0.15, 'ATP_e': 0, 'AMP_e': 0, 'K_DT': 2.4e-05, 'K_DD': 0.000347, 'K_AK': 0.4331, 'W_m': 0.72376, 'gamma': 5.99, 'Ctot': 0.0027, 'Qtot': 0.00135, 'NADtot': 0.00297, 'k_Pi1': 0.00013413, 'k_Pi2': 0.00067668, 'k_Pi3': 0.00019172, 'k_Pi4': 0.02531, 'k_PiH': 0.00045082, 'r': 4.5807, 'x_DH': 0.09183, 'x_C1': 0.36923, 'x_C3': 0.091737, 'x_C4': 3.2562e-05, 'x_F1': 150.93, 'x_ANT': 0.0079204, 'x_Pi1': 339430, 'x_KH': 29802000, 'x_Hle': 250, 'x_K': 0, 'k_mADP': 3.5e-06, 'x_AK': 0, 'p_A': 85, 'k_O2': 0.00012, 'x_buff': 100, 'x_MgA': 1000000, 'x_Pi2': 327, 'mincond': 1e-12, 'C_im': 6.756756756756757e-06, 'k_dHPi': 1.7782794100389227e-07, 'k_dHatp': 3.311311214825908e-07, 'k_dHadp': 5.128613839913648e-07, 'constant_59': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'dPsi': 160, 'H_x': 6.30957344480193e-08, 'NADH_x': 0.0015, 'Pi_x': 0.001, 'QH2': 0.0008, 'Cred': 0.001, 'O2': 2.6e-05, 'ADP_mx': 0, 'ATP_mx': 0, 'ATP_x': 0, 'Mg_x': 0.005, 'ADP_x': 0.01, 'ATP_i': 0, 'ATP_mi': 0, 'ADP_i': 0, 'ADP_mi': 0, 'AMP_i': 0, 'Pi_i': 0.001, 'K_x': 0.14}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Mg_tot', 'Pi_e', 'ADP_e', 'RT', 'F', 'n_A', 'dG_C1o', 'dG_C3o', 'dG_C4o', 'dG_F1o', 'pH_e', 'K_e', 'ATP_e', 'AMP_e', 'K_DT', 'K_DD', 'K_AK', 'W_m', 'gamma', 'Ctot', 'Qtot', 'NADtot', 'k_Pi1', 'k_Pi2', 'k_Pi3', 'k_Pi4', 'k_PiH', 'r', 'x_DH', 'x_C1', 'x_C3', 'x_C4', 'x_F1', 'x_ANT', 'x_Pi1', 'x_KH', 'x_Hle', 'x_K', 'k_mADP', 'x_AK', 'p_A', 'k_O2', 'x_buff', 'x_MgA', 'x_Pi2', 'mincond', 'C_im', 'k_dHPi', 'k_dHatp', 'k_dHadp', 'constant_59']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['dPsi', 'H_x', 'NADH_x', 'Pi_x', 'QH2', 'Cred', 'O2', 'ADP_mx', 'ATP_mx', 'ATP_x', 'Mg_x', 'ADP_x', 'ATP_i', 'ATP_mi', 'ADP_i', 'ADP_mi', 'AMP_i', 'Pi_i', 'K_x']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (19)
        dPsi = y[:, 0]
        H_x = y[:, 1]
        NADH_x = y[:, 2]
        Pi_x = y[:, 3]
        QH2 = y[:, 4]
        Cred = y[:, 5]
        O2 = y[:, 6]
        ADP_mx = y[:, 7]
        ATP_mx = y[:, 8]
        ATP_x = y[:, 9]
        Mg_x = y[:, 10]
        ADP_x = y[:, 11]
        ATP_i = y[:, 12]
        ATP_mi = y[:, 13]
        ADP_i = y[:, 14]
        ADP_mi = y[:, 15]
        AMP_i = y[:, 16]
        Pi_i = y[:, 17]
        K_x = y[:, 18]

        # Independent parameters (source constant order)
        Mg_tot = theta[:, 0]  # constants[0]
        Pi_e = theta[:, 1]  # constants[1]
        ADP_e = theta[:, 2]  # constants[2]
        RT = theta[:, 3]  # constants[3]
        F = theta[:, 4]  # constants[4]
        n_A = theta[:, 5]  # constants[5]
        dG_C1o = theta[:, 6]  # constants[6]
        dG_C3o = theta[:, 7]  # constants[7]
        dG_C4o = theta[:, 8]  # constants[8]
        dG_F1o = theta[:, 9]  # constants[9]
        pH_e = theta[:, 10]  # constants[10]
        K_e = theta[:, 11]  # constants[11]
        ATP_e = theta[:, 12]  # constants[12]
        AMP_e = theta[:, 13]  # constants[13]
        K_DT = theta[:, 14]  # constants[14]
        K_DD = theta[:, 15]  # constants[15]
        K_AK = theta[:, 16]  # constants[16]
        W_m = theta[:, 17]  # constants[17]
        gamma = theta[:, 18]  # constants[18]
        Ctot = theta[:, 19]  # constants[19]
        Qtot = theta[:, 20]  # constants[20]
        NADtot = theta[:, 21]  # constants[21]
        k_Pi1 = theta[:, 22]  # constants[22]
        k_Pi2 = theta[:, 23]  # constants[23]
        k_Pi3 = theta[:, 24]  # constants[24]
        k_Pi4 = theta[:, 25]  # constants[25]
        k_PiH = theta[:, 26]  # constants[26]
        r = theta[:, 27]  # constants[27]
        x_DH = theta[:, 28]  # constants[28]
        x_C1 = theta[:, 29]  # constants[29]
        x_C3 = theta[:, 30]  # constants[30]
        x_C4 = theta[:, 31]  # constants[31]
        x_F1 = theta[:, 32]  # constants[32]
        x_ANT = theta[:, 33]  # constants[33]
        x_Pi1 = theta[:, 34]  # constants[34]
        x_KH = theta[:, 35]  # constants[35]
        x_Hle = theta[:, 36]  # constants[36]
        x_K = theta[:, 37]  # constants[37]
        k_mADP = theta[:, 38]  # constants[38]
        x_AK = theta[:, 39]  # constants[39]
        p_A = theta[:, 40]  # constants[40]
        k_O2 = theta[:, 41]  # constants[41]
        x_buff = theta[:, 42]  # constants[42]
        x_MgA = theta[:, 43]  # constants[43]
        x_Pi2 = theta[:, 44]  # constants[44]
        mincond = theta[:, 45]  # constants[45]
        C_im = theta[:, 46]  # constants[46]
        k_dHPi = theta[:, 47]  # constants[48]
        k_dHatp = theta[:, 48]  # constants[49]
        k_dHadp = theta[:, 49]  # constants[50]
        constant_59 = theta[:, 50]  # constants[59]

        # Derived constants from the original initConsts()
        H_e = 1.0 * 10.0 ** (-pH_e)  # constants[47]
        K_i = K_e  # constants[51]
        W_x = 0.9 * W_m  # constants[52]
        ADP_me = (K_DD + ADP_e + Mg_tot - ((K_DD + ADP_e + Mg_tot) ** 2.0 - 4.0 * Mg_tot * ADP_e) ** (1.0 / 2)) / 2.0  # constants[53]
        H_i = H_e  # constants[54]
        W_i = 0.1 * W_m  # constants[55]
        Mg_e = Mg_tot - ADP_me  # constants[57]
        Mg_i = Mg_e  # constants[58]

        # Initialize rates and algebraic variables, as in computeRates()
        d_dPsi = torch.zeros_like(y[:, 0])
        d_H_x = torch.zeros_like(y[:, 0])
        d_NADH_x = torch.zeros_like(y[:, 0])
        d_Pi_x = torch.zeros_like(y[:, 0])
        d_QH2 = torch.zeros_like(y[:, 0])
        d_Cred = torch.zeros_like(y[:, 0])
        d_O2 = torch.zeros_like(y[:, 0])
        d_ADP_mx = torch.zeros_like(y[:, 0])
        d_ATP_mx = torch.zeros_like(y[:, 0])
        d_ATP_x = torch.zeros_like(y[:, 0])
        d_Mg_x = torch.zeros_like(y[:, 0])
        d_ADP_x = torch.zeros_like(y[:, 0])
        d_ATP_i = torch.zeros_like(y[:, 0])
        d_ATP_mi = torch.zeros_like(y[:, 0])
        d_ADP_i = torch.zeros_like(y[:, 0])
        d_ADP_mi = torch.zeros_like(y[:, 0])
        d_AMP_i = torch.zeros_like(y[:, 0])
        d_Pi_i = torch.zeros_like(y[:, 0])
        d_K_x = torch.zeros_like(y[:, 0])
        alg_0_dG_H = torch.zeros_like(y[:, 0])
        alg_1_dG_C4op = torch.zeros_like(y[:, 0])
        alg_2_J_F1 = torch.zeros_like(y[:, 0])
        alg_3_Psi_x = torch.zeros_like(y[:, 0])
        alg_4_Psi_i = torch.zeros_like(y[:, 0])
        alg_5_ATP_fx = torch.zeros_like(y[:, 0])
        alg_6_J_MgATPx = torch.zeros_like(y[:, 0])
        alg_7_ADP_fx = torch.zeros_like(y[:, 0])
        alg_8_ATP_fi = torch.zeros_like(y[:, 0])
        alg_9_J_MgADPx = torch.zeros_like(y[:, 0])
        alg_10_ADP_fi = torch.zeros_like(y[:, 0])
        alg_11_J_MgATPi = torch.zeros_like(y[:, 0])
        alg_12_J_ANT = torch.zeros_like(y[:, 0])
        alg_13_J_MgADPi = torch.zeros_like(y[:, 0])
        alg_14_J_Pi2 = torch.zeros_like(y[:, 0])
        alg_15_J_ATP = torch.zeros_like(y[:, 0])
        alg_16_H2PIi = torch.zeros_like(y[:, 0])
        alg_17_J_ADP = torch.zeros_like(y[:, 0])
        alg_18_H2PIx = torch.zeros_like(y[:, 0])
        alg_19_J_AMP = torch.zeros_like(y[:, 0])
        alg_20_J_Pi1 = torch.zeros_like(y[:, 0])
        alg_21_J_AKi = torch.zeros_like(y[:, 0])
        alg_22_J_Hle = torch.zeros_like(y[:, 0])
        alg_23_J_K = torch.zeros_like(y[:, 0])
        alg_24_J_KH = torch.zeros_like(y[:, 0])
        alg_25_NAD_x = torch.zeros_like(y[:, 0])
        alg_26_J_DH = torch.zeros_like(y[:, 0])
        alg_27_Q = torch.zeros_like(y[:, 0])
        alg_28_dG_C1op = torch.zeros_like(y[:, 0])
        alg_29_J_C1 = torch.zeros_like(y[:, 0])
        alg_30_dG_C3op = torch.zeros_like(y[:, 0])
        alg_31_Cox = torch.zeros_like(y[:, 0])
        alg_32_J_C3 = torch.zeros_like(y[:, 0])
        alg_33_J_C4 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_O2 = torch.zeros_like(y[:, 0]) + (constant_59)
        alg_5_ATP_fx = ATP_x - ATP_mx
        alg_6_J_MgATPx = x_MgA * (alg_5_ATP_fx * Mg_x - K_DT * ATP_mx)
        d_ATP_mx = torch.zeros_like(y[:, 0]) + (alg_6_J_MgATPx / W_x)
        alg_7_ADP_fx = ADP_x - ADP_mx
        alg_9_J_MgADPx = x_MgA * (alg_7_ADP_fx * Mg_x - K_DD * ADP_mx)
        d_Mg_x = torch.zeros_like(y[:, 0]) + ((-alg_6_J_MgATPx - alg_9_J_MgADPx) / W_x)
        d_ADP_mx = torch.zeros_like(y[:, 0]) + (alg_9_J_MgADPx / W_x)
        alg_8_ATP_fi = ATP_i - ATP_mi
        alg_11_J_MgATPi = x_MgA * (alg_8_ATP_fi * Mg_i - K_DT * ATP_mi)
        d_ATP_mi = torch.zeros_like(y[:, 0]) + (alg_11_J_MgATPi / W_i)
        alg_0_dG_H = F * dPsi + RT * torch.log(H_i / H_x)
        alg_2_J_F1 = x_F1 * (torch.exp(-(dG_F1o - n_A * alg_0_dG_H) / RT) * K_DD / K_DT * ADP_mx * Pi_x - ATP_mx * 1.0)
        alg_3_Psi_x = -0.65 * dPsi
        alg_4_Psi_i = 0.35 * dPsi
        alg_10_ADP_fi = ADP_i - ADP_mi
        alg_12_J_ANT = torch.where((alg_10_ADP_fi > mincond) | (alg_8_ATP_fi > mincond), x_ANT * (alg_10_ADP_fi / (alg_10_ADP_fi + alg_8_ATP_fi * torch.exp(-F * alg_4_Psi_i / RT)) - alg_7_ADP_fx / (alg_7_ADP_fx + alg_5_ATP_fx * torch.exp(-F * alg_3_Psi_x / RT))) * alg_10_ADP_fi / (k_mADP + alg_10_ADP_fi), 0.0)
        d_ATP_x = torch.zeros_like(y[:, 0]) + ((alg_2_J_F1 - alg_12_J_ANT) / W_x)
        d_ADP_x = torch.zeros_like(y[:, 0]) + ((-alg_2_J_F1 + alg_12_J_ANT) / W_x)
        alg_13_J_MgADPi = x_MgA * (alg_10_ADP_fi * Mg_i - K_DD * ADP_mi)
        d_ADP_mi = torch.zeros_like(y[:, 0]) + (alg_13_J_MgADPi / W_i)
        alg_16_H2PIi = Pi_i * H_i / (H_i + k_dHPi)
        alg_18_H2PIx = Pi_x * H_x / (H_x + k_dHPi)
        alg_20_J_Pi1 = x_Pi1 * (H_x * alg_16_H2PIi - H_i * alg_18_H2PIx) / (alg_16_H2PIi + k_PiH)
        d_Pi_x = torch.zeros_like(y[:, 0]) + ((-alg_2_J_F1 + alg_20_J_Pi1) / W_x)
        alg_15_J_ATP = gamma * p_A * (ATP_e - ATP_i)
        alg_21_J_AKi = x_AK * (K_AK * ADP_i * ADP_i - AMP_i * ATP_i)
        d_ATP_i = torch.zeros_like(y[:, 0]) + ((alg_15_J_ATP + alg_12_J_ANT + alg_21_J_AKi) / W_i)
        alg_17_J_ADP = gamma * p_A * (ADP_e - ADP_i)
        d_ADP_i = torch.zeros_like(y[:, 0]) + ((alg_17_J_ADP - alg_12_J_ANT - 2.0 * alg_21_J_AKi) / W_i)
        alg_19_J_AMP = gamma * p_A * (AMP_e - AMP_i)
        d_AMP_i = torch.zeros_like(y[:, 0]) + ((alg_19_J_AMP + alg_21_J_AKi) / W_i)
        alg_14_J_Pi2 = gamma * x_Pi2 * (Pi_e - Pi_i)
        d_Pi_i = torch.zeros_like(y[:, 0]) + ((-alg_20_J_Pi1 + alg_14_J_Pi2) / W_i)
        alg_23_J_K = x_K * dPsi * (K_i * torch.exp(F * dPsi / RT) - K_x) / (torch.exp(F * dPsi / RT) - 1.0)
        alg_24_J_KH = x_KH * (K_i * H_x - K_x * H_i)
        d_K_x = torch.zeros_like(y[:, 0]) + ((alg_24_J_KH + alg_23_J_K) / W_x)
        alg_25_NAD_x = NADtot - NADH_x
        alg_26_J_DH = x_DH * (r * alg_25_NAD_x - NADH_x) * (1.0 + Pi_x / k_Pi1) / (1.0 + Pi_x / k_Pi2)
        alg_27_Q = Qtot - QH2
        alg_28_dG_C1op = dG_C1o - RT * torch.log(H_x / 1e-07) - RT * torch.log(alg_27_Q / QH2)
        alg_29_J_C1 = x_C1 * (torch.exp(-(alg_28_dG_C1op + 4.0 * alg_0_dG_H) / RT) * NADH_x - alg_25_NAD_x)
        d_NADH_x = torch.zeros_like(y[:, 0]) + ((alg_26_J_DH - alg_29_J_C1) / W_x)
        alg_30_dG_C3op = dG_C3o + 2.0 * RT * torch.log(H_x / 1e-07) - RT * torch.log(QH2 / alg_27_Q)
        alg_31_Cox = Ctot - Cred
        alg_32_J_C3 = x_C3 * (1.0 + Pi_x / k_Pi3) / (1.0 + Pi_x / k_Pi4) * (torch.exp(-(alg_30_dG_C3op + 4.0 * alg_0_dG_H - 2.0 * F * dPsi) / (2.0 * RT)) * alg_31_Cox - Cred)
        d_QH2 = torch.zeros_like(y[:, 0]) + ((alg_29_J_C1 - alg_32_J_C3) / W_x)
        alg_1_dG_C4op = dG_C4o - 2.0 * RT * torch.log(H_x / 1e-07) - RT / 2.0 * torch.log(O2 / 1.0)
        alg_33_J_C4 = x_C4 * 1.0 / (1.0 + k_O2 / O2) * Cred / Ctot * (torch.exp(-(alg_1_dG_C4op + 2.0 * alg_0_dG_H) / (2.0 * RT)) * Cred - alg_31_Cox * torch.exp(F * dPsi / RT))
        alg_22_J_Hle = x_Hle * dPsi * (H_i * torch.exp(F * dPsi / RT) - H_x) / (torch.exp(F * dPsi / RT) - 1.0)
        d_H_x = torch.zeros_like(y[:, 0]) + (x_buff * H_x * (alg_26_J_DH - 5.0 * alg_29_J_C1 - 2.0 * alg_32_J_C3 - 4.0 * alg_33_J_C4 + (n_A - 1.0) * alg_2_J_F1 + 2.0 * alg_20_J_Pi1 + alg_22_J_Hle - alg_24_J_KH) / W_x)
        d_Cred = torch.zeros_like(y[:, 0]) + ((2.0 * alg_32_J_C3 - 2.0 * alg_33_J_C4) / W_i)
        d_dPsi = torch.zeros_like(y[:, 0]) + ((4.0 * alg_29_J_C1 + 2.0 * alg_32_J_C3 + 4.0 * alg_33_J_C4 - n_A * alg_2_J_F1 - alg_12_J_ANT - alg_22_J_Hle - alg_23_J_K) / C_im)

        return torch.stack(
            [d_dPsi, d_H_x, d_NADH_x, d_Pi_x, d_QH2, d_Cred, d_O2, d_ADP_mx, d_ATP_mx, d_ATP_x, d_Mg_x, d_ADP_x, d_ATP_i, d_ATP_mi, d_ADP_i, d_ADP_mi, d_AMP_i, d_Pi_i, d_K_x],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
