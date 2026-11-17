import torch
import torch.nn as nn
import math


class MetabolismTran2009ODE(nn.Module):
    """
    PyTorch translation of metabolism/tran_2009/model.py.
    Dynamic states: 11; independent constants: 60; derived constants used by rates: 11.
    Original constant slots: 71; algebraic slots: 50.
    """
    source_file = 'metabolism/tran_2009/model.py'
    source_state_dim = 11
    source_constant_dim = 71
    algebraic_dim = 50
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 11
        self.parameter_dim = 60

        self.parameters = (
            parameters
            if parameters is not None
            else {'len_thin': 1.2, 'len_thick': 1.65, 'len_hbare': 0.1, 'Qkon': 1.5, 'Qkoff': 1.3, 'Qkn_p': 1.6, 'Qkp_n': 1.6, 'kon': 0.05, 'koffL': 0.25, 'koffH': 0.025, 'perm50': 0.5, 'nperm': 15, 'kn_p': 0.5, 'kp_n': 0.05, 'koffmod': 1, 'pH': 7.15, 'm': 1, 'kdHCa': 0.02, 'TmpC': 22, 'Cai': 200, 'fapp': 0.5, 'gapp': 0.07, 'hf': 2, 'hb': 0.4, 'gxb': 0.07, 'gslmod': 6, 'hfmdc': 5, 'hbmdc': 0, 'sigmap': 8, 'sigman': 1, 'xbmodsp': 0.2, 'Qfapp': 6.25, 'Qgapp': 2.5, 'Qhf': 6.25, 'Qhb': 6.25, 'Qgxb': 6.25, 'x_0': 0.007, 'kMgATP': 15400000000, 'kdADP': 4, 'xPi_cons': 2000, 'MgATP_cons': 5000, 'MgADP_cons': 36, 'xPi': 2000, 'MgATP': 5000, 'MgADP': 36.3, 'xPsi': 2, 'kxb': 120, 'SL_c': 2.25, 'SLrest': 1.85, 'SLset': 1.9, 'PCon_t': 0.002, 'PExp_t': 10, 'PCon_c': 0.02, 'PExp_c': 70, 'KSE': 1, 'SEon': 1, 'kxb_2': 120, 'Trop_conc': 70, 'dSL': 0, 'H_cons': 0.7079457843841372}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'SL': 2.2, 'TRPNCaL': 0.0147730085063734, 'TRPNCaH': 0.13066096561522, 'xXBpostr': 0.00700005394873882, 'xXBprer': 3.41212828972468e-08, 'XBpostr': 1.81017564383744e-06, 'XBprer': 3.0494964880038e-07, 'N_NoXB': 0.999999959256274, 'P_NoXB': 4.07437173988636e-08, 'P': 0.999997834540066, 'intf': -4.5113452510363e-06}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['len_thin', 'len_thick', 'len_hbare', 'Qkon', 'Qkoff', 'Qkn_p', 'Qkp_n', 'kon', 'koffL', 'koffH', 'perm50', 'nperm', 'kn_p', 'kp_n', 'koffmod', 'pH', 'm', 'kdHCa', 'TmpC', 'Cai', 'fapp', 'gapp', 'hf', 'hb', 'gxb', 'gslmod', 'hfmdc', 'hbmdc', 'sigmap', 'sigman', 'xbmodsp', 'Qfapp', 'Qgapp', 'Qhf', 'Qhb', 'Qgxb', 'x_0', 'kMgATP', 'kdADP', 'xPi_cons', 'MgATP_cons', 'MgADP_cons', 'xPi', 'MgATP', 'MgADP', 'xPsi', 'kxb', 'SL_c', 'SLrest', 'SLset', 'PCon_t', 'PExp_t', 'PCon_c', 'PExp_c', 'KSE', 'SEon', 'kxb_2', 'Trop_conc', 'dSL', 'H_cons']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['SL', 'TRPNCaL', 'TRPNCaH', 'xXBpostr', 'xXBprer', 'XBpostr', 'XBprer', 'N_NoXB', 'P_NoXB', 'P', 'intf']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (11)
        SL = y[:, 0]
        TRPNCaL = y[:, 1]
        TRPNCaH = y[:, 2]
        xXBpostr = y[:, 3]
        xXBprer = y[:, 4]
        XBpostr = y[:, 5]
        XBprer = y[:, 6]
        N_NoXB = y[:, 7]
        P_NoXB = y[:, 8]
        P = y[:, 9]
        intf = y[:, 10]

        # Independent parameters (source constant order)
        len_thin = theta[:, 0]  # constants[0]
        len_thick = theta[:, 1]  # constants[1]
        len_hbare = theta[:, 2]  # constants[2]
        Qkon = theta[:, 3]  # constants[3]
        Qkoff = theta[:, 4]  # constants[4]
        Qkn_p = theta[:, 5]  # constants[5]
        Qkp_n = theta[:, 6]  # constants[6]
        kon = theta[:, 7]  # constants[7]
        koffL = theta[:, 8]  # constants[8]
        koffH = theta[:, 9]  # constants[9]
        perm50 = theta[:, 10]  # constants[10]
        nperm = theta[:, 11]  # constants[11]
        kn_p = theta[:, 12]  # constants[12]
        kp_n = theta[:, 13]  # constants[13]
        koffmod = theta[:, 14]  # constants[14]
        pH = theta[:, 15]  # constants[15]
        m = theta[:, 16]  # constants[16]
        kdHCa = theta[:, 17]  # constants[17]
        TmpC = theta[:, 18]  # constants[18]
        Cai = theta[:, 19]  # constants[19]
        fapp = theta[:, 20]  # constants[20]
        gapp = theta[:, 21]  # constants[21]
        hf = theta[:, 22]  # constants[22]
        hb = theta[:, 23]  # constants[23]
        gxb = theta[:, 24]  # constants[24]
        gslmod = theta[:, 25]  # constants[25]
        hfmdc = theta[:, 26]  # constants[26]
        hbmdc = theta[:, 27]  # constants[27]
        sigmap = theta[:, 28]  # constants[28]
        sigman = theta[:, 29]  # constants[29]
        xbmodsp = theta[:, 30]  # constants[30]
        Qfapp = theta[:, 31]  # constants[31]
        Qgapp = theta[:, 32]  # constants[32]
        Qhf = theta[:, 33]  # constants[33]
        Qhb = theta[:, 34]  # constants[34]
        Qgxb = theta[:, 35]  # constants[35]
        x_0 = theta[:, 36]  # constants[36]
        kMgATP = theta[:, 37]  # constants[37]
        kdADP = theta[:, 38]  # constants[38]
        xPi_cons = theta[:, 39]  # constants[39]
        MgATP_cons = theta[:, 40]  # constants[40]
        MgADP_cons = theta[:, 41]  # constants[41]
        xPi = theta[:, 42]  # constants[42]
        MgATP = theta[:, 43]  # constants[43]
        MgADP = theta[:, 44]  # constants[44]
        xPsi = theta[:, 45]  # constants[45]
        kxb = theta[:, 46]  # constants[46]
        SL_c = theta[:, 47]  # constants[47]
        SLrest = theta[:, 48]  # constants[48]
        SLset = theta[:, 49]  # constants[49]
        PCon_t = theta[:, 50]  # constants[50]
        PExp_t = theta[:, 51]  # constants[51]
        PCon_c = theta[:, 52]  # constants[52]
        PExp_c = theta[:, 53]  # constants[53]
        KSE = theta[:, 54]  # constants[54]
        SEon = theta[:, 55]  # constants[55]
        kxb_2 = theta[:, 56]  # constants[56]
        Trop_conc = theta[:, 57]  # constants[57]
        dSL = theta[:, 58]  # constants[61]
        H_cons = theta[:, 59]  # constants[62]

        # Derived constants from the original initConsts()
        koffLT = koffL * koffmod * Qkoff ** ((TmpC - 37.0) / 10.0)  # constants[58]
        koffHT = koffH * koffmod * Qkoff ** ((TmpC - 37.0) / 10.0)  # constants[59]
        H = 10000000.0 * 10.0 ** (-pH)  # constants[60]
        constant_70 = dSL  # constants[70]
        konT = (kdHCa ** m + H_cons ** m) / (kdHCa ** m + H ** m) * (kon * Qkon ** ((TmpC - 37.0) / 10.0))  # constants[63]
        fappT = fapp * xbmodsp * Qfapp ** ((TmpC - 37.0) / 10.0)  # constants[64]
        alpha1_plus = fappT  # constants[65]
        fxb = kdADP * fapp * hf * (gxb / MgATP_cons) / (gapp / xPi_cons * (hb / H_cons) * kMgATP)  # constants[66]
        SSXBpostr = (fapp * hf + fxb * gapp + fxb * hb) / (hf * gxb + hb * gapp + gxb * gapp + fxb * hb + gxb * fapp + hb * fapp + fxb * gapp + fapp * hf + fxb * hb)  # constants[67]
        Fnordv = kxb * x_0 * SSXBpostr  # constants[68]
        preload = torch.abs(SLset - SLrest) / (SLset - SLrest) * PCon_t * (torch.exp(PExp_t * torch.abs(SLset - SLrest)) - 1.0)  # constants[69]

        # Initialize rates and algebraic variables, as in computeRates()
        d_SL = torch.zeros_like(y[:, 0])
        d_TRPNCaL = torch.zeros_like(y[:, 0])
        d_TRPNCaH = torch.zeros_like(y[:, 0])
        d_xXBpostr = torch.zeros_like(y[:, 0])
        d_xXBprer = torch.zeros_like(y[:, 0])
        d_XBpostr = torch.zeros_like(y[:, 0])
        d_XBprer = torch.zeros_like(y[:, 0])
        d_N_NoXB = torch.zeros_like(y[:, 0])
        d_P_NoXB = torch.zeros_like(y[:, 0])
        d_P = torch.zeros_like(y[:, 0])
        d_intf = torch.zeros_like(y[:, 0])
        alg_0_sovr_ze = torch.zeros_like(y[:, 0])
        alg_1_sovr_cle = torch.zeros_like(y[:, 0])
        alg_2_len_sovr = torch.zeros_like(y[:, 0])
        alg_3_SOVFThick = torch.zeros_like(y[:, 0])
        alg_4_SOVFThin = torch.zeros_like(y[:, 0])
        alg_5_force = torch.zeros_like(y[:, 0])
        alg_6_Tropreg = torch.zeros_like(y[:, 0])
        alg_7_dTRPNCaL = torch.zeros_like(y[:, 0])
        alg_8_active = torch.zeros_like(y[:, 0])
        alg_9_permtot = torch.zeros_like(y[:, 0])
        alg_10_dTRPNCaH = torch.zeros_like(y[:, 0])
        alg_11_ppforce_t = torch.zeros_like(y[:, 0])
        alg_12_kn_pT = torch.zeros_like(y[:, 0])
        alg_13_gapslmd = torch.zeros_like(y[:, 0])
        alg_14_ppforce_c = torch.zeros_like(y[:, 0])
        alg_15_inprmt = torch.zeros_like(y[:, 0])
        alg_16_gappT = torch.zeros_like(y[:, 0])
        alg_17_ppforce = torch.zeros_like(y[:, 0])
        alg_18_kp_nT = torch.zeros_like(y[:, 0])
        alg_19_hfmd = torch.zeros_like(y[:, 0])
        alg_20_afterload = torch.zeros_like(y[:, 0])
        alg_21_hbmd = torch.zeros_like(y[:, 0])
        alg_22_N = torch.zeros_like(y[:, 0])
        alg_23_hfT = torch.zeros_like(y[:, 0])
        alg_24_hbT = torch.zeros_like(y[:, 0])
        alg_25_gxbmd = torch.zeros_like(y[:, 0])
        alg_26_gxbT = torch.zeros_like(y[:, 0])
        alg_27_alpha2_plus = torch.zeros_like(y[:, 0])
        alg_28_alpha3_plus = torch.zeros_like(y[:, 0])
        alg_29_alpha1_minus = torch.zeros_like(y[:, 0])
        alg_30_alpha2_minus = torch.zeros_like(y[:, 0])
        alg_31_dXBprer = torch.zeros_like(y[:, 0])
        alg_32_fxbT = torch.zeros_like(y[:, 0])
        alg_33_alpha3_minus = torch.zeros_like(y[:, 0])
        alg_34_dXBpostr = torch.zeros_like(y[:, 0])
        alg_35_dutyprer = torch.zeros_like(y[:, 0])
        alg_36_dxXBprer = torch.zeros_like(y[:, 0])
        alg_37_dutypostr = torch.zeros_like(y[:, 0])
        alg_38_dxXBpostr = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_SL = torch.zeros_like(y[:, 0]) + (constant_70)
        alg_7_dTRPNCaL = konT * Cai * (1.0 - TRPNCaL) - koffLT * TRPNCaL
        d_TRPNCaL = torch.zeros_like(y[:, 0]) + (alg_7_dTRPNCaL)
        alg_10_dTRPNCaH = konT * Cai * (1.0 - TRPNCaH) - koffHT * TRPNCaH
        d_TRPNCaH = torch.zeros_like(y[:, 0]) + (alg_10_dTRPNCaH)
        alg_0_sovr_ze = torch.where(len_thick / 2.0 < SL / 2.0, len_thick / 2.0, SL / 2.0)
        alg_1_sovr_cle = torch.where(SL / 2.0 - (SL - len_thin) > len_hbare / 2.0, SL / 2.0 - (SL - len_thin), len_hbare / 2.0)
        alg_2_len_sovr = alg_0_sovr_ze - alg_1_sovr_cle
        alg_4_SOVFThin = alg_2_len_sovr / len_thin
        alg_6_Tropreg = (1.0 - alg_4_SOVFThin) * TRPNCaL + alg_4_SOVFThin * TRPNCaH
        alg_9_permtot = torch.abs(1.0 / (1.0 + (perm50 / alg_6_Tropreg) ** nperm)) ** (1.0 / 2)
        alg_12_kn_pT = kn_p * alg_9_permtot * Qkn_p ** ((TmpC - 37.0) / 10.0)
        alg_15_inprmt = torch.where(1.0 / alg_9_permtot < 100.0, 1.0 / alg_9_permtot, 100.0)
        alg_18_kp_nT = kp_n * alg_15_inprmt * Qkp_n ** ((TmpC - 37.0) / 10.0)
        d_N_NoXB = torch.zeros_like(y[:, 0]) + (alg_18_kp_nT * P_NoXB - alg_12_kn_pT * N_NoXB)
        d_P_NoXB = torch.zeros_like(y[:, 0]) + (alg_12_kn_pT * N_NoXB - alg_18_kp_nT * P_NoXB)
        alg_3_SOVFThick = alg_2_len_sovr * 2.0 / (len_thick - len_hbare)
        alg_5_force = kxb * alg_3_SOVFThick * (xXBpostr * XBpostr + xXBprer * XBprer)
        alg_8_active = 1.0 * alg_5_force / Fnordv
        alg_11_ppforce_t = (SL - SLrest) / torch.abs(SL - SLrest) * PCon_t * (torch.exp(PExp_t * torch.abs(SL - SLrest)) - 1.0)
        alg_14_ppforce_c = torch.where(SL > SL_c, PCon_c * (torch.exp(PExp_c * torch.abs(SL - SL_c)) - 1.0), 0.0)
        alg_17_ppforce = alg_11_ppforce_t + alg_14_ppforce_c
        alg_20_afterload = torch.where(SEon == 1.0, KSE * (SLset - SL), 0.0)
        d_intf = torch.zeros_like(y[:, 0]) + (preload + alg_20_afterload - (alg_17_ppforce + alg_8_active))
        alg_22_N = 1.0 - (P + XBprer + XBpostr)
        d_P = torch.zeros_like(y[:, 0]) + (alg_12_kn_pT * alg_22_N - (alg_18_kp_nT + alpha1_plus) * P)
        alg_19_hfmd = torch.exp(-xXBprer / torch.abs(xXBprer) * hfmdc * (xXBprer / x_0) ** 2.0)
        alg_23_hfT = hf * alg_19_hfmd * xbmodsp * Qhf ** ((TmpC - 37.0) / 10.0)
        alg_27_alpha2_plus = alg_23_hfT
        alg_13_gapslmd = 1.0 + (1.0 - alg_3_SOVFThick) * gslmod
        alg_16_gappT = gapp * alg_13_gapslmd * xbmodsp * Qgapp ** ((TmpC - 37.0) / 10.0)
        alg_29_alpha1_minus = xPi * (alg_16_gappT / xPi_cons)
        alg_21_hbmd = torch.exp((xXBpostr - x_0) / torch.abs(xXBpostr - x_0) * hbmdc * ((xXBpostr - x_0) / x_0) ** 2.0)
        alg_24_hbT = hb * alg_21_hbmd * xbmodsp * Qhb ** ((TmpC - 37.0) / 10.0)
        alg_30_alpha2_minus = H * (alg_24_hbT / H_cons) * ((kdADP + MgADP_cons) / MgADP_cons) * (MgADP / (kdADP + MgADP))
        alg_31_dXBprer = alpha1_plus * P + alg_30_alpha2_minus * XBpostr - (alg_29_alpha1_minus + alg_27_alpha2_plus) * XBprer
        d_XBprer = torch.zeros_like(y[:, 0]) + (alg_31_dXBprer)
        alg_25_gxbmd = torch.where(xXBpostr < x_0, torch.exp(sigmap * ((x_0 - xXBpostr) / x_0) ** 2.0), torch.exp(sigman * ((xXBpostr - x_0) / x_0) ** 2.0))
        alg_26_gxbT = gxb * alg_25_gxbmd * xbmodsp * Qgxb ** ((TmpC - 37.0) / 10.0)
        alg_28_alpha3_plus = MgATP * (alg_26_gxbT / MgATP_cons) * ((kdADP + MgADP_cons) / (kdADP + MgADP))
        alg_32_fxbT = kdADP * fappT * alg_23_hfT * (alg_26_gxbT / MgATP_cons) / (alg_16_gappT / xPi_cons * (alg_24_hbT / H_cons) * kMgATP)
        alg_33_alpha3_minus = alg_32_fxbT
        alg_34_dXBpostr = alg_33_alpha3_minus * P + alg_27_alpha2_plus * XBprer - (alg_30_alpha2_minus + alg_28_alpha3_plus) * XBpostr
        d_XBpostr = torch.zeros_like(y[:, 0]) + (alg_34_dXBpostr)
        alg_35_dutyprer = (alg_33_alpha3_minus * alg_30_alpha2_minus + alg_28_alpha3_plus * alpha1_plus + alg_30_alpha2_minus * alpha1_plus) / (alpha1_plus * alg_27_alpha2_plus + alg_33_alpha3_minus * alg_29_alpha1_minus + alg_33_alpha3_minus * alg_27_alpha2_plus + alg_33_alpha3_minus * alg_30_alpha2_minus + alg_28_alpha3_plus * alpha1_plus + alg_30_alpha2_minus * alpha1_plus + alg_27_alpha2_plus * alg_28_alpha3_plus + alg_33_alpha3_minus * alg_29_alpha1_minus + alg_28_alpha3_plus * alg_29_alpha1_minus)
        alg_36_dxXBprer = dSL / 2.0 + xPsi / alg_35_dutyprer * (-(alpha1_plus * xXBprer) + alg_30_alpha2_minus * (xXBpostr - (x_0 + xXBprer)))
        d_xXBprer = torch.zeros_like(y[:, 0]) + (alg_36_dxXBprer)
        alg_37_dutypostr = (alpha1_plus * alg_27_alpha2_plus + alg_33_alpha3_minus * alg_29_alpha1_minus + alg_33_alpha3_minus * alg_27_alpha2_plus) / (alpha1_plus * alg_27_alpha2_plus + alg_33_alpha3_minus * alg_29_alpha1_minus + alg_33_alpha3_minus * alg_27_alpha2_plus + alg_33_alpha3_minus * alg_30_alpha2_minus + alg_28_alpha3_plus * alpha1_plus + alg_30_alpha2_minus * alpha1_plus + alg_27_alpha2_plus * alg_28_alpha3_plus + alg_33_alpha3_minus * alg_29_alpha1_minus + alg_28_alpha3_plus * alg_29_alpha1_minus)
        alg_38_dxXBpostr = dSL / 2.0 + xPsi / alg_37_dutypostr * (alg_27_alpha2_plus * (xXBprer + x_0 - xXBpostr))
        d_xXBpostr = torch.zeros_like(y[:, 0]) + (alg_38_dxXBpostr)

        return torch.stack(
            [d_SL, d_TRPNCaL, d_TRPNCaH, d_xXBpostr, d_xXBprer, d_XBpostr, d_XBprer, d_N_NoXB, d_P_NoXB, d_P, d_intf],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
