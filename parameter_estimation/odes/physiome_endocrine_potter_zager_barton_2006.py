import torch
import torch.nn as nn
import math


class EndocrinePotterZagerBarton2006ODE(nn.Module):
    """
    PyTorch translation of endocrine/potter_zager_barton_2006/model.py.
    Dynamic states: 33; independent constants: 90; derived constants used by rates: 7.
    Original constant slots: 97; algebraic slots: 24.
    """
    source_file = 'endocrine/potter_zager_barton_2006/model.py'
    source_state_dim = 33
    source_constant_dim = 97
    algebraic_dim = 24
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 33
        self.parameter_dim = 90

        self.parameters = (
            parameters
            if parameters is not None
            else {'k2T': 0.17, 'Ql': 1.06, 'Qt': 0.061, 'CTbl': 7.6, 'ts': 0.56, 'CDbl': 0.59, 'kTA_on': 0.18, 'kTA_off': 8131, 'kDA_on': 0.15, 'kDA_off': 5407, 'CA': 500000, 'k1T': 87.93, 'km5alpha': 40, 'Vmax1': 3.65, 'PTl': 2.75, 'k1D': 77.2, 'PDl': 2, 'CTl': 9.92, 'CDl': 0.92, 'mu': 1000, 'PTif': 1.96, 'CTif': 307.29, 'CTst': 307.29, 'CLH': 0.104, 'PTst': 1, 'Vmax0': 2.89, 'kTR_on': 0.14, 'kTR_off': 0.069, 'kDR_on': 0.053, 'kDR_off': 0.018, 'kDT_on': 0.14, 'kDT_off': 3.13, 'kDD_on': 0.14, 'kDD_off': 3.13, 'kTT_on': 0.14, 'kTT_off': 3.13, 'kDNA_onDD5a': 0.14, 'kDNA_offDD5a': 0.07, 'kDNA_onDDcd': 0.14, 'kDNA_offDDcd': 0.1, 'kDNA_onDDsec': 0.14, 'kDNA_offDDsec': 0.2, 'kDNA_onDDcp': 0.14, 'kDNA_offDDcp': 0.7, 'kDNA_onDT5a': 0.14, 'kDNA_offDT5a': 0.56, 'kDNA_onDTcd': 0.14, 'kDNA_offDTcd': 0.8, 'kDNA_onDTsec': 0.14, 'kDNA_offDTsec': 1.6, 'kDNA_onDTcp': 0.14, 'kDNA_offDTcp': 5.6, 'kDNA_onTT5a': 0.14, 'kDNA_offTT5a': 0.84, 'kDNA_onTTcd': 0.14, 'kDNA_offTTcd': 1.2, 'kDNA_onTTsec': 0.14, 'kDNA_offTTsec': 2.4, 'kDNA_onTTcp': 0.14, 'kDNA_offTTcp': 8.4, 'NTp': 0.79, 'CTp': 11.38, 'NDp': 1, 'CDp': 52.46, 'CDNAcd': 0.075, 'CDNAsec': 0.075, 'CDNA5a': 0.075, 'CDNAcp': 0.075, 'CTb': 0.6275, 'PTb': 1, 'CDb': 0.019884, 'PDb': 0.62, 'kLH1': 0.13, 'kLH2': 0.026, 'kLH3': 0.000168, 'kLH4': 0.8, 'alpha_T': 0.25, 'alpha_D': 0.75, 'k1R': 68.15, 'keR': 71.9, 'kQp': 73.84, 'VPC2': 5.08e-05, 'k5a': 3.02, 'kcpl': 1.17e-07, 'kcd1': 0.014, 'VPC1b': 0.0001, 'ksec': 8.44e-06, 'kflo': 0.043, 'VPL2': 3.08e-06, 'Qc': 6.08}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'ATbl': 0, 'ADbl': 0, 'CT_A': 0, 'CD_A': 0, 'ATl': 0, 'ADl': 0, 'ATif': 0, 'ATst': 0, 'ATp': 0, 'ADp': 0, 'CT_R': 0, 'CD_R': 0, 'CDT': 0, 'CDD': 0, 'CTT': 0, 'CDD5a': 0, 'CDDcd': 0, 'CDDsec': 0, 'CDDcp': 0, 'CDT5a': 0, 'CDTcd': 0, 'CDTsec': 0, 'CDTcp': 0, 'CTT5a': 0, 'CTTcd': 0, 'CTTsec': 0, 'CTTcp': 0, 'ATb': 0, 'ADb': 0, 'ALH': 0, 'AR': 0, 'VPC1': 0, 'VPL1': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k2T', 'Ql', 'Qt', 'CTbl', 'ts', 'CDbl', 'kTA_on', 'kTA_off', 'kDA_on', 'kDA_off', 'CA', 'k1T', 'km5alpha', 'Vmax1', 'PTl', 'k1D', 'PDl', 'CTl', 'CDl', 'mu', 'PTif', 'CTif', 'CTst', 'CLH', 'PTst', 'Vmax0', 'kTR_on', 'kTR_off', 'kDR_on', 'kDR_off', 'kDT_on', 'kDT_off', 'kDD_on', 'kDD_off', 'kTT_on', 'kTT_off', 'kDNA_onDD5a', 'kDNA_offDD5a', 'kDNA_onDDcd', 'kDNA_offDDcd', 'kDNA_onDDsec', 'kDNA_offDDsec', 'kDNA_onDDcp', 'kDNA_offDDcp', 'kDNA_onDT5a', 'kDNA_offDT5a', 'kDNA_onDTcd', 'kDNA_offDTcd', 'kDNA_onDTsec', 'kDNA_offDTsec', 'kDNA_onDTcp', 'kDNA_offDTcp', 'kDNA_onTT5a', 'kDNA_offTT5a', 'kDNA_onTTcd', 'kDNA_offTTcd', 'kDNA_onTTsec', 'kDNA_offTTsec', 'kDNA_onTTcp', 'kDNA_offTTcp', 'NTp', 'CTp', 'NDp', 'CDp', 'CDNAcd', 'CDNAsec', 'CDNA5a', 'CDNAcp', 'CTb', 'PTb', 'CDb', 'PDb', 'kLH1', 'kLH2', 'kLH3', 'kLH4', 'alpha_T', 'alpha_D', 'k1R', 'keR', 'kQp', 'VPC2', 'k5a', 'kcpl', 'kcd1', 'VPC1b', 'ksec', 'kflo', 'VPL2', 'Qc']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['ATbl', 'ADbl', 'CT_A', 'CD_A', 'ATl', 'ADl', 'ATif', 'ATst', 'ATp', 'ADp', 'CT_R', 'CD_R', 'CDT', 'CDD', 'CTT', 'CDD5a', 'CDDcd', 'CDDsec', 'CDDcp', 'CDT5a', 'CDTcd', 'CDTsec', 'CDTcp', 'CTT5a', 'CTTcd', 'CTTsec', 'CTTcp', 'ATb', 'ADb', 'ALH', 'AR', 'VPC1', 'VPL1']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (33)
        ATbl = y[:, 0]
        ADbl = y[:, 1]
        CT_A = y[:, 2]
        CD_A = y[:, 3]
        ATl = y[:, 4]
        ADl = y[:, 5]
        ATif = y[:, 6]
        ATst = y[:, 7]
        ATp = y[:, 8]
        ADp = y[:, 9]
        CT_R = y[:, 10]
        CD_R = y[:, 11]
        CDT = y[:, 12]
        CDD = y[:, 13]
        CTT = y[:, 14]
        CDD5a = y[:, 15]
        CDDcd = y[:, 16]
        CDDsec = y[:, 17]
        CDDcp = y[:, 18]
        CDT5a = y[:, 19]
        CDTcd = y[:, 20]
        CDTsec = y[:, 21]
        CDTcp = y[:, 22]
        CTT5a = y[:, 23]
        CTTcd = y[:, 24]
        CTTsec = y[:, 25]
        CTTcp = y[:, 26]
        ATb = y[:, 27]
        ADb = y[:, 28]
        ALH = y[:, 29]
        AR = y[:, 30]
        VPC1 = y[:, 31]
        VPL1 = y[:, 32]

        # Independent parameters (source constant order)
        k2T = theta[:, 0]  # constants[0]
        Ql = theta[:, 1]  # constants[1]
        Qt = theta[:, 2]  # constants[2]
        CTbl = theta[:, 3]  # constants[3]
        ts = theta[:, 4]  # constants[4]
        CDbl = theta[:, 5]  # constants[5]
        kTA_on = theta[:, 6]  # constants[6]
        kTA_off = theta[:, 7]  # constants[7]
        kDA_on = theta[:, 8]  # constants[8]
        kDA_off = theta[:, 9]  # constants[9]
        CA = theta[:, 10]  # constants[10]
        k1T = theta[:, 11]  # constants[11]
        km5alpha = theta[:, 12]  # constants[12]
        Vmax1 = theta[:, 13]  # constants[13]
        PTl = theta[:, 14]  # constants[14]
        k1D = theta[:, 15]  # constants[15]
        PDl = theta[:, 16]  # constants[16]
        CTl = theta[:, 17]  # constants[17]
        CDl = theta[:, 18]  # constants[18]
        mu = theta[:, 19]  # constants[19]
        PTif = theta[:, 20]  # constants[20]
        CTif = theta[:, 21]  # constants[21]
        CTst = theta[:, 22]  # constants[22]
        CLH = theta[:, 23]  # constants[23]
        PTst = theta[:, 24]  # constants[24]
        Vmax0 = theta[:, 25]  # constants[25]
        kTR_on = theta[:, 26]  # constants[26]
        kTR_off = theta[:, 27]  # constants[27]
        kDR_on = theta[:, 28]  # constants[28]
        kDR_off = theta[:, 29]  # constants[29]
        kDT_on = theta[:, 30]  # constants[30]
        kDT_off = theta[:, 31]  # constants[31]
        kDD_on = theta[:, 32]  # constants[32]
        kDD_off = theta[:, 33]  # constants[33]
        kTT_on = theta[:, 34]  # constants[34]
        kTT_off = theta[:, 35]  # constants[35]
        kDNA_onDD5a = theta[:, 36]  # constants[36]
        kDNA_offDD5a = theta[:, 37]  # constants[37]
        kDNA_onDDcd = theta[:, 38]  # constants[38]
        kDNA_offDDcd = theta[:, 39]  # constants[39]
        kDNA_onDDsec = theta[:, 40]  # constants[40]
        kDNA_offDDsec = theta[:, 41]  # constants[41]
        kDNA_onDDcp = theta[:, 42]  # constants[42]
        kDNA_offDDcp = theta[:, 43]  # constants[43]
        kDNA_onDT5a = theta[:, 44]  # constants[44]
        kDNA_offDT5a = theta[:, 45]  # constants[45]
        kDNA_onDTcd = theta[:, 46]  # constants[46]
        kDNA_offDTcd = theta[:, 47]  # constants[47]
        kDNA_onDTsec = theta[:, 48]  # constants[48]
        kDNA_offDTsec = theta[:, 49]  # constants[49]
        kDNA_onDTcp = theta[:, 50]  # constants[50]
        kDNA_offDTcp = theta[:, 51]  # constants[51]
        kDNA_onTT5a = theta[:, 52]  # constants[52]
        kDNA_offTT5a = theta[:, 53]  # constants[53]
        kDNA_onTTcd = theta[:, 54]  # constants[54]
        kDNA_offTTcd = theta[:, 55]  # constants[55]
        kDNA_onTTsec = theta[:, 56]  # constants[56]
        kDNA_offTTsec = theta[:, 57]  # constants[57]
        kDNA_onTTcp = theta[:, 58]  # constants[58]
        kDNA_offTTcp = theta[:, 59]  # constants[59]
        NTp = theta[:, 60]  # constants[60]
        CTp = theta[:, 61]  # constants[61]
        NDp = theta[:, 62]  # constants[62]
        CDp = theta[:, 63]  # constants[63]
        CDNAcd = theta[:, 64]  # constants[64]
        CDNAsec = theta[:, 65]  # constants[65]
        CDNA5a = theta[:, 66]  # constants[66]
        CDNAcp = theta[:, 67]  # constants[67]
        CTb = theta[:, 68]  # constants[68]
        PTb = theta[:, 69]  # constants[69]
        CDb = theta[:, 70]  # constants[70]
        PDb = theta[:, 71]  # constants[71]
        kLH1 = theta[:, 72]  # constants[72]
        kLH2 = theta[:, 73]  # constants[73]
        kLH3 = theta[:, 74]  # constants[74]
        kLH4 = theta[:, 75]  # constants[75]
        alpha_T = theta[:, 76]  # constants[76]
        alpha_D = theta[:, 77]  # constants[77]
        k1R = theta[:, 78]  # constants[78]
        keR = theta[:, 79]  # constants[79]
        kQp = theta[:, 80]  # constants[80]
        VPC2 = theta[:, 81]  # constants[81]
        k5a = theta[:, 82]  # constants[82]
        kcpl = theta[:, 83]  # constants[83]
        kcd1 = theta[:, 84]  # constants[84]
        VPC1b = theta[:, 85]  # constants[85]
        ksec = theta[:, 86]  # constants[86]
        kflo = theta[:, 87]  # constants[87]
        VPL2 = theta[:, 88]  # constants[88]
        Qc = theta[:, 89]  # constants[89]

        # Derived constants from the original initConsts()
        CTlf = CTl / PTl  # constants[90]
        B = kLH3  # constants[91]
        constant_96 = mu * (CTif - CTst / PTst)  # constants[96]
        CDlf = CDl / PDl  # constants[92]
        CTtv = CTif / PTif  # constants[93]
        CTbf = CTb / PTb  # constants[94]
        CDbf = CDb / PDb  # constants[95]

        # Initialize rates and algebraic variables, as in computeRates()
        d_ATbl = torch.zeros_like(y[:, 0])
        d_ADbl = torch.zeros_like(y[:, 0])
        d_CT_A = torch.zeros_like(y[:, 0])
        d_CD_A = torch.zeros_like(y[:, 0])
        d_ATl = torch.zeros_like(y[:, 0])
        d_ADl = torch.zeros_like(y[:, 0])
        d_ATif = torch.zeros_like(y[:, 0])
        d_ATst = torch.zeros_like(y[:, 0])
        d_ATp = torch.zeros_like(y[:, 0])
        d_ADp = torch.zeros_like(y[:, 0])
        d_CT_R = torch.zeros_like(y[:, 0])
        d_CD_R = torch.zeros_like(y[:, 0])
        d_CDT = torch.zeros_like(y[:, 0])
        d_CDD = torch.zeros_like(y[:, 0])
        d_CTT = torch.zeros_like(y[:, 0])
        d_CDD5a = torch.zeros_like(y[:, 0])
        d_CDDcd = torch.zeros_like(y[:, 0])
        d_CDDsec = torch.zeros_like(y[:, 0])
        d_CDDcp = torch.zeros_like(y[:, 0])
        d_CDT5a = torch.zeros_like(y[:, 0])
        d_CDTcd = torch.zeros_like(y[:, 0])
        d_CDTsec = torch.zeros_like(y[:, 0])
        d_CDTcp = torch.zeros_like(y[:, 0])
        d_CTT5a = torch.zeros_like(y[:, 0])
        d_CTTcd = torch.zeros_like(y[:, 0])
        d_CTTsec = torch.zeros_like(y[:, 0])
        d_CTTcp = torch.zeros_like(y[:, 0])
        d_ATb = torch.zeros_like(y[:, 0])
        d_ADb = torch.zeros_like(y[:, 0])
        d_ALH = torch.zeros_like(y[:, 0])
        d_AR = torch.zeros_like(y[:, 0])
        d_VPC1 = torch.zeros_like(y[:, 0])
        d_VPL1 = torch.zeros_like(y[:, 0])
        alg_0_CTblf = torch.zeros_like(y[:, 0])
        alg_1_DNAocp = torch.zeros_like(y[:, 0])
        alg_2_DNAosec = torch.zeros_like(y[:, 0])
        alg_4_CDblf = torch.zeros_like(y[:, 0])
        alg_5_DNAocd = torch.zeros_like(y[:, 0])
        alg_7_T = torch.zeros_like(y[:, 0])
        alg_8_CAf = torch.zeros_like(y[:, 0])
        alg_9_A = torch.zeros_like(y[:, 0])
        alg_10_CTpf = torch.zeros_like(y[:, 0])
        alg_11_D = torch.zeros_like(y[:, 0])
        alg_12_CDpf = torch.zeros_like(y[:, 0])
        alg_13_Vpc = torch.zeros_like(y[:, 0])
        alg_14_CDNAcdf = torch.zeros_like(y[:, 0])
        alg_15_Qp = torch.zeros_like(y[:, 0])
        alg_16_CDNAsecf = torch.zeros_like(y[:, 0])
        alg_17_Qb = torch.zeros_like(y[:, 0])
        alg_18_Qbt = torch.zeros_like(y[:, 0])
        alg_19_CDNA5af = torch.zeros_like(y[:, 0])
        alg_20_CDNAcpf = torch.zeros_like(y[:, 0])
        alg_21_Vp = torch.zeros_like(y[:, 0])
        alg_22_CR = torch.zeros_like(y[:, 0])
        alg_23_CRf = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_ATst = torch.zeros_like(y[:, 0]) + (constant_96)
        d_ATl = torch.zeros_like(y[:, 0]) + (Ql * (CTbl - CTlf) - (Vmax1 * CTlf / (km5alpha + CTlf) + k1T * (ATl / PTl)))
        d_ADl = torch.zeros_like(y[:, 0]) + (Ql * (CDbl - CDlf) - (Vmax1 * CTlf / (km5alpha + CTlf) + k1D * (ADl / PDl)))
        d_CDT = torch.zeros_like(y[:, 0]) + (kDT_on * CD_R * CT_R - kDT_off * CDT)
        d_CDD = torch.zeros_like(y[:, 0]) + (kDD_on * CD_R ** 2.0 - kDD_off * CDD)
        d_CTT = torch.zeros_like(y[:, 0]) + (kTT_on * CT_R ** 2.0 - kTT_off * CTT)
        alg_0_CTblf = CTbl - CT_A
        d_ATif = torch.zeros_like(y[:, 0]) + (Qt * (1.0 - ts) * (alg_0_CTblf - CTif / PTif) + mu * (CTst / PTst - CTif) + k1T * CLH)
        alg_2_DNAosec = (CTTsec + CDDsec + CDTsec) / CDNAsec
        d_VPL1 = torch.zeros_like(y[:, 0]) + (VPC1 / VPC1b * ksec * alg_2_DNAosec - kflo * VPL1)
        alg_1_DNAocp = (CTTcp + CDDcp + CDTcp) / CDNAcp
        alg_5_DNAocd = (CTTcd + CDDcd + CDTcd) / CDNAcd
        d_VPC1 = torch.zeros_like(y[:, 0]) + (VPC1 / VPC1b * kcpl * alg_1_DNAocp - kcd1 * (1.0 - alg_5_DNAocd) * VPC1)
        alg_8_CAf = CA - (CT_A + CD_A)
        d_CT_A = torch.zeros_like(y[:, 0]) + (kTA_on * alg_0_CTblf * alg_8_CAf - kTA_off * CT_A)
        alg_4_CDblf = CDbl - CD_A
        d_CD_A = torch.zeros_like(y[:, 0]) + (kDA_on * alg_4_CDblf * alg_8_CAf - kDA_off * CD_A)
        alg_9_A = -kLH1 * (alpha_T * alg_0_CTblf + alpha_D * alg_4_CDblf) + kLH2
        d_ALH = torch.zeros_like(y[:, 0]) + (torch.where(alg_9_A > B, alg_9_A - kLH4 * ALH, B - kLH4 * ALH))
        alg_14_CDNAcdf = CDNAcd - (CTTcd + CDDcd + CDTcd)
        d_CDDcd = torch.zeros_like(y[:, 0]) + (kDNA_onDDcd * CDD * alg_14_CDNAcdf - kDNA_offDDcd * CDDcd)
        d_CDTcd = torch.zeros_like(y[:, 0]) + (kDNA_onDTcd * CDT * alg_14_CDNAcdf - kDNA_offDTcd * CDTcd)
        d_CTTcd = torch.zeros_like(y[:, 0]) + (kDNA_onTTcd * CTT * alg_14_CDNAcdf - kDNA_offTTcd * CTTcd)
        alg_13_Vpc = VPC1 + VPC2
        alg_15_Qp = kQp * alg_13_Vpc
        alg_7_T = CTTsec + CDTsec + CTT5a + CDT5a + CTTcd + CDTcd + CTTcp + CDTcp
        alg_10_CTpf = (CTp - (CT_R + 2.0 * CTT + CDT + alg_7_T)) / (1.0 + NTp)
        d_ATp = torch.zeros_like(y[:, 0]) + (alg_15_Qp * (CTbl - alg_10_CTpf) - Vmax0 * alg_10_CTpf / (km5alpha + alg_10_CTpf))
        alg_11_D = CDDsec + CDTsec + CDD5a + CDT5a + CDDcd + CDTcd + CDDcp + CDTcp
        alg_12_CDpf = (CDp - (CD_R + 2.0 * CDD + CDT + alg_11_D)) / (1.0 + NDp)
        d_ADp = torch.zeros_like(y[:, 0]) + (alg_15_Qp * (CDbl - alg_12_CDpf) - Vmax0 * alg_10_CTpf / (km5alpha + alg_10_CTpf))
        alg_16_CDNAsecf = CDNAsec - (CTTsec + CDDsec + CDTsec)
        d_CDDsec = torch.zeros_like(y[:, 0]) + (kDNA_onDDsec * CDD * alg_16_CDNAsecf - kDNA_offDDsec * CDDsec)
        d_CDTsec = torch.zeros_like(y[:, 0]) + (kDNA_onDTsec * CDT * alg_16_CDNAsecf - kDNA_offDTsec * CDTsec)
        d_CTTsec = torch.zeros_like(y[:, 0]) + (kDNA_onTTsec * CTT * alg_16_CDNAsecf - kDNA_offTTsec * CTTsec)
        alg_17_Qb = Qc - (alg_15_Qp + Qt + Ql)
        d_ATbl = torch.zeros_like(y[:, 0]) + (alg_17_Qb * (CTbf - alg_0_CTblf) + alg_15_Qp * (alg_10_CTpf - CTbl) + Ql * (CTlf - CTbl) + Qt * (1.0 - ts) * (CTtv - alg_0_CTblf) + k2T)
        alg_18_Qbt = Qc - (alg_15_Qp + Ql)
        d_ADbl = torch.zeros_like(y[:, 0]) + (alg_18_Qbt * (CDbf - alg_4_CDblf) + alg_15_Qp * (alg_12_CDpf - alg_4_CDblf) + Ql * (CDlf - CDbl))
        alg_19_CDNA5af = CDNA5a - (CTT5a + CDD5a + CDT5a)
        d_CDD5a = torch.zeros_like(y[:, 0]) + (kDNA_onDD5a * CDD * alg_19_CDNA5af - kDNA_offDD5a * CDD5a)
        d_CDT5a = torch.zeros_like(y[:, 0]) + (kDNA_onDT5a * CDT * alg_19_CDNA5af - kDNA_offDT5a * CDT5a)
        d_CTT5a = torch.zeros_like(y[:, 0]) + (kDNA_onTT5a * CTT * alg_19_CDNA5af - kDNA_offTT5a * CTT5a)
        d_ATb = torch.zeros_like(y[:, 0]) + (alg_17_Qb * (alg_0_CTblf - CTb / PTb))
        d_ADb = torch.zeros_like(y[:, 0]) + (alg_18_Qbt * (alg_4_CDblf - CDb / PDb))
        alg_20_CDNAcpf = CDNAcp - (CTTcp + CDDcp + CDTcp)
        d_CDDcp = torch.zeros_like(y[:, 0]) + (kDNA_onDDcp * CDD * alg_20_CDNAcpf - kDNA_offDDcp * CDDcp)
        d_CDTcp = torch.zeros_like(y[:, 0]) + (kDNA_onDTcp * CDT * alg_20_CDNAcpf - kDNA_offDTcp * CDTcp)
        d_CTTcp = torch.zeros_like(y[:, 0]) + (kDNA_onTTcp * CTT * alg_20_CDNAcpf - kDNA_offTTcp * CTTcp)
        alg_21_Vp = VPC1 + VPC2 + VPL1 + VPL2
        alg_22_CR = AR / alg_21_Vp
        alg_23_CRf = alg_22_CR - (CT_R + CD_R + 2.0 * (CTT + CDD + CDT) + 2.0 * (alg_14_CDNAcdf + alg_16_CDNAsecf + alg_20_CDNAcpf + alg_19_CDNA5af))
        d_CT_R = torch.zeros_like(y[:, 0]) + (kTR_on * alg_10_CTpf * alg_23_CRf - kTR_off * CT_R)
        d_CD_R = torch.zeros_like(y[:, 0]) + (kDR_on * alg_12_CDpf * alg_23_CRf - kDR_off * CD_R)
        d_AR = torch.zeros_like(y[:, 0]) + (k1R - keR * alg_23_CRf)

        return torch.stack(
            [d_ATbl, d_ADbl, d_CT_A, d_CD_A, d_ATl, d_ADl, d_ATif, d_ATst, d_ATp, d_ADp, d_CT_R, d_CD_R, d_CDT, d_CDD, d_CTT, d_CDD5a, d_CDDcd, d_CDDsec, d_CDDcp, d_CDT5a, d_CDTcd, d_CDTsec, d_CDTcp, d_CTT5a, d_CTTcd, d_CTTsec, d_CTTcp, d_ATb, d_ADb, d_ALH, d_AR, d_VPC1, d_VPL1],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
