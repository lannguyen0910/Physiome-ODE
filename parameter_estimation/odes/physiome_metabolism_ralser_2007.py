import torch
import torch.nn as nn
import math


class MetabolismRalser2007ODE(nn.Module):
    """
    PyTorch translation of metabolism/ralser_2007/model.py.
    Dynamic states: 24; independent constants: 142; derived constants used by rates: 0.
    Original constant slots: 142; algebraic slots: 32.
    """
    source_file = 'metabolism/ralser_2007/model.py'
    source_state_dim = 24
    source_constant_dim = 142
    algebraic_dim = 32
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 24
        self.parameter_dim = 142

        self.parameters = (
            parameters
            if parameters is not None
            else {'k_rel_TPI': 1, 'k_rel_GAPDH': 1, 'SUMAXP': 4.1, 'CO2': 1, 'ETOH': 50, 'SUCC': 0.1, 'GLY': 0.15, 'GLCo': 50, 'cytoplasm': 1, 'X': 0.1, 'F26BP': 0.02, 'VmGLK': 226.452, 'KeqAK': 0.45, 'KeqGLK': 3800, 'KmGLKATP': 0.15, 'KmGLKGLCi': 0.08, 'KmGLKG6P': 30, 'KmGLKADP': 0.23, 'VmPGI': 339.677, 'KmPGIG6P': 1.4, 'KeqPGI': 0.314, 'KmPGIF6P': 0.3, 'gR': 5.12, 'VmPFK': 182.903, 'KeqAK_2': 0.45, 'KmPFKF6P': 0.1, 'KmPFKATP': 0.71, 'L0': 0.66, 'CPFKF26BP': 0.0174, 'KPFKF26BP': 0.000682, 'CPFKF16BP': 0.397, 'KPFKF16BP': 0.111, 'CPFKAMP': 0.0845, 'KPFKAMP': 0.0995, 'CiPFKATP': 100, 'KiPFKATP': 0.65, 'CPFKATP': 3, 'VmALD': 322.258, 'KeqTPI': 0.045, 'KeqALD': 0.069, 'KmALDF16P': 0.3, 'KmALDDHAP': 2.4, 'KmALDGAP': 2, 'KmALDGAPi': 10, 'VmG3PDH': 70.15, 'KeqG3PDH': 4300, 'KeqTPI_2': 0.045, 'KmG3PDHDHAP': 0.4, 'KmG3PDHNADH': 0.023, 'KmG3PDHNAD': 0.93, 'KmG3PDHGLY': 1, 'VmGAPDHr': 6549.68, 'KmGAPDHBPG': 0.0098, 'KmGAPDHNADH': 0.06, 'KeqTPI_3': 0.045, 'VmGAPDHf': 1184.52, 'KmGAPDHGAP': 0.21, 'KmGAPDHNAD': 0.09, 'KeqGAPDH': 0.005, 'VmPGK': 1306.45, 'KeqPGK': 3200, 'KeqAK_3': 0.45, 'KmPGKATP': 0.3, 'KmPGKP3G': 0.53, 'KmPGKADP': 0.2, 'KmPGKBPG': 0.003, 'VmPGM': 2525.81, 'KmPGMP3G': 1.2, 'KeqPGM': 0.19, 'KmPGMP2G': 0.08, 'VmENO': 365.806, 'KmENOP2G': 0.04, 'KeqENO': 6.7, 'KmENOPEP': 0.5, 'VmPYK': 1088.71, 'KmPYKPEP': 0.14, 'KmPYKADP': 0.53, 'KeqAK_4': 0.45, 'KeqPYK': 6500, 'KmPYKPYR': 21, 'KmPYKATP': 1.5, 'VmPDC': 174.194, 'nPDC': 1.9, 'KmPDCPYR': 4.33, 'KSUCC': 21.4, 'VmADH': 810, 'KiADHNAD': 0.92, 'KmADHETOH': 17, 'KeqADH': 6.9e-05, 'KmADHNAD': 0.17, 'KmADHNADH': 0.11, 'KiADHNADH': 0.031, 'KmADHACE': 1.11, 'KiADHACE': 1.1, 'KiADHETOH': 90, 'KATPASE': 39.5, 'KeqAK_5': 0.45, 'KmGA3P': 1.27, 'KmDHAP': 1.23, 'VmDHAP': 10900, 'VmGA3P': 555, 'VmG6PDH': 4, 'KmG6P': 0.04, 'KmNADP': 0.02, 'KiNADPH': 0.017, 'Vm6PGL': 4, 'Km6PGL': 0.8, 'VmGluDH': 4, 'KmGluconate': 0.02, 'KmNADP_2': 0.03, 'KiNADPH_2': 0.03, 'VmPPIf': 3458, 'VmPPIr': 3458, 'KmRibu5P': 1.6, 'KmRibo5P': 1.6, 'VmTransk1f': 4, 'VmTransk1r': 2, 'KmRibose5P': 0.1, 'KmXyl5P': 0.15, 'KmGA3P_2': 0.1, 'KmSeduhept': 0.15, 'VmR5PIf': 1039, 'VmR5PIr': 1039, 'KmRibu5P_2': 1.5, 'KmXyl': 1.5, 'VmTransaldf': 55, 'VmTransaldr': 10, 'KmGA3P_3': 0.22, 'KmSeduhept_2': 0.18, 'KmF6P': 0.32, 'KmEry4P': 0.018, 'VmTransk2f': 3.2, 'VmTransk2r': 43, 'KmXyl5P_2': 0.16, 'KmEry4P_2': 0.09, 'KmF6P_2': 1.1, 'KmGA3P_4': 2.1, 'kNADPH': 2, 'VmGLT': 97.264, 'KeqGLT': 1, 'KmGLTGLCo': 1.1918, 'KmGLTGLCi': 1.1918}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'P': 5, 'G6P': 1.39, 'F6P': 0.28, 'F16P': 0.1, 'NADH': 0.39, 'NAD': 1.2, 'BPG': 0.1, 'P3G': 0.1, 'P2G': 0.1, 'PEP': 0.1, 'PYR': 3.36, 'ACE': 0.04, 'GA3P': 0.05, 'DHAP': 1, 'D6PGluconoLactone': 0.1, 'D6PGluconate': 0.1, 'NADP': 0.4, 'NADPH': 1.6, 'Ribulose5P': 0.1, 'Ribose5P': 0.1, 'Xyl5P': 0.1, 'Seduhept7P': 0.1, 'Erythrose4P': 0, 'GLCi': 0.087}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k_rel_TPI', 'k_rel_GAPDH', 'SUMAXP', 'CO2', 'ETOH', 'SUCC', 'GLY', 'GLCo', 'cytoplasm', 'X', 'F26BP', 'VmGLK', 'KeqAK', 'KeqGLK', 'KmGLKATP', 'KmGLKGLCi', 'KmGLKG6P', 'KmGLKADP', 'VmPGI', 'KmPGIG6P', 'KeqPGI', 'KmPGIF6P', 'gR', 'VmPFK', 'KeqAK_2', 'KmPFKF6P', 'KmPFKATP', 'L0', 'CPFKF26BP', 'KPFKF26BP', 'CPFKF16BP', 'KPFKF16BP', 'CPFKAMP', 'KPFKAMP', 'CiPFKATP', 'KiPFKATP', 'CPFKATP', 'VmALD', 'KeqTPI', 'KeqALD', 'KmALDF16P', 'KmALDDHAP', 'KmALDGAP', 'KmALDGAPi', 'VmG3PDH', 'KeqG3PDH', 'KeqTPI_2', 'KmG3PDHDHAP', 'KmG3PDHNADH', 'KmG3PDHNAD', 'KmG3PDHGLY', 'VmGAPDHr', 'KmGAPDHBPG', 'KmGAPDHNADH', 'KeqTPI_3', 'VmGAPDHf', 'KmGAPDHGAP', 'KmGAPDHNAD', 'KeqGAPDH', 'VmPGK', 'KeqPGK', 'KeqAK_3', 'KmPGKATP', 'KmPGKP3G', 'KmPGKADP', 'KmPGKBPG', 'VmPGM', 'KmPGMP3G', 'KeqPGM', 'KmPGMP2G', 'VmENO', 'KmENOP2G', 'KeqENO', 'KmENOPEP', 'VmPYK', 'KmPYKPEP', 'KmPYKADP', 'KeqAK_4', 'KeqPYK', 'KmPYKPYR', 'KmPYKATP', 'VmPDC', 'nPDC', 'KmPDCPYR', 'KSUCC', 'VmADH', 'KiADHNAD', 'KmADHETOH', 'KeqADH', 'KmADHNAD', 'KmADHNADH', 'KiADHNADH', 'KmADHACE', 'KiADHACE', 'KiADHETOH', 'KATPASE', 'KeqAK_5', 'KmGA3P', 'KmDHAP', 'VmDHAP', 'VmGA3P', 'VmG6PDH', 'KmG6P', 'KmNADP', 'KiNADPH', 'Vm6PGL', 'Km6PGL', 'VmGluDH', 'KmGluconate', 'KmNADP_2', 'KiNADPH_2', 'VmPPIf', 'VmPPIr', 'KmRibu5P', 'KmRibo5P', 'VmTransk1f', 'VmTransk1r', 'KmRibose5P', 'KmXyl5P', 'KmGA3P_2', 'KmSeduhept', 'VmR5PIf', 'VmR5PIr', 'KmRibu5P_2', 'KmXyl', 'VmTransaldf', 'VmTransaldr', 'KmGA3P_3', 'KmSeduhept_2', 'KmF6P', 'KmEry4P', 'VmTransk2f', 'VmTransk2r', 'KmXyl5P_2', 'KmEry4P_2', 'KmF6P_2', 'KmGA3P_4', 'kNADPH', 'VmGLT', 'KeqGLT', 'KmGLTGLCo', 'KmGLTGLCi']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['P', 'G6P', 'F6P', 'F16P', 'NADH', 'NAD', 'BPG', 'P3G', 'P2G', 'PEP', 'PYR', 'ACE', 'GA3P', 'DHAP', 'D6PGluconoLactone', 'D6PGluconate', 'NADP', 'NADPH', 'Ribulose5P', 'Ribose5P', 'Xyl5P', 'Seduhept7P', 'Erythrose4P', 'GLCi']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (24)
        P = y[:, 0]
        G6P = y[:, 1]
        F6P = y[:, 2]
        F16P = y[:, 3]
        NADH = y[:, 4]
        NAD = y[:, 5]
        BPG = y[:, 6]
        P3G = y[:, 7]
        P2G = y[:, 8]
        PEP = y[:, 9]
        PYR = y[:, 10]
        ACE = y[:, 11]
        GA3P = y[:, 12]
        DHAP = y[:, 13]
        D6PGluconoLactone = y[:, 14]
        D6PGluconate = y[:, 15]
        NADP = y[:, 16]
        NADPH = y[:, 17]
        Ribulose5P = y[:, 18]
        Ribose5P = y[:, 19]
        Xyl5P = y[:, 20]
        Seduhept7P = y[:, 21]
        Erythrose4P = y[:, 22]
        GLCi = y[:, 23]

        # Independent parameters (source constant order)
        k_rel_TPI = theta[:, 0]  # constants[0]
        k_rel_GAPDH = theta[:, 1]  # constants[1]
        SUMAXP = theta[:, 2]  # constants[2]
        CO2 = theta[:, 3]  # constants[3]
        ETOH = theta[:, 4]  # constants[4]
        SUCC = theta[:, 5]  # constants[5]
        GLY = theta[:, 6]  # constants[6]
        GLCo = theta[:, 7]  # constants[7]
        cytoplasm = theta[:, 8]  # constants[8]
        X = theta[:, 9]  # constants[9]
        F26BP = theta[:, 10]  # constants[10]
        VmGLK = theta[:, 11]  # constants[11]
        KeqAK = theta[:, 12]  # constants[12]
        KeqGLK = theta[:, 13]  # constants[13]
        KmGLKATP = theta[:, 14]  # constants[14]
        KmGLKGLCi = theta[:, 15]  # constants[15]
        KmGLKG6P = theta[:, 16]  # constants[16]
        KmGLKADP = theta[:, 17]  # constants[17]
        VmPGI = theta[:, 18]  # constants[18]
        KmPGIG6P = theta[:, 19]  # constants[19]
        KeqPGI = theta[:, 20]  # constants[20]
        KmPGIF6P = theta[:, 21]  # constants[21]
        gR = theta[:, 22]  # constants[22]
        VmPFK = theta[:, 23]  # constants[23]
        KeqAK_2 = theta[:, 24]  # constants[24]
        KmPFKF6P = theta[:, 25]  # constants[25]
        KmPFKATP = theta[:, 26]  # constants[26]
        L0 = theta[:, 27]  # constants[27]
        CPFKF26BP = theta[:, 28]  # constants[28]
        KPFKF26BP = theta[:, 29]  # constants[29]
        CPFKF16BP = theta[:, 30]  # constants[30]
        KPFKF16BP = theta[:, 31]  # constants[31]
        CPFKAMP = theta[:, 32]  # constants[32]
        KPFKAMP = theta[:, 33]  # constants[33]
        CiPFKATP = theta[:, 34]  # constants[34]
        KiPFKATP = theta[:, 35]  # constants[35]
        CPFKATP = theta[:, 36]  # constants[36]
        VmALD = theta[:, 37]  # constants[37]
        KeqTPI = theta[:, 38]  # constants[38]
        KeqALD = theta[:, 39]  # constants[39]
        KmALDF16P = theta[:, 40]  # constants[40]
        KmALDDHAP = theta[:, 41]  # constants[41]
        KmALDGAP = theta[:, 42]  # constants[42]
        KmALDGAPi = theta[:, 43]  # constants[43]
        VmG3PDH = theta[:, 44]  # constants[44]
        KeqG3PDH = theta[:, 45]  # constants[45]
        KeqTPI_2 = theta[:, 46]  # constants[46]
        KmG3PDHDHAP = theta[:, 47]  # constants[47]
        KmG3PDHNADH = theta[:, 48]  # constants[48]
        KmG3PDHNAD = theta[:, 49]  # constants[49]
        KmG3PDHGLY = theta[:, 50]  # constants[50]
        VmGAPDHr = theta[:, 51]  # constants[51]
        KmGAPDHBPG = theta[:, 52]  # constants[52]
        KmGAPDHNADH = theta[:, 53]  # constants[53]
        KeqTPI_3 = theta[:, 54]  # constants[54]
        VmGAPDHf = theta[:, 55]  # constants[55]
        KmGAPDHGAP = theta[:, 56]  # constants[56]
        KmGAPDHNAD = theta[:, 57]  # constants[57]
        KeqGAPDH = theta[:, 58]  # constants[58]
        VmPGK = theta[:, 59]  # constants[59]
        KeqPGK = theta[:, 60]  # constants[60]
        KeqAK_3 = theta[:, 61]  # constants[61]
        KmPGKATP = theta[:, 62]  # constants[62]
        KmPGKP3G = theta[:, 63]  # constants[63]
        KmPGKADP = theta[:, 64]  # constants[64]
        KmPGKBPG = theta[:, 65]  # constants[65]
        VmPGM = theta[:, 66]  # constants[66]
        KmPGMP3G = theta[:, 67]  # constants[67]
        KeqPGM = theta[:, 68]  # constants[68]
        KmPGMP2G = theta[:, 69]  # constants[69]
        VmENO = theta[:, 70]  # constants[70]
        KmENOP2G = theta[:, 71]  # constants[71]
        KeqENO = theta[:, 72]  # constants[72]
        KmENOPEP = theta[:, 73]  # constants[73]
        VmPYK = theta[:, 74]  # constants[74]
        KmPYKPEP = theta[:, 75]  # constants[75]
        KmPYKADP = theta[:, 76]  # constants[76]
        KeqAK_4 = theta[:, 77]  # constants[77]
        KeqPYK = theta[:, 78]  # constants[78]
        KmPYKPYR = theta[:, 79]  # constants[79]
        KmPYKATP = theta[:, 80]  # constants[80]
        VmPDC = theta[:, 81]  # constants[81]
        nPDC = theta[:, 82]  # constants[82]
        KmPDCPYR = theta[:, 83]  # constants[83]
        KSUCC = theta[:, 84]  # constants[84]
        VmADH = theta[:, 85]  # constants[85]
        KiADHNAD = theta[:, 86]  # constants[86]
        KmADHETOH = theta[:, 87]  # constants[87]
        KeqADH = theta[:, 88]  # constants[88]
        KmADHNAD = theta[:, 89]  # constants[89]
        KmADHNADH = theta[:, 90]  # constants[90]
        KiADHNADH = theta[:, 91]  # constants[91]
        KmADHACE = theta[:, 92]  # constants[92]
        KiADHACE = theta[:, 93]  # constants[93]
        KiADHETOH = theta[:, 94]  # constants[94]
        KATPASE = theta[:, 95]  # constants[95]
        KeqAK_5 = theta[:, 96]  # constants[96]
        KmGA3P = theta[:, 97]  # constants[97]
        KmDHAP = theta[:, 98]  # constants[98]
        VmDHAP = theta[:, 99]  # constants[99]
        VmGA3P = theta[:, 100]  # constants[100]
        VmG6PDH = theta[:, 101]  # constants[101]
        KmG6P = theta[:, 102]  # constants[102]
        KmNADP = theta[:, 103]  # constants[103]
        KiNADPH = theta[:, 104]  # constants[104]
        Vm6PGL = theta[:, 105]  # constants[105]
        Km6PGL = theta[:, 106]  # constants[106]
        VmGluDH = theta[:, 107]  # constants[107]
        KmGluconate = theta[:, 108]  # constants[108]
        KmNADP_2 = theta[:, 109]  # constants[109]
        KiNADPH_2 = theta[:, 110]  # constants[110]
        VmPPIf = theta[:, 111]  # constants[111]
        VmPPIr = theta[:, 112]  # constants[112]
        KmRibu5P = theta[:, 113]  # constants[113]
        KmRibo5P = theta[:, 114]  # constants[114]
        VmTransk1f = theta[:, 115]  # constants[115]
        VmTransk1r = theta[:, 116]  # constants[116]
        KmRibose5P = theta[:, 117]  # constants[117]
        KmXyl5P = theta[:, 118]  # constants[118]
        KmGA3P_2 = theta[:, 119]  # constants[119]
        KmSeduhept = theta[:, 120]  # constants[120]
        VmR5PIf = theta[:, 121]  # constants[121]
        VmR5PIr = theta[:, 122]  # constants[122]
        KmRibu5P_2 = theta[:, 123]  # constants[123]
        KmXyl = theta[:, 124]  # constants[124]
        VmTransaldf = theta[:, 125]  # constants[125]
        VmTransaldr = theta[:, 126]  # constants[126]
        KmGA3P_3 = theta[:, 127]  # constants[127]
        KmSeduhept_2 = theta[:, 128]  # constants[128]
        KmF6P = theta[:, 129]  # constants[129]
        KmEry4P = theta[:, 130]  # constants[130]
        VmTransk2f = theta[:, 131]  # constants[131]
        VmTransk2r = theta[:, 132]  # constants[132]
        KmXyl5P_2 = theta[:, 133]  # constants[133]
        KmEry4P_2 = theta[:, 134]  # constants[134]
        KmF6P_2 = theta[:, 135]  # constants[135]
        KmGA3P_4 = theta[:, 136]  # constants[136]
        kNADPH = theta[:, 137]  # constants[137]
        VmGLT = theta[:, 138]  # constants[138]
        KeqGLT = theta[:, 139]  # constants[139]
        KmGLTGLCo = theta[:, 140]  # constants[140]
        KmGLTGLCi = theta[:, 141]  # constants[141]

        # Initialize rates and algebraic variables, as in computeRates()
        d_P = torch.zeros_like(y[:, 0])
        d_G6P = torch.zeros_like(y[:, 0])
        d_F6P = torch.zeros_like(y[:, 0])
        d_F16P = torch.zeros_like(y[:, 0])
        d_NADH = torch.zeros_like(y[:, 0])
        d_NAD = torch.zeros_like(y[:, 0])
        d_BPG = torch.zeros_like(y[:, 0])
        d_P3G = torch.zeros_like(y[:, 0])
        d_P2G = torch.zeros_like(y[:, 0])
        d_PEP = torch.zeros_like(y[:, 0])
        d_PYR = torch.zeros_like(y[:, 0])
        d_ACE = torch.zeros_like(y[:, 0])
        d_GA3P = torch.zeros_like(y[:, 0])
        d_DHAP = torch.zeros_like(y[:, 0])
        d_D6PGluconoLactone = torch.zeros_like(y[:, 0])
        d_D6PGluconate = torch.zeros_like(y[:, 0])
        d_NADP = torch.zeros_like(y[:, 0])
        d_NADPH = torch.zeros_like(y[:, 0])
        d_Ribulose5P = torch.zeros_like(y[:, 0])
        d_Ribose5P = torch.zeros_like(y[:, 0])
        d_Xyl5P = torch.zeros_like(y[:, 0])
        d_Seduhept7P = torch.zeros_like(y[:, 0])
        d_Erythrose4P = torch.zeros_like(y[:, 0])
        d_GLCi = torch.zeros_like(y[:, 0])
        alg_0_vGLK = torch.zeros_like(y[:, 0])
        alg_2_vPGI = torch.zeros_like(y[:, 0])
        alg_3_vGLT = torch.zeros_like(y[:, 0])
        alg_4_numerator1 = torch.zeros_like(y[:, 0])
        alg_5_numerator2 = torch.zeros_like(y[:, 0])
        alg_6_denom1 = torch.zeros_like(y[:, 0])
        alg_7_denom2 = torch.zeros_like(y[:, 0])
        alg_8_denom3 = torch.zeros_like(y[:, 0])
        alg_9_denominator = torch.zeros_like(y[:, 0])
        alg_10_vPFK = torch.zeros_like(y[:, 0])
        alg_11_vALD = torch.zeros_like(y[:, 0])
        alg_12_vG3PDH = torch.zeros_like(y[:, 0])
        alg_13_vGAPDH = torch.zeros_like(y[:, 0])
        alg_14_vPGK = torch.zeros_like(y[:, 0])
        alg_15_vTPI = torch.zeros_like(y[:, 0])
        alg_16_vPGM = torch.zeros_like(y[:, 0])
        alg_17_vG6PDH = torch.zeros_like(y[:, 0])
        alg_18_vENO = torch.zeros_like(y[:, 0])
        alg_19_v6PGL = torch.zeros_like(y[:, 0])
        alg_20_vPYK = torch.zeros_like(y[:, 0])
        alg_21_vGluDH = torch.zeros_like(y[:, 0])
        alg_22_vPDC = torch.zeros_like(y[:, 0])
        alg_23_vATP = torch.zeros_like(y[:, 0])
        alg_24_vPPI = torch.zeros_like(y[:, 0])
        alg_25_vNADPH = torch.zeros_like(y[:, 0])
        alg_26_vSUC = torch.zeros_like(y[:, 0])
        alg_27_vTransk1 = torch.zeros_like(y[:, 0])
        alg_28_vADH = torch.zeros_like(y[:, 0])
        alg_29_vR5PI = torch.zeros_like(y[:, 0])
        alg_30_vTransald = torch.zeros_like(y[:, 0])
        alg_31_vTransk2 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_vGLK = cytoplasm * VmGLK * (-G6P * (SUMAXP - (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK * P ** 2.0) ** (1.0 / 2)) / ((1.0 - 4.0 * KeqAK) * KeqGLK) + GLCi * (-SUMAXP + P - 4.0 * KeqAK * P + (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK * P ** 2.0) ** (1.0 / 2)) / (2.0 - 8.0 * KeqAK)) / (KmGLKATP * KmGLKGLCi * (1.0 + G6P / KmGLKG6P + GLCi / KmGLKGLCi) * (1.0 + (SUMAXP - (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK * P ** 2.0) ** (1.0 / 2)) / ((1.0 - 4.0 * KeqAK) * KmGLKADP) + (-SUMAXP + P - 4.0 * KeqAK * P + (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK * P ** 2.0) ** (1.0 / 2)) / ((2.0 - 8.0 * KeqAK) * KmGLKATP)))
        alg_3_vGLT = cytoplasm * VmGLT * (GLCo - GLCi / KeqGLT) / (KmGLTGLCo * (1.0 + GLCo / KmGLTGLCo + GLCi / KmGLTGLCi + 0.91 * GLCo * GLCi / (KmGLTGLCi * KmGLTGLCo)))
        d_GLCi = torch.zeros_like(y[:, 0]) + ((alg_3_vGLT - alg_0_vGLK) / cytoplasm)
        alg_4_numerator1 = cytoplasm * gR * VmPFK * F6P * (-SUMAXP + P - 4.0 * KeqAK_2 * P + (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_2 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_2 * P ** 2.0) ** (1.0 / 2))
        alg_5_numerator2 = 1.0 + F6P / KmPFKF6P + (-SUMAXP + P - 4.0 * KeqAK_2 * P + (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_2 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_2 * P ** 2.0) ** (1.0 / 2)) / ((2.0 - 8.0 * KeqAK_2) * KmPFKATP) + gR * F6P * (-SUMAXP + P - 4.0 * KeqAK_2 * P + (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_2 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_2 * P ** 2.0) ** (1.0 / 2)) / ((2.0 - 8.0 * KeqAK_2) * KmPFKATP * KmPFKF6P)
        alg_6_denom1 = (1.0 + CPFKF26BP * F26BP / KPFKF26BP + CPFKF16BP * F16P / KPFKF16BP) ** 2.0 * (1.0 + 2.0 * CPFKAMP * KeqAK_2 * (SUMAXP - (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_2 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_2 * P ** 2.0) ** (1.0 / 2)) ** 2.0 / ((-1.0 + 4.0 * KeqAK_2) * KPFKAMP * (SUMAXP - P + 4.0 * KeqAK_2 * P - (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_2 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_2 * P ** 2.0) ** (1.0 / 2)))) ** 2.0
        alg_7_denom2 = (1.0 + CiPFKATP * (-SUMAXP + P - 4.0 * KeqAK_2 * P + (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_2 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_2 * P ** 2.0) ** (1.0 / 2)) / ((2.0 - 8.0 * KeqAK_2) * KiPFKATP)) ** 2.0 * (1.0 + CPFKATP * (-SUMAXP + P - 4.0 * KeqAK_2 * P + (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_2 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_2 * P ** 2.0) ** (1.0 / 2)) / ((2.0 - 8.0 * KeqAK_2) * KmPFKATP)) ** 2.0
        alg_8_denom3 = (1.0 + F26BP / KPFKF26BP + F16P / KPFKF16BP) ** 2.0 * (1.0 + 2.0 * KeqAK_2 * (SUMAXP - (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_2 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_2 * P ** 2.0) ** (1.0 / 2)) ** 2.0 / ((-1.0 + 4.0 * KeqAK_2) * KPFKAMP * (SUMAXP - P + 4.0 * KeqAK_2 * P - (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_2 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_2 * P ** 2.0) ** (1.0 / 2)))) ** 2.0 * (1.0 + (-SUMAXP + P - 4.0 * KeqAK_2 * P + (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_2 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_2 * P ** 2.0) ** (1.0 / 2)) / ((2.0 - 8.0 * KeqAK_2) * KiPFKATP)) ** 2.0
        alg_9_denominator = (2.0 - 8.0 * KeqAK_2) * KmPFKATP * KmPFKF6P * (L0 * alg_6_denom1 * alg_7_denom2 / alg_8_denom3 + (1.0 + F6P / KmPFKF6P + (-SUMAXP + P - 4.0 * KeqAK_2 * P + (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_2 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_2 * P ** 2.0) ** (1.0 / 2)) / ((2.0 - 8.0 * KeqAK_2) * KmPFKATP) + gR * F6P * (-SUMAXP + P - 4.0 * KeqAK_2 * P + (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_2 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_2 * P ** 2.0) ** (1.0 / 2)) / ((2.0 - 8.0 * KeqAK_2) * KmPFKATP * KmPFKF6P)) ** 2.0)
        alg_10_vPFK = alg_4_numerator1 * alg_5_numerator2 / alg_9_denominator
        alg_11_vALD = cytoplasm * VmALD * F16P / KmALDF16P * (1.0 - DHAP * GA3P / (F16P * KeqALD)) / (1.0 + F16P / KmALDF16P + DHAP / KmALDDHAP + GA3P / KmALDGAP + F16P * GA3P / (KmALDF16P * KmALDGAPi) + DHAP * GA3P / (KmALDDHAP * KmALDGAP))
        d_F16P = torch.zeros_like(y[:, 0]) + ((alg_10_vPFK - alg_11_vALD) / cytoplasm)
        alg_13_vGAPDH = cytoplasm * k_rel_GAPDH * VmGAPDHf * GA3P * NAD / (KmGAPDHGAP * KmGAPDHNAD) * (1.0 - BPG * NADH / (GA3P * NAD * KeqGAPDH)) / ((1.0 + GA3P / KmGAPDHGAP + BPG / KmGAPDHBPG) * (1.0 + NAD / KmGAPDHNAD + NADH / KmGAPDHNADH))
        alg_14_vPGK = 1.0 * cytoplasm * VmPGK * (KeqPGK * BPG * (SUMAXP - (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_3 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_3 * P ** 2.0) ** (1.0 / 2)) / (1.0 - 4.0 * KeqAK_3) - (-SUMAXP + P - 4.0 * KeqAK_3 * P + (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_3 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_3 * P ** 2.0) ** (1.0 / 2)) * P3G / (2.0 - 8.0 * KeqAK_3)) / (KmPGKATP * KmPGKP3G * (1.0 + (SUMAXP - (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_3 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_3 * P ** 2.0) ** (1.0 / 2)) / ((1.0 - 4.0 * KeqAK_3) * KmPGKADP) + (-SUMAXP + P - 4.0 * KeqAK_3 * P + (SUMAXP ** 2.0 - 2.0 * SUMAXP * P + 8.0 * KeqAK_3 * SUMAXP * P + P ** 2.0 - 4.0 * KeqAK_3 * P ** 2.0) ** (1.0 / 2)) / ((2.0 - 8.0 * KeqAK_3) * KmPGKATP)) * (1.0 + BPG / KmPGKBPG + P3G / KmPGKP3G))
        d_BPG = torch.zeros_like(y[:, 0]) + ((alg_13_vGAPDH - alg_14_vPGK) / cytoplasm)
        alg_12_vG3PDH = cytoplasm * VmG3PDH * (-GLY * NAD / KeqG3PDH + NADH * DHAP / (1.0 + KeqTPI_2)) / (KmG3PDHDHAP * KmG3PDHNADH * (1.0 + NAD / KmG3PDHNAD + NADH / KmG3PDHNADH) * (1.0 + GLY / KmG3PDHGLY + DHAP / ((1.0 + KeqTPI_2) * KmG3PDHDHAP)))
        alg_15_vTPI = cytoplasm * k_rel_TPI * (VmDHAP * GA3P / KmGA3P - VmGA3P * DHAP / KmDHAP) / (1.0 + GA3P / KmGA3P + DHAP / KmDHAP)
        d_DHAP = torch.zeros_like(y[:, 0]) + ((alg_11_vALD - alg_12_vG3PDH + alg_15_vTPI) / cytoplasm)
        alg_2_vPGI = cytoplasm * VmPGI / KmPGIG6P * (G6P - F6P / KeqPGI) / (1.0 + G6P / KmPGIG6P + F6P / KmPGIF6P)
        alg_17_vG6PDH = cytoplasm * VmG6PDH * G6P * NADP / (KmG6P * KmNADP) / ((1.0 + G6P / KmG6P + NADPH / KiNADPH) * (1.0 + NADP / KmNADP))
        d_G6P = torch.zeros_like(y[:, 0]) + ((alg_0_vGLK - alg_2_vPGI - alg_17_vG6PDH) / cytoplasm)
        alg_16_vPGM = cytoplasm * VmPGM / KmPGMP3G * (P3G - P2G / KeqPGM) / (1.0 + P3G / KmPGMP3G + P2G / KmPGMP2G)
        d_P3G = torch.zeros_like(y[:, 0]) + ((alg_14_vPGK - alg_16_vPGM) / cytoplasm)
        alg_18_vENO = cytoplasm * VmENO / KmENOP2G * (P2G - PEP / KeqENO) / (1.0 + P2G / KmENOP2G + PEP / KmENOPEP)
        d_P2G = torch.zeros_like(y[:, 0]) + ((alg_16_vPGM - alg_18_vENO) / cytoplasm)
        alg_19_v6PGL = cytoplasm * Vm6PGL * D6PGluconoLactone / (Km6PGL + D6PGluconoLactone)
        d_D6PGluconoLactone = torch.zeros_like(y[:, 0]) + ((alg_17_vG6PDH - alg_19_v6PGL) / cytoplasm)
        alg_20_vPYK = cytoplasm * VmPYK / (KmPYKPEP * KmPYKADP) * (PEP * (SUMAXP - (P ** 2.0 - 4.0 * KeqAK_4 * P ** 2.0 - 2.0 * P * SUMAXP + 8.0 * KeqAK_4 * P * SUMAXP + SUMAXP ** 2.0) ** (1.0 / 2)) / (1.0 - 4.0 * KeqAK_4) - PYR * (P - 4.0 * KeqAK_4 * P - SUMAXP + (P ** 2.0 - 4.0 * KeqAK_4 * P ** 2.0 - 2.0 * P * SUMAXP + 8.0 * KeqAK_4 * P * SUMAXP + SUMAXP ** 2.0) ** (1.0 / 2)) / (2.0 - 8.0 * KeqAK_4) / KeqPYK) / ((1.0 + PEP / KmPYKPEP + PYR / KmPYKPYR) * (1.0 + (P - 4.0 * KeqAK_4 * P - SUMAXP + (P ** 2.0 - 4.0 * KeqAK_4 * P ** 2.0 - 2.0 * P * SUMAXP + 8.0 * KeqAK_4 * P * SUMAXP + SUMAXP ** 2.0) ** (1.0 / 2)) / (2.0 - 8.0 * KeqAK_4) / KmPYKATP + (SUMAXP - (P ** 2.0 - 4.0 * KeqAK_4 * P ** 2.0 - 2.0 * P * SUMAXP + 8.0 * KeqAK_4 * P * SUMAXP + SUMAXP ** 2.0) ** (1.0 / 2)) / (1.0 - 4.0 * KeqAK_4) / KmPYKADP))
        d_PEP = torch.zeros_like(y[:, 0]) + ((alg_18_vENO - alg_20_vPYK) / cytoplasm)
        alg_21_vGluDH = cytoplasm * VmGluDH * D6PGluconate * NADP / (KmGluconate * KmNADP_2) / ((1.0 + D6PGluconate / KmGluconate + NADPH / KiNADPH_2) * (1.0 + NADP / KmNADP_2))
        d_D6PGluconate = torch.zeros_like(y[:, 0]) + ((alg_19_v6PGL - alg_21_vGluDH) / cytoplasm)
        alg_23_vATP = cytoplasm * KATPASE * (P - 4.0 * KeqAK_5 * P - SUMAXP + (P ** 2.0 - 4.0 * KeqAK_5 * P ** 2.0 - 2.0 * P * SUMAXP + 8.0 * KeqAK_5 * P * SUMAXP + SUMAXP ** 2.0) ** (1.0 / 2)) / (2.0 - 8.0 * KeqAK_5)
        d_P = torch.zeros_like(y[:, 0]) + ((alg_14_vPGK - (alg_0_vGLK + alg_10_vPFK) + alg_20_vPYK - alg_23_vATP) / cytoplasm)
        alg_22_vPDC = cytoplasm * VmPDC * PYR ** nPDC / KmPDCPYR ** nPDC / (1.0 + PYR ** nPDC / KmPDCPYR ** nPDC)
        d_PYR = torch.zeros_like(y[:, 0]) + ((alg_20_vPYK - alg_22_vPDC) / cytoplasm)
        alg_25_vNADPH = cytoplasm * kNADPH * NADPH
        d_NADP = torch.zeros_like(y[:, 0]) + ((alg_25_vNADPH - (alg_17_vG6PDH + alg_21_vGluDH)) / cytoplasm)
        d_NADPH = torch.zeros_like(y[:, 0]) + ((alg_17_vG6PDH + alg_21_vGluDH - alg_25_vNADPH) / cytoplasm)
        alg_24_vPPI = cytoplasm * (VmPPIf * Ribulose5P / KmRibu5P - VmPPIr * Ribose5P / KmRibo5P) / (1.0 + Ribulose5P / KmRibu5P + Ribose5P / KmRibo5P)
        alg_27_vTransk1 = cytoplasm * (VmTransk1f * Ribose5P * Xyl5P / (KmRibose5P * KmXyl5P) - VmTransk1r * GA3P * Seduhept7P / (KmGA3P_2 * KmSeduhept)) / ((1.0 + Ribose5P / KmRibose5P + GA3P / KmGA3P_2) * (1.0 + Xyl5P / KmXyl5P + Seduhept7P / KmSeduhept))
        d_Ribose5P = torch.zeros_like(y[:, 0]) + ((alg_24_vPPI - alg_27_vTransk1) / cytoplasm)
        alg_26_vSUC = cytoplasm * KSUCC * ACE
        alg_28_vADH = cytoplasm * -VmADH / (KiADHNAD * KmADHETOH) * (NAD * ETOH - NADH * ACE / KeqADH) / (1.0 + NAD / KiADHNAD + KmADHNAD * ETOH / (KiADHNAD * KmADHETOH) + KmADHNADH * ACE / (KiADHNADH * KmADHACE) + NADH / KiADHNADH + NAD * ETOH / (KiADHNAD * KmADHETOH) + KmADHNADH * NAD * ACE / (KiADHNAD * KiADHNADH * KmADHACE) + KmADHNAD * ETOH * NADH / (KiADHNAD * KmADHETOH * KiADHNADH) + NADH * ACE / (KiADHNADH * KmADHACE) + NAD * ETOH * ACE / (KiADHNAD * KmADHETOH * KiADHACE) + ETOH * NADH * ACE / (KiADHETOH * KiADHNADH * KmADHACE))
        d_NADH = torch.zeros_like(y[:, 0]) + ((alg_13_vGAPDH - alg_12_vG3PDH + 3.0 * alg_26_vSUC - alg_28_vADH) / cytoplasm)
        d_NAD = torch.zeros_like(y[:, 0]) + ((alg_12_vG3PDH - alg_13_vGAPDH - 3.0 * alg_26_vSUC + alg_28_vADH) / cytoplasm)
        d_ACE = torch.zeros_like(y[:, 0]) + ((alg_22_vPDC - 2.0 * alg_26_vSUC - alg_28_vADH) / cytoplasm)
        alg_29_vR5PI = cytoplasm * (VmR5PIf * Ribulose5P / KmRibu5P_2 - VmR5PIr * Xyl5P / KmXyl) / (1.0 + Ribulose5P / KmRibu5P_2 + Xyl5P / KmXyl)
        d_Ribulose5P = torch.zeros_like(y[:, 0]) + ((alg_21_vGluDH - alg_24_vPPI - alg_29_vR5PI) / cytoplasm)
        alg_30_vTransald = cytoplasm * (VmTransaldf * GA3P * Seduhept7P / (KmGA3P_3 * KmSeduhept_2) - VmTransaldr * F6P * Erythrose4P / (KmF6P * KmEry4P)) / ((1.0 + GA3P / KmGA3P_3 + F6P / KmF6P) * (1.0 + Seduhept7P / KmSeduhept_2 + Erythrose4P / KmEry4P))
        d_Seduhept7P = torch.zeros_like(y[:, 0]) + ((alg_27_vTransk1 - alg_30_vTransald) / cytoplasm)
        alg_31_vTransk2 = cytoplasm * (VmTransk2f * Erythrose4P * Xyl5P / (KmEry4P_2 * KmXyl5P_2) - VmTransk2r * F6P * GA3P / (KmF6P_2 * KmGA3P_4)) / ((1.0 + Xyl5P / KmXyl5P_2 + GA3P / KmGA3P_4) * (1.0 + Erythrose4P / KmEry4P_2 + F6P / KmF6P_2))
        d_F6P = torch.zeros_like(y[:, 0]) + ((alg_2_vPGI - alg_10_vPFK + alg_30_vTransald + alg_31_vTransk2) / cytoplasm)
        d_GA3P = torch.zeros_like(y[:, 0]) + ((alg_11_vALD - alg_13_vGAPDH - alg_15_vTPI + alg_27_vTransk1 - alg_30_vTransald + alg_31_vTransk2) / cytoplasm)
        d_Xyl5P = torch.zeros_like(y[:, 0]) + ((alg_29_vR5PI - alg_27_vTransk1 - alg_31_vTransk2) / cytoplasm)
        d_Erythrose4P = torch.zeros_like(y[:, 0]) + ((alg_30_vTransald - alg_31_vTransk2) / cytoplasm)

        return torch.stack(
            [d_P, d_G6P, d_F6P, d_F16P, d_NADH, d_NAD, d_BPG, d_P3G, d_P2G, d_PEP, d_PYR, d_ACE, d_GA3P, d_DHAP, d_D6PGluconoLactone, d_D6PGluconate, d_NADP, d_NADPH, d_Ribulose5P, d_Ribose5P, d_Xyl5P, d_Seduhept7P, d_Erythrose4P, d_GLCi],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
