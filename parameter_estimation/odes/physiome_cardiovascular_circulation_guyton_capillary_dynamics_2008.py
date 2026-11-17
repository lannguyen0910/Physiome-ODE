import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonCapillaryDynamics2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_capillary_dynamics_2008/model.py.
    Dynamic states: 4; independent constants: 24; derived constants used by rates: 3.
    Original constant slots: 27; algebraic slots: 26.
    """
    source_file = 'cardiovascular_circulation/guyton_capillary_dynamics_2008/model.py'
    source_state_dim = 4
    source_constant_dim = 27
    algebraic_dim = 26
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 4
        self.parameter_dim = 24

        self.parameters = (
            parameters
            if parameters is not None
            else {'VEC': 14.8548, 'PPD': 4.4805e-06, 'RVS': 2.77751, 'DFP': -4.078e-07, 'VPF': 0.0123238, 'BFN': 2.79521, 'PVS': 3.71612, 'CFC': 0.01167, 'TRPL': 0, 'CPR': 40, 'LPPR': 0.03, 'LPDE': 8, 'LPK': 2.728e-14, 'PCR': 15, 'CPK': 0.000253, 'PCE': 1, 'TSSLML': 0.15, 'TSSLTC': 0.005, 'HYL': 60, 'CMPTSS': 2, 'PGHF': -2, 'GCOPF': 0.8092, 'VTSF': 6, 'PLDF': 4.2}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'VP': 3.00449, 'PRP': 216.243, 'VTS2': 0, 'TSP': 279.945}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['VEC', 'PPD', 'RVS', 'DFP', 'VPF', 'BFN', 'PVS', 'CFC', 'TRPL', 'CPR', 'LPPR', 'LPDE', 'LPK', 'PCR', 'CPK', 'PCE', 'TSSLML', 'TSSLTC', 'HYL', 'CMPTSS', 'PGHF', 'GCOPF', 'VTSF', 'PLDF']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['VP', 'PRP', 'VTS2', 'TSP']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        VP = y[:, 0]
        PRP = y[:, 1]
        VTS2 = y[:, 2]
        TSP = y[:, 3]

        # Independent parameters (source constant order)
        VEC = theta[:, 0]  # constants[0]
        PPD = theta[:, 1]  # constants[1]
        RVS = theta[:, 2]  # constants[2]
        DFP = theta[:, 3]  # constants[3]
        VPF = theta[:, 4]  # constants[4]
        BFN = theta[:, 5]  # constants[5]
        PVS = theta[:, 6]  # constants[6]
        CFC = theta[:, 7]  # constants[7]
        TRPL = theta[:, 8]  # constants[8]
        CPR = theta[:, 9]  # constants[9]
        LPPR = theta[:, 10]  # constants[10]
        LPDE = theta[:, 11]  # constants[11]
        LPK = theta[:, 12]  # constants[12]
        PCR = theta[:, 13]  # constants[13]
        CPK = theta[:, 14]  # constants[14]
        PCE = theta[:, 15]  # constants[15]
        TSSLML = theta[:, 16]  # constants[16]
        TSSLTC = theta[:, 17]  # constants[17]
        HYL = theta[:, 18]  # constants[18]
        CMPTSS = theta[:, 19]  # constants[19]
        PGHF = theta[:, 20]  # constants[20]
        GCOPF = theta[:, 21]  # constants[21]
        VTSF = theta[:, 22]  # constants[22]
        PLDF = theta[:, 23]  # constants[23]

        # Derived constants from the original initConsts()
        PC = RVS * 1.7 * BFN + PVS  # constants[24]
        PRCD = torch.where(PC - PCR < 0.0, 0.0, PC - PCR)  # constants[25]
        VTCPL = (PRCD * CPK) ** PCE  # constants[26]

        # Initialize rates and algebraic variables, as in computeRates()
        d_VP = torch.zeros_like(y[:, 0])
        d_PRP = torch.zeros_like(y[:, 0])
        d_VTS2 = torch.zeros_like(y[:, 0])
        d_TSP = torch.zeros_like(y[:, 0])
        alg_0_CPP = torch.zeros_like(y[:, 0])
        alg_1_CPPD = torch.zeros_like(y[:, 0])
        alg_2_DLP = torch.zeros_like(y[:, 0])
        alg_3_PPC = torch.zeros_like(y[:, 0])
        alg_4_VTS = torch.zeros_like(y[:, 0])
        alg_5_VTS1 = torch.zeros_like(y[:, 0])
        alg_7_CPI = torch.zeros_like(y[:, 0])
        alg_9_DPC = torch.zeros_like(y[:, 0])
        alg_10_PTCPR = torch.zeros_like(y[:, 0])
        alg_11_CHY = torch.zeros_like(y[:, 0])
        alg_12_POSHYL = torch.zeros_like(y[:, 0])
        alg_13_PTC = torch.zeros_like(y[:, 0])
        alg_14_PTT = torch.zeros_like(y[:, 0])
        alg_15_PGH = torch.zeros_like(y[:, 0])
        alg_16_VTC = torch.zeros_like(y[:, 0])
        alg_17_PIF = torch.zeros_like(y[:, 0])
        alg_18_PLD1 = torch.zeros_like(y[:, 0])
        alg_20_PLD = torch.zeros_like(y[:, 0])
        alg_21_VTL = torch.zeros_like(y[:, 0])
        alg_22_VPD = torch.zeros_like(y[:, 0])
        alg_23_DPL = torch.zeros_like(y[:, 0])
        alg_24_DPP = torch.zeros_like(y[:, 0])
        alg_25_DPI = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_4_VTS = VEC - VP - VPF
        d_VTS2 = torch.zeros_like(y[:, 0]) + (((alg_4_VTS - 12.0) * TSSLML - VTS2) * TSSLTC)
        alg_5_VTS1 = alg_4_VTS - VTS2
        alg_14_PTT = ((alg_5_VTS1 - VTSF) / VTSF) ** 2.0 * 1.0
        alg_11_CHY = (HYL / alg_4_VTS / 5.0) ** CMPTSS
        alg_15_PGH = alg_11_CHY * PGHF + alg_14_PTT
        alg_7_CPI = TSP / alg_4_VTS
        alg_10_PTCPR = 0.28 * alg_7_CPI + 0.0019 * alg_7_CPI ** 2.0
        alg_12_POSHYL = alg_11_CHY * 2.0
        alg_13_PTC = alg_12_POSHYL * alg_10_PTCPR * GCOPF
        alg_0_CPP = PRP / VP
        alg_3_PPC = 0.28 * alg_0_CPP + 0.0019 * alg_0_CPP ** 2.0
        alg_16_VTC = (PC - alg_3_PPC - alg_15_PGH + alg_13_PTC) * CFC + VTCPL
        alg_17_PIF = alg_15_PGH - alg_12_POSHYL
        alg_18_PLD1 = alg_17_PIF + PLDF - alg_14_PTT
        alg_20_PLD = torch.where(alg_18_PLD1 > 7.0, 7.0, alg_18_PLD1)
        alg_21_VTL = torch.where(alg_20_PLD < 0.0, 0.0, alg_20_PLD * 0.02)
        alg_22_VPD = alg_21_VTL - alg_16_VTC - DFP + TRPL
        d_VP = torch.zeros_like(y[:, 0]) + (alg_22_VPD)
        alg_1_CPPD = torch.where(alg_0_CPP - CPR < 0.0, 0.0, alg_0_CPP - CPR)
        alg_2_DLP = LPPR - alg_1_CPPD ** LPDE * LPK
        alg_9_DPC = VTCPL * alg_0_CPP + (alg_0_CPP - alg_7_CPI) * 0.00104
        alg_23_DPL = alg_7_CPI * alg_21_VTL
        alg_24_DPP = alg_2_DLP + alg_23_DPL - alg_9_DPC - PPD + TRPL * 72.0
        d_PRP = torch.zeros_like(y[:, 0]) + (alg_24_DPP)
        alg_25_DPI = alg_9_DPC - alg_23_DPL
        d_TSP = torch.zeros_like(y[:, 0]) + (alg_25_DPI)

        return torch.stack(
            [d_VP, d_PRP, d_VTS2, d_TSP],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
