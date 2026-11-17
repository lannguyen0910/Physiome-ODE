import torch
import torch.nn as nn
import math


class SignalTransductionFarhyBowersVeldhuis2007BODE(nn.Module):
    """
    PyTorch translation of signal_transduction/farhy_bowers_veldhuis_2007_b/model.py.
    Dynamic states: 5; independent constants: 33; derived constants used by rates: 2.
    Original constant slots: 35; algebraic slots: 2.
    """
    source_file = 'signal_transduction/farhy_bowers_veldhuis_2007_b/model.py'
    source_state_dim = 5
    source_constant_dim = 35
    algebraic_dim = 2
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 5
        self.parameter_dim = 33

        self.parameters = (
            parameters
            if parameters is not None
            else {'GHS': 20, 'k1': 3, 'kr1': 600, 't1': 400, 'n1': 5, 'n2': 2, 'g0': 1, 'ng0': 2.9, 'tg0': 200, 't2': 10, 'k4': 25, 'kr4': 20400, 't5': 10, 'n5': 2, 'S_basal': 900, 'k2': 25, 'kr2': 2200, 't3': 400, 'n3': 2, 'k3': 40, 'kr3': 63000, 't4': 28, 'n4': 5, 'g1': 45000, 'g2': 100, 'tg1': 390, 'tg2': 10000, 'ng1': 3, 'ng2': 2, 'kghr': 15, 'C': 10000, 'onset': 2, 'duration': 0.2}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'GH': 0, 'SRIF_PeV': 0, 'GHRH': 0, 'ghr_GHRH': 0, 'SRIF_ArC': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['GHS', 'k1', 'kr1', 't1', 'n1', 'n2', 'g0', 'ng0', 'tg0', 't2', 'k4', 'kr4', 't5', 'n5', 'S_basal', 'k2', 'kr2', 't3', 'n3', 'k3', 'kr3', 't4', 'n4', 'g1', 'g2', 'tg1', 'tg2', 'ng1', 'ng2', 'kghr', 'C', 'onset', 'duration']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['GH', 'SRIF_PeV', 'GHRH', 'ghr_GHRH', 'SRIF_ArC']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        GH = y[:, 0]
        SRIF_PeV = y[:, 1]
        GHRH = y[:, 2]
        ghr_GHRH = y[:, 3]
        SRIF_ArC = y[:, 4]

        # Independent parameters (source constant order)
        GHS = theta[:, 0]  # constants[0]
        k1 = theta[:, 1]  # constants[1]
        kr1 = theta[:, 2]  # constants[2]
        t1 = theta[:, 3]  # constants[3]
        n1 = theta[:, 4]  # constants[4]
        n2 = theta[:, 5]  # constants[5]
        g0 = theta[:, 6]  # constants[6]
        ng0 = theta[:, 7]  # constants[7]
        tg0 = theta[:, 8]  # constants[8]
        t2 = theta[:, 9]  # constants[9]
        k4 = theta[:, 10]  # constants[10]
        kr4 = theta[:, 11]  # constants[11]
        t5 = theta[:, 12]  # constants[12]
        n5 = theta[:, 13]  # constants[13]
        S_basal = theta[:, 14]  # constants[14]
        k2 = theta[:, 15]  # constants[15]
        kr2 = theta[:, 16]  # constants[16]
        t3 = theta[:, 17]  # constants[17]
        n3 = theta[:, 18]  # constants[18]
        k3 = theta[:, 19]  # constants[19]
        kr3 = theta[:, 20]  # constants[20]
        t4 = theta[:, 21]  # constants[21]
        n4 = theta[:, 22]  # constants[22]
        g1 = theta[:, 23]  # constants[23]
        g2 = theta[:, 24]  # constants[24]
        tg1 = theta[:, 25]  # constants[25]
        tg2 = theta[:, 26]  # constants[26]
        ng1 = theta[:, 27]  # constants[27]
        ng2 = theta[:, 28]  # constants[28]
        kghr = theta[:, 29]  # constants[29]
        C = theta[:, 30]  # constants[30]
        onset = theta[:, 31]  # constants[31]
        duration = theta[:, 32]  # constants[32]

        # Derived constants from the original initConsts()
        F1_GHS = g1 * ((GHS / tg1) ** ng1 / (1.0 + (GHS / tg1) ** ng1))  # constants[33]
        F2_GHS = g2 * ((GHS / tg2) ** ng2 / (1.0 + (GHS / tg2) ** ng2))  # constants[34]

        # Initialize rates and algebraic variables, as in computeRates()
        d_GH = torch.zeros_like(y[:, 0])
        d_SRIF_PeV = torch.zeros_like(y[:, 0])
        d_GHRH = torch.zeros_like(y[:, 0])
        d_ghr_GHRH = torch.zeros_like(y[:, 0])
        d_SRIF_ArC = torch.zeros_like(y[:, 0])
        alg_0_inject = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_GH = torch.zeros_like(y[:, 0]) + (kr1 * (((GHRH + ghr_GHRH) / t1) ** n1 / (((GHRH + ghr_GHRH) / t1) ** n1 + 1.0) + g0 * ((GHS / tg0) ** ng0 / (1.0 + (GHS / tg0) ** ng0)) + (1.0 + F1_GHS) / (1.0 + (SRIF_PeV / t2) ** n2 + F1_GHS)) - k1 * GH)
        d_SRIF_PeV = torch.zeros_like(y[:, 0]) + (-(k4 * SRIF_PeV) + kr4 * ((GH / t5) ** n5 / ((GH / t5) ** n5 + 1.0)) + S_basal)
        d_SRIF_ArC = torch.zeros_like(y[:, 0]) + (kr2 * (((GHRH + ghr_GHRH) / t3) ** n3 / (1.0 + ((GHRH + ghr_GHRH) / t3) ** n3)) - k2 * SRIF_ArC)
        d_GHRH = torch.zeros_like(y[:, 0]) + (kr3 * ((1.0 + F2_GHS) / (1.0 + ((SRIF_PeV + SRIF_ArC) / t4) ** n4 + F2_GHS)) + ghr_GHRH * 1.0 - k3 * GHRH)
        alg_0_inject = torch.where(t < onset, 0.0, torch.where((t >= onset) & (t <= onset + duration), C, torch.where(t > onset + duration, 0.0, (1e309-1e309))))
        d_ghr_GHRH = torch.zeros_like(y[:, 0]) + (alg_0_inject - kghr * ghr_GHRH)

        return torch.stack(
            [d_GH, d_SRIF_PeV, d_GHRH, d_ghr_GHRH, d_SRIF_ArC],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
