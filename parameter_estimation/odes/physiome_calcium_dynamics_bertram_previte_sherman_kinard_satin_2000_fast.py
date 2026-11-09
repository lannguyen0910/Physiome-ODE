import torch
import torch.nn as nn
import math


class CalciumDynamicsBertramPreviteShermanKinardSatin2000FastODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/bertram_previte_sherman_kinard_satin_2000_fast/model.py.
    Dynamic states: 4; independent constants: 21; derived constants used by rates: 0.
    Original constant slots: 21; algebraic slots: 10.
    """
    source_file = 'calcium_dynamics/bertram_previte_sherman_kinard_satin_2000_fast/model.py'
    source_state_dim = 4
    source_constant_dim = 21
    algebraic_dim = 10
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
        self.parameter_dim = 21

        self.parameters = (
            parameters
            if parameters is not None
            else {'Cm': 4524, 'Vm': -22, 'VCa': 100, 'gCa': 280, 'sm': 7.5, 'VK': -80, 'gK': 1300, 'lambda_var': 1.1, 'tnbar': 9.09, 'Vn': -9, 'sn': 10, 'gs1': 20, 'Vs1': -40, 'ss1': 0.5, 'taus1': 1000, 'Vs2': -42, 'ss2': 0.4, 'gs2': 32, 'taus2': 120000, 'gl': 25, 'Vl': -40}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': -43, 'n': 0.03, 's1': 0.1, 's2': 0.434}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Cm', 'Vm', 'VCa', 'gCa', 'sm', 'VK', 'gK', 'lambda_var', 'tnbar', 'Vn', 'sn', 'gs1', 'Vs1', 'ss1', 'taus1', 'Vs2', 'ss2', 'gs2', 'taus2', 'gl', 'Vl']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'n', 's1', 's2']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        V = y[:, 0]
        n = y[:, 1]
        s1 = y[:, 2]
        s2 = y[:, 3]

        # Independent parameters (source constant order)
        Cm = theta[:, 0]  # constants[0]
        Vm = theta[:, 1]  # constants[1]
        VCa = theta[:, 2]  # constants[2]
        gCa = theta[:, 3]  # constants[3]
        sm = theta[:, 4]  # constants[4]
        VK = theta[:, 5]  # constants[5]
        gK = theta[:, 6]  # constants[6]
        lambda_var = theta[:, 7]  # constants[7]
        tnbar = theta[:, 8]  # constants[8]
        Vn = theta[:, 9]  # constants[9]
        sn = theta[:, 10]  # constants[10]
        gs1 = theta[:, 11]  # constants[11]
        Vs1 = theta[:, 12]  # constants[12]
        ss1 = theta[:, 13]  # constants[13]
        taus1 = theta[:, 14]  # constants[14]
        Vs2 = theta[:, 15]  # constants[15]
        ss2 = theta[:, 16]  # constants[16]
        gs2 = theta[:, 17]  # constants[17]
        taus2 = theta[:, 18]  # constants[18]
        gl = theta[:, 19]  # constants[19]
        Vl = theta[:, 20]  # constants[20]

        # Initialize rates and algebraic variables, as in computeRates()
        d_V = torch.zeros_like(y[:, 0])
        d_n = torch.zeros_like(y[:, 0])
        d_s1 = torch.zeros_like(y[:, 0])
        d_s2 = torch.zeros_like(y[:, 0])
        alg_0_minf = torch.zeros_like(y[:, 0])
        alg_1_ninf = torch.zeros_like(y[:, 0])
        alg_2_s1inf = torch.zeros_like(y[:, 0])
        alg_3_s2inf = torch.zeros_like(y[:, 0])
        alg_4_ICa = torch.zeros_like(y[:, 0])
        alg_5_taun = torch.zeros_like(y[:, 0])
        alg_6_IK = torch.zeros_like(y[:, 0])
        alg_7_Is1 = torch.zeros_like(y[:, 0])
        alg_8_Is2 = torch.zeros_like(y[:, 0])
        alg_9_Il = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_2_s1inf = 1.0 / (1.0 + torch.exp((Vs1 - V) / ss1))
        d_s1 = torch.zeros_like(y[:, 0]) + ((alg_2_s1inf - s1) / (taus1 * 1.0))
        alg_3_s2inf = 1.0 / (1.0 + torch.exp((Vs2 - V) / ss2))
        d_s2 = torch.zeros_like(y[:, 0]) + ((alg_3_s2inf - s2) / (taus2 * 1.0))
        alg_5_taun = tnbar / (1.0 + torch.exp((V - Vn) / sn))
        alg_1_ninf = 1.0 / (1.0 + torch.exp((Vn - V) / sn))
        d_n = torch.zeros_like(y[:, 0]) + (lambda_var * (alg_1_ninf - n) / (alg_5_taun * 1.0))
        alg_0_minf = 1.0 / (1.0 + torch.exp((Vm - V) / sm))
        alg_4_ICa = gCa * alg_0_minf * (V - VCa)
        alg_6_IK = gK * n * (V - VK)
        alg_9_Il = gl * (V - Vl)
        alg_7_Is1 = gs1 * s1 * (V - VK)
        alg_8_Is2 = gs2 * s2 * (V - VK)
        d_V = torch.zeros_like(y[:, 0]) + (-(alg_4_ICa + alg_6_IK + alg_9_Il + alg_7_Is1 + alg_8_Is2) / Cm)

        return torch.stack(
            [d_V, d_n, d_s1, d_s2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
