import torch
import torch.nn as nn
import math


class SignalTransductionVilarRnx2006ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/vilar_rnx_2006/model.py.
    Dynamic states: 6; independent constants: 7; derived constants used by rates: 2.
    Original constant slots: 9; algebraic slots: 12.
    """
    source_file = 'signal_transduction/vilar_rnx_2006/model.py'
    source_state_dim = 6
    source_constant_dim = 9
    algebraic_dim = 12
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 6
        self.parameter_dim = 7

        self.parameters = (
            parameters
            if parameters is not None
            else {'ka': 1, 'kcd': 0.0277777778, 'klid': 0.25, 'ki': 0.333333333, 'p_RI': 8, 'kr': 0.033333333, 'p_RII': 4}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'RI': 20, 'RII': 20, 'l_RI_RII': 0, 'RI_endo': 0, 'l_RI_RII_endo': 40, 'RII_endo': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['ka', 'kcd', 'klid', 'ki', 'p_RI', 'kr', 'p_RII']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['RI', 'RII', 'l_RI_RII', 'RI_endo', 'l_RI_RII_endo', 'RII_endo']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (6)
        RI = y[:, 0]
        RII = y[:, 1]
        l_RI_RII = y[:, 2]
        RI_endo = y[:, 3]
        l_RI_RII_endo = y[:, 4]
        RII_endo = y[:, 5]

        # Independent parameters (source constant order)
        ka = theta[:, 0]  # constants[0]
        kcd = theta[:, 1]  # constants[1]
        klid = theta[:, 2]  # constants[2]
        ki = theta[:, 3]  # constants[3]
        p_RI = theta[:, 4]  # constants[4]
        kr = theta[:, 5]  # constants[5]
        p_RII = theta[:, 6]  # constants[6]

        # Derived constants from the original initConsts()
        v5 = p_RI  # constants[7]
        v10 = p_RII  # constants[8]

        # Initialize rates and algebraic variables, as in computeRates()
        d_RI = torch.zeros_like(y[:, 0])
        d_RII = torch.zeros_like(y[:, 0])
        d_l_RI_RII = torch.zeros_like(y[:, 0])
        d_RI_endo = torch.zeros_like(y[:, 0])
        d_l_RI_RII_endo = torch.zeros_like(y[:, 0])
        d_RII_endo = torch.zeros_like(y[:, 0])
        alg_0_v2 = torch.zeros_like(y[:, 0])
        alg_1_v3 = torch.zeros_like(y[:, 0])
        alg_2_v4 = torch.zeros_like(y[:, 0])
        alg_3_v6 = torch.zeros_like(y[:, 0])
        alg_4_v7 = torch.zeros_like(y[:, 0])
        alg_5_v8 = torch.zeros_like(y[:, 0])
        alg_6_v9 = torch.zeros_like(y[:, 0])
        alg_7_v11 = torch.zeros_like(y[:, 0])
        alg_8_v12 = torch.zeros_like(y[:, 0])
        alg_9_v13 = torch.zeros_like(y[:, 0])
        alg_10_l = torch.zeros_like(y[:, 0])
        alg_11_v1 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_4_v7 = ki * RI
        alg_5_v8 = kr * RI_endo
        d_RI_endo = torch.zeros_like(y[:, 0]) + (alg_4_v7 - alg_5_v8)
        alg_2_v4 = ki * l_RI_RII
        alg_6_v9 = kr * l_RI_RII_endo
        d_l_RI_RII_endo = torch.zeros_like(y[:, 0]) + (alg_2_v4 - alg_6_v9)
        alg_8_v12 = ki * RII
        alg_9_v13 = kr * RII_endo
        d_RII_endo = torch.zeros_like(y[:, 0]) + (alg_8_v12 - alg_9_v13)
        alg_10_l = torch.where(t >= 2500.0, 0.01, 3e-05)
        alg_11_v1 = ka * alg_10_l * RI * RII
        alg_0_v2 = kcd * l_RI_RII
        alg_1_v3 = klid * l_RI_RII
        d_l_RI_RII = torch.zeros_like(y[:, 0]) + (alg_11_v1 - (alg_0_v2 + alg_1_v3 + alg_2_v4))
        alg_3_v6 = kcd * RI
        d_RI = torch.zeros_like(y[:, 0]) + (v5 + alg_5_v8 + alg_6_v9 - (alg_11_v1 + alg_3_v6 + alg_4_v7))
        alg_7_v11 = kcd * RII
        d_RII = torch.zeros_like(y[:, 0]) + (alg_6_v9 + v10 + alg_9_v13 - (alg_11_v1 + alg_7_v11 + alg_8_v12))

        return torch.stack(
            [d_RI, d_RII, d_l_RI_RII, d_RI_endo, d_l_RI_RII_endo, d_RII_endo],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
