import torch
import torch.nn as nn
import math


class EndocrineKyrylovSeveryanovaVieira2005ODE(nn.Module):
    """
    PyTorch translation of endocrine/kyrylov_severyanova_vieira_2005/model.py.
    Dynamic states: 5; independent constants: 24; derived constants used by rates: 0.
    Original constant slots: 24; algebraic slots: 15.
    """
    source_file = 'endocrine/kyrylov_severyanova_vieira_2005/model.py'
    source_state_dim = 5
    source_constant_dim = 24
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

        self.state_dim = 5
        self.parameter_dim = 24

        self.parameters = (
            parameters
            if parameters is not None
            else {'S0': 0.01, 'a00': -0.00843, 'a02': -0.44, 'c0': 0.443, 'e0': 0, 'epsilon': 0.5, 'S1': 0.01, 'a10': 0.082, 'a12': -0.0668, 'a11': -0.004, 'e1': 0, 'S2': 0.01, 'a23': 0.0576, 'a24': 0.000325, 'a20': 0, 'a21': 0.031, 'a22': -0.0957, 'e2': 0, 'S3': 0.01, 'a32': 0.00869, 'a33': -0.00857, 'S4': 0.01, 'a42': 0.000139, 'a44': -0.000143}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'y0': 0.4, 'y2': 1.17, 'y1': 1.4, 'y3': 0.95, 'y4': 0.65}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['S0', 'a00', 'a02', 'c0', 'e0', 'epsilon', 'S1', 'a10', 'a12', 'a11', 'e1', 'S2', 'a23', 'a24', 'a20', 'a21', 'a22', 'e2', 'S3', 'a32', 'a33', 'S4', 'a42', 'a44']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['y0', 'y2', 'y1', 'y3', 'y4']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        y0 = y[:, 0]
        y2 = y[:, 1]
        y1 = y[:, 2]
        y3 = y[:, 3]
        y4 = y[:, 4]

        # Independent parameters (source constant order)
        S0 = theta[:, 0]  # constants[0]
        a00 = theta[:, 1]  # constants[1]
        a02 = theta[:, 2]  # constants[2]
        c0 = theta[:, 3]  # constants[3]
        e0 = theta[:, 4]  # constants[4]
        epsilon = theta[:, 5]  # constants[5]
        S1 = theta[:, 6]  # constants[6]
        a10 = theta[:, 7]  # constants[7]
        a12 = theta[:, 8]  # constants[8]
        a11 = theta[:, 9]  # constants[9]
        e1 = theta[:, 10]  # constants[10]
        S2 = theta[:, 11]  # constants[11]
        a23 = theta[:, 12]  # constants[12]
        a24 = theta[:, 13]  # constants[13]
        a20 = theta[:, 14]  # constants[14]
        a21 = theta[:, 15]  # constants[15]
        a22 = theta[:, 16]  # constants[16]
        e2 = theta[:, 17]  # constants[17]
        S3 = theta[:, 18]  # constants[18]
        a32 = theta[:, 19]  # constants[19]
        a33 = theta[:, 20]  # constants[20]
        S4 = theta[:, 21]  # constants[21]
        a42 = theta[:, 22]  # constants[22]
        a44 = theta[:, 23]  # constants[23]

        # Initialize rates and algebraic variables, as in computeRates()
        d_y0 = torch.zeros_like(y[:, 0])
        d_y2 = torch.zeros_like(y[:, 0])
        d_y1 = torch.zeros_like(y[:, 0])
        d_y3 = torch.zeros_like(y[:, 0])
        d_y4 = torch.zeros_like(y[:, 0])
        alg_0_h0 = torch.zeros_like(y[:, 0])
        alg_1_h1 = torch.zeros_like(y[:, 0])
        alg_2_h2 = torch.zeros_like(y[:, 0])
        alg_3_h3 = torch.zeros_like(y[:, 0])
        alg_4_h4 = torch.zeros_like(y[:, 0])
        alg_5_r0 = torch.zeros_like(y[:, 0])
        alg_6_r1 = torch.zeros_like(y[:, 0])
        alg_7_r2 = torch.zeros_like(y[:, 0])
        alg_8_r3 = torch.zeros_like(y[:, 0])
        alg_9_r4 = torch.zeros_like(y[:, 0])
        alg_10_g0 = torch.zeros_like(y[:, 0])
        alg_11_g1 = torch.zeros_like(y[:, 0])
        alg_12_g2 = torch.zeros_like(y[:, 0])
        alg_13_g3 = torch.zeros_like(y[:, 0])
        alg_14_g4 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_5_r0 = torch.where((y0 < epsilon) & (a00 * y0 + a02 * y2 < 0.0), 1.0 - torch.exp(S0 * y0 ** 2.0 / ((a00 * y0 + a02 * y2) * (epsilon - y0) ** 2.0)), 1.0)
        alg_0_h0 = torch.where(a00 * y0 + a02 * y2 > 0.0, (a00 * y0 + a02 * y2) / (1.0 + (a00 * y0 + a02 * y2) / S0 * (1.0 - torch.exp(-((a00 * y0 + a02 * y2) / S0)))), torch.where(a00 * y0 + a02 * y2 <= 0.0, a00 * y0 + a02 * y2, (1e309-1e309)))
        alg_10_g0 = alg_0_h0 * alg_5_r0
        d_y0 = torch.zeros_like(y[:, 0]) + (1.0 * alg_10_g0 + c0 + e0)
        alg_6_r1 = torch.where((y1 < epsilon) & (a10 * y0 + a11 * y1 + a12 * y2 < 0.0), 1.0 - torch.exp(S1 * y1 ** 2.0 / ((a10 * y0 + a11 * y1 + a12 * y2) * (epsilon - y1) ** 2.0)), 1.0)
        alg_1_h1 = torch.where(a10 * y0 + a11 * y1 + a12 * y2 > 0.0, (a10 * y0 + a11 * y1 + a12 * y2) / (1.0 + (a10 * y0 + a11 * y1 + a12 * y2) / S1 * (1.0 - torch.exp(-((a10 * y0 + a11 * y1 + a12 * y2) / S1)))), torch.where(a10 * y0 + a11 * y1 + a12 * y2 <= 0.0, a10 * y0 + a11 * y1 + a12 * y2, (1e309-1e309)))
        alg_11_g1 = alg_1_h1 * alg_6_r1
        d_y1 = torch.zeros_like(y[:, 0]) + (1.0 * alg_11_g1 + e1)
        alg_7_r2 = torch.where((y2 < epsilon) & (a20 * y0 + a21 * y1 + a22 * y2 + a23 * y3 + a24 * y4 < 0.0), 1.0 - torch.exp(S2 * y2 ** 2.0 / ((a20 * y0 + a21 * y1 + a22 * y2 + a23 * y3 + a24 * y4) * (epsilon - y2) ** 2.0)), 1.0)
        alg_2_h2 = torch.where(a20 * y0 + a21 * y1 + a22 * y2 + a23 * y3 + a24 * y4 > 0.0, (a20 * y0 + a21 * y1 + a22 * y2 + a23 * y3 + a24 * y4) / (1.0 + (a20 * y0 + a21 * y1 + a22 * y2 + a23 * y3 + a24 * y4) / S2 * (1.0 - torch.exp(-((a20 * y0 + a21 * y1 + a22 * y2 + a23 * y3 + a24 * y4) / S2)))), torch.where(a20 * y0 + a21 * y1 + a22 * y2 + a23 * y3 + a24 * y4 <= 0.0, a20 * y0 + a21 * y1 + a22 * y2 + a23 * y3 + a24 * y4, (1e309-1e309)))
        alg_12_g2 = alg_2_h2 * alg_7_r2
        d_y2 = torch.zeros_like(y[:, 0]) + (1.0 * alg_12_g2 + e2)
        alg_8_r3 = torch.where((y3 < epsilon) & (a32 * y2 + a33 * y3 < 0.0), 1.0 - torch.exp(S3 * y3 ** 2.0 / ((a32 * y2 + a33 * y3) * (epsilon - y3) ** 2.0)), 1.0)
        alg_3_h3 = torch.where(a32 * y2 + a33 * y3 > 0.0, (a32 * y2 + a33 * y3) / (1.0 + (a32 * y2 + a33 * y3) / S3 * (1.0 - torch.exp(-((a32 * y2 + a33 * y3) / S3)))), torch.where(a32 * y2 + a33 * y3 <= 0.0, a32 * y2 + a33 * y3, (1e309-1e309)))
        alg_13_g3 = alg_3_h3 * alg_8_r3
        d_y3 = torch.zeros_like(y[:, 0]) + (1.0 * alg_13_g3)
        alg_9_r4 = torch.where((y4 < epsilon) & (a42 * y2 + a44 * y4 < 0.0), 1.0 - torch.exp(S4 * y4 ** 2.0 / ((a42 * y2 + a44 * y4) * (epsilon - y4) ** 2.0)), 1.0)
        alg_4_h4 = torch.where(a42 * y2 + a44 * y4 > 0.0, (a42 * y2 + a44 * y4) / (1.0 + (a42 * y2 + a44 * y4) / S4 * (1.0 - torch.exp(-((a42 * y2 + a44 * y4) / S4)))), torch.where(a42 * y2 + a44 * y4 <= 0.0, a42 * y2 + a44 * y4, (1e309-1e309)))
        alg_14_g4 = alg_4_h4 * alg_9_r4
        d_y4 = torch.zeros_like(y[:, 0]) + (1.0 * alg_14_g4)

        return torch.stack(
            [d_y0, d_y2, d_y1, d_y3, d_y4],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
