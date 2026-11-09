import torch
import torch.nn as nn
import math


class CalciumDynamicsCuiKaandorp2006ODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/cui_kaandorp_2006/model.py.
    Dynamic states: 4; independent constants: 21; derived constants used by rates: 1.
    Original constant slots: 22; algebraic slots: 6.
    """
    source_file = 'calcium_dynamics/cui_kaandorp_2006/model.py'
    source_state_dim = 4
    source_constant_dim = 22
    algebraic_dim = 6
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
            else {'kM_plus': 500, 'kM_minus': 100, 'CaMtotal': 25, 'kN_plus': 5, 'kN_minus': 5, 'CaNtotal': 25, 'd': 0.4, 'f': 0.1, 'lamda': 5, 'N': 13, 'Vx': 1000, 'Kx': 500, 'V1': 30000, 'K1': 4.3, 'V2': 100, 'K2': 0.1, 'V3': 10000, 'K3': 100, 'kc': 10, 'alpha': 0.006, 'Caex': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'m': 0, 'x': 0, 'z': 1e-08, 'h': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['kM_plus', 'kM_minus', 'CaMtotal', 'kN_plus', 'kN_minus', 'CaNtotal', 'd', 'f', 'lamda', 'N', 'Vx', 'Kx', 'V1', 'K1', 'V2', 'K2', 'V3', 'K3', 'kc', 'alpha', 'Caex']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['m', 'x', 'z', 'h']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        m = y[:, 0]
        x = y[:, 1]
        z = y[:, 2]
        h = y[:, 3]

        # Independent parameters (source constant order)
        kM_plus = theta[:, 0]  # constants[0]
        kM_minus = theta[:, 1]  # constants[1]
        CaMtotal = theta[:, 2]  # constants[2]
        kN_plus = theta[:, 3]  # constants[3]
        kN_minus = theta[:, 4]  # constants[4]
        CaNtotal = theta[:, 5]  # constants[5]
        d = theta[:, 6]  # constants[6]
        f = theta[:, 7]  # constants[7]
        lamda = theta[:, 8]  # constants[8]
        N = theta[:, 9]  # constants[9]
        Vx = theta[:, 10]  # constants[10]
        Kx = theta[:, 11]  # constants[11]
        V1 = theta[:, 12]  # constants[12]
        K1 = theta[:, 13]  # constants[13]
        V2 = theta[:, 14]  # constants[14]
        K2 = theta[:, 15]  # constants[15]
        V3 = theta[:, 16]  # constants[16]
        K3 = theta[:, 17]  # constants[17]
        kc = theta[:, 18]  # constants[18]
        alpha = theta[:, 19]  # constants[19]
        Caex = theta[:, 20]  # constants[20]

        # Derived constants from the original initConsts()
        L0 = 10.0 ** (-(N / 2.0))  # constants[21]

        # Initialize rates and algebraic variables, as in computeRates()
        d_m = torch.zeros_like(y[:, 0])
        d_x = torch.zeros_like(y[:, 0])
        d_z = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        alg_2_y = torch.zeros_like(y[:, 0])
        alg_3_phi = torch.zeros_like(y[:, 0])
        alg_4_psi = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_m = torch.zeros_like(y[:, 0]) + (kM_plus * (CaMtotal - m) * x ** 3.0 - kM_minus * m)
        d_z = torch.zeros_like(y[:, 0]) + (kN_plus * (CaNtotal - z) * m - kN_minus * z)
        alg_2_y = 1.0 / z
        alg_3_phi = 1.0 / (1.0 + L0 * ((lamda * alg_2_y) ** (N + 1.0) - 1.0) / ((lamda * alg_2_y - 1.0) * ((alg_2_y - 1.0) / (alg_2_y ** (N + 1.0) - 1.0))))
        d_h = torch.zeros_like(y[:, 0]) + (d * alg_3_phi * (1.0 / z) * (1.0 - h) - f * (1.0 - alg_3_phi * (1.0 / z)) * h)
        alg_4_psi = (1.0 + L0) / ((alg_2_y ** (N + 1.0) - 1.0) / (alg_2_y - 1.0) + L0 * (((lamda * alg_2_y) ** (N + 1.0) - 1.0) / (lamda * alg_2_y - 1.0)))
        d_x = torch.zeros_like(y[:, 0]) + (Vx * Caex / (Kx + Caex) - (h * alg_4_psi * (1.0 / z) * (V1 * x / (K1 + x)) + h * alg_4_psi * (1.0 / z) * (V2 * x / (K2 + x)) + 1.0 / (1.0 + kc * z) * (V3 * x / (K3 + x)) + alpha * x))

        return torch.stack(
            [d_m, d_x, d_z, d_h],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
