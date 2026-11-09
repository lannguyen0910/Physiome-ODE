import torch
import torch.nn as nn
import math


class EndocrineMacgregorLeng2005AODE(nn.Module):
    """
    PyTorch translation of endocrine/macgregor_leng_2005_a/model.py.
    Dynamic states: 4; independent constants: 14; derived constants used by rates: 0.
    Original constant slots: 14; algebraic slots: 4.
    """
    source_file = 'endocrine/macgregor_leng_2005_a/model.py'
    source_state_dim = 4
    source_constant_dim = 14
    algebraic_dim = 4
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
        self.parameter_dim = 14

        self.parameters = (
            parameters
            if parameters is not None
            else {'k6': 5, 'k7': 5, 'k1': 0.1, 'k2': 0.002, 'k3': 0.018, 'sb': 0.029, 'delta_b': 0.3, 'c': 0.01, 'k4': 9, 'k5': 71, 'sr': -0.56, 'delta_r': 0.3, 'k8': 0.07, 'j1': 10}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'r': 0, 's': 0, 'f': 0.3, 'h': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k6', 'k7', 'k1', 'k2', 'k3', 'sb', 'delta_b', 'c', 'k4', 'k5', 'sr', 'delta_r', 'k8', 'j1']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['r', 's', 'f', 'h']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        r = y[:, 0]
        s = y[:, 1]
        f = y[:, 2]
        h = y[:, 3]

        # Independent parameters (source constant order)
        k6 = theta[:, 0]  # constants[0]
        k7 = theta[:, 1]  # constants[1]
        k1 = theta[:, 2]  # constants[2]
        k2 = theta[:, 3]  # constants[3]
        k3 = theta[:, 4]  # constants[4]
        sb = theta[:, 5]  # constants[5]
        delta_b = theta[:, 6]  # constants[6]
        c = theta[:, 7]  # constants[7]
        k4 = theta[:, 8]  # constants[8]
        k5 = theta[:, 9]  # constants[9]
        sr = theta[:, 10]  # constants[10]
        delta_r = theta[:, 11]  # constants[11]
        k8 = theta[:, 12]  # constants[12]
        j1 = theta[:, 13]  # constants[13]

        # Initialize rates and algebraic variables, as in computeRates()
        d_r = torch.zeros_like(y[:, 0])
        d_s = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        alg_0_Ir = torch.zeros_like(y[:, 0])
        alg_1_Is = torch.zeros_like(y[:, 0])
        alg_2_phi_b_s = torch.zeros_like(y[:, 0])
        alg_3_phi_r_s = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_Ir = torch.where((t >= 0.0) & (t <= 90.0), 0.0, torch.where((t >= 91.0) & (t <= 92.0), 10.0, torch.where((t >= 93.0) & (t <= 113.0), 0.0, torch.where((t >= 114.0) & (t <= 115.0), 10.0, torch.where((t >= 116.0) & (t <= 136.0), 0.0, torch.where((t >= 137.0) & (t <= 138.0), 10.0, torch.where((t >= 139.0) & (t <= 159.0), 0.0, torch.where((t >= 160.0) & (t <= 161.0), 10.0, torch.where((t >= 162.0) & (t <= 252.0), 0.0, torch.where((t >= 253.0) & (t <= 254.0), 10.0, torch.where((t >= 255.0) & (t <= 275.0), 0.0, torch.where((t >= 276.0) & (t <= 277.0), 10.0, torch.where((t >= 278.0) & (t <= 298.0), 0.0, torch.where((t >= 299.0) & (t <= 300.0), 10.0, torch.where((t >= 301.0) & (t <= 321.0), 0.0, torch.where((t >= 322.0) & (t <= 323.0), 10.0, 0.0))))))))))))))))
        d_r = torch.zeros_like(y[:, 0]) + (alg_0_Ir - k6 * r)
        alg_1_Is = torch.where((t > 0.0) & (t <= 90.0), 10.0, torch.where((t > 90.0) & (t <= 180.0), 0.0, torch.where((t > 180.0) & (t <= 270.0), 10.0, torch.where((t > 270.0) & (t <= 360.0), 0.0, 0.0))))
        d_s = torch.zeros_like(y[:, 0]) + (alg_1_Is - k7 * s)
        alg_2_phi_b_s = 1.0 / (1.0 + torch.exp(-(torch.log(1.0 * s) / 2.302585092994046 - sb) / delta_b))
        d_f = torch.zeros_like(y[:, 0]) + (-(k1 * (r + c) * f) + (k2 + k3 * alg_2_phi_b_s) * (1.0 - f))
        alg_3_phi_r_s = 1.0 / (1.0 + torch.exp(-(torch.log(1.0 * s) / 2.302585092994046 - sr) / delta_r))
        d_h = torch.zeros_like(y[:, 0]) + (j1 * ((k4 + k5 * (1.0 - alg_3_phi_r_s)) * ((r + c) * f) - k8 * h))

        return torch.stack(
            [d_r, d_s, d_f, d_h],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
