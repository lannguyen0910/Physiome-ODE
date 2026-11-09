import torch
import torch.nn as nn
import math


class EndocrineMacgregorLeng2005BODE(nn.Module):
    """
    PyTorch translation of endocrine/macgregor_leng_2005_b/model.py.
    Dynamic states: 7; independent constants: 28; derived constants used by rates: 0.
    Original constant slots: 28; algebraic slots: 10.
    """
    source_file = 'endocrine/macgregor_leng_2005_b/model.py'
    source_state_dim = 7
    source_constant_dim = 28
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

        self.state_dim = 7
        self.parameter_dim = 28

        self.parameters = (
            parameters
            if parameters is not None
            else {'k6': 5, 'n1': 4, 'theta_1': 1, 'n3': 5, 'theta_3': 30, 'j1': 10, 'Is2': 50, 'k7': 5, 'Ise': 10, 'k1': 0.1, 'k2': 0.002, 'k3': 0.018, 'sb': 0.029, 'delta_b': 0.3, 'c': 0.01, 'k4': 0.5, 'k5': 71, 'sr': -0.56, 'delta_r': 0.2, 'k8': 0.07, 'g1': 1, 'gmax': 5, 'g2': 0.008, 'n2': 2, 'theta_2': 30, 'Ih': 50, 'delta': 60, 'delta1': 15}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'r': 0, 's': 0, 'g': 2, 'se': 0, 'f': 0.3, 'h': 0, 'hh': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k6', 'n1', 'theta_1', 'n3', 'theta_3', 'j1', 'Is2', 'k7', 'Ise', 'k1', 'k2', 'k3', 'sb', 'delta_b', 'c', 'k4', 'k5', 'sr', 'delta_r', 'k8', 'g1', 'gmax', 'g2', 'n2', 'theta_2', 'Ih', 'delta', 'delta1']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['r', 's', 'g', 'se', 'f', 'h', 'hh']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (7)
        r = y[:, 0]
        s = y[:, 1]
        g = y[:, 2]
        se = y[:, 3]
        f = y[:, 4]
        h = y[:, 5]
        hh = y[:, 6]

        # Independent parameters (source constant order)
        k6 = theta[:, 0]  # constants[0]
        n1 = theta[:, 1]  # constants[1]
        theta_1 = theta[:, 2]  # constants[2]
        n3 = theta[:, 3]  # constants[3]
        theta_3 = theta[:, 4]  # constants[4]
        j1 = theta[:, 5]  # constants[5]
        Is2 = theta[:, 6]  # constants[6]
        k7 = theta[:, 7]  # constants[7]
        Ise = theta[:, 8]  # constants[8]
        k1 = theta[:, 9]  # constants[9]
        k2 = theta[:, 10]  # constants[10]
        k3 = theta[:, 11]  # constants[11]
        sb = theta[:, 12]  # constants[12]
        delta_b = theta[:, 13]  # constants[13]
        c = theta[:, 14]  # constants[14]
        k4 = theta[:, 15]  # constants[15]
        k5 = theta[:, 16]  # constants[16]
        sr = theta[:, 17]  # constants[17]
        delta_r = theta[:, 18]  # constants[18]
        k8 = theta[:, 19]  # constants[19]
        g1 = theta[:, 20]  # constants[20]
        gmax = theta[:, 21]  # constants[21]
        g2 = theta[:, 22]  # constants[22]
        n2 = theta[:, 23]  # constants[23]
        theta_2 = theta[:, 24]  # constants[24]
        Ih = theta[:, 25]  # constants[25]
        delta = theta[:, 26]  # constants[26]
        delta1 = theta[:, 27]  # constants[27]

        # Initialize rates and algebraic variables, as in computeRates()
        d_r = torch.zeros_like(y[:, 0])
        d_s = torch.zeros_like(y[:, 0])
        d_g = torch.zeros_like(y[:, 0])
        d_se = torch.zeros_like(y[:, 0])
        d_f = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        d_hh = torch.zeros_like(y[:, 0])
        alg_0_ract = torch.zeros_like(y[:, 0])
        alg_1_h_delta = torch.zeros_like(y[:, 0])
        alg_2_phi_b_s = torch.zeros_like(y[:, 0])
        alg_3_phi_r_s = torch.zeros_like(y[:, 0])
        alg_4_Ir = torch.zeros_like(y[:, 0])
        alg_5_hact = torch.zeros_like(y[:, 0])
        alg_6_Ir2 = torch.zeros_like(y[:, 0])
        alg_7_Is = torch.zeros_like(y[:, 0])
        alg_8_h_delta1 = torch.zeros_like(y[:, 0])
        alg_9_hr = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_se = torch.zeros_like(y[:, 0]) + (Ise - k7 * se)
        d_hh = torch.zeros_like(y[:, 0]) + (j1 * (Ih - k8 * hh))
        alg_2_phi_b_s = 1.0 / (1.0 + torch.exp(-(torch.log(1.0 * (s + se)) / 2.302585092994046 - sb) / delta_b))
        d_f = torch.zeros_like(y[:, 0]) + (-(k1 * (r + c) * f) + (k2 + k3 * alg_2_phi_b_s) * (1.0 - f))
        alg_3_phi_r_s = 1.0 / (1.0 + torch.exp(-(torch.log(1.0 * (s + se)) / 2.302585092994046 - sr) / delta_r))
        d_h = torch.zeros_like(y[:, 0]) + (j1 * ((k4 + k5 * (1.0 - alg_3_phi_r_s)) * ((r + c) * f) - k8 * h))
        alg_7_Is = torch.where((t > 0.0) & (t <= 90.0), 10.0, torch.where((t > 90.0) & (t <= 180.0), 0.0, torch.where((t > 180.0) & (t <= 270.0), 10.0, torch.where((t > 270.0) & (t <= 360.0), 0.0, 0.0))))
        d_s = torch.zeros_like(y[:, 0]) + (alg_7_Is * g - k7 * s + Is2)
        alg_1_h_delta = (h + hh) * delta
        alg_5_hact = alg_1_h_delta ** n2 / (alg_1_h_delta ** n2 + (j1 * theta_2) ** n2)
        d_g = torch.zeros_like(y[:, 0]) + (g1 * alg_5_hact * ((gmax - g) / gmax) - g2 * alg_7_Is * g)
        alg_0_ract = 1.0 - s ** n1 / (s ** n1 + theta_1 ** n1)
        alg_8_h_delta1 = (h + hh) * delta1
        alg_9_hr = 1.0 - alg_8_h_delta1 ** n3 / (alg_8_h_delta1 ** n3 + (j1 * theta_3) ** n3)
        alg_4_Ir = torch.where((t >= 0.0) & (t <= 90.0), 0.0, torch.where((t >= 91.0) & (t <= 92.0), 10.0, torch.where((t >= 93.0) & (t <= 113.0), 0.0, torch.where((t >= 114.0) & (t <= 115.0), 10.0, torch.where((t >= 116.0) & (t <= 136.0), 0.0, torch.where((t >= 137.0) & (t <= 138.0), 10.0, torch.where((t >= 139.0) & (t <= 159.0), 0.0, torch.where((t >= 160.0) & (t <= 161.0), 10.0, torch.where((t >= 162.0) & (t <= 252.0), 0.0, torch.where((t >= 253.0) & (t <= 254.0), 10.0, torch.where((t >= 255.0) & (t <= 275.0), 0.0, torch.where((t >= 276.0) & (t <= 277.0), 10.0, torch.where((t >= 278.0) & (t <= 298.0), 0.0, torch.where((t >= 299.0) & (t <= 300.0), 10.0, torch.where((t >= 301.0) & (t <= 321.0), 0.0, torch.where((t >= 322.0) & (t <= 323.0), 10.0, 0.0))))))))))))))))
        alg_6_Ir2 = torch.where((t >= 0.0) & (t <= 5.0), 0.0, torch.where((t >= 6.0) & (t <= 7.0), 1000.0, torch.where((t >= 8.0) & (t <= 12.0), 0.0, torch.where((t >= 13.0) & (t <= 14.0), 1000.0, torch.where((t >= 15.0) & (t <= 21.0), 0.0, torch.where((t >= 22.0) & (t <= 23.0), 1000.0, torch.where((t >= 24.0) & (t <= 204.0), 0.0, torch.where((t >= 205.0) & (t <= 206.0), 1000.0, torch.where((t >= 207.0) & (t <= 217.0), 0.0, torch.where((t >= 218.0) & (t <= 219.0), 1000.0, torch.where((t >= 220.0) & (t <= 227.0), 0.0, torch.where((t >= 228.0) & (t <= 229.0), 1000.0, torch.where((t >= 230.0) & (t <= 310.0), 0.0, torch.where((t >= 311.0) & (t <= 312.0), 1000.0, torch.where((t >= 313.0) & (t <= 321.0), 0.0, torch.where((t >= 322.0) & (t <= 323.0), 1000.0, 0.0))))))))))))))))
        d_r = torch.zeros_like(y[:, 0]) + (alg_0_ract * alg_9_hr * alg_4_Ir - k6 * r + alg_6_Ir2)

        return torch.stack(
            [d_r, d_s, d_g, d_se, d_f, d_h, d_hh],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
