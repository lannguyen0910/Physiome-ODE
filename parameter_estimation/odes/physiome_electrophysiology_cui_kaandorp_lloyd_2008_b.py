import torch
import torch.nn as nn
import math


class ElectrophysiologyCuiKaandorpLloyd2008BODE(nn.Module):
    """
    PyTorch translation of electrophysiology/cui_kaandorp_lloyd_2008_b/model.py.
    Dynamic states: 20; independent constants: 17; derived constants used by rates: 0.
    Original constant slots: 17; algebraic slots: 1.
    """
    source_file = 'electrophysiology/cui_kaandorp_lloyd_2008_b/model.py'
    source_state_dim = 20
    source_constant_dim = 17
    algebraic_dim = 1
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 20
        self.parameter_dim = 17

        self.parameters = (
            parameters
            if parameters is not None
            else {'r3': 44100000000, 'r4': 0.009, 'k_1': 0.9, 'k1a': 1, 'k2': 0.02, 'k_2': 0.3, 'r1': 273, 'r2': 0.0003437, 'k1b': 0.01253, 'k1': 0.025, 'r5': 30000, 'r6': 0.01506, 'k2a': 5e-05, 'k2b': 0.0002, 'k2c': 0.0037, 'td0': 1800, 'td': 2700}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Py': 25, 'Zn': 10000, 'Py1': 0, 'Dw': 4, 'Qw2': 0, 'Qw1': 0, 'Rw': 50, 'Mw': 0, 'Px': 50, 'Px1': 0, 'Dz': 2, 'Qz4': 0, 'Qz2': 0, 'Tp': 10000, 'Tp1': 0, 'Qz1': 0, 'Rz': 100, 'Qz3': 0, 'Qz5': 0, 'Mz': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['r3', 'r4', 'k_1', 'k1a', 'k2', 'k_2', 'r1', 'r2', 'k1b', 'k1', 'r5', 'r6', 'k2a', 'k2b', 'k2c', 'td0', 'td']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Py', 'Zn', 'Py1', 'Dw', 'Qw2', 'Qw1', 'Rw', 'Mw', 'Px', 'Px1', 'Dz', 'Qz4', 'Qz2', 'Tp', 'Tp1', 'Qz1', 'Rz', 'Qz3', 'Qz5', 'Mz']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (20)
        Py = y[:, 0]
        Zn = y[:, 1]
        Py1 = y[:, 2]
        Dw = y[:, 3]
        Qw2 = y[:, 4]
        Qw1 = y[:, 5]
        Rw = y[:, 6]
        Mw = y[:, 7]
        Px = y[:, 8]
        Px1 = y[:, 9]
        Dz = y[:, 10]
        Qz4 = y[:, 11]
        Qz2 = y[:, 12]
        Tp = y[:, 13]
        Tp1 = y[:, 14]
        Qz1 = y[:, 15]
        Rz = y[:, 16]
        Qz3 = y[:, 17]
        Qz5 = y[:, 18]
        Mz = y[:, 19]

        # Independent parameters (source constant order)
        r3 = theta[:, 0]  # constants[0]
        r4 = theta[:, 1]  # constants[1]
        k_1 = theta[:, 2]  # constants[2]
        k1a = theta[:, 3]  # constants[3]
        k2 = theta[:, 4]  # constants[4]
        k_2 = theta[:, 5]  # constants[5]
        r1 = theta[:, 6]  # constants[6]
        r2 = theta[:, 7]  # constants[7]
        k1b = theta[:, 8]  # constants[8]
        k1 = theta[:, 9]  # constants[9]
        r5 = theta[:, 10]  # constants[10]
        r6 = theta[:, 11]  # constants[11]
        k2a = theta[:, 12]  # constants[12]
        k2b = theta[:, 13]  # constants[13]
        k2c = theta[:, 14]  # constants[14]
        td0 = theta[:, 15]  # constants[15]
        td = theta[:, 16]  # constants[16]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Py = torch.zeros_like(y[:, 0])
        d_Zn = torch.zeros_like(y[:, 0])
        d_Py1 = torch.zeros_like(y[:, 0])
        d_Dw = torch.zeros_like(y[:, 0])
        d_Qw2 = torch.zeros_like(y[:, 0])
        d_Qw1 = torch.zeros_like(y[:, 0])
        d_Rw = torch.zeros_like(y[:, 0])
        d_Mw = torch.zeros_like(y[:, 0])
        d_Px = torch.zeros_like(y[:, 0])
        d_Px1 = torch.zeros_like(y[:, 0])
        d_Dz = torch.zeros_like(y[:, 0])
        d_Qz4 = torch.zeros_like(y[:, 0])
        d_Qz2 = torch.zeros_like(y[:, 0])
        d_Tp = torch.zeros_like(y[:, 0])
        d_Tp1 = torch.zeros_like(y[:, 0])
        d_Qz1 = torch.zeros_like(y[:, 0])
        d_Rz = torch.zeros_like(y[:, 0])
        d_Qz3 = torch.zeros_like(y[:, 0])
        d_Qz5 = torch.zeros_like(y[:, 0])
        d_Mz = torch.zeros_like(y[:, 0])
        alg_0_k3 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Py = torch.zeros_like(y[:, 0]) + (r4 * Py1 - r3 * Zn ** 2.0 * Py)
        d_Py1 = torch.zeros_like(y[:, 0]) + (r3 * Zn ** 2.0 * Py + k_1 * Qw2 - (r4 * Py1 + k1a * Dw * Py1))
        d_Qw2 = torch.zeros_like(y[:, 0]) + (k1a * Dw * Py1 - k_1 * Qw2)
        d_Px = torch.zeros_like(y[:, 0]) + (r2 * Px1 + k_1 * Qz4 - (r1 * Zn * Px + k1b * Dz * Px))
        d_Px1 = torch.zeros_like(y[:, 0]) + (r1 * Zn * Px + k_1 * Qz2 - (r2 * Px1 + k1 * Dz * Px1))
        d_Tp = torch.zeros_like(y[:, 0]) + (r6 * Tp1 - r5 * Zn * Tp)
        d_Tp1 = torch.zeros_like(y[:, 0]) + (r5 * Zn * Tp - r6 * Tp1)
        d_Zn = torch.zeros_like(y[:, 0]) + (r2 * Px1 + r6 * Tp1 - (r1 * Zn * Px + r5 * Zn * Tp))
        alg_0_k3 = torch.where((t >= 0.0) & (t < td0), 0.0, torch.where((t >= td0) & (t < td), 0.011, (1e309-1e309)))
        d_Dw = torch.zeros_like(y[:, 0]) + (k_1 * Qw2 + alg_0_k3 * Qw1 + k_2 * Qw1 - (k1a * Dw * Py1 + k2 * Dw * Rw))
        d_Rw = torch.zeros_like(y[:, 0]) + (alg_0_k3 * Qw1 + k_2 * Qw1 - k2 * Dw * Rw)
        d_Qw1 = torch.zeros_like(y[:, 0]) + (k2 * Dw * Rw - (alg_0_k3 * Qw1 + k_2 * Qw1))
        d_Mw = torch.zeros_like(y[:, 0]) + (alg_0_k3 * Qw1)
        d_Dz = torch.zeros_like(y[:, 0]) + (k_1 * Qz2 + alg_0_k3 * Qz1 + k_2 * Qz1 + k_1 * Qz4 - (k1b * Dz * Px + k1 * Dz * Px1 + k2a * Dz * Rz))
        d_Rz = torch.zeros_like(y[:, 0]) + (alg_0_k3 * Qz1 + k_2 * Qz1 + alg_0_k3 * Qz3 + k_2 * Qz3 + alg_0_k3 * Qz5 + k_2 * Qz5 - (k2a * Dz * Rz + k2b * Qz4 * Rz + k2c * Qz2 * Rz))
        d_Qz1 = torch.zeros_like(y[:, 0]) + (k2a * Dz * Rz - (alg_0_k3 * Qz1 + k_2 * Qz1))
        d_Qz2 = torch.zeros_like(y[:, 0]) + (k1 * Dz * Px1 + alg_0_k3 * Qz3 + k_2 * Qz3 - (k_1 * Qz2 + k2c * Qz2 * Rz))
        d_Qz3 = torch.zeros_like(y[:, 0]) + (k2c * Qz2 * Rz - (alg_0_k3 * Qz3 + k_2 * Qz3))
        d_Qz4 = torch.zeros_like(y[:, 0]) + (k1b * Dz * Px + alg_0_k3 * Qz5 + k_2 * Qz5 - (k_1 * Qz4 + k2b * Qz4 * Rz))
        d_Qz5 = torch.zeros_like(y[:, 0]) + (k2b * Qz4 * Rz - (alg_0_k3 * Qz5 + k_2 * Qz5))
        d_Mz = torch.zeros_like(y[:, 0]) + (alg_0_k3 * Qz1 + alg_0_k3 * Qz3 + alg_0_k3 * Qz5)

        return torch.stack(
            [d_Py, d_Zn, d_Py1, d_Dw, d_Qw2, d_Qw1, d_Rw, d_Mw, d_Px, d_Px1, d_Dz, d_Qz4, d_Qz2, d_Tp, d_Tp1, d_Qz1, d_Rz, d_Qz3, d_Qz5, d_Mz],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
