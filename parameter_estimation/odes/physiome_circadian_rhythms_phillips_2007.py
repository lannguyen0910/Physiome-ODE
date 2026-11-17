import torch
import torch.nn as nn
import math


class CircadianRhythmsPhillips2007ODE(nn.Module):
    """
    PyTorch translation of circadian_rhythms/phillips_2007/model.py.
    Dynamic states: 3; independent constants: 15; derived constants used by rates: 0.
    Original constant slots: 17; algebraic slots: 4.
    """
    source_file = 'circadian_rhythms/phillips_2007/model.py'
    source_state_dim = 3
    source_constant_dim = 17
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

        self.state_dim = 3
        self.parameter_dim = 15

        self.parameters = (
            parameters
            if parameters is not None
            else {'tau_v': 10, 'v_vm': -1.9, 'Qmax': 100, 'theta': 10, 'sigma': 3, 'Vao': 1, 'tau_m': 10, 'v_mv': -1.9, 'v_maQao': 1, 'chi': 10.8, 'mu': 3.6, 'c0': 1, 'v_vc': -6.3, 'v_vh': 0.19, 'omega': 0.2617993877991494}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Vv': 0, 'Vm': 0, 'H': 15}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['tau_v', 'v_vm', 'Qmax', 'theta', 'sigma', 'Vao', 'tau_m', 'v_mv', 'v_maQao', 'chi', 'mu', 'c0', 'v_vc', 'v_vh', 'omega']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Vv', 'Vm', 'H']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        Vv = y[:, 0]
        Vm = y[:, 1]
        H = y[:, 2]

        # Independent parameters (source constant order)
        tau_v = theta[:, 0]  # constants[0]
        v_vm = theta[:, 1]  # constants[1]
        Qmax = theta[:, 2]  # constants[2]
        const_theta = theta[:, 3]  # constants[3]
        sigma = theta[:, 4]  # constants[4]
        Vao = theta[:, 5]  # constants[5]
        tau_m = theta[:, 6]  # constants[6]
        v_mv = theta[:, 7]  # constants[7]
        v_maQao = theta[:, 8]  # constants[8]
        chi = theta[:, 9]  # constants[9]
        mu = theta[:, 10]  # constants[10]
        c0 = theta[:, 11]  # constants[11]
        v_vc = theta[:, 12]  # constants[12]
        v_vh = theta[:, 13]  # constants[13]
        omega = theta[:, 14]  # constants[15]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Vv = torch.zeros_like(y[:, 0])
        d_Vm = torch.zeros_like(y[:, 0])
        d_H = torch.zeros_like(y[:, 0])
        alg_0_Qv = torch.zeros_like(y[:, 0])
        alg_1_Qm = torch.zeros_like(y[:, 0])
        alg_2_C = torch.zeros_like(y[:, 0])
        alg_3_D = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_Qv = Qmax / (1.0 + torch.exp(-(Vv - const_theta) / sigma))
        d_Vm = torch.zeros_like(y[:, 0]) + ((v_maQao + v_mv * alg_0_Qv - Vm) / (tau_m / 3600.0))
        alg_1_Qm = Qmax / (1.0 + torch.exp(-(Vm - const_theta) / sigma))
        d_H = torch.zeros_like(y[:, 0]) + ((mu * alg_1_Qm - H) / chi)
        alg_2_C = c0 + torch.cos(omega * t)
        alg_3_D = v_vc * alg_2_C + v_vh * H
        d_Vv = torch.zeros_like(y[:, 0]) + ((v_vm * alg_1_Qm + alg_3_D - Vv) / (tau_v / 3600.0))

        return torch.stack(
            [d_Vv, d_Vm, d_H],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
