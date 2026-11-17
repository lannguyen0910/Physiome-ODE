import torch
import torch.nn as nn
import math


class CalciumDynamicsSternSongShamYangBohelerRios1999ODE(nn.Module):
    """
    PyTorch translation of calcium_dynamics/stern_song_sham_yang_boheler_rios_1999/model.py.
    Dynamic states: 4; independent constants: 5; derived constants used by rates: 0.
    Original constant slots: 5; algebraic slots: 0.
    """
    source_file = 'calcium_dynamics/stern_song_sham_yang_boheler_rios_1999/model.py'
    source_state_dim = 4
    source_constant_dim = 5
    algebraic_dim = 0
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
        self.parameter_dim = 5

        self.parameters = (
            parameters
            if parameters is not None
            else {'ki': 0.5, 'ko': 35, 'kim': 0.005, 'kom': 0.06, 'Ca': 0.0001}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'R': 1, 'RI': 1, 'O': 1, 'I': 1}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['ki', 'ko', 'kim', 'kom', 'Ca']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['R', 'RI', 'O', 'I']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        R = y[:, 0]
        RI = y[:, 1]
        O = y[:, 2]
        I = y[:, 3]

        # Independent parameters (source constant order)
        ki = theta[:, 0]  # constants[0]
        ko = theta[:, 1]  # constants[1]
        kim = theta[:, 2]  # constants[2]
        kom = theta[:, 3]  # constants[3]
        Ca = theta[:, 4]  # constants[4]

        # Initialize rates and algebraic variables, as in computeRates()
        d_R = torch.zeros_like(y[:, 0])
        d_RI = torch.zeros_like(y[:, 0])
        d_O = torch.zeros_like(y[:, 0])
        d_I = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_R = torch.zeros_like(y[:, 0]) + (kom * O + kim * RI - (ko * Ca ** 2.0 * R + ki * Ca * R))
        d_O = torch.zeros_like(y[:, 0]) + (kim * I + ko * Ca ** 2.0 * R - (kom * O + ki * Ca * O))
        d_I = torch.zeros_like(y[:, 0]) + (ki * Ca * O + ko * Ca ** 2.0 * RI - (kim * I + kom * I))
        d_RI = torch.zeros_like(y[:, 0]) + (ki * Ca * R + kom * I - (kim * RI + ko * Ca ** 2.0 * RI))

        return torch.stack(
            [d_R, d_RI, d_O, d_I],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
