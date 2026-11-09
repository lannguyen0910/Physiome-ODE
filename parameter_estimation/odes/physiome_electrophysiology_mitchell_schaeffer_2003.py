import torch
import torch.nn as nn
import math


class ElectrophysiologyMitchellSchaeffer2003ODE(nn.Module):
    """
    PyTorch translation of electrophysiology/mitchell_schaeffer_2003/model.py.
    Dynamic states: 2; independent constants: 10; derived constants used by rates: 0.
    Original constant slots: 10; algebraic slots: 3.
    """
    source_file = 'electrophysiology/mitchell_schaeffer_2003/model.py'
    source_state_dim = 2
    source_constant_dim = 10
    algebraic_dim = 3
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 2
        self.parameter_dim = 10

        self.parameters = (
            parameters
            if parameters is not None
            else {'IstimStart': 0, 'IstimEnd': 50000, 'IstimAmplitude': 0.2, 'IstimPeriod': 500, 'IstimPulseDuration': 1, 'tau_in': 0.3, 'tau_open': 120, 'tau_close': 150, 'V_gate': 0.13, 'tau_out': 6}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Vm': 8.20413566106744e-06, 'h': 0.8789655121804799}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['IstimStart', 'IstimEnd', 'IstimAmplitude', 'IstimPeriod', 'IstimPulseDuration', 'tau_in', 'tau_open', 'tau_close', 'V_gate', 'tau_out']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Vm', 'h']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        Vm = y[:, 0]
        h = y[:, 1]

        # Independent parameters (source constant order)
        IstimStart = theta[:, 0]  # constants[0]
        IstimEnd = theta[:, 1]  # constants[1]
        IstimAmplitude = theta[:, 2]  # constants[2]
        IstimPeriod = theta[:, 3]  # constants[3]
        IstimPulseDuration = theta[:, 4]  # constants[4]
        tau_in = theta[:, 5]  # constants[5]
        tau_open = theta[:, 6]  # constants[6]
        tau_close = theta[:, 7]  # constants[7]
        V_gate = theta[:, 8]  # constants[8]
        tau_out = theta[:, 9]  # constants[9]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Vm = torch.zeros_like(y[:, 0])
        d_h = torch.zeros_like(y[:, 0])
        alg_0_J_stim = torch.zeros_like(y[:, 0])
        alg_1_J_in = torch.zeros_like(y[:, 0])
        alg_2_J_out = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_h = torch.zeros_like(y[:, 0]) + (torch.where(Vm < V_gate, (1.0 - h) / tau_open, -h / tau_close))
        alg_0_J_stim = torch.where((t >= IstimStart) & (t <= IstimEnd) & (t - IstimStart - torch.floor((t - IstimStart) / IstimPeriod) * IstimPeriod <= IstimPulseDuration), IstimAmplitude, 0.0)
        alg_1_J_in = h * (Vm ** 2.0 * (1.0 - Vm)) / tau_in
        alg_2_J_out = -(Vm / tau_out)
        d_Vm = torch.zeros_like(y[:, 0]) + (alg_1_J_in + alg_2_J_out + alg_0_J_stim)

        return torch.stack(
            [d_Vm, d_h],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
