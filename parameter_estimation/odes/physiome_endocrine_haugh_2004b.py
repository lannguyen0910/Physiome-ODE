import torch
import torch.nn as nn
import math


class EndocrineHaugh2004bODE(nn.Module):
    """
    PyTorch translation of endocrine/haugh_2004b/model.py.
    Dynamic states: 1; independent constants: 12; derived constants used by rates: 0.
    Original constant slots: 19; algebraic slots: 0.
    """
    source_file = 'endocrine/haugh_2004b/model.py'
    source_state_dim = 1
    source_constant_dim = 19
    algebraic_dim = 0
    requires_dae_solver = True

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 1
        self.parameter_dim = 12

        self.parameters = (
            parameters
            if parameters is not None
            else {'kf1': 0.1, 'kt': 0.005, 'ke': 0.1, 'L': 0.01, 'kx2': 4.83, 'k_x2': 0.016, 'R_initial': 2000, 'krec': 0, 'kdeg': 0.05, 'kappaE': 0.2, 'Vs': 10, 'KD': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Ri': 200}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['kf1', 'kt', 'ke', 'L', 'kx2', 'k_x2', 'R_initial', 'krec', 'kdeg', 'kappaE', 'Vs', 'KD']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Ri']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        raise NotImplementedError(
            'This model requires a differentiable algebraic/DAE solve: initConsts() creates constants through root finding; source constants are not explicitly initialized: 15, 16'
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
