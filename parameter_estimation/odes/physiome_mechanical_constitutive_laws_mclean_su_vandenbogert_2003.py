import torch
import torch.nn as nn
import math


class MechanicalConstitutiveLawsMcleanSuVandenbogert2003ODE(nn.Module):
    """
    PyTorch translation of mechanical_constitutive_laws/mclean_su_vandenbogert_2003/model.py.
    Dynamic states: 1; independent constants: 24; derived constants used by rates: 0.
    Original constant slots: 29; algebraic slots: 9.
    """
    source_file = 'mechanical_constitutive_laws/mclean_su_vandenbogert_2003/model.py'
    source_state_dim = 1
    source_constant_dim = 29
    algebraic_dim = 9
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
        self.parameter_dim = 24

        self.parameters = (
            parameters
            if parameters is not None
            else {'a': 0.8, 'F_min': 10, 'F_max': 7000, 'L_CE_opt': 0.093, 'W': 0.63, 'V_max': 0.93, 'A': 0.25, 'g_max': 1.5, 'gamma': 5.67, 'V_max_2': 0.93, 'A_2': 0.25, 'g_max_2': 1.5, 'S': 2, 'S_2': 2, 'A_3': 0.25, 'V_max_3': 0.93, 'gamma_2': 5.67, 'g_max_3': 1.5, 'gamma_3': 5.67, 'k_SEE': 1000000, 'L_slack': 0.0025, 'L_slack_2': 0.0025, 'W_2': 0.63, 'L_CE_opt_2': 0.01}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'L_CE': 0.038}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['a', 'F_min', 'F_max', 'L_CE_opt', 'W', 'V_max', 'A', 'g_max', 'gamma', 'V_max_2', 'A_2', 'g_max_2', 'S', 'S_2', 'A_3', 'V_max_3', 'gamma_2', 'g_max_3', 'gamma_3', 'k_SEE', 'L_slack', 'L_slack_2', 'W_2', 'L_CE_opt_2']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['L_CE']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        raise NotImplementedError(
            'This model requires a differentiable algebraic/DAE solve: computeRates() calls an implicit rootfind function'
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
