import torch
import torch.nn as nn
import math


class CardiovascularCirculationMainWestkesselODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/MainWestkessel/model.py.
    Dynamic states: 3; independent constants: 21; derived constants used by rates: 0.
    Original constant slots: 21; algebraic slots: 15.
    """
    source_file = 'cardiovascular_circulation/MainWestkessel/model.py'
    source_state_dim = 3
    source_constant_dim = 21
    algebraic_dim = 15
    requires_dae_solver = True

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 3
        self.parameter_dim = 21

        self.parameters = (
            parameters
            if parameters is not None
            else {'CVao': 350, 'PlvIni': 1, 'VlvIni': 5, 'Vlv0': 500, 'CVmi': 400, 'PlaIni': 1, 'VlaIni': 4, 'Vla0': 20, 'ElvMax': 2.5, 'ElvMin': 0.1, 'T': 1, 'Ts1': 0.3, 'Ts2': 0.45, 'ElaMax': 0.25, 'ElaMin': 0.15, 'Tpwb': 0.92, 'Tpww': 0.09, 'Rc': 0.005, 'Rv': 0.65, 'Cv': 2.6, 'P0v': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'V': 500, 'V_2': 20, 'Pi': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['CVao', 'PlvIni', 'VlvIni', 'Vlv0', 'CVmi', 'PlaIni', 'VlaIni', 'Vla0', 'ElvMax', 'ElvMin', 'T', 'Ts1', 'Ts2', 'ElaMax', 'ElaMin', 'Tpwb', 'Tpww', 'Rc', 'Rv', 'Cv', 'P0v']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['V', 'V_2', 'Pi']
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
