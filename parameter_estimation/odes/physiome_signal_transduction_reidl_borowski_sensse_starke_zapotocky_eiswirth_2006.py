import torch
import torch.nn as nn
import math


class SignalTransductionReidlBorowskiSensseStarkeZapotockyEiswirth2006ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/reidl_borowski_sensse_starke_zapotocky_eiswirth_2006/model.py.
    Dynamic states: 4; independent constants: 11; derived constants used by rates: 0.
    Original constant slots: 11; algebraic slots: 7.
    """
    source_file = 'signal_transduction/reidl_borowski_sensse_starke_zapotocky_eiswirth_2006/model.py'
    source_state_dim = 4
    source_constant_dim = 11
    algebraic_dim = 7
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
        self.parameter_dim = 11

        self.parameters = (
            parameters
            if parameters is not None
            else {'sigma': 5e-07, 'CNG_tot': 1.3e-13, 'CaM_tot': 2e-05, 'km_CNG_0': 0.01, 'km_CaM4': 2.5, 'kp_CaM4': 1100000000, 'kp_CNG_i': 2100000, 'km_CNG_i': 0.34, 'i_Ca': 20000, 'k_Ca': 1e-10, 'K_Ca': 1.2e-07}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'CNG_o': 0, 'CNG_i': 0, 'CaM4': 0, 'Ca': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['sigma', 'CNG_tot', 'CaM_tot', 'km_CNG_0', 'km_CaM4', 'kp_CaM4', 'kp_CNG_i', 'km_CNG_i', 'i_Ca', 'k_Ca', 'K_Ca']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['CNG_o', 'CNG_i', 'CaM4', 'Ca']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (4)
        CNG_o = y[:, 0]
        CNG_i = y[:, 1]
        CaM4 = y[:, 2]
        Ca = y[:, 3]

        # Independent parameters (source constant order)
        sigma = theta[:, 0]  # constants[0]
        CNG_tot = theta[:, 1]  # constants[1]
        CaM_tot = theta[:, 2]  # constants[2]
        km_CNG_0 = theta[:, 3]  # constants[3]
        km_CaM4 = theta[:, 4]  # constants[4]
        kp_CaM4 = theta[:, 5]  # constants[5]
        kp_CNG_i = theta[:, 6]  # constants[6]
        km_CNG_i = theta[:, 7]  # constants[7]
        i_Ca = theta[:, 8]  # constants[8]
        k_Ca = theta[:, 9]  # constants[9]
        K_Ca = theta[:, 10]  # constants[10]

        # Initialize rates and algebraic variables, as in computeRates()
        d_CNG_o = torch.zeros_like(y[:, 0])
        d_CNG_i = torch.zeros_like(y[:, 0])
        d_CaM4 = torch.zeros_like(y[:, 0])
        d_Ca = torch.zeros_like(y[:, 0])
        alg_0_kp_act = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Ca = torch.zeros_like(y[:, 0]) + (CNG_o / sigma * i_Ca - k_Ca / sigma * Ca / (Ca + K_Ca) - 4.0 * (kp_CaM4 * Ca ** 2.0 * (CaM_tot - CaM4 - CNG_i / sigma) - km_CaM4 * CaM4))
        d_CaM4 = torch.zeros_like(y[:, 0]) + (kp_CaM4 * Ca ** 2.0 * (CaM_tot - CaM4 - CNG_i / sigma) - km_CaM4 * CaM4 - kp_CNG_i / sigma * CaM4 * (CNG_tot - CNG_o) + km_CNG_i / sigma * CNG_i)
        d_CNG_i = torch.zeros_like(y[:, 0]) + (-km_CNG_i * CNG_i + kp_CNG_i * CaM4 * (CNG_tot - CNG_i))
        alg_0_kp_act = torch.where((t > 0.1) & (t < 0.2), 5.5, torch.where((t > 4.1) & (t < 4.2), 5.5, 1.6e-05))
        d_CNG_o = torch.zeros_like(y[:, 0]) + (alg_0_kp_act * (CNG_tot - CNG_o - CNG_i) - km_CNG_0 * CNG_o - kp_CNG_i * CNG_o * CaM4)

        return torch.stack(
            [d_CNG_o, d_CNG_i, d_CaM4, d_Ca],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
