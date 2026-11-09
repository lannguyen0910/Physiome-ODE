import torch
import torch.nn as nn
import math


class EndocrinePotterGrellerChoNuttallStroupSuvaTobin2005BODE(nn.Module):
    """
    PyTorch translation of endocrine/potter_greller_cho_nuttall_stroup_suva_tobin_2005_b/model.py.
    Dynamic states: 5; independent constants: 15; derived constants used by rates: 1.
    Original constant slots: 16; algebraic slots: 4.
    """
    source_file = 'endocrine/potter_greller_cho_nuttall_stroup_suva_tobin_2005_b/model.py'
    source_state_dim = 5
    source_constant_dim = 16
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

        self.state_dim = 5
        self.parameter_dim = 15

        self.parameters = (
            parameters
            if parameters is not None
            else {'k1': 1e-06, 'k1_2': 0.001, 'k2': 1e-07, 'k2_2': 0.001, 'kcl': 0.005, 'k3': 0.001, 'k3_2': 0.0001, 'k4': 0.002, 'k4_2': 0.4, 'De': 0, 'dmax': 7.5, 'dmin': 0, 'tau_on': 0.5, 'tau_off': 0.5, 'j': 9}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'P': 3, 'Ca': 0.0004, 'Ci': 0.05, 'Ra': 16.9, 'Ri': 1.7}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k1', 'k1_2', 'k2', 'k2_2', 'kcl', 'k3', 'k3_2', 'k4', 'k4_2', 'De', 'dmax', 'dmin', 'tau_on', 'tau_off', 'j']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['P', 'Ca', 'Ci', 'Ra', 'Ri']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (5)
        P = y[:, 0]
        Ca = y[:, 1]
        Ci = y[:, 2]
        Ra = y[:, 3]
        Ri = y[:, 4]

        # Independent parameters (source constant order)
        k1 = theta[:, 0]  # constants[0]
        k1_2 = theta[:, 1]  # constants[1]
        k2 = theta[:, 2]  # constants[2]
        k2_2 = theta[:, 3]  # constants[3]
        kcl = theta[:, 4]  # constants[4]
        k3 = theta[:, 5]  # constants[5]
        k3_2 = theta[:, 6]  # constants[6]
        k4 = theta[:, 7]  # constants[7]
        k4_2 = theta[:, 8]  # constants[8]
        De = theta[:, 9]  # constants[9]
        dmax = theta[:, 10]  # constants[10]
        dmin = theta[:, 11]  # constants[11]
        tau_on = theta[:, 12]  # constants[12]
        tau_off = theta[:, 13]  # constants[13]
        j = theta[:, 14]  # constants[14]

        # Derived constants from the original initConsts()
        cycle_length = tau_on + tau_off  # constants[15]

        # Initialize rates and algebraic variables, as in computeRates()
        d_P = torch.zeros_like(y[:, 0])
        d_Ca = torch.zeros_like(y[:, 0])
        d_Ci = torch.zeros_like(y[:, 0])
        d_Ra = torch.zeros_like(y[:, 0])
        d_Ri = torch.zeros_like(y[:, 0])
        alg_0_time_hour = torch.zeros_like(y[:, 0])
        alg_2_Dd = torch.zeros_like(y[:, 0])
        alg_3_D = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Ra = torch.zeros_like(y[:, 0]) + (k1_2 * Ca + k3 * Ri - (k1 * Ra * P + k3_2 * Ra))
        d_Ri = torch.zeros_like(y[:, 0]) + (k2_2 * Ci + k3_2 * Ra - (k2 * Ri * P + k3 * Ri))
        d_Ca = torch.zeros_like(y[:, 0]) + (k1 * Ra * P + k4 * Ci - (k1_2 * Ca + k4_2 * Ca))
        d_Ci = torch.zeros_like(y[:, 0]) + (k2 * Ri * P + k4_2 * Ca - (k2_2 * Ci + k4 * Ci))
        alg_0_time_hour = t * 1.0 / 3600.0
        alg_2_Dd = torch.where(alg_0_time_hour / cycle_length >= j, dmin, torch.where((alg_0_time_hour - torch.floor(alg_0_time_hour / cycle_length) * cycle_length >= 0.0) & (alg_0_time_hour - torch.floor(alg_0_time_hour / cycle_length) * cycle_length < tau_on), dmax, torch.where((alg_0_time_hour - torch.floor(alg_0_time_hour / cycle_length) * cycle_length >= tau_on) & (alg_0_time_hour - torch.floor(alg_0_time_hour / cycle_length) * cycle_length < cycle_length), dmin, (1e309-1e309))))
        alg_3_D = De + alg_2_Dd
        d_P = torch.zeros_like(y[:, 0]) + (k1_2 * Ca + k2_2 * Ci + alg_3_D - (k1 * Ra * P + k2 * Ri * P + kcl * P))

        return torch.stack(
            [d_P, d_Ca, d_Ci, d_Ra, d_Ri],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
