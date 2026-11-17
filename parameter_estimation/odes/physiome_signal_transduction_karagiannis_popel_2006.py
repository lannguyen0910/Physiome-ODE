import torch
import torch.nn as nn
import math


class SignalTransductionKaragiannisPopel2006ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/karagiannis_popel_2006/model.py.
    Dynamic states: 16; independent constants: 21; derived constants used by rates: 0.
    Original constant slots: 21; algebraic slots: 0.
    """
    source_file = 'signal_transduction/karagiannis_popel_2006/model.py'
    source_state_dim = 16
    source_constant_dim = 21
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

        self.state_dim = 16
        self.parameter_dim = 21

        self.parameters = (
            parameters
            if parameters is not None
            else {'kM2T2_on': 5900000, 'kM2T2_off': 6.3, 'kM2T2_iso': 33, 'kM2T2_negativeiso': 2e-08, 'kM2C1_on': 2600, 'kM2C1_off': 0.0021, 'kM2C1_cat': 0.0045, 'kMT1T2_on': 2980000, 'kMT1T2_off': 0.202, 'kMT1T2M2pro_on': 140000, 'kMT1T2M2pro_off': 0.0047, 'kM2_act': 0.02, 'kMT1_shedeff': 2800, 'kMT1C1_cat': 0.00197, 'kMT1C1_on': 1000, 'kMT1C1_off': 1, 'kMT1T2M2proMT1_on': 3000, 'kMT1T2M2proMT1_off': 0.0009, 'qMT1': 0, 'qT2': 0, 'qpro': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'M2T2': 7.2e-09, 'M2C1': 8.5e-06, 'MT1T2': 1.39e-09, 'MT1T2M2proMT1': 5.6e-10, 'MT1C1': 2.9e-06, 'M2': 0, 'MT1': 0, 'M2_p': 0, 'T2': 0, 'C1_D': 0, 'C1': 0, 'MT1_cat': 0, 'MT1_t': 0, 'MT1T2M2pro': 0, 'MT1T2_star': 0, 'M2T2_star': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['kM2T2_on', 'kM2T2_off', 'kM2T2_iso', 'kM2T2_negativeiso', 'kM2C1_on', 'kM2C1_off', 'kM2C1_cat', 'kMT1T2_on', 'kMT1T2_off', 'kMT1T2M2pro_on', 'kMT1T2M2pro_off', 'kM2_act', 'kMT1_shedeff', 'kMT1C1_cat', 'kMT1C1_on', 'kMT1C1_off', 'kMT1T2M2proMT1_on', 'kMT1T2M2proMT1_off', 'qMT1', 'qT2', 'qpro']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['M2T2', 'M2C1', 'MT1T2', 'MT1T2M2proMT1', 'MT1C1', 'M2', 'MT1', 'M2_p', 'T2', 'C1_D', 'C1', 'MT1_cat', 'MT1_t', 'MT1T2M2pro', 'MT1T2_star', 'M2T2_star']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (16)
        M2T2 = y[:, 0]
        M2C1 = y[:, 1]
        MT1T2 = y[:, 2]
        MT1T2M2proMT1 = y[:, 3]
        MT1C1 = y[:, 4]
        M2 = y[:, 5]
        MT1 = y[:, 6]
        M2_p = y[:, 7]
        T2 = y[:, 8]
        C1_D = y[:, 9]
        C1 = y[:, 10]
        MT1_cat = y[:, 11]
        MT1_t = y[:, 12]
        MT1T2M2pro = y[:, 13]
        MT1T2_star = y[:, 14]
        M2T2_star = y[:, 15]

        # Independent parameters (source constant order)
        kM2T2_on = theta[:, 0]  # constants[0]
        kM2T2_off = theta[:, 1]  # constants[1]
        kM2T2_iso = theta[:, 2]  # constants[2]
        kM2T2_negativeiso = theta[:, 3]  # constants[3]
        kM2C1_on = theta[:, 4]  # constants[4]
        kM2C1_off = theta[:, 5]  # constants[5]
        kM2C1_cat = theta[:, 6]  # constants[6]
        kMT1T2_on = theta[:, 7]  # constants[7]
        kMT1T2_off = theta[:, 8]  # constants[8]
        kMT1T2M2pro_on = theta[:, 9]  # constants[9]
        kMT1T2M2pro_off = theta[:, 10]  # constants[10]
        kM2_act = theta[:, 11]  # constants[11]
        kMT1_shedeff = theta[:, 12]  # constants[12]
        kMT1C1_cat = theta[:, 13]  # constants[13]
        kMT1C1_on = theta[:, 14]  # constants[14]
        kMT1C1_off = theta[:, 15]  # constants[15]
        kMT1T2M2proMT1_on = theta[:, 16]  # constants[16]
        kMT1T2M2proMT1_off = theta[:, 17]  # constants[17]
        qMT1 = theta[:, 18]  # constants[18]
        qT2 = theta[:, 19]  # constants[19]
        qpro = theta[:, 20]  # constants[20]

        # Initialize rates and algebraic variables, as in computeRates()
        d_M2T2 = torch.zeros_like(y[:, 0])
        d_M2C1 = torch.zeros_like(y[:, 0])
        d_MT1T2 = torch.zeros_like(y[:, 0])
        d_MT1T2M2proMT1 = torch.zeros_like(y[:, 0])
        d_MT1C1 = torch.zeros_like(y[:, 0])
        d_M2 = torch.zeros_like(y[:, 0])
        d_MT1 = torch.zeros_like(y[:, 0])
        d_M2_p = torch.zeros_like(y[:, 0])
        d_T2 = torch.zeros_like(y[:, 0])
        d_C1_D = torch.zeros_like(y[:, 0])
        d_C1 = torch.zeros_like(y[:, 0])
        d_MT1_cat = torch.zeros_like(y[:, 0])
        d_MT1_t = torch.zeros_like(y[:, 0])
        d_MT1T2M2pro = torch.zeros_like(y[:, 0])
        d_MT1T2_star = torch.zeros_like(y[:, 0])
        d_M2T2_star = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_M2 = torch.zeros_like(y[:, 0]) + (-kM2T2_on * M2 * T2 + kM2T2_off * M2T2 - kM2C1_on * M2 * C1 + (kM2C1_off + kM2C1_cat) * M2C1 + kM2_act * MT1T2M2proMT1)
        d_MT1 = torch.zeros_like(y[:, 0]) + (qMT1 - kMT1_shedeff * MT1 * MT1 - kMT1T2_on * MT1 * T2 + kMT1T2_off * MT1T2 - kMT1C1_on * MT1 * C1 + (kMT1C1_off + kMT1C1_cat) * MT1C1 - kMT1T2M2proMT1_on * MT1 * MT1T2M2pro + kMT1T2M2proMT1_off * MT1T2M2proMT1 + kM2_act * MT1T2M2proMT1)
        d_MT1_t = torch.zeros_like(y[:, 0]) + (kMT1_shedeff * MT1 * MT1)
        d_MT1C1 = torch.zeros_like(y[:, 0]) + (kMT1C1_on * MT1 * C1 - (kMT1C1_off + kMT1C1_cat) * MT1C1)
        d_MT1T2 = torch.zeros_like(y[:, 0]) + (kMT1T2_on * MT1 * T2 - kMT1T2_off * MT1T2 - kMT1T2M2pro_on * MT1T2 * M2_p + kMT1T2M2pro_off * MT1T2M2pro)
        d_MT1T2M2pro = torch.zeros_like(y[:, 0]) + (kMT1T2M2pro_on * MT1T2 * M2_p - kMT1T2M2pro_off * MT1T2M2pro - kMT1T2M2proMT1_on * MT1 * MT1T2M2pro + kMT1T2M2proMT1_off * MT1T2M2proMT1)
        d_MT1T2M2proMT1 = torch.zeros_like(y[:, 0]) + (kMT1T2M2proMT1_on * MT1 * MT1T2M2pro - kMT1T2M2proMT1_off * MT1T2M2proMT1 - kM2_act * MT1T2M2proMT1)
        d_C1 = torch.zeros_like(y[:, 0]) + (-kMT1C1_on * MT1 * C1 + kMT1C1_off * MT1C1 - kM2C1_on * M2 * C1 + kM2C1_off * M2C1)
        d_C1_D = torch.zeros_like(y[:, 0]) + (kM2C1_cat * M2C1 + kMT1C1_cat * MT1C1)
        d_MT1_cat = torch.zeros_like(y[:, 0]) + (kMT1_shedeff * MT1 * MT1)
        d_T2 = torch.zeros_like(y[:, 0]) + (-kM2T2_on * M2 * T2 + kM2T2_off * M2T2 + qT2 - kMT1T2_on * MT1 * T2 + kMT1T2_off * MT1T2)
        d_M2_p = torch.zeros_like(y[:, 0]) + (qpro - kMT1T2M2pro_on * MT1T2 * M2_p + kMT1T2M2pro_off * MT1T2M2pro)
        d_M2T2 = torch.zeros_like(y[:, 0]) + (kM2T2_on * M2 * T2 - kM2T2_off * M2T2 - kM2T2_iso * M2T2 + kM2T2_negativeiso * M2T2_star)
        d_M2T2_star = torch.zeros_like(y[:, 0]) + (kM2T2_iso * M2T2 - kM2T2_negativeiso * M2T2_star)
        d_M2C1 = torch.zeros_like(y[:, 0]) + (kM2C1_on * M2 * C1 - (kM2C1_off + kM2C1_cat) * M2C1)
        d_MT1T2_star = torch.zeros_like(y[:, 0]) + (kM2_act * MT1T2M2proMT1)

        return torch.stack(
            [d_M2T2, d_M2C1, d_MT1T2, d_MT1T2M2proMT1, d_MT1C1, d_M2, d_MT1, d_M2_p, d_T2, d_C1_D, d_C1, d_MT1_cat, d_MT1_t, d_MT1T2M2pro, d_MT1T2_star, d_M2T2_star],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
