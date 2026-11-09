import torch
import torch.nn as nn
import math


class SignalTransductionKaragiannisPopel2004ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/karagiannis_popel_2004/model.py.
    Dynamic states: 13; independent constants: 15; derived constants used by rates: 0.
    Original constant slots: 15; algebraic slots: 9.
    """
    source_file = 'signal_transduction/karagiannis_popel_2004/model.py'
    source_state_dim = 13
    source_constant_dim = 15
    algebraic_dim = 9
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 13
        self.parameter_dim = 15

        self.parameters = (
            parameters
            if parameters is not None
            else {'kshed_eff': 2800, 'kon_MT1T2': 3540000, 'ki_MT1T2': 4.9e-09, 'kon_MT1T2M2P': 140000, 'koff_MT1T2M2P': 0.0047, 'kact_eff_m2': 2800, 'kon_M2T2': 5900000, 'ki_M2T2': 1.07e-06, 'kiso_M2T2': 33, 'k_iso_M2T2': 2e-08, 'kon_M2C1': 2600, 'koff_M2C1': 0.0021, 'kcat_M2C1': 0.0045, 'kcat_MT1C1': 0.00197, 'km_MT1C1': 2.9e-06}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'MT1': 1e-07, 'MT1cat': 0, 'MT1T2': 0, 'MT1T2M2P': 0, 'M2P': 5e-08, 'M2': 0, 'M2T2': 0, 'M2T2star': 0, 'M2C1': 0, 'C1': 1e-06, 'C1dmt1': 0, 'C1dm2': 0, 'T2': 5e-08}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['kshed_eff', 'kon_MT1T2', 'ki_MT1T2', 'kon_MT1T2M2P', 'koff_MT1T2M2P', 'kact_eff_m2', 'kon_M2T2', 'ki_M2T2', 'kiso_M2T2', 'k_iso_M2T2', 'kon_M2C1', 'koff_M2C1', 'kcat_M2C1', 'kcat_MT1C1', 'km_MT1C1']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['MT1', 'MT1cat', 'MT1T2', 'MT1T2M2P', 'M2P', 'M2', 'M2T2', 'M2T2star', 'M2C1', 'C1', 'C1dmt1', 'C1dm2', 'T2']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (13)
        MT1 = y[:, 0]
        MT1cat = y[:, 1]
        MT1T2 = y[:, 2]
        MT1T2M2P = y[:, 3]
        M2P = y[:, 4]
        M2 = y[:, 5]
        M2T2 = y[:, 6]
        M2T2star = y[:, 7]
        M2C1 = y[:, 8]
        C1 = y[:, 9]
        C1dmt1 = y[:, 10]
        C1dm2 = y[:, 11]
        T2 = y[:, 12]

        # Independent parameters (source constant order)
        kshed_eff = theta[:, 0]  # constants[0]
        kon_MT1T2 = theta[:, 1]  # constants[1]
        ki_MT1T2 = theta[:, 2]  # constants[2]
        kon_MT1T2M2P = theta[:, 3]  # constants[3]
        koff_MT1T2M2P = theta[:, 4]  # constants[4]
        kact_eff_m2 = theta[:, 5]  # constants[5]
        kon_M2T2 = theta[:, 6]  # constants[6]
        ki_M2T2 = theta[:, 7]  # constants[7]
        kiso_M2T2 = theta[:, 8]  # constants[8]
        k_iso_M2T2 = theta[:, 9]  # constants[9]
        kon_M2C1 = theta[:, 10]  # constants[10]
        koff_M2C1 = theta[:, 11]  # constants[11]
        kcat_M2C1 = theta[:, 12]  # constants[12]
        kcat_MT1C1 = theta[:, 13]  # constants[13]
        km_MT1C1 = theta[:, 14]  # constants[14]

        # Initialize rates and algebraic variables, as in computeRates()
        d_MT1 = torch.zeros_like(y[:, 0])
        d_MT1cat = torch.zeros_like(y[:, 0])
        d_MT1T2 = torch.zeros_like(y[:, 0])
        d_MT1T2M2P = torch.zeros_like(y[:, 0])
        d_M2P = torch.zeros_like(y[:, 0])
        d_M2 = torch.zeros_like(y[:, 0])
        d_M2T2 = torch.zeros_like(y[:, 0])
        d_M2T2star = torch.zeros_like(y[:, 0])
        d_M2C1 = torch.zeros_like(y[:, 0])
        d_C1 = torch.zeros_like(y[:, 0])
        d_C1dmt1 = torch.zeros_like(y[:, 0])
        d_C1dm2 = torch.zeros_like(y[:, 0])
        d_T2 = torch.zeros_like(y[:, 0])
        alg_0_v1 = torch.zeros_like(y[:, 0])
        alg_1_v2 = torch.zeros_like(y[:, 0])
        alg_2_v3 = torch.zeros_like(y[:, 0])
        alg_3_v4 = torch.zeros_like(y[:, 0])
        alg_4_v5 = torch.zeros_like(y[:, 0])
        alg_5_v6 = torch.zeros_like(y[:, 0])
        alg_6_v7 = torch.zeros_like(y[:, 0])
        alg_7_v8 = torch.zeros_like(y[:, 0])
        alg_8_v9 = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_v1 = kshed_eff * MT1 * MT1
        d_MT1cat = torch.zeros_like(y[:, 0]) + (alg_0_v1)
        alg_1_v2 = kon_MT1T2 * MT1 * T2 - ki_MT1T2 * kon_MT1T2 * MT1T2
        d_MT1 = torch.zeros_like(y[:, 0]) + (-alg_0_v1 - alg_1_v2)
        alg_2_v3 = kon_MT1T2M2P * MT1T2 * M2P - koff_MT1T2M2P * MT1T2M2P
        d_MT1T2 = torch.zeros_like(y[:, 0]) + (alg_1_v2 - alg_2_v3)
        d_M2P = torch.zeros_like(y[:, 0]) + (-alg_2_v3)
        alg_3_v4 = kact_eff_m2 * MT1 * MT1T2M2P
        d_MT1T2M2P = torch.zeros_like(y[:, 0]) + (alg_2_v3 - alg_3_v4)
        alg_4_v5 = kon_M2T2 * M2 * T2 - kon_M2T2 * ki_M2T2 * M2T2
        d_T2 = torch.zeros_like(y[:, 0]) + (-alg_1_v2 - alg_4_v5)
        alg_6_v7 = kon_M2C1 * M2 * C1 - koff_M2C1 * M2C1
        d_M2 = torch.zeros_like(y[:, 0]) + (alg_3_v4 - alg_4_v5 - alg_6_v7)
        alg_5_v6 = kiso_M2T2 * M2T2 - k_iso_M2T2 * M2T2star
        d_M2T2 = torch.zeros_like(y[:, 0]) + (alg_4_v5 - alg_5_v6)
        d_M2T2star = torch.zeros_like(y[:, 0]) + (alg_5_v6)
        alg_7_v8 = kcat_M2C1 * M2C1
        d_M2C1 = torch.zeros_like(y[:, 0]) + (alg_6_v7 - alg_7_v8)
        alg_8_v9 = kcat_MT1C1 / km_MT1C1 * MT1 * C1
        d_C1 = torch.zeros_like(y[:, 0]) + (-alg_6_v7 - alg_8_v9)
        d_C1dmt1 = torch.zeros_like(y[:, 0]) + (alg_8_v9)
        d_C1dm2 = torch.zeros_like(y[:, 0]) + (alg_7_v8)

        return torch.stack(
            [d_MT1, d_MT1cat, d_MT1T2, d_MT1T2M2P, d_M2P, d_M2, d_M2T2, d_M2T2star, d_M2C1, d_C1, d_C1dmt1, d_C1dm2, d_T2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
