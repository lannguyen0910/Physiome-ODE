import torch
import torch.nn as nn
import math


class PKPDTham2008ODE(nn.Module):
    """
    PyTorch translation of PKPD/tham_2008/model.py.
    Dynamic states: 2; independent constants: 12; derived constants used by rates: 3.
    Original constant slots: 15; algebraic slots: 3.
    """
    source_file = 'PKPD/tham_2008/model.py'
    source_state_dim = 2
    source_constant_dim = 15
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
        self.parameter_dim = 12

        self.parameters = (
            parameters
            if parameters is not None
            else {'Dose': 5203.84, 'Dose_Int1': 0, 'Dose_Int2': 1, 'Dose_Length': 0.44359, 'Cycle_Int': 3, 'N_Cycle': 6, 'conversion_factor': 604800, 'E_max': 1, 'Amt_50': 10600, 't_half_eq': 7.67, 'Size_0': 6.66, 'T_Turnover': 21.8}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Ce': 0, 'Size': 6.66}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['Dose', 'Dose_Int1', 'Dose_Int2', 'Dose_Length', 'Cycle_Int', 'N_Cycle', 'conversion_factor', 'E_max', 'Amt_50', 't_half_eq', 'Size_0', 'T_Turnover']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Ce', 'Size']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        Ce = y[:, 0]
        Size = y[:, 1]

        # Independent parameters (source constant order)
        Dose = theta[:, 0]  # constants[0]
        Dose_Int1 = theta[:, 1]  # constants[1]
        Dose_Int2 = theta[:, 2]  # constants[2]
        Dose_Length = theta[:, 3]  # constants[3]
        Cycle_Int = theta[:, 4]  # constants[4]
        N_Cycle = theta[:, 5]  # constants[5]
        conversion_factor = theta[:, 6]  # constants[6]
        E_max = theta[:, 7]  # constants[7]
        Amt_50 = theta[:, 8]  # constants[8]
        t_half_eq = theta[:, 9]  # constants[9]
        Size_0 = theta[:, 10]  # constants[10]
        T_Turnover = theta[:, 11]  # constants[11]

        # Derived constants from the original initConsts()
        k_1 = 0.6931471805599453 / t_half_eq  # constants[12]
        k_2 = 0.6931471805599453 / T_Turnover  # constants[13]
        RateIn = Size_0 * k_2  # constants[14]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Ce = torch.zeros_like(y[:, 0])
        d_Size = torch.zeros_like(y[:, 0])
        alg_0_rem_time = torch.zeros_like(y[:, 0])
        alg_1_Effect = torch.zeros_like(y[:, 0])
        alg_2_Exposure = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_Effect = 1.0 - E_max * Ce / (Amt_50 + Ce)
        d_Size = torch.zeros_like(y[:, 0]) + ((RateIn * alg_1_Effect - k_2 * Size) * Size)
        alg_0_rem_time = t * conversion_factor % Cycle_Int * conversion_factor / conversion_factor
        alg_2_Exposure = torch.where((t < Cycle_Int * N_Cycle) & (Dose_Int1 < alg_0_rem_time) & (alg_0_rem_time < Dose_Length), Dose, torch.where((t < Cycle_Int * N_Cycle) & (Dose_Int2 < alg_0_rem_time) & (alg_0_rem_time < Dose_Int2 + Dose_Length), Dose, 0.0))
        d_Ce = torch.zeros_like(y[:, 0]) + (alg_2_Exposure / 1.0 - Ce * k_1)

        return torch.stack(
            [d_Ce, d_Size],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
