import torch
import torch.nn as nn
import math


class PKPDGrange2001ODE(nn.Module):
    """
    PyTorch translation of PKPD/grange_2001/model.py.
    Dynamic states: 9; independent constants: 22; derived constants used by rates: 8.
    Original constant slots: 30; algebraic slots: 2.
    """
    source_file = 'PKPD/grange_2001/model.py'
    source_state_dim = 9
    source_constant_dim = 30
    algebraic_dim = 2
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 9
        self.parameter_dim = 22

        self.parameters = (
            parameters
            if parameters is not None
            else {'ka_b_dopa': 2.11, 'Q': 0.828, 'f_H': 0.13, 'CL_dopa_0': 0.823, 'V_dopa': 0.496, 'CL_OMD_b': 0.012, 'V_OMD_b': 0.196, 'ki': 0.00246, 'ka_Bens': 0.94, 'F_Bens': 0.022, 'V1_B': 0.202, 'CLd_B': 0.072, 'V2_B': 0.127, 'CL_B': 1.67, 'fm': 0.15, 'ka_M': 2.47, 'F_Ro': 1, 'V1_M': 0.0691, 'CLd_M': 1.06, 'CL_M': 4.29, 'V2_M': 3.2, 'F_G': 0.244}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'A_dopa_b': 101, 'C_dopa_b': 0, 'C_OMD_b': 0, 'C_Ro_central': 0, 'A_bens': 19.51, 'C_bens_central': 0, 'C_bens_peripheral': 0, 'A_Ro': 1.3658, 'C_Ro_peripheral': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['ka_b_dopa', 'Q', 'f_H', 'CL_dopa_0', 'V_dopa', 'CL_OMD_b', 'V_OMD_b', 'ki', 'ka_Bens', 'F_Bens', 'V1_B', 'CLd_B', 'V2_B', 'CL_B', 'fm', 'ka_M', 'F_Ro', 'V1_M', 'CLd_M', 'CL_M', 'V2_M', 'F_G']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['A_dopa_b', 'C_dopa_b', 'C_OMD_b', 'C_Ro_central', 'A_bens', 'C_bens_central', 'C_bens_peripheral', 'A_Ro', 'C_Ro_peripheral']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (9)
        A_dopa_b = y[:, 0]
        C_dopa_b = y[:, 1]
        C_OMD_b = y[:, 2]
        C_Ro_central = y[:, 3]
        A_bens = y[:, 4]
        C_bens_central = y[:, 5]
        C_bens_peripheral = y[:, 6]
        A_Ro = y[:, 7]
        C_Ro_peripheral = y[:, 8]

        # Independent parameters (source constant order)
        ka_b_dopa = theta[:, 0]  # constants[0]
        Q = theta[:, 1]  # constants[1]
        f_H = theta[:, 2]  # constants[2]
        CL_dopa_0 = theta[:, 3]  # constants[3]
        V_dopa = theta[:, 4]  # constants[4]
        CL_OMD_b = theta[:, 5]  # constants[5]
        V_OMD_b = theta[:, 6]  # constants[6]
        ki = theta[:, 7]  # constants[7]
        ka_Bens = theta[:, 8]  # constants[8]
        F_Bens = theta[:, 9]  # constants[9]
        V1_B = theta[:, 10]  # constants[10]
        CLd_B = theta[:, 11]  # constants[11]
        V2_B = theta[:, 12]  # constants[12]
        CL_B = theta[:, 13]  # constants[13]
        fm = theta[:, 14]  # constants[14]
        ka_M = theta[:, 15]  # constants[15]
        F_Ro = theta[:, 16]  # constants[16]
        V1_M = theta[:, 17]  # constants[17]
        CLd_M = theta[:, 18]  # constants[18]
        CL_M = theta[:, 19]  # constants[19]
        V2_M = theta[:, 20]  # constants[20]
        F_G = theta[:, 21]  # constants[21]

        # Derived constants from the original initConsts()
        CL_COMT = CL_dopa_0 * 0.1  # constants[22]
        CL_AADC0 = CL_dopa_0 * 0.69  # constants[23]
        CL_bens_total = CL_B / (1.0 - fm)  # constants[24]
        CL_H = f_H * CL_dopa_0  # constants[25]
        CL_REST = CL_dopa_0 * 0.21  # constants[26]
        CL_Ro = CL_bens_total * fm  # constants[27]
        F_H = 1.0 - CL_H / Q  # constants[28]
        F_b_dopa = F_H * F_G  # constants[29]

        # Initialize rates and algebraic variables, as in computeRates()
        d_A_dopa_b = torch.zeros_like(y[:, 0])
        d_C_dopa_b = torch.zeros_like(y[:, 0])
        d_C_OMD_b = torch.zeros_like(y[:, 0])
        d_C_Ro_central = torch.zeros_like(y[:, 0])
        d_A_bens = torch.zeros_like(y[:, 0])
        d_C_bens_central = torch.zeros_like(y[:, 0])
        d_C_bens_peripheral = torch.zeros_like(y[:, 0])
        d_A_Ro = torch.zeros_like(y[:, 0])
        d_C_Ro_peripheral = torch.zeros_like(y[:, 0])
        alg_0_CL_AADC = torch.zeros_like(y[:, 0])
        alg_1_CL_dopa = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_A_dopa_b = torch.zeros_like(y[:, 0]) + (-ka_b_dopa * A_dopa_b)
        d_C_OMD_b = torch.zeros_like(y[:, 0]) + (1.0 / V_OMD_b * (CL_COMT * C_dopa_b - CL_OMD_b * C_OMD_b))
        d_A_bens = torch.zeros_like(y[:, 0]) + (-ka_Bens * A_bens)
        d_C_bens_central = torch.zeros_like(y[:, 0]) + (1.0 / V1_B * (ka_Bens * A_bens * F_Bens - CL_bens_total * C_bens_central + CLd_B * (C_bens_peripheral - C_bens_central)))
        d_C_bens_peripheral = torch.zeros_like(y[:, 0]) + (1.0 / V2_B * CLd_B * (C_bens_central - C_bens_peripheral))
        d_A_Ro = torch.zeros_like(y[:, 0]) + (-ka_M * A_Ro)
        d_C_Ro_central = torch.zeros_like(y[:, 0]) + (1.0 / V1_M * (ka_M * A_Ro * F_Ro - CL_M * C_Ro_central + CL_Ro * C_bens_central + CLd_M * (C_Ro_peripheral - C_Ro_central)))
        d_C_Ro_peripheral = torch.zeros_like(y[:, 0]) + (1.0 / V2_M * CLd_M * (C_Ro_central - C_Ro_peripheral))
        alg_0_CL_AADC = CL_AADC0 / (1.0 + C_Ro_central / ki)
        alg_1_CL_dopa = alg_0_CL_AADC + CL_COMT + CL_REST
        d_C_dopa_b = torch.zeros_like(y[:, 0]) + (1.0 / V_dopa * (ka_b_dopa * A_dopa_b * F_b_dopa - alg_1_CL_dopa * C_dopa_b))

        return torch.stack(
            [d_A_dopa_b, d_C_dopa_b, d_C_OMD_b, d_C_Ro_central, d_A_bens, d_C_bens_central, d_C_bens_peripheral, d_A_Ro, d_C_Ro_peripheral],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
