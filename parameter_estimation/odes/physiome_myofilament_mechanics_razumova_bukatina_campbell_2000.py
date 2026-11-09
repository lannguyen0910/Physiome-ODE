import torch
import torch.nn as nn
import math


class MyofilamentMechanicsRazumovaBukatinaCampbell2000ODE(nn.Module):
    """
    PyTorch translation of myofilament_mechanics/razumova_bukatina_campbell_2000/model.py.
    Dynamic states: 3; independent constants: 14; derived constants used by rates: 4.
    Original constant slots: 19; algebraic slots: 13.
    """
    source_file = 'myofilament_mechanics/razumova_bukatina_campbell_2000/model.py'
    source_state_dim = 3
    source_constant_dim = 19
    algebraic_dim = 13
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 3
        self.parameter_dim = 14

        self.parameters = (
            parameters
            if parameters is not None
            else {'R_T': 1, 'k_0_on': 0, 'k_0_off': 100, 'k_Ca_on': 120, 'k_Ca_off': 50, 'f_0': 50, 'f_prime_0': 400, 'h': 8, 'h_prime': 6, 'g': 4, 'n_H': 1, 'u': 1, 'w': 1, 'v': 1}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'D': 0, 'A_1': 0, 'A_2': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['R_T', 'k_0_on', 'k_0_off', 'k_Ca_on', 'k_Ca_off', 'f_0', 'f_prime_0', 'h', 'h_prime', 'g', 'n_H', 'u', 'w', 'v']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['D', 'A_1', 'A_2']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (3)
        D = y[:, 0]
        A_1 = y[:, 1]
        A_2 = y[:, 2]

        # Independent parameters (source constant order)
        R_T = theta[:, 0]  # constants[0]
        k_0_on = theta[:, 1]  # constants[1]
        k_0_off = theta[:, 2]  # constants[2]
        k_Ca_on = theta[:, 3]  # constants[3]
        k_Ca_off = theta[:, 4]  # constants[4]
        f_0 = theta[:, 5]  # constants[5]
        f_prime_0 = theta[:, 6]  # constants[6]
        h = theta[:, 7]  # constants[7]
        h_prime = theta[:, 8]  # constants[8]
        g = theta[:, 9]  # constants[9]
        n_H = theta[:, 10]  # constants[10]
        u = theta[:, 11]  # constants[11]
        w = theta[:, 12]  # constants[12]
        v = theta[:, 13]  # constants[13]

        # Derived constants from the original initConsts()
        Ca_50 = k_Ca_off / k_Ca_on  # constants[14]
        Ca = Ca_50 * 100.0  # constants[15]
        k_u_on = k_0_on + (k_Ca_on - k_0_on) * Ca / (Ca_50 + Ca)  # constants[17]
        k_u_off = k_0_off + (k_Ca_off - k_0_off) * Ca / (Ca_50 + Ca)  # constants[18]

        # Initialize rates and algebraic variables, as in computeRates()
        d_D = torch.zeros_like(y[:, 0])
        d_A_1 = torch.zeros_like(y[:, 0])
        d_A_2 = torch.zeros_like(y[:, 0])
        alg_0_R_off = torch.zeros_like(y[:, 0])
        alg_5_lambda_on = torch.zeros_like(y[:, 0])
        alg_6_lambda_A_2 = torch.zeros_like(y[:, 0])
        alg_7_k_w_on = torch.zeros_like(y[:, 0])
        alg_8_k_w_off = torch.zeros_like(y[:, 0])
        alg_9_f = torch.zeros_like(y[:, 0])
        alg_10_f_prime = torch.zeros_like(y[:, 0])
        alg_11_k_on = torch.zeros_like(y[:, 0])
        alg_12_k_off = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_A_2 = torch.zeros_like(y[:, 0]) + (h * A_1 - (h_prime + g) * A_2)
        alg_6_lambda_A_2 = A_2 / R_T
        alg_9_f = f_0 * (1.0 + alg_6_lambda_A_2 * (torch.exp(v - 1.0) - 1.0)) ** 2.0
        alg_10_f_prime = f_prime_0 * (1.0 + alg_6_lambda_A_2 * (torch.exp(-(v - 1.0)) - 1.0)) ** 2.0
        d_A_1 = torch.zeros_like(y[:, 0]) + (alg_9_f * D + h_prime * A_2 - (alg_10_f_prime + h) * A_1)
        alg_0_R_off = R_T - (D + A_1 + A_2)
        alg_5_lambda_on = (D + A_1 + A_2) / R_T
        alg_7_k_w_on = k_u_on * (1.0 + alg_5_lambda_on * (u - 1.0)) ** 2.0
        alg_11_k_on = alg_7_k_w_on * (1.0 + alg_6_lambda_A_2 * (torch.exp(w - 1.0) - 1.0)) ** 2.0
        alg_8_k_w_off = k_u_off * (u - alg_5_lambda_on * (u - 1.0)) ** 2.0
        alg_12_k_off = alg_8_k_w_off * (1.0 + alg_6_lambda_A_2 * (torch.exp(-(w - 1.0)) - 1.0)) ** 2.0
        d_D = torch.zeros_like(y[:, 0]) + (alg_11_k_on * alg_0_R_off + alg_10_f_prime * A_1 + g * A_2 - (alg_12_k_off + alg_9_f) * D)

        return torch.stack(
            [d_D, d_A_1, d_A_2],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
