import torch
import torch.nn as nn


class CalciumDynamicsBorghansDupontGoldbeter1997aODE(nn.Module):

    source_file = (
        "calcium_dynamics/"
        "borghans_dupont_goldbeter_1997a/model.py"
    )

    source_state_dim = 3
    source_constant_dim = 14
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

        self.state_dim = 3
        self.parameter_dim = 13

        self.parameters = (
            parameters
            if parameters is not None
            else {
                "K_f": 1.0,
                "K": 10.0,
                "beta": 1.0,
                "v_0": 1.0,
                "v_1": 1.0,
                "V_M2": 6.5,
                "K_2": 0.1,
                "K_y": 0.2,
                "V_M3": 50.0,
                "k_d": 5000.0,
                "k_r": 5.0,
                "a": 10000.0,
                "d": 100.0,
            }
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {
                "Z": 0.3,
                "Y": 2.7,
                "rho": 0.2,
            }
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else [
                "K_f", "K", "beta", "v_0", "v_1",
                "V_M2", "K_2", "K_y", "V_M3",
                "k_d", "k_r", "a", "d",
            ]
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ["Z", "Y", "rho"]
        )

    def dynamics(self, t, y, theta):

        # States
        Z = y[:, 0]
        Y = y[:, 1]
        rho = y[:, 2]

        # Independent parameters
        (
            K_f, K, beta, v_0, v_1,
            V_M2, K_2, K_y, V_M3,
            k_d, k_r, a, d, _
        ) = theta.unbind(dim=-1)

        eps = 1e-8

        # Derived constant from original CellML model
        V_in = v_0 + v_1 * beta

        # Algebraic variables
        V_2 = V_M2 * (
            Z.square()
            / (K_2.square() + Z.square()).clamp_min(eps)
        )

        gamma = (a / d.clamp_min(eps)) * Z.pow(4)

        R_plus = (
            gamma * rho
            / (1.0 + gamma).clamp_min(eps)
        )

        V_3 = (
            beta
            * R_plus
            * V_M3
            * (
                Y.square()
                / (K_y.square() + Y.square()).clamp_min(eps)
            )
        )

        # State derivatives
        d_rho = (
            -k_d * Z.pow(4) * rho
            + k_r * (1.0 - rho)
        )

        d_Z = (
            V_in
            - V_2
            + V_3
            + K_f * Y
            - K * Z
        )

        d_Y = (
            V_2
            - V_3
            - K_f * Y
        )

        return torch.stack(
            [d_Z, d_Y, d_rho],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
