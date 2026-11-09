import torch
import torch.nn as nn


class CalciumDynamicsBorghansDupontGoldbeter1997aODE(nn.Module):
    """
    PyTorch translation of
    calcium_dynamics/borghans_dupont_goldbeter_1997a/model.py.

    Dynamic states: 3
    Source constant slots: 14
    Algebraic variables: 4

    Note:
        constants[13] (V_in) is a derived constant in the original
        initConsts():

            V_in = v_0 + v_1 * beta

        It is nevertheless included in theta because the generated
        dataset contains all 14 source constant slots.
    """

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
        self.parameter_dim = 14

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
                "V_in": 2.0,
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
                "K_f",
                "K",
                "beta",
                "v_0",
                "v_1",
                "V_M2",
                "K_2",
                "K_y",
                "V_M3",
                "k_d",
                "k_r",
                "a",
                "d",
                "V_in",
            ]
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else [
                "Z",
                "Y",
                "rho",
            ]
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):
        # States
        Z = y[:, 0]
        Y = y[:, 1]
        rho = y[:, 2]

        # Parameters
        K_f = theta[:, 0]      # constants[0]
        K = theta[:, 1]        # constants[1]
        beta = theta[:, 2]     # constants[2]
        v_0 = theta[:, 3]      # constants[3]
        v_1 = theta[:, 4]      # constants[4]
        V_M2 = theta[:, 5]     # constants[5]
        K_2 = theta[:, 6]      # constants[6]
        K_y = theta[:, 7]      # constants[7]
        V_M3 = theta[:, 8]     # constants[8]
        k_d = theta[:, 9]      # constants[9]
        k_r = theta[:, 10]     # constants[10]
        a = theta[:, 11]       # constants[11]
        d = theta[:, 12]       # constants[12]
        V_in = theta[:, 13]    # constants[13]

        # ---------------------------------------------------------
        # Algebraic variables
        # ---------------------------------------------------------

        # V_2
        V_2 = V_M2 * (
            Z**2 / (K_2**2 + Z**2)
        )

        # gamma
        gamma = (
            a / d
        ) * Z**4

        # R_plus
        R_plus = (
            gamma * rho / (1.0 + gamma)
        )

        # V_3
        V_3 = (
            beta
            * R_plus
            * V_M3
            * (Y**2 / (K_y**2 + Y**2))
        )

        # ---------------------------------------------------------
        # State derivatives
        # ---------------------------------------------------------

        # d rho / dt
        d_rho = (
            -(k_d * Z**4 * rho)
            + k_r * (1.0 - rho)
        )

        # d Z / dt
        d_Z = (
            V_in
            - V_2
            + V_3
            + K_f * Y
            - K * Z
        )

        # d Y / dt
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
