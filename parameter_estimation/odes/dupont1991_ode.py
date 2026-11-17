import torch
import torch.nn as nn


class Dupont1991_ODE(nn.Module):
    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 2
        self.parameter_dim = 14

        self.parameters = (
            parameters
            if parameters is not None
            else {}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {}
        )

        self.parameter_names = (
            parameter_names
            if parameter_names is not None
            else [
                "v0",
                "v1",
                "VM2",
                "VM3",
                "KR",
                "KA",
                "kf",
                "k",
                "K2",
                "n",
                "m",
                "p",
                "betaf",
                "tp",
            ]
        )

        self.state_names = (
            state_names
            if state_names is not None
            else [
                "Z",
                "Y",
            ]
        )

    @staticmethod
    def hill(
        x,
        K,
        n,
        eps=1e-8,
    ):
        """
        Stable implementation of

            x^n / (K^n + x^n)

        using logarithms.
        """

        x = torch.clamp(
            x,
            min=eps,
        )

        K = torch.clamp(
            K,
            min=eps,
        )

        log_ratio = (
            n
            * (
                torch.log(K)
                -
                torch.log(x)
            )
        )

        return torch.sigmoid(
            -log_ratio
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        Z = y[:, 0]
        Y = y[:, 1]

        eps = 1e-8

        Z_safe = torch.clamp(
            Z,
            min=eps,
        )

        Y_safe = torch.clamp(
            Y,
            min=eps,
        )

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------

        v0 = theta[:, 0]
        v1 = theta[:, 1]
        VM2 = theta[:, 2]
        VM3 = theta[:, 3]
        KR = theta[:, 4]
        KA = theta[:, 5]
        kf = theta[:, 6]
        k = theta[:, 7]
        K2 = theta[:, 8]
        n = theta[:, 9]
        m = theta[:, 10]
        p = theta[:, 11]
        betaf = theta[:, 12]
        tp = theta[:, 13]

        # Positive parameters
        VM2 = torch.clamp(
            VM2,
            min=eps,
        )

        VM3 = torch.clamp(
            VM3,
            min=eps,
        )

        KR = torch.clamp(
            KR,
            min=eps,
        )

        KA = torch.clamp(
            KA,
            min=eps,
        )

        K2 = torch.clamp(
            K2,
            min=eps,
        )

        n = torch.clamp(
            n,
            min=eps,
        )

        m = torch.clamp(
            m,
            min=eps,
        )

        p = torch.clamp(
            p,
            min=eps,
        )

        # --------------------------------------------------
        # v2
        # --------------------------------------------------

        activation_v2 = self.hill(
            Z_safe,
            K2,
            n,
            eps,
        )

        v2 = (
            VM2
            * activation_v2
        )

        # --------------------------------------------------
        # v3
        # --------------------------------------------------

        y_activation = self.hill(
            Y_safe,
            KR,
            m,
            eps,
        )

        z_activation = self.hill(
            Z_safe,
            KA,
            p,
            eps,
        )

        v3 = (
            VM3
            * y_activation
            * z_activation
        )

        # --------------------------------------------------
        # beta(t)
        # --------------------------------------------------

        beta = torch.where(
            t < tp,
            torch.zeros_like(
                betaf
            ),
            betaf
            * torch.exp(
                -0.2
                * (
                    t
                    - tp
                )
            ),
        )

        # --------------------------------------------------
        # ODE
        # --------------------------------------------------

        dY = (
            v2
            - v3
            - kf * Y
        )

        dZ = (
            v0
            + v1 * beta
            - v2
            + v3
            + kf * Y
            - k * Z
        )

        dydt = torch.stack(
            [
                dZ,
                dY,
            ],
            dim=-1,
        )

        # Final safety check
        dydt = torch.nan_to_num(
            dydt,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        return dydt

    def forward(
        self,
        t,
        y,
        theta,
    ):

        return self.dynamics(
            t,
            y,
            theta,
        )
