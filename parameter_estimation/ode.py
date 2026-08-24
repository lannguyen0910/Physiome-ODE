import torch
import torch.nn as nn


class PhysiomeODE(nn.Module):
    def __init__(self):
        super().__init__()

        self.state_dim = 2
        self.parameter_dim = 14

    @staticmethod
    def hill(x, K, n, eps=1e-8):
        x = torch.clamp(x, min=eps)
        K = torch.clamp(K, min=eps)

        log_ratio = n * (
            torch.log(K) - torch.log(x)
        )

        return torch.sigmoid(-log_ratio)

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dy/dt for the Physiome Dupont 1991b model.

        Parameters
        ----------
        t:
            Current time.
            Scalar tensor.

        y:
            Current states.
            Shape: [batch, 2]

            y[:, 0] = Z
            y[:, 1] = Y

        theta:
            ODE constants.
            Shape: [batch, 14]

            theta[:, 0]  = v0
            theta[:, 1]  = v1
            theta[:, 2]  = VM2
            theta[:, 3]  = VM3
            theta[:, 4]  = KR
            theta[:, 5]  = KA
            theta[:, 6]  = kf
            theta[:, 7]  = k
            theta[:, 8]  = K2
            theta[:, 9]  = n
            theta[:, 10] = m
            theta[:, 11] = p
            theta[:, 12] = betaf
            theta[:, 13] = tp

        Returns
        -------
        dydt:
            Shape [batch, 2]
        """

        Z = y[:, 0]
        Y = y[:, 1]

        # States should be non-negative
        Z_safe = torch.clamp(Z, min=1e-8)
        Y_safe = torch.clamp(Y, min=1e-8)

        # Parameters
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

        eps = 1e-8

        Z_safe = torch.clamp(Z, min=eps)
        Y_safe = torch.clamp(Y, min=eps)

        VM2 = torch.clamp(VM2, min=eps)
        VM3 = torch.clamp(VM3, min=eps)

        KR = torch.clamp(KR, min=eps)
        KA = torch.clamp(KA, min=eps)
        K2 = torch.clamp(K2, min=eps)

        n = torch.clamp(n, min=eps)
        m = torch.clamp(m, min=eps)
        p = torch.clamp(p, min=eps)

        # v2
        Z_n = Z_safe ** n
        K2_n = K2 ** n

        v2 = VM2 * Z_n / (
            K2_n + Z_n + eps
        )

        # v3
        Y_m = Y_safe ** m
        KR_m = KR ** m

        Z_p = Z_safe ** p
        KA_p = KA ** p

        y_activation = Y_m / (
            KR_m + Y_m + eps
        )

        z_activation = Z_p / (
            KA_p + Z_p + eps
        )

        v3 = VM3 * y_activation * z_activation

        # beta(t) =
        #     0                                  if t < tp
        #     betaf * exp(-0.2*(t-tp))           if t >= tp
        if t.ndim == 0:
            beta = torch.where(
                t < tp,
                torch.zeros_like(tp),
                betaf * torch.exp(
                    -0.2 * (t - tp)
                ),
            )
        else:
            beta = torch.where(
                t.unsqueeze(-1) < tp,
                torch.zeros_like(tp),
                betaf * torch.exp(
                    -0.2 * (t.unsqueeze(-1) - tp)
                ),
            )

        # ODE equations
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

        return torch.stack(
            [dZ, dY],
            dim=-1,
        )


class RK4Solver:
    """
    Differentiable fourth-order Runge-Kutta solver.

    Input
    -----
    y0:
        [batch, state_dim]

    t:
        [time]

    theta:
        [batch, parameter_dim]

    Output
    ------
    trajectory:
        [batch, time, state_dim]
    """

    def __init__(self, dynamics):
        self.dynamics = dynamics

    def __call__(
        self,
        y0: torch.Tensor,
        t: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:

        return self.solve(
            y0=y0,
            t=t,
            theta=theta,
        )

    def solve(
        self,
        y0: torch.Tensor,
        t: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:

        trajectory = []
        y = y0
        trajectory.append(y)

        for i in range(len(t) - 1):
            t0 = t[i]
            t1 = t[i + 1]

            dt = t1 - t0

            # RK4
            k1 = self.dynamics(
                t0,
                y,
                theta,
            )

            k2 = self.dynamics(
                t0 + dt / 2,
                y + dt * k1 / 2,
                theta,
            )

            k3 = self.dynamics(
                t0 + dt / 2,
                y + dt * k2 / 2,
                theta,
            )

            k4 = self.dynamics(
                t1,
                y + dt * k3,
                theta,
            )

            y = y + (
                dt / 6.0
                * (
                    k1
                    + 2.0 * k2
                    + 2.0 * k3
                    + k4
                )
            )

            trajectory.append(y)

        trajectory = torch.stack(
            trajectory,
            dim=1,
        )

        return trajectory
