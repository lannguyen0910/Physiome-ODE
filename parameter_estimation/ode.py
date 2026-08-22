import torch
import torch.nn as nn


class PhysiomeODE(nn.Module):
    def __init__(self):
        super().__init__()

        self.state_dim = 2
        self.parameter_dim = 13

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute dy/dt.

        Parameters
        ----------
        t:
            Current time.

            Shape:
                scalar
                or [batch]

        y:
            Current state.

            Shape:
                [batch, 2]

        theta:
            ODE parameters.

            Shape:
                [batch, 13]

        Returns
        -------
        dydt:
            Shape:
                [batch, 2]
        """

        # States
        Z = y[:, 0]
        Y = y[:, 1]

        # Parameters
        v0 = theta[:, 0]
        v1 = theta[:, 1]
        beta = theta[:, 2]

        VM2 = theta[:, 3]
        VM3 = theta[:, 4]

        KR = theta[:, 5]
        KA = theta[:, 6]

        kf = theta[:, 7]
        k = theta[:, 8]

        K2 = theta[:, 9]

        n = theta[:, 10]
        m = theta[:, 11]
        p = theta[:, 12]

        # Algebraic variables
        v2 = (
            VM2
            * Z**n
            / (K2**n + Z**n)
        )

        v3 = (
            VM3
            * (
                Y**m
                / (KR**m + Y**m)
            )
            * (
                Z**p
                / (KA**p + Z**p)
            )
        )

        # ODE equations
        dZ = (
            v0
            + v1
            - v2
            - kf * Z
            + k * Y
        )

        dY = (
            beta * v2
            - v3
            - k * Y
        )

        return torch.stack(
            [dZ, dY],
            dim=-1,
        )


class RK4Solver:
    """
    Differentiable fourth-order Runge-Kutta solver.
    The solver supports batched initial states and parameters.

    Input:
        y0    [batch, state_dim]
        t     [time]
        theta [batch, parameter_dim]

    Output:
        trajectory [batch, time, state_dim]
    """

    def __init__(
        self,
        dynamics,
    ):
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
                dt
                / 6.0
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
