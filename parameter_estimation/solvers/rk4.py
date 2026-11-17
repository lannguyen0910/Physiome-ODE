import torch


class RK4Solver:
    def __init__(self):
        pass

    @staticmethod
    def step(
        ode,
        t0,
        y,
        theta,
        dt,
    ):

        k1 = ode(
            t0,
            y,
            theta,
        )

        k2 = ode(
            t0 + dt / 2.0,
            y + dt * k1 / 2.0,
            theta,
        )

        k3 = ode(
            t0 + dt / 2.0,
            y + dt * k2 / 2.0,
            theta,
        )

        k4 = ode(
            t0 + dt,
            y + dt * k3,
            theta,
        )

        y_next = (
            y
            +
            (
                dt / 6.0
            )
            * (
                k1
                + 2.0 * k2
                + 2.0 * k3
                + k4
            )
        )

        return torch.nan_to_num(
            y_next,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

    def solve(
        self,
        ode,
        y0,
        t,
        theta,
    ):

        if t.ndim != 1:

            raise ValueError(
                "RK4Solver expects time "
                "shape [T], got "
                f"{tuple(t.shape)}."
            )

        if y0.ndim != 2:

            raise ValueError(
                "y0 must have shape [B,D]."
            )

        if theta.ndim != 2:

            raise ValueError(
                "theta must have shape [B,P]."
            )

        trajectory = [
            y0
        ]

        y = y0

        for i in range(
            len(t) - 1
        ):

            t0 = t[i]
            t1 = t[i + 1]

            dt = t1 - t0

            y = self.step(
                ode=ode,
                t0=t0,
                y=y,
                theta=theta,
                dt=dt,
            )

            trajectory.append(y)

        return torch.stack(
            trajectory,
            dim=1,
        )

    def __call__(
        self,
        ode,
        y0,
        theta,
        t,
    ):

        return self.solve(
            ode=ode,
            y0=y0,
            theta=theta,
            t=t,
        )
