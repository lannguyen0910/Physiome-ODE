
import torch
import torch.nn as nn
from torchdiffeq import odeint


class TorchDiffEqSolver(nn.Module):
    """Differentiable ODE solver backed by torchdiffeq."""

    def __init__(
        self,
        method="dopri5",
        rtol=1e-5,
        atol=1e-7,
        step_size=None,
    ):
        super().__init__()

        self.method = method
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.step_size = step_size

    def forward(self, ode, y0, theta, t):
        # torchdiffeq requires a 1-D, ordered time vector.
        t = torch.as_tensor(t, dtype=y0.dtype, device=y0.device)

        if t.ndim != 1:
            raise ValueError(
                f"integration_times must be 1-D, got shape {tuple(t.shape)}"
            )

        if t.numel() < 2:
            raise ValueError("At least two integration times are required.")

        if not torch.all(t[1:] > t[:-1]):
            raise ValueError(
                "integration_times must be strictly increasing."
            )

        # Pass the estimated ODE parameters into every RHS evaluation.
        def dynamics(time, state):
            return ode(time, state, theta)

        options = {}
        if self.step_size is not None:
            if self.method not in {"rk4", "euler", "midpoint"}:
                raise ValueError(
                    "step_size is configured; use a fixed-step method "
                    "such as rk4, euler, or midpoint."
                )
            options["step_size"] = float(self.step_size)

        return odeint(
            func=dynamics,
            y0=y0,
            t=t,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            options=options or None,
        )
