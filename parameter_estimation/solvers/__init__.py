from registry import Registry

from .rk4 import RK4Solver
from .torch_diffeq import TorchDiffEqSolver

SOLVER_REGISTRY = Registry("SOLVER")


SOLVER_REGISTRY.register(RK4Solver)
SOLVER_REGISTRY.register(TorchDiffEqSolver)


__all__ = [
    "SOLVER_REGISTRY",
    "RK4Solver",
    "TorchDiffEqSolver",
]
