from registry import Registry

from .rk4 import RK4Solver


SOLVER_REGISTRY = Registry("SOLVER")


SOLVER_REGISTRY.register(RK4Solver)


__all__ = [
    "SOLVER_REGISTRY",
    "RK4Solver",
]
