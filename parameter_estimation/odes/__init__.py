from registry import Registry

from .dupont1991_ode import Dupont1991_ODE


ODE_REGISTRY = Registry("ODE")


ODE_REGISTRY.register(Dupont1991_ODE)


__all__ = [
    "ODE_REGISTRY",
    "Dupont1991_ODE",
]
