from registry import Registry

from .parameter_estimator import ParameterEstimator


MODEL_REGISTRY = Registry("MODEL")


MODEL_REGISTRY.register(ParameterEstimator)


__all__ = [
    "MODEL_REGISTRY",
    "ParameterEstimator",
]
