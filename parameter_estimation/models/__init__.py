from registry import Registry

from .parameter_estimator import ParameterEstimator


MODEL_REGISTRY = Registry("MODEL")


MODEL_REGISTRY.register(
    ParameterEstimator,
    name="parameter_estimator",
)


__all__ = [
    "MODEL_REGISTRY",
    "ParameterEstimator",
]
