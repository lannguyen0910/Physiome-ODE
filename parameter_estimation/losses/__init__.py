from registry import Registry

from .loss import ParameterEstimationLoss


LOSS_REGISTRY = Registry("LOSS")


LOSS_REGISTRY.register(
    ParameterEstimationLoss,
    name="parameter_estimation_loss",
)


__all__ = [
    "LOSS_REGISTRY",
    "ParameterEstimationLoss",
]
