from registry import Registry

from .parameter import ParameterMetrics
from .trajectory import TrajectoryMetrics
from .jgd import JGDMetrics


METRIC_REGISTRY = Registry("METRIC")


METRIC_REGISTRY.register(ParameterMetrics)
METRIC_REGISTRY.register(TrajectoryMetrics)
METRIC_REGISTRY.register(JGDMetrics)


__all__ = [
    "METRIC_REGISTRY",
    "ParameterMetrics",
    "TrajectoryMetrics",
    "JGDMetrics",
]
