from registry import Registry

from .gru import GRUEncoder
from .tcn import TCNEncoder
from .transformer import TransformerEncoder
from .neural_ode import NeuralODEEncoder


ENCODER_REGISTRY = Registry("ENCODER")


ENCODER_REGISTRY.register(GRUEncoder)
ENCODER_REGISTRY.register(TCNEncoder)
ENCODER_REGISTRY.register(TransformerEncoder)
ENCODER_REGISTRY.register(NeuralODEEncoder)


__all__ = [
    "ENCODER_REGISTRY",
    "GRUEncoder",
    "TCNEncoder",
    "TransformerEncoder",
    "NeuralODEEncoder",
]
