from registry import Registry

from .gru import GRUEncoder
from .tcn import TCNEncoder
from .transformer import TransformerEncoder
from .neural_ode import NeuralODEEncoder


ENCODER_REGISTRY = Registry("ENCODER")


ENCODER_REGISTRY.register(
    GRUEncoder,
    name="gru",
)

ENCODER_REGISTRY.register(
    TCNEncoder,
    name="tcn",
)

ENCODER_REGISTRY.register(
    TransformerEncoder,
    name="transformer",
)

ENCODER_REGISTRY.register(
    NeuralODEEncoder,
    name="neural_ode",
)


__all__ = [
    "ENCODER_REGISTRY",
    "GRUEncoder",
    "TCNEncoder",
    "TransformerEncoder",
    "NeuralODEEncoder",
]
